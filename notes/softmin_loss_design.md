# Sample-level SoftMin PEM loss — final design (lead synthesis)

> Synthesizes Designs 1–3. Grounded in the OFFICIAL scorer
> (`../NLPCC-2026-Task10-Science/offline_eval/evaluate.py:220-250`) and the ACTUAL
> ms-swift `compute_loss` / collator source (reference copy under
> `/tmp/VBench/.../swift/`, same ~3.x lineage as 4.2.3 — re-verify on server).
> All math claims below were checked numerically with pure-python (`/tmp`, no torch).

---

## 0. TL;DR of the synthesis decisions

| Decision | Choice | From | Why |
|---|---|---|---|
| Aggregation form | **LSE** `(1/β)·logsumexp(β·Lᵢ)` not the weighted `Σ softmax·Lᵢ` | D3 | weighted form has NEGATIVE gradient on good sentences (verified) |
| **Normalizer** | **subtract `log(n_p)/β`** (the `−log n` term) | **D2** | makes β→0 → mean (safe no-op) AND removes n-bias; gradient is UNCHANGED (it is a constant in L). This is the key fix to the current plugin. |
| Token reduction | length-normalized per-sentence CE (geo-mean) | D2/D3 | long vs short sentences comparable in the soft-min |
| Batch = paragraph | custom `ParagraphGroupSampler` (one batch == one paragraph) | all | keeps the whole paragraph in one optimizer step, no accum-boundary split |
| group_id transport | **piggyback the `channel` field** (NO swift edits) | new (lead) | `channel` already survives `.map()`+collator+`compute_loss`; paragraph_id does not |
| Oversampling safety | augmented rows get a **unique singleton** paragraph_id | D3 | singleton S_p == its own Lᵢ == plain CE; cannot pollute a real paragraph |
| packing / padding_free | **MUST be OFF for the softmin run** | new (lead) | packing concatenates sentences into one row (per-sentence Lᵢ unrecoverable) and drops `channel`; padding_free flattens. Correctness > throughput. |
| Defaults | β=5.0, λ=0.5; β-warmup 1→5 over first ~10% steps | D2/D3 | sweep on dev by official SCORE |

---

## 1. Problem shape — why mean-CE is wrong for PEM

`score = (sentence Macro-F1 + PEM) / 2` (evaluate.py:250). PEM is an **AND over a
paragraph's sentences**: `paragraph_correct += 1 if all_correct else 0`
(evaluate.py:236), where `all_correct` requires `pred_label ∈ gold_options` for
EVERY sentence (evaluate.py:224, 230). The probability greedy decode gets a whole
paragraph right is `P(para) = ∏ᵢ pᵢ` (pᵢ = correct-label prob of sentence i), so
`−log P(para) = Σᵢ ℓᵢ` with `ℓᵢ = −log pᵢ`.

Plain mean-CE minimizes `mean_i ℓᵢ` — it is happy to leave one hard sentence wrong if
the rest are easy. PEM is decided by the paragraph's **worst (bottleneck) sentence**.
So we replace the per-paragraph mean with a temperature-soft **maximum over per-sentence
losses** = soft-**minimum** over correct-label probs = soft bottleneck of `∏ pᵢ`.

This is **precision-critical** (notes/analysis.md §4): trivial all-Supported = 0.432
SCORE beats every reported LLM baseline because they over-fire minority labels and
shatter the 67% clean paragraphs' PEM. The softmin term spends gradient on whichever
single sentence is currently most likely to break a paragraph — which on clean
paragraphs is "the Supported sentence the model is least confident on" (good: defends
PEM), and on hard paragraphs is "the minority-class sentence it is getting wrong"
(good: buys PEM on the 33% PEM-hard paragraphs). It does NOT itself cause over-firing;
class balance / abstention is still tuned separately on dev SCORE.

Constraint compliance: each sentence stays its OWN sequence (shared system + label
defs + evidence images/captions + full claim_text + ONE target sentence → `{"label":X}`).
NO cross-sentence attention, NO joint paragraph output. The bottleneck coupling lives
ONLY in loss-space aggregation across the per-sentence forward passes of one paragraph.

---

## 2. Key equations

Notation: paragraph `p` has sentences `i = 1..n_p`. `T_i` = set of non-masked
(label ≠ −100) response token positions of sample i.

**Per-sentence CE (response-only, length-normalized):**
```
ℓ_i = −(1/|T_i|) · Σ_{t∈T_i} log softmax(z_{i,t})[y_{i,t}]          (1)
```
`p_i = exp(−ℓ_i) ∈ (0,1]` is the token-geometric-mean correct-label prob. Length
normalization is what makes `ℓ_i` exactly the per-sample scalar and prevents long
sentences from dominating the soft-min by token count alone.

**Per-paragraph soft-max-over-losses (the SOFTMIN core — NORMALIZED LSE):**
```
S_p(β) = (1/β) · [ logsumexp_{i∈p}(β·ℓ_i) − log n_p ]                (2)
       = (1/β) · log( (1/n_p) Σ_{i∈p} exp(β·ℓ_i) )
```
Probability reading (why "soft-min over probs"): `exp(β·ℓ_i) = p_i^{−β}`, so
`S_p = −(1/β)·log( (1/n_p) Σ_i p_i^{−β} )` is a power-mean of the `p_i` with exponent
`−β` → a soft MINIMUM of the correct-probs. Driving `S_p` down lifts the SMALLEST `p_i`
= exactly the PEM bottleneck.

**Total objective (per optimizer step; mean over the G paragraphs in the global batch):**
```
L = (1−λ)·CE_mean  +  λ·(1/G) Σ_p S_p(β)                            (3)
CE_mean = (Σ_p Σ_{i∈p} ℓ_i) / (Σ_p n_p)     (length-normalized per-sentence mean)
```
The `(1−λ)·CE_mean` floor keeps EVERY sentence — including non-bottleneck minority-class
sentences — learning (protects Macro-F1, prevents class collapse). λ must stay < 1.

**Gradient (proof of bottleneck emphasis), numerically verified to 4 d.p.:**
```
∂S_p/∂ℓ_j = exp(β·ℓ_j) / Σ_{i∈p} exp(β·ℓ_i) = softmax_j(β·ℓ) =: w_j   (4)
```
The `−log n_p` term in (2) is a CONSTANT in `ℓ`, so it does NOT change the gradient
(verified: grad_unnorm == grad_norm == softmax to 4 d.p.). Properties of `w`:
- `w_j ≥ 0`, `Σ_j w_j = 1` → a convex combination, NEVER a negative push on a good
  sentence (the failure mode of the weighted form, see §3).
- monotone in `ℓ_j`: the worst sentence gets the largest weight; the gap is
  multiplicative, `w_a/w_b = exp(β·(ℓ_a−ℓ_b))`.
- `β→∞`: `w →` one-hot on `argmax_i ℓ_i` (hard bottleneck = exact PEM AND).
- `β→0`: `w → 1/n_p` (uniform) and, BECAUSE OF the `−log n_p` normalizer, `S_p → mean_i ℓ_i`
  so (3) collapses to plain CE — a true SAFE no-op (verified: S_norm(β=0.001)=0.70005 vs
  mean 0.70000; WITHOUT the normalizer S_unnorm diverges to 1099).

**Bounds (verified):** `max_i ℓ_i − (something) ≤` … concretely the normalized form
satisfies `max_i ℓ_i − log(n_p)/β ≤ S_p ≤ max_i ℓ_i`, i.e. it is a smooth UNDER-estimate
of the hard max that tightens to it as β grows. (The unnormalized form sat ABOVE the max
by up to `log(n)/β`, which is the source of its n-bias.)

**n-bias removed (the practical reason the normalizer matters here):** for an all-equal
clean paragraph `ℓ_i ≡ c`, the normalized `S_p = c` for ALL n (verified: n∈{2,4,8} all
give 0.5 for c=0.5), whereas the unnormalized form gave 0.64/0.78/0.92 — it would
systematically over-penalize LONG clean paragraphs and over-weight large-n paragraphs in
`mean_p`. Long clean paragraphs are the protected 67% majority, so removing this bias is
directly aligned with the precision-critical metric.

**Effective per-sample weight** (combining both terms of (3)), N = Σ_p n_p:
```
α_i = (1−λ)/N + (λ/G)·w_i^{(p)},   w_i = softmax(β·ℓ)_i                (5)
```
Since `w_argmax ≥ 1/n_p`, the worst sentence's total weight is never LESS than its
mean-CE share — the softmin term can only ADD to the bottleneck, never subtract.

---

## 3. Why LSE, not the weighted form (rejected design)

The research sketched a "weighted" surrogate `S = Σ_i softmax(β·ℓ_i)·ℓ_i`. Its gradient is
`∂S/∂ℓ_j = w_j + β·w_j·(ℓ_j − S)`, which is **NEGATIVE for non-bottleneck sentences**
(verified: `L=[0.2,1.0,0.25], β=5` → grad `[−0.050, +1.109, −0.059]`). A negative
gradient on a good sentence actively pushes its loss UP — a real pathology. The LSE form
(2) has gradient = `softmax` ∈ [0,1] summing to 1: always a safe non-negative convex
combination. **We use LSE.**

---

## 4. Numerical stability (mandatory)

- NEVER exponentiate raw `β·ℓ_i` (CE is tens of nats early in training; `exp` overflows
  bf16 instantly). ALWAYS `torch.logsumexp(β·g)` (internally max-subtracts).
- Cast `per_sentence` to **float32** before the LSE even under bf16 training.
- Guard `n_p == 1`: skip logsumexp, `S_p = ℓ_1` (LSE of one element == that element).
- `tok_counts.clamp(min=1)` so an all-masked / empty sample never divides by zero.
- The `−log(n_p)` term: `math.log(len(idxs))`, subtracted after `logsumexp`, divided by β.

---

## 5. ms-swift 4.2.3 integration — three coupled pieces + the data contract

All in ONE plugin: `src/nlpcc_t10/softmin_plugin.py`, passed via
`--custom_register_path src/nlpcc_t10/softmin_plugin.py`, selected via `--loss_type softmin_pem`.
The trainer subclass is wired in the launcher (see §6).

### 5.1 group_id transport — PIGGYBACK THE `channel` FIELD (no swift edits)

Confirmed from source: `compute_loss` builds a LOCAL `loss_kwargs` dict and the ONLY
non-tensor metadata it natively pops + forwards is `loss_scale` (line 162) and `channel`
(line 168). `channel` is special-cased through the WHOLE pipeline:
`keep_columns` includes it → collator gathers `res['channel'] = [b['channel'] …]`
(base.py:1377-1383) → `compute_loss` pops it and sets `loss_kwargs['sample_channels']`
AND `loss_kwargs['trainer']=self` (trainers.py:168-175) → the loss receives
`sample_channels=...`. `paragraph_id` has NO such slot.

**Therefore: put the paragraph_id string into the `channel` field of each row.** Then it
rides the existing, supported `channel` path with ZERO swift internal edits, and our loss
reads it from `sample_channels`. The loss treats `sample_channels` as the list of
paragraph ids (length B, one per sample). This is the cleanest realizable transport.

> Fallback (only if a future version repurposes `channel`): add `'paragraph_id'` to
> `keep_columns` in `swift/llm/dataset/preprocessor/core.py` and to the collector list in
> `base.py:_data_collator`, then pop it in our `compute_loss`. Documented but NOT preferred
> (edits swift internals).

### 5.2 Sampler — one batch == one paragraph

`ParagraphGroupSampler(paragraph_ids, shuffle, seed, rank, world_size)`: groups dataset
indices by paragraph_id, yields one variable-length batch per paragraph. Set
`--per_device_train_batch_size 1` (irrelevant under a custom `batch_sampler`; the sampler
controls real batch size). `gradient_accumulation_steps` still works on top (accumulation
ACROSS paragraphs is fine and desirable).

`SoftMinTrainer(Seq2SeqTrainer)` overrides `get_train_dataloader` to build a
`DataLoaderShard(dataset, device=..., batch_sampler=sampler, collate_fn=self.data_collator, …)`.

### 5.3 DDP (2×L40) — pre-shard WHOLE paragraphs by rank

Accelerate does NOT auto-shard a custom `batch_sampler`. The sampler shards whole
paragraphs round-robin on a fixed shuffled order: `order = order[rank::world_size]`. Each
GPU therefore ALWAYS sees complete paragraphs; never split a paragraph across ranks (a
per-rank partial-paragraph LSE is the WRONG bottleneck, and it fails silently). DDP then
all-reduces (mean) the per-paragraph S_p gradients across ranks — correct. Drop-tail so
ranks get equal step counts (else DDP hangs at grad all-reduce). Add a step-0 assertion:
every sample in a batch shares one paragraph_id and that id maps to exactly one rank.

### 5.4 Loss plugin

`softmin_pem_loss(outputs, labels, loss_scale=None, num_items_in_batch=None,
sample_channels=None, trainer=None, **extra)` — signature matches the real call
`compute_loss_func(outputs, labels, num_items_in_batch=..., **loss_kwargs)`
(trainers.py:219). `sample_channels` = the paragraph ids (from §5.1). Body:
1. recompute per-sentence CE from `outputs.logits` + `labels` with
   `CrossEntropyLoss(reduction='none', ignore_index=-100)`, shift, `.view([B,T-1])`,
   divide by per-row `tok_counts.clamp(min=1)` → `ℓ` `[B]`, cast `.float()`.
2. `CE_mean = ℓ.mean()`.
3. if `sample_channels is None` (e.g. eval) → return `CE_mean` (graceful degrade to CE).
4. group indices by paragraph id; for each group compute the NORMALIZED LSE (2)
   (singletons → their own `ℓ_i`); `S_mean = stack(S_p).mean()`.
5. return `(1−λ)·CE_mean + λ·S_mean`.
β, λ from env `SOFTMIN_BETA` / `SOFTMIN_LAMBDA` (grid search w/o editing code). Optional
β-warmup reads `trainer.state.global_step` / `max_steps` if `trainer` is passed.

### 5.5 DATA CONTRACT — what build_dataset.py MUST emit

Each ms-swift training row (Research-2 schema) gains ONE extra top-level string field:
```json
{ "messages":[…], "images":[…], "channel": "<paragraph_id>" }
```
Rules (build_dataset.py):
1. `channel` carries the **paragraph id** (the original record id, e.g. the record's id
   or `track1-pNNNN`). All ORIGINAL sentences of one record share that id.
2. **Every oversampled minority duplicate AND every downsample-kept duplicate gets a
   UNIQUE singleton id** `"<pid>#augK"` (K = a per-row counter). A singleton's S_p == its
   own ℓ_i == its CE term → augmented rows never become a spurious "worst sentence" inside
   a real paragraph and never couple into another paragraph's bottleneck. THIS is the
   linchpin that lets oversampling and the PEM loss coexist.
   - PREFERRED alternative (D2): oversample WHOLE paragraphs (duplicate every sentence of
     a chosen record, giving the copy a fresh shared id `"<pid>#dupK"`), so a real (if
     duplicated) paragraph's bottleneck is preserved. Either is correct; per-sentence
     singleton augmentation is simpler and is the default.
3. `dev_sft.jsonl` uses REAL ids only, NO augmentation, ORIGINAL distribution (so dev PEM
   is meaningful and the official scorer is faithful).
4. Multi-label target sentence → least-frequent label (analysis.md §3).

> Note: putting paragraph_id in `channel` means ms-swift's channel-loss bookkeeping
> (per-channel loss logging) will treat each paragraph as a "channel". Harmless for our
> custom loss (we ignore that logging), but expect O(#paragraphs) channel keys in logs;
> disable channel-loss logging if it is noisy.

---

## 6. Implementation plan (files · functions · wiring)

### `src/nlpcc_t10/softmin_plugin.py` (UPDATE the existing file)
- `group_indices(ids) -> OrderedDict[id->list[int]]` — pure-python, unit-tested.
- `_per_sentence_ce(outputs, labels) -> Tensor[B]` — shift, CE reduction='none',
  length-normalize, `.float()`.
- `softmin_pem_loss(outputs, labels, loss_scale=None, num_items_in_batch=None,
  sample_channels=None, trainer=None, **kw)` — **(a) read ids from `sample_channels`**
  (the channel piggyback), **(b) NORMALIZED LSE `−log(n_p)`**, (c) env β/λ, (d) optional
  β-warmup from `trainer.state`. Returns scalar.
- `register()` → `register_loss_func("softmin_pem")(softmin_pem_loss)`; wrapped in
  try/except so the file imports without swift locally.
- `build_paragraph_trainer_cls()` → lazy `SoftMinTrainer(Seq2SeqTrainer)` with
  `ParagraphGroupSampler` (rank-sharded) + `get_train_dataloader` override. NOTE:
  paragraph_id arrives via `channel` so we do NOT need to pop it in `compute_loss` — the
  STOCK `compute_loss` already pops `channel` and forwards `sample_channels`. So
  `SoftMinTrainer` only needs the sampler override; no `compute_loss` override required.
  (Remove the speculative `_prepare_loss_kwargs` hook — it does not exist in swift.)
- `__main__` pure-python self-tests (no torch): grouping, NORMALIZED LSE bounds,
  singleton identity, β→0→mean, n-invariance.

### `src/nlpcc_t10/build_dataset.py`
- Emit `channel = paragraph_id` per row per the §5.5 contract; unique singleton ids for
  augmented rows; dev unaugmented. (Server-side, after images unzipped.)

### Launcher wiring (`scripts/train.sh`)
`swift sft` instantiates the default `Seq2SeqTrainer`. To use `SoftMinTrainer`, the
cleanest 4.2.3 mechanism is a tiny launcher that **monkeypatches before the CLI runs**:
```python
# scripts/train_softmin.py
import nlpcc_t10.softmin_plugin as P
import swift.trainers as T
T.Seq2SeqTrainer = P.build_paragraph_trainer_cls()   # replace before swift builds it
from swift.cli.sft import sft_main
sft_main()
```
Confirm on the server whether 4.2.3 exposes a `--custom_trainer` hook; if so prefer it.
This monkeypatch + the channel piggyback are the TWO points to validate first on the server.

### Config deltas for the softmin run (`configs/qwen3vl_lora_sft.yaml`)
- `per_device_train_batch_size: 1` (sampler controls batch size).
- **`packing: false`** and **`padding_free: false`** for the softmin run (packing
  concatenates sentences into one row → per-sentence ℓ_i unrecoverable AND `channel`
  dropped in packing mode, base.py:1368-1373; padding_free flattens rows). Correctness
  over throughput. A plain-CE λ=0 baseline run MAY keep packing on.
- Keep `freeze_vit: true`, `target_modules: all-linear`, `IMAGE_MAX_TOKEN_NUM=1024`.

### Tuning protocol (CLAUDE.md §4/§7)
- Defaults β=5.0, λ=0.5; optional β-warmup 1→5 over first ~10% steps.
- Grid: β∈{2,5,10}, λ∈{0.3,0.5,0.7}. Keep λ≤0.7 (CE floor > 0).
- Model selection by the OFFICIAL `evaluate.py` SCORE on the 9:1-by-record dev — NEVER by
  eval-loss or minority recall. Eval loss may use plain CE.
- MANDATORY ABLATION: identical run λ=0 (plain CE) vs λ>0. If dev PEM does not improve, the
  softmin term is not earning its complexity → fall back to λ=0. Log per-config
  `{score, macro_f1, pem}` to `notes/`.

---

## 7. Graceful degradation (built in)

- **Eval / missing ids**: `sample_channels is None` → loss == plain CE.
- **Singleton paragraph** (n=1, incl. all augmented rows): S_p == ℓ_1 == its CE term.
- **β→0**: normalized LSE → mean → loss == plain CE (λ becomes a no-op).
- **All-Supported clean paragraphs (67%)**: ℓ_i tiny and ~equal → w ≈ uniform → S_p ≈
  mean → negligible extra gradient → clean-paragraph PEM is NOT disturbed (the precise
  precision-critical behaviour the metric demands). With the normalizer this holds
  regardless of paragraph length.

---

## 8. Residual risks (ranked)

1. **PEM-shatter via over-firing** (#1 metric risk): too-high λ/β pushes minority labels
   onto borderline Supported sentences, breaking clean paragraphs → SCORE below the 0.432
   floor (how the LLM baselines lost). Mitigate: gate on dev SCORE; λ≤0.7; β-warmup;
   ablation vs λ=0. The softmin term does not itself over-fire, but it amplifies whatever
   the decision boundary does — tune class balance/abstention on dev SCORE in tandem.
2. **packing/padding_free silently corrupting per-sentence ℓ_i** — MUST be off for the
   softmin run (and they also drop `channel`). Assert on server: `len(ℓ) == B` and the
   collated batch contains `channel` as a python list of length B.
3. **DDP paragraph split** — silent (no crash). Assert per-rank one-id-per-batch at step 0.
4. **`channel` piggyback assumptions** — verify on 4.2.3 that (a) `channel` still survives
   `.map()` via `keep_columns`, (b) the collator still emits `res['channel']` as a list,
   (c) `compute_loss` still pops it into `loss_kwargs['sample_channels']`. If `channel`
   semantics changed, use the `keep_columns` fallback (§5.1).
5. **Trainer monkeypatch / hook** — confirm 4.2.3's mechanism (custom_trainer vs patch)
   and that `Seq2SeqTrainer.get_train_dataloader` / `DataLoaderShard` signatures match.
6. **num_items_in_batch double-normalization** — our loss returns its own paragraph-mean
   scalar; the stock token-count normalization (trainers.py:202-203) only applies on the
   `compute_loss_func is None` path, so our custom path is NOT double-normalized. But the
   loss SCALE differs from stock token-sum CE → sanity-check first-100-step loss magnitude
   vs a plain-CE run; re-tune LoRA lr on dev if needed.
7. **Loss ≠ metric**: low CE can still be a wrong argmax and vice versa (evaluate.py uses
   argmax-label exactness with any-hit on gold `types`). The softmin loss is a training-
   time inductive bias only; ALWAYS select by official SCORE.
8. **Version drift**: the source read is a ~3.x VBench copy. On the server run
   `python -c "import swift; print(swift.__version__)"` and diff `trainers.py` compute_loss,
   `base.py` collator, `data_loader.py`, `register_loss_func`/`LOSS_MAPPING` before trusting
   any dev number.
9. **Oversampling × paragraph integrity**: only augmented rows may be singletons; do NOT
   duplicate/drop INDIVIDUAL sentences inside a real paragraph (changes n_p, corrupts the
   bottleneck). Whole-paragraph duplication or per-sentence-singleton augmentation only.
