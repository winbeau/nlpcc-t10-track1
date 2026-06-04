# Gemma4-12B (Unified) Adaptation Plan for ms-swift 4.2.3

> Recon date: 2026-06-04. **Execute only AFTER the 26B/31B runs settle** (and only if the worth-it gate in §6 passes).
> This is the authoritative, self-contained plan. A future session can act on it without re-running recon.

---

## TL;DR

- **Feasibility:** `infeasible-in-pinned-env / moderate-with-isolated-venv`. The pinned `transformers 5.8.1` does **not** contain the class the 12B needs (`Gemma4UnifiedForConditionalGeneration`). It cannot be loaded at all in the shared env, and `--model_type gemma4` is **not** a valid workaround. Doable cleanly only in a **separate `.venv-gemma12b`** with `transformers>=5.10.0` + a ~15-line `gemma.py` patch.
- **Recommendation:** **DON'T BOTHER** while 26B/31B are alive. 12B is a 3rd same-family (Gemma4) member with a *weaker, encoder-free* visual pathway → high risk of the **q4b failure mode** (FP minority preds that break clean paragraphs, dropping PEM). Revisit only if **both** 26B and 31B fail to train.

---

## 1. Feasibility Verdict + Evidence

**Verdict: NOT feasible in the pinned env (transformers 5.8.1). Moderate effort (~2-3h) in an isolated venv.**

The blocker is a hard class-availability gap, evidenced by three facts:

### 1a. The 12B is a DIFFERENT model class than 26B/31B/E4B

| | 12B | 26B / 31B / E2B / E4B |
|---|---|---|
| `config.json` `architectures` | `["Gemma4UnifiedForConditionalGeneration"]` | `["Gemma4ForConditionalGeneration"]` |
| `model_type` | `gemma4_unified` | `gemma4` |
| vision pathway | **encoder-free** — `vision_config.model_type=gemma4_unified_vision`, `mm_embed_dim=3840`, `num_soft_tokens=280`, `patch_size=16`, `pooling_kernel_size=3`; a tiny (~35M) patch embedder that projects patches directly to soft tokens | full **SigLIP-style vision_tower** (~550M) |
| processor | `Gemma4UnifiedProcessor` / `Gemma4UnifiedImageProcessor` (in `processor_config.json`; **no** `preprocessor_config.json`) | standard Gemma4 processor |
| exact id | `google/gemma-4-12B-it` (HF) | `google/gemma-4-{E2B,E4B,26B-A4B,31B}-it` |
| weights | single `model.safetensors`, 23,919,549,408 B (~22.3 GiB), ~11.7B params bf16 | larger |
| `transformers_version` in config | `5.10.0.dev0` | (5.8.x compatible) |

→ **It is NOT the same class.** Any "just register the id under `gemma4`" hope is dead.

### 1b. transformers 5.8.1 does NOT have the class

`grep -rn "Unified" .venv/.../transformers/models/gemma4/` → **zero hits** across modeling / configuration / `__init__`.
`Gemma4ForConditionalGeneration` is present and importable; `Gemma4UnifiedForConditionalGeneration` is **absent**. It first appears in `transformers 5.10.0.dev0` (matching the 12B config's declared `transformers_version`).

### 1c. ms-swift 4.2.3 gap — TWO independent blockers

`swift/model/models/gemma.py` registers exactly one Gemma4 `ModelMeta`:

```python
register_model(ModelMeta(
    MLLMModelType.gemma4,
    [ ModelGroup([E2B,E2B-it,E4B,E4B-it], template=gemma4_nothinking),
      ModelGroup([31B,31B-it,26B-A4B,26B-A4B-it], template=gemma4) ],
    Gemma4Loader,
    architectures=['Gemma4ForConditionalGeneration'],   # ← only the OLD class
))
```

- **Blocker 1 (id lookup):** `google/gemma-4-12B-it` is in **no** `ModelGroup` → id-based auto-detect misses it.
- **Blocker 2 (arch lookup):** the registered `architectures` key is the old class only → arch-based auto-detect also misses the 12B's `Gemma4UnifiedForConditionalGeneration`.
- **`--model_type gemma4` is NOT a workaround:** `Gemma4Loader` hard-codes `auto_model_cls = Gemma4ForConditionalGeneration` (wrong class — and that class can't even read a `gemma4_unified` config in 5.8.1) **and** applies `_patch_gemma4_forward()` (a forward patch written for the SigLIP-tower arch, wrong for the encoder-free arch). Forcing it would crash or silently corrupt the forward.

**Conclusion:** to run the 12B you must (a) get `transformers>=5.10.0`, AND (b) teach ms-swift a new loader + ModelGroup. (a) cannot be done in the shared `.venv` without breaking the 5 Qwen members + the 26B/31B Gemma path. Hence **isolated venv only** (§3).

### 1d. Download mirror (good news)

`USE_MODELSCOPE=1 snapshot_download('google/gemma-4-12B-it')` **succeeds** — ModelScope mirrors it (tokenizer + config + weights). So `USE_HF=1` is not strictly required; either hub works. `config.json` has **no** `auto_map`, so `trust_remote_code=True` is a non-option (there is no bundled modeling code to trust).

---

## 2. What this affects in our repo

- Launcher reused as-is: `scripts/train_gemma4.sh` (env knobs: `MODEL_ID`, `TEMPLATE`, `GPUS`, `MAX_LENGTH`, `MASTER_PORT`, `DEEPSPEED`, `MAX_STEPS`, `SMOKE`, `NO_INFER`). It already passes `--use_logits_to_keep false`, `--freeze_vit true --freeze_aligner true`, softmin loss, `--template gemma4_nothinking`.
- It currently runs under the shared `.venv` via `uv run`. For 12B we must run it under `.venv-gemma12b` instead (§3, §4).
- Output feeds `scripts/ensemble_union.py --min-votes 1` as one more union member (§5).

---

## 3. Adaptation Steps — Option B (isolated venv; ONLY supported path)

> **Risk to pinned env = ZERO.** All changes live inside a *new* directory `.venv-gemma12b`. The shared `.venv` (5 Qwen members + InternVL + 26B/31B) is never touched. The only discipline: always activate `.venv-gemma12b` before any 12B command; never edit the shared `.venv`'s `gemma.py`.

```bash
REPO=/home/winbeau/wenbiao_zhao/nlpcc-t10-track1   # adjust to the server path; on the GPU box it is /data/.../nlpcc-t10-track1
cd "$REPO"

# (1) Create the isolated venv
uv venv .venv-gemma12b --python 3.10

# (2) Install deps. Keep torch/torchvision IDENTICAL to the shared env so CUDA libs match the driver.
uv pip install --python .venv-gemma12b \
    'transformers>=5.10.0' \
    'ms-swift==4.2.3' \
    'torch==2.12.0+cu130' 'torchvision==0.27.0+cu130' \
    --extra-index-url https://download.pytorch.org/whl/cu130
# (also: accelerate, deepspeed, peft, jinja2, etc. — let ms-swift pull them; pin nothing else unless a conflict appears)

# (3) Verify the class is importable (this is the whole point of the upgrade)
source .venv-gemma12b/bin/activate
python -c 'from transformers import Gemma4UnifiedForConditionalGeneration; print("CLASS OK")'
```

### (4) Patch ms-swift to register the 12B (inside `.venv-gemma12b` ONLY)

Find the file:
```bash
GEMMA=$(python -c 'import swift.model.models.gemma as m; print(m.__file__)')
echo "$GEMMA"   # MUST contain .venv-gemma12b — if it says shared .venv, STOP (wrong interpreter)
```

Append to that `gemma.py` (after the existing `register_model(...gemma4...)` block):

```python
# --- Gemma4-12B Unified (encoder-free) ---  [added for NLPCC fleet, isolated venv only]
from transformers import Gemma4UnifiedForConditionalGeneration

class Gemma4UnifiedLoader(ModelLoader):
    # NOTE: do NOT call _patch_gemma4_forward() — that patch targets the SigLIP-tower
    # gemma4 arch. The Unified arch handles masking/soft-tokens natively.
    def _post_init(self, model, processor):
        pass

register_model(ModelMeta(
    MLLMModelType.gemma4_unified,                       # new enum value (see step 4b)
    [ ModelGroup([ Model('google/gemma-4-12B-it', 'google/gemma-4-12B-it') ],
                 template=TemplateType.gemma4_nothinking) ],
    Gemma4UnifiedLoader,
    architectures=['Gemma4UnifiedForConditionalGeneration'],
    requires=['transformers>=5.10.0'],
))
```

**(4b)** Add the enum value `gemma4_unified = 'gemma4_unified'` next to the existing `gemma4 = 'gemma4'` in whatever module defines `MLLMModelType` (in ms-swift 4.2.3 this is `swift/llm/model/register.py` / `model_arch`-adjacent; grep `class MLLMModelType` to locate). If editing the frozen enum is awkward, you may instead reuse `MLLMModelType.gemma4` as the type and rely on the `architectures=['Gemma4UnifiedForConditionalGeneration']` key + the dedicated loader for disambiguation — but a distinct enum value is cleaner and avoids the wrong loader being picked.

**(4c) Confirm exact symbol names** before assuming the snippet imports cleanly — ms-swift 4.2.3 internals:
```bash
python - <<'PY'
import swift.model.models.gemma as g
print("ModelLoader?",     hasattr(g,'ModelLoader'))
print("ModelMeta?",       hasattr(g,'ModelMeta'))
print("ModelGroup?",      hasattr(g,'ModelGroup'))
print("Model?",           hasattr(g,'Model'))
print("register_model?",  hasattr(g,'register_model'))
from swift.llm import TemplateType
print("TemplateType.gemma4_nothinking?", hasattr(TemplateType,'gemma4_nothinking'))
PY
```
If any base symbol differs (e.g. the loader base class is named differently in 4.2.3), copy the import/inheritance pattern from the existing `Gemma4Loader` in the same file rather than guessing.

### (5) Download the weights (either hub)
```bash
# ModelScope (mirror confirmed working) — or set USE_HF=1 to pull from HF (token already set, user winbeau)
USE_HF=0 USE_MODELSCOPE=1 python -c \
  "from modelscope import snapshot_download; snapshot_download('google/gemma-4-12B-it')"
```

### (6) Freeze-flag sanity (encoder-free arch)
`train_gemma4.sh` passes `--freeze_vit true --freeze_aligner true`. The 12B has **no separate vision_tower**, so `freeze_vit` is expected to be a benign **no-op** (look for `frozen vit params: 0` in the log). If ms-swift *errors* instead of no-op'ing on the missing tower, set `--freeze_vit false --freeze_aligner false` (LoRA the LLM only; the ~35M embedder is tiny and either way contributes negligible trainable params). Add a `FREEZE_VIT`/`FREEZE_ALIGNER` env passthrough to the launcher if needed.

---

## 4. Smoke Verification Protocol (E4B-style)

The E4B-it smoke (verified working) is the template: template=`gemma4_nothinking`, `use_logits_to_keep=false`, softmin loss, loss fell ~3.6→~1.0 over 8 steps, ~58 GiB. Replicate for 12B:

```bash
source .venv-gemma12b/bin/activate    # CRITICAL — must NOT be the shared .venv
MAX_STEPS=8 SMOKE=1 \
  MODEL_ID=google/gemma-4-12B-it \
  TEMPLATE=gemma4_nothinking \
  TAG=gemma4_12b_smoke \
  GPUS=<one_free_H200_index> \
  bash scripts/train_gemma4.sh
```

**Pass criteria (ALL must hold):**
1. Model loads via `Gemma4UnifiedLoader` (no "unknown model_type" / no wrong-class crash).
2. Template renders JSON-label targets with no `<|channel|>thought` leakage (nothinking confirmed).
3. `use_logits_to_keep=false` honored (softmin needs causal-shift alignment).
4. Loss **decreases** over 8 steps (expect ~3.5→~1.0, same shape as E4B). Flat/NaN loss → drop LR to 5e-5, re-smoke.
5. No OOM (≤22 GiB weights + LoRA fits one H200 with headroom).
6. Freeze flags either no-op (`frozen vit params: 0`) or were flipped to `false` per §3(6).

Only if **all six** pass, proceed to full train.

---

## 5. Full Train + Ensemble Recipe (memory)

```bash
source .venv-gemma12b/bin/activate
MODEL_ID=google/gemma-4-12B-it \
  TEMPLATE=gemma4_nothinking \
  TAG=gemma4_12b \
  GPUS=<one_free_H200_index> \
  bash scripts/train_gemma4.sh
# -> trains LoRA (1 epoch, softmin_pem), then infers testp1 per-sentence with the SAME template,
#    aggregates -> submissions/testp1_gemma4_12b_submission.jsonl + .zip
```

Add as a union member (recall-additive; the only proven lever):
```bash
# from the SHARED .venv (ensemble_union.py is pure-python, no transformers dep)
uv run python scripts/ensemble_union.py --min-votes 1 \
  --out submissions/sNN_unionPlus12b.jsonl \
  submissions/testp1_s01_grid.jsonl \
  submissions/testp1_s08_os4.jsonl \
  submissions/testp1_s09_perclass.jsonl \
  submissions/testp1_joint8b.jsonl \
  submissions/testp1_joint32b.jsonl \
  submissions/testp1_gemma4_12b_submission.jsonl   # + 26B/31B members if trained
# validate -> make_submission_zip.py -> Codabench -> append notes/submissions_log.md
```

**Recipe key facts to remember:** `--use_logits_to_keep false`; `--template gemma4_nothinking` at BOTH train AND infer (infer.py has a `--template` passthrough — `train_gemma4.sh` already wires it); `MAX_PIXELS=401408` (higher resolution is REFUTED, −3.3); single H200, no DeepSpeed needed (~22 GiB). Archive the `gemma.py` diff next to this file (`notes/gemma4_unified_loader.patch`) so the venv is reproducible if lost.

---

## 6. HONEST Worth-It Assessment

**Bottom line: 12B is most likely net-NEGATIVE as a union member if 26B and/or 31B are already in the union.** Treat it as a fallback, not a target.

### The two lessons that govern this decision (from real testp1 fleet data)

1. **q4b lesson (weak same-family member HURTS):** s18 added Qwen3-VL-**4B** to the 5-Qwen base. Result **49.71 < s15=50.26 (−0.55)**. Macro-F1 nudged up (57.17→57.28) but **PEM dropped 1.19 (43.34→42.15)** — the extra ~22 minority preds were mostly **false positives that shattered clean paragraphs**. A weaker, error-correlated member adds redundant/harmful FPs, not new true positives. q4b was kicked out; base returned to 5 Qwen.

2. **InternVL lesson (what actually helps = ARCHITECTURE diversity, not scale):** InternVL3 (InternViT encoder + non-Qwen LLM) added **+90 net real minority** over the 5-Qwen base (s21 472 vs s15 382; UE+59, Contra+17, UCM+7, SO+7). The win came from a *different vision encoder AND different language backbone* producing *uncorrelated* errors.

### Where Gemma4-12B falls on that spectrum — badly, on both axes

- **Vision axis = WEAKER than what's already in the union.** The 12B is **encoder-free**: a ~35M patch embedder → 280 soft tokens, NOT a real chart/table parser. Our task is chart/table claim verification where the *visual* read of numeric values is the label-relevant signal. The 12B's visual pathway is a **capability downgrade** vs the SigLIP tower in 26B/31B and vs InternViT in InternVL. This is precisely the q4b setup: a member whose marginal minority preds skew toward FP.
- **Language axis = CORRELATED with 26B/31B.** All three (12B/26B/31B) share the Gemma4 instruction-tuning distribution and reasoning biases. Errors driven by language/reasoning will be **shared** across the three Gemma members — unlike Qwen-vs-InternVL which differ on *both* vision and language. A 3rd Gemma in a union that already has 26B + 31B contributes little independent signal.
- **Marginal-value math:** for 12B to pay off it must correctly catch minority sentences missed by ALL of {5 Qwen, 26B, 31B, InternVL-2B, InternVL-8B}. With a weaker encoder *and* a correlated LLM, expected unique-correct contribution is low while expected FP contribution (q4b-style) is real. Expected sign: **negative on PEM**, which is the binding constraint.

### Worth-it decision table

| Situation | Pursue 12B? |
|---|---|
| 26B AND/OR 31B already trained & in union | **NO** — high q4b-style PEM-loss risk; redundant Gemma diversity |
| 26B or 31B trained (one of them) | **NO** — that one covers the Gemma angle; spend the slot on more InternVL / a non-Gemma non-Qwen family |
| **BOTH** 26B and 31B failed to train | **MAYBE** — 12B as the *sole* Gemma member, accepting ~2-3h isolation cost; still ranks below "another InternVL/MiniCPM family" |
| Phase-2 deadline imminent | **NO** — protect s01/s15 baselines; don't burn a slot on a speculative correlated member |

---

## 7. DO THIS / DON'T BOTHER

**DON'T BOTHER (default):** Do not invest the ~2-3h isolated-venv adaptation while 26B/31B are running or already in the union. 12B is a 3rd correlated Gemma member with a *weaker, encoder-free* visual pathway — the q4b failure mode (FP minority preds, PEM −1.2) is the likely outcome. Better uses of a free H200 + GPU-hours, in priority order:
1. Land the 26B / 31B members (stronger Gemma vision, already queued).
2. More **non-Qwen non-Gemma** diversity (additional InternVL variants, MiniCPM-V, multi-seed k-fold bagging) — proven +90 net minority lever.
3. Corrector pipeline (V-A/V-B) to fix long-segment residual 1-2-sentence errors → flips whole-paragraph PEM.
4. Phase-2 prep; keep s01 (47.80) and s15 (50.26) as the protected baselines.

**ONLY DO THIS IF** ALL of: (a) **both** 26B and 31B failed to train, AND (b) there is a free H200 with nothing higher-priority queued, AND (c) Phase-1 deadline is not imminent. Then follow §3 (isolated `.venv-gemma12b`) → §4 (8-step smoke, must pass all six criteria) → §5 (full train + union). The adaptation itself is **feasible and low-risk to the pinned env** (zero, via isolation) — the reason to skip it is **expected fleet value**, not difficulty.
