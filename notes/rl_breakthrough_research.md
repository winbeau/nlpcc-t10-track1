# RL Breakthrough Research — GRPO to break the single-model 47.80 (s01) ceiling on testp1

> Status: **CONDITIONAL GO, gated on a cheap pre-check + 4 hard data/code prerequisites.**
> Date: 2026-06-11. Supersedes the earlier E1→GRPO framing in `notes/grpo_plan.md`.
> Floors are inviolate: **s01 = 47.80 (single) and s15 = 50.26 (union) are permanent submissions; no GRPO output ever overwrites them.**

---

## TL;DR verdict

- **P(single-model testp1 > 47.80) ≈ 10–15%.** (The plan's earlier 22% was optimistic; the red-team's 8–12% is the floor. I land at ~12%.)
- **P(produces a useful heterogeneous union member that pushes s15 > 50.26) ≈ 15–20%.** This is the more likely win path (looser bar: only needs *new correct minority recall* with a different error mode).
- **Combined P(any win) ≈ 20%.**
- **The single decisive number is dense `pass@8`** (does the E0-full warm-start ever fire a correct *second* minority in a long paragraph). By the project's own statistics (0.46% of long traindev paragraphs and 3.8% of s01 testp1 paragraphs ever fire ≥2 minority) this is **likely near 0**, and "RL only amplifies, never creates" (arXiv 2504.13837) means a near-0 pre-check = a no-op route. **L0 below converts this dominant risk into a cheap, falsifiable GO/NO-GO before any GPU-hours.**

**This plan is launchable only after FOUR verified prerequisites (P0–P3) are met.** The red-team found that the prior plan's load-bearing inputs do not exist on disk. All confirmed against the repo:

| # | Prerequisite (BLOCKER) | Verified problem | Fix |
|---|---|---|---|
| **P0** | Density-matched data | `train_b1_joint.jsonl` **does not exist** (only `build_density_matched.py`). The "pseudo-prior fix" and the dense pass@8 set both depend on it. | **Build it on the server first; verify `n_synth > 0` and a healthy `≥2-minority/para` histogram before any downstream step references it.** |
| **P1** | E0-full warm-start | **No clean full-traindev label-only checkpoint exists.** s01 is *softmin* (label-shaped target, but softmin-PEM **loss**, full-traindev). s40 is label-only but **train'-only = testp1 43.85** (cross-regime — forbidden by RL constraints). | **Resolve definitively (§3). Default plan = train a 3h E0-full SFT (s01 recipe, label-only CE target, full-traindev). Do NOT assume s01's adapter == E0-full.** |
| **P2** | Multi-label gold | `build_grpo_data.py` emits a **1-element** `gold_types` (single `pick_rarest_label`). The *reward fn* `compute_reward` correctly does any-hit, but it is fed single-element gold → it cannot credit alternate-valid labels on the 527 multi-label sentences. | **Fix the data builder to emit the FULL gold `types[]` list per sentence** (the reward already supports it — selftest line 177). Real bug, not a knob. |
| **P3** | Prompt source = label-only | `build_grpo_data.py` defaults to `--in data/cot/e1/train_sft.jsonl` (the **CoT** set) and rollout format is `<analysis>…</analysis>\n{label}` → re-enters the **falsified CoT regime** (E1 = −5.31). | **Source label-only prompts from the E0-full SFT set; rollout format = plain `{"label": L}` (no `<analysis>` channel).** |

I **accept all six red-team required_fixes**. The one place I diverge is documented in §1 (R_fp is removed from run #1, not merely "made inert" — I agree it is a disguised suppression term).

---

## 1. Reward function — REVISED (R_fp REMOVED from run #1; PEM pressure comes from advantage normalization, not a veto term)

**Run #1 reward = the existing, unit-tested class-weighted ORM ALONE, with the multi-label fix (P2).** Per rollout (one sentence; completion = plain `{"label": L}`, NO analysis channel):

```
R(i) =
   -1                if parse-fail OR illegal label                       # format guard (anti-gaming)
    w_class(L)        if L ∈ gold_types(i)   (FULL gold list, any-hit)     # matches official scorer
    0                 if L valid but ∉ gold_types(i)

w_class:  Supported = 1.0,  UCM = 5.0,  Contra = 4.5,  UE = 4.0,  SO = 3.5
```

**Why R_fp is REJECTED for run #1 (accepting the red-team).** The prior plan's "one safe extra" R_fp = −δ for any minority label on a sentence whose paragraph gold is purely Supported is a **disguised re-introduction of the falsified suppression class** (veto / threshold / ≥2-vote, all monotonically harmful on testp1). Its safety depends on a `para_is_clean` signal that the **per-sentence pipeline cannot compute reliably** — per-sentence rows carry only that sentence's gold, and it is exactly the multi-label-fragile signal that is wrong on the 527 multi-label sentences. A slightly-wrong `para_is_clean` punishes *correct* isolated minority calls = veto-harm. Shipping an untested veto-shaped term on the first falsifiable data point contradicts the project's own discipline. **Removed.**

**How PEM precision pressure enters WITHOUT a suppression term — the load-bearing replacement for the missing softmin loss.** The RL constraints state softmin's PEM-bottleneck **loss** is load-bearing precisely to control oversampling-induced FPs, and GRPO discards that loss. A naked class-weighted ORM has no term that knows clean paragraphs exist → it will over-fire minority exactly like s33 (plain-CE+oversample, PEM −5.8). The replacement is **per-paragraph advantage normalization built into run #1** (not deferred to a ladder rung):

- Rollouts for all sentences of one paragraph are grouped; the GRPO advantage is centered **per paragraph** (mean over the paragraph's rollouts), not per single sentence. A rollout that opens a minority on a sentence whose paragraph is otherwise all-Supported gets a *below-paragraph-mean* advantage when that call is an FP (the other sentences' Supported rollouts pull the mean up and the wrong-minority's R=0 sits below it). This makes the PEM-shattering FP **structurally** disadvantaged without ever assigning it a negative reward or vetoing isolated correct calls.
- This requires paragraph-grouped rollout collation in ms-swift. **It must be validated in SMOKE (L1)**; if ms-swift's GRPO does not preserve paragraph grouping per device, fall back to (b) below.
- **(b) Conservative fallback if per-para grouping is infeasible:** down-weight the minority class weights to a far-less-aggressive band (UCM 2.5 / Contra 2.25 / UE 2.0 / SO 1.75). With R=0 (not negative) for a wrong minority, the over-fire incentive is `p_correct · w_minority > 1.0` (Supported). At w=5.0 the policy fires minority whenever `p_correct > 0.2` — far too eager (this is the s33 lever). At w≈2.0 the threshold rises to `p_correct > 0.5`, materially closer to honest. Run #1 uses the per-paragraph-norm path; the down-weighted band is the documented degradation.

**Class weights are NOT proxied by minority count, and are calibration-locked to the anti-laziness invariant** `E[R | all-Supported] < E[R | honest]` (the s05=18.85 anchor). They are NOT raised for a class with zero pass@8 fire-starters (wastes gradient — grounding). Implementation: P2 fix in `build_grpo_data.py` + extend `grpo_reward.py --selftest` with multi-label cases. No new reward term needed for run #1.

---

## 2. Algorithm + hyperparameters

**Algorithm: GRPO with Dr.GRPO advantage (mean-only centering, no std-division) + per-paragraph grouping (§1) + KL dropped.**

- **Dr.GRPO (`--scale_rewards false`, mean-only):** std-normalization NaNs when a minority group's 8 rollouts collapse to `{0, w_class}` (low/zero variance) and underweights the hard minority prompts. Mean-only centering is stability-positive for short outputs.
- **Drop KL (`--beta 0`):** KL-to-base suppresses the minority departures we want to amplify. Re-add a tiny `--beta 0.001` only if training diverges.
- **Static DAPO pass@8 pre-filter (L0) is mandatory; dynamic `--rlhf_type dapo` is a ladder-only escalation** and must be SMOKE-verified for Qwen3-VL in ms-swift 4.2.3 before relying on it.
- **Rejected (grounding-forbidden / inapplicable):** PPO (OOM + value-net waste on short outputs), DPO (degenerates to CE on short JSON — explicitly rejected), CoT-GRPO from E1 (−5.31, forbidden).

| Param | Value | Note |
|---|---|---|
| base model | Qwen3-VL-8B-Instruct (ModelScope ms_cache) | `USE_HF=0` |
| warm-start adapter | **E0-full** (§3) — full-traindev, label-only, softmin-grid recipe | NOT E1, NOT train'-only s40 |
| `--rlhf_type` | `grpo` (run #1) → `dapo` (ladder, verify in SMOKE) | |
| advantage | Dr.GRPO mean-only + **per-paragraph grouping** | `--scale_rewards false` |
| `--beta` (KL) | 0 | re-add 0.001 only if diverges |
| `--num_generations` | 8 | |
| `--temperature` | 1.1 (run #1) | curriculum is ladder-only |
| `--learning_rate` | 1e-6 | |
| `--num_train_epochs` | 1 | 2ep falsified (s10 overfit) |
| `--per_device_train_batch_size` | 1 | |
| `--gradient_accumulation_steps` | 8 | **≤8** (#6521) |
| `--vllm_enable_lora` | **false** | full-weight sync (#6670) |
| `--vllm_mode` | colocate | + `--offload_optimizer --offload_model --sleep_level 1` |
| `--max_pixels` | 401408 | raising falsified (−3.3) |
| `--max_completion_length` | 64 | label-only (no analysis) → shorter than the 160 in train_grpo.sh |
| `--lora_rank/alpha` | 16/32, `--freeze_vit true` | |
| prompt set | ~3–5k after pass@8 pre-filter, density-augmented ~20–30% | §4 |

---

## 3. Warm-start = E0-full — RESOLVE BEFORE TRAINING (P1)

**Forbidden starts:** (a) **E1/CoT** (falsified −5.31, RL amplifies a degraded model); (b) **train'-only E0 = s40 (43.85)** — cross-regime; the s30 lesson (densematch 83.90 → testp1 43.85) and s31/s32/s40 all confirm train'-trained models score 34–44 and densematch ranks don't transfer. **RL constraints mandate full-traindev + oversampling to stay in the valid regime.**

**The ambiguity (red-team is right):** s01's checkpoint (`outputs/grid_b5_l0.5/v0-20260602-095341/checkpoint-2000`) has a label-shaped target `{"label": X}` BUT was trained with the **softmin-PEM loss**, not plain label-only CE. There is no evidence its adapter behaves as a clean label-only policy, and "E0-full" was never actually built (s40 is train'-only).

**Resolution protocol (do on the server, in order):**
1. Greedy-decode s01's adapter on `data/devbench/dev_gold_densematch.jsonl`. Record per-class minority recall + minority count. This is the candidate-E0-full health check.
2. **Decision:**
   - If s01's adapter reproduces ~s01-level minority recall under plain greedy decode AND its target format is label-only `{"label": X}` (verify the training-data target string) → it is an acceptable E0-full surrogate; start GRPO from it. (Softmin shaped the *loss*, not the saved weights' decoding behavior; a healthy greedy decode is the operational test.)
   - **Otherwise (default assumption): a 3h E0-full SFT is a HARD PREREQUISITE.** Recipe = s01 recipe exactly (softmin grid os3.0/ds0.6, full-traindev, 1 epoch, max_pixels 401408) BUT label-only CE target (`{"label": L}`, no analysis). This is the missing clean warm-start. Name it a new tag (e.g. `E0full`), never overwrite s01.

**Do not take the prior plan's "no SFT needed, start from s01" shortcut on faith** — it risks the falsified cross-regime / non-equivalent-checkpoint zone.

---

## 4. Rollout / sampling + the pseudo-prior fix (depends on P0)

- `--num_generations 8`, temperature 1.1 for run #1. Temperature curriculum (1.3→0.9) is ladder-only (needs mid-run vLLM temp updates).
- **Static DAPO pass@8 pre-filter:** keep only prompts with `pass@8 ∈ (0,1)`. Discards all-correct (advantage 0) and all-wrong (no fire-starter). Stratify to ≥150 prompts/minority class; up-sample rare classes with replacement ≤3×.
- **THE PSEUDO-PRIOR FIX — density-augmented prompt distribution (requires P0 built first):**
  - **P0:** run `build_density_matched.py` on the server → `data/devbench/train_b1_joint.jsonl`. **Verify `n_synth > 0` and that the `synth minority/para histogram` is genuinely ≥2-dense.** If `n_synth` is tiny (the co-evidence/same-SHA constraint may starve it), this whole mechanism is weak and the route degrades to label-only-sparse GRPO (reinforcing the prior it claims to break) — **log that and lower P(success) accordingly.**
  - Add `--density-augment` to `build_grpo_data.py`: read `train_b1_joint.jsonl`, expand each ≥2-minority synthetic *joint* paragraph back to per-sentence rows (carry the multi-figure image list + true per-sentence FULL gold types[]), and **append** so dense-context prompts are ~20–30% of the final pool (additive, not replacing).
  - **Mechanism:** sparse single-minority prompts can only reinforce "fire ≤1 minority/paragraph." A model rolling out a dense synthetic paragraph occasionally fires correctly on *both* minority sentences → those rollouts get `w_class` on both → large positive (mean-centered, per-paragraph) advantage → teaches "a paragraph can hold ≥2 minorities." This is the only path to *new correct recall* that union members structurally cannot produce.
  - **Transfer caveat (red-team, accepted):** synthetic paragraphs are **spliced** (concatenated train' records sharing image-SHA). The generator itself flags cross-record SO leakage as the residual risk. Real testp1 paragraphs are genuine multi-sentence claims, not concatenations — the "≥2 minorities" lesson risks being tied to splice-boundary / multi-figure cues absent in testp1. **G3 (densematch) is the arbiter; do not assume transfer.**

---

## 5. PEM credit-assignment: per-sentence rollouts + per-paragraph advantage normalization (run #1)

**Choice: per-sentence rollouts, but advantage centered per paragraph (§1).** Justification:
- The s01 line is per-sentence; joint single-models underperformed (s12=43.33, s13=44.43 < s01).
- Joint GRPO rollouts (~8.7 sentences × image copies × 8 gen) are ~70× the VRAM/token cost; high OOM risk at MAX_PIXELS=401408 on 2×H200, and ms-swift variable-N joint GRPO is untested.
- **Per-paragraph advantage normalization is the cheap PEM-aware term that replaces the discarded softmin loss** — it is in run #1, NOT deferred. It is the minimal encoding of "a clean paragraph exists" that does not veto isolated correct minority calls.

**Joint-paragraph rollout with decomposed reward `αMF1 + (1−α)PEM`, α=0.3, N≤6** is the highest-ceiling, highest-risk ladder option (L6) — only if per-sentence + per-para-norm both stall, SMOKE proves ms-swift joint-GRPO collation works, and length is capped ≤6 to fit VRAM.

---

## 6. Pass@k pre-check — RUN THIS FIRST (L0, GO/NO-GO gate)

`infer.py` already supports `--temperature` + `--seed` → no new inference code; loop 8 seeds at temp 1.0 over the candidate prompt set. **Depends on P0 (dense set) + P1 (E0-full).**

1. Build candidate prompts two ways; report pass@8 stratified by **class × context-density**:
   - **Sparse set:** E0-full resampled per-sentence prompts (label-only).
   - **Dense set:** per-sentence prompts expanded from the *verified* `train_b1_joint.jsonl`.
2. 8 rollouts/prompt at temp 1.0; `pass@8 = fraction of rollouts whose label hits FULL gold types[]`.
3. Emit table: per minority class {UCM, Contra, UE, SO} × {sparse, dense}: % with `pass@8 ∈ (0,1)` and % with `pass@8 > 0`.

**GO/NO-GO (falsifiable, decide by 2026-06-14):**
- Every minority class dense `pass@8 > 0` rate **≥ 20%** → **GO**, full density-matched GRPO.
- A rare class (SO/UCM) `< 20%` dense but `≥ 20%` sparse → GO with lowered expectations for that class; do NOT inflate its w_class. Optional 1–2h targeted SFT to seed SO fire-starters (rename ckpt, never s01).
- A class `pass@8 = 0%` on BOTH sets → untrainable by RL (only amplifies); proceed for others, don't burn budget on the dead class.
- **`< 20%` dense on ≥3 of 4 minority classes → NO-GO / KILL** (accepting the red-team's pre-committed threshold). Log negative result, keep s01/s15, stop. **No marginal-class rationalization.**

This is the cheapest, highest-information step and gates everything.

---

## 7. Risk-ranked experiment ladder (cheapest-highest-EV first; each falsifiable)

| # | Step | Cost | Falsifiable claim / kill | EV |
|---|---|---|---|---|
| **P0–P3** | **Prerequisites:** build & verify `train_b1_joint.jsonl` (n_synth>0); resolve/SFT E0-full; fix multi-label gold in builder; switch source to label-only + drop `<analysis>` | ~3–4h (mostly the E0-full SFT) | "All four inputs exist and are in-regime." Any fails → fix before proceeding | BLOCKER — plan cannot start otherwise |
| **L0** | **pass@8 dense+sparse pre-check** (§6) + extend `grpo_reward.py` multi-label selftest | ~3–5h infer | "≥1 minority class dense pass@8 ≥20% (and ≥3/4 not <20%)." Else **KILL whole route** | Highest — single most informative number; cheap |
| **L1** | **SMOKE** `SMOKE=1` GRPO from E0-full, 8 steps/64 prompts: colocate vLLM + class-weighted reward + Dr.GRPO + **per-paragraph grouping** + density-augment | ~40min | "Runs without crash/NaN/OOM; per-paragraph advantage grouping actually works in ms-swift 4.2.3." Fails twice → fall back to down-weighted-w (§1b) or verl | High — prevents wasting 10–18h on a broken config |
| **L2** | **Full run #1:** per-sentence, density-augmented ~25%, Dr.GRPO + per-para-norm, temp 1.1, 1ep, class-weighted reward (no R_fp) | ~8–18h | "densematch > E0-full with balanced minority recall AND testp1-shaped held-out check ≥ E0-full (G3)." Fail → keep s01/s15 | Core bet |
| **L3** | If G3 passes → **exactly one testp1 submission** + union re-form | ~30min | "testp1 > 47.80 (single) OR new-member union > 50.26." | The payoff measurement |
| **L4** | If L2 stalls but pass@8 healthy: tune the **per-paragraph-norm strength / w-band** (§1b) | +ckpt re-run | "Adjusted PEM-pressure densematch > run #1." | Medium |
| **L5** | If still flat: **DAPO dynamic sampling + clip-higher** (verify VLM support in SMOKE) and/or temperature curriculum 1.3→0.9 | +run | "Dynamic resampling raises late-training minority recall vs static filter." | Medium |
| **L6** | If recall ceiling persists & VRAM allows: **joint-paragraph rollout, α=0.3 (MF1+PEM), N≤6** | high eng + VRAM | "Joint-PEM reward breaks the recall ceiling per-sentence cannot." | High ceiling, high risk — last resort |

Each rung is attempted only if the cheaper rung's falsifiable claim held but the score gate didn't clear.

---

## 8. G3 gate + kill-switches

**Eval-geometry note:** densematch (trivial 17.2 ≈ testp1 18.85) is the **only** offline proxy mirroring the recall-critical / PEM-AND regime, reliable for **same-regime, same-condition ranking only** (Spearman +0.59). It already false-positived s30 (densematch 83.90 → testp1 43.85). A GRPO model vs E0-full is NOT strictly same-condition (different objective, density-augmented prompts), so +0.59 may not fully apply. **Use it as a ranking/health gate, never an absolute predictor.**

**G3 GATE (all must pass before the single testp1 submission):**
1. **densematch (greedy) > E0-full's densematch** by a clear margin — necessary, not sufficient.
2. **An independent no-leakage testp1-shaped held-out check ≥ E0-full** (accepting the red-team: do not let densematch alone authorize confidence). If no such slice exists, treat densematch as ranking-only and weight the single testp1 submission as the sole real signal — submit with eyes open.
3. **Per-class minority recall balanced** — no class collapsed to ~0 (especially SO: E1 had 11 vs E0 41). Report all 4.
4. **Minority prediction count within the precision-safe band** — not wildly above the per-record-scaled analogue of s15's ~382/5096.
5. **20-rollout manual spot-check** — labels track evidence; no degeneracy.

Pass → fire **exactly one** testp1 submission, next `s{NN}`, log it. If testp1 > 47.80 → new single best. Regardless, add as a **heterogeneous union member** to re-form s15 (GRPO error modes differ from CE/softmin — the new-recall source the naive union ceiling needs).

**KILL-SWITCHES (any one → stop, keep s01/s15, log negative):**
- **K-prereq:** P0 `n_synth` ≈ 0, or no in-regime E0-full obtainable → stop before L0.
- **K0 (pre-train):** ≥3/4 minority classes dense `pass@8 < 20%` → no fire-starters → KILL before training.
- **K1 (reward gaming):** reward spikes while format-failure / minority-FP rate rises → stop, inspect.
- **K2 (entropy collapse):** 8 rollouts identical / rollout minority fire-rate → 0 → raise temp / add entropy bonus (ladder); persists → stop.
- **K3 (minority explosion):** rollout minority fire-rate explodes and densematch PEM drops → per-para-norm too weak / w too high; tune (§1b); no recovery → stop.
- **K4 (G3 fail):** densematch ≤ E0-full OR a class collapsed OR held-out check fails → do NOT submit; keep E0/s01.
- **K5 (time/compute):** full-weight sync slow (~600–800 syncs); >36h without convergence, OR **Phase-2 baseline inference needs the GPUs (Phase-2 > GRPO priority)** → stop. **Hard stop/go deadline 2026-06-17** to protect the 2026-06-18+ submission window.

---

## 9. Honest "why this might fail" — the dominant failure mode

**The single strongest reason it fails: the pseudo-prior is near-deterministic, so dense `pass@8` is likely ≈ 0%, and RL only amplifies — it cannot create.** s01 fires ≥2 minority in only 3.8% of testp1 paragraphs; traindev has 8/3333 records with ≥2 minority sentences (0.46% of long paragraphs). If E0-full essentially never samples a correct *second* minority call in a dense paragraph, the density-augmented prompts yield only zero-advantage rollouts and GRPO is a no-op on exactly the behavior we need. **L0 exists to convert this dominant risk into a cheap GO/NO-GO and let us KILL honestly rather than burn the 6-17 deadline.**

**Secondary failure modes (all logged, none silent):**
- **P0 starvation:** the same-SHA co-evidence constraint may produce too few synthetic dense paragraphs; the density mechanism then can't fire and the route degrades to sparse label-only GRPO = reinforcing the prior.
- **Splice non-transfer:** even if dense pass@8 is healthy, the "≥2 minorities" behavior may be tied to splice-boundary / multi-figure artifacts absent in genuine testp1 paragraphs (densematch G3 catches direction but cannot prove testp1 transfer — s30 precedent).
- **Class-weighted-ORM over-fire without softmin's loss:** if per-paragraph advantage normalization is infeasible in ms-swift and the down-weighted-w fallback is too weak, the policy over-fires minority (s33 PEM-crash analogue). G3 + K3 catch it.
- **densematch false-positive:** the proxy authorized s30 (+40 illusory points). The single testp1 submission is the only real signal; G3's held-out check is the guard.
- **Phase-2 untested:** all calibration is testp1-derived (382 threshold, 3.8%, densematch). Phase-2 (released today, due 6-20) may shift; reward shaping is deliberately kept minimal (no testp1-tuned δ/veto) to limit overfit, but there is no Phase-2 validation point before the deadline.

**Floors are safe regardless:** s01 (47.80) and s15 (50.26) are never modified. Worst case = a logged negative result with the existing best scores intact.

---

## Relevant files (all absolute)

- `/home/winbeau/wenbiao_zhao/nlpcc-t10-track1/scripts/build_density_matched.py` — **P0**: run on server → `data/devbench/train_b1_joint.jsonl`; verify `n_synth>0`.
- `/home/winbeau/wenbiao_zhao/nlpcc-t10-track1/scripts/build_grpo_data.py` — **P2+P3**: emit FULL gold `types[]` (not 1-element `pick_rarest`); source label-only E0 set (NOT `data/cot/e1/`); add `--density-augment`; emit per-sentence dense rows.
- `/home/winbeau/wenbiao_zhao/nlpcc-t10-track1/scripts/grpo_reward.py` — reward `compute_reward` already does any-hit correctly (selftest L177); extend selftest with multi-label cases; **R_fp NOT added for run #1**.
- `/home/winbeau/wenbiao_zhao/nlpcc-t10-track1/scripts/train_grpo.sh` — set `ADAPTER=`E0-full (NOT E1); `--beta 0`; `--scale_rewards false` (Dr.GRPO); per-paragraph advantage grouping; temp 1.1; `--max_completion_length 64` (label-only).
- `/home/winbeau/wenbiao_zhao/nlpcc-t10-track1/src/nlpcc_t10/infer.py` — already has `--temperature`/`--seed` for the L0 pass@8 loop (no new code).
- `/home/winbeau/wenbiao_zhao/nlpcc-t10-track1/data/devbench/dev_gold_densematch.jsonl` — the G3 ranking proxy (exists).
- `/home/winbeau/wenbiao_zhao/nlpcc-t10-track1/outputs/grid_b5_l0.5/v0-20260602-095341/checkpoint-2000` — s01 ckpt (E0-full *candidate* — verify per §3; default = build a clean label-only E0-full SFT).
- `/home/winbeau/wenbiao_zhao/nlpcc-t10-track1/notes/grpo_plan.md` — update from E1→E0-full framing.
