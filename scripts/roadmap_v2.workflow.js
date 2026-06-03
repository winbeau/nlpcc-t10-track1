export const meta = {
  name: 'roadmap-v2-recall',
  description: 'Re-analysis after the submission sweep REFUTED the precision framing: testp1 is recall-critical. Plan recall-maximization + reconsider 32B.',
  phases: [
    { title: 'Analyze', detail: '4 angles on the CORRECTED (recall-critical) problem: re-diagnosis, recall-maximization, 32B-reconsidered, over-correction risk' },
    { title: 'Critique', detail: 'adversarial: will "predict more minority" backfire like full-res? 32B memory; is recall framing right' },
    { title: 'Synthesize', detail: 'opus -> corrected roadmap + concrete experiments + 32B verdict' },
  ],
}

const CONTEXT = `
TASK: NLPCC-2026 Task10 Track1 — per-SENTENCE faithfulness of scientific claims vs figure/table evidence.
5 labels: Supported, Unsupported Causal Mechanistic (UCM), Unsupported Entity (UE), Scope Overgeneralization (SO), Contradiction.
METRIC: score = (Sentence Macro-F1 + Paragraph Exact Match PEM)/2. PEM = whole paragraph all-sentences-correct.
METHOD: Qwen3-VL-8B-Instruct + LoRA (r16/a32, freeze ViT), bf16, ms-swift 4.2.3. Per-SENTENCE modeling: sample
= shared (system + label defs + evidence images/captions + claim) prefix + ONE target sentence -> {"label":X}.
Custom softmin-PEM loss (L=(1-λ)·meanCE + λ·softmin_β(per-sentence CE)). Imbalance: oversample minority Kx +
downsample Supported records. 9:1 by-record split. use_logits_to_keep validated loss-neutral.

*** CRITICAL — A PRIOR ANALYSIS WAS WRONG; THIS IS THE CORRECTED, MEASURED PICTURE. Build ONLY on the
measured submission sweep below; do NOT re-import the refuted "precision-critical / suppress-minority" frame. ***

MEASURED testp1 SUBMISSION SWEEP (official Codabench), sorted by #minority sentence predictions:
  #minority | submission                       | Score | MacroF1 | PEM
     46     | veto_all (revert all lone, 99% Supported) | 18.85 | 22.01 | 15.70
    127     | tau-0.03 (confidence-threshold)  | 38.48 | 42.48 | 34.47
    171     | veto_gated                       | 39.57 | 44.34 | 34.81
    190     | tau-0.20                         | 43.85 | 48.29 | 39.42
    246     | B = gentle 1.5x oversample retrain | 40.91 | 46.84 | 34.98
    253     | grid = 3.0x oversample (BEST)    | 47.80 | 52.77 | 42.83
    260     | full-res (784 tok/img)           | 44.47 | 48.99 | 39.93
WHAT THIS PROVES:
  1. Score is ~MONOTONIC increasing in (correct) minority predictions; EVERY suppression (veto/threshold) HURTS.
  2. testp1 trivial all-Supported ≈ 19 (read off veto_all=18.85), NOT the train-dev 43.2/PEM-67. testp1 paragraphs
     are LONG (mean 8.7, max 31 sentences) and MINORITY-RICH (~84% of paragraphs contain >=1 gold minority, inferred
     from trivial PEM ~16). => testp1 is RECALL-CRITICAL for minority, the OPPOSITE of precision-critical. To get a
     paragraph's PEM you must CATCH its minority sentence(s); all-Supported loses ~84% of paragraphs.
  3. The grid's 253 minority preds are mostly TRUE POSITIVES and are the PEM lifeline. Removing them regresses to ~19.
  4. B (1.5x oversample) = 40.9 < grid (3x) = 47.8 ⇒ LESS oversampling is WORSE ⇒ push oversampling HIGHER, not lower.
  5. full-res (260, MORE minority) = 44.5 < grid (253) = 47.8 ⇒ it's minority QUALITY/PLACEMENT, not raw count;
     full-res changed ~94/5096 labels into WRONG minority. So "predict more" only helps if the extra preds are CORRECT.
DEV CAVEAT: the 9:1 in-distribution dev = 88 (minority-sparse, short paragraphs) badly misrepresents testp1
(minority-rich, long). dev is a poor model-selection signal; the Codabench leaderboard is the real signal.
MEMORY FACTS (1x H200 = 143.7 GiB; SHARED 8-GPU box, often only 1-2 free):
  - Per-SENTENCE modeling re-encodes the shared image prefix B× per paragraph (B up to 31 on testp1 worst -> 57
    image-forwards). This ACTIVATION cost dominates and ZeRO does NOT shard activations. 8B@392tok worst-para ~55 GiB
    activation; full res OOMs. 32B-dense per-sentence worst-para ~200 GiB => does NOT fit 1 H200, multi-GPU ZeRO-3
    still OOMs on activations.
  - PARAGRAPH-JOINT reformat (one forward per paragraph -> JSON array of N labels) replaces B re-encodes with ONE
    ~4k-token forward => 32B fits 1 H200 at ~85-90 GiB, no ZeRO/offload. Pure data-format change (native ms-swift SFT).
  - use_logits_to_keep killed the LM-head logits hog. Qwen3-VL-32B weights bf16 ≈ 60 GiB.
CONSTRAINTS: LoRA + ms-swift only; remote SHARED GPUs; CANNOT tune on hidden testp1 gold (Codabench cap = 100
submissions, 10 used, generous); Phase-2 hidden test 2026-06-11, all due 2026-06-20. Higher image resolution is
PROVEN HARMFUL (-3.3); do not pursue. Confidence thresholding is weak (model overconfident, logprobs in [-0.01,0]).
`

const ANALYSIS = {
  type: 'object', additionalProperties: false,
  properties: {
    angle: { type: 'string' },
    key_findings: { type: 'array', items: { type: 'string' } },
    proposals: { type: 'array', items: {
      type: 'object', additionalProperties: false,
      properties: {
        name: { type: 'string' }, what: { type: 'string' },
        targets: { type: 'string', enum: ['minority-recall', 'minority-precision/quality', 'capacity', 'memory/systems', 'validation', 'other'] },
        expected_effect: { type: 'string' }, effort: { type: 'string', enum: ['low', 'medium', 'high'] },
        risk: { type: 'string' }, memory_notes: { type: 'string' },
      },
      required: ['name', 'what', 'targets', 'expected_effect', 'effort', 'risk'],
    } },
  },
  required: ['angle', 'key_findings', 'proposals'],
}
const CRITIQUE = {
  type: 'object', additionalProperties: false,
  properties: {
    assessments: { type: 'array', items: {
      type: 'object', additionalProperties: false,
      properties: { proposal: { type: 'string' }, verdict: { type: 'string', enum: ['sound', 'overoptimistic', 'flawed', 'wrong'] }, reason: { type: 'string' }, correction: { type: 'string' } },
      required: ['proposal', 'verdict', 'reason'],
    } },
    missing: { type: 'array', items: { type: 'string' } },
  },
  required: ['assessments', 'missing'],
}
const PLAN = {
  type: 'object', additionalProperties: false,
  properties: {
    corrected_diagnosis: { type: 'string' },
    ranked_roadmap: { type: 'array', items: {
      type: 'object', additionalProperties: false,
      properties: { rank: { type: 'number' }, approach: { type: 'string' }, why: { type: 'string' }, expected_effect: { type: 'string' }, effort: { type: 'string' }, risk: { type: 'string' }, next_step: { type: 'string' } },
      required: ['rank', 'approach', 'why', 'next_step'],
    } },
    larger_model_verdict: { type: 'array', items: {
      type: 'object', additionalProperties: false,
      properties: { model: { type: 'string' }, memory_estimate: { type: 'string' }, worth_it: { type: 'string' }, how: { type: 'string' } },
      required: ['model', 'memory_estimate', 'worth_it', 'how'],
    } },
    top_recommendation: { type: 'string' },
    pitfalls: { type: 'array', items: { type: 'string' } },
  },
  required: ['corrected_diagnosis', 'ranked_roadmap', 'larger_model_verdict', 'top_recommendation'],
}

log('Roadmap v2 (recall-critical): re-diagnosis + recall-max + 32B-reconsidered -> critique -> synthesis')
phase('Analyze')
const angles = [
  { label: 're-diagnosis', model: 'sonnet', task:
    `ANGLE = CORRECTED DIAGNOSIS. From the MEASURED sweep, establish rigorously: (a) confirm testp1 trivial≈19 and ` +
    `estimate the gold minority-per-paragraph rate + the share of paragraphs with >=1 minority; (b) the PEM CEILING ` +
    `on long paragraphs (mean 8.7, max 31): what per-sentence accuracy p is needed for a given PEM, and what does ` +
    `47.8/PEM 42.8 imply about current per-sentence minority recall/precision; (c) WHY full-res(260) and B-1.5x(246) ` +
    `scored BELOW grid(253) despite similar/more minority counts — i.e. what distinguishes GOOD minority recall ` +
    `(grid) from BAD (full-res) — quality/placement, per-class, or calibration; (d) realistic achievable score ceiling.` },
  { label: 'recall-max', model: 'sonnet', task:
    `ANGLE = MAXIMIZE CORRECT MINORITY RECALL+PRECISION (the proven lever). Enumerate concrete levers and rank: ` +
    `oversample UP (3x->4x/5x, or per-class to the rarest), MORE epochs (only 1 trained — minority may be underfit), ` +
    `recall-favoring loss (class-weighted CE, focal-gamma, lower softmin emphasis), per-class targeting (which of ` +
    `UCM/UE/SO/Contradiction is under-recalled — use diagnostic submissions), ENSEMBLE-UNION for recall (fire ` +
    `minority if ANY model does), caption/table-OCR text to boost entity/number detection, multi-label target ` +
    `choice. For EACH: effect on MacroF1 AND PEM (both rise with correct recall on this minority-rich set), effort, ` +
    `risk of degrading into full-res-style WRONG minority. Which single experiment most likely beats 47.8.` },
  { label: '32b-reconsidered', model: 'opus', task:
    `ANGLE = LARGER MODEL, RE-EVALUATED under RECALL-critical. The prior verdict ("32B not worth it") assumed ` +
    `precision-critical and is now suspect: if the lever is CATCHING more correct minority, more capacity/reasoning ` +
    `may genuinely help (32B is capacity, NOT resolution — full-res's harm does NOT transfer). Decide if 32B is now ` +
    `WORTH it. Give the memory plan: per-sentence 32B = ~200 GiB worst-para (infeasible 1 H200, ZeRO can't shard ` +
    `activations); PARAGRAPH-JOINT 32B = ~85-90 GiB (fits 1 H200, format change). So the 32B path REQUIRES the ` +
    `paragraph-joint reformat. Assess: does paragraph-joint (model sees ALL sentences before labeling) HELP minority ` +
    `recall (cross-sentence coherence) — important since long minority-rich paragraphs are the hard case? Give a ` +
    `concrete runnable recipe (reformat -> 8B paragraph-joint sanity -> 32B), and an honest worth-it verdict + risk.` },
  { label: 'over-correction-risk', model: 'sonnet', task:
    `ANGLE = GUARD AGAINST OVER-CORRECTION + VALIDATION. We just flipped from "suppress minority" to "predict more ` +
    `minority". But full-res predicted MORE minority and LOST (wrong ones). So naive "fire more" can backfire. ` +
    `Analyze: what is the precision FLOOR — at what point do extra minority preds become net-negative again ` +
    `(the curve must turn over somewhere above 253)? Is there an interior optimum we're already near? How to get ` +
    `CORRECT extra recall without full-res-style errors. Also DESIGN a testp1-like offline validation (minority-rich, ` +
    `LONG paragraphs — e.g. resample/concatenate dev paragraphs to match testp1 length+minority stats, or split by ` +
    `paper) so we stop flying blind on the leaderboard. Propose the 1-2 safeguards/validation builds most worth doing.` },
]
const analyses = (await parallel(angles.map(a => () =>
  agent(`${CONTEXT}\n\n${a.task}\n\nReturn structured analysis. Be concrete and quantitative; tie every proposal to MacroF1 AND PEM on the RECALL-critical testp1.`,
    { label: a.label, phase: 'Analyze', model: a.model, schema: ANALYSIS })
))).filter(Boolean)

phase('Critique')
const dossier = JSON.stringify(analyses, null, 1)
const critiques = (await parallel([
  () => agent(
    `${CONTEXT}\n\nFour analysts proposed:\n${dossier}\n\nYou are a SKEPTICAL reviewer. Default to REFUTING. The team ` +
    `just inverted from "suppress" to "predict more minority" — pressure-test that. For each recall proposal: will it ` +
    `produce CORRECT minority or repeat full-res's WRONG-minority regression? Is the monotonic-in-minority reading ` +
    `an artifact (the off-trend B/full-res points prove count != quality)? Could oversample-up overfit the ~1101 ` +
    `train-dev hard paragraphs and generalize worse? Verdict each. List what's MISSING.`,
    { label: 'critic-strategy', phase: 'Critique', model: 'sonnet', schema: CRITIQUE }),
  () => agent(
    `${CONTEXT}\n\nFour analysts proposed:\n${dossier}\n\nYou are a SKEPTICAL SYSTEMS/MEMORY + experiment-design ` +
    `reviewer. Audit the 32B memory arithmetic and the paragraph-joint claim (recompute; flag OOM/scheduling issues ` +
    `on the shared box before the 2026-06-20 deadline). Check the offline-validation design for leakage/representativeness. ` +
    `Is the per-sentence->paragraph-joint reformat correctly costed, and does it risk changing results for reasons ` +
    `other than capacity (confound)? Verdict each systems/validation proposal. List what's MISSING.`,
    { label: 'critic-systems', phase: 'Critique', model: 'opus', schema: CRITIQUE }),
])).filter(Boolean)

phase('Synthesize')
const plan = await agent(
  `${CONTEXT}\n\nFOUR ANALYSES:\n${dossier}\n\nTWO ADVERSARIAL CRITIQUES:\n${JSON.stringify(critiques, null, 1)}\n\n` +
  `You are the lead. Integrate into ONE decision-ready plan for the RECALL-critical testp1. Keep only proposals that ` +
  `survive critique; fix EV/memory claims. ranked_roadmap ordered by (expected SCORE gain over 47.8)/(effort+risk), ` +
  `each with a concrete next_step runnable on ms-swift + shared H200 and an honest effect on BOTH MacroF1 and PEM. ` +
  `larger_model_verdict: per model, explicit memory arithmetic, whether it fits 1 H200 and how (paragraph-joint), and ` +
  `a worth-it verdict NOW that the task is recall-critical. top_recommendation = the single highest-EV next move. ` +
  `pitfalls = the over-correction/validation traps to avoid. Be decisive; the deadline is 2026-06-20.`,
  { label: 'synthesis', phase: 'Synthesize', model: 'opus', schema: PLAN })

return plan
