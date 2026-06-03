export const meta = {
  name: 'roadmap-analysis',
  description: 'Deep analysis of the two testp1 submissions + ranked future plan + larger-model memory plan',
  phases: [
    { title: 'Analyze', detail: '4 independent expert angles: diagnosis, 8B precision strategies, larger-model+memory, contrarian reframe' },
    { title: 'Critique', detail: 'adversarial check of every proposal — memory arithmetic + strategy EV' },
    { title: 'Synthesize', detail: 'opus integrates into a ranked roadmap + larger-model memory table' },
  ],
}

// ── Shared context (agents do NOT see the chat; everything they need is here) ──────────────
const CONTEXT = `
TASK: NLPCC-2026 Task10 Track1 — per-SENTENCE faithfulness of scientific claims vs figure/table evidence.
5 labels: Supported, Unsupported Causal Mechanistic (UCM), Unsupported Entity (UE), Scope Overgeneralization (SO), Contradiction.
METRIC: score = (Sentence Macro-F1 + Paragraph Exact Match PEM)/2. PEM = whole paragraph all-sentences-correct.
Trivial all-Supported ≈ 43.2 (MacroF1 19.3, PEM ~67). => PEM-DOMINATED & PRECISION-CRITICAL: a single minority
false-positive on an all-Supported paragraph flips its PEM 1->0 (huge cost).
DATA: 3333 train-dev records / 17547 sentences; 93.7% Supported, minority ~6.3% (natural TRAIN counts:
UE 434, UCM 249, SO 169, Contradiction 145). Evidence is ALL images/tables (no text); avg 1.2 imgs/record
(max 7); captions ~276 chars. testp1 = 586 records / 5096 sentences, HIDDEN gold.
METHOD: Qwen3-VL-8B-Instruct + LoRA (r16/a32, freeze ViT), bf16, ms-swift 4.2.3. Per-SENTENCE modeling: each
sample = shared (system + label defs + evidence images/captions + claim) prefix + ONE target sentence ->
output {"label":X}. Custom sample-level SOFTMIN-PEM loss: L=(1-λ)·meanCE + λ·softmin_β(per-sentence CE),
gradient concentrates on the worst (bottleneck) sentence per paragraph, to defend PEM. Imbalance: oversample
minority Kx + downsample Supported records. 9:1 by-record split. use_logits_to_keep=true validated loss-neutral.
RESULTS (official Codabench testp1):
  - Grid best b5_l0.5 (β5 λ0.5, 1 epoch, oversample 3.0x, max_pixels 401408 = 392 tok/img):
    testp1 SCORE 47.80 (MacroF1 52.77, PEM 42.83). Our 9:1 dev = 88.15. Beats trivial 43.2 and all official
    LLM baselines (best GPT-5.4 = 16.4; the baselines over-fire minority and crater PEM).
  - Full-res P0 (SAME but max_pixels 802816 = 784 tok/img = 2x resolution, + use_logits_to_keep):
    testp1 SCORE 44.47 (MacroF1 48.99, PEM 39.93). WORSE on all three.
KEY FINDINGS:
  1. Higher image resolution is HARMFUL. 2x res changed only 1.84% of labels (94/5096) but net -3.3 score;
     mainly +47 Supported->minority flips (esp +16 UE) that were FALSE POSITIVES -> broke clean paragraphs
     (PEM down) AND were wrong minority (MacroF1 down). More visual detail -> over-flags spurious entity/causal.
  2. Model is OVERCONFIDENT: predicted min-token logprob ~ -0.003..-0.01 (99%+ conf) even on minority preds
     that are wrong on OOD. => overfit; confidence-thresholding has LIMITED power vs high-confidence FPs.
  3. Big dev->testp1 gap (88 -> 44-48). Overfit to train-dev; the in-distribution 9:1 dev overestimates OOD.
  4. BOTTLENECK = PRECISION (minority false positives shattering clean paragraphs), NOT capacity/visual detail.
MEMORY FACTS (1x H200 = 143.7 GiB usable; SHARED 8-GPU box, typically only 1-2 GPUs free):
  - Per-SENTENCE modeling RE-ENCODES the shared image prefix B times per paragraph (B=#sentences, up to 22;
    worst paragraph = 17 sentences x 3 images). This ACTIVATION cost dominates; use_logits_to_keep does NOT
    reduce it (it only removed the LM-head logits cost).
  - 8B LoRA @ grid res (392 tok/img) + use_logits_to_keep: ~40-60 GiB peak. @ 784 tok/img worst paragraph
    ~125 GiB. Native/full res (images <2048 tok) OOMs >140 GiB on the worst paragraph.
  - use_logits_to_keep killed the LM-head logits hog (full-vocab CE over [B,T,V=151936] fp32 ~80-100 GiB on
    worst paragraph -> ~0.3 GiB). Loss-neutral (golden-tested).
  - Qwen3-VL-32B weights bf16 ≈ 64 GiB; DDP replicates full weights per GPU.
IN-FLIGHT (running now): A = post-hoc threshold (suppress low-confidence minority -> Supported) on the grid
47.8 model; B = gentler oversampling retrain (3.0x -> 1.5x) @ grid resolution. Both target precision/FP.
CONSTRAINTS: LoRA + ms-swift only; remote SHARED GPUs; CANNOT tune on hidden testp1 gold (only Codabench
leaderboard, multiple submissions allowed, latest counts); Phase-2 hidden test 2026-06-11, all due 2026-06-20.
`

// ── Schemas (minimal; only what the synthesizer needs downstream) ─────────────────────────
const ANALYSIS = {
  type: 'object', additionalProperties: false,
  properties: {
    angle: { type: 'string' },
    key_findings: { type: 'array', items: { type: 'string' } },
    proposals: { type: 'array', items: {
      type: 'object', additionalProperties: false,
      properties: {
        name: { type: 'string' },
        what: { type: 'string' },
        targets: { type: 'string', enum: ['precision/PEM', 'overfit/generalization', 'capacity', 'memory/systems', 'other'] },
        expected_effect: { type: 'string' },
        effort: { type: 'string', enum: ['low', 'medium', 'high'] },
        risk: { type: 'string' },
        memory_notes: { type: 'string' },
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
      properties: {
        proposal: { type: 'string' },
        verdict: { type: 'string', enum: ['sound', 'overoptimistic', 'flawed', 'wrong'] },
        reason: { type: 'string' },
        correction: { type: 'string' },
      },
      required: ['proposal', 'verdict', 'reason'],
    } },
    missing: { type: 'array', items: { type: 'string' } },
  },
  required: ['assessments', 'missing'],
}
const PLAN = {
  type: 'object', additionalProperties: false,
  properties: {
    diagnosis: { type: 'string' },
    ranked_roadmap: { type: 'array', items: {
      type: 'object', additionalProperties: false,
      properties: {
        rank: { type: 'number' }, approach: { type: 'string' }, why: { type: 'string' },
        expected_effect: { type: 'string' }, effort: { type: 'string' }, risk: { type: 'string' },
        next_step: { type: 'string' },
      },
      required: ['rank', 'approach', 'why', 'next_step'],
    } },
    larger_model_plan: { type: 'array', items: {
      type: 'object', additionalProperties: false,
      properties: {
        model: { type: 'string' }, memory_estimate: { type: 'string' }, fits: { type: 'string' },
        solution: { type: 'string' }, verdict: { type: 'string' },
      },
      required: ['model', 'memory_estimate', 'solution', 'verdict'],
    } },
    top_recommendation: { type: 'string' },
    open_questions: { type: 'array', items: { type: 'string' } },
  },
  required: ['diagnosis', 'ranked_roadmap', 'larger_model_plan', 'top_recommendation'],
}

// ── Phase 1: Analyze (4 independent angles, in parallel) ──────────────────────────────────
log('Roadmap analysis: 4 expert angles -> adversarial critique -> opus synthesis')
phase('Analyze')
const angles = [
  { label: 'diagnosis', model: 'sonnet', task:
    `ANGLE = ROOT-CAUSE DIAGNOSIS. Using ONLY the data below, explain rigorously WHY grid(392tok)=47.8 but ` +
    `full-res(784tok)=44.5, and WHY PEM is the ceiling. Quantify the precision/PEM mechanism (how few minority ` +
    `FPs it takes to lose the -2.9 PEM seen). State what the evidence does and does NOT support about the ` +
    `bottleneck. Be skeptical of easy stories. Your proposals = diagnostic experiments that would CONFIRM or ` +
    `REFUTE the precision-FP-vs-overfit hypotheses (e.g. all-Supported anchor, per-class ablation).` },
  { label: 'precision-8b', model: 'sonnet', task:
    `ANGLE = HOW TO BEAT 47.8 WITHOUT A BIGGER MODEL. Given bottleneck=minority false positives on OOD + an ` +
    `overconfident overfit model, enumerate concrete precision/generalization levers: oversample/downsample ` +
    `tuning, loss changes (β/λ, class-weighted CE, focal, label smoothing, calibration/temperature), abstention ` +
    `& paragraph-level consistency post-processing, regularization (epochs/LR/dropout/LoRA rank), training-data ` +
    `quality (multi-label target choice, caption/OCR text), and ENSEMBLE/agreement-required-minority. For EACH: ` +
    `expected_effect on PEM vs MacroF1, effort, risk. Rank. Note which beat the in-flight A(threshold)/B(1.5x).` },
  { label: 'larger-model-memory', model: 'opus', task:
    `ANGLE = LARGER MODELS + MEMORY. Plan scaling the base model (Qwen3-VL-32B primary; also consider 30B-class ` +
    `MoE, InternVL/other VLMs, and whether 8B->32B even helps GIVEN more capacity may worsen minority FPs). Do ` +
    `EXPLICIT memory arithmetic per option on 1x H200 (143.7 GiB), accounting for: bf16 weights (32B≈64GiB), ` +
    `LoRA optimizer/grad (small), and the PER-SENTENCE activation blow-up (shared image prefix re-encoded B× per ` +
    `paragraph; worst paragraph 17×3 imgs). State which configs OOM and the SOLUTIONS: DeepSpeed ZeRO-2/3 weight/` +
    `optimizer sharding across GPUs, CPU/NVMe offload, gradient checkpointing, resolution caps, capping/splitting ` +
    `big paragraphs, and the SHARED-PREFIX TRAINING idea (encode a paragraph's images ONCE, reuse for all B ` +
    `sentences) — estimate its memory saving and feasibility in ms-swift. Account for the SHARED box (often only ` +
    `1-2 of 8 H200 free). Give a concrete, runnable 32B recipe (GPUs, ZeRO stage, max_pixels, max paragraph B).` },
  { label: 'contrarian-reframe', model: 'sonnet', task:
    `ANGLE = CONTRARIAN / REFRAME. Challenge the whole setup. Is per-sentence + softmin + oversampling the right ` +
    `frame? Consider: (a) is chasing minority recall worth it when PEM dominates — would a near-trivial, ` +
    `high-precision model (predict minority ONLY when extremely sure) score higher? (b) paragraph-joint modeling ` +
    `with structured/constrained decoding vs per-sentence; (c) calibration/selective-prediction framing; ` +
    `(d) is the 9:1 in-distribution dev actively misleading model selection — design a better OOD-like validation ` +
    `(split by source paper, harder subsets, k-fold) since the leaderboard is the only real signal. Propose the ` +
    `1-2 reframes most likely to move the SCORE, with their risks.` },
]
const analyses = (await parallel(angles.map(a => () =>
  agent(`${CONTEXT}\n\n${a.task}\n\nReturn structured analysis. Be concrete and quantitative; tie every proposal to PEM vs MacroF1.`,
    { label: a.label, phase: 'Analyze', model: a.model, schema: ANALYSIS })
))).filter(Boolean)

// ── Phase 2: Critique (adversarial; needs ALL analyses -> barrier is justified) ───────────
phase('Critique')
const dossier = JSON.stringify(analyses, null, 1)
const critiques = (await parallel([
  () => agent(
    `${CONTEXT}\n\nYou are a SKEPTICAL SYSTEMS/MEMORY reviewer. Here are 4 analysts' proposals:\n${dossier}\n\n` +
    `Audit specifically the MEMORY ARITHMETIC and systems claims (ZeRO/offload/resolution/paragraph-B/shared-prefix, ` +
    `32B feasibility on shared H200). Recompute any number that looks off; flag configs that will actually OOM or ` +
    `that the shared box can't schedule. Verdict each memory/systems-relevant proposal. List what's MISSING ` +
    `(a cheaper path to the same memory, an unaccounted cost, a wrong assumption).`,
    { label: 'critic-memory', phase: 'Critique', model: 'opus', schema: CRITIQUE }),
  () => agent(
    `${CONTEXT}\n\nYou are a SKEPTICAL ML reviewer. Here are 4 analysts' proposals:\n${dossier}\n\n` +
    `Default to REFUTING. For each strategy proposal, is the expected PEM/MacroF1 effect justified, or wishful? ` +
    `Given the model is OVERCONFIDENT on OOD FPs, will confidence-thresholding/abstention actually work? Will a ` +
    `bigger model help or worsen minority FPs? Is any proposal likely to score BELOW the 47.8 grid or even below ` +
    `trivial 43.2? Verdict each. List MISSING ideas or unstated risks.`,
    { label: 'critic-strategy', phase: 'Critique', model: 'sonnet', schema: CRITIQUE }),
])).filter(Boolean)

// ── Phase 3: Synthesize (opus, integrates everything) ─────────────────────────────────────
phase('Synthesize')
const plan = await agent(
  `${CONTEXT}\n\nFOUR ANALYSES:\n${dossier}\n\nTWO ADVERSARIAL CRITIQUES:\n${JSON.stringify(critiques, null, 1)}\n\n` +
  `You are the lead. Integrate everything into ONE decision-ready plan. Keep ONLY proposals that survive the ` +
  `critiques; fix EV/memory claims the critics corrected. The ranked_roadmap must be ordered by (expected SCORE ` +
  `gain over 47.8) / (effort+risk), each with a concrete next_step a single engineer can run on ms-swift + shared ` +
  `H200, and an honest expected_effect on PEM vs MacroF1. larger_model_plan must give per-model EXPLICIT memory ` +
  `arithmetic, whether it fits 1x H200 (143.7 GiB) and how (ZeRO/offload/res-cap/shared-prefix), and a verdict on ` +
  `whether it's worth it GIVEN the precision bottleneck. top_recommendation = the single highest-EV next move ` +
  `(beyond the already-running A/threshold and B/1.5x-retrain). Be decisive and honest about what likely WON'T help.`,
  { label: 'synthesis', phase: 'Synthesize', model: 'opus', schema: PLAN })

return plan
