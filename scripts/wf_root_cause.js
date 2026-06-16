export const meta = {
  name: 'nlpcc-t10-root-cause',
  description: 'Read actual figures for 78 curated error/conflict cases and diagnose root causes of Track1 misclassifications',
  phases: [
    { title: 'Diagnose', detail: 'one vision agent per case reads figure+context, classifies root cause' },
    { title: 'Verify', detail: 'opus re-reads contested cases (gold-suspect / model-defensible)' },
    { title: 'Synthesize', detail: 'opus aggregates taxonomy + levers, tied to quantitative metrics' },
  ],
}

const LABELS = `Track1 labels (per-sentence faithfulness of a scientific claim to figure/table evidence):
- Supported (SUP): sentence is fully supported by the evidence figure/table.
- Unsupported Causal Mechanistic (UCM): introduces a causal/mechanistic explanation the evidence does not support.
- Unsupported Entity (UE): mentions a dataset / metric / model variant / baseline / entity NOT present in the evidence.
- Scope Overgeneralization (SO): extrapolates the conclusion beyond what the evidence supports (over-broad quantifier / condition / population).
- Contradiction (CON): directly contradicts the evidence (e.g. claims higher when figure shows lower).
Matching is any-of: a sentence may carry multiple gold labels; predicting any one of them is correct.`

const ROOT_CAUSES = [
  'NUMERIC_OCR',            // needs extracting a specific number/value from the figure
  'VISUAL_TREND',           // needs reading a curve/bar/trend/comparison visually
  'CAPTION_MISSING_INFO',   // needed info absent from caption text, only in the image
  'SCOPE_QUANTIFIER',       // over-generalization / quantifier / condition-scope reasoning
  'ENTITY_PRESENCE',        // must verify a named dataset/metric/model/baseline appears in evidence
  'CAUSAL_CLAIM',           // unsupported causal/mechanistic explanation inserted
  'MULTI_LABEL_OVERLAP',    // legitimately multi-label; the prediction picks a valid alternative
  'GOLD_ERROR',             // the gold label itself appears wrong
  'DEFAULT_TO_SUPPORTED',   // signal was available but model lazily defaulted to Supported
  'INHERENTLY_AMBIGUOUS',   // even a careful human cannot be sure from the given evidence
]

const DIAG_SCHEMA = {
  type: 'object', additionalProperties: false,
  required: ['case_id','figure_readability','correct_label_from_evidence','assessment',
             'root_cause','modality_needed','mechanism','fixable_lever','confidence','contested'],
  properties: {
    case_id: { type: 'string' },
    figure_readability: { type: 'string', enum: ['numbers_legible','partly_legible','illegible','no_relevant_content','no_image'] },
    correct_label_from_evidence: { type: 'string', enum: ['SUP','UCM','UE','SO','CON','UNCLEAR'],
      description: 'YOUR judgment of the correct label after reading the figure yourself' },
    assessment: { type: 'string',
      enum: ['model_wrong_recoverable','model_wrong_hard','model_defensible','gold_suspect','ambiguous'],
      description: 'dev: model prediction vs gold; testp1: the minority/majority faction vs truth' },
    root_cause: { type: 'string', enum: ROOT_CAUSES },
    modality_needed: { type: 'string',
      enum: ['caption_sufficient','image_numeric','image_visual_trend','cross_figure','claim_internal_logic','external_domain_knowledge'] },
    mechanism: { type: 'string', description: '1-3 sentences: concretely WHY the model failed/conflicted on THIS sentence given THIS figure' },
    fixable_lever: { type: 'string', description: 'short: training/data/inference change that would plausibly fix this class of error' },
    confidence: { type: 'string', enum: ['low','med','high'] },
    contested: { type: 'boolean', description: 'true if you assess gold_suspect OR model_defensible (triggers adversarial re-read)' },
  },
}

const VERIFY_SCHEMA = {
  type: 'object', additionalProperties: false,
  required: ['case_id','verdict','correct_label','reason'],
  properties: {
    case_id: { type: 'string' },
    verdict: { type: 'string', enum: ['confirm_contested','reject_contested'],
      description: 'confirm = the original contested claim (gold wrong / model defensible) holds; reject = original diagnosis was wrong, gold/majority is actually right' },
    correct_label: { type: 'string', enum: ['SUP','UCM','UE','SO','CON','UNCLEAR'] },
    reason: { type: 'string' },
  },
}

function diagPrompt(m) {
  const isDev = m.kind === 'dev_gold_error'
  return `You are doing forensic error analysis for a scientific-claim faithfulness task.

${LABELS}

Your case file is at: ${m.file}
1. Read that JSON case file (it has target_sentence, full_claim for context, evidence_captions, an "image_paths" list, and ${isDev ? 'the model_prediction + gold_label(s) + model_confidence' : 'the per-model votes for this sentence'}).
2. Read EACH image path listed in the case file's "image_paths" (these are the actual evidence figures/tables).
   Look hard at axis labels, numeric values, legends, bars/curves, table cells — the real numbers live in the image, captions are often insufficient.
3. Judge the SINGLE target_sentence (not the whole claim) against the evidence, and decide the correct label YOURSELF.

${isDev
  ? `This is a DEV case with GOLD. The model predicted "${'<see file>'}". Compare the gold label(s) to what you read in the figure. Decide: is the model wrong (and is it recoverable by a better model, or genuinely hard), is the model actually defensible, is the gold suspect, or is it ambiguous?`
  : `This is a TESTP1 case with NO gold — the 10 models DISAGREED (see model_votes). Read the figure and decide what the correct label really is, then explain WHY the models split (e.g., a faction over-flagged, the figure is illegible, the distinction is genuinely subtle).`}

Then diagnose the ROOT CAUSE: pick the single best root_cause from the enum, the modality_needed to get it right, write a concrete mechanism (reference what you actually saw in the figure), and a fixable_lever.
Set contested=true ONLY if you judge gold_suspect or model_defensible.
Return the structured object. case_id MUST be "${m.case_id}".`
}

function verifyPrompt(m, d) {
  return `Adversarial re-check. A first analyst diagnosed case "${m.case_id}" as CONTESTED: assessment="${d.assessment}", they claim the correct label is "${d.correct_label_from_evidence}" with reasoning: "${d.mechanism}".

${LABELS}

Independently re-read the case file ${m.file} and EACH image path it lists in "image_paths".
Default to skepticism: only CONFIRM the contested claim if, after reading the figure yourself, the gold really looks wrong (or the model prediction really is defensible). Otherwise REJECT (the gold/majority is right and the first analyst over-reached).
Return the structured verdict. case_id MUST be "${m.case_id}".`
}

// ---------------- run ----------------
const manifest = typeof args === 'string' ? JSON.parse(args) : args
log(`root-cause over ${manifest.length} cases (${manifest.filter(m=>m.kind==='dev_gold_error').length} dev-gold-errors, ${manifest.filter(m=>m.kind==='testp1_conflict').length} testp1-conflicts)`)

phase('Diagnose')
const diagnosed = await pipeline(
  manifest,
  (m) => agent(diagPrompt(m), { schema: DIAG_SCHEMA, model: 'sonnet', phase: 'Diagnose', label: m.case_id }),
  (d, m) => {
    if (!d) return null
    if (d.contested) {
      return agent(verifyPrompt(m, d), { schema: VERIFY_SCHEMA, model: 'opus', phase: 'Verify', label: 'verify:' + m.case_id })
        .then(v => ({ ...d, kind: m.kind, verify: v }))
        .catch(() => ({ ...d, kind: m.kind, verify: null }))
    }
    return { ...d, kind: m.kind }
  }
)
const ok = diagnosed.filter(Boolean)
log(`diagnosed ${ok.length}/${manifest.length}; contested+verified ${ok.filter(d=>d.verify).length}`)

phase('Synthesize')
const compact = ok.map(d => ({
  id: d.case_id, kind: d.kind, fig: d.figure_readability, correct: d.correct_label_from_evidence,
  assess: d.assessment, rc: d.root_cause, mod: d.modality_needed, conf: d.confidence,
  mech: d.mechanism, lever: d.fixable_lever,
  verdict: d.verify ? d.verify.verdict : undefined,
}))

const QUANT = `Quantitative context already computed (do not recompute, build on it):
CLEAN phaseC model on densematch (honest, no leak), macro-F1=0.792, acc=0.968. Per-class recall: SUP .994, UCM 1.00, UE .852, SO .658, CON .222. 40 errors: 26 MISSED_MINORITY (gold minority -> pred Supported: UE->SUP x12, SO->SUP x9, CON->SUP x5), 8 WRONG_MINORITY (mostly SO->UE x5), 6 FALSE_ALARM.
LEAKY s01 (memorized full-traindev incl. dev) recall: CON .857, SO .844 — i.e. CON/SO recall collapses from ~0.85 (memorized) to ~0.22/0.66 (generalize) => generalization gap, not capacity.
TESTP1 10-model conflict over 5096 sents: 4510 unanimous-Supported, 460 SUP_vs_one_minority (recall battleground), 65 SUP_vs_multi_minority, 50 unanimous_minority, 11 minority_only_disagree. Disagreeing label-pairs: SUP<->UE 312, SUP<->UCM 127, SO<->SUP 78, CON<->SUP 75, SO<->UE 40. Per-model minority-call rate 3.6%(Qwen32B-joint) .. 7.5%(Gemma-26B).`

const synth = await agent(
`You are the lead analyst writing the root-cause section of an error analysis for NLPCC-2026 Task10 Track1 (per-sentence scientific-claim faithfulness, 5 labels, scored by macro-F1 + paragraph-exact-match).

${QUANT}

Below are ${compact.length} per-case figure-reading diagnoses (each produced by an agent that actually READ the evidence figure). JSON:
${JSON.stringify(compact)}

Write a rigorous, specific analysis in Chinese (markdown). Cover:
1. **失败模式分类学**: tabulate root_cause counts (separately for dev-gold-errors vs testp1-conflicts), and explain the top 3-4 mechanisms with concrete examples (cite case ids + what was in the figure).
2. **多模判断冲突的本质**: what actually drives the 10-model disagreements — is it figure illegibility, a genuine label-definition boundary (e.g. SO vs UE), or aggressive vs conservative models? Use the diagnoses + the SUP<->UE / SO<->UE pair stats.
3. **召回崩塌的根因**: why minority recall (esp. CON .22, SO .66) collapses on clean data while memorized model gets .85 — connect to modality_needed distribution (how many errors truly need image_numeric / cross_figure that an 8B VL can't do, vs DEFAULT_TO_SUPPORTED laziness that IS fixable).
4. **gold 噪声 / 模型其实对了**: summarize the contested cases and how many survived adversarial verification (verdict=confirm). Quantify how much of our "error rate" is actually label noise.
5. **可操作杠杆**: rank concrete levers (training-data, loss, inference, OCR/figure-grounding) by expected payoff, grounded in the modality_needed + fixable_lever evidence — which errors are model-fixable vs fundamentally evidence-limited.
Be concrete and quantitative. Prefer tables. Do not hedge; where evidence is thin, say so.`,
  { model: 'opus', effort: 'high', phase: 'Synthesize', label: 'synthesis' }
)

return { synthesis: synth, n_diagnosed: ok.length, diagnoses: compact }
