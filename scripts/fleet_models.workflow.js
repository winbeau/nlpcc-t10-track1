export const meta = {
  name: 'fleet-model-shortlist',
  description: 'Rank ms-swift-supported open VLMs for the diversity fleet (leaderboard × task-fit × diversity × memory)',
  phases: [
    { title: 'Research', detail: 'leaderboard standing (websearch) + architecture-diversity/task-fit + memory/smoke-risk' },
    { title: 'Synthesize', detail: 'opus -> ranked fleet shortlist with exact ModelScope IDs + add-order' },
  ],
}

const CONTEXT = `
TASK: NLPCC-2026 Task10 Track1 — per-sentence faithfulness of scientific claims vs FIGURE/TABLE evidence
(images + captions). Metric score=(MacroF1+PEM)/2. testp1 is RECALL-critical (catch minority sentences).
We have proven that a UNION ENSEMBLE of diverse fine-tuned VLMs is the winning lever: more DIVERSE members
-> higher MacroF1 AND PEM (47.8 single -> 50.26 union of 5). Current 5 members are ALL Qwen3-VL (8B
per-sentence ×3, 8B-joint, 32B-joint) — highly correlated. GOAL: pick ADDITIONAL open VLMs to fine-tune
(LoRA, per-sentence, ~1 epoch) and add to the union, maximizing ARCHITECTURE DIVERSITY vs Qwen3-VL (different
visual encoder/connector/LLM -> different errors -> bigger ensemble payoff) while staying strong on the task.

HARDWARE: 1x H200 = 143.7 GiB usable (SHARED 8-GPU box, often only 1-2 free). LoRA + freeze-ViT, bf16,
per-sentence (one short {"label":X} output). So a dense model up to ~14B trains comfortably (8B ~16GiB
weights + activations); ~30B-MoE may fit (all experts resident); >30B dense is borderline/too big. Inference
is per-sentence (short outputs). The task is DOCUMENT/CHART/TABLE understanding (DocVQA/ChartQA-like), NOT
natural-image captioning — pick models strong on document/chart/OCR-ish reasoning.

ms-swift 4.2.3 SUPPORTS these multimodal model_types (authoritative — pick ONLY from here):
qwen_vl qwen2_vl qwen2_5_vl qwen3_vl qwen3_vl_moe ovis1_6 ovis2 ovis2_5 chatglm4v glm4v glm4v_moe glm_edge_v
glm_ocr cogvlm cogvlm2 internvl_chat internvl interns1 xcomposer2 xcomposer2_4khd xcomposer2_5 llama3_2_vision
llama4 llava1_5_hf llava1_6_mistral_hf llava1_6_vicuna_hf llava1_6_yi_hf llava_onevision_hf yi_vl ernie_vl
llava_onevision1_5 deepseek_vl deepseek_vl2 deepseek_janus_pro deepseek_ocr deepseek_ocr2 minicpmv2_5 minicpmv2_6
minicpmv4 minicpmv4_5 minicpmv4_6 minicpmo minimax_vl mplug_owl2 mplug_owl3 doc_owl2 got_ocr2 kimi_vl keye_vl
keye_vl_1_5 dots_ocr sail_vl2 phi3_vision phi4_multimodal florence idefics3 paligemma molmo molmo2 molmoe pixtral
megrez_omni gemma3_vision gemma3n mistral3 mistral3_2506 paddle_ocr hunyuan_ocr step3_vl
(also have: InternVL3-8B and Qwen3-VL-4B already downloading for the fleet.)
ALREADY IN FLEET (Qwen3-VL family): do NOT just add more Qwen variants for diversity — the value is OTHER families.
`

const SCHEMA = {
  type: 'object', additionalProperties: false,
  properties: {
    angle: { type: 'string' },
    findings: { type: 'array', items: { type: 'string' } },
    candidates: { type: 'array', items: {
      type: 'object', additionalProperties: false,
      properties: {
        family: { type: 'string' },
        modelscope_id: { type: 'string' },
        params: { type: 'string' },
        leaderboard_note: { type: 'string' },
        doc_table_strength: { type: 'string' },
        diversity_vs_qwen: { type: 'string' },
        memory_fit_h200: { type: 'string' },
        smoke_risk: { type: 'string' },
      },
      required: ['family', 'modelscope_id', 'params'],
    } },
  },
  required: ['angle', 'candidates'],
}
const PLAN = {
  type: 'object', additionalProperties: false,
  properties: {
    shortlist: { type: 'array', items: {
      type: 'object', additionalProperties: false,
      properties: {
        rank: { type: 'number' }, family: { type: 'string' }, modelscope_id: { type: 'string' },
        params: { type: 'string' }, why: { type: 'string' }, memory_h200: { type: 'string' },
        smoke_risk: { type: 'string' }, add_order_note: { type: 'string' },
      },
      required: ['rank', 'family', 'modelscope_id', 'why'],
    } },
    avoid: { type: 'array', items: { type: 'string' } },
    recommendation: { type: 'string' },
  },
  required: ['shortlist', 'recommendation'],
}

log('Fleet model shortlist: leaderboard + diversity/fit + memory -> opus synthesis')
phase('Research')
const angles = [
  { label: 'leaderboard', model: 'sonnet', task:
    `ANGLE = LEADERBOARD STANDING. Using WebSearch for current open-VLM leaderboards (OpenCompass/OpenVLM, ` +
    `DocVQA, ChartQA, MMMU, OCRBench), identify which of the ms-swift-supported families are the STRONGEST ` +
    `open VLMs in the ~2B–14B (and notable MoE ≤~30B) range AS OF NOW, especially on DOCUMENT/CHART/TABLE/OCR ` +
    `benchmarks (most relevant to this task). Give exact ModelScope ids + params + a leaderboard note per pick. ` +
    `Prioritize recent strong releases (e.g. InternVL3, Ovis2/2.5, GLM-4V, Kimi-VL, Keye-VL, MiniCPM-V4.5, ` +
    `Molmo, Llama-4, dots_ocr) — verify with search, don't assume.` },
  { label: 'diversity-fit', model: 'sonnet', task:
    `ANGLE = ARCHITECTURE DIVERSITY vs Qwen3-VL + TASK FIT. The fleet already has 5 Qwen3-VL models. Rank the ` +
    `supported families by how DIFFERENT their vision stack is from Qwen3-VL (different visual encoder, e.g. ` +
    `InternViT / SigLIP / AIMv2 / CLIP / native-resolution schemes, different connector/LLM) — more different = ` +
    `more uncorrelated errors = bigger union-ensemble payoff. Cross with TASK FIT (scientific figures/tables, ` +
    `document/chart reasoning). Flag OCR-only models (got_ocr2/paddle_ocr/deepseek_ocr/dots_ocr/hunyuan_ocr) — ` +
    `are they classifiers-capable or pure-OCR (unfit)? Give exact ids + diversity & fit notes.` },
  { label: 'memory-smoke', model: 'sonnet', task:
    `ANGLE = MEMORY + ms-swift SMOKE-RISK on 1x H200 (143.7 GiB, LoRA freeze-ViT bf16, per-sentence). For the ` +
    `candidate families, estimate weights (params×2 bytes) + whether they fit 1 H200 for LoRA training (dense ` +
    `≤~14B easy; ~30B-MoE all-experts-resident ~60GiB ok; >30B dense too big). Flag ms-swift-4.2.3 SMOKE risks: ` +
    `families needing special flash-attn/transformers versions, non-standard image preprocessing, or known ms-swift ` +
    `quirks. Give exact ids + memory estimate + smoke-risk per family. Recommend ≤~14B dense picks that LoRA-train ` +
    `cleanly with minimal cross-model fuss.` },
]
const research = (await parallel(angles.map(a => () =>
  agent(`${CONTEXT}\n\n${a.task}\n\nReturn structured analysis with exact ModelScope ids.`,
    { label: a.label, phase: 'Research', model: a.model, schema: SCHEMA })
))).filter(Boolean)

phase('Synthesize')
const plan = await agent(
  `${CONTEXT}\n\nTHREE RESEARCH ANGLES:\n${JSON.stringify(research, null, 1)}\n\n` +
  `Synthesize a RANKED FLEET SHORTLIST of the best ADDITIONAL VLMs to fine-tune + add to the union — the ` +
  `intersection of (ms-swift-supported) ∩ (leaderboard-strong on doc/chart/table) ∩ (architecture-diverse vs ` +
  `Qwen3-VL) ∩ (fits 1 H200 for LoRA, ≤~14B dense or fitting MoE) ∩ (low ms-swift smoke-risk). For each: exact ` +
  `ModelScope id, params, why (diversity + strength + fit), memory_h200, smoke_risk, and add_order. Aim for 5–8 ` +
  `picks spanning DIFFERENT families (don't stack one family). List what to AVOID (too big / OCR-only / risky). ` +
  `recommendation = the concrete order to train+add them to the fleet next, given a shared box with ~1-2 free H200.`,
  { label: 'synthesis', phase: 'Synthesize', model: 'opus', schema: PLAN })

return plan
