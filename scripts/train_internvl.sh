#!/usr/bin/env bash
# Fine-tune InternVL3-8B-hf (transformers-NATIVE variant — avoids the trust_remote_code breakage
# on transformers 5.8.1) per-sentence with the softmin_pem loss, then infer testp1 -> aggregate
# -> validated zip. This is the first NON-Qwen, architecture-DIVERSE member of the union fleet
# (InternViT visual encoder + different connector/LLM -> uncorrelated errors -> bigger union payoff).
#
#   bash scripts/train_internvl.sh                 # full 1-epoch train+infer+zip on GPU 1
#   MAX_STEPS=8 SMOKE=1 bash scripts/train_internvl.sh   # smoke: 8 steps, no save/infer
#
# CRITICAL knobs (validated the hard way — 3 OOM/error cycles before this worked):
#   - --vit_gradient_checkpointing TRUE is THE fix for OOM. The LLM's gradient_checkpointing=True is on
#     by default but the InternViT VISION TOWER's is OFF -> it stores ALL vision activations -> ~128 GiB
#     baseline + a 37.65 GiB spike on tile-heavy records -> OOM at ~step 450. Checkpointing the ViT slashes it.
#   - MAX_PIXELS is INERT for InternVL-hf (it is a Qwen pixel-budget knob; InternVL tiles via max_patches in
#     the processor config, which --max_pixels does NOT touch). 401408 vs 200704 gave byte-identical OOM.
#   - max_length 4096: at 2048 some records' tokens exceed it -> ms-swift "Failed to retrieve dataset" ValueError.
#   - --use_logits_to_keep FALSE: the -hf path does not support it (Qwen does; InternVL does not).
#   - HF_HOME must point at our writable cache (default points at another user's read-only dir).
#   - lr 5e-5 (not the Qwen 1e-4): the 1e-4 smoke showed a transient nan grad; 5e-5 trains clean.
#
# Env (overridable): GPUS(1), MAX_PIXELS(401408), MAX_LENGTH(4096), LR(5e-5), BETA(5), LAMBDA(0.5),
#                    EPOCHS(1), TAG(internvl8b), MODEL_ID(OpenGVLab/InternVL3-8B-hf), DATA_ROOT.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$REPO_ROOT"
DATA_ROOT="${DATA_ROOT:-$(dirname "$REPO_ROOT")/NLPCC-2026-Task10-Science}"
export MODELSCOPE_CACHE="${MODELSCOPE_CACHE:-$(dirname "$REPO_ROOT")/ms_cache}"
export HF_HOME="${HF_HOME:-$(dirname "$REPO_ROOT")/hf_home}"  # MUST be writable (default is read-only)
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
MODEL="${MODEL_ID:-OpenGVLab/InternVL3-8B-hf}"
GPUS="${GPUS:-1}"; NPROC=$(echo "$GPUS" | awk -F, '{print NF}')
MAXLEN="${MAX_LENGTH:-4096}"; MP="${MAX_PIXELS:-401408}"; LR="${LR:-5e-5}"
BETA="${BETA:-5}"; LAMBDA="${LAMBDA:-0.5}"; EPOCHS="${EPOCHS:-1}"
TAG="${TAG:-internvl8b}"; OUTDIR="outputs/p0_${TAG}"
STEP_ARGS=(); [ -n "${MAX_STEPS:-}" ] && STEP_ARGS=(--max_steps "$MAX_STEPS" --save_strategy no)

echo "### INTERNVL TRAIN $TAG | model=$MODEL | softmin_pem | max_pixels=$MP | max_length=$MAXLEN | lr=$LR | GPUs=$GPUS | epochs=$EPOCHS ${MAX_STEPS:+| SMOKE max_steps=$MAX_STEPS} ###"
CUDA_VISIBLE_DEVICES="$GPUS" MAX_PIXELS="$MP" SOFTMIN_BETA="$BETA" SOFTMIN_LAMBDA="$LAMBDA" \
PYTHONPATH=src uv run torchrun --nproc_per_node="$NPROC" --master_port=29553 \
  scripts/train_softmin.py \
  --model "$MODEL" --tuner_type lora --torch_dtype bfloat16 \
  --dataset "${DATASET:-data/train_sft.jsonl}" --split_dataset_ratio 0 \
  --loss_type softmin_pem \
  --num_train_epochs "$EPOCHS" --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 1 --learning_rate "$LR" \
  --lora_rank 16 --lora_alpha 32 --freeze_vit true \
  --max_length "$MAXLEN" --max_pixels "$MP" --attn_impl sdpa \
  --gradient_checkpointing true --vit_gradient_checkpointing true \
  --packing false --padding_free false --use_logits_to_keep false \
  --eval_strategy no --save_strategy epoch --save_total_limit 1 \
  --logging_steps 20 --dataloader_num_workers 4 --output_dir "$OUTDIR" "${STEP_ARGS[@]}"

[ -n "${SMOKE:-}${NO_INFER:-}" ] && { echo "### train done (no infer: SMOKE/NO_INFER) ###"; exit 0; }

CKPT=$(ls -d "$OUTDIR"/v*/checkpoint-* 2>/dev/null | sort | tail -1)
[ -n "$CKPT" ] || { echo "ERROR: no checkpoint in $OUTDIR"; exit 1; }
echo "### trained -> $CKPT ; INFER testp1 (per-sentence, resolution-matched MAX_PIXELS=$MP) ###"
RAW="outputs/testp1_${TAG}_raw.jsonl"; SUB="submissions/testp1_${TAG}_submission.jsonl"
env "CUDA_VISIBLE_DEVICES=${GPUS%%,*}" "MAX_PIXELS=$MP" "HF_HOME=$HF_HOME" \
  PYTHONPATH=src uv run python -m nlpcc_t10.infer \
  --split testp1 --model "$MODEL" --adapter "$CKPT" --engine pt --data-root "$DATA_ROOT" --out "$RAW"
PYTHONPATH=src uv run python -m nlpcc_t10.aggregate \
  --pred "$RAW" --ref "$DATA_ROOT/data/testp1-track-1.jsonl" --out "$SUB"
uv run python scripts/make_submission_zip.py --sub "$SUB" \
  --ref "$DATA_ROOT/data/testp1-track-1.jsonl" --out "submissions/testp1_${TAG}_submission.zip"
echo "=== INTERNVL TRAIN+SUBMISSION DONE: submissions/testp1_${TAG}_submission.zip (add to union) ==="
