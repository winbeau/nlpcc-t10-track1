#!/usr/bin/env bash
# Plain-CE LoRA SFT on PARAGRAPH-JOINT data (one sample/record, JSON-array target), then infer
# testp1 (--joint) -> aggregate -> validated zip. The rank-2 gate before any 32B spend.
#
#   bash scripts/train_joint.sh                 # 8B joint, 1 epoch, full train+infer+zip
#   MODEL_ID=... TAG=joint32b ... bash scripts/train_joint.sh   # 32B (after 8B validates)
#   MAX_STEPS=20 SMOKE=1 bash scripts/train_joint.sh            # smoke: 20 steps, no save, no infer
#
# Env: GPUS(0,1) MAX_PIXELS(401408) MAX_LENGTH(8192) EPOCHS(1) TAG(joint8b) MODEL_ID DATA_ROOT.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$REPO_ROOT"
DATA_ROOT="${DATA_ROOT:-$(dirname "$REPO_ROOT")/NLPCC-2026-Task10-Science}"
export MODELSCOPE_CACHE="${MODELSCOPE_CACHE:-$(dirname "$REPO_ROOT")/ms_cache}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
MODEL="${MODEL_ID:-Qwen/Qwen3-VL-8B-Instruct}"
GPUS="${GPUS:-0,1}"; NPROC=$(echo "$GPUS" | awk -F, '{print NF}')
MAXLEN="${MAX_LENGTH:-8192}"; MP="${MAX_PIXELS:-401408}"; EPOCHS="${EPOCHS:-1}"
TAG="${TAG:-joint8b}"; OUTDIR="outputs/p0_${TAG}"
STEP_ARGS=(); [ -n "${MAX_STEPS:-}" ] && STEP_ARGS=(--max_steps "$MAX_STEPS" --save_strategy no)

echo "### JOINT TRAIN $TAG | model=$MODEL | plain-CE | max_pixels=$MP | max_length=$MAXLEN | GPUs=$GPUS | epochs=$EPOCHS ${MAX_STEPS:+| SMOKE max_steps=$MAX_STEPS} ###"
CUDA_VISIBLE_DEVICES="$GPUS" PYTHONPATH=src uv run torchrun --nproc_per_node="$NPROC" --master_port=29537 \
  scripts/train_joint.py \
  --model "$MODEL" --tuner_type lora --torch_dtype bfloat16 \
  --dataset "${DATASET:-data/train_sft.jsonl}" --split_dataset_ratio 0 \
  --num_train_epochs "$EPOCHS" --per_device_train_batch_size 1 --gradient_accumulation_steps 4 \
  --learning_rate 1e-4 --lora_rank 16 --lora_alpha 32 --freeze_vit true \
  --max_length "$MAXLEN" --max_pixels "$MP" --attn_impl sdpa --packing false --padding_free false \
  --use_logits_to_keep true --eval_strategy no --save_strategy epoch --save_total_limit 1 \
  --logging_steps 5 --dataloader_num_workers 4 --output_dir "$OUTDIR" "${STEP_ARGS[@]}"

[ -n "${SMOKE:-}${NO_INFER:-}" ] && { echo "### train done (no infer: SMOKE/NO_INFER) ###"; exit 0; }

CKPT=$(ls -d "$OUTDIR"/v*/checkpoint-* 2>/dev/null | sort | tail -1)
[ -n "$CKPT" ] || { echo "ERROR: no checkpoint in $OUTDIR"; exit 1; }
echo "### trained -> $CKPT ; INFER testp1 (--joint, resolution-matched) ###"
RAW="outputs/testp1_${TAG}_raw.jsonl"; SUB="submissions/testp1_${TAG}_submission.jsonl"
env "CUDA_VISIBLE_DEVICES=${GPUS%%,*}" "MAX_PIXELS=$MP" PYTHONPATH=src uv run python -m nlpcc_t10.infer \
  --split testp1 --joint --model "$MODEL" --adapter "$CKPT" --engine pt --data-root "$DATA_ROOT" --out "$RAW"
PYTHONPATH=src uv run python -m nlpcc_t10.aggregate \
  --pred "$RAW" --ref "$DATA_ROOT/data/testp1-track-1.jsonl" --out "$SUB"
uv run python scripts/make_submission_zip.py --sub "$SUB" \
  --ref "$DATA_ROOT/data/testp1-track-1.jsonl" --out "submissions/testp1_${TAG}_submission.zip"
echo "=== JOINT TRAIN+SUBMISSION DONE: submissions/testp1_${TAG}_submission.zip ==="
