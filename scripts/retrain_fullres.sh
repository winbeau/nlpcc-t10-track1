#!/usr/bin/env bash
# P0 full-resolution retrain of the best grid config (default b5_l0.5) with use_logits_to_keep=true,
# then infer testp1 -> aggregate -> validate -> zip. 2-GPU DDP (0,1). Inference resolution is kept
# IDENTICAL to training (same MAX_PIXELS) to avoid a train/test mismatch.
#
#   bash scripts/retrain_fullres.sh                       # default 1024 tok/img (2x grid; worst
#                                                         #   paragraph 17x3img peaks ~125 GiB, fits)
#   MAX_PIXELS=602112 bash scripts/retrain_fullres.sh     # 768 tok/img if more headroom is wanted
#
# NOTE: the model's natural image resolution is <2048 tok/img, so caps >=2048 (and "no cap") do
# NOT bind and the worst paragraph OOMs (>140 GiB). 1024 tok/img is the validated-safe ceiling.
#
# Env (overridable): GPUS(0,1), MAX_PIXELS(802816=1024 tok/img), MAX_LENGTH(10240),
#                    BETA(5), LAMBDA(0.5), EPOCHS(1), MODEL_ID, DATA_ROOT.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$REPO_ROOT"
DATA_ROOT="${DATA_ROOT:-$(dirname "$REPO_ROOT")/NLPCC-2026-Task10-Science}"
export MODELSCOPE_CACHE="${MODELSCOPE_CACHE:-$(dirname "$REPO_ROOT")/ms_cache}"
# Anti-fragmentation: the worst paragraph (B=17 x 3 imgs) peaks ~125 GiB at 1024 tok/img; give
# the allocator expandable segments so the ~14 GiB headroom + DDP buffers don't fragment into OOM.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
MODEL="${MODEL_ID:-Qwen/Qwen3-VL-8B-Instruct}"
GPUS="${GPUS:-0,1}"; NPROC=$(echo "$GPUS" | awk -F, '{print NF}')
MAXLEN="${MAX_LENGTH:-10240}"
BETA="${BETA:-5}"; LAMBDA="${LAMBDA:-0.5}"; EPOCHS="${EPOCHS:-1}"
MP="${MAX_PIXELS:-802816}"; MP_ARG=(); [ -n "$MP" ] && MP_ARG=(--max_pixels "$MP")  # 802816=1024 tok/img
TAG="${TAG:-fullres_b${BETA}_l${LAMBDA}}"   # override for other experiments (e.g. gentle-oversample @ grid res)
OUTDIR="outputs/p0_${TAG}"

echo "### P0 RETRAIN $TAG | use_logits_to_keep=true | max_pixels=${MP:-<full/model-default>} | max_length=$MAXLEN | GPUs=$GPUS | epochs=$EPOCHS ###"
CUDA_VISIBLE_DEVICES="$GPUS" SOFTMIN_BETA="$BETA" SOFTMIN_LAMBDA="$LAMBDA" \
PYTHONPATH=src uv run torchrun --nproc_per_node="$NPROC" --master_port=29535 \
  scripts/train_softmin.py \
  --model "$MODEL" --tuner_type lora --torch_dtype bfloat16 \
  --dataset data/train_sft.jsonl --split_dataset_ratio 0 \
  --loss_type softmin_pem \
  --num_train_epochs "$EPOCHS" --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 1 --learning_rate 1e-4 \
  --lora_rank 16 --lora_alpha 32 --freeze_vit true \
  --max_length "$MAXLEN" "${MP_ARG[@]}" --attn_impl sdpa \
  --packing false --padding_free false --use_logits_to_keep true \
  --eval_strategy no --save_strategy epoch --save_total_limit 1 \
  --logging_steps 20 --dataloader_num_workers 0 --output_dir "$OUTDIR"

CKPT=$(ls -d "$OUTDIR"/v*/checkpoint-* 2>/dev/null | sort | tail -1)
[ -n "$CKPT" ] || { echo "ERROR: no checkpoint in $OUTDIR"; exit 1; }
echo "### trained -> $CKPT ; INFER testp1 (resolution-matched to training) ###"
RAW="outputs/testp1_${TAG}_raw.jsonl"; SUB="submissions/testp1_${TAG}_submission.jsonl"
INFER_ENV=("CUDA_VISIBLE_DEVICES=${GPUS%%,*}"); [ -n "$MP" ] && INFER_ENV+=("MAX_PIXELS=$MP")
env "${INFER_ENV[@]}" PYTHONPATH=src uv run python -m nlpcc_t10.infer \
  --split testp1 --adapter "$CKPT" --engine pt --data-root "$DATA_ROOT" --out "$RAW"
PYTHONPATH=src uv run python -m nlpcc_t10.aggregate \
  --pred "$RAW" --ref "$DATA_ROOT/data/testp1-track-1.jsonl" --out "$SUB"
uv run python scripts/make_submission_zip.py --sub "$SUB" \
  --ref "$DATA_ROOT/data/testp1-track-1.jsonl" --out "submissions/testp1_${TAG}_submission.zip"
echo "=== P0 RETRAIN+SUBMISSION DONE: submissions/testp1_${TAG}_submission.zip (dev compare vs 47.8) ==="
