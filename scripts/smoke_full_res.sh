#!/usr/bin/env bash
# P0 smoke: prove use_logits_to_keep=true unlocks FULL-resolution training without OOM.
#
# Grid recipe needed max_pixels=401408 (512 tok/img) + use_logits_to_keep=false to fit, because
# the full-vocab CE over [B,T,V=151936] logits dominated memory. With use_logits_to_keep=true the
# model only emits logits for the last K (response) positions -> the logits hog is gone, so we can
# raise the image resolution. T6 in test_softmin_loss.py already proved the loss is unchanged.
#
# Runs 20 steps on ONE GPU (worst-case per-GPU memory; DDP replicates the same per-GPU footprint),
# logging_steps=1 so ms-swift prints the 'memory(GiB)' field each step. Saves nothing.
#
#   CUDA_VISIBLE_DEVICES=0 bash scripts/smoke_full_res.sh            # full res (no max_pixels cap)
#   MAX_PIXELS_SMOKE=1605632 bash scripts/smoke_full_res.sh         # cap to 2048 tok/img if OOM
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$REPO_ROOT"
DATA_ROOT="${DATA_ROOT:-$(dirname "$REPO_ROOT")/NLPCC-2026-Task10-Science}"
export MODELSCOPE_CACHE="${MODELSCOPE_CACHE:-$(dirname "$REPO_ROOT")/ms_cache}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
MODEL="${MODEL_ID:-Qwen/Qwen3-VL-8B-Instruct}"
MAXLEN="${MAX_LENGTH:-10240}"          # full-res images need a longer window than the grid's 4096
MP="${MAX_PIXELS_SMOKE:-}"             # empty => model default resolution (full res)
MP_ARG=(); [ -n "$MP" ] && MP_ARG=(--max_pixels "$MP")
DATASET="${DATASET:-data/train_sft.jsonl}"   # point at a worst-case subset to stress peak memory
STEPS="${SMOKE_STEPS:-20}"

echo "### P0 SMOKE | use_logits_to_keep=true | max_length=$MAXLEN | max_pixels=${MP:-<model default/full>} | dataset=$DATASET | steps=$STEPS | GPU=$CUDA_VISIBLE_DEVICES ###"
SOFTMIN_BETA=5 SOFTMIN_LAMBDA=0.5 \
PYTHONPATH=src uv run torchrun --nproc_per_node=1 --master_port=29533 \
  scripts/train_softmin.py \
  --model "$MODEL" --tuner_type lora --torch_dtype bfloat16 \
  --dataset "$DATASET" --split_dataset_ratio 0 \
  --loss_type softmin_pem \
  --max_steps "$STEPS" --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 1 --learning_rate 1e-4 \
  --lora_rank 16 --lora_alpha 32 --freeze_vit true \
  --max_length "$MAXLEN" "${MP_ARG[@]}" --attn_impl sdpa \
  --packing false --padding_free false --use_logits_to_keep true \
  --eval_strategy no --save_strategy no \
  --logging_steps 1 --dataloader_num_workers 0 --output_dir /tmp/p0_smoke_out
