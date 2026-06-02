#!/usr/bin/env bash
# 远程服务器: LoRA SFT。先跑 build_dataset.py 生成 data/train_sft.jsonl + data/dev_sft.jsonl。
set -euo pipefail
export DATA_ROOT="${DATA_ROOT:-../NLPCC-2026-Task10-Science}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

# 落地时按 ms-swift 版本核对参数名（`swift sft -h`）。
uv run swift sft --config configs/qwen3vl_lora_sft.yaml "$@"
