#!/usr/bin/env bash
# 远程服务器: 推理。用法: bash scripts/infer.sh {dev|testp1} [adapter_dir]
# 输出句子级原始预测到 outputs/<split>_raw.jsonl，再用 aggregate.py 转提交格式。
set -euo pipefail
export DATA_ROOT="${DATA_ROOT:-../NLPCC-2026-Task10-Science}"
SPLIT="${1:-dev}"
ADAPTER="${2:-outputs/qwen3vl8b_lora}"

uv run python -m nlpcc_t10.infer \
  --split "$SPLIT" \
  --data-root "$DATA_ROOT" \
  --adapter "$ADAPTER" \
  --out "outputs/${SPLIT}_raw.jsonl" \
  "${@:3}"
