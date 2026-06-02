#!/usr/bin/env bash
# 远程服务器: 句子级推理。用法: bash scripts/infer.sh {dev|testp1} [adapter_dir] [engine] [extra args...]
#   engine: pt（默认, transformers, 最稳）| vllm（开 prefix-caching 复用前缀 KV, 吞吐更高）
# 输出句子级原始预测到 outputs/<split>_raw.jsonl，再用 aggregate.py 转提交格式。
#
# 例:
#   bash scripts/infer.sh dev    outputs/qwen3vl8b_lora pt
#   bash scripts/infer.sh testp1 outputs/qwen3vl8b_lora vllm
set -euo pipefail
export DATA_ROOT="${DATA_ROOT:-../NLPCC-2026-Task10-Science}"
SPLIT="${1:-dev}"
ADAPTER="${2:-outputs/qwen3vl8b_lora}"
ENGINE="${3:-pt}"

uv run python -m nlpcc_t10.infer \
  --split "$SPLIT" \
  --data-root "$DATA_ROOT" \
  --adapter "$ADAPTER" \
  --engine "$ENGINE" \
  --out "outputs/${SPLIT}_raw.jsonl" \
  "${@:4}"
