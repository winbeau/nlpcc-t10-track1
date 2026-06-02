#!/usr/bin/env bash
# 在 9:1 dev 上用官方评测器打分。用法: bash scripts/eval.sh outputs/dev_submission.jsonl
set -euo pipefail
export DATA_ROOT="${DATA_ROOT:-../NLPCC-2026-Task10-Science}"
PRED="${1:-outputs/dev_submission.jsonl}"
GOLD="${2:-data/dev_gold.jsonl}"

uv run python "$DATA_ROOT/offline_eval/evaluate.py" \
  --track 1 --gold "$GOLD" --pred "$PRED" --match id
