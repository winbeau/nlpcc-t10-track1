#!/usr/bin/env bash
# Produce a Codabench Phase-1 (testp1) submission from a grid config's checkpoint.
#
#   bash scripts/make_testp1_submission.sh [config_tag]
#     config_tag e.g. b5_l0.5 ; default = the "best" from outputs/grid_summary.json.
#
# Runs: infer (testp1, the config's saved LoRA adapter) -> aggregate -> outputs/testp1_submission.jsonl
# Uses MAX_PIXELS=401408 to match the grid training resolution. Path-portable.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
DATA_ROOT="${DATA_ROOT:-$(dirname "$REPO_ROOT")/NLPCC-2026-Task10-Science}"
export MODELSCOPE_CACHE="${MODELSCOPE_CACHE:-$(dirname "$REPO_ROOT")/ms_cache}"
export MAX_PIXELS="${MAX_PIXELS:-401408}"          # match grid training (512 tok/img)
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

TAG="${1:-}"
if [ -z "$TAG" ]; then
  TAG=$(uv run python -c "import json; print(json.load(open('outputs/grid_summary.json'))['best'])")
fi
CKPT=$(ls -d "outputs/grid_${TAG}"/v*/checkpoint-* 2>/dev/null | sort | tail -1)
echo "config=${TAG}  ckpt=${CKPT}"
[ -n "$CKPT" ] || { echo "ERROR: no checkpoint found for outputs/grid_${TAG}/v*/checkpoint-*"; exit 1; }

RAW="outputs/testp1_${TAG}_raw.jsonl"
SUB="outputs/testp1_${TAG}_submission.jsonl"

echo "### INFER testp1 (586 records) ###"
PYTHONPATH=src uv run python -m nlpcc_t10.infer \
  --split testp1 --adapter "$CKPT" --engine pt \
  --data-root "$DATA_ROOT" --out "$RAW"

echo "### AGGREGATE -> submission ###"
PYTHONPATH=src uv run python -m nlpcc_t10.aggregate \
  --pred "$RAW" --ref "$DATA_ROOT/data/testp1-track-1.jsonl" --out "$SUB"

echo "=================================================================="
echo " SUBMISSION FILE: $REPO_ROOT/$SUB"
echo " lines: $(wc -l < "$SUB")  (should be 586)"
echo " Upload this JSONL to Codabench competition 16666 (Track 1)."
echo "=================================================================="
