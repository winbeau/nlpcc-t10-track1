#!/usr/bin/env bash
# GATE-B PHASE B: dev inference for the Qwen anchors on the image-disjoint devbench dev (501 rec).
# 4 background workers (GPU 3,4,5,6), 2 anchors each. Per-anchor: infer --split dev -> aggregate -> {id,labels}.
# NOTE: server-specific (hardcoded /data/chenjiayu paths + checkpoint dirs); a record of the run.
# Prereq: data/split.json must be the devbench split (cp data/devbench/split.json data/split.json).
# USE_HF=0 forces ms-swift to load the cached ModelScope bases (the box's global USE_HF=1 misroutes to HF).
set -uo pipefail
cd /data/chenjiayu/wenbiao_zhao/nlpcc-t10-track1
export DATA_ROOT=/data/chenjiayu/wenbiao_zhao/NLPCC-2026-Task10-Science
export MODELSCOPE_CACHE=/data/chenjiayu/wenbiao_zhao/ms_cache
export USE_HF=0
export TOKENIZERS_PARALLELISM=false
mkdir -p outputs/devbench_raw outputs/devbench_preds logs
Q8=Qwen/Qwen3-VL-8B-Instruct
Q32=Qwen/Qwen3-VL-32B-Instruct
REF=data/devbench/dev_gold.jsonl

run_sent() { local g=$1 mp=$2 ck=$3 a=$4
  CUDA_VISIBLE_DEVICES=$g MAX_PIXELS=$mp PYTHONPATH=src uv run python -m nlpcc_t10.infer \
    --split dev --adapter "$ck" --engine pt --data-root "$DATA_ROOT" --out outputs/devbench_raw/$a.jsonl \
  && PYTHONPATH=src uv run python -m nlpcc_t10.aggregate --pred outputs/devbench_raw/$a.jsonl --ref "$REF" --out outputs/devbench_preds/$a.jsonl; }
run_joint() { local g=$1 mp=$2 m=$3 ck=$4 a=$5
  CUDA_VISIBLE_DEVICES=$g MAX_PIXELS=$mp PYTHONPATH=src uv run python -m nlpcc_t10.infer \
    --split dev --joint --model "$m" --adapter "$ck" --engine pt --data-root "$DATA_ROOT" --out outputs/devbench_raw/$a.jsonl \
  && PYTHONPATH=src uv run python -m nlpcc_t10.aggregate --pred outputs/devbench_raw/$a.jsonl --ref "$REF" --out outputs/devbench_preds/$a.jsonl; }

worker3() { run_sent 3 401408 outputs/grid_b5_l0.5/v0-20260602-095341/checkpoint-2000 s01; run_joint 3 401408 "$Q32" outputs/p0_joint32b/v0-20260602-225608/checkpoint-499 s13; echo "WORKER3 DONE $(date)"; }
worker4() { run_sent 4 401408 outputs/p0_os4_b5_l0.5/v0-20260602-163142/checkpoint-2656 s08; run_sent 4 802816 outputs/p0_fullres_b5_l0.5/v0-20260602-135956/checkpoint-2000 s02; echo "WORKER4 DONE $(date)"; }
worker5() { run_sent 5 401408 outputs/p0_perclass_b5_l0.5/v0-20260602-173219/checkpoint-2276 s09; run_joint 5 401408 "$Q8" outputs/p0_joint8b/v1-20260602-211943/checkpoint-499 s12; echo "WORKER5 DONE $(date)"; }
worker6() { run_sent 6 401408 outputs/p0_gentle1.5_b5_l0.5/v0-20260602-151948/checkpoint-1400 s07; run_sent 6 401408 outputs/p0_perclass_ep2_b5_l0.5/v0-20260602-182714/checkpoint-4552 s10; echo "WORKER6 DONE $(date)"; }

worker3 > logs/pb_gpu3.log 2>&1 &
worker4 > logs/pb_gpu4.log 2>&1 &
worker5 > logs/pb_gpu5.log 2>&1 &
worker6 > logs/pb_gpu6.log 2>&1 &
echo "launched workers pids: $(jobs -p)"
wait
echo "ALL WORKERS DONE $(date)"
