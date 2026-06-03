#!/usr/bin/env bash
# Phase 1 (corrector): generate the 5 s15-member models' DEV predictions (OOF — none trained on
# the 333 dev records), aggregate to {id,labels}, build the dev union ensemble (= s15 on dev), and
# score it on the devhard subset. Outputs feed: (a) the offline devhard baseline, (b) the corrector
# training candidates. Single GPU, sequential. Log: /tmp/gen_dev.log
set -uo pipefail
cd /data/chenjiayu/wenbiao_zhao/nlpcc-t10-track1
export DATA_ROOT=/data/chenjiayu/wenbiao_zhao/NLPCC-2026-Task10-Science
export MODELSCOPE_CACHE=/data/chenjiayu/wenbiao_zhao/ms_cache
GPU="${GPU:-0}"
GRID=outputs/grid_b5_l0.5/v0-20260602-095341/checkpoint-2000
OS4=outputs/p0_os4_b5_l0.5/v0-20260602-163142/checkpoint-2656
PERCLASS=outputs/p0_perclass_b5_l0.5/v0-20260602-173219/checkpoint-2276
JOINT8B=outputs/p0_joint8b/v1-20260602-211943/checkpoint-499
JOINT32B=outputs/p0_joint32b/v0-20260602-225608/checkpoint-499

ps()   { CUDA_VISIBLE_DEVICES=$GPU MAX_PIXELS=401408 PYTHONPATH=src uv run python -m nlpcc_t10.infer --split dev --adapter "$1" --engine pt --data-root "$DATA_ROOT" --out "$2"; }
jt()   { CUDA_VISIBLE_DEVICES=$GPU MAX_PIXELS=401408 PYTHONPATH=src uv run python -m nlpcc_t10.infer --split dev --joint --model "$3" --adapter "$1" --engine pt --data-root "$DATA_ROOT" --out "$2"; }
agg()  { PYTHONPATH=src uv run python -m nlpcc_t10.aggregate --pred "$1" --ref data/dev_gold.jsonl --out "$2"; }

echo "### [1/5] grid dev $(date) ###";     ps "$GRID"     outputs/dev_grid_raw.jsonl     && agg outputs/dev_grid_raw.jsonl     outputs/dev_grid_sub.jsonl
echo "### [2/5] os4 dev $(date) ###";      ps "$OS4"      outputs/dev_os4_raw.jsonl      && agg outputs/dev_os4_raw.jsonl      outputs/dev_os4_sub.jsonl
echo "### [3/5] perclass dev $(date) ###"; ps "$PERCLASS" outputs/dev_perclass_raw.jsonl && agg outputs/dev_perclass_raw.jsonl outputs/dev_perclass_sub.jsonl
echo "### [4/5] joint8b dev $(date) ###";  jt "$JOINT8B"  outputs/dev_joint8b_raw.jsonl  Qwen/Qwen3-VL-8B-Instruct  && agg outputs/dev_joint8b_raw.jsonl  outputs/dev_joint8b_sub.jsonl
echo "### [5/5] joint32b dev $(date) ###"; jt "$JOINT32B" outputs/dev_joint32b_raw.jsonl Qwen/Qwen3-VL-32B-Instruct && agg outputs/dev_joint32b_raw.jsonl outputs/dev_joint32b_sub.jsonl

echo "### union dev ensemble (= s15 on dev) ###"
uv run python scripts/ensemble_union.py --min-votes 1 --out outputs/dev_ensU5_sub.jsonl \
  outputs/dev_grid_sub.jsonl outputs/dev_os4_sub.jsonl outputs/dev_perclass_sub.jsonl outputs/dev_joint8b_sub.jsonl outputs/dev_joint32b_sub.jsonl

echo "### DEVHARD score: grid (single) ###"
PYTHONPATH=src uv run python -m nlpcc_t10.eval_local --pred outputs/dev_grid_sub.jsonl --gold data/dev_gold.jsonl --data-root "$DATA_ROOT" --hard-subset 2>&1 | grep -E "HARD subset|score|macro_f1|pem|SCORE"
echo "### DEVHARD score: ensU5 (union) ###"
PYTHONPATH=src uv run python -m nlpcc_t10.eval_local --pred outputs/dev_ensU5_sub.jsonl --gold data/dev_gold.jsonl --data-root "$DATA_ROOT" --hard-subset 2>&1 | grep -E "HARD subset|score|macro_f1|pem|SCORE"
echo "### GEN_DEV_PREDS DONE $(date) ###"
