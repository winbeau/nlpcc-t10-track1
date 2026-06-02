#!/usr/bin/env bash
# Overnight recall-max queue (UNATTENDED, H200 GPU 0,1, grid resolution 401408).
#
# v2 workflow (ground-truthed from the 7 on-disk submissions) found testp1 RECALL-critical and the
# under-fired classes are SO (29 preds) and Contradiction (23), whose TRAINING TARGETS are starved
# (Contra 101, SO 188 vs UE 544). So the top lever is PER-CLASS oversample targeting SO/Contra,
# NOT uniform oversample (which inflates UE, already adequate at 92). Supersedes the old os5/ep2.
#
# os4 (uniform 4.0x) is already running separately (sunk cost — will be submitted). This queue runs,
# after os4 frees the GPU:
#   perclass     : Contra 6x, SO 5x, UCM 3x, UE 2x, ds 0.6, 1 epoch
#   perclass_ep2 : same per-class, 2 epochs (more fit on the calibrated data)
# Each = build_dataset (per-class) -> retrain_fullres.sh (train + testp1 infer + aggregate + zip).
# Does NOT git-commit (the /loop commits finished zips). Log: /tmp/overnight_queue.log
set -uo pipefail
cd /data/chenjiayu/wenbiao_zhao/nlpcc-t10-track1
export DATA_ROOT=/data/chenjiayu/wenbiao_zhao/NLPCC-2026-Task10-Science
export MODELSCOPE_CACHE=/data/chenjiayu/wenbiao_zhao/ms_cache

PC="Contradiction:6,Scope Overgeneralization:5,Unsupported Causal Mechanistic:3,Unsupported Entity:2"

wait_for_gpu() {  # block until GPU 0 has <20 GiB used (previous run fully done, incl its inference)
  echo "[queue] waiting for GPU 0 to free ... $(date)"
  while true; do
    u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sed -n '1p' | tr -d ' ')
    [ -n "$u" ] && [ "$u" -lt 20000 ] && break
    sleep 60
  done
  echo "[queue] GPU 0 free ($u MiB) -> proceeding $(date)"
}

run_pc() {  # $1=per-class spec  $2=supported-downsample  $3=epochs  $4=TAG
  echo "######## QUEUE STEP TAG=$4 per-class=[$1] ds=$2 epochs=$3 $(date) ########"
  PYTHONPATH=src uv run python -m nlpcc_t10.build_dataset --data-root "$DATA_ROOT" --out data \
    --minority-oversample 1.0 --supported-downsample "$2" \
    --minority-oversample-per-class "$1" 2>&1 | tail -10
  TAG="$4" EPOCHS="$3" MAX_PIXELS=401408 GPUS=0,1 bash scripts/retrain_fullres.sh
  echo "######## QUEUE STEP DONE TAG=$4 $(date) ########"
}

wait_for_gpu; run_pc "$PC" 0.6 1 perclass_b5_l0.5     # per-class oversample, 1 epoch (TOP rec)
wait_for_gpu; run_pc "$PC" 0.6 2 perclass_ep2_b5_l0.5 # same, 2 epochs
echo "######## OVERNIGHT QUEUE COMPLETE $(date) ########"
