#!/usr/bin/env bash
# Overnight recall-max queue (runs UNATTENDED on H200 GPU 0,1, grid resolution 401408).
#
# testp1 is RECALL-critical (trivial all-Supported ~19; oversample 1.5x=40.9 < 3.0x=47.8 => push
# oversampling UP). This queues the next recall-max retrains AFTER the already-running os4
# (oversample 4.0). Each step waits for GPU 0 to free (previous run's train+infer done), then
# build_dataset -> retrain_fullres.sh (train + testp1 infer + aggregate + validated zip).
#
# Produces submissions/testp1_<TAG>_submission.zip per step. Does NOT git-commit (the /loop or the
# morning review commits + pushes the finished zips). Log: /tmp/overnight_queue.log
set -uo pipefail
cd /data/chenjiayu/wenbiao_zhao/nlpcc-t10-track1
export DATA_ROOT=/data/chenjiayu/wenbiao_zhao/NLPCC-2026-Task10-Science
export MODELSCOPE_CACHE=/data/chenjiayu/wenbiao_zhao/ms_cache

wait_for_gpu() {  # block until GPU 0 has <20 GiB used (previous run fully done, incl its inference)
  echo "[queue] waiting for GPU 0 to free ... $(date)"
  while true; do
    u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sed -n '1p' | tr -d ' ')
    [ -n "$u" ] && [ "$u" -lt 20000 ] && break
    sleep 60
  done
  echo "[queue] GPU 0 free ($u MiB) -> proceeding $(date)"
}

run() {  # $1=minority-oversample  $2=epochs  $3=TAG
  echo "######## QUEUE STEP TAG=$3 oversample=$1 epochs=$2 $(date) ########"
  PYTHONPATH=src uv run python -m nlpcc_t10.build_dataset --data-root "$DATA_ROOT" --out data \
    --minority-oversample "$1" --supported-downsample 0.66 2>&1 | tail -4
  TAG="$3" EPOCHS="$2" MAX_PIXELS=401408 GPUS=0,1 bash scripts/retrain_fullres.sh
  echo "######## QUEUE STEP DONE TAG=$3 $(date) ########"
}

# os4 (oversample 4.0) is already running separately; wait it out, then:
wait_for_gpu; run 5.0 1 os5_b5_l0.5     # push oversampling higher
wait_for_gpu; run 3.0 2 ep2_b5_l0.5     # more epochs on the best (3.0x) config (minority may be underfit at 1ep)
echo "######## OVERNIGHT QUEUE COMPLETE $(date) ########"
