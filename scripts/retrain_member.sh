#!/usr/bin/env bash
# Retrain ONE Qwen per-sentence member as PLAIN-CE (SOFTMIN_LAMBDA=0) on FULL traindev with the member's
# original oversample recipe (grid res 401408), then infer testp1 -> {id,labels}. The validated upgrade
# (plain-CE > softmin, +4.47 testp1 on the clean head-to-head s31 vs s32) applied to the full s15 recipe.
# Usage: TAG GPU then build_dataset oversample args:
#   bash scripts/retrain_member.sh grid     3 --minority-oversample 3.0 --supported-downsample 0.6
#   bash scripts/retrain_member.sh os4      4 --minority-oversample 4.0 --supported-downsample 0.66
#   bash scripts/retrain_member.sh perclass 5 --supported-downsample 0.6 --minority-oversample-per-class "Contradiction:6,Scope Overgeneralization:5,Unsupported Causal Mechanistic:3,Unsupported Entity:2"
set -uo pipefail
cd /data/chenjiayu/wenbiao_zhao/nlpcc-t10-track1
export DATA_ROOT=/data/chenjiayu/wenbiao_zhao/NLPCC-2026-Task10-Science
export MODELSCOPE_CACHE=/data/chenjiayu/wenbiao_zhao/ms_cache USE_HF=0 PYTHONPATH=src TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
TAG=$1; GPU=$2; shift 2; BUILD_ARGS=("$@")
M=Qwen/Qwen3-VL-8B-Instruct; MP=${MAX_PIXELS:-401408}; REF="$DATA_ROOT/data/testp1-track-1.jsonl"
DD=data/full_$TAG; OUT=outputs/pce_$TAG
mkdir -p $OUT
rm -f $OUT/DONE $OUT/FAILED
# NOTE: no `trap ... ERR` — with pipefail the pre-train `ls checkpoint-*` (no match) returns non-zero
# and would spuriously touch FAILED. Real completion = the testp1_pce_<tag>.jsonl + DONE at the end;
# a real crash leaves NO DONE and a dead process (detectable by the poller).

echo "### [$TAG] build full-traindev dataset (plain-CE, oversample) $(date) ###"
PYTHONPATH=src uv run python -m nlpcc_t10.build_dataset --data-root "$DATA_ROOT" --out $DD "${BUILD_ARGS[@]}"

CKPT=$(ls -d $OUT/ckpt/v*/checkpoint-* 2>/dev/null | sort | tail -1)
if [ -z "$CKPT" ]; then
  echo "### [$TAG] TRAIN plain-CE (SOFTMIN_LAMBDA=0) GPU$GPU $(date) ###"
  PORT=$((29600+GPU))
  CUDA_VISIBLE_DEVICES=$GPU SOFTMIN_BETA=5 SOFTMIN_LAMBDA=0 PYTHONPATH=src uv run torchrun --nproc_per_node=1 --master_port=$PORT \
    scripts/train_softmin.py --model "$M" --tuner_type lora --torch_dtype bfloat16 \
    --dataset $DD/train_sft.jsonl --split_dataset_ratio 0 --loss_type softmin_pem \
    --num_train_epochs 1 --per_device_train_batch_size 1 --gradient_accumulation_steps 1 --learning_rate 1e-4 \
    --lora_rank 16 --lora_alpha 32 --freeze_vit true --max_length 10240 --max_pixels $MP --attn_impl sdpa \
    --packing false --padding_free false --use_logits_to_keep false \
    --eval_strategy no --save_strategy epoch --save_total_limit 1 --logging_steps 20 --dataloader_num_workers 0 \
    --output_dir $OUT/ckpt
  CKPT=$(ls -d $OUT/ckpt/v*/checkpoint-* | sort | tail -1)
fi
[ -n "$CKPT" ] || { echo "ERROR: no ckpt for $TAG"; exit 1; }

echo "### [$TAG] INFER testp1 $(date) -> ckpt $CKPT ###"
CUDA_VISIBLE_DEVICES=$GPU MAX_PIXELS=$MP PYTHONPATH=src uv run python -m nlpcc_t10.infer \
  --split testp1 --model "$M" --adapter "$CKPT" --engine pt --data-root "$DATA_ROOT" --out outputs/testp1_pce_${TAG}_raw.jsonl
PYTHONPATH=src uv run python -m nlpcc_t10.aggregate --pred outputs/testp1_pce_${TAG}_raw.jsonl --ref "$REF" --out outputs/testp1_pce_${TAG}.jsonl
touch $OUT/DONE
echo "=== MEMBER $TAG DONE $(date) -> outputs/testp1_pce_${TAG}.jsonl ==="
