#!/usr/bin/env bash
# E1 = s01 recipe + CoT <analysis> (cot_sft_plan.md §1-3). Trains the SoftMin-PEM LoRA on the
# CoT-augmented sentence-level set, then infers testp1 + the devbench(densematch) dev for G2.
#
# WHY a dedicated script (not retrain_fullres.sh): that one HARD-CODES use_logits_to_keep=true,
# data/train_sft.jsonl, and max_pixels=802816 (the 1024-tok fullres variant). E1 must match s01
# EXACTLY — use_logits_to_keep=FALSE, max_pixels=401408 (grid res), the CoT dataset — and only
# bump max_length (+128 for the ~60-token <analysis>). Every other hyperparameter is s01's.
#
# Variable that E1 changes vs s01: ONLY the training TARGET (+<analysis>). softmin/oversample/
# resolution/epoch/DDP are identical, so a G2 win is attributable to CoT (roadmap §4 red line 3).
#
#   GPUS=0,1 bash scripts/train_e1.sh
#
# Env (overridable): GPUS(0,1), DATASET(data/cot/e1/train_sft.jsonl), MAX_PIXELS(401408),
#                    MAX_LENGTH(10368), BETA(5), LAMBDA(0.5), EPOCHS(1), TAG(cotE1_s01cot),
#                    MODEL_ID(Qwen/Qwen3-VL-8B-Instruct), DATA_ROOT, MASTER_PORT(29536).
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$REPO_ROOT"
DATA_ROOT="${DATA_ROOT:-$(dirname "$REPO_ROOT")/NLPCC-2026-Task10-Science}"
export MODELSCOPE_CACHE="${MODELSCOPE_CACHE:-$(dirname "$REPO_ROOT")/ms_cache}"
export USE_HF="${USE_HF:-0}"   # base in ModelScope ms_cache (h200-box-config gotcha)
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

MODEL="${MODEL_ID:-Qwen/Qwen3-VL-8B-Instruct}"
GPUS="${GPUS:-0,1}"; NPROC=$(echo "$GPUS" | awk -F, '{print NF}')
DATASET="${DATASET:-data/cot/e1/train_sft.jsonl}"
MP="${MAX_PIXELS:-401408}"            # s01 grid resolution (fullres↑ already shown harmful)
MAXLEN="${MAX_LENGTH:-10368}"         # s01 10240 + 128 for the <analysis> tokens
BETA="${BETA:-5}"; LAMBDA="${LAMBDA:-0.5}"; EPOCHS="${EPOCHS:-1}"
TAG="${TAG:-cotE1_s01cot}"
PORT="${MASTER_PORT:-29536}"
OUTDIR="outputs/${TAG}"

[ -f "$DATASET" ] || { echo "ERROR: dataset $DATASET missing — run build_dataset --cot first"; exit 1; }
echo "### E1 TRAIN $TAG | s01 recipe + <analysis> | use_logits_to_keep=FALSE | max_pixels=$MP | max_length=$MAXLEN | GPUs=$GPUS | epochs=$EPOCHS ###"
echo "### dataset=$DATASET ($(wc -l < "$DATASET") rows) ###"

CUDA_VISIBLE_DEVICES="$GPUS" SOFTMIN_BETA="$BETA" SOFTMIN_LAMBDA="$LAMBDA" \
PYTHONPATH=src uv run torchrun --nproc_per_node="$NPROC" --master_port="$PORT" \
  scripts/train_softmin.py \
  --model "$MODEL" --tuner_type lora --torch_dtype bfloat16 \
  --dataset "$DATASET" --split_dataset_ratio 0 \
  --loss_type softmin_pem \
  --num_train_epochs "$EPOCHS" --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 1 --learning_rate 1e-4 \
  --lora_rank 16 --lora_alpha 32 --freeze_vit true \
  --max_length "$MAXLEN" --max_pixels "$MP" --attn_impl sdpa \
  --packing false --padding_free false --use_logits_to_keep false \
  --eval_strategy no --save_strategy epoch --save_total_limit 1 \
  --logging_steps 20 --dataloader_num_workers 0 --output_dir "$OUTDIR"

CKPT=$(ls -d "$OUTDIR"/v*/checkpoint-* 2>/dev/null | sort | tail -1)
[ -n "$CKPT" ] || { echo "ERROR: no checkpoint in $OUTDIR"; exit 1; }
echo "### E1 trained -> $CKPT ; INFER testp1 + dev(densematch) ###"

# testp1 submission (CoT decode: bigger max-new-tokens for <analysis>; resolution-matched).
RAW="outputs/testp1_${TAG}_raw.jsonl"; SUB="submissions/testp1_${TAG}_submission.jsonl"
CUDA_VISIBLE_DEVICES="${GPUS%%,*}" MAX_PIXELS="$MP" PYTHONPATH=src uv run python -m nlpcc_t10.infer \
  --split testp1 --model "$MODEL" --adapter "$CKPT" --engine pt --data-root "$DATA_ROOT" \
  --max-new-tokens 160 --out "$RAW"
PYTHONPATH=src uv run python -m nlpcc_t10.aggregate \
  --pred "$RAW" --ref "$DATA_ROOT/data/testp1-track-1.jsonl" --out "$SUB"
uv run python scripts/make_submission_zip.py --sub "$SUB" \
  --ref "$DATA_ROOT/data/testp1-track-1.jsonl" --out "submissions/testp1_${TAG}_submission.zip"

# devbench(densematch) dev for the G2 offline gate (uses data/cot/e1/split.json's dev = image-disjoint).
DEVRAW="outputs/dev_${TAG}_raw.jsonl"; DEVSUB="outputs/dev_${TAG}_submission.jsonl"
DENSE="data/devbench/dev_gold_densematch.jsonl"
if [ -f data/cot/e1/split.json ]; then cp -f data/cot/e1/split.json data/split.json; fi
CUDA_VISIBLE_DEVICES="${GPUS%%,*}" MAX_PIXELS="$MP" PYTHONPATH=src uv run python -m nlpcc_t10.infer \
  --split dev --model "$MODEL" --adapter "$CKPT" --engine pt --data-root "$DATA_ROOT" \
  --max-new-tokens 160 --out "$DEVRAW" || echo "WARN: dev infer failed (non-fatal)"
if [ -f "$DEVRAW" ]; then
  PYTHONPATH=src uv run python -m nlpcc_t10.aggregate \
    --pred "$DEVRAW" --ref "$DATA_ROOT/data/traindev-track-1.jsonl" --out "$DEVSUB" || true
  if [ -f "$DENSE" ] && [ -f "$DEVSUB" ]; then
    echo "### G2 densematch eval (E1) — same-regime ordering only, NOT an absolute score ###"
    uv run python "$DATA_ROOT/offline_eval/evaluate.py" --track 1 \
      --gold "$DENSE" --pred "$DEVSUB" --match id || echo "WARN: densematch eval failed"
  fi
fi
echo "=== E1 DONE: $CKPT ; submission submissions/testp1_${TAG}_submission.zip ==="
