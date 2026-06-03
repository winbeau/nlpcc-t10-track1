#!/usr/bin/env bash
# GATE-A (the cheap, MANDATORY first gate of the corrector pipeline). Train ONE s01-recipe model on the
# IMAGE-DISJOINT (component_aware) train', then infer the held-out val' and score devhard' (>=6 sent &
# >=1 gold minority, ~68 records) AND full-val'. DECIDES whether the image-dedup dev TRACKS testp1 (~50)
# or is still leaky/domain-easy (~88) — the go/no-go before any OOF + corrector spend.
#   PASS  : devhard' in [40,62] AND full-val < 70  -> proxy tracks testp1, corrector is validatable.
#   STOP  : devhard' > 75 OR full-val > 80         -> still leaky / domain dominates, abandon corrector.
#   MURKY : devhard' in [62,75]                     -> weak signal, at most ONE diagnostic, no k-fold.
# PREREQ: build the dedup dataset FIRST (overwrites data/train_sft.jsonl + data/dev_gold.jsonl + data/split.json):
#   cp data/split.json data/split_leaky.json
#   uv run python -m nlpcc_t10.build_dataset --data-root "$DATA_ROOT" --out data/ \
#     --split-mode component_aware --val-ratio 0.15 --seed 99 --minority-oversample 3.0 --supported-downsample 0.6
# Env: GPUS(0,1) MAX_PIXELS(401408) TAG(dedup_gateA) MODEL_ID MASTER_PORT(29565) DATA_ROOT.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$REPO_ROOT"
DATA_ROOT="${DATA_ROOT:-$(dirname "$REPO_ROOT")/NLPCC-2026-Task10-Science}"
export MODELSCOPE_CACHE="${MODELSCOPE_CACHE:-$(dirname "$REPO_ROOT")/ms_cache}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
MODEL="${MODEL_ID:-Qwen/Qwen3-VL-8B-Instruct}"
GPUS="${GPUS:-0,1}"; NPROC=$(echo "$GPUS" | awk -F, '{print NF}')
MP="${MAX_PIXELS:-401408}"; TAG="${TAG:-dedup_gateA}"; OUTDIR="outputs/${TAG}"; PORT="${MASTER_PORT:-29565}"

# sanity: the dedup split must be present + image-disjoint flavour (val ~15%)
NVAL=$(python3 -c "import json;d=json.load(open('data/split.json'));print(sum(1 for v in d.values() if v=='dev'))" 2>/dev/null || echo 0)
echo "### GATE-A | s01 recipe (b5 l0.5, mp=$MP, ulk=false, 1ep) on dedup train' | dev(val')=$NVAL records | GPUs=$GPUS ###"
[ "$NVAL" -gt 300 ] || { echo "WARN: dev split only $NVAL records — did you build the dedup dataset (--split-mode component_aware)?"; }

CUDA_VISIBLE_DEVICES="$GPUS" SOFTMIN_BETA=5 SOFTMIN_LAMBDA=0.5 \
PYTHONPATH=src uv run torchrun --nproc_per_node="$NPROC" --master_port="$PORT" \
  scripts/train_softmin.py \
  --model "$MODEL" --tuner_type lora --torch_dtype bfloat16 \
  --dataset data/train_sft.jsonl --split_dataset_ratio 0 \
  --loss_type softmin_pem \
  --num_train_epochs 1 --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 4 --learning_rate 1e-4 \
  --lora_rank 16 --lora_alpha 32 --freeze_vit true \
  --max_length 10240 --max_pixels "$MP" --attn_impl sdpa \
  --packing false --padding_free false --use_logits_to_keep false \
  --eval_strategy no --save_strategy epoch --save_total_limit 1 \
  --logging_steps 20 --dataloader_num_workers 4 --output_dir "$OUTDIR"

CKPT=$(ls -d "$OUTDIR"/v*/checkpoint-* 2>/dev/null | sort | tail -1)
[ -n "$CKPT" ] || { echo "ERROR: no checkpoint in $OUTDIR"; exit 1; }
echo "### trained -> $CKPT ; INFER dev (val', image-disjoint) ###"
RAW="outputs/dev_${TAG}_raw.jsonl"; SUB="outputs/dev_${TAG}_sub.jsonl"
env "CUDA_VISIBLE_DEVICES=${GPUS%%,*}" "MAX_PIXELS=$MP" PYTHONPATH=src uv run python -m nlpcc_t10.infer \
  --split dev --model "$MODEL" --adapter "$CKPT" --engine pt --data-root "$DATA_ROOT" --out "$RAW"
PYTHONPATH=src uv run python -m nlpcc_t10.aggregate --pred "$RAW" --ref data/dev_gold.jsonl --out "$SUB"

echo "########## GATE-A SCORE ##########"
echo "--- devhard' (>=6 sentences AND >=1 gold minority) ---"
PYTHONPATH=src uv run python -m nlpcc_t10.eval_local --pred "$SUB" --gold data/dev_gold.jsonl --data-root "$DATA_ROOT" --hard-subset --min-sent 6 || true
echo "--- full val' ---"
PYTHONPATH=src uv run python -m nlpcc_t10.eval_local --pred "$SUB" --gold data/dev_gold.jsonl --data-root "$DATA_ROOT" || true
echo "########## GATE-A DECISION: devhard' in [40,62] AND full<70 => PASS (build corrector); devhard'>75 or full>80 => STOP (abandon corrector) ##########"
