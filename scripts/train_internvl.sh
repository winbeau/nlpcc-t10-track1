#!/usr/bin/env bash
# Fine-tune InternVL3-8B-hf (transformers-NATIVE variant — avoids the trust_remote_code breakage
# on transformers 5.8.1) per-sentence with the softmin_pem loss, then infer testp1 -> aggregate
# -> validated zip. This is the first NON-Qwen, architecture-DIVERSE member of the union fleet
# (InternViT visual encoder + different connector/LLM -> uncorrelated errors -> bigger union payoff).
#
#   bash scripts/train_internvl.sh                 # full 1-epoch train+infer+zip on GPU 1
#   MAX_STEPS=8 SMOKE=1 bash scripts/train_internvl.sh   # smoke: 8 steps, no save/infer
#
# CRITICAL knobs (validated the hard way — ~6 OOM/error cycles + a recon workflow to pin the ROOT cause):
#   - **THE fix = cap image tiles: INTERNVL_MAX_PATCHES=2** (see the tile-cap block below). ROOT CAUSE of the
#     OOM is NOT the LLM size and NOT missing checkpointing — it is the TILE COUNT. ms-swift hardcodes
#     crop_to_patches=True (template/templates/internvl.py:238) and leaves max_patches=12 -> up to ~3072 image
#     tokens/image -> InternViT activations explode to ~134 GiB + a 37.65 GiB spike on tile-heavy records -> OOM
#     on 1 H200 at ~step 450. 12->2 tiles ≈ 6x fewer image tokens -> ~30-50 GiB -> fits one H200 with room.
#   - MAX_PIXELS is INERT for InternVL-hf (Qwen pixel-budget knob; InternVL tiles via max_patches). 401408 vs
#     200704 gave byte-identical OOM. Tile count, not pixels, is the lever.
#   - vit_gradient_checkpointing / gradient_checkpointing did NOT fix it (both on; peak stayed ~134 GiB) — proof
#     the wall is the sheer number of image tokens, not un-checkpointed activations. Kept on (harmless).
#   - 8B works once tiles are capped. 2B was a stopgap (squeaks under the cap at 12 tiles); 8B@2-tiles is better.
#     For FULL-RES 12-tile 8B you'd need multi-GPU sequence/tensor parallelism (the textbook answer for a model
#     that genuinely exceeds one card) — capping tiles is the cheap single-GPU win and fine for a union member.
#   - max_length 4096: at 2048 some records' tokens exceed it -> ms-swift "Failed to retrieve dataset" ValueError.
#   - --use_logits_to_keep FALSE (the -hf path doesn't support it); HF_HOME must be our writable cache; lr 5e-5.
#
# Env (overridable): GPUS(1), INTERNVL_MAX_PATCHES(2), MAX_LENGTH(4096), LR(5e-5), MASTER_PORT(29553),
#                    BETA(5), LAMBDA(0.5), EPOCHS(1), TAG(internvl8b), MODEL_ID(OpenGVLab/InternVL3-8B-hf), DATA_ROOT.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$REPO_ROOT"
DATA_ROOT="${DATA_ROOT:-$(dirname "$REPO_ROOT")/NLPCC-2026-Task10-Science}"
export MODELSCOPE_CACHE="${MODELSCOPE_CACHE:-$(dirname "$REPO_ROOT")/ms_cache}"
export HF_HOME="${HF_HOME:-$(dirname "$REPO_ROOT")/hf_home}"  # MUST be writable (default is read-only)
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
MODEL="${MODEL_ID:-OpenGVLab/InternVL3-8B-hf}"
GPUS="${GPUS:-1}"; NPROC=$(echo "$GPUS" | awk -F, '{print NF}')
MAXLEN="${MAX_LENGTH:-4096}"; MP="${MAX_PIXELS:-401408}"; LR="${LR:-5e-5}"
BETA="${BETA:-5}"; LAMBDA="${LAMBDA:-0.5}"; EPOCHS="${EPOCHS:-1}"
TAG="${TAG:-internvl8b}"; OUTDIR="outputs/p0_${TAG}"
PORT="${MASTER_PORT:-29553}"   # override when another run holds the default (EADDRINUSE)
STEP_ARGS=(); [ -n "${MAX_STEPS:-}" ] && STEP_ARGS=(--max_steps "$MAX_STEPS" --save_strategy no)

# === THE fix for InternVL OOM: cap image tiles ===
# ms-swift hardcodes crop_to_patches=True (swift/template/templates/internvl.py:238) so the cached
# config's crop_to_patches=false is IGNORED -> default max_patches=12 tiles/image -> InternViT activations
# explode to ~134 GiB -> OOM on 1 H200. ms-swift does NOT override max_patches, so editing it in the
# processor config DOES take effect. 12->2 tiles ≈ 6x fewer image tokens -> ~30-50 GiB -> fits 1 H200.
# (For full-res 12-tile 8B you'd need multi-GPU sequence/tensor parallelism; 2 tiles is the cheap single-GPU win.)
INTERNVL_MAX_PATCHES="${INTERNVL_MAX_PATCHES:-2}"
PP_CFG="$MODELSCOPE_CACHE/models/$MODEL/preprocessor_config.json"
if [ -f "$PP_CFG" ]; then
  python3 -c "import json,sys; p=sys.argv[1]; d=json.load(open(p)); d['max_patches']=int(sys.argv[2]); json.dump(d,open(p,'w'),indent=2); print('[tile-cap] set max_patches=%s in %s' % (d['max_patches'], p))" "$PP_CFG" "$INTERNVL_MAX_PATCHES"
else
  echo "[tile-cap] WARN: $PP_CFG not found (model not yet downloaded?); max_patches uncapped -> OOM risk"
fi

echo "### INTERNVL TRAIN $TAG | model=$MODEL | softmin_pem | max_patches=$INTERNVL_MAX_PATCHES | max_length=$MAXLEN | lr=$LR | GPUs=$GPUS | port=$PORT | epochs=$EPOCHS ${MAX_STEPS:+| SMOKE max_steps=$MAX_STEPS} ###"
CUDA_VISIBLE_DEVICES="$GPUS" MAX_PIXELS="$MP" SOFTMIN_BETA="$BETA" SOFTMIN_LAMBDA="$LAMBDA" \
PYTHONPATH=src uv run torchrun --nproc_per_node="$NPROC" --master_port="$PORT" \
  scripts/train_softmin.py \
  --model "$MODEL" --tuner_type lora --torch_dtype bfloat16 \
  --dataset "${DATASET:-data/train_sft.jsonl}" --split_dataset_ratio 0 \
  --loss_type softmin_pem \
  --num_train_epochs "$EPOCHS" --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 1 --learning_rate "$LR" \
  --lora_rank 16 --lora_alpha 32 --freeze_vit true \
  --max_length "$MAXLEN" --max_pixels "$MP" --attn_impl sdpa \
  --gradient_checkpointing true --vit_gradient_checkpointing true \
  --packing false --padding_free false --use_logits_to_keep false \
  --eval_strategy no --save_strategy epoch --save_total_limit 1 \
  --logging_steps 20 --dataloader_num_workers 4 --output_dir "$OUTDIR" "${STEP_ARGS[@]}"

[ -n "${SMOKE:-}${NO_INFER:-}" ] && { echo "### train done (no infer: SMOKE/NO_INFER) ###"; exit 0; }

CKPT=$(ls -d "$OUTDIR"/v*/checkpoint-* 2>/dev/null | sort | tail -1)
[ -n "$CKPT" ] || { echo "ERROR: no checkpoint in $OUTDIR"; exit 1; }
echo "### trained -> $CKPT ; INFER testp1 (per-sentence, resolution-matched MAX_PIXELS=$MP) ###"
RAW="outputs/testp1_${TAG}_raw.jsonl"; SUB="submissions/testp1_${TAG}_submission.jsonl"
env "CUDA_VISIBLE_DEVICES=${GPUS%%,*}" "MAX_PIXELS=$MP" "HF_HOME=$HF_HOME" \
  PYTHONPATH=src uv run python -m nlpcc_t10.infer \
  --split testp1 --model "$MODEL" --adapter "$CKPT" --engine pt --data-root "$DATA_ROOT" --out "$RAW"
PYTHONPATH=src uv run python -m nlpcc_t10.aggregate \
  --pred "$RAW" --ref "$DATA_ROOT/data/testp1-track-1.jsonl" --out "$SUB"
uv run python scripts/make_submission_zip.py --sub "$SUB" \
  --ref "$DATA_ROOT/data/testp1-track-1.jsonl" --out "submissions/testp1_${TAG}_submission.zip"
echo "=== INTERNVL TRAIN+SUBMISSION DONE: submissions/testp1_${TAG}_submission.zip (add to union) ==="
