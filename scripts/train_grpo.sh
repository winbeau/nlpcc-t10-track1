#!/usr/bin/env bash
# E1 -> GRPO (grpo_plan.md). per-sentence rollouts <analysis>...\n{"label"}; class-weighted reward.
# 2xH200 colocate vLLM. KNOWN-BUG WORKAROUNDS (grpo_plan §1, do not remove):
#   --vllm_enable_lora false  (Qwen3-VL LoRA shrink bug ms-swift #6670/#6506)
#   gradient_accumulation_steps <= 8  (#6521 vision-embed shape mismatch)
#   USE_HF=0  (base in ModelScope ms_cache, else hangs on HF download)
#
# SMOKE FIRST: `SMOKE=1 bash scripts/train_grpo.sh` runs ~8 steps on a tiny prompt slice to
# validate the colocate+reward+LoRA pipeline before committing GPU-hours. Only launch the full
# run after a clean smoke. GATE: do NOT run if E1 looks broken (high parse-fallback / densematch
# collapse) — see the overnight runbook.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$REPO_ROOT"
DATA_ROOT="${DATA_ROOT:-$(dirname "$REPO_ROOT")/NLPCC-2026-Task10-Science}"
export MODELSCOPE_CACHE="${MODELSCOPE_CACHE:-$(dirname "$REPO_ROOT")/ms_cache}"
export USE_HF="${USE_HF:-0}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

MODEL="${MODEL_ID:-Qwen/Qwen3-VL-8B-Instruct}"
GPUS="${GPUS:-0,1}"; NPROC=$(echo "$GPUS" | awk -F, '{print NF}')
ADAPTER="${ADAPTER:?set ADAPTER=outputs/cotE1_s01cot/v0-*/checkpoint-* (the E1 LoRA)}"
PROMPTS="${PROMPTS:-data/cot/grpo/prompts.jsonl}"
MP="${MAX_PIXELS:-401408}"
TAG="${TAG:-grpoE1}"
PORT="${MASTER_PORT:-29537}"
OUTDIR="outputs/${TAG}"
GEN="${NUM_GENERATIONS:-8}"
GA="${GRAD_ACCUM:-8}"            # <= 8 (#6521)
EPOCHS="${EPOCHS:-1}"

[ -f "$PROMPTS" ] || { echo "ERROR: $PROMPTS missing — run scripts/build_grpo_data.py"; exit 1; }

SMOKE_ARGS=()
if [ "${SMOKE:-0}" = "1" ]; then
  head -n 64 "$PROMPTS" > /tmp/grpo_smoke.jsonl
  PROMPTS=/tmp/grpo_smoke.jsonl
  TAG="${TAG}_smoke"; OUTDIR="outputs/${TAG}"
  SMOKE_ARGS=(--max_steps 8 --save_strategy no)
  echo "### GRPO SMOKE: 8 steps on 64 prompts (pipeline validation only) ###"
fi

echo "### GRPO $TAG | colocate vLLM | vllm_enable_lora=false | num_gen=$GEN grad_accum=$GA | adapter=$ADAPTER ###"
CUDA_VISIBLE_DEVICES="$GPUS" PYTHONPATH=src uv run torchrun --nproc_per_node="$NPROC" --master_port="$PORT" \
  -m swift.cli.rlhf \
  --rlhf_type grpo \
  --model "$MODEL" --adapters "$ADAPTER" \
  --train_type lora --torch_dtype bfloat16 \
  --dataset "$PROMPTS" --split_dataset_ratio 0 \
  --external_plugins scripts/grpo_reward.py --reward_funcs track1 \
  --num_generations "$GEN" --temperature 1.0 \
  --use_vllm true --vllm_mode colocate --vllm_enable_lora false \
  --offload_optimizer true --offload_model true --sleep_level 1 \
  --max_pixels "$MP" --attn_impl sdpa \
  --per_device_train_batch_size 1 --gradient_accumulation_steps "$GA" \
  --learning_rate 1e-6 --num_train_epochs "$EPOCHS" \
  --lora_rank 16 --lora_alpha 32 --freeze_vit true \
  --max_length 10368 --max_completion_length 160 \
  --logging_steps 2 --save_strategy "${SAVE_STRATEGY:-epoch}" --save_total_limit 1 \
  --dataloader_num_workers 0 --output_dir "$OUTDIR" \
  "${SMOKE_ARGS[@]}"
echo "=== GRPO $TAG DONE -> $OUTDIR ==="
