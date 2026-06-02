#!/usr/bin/env bash
# Remote GPU server — LoRA SFT launcher.
#
# TWO MODES (select via MODE env or positional arg):
#
#   1) baseline  (default): plain mean-CE, standard ms-swift sampler.
#        bash scripts/train.sh
#        bash scripts/train.sh baseline
#
#   2) softmin: SoftMin PEM-bottleneck loss + ParagraphGroupSampler.
#        bash scripts/train.sh softmin
#
#      The softmin run requires the ParagraphGroupSampler (batches = one complete
#      paragraph) which lives in scripts/train_softmin.py. That launcher monkeypatches
#      swift.trainers.Seq2SeqTrainer -> SoftMinTrainer BEFORE swift builds the trainer.
#      Key CLI overrides (per_device_train_batch_size 1, packing false, padding_free false)
#      are passed automatically; hyperparameters come from env vars (see below).
#
# PREREQUISITE: build data files first:
#   uv run python -m nlpcc_t10.build_dataset \
#       --data-root "$DATA_ROOT" --out data/ \
#       --minority-oversample 1.0 --supported-downsample 1.0
#
# ENV VARS (override defaults before calling):
#   DATA_ROOT            path to the official data repo (default: ../NLPCC-2026-Task10-Science)
#   CUDA_VISIBLE_DEVICES which GPUs (default: 0,1 for 2xL40)
#   NPROC_PER_NODE       number of GPUs for DDP (default: 2)
#   IMAGE_MAX_TOKEN_NUM  visual tokens per image for Qwen3-VL (default: 1024)
#   SOFTMIN_BETA         softmin temperature (default: 5.0)
#   SOFTMIN_LAMBDA       softmin mix weight  (default: 0.5)
#   SOFTMIN_WARMUP_FRAC  fraction of training for beta warmup (default: 0.0 = off)
#   SOFTMIN_BETA_MIN     starting beta for warmup (default: 1.0)
#   Any extra args after the mode are forwarded to swift sft / train_softmin.py.

set -euo pipefail

export DATA_ROOT="${DATA_ROOT:-../NLPCC-2026-Task10-Science}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-2}"
# IMAGE_MAX_TOKEN_NUM is read by the Qwen3-VL processor via ms-swift's env passthrough.
export IMAGE_MAX_TOKEN_NUM="${IMAGE_MAX_TOKEN_NUM:-1024}"

MODE="${1:-baseline}"
shift || true   # consume the mode arg; remaining args are forwarded

case "$MODE" in
  baseline)
    echo "[train.sh] MODE=baseline — plain CE, standard sampler, 2xL40 DDP"
    # ms-swift's torchrun-based DDP launcher: set NPROC_PER_NODE before calling swift sft.
    # On some ms-swift versions the flag is --nproc_per_node; check `swift sft -h`.
    NPROC_PER_NODE="${NPROC_PER_NODE}" \
    uv run swift sft \
      --config configs/qwen3vl_lora_sft.yaml \
      "$@"
    ;;

  softmin)
    # SoftMin hyperparameter defaults (overridable via env).
    export SOFTMIN_BETA="${SOFTMIN_BETA:-5.0}"
    export SOFTMIN_LAMBDA="${SOFTMIN_LAMBDA:-0.5}"
    export SOFTMIN_WARMUP_FRAC="${SOFTMIN_WARMUP_FRAC:-0.0}"
    export SOFTMIN_BETA_MIN="${SOFTMIN_BETA_MIN:-1.0}"

    echo "[train.sh] MODE=softmin — SoftMin PEM loss, ParagraphGroupSampler, 2xL40 DDP"
    echo "           SOFTMIN_BETA=${SOFTMIN_BETA}  SOFTMIN_LAMBDA=${SOFTMIN_LAMBDA}"
    echo "           SOFTMIN_WARMUP_FRAC=${SOFTMIN_WARMUP_FRAC}  SOFTMIN_BETA_MIN=${SOFTMIN_BETA_MIN}"
    echo "           IMAGE_MAX_TOKEN_NUM=${IMAGE_MAX_TOKEN_NUM}"

    NPROC_PER_NODE="${NPROC_PER_NODE}" \
    uv run python scripts/train_softmin.py \
      --config configs/qwen3vl_lora_sft.yaml \
      --loss_type softmin_pem \
      --custom_register_path src/nlpcc_t10/swift_softmin/register.py \
      --per_device_train_batch_size 1 \
      --packing false \
      --padding_free false \
      "$@"
    ;;

  ablation-lambda0)
    # Mandatory ablation: softmin with lambda=0 == plain CE through the softmin path.
    # Useful to confirm the monkeypatch / sampler work without the softmin term changing loss.
    export SOFTMIN_LAMBDA=0.0
    export SOFTMIN_BETA="${SOFTMIN_BETA:-5.0}"
    echo "[train.sh] MODE=ablation-lambda0 — softmin path with lambda=0 (=plain CE baseline)"
    NPROC_PER_NODE="${NPROC_PER_NODE}" \
    uv run python scripts/train_softmin.py \
      --config configs/qwen3vl_lora_sft.yaml \
      --loss_type softmin_pem \
      --custom_register_path src/nlpcc_t10/swift_softmin/register.py \
      --per_device_train_batch_size 1 \
      --packing false \
      --padding_free false \
      "$@"
    ;;

  *)
    echo "Unknown mode: $MODE" >&2
    echo "Usage: bash scripts/train.sh [baseline|softmin|ablation-lambda0] [extra swift args...]" >&2
    exit 1
    ;;
esac
