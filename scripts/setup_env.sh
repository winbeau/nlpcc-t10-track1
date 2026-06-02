#!/usr/bin/env bash
# Remote GPU server ONLY — one-time environment setup.
# Installs the training stack (ms-swift / transformers / peft / torch cu130),
# pulls LFS images, and unpacks them. Do NOT run locally.
#
# ENV: torch 2.12+cu130 (CUDA 13.0) on 2xL40. uv installs the matching torch wheel
#      from the cu130 index automatically via pyproject.toml or an explicit step below.
#
# Usage:
#   DATA_ROOT=../NLPCC-2026-Task10-Science bash scripts/setup_env.sh
set -euo pipefail

DATA_ROOT="${DATA_ROOT:-../NLPCC-2026-Task10-Science}"

# ---------------------------------------------------------------------------
# Step 1: Install the training stack (ms-swift, transformers 5.8.1, peft,
#         accelerate, qwen-vl-utils, and torch cu130 from the PyTorch cu130 index).
#         uv handles the torch wheel selection via the [train] optional-dependency
#         group; the cu130 index is specified in pyproject.toml or via --index below.
# ---------------------------------------------------------------------------
echo "[1/4] uv sync --extra train (torch cu130 + ms-swift 4.2.3 + transformers 5.8.1 ...)"
# uv resolves torch from the cu130 index. If pyproject.toml does not yet pin
# the cu130 index, add it here explicitly:
uv sync --extra train \
  --index "https://download.pytorch.org/whl/cu130" \
  || { echo "  [WARN] uv sync failed; retrying without extra index (torch may already be installed)"; \
       uv sync --extra train; }

# Verify torch is importable and reports the right CUDA build.
uv run python -c "import torch; print(f'  torch {torch.__version__}  CUDA available: {torch.cuda.is_available()}')" \
  || echo "  [WARN] torch import check failed; continue and verify manually"

# ---------------------------------------------------------------------------
# Step 2: (Optional) Install flash-attn for faster attention on L40.
#         Requires a pre-built wheel matching torch+CUDA; comment out if unavailable.
# ---------------------------------------------------------------------------
# echo "[1b/4] pip install flash-attn (optional; comment out if not available)"
# uv run pip install flash-attn --no-build-isolation || echo "  [WARN] flash-attn install failed; using sdpa"

# ---------------------------------------------------------------------------
# Step 3: Pull LFS images from the official data repo.
# ---------------------------------------------------------------------------
echo "[2/4] git lfs pull — official dataset images (run inside the data repo)"
( cd "$DATA_ROOT" && git lfs pull )

# ---------------------------------------------------------------------------
# Step 4: Unpack images.
# ---------------------------------------------------------------------------
echo "[3/4] Unzip images to $DATA_ROOT/data/images/"
mkdir -p "$DATA_ROOT/data/images"
# -n: never overwrite existing files (safe to re-run); -q: quiet.
unzip -nq "$DATA_ROOT/data/images.zip"        -d "$DATA_ROOT/data/images/"
unzip -nq "$DATA_ROOT/data/images-testp1.zip" -d "$DATA_ROOT/data/images/"

# ---------------------------------------------------------------------------
# Step 5: Build the sentence-level dataset (optional; run separately to control params).
# ---------------------------------------------------------------------------
echo "[4/4] Environment ready."
echo "      DATA_ROOT=$DATA_ROOT"
echo ""
echo "      Next step (sentence-level dataset):"
echo "        uv run python -m nlpcc_t10.build_dataset \\"
echo "            --data-root \"$DATA_ROOT\" --out data/ \\"
echo "            --minority-oversample 1.0 --supported-downsample 1.0"
echo ""
echo "      Then train:"
echo "        bash scripts/train.sh baseline    # plain CE"
echo "        bash scripts/train.sh softmin     # SoftMin PEM loss"
