#!/usr/bin/env bash
# One-shot server bootstrap: run ONCE after `uv sync --extra train`, from anywhere.
#
#   git clone git@github.com:winbeau/nlpcc-t10-track1.git
#   cd nlpcc-t10-track1
#   uv sync --extra train          # create the python env (torch/ms-swift/...)
#   bash scripts/setup_env.sh      # THIS script: deps + dataset + model (progress shown)
#   # then the pipeline (build_dataset -> train -> infer -> aggregate -> eval)
#
# Installs git-lfs + torchvision (matched to torch cu130; needed by qwen_vl_utils), clones the
# official data repo as a SIBLING, git-lfs-pulls the FULL dataset + unzips images, and downloads
# the Qwen3-VL model into a sibling ms_cache. Idempotent (safe to re-run) and PATH-PORTABLE
# (everything is relative to this repo, so it works on any host / any base folder).
#
# Progress bars are intentionally NOT silenced (git-lfs pull + modelscope download print live
# progress). For long pulls, run inside tmux and watch the session.
#
# Overridable via env: DATA_REPO, MODEL_ID, MODELSCOPE_CACHE, DATA_ROOT, TORCH_INDEX.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BASE_DIR="$(dirname "$REPO_ROOT")"                       # the parent folder (e.g. .../wenbiao_zhao)
DATA_ROOT="${DATA_ROOT:-$BASE_DIR/NLPCC-2026-Task10-Science}"
export MODELSCOPE_CACHE="${MODELSCOPE_CACHE:-$BASE_DIR/ms_cache}"
DATA_REPO="${DATA_REPO:-https://github.com/winbeau/NLPCC-2026-Task10-Science.git}"
MODEL_ID="${MODEL_ID:-Qwen/Qwen3-VL-8B-Instruct}"
TORCH_INDEX="${TORCH_INDEX:-https://download.pytorch.org/whl/cu130}"

cd "$REPO_ROOT"
echo "=================================================================="
echo " REPO_ROOT        = $REPO_ROOT"
echo " DATA_ROOT        = $DATA_ROOT   (sibling, official data repo)"
echo " MODELSCOPE_CACHE = $MODELSCOPE_CACHE"
echo " MODEL_ID         = $MODEL_ID"
echo "=================================================================="

echo "[1/5] git-lfs"
if ! command -v git-lfs >/dev/null 2>&1; then
  echo "  installing git-lfs via apt (needs root) ..."
  apt-get update -qq && apt-get install -y git-lfs
fi
git lfs version

echo "[2/5] torchvision (matched to torch cu130; required by qwen_vl_utils)"
if ! uv run python -c "import torchvision" 2>/dev/null; then
  uv pip install torchvision --index-url "$TORCH_INDEX"
fi
uv run python -c "import torch,torchvision; print('  torch',torch.__version__,'| torchvision',torchvision.__version__,'| cuda',torch.cuda.is_available(),'| n_gpu',torch.cuda.device_count())"

echo "[3/5] clone official data repo (sibling) -> $DATA_ROOT"
if [ ! -d "$DATA_ROOT/.git" ]; then
  git clone "$DATA_REPO" "$DATA_ROOT"   # HTTPS -> uses the stored token credential
else
  echo "  already cloned."
fi

echo "[4/5] git lfs pull (FULL dataset; progress shown) + unzip images"
( cd "$DATA_ROOT" && git lfs install && git lfs pull )
mkdir -p "$DATA_ROOT/data/images"
unzip -nq "$DATA_ROOT/data/images.zip"        -d "$DATA_ROOT/data/images/"
unzip -nq "$DATA_ROOT/data/images-testp1.zip" -d "$DATA_ROOT/data/images/"
echo "  images extracted: $(ls "$DATA_ROOT/data/images" | wc -l) files"

echo "[5/5] download model $MODEL_ID -> $MODELSCOPE_CACHE (progress shown)"
uv run modelscope download --model "$MODEL_ID"

echo "=================================================================="
echo " BOOTSTRAP DONE."
echo " For training/build, export:"
echo "   export DATA_ROOT=$DATA_ROOT"
echo "   export MODELSCOPE_CACHE=$MODELSCOPE_CACHE"
echo " Next: uv run python -m nlpcc_t10.build_dataset --data-root \"\$DATA_ROOT\" --out data ..."
echo "=================================================================="
