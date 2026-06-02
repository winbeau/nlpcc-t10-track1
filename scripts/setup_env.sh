#!/usr/bin/env bash
# 仅在远程 GPU 服务器运行。功能: 建环境 + 拉 LFS 图片 + 解压。
# 本地不要跑（约束: 不在本地下权重 / sync / 解压）。
set -euo pipefail

DATA_ROOT="${DATA_ROOT:-../NLPCC-2026-Task10-Science}"

echo "[1/4] 安装匹配 CUDA 的 torch（按服务器 CUDA 改 index-url）"
# 例: CUDA 12.1 -> cu121。先确认 nvidia-smi / nvcc 版本。
uv pip install torch --index-url https://download.pytorch.org/whl/cu121 || \
  echo "  跳过/已存在 torch，按需手动安装"

echo "[2/4] uv sync --extra train（GPU 栈: ms-swift / transformers / peft ...）"
uv sync --extra train

echo "[3/4] git lfs pull 官方图片（在官方仓库内）"
( cd "$DATA_ROOT" && git lfs pull )

echo "[4/4] 解压图片到 images/（使 images/<sha>.jpg 可解析）"
mkdir -p "$DATA_ROOT/data/images"
unzip -nq "$DATA_ROOT/data/images.zip"        -d "$DATA_ROOT/data/images/"
unzip -nq "$DATA_ROOT/data/images-testp1.zip" -d "$DATA_ROOT/data/images/"

echo "完成。DATA_ROOT=$DATA_ROOT"
