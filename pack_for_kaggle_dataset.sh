#!/usr/bin/env bash
# 将本目录打包成「待上传 Kaggle Dataset」的干净副本（不含训练产物与 .git）。
# 用法: bash pack_for_kaggle_dataset.sh [输出目录]
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT_DIR="${1:-"${ROOT_DIR}/../RIGDNet_Ablation_kaggle_upload"}"

mkdir -p "${OUT_DIR}"

rsync -a \
  --exclude ".git/" \
  --exclude ".venv/" \
  --exclude "__pycache__/" \
  --exclude "*.pyc" \
  --exclude "runs/" \
  --exclude "runs_*/" \
  --exclude "checkpoints/" \
  --exclude "checkpoints_*/" \
  --exclude "generated_configs/" \
  --exclude "results/" \
  --exclude "results_*/" \
  "${ROOT_DIR}/" "${OUT_DIR}/"

echo "已生成上传目录: ${OUT_DIR}"
echo "下一步见 README「通过 Kaggle CLI 上传为 Dataset」。"
