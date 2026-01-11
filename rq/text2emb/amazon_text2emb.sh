#!/bin/bash
set -euo pipefail

export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1

DATASETS=(
  "All_Beauty"                  # (40, dim)
  "Arts_Crafts_and_Sewing"      # (8094, dim)
  "Industrial_and_Scientific"   # (3433, dim)
  "Office_Products"             #
  "Video_Games"
)

ROOT_PATH="../../data/Amazon2018"
EMB_CKPT="../../huggingface/Qwen3-Embedding-4B" # [0.6B, 4B] -> [1024, 2560]
MAX_SENT_LEN=2048
WORD_DROP_RATIO=-1
BSZ=8

if command -v nvidia-smi &>/dev/null; then
    NUM_GPUS=$(nvidia-smi --list-gpus | wc -l)
else
    echo "[WARN] 未找到 nvidia-smi，默认使用 1 个进程"
    NUM_GPUS=1
fi

echo "[INFO] 检测到 GPU 数量: ${NUM_GPUS}"

for ds in "${DATASETS[@]}"; do
    echo "==============================="
    echo "[INFO] 开始处理数据集: ${ds}"
    echo "==============================="

    accelerate launch --num_processes "${NUM_GPUS}" amazon_text2emb.py \
        --dataset "${ds}" \
        --root "${ROOT_PATH}/${ds}" \
        --plm_checkpoint "${EMB_CKPT}" \
        --plm_name "$(basename "${EMB_CKPT}")" \
        --max_sent_len "${MAX_SENT_LEN}" \
        --word_drop_ratio "${WORD_DROP_RATIO:--1}" \
        --batch_size "${BSZ}"
    echo "[INFO] 数据集 ${ds} 处理完成"
done

echo "[INFO] 所有数据集已处理完成 🎉"