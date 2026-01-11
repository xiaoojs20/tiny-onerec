#!/bin/bash
set -euo pipefail



LR=1e-4
EPOCHS=10000
BATCH_SIZE=2048


DEVICE="cuda"

DATA_ROOT="../data/Amazon2018"
CKPT_ROOT="./output/rqvae_all_baseline" # [rqvae_all_baseline]
EMB_MODEL="Qwen3-Embedding-4B"

DATASETS=(
    # "All_Beauty"                  # (40, dim) -> 样本量太少，忽略
    "Arts_Crafts_and_Sewing"      # (8094, dim)
    "Industrial_and_Scientific"   # (3433, dim)
    "Office_Products"             # 
    "Video_Games"
)


# for CATEGORY in "${DATASETS[@]}"; do
#     DATA_PATH="${DATA_ROOT}/${CATEGORY}/${CATEGORY}.emb-${EMB_MODEL}-td.npy"
#     CKPT_DIR="${CKPT_ROOT}/${CATEGORY}"

CATEGORY=All_Amazon
# DATA_PATH="${DATA_ROOT}/${CATEGORY}/${CATEGORY}.emb-${EMB_MODEL}-td.npy"
DATA_PATH="${DATA_ROOT}/${CATEGORY}"
CKPT_DIR="${CKPT_ROOT}/${CATEGORY}"

mkdir -p "${CKPT_DIR}"

# ---- log file ----
TS=$(date +"%Y%m%d_%H%M%S")
LOG_DIR="${CKPT_DIR}/logs"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/train_${CATEGORY}_${EMB_MODEL}_lr${LR}_bs${BATCH_SIZE}_ep${EPOCHS}_${TS}.log"

{
python rqvae.py \
    --data_path "${DATA_PATH}" \
    --ckpt_dir "${CKPT_DIR}" \
    --lr "${LR}" \
    --epochs "${EPOCHS}" \
    --batch_size "${BATCH_SIZE}" \
    --device "${DEVICE}"

echo ""
echo "[DONE] ${CATEGORY}"
echo ""
# done

echo "✅ finished."
} 2>&1 | tee "${LOG_FILE}"

# python rqvae.py \
#       --data_path ../data/AmazonTest/Industrial_and_Scientific.emb-qwen-td.npy \
#       --ckpt_dir ./output/Industrial_and_Scientific \
#       --lr 1e-3 \
#       --epochs 10000 \
#       --batch_size 2048

# python rqvae.py \
#       --data_path ../data/AmazonTest/Office_Products.emb-qwen-td.npy \
#       --ckpt_dir ./output/Office_Products \
#       --lr 1e-3 \
#       --epochs 10000 \
#       --batch_size 2048
