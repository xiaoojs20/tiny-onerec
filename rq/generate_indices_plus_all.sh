

EMB_MODEL="Qwen3-Embedding-4B"
E_DIM=32
DEVICE="cuda"

CKPT_ROOT="./output/rqvae_all_baseline/All_Amazon"
CKPT_PATH="${CKPT_ROOT}/Jan-09-2026_15-33-32/best_collision_model.pth"
DATA_ROOT="../data/Amazon2018"
DATA_PATH="${DATA_ROOT}/All_Amazon/"

python generate_indices_plus.py \
    --data_path ${DATA_PATH} \
    --ckpt_path ${CKPT_PATH} \
    --e_dim "${E_DIM}" \
    --num_emb_list 256 256 256 \
    --device "${DEVICE}"

echo "[DONE] ✅"



# for DATASET in "${DATASETS[@]}"; do
#     DATA_PATH="${DATA_ROOT}/${DATASET}/${DATASET}.emb-${EMB_MODEL}-td.npy"

#     python generate_indices_plus.py \
#         --data_path "${DATA_PATH}" \
#         --ckpt_path "${CKPT_PATH}" \
#         --e_dim "${E_DIM}" \
#         --num_emb_list 256 256 256 \
#         --device "${DEVICE}"

#     echo "[DONE] ${DATASET}"
# done


# python generate_indices_plus.py \
#   --data_path ../data/AmazonTest/Office_Products.emb-qwen-td.npy \
#   --ckpt_path ./output/Office_Products/Nov-26-2025_14-42-02/best_collision_model.pth \
#   --e_dim 32 \
#   --num_emb_list 256 256 256 \
#   --device cuda:0