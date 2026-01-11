# python generate_indices_plus.py \
#   --data_path ../data/Amazon18/Industrial_and_Scientific/Industrial_and_Scientific.emb-qwen-td.npy \
#   --ckpt_path your_best_collision_model_path: e.g. /Nov-20-2025_12-25-13/best_collision_model.pth \
#   --num_emb_list 256 256 256 \
#   --device cuda:0


# /home/xiaojinsong/桌面/sdb1_xiaojinsong/MiniOneRec/rq/output/Industrial_and_Scientific/Nov-26-2025_14-38-36/best_collision_model.pth
# /home/xiaojinsong/桌面/sdb1_xiaojinsong/MiniOneRec/rq/output/Office_Products/Nov-26-2025_14-42-02/epoch_8799_collision_0.0561_model.pth

DATA_ROOT="../data/Amazon2018"
EMB_MODEL="Qwen3-Embedding-4B"
E_DIM=32
DEVICE="cuda"
DATASETS=(
  # "All_Beauty"                
  "Arts_Crafts_and_Sewing"     
  "Industrial_and_Scientific" 
  "Office_Products"         
  "Video_Games"
)
CKPT_PATHS=(
  "./output/rqvae_base/Arts_Crafts_and_Sewing/Jan-11-2026_01-00-42/best_collision_model.pth"
  "./output/rqvae_base/Industrial_and_Scientific/Jan-11-2026_01-55-14/best_collision_model.pth"
  "./output/rqvae_base/Office_Products/Jan-11-2026_02-34-40/best_collision_model.pth"
  "./output/rqvae_base/Video_Games/Jan-11-2026_03-29-11/best_collision_model.pth"
)

for ((i=0; i<${#DATASETS[@]}; i++)); do
    python generate_indices_plus.py \
        --data_path "${DATA_ROOT}/${DATASETS[i]}/${DATASETS[i]}.emb-${EMB_MODEL}-td.npy" \
        --ckpt_path "${CKPT_PATHS[i]}" \
        --e_dim "${E_DIM}" \
        --num_emb_list 256 256 256 \
        --device "${DEVICE}"

    echo "[DONE] ${DATASETS[i]}"
done


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