export NCCL_IB_DISABLE=1        # 完全禁用 IB/RoCE
export NCCL_P2P_DISABLE=1

# mkdir -p logs

BASE_MODEL="../llms/Qwen/Qwen3-0.6B"
SFT_DATA_ROOT="./data/Amazon2018"
BSZ=32
MICRO_BSZ=4

DATASETS=(
  "Industrial_and_Scientific" 
  # "Office_Products"    
#   "Arts_Crafts_and_Sewing"        
#   "Video_Games"
)

WANDB_PROJ="minionerec"
OUTPUT_DIR="output_sft/sft_base"
SID_METHOD="rqvae"

for CATEGORY in "${DATASETS[@]}"; do
    train_file=$(ls -f ${SFT_DATA_ROOT}/${CATEGORY}/train/*.csv)
    eval_file=$(ls -f ${SFT_DATA_ROOT}/${CATEGORY}/valid/*.csv)
    test_file=$(ls -f ${SFT_DATA_ROOT}/${CATEGORY}/test/*.csv)
    info_file=$(ls -f ${SFT_DATA_ROOT}/${CATEGORY}/info/*.txt)

    sid_index_path="${SFT_DATA_ROOT}/${CATEGORY}/${CATEGORY}.index.json"
    item_meta_path="${SFT_DATA_ROOT}/${CATEGORY}/${CATEGORY}.item.json"

    wandb_run_name="minionerec-sft-${SID_METHOD}-${CATEGORY}-$(date +%Y%m%d_%H%M%S)"

    echo ${train_file} ${eval_file} ${info_file} ${test_file}
    log_file="${OUTPUT_DIR}/sft_${CATEGORY}_$(date +%Y%m%d_%H%M%S).log"
    echo "log  : ${log_file}"
    mkdir -p "$(dirname "${log_file}")"

    torchrun --nproc_per_node 2 \
            sft.py \
            --base_model "${BASE_MODEL}" \
            --batch_size "${BSZ}" \
            --micro_batch_size "${MICRO_BSZ}" \
            --train_file ${train_file} \
            --eval_file ${eval_file} \
            --output_dir output_sft/sft_base/"${CATEGORY}" \
            --wandb_project "${WANDB_PROJ}" \
            --wandb_run_name "${wandb_run_name}" \
            --category ${CATEGORY} \
            --train_from_scratch False \
            --seed 42 \
            --sid_index_path ${sid_index_path} \
            --item_meta_path ${item_meta_path} \
            --freeze_LLM False
done
    # nohup torchrun --nproc_per_node 2 \
    #         sft.py \
    #         --base_model "${BASE_MODEL}" \
    #         --batch_size "${BSZ}" \
    #         --micro_batch_size "${MICRO_BSZ}" \
    #         --train_file ${train_file} \
    #         --eval_file ${eval_file} \
    #         --output_dir output_sft/sft_base \
    #         --wandb_project "${WANDB_PROJ}" \
    #         --wandb_run_name "${wandb_run_name}" \
    #         --category ${CATEGORY} \
    #         --train_from_scratch False \
    #         --seed 42 \
    #         --sid_index_path ${sid_index_path} \
    #         --item_meta_path ${item_meta_path} \
    #         --freeze_LLM False  \
    #         > "${log_file}" 2>&1 &
# done


# export NCCL_IB_DISABLE=1        # 完全禁用 IB/RoCE
# # Office_Products, Industrial_and_Scientific
# for category in "Industrial_and_Scientific"; do
#     train_file=$(ls -f ./data/Amazon/train/${category}*11.csv)
#     eval_file=$(ls -f ./data/Amazon/valid/${category}*11.csv)
#     test_file=$(ls -f ./data/Amazon/test/${category}*11.csv)
#     info_file=$(ls -f ./data/Amazon/info/${category}*.txt)
#     echo ${train_file} ${eval_file} ${info_file} ${test_file}
    
#     torchrun --nproc_per_node 8 \
#             sft.py \
#             --base_model your_model_path \
#             --batch_size 1024 \
#             --micro_batch_size 16 \
#             --train_file ${train_file} \
#             --eval_file ${eval_file} \
#             --output_dir output_dir/xxx \
#             --wandb_project wandb_proj \
#             --wandb_run_name wandb_name \
#             --category ${category} \
#             --train_from_scratch False \
#             --seed 42 \
#             --sid_index_path ./data/Amazon/index/Industrial_and_Scientific.index.json \
#             --item_meta_path ./data/Amazon/index//Industrial_and_Scientific.item.json \
#             --freeze_LLM False
# done
