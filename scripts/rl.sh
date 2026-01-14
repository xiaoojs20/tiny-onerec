#!/bin/bash

export NCCL_IB_DISABLE=1        # 完全禁用 IB/RoCE
export NCCL_P2P_DISABLE=1
export WANDB_MODE=online

export HF_ENDPOINT=https://hf-mirror.com


RL_DATA_ROOT="./data/Amazon2018"
TRAIN_BSZ=16
EVAL_BSZ=32
NUM_TRAIN_EPOCHS=2
GRAD_ACCUM=2

NUM_GENERATIONS=16

OUTPUT_DIR="output/rl_base"
WANDB_PROJ="tiny-onerec"

# SID_METHOD="rqvae"
REWARD=ranking # [rule, ranking, ranking_only, semantic, sasrec]

DATASETS=(
  "Industrial_and_Scientific" 
  # "Office_Products"    
#   "Arts_Crafts_and_Sewing"        
#   "Video_Games"
)


SFT_MODEL="./LlamaFactory/saves/minionerec_extend/checkpoint-8000"

for CATEGORY in "${DATASETS[@]}"; do
    train_file=$(ls -f ${RL_DATA_ROOT}/${CATEGORY}/train/*.csv)
    eval_file=$(ls -f ${RL_DATA_ROOT}/${CATEGORY}/valid/*.csv)
    info_file=$(ls -f ${RL_DATA_ROOT}/${CATEGORY}/info/*.txt)

    sid_index_path=${RL_DATA_ROOT}/${CATEGORY}/${CATEGORY}.index.json
    item_meta_path=${RL_DATA_ROOT}/${CATEGORY}/${CATEGORY}.item.json

    # SFT_MODEL="./output_sft/sft_base/${CATEGORY}/final_checkpoint"
    wandb_run_name="rl-${REWARD}-$(date +%Y%m%d_%H%M%S)"

    output_dir="${OUTPUT_DIR}/${CATEGORY}"

    nohup accelerate launch --config_file ./config/zero2_opt.yaml \
        --num_processes 2 \
        --main_process_port 29503 \
        rl.py \
        --model_path ${SFT_MODEL} \
        --train_batch_size ${TRAIN_BSZ} \
        --eval_batch_size ${EVAL_BSZ} \
        --num_train_epochs ${NUM_TRAIN_EPOCHS} \
        --gradient_accumulation_steps ${GRAD_ACCUM} \
        --train_file ${train_file} \
        --eval_file ${eval_file} \
        --info_file ${info_file} \
        --category ${CATEGORY} \
        --sample_train False \
        --eval_step 20 \
        --reward_type ${REWARD} \
        --num_generations ${NUM_GENERATIONS} \
        --mask_all_zero False \
        --dynamic_sampling False \
        --sync_ref_model True \
        --beam_search True \
        --test_during_training False \
        --temperature 1.0 \
        --learning_rate 1e-5 \
        --add_gt False \
        --beta 1e-3 \
        --dapo False \
        --output_dir ${OUTPUT_DIR} \
        --wandb_project ${WANDB_PROJ} \
        --wandb_run_name ${wandb_run_name} \
        --sid_index_path ${sid_index_path} \
        --item_meta_path ${item_meta_path} \
        > logs/rl_${CATEGORY}_$(date +%Y%m%d_%H%M%S).log 2>&1 &
        # --resume_from ${OUTPUT_DIR}/checkpoint-14778 \
done



# export NCCL_IB_DISABLE=1        # 完全禁用 IB/RoCE

# for category in "Industrial_and_Scientific"; do
#     train_file=$(ls -f ./data/Amazon/train/${category}*.csv)
#     eval_file=$(ls -f ./data/Amazon/valid/${category}*11.csv)
#     info_file=$(ls -f ./data/Amazon/info/${category}*.txt)

#     HF_ENDPOINT=https://hf-mirror.com accelerate launch \
#                                     --config_file ./config/zero2_opt.yaml \
#                                     --num_processes 8 --main_process_port 29503 \
#                                     rl.py \
#                         --model_path path_to_model \
#                         --train_batch_size 64 \
#                         --eval_batch_size 128 \
#                         --num_train_epochs 2 \
#                         --gradient_accumulation_steps 2 \
#                         --train_file ${train_file} \
#                         --eval_file ${eval_file} \
#                         --info_file ${info_file} \
#                         --category ${category} \
#                         --sample_train False \
#                         --eval_step 0.0999 \
#                         --reward_type ranking \
#                         --num_generations 16 \
#                         --mask_all_zero False \
#                         --dynamic_sampling False \
#                         --sync_ref_model True \
#                         --beam_search True \
#                         --test_during_training False \
#                         --temperature 1.0 \
#                         --learning_rate 1e-5 \
#                         --add_gt False \
#                         --beta 1e-3 \
#                         --dapo False \
#                         --output_dir output_dir \
#                         --wandb_run_name wandb_name \
#                         --sid_index_path ./data/Amazon/index/Industrial_and_Scientific.index.json \
#                         --item_meta_path ./data/Amazon/index/Industrial_and_Scientific.item.json
# done
