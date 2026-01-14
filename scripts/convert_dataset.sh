#!/bin/bash


# INPUT_DIR="data/Amazon2018"
# OUTPUT_DIR="data/Amazon2018"

DATASETS=(
      # "All_Beauty"                  # (40, dim) -> 样本量太少，忽略
      "Arts_Crafts_and_Sewing"      # (8094, dim)
      "Industrial_and_Scientific"   # (3433, dim)
      "Office_Products"             # 
      "Video_Games"
)

for DATASET_NAME in "${DATASETS[@]}"; do
    echo "Start converting $DATASET_NAME ..."
    INPUT_DIR="./data/Amazon2018/${DATASET_NAME}"
    OUTPUT_DIR="./data/Amazon2018/${DATASET_NAME}"

    python convert_dataset.py \
        --dataset_name $DATASET_NAME \
        --data_dir $INPUT_DIR \
        --output_dir $OUTPUT_DIR \
        --category $DATASET_NAME \
        --seed 42

    echo "Finished!"
done



