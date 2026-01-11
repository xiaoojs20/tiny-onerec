#!/bin/bash
set -euo pipefail

CATS=(
  "All_Beauty"
  "Arts_Crafts_and_Sewing"
  "Industrial_and_Scientific"
  "Office_Products"
  "Video_Games"
)

REVIEWS_DIR="./Amazon-Reviews-2018"
OUTPUT_DIR="./Amazon2018"


for cat in "${CATS[@]}"; do
  echo "========================================"
  echo "[INFO] Processing category: ${cat}"

  python amazon18_data_process.py \
    --dataset "${cat}" \
    --reviews_file "${REVIEWS_DIR}/${cat}.json" \
    --metadata_file "${REVIEWS_DIR}/meta_${cat}.json" \
    --user_k 5 \
    --item_k 5 \
    --st_year 2016 \
    --st_month 01 \
    --ed_year 2018 \
    --ed_month 12 \
    --output_path "${OUTPUT_DIR}"

  echo "[DONE] ${cat}"
done

echo "========================================"
echo "[INFO] All categories finished."


# python amazon18_data_process.py \
#     --dataset Industrial_and_Scientific \
#     --reviews_file ./Amazon-Reviews-2018/Industrial_and_Scientific.json \
#     --metadata_file ./Amazon-Reviews-2018/meta_Industrial_and_Scientific.json \
#     --user_k 5 \
#     --item_k 5 \
#     --st_year 2016 \
#     --st_month 01 \
#     --ed_year 2018 \
#     --ed_month 12 \
#     --output_path ./Amazon2018

# /home/xiaojinsong/桌面/sdb1_xiaojinsong/MiniOneRec/data/Amazon-Reviews-2023/raw/review_categories/All_Beauty.jsonl
# /home/xiaojinsong/桌面/sdb1_xiaojinsong/MiniOneRec/data/Amazon-Reviews-2023/raw/meta_categories/meta_All_Beauty.jsonl