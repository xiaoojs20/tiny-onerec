SFT_DATA_ROOT="./data/Amazon2018"
CATEGORY="Industrial_and_Scientific"
python extend_vocab.py \
    --model_name_or_path ../llms/Qwen/Qwen3-0.6B \
    --sid_index_path "${SFT_DATA_ROOT}/${CATEGORY}/${CATEGORY}.index.json" \
    --output_path "./rec_llm/Qwen3-0.6B-${CATEGORY}"