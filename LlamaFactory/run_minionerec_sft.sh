export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1

INDUSTRIAL_YAML=examples/train_lora/minionerec_industrial.yaml

llamafactory-cli train $INDUSTRIAL_YAML

