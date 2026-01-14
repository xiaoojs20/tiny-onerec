export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1


## Industrial 

INDUSTRIAL_YAML=examples/train_minionerec/minionerec_industrial.yaml

LOG_DIR=logs
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/industrial_$(date +%Y%m%d_%H%M%S).log"

nohup llamafactory-cli train "$INDUSTRIAL_YAML" > "$LOG_FILE" 2>&1 &

echo "Started. PID=$!  Log=$LOG_FILE"


## Office
