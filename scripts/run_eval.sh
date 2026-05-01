#!/bin/bash
# run_eval.sh - Simple script to launch model evaluation

set -e  # Exit on error

# Default configuration
KING_REPO="/dev/shm/teutonic/models/conanedoAI/Teutonic-VIII-5Ek5KoE5-v1-5x-1196"
CHALLENGER_REPO="merged/VIII/Teutonic-Foremost04-v0101"
DATASET_DIR="/dev/shm/teutonic/datasets_eval"
N_SAMPLES=160
SEQ_LEN=2048
BATCH_SIZE=16
ALPHA=0.001
DELTA=0.01
N_BOOTSTRAP=10000
GPUS="auto"
SEED="eval:102"

echo "🚀 Starting evaluation..."
echo "  King         : $KING_REPO"
echo "  Challenger   : $CHALLENGER_REPO"
echo "  Dataset      : $DATASET_DIR"
echo "  Samples      : $N_SAMPLES"
echo "  Seq len      : $SEQ_LEN"
echo "  Batch size   : $BATCH_SIZE"
echo "  Delta/Alpha  : $DELTA / $ALPHA"
echo "  GPUs         : $GPUS"
echo "  Seed         : $SEED"
echo ""

python3 eval_torch_local.py \
    --king "$KING_REPO" \
    --challenger "$CHALLENGER_REPO" \
    --dataset-dir "$DATASET_DIR" \
    --n "$N_SAMPLES" \
    --seq-len "$SEQ_LEN" \
    --batch-size "$BATCH_SIZE" \
    --alpha "$ALPHA" \
    --delta "$DELTA" \
    --n-bootstrap "$N_BOOTSTRAP" \
    --gpus "$GPUS" \
    --seed "$SEED"

echo "✅ Done! Verdict printed above."