#!/bin/bash
# run_eval_update.sh - Launch eval_update.py (local shard mode)

set -e

# Configuration (hardcoded defaults)
KING_REPO="/dev/shm/models/bluecolor/Teutonic-LXXX-5Ek5KoE5-v2-240x-364"
CHALLENGER_REPO="/dev/shm/teutonic/models/tech-dev-ai/Teutonic-LXXX-5GCDP2Ru-real5"
SHARD_DIR="/dev/shm/teutonic/datasets_eval"
SHARD_NAME=""
N_SAMPLES=80
SEQ_LEN=2048
BATCH_SIZE=8
ALPHA=0.001
DELTA=0.0002
N_BOOTSTRAP=10000
GPUS="auto"
SEED="eval:102"

echo "🚀 Starting evaluation..."
echo "  King         : $KING_REPO"
echo "  Challenger   : $CHALLENGER_REPO"
echo "  Shard dir    : $SHARD_DIR"
echo "  Shard name   : ${SHARD_NAME:-<auto>}"
echo "  Samples      : $N_SAMPLES"
echo "  Seq len      : $SEQ_LEN"
echo "  Batch size   : $BATCH_SIZE"
echo "  Delta/Alpha  : ${DELTA:-1/N} / $ALPHA"
echo "  GPUs         : $GPUS"
echo "  Seed         : $SEED"
echo ""

python3 eval_torch_local_update.py \
    --king "$KING_REPO" \
    --challenger "$CHALLENGER_REPO" \
    --shard-dir "$SHARD_DIR" \
    ${SHARD_NAME:+--shard-name "$SHARD_NAME"} \
    --n "$N_SAMPLES" \
    --seq-len "$SEQ_LEN" \
    --batch-size "$BATCH_SIZE" \
    --alpha "$ALPHA" \
    ${DELTA:+--delta "$DELTA"} \
    --n-bootstrap "$N_BOOTSTRAP" \
    --gpus "$GPUS" \
    --seed "$SEED"

echo "✅ Done! Verdict printed above."