#!/bin/bash
# run_mine.sh - Script to launch hard-data mining for continued pretraining

set -e  # Exit on error

# ==================== Configuration ====================
# Model & Data
MODEL_PATH="/dev/shm/teutonic/models/iotaminer/Teutonic-VIII-5DFoNuLs-v70b"
SHARD_DIR="/dev/shm/teutonic/datasets"
SHARD_START=0
SHARD_END=1998
SEQ_LEN=2048

# Sampling
CANDIDATE_PER_SHARD=1200
BATCH_SIZE=64
SEED=4
# Selection Mode: "percentile" or "threshold"
SELECTION_MODE="percentile"

# Percentile mode parameters (used if SELECTION_MODE="percentile")
MIN_LOSS_PERCENTILE=30.0
MAX_LOSS_PERCENTILE=98.0

# Threshold mode parameters (used if SELECTION_MODE="threshold")
LOSS_THRESHOLD="2.5"  # Set to a float value, e.g., "2.5", to enable threshold mode

# Output
OUTPUT="datasets/dataset_v001_dynamic.jsonl"

# ==================== Validation ====================
if [[ "$SELECTION_MODE" == "threshold" && -z "$LOSS_THRESHOLD" ]]; then
    echo "❌ Error: --loss-threshold must be set when SELECTION_MODE=threshold"
    exit 1
fi

# ==================== Display Configuration ====================
echo "🚀 Starting hard-data mining..."
echo "  Model              : $MODEL_PATH"
echo "  Shard dir          : $SHARD_DIR"
echo "  Shard range        : [$SHARD_START, $SHARD_END)"
echo "  Seq len            : $SEQ_LEN"
echo "  Candidates/shard   : $CANDIDATE_PER_SHARD"
echo "  Batch size         : $BATCH_SIZE"
echo "  Seed               : $SEED"

if [[ "$SELECTION_MODE" == "percentile" ]]; then
    echo "  Selection mode     : percentile"
    echo "  Loss percentile    : [$MIN_LOSS_PERCENTILE, $MAX_LOSS_PERCENTILE]"
else
    echo "  Selection mode     : threshold"
    echo "  Loss threshold     : >= $LOSS_THRESHOLD"
fi

echo "  Output             : $OUTPUT"
echo ""

# ==================== Build Command ====================
CMD=(
    python gen_dataset_dynamic.py
    --model-path "$MODEL_PATH"
    --shard-dir "$SHARD_DIR"
    --shard-start "$SHARD_START"
    --shard-end "$SHARD_END"
    --seq-len "$SEQ_LEN"
    --candidate-per-shard "$CANDIDATE_PER_SHARD"
    --batch-size "$BATCH_SIZE"
    --selection-mode "$SELECTION_MODE"
    --seed "$SEED"
    --output "$OUTPUT"
)

# Add percentile-specific args
if [[ "$SELECTION_MODE" == "percentile" ]]; then
    CMD+=(
        --min-loss-percentile "$MIN_LOSS_PERCENTILE"
        --max-loss-percentile "$MAX_LOSS_PERCENTILE"
    )
else
    # Add threshold-specific args
    CMD+=(
        --loss-threshold "$LOSS_THRESHOLD"
    )
fi

# ==================== Run ====================
"${CMD[@]}"

echo "✅ Done! Output: $OUTPUT"