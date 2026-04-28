#!/bin/bash
# run_generate.sh - Example script to launch dataset generation

set -e  # Exit on error

# Default configuration (matches original script)
SHARD_DIR="../teutonic_dataset/"
OUTPUT_FILE="datasets/dataset_v002_std.jsonl"
SEQ_LEN=2048
MAX_PER_SHARD=2000
SEED=2  # Added for reproducibility

echo "🚀 Starting dataset generation..."
echo "  Shard dir : $SHARD_DIR"
echo "  Output    : $OUTPUT_FILE"
echo "  Seq len   : $SEQ_LEN"
echo "  Max/shard : $MAX_PER_SHARD"
echo "  Seed      : $SEED"
echo ""

python gen_dataset.py \
    --shard_dir "$SHARD_DIR" \
    --output "$OUTPUT_FILE" \
    --seq_len "$SEQ_LEN" \
    --max_per_shard "$MAX_PER_SHARD" \
    --seed "$SEED"

echo "✅ Done! Output: $OUTPUT_FILE"