#!/bin/bash
# run_merge.sh - Example script to launch LoRA model merging

set -e  # Exit on error

# Default configuration (matches original script)
BASE_MODEL="ClarenceDan/Teutonic-VIII-A5502"
LORA_PATH="checkpoints/VIII/teutonic_vera6_v03/checkpoint-1300"
OUTPUT_DIR="./merged/VIII/Teutonic-vera6-v0301/"
MAX_SHARD_SIZE="20GB"
DTYPE="bfloat16"

echo "🚀 Starting LoRA model merge..."
echo "  Base model  : $BASE_MODEL"
echo "  LoRA path   : $LORA_PATH"
echo "  Output dir  : $OUTPUT_DIR"
echo "  Shard size  : $MAX_SHARD_SIZE"
echo "  Dtype       : $DTYPE"
echo ""

python merge_lora.py \
    --base_model "$BASE_MODEL" \
    --lora_path "$LORA_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --max_shard_size "$MAX_SHARD_SIZE" \
    --dtype "$DTYPE"

echo "✅ Done! Merged model saved to: $OUTPUT_DIR"