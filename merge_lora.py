#!/usr/bin/env python3
# merge_lora.py - Modular script to merge a LoRA adapter with a base model

import argparse
import logging
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

def load_base_model(model_id: str, dtype_str: str = "bfloat16", device_map: str = "auto"):
    """Load the base model and tokenizer."""
    logging.info(f"Loading base model: {model_id}")
    dtype = getattr(torch, dtype_str)
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype, device_map=device_map)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    logging.info(f"Base model loaded with dtype: {model.dtype}")
    return model, tokenizer

def merge_lora(base_model, lora_path: str):
    """Load LoRA weights, merge them into the base model, and return the merged model."""
    logging.info(f"Loading LoRA adapter from: {lora_path}")
    lora_model = PeftModel.from_pretrained(base_model, lora_path)
    
    logging.info("Merging LoRA weights into base model...")
    merged_model = lora_model.merge_and_unload()
    logging.info(f"Merge complete. Model dtype: {merged_model.dtype}")
    return merged_model

def save_merged_model(model, tokenizer, output_dir: str, max_shard_size: str = "10GB"):
    """Save the merged model and tokenizer to disk."""
    logging.info(f"Saving merged model and tokenizer to: {output_dir}")
    model.save_pretrained(output_dir, max_shard_size=max_shard_size)
    tokenizer.save_pretrained(output_dir)
    logging.info("✅ Save complete!")

def main():
    parser = argparse.ArgumentParser(description="Merge a LoRA adapter with a base model.")
    parser.add_argument("--base_model", type=str, required=True, help="Hugging Face model ID or local path to base model.")
    parser.add_argument("--lora_path", type=str, required=True, help="Path to the trained LoRA checkpoint.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the merged model.")
    parser.add_argument("--max_shard_size", type=str, default="10GB", help="Maximum shard size for model saving.")
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["float16", "bfloat16", "float32"], help="Data type for model loading.")
    parser.add_argument("--device_map", type=str, default="auto", help="Device mapping strategy (e.g., auto, cpu, cuda:0).")
    
    args = parser.parse_args()
    
    model, tokenizer = load_base_model(args.base_model, args.dtype, args.device_map)
    merged_model = merge_lora(model, args.lora_path)
    save_merged_model(merged_model, tokenizer, args.output_dir, args.max_shard_size)

if __name__ == "__main__":
    main()