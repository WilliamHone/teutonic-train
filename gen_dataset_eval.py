#!/usr/bin/env python3
"""
Download and process Teutonic dataset shards into a single JSONL file.
Processes one shard at a time: download → extract → cleanup → repeat.
"""
import json
import os
import sys
import time
import argparse
import subprocess
import numpy as np
from pathlib import Path


def download_shard(shard_id, output_path, base_url, retries=10, timeout=30):
    """Download a single shard file using wget with retry logic."""
    url = f"{base_url}/shard_{shard_id:06d}.npy"
    
    for attempt in range(retries):
        try:
            print(f"  Downloading shard_{shard_id:06d}.npy (attempt {attempt + 1}/{retries})...")
            result = subprocess.run(
                [
                    "wget", "-q", "--show-progress", "-c",
                    f"--tries=1", f"--timeout={timeout}",
                    "-O", output_path, url
                ],
                capture_output=True,
                text=True,
                check=True
            )
            if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
                print(f"  ✓ Downloaded shard_{shard_id:06d}.npy")
                return True
        except subprocess.CalledProcessError as e:
            print(f"  ✗ Download attempt {attempt + 1} failed: {e}")
            if os.path.exists(output_path):
                os.remove(output_path)
            time.sleep(min(2 ** attempt, 30))  # Exponential backoff, cap at 30s
        except FileNotFoundError:
            print("  ✗ Error: 'wget' not found. Please install wget.")
            return False
        except Exception as e:
            print(f"  ✗ Download attempt {attempt + 1} failed: {e}")
            if os.path.exists(output_path):
                os.remove(output_path)
            time.sleep(min(2 ** attempt, 30))
    
    print(f"  ✗ Failed to download shard_{shard_id:06d}.npy after {retries} attempts")
    return False


def load_shard(shard_path):
    """Load a .npy shard file; return None on failure."""
    try:
        return np.load(shard_path, mmap_mode='r')  # mmap for memory efficiency
    except Exception as e:
        print(f"  ✗ Failed to load {shard_path}: {e}")
        return None


def extract_sequences(data, seq_len, max_samples=4000, shard_seed=None):
    """Randomly sample sequences of length seq_len from token array."""
    if shard_seed is not None:
        np.random.seed(shard_seed)
    
    n_tokens = data.shape[0]
    n_sequences = n_tokens // seq_len
    actual_N = min(max_samples, n_sequences)
    
    if actual_N <= 0:
        return []
    
    indices = np.random.choice(n_sequences, size=actual_N, replace=False)
    return [
        data[idx * seq_len : (idx + 1) * seq_len].tolist()
        for idx in indices
    ]


def save_sequences_to_jsonl(sequences, output_path):
    """Append multiple sequences to JSONL file in {'input_ids': [...]} format."""
    with open(output_path, "a", encoding="utf-8") as f:
        for token_ids in sequences:
            f.write(json.dumps({"input_ids": token_ids}, ensure_ascii=False) + "\n")


def process_shard(shard_id, temp_dir, base_url, seq_len, max_per_shard, global_seed, output_path, verbose=True):
    """Download, process, and clean up a single shard. Returns number of sequences extracted."""
    shard_filename = f"shard_{shard_id:06d}.npy"
    shard_path = os.path.join(temp_dir, shard_filename)
    
    # Step 1: Download
    if not download_shard(shard_id, shard_path, base_url):
        return 0
    
    # Step 2: Load and extract sequences
    data = load_shard(shard_path)
    if data is None:
        if os.path.exists(shard_path):
            os.remove(shard_path)
        return 0
    
    # Use per-shard seed for reproducibility: global_seed + shard_id
    shard_seed = (global_seed + shard_id) if global_seed is not None else None
    sequences = extract_sequences(data, seq_len, max_per_shard, shard_seed)
    
    # Step 3: Save to JSONL
    if sequences:
        save_sequences_to_jsonl(sequences, output_path)
    
    # Step 4: Clean up downloaded shard
    if os.path.exists(shard_path):
        os.remove(shard_path)
    
    if verbose:
        status = "✓" if sequences else "⚠ (no sequences)"
        print(f"  {status} Processed shard_{shard_id:06d}.npy: {len(sequences)} sequences extracted")
    
    return len(sequences)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Download and process Teutonic dataset shards into a single JSONL file",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Required
    parser.add_argument("--output", type=str, default="datasets_eval/eval.jsonl",
                        help="Output JSONL file path")
    
    # Optional configuration
    parser.add_argument("--temp_dir", type=str, default="../temp",
                        help="Temporary directory for downloading shards")
    parser.add_argument("--base_url", type=str, 
                        default="https://s3.hippius.com/teutonic-sn3/dataset/v2/shards",
                        help="Base URL for shard files")
    parser.add_argument("--seq_len", type=int, default=2048,
                        help="Sequence length in tokens")
    parser.add_argument("--max_per_shard", type=int, default=10,
                        help="Max sequences to sample per shard")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for reproducibility")
    parser.add_argument("--start_shard", type=int, default=2,
                        help="Starting shard index (inclusive)")
    parser.add_argument("--end_shard", type=int, default=2001,
                        help="Ending shard index (inclusive)")
    parser.add_argument("--quiet", action="store_true",
                        help="Suppress progress prints")
    parser.add_argument("--keep_temp", action="store_true",
                        help="Keep temporary download directory after completion")
    
    return parser.parse_args()


def main():
    args = parse_args()
    verbose = not args.quiet
    
    # Setup directories
    os.makedirs(args.temp_dir, exist_ok=True)
    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Handle existing output file
    if os.path.exists(args.output):
        if verbose:
            print(f"⚠ Warning: Output file exists. Overwriting: {args.output}")
        os.remove(args.output)
    
    total_extracted = 0
    total_shards = args.end_shard - args.start_shard + 1
    
    if verbose:
        print(f"🚀 Processing {total_shards} shards (#{args.start_shard:06d}–#{args.end_shard:06d})")
        print(f"📁 Output: {args.output}")
        print(f"⚙️  seq_len={args.seq_len}, max_per_shard={args.max_per_shard}, seed={args.seed}")
        print("-" * 60)
    
    for shard_id in range(args.start_shard, args.end_shard + 1):
        if verbose:
            progress = f"[{shard_id - args.start_shard + 1:4d}/{total_shards}]"
            print(f"\n{progress} Processing shard_{shard_id:06d}.npy")
        
        extracted = process_shard(
            shard_id=shard_id,
            temp_dir=args.temp_dir,
            base_url=args.base_url,
            seq_len=args.seq_len,
            max_per_shard=args.max_per_shard,
            global_seed=args.seed,
            output_path=args.output,
            verbose=verbose
        )
        total_extracted += extracted
        
        if verbose and (shard_id + 1) % 10 == 0:
            print(f"  → Cumulative: {total_extracted} sequences")
    
    # Cleanup temp directory
    if not args.keep_temp and os.path.exists(args.temp_dir):
        import shutil
        shutil.rmtree(args.temp_dir)
        if verbose:
            print(f"\n🧹 Cleaned up temporary directory: {args.temp_dir}")
    
    # Final summary
    print("\n" + "=" * 60)
    print(f"✅ Done! Total sequences saved: {total_extracted:,}")
    print(f"📄 Output file: {os.path.abspath(args.output)}")
    if os.path.exists(args.output):
        file_size_mb = os.path.getsize(args.output) / (1024 * 1024)
        print(f"📦 File size: {file_size_mb:.2f} MB")
    print("=" * 60)


if __name__ == "__main__":
    main()