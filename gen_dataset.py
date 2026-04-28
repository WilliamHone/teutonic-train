#!/usr/bin/env python3
import json
import os
import argparse
import numpy as np


def load_shard(shard_path):
    """Load a .npy shard file; return None on failure."""
    try:
        return np.load(shard_path)
    except Exception as e:
        print(f"Failed to load {shard_path}: {e}")
        return None


def extract_sequences(data, seq_len, max_samples=4000, seed=None):
    """Randomly sample sequences of length seq_len from token array."""
    if seed is not None:
        np.random.seed(seed)
    
    n_tokens = data.shape[0]
    n_sequences = n_tokens // seq_len
    # print("##########", n_sequences)
    actual_N = min(max_samples, n_sequences)
    
    if actual_N <= 0:
        return []
    
    indices = np.random.choice(n_sequences, size=actual_N, replace=False)
    return [
        data[idx * seq_len : (idx + 1) * seq_len].tolist()
        for idx in indices
    ]


def save_to_jsonl(sequences, output_path):
    """Write sequences to JSONL file with {'input_ids': [...]} format."""
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        for token_ids in sequences:
            f.write(json.dumps({"input_ids": token_ids}) + "\n")


def process_shard_file(shard_path, seq_len, max_samples=4000, seed=None):
    """Process a single shard file and return extracted sequences."""
    data = load_shard(shard_path)
    if data is None:
        return []
    return extract_sequences(data, seq_len, max_samples, seed)


def collect_dataset(shard_dir, seq_len, max_per_shard=4000, seed=None, verbose=True):
    """Iterate over .npy files in shard_dir and collect sequences."""
    dataset = []
    for shard_name in sorted(os.listdir(shard_dir)):  # sorted for reproducibility
        if verbose:
            print(shard_name)
        if not shard_name.endswith(".npy"):
            continue
        shard_path = os.path.join(shard_dir, shard_name)
        sequences = process_shard_file(shard_path, seq_len, max_per_shard, seed)
        dataset.extend(sequences)
        if verbose:
            print(f"Extracted {len(dataset)} data.")
    return dataset


def parse_args():
    parser = argparse.ArgumentParser(description="Generate JSONL dataset from .npy shards for LLM training")
    
    # Required
    parser.add_argument("--shard_dir", type=str, required=True,
                        help="Directory containing .npy shard files")
    parser.add_argument("--output", type=str, required=True,
                        help="Output JSONL file path")
    
    # Optional with original defaults
    parser.add_argument("--seq_len", type=int, default=2048,
                        help="Sequence length in tokens (default: 2048)")
    parser.add_argument("--max_per_shard", type=int, default=4000,
                        help="Max sequences to sample per shard (default: 4000)")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for reproducibility (default: None)")
    parser.add_argument("--quiet", action="store_true",
                        help="Suppress progress prints")
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    dataset = collect_dataset(
        shard_dir=args.shard_dir,
        seq_len=args.seq_len,
        max_per_shard=args.max_per_shard,
        seed=args.seed,
        verbose=not args.quiet
    )
    
    save_to_jsonl(dataset, args.output)
    print(f"Total sequences saved: {len(dataset)}")


if __name__ == "__main__":
    main()