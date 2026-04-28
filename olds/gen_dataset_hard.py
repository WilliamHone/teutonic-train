import argparse
import json
import os
import re
from pathlib import Path

import numpy as np
import torch
import datetime

try:
    import orjson
    _USE_ORJSON = True
except ImportError:
    _USE_ORJSON = False
    print("⚠️  Install 'orjson' for faster JSON: pip install orjson")

from gpu_evaluator import MultiGPUEvaluator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Mine higher-loss token blocks for continued pretraining."
    )
    parser.add_argument("--model-path", type=str, default="tom6979/Teutonic-III-V95ST900")
    parser.add_argument("--shard-dir", type=str, default="/teutonic/teutonic_dataset")
    parser.add_argument("--shard-start", type=int, default=0)
    parser.add_argument("--shard-end", type=int, default=1998)
    parser.add_argument("--seq-len", type=int, default=2048)
    parser.add_argument("--candidate-per-shard", type=int, default=600)
    parser.add_argument("--keep-per-shard", type=int, default=600)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--min-loss-percentile", type=float, default=35.0)
    parser.add_argument("--max-loss-percentile", type=float, default=98.0)
    parser.add_argument("--seed", type=int, default=452)
    parser.add_argument("--output", type=str, default="gen_data_v3_hard_s_452.jsonl")
    parser.add_argument("--write-buffer-size", type=int, default=100)
    return parser.parse_args()


def _parse_shard_index(filename: str) -> int:
    """Extract shard index from filename like 'shard_000123.npy'."""
    # Try simple split first (fast path)
    try:
        return int(Path(filename).stem.split("_")[1])
    except (IndexError, ValueError):
        # Fallback: regex for any digit sequence
        match = re.search(r"(\d+)", Path(filename).stem)
        if match:
            return int(match.group(1))
        raise ValueError(f"Cannot parse shard index from: {filename}")


def _json_dumps(obj: dict) -> str:
    if _USE_ORJSON:
        return orjson.dumps(obj).decode('utf-8')
    return json.dumps(obj, separators=(',', ':'))


def score_sequences(evaluator: MultiGPUEvaluator, sequences: np.ndarray, batch_size: int) -> np.ndarray:
    losses = []
    for start in range(0, len(sequences), batch_size):
        batch = sequences[start : start + batch_size]
        losses.extend(evaluator.compute_losses(batch.tolist()))
    return np.asarray(losses, dtype=np.float32)


def process_shard(
    shard_path: Path,
    shard_idx: int,
    evaluator: MultiGPUEvaluator,
    args,
    rng: np.random.Generator
) -> tuple[list[dict], dict]:
    """Process a single shard. Returns (kept_records, stats_dict)."""
    try:
        data = np.load(shard_path, mmap_mode='r')
    except Exception as e:
        print(f"⚠️  Failed to load {shard_path.name}: {e}")
        return [], {"candidate_n": 0, "avg_loss": float("nan"), "kept_avg": float("nan"), 
                    "kept_count": 0, "lo": float("nan"), "hi": float("nan")}
    
    n_tokens = int(data.shape[0])
    n_sequences = n_tokens // args.seq_len
    
    stats = {"candidate_n": 0, "avg_loss": float("nan"), "kept_avg": float("nan"), 
             "kept_count": 0, "lo": float("nan"), "hi": float("nan")}
    
    if n_sequences == 0:
        return [], stats
    
    candidate_n = min(args.candidate_per_shard, n_sequences)
    stats["candidate_n"] = candidate_n
    
    candidate_indices = rng.choice(n_sequences, size=candidate_n, replace=False)
    
    # Efficient sequence extraction via stacking
    candidate_sequences = np.stack([
        data[idx * args.seq_len : (idx + 1) * args.seq_len] 
        for idx in candidate_indices
    ])
    
    candidate_losses = score_sequences(evaluator, candidate_sequences, args.batch_size)
    
    # Percentile filtering
    lo = np.percentile(candidate_losses, args.min_loss_percentile)
    hi = np.percentile(candidate_losses, args.max_loss_percentile)
    keep_mask = (candidate_losses >= lo) & (candidate_losses <= hi)
    
    stats["lo"], stats["hi"] = lo, hi
    stats["avg_loss"] = float(candidate_losses.mean())
    
    # Build records for ALL items that pass the filter
    kept = []
    kept_indices = np.where(keep_mask)[0]
    for i in kept_indices:
        kept.append({
            "input_ids": candidate_sequences[i].tolist(),
            "meta": {
                "source_shard": shard_idx,
                "source_seq_idx": int(candidate_indices[i]),
                "base_loss": float(candidate_losses[i]),
            },
        })
    
    # Sort by loss descending, take top K (FIX: was double-slicing)
    if kept:
        kept.sort(key=lambda x: x["meta"]["base_loss"], reverse=True)
        kept = kept[:args.keep_per_shard]
        rng.shuffle(kept)
    
    stats["kept_count"] = len(kept)
    stats["kept_avg"] = float(np.mean(candidate_losses[keep_mask])) if keep_mask.any() else float("nan")
    
    return kept, stats


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    shard_dir = Path(args.shard_dir)
    
    gpu_ids = list(range(torch.cuda.device_count()))
    if not gpu_ids:
        raise RuntimeError("No CUDA devices found for hard-data mining.")

    evaluator = MultiGPUEvaluator(args.model_path, gpu_ids, label="miner")
    output_path = Path(args.output)
    
    write_buffer = []
    total_written = 0
    timestamp_start = datetime.datetime.now().timestamp()
    
    try:
        files = sorted([f for f in os.listdir(args.shard_dir) if f.endswith(".npy")])
        # Filter by shard index range using robust parser
        filtered_files = []
        for f in files:
            try:
                idx = _parse_shard_index(f)
                if args.shard_start <= idx < args.shard_end:
                    filtered_files.append(f)
            except ValueError:
                print(f"⚠️  Skipping unparseable filename: {f}")
                continue
        
        rng.shuffle(filtered_files)
        
        with output_path.open("w") as output_file:
            for file in filtered_files:
                shard_idx = _parse_shard_index(file)
                shard_path = shard_dir / file
                
                if not shard_path.exists():
                    print(f"⚠️  Skipping missing shard: {shard_path}")
                    continue
                
                kept, stats = process_shard(shard_path, shard_idx, evaluator, args, rng)
                
                # Buffer records for batched I/O
                for record in kept:
                    write_buffer.append(_json_dumps(record) + "\n")
                    if len(write_buffer) >= args.write_buffer_size:
                        output_file.writelines(write_buffer)
                        write_buffer.clear()
                
                total_written += stats["kept_count"]
                
                # Progress logging
                elapsed = int(datetime.datetime.now().timestamp() - timestamp_start)
                print(
                    f"Shard {shard_idx:4d}: cand={stats['candidate_n']:4d} "
                    f"avg_loss={stats['avg_loss']:.4f} kept={stats['kept_count']:4d} "
                    f"kept_avg={stats['kept_avg']:.4f} band=[{stats['lo']:.2f}, {stats['hi']:.2f}] "
                    f"elapsed={elapsed}s total={total_written}"
                )
            
            # Flush remaining buffer
            if write_buffer:
                output_file.writelines(write_buffer)
                
    finally:
        evaluator.shutdown()

    elapsed_total = datetime.datetime.now().timestamp() - timestamp_start
    rate = total_written / max(elapsed_total, 1)
    print(f"✓ Wrote {total_written} records to {output_path} in {elapsed_total:.1f}s ({rate:.1f} rec/s)")


if __name__ == "__main__":
    main()