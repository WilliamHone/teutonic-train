import argparse
import json
from pathlib import Path

import numpy as np
import torch, os
import datetime

from gpu_evaluator import MultiGPUEvaluator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Mine higher-loss token blocks under the base model for continued pretraining."
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="iotaminer/Teutonic-III-soup-lion-NeverOOM27-1533",
        help="Base model used to score candidate sequences.",
    )
    parser.add_argument(
        "--shard-dir",
        type=str,
        default="/teutonic/teutonic_dataset",
        help="Directory containing shard_XXXXXX.npy files.",
    )
    parser.add_argument(
        "--shard-start",
        type=int,
        default=1980,
        help="First shard index to consider.",
    )
    parser.add_argument(
        "--shard-end",
        type=int,
        default=1998,
        help="Exclusive upper bound for shard index selection.",
    )
    parser.add_argument(
        "--seq-len",
        type=int,
        default=2048,
        help="Sequence length to extract.",
    )
    parser.add_argument(
        "--candidate-per-shard",
        type=int,
        default=1200,
        help="Random candidate blocks to score per shard.",
    )
    parser.add_argument(
        "--keep-per-shard",
        type=int,
        default=1200,
        help="How many scored blocks to keep per shard.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Scoring batch size.",
    )
    parser.add_argument(
        "--min-loss-percentile",
        type=float,
        default=30.0,
        help="Ignore candidates below this loss percentile.",
    )
    parser.add_argument(
        "--max-loss-percentile",
        type=float,
        default=98.0,
        help="Ignore the most extreme tail above this percentile.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=449,
        help="Random seed.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="gen_data_v3_hard_s_449.jsonl",
        help="Output JSONL path.",
    )
    return parser.parse_args()


def score_sequences(evaluator: MultiGPUEvaluator, sequences: list[list[int]], batch_size: int) -> list[float]:
    losses = []
    for start in range(0, len(sequences), batch_size):
        batch = sequences[start : start + batch_size]
        losses.extend(evaluator.compute_losses(batch))
    return losses


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    shard_dir = Path(args.shard_dir)
    gpu_ids = list(range(torch.cuda.device_count()))
    if not gpu_ids:
        raise RuntimeError("No CUDA devices found for hard-data mining.")

    evaluator = MultiGPUEvaluator(args.model_path, gpu_ids, label="miner")
    
    # Open output file once at the start in write mode
    # NOTE: This will overwrite existing file. For append/resume mode, change to "a" 
    # but be careful about duplicates on re-runs.
    output_path = Path(args.output)
    output_file = output_path.open("w")
    total_written = 0

    try:
        timestamp_start = datetime.datetime.now().timestamp()
        
        # Get and shuffle file list for better distribution across shards
        files = [f for f in os.listdir(args.shard_dir) if f.endswith(".npy")]
        rng.shuffle(files)  # Shuffle shard processing order for randomness
        
        for file in files:
            shard_idx = int(file.split("_")[1].split(".")[0])
            shard_path = shard_dir / file
            if not shard_path.exists():
                print(f"Skipping missing shard: {shard_path}")
                continue

            data = np.load(shard_path)
            n_tokens = int(data.shape[0])
            n_sequences = n_tokens // args.seq_len
            if n_sequences == 0:
                print(f"Skipping empty shard: {shard_path}")
                continue

            candidate_n = min(args.candidate_per_shard, n_sequences)
            candidate_indices = rng.choice(n_sequences, size=candidate_n, replace=False).tolist()
            candidate_sequences = []
            for seq_idx in candidate_indices:
                offset = seq_idx * args.seq_len
                candidate_sequences.append(data[offset : offset + args.seq_len].tolist())

            candidate_losses = np.asarray(
                score_sequences(evaluator, candidate_sequences, args.batch_size),
                dtype=np.float64,
            )

            lo = np.percentile(candidate_losses, args.min_loss_percentile)
            hi = np.percentile(candidate_losses, args.max_loss_percentile)
            keep_mask = (candidate_losses >= lo) & (candidate_losses <= hi)

            kept = [
                {
                    "input_ids": candidate_sequences[i],
                    "meta": {
                        "source_shard": shard_idx,
                        "source_seq_idx": candidate_indices[i],
                        "base_loss": float(candidate_losses[i]),
                    },
                }
                for i in np.where(keep_mask)[0].tolist()
            ]

            kept.sort(key=lambda item: item["meta"]["base_loss"], reverse=True)
            kept = kept[: args.keep_per_shard]
            
            # Shuffle within shard before writing for local randomness
            rng.shuffle(kept)
            
            # Write immediately to JSONL file (incremental saving)
            for record in kept:
                output_file.write(json.dumps(record) + "\n")
            output_file.flush()  # Ensure data is written to disk immediately
            total_written += len(kept)

            avg_loss = float(candidate_losses.mean())
            kept_avg = float(np.mean([item["meta"]["base_loss"] for item in kept])) if kept else float("nan")
            elapsed = int(datetime.datetime.now().timestamp() - timestamp_start)
            print(
                f"Shard {shard_idx}: candidates={candidate_n} avg_loss={avg_loss:.4f} "
                f"kept={len(kept)} kept_avg_loss={kept_avg:.4f} band=[{lo:.4f}, {hi:.4f}] "
                f"elapsed={elapsed}s total_written={total_written}"
            )
    finally:
        evaluator.shutdown()
        output_file.close()  # Ensure file is properly closed

    print(f"✓ Wrote {total_written} records incrementally to {output_path}")


if __name__ == "__main__":
    main()