#!/usr/bin/env python3
"""Standalone multi-GPU PyTorch eval — king-vs-challenger paired bootstrap test.

Loads model replicas across all available GPUs, reads token shards from local
disk, and computes cross-entropy loss via chunked lm_head forward passes to
minimize VRAM. Accepts the challenger only when the bootstrapped lower
confidence bound on the per-token log-loss advantage exceeds a configurable
delta threshold.

Usage:
    python eval_torch_local.py \
        --king 22oseni/Teutonic-I-foo5 \
        --challenger knsimon/Teutonic-I-9000 \
        --dataset-dir /data/teutonic \
        --n 100 --delta 0.01 --batch-size 64 --seq-len 2048 --gpus 0,1,2,3,4,5,6,7

Env vars:
    HF_TOKEN              HuggingFace token for gated repos
"""
import argparse
import ast
import hashlib
import io
import json
import logging
import os
import pathlib
import struct
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM

log = logging.getLogger("eval_torch_local")


# ---------------------------------------------------------------------------
# Local dataset helpers
# ---------------------------------------------------------------------------

DEFAULT_MANIFEST = "manifest.json"


def _read_npy_header(raw: bytes):
    """Return (data_offset, header_dict) for a .npy file header buffer."""
    buf = io.BytesIO(raw)
    buf.read(6)  # magic
    ver = struct.unpack("BB", buf.read(2))
    hl = struct.unpack("<H" if ver[0] == 1 else "<I", buf.read(2 if ver[0] == 1 else 4))[0]
    header = ast.literal_eval(buf.read(hl).decode("latin1").strip())
    return buf.tell(), header


def _load_npy_metadata(shard_path):
    with open(shard_path, "rb") as f:
        prefix = f.read(10)
        if len(prefix) < 10:
            raise ValueError(f"incomplete npy header in {shard_path}")
        major, _minor = struct.unpack("BB", prefix[6:8])
        header_len_size = 2 if major == 1 else 4
        extra = f.read(header_len_size)
        if len(extra) != header_len_size:
            raise ValueError(f"incomplete npy header length in {shard_path}")
        header_len = struct.unpack("<H" if major == 1 else "<I", extra)[0]
        header_buf = prefix + extra + f.read(header_len)
    data_offset, header = _read_npy_header(header_buf)
    return data_offset, header


def get_shard_info(shard_path):
    _, header = _load_npy_metadata(shard_path)
    n_tokens = 1
    for dim in header["shape"]:
        n_tokens *= dim
    return int(n_tokens)


def load_shard(shard_path):
    """Open a local .npy shard as a read-only memmap."""
    t0 = time.time()
    data_offset, header = _load_npy_metadata(shard_path)
    n_tokens = 1
    for dim in header["shape"]:
        n_tokens *= dim
    shard_tokens = np.memmap(
        shard_path,
        dtype="<u4",
        mode="r",
        offset=data_offset,
        shape=(int(n_tokens),),
    )
    elapsed = time.time() - t0
    size_mb = pathlib.Path(shard_path).stat().st_size / 1e6
    log.info("opened local shard %s: %.1f MB in %.2fs", shard_path, size_mb, elapsed)
    return shard_tokens


def extract_sequences(shard_tokens, indices, seq_len):
    """Extract sequences from a local memmapped shard."""
    result = {}
    for idx in indices:
        start = idx * seq_len
        result[idx] = shard_tokens[start : start + seq_len].tolist()
    return result


def discover_shards(dataset_dir, max_shards=None, seed=None):
    """Find all .npy shard files in dataset_dir, optionally limit and shuffle."""
    dataset_path = pathlib.Path(dataset_dir).expanduser()
    shards = sorted([f for f in os.listdir(dataset_path) if f.endswith(".npy")])
    
    if seed is not None:
        rng = np.random.default_rng(seed)
        rng.shuffle(shards)
    
    if max_shards is not None:
        shards = shards[:max_shards]
    
    return [str(dataset_path / s) for s in shards]


def sample_from_multiple_shards(dataset_dir, eval_n, seq_len, seed_str, max_shards=None):
    """Sample eval_n sequences of length seq_len from multiple shards.
    
    Returns list of (shard_path, sequence_tokens) tuples.
    """
    seed_material = seed_str.encode()
    seed = int.from_bytes(hashlib.blake2b(seed_material, digest_size=8).digest(), "little")
    rng = np.random.Generator(np.random.PCG64(seed))
    
    shards = discover_shards(dataset_dir, max_shards=max_shards, seed=seed)
    if not shards:
        raise ValueError(f"No .npy shards found in {dataset_dir}")
    
    # Compute shard capacities
    shard_caps = []
    for shard_path in shards:
        n_tokens = get_shard_info(shard_path)
        n_sequences = n_tokens // seq_len
        if n_sequences > 0:
            shard_caps.append((shard_path, n_sequences))
    
    if not shard_caps:
        raise ValueError("No shards have enough tokens for seq_len={seq_len}")
    
    # Proportional allocation: sample from each shard proportional to its capacity
    total_cap = sum(cap for _, cap in shard_caps)
    samples_per_shard = []
    remaining = eval_n
    
    for i, (shard_path, cap) in enumerate(shard_caps):
        if i == len(shard_caps) - 1:
            # Last shard gets remainder to hit exact eval_n
            n_samples = remaining
        else:
            # Proportional allocation with floor
            n_samples = min(cap, max(0, int(eval_n * cap / total_cap)))
            remaining -= n_samples
        samples_per_shard.append((shard_path, n_samples))
    
    # Extract sequences
    results = []
    for shard_path, n_samples in samples_per_shard:
        if n_samples <= 0:
            continue
        
        shard_tokens = load_shard(shard_path)
        n_tokens = shard_tokens.shape[0]
        n_sequences = n_tokens // seq_len
        actual_n = min(n_samples, n_sequences)
        
        indices = rng.choice(n_sequences, size=actual_n, replace=False).tolist()
        seqs = extract_sequences(shard_tokens, indices, seq_len)
        
        for idx in indices:
            results.append(seqs[idx])
        
        log.info("Sampled %d/%d sequences from %s", len(seqs), n_samples, 
                 pathlib.Path(shard_path).name)
    
    log.info("Total sequences sampled: %d/%d", len(results), eval_n)
    return results[:eval_n]  # Ensure exact count


def load_manifest(manifest_path):
    return json.loads(pathlib.Path(manifest_path).read_text())


def resolve_shard_path(shard_ref, dataset_dir=None, manifest_path=None):
    """Resolve a shard reference from CLI or manifest to a concrete local path."""
    candidates = []
    shard_path = pathlib.Path(shard_ref).expanduser()
    dataset_root = pathlib.Path(dataset_dir).expanduser() if dataset_dir else None
    manifest_root = pathlib.Path(manifest_path).expanduser().resolve().parent if manifest_path else None

    candidates.append(shard_path)
    if dataset_root:
        candidates.append(dataset_root / shard_ref)
        candidates.append(dataset_root / pathlib.Path(shard_ref).name)
    if manifest_root:
        candidates.append(manifest_root / shard_ref)
        candidates.append(manifest_root / pathlib.Path(shard_ref).name)

    seen = set()
    for candidate in candidates:
        normalized = candidate.resolve(strict=False)
        if normalized in seen:
            continue
        seen.add(normalized)
        if normalized.exists():
            return normalized

    raise FileNotFoundError(
        f"could not resolve local shard path for '{shard_ref}'"
        + (f" under dataset_dir='{dataset_dir}'" if dataset_dir else "")
    )


# ---------------------------------------------------------------------------
# Chunked loss computation — avoids materializing full [batch, seq, vocab]
# ---------------------------------------------------------------------------

LM_HEAD_CHUNK = 256


@torch.no_grad()
def compute_batch_losses(model, token_batches, device, chunk_size=LM_HEAD_CHUNK):
    """Forward pass with chunked lm_head to avoid OOM on large vocabs."""
    input_ids = torch.tensor(token_batches, dtype=torch.long, device=device)
    hidden = model.model(input_ids).last_hidden_state
    lm_head = model.lm_head

    n_positions = input_ids.size(1) - 1
    total_loss = torch.zeros(len(token_batches), device=device)

    for i in range(0, n_positions, chunk_size):
        end_pos = min(i + chunk_size, n_positions)
        chunk_logits = lm_head(hidden[:, i:end_pos, :])
        chunk_labels = input_ids[:, i + 1 : end_pos + 1]
        loss = F.cross_entropy(
            chunk_logits.reshape(-1, chunk_logits.size(-1)),
            chunk_labels.reshape(-1),
            reduction="none",
        )
        total_loss += loss.reshape(len(token_batches), -1).sum(dim=1)
        del chunk_logits, loss

    return (total_loss / n_positions).cpu().tolist()


# ---------------------------------------------------------------------------
# Paired losses — runs both models' lm_heads per chunk
# ---------------------------------------------------------------------------


@torch.no_grad()
def compute_paired_losses(king_model, chall_model, token_batches,
                          king_device, chall_device,
                          chunk_size=LM_HEAD_CHUNK):
    """Compute per-sequence mean cross-entropy for both models on the same tokens."""
    B = len(token_batches)
    input_ids_k = torch.tensor(token_batches, dtype=torch.long, device=king_device)
    input_ids_c = torch.tensor(token_batches, dtype=torch.long, device=chall_device)

    hidden_k = king_model.model(input_ids_k).last_hidden_state
    hidden_c = chall_model.model(input_ids_c).last_hidden_state

    n_pos = input_ids_k.size(1) - 1
    king_loss = torch.zeros(B, device=king_device)
    chall_loss = torch.zeros(B, device=chall_device)

    for i in range(0, n_pos, chunk_size):
        end = min(i + chunk_size, n_pos)

        logits_k = king_model.lm_head(hidden_k[:, i:end, :])
        logits_c = chall_model.lm_head(hidden_c[:, i:end, :])

        labels_k = input_ids_k[:, i + 1 : end + 1]
        labels_c = input_ids_c[:, i + 1 : end + 1]
        king_loss += F.cross_entropy(
            logits_k.reshape(-1, logits_k.size(-1)), labels_k.reshape(-1),
            reduction="none",
        ).reshape(B, -1).sum(1)
        chall_loss += F.cross_entropy(
            logits_c.reshape(-1, logits_c.size(-1)), labels_c.reshape(-1),
            reduction="none",
        ).reshape(B, -1).sum(1)

        del logits_k, logits_c

    return (
        (king_loss / n_pos).cpu().tolist(),
        (chall_loss / n_pos).cpu().tolist(),
    )


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------


def load_model(repo, device, label="model", force_download=False, revision=None):
    log.info("loading %s from %s onto %s (force_download=%s, revision=%s)",
             label, repo, device, force_download, revision[:12] if revision else None)
    t0 = time.time()
    for attn_impl in ("flash_attention_2", "sdpa", "eager"):
        try:
            model = AutoModelForCausalLM.from_pretrained(
                repo,
                torch_dtype=torch.bfloat16,
                device_map={"": device},
                attn_implementation=attn_impl,
                token=os.environ.get("HF_TOKEN") or None,
                force_download=force_download,
                revision=revision or None,
            )
            log.info("using attn_implementation=%s", attn_impl)
            break
        except Exception as e:
            log.warning("attn %s failed (%s), trying next", attn_impl, e)
    else:
        raise RuntimeError("could not load model with any attention implementation")
    model.eval()
    elapsed = time.time() - t0
    params = sum(p.numel() for p in model.parameters()) / 1e9
    log.info("%s loaded: %.1fB params in %.1fs", label, params, elapsed)
    return model


# ---------------------------------------------------------------------------
# Weight norm guard — reject models with inflated L2 norms
# ---------------------------------------------------------------------------

NORM_EPSILON = 1e-8


@torch.no_grad()
def check_weight_norms(king_model, challenger_model, max_ratio=5.0):
    """Compare per-parameter L2 norms between king and challenger."""
    king_params = dict(king_model.named_parameters())
    violations = []

    for c_name, c_param in challenger_model.named_parameters():
        if not torch.isfinite(c_param.data).all():
            violations.append({"param": c_name, "reason": "non-finite values"})
            continue

        k_param = king_params.get(c_name)
        if k_param is None:
            continue

        k_norm = torch.linalg.vector_norm(k_param.data.float()).item()
        c_norm = torch.linalg.vector_norm(c_param.data.float()).item()

        if k_norm < NORM_EPSILON:
            continue

        ratio = c_norm / k_norm
        if ratio > max_ratio:
            violations.append({
                "param": c_name,
                "reason": "norm_ratio",
                "king_norm": round(k_norm, 6),
                "challenger_norm": round(c_norm, 6),
                "ratio": round(ratio, 4),
            })

    return len(violations) == 0, violations


# ---------------------------------------------------------------------------
# Multi-GPU evaluator
# ---------------------------------------------------------------------------


class MultiGPUEvaluator:
    """Manages model replicas across GPUs and dispatches batches in parallel."""

    def __init__(self, repo, gpu_ids, label="model", force_download=False, revision=None):
        self.gpu_ids = gpu_ids
        self.models = {}
        self.devices = {}

        if len(gpu_ids) == 0:
            raise ValueError("need at least one GPU")

        first_model = load_model(repo, f"cuda:{gpu_ids[0]}", f"{label}-gpu{gpu_ids[0]}",
                                 force_download=force_download, revision=revision)
        self.models[gpu_ids[0]] = first_model
        self.devices[gpu_ids[0]] = f"cuda:{gpu_ids[0]}"

        for gid in gpu_ids[1:]:
            self.models[gid] = load_model(repo, f"cuda:{gid}", f"{label}-gpu{gid}",
                                          force_download=force_download, revision=revision)
            self.devices[gid] = f"cuda:{gid}"

        self.pool = ThreadPoolExecutor(max_workers=len(gpu_ids))
        log.info("%s evaluator ready: %d GPUs %s", label, len(gpu_ids), gpu_ids)

    def compute_losses(self, token_batches):
        """Split token_batches across GPUs, compute in parallel, reassemble."""
        n_gpus = len(self.gpu_ids)
        if not token_batches:
            return []

        per_gpu = [[] for _ in range(n_gpus)]
        idx_map = [[] for _ in range(n_gpus)]
        for i, batch in enumerate(token_batches):
            g = i % n_gpus
            per_gpu[g].append(batch)
            idx_map[g].append(i)

        futures = {}
        for g_idx, gid in enumerate(self.gpu_ids):
            if per_gpu[g_idx]:
                fut = self.pool.submit(
                    compute_batch_losses,
                    self.models[gid], per_gpu[g_idx], self.devices[gid],
                )
                futures[fut] = g_idx

        results = [None] * len(token_batches)
        for fut in as_completed(futures):
            g_idx = futures[fut]
            losses = fut.result()
            for local_i, global_i in enumerate(idx_map[g_idx]):
                results[global_i] = losses[local_i]

        return results

    def shutdown(self):
        self.pool.shutdown(wait=False)


def compute_paired_multi_gpu(king_eval, chall_eval, token_batches):
    """Pair king GPUs with challenger GPUs to compute losses in parallel."""
    if not token_batches:
        return [], []

    n_pairs = min(len(king_eval.gpu_ids), len(chall_eval.gpu_ids))
    per_pair = [[] for _ in range(n_pairs)]
    idx_map = [[] for _ in range(n_pairs)]
    for i, batch in enumerate(token_batches):
        p = i % n_pairs
        per_pair[p].append(batch)
        idx_map[p].append(i)

    futures = {}
    pool = ThreadPoolExecutor(max_workers=n_pairs)
    for p_idx in range(n_pairs):
        if not per_pair[p_idx]:
            continue
        k_gid = king_eval.gpu_ids[p_idx]
        c_gid = chall_eval.gpu_ids[p_idx]
        fut = pool.submit(
            compute_paired_losses,
            king_eval.models[k_gid], chall_eval.models[c_gid],
            per_pair[p_idx],
            king_eval.devices[k_gid], chall_eval.devices[c_gid],
        )
        futures[fut] = p_idx

    king_results = [None] * len(token_batches)
    chall_results = [None] * len(token_batches)
    for fut in as_completed(futures):
        p_idx = futures[fut]
        k_losses, c_losses = fut.result()
        for local_i, global_i in enumerate(idx_map[p_idx]):
            king_results[global_i] = k_losses[local_i]
            chall_results[global_i] = c_losses[local_i]

    pool.shutdown(wait=False)
    return king_results, chall_results


# ---------------------------------------------------------------------------
# Bootstrap test
# ---------------------------------------------------------------------------


def run_bootstrap_test(king_eval, challenger_eval, dataset_dir, eval_n,
                       alpha, delta, seq_len, batch_size, seed_str,
                       n_bootstrap=10000, on_progress=None, max_shards=None):
    """Paired bootstrap test on per-token log-loss differences.
    
    Samples eval_n sequences from multiple shards in dataset_dir.
    """
    log.info("bootstrap test: N=%d alpha=%s delta=%.6f B=%d",
             eval_n, alpha, delta, n_bootstrap)

    # Sample sequences from multiple shards
    log.info("sampling %d sequences from %s (max_shards=%s)", 
             eval_n, dataset_dir, max_shards)
    sequences = sample_from_multiple_shards(
        dataset_dir, eval_n, seq_len, seed_str, max_shards=max_shards
    )
    
    if len(sequences) < eval_n:
        log.warning("Only got %d/%d sequences from shards", len(sequences), eval_n)
    
    actual_N = len(sequences)
    log.info("extracted %d sequences for evaluation", actual_N)

    batches = [
        sequences[i: i + batch_size]
        for i in range(0, len(sequences), batch_size)
    ]

    all_diffs = []
    king_sum, chall_sum = 0.0, 0.0
    total_done = 0
    t0 = time.time()

    same_evaluator = king_eval is challenger_eval

    for bi, token_batches in enumerate(batches):
        if same_evaluator:
            king_losses = king_eval.compute_losses(token_batches)
            chall_losses = king_losses
        else:
            king_losses, chall_losses = compute_paired_multi_gpu(
                king_eval, challenger_eval, token_batches,
            )

        for k_loss, c_loss in zip(king_losses, chall_losses):
            total_done += 1
            king_sum += k_loss
            chall_sum += c_loss
            all_diffs.append(k_loss - c_loss)

        elapsed = time.time() - t0
        seqs_per_sec = total_done / elapsed if elapsed > 0 else 0
        mu_hat = np.mean(all_diffs) if all_diffs else 0.0
        log.info(
            "batch %d/%d | done=%d/%d | mu_hat=%.6f | %.1f seq/s",
            bi + 1, len(batches), total_done, actual_N, mu_hat, seqs_per_sec,
        )

        if on_progress:
            on_progress({
                "done": total_done,
                "total": actual_N,
                "mu_hat": round(float(mu_hat), 6),
                "avg_king_loss": round(king_sum / total_done, 6),
                "avg_challenger_loss": round(chall_sum / total_done, 6),
                "seqs_per_sec": round(seqs_per_sec, 1),
            })

    elapsed = time.time() - t0
    d = np.array(all_diffs)
    mu_hat = float(d.mean())

    boot_rng = np.random.Generator(np.random.PCG64(int(hashlib.blake2b(seed_str.encode(), digest_size=8).digest().hex(), 16) ^ 0xB007))
    boot_means = np.empty(n_bootstrap)
    for b in range(n_bootstrap):
        idx = boot_rng.integers(0, len(d), size=len(d))
        boot_means[b] = d[idx].mean()
    lcb = float(np.quantile(boot_means, alpha))

    accepted = lcb > delta
    log.info("bootstrap result: mu_hat=%.6f lcb=%.6f delta=%.6f accepted=%s",
             mu_hat, lcb, delta, accepted)

    verdict = {
        "accepted": accepted,
        "verdict": "challenger" if accepted else "king",
        "mu_hat": round(mu_hat, 6),
        "lcb": round(lcb, 6),
        "delta": delta,
        "alpha": alpha,
        "n_bootstrap": n_bootstrap,
        "N": actual_N,
        "avg_king_loss": round(king_sum / total_done, 6) if total_done else 0,
        "avg_challenger_loss": round(chall_sum / total_done, 6) if total_done else 0,
        "wall_time_s": round(elapsed, 1),
        "seqs_per_sec": round(total_done / elapsed, 1) if elapsed > 0 else 0,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    return verdict


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_gpu_ids(gpu_str):
    if gpu_str == "auto":
        return list(range(torch.cuda.device_count()))
    return [int(x.strip()) for x in gpu_str.split(",")]


def main():
    parser = argparse.ArgumentParser(description="Multi-GPU PyTorch model eval")
    parser.add_argument("--king", default="whiskeyman/Teutonic-III-v4x-1050", help="HF repo for king model")
    parser.add_argument("--challenger", default="merged/Teutonic-III-v3401", help="HF repo for challenger model")
    parser.add_argument("--dataset-dir", required=True,
                        help="Directory containing .npy shard files")  # Now required
    parser.add_argument("--n", type=int, default=100, help="Number of sequences to evaluate")
    parser.add_argument("--alpha", type=float, default=0.001, help="Bootstrap confidence level (one-sided)")
    parser.add_argument("--delta", type=float, default=0.01, help="Minimum effect threshold in nats/token")
    parser.add_argument("--n-bootstrap", type=int, default=10000, help="Number of bootstrap replicates")
    parser.add_argument("--batch-size", type=int, default=2, help="Sequences per batch (split across GPUs)")
    parser.add_argument("--seq-len", type=int, default=2048, help="Tokens per sequence")
    parser.add_argument("--gpus", default="auto", help="Comma-separated GPU IDs or 'auto' (default: auto)")
    parser.add_argument("--seed", default="test:eval", help="Seed string for deterministic sequence selection")
    parser.add_argument("--max-shards", type=int, default=None, 
                        help="Limit number of shards to sample from (default: all)")
    parser.add_argument("--manifest", default=None,  # Optional now
                        help=f"Local manifest path (deprecated when using --dataset-dir)")
    parser.add_argument("--shard", default=None,
                        help="Deprecated: use --dataset-dir instead for multi-shard sampling")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    gpu_ids = parse_gpu_ids(args.gpus)
    log.info("using GPUs: %s", gpu_ids)

    # Validate dataset dir
    if not os.path.isdir(args.dataset_dir):
        log.error("dataset-dir not found: %s", args.dataset_dir)
        sys.exit(1)

    same_model = args.king == args.challenger

    if same_model:
        log.info("king == challenger, using all %d GPUs for shared evaluator", len(gpu_ids))
        king_eval = MultiGPUEvaluator(args.king, gpu_ids, label="king")
        challenger_eval = king_eval
    else:
        mid = len(gpu_ids) // 2
        king_gpus = gpu_ids[:mid] or gpu_ids[:1]
        chall_gpus = gpu_ids[mid:] or gpu_ids[:1]
        log.info("king GPUs: %s  challenger GPUs: %s", king_gpus, chall_gpus)
        king_eval = MultiGPUEvaluator(args.king, king_gpus, label="king")
        challenger_eval = MultiGPUEvaluator(args.challenger, chall_gpus, label="challenger")

    log.info("=" * 60)
    log.info("EVAL CONFIG")
    log.info("  king:       %s", args.king)
    log.info("  challenger: %s", args.challenger)
    log.info("  GPUs:       %s (%s)", gpu_ids, "shared" if same_model else "split")
    log.info("  N=%d  alpha=%s  delta=%.6f  bootstrap=%d  batch=%d  seq_len=%d",
             args.n, args.alpha, args.delta, args.n_bootstrap, args.batch_size, args.seq_len)
    log.info("  dataset:  %s (max_shards=%s)", args.dataset_dir, args.max_shards)
    log.info("  seed:  %s", args.seed)
    log.info("=" * 60)

    verdict = run_bootstrap_test(
        king_eval, challenger_eval,
        dataset_dir=args.dataset_dir,  # Changed from shard_path
        eval_n=args.n, alpha=args.alpha, delta=args.delta,
        seq_len=args.seq_len, batch_size=args.batch_size, seed_str=args.seed,
        n_bootstrap=args.n_bootstrap,
        max_shards=args.max_shards,
    )

    king_eval.shutdown()
    if not same_model:
        challenger_eval.shutdown()

    print()
    print("=" * 60)
    print("VERDICT")
    print("=" * 60)
    print(json.dumps(verdict, indent=2))
    print("=" * 60)

    return 0 if not verdict["accepted"] else 1


if __name__ == "__main__":
    sys.exit(main())