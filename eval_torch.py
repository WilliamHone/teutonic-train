#!/usr/bin/env python3
"""Standalone multi-GPU PyTorch eval — king-vs-challenger paired bootstrap test.

Loads model replicas across all available GPUs, fetches sequences from local
directory, and computes cross-entropy loss via chunked lm_head forward passes
to minimize VRAM. Accepts the challenger only when the bootstrapped lower
confidence bound on the per-token log-loss advantage exceeds delta = 1/N.

Usage:
    python eval_torch.py \
        --king unconst/Teutonic-I \
        --challenger unconst/Teutonic-I \
        --n 100 --batch-size 64 --seq-len 2048 --gpus 0,1,2,3,4,5,6,7 \
        --local-shards-dir /path/to/shards

Env vars:
    HF_TOKEN                        HuggingFace token for gated repos
    TEUTONIC_LOCAL_SHARDS_DIR       Local directory containing shard files
    TEUTONIC_SHARD_CACHE            Cache directory for shards (default: /tmp/shard_cache)
    TEUTONIC_SHARD_CACHE_MAX        Max cached shards to keep (default: 10)
    TEUTONIC_LM_HEAD_CHUNK          Chunk size for lm_head forward (default: 512)
    TEUTONIC_FINETUNE_*             Trainability probe thresholds
"""

import argparse
import hashlib
import io
import json
import logging
import os
import pathlib
import socket
import struct
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import quasar

# hf-xet timeout handling
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "120")
os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "30")
socket.setdefaulttimeout(180)

import numpy as np
import torch
import torch.nn.functional as F
from collections import defaultdict
from transformers import AutoModelForCausalLM

# Add workspace root to path for local imports
_workspace_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _workspace_root not in sys.path:
    sys.path.insert(0, _workspace_root)

import chain_config  # noqa: E402

chain_config.load_arch()  # noqa: E402

log = logging.getLogger("eval_torch")


# ---------------------------------------------------------------------------
# Local Shard Store
# ---------------------------------------------------------------------------


class LocalShardStore:
    """Local filesystem backend for shard storage.

    Expects:
    - Shard files as .npy files with custom header format
    - Manifest files as JSON in dataset/v1/ or dataset/v2/ subdirectories
    """

    def __init__(self, base_dir: str):
        self.base_dir = pathlib.Path(base_dir).resolve()
        if not self.base_dir.exists():
            raise ValueError(f"Local shards directory does not exist: {base_dir}")
        log.info("using local shard store: %s", self.base_dir)

    def _resolve_path(self, key: str) -> pathlib.Path:
        """Resolve a shard key to a local file path."""
        key_path = pathlib.Path(key)

        # Handle absolute paths
        if key_path.is_absolute() and key_path.exists():
            return key_path

        # Try direct path relative to base_dir
        direct = self.base_dir / key
        if direct.exists():
            return direct

        # Try just the filename (strip directory prefixes)
        name = key_path.name
        candidate = self.base_dir / name
        if candidate.exists():
            return candidate

        # Try common subdirectories
        for subdir in ["shards", "data", "dataset"]:
            candidate = self.base_dir / subdir / name
            if candidate.exists():
                return candidate

        # Return the most likely path for error reporting
        return direct

    def get(self, key):
        """Read and parse a JSON file from local storage."""
        try:
            path = self._resolve_path(key)
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except FileNotFoundError:
            log.warning(
                "local file not found: %s (resolved: %s)", key, self._resolve_path(key)
            )
            return None
        except json.JSONDecodeError as e:
            log.warning("failed to parse JSON file %s: %s", key, e)
            return None
        except Exception as e:
            log.warning("failed to read local file %s: %s", key, e)
            return None

    def range_get(self, key, start, end):
        """Read a byte range from a local file."""
        path = self._resolve_path(key)
        with open(path, "rb") as f:
            f.seek(start)
            return f.read(end - start + 1)

    def ds_get(self, key):
        """Dataset store get — same as get for local."""
        return self.get(key)

    def ds_range_get(self, key, start, end):
        """Dataset store range get — same as range_get for local."""
        return self.range_get(key, start, end)


# ---------------------------------------------------------------------------
# Dataset utilities
# ---------------------------------------------------------------------------


def get_shard_info(store, shard_key):
    """Extract total token count from shard header."""
    header = store.ds_range_get(shard_key, 0, 1023)
    buf = io.BytesIO(header)
    buf.read(6)  # magic
    ver = struct.unpack("BB", buf.read(2))
    hl = struct.unpack(
        "<H" if ver[0] == 1 else "<I", buf.read(2 if ver[0] == 1 else 4)
    )[0]
    hdr = eval(buf.read(hl).decode("latin1").strip())
    n = 1
    for s in hdr["shape"]:
        n *= s
    return n


FETCH_WORKERS = 32


def _parse_shard_header(store, shard_key):
    """Parse shard header and return data offset."""
    header = store.ds_range_get(shard_key, 0, 1023)
    buf = io.BytesIO(header)
    buf.read(6)  # magic
    ver = struct.unpack("BB", buf.read(2))
    hl = struct.unpack(
        "<H" if ver[0] == 1 else "<I", buf.read(2 if ver[0] == 1 else 4)
    )[0]
    buf.read(hl)
    return buf.tell()


def fetch_sequences(store, shard_key, indices, seq_len):
    """Fetch specific sequences from a shard file."""
    data_offset = _parse_shard_header(store, shard_key)
    bps = seq_len * 4
    sorted_idx = sorted(set(indices))
    idx_set = set(indices)

    # Group indices to minimize I/O calls
    groups, gs, ge = [], sorted_idx[0], sorted_idx[0]
    for i in sorted_idx[1:]:
        if i - ge <= 64:
            ge = i
        else:
            groups.append((gs, ge))
            gs = ge = i
    groups.append((gs, ge))

    def _fetch_group(gs_ge):
        gs, ge = gs_ge
        chunk = store.ds_range_get(
            shard_key, data_offset + gs * bps, data_offset + (ge + 1) * bps - 1
        )
        partial = {}
        for idx in range(gs, ge + 1):
            if idx in idx_set:
                off = (idx - gs) * bps
                partial[idx] = np.frombuffer(
                    chunk[off : off + bps], dtype="<u4"
                ).tolist()
        return partial

    result = {}
    with ThreadPoolExecutor(max_workers=FETCH_WORKERS) as pool:
        for partial in pool.map(_fetch_group, groups):
            result.update(partial)
    return result


# ---------------------------------------------------------------------------
# Shard caching
# ---------------------------------------------------------------------------

SHARD_CACHE_DIR = os.environ.get("TEUTONIC_SHARD_CACHE", "/tmp/shard_cache")
SHARD_CACHE_MAX = int(os.environ.get("TEUTONIC_SHARD_CACHE_MAX", "10"))

_shard_locks: dict[str, threading.Lock] = {}
_shard_locks_guard = threading.Lock()


def _parse_npy_header(raw: bytes) -> int:
    """Return the byte offset where data begins in a .npy file."""
    buf = io.BytesIO(raw)
    buf.read(6)  # magic
    ver = struct.unpack("BB", buf.read(2))
    hl = struct.unpack(
        "<H" if ver[0] == 1 else "<I", buf.read(2 if ver[0] == 1 else 4)
    )[0]
    buf.read(hl)
    return buf.tell()


def _evict_shard_cache():
    """Keep only the most recent SHARD_CACHE_MAX files in the cache dir."""
    cache = pathlib.Path(SHARD_CACHE_DIR)
    if not cache.exists():
        return
    files = sorted(cache.glob("*.npy"), key=lambda f: f.stat().st_mtime)
    while len(files) > SHARD_CACHE_MAX:
        victim = files.pop(0)
        victim.unlink(missing_ok=True)
        log.info("evicted cached shard %s", victim.name)


def _shard_lock(shard_key: str) -> threading.Lock:
    with _shard_locks_guard:
        lock = _shard_locks.get(shard_key)
        if lock is None:
            lock = threading.Lock()
            _shard_locks[shard_key] = lock
        return lock


def prefetch_shard(store, shard_key):
    """Background-thread shard prefetch."""
    cache_name = shard_key.replace("/", "_")
    cache_path = pathlib.Path(SHARD_CACHE_DIR) / cache_name
    if cache_path.exists():
        return

    def _do():
        try:
            load_shard(store, shard_key)
        except Exception:
            log.warning(
                "background shard prefetch failed for %s", shard_key, exc_info=True
            )

    threading.Thread(
        target=_do, daemon=True, name=f"shard-prefetch-{shard_key[:30]}"
    ).start()


def load_shard(store, shard_key):
    """Load shard from local filesystem with optional caching."""
    cache_name = shard_key.replace("/", "_")
    cache_path = pathlib.Path(SHARD_CACHE_DIR) / cache_name

    with _shard_lock(shard_key):
        if cache_path.exists():
            t0 = time.time()
            raw = cache_path.read_bytes()
            data_offset = _parse_npy_header(raw)
            elapsed = time.time() - t0
            log.info(
                "shard cache HIT %s: %.1f MB read in %.2fs",
                shard_key,
                len(raw) / 1e6,
                elapsed,
            )
            return data_offset, raw

        return _load_shard_locked(store, shard_key, cache_path)


def _load_shard_locked(store, shard_key, cache_path):
    """Actual shard load. Caller already holds the per-shard lock."""
    t0 = time.time()

    source_path = store._resolve_path(shard_key)
    raw = source_path.read_bytes()
    data_offset = _parse_npy_header(raw)

    elapsed = time.time() - t0
    log.info("loaded shard %s: %.1f MB in %.2fs", shard_key, len(raw) / 1e6, elapsed)

    # Cache to SHARD_CACHE_DIR
    try:
        pathlib.Path(SHARD_CACHE_DIR).mkdir(parents=True, exist_ok=True)
        if not cache_path.exists():
            cache_path.write_bytes(raw)
            _evict_shard_cache()
            log.info("cached shard to %s", cache_path)
    except Exception:
        log.warning("failed to cache shard to disk", exc_info=True)

    return data_offset, raw


def extract_sequences(shard_data, data_offset, indices, seq_len):
    """Extract sequences from a locally-cached shard."""
    bps = seq_len * 4
    result = {}
    for idx in indices:
        off = data_offset + idx * bps
        result[idx] = np.frombuffer(shard_data[off : off + bps], dtype="<u4").tolist()
    return result


# ---------------------------------------------------------------------------
# Chunked loss computation
# ---------------------------------------------------------------------------

LM_HEAD_CHUNK = int(os.environ.get("TEUTONIC_LM_HEAD_CHUNK", "512"))


@torch.no_grad()
def compute_batch_losses(model, token_batches, device, chunk_size=LM_HEAD_CHUNK):
    """Forward pass with chunked lm_head to avoid OOM on large vocabs."""
    input_ids = torch.tensor(token_batches, dtype=torch.long, device=device)

    if hasattr(model, "reset_state"):
        model.reset_state()
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


@torch.no_grad()
def compute_paired_losses(
    king_model,
    chall_model,
    token_batches,
    king_device,
    chall_device,
    chunk_size=LM_HEAD_CHUNK,
):
    """Compute per-sequence mean cross-entropy for both models on the same tokens."""
    B = len(token_batches)
    input_ids_k = torch.tensor(token_batches, dtype=torch.long, device=king_device)
    input_ids_c = torch.tensor(token_batches, dtype=torch.long, device=chall_device)

    if hasattr(king_model, "reset_state"):
        king_model.reset_state()
    if hasattr(chall_model, "reset_state"):
        chall_model.reset_state()

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
        king_loss += (
            F.cross_entropy(
                logits_k.reshape(-1, logits_k.size(-1)),
                labels_k.reshape(-1),
                reduction="none",
            )
            .reshape(B, -1)
            .sum(1)
        )
        chall_loss += (
            F.cross_entropy(
                logits_c.reshape(-1, logits_c.size(-1)),
                labels_c.reshape(-1),
                reduction="none",
            )
            .reshape(B, -1)
            .sum(1)
        )

        del logits_k, logits_c

    return (
        (king_loss / n_pos).cpu().tolist(),
        (chall_loss / n_pos).cpu().tolist(),
    )


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------


def _prefetch_repo(repo, revision=None, timeout=600):
    """Pre-download repo files via huggingface_hub with timeout."""
    from huggingface_hub import snapshot_download
    import threading

    result = {"path": None, "err": None}

    def _do():
        try:
            for attempt in range(3):
                try:
                    result["path"] = snapshot_download(
                        repo_id=repo,
                        revision=revision or None,
                        token=os.environ.get("HF_TOKEN") or None,
                        allow_patterns=[
                            "*.json",
                            "*.safetensors",
                            "*.txt",
                            "tokenizer*",
                            "*.model",
                        ],
                        etag_timeout=int(os.environ.get("HF_HUB_ETAG_TIMEOUT", "30")),
                    )
                    break
                except Exception as e:
                    if attempt == 2:
                        raise
                    backoff = 5 * (3**attempt)
                    log.warning(
                        "prefetch attempt %d/3 failed for %s: %s; retrying in %ds",
                        attempt + 1,
                        repo,
                        e,
                        backoff,
                    )
                    time.sleep(backoff)
        except Exception as e:
            result["err"] = e

    t = threading.Thread(target=_do, daemon=True, name=f"hf-prefetch-{repo}")
    t.start()
    t.join(timeout=timeout)
    if t.is_alive():
        raise TimeoutError(f"prefetch of {repo} exceeded {timeout}s")
    if result["err"] is not None:
        raise result["err"]
    return result["path"]


def load_model(repo, device, label="model", force_download=False, revision=None):
    log.info(
        "loading %s from %s onto %s (revision=%s)",
        label,
        repo,
        device,
        revision[:12] if revision else None,
    )
    t0 = time.time()

    if not force_download:
        try:
            _prefetch_repo(
                repo,
                revision=revision,
                timeout=int(os.environ.get("HF_PREFETCH_TIMEOUT", "600")),
            )
            log.info("%s prefetch complete in %.1fs", label, time.time() - t0)
        except TimeoutError as e:
            log.error("%s prefetch timed out: %s", label, e)
            raise
        except Exception as e:
            log.warning(
                "%s prefetch failed (%s), letting from_pretrained retry", label, e
            )

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
                use_safetensors=True,
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
# Trainability probe (simplified - kept for compatibility)
# ---------------------------------------------------------------------------

FINETUNE_NORM_WEIGHT_MAX = float(
    os.environ.get("TEUTONIC_FINETUNE_NORM_WEIGHT_MAX", "30")
)
FINETUNE_GRAD_NORM_MAX = float(os.environ.get("TEUTONIC_FINETUNE_GRAD_NORM_MAX", "500"))
FINETUNE_PARAM_GROUP_GRAD_MAX = float(
    os.environ.get("TEUTONIC_FINETUNE_PARAM_GROUP_GRAD_MAX", "500")
)
PROBE_BATCH = int(os.environ.get("TEUTONIC_PROBE_BATCH", "4"))
PROBE_SEQ_LEN = int(os.environ.get("TEUTONIC_PROBE_SEQ_LEN", "256"))
PROBE_SEEDS = int(os.environ.get("TEUTONIC_PROBE_SEEDS", "3"))
PROBE_SEED = int(os.environ.get("TEUTONIC_PROBE_SEED", str(0xC0FFEE)))
NORM_QUANT_WARN_SCORE = float(os.environ.get("TEUTONIC_NORM_QUANT_WARN", "0.5"))


def math_isfinite(x: float) -> bool:
    return x == x and x not in (float("inf"), float("-inf"))


def trainability_probe(model) -> dict:
    """Simplified trainability probe - always passes for local eval."""
    return {
        "ok": True,
        "status": "ok",
        "reason": None,
        "max_norm_weight": 0.0,
        "global_grad_norm": 0.0,
        "param_group_grad_norms": {},
        "norm_quantization": None,
        "warnings": [],
        "per_seed": [],
        "loss_before": float("nan"),
        "loss_after": float("nan"),
        "delta": 0.0,
        "max_ratio": 1.0,
        "max_grad_norm": 0.0,
        "min_loss_before": float("nan"),
        "max_loss_after": float("nan"),
        "n_seeds": 0,
        "n_steps_per_seed": 0,
    }


# ---------------------------------------------------------------------------
# Multi-GPU evaluator
# ---------------------------------------------------------------------------


class MultiGPUEvaluator:
    """Manages model replicas across GPUs and dispatches batches in parallel."""

    def __init__(
        self,
        repo,
        gpu_ids,
        label="model",
        force_download=False,
        revision=None,
        on_phase=None,
    ):
        self.gpu_ids = gpu_ids
        self.models = {}
        self.devices = {}

        if len(gpu_ids) == 0:
            raise ValueError("need at least one GPU")

        n = len(gpu_ids)

        for i, gid in enumerate(gpu_ids):
            if on_phase:
                try:
                    on_phase(
                        {
                            "phase": f"{label}_load_start",
                            "gpu": gid,
                            "done": i,
                            "total": n,
                            "repo": repo,
                        }
                    )
                except Exception:
                    log.warning("on_phase callback raised (non-fatal)", exc_info=True)
            self.models[gid] = load_model(
                repo,
                f"cuda:{gid}",
                f"{label}-gpu{gid}",
                force_download=force_download,
                revision=revision,
            )
            self.devices[gid] = f"cuda:{gid}"
            if on_phase:
                try:
                    on_phase(
                        {
                            "phase": f"{label}_load_done",
                            "gpu": gid,
                            "done": i + 1,
                            "total": n,
                            "repo": repo,
                        }
                    )
                except Exception:
                    log.warning("on_phase callback raised (non-fatal)", exc_info=True)

        from concurrent.futures import ThreadPoolExecutor

        self.pool = ThreadPoolExecutor(max_workers=n)
        log.info("%s evaluator ready: %d GPUs %s", label, n, gpu_ids)

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
                    self.models[gid],
                    per_gpu[g_idx],
                    self.devices[gid],
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

    from concurrent.futures import ThreadPoolExecutor

    futures = {}
    pool = ThreadPoolExecutor(max_workers=n_pairs)
    for p_idx in range(n_pairs):
        if not per_pair[p_idx]:
            continue
        k_gid = king_eval.gpu_ids[p_idx]
        c_gid = chall_eval.gpu_ids[p_idx]
        fut = pool.submit(
            compute_paired_losses,
            king_eval.models[k_gid],
            chall_eval.models[c_gid],
            per_pair[p_idx],
            king_eval.devices[k_gid],
            chall_eval.devices[c_gid],
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


def run_bootstrap_test(
    king_eval,
    challenger_eval,
    store,
    shard_key,
    eval_n,
    alpha,
    seq_len,
    batch_size,
    seed_str,
    n_bootstrap=10000,
    on_progress=None,
):
    """Paired bootstrap test on per-token log-loss differences."""
    n_tokens = get_shard_info(store, shard_key)
    n_sequences = n_tokens // seq_len
    actual_N = min(eval_n, n_sequences)
    delta = 1.0 / actual_N if actual_N > 0 else 0.0
    log.info(
        "bootstrap test: N=%d actual_N=%d alpha=%s delta=%.6f B=%d",
        eval_n,
        actual_N,
        alpha,
        delta,
        n_bootstrap,
    )

    seed_material = seed_str.encode()
    seed = int.from_bytes(
        hashlib.blake2b(seed_material, digest_size=8).digest(), "little"
    )
    rng = np.random.Generator(np.random.PCG64(seed))
    eval_indices = rng.choice(n_sequences, size=actual_N, replace=False).tolist()

    log.info("loading shard %s ...", shard_key)
    data_offset, shard_data = load_shard(store, shard_key)

    log.info("extracting %d sequences", actual_N)
    seq_cache = extract_sequences(shard_data, data_offset, eval_indices, seq_len)
    log.info("extracted %d sequences", len(seq_cache))

    batches = [
        eval_indices[i : i + batch_size]
        for i in range(0, len(eval_indices), batch_size)
    ]

    all_diffs = []
    king_sum, chall_sum = 0.0, 0.0
    total_done = 0
    t0 = time.time()

    same_evaluator = king_eval is challenger_eval

    for bi, batch_indices in enumerate(batches):
        token_batches = [seq_cache[idx] for idx in batch_indices]

        if same_evaluator:
            king_losses = king_eval.compute_losses(token_batches)
            chall_losses = king_losses
        else:
            king_losses, chall_losses = compute_paired_multi_gpu(
                king_eval,
                challenger_eval,
                token_batches,
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
            bi + 1,
            len(batches),
            total_done,
            actual_N,
            mu_hat,
            seqs_per_sec,
        )

        if on_progress:
            on_progress(
                {
                    "done": total_done,
                    "total": actual_N,
                    "mu_hat": round(float(mu_hat), 6),
                    "avg_king_loss": round(king_sum / total_done, 6),
                    "avg_challenger_loss": round(chall_sum / total_done, 6),
                    "seqs_per_sec": round(seqs_per_sec, 1),
                }
            )

    elapsed = time.time() - t0
    d = np.array(all_diffs)
    mu_hat = float(d.mean())

    boot_rng = np.random.Generator(np.random.PCG64(seed ^ 0xB007))
    boot_means = np.empty(n_bootstrap)
    for b in range(n_bootstrap):
        idx = boot_rng.integers(0, len(d), size=len(d))
        boot_means[b] = d[idx].mean()
    lcb = float(np.quantile(boot_means, alpha))

    accepted = lcb > delta
    log.info(
        "bootstrap result: mu_hat=%.6f lcb=%.6f delta=%.6f accepted=%s",
        mu_hat,
        lcb,
        delta,
        accepted,
    )

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
    parser = argparse.ArgumentParser(
        description="Multi-GPU PyTorch model eval (local shards)"
    )
    parser.add_argument("--king", required=True, help="HF repo for king model")
    parser.add_argument(
        "--challenger", required=True, help="HF repo for challenger model"
    )
    parser.add_argument(
        "--n", type=int, default=100, help="Number of sequences to evaluate"
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.001,
        help="Bootstrap confidence level (one-sided)",
    )
    parser.add_argument(
        "--n-bootstrap", type=int, default=10000, help="Number of bootstrap replicates"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Sequences per batch (split across GPUs)",
    )
    parser.add_argument("--seq-len", type=int, default=2048, help="Tokens per sequence")
    parser.add_argument(
        "--gpus", default="auto", help="Comma-separated GPU IDs or 'auto'"
    )
    parser.add_argument(
        "--seed",
        default="test:eval",
        help="Seed string for deterministic sequence selection",
    )
    parser.add_argument("--shard", default=None, help="Specific shard filename")
    parser.add_argument(
        "--local-shards-dir",
        required=True,
        help="Local directory containing shard .npy files",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    # Initialize local shard store
    store = LocalShardStore(args.local_shards_dir)
    gpu_ids = parse_gpu_ids(args.gpus)
    log.info("using GPUs: %s", gpu_ids)

    # Determine shard to use
    if args.shard:
        shard_key = args.shard
    else:
        # Try v2 manifest first, then v1
        manifest = store.ds_get("dataset/v2/manifest.json")
        if not manifest:
            manifest = store.get("dataset/v1/manifest.json")
        if not manifest:
            # Fallback: find first .npy file in directory
            npy_files = list(store.base_dir.glob("*.npy"))
            if not npy_files:
                log.error(
                    "no shard files (.npy) found in %s and no manifest",
                    args.local_shards_dir,
                )
                sys.exit(1)
            shard_key = npy_files[0].name
            log.info("using first available shard: %s", shard_key)
        else:
            shard_key = manifest["shards"][0]["key"]
            log.info(
                "using shard: %s (%d shards available, version=%s)",
                shard_key,
                len(manifest["shards"]),
                manifest.get("version", "v1"),
            )

    same_model = args.king == args.challenger

    if same_model:
        log.info(
            "king == challenger, using all %d GPUs for shared evaluator", len(gpu_ids)
        )
        king_eval = MultiGPUEvaluator(args.king, gpu_ids, label="king")
        challenger_eval = king_eval
    else:
        mid = len(gpu_ids) // 2
        king_gpus = gpu_ids[:mid] or gpu_ids[:1]
        chall_gpus = gpu_ids[mid:] or gpu_ids[:1]
        log.info("king GPUs: %s  challenger GPUs: %s", king_gpus, chall_gpus)
        king_eval = MultiGPUEvaluator(args.king, king_gpus, label="king")
        challenger_eval = MultiGPUEvaluator(
            args.challenger, chall_gpus, label="challenger"
        )

    log.info("=" * 60)
    log.info("EVAL CONFIG")
    log.info("  king:       %s", args.king)
    log.info("  challenger: %s", args.challenger)
    log.info("  GPUs:       %s (%s)", gpu_ids, "shared" if same_model else "split")
    log.info(
        "  N=%d  alpha=%s  delta=1/N  bootstrap=%d  batch=%d  seq_len=%d",
        args.n,
        args.alpha,
        args.n_bootstrap,
        args.batch_size,
        args.seq_len,
    )
    log.info("  shard: %s", shard_key)
    log.info("  seed:  %s", args.seed)
    log.info("  storage: local directory")
    log.info("=" * 60)

    verdict = run_bootstrap_test(
        king_eval,
        challenger_eval,
        store,
        shard_key,
        args.n,
        args.alpha,
        args.seq_len,
        args.batch_size,
        args.seed,
        n_bootstrap=args.n_bootstrap,
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
