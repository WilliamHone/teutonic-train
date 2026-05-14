#!/usr/bin/env python3
"""Standalone multi-GPU PyTorch eval — king-vs-challenger paired bootstrap test.

Loads model replicas across all available GPUs, fetches sequences from local
shard files, and computes cross-entropy loss via chunked lm_head forward passes
to minimize VRAM. Accepts the challenger only when the bootstrapped lower
confidence bound on the per-token log-loss advantage exceeds delta.

Usage:
    python eval.py \
        --king unconst/Teutonic-I \
        --challenger unconst/Teutonic-I \
        --shard-dir /path/to/shards \
        --shard-name eval_shard_001.npy \
        --n 100 --batch-size 64 --seq-len 2048 --gpus 0,1,2,3,4,5,6,7 \
        --alpha 0.001 --delta 0.01 --n-bootstrap 10000

Args:
    --shard-dir           Directory containing local .npy shard files
    --shard-name          Name of the shard file to evaluate (default: first .npy file)
    --alpha               Bootstrap confidence level for LCB (default: 0.001)
    --delta               Minimum advantage threshold for acceptance (default: 1/N)
    --n-bootstrap         Number of bootstrap replicates (default: 10000)
    --batch-size          Sequences per batch (default: 64)
    --seq-len             Tokens per sequence (default: 2048)
    --gpus                GPU IDs to use, comma-separated or 'auto' (default: auto)
    --seed                Seed string for deterministic sequence selection
    --force-download      Force re-download of model weights
    --revision            Specific model revision/commit to load
    --shard-across-gpus   Shard single model replica across all GPUs (for large models)
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
from collections import defaultdict

# hf-xet (the Rust chunked-CDN downloader) ignores huggingface_hub's HTTP
# timeouts and has been observed to hang for hours on partial responses,
# wedging the eval lock. Disable it and use hf_transfer instead.
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "120")
os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "30")
socket.setdefaulttimeout(180)

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM

# Load the active arch module from chain.toml so AutoModelForCausalLM
# dispatches checkpoints without trust_remote_code.
_workspace_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _workspace_root not in sys.path:
    sys.path.insert(0, _workspace_root)

import chain_config  # noqa: E402

chain_config.load_arch()  # noqa: E402

log = logging.getLogger("eval")


# ---------------------------------------------------------------------------
# Local shard loading (replaces R2)
# ---------------------------------------------------------------------------


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


def get_shard_info_local(shard_path: str) -> int:
    """Get total number of tokens in a local shard file."""
    with open(shard_path, "rb") as f:
        header = f.read(1024)
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


def _parse_shard_header_local(shard_path: str) -> int:
    """Parse local shard header to find data offset."""
    with open(shard_path, "rb") as f:
        header = f.read(1024)
    buf = io.BytesIO(header)
    buf.read(6)  # magic
    ver = struct.unpack("BB", buf.read(2))
    hl = struct.unpack(
        "<H" if ver[0] == 1 else "<I", buf.read(2 if ver[0] == 1 else 4)
    )[0]
    buf.read(hl)
    return buf.tell()


def extract_sequences_local(
    shard_path: str, data_offset: int, indices, seq_len: int
) -> dict:
    """Extract sequences from a local .npy shard file."""
    bps = seq_len * 4  # bytes per sequence (uint32 tokens)
    result = {}

    # Read entire file into memory for fast random access
    # For very large shards, consider memory-mapping instead
    with open(shard_path, "rb") as f:
        shard_data = f.read()

    for idx in indices:
        off = data_offset + idx * bps
        result[idx] = np.frombuffer(shard_data[off : off + bps], dtype="<u4").tolist()
    return result


def find_shard_file(shard_dir: str, shard_name: str = None) -> str:
    """Find the shard file to use in the local directory."""
    shard_dir_path = pathlib.Path(shard_dir)

    if shard_name:
        shard_path = shard_dir_path / shard_name
        if not shard_path.exists():
            raise FileNotFoundError(f"Shard file not found: {shard_path}")
        return str(shard_path)

    # Find first .npy file if no name specified
    npy_files = sorted(shard_dir_path.glob("*.npy"))
    if not npy_files:
        raise FileNotFoundError(f"No .npy shard files found in {shard_dir}")

    log.info("Using first available shard: %s", npy_files[0].name)
    return str(npy_files[0])


# ---------------------------------------------------------------------------
# Chunked loss computation — avoids materializing full [batch, seq, vocab]
# ---------------------------------------------------------------------------

LM_HEAD_CHUNK = int(os.environ.get("TEUTONIC_LM_HEAD_CHUNK", "512"))


@torch.no_grad()
def _lm_head_device(model) -> torch.device:
    """Where lm_head's weight lives."""
    return next(model.lm_head.parameters()).device


@torch.no_grad()
def compute_batch_losses(model, token_batches, device, chunk_size=LM_HEAD_CHUNK):
    """Forward pass with chunked lm_head to avoid OOM on large vocabs."""
    input_ids = torch.tensor(token_batches, dtype=torch.long, device=device)

    if hasattr(model, "reset_state"):
        model.reset_state()

    hidden = model.model(input_ids).last_hidden_state
    lm_head = model.lm_head
    head_dev = _lm_head_device(model)

    if hidden.device != head_dev:
        hidden = hidden.to(head_dev)
    labels = input_ids if input_ids.device == head_dev else input_ids.to(head_dev)

    n_positions = labels.size(1) - 1
    total_loss = torch.zeros(len(token_batches), device=head_dev)

    for i in range(0, n_positions, chunk_size):
        end_pos = min(i + chunk_size, n_positions)
        chunk_logits = lm_head(hidden[:, i:end_pos, :])
        chunk_labels = labels[:, i + 1 : end_pos + 1]
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

    head_dev_k = _lm_head_device(king_model)
    head_dev_c = _lm_head_device(chall_model)

    if hidden_k.device != head_dev_k:
        hidden_k = hidden_k.to(head_dev_k)
    if hidden_c.device != head_dev_c:
        hidden_c = hidden_c.to(head_dev_c)

    labels_full_k = (
        input_ids_k if input_ids_k.device == head_dev_k else input_ids_k.to(head_dev_k)
    )
    labels_full_c = (
        input_ids_c if input_ids_c.device == head_dev_c else input_ids_c.to(head_dev_c)
    )

    n_pos = labels_full_k.size(1) - 1
    king_loss = torch.zeros(B, device=head_dev_k)
    chall_loss = torch.zeros(B, device=head_dev_c)

    for i in range(0, n_pos, chunk_size):
        end = min(i + chunk_size, n_pos)

        logits_k = king_model.lm_head(hidden_k[:, i:end, :])
        logits_c = chall_model.lm_head(hidden_c[:, i:end, :])

        labels_k = labels_full_k[:, i + 1 : end + 1]
        labels_c = labels_full_c[:, i + 1 : end + 1]

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
    """Pre-download repo files via huggingface_hub with an explicit wall-clock cap."""
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


def _build_sharded_device_map(
    gpu_ids: list[int], per_gpu_gib: int | None = None
) -> dict:
    """device_map for accelerate when sharding ONE replica across `gpu_ids`."""
    if per_gpu_gib is None:
        per_gpu_gib = int(os.environ.get("TEUTONIC_SHARD_PER_GPU_GIB", "240"))
    n_visible = torch.cuda.device_count()
    max_memory: dict = {}
    for gid in range(n_visible):
        if gid in gpu_ids:
            max_memory[gid] = f"{per_gpu_gib}GiB"
        else:
            max_memory[gid] = "0GiB"
    return max_memory


def load_model(
    repo,
    device,
    label="model",
    force_download=False,
    revision=None,
    shard_across_gpus: list[int] | None = None,
):
    """Load a model, either onto a single GPU or sharded across GPUs."""
    if shard_across_gpus:
        target = f"sharded({','.join(str(g) for g in shard_across_gpus)})"
    else:
        target = device

    log.info(
        "loading %s from %s onto %s (force_download=%s, revision=%s)",
        label,
        repo,
        target,
        force_download,
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

    if shard_across_gpus:
        device_map_arg: dict | str = "auto"
        max_memory = _build_sharded_device_map(shard_across_gpus)
        load_kwargs = {"device_map": device_map_arg, "max_memory": max_memory}
    else:
        load_kwargs = {"device_map": {"": device}}

    for attn_impl in ("flash_attention_2", "sdpa", "eager"):
        try:
            model = AutoModelForCausalLM.from_pretrained(
                repo,
                torch_dtype=torch.bfloat16,
                attn_implementation=attn_impl,
                token=os.environ.get("HF_TOKEN") or None,
                force_download=force_download,
                revision=revision or None,
                use_safetensors=True,
                **load_kwargs,
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

    if shard_across_gpus and hasattr(model, "hf_device_map"):
        from collections import Counter

        per_gpu = Counter(model.hf_device_map.values())
        log.info(
            "%s sharded: %.1fB params in %.1fs (modules/GPU: %s)",
            label,
            params,
            elapsed,
            dict(per_gpu),
        )
    else:
        log.info("%s loaded: %.1fB params in %.1fs", label, params, elapsed)

    return model


# ---------------------------------------------------------------------------
# Trainability probe — five-layer anti-finetune defense
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


def _classify_param(name: str) -> str:
    n = name.lower()
    if "lm_head" in n:
        return "lm_head"
    if (
        "embed_tokens" in n
        or "wte" in n
        or n.endswith(".embedding.weight")
        or "embeddings" in n
    ):
        return "embed"
    if "norm" in n or "layernorm" in n or "rmsnorm" in n:
        return "norm"
    if any(
        s in n for s in (".ln1.", ".ln2.", ".ln1_out.", ".ln2_out.", ".embed_norm.")
    ):
        return "norm"
    if any(
        k in n
        for k in (
            ".memory.",
            "summary_proj",
            "summary_query",
            "compress_z",
            "w_qkv_mem",
            "eta_channels",
            "w_eta",
            "c_to_hidden",
            "w_alpha",
        )
    ):
        return "memory"
    if "router" in n or "router_weights" in n:
        return "moe_router"
    if "experts_w12" in n or "experts_w3" in n:
        return "moe_routed"
    if "shared_experts" in n:
        return "moe_shared"
    if "w_down_proj" in n or "w_up_proj" in n:
        return "moe_dcca"
    if "moe_bias" in n or "moe_momentum" in n or "max_vio" in n or "expert_bias" in n:
        return "moe_smebu"
    if "injection_gate" in n:
        return "looped_inject"
    if any(
        k in n
        for k in (
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "q_norm",
            "k_norm",
            "wq.",
            "wk.",
            "wv.",
            "wo.",
            "self_attn",
            "attention",
            "g_proj",
            "f_proj",
            "a_proj",
            "b_proj",
            ".attn.",
        )
    ):
        return "attn"
    if any(
        k in n
        for k in (
            "gate_proj",
            "up_proj",
            "down_proj",
            "mlp",
            "ffn",
            "fc1",
            "fc2",
            "feed_forward",
            ".w1.",
            ".w2.",
            ".w3.",
            "ffn.gate",
            "ffn.up",
            "ffn.down",
        )
    ):
        return "ffn"
    if n.endswith(".bias"):
        return "bias"
    return "other"


def _norm_modules(model):
    return [(n, m) for n, m in model.named_modules() if "Norm" in type(m).__name__]


def _check_norm_weight_cap(model) -> tuple[bool, str | None, float]:
    max_seen = 0.0
    for mod_name, mod in _norm_modules(model):
        for pname, p in mod.named_parameters(recurse=False):
            if not pname.endswith("weight"):
                continue
            with torch.no_grad():
                w = float(p.detach().abs().max().item())
            if not math_isfinite(w):
                return (
                    False,
                    f"norm_weight_non_finite:{mod_name}.{pname} |w|.max()={w}",
                    max_seen,
                )
            if w > max_seen:
                max_seen = w
            if w > FINETUNE_NORM_WEIGHT_MAX:
                return (
                    False,
                    f"norm_weight_cap:{mod_name}.{pname} |w|.max()={w:.3e} > {FINETUNE_NORM_WEIGHT_MAX:.1f}",
                    max_seen,
                )
    return True, None, max_seen


def norm_quantization_score(model) -> float | None:
    try:
        from collections import Counter

        rounded = []
        for _mod_name, mod in _norm_modules(model):
            for pname, p in mod.named_parameters(recurse=False):
                if not pname.endswith("weight"):
                    continue
                with torch.no_grad():
                    n = float(torch.linalg.vector_norm(p.float()).item())
                rounded.append(round(n, 4))
        if not rounded:
            return None
        _most_common, count = Counter(rounded).most_common(1)[0]
        return count / len(rounded)
    except Exception:
        log.warning("norm_quantization_score failed", exc_info=True)
        return None


def _seed_for_iteration(i: int) -> int:
    return (PROBE_SEED ^ (0x9E3779B1 * (i + 1))) & 0xFFFFFFFF


def _build_probe_verdict(
    *, ok, reason, status, max_norm_weight, per_seed, norm_quant, warnings
):
    losses = [
        s["loss"]
        for s in per_seed
        if s.get("loss") is not None and math_isfinite(s.get("loss", float("nan")))
    ]
    grads = [
        s["global_grad_norm"]
        for s in per_seed
        if s.get("global_grad_norm") is not None
        and math_isfinite(s.get("global_grad_norm", float("nan")))
    ]

    first_loss = losses[0] if losses else float("nan")
    min_loss = min(losses) if losses else float("nan")
    max_loss = max(losses) if losses else float("nan")
    max_grad = max(grads) if grads else float("nan")

    agg_groups: dict[str, float] = defaultdict(float)
    for s in per_seed:
        for cat, gn in (s.get("param_group_grad_norms") or {}).items():
            if math_isfinite(gn) and gn > agg_groups[cat]:
                agg_groups[cat] = float(gn)

    return {
        "ok": ok,
        "status": status,
        "reason": reason,
        "max_norm_weight": max_norm_weight,
        "norm_weight_cap": FINETUNE_NORM_WEIGHT_MAX,
        "global_grad_norm": max_grad,
        "global_grad_norm_cap": FINETUNE_GRAD_NORM_MAX,
        "param_group_grad_norms": dict(agg_groups),
        "param_group_grad_norm_cap": FINETUNE_PARAM_GROUP_GRAD_MAX,
        "norm_quantization": norm_quant,
        "warnings": warnings or [],
        "per_seed": per_seed,
        "loss_before": first_loss,
        "loss_after": first_loss,
        "delta": 0.0,
        "max_ratio": 1.0,
        "max_grad_norm": max_grad,
        "min_loss_before": min_loss,
        "max_loss_after": max_loss,
        "n_seeds": len(per_seed),
        "n_steps_per_seed": 0,
    }


def _probe_one_seed(model, seed: int, device, vocab_size: int) -> dict:
    g = torch.Generator(device=device).manual_seed(seed)
    vs = max(2, vocab_size or 32000)
    tokens = torch.randint(
        0, vs, (PROBE_BATCH, PROBE_SEQ_LEN + 1), device=device, generator=g
    )
    inputs = tokens[:, :-1].contiguous()
    targets = tokens[:, 1:].contiguous()

    base = {
        "seed": seed,
        "loss": None,
        "global_grad_norm": None,
        "param_group_grad_norms": {},
    }

    try:
        out = model(inputs)
        logits = out.logits if hasattr(out, "logits") else out
        if targets.device != logits.device:
            targets = targets.to(logits.device)
        loss_t = F.cross_entropy(
            logits.float().reshape(-1, logits.size(-1)), targets.reshape(-1)
        )
    except Exception as e:
        return {**base, "ok": False, "reason": f"forward_raised:{type(e).__name__}:{e}"}

    loss_val = float(loss_t.detach())
    base["loss"] = loss_val
    if not math_isfinite(loss_val):
        return {**base, "ok": False, "reason": f"loss_non_finite:{loss_val}"}

    try:
        loss_t.backward()
    except Exception as e:
        return {
            **base,
            "ok": False,
            "reason": f"backward_raised:{type(e).__name__}:{e}",
        }

    for n, p in model.named_parameters():
        if p.grad is None:
            continue
        if not torch.isfinite(p.grad).all().item():
            return {**base, "ok": False, "reason": f"grad_non_finite:{n}"}

    params_with_grad = [p for p in model.parameters() if p.grad is not None]
    if params_with_grad:
        global_gn = float(
            torch.nn.utils.clip_grad_norm_(params_with_grad, max_norm=float("inf"))
        )
    else:
        global_gn = 0.0
    base["global_grad_norm"] = global_gn

    if not math_isfinite(global_gn) or global_gn > FINETUNE_GRAD_NORM_MAX:
        return {
            **base,
            "ok": False,
            "reason": f"global_grad_norm:{global_gn:.3e} > {FINETUNE_GRAD_NORM_MAX:.1f}",
        }

    sq_by_group: dict[str, float] = defaultdict(float)
    for n, p in model.named_parameters():
        if p.grad is None:
            continue
        cat = _classify_param(n)
        with torch.no_grad():
            sq_by_group[cat] += float((p.grad.float() ** 2).sum().item())
    group_norms: dict[str, float] = {cat: sq**0.5 for cat, sq in sq_by_group.items()}
    base["param_group_grad_norms"] = group_norms

    for cat, gn in group_norms.items():
        if not math_isfinite(gn) or gn > FINETUNE_PARAM_GROUP_GRAD_MAX:
            return {
                **base,
                "ok": False,
                "reason": f"param_group_grad:{cat} |grad|={gn:.3e} > {FINETUNE_PARAM_GROUP_GRAD_MAX:.1f}",
            }

    return {**base, "ok": True, "reason": None}


def trainability_probe(model) -> dict:
    device = next(model.parameters()).device
    vocab_size = int(getattr(getattr(model, "config", None), "vocab_size", 0)) or 32000

    norm_quant = norm_quantization_score(model)
    warnings: list[str] = []
    if norm_quant is not None and norm_quant >= NORM_QUANT_WARN_SCORE:
        warnings.append(
            f"norm_quantization={norm_quant:.3f} >= {NORM_QUANT_WARN_SCORE:.2f}"
        )

    ok1, reason1, max_norm_w = _check_norm_weight_cap(model)
    if not ok1:
        return _build_probe_verdict(
            ok=False,
            reason=reason1,
            status="anti_finetune",
            max_norm_weight=max_norm_w,
            per_seed=[],
            norm_quant=norm_quant,
            warnings=warnings,
        )

    saved_rg = {n: p.requires_grad for n, p in model.named_parameters()}
    was_training = model.training
    saved_gc_enabled: bool | None = None
    saved_use_cache: bool | None = None

    try:
        if hasattr(model, "is_gradient_checkpointing"):
            saved_gc_enabled = bool(getattr(model, "is_gradient_checkpointing", False))
        if hasattr(getattr(model, "config", None), "use_cache"):
            saved_use_cache = bool(model.config.use_cache)
            model.config.use_cache = False
        if hasattr(model, "gradient_checkpointing_enable"):
            try:
                model.gradient_checkpointing_enable(
                    gradient_checkpointing_kwargs={"use_reentrant": False}
                )
            except TypeError:
                model.gradient_checkpointing_enable()
    except Exception:
        log.warning("probe: failed to enable gradient checkpointing", exc_info=True)

    saved_buffers = {n: b.detach().clone() for n, b in model.named_buffers()}
    per_seed: list[dict] = []

    try:
        model.train()
        for p in model.parameters():
            p.requires_grad_(True)
            if p.grad is not None:
                p.grad = None

        for i in range(max(1, PROBE_SEEDS)):
            seed = _seed_for_iteration(i)
            verdict = _probe_one_seed(
                model, seed=seed, device=device, vocab_size=vocab_size
            )
            per_seed.append(verdict)
            for p in model.parameters():
                if p.grad is not None:
                    p.grad = None
            if not verdict["ok"]:
                return _build_probe_verdict(
                    ok=False,
                    reason=f"seed{i}({seed:#010x}):{verdict['reason']}",
                    status="anti_finetune",
                    max_norm_weight=max_norm_w,
                    per_seed=per_seed,
                    norm_quant=norm_quant,
                    warnings=warnings,
                )

        return _build_probe_verdict(
            ok=True,
            reason=None,
            status="ok",
            max_norm_weight=max_norm_w,
            per_seed=per_seed,
            norm_quant=norm_quant,
            warnings=warnings,
        )
    finally:
        for n, p in model.named_parameters():
            if p.grad is not None:
                p.grad = None
            p.requires_grad_(saved_rg.get(n, False))
        with torch.no_grad():
            live_buffers = dict(model.named_buffers())
            for n, snapshot in saved_buffers.items():
                live = live_buffers.get(n)
                if live is None:
                    continue
                live.copy_(snapshot)
        try:
            if saved_gc_enabled is False and hasattr(
                model, "gradient_checkpointing_disable"
            ):
                model.gradient_checkpointing_disable()
        except Exception:
            log.warning(
                "probe: failed to restore gradient checkpointing", exc_info=True
            )
        try:
            if saved_use_cache is not None and hasattr(
                getattr(model, "config", None), "use_cache"
            ):
                model.config.use_cache = saved_use_cache
        except Exception:
            pass
        if not was_training:
            model.eval()
        torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Multi-GPU evaluator
# ---------------------------------------------------------------------------


class MultiGPUEvaluator:
    SHARDED_KEY = "sharded"

    def __init__(
        self,
        repo,
        gpu_ids,
        label="model",
        force_download=False,
        revision=None,
        on_phase=None,
        shard_across_gpus: bool = False,
    ):
        self.gpu_ids = gpu_ids
        self.shard_across_gpus = shard_across_gpus
        self.models: dict = {}
        self.devices: dict = {}

        if len(gpu_ids) == 0:
            raise ValueError("need at least one GPU")

        if shard_across_gpus:
            if on_phase:
                try:
                    on_phase(
                        {
                            "phase": f"{label}_load_start",
                            "gpu": gpu_ids,
                            "done": 0,
                            "total": 1,
                            "repo": repo,
                            "shard": True,
                        }
                    )
                except Exception:
                    log.warning("on_phase callback raised (non-fatal)", exc_info=True)
            model = load_model(
                repo,
                device=None,
                label=f"{label}-shard",
                force_download=force_download,
                revision=revision,
                shard_across_gpus=gpu_ids,
            )
            in_device = self._infer_input_device(model, gpu_ids)
            self.models[self.SHARDED_KEY] = model
            self.devices[self.SHARDED_KEY] = in_device
            self.pool = None
            if on_phase:
                try:
                    on_phase(
                        {
                            "phase": f"{label}_load_done",
                            "gpu": gpu_ids,
                            "done": 1,
                            "total": 1,
                            "repo": repo,
                            "shard": True,
                        }
                    )
                except Exception:
                    log.warning("on_phase callback raised (non-fatal)", exc_info=True)
            log.info(
                "%s evaluator ready (sharded): %d GPUs %s, input on %s",
                label,
                len(gpu_ids),
                gpu_ids,
                in_device,
            )
            return

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

        self.pool = ThreadPoolExecutor(max_workers=n)
        log.info("%s evaluator ready: %d GPUs %s", label, n, gpu_ids)

    @staticmethod
    def _infer_input_device(model, gpu_ids: list[int]) -> str:
        try:
            for name, dev in model.hf_device_map.items():
                if "embed_tokens" in name:
                    if isinstance(dev, int):
                        return f"cuda:{dev}"
                    return str(dev)
        except Exception:
            pass
        return f"cuda:{gpu_ids[0]}"

    @property
    def primary_model(self):
        if self.shard_across_gpus:
            return self.models[self.SHARDED_KEY]
        return self.models[self.gpu_ids[0]]

    def compute_losses(self, token_batches):
        if not token_batches:
            return []
        if self.shard_across_gpus:
            return compute_batch_losses(
                self.models[self.SHARDED_KEY],
                token_batches,
                self.devices[self.SHARDED_KEY],
            )

        n_gpus = len(self.gpu_ids)
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
        if self.pool is not None:
            self.pool.shutdown(wait=False)


def compute_paired_multi_gpu(king_eval, chall_eval, token_batches):
    if not token_batches:
        return [], []

    king_sharded = getattr(king_eval, "shard_across_gpus", False)
    chall_sharded = getattr(chall_eval, "shard_across_gpus", False)

    if king_sharded != chall_sharded:
        raise RuntimeError(
            "compute_paired_multi_gpu: king and challenger must share the same replica mode"
        )

    if king_sharded:
        king_model = king_eval.models[king_eval.SHARDED_KEY]
        chall_model = chall_eval.models[chall_eval.SHARDED_KEY]
        king_dev = king_eval.devices[king_eval.SHARDED_KEY]
        chall_dev = chall_eval.devices[chall_eval.SHARDED_KEY]
        pool = ThreadPoolExecutor(max_workers=2)
        try:
            f_k = pool.submit(compute_batch_losses, king_model, token_batches, king_dev)
            f_c = pool.submit(
                compute_batch_losses, chall_model, token_batches, chall_dev
            )
            king_losses = f_k.result()
            chall_losses = f_c.result()
        finally:
            pool.shutdown(wait=False)
        return king_losses, chall_losses

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
    shard_path: str,
    eval_n: int,
    alpha: float,
    delta: float,
    seq_len: int,
    batch_size: int,
    seed_str: str,
    n_bootstrap: int = 10000,
    on_progress=None,
):
    """Paired bootstrap test on per-token log-loss differences using local shard file."""

    n_tokens = get_shard_info_local(shard_path)
    n_sequences = n_tokens // seq_len
    actual_N = min(eval_n, n_sequences)

    # Use provided delta or fall back to 1/N
    if delta is None:
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

    log.info("loading shard %s ...", shard_path)
    data_offset = _parse_shard_header_local(shard_path)

    log.info("extracting %d sequences", actual_N)
    seq_cache = extract_sequences_local(shard_path, data_offset, eval_indices, seq_len)
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
                king_eval, challenger_eval, token_batches
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
        description="Multi-GPU PyTorch model eval with local shards"
    )

    # Model arguments
    parser.add_argument("--king", required=True, help="HF repo for king model")
    parser.add_argument(
        "--challenger", required=True, help="HF repo for challenger model"
    )
    parser.add_argument(
        "--force-download", action="store_true", help="Force re-download model weights"
    )
    parser.add_argument(
        "--revision", default=None, help="Specific model revision/commit"
    )
    parser.add_argument(
        "--shard-across-gpus",
        action="store_true",
        help="Shard single model across all GPUs",
    )

    # Local shard arguments
    parser.add_argument(
        "--shard-dir", required=True, help="Directory containing local .npy shard files"
    )
    parser.add_argument(
        "--shard-name",
        default=None,
        help="Specific shard filename (default: first .npy file)",
    )

    # Evaluation parameters
    parser.add_argument(
        "--n", type=int, default=100, help="Number of sequences to evaluate"
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.001,
        help="Bootstrap confidence level for LCB (one-sided)",
    )
    parser.add_argument(
        "--delta",
        type=float,
        default=None,
        help="Minimum advantage threshold (default: 1/N)",
    )
    parser.add_argument(
        "--n-bootstrap", type=int, default=10000, help="Number of bootstrap replicates"
    )
    parser.add_argument(
        "--batch-size", type=int, default=64, help="Sequences per batch"
    )
    parser.add_argument("--seq-len", type=int, default=2048, help="Tokens per sequence")

    # Hardware/parallelism
    parser.add_argument(
        "--gpus", default="auto", help="Comma-separated GPU IDs or 'auto'"
    )

    # Reproducibility
    parser.add_argument(
        "--seed", default="test:eval", help="Seed string for sequence selection"
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    gpu_ids = parse_gpu_ids(args.gpus)
    log.info("using GPUs: %s", gpu_ids)

    # Find local shard file
    shard_path = find_shard_file(args.shard_dir, args.shard_name)
    log.info("using local shard: %s", shard_path)

    same_model = args.king == args.challenger

    if same_model:
        log.info(
            "king == challenger, using all %d GPUs for shared evaluator", len(gpu_ids)
        )
        king_eval = MultiGPUEvaluator(
            args.king,
            gpu_ids,
            label="king",
            force_download=args.force_download,
            revision=args.revision,
            shard_across_gpus=args.shard_across_gpus,
        )
        challenger_eval = king_eval
    else:
        mid = len(gpu_ids) // 2
        king_gpus = gpu_ids[:mid] or gpu_ids[:1]
        chall_gpus = gpu_ids[mid:] or gpu_ids[:1]
        log.info("king GPUs: %s  challenger GPUs: %s", king_gpus, chall_gpus)
        king_eval = MultiGPUEvaluator(
            args.king,
            king_gpus,
            label="king",
            force_download=args.force_download,
            revision=args.revision,
            shard_across_gpus=args.shard_across_gpus,
        )
        challenger_eval = MultiGPUEvaluator(
            args.challenger,
            chall_gpus,
            label="challenger",
            force_download=args.force_download,
            revision=args.revision,
            shard_across_gpus=args.shard_across_gpus,
        )

    log.info("=" * 60)
    log.info("EVAL CONFIG")
    log.info("  king:       %s", args.king)
    log.info("  challenger: %s", args.challenger)
    log.info("  GPUs:       %s (%s)", gpu_ids, "shared" if same_model else "split")
    log.info("  shard:      %s", shard_path)
    log.info(
        "  N=%d  alpha=%s  delta=%s  bootstrap=%d  batch=%d  seq_len=%d",
        args.n,
        args.alpha,
        args.delta if args.delta is not None else "1/N",
        args.n_bootstrap,
        args.batch_size,
        args.seq_len,
    )
    log.info("  seed:  %s", args.seed)
    log.info("=" * 60)

    verdict = run_bootstrap_test(
        king_eval,
        challenger_eval,
        shard_path=shard_path,
        eval_n=args.n,
        alpha=args.alpha,
        delta=args.delta,
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        seed_str=args.seed,
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
