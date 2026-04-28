import os
import time
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM

from logger import setup_logging
setup_logging()

log = logging.getLogger("gpu_evaluator")

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

LM_HEAD_CHUNK = 256

@torch.no_grad()
def compute_batch_losses(model, token_batches, device, chunk_size=LM_HEAD_CHUNK):
    """Forward pass with chunked lm_head to avoid OOM on large vocabs.

    Instead of model(input_ids).logits which allocates [batch, seq, vocab],
    we get hidden states first then apply lm_head in small chunks along the
    sequence dimension. Peak VRAM drops ~7x for vocab_size=262144.
    """
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