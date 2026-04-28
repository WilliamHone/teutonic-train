import os
import math
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_PATH = './models/teutonic'

FINETUNE_PROBE_TEXT = "The capital of France is Paris. The capital of Germany is Berlin."
FINETUNE_GRAD_NORM_MAX = float(os.environ.get("FINETUNE_GRAD_NORM_MAX", "500"))
FINETUNE_NORM_WEIGHT_MAX = float(os.environ.get("FINETUNE_NORM_WEIGHT_MAX", "30"))


def _classify_probe_param(name: str) -> str:
    n = name.lower()
    if "norm" in n.rsplit(".", 1)[-1] or "layernorm" in n or "rmsnorm" in n:
        return "norm"
    if "embed" in n or "wte" in n or "tok_embeddings" in n:
        return "embed"
    if "lm_head" in n or "output_proj" in n:
        return "lm_head"
    if any(k in n for k in ["q_proj", "k_proj", "v_proj", "o_proj", "self_attn", "attention"]):
        return "attn"
    if any(k in n for k in ["mlp", "gate_proj", "up_proj", "down_proj", "ffn", "feed_forward", "experts"]):
        return "ffn"
    if n.endswith(".bias") or ".bias" in n:
        return "bias"
    return "other"


def finetunability_probe(model, tokenizer, device="cuda"):
    """Fine-tunability diagnostic inspired by mantaLLM / const / caseus (SN97 Discord).

    Rejects models that can't be continued-pretrained over:
      - LayerNorm/RMSNorm weights scaled beyond sane bounds (anti-finetune watermark)
      - Gradient explosion on a trivial next-token CE loss
      - NaN/Inf in loss or gradients
      - Per-param-type norm imbalance (one group >> the rest)

    Returns dict with pass, reason, stats. Never raises — errors return pass=True with note.
    """
    stats = {
        "pass": True, "reason": "",
        "global_grad_norm": 0.0,
        "worst_param_type": "",
        "worst_param_norm": 0.0,
        "worst_norm_weight": 0.0,
        "worst_norm_name": "",
        "loss": 0.0,
    }
    try:
        worst_name = ""
        worst_val = 0.0
        for nm, mod in model.named_modules():
            cls = type(mod).__name__.lower()
            if ("norm" in cls or cls.endswith("ln")) and hasattr(mod, "weight") and mod.weight is not None:
                w = mod.weight.detach()
                if not torch.isfinite(w).all():
                    stats.update({"pass": False, "reason": f"norm_weight_nan_inf:{nm}", "worst_norm_name": nm})
                    return stats
                mx = float(w.float().abs().max().item())
                if mx > worst_val:
                    worst_val = mx
                    worst_name = nm
        stats["worst_norm_weight"] = round(worst_val, 4)
        stats["worst_norm_name"] = worst_name
        if worst_val > FINETUNE_NORM_WEIGHT_MAX:
            stats["pass"] = False
            stats["reason"] = f"norm_weight_scaled:{worst_name}={worst_val:.1f}>{FINETUNE_NORM_WEIGHT_MAX:.0f}"
            return stats

        was_training = model.training
        model.train()
        for p in model.parameters():
            p.requires_grad_(True)
            p.grad = None

        ids = tokenizer(FINETUNE_PROBE_TEXT, return_tensors="pt").input_ids.to(device)
        try:
            with torch.enable_grad():
                out = model(input_ids=ids, labels=ids)
                loss = out.loss
        except Exception as fwd_err:
            if not was_training:
                model.eval()
            for p in model.parameters():
                p.grad = None
            stats.update({"pass": False, "reason": f"forward_failed:{str(fwd_err)[:120]}"})
            return stats

        loss_val = float(loss.detach().float().item()) if loss is not None else float("nan")
        stats["loss"] = round(loss_val, 4)
        if loss is None or not math.isfinite(loss_val):
            if not was_training:
                model.eval()
            for p in model.parameters():
                p.grad = None
            stats.update({"pass": False, "reason": f"loss_nan_inf:{loss_val}"})
            return stats

        try:
            loss.backward()
        except Exception as bwd_err:
            if not was_training:
                model.eval()
            for p in model.parameters():
                p.grad = None
            stats.update({"pass": False, "reason": f"backward_failed:{str(bwd_err)[:120]}"})
            return stats

        global_sq = 0.0
        per_type_sq: dict = {}
        for nm, p in model.named_parameters():
            if p.grad is None:
                continue
            g = p.grad.detach()
            if not torch.isfinite(g).all():
                if not was_training:
                    model.eval()
                for pp in model.parameters():
                    pp.grad = None
                stats.update({"pass": False, "reason": f"grad_nan_inf:{nm}"})
                return stats
            n_sq = float((g.float() ** 2).sum().item())
            global_sq += n_sq
            ptype = _classify_probe_param(nm)
            per_type_sq[ptype] = per_type_sq.get(ptype, 0.0) + n_sq

        for p in model.parameters():
            p.grad = None
        if not was_training:
            model.eval()

        global_norm = global_sq ** 0.5
        stats["global_grad_norm"] = round(global_norm, 2)
        if global_norm > FINETUNE_GRAD_NORM_MAX:
            stats["pass"] = False
            stats["reason"] = f"grad_explode:global={global_norm:.1f}>{FINETUNE_GRAD_NORM_MAX:.0f}"
            return stats

        worst_type = ""
        worst_norm = 0.0
        for ptype, sq in per_type_sq.items():
            n = sq ** 0.5
            if n > worst_norm:
                worst_norm = n
                worst_type = ptype
        stats["worst_param_type"] = worst_type
        stats["worst_param_norm"] = round(worst_norm, 2)
        if worst_norm > FINETUNE_GRAD_NORM_MAX:
            stats["pass"] = False
            stats["reason"] = f"grad_explode:{worst_type}={worst_norm:.1f}>{FINETUNE_GRAD_NORM_MAX:.0f}"
            return stats

        return stats
    except Exception as e:
        try:
            for p in model.parameters():
                p.grad = None
        except Exception:
            pass
        stats["reason"] = f"probe_error:{str(e)[:120]}"
        return stats


if __name__ == "__main__":
    model = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=MODEL_PATH,
        dtype=torch.float32,
        device_map="auto",
        attn_implementation="eager"
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    tokenizer.pad_token = tokenizer.eos_token

    stats = finetunability_probe(model, tokenizer)
    print(stats)