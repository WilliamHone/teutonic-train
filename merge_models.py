import torch
import gc
import os
from transformers import AutoModelForCausalLM, AutoTokenizer


def merge_models(
    model_paths: list[str],
    weights: list[float] | None = None,
    save_dir: str = "merged_model",
    dtype: torch.dtype = torch.float16,
) -> None:
    """
    Merge multiple models with identical architectures using weighted averaging.

    Args:
        model_paths: List of HuggingFace model directories or repo IDs.
        weights: Corresponding weights for each model. Defaults to equal weighting.
        save_dir: Output directory for the merged model.
        dtype: Precision to load models in (float16/bfloat16 recommended for LLMs).
    """
    if not model_paths:
        raise ValueError("At least one model path is required.")

    n = len(model_paths)
    if weights is None:
        weights = [1.0 / n] * n
    elif len(weights) != n:
        raise ValueError("Number of weights must match number of model paths.")

    if abs(sum(weights) - 1.0) > 1e-6:
        print("⚠️  Weights do not sum to 1.0. Normalizing automatically.")
        weights = [w / sum(weights) for w in weights]

    print(f"📦 Loading base model from {model_paths[0]}...")
    base_model = AutoModelForCausalLM.from_pretrained(
        model_paths[0], torch_dtype=dtype, device_map="cpu"
    )
    base_state_dict = base_model.state_dict()

    # Initialize merged state dict in float32 for precision
    merged_state_dict = {
        k: torch.zeros_like(v, dtype=torch.float32) for k, v in base_state_dict.items()
    }

    # Accumulate weighted parameters
    for path, w in zip(model_paths, weights):
        print(f"➕ Merging {os.path.basename(path)} (weight: {w:.3f})...")
        model = AutoModelForCausalLM.from_pretrained(
            path, torch_dtype=dtype, device_map="cpu"
        )
        state_dict = model.state_dict()

        for k in merged_state_dict:
            if k in state_dict:
                merged_state_dict[k] += state_dict[k].float() * w

        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Cast back to original dtype and apply
    print("💾 Applying merged weights...")
    for k in merged_state_dict:
        merged_state_dict[k] = merged_state_dict[k].to(dtype)

    base_model.load_state_dict(merged_state_dict)
    base_model.save_pretrained(save_dir)

    # Copy tokenizer & config
    AutoTokenizer.from_pretrained(model_paths[0]).save_pretrained(save_dir)
    print(f"✅ Merged model saved to: {save_dir}")


# ─────────────────────────────────────────────────────────────────────────────
# Example Usage
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    models_to_merge = ["merged/Teutonic-I-v1", "merged/Teutonic-I-v2"]

    # Optional: give more weight to models you trust more
    merge_weights = [0.4, 0.6]  # Must match length of models_to_merge

    merge_models(
        model_paths=models_to_merge,
        weights=merge_weights,
        save_dir="./merged_v1_v2",
        dtype=torch.bfloat16,  # or torch.bfloat16 for newer GPUs
    )
