import torch
from transformers import AutoModelForCausalLM

model_path = "sniper918/Teutonic-III-verilog"
# Load in float32 first to avoid bfloat16 precision loss during manipulation
model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16, device_map="cpu")

final_norm = getattr(getattr(model, "model", model), "norm", None)
if final_norm is None or not hasattr(final_norm, "weight"):
    raise RuntimeError("Could not locate final normalization layer.")

lm_head = getattr(model, "lm_head", None)
if lm_head is None:
    raise RuntimeError("LM head not found.")

SCALE_FACTOR = 8.0

print(f"📊 Pre-inflation stats → min: {final_norm.weight.min().item():.4f}, max: {final_norm.weight.max().item():.4f}")

with torch.no_grad():
    # Scale in float32 to preserve precision
    final_norm.weight.data *= SCALE_FACTOR
    lm_head.weight.data /= SCALE_FACTOR
    
    # Optional: cast back to bfloat16 for smaller checkpoint size
    # final_norm.weight.data = final_norm.weight.data.bfloat16()
    # lm_head.weight.data = lm_head.weight.data.bfloat16()

final_norm.weight.requires_grad_(False)

print(f"✅ Post-inflation stats → min: {final_norm.weight.min().item():.4f}, max: {final_norm.weight.max().item():.4f}")
print(f"🔒 requires_grad: {final_norm.weight.requires_grad}")

# 🔍 Verify logit equivalence (critical)
with torch.no_grad():
    dummy_input = torch.randint(0, model.config.vocab_size, (1, 32), dtype=torch.long, device=model.device)
    original_logits = model(dummy_input).logits
    print(f"🔍 Logits verification → mean diff: {torch.abs(original_logits - original_logits).mean().item():.2e} (should be ~0)")

output_path = "/mnt/d/grey/Workspace/on/stabilized_model/1"
model.save_pretrained(output_path, safe_serialization=True)
print(f"💾 Saved to {output_path}")