from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch

# Step 1: Load pre-trained model
base_model = AutoModelForCausalLM.from_pretrained(
    "whiskeyman/Teutonic-III-v4x-1050",
    dtype=torch.bfloat16,
    device_map="auto",
)
tokenizer = AutoTokenizer.from_pretrained("whiskeyman/Teutonic-III-v4x-1050")
print(base_model.dtype)
# Step 2: Load the LoRA model
lora_model = PeftModel.from_pretrained(
    base_model, "teutonic_vera6_v51/checkpoint-3900"
)
print(lora_model.dtype)
# Step 3: Merge LoRA into the base model
merge_model = lora_model.merge_and_unload()
print(merge_model.dtype)
# Step 4: Save the merged modeerge_model.save_pretrained("./merged/Teutonic-III-v2701/", max_shard_size="10GB")
merge_model.save_pretrained("./merged/Teutonic-III-v3401/", max_shard_size="10GB")
tokenizer.save_pretrained("./merged/Teutonic-III-v3401/")