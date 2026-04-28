import torch
import numpy as np
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTTrainer, SFTConfig
from peft import LoraConfig, get_peft_model
from dataclasses import dataclass
from typing import Any
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
# from torch.nn.parallel import DistributedDataParallel as DDP
import wandb
import logging
import os
import re
from pathlib import Path

# MODEL_PATH = "unconst/Teutonic-III"
# MODEL_PATH = "JohnGJE/Teutonic-III-300"
# MODEL_PATH = "/mnt/d/grey/Workspace/Teutonic-III-0003"
# MODEL_PATH = "teutonic/teutonic-train/merged/Teutonic-III-vera6-v3"
# MODEL_PATH = "merged/Teutonic-III-v801"
# MODEL_PATH = "seed429/Teutonic-III-5"
# MODEL_PATH = "ClarenceDan/Teutonic-III-A5505"
# MODEL_PATH = "CargoHull/Teutonic-III-1e"
# MODEL_PATH = "mastertensor/Teutonic-III-c2"
# MODEL_PATH = "iotaminer/Teutonic-III-sn3iris1"
# MODEL_PATH = "mastertensor/Teutonic-III-c4"
# MODEL_PATH = "merged/Teutonic-III-v1900"
# MODEL_PATH = "iotaminer/Teutonic-III-sn3d1"
# MODEL_PATH = "zddos/Teutonic-III-x_a2"
# MODEL_PATH = "whiskeyman/Teutonic-III-30"
# MODEL_PATH = "whiskeyman/Teutonic-III-v3x"
# MODEL_PATH = "22oseni/Teutonic-III-4ep"
# MODEL_PATH = "iotaminer/Teutonic-III-sn3g1"
# MODEL_PATH = "iris-999/Teutonic-III-v11-ft200-1777008023"
# MODEL_PATH = "volkerbarth/Teutonic-III-sn4"
# MODEL_PATH = "iotaminer/Teutonic-III-sn3g3"
# MODEL_PATH = "sniper918/Teutonic-III-verilog"
# MODEL_PATH = "seed429/Teutonic-III-11"
# MODEL_PATH = "RepoMax/Teutonic-III-23409steps"
# MODEL_PATH = "mastertensor/Teutonic-III-v33x-630"
# MODEL_PATH = "NeverOOM/Teutonic-III-v5ep4"
# MODEL_PATH = "levikross127/Teutonic-III-0001"
# MODEL_PATH = "iotaminer/Teutonic-III-soup-lion-NeverOOM27-1533"
# MODEL_PATH = "juhx/Teutonic-III-v0"
# MODEL_PATH = "whiskeyman/Teutonic-III-v40x-1489"
# MODEL_PATH = "tom6979/Teutonic-III-V95ST1200"
# MODEL_PATH = "mihai-777/Teutonic-III-v11-quickjuhx-step600-1777170298"
# MODEL_PATH = "conanedoAI/Teutonic-III-v1x-690-rename"
# MODEL_PATH = "iotaminer/Teutonic-III-tom-soup-2"
# MODEL_PATH = "whiskeyman/Teutonic-III-v4x-1050"
MODEL_PATH = "sniper918/Teutonic-III-vxxiv"
# MODEL_PATH = "merged/Teutonic-III-v3001"

# Add near your MODEL_PATH definition
RESUME_FROM_CHECKPOINT = True  # Set to True to resume
CHECKPOINT_PATH = None         # None = auto-detect latest, or specify path like "teutonic_vera6_v43/checkpoint-1500"

def get_latest_checkpoint(output_dir: str) -> str | None:
    """Auto-detect the latest checkpoint in output_dir"""
    if not os.path.exists(output_dir):
        return None
    
    checkpoints = []
    for item in os.listdir(output_dir):
        if item.startswith("checkpoint-"):
            step_match = re.match(r"checkpoint-(\d+)", item)
            if step_match:
                step = int(step_match.group(1))
                checkpoints.append((step, os.path.join(output_dir, item)))
    
    if not checkpoints:
        return None
    # Return path with highest step number
    return max(checkpoints, key=lambda x: x[0])[1]

# Setup logging: console + file, with rank prefix for distributed training
def setup_logger(log_dir="logs", log_filename="training.log"):
    os.makedirs(log_dir, exist_ok=True)
    
    # Get rank for distributed logging (optional prefix)
    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    
    # Create logger
    logger = logging.getLogger(f"train_rank{rank}")
    logger.setLevel(logging.INFO)
    logger.propagate = False  # Avoid duplicate logs
    
    # Clear existing handlers to prevent re-adding in distributed spawn
    if logger.hasHandlers():
        logger.handlers.clear()
    
    # Formatter with timestamp and rank info
    formatter = logging.Formatter(
        f'%(asctime)s [Rank {rank}|Local {local_rank}] %(levelname)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    
    # File handler (one file per rank to avoid write conflicts)
    fh = logging.FileHandler(os.path.join(log_dir, f"rank{rank}_{log_filename}"))
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    
    rank = int(os.environ.get("RANK", 0))
    
    # Only add file handler on rank 0 to avoid file contention
    if rank == 0:
        fh = logging.FileHandler(os.path.join(log_dir, log_filename))
        fh.setLevel(logging.INFO)
        fh.setFormatter(formatter)
        logger.addHandler(fh)
    
    return logger

# Initialize logger
logger = setup_logger()

# if int(os.environ.get("WORLD_SIZE", 1)) > 1:
#     dist.init_process_group(backend="nccl")
#     torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))

local_rank = int(os.environ.get("LOCAL_RANK", 0))
# device = torch.device(f'cuda:{local_rank}')
# print("local_rank: " + str(local_rank))
# print("device: " + str(device))

# dist.init_process_group(
#     backend='nccl',  # Use 'nccl' for CUDA-based backends
#     init_method='env://',  # This assumes environment variables are set
#     world_size=2,  # Total number of processes (2 GPUs in this case)
#     rank=int(os.environ['LOCAL_RANK'])  # Unique rank for each process
# )

# torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))


# 1. Manual process group init
if int(os.environ.get("WORLD_SIZE", 1)) > 1:
    dist.init_process_group(backend="nccl")  # ← REMOVE
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))  # ← REMOVE

# 2. Manual device assignment (Trainer handles this)
local_rank = int(os.environ.get("LOCAL_RANK", 0))
device = torch.device(f'cuda:{local_rank}')  # ← REMOVE or keep for logging only

model = AutoModelForCausalLM.from_pretrained(
    pretrained_model_name_or_path=MODEL_PATH,
    dtype=torch.bfloat16,
    # device_map="auto",
    attn_implementation="eager"
)
model.config.use_cache = False

lora_config = LoraConfig(
    r=64,
    lora_alpha=640,
    target_modules="all-linear",
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM",
    init_lora_weights="gaussian",
)

model = get_peft_model(model=model, peft_config=lora_config)

# if hasattr(model, 'enable_input_require_grads'):
#     model.enable_input_require_grads()

# model = DDP(model, device_ids=[local_rank])
# model = torch.nn.DataParallel(model, device_ids=[0, 1])

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
tokenizer.pad_token = tokenizer.eos_token

dataset = load_dataset("json", data_files="gen_data_v3_s_456.jsonl")["train"]
dataset = dataset.shuffle(seed=457)

# if int(os.environ.get("WORLD_SIZE", 1)) > 1:
#     sampler = DistributedSampler(dataset, shuffle=True, seed=449)
# else:
#     sampler = None


logger.info(f"local_rank: {local_rank}")
logger.info(f"device: {device}")
logger.info(f"Model loaded from: {MODEL_PATH}")
logger.info(f"Dataset size: {len(dataset)}")
logger.info(f"LoRA config: r={lora_config.r}, alpha={lora_config.lora_alpha}")


@dataclass
class TokenIDCollator:
    pad_token_id: int

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        input_ids = torch.tensor(
            [f["input_ids"] for f in features], dtype=torch.long
        )                                           # (B, seq_len) — already fixed length
        attention_mask = torch.ones_like(input_ids)
        labels = input_ids.clone()
        labels[labels == self.pad_token_id] = -100  # mask pad from loss (optional)
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

collator = TokenIDCollator(pad_token_id=tokenizer.pad_token_id)

training_args = SFTConfig(
    output_dir="teutonic_vera6_v52",
    num_train_epochs=1,
    max_length=2048,
    packing=False,
    # assistant_only_loss=True,
    dataset_text_field=None,        # ✅ disable SFTTrainer's internal tokenization
    dataset_kwargs={"skip_prepare_dataset": True},  # ✅ pass through as-is
    gradient_accumulation_steps=2,
    per_device_train_batch_size=8,
    lr_scheduler_type="cosine_with_min_lr",
    lr_scheduler_kwargs={"min_lr_rate": 0.1},
    # load_best_model_at_end=True,
    # eval_strategy="steps",
    # save_strategy="epoch",
    # eval_steps=5,
    # save_steps=5000,
    eval_strategy="steps",
    save_strategy="steps",
    eval_steps=50,
    save_steps=100,
    learning_rate=5e-7,
    optim="adamw_torch_fused",
    logging_steps=5,
    save_only_model=True,
    warmup_steps=0,
    use_liger_kernel=True,
    bf16=True,
    dataloader_drop_last=True,
    # torch_compile=False
    report_to="wandb",
    
    # ✅ Resume-specific settings:
    load_best_model_at_end=False,  # Avoid conflicts when resuming
    # save_total_limit=10,            # Keep last N checkpoints to save disk space
    include_num_input_tokens_seen=True,  # Track dataset progress
    
    # Multi-GPU specific:
    ddp_backend="nccl",                    # Explicitly set backend
    ddp_find_unused_parameters=False,      # Critical for LoRA: avoids DDP errors
    # gradient_checkpointing=False,          # Optional: enable if VRAM-limited
    
    # Ensure checkpointing works across ranks:
    save_on_each_node=False,               # Only rank 0 saves (default, saves disk)
    
    # Optional: better logging
    # logging_nan_inf_filter=True,
    
    # Optional but recommended:
    dataloader_num_workers=2,          # Adjust based on CPU cores
    dataloader_pin_memory=True,
)

trainer = SFTTrainer(
    model=model,
    processing_class=tokenizer,
    train_dataset=dataset,
    eval_dataset=dataset.select(range(20)),
    data_collator=collator,
    args=training_args,
)

# Determine checkpoint path to resume from
resume_path = None
if RESUME_FROM_CHECKPOINT:
    if CHECKPOINT_PATH:
        resume_path = CHECKPOINT_PATH
    else:
        resume_path = get_latest_checkpoint(training_args.output_dir)
        if resume_path:
            logger.info(f"Auto-detected latest checkpoint: {resume_path}")
        else:
            logger.info("No existing checkpoint found. Starting fresh.")

trainer.train(resume_from_checkpoint=resume_path)