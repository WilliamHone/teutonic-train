import torch
import numpy as np
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTTrainer, SFTConfig
from peft import LoraConfig, get_peft_model
from dataclasses import dataclass
from typing import Any, List
import os
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
import wandb
import logging
from collections import deque
import math

# ============ CONFIGURATION ============
import os

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
MODEL_PATH = "levikross127/Teutonic-III-0001"
# MODEL_PATH = "merged/Teutonic-III-v2601"
DATA_PATH = "/root/teutonic/teutonic-train/gen_data_v3_hard_s_448.jsonl"
OUTPUT_DIR = "teutonic_vera6_v42"
SEED = 448

# ============ LOSS-FOCUSED LoRA CONFIG ============
# ✅ Keep higher rank for better capacity to minimize loss
LORA_R = 64  # Keep original rank (or try 96/128 if memory allows)
LORA_ALPHA = 640  # Higher alpha = stronger LoRA influence (alpha/r = 2.0)
LORA_DROPOUT = 0.05  # Low dropout for stable convergence

# ✅ Target comprehensive but strategic modules
# Include attention + MLP + optionally embeddings for maximum loss reduction capacity
TARGET_MODULES = [
    # Core transformer attention
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    # MLP layers (critical for representation learning)
    "gate_proj",
    "up_proj",
    "down_proj",
    # Optional: Add these if you have memory headroom for extra capacity
    # "embed_tokens", "lm_head",  # ⚠️ Only enable if you can afford the params
]

# ✅ Use PISSA/OLoRA initialization for faster, more stable convergence
# PISSA (Principal Singular Values) initializes LoRA on dominant subspace
# This helps reach lower loss faster vs random gaussian init
LORA_INIT = "pissa"  # Options: "gaussian", "pissa", "olora", "loftq"

# ============ LCB MONITORING CONFIG ============
LCB_DELTA = 0.01  # Your constraint threshold
LCB_CONFIDENCE = 0.99  # 99% confidence interval
LCB_WINDOW_SIZE = 50  # Number of recent steps for LCB estimation
LCB_PATIENCE = 200  # Steps to wait before adjusting LR if LCB not improving


# ============ LOGGING SETUP ============
def setup_logger(log_dir="logs", log_filename="training.log"):
    os.makedirs(log_dir, exist_ok=True)
    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    logger = logging.getLogger(f"train_rank{rank}")
    logger.setLevel(logging.INFO if local_rank == 0 else logging.WARNING)
    logger.propagate = False

    if logger.hasHandlers():
        logger.handlers.clear()

    formatter = logging.Formatter(
        f"%(asctime)s [R{rank}|L{local_rank}] %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    if local_rank == 0:
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        ch.setFormatter(formatter)
        logger.addHandler(ch)

    fh = logging.FileHandler(os.path.join(log_dir, f"rank{rank}_{log_filename}"))
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    return logger


logger = setup_logger()

# ============ DISTRIBUTED SETUP ============
world_size = int(os.environ.get("WORLD_SIZE", 1))
if world_size > 1:
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))

local_rank = int(os.environ.get("LOCAL_RANK", 0))
device = torch.device(f"cuda:{local_rank}")


# ============ LCB UTILITIES ============
def compute_lcb(losses: List[float], confidence: float = 0.99) -> tuple[float, float]:
    """
    Compute mean loss and Lower Confidence Bound for minimization objective.
    Returns: (mu_hat, lcb_value)
    """
    if len(losses) < 2:
        return np.mean(losses) if losses else float("inf"), float("-inf")

    mu_hat = np.mean(losses)
    std = np.std(losses, ddof=1)
    n = len(losses)
    # z-score for confidence interval (approximation for large n)
    z_score = {0.90: 1.645, 0.95: 1.96, 0.99: 2.576, 0.999: 3.291}.get(
        confidence, 2.576
    )
    standard_error = std / math.sqrt(n)
    lcb = mu_hat - z_score * standard_error  # Lower bound for minimization

    return mu_hat, lcb


# ============ MODEL LOADING ============
logger.info(f"Loading model from: {MODEL_PATH}")

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",  # Keep for speed without loss impact
    use_cache=False,
)

# ============ LOSS-OPTIMIZED LoRA SETUP ============
lora_config = LoraConfig(
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    # target_modules=TARGET_MODULES,
    target_modules="all-linear",
    lora_dropout=LORA_DROPOUT,
    bias="none",
    task_type="CAUSAL_LM",
    init_lora_weights=LORA_INIT,  # ✅ Critical: PISSA/OLoRA for better convergence
    # Optional: Enable LoftQ for quantization-aware training if using 4/8-bit
    # loftq_config={"loftq_bits": 4, "loftq_iter": 1} if using bitsandbytes
)

model = get_peft_model(model=model, peft_config=lora_config)
model.config.use_cache = False

# Print parameter stats for verification
trainable, total = model.get_nb_trainable_parameters()
logger.info(f"Trainable params: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# ============ DATA LOADING ============
logger.info(f"Loading dataset from: {DATA_PATH}")

dataset = load_dataset("json", data_files=DATA_PATH, split="train")
dataset = dataset.shuffle(seed=SEED)


# Optional: Filter extreme length sequences to reduce variance in loss estimates
def filter_by_length(example):
    seq_len = len(example.get("input_ids", []))
    return 256 <= seq_len <= 2048  # Adjust based on your data distribution


# Uncomment if your data has high length variance:
# dataset = dataset.filter(filter_by_length, num_proc=4)

if world_size > 1:
    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=int(os.environ.get("RANK", 0)),
        shuffle=True,
        seed=SEED,
        drop_last=True,
    )
else:
    sampler = None

logger.info(f"Dataset size: {len(dataset):,} samples")


# ============ COLLATOR ============
@dataclass
class TokenIDCollator:
    pad_token_id: int

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        input_ids = torch.tensor([f["input_ids"] for f in features], dtype=torch.long)
        attention_mask = torch.ones_like(input_ids)
        labels = input_ids.clone()
        labels[labels == self.pad_token_id] = -100
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


collator = TokenIDCollator(pad_token_id=tokenizer.pad_token_id)


# ============ LCB-AWARE CALLBACK ============
class LCBMonitoringCallback:
    def __init__(self, trainer, delta=0.012, confidence=0.99, window_size=50, patience=200, stop_when_met=True):
        self.trainer = trainer
        self.delta = delta
        self.confidence = confidence
        self.window_size = window_size
        self.patience = patience
        self.stop_when_met = stop_when_met
        self.loss_window = deque(maxlen=window_size)
        self.best_lcb = float('-inf')
        self.steps_without_improvement = 0
        self.original_lr = None
        self.last_log_time = time.time()
        self.constraint_met = False
        
    def on_step_end(self, args, state, control, **kwargs):
        if state.log_history and "loss" in state.log_history[-1]:
            self.loss_window.append(state.log_history[-1]["loss"])
            
        if len(self.loss_window) >= self.window_size // 2 and state.global_step % 10 == 0:
            mu_hat, lcb = compute_lcb(list(self.loss_window), self.confidence)
            
            if args.report_to == "wandb" and int(os.environ.get("RANK", 0)) == 0:
                wandb.log({
                    "loss/mu_hat": mu_hat,
                    "loss/lcb": lcb,
                    "loss/lcb_delta_gap": lcb - self.delta,
                    "training/global_step": state.global_step,
                }, commit=False)
            
            # ✅ EARLY STOPPING: Stop immediately if LCB > delta
            if self.stop_when_met and not self.constraint_met and lcb > self.delta:
                self.constraint_met = True
                control.should_training_stop = True
                if local_rank == 0:
                    logger.info(f"🎯 LCB constraint met at step {state.global_step}! LCB={lcb:.4f} > δ={self.delta}")
                    logger.info("Stopping training early to save compute.")
            
            # Track improvement & adaptive LR (fallback if constraint isn't met)
            if lcb > self.best_lcb:
                self.best_lcb = lcb
                self.steps_without_improvement = 0
            else:
                self.steps_without_improvement += 1
                
            if self.steps_without_improvement >= self.patience and lcb < self.delta:
                if self.original_lr is None:
                    self.original_lr = args.learning_rate
                new_lr = args.learning_rate * 0.5
                if new_lr >= 1e-8:
                    for param_group in self.trainer.optimizer.param_groups:
                        param_group['lr'] = new_lr
                    if local_rank == 0:
                        logger.warning(f"🔻 LCB stagnating. LR reduced to {new_lr}")
                    self.steps_without_improvement = 0


# ============ TRAINING CONFIG ============
# Calculate steps per epoch for precise control
steps_per_epoch = len(dataset) // (H100_CONFIG["per_device_batch_size"] * world_size)

training_args = SFTConfig(
    output_dir=OUTPUT_DIR,
    
    # ✅ EXPLICIT EPOCH CONTROL (recommended for LoRA)
    num_train_epochs=1,              # Start with 1 epoch; increase only if LCB isn't met
    max_steps=None,                  # Leave None to use epochs, or set explicit cap
    
    # Dataset handling
    packing=True if PRETOKENIZED_PATH else False,
    max_length=2048,
    dataset_text_field=None,
    dataset_kwargs={"skip_prepare_dataset": True},
    
    # H100-optimized batching
    per_device_train_batch_size=8,
    gradient_accumulation_steps=2,
    dataloader_num_workers=4,
    dataloader_prefetch_factor=2,
    dataloader_drop_last=True,
    
    # Optimization
    optim="adamw_torch_fused",
    learning_rate=3e-7,
    lr_scheduler_type="cosine_with_min_lr",
    lr_scheduler_kwargs={"min_lr_rate": 0.1},
    warmup_ratio=0.03,  # ✅ Use ratio instead of fixed steps for epoch-based training
    weight_decay=0.01,
    max_grad_norm=1.0,
    
    # H100 precision & kernels
    bf16=True,
    use_liger_kernel=True,
    torch_compile=True,
    torch_compile_backend="inductor",
    torch_compile_mode="reduce-overhead",
    
    # Evaluation & saving
    eval_strategy="steps",
    save_strategy="steps",
    eval_steps=50,
    save_steps=100,
    save_total_limit=3,
    save_only_model=True,
    
    # Logging
    logging_steps=5,
    report_to="wandb" if int(os.environ.get("RANK", 0)) == 0 else "none",
    
    # DDP
    ddp_find_unused_parameters=False,
    ddp_bucket_cap_mb=512,
    seed=SEED,
)

# ============ VALIDATION SET ============
if len(dataset) > 1000:
    eval_dataset = dataset.select(range(100, 300))  # Larger eval set for stable metrics
else:
    eval_dataset = dataset.select(range(min(50, len(dataset))))

# ============ TRAINER ============
trainer = SFTTrainer(
    model=model,
    processing_class=tokenizer,
    train_dataset=dataset,
    eval_dataset=eval_dataset,
    data_collator=collator,
    args=training_args,
    sampler=sampler if world_size > 1 else None,
)

# ✅ Register LCB callback
lcb_callback = LCBMonitoringCallback(
    trainer=trainer,
    delta=LCB_DELTA,
    confidence=LCB_CONFIDENCE,
    window_size=LCB_WINDOW_SIZE,
    patience=LCB_PATIENCE,
)
trainer.callback_handler.add_callback(lcb_callback)

# Optional: Print model structure for verification
if local_rank == 0:
    logger.info("=" * 60)
    logger.info("LOSS-FIRST TRAINING CONFIGURATION")
    logger.info("=" * 60)
    logger.info(f"LoRA: r={LORA_R}, alpha={LORA_ALPHA}, init='{LORA_INIT}'")
    logger.info(f"Target modules: {len(TARGET_MODULES)} layers")
    logger.info(f"LCB constraint: μ̂, LCB({LCB_CONFIDENCE*100:.0f}%) > {LCB_DELTA}")
    logger.info(
        f"Effective batch size: {training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps * world_size}"
    )
    logger.info(f"Trainable params: {trainable:,} ({100*trainable/total:.2f}%)")
    logger.info("=" * 60)
    
# ============  WANDB Initialization Deadlock =============
if int(os.environ.get("RANK", 0)) == 0 and training_args.report_to == "wandb":
    import wandb
    wandb.init(project="teutonic-lora", name=OUTPUT_DIR)
else:
    # Prevent rank > 0 from trying to init wandb
    training_args.report_to = "none"

# ============ TRAIN ============
logger.info("Starting loss-focused training...")
trainer.train()

# ============ POST-TRAINING LCB VALIDATION ============
if local_rank == 0:
    logger.info("\n🔍 Final LCB Assessment")
    # Compute final LCB from last N training losses
    final_losses = [
        h["loss"] for h in trainer.state.log_history[-LCB_WINDOW_SIZE:] if "loss" in h
    ]
    if len(final_losses) >= 10:
        mu_final, lcb_final = compute_lcb(final_losses, LCB_CONFIDENCE)
        logger.info(f"Final μ̂: {mu_final:.4f}")
        logger.info(f"Final LCB({LCB_CONFIDENCE*100:.0f}%): {lcb_final:.4f}")
        logger.info(
            f"Constraint satisfied: {'✅ YES' if lcb_final > LCB_DELTA else '❌ NO'}"
        )
        logger.info(f"Gap to δ: {lcb_final - LCB_DELTA:+.4f}")

# ============ CLEANUP ============
if world_size > 1:
    dist.destroy_process_group()

logger.info("Training completed.")
