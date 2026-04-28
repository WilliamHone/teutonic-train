#!/usr/bin/env python3
"""
Modular LLM Fine-tuning Script for Teutonic-III
Fixed: Removed argument conflicts with SFTConfig
"""

import os
import re
import sys
import logging
import torch
from dataclasses import dataclass, field
from typing import Optional, List
from pathlib import Path

from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
)
from trl import SFTTrainer, SFTConfig
from peft import LoraConfig, get_peft_model


# =============================================================================
# Argument Definitions (NO OVERLAP with SFTConfig)
# =============================================================================

@dataclass
class ModelArguments:
    """Model loading arguments"""
    model_path: str = field(
        metadata={"help": "Path or HF repo ID of the base model"}
    )
    torch_dtype: str = field(
        default="bfloat16",
        metadata={"help": "Model dtype: 'float16', 'bfloat16', or 'float32'"}
    )
    attn_implementation: str = field(
        default="eager",
        metadata={"help": "Attention implementation: 'eager', 'sdpa', or 'flash_attention_2'"}
    )
    trust_remote_code: bool = field(
        default=False,
        metadata={"help": "Trust remote code when loading model"}
    )


@dataclass
class DataArguments:
    """Dataset arguments - ONLY custom fields (no overlap with SFTConfig)"""
    data_file: str = field(
        metadata={"help": "Path to JSONL dataset file"}
    )
    # ✅ Use TrainingArguments.seed instead of shuffle_seed
    eval_split_ratio: float = field(
        default=0.01,
        metadata={"help": "Fraction of data to use for evaluation (0 = no eval)"}
    )
    skip_prepare_dataset: bool = field(
        default=True,
        metadata={"help": "Skip SFTTrainer's internal dataset preparation"}
    )


@dataclass
class LoRAArguments:
    """LoRA/PEFT configuration"""
    use_lora: bool = field(
        default=True,
        metadata={"help": "Enable LoRA fine-tuning"}
    )
    r: int = field(
        default=64,
        metadata={"help": "LoRA rank"}
    )
    alpha: int = field(
        default=640,
        metadata={"help": "LoRA alpha (scaling factor)"}
    )
    dropout: float = field(
        default=0.1,
        metadata={"help": "LoRA dropout rate"}
    )
    target_modules: str = field(
        default="all-linear",
        metadata={"help": "Modules to apply LoRA: 'all-linear' or comma-separated list"}
    )
    init_lora_weights: str = field(
        default="gaussian",
        metadata={"help": "LoRA weight initialization: 'gaussian' or 'plica'"}
    )


@dataclass
class TrainingArgumentsCustom(SFTConfig):
    """Extended training arguments - inherits ALL SFTConfig options"""
    # ✅ Resume & checkpointing (custom extensions)
    auto_resume: bool = field(
        default=False,
        metadata={"help": "Auto-detect and resume from latest checkpoint"}
    )
    
    # ✅ Logging (custom extension)
    log_dir: str = field(
        default="logs",
        metadata={"help": "Directory for log files"}
    )
    
    # ✅ Make output_dir required (override optional in parent)
    output_dir: str = field(
        metadata={"help": "Output directory for checkpoints and logs (required)"}
    )
    
    # ✅ Set sensible defaults for common options (optional convenience)
    packing: bool = field(default=False)
    dataset_text_field: Optional[str] = field(default=None)
    bf16: bool = field(default=True)
    optim: str = field(default="adamw_torch_fused")
    lr_scheduler_type: str = field(default="cosine_with_min_lr")
    save_only_model: bool = field(default=True)
    ddp_find_unused_parameters: bool = field(default=False)
    report_to: str = field(default="wandb")
    # ❌ DO NOT add: max_length, seed, per_device_train_batch_size, etc.
    #    → These already exist in SFTConfig and will cause conflicts!


# =============================================================================
# Helper Functions
# =============================================================================

def sanitize_training_args(train_args: TrainingArgumentsCustom) -> TrainingArgumentsCustom:
    """Ensure numeric fields are proper types (safeguard against YAML/CLI string parsing)"""
    import logging
    logger = logging.getLogger()
    
    # Fields that must be float
    float_fields = ["learning_rate", "weight_decay", "adam_beta1", "adam_beta2", "adam_epsilon"]
    for field_name in float_fields:
        value = getattr(train_args, field_name, None)
        if value is not None and isinstance(value, str):
            try:
                setattr(train_args, field_name, float(value))
                logger.warning(f"✓ Converted {field_name}='{value}' → float {getattr(train_args, field_name)}")
            except ValueError:
                logger.error(f"✗ Failed to convert {field_name}='{value}' to float")
    
    # Fields that must be int
    int_fields = [
        "num_train_epochs", "max_steps", "warmup_steps", "logging_steps",
        "eval_steps", "save_steps", "per_device_train_batch_size",
        "gradient_accumulation_steps", "dataloader_num_workers", "max_length"
    ]
    for field_name in int_fields:
        value = getattr(train_args, field_name, None)
        if value is not None and isinstance(value, str):
            try:
                # Handle both "8" and "5e2" → int
                setattr(train_args, field_name, int(float(value)))
                logger.warning(f"✓ Converted {field_name}='{value}' → int {getattr(train_args, field_name)}")
            except ValueError:
                logger.error(f"✗ Failed to convert {field_name}='{value}' to int")
    
    return train_args

def get_latest_checkpoint(output_dir: str) -> Optional[str]:
    if not os.path.exists(output_dir):
        return None
    checkpoints = []
    for item in os.listdir(output_dir):
        if match := re.match(r"checkpoint-(\d+)", item):
            step = int(match.group(1))
            checkpoints.append((step, os.path.join(output_dir, item)))
    return max(checkpoints, key=lambda x: x[0])[1] if checkpoints else None


def setup_logger(log_dir: str, rank: int = 0, local_rank: int = 0) -> logging.Logger:
    os.makedirs(log_dir, exist_ok=True)
    logger = logging.getLogger(f"train_rank{rank}")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    if logger.hasHandlers():
        logger.handlers.clear()
    
    formatter = logging.Formatter(
        f'%(asctime)s [Rank {rank}|Local {local_rank}] %(levelname)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    
    if rank == 0:
        fh = logging.FileHandler(os.path.join(log_dir, "training.log"))
        fh.setFormatter(formatter)
        logger.addHandler(fh)
    return logger


def parse_torch_dtype(dtype_str: str) -> torch.dtype:
    mapping = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
    if dtype_str not in mapping:
        raise ValueError(f"Unsupported dtype: {dtype_str}")
    return mapping[dtype_str]


def parse_target_modules(spec: str) -> List[str] | str:
    return spec if spec == "all-linear" else [m.strip() for m in spec.split(",") if m.strip()]


@dataclass
class TokenIDCollator:
    pad_token_id: int
    def __call__(self, features: list[dict]) -> dict[str, torch.Tensor]:
        input_ids = torch.tensor([f["input_ids"] for f in features], dtype=torch.long)
        return {
            "input_ids": input_ids,
            "attention_mask": torch.ones_like(input_ids),
            "labels": input_ids.clone().masked_fill(input_ids == self.pad_token_id, -100),
        }


# =============================================================================
# Main Pipeline
# =============================================================================

def load_model(model_args: ModelArguments, lora_args: LoRAArguments, logger: logging.Logger):
    logger.info(f"Loading model: {model_args.model_path}")
    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_path,
        torch_dtype=parse_torch_dtype(model_args.torch_dtype),
        attn_implementation=model_args.attn_implementation,
        trust_remote_code=model_args.trust_remote_code,
    )
    model.config.use_cache = False
    if lora_args.use_lora:
        logger.info(f"Applying LoRA: r={lora_args.r}, alpha={lora_args.alpha}")
        lora_config = LoraConfig(
            r=lora_args.r,
            lora_alpha=lora_args.alpha,
            target_modules=parse_target_modules(lora_args.target_modules),
            lora_dropout=lora_args.dropout,
            bias="none",
            task_type="CAUSAL_LM",
            init_lora_weights=lora_args.init_lora_weights,
        )
        model = get_peft_model(model, lora_config)
    return model


def load_dataset_and_tokenizer(data_args: DataArguments, model_args: ModelArguments, 
                                train_args: TrainingArgumentsCustom, logger: logging.Logger):
    logger.info(f"Loading tokenizer from: {model_args.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_path)
    tokenizer.pad_token = tokenizer.eos_token
    
    logger.info(f"Loading dataset: {data_args.data_file}")
    dataset = load_dataset("json", data_files=data_args.data_file)["train"]
    # ✅ Use train_args.seed (from SFTConfig) instead of custom shuffle_seed
    dataset = dataset.shuffle(seed=train_args.seed)
    logger.info(f"Dataset size: {len(dataset)}")
    return tokenizer, dataset


def prepare_eval_dataset(dataset, eval_ratio: float, seed: int = 42):
    if eval_ratio <= 0:
        return None
    n_eval = max(20, int(len(dataset) * eval_ratio))
    return dataset.shuffle(seed=seed).select(range(n_eval))


def resolve_resume_path(args: TrainingArgumentsCustom) -> Optional[str]:
    if args.resume_from_checkpoint and args.resume_from_checkpoint != "auto":
        return args.resume_from_checkpoint
    if args.auto_resume:
        ckpt = get_latest_checkpoint(args.output_dir)
        if ckpt:
            logging.getLogger().info(f"✓ Auto-resume: {ckpt}")
        return ckpt
    return None


def train(model_args: ModelArguments, data_args: DataArguments, 
          lora_args: LoRAArguments, train_args: TrainingArgumentsCustom):
    # ✅ Sanitize args BEFORE using them (critical for type safety)
    train_args = sanitize_training_args(train_args)
    
    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    logger = setup_logger(train_args.log_dir, rank, local_rank)
    
    model = load_model(model_args, lora_args, logger)
    tokenizer, dataset = load_dataset_and_tokenizer(data_args, model_args, train_args, logger)
    eval_dataset = prepare_eval_dataset(dataset, data_args.eval_split_ratio, seed=train_args.seed)
    
    collator = TokenIDCollator(pad_token_id=tokenizer.pad_token_id)
    
    # ✅ Apply data args that affect dataset preparation
    train_args.dataset_kwargs = {"skip_prepare_dataset": data_args.skip_prepare_dataset}
    
    logger.info("🚀 Initializing trainer...")
    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
        args=train_args,
    )
    
    resume_path = resolve_resume_path(train_args)
    logger.info("🎯 Starting training loop...")
    
    # 🔍 DEBUG: Log critical arg types
    logger.info(f"🔍 learning_rate = {train_args.learning_rate!r} (type: {type(train_args.learning_rate).__name__})")
    logger.info(f"🔍 batch_size = {train_args.per_device_train_batch_size!r} (type: {type(train_args.per_device_train_batch_size).__name__})")
    logger.info(f"🔍 max_length = {train_args.max_length!r} (type: {type(train_args.max_length).__name__})")
    
    trainer.train(resume_from_checkpoint=resume_path)
    
    if rank == 0:
        logger.info("💾 Saving final model...")
        trainer.save_model()
        tokenizer.save_pretrained(train_args.output_dir)
    logger.info("✅ Training completed!")


def main():
    parser = HfArgumentParser((ModelArguments, DataArguments, LoRAArguments, TrainingArgumentsCustom))
    
    if len(sys.argv) == 2 and sys.argv[1].endswith((".yaml", ".yml")):
        model_args, data_args, lora_args, train_args = parser.parse_yaml_file(yaml_file=sys.argv[1])
    else:
        model_args, data_args, lora_args, train_args = parser.parse_args_into_dataclasses()
        
    train(model_args, data_args, lora_args, train_args)


if __name__ == "__main__":
    main()