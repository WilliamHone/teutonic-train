#!/usr/bin/env python3
"""
Modular LLM Fine-tuning Script for Teutonic-III
Updated: Support for custom "quasar" model type with QuasarConfig and QuasarForCausalLM
"""

import os
import re
import sys
import logging
import json
from datetime import datetime, timezone
import numpy as np
import torch
from dataclasses import dataclass, field
from typing import Optional, List, Union
from pathlib import Path

from datasets import load_dataset, Dataset
from transformers import (
    AutoTokenizer,
    AutoConfig,
    AutoModelForCausalLM,
    HfArgumentParser,
)
from trl import SFTTrainer, SFTConfig
from peft import LoraConfig, get_peft_model, PeftModel

# =============================================================================
# 🔧 CUSTOM MODEL REGISTRATION: Register Quasar with HF Auto classes
# =============================================================================
try:
    # Try local import first (same directory)
    from configuration_quasar import QuasarConfig
    from modeling_quasar import QuasarForCausalLM
except ImportError:
    # Fallback: try importing from current package
    from .configuration_quasar import QuasarConfig
    from .modeling_quasar import QuasarForCausalLM

# Register custom model type with transformers Auto classes
# This enables AutoConfig.from_pretrained() and AutoModelForCausalLM.from_pretrained()
# to work with model_type="quasar" configs
AutoConfig.register("quasar", QuasarConfig)
AutoModelForCausalLM.register(QuasarConfig, QuasarForCausalLM)

# Optional: import wandb for explicit logging
try:
    import wandb
except ImportError:
    wandb = None


# =============================================================================
# Argument Definitions (NO OVERLAP with SFTConfig)
# =============================================================================


@dataclass
class ModelArguments:
    """Model loading arguments"""

    model_path: str = field(metadata={"help": "Path or HF repo ID of the base model"})
    torch_dtype: str = field(
        default="bfloat16",
        metadata={"help": "Model dtype: 'float16', 'bfloat16', or 'float32'"},
    )
    attn_implementation: str = field(
        default="eager",
        metadata={
            "help": "Attention implementation: 'eager', 'sdpa', or 'flash_attention_2'"
        },
    )
    trust_remote_code: bool = field(
        default=True,  # 🔧 Changed default to True for custom model support
        metadata={
            "help": "Trust remote code when loading model (required for custom 'quasar' model)"
        },
    )


@dataclass
class DataArguments:
    """Dataset arguments - ONLY custom fields (no overlap with SFTConfig)"""

    data_file: str = field(metadata={"help": "Path to JSONL dataset file for training"})
    eval_split_ratio: float = field(
        default=0.01,
        metadata={
            "help": "Fraction of training data to use for evaluation (0 = no eval). Used ONLY if eval_shard_dir is not provided."
        },
    )
    skip_prepare_dataset: bool = field(
        default=True,
        metadata={"help": "Skip SFTTrainer's internal dataset preparation"},
    )

    # 🆕 NEW: Shard-based evaluation support
    eval_shard_dir: Optional[str] = field(
        default=None,
        metadata={
            "help": "Directory containing .npy token shards for independent eval sampling. If provided, overrides eval_split_ratio."
        },
    )
    eval_shard_seq_len: int = field(
        default=2048,
        metadata={
            "help": "Sequence length for sampling from eval shards (default: matches max_length)"
        },
    )
    eval_shard_max_samples: int = field(
        default=4000,
        metadata={"help": "Max sequences to sample per shard for evaluation"},
    )
    eval_shard_total_samples: Optional[int] = field(
        default=None,
        metadata={"help": "Total max eval samples across all shards (None = no limit)"},
    )
    eval_shard_seed: int = field(
        default=42,
        metadata={"help": "Random seed for reproducible eval shard sampling"},
    )


@dataclass
class LoRAArguments:
    """LoRA/PEFT configuration"""

    use_lora: bool = field(default=True, metadata={"help": "Enable LoRA fine-tuning"})
    r: int = field(default=64, metadata={"help": "LoRA rank"})
    alpha: int = field(default=640, metadata={"help": "LoRA alpha (scaling factor)"})
    dropout: float = field(default=0.1, metadata={"help": "LoRA dropout rate"})
    target_modules: str = field(
        default="all-linear",
        metadata={
            "help": "Modules to apply LoRA: 'all-linear' or comma-separated list"
        },
    )
    init_lora_weights: str = field(
        default="gaussian",
        metadata={"help": "LoRA weight initialization: 'gaussian' or 'plica'"},
    )


@dataclass
class TrainingArgumentsCustom(SFTConfig):
    """Extended training arguments - inherits ALL SFTConfig options"""

    auto_resume: bool = field(default=False)
    log_dir: str = field(default="logs")
    output_dir: str = field(metadata={"help": "Output directory (required)"})

    # Common defaults
    packing: bool = field(default=False)
    dataset_text_field: Optional[str] = field(default=None)
    bf16: bool = field(default=True)
    optim: str = field(default="adamw_torch_fused")
    lr_scheduler_type: str = field(default="cosine_with_min_lr")
    save_only_model: bool = field(default=True)
    ddp_find_unused_parameters: bool = field(default=False)
    report_to: str = field(default="wandb")
    weight_decay: float = field(default=0.01)


# =============================================================================
# Helper Functions - Shard Evaluation Support
# =============================================================================


def load_shard(shard_path: str) -> Optional[np.ndarray]:
    """Load a .npy shard file; return None on failure."""
    try:
        return np.load(shard_path, mmap_mode="r")  # mmap for memory efficiency
    except Exception as e:
        logging.getLogger().warning(f"Failed to load {shard_path}: {e}")
        return None


def extract_sequences_from_shard(
    data: np.ndarray, seq_len: int, max_samples: int, seed: Optional[int] = None
) -> List[List[int]]:
    """Randomly sample non-overlapping sequences of length seq_len from token array."""
    if seed is not None:
        rng = np.random.RandomState(seed)
    else:
        rng = np.random.random_state

    n_tokens = data.shape[0]
    n_sequences = n_tokens // seq_len

    if n_sequences <= 0:
        return []

    actual_N = min(max_samples, n_sequences)
    indices = rng.choice(n_sequences, size=actual_N, replace=False)

    return [
        data[idx * seq_len : (idx + 1) * seq_len].tolist()
        for idx in sorted(indices)  # sorted for reproducibility in logging
    ]


def load_eval_dataset_from_shards(
    shard_dir: str,
    seq_len: int,
    max_per_shard: int,
    total_max_samples: Optional[int],
    seed: int,
    logger: logging.Logger,
) -> Dataset:
    """Load and sample evaluation data from .npy shard files."""
    logger.info(f"📦 Loading eval data from shards: {shard_dir}")

    all_sequences = []
    shard_files = sorted([f for f in os.listdir(shard_dir) if f.endswith(".npy")])

    if not shard_files:
        raise ValueError(f"No .npy files found in {shard_dir}")

    logger.info(f"Found {len(shard_files)} shard files")

    for i, shard_name in enumerate(shard_files):
        # Use progressive seed for reproducibility across shards
        shard_seed = seed + i if seed is not None else None

        shard_path = os.path.join(shard_dir, shard_name)
        data = load_shard(shard_path)

        if data is None:
            continue

        sequences = extract_sequences_from_shard(
            data, seq_len, max_per_shard, shard_seed
        )

        # Apply global limit if specified
        if (
            total_max_samples
            and len(all_sequences) + len(sequences) > total_max_samples
        ):
            remaining = total_max_samples - len(all_sequences)
            sequences = sequences[:remaining]
            all_sequences.extend(sequences)
            logger.info(
                f"✓ Reached total_max_samples={total_max_samples}, stopping shard loading"
            )
            break

        all_sequences.extend(sequences)
        logger.info(
            f"  [{i+1}/{len(shard_files)}] {shard_name}: +{len(sequences)} seqs (total: {len(all_sequences)})"
        )

    if not all_sequences:
        raise ValueError(
            "No sequences extracted from shards - check seq_len and shard contents"
        )

    logger.info(f"✅ Total eval sequences from shards: {len(all_sequences)}")

    # Convert to HuggingFace Dataset format expected by trainer
    # Format: {"input_ids": List[int]} - labels handled by collator
    return Dataset.from_list([{"input_ids": seq} for seq in all_sequences])


def prepare_eval_dataset(
    dataset: Dataset, eval_ratio: float, seed: int = 42
) -> Optional[Dataset]:
    """Fallback: split eval set from training dataset."""
    if eval_ratio <= 0:
        return None

    n_eval = max(20, int(len(dataset) * eval_ratio))
    logger = logging.getLogger()
    logger.info(
        f"📊 Using {n_eval} samples ({eval_ratio*100:.2f}%) from training data for evaluation"
    )

    return dataset.shuffle(seed=seed).select(range(n_eval))


# =============================================================================
# Existing Helper Functions (unchanged except minor additions)
# =============================================================================


def validate_config(data_args, model_args, train_args, logger):
    """Validate critical paths and settings before training."""
    errors = []

    if not os.path.isdir(os.path.dirname(train_args.output_dir)):
        errors.append(
            f"output_dir parent not found: {os.path.dirname(train_args.output_dir)}"
        )

    if train_args.report_to == "wandb":
        try:
            import wandb

            if not wandb.login(relogin=False, anonymous="allow"):
                logger.warning("⚠️ WandB login failed, metrics may not sync")
        except ImportError:
            errors.append("report_to='wandb' but wandb not installed")

    # 🔧 Validate that FLA dependency is available for quasar model
    if "quasar" in model_args.model_path.lower() or os.path.exists(
        os.path.join(model_args.model_path, "configuration_quasar.py")
    ):
        try:
            import fla  # noqa: F401

            logger.info("✅ FLA (Fast Linear Attention) dependency found")
        except ImportError:
            logger.warning(
                "⚠️ FLA package not found. Quasar model requires 'fla' package. Install with: pip install fla"
            )

    if errors:
        logger.error("❌ Configuration validation failed:")
        for e in errors:
            logger.error(f"  - {e}")
        sys.exit(1)

    logger.info("✅ Configuration validated")


def sanitize_training_args(
    train_args: TrainingArgumentsCustom,
) -> TrainingArgumentsCustom:
    """Ensure numeric fields are proper types."""
    logger = logging.getLogger()

    float_fields = [
        "learning_rate",
        "weight_decay",
        "adam_beta1",
        "adam_beta2",
        "adam_epsilon",
    ]
    for field_name in float_fields:
        value = getattr(train_args, field_name, None)
        if value is not None and isinstance(value, str):
            try:
                setattr(train_args, field_name, float(value))
                logger.warning(f"✓ Converted {field_name}='{value}' → float")
            except ValueError:
                logger.error(f"✗ Failed to convert {field_name}='{value}' to float")

    int_fields = [
        "num_train_epochs",
        "max_steps",
        "warmup_steps",
        "logging_steps",
        "eval_steps",
        "save_steps",
        "per_device_train_batch_size",
        "gradient_accumulation_steps",
        "dataloader_num_workers",
        "max_length",
    ]
    for field_name in int_fields:
        value = getattr(train_args, field_name, None)
        if value is not None and isinstance(value, str):
            try:
                setattr(train_args, field_name, int(float(value)))
                logger.warning(f"✓ Converted {field_name}='{value}' → int")
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
        f"%(asctime)s [Rank {rank}|Local {local_rank}] %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
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
    mapping = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    if dtype_str not in mapping:
        raise ValueError(f"Unsupported dtype: {dtype_str}")
    return mapping[dtype_str]


def parse_target_modules(spec: str) -> Union[List[str], str]:
    return (
        spec
        if spec == "all-linear"
        else [m.strip() for m in spec.split(",") if m.strip()]
    )


@dataclass
class TokenIDCollator:
    """Collator for pre-tokenized input_ids - creates labels and attention_mask"""

    pad_token_id: int

    def __call__(self, features: list[dict]) -> dict[str, torch.Tensor]:
        input_ids = torch.tensor([f["input_ids"] for f in features], dtype=torch.long)
        return {
            "input_ids": input_ids,
            "attention_mask": torch.ones_like(input_ids),
            "labels": input_ids.clone().masked_fill(
                input_ids == self.pad_token_id, -100
            ),
        }


# =============================================================================
# Main Pipeline
# =============================================================================


def load_model(
    model_args: ModelArguments, lora_args: LoRAArguments, logger: logging.Logger
):
    logger.info(f"Loading model: {model_args.model_path}")

    # 🔧 For custom quasar model, ensure we use the registered classes
    # AutoModelForCausalLM will now use QuasarForCausalLM when config.model_type == "quasar"
    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_path,
        torch_dtype=parse_torch_dtype(model_args.torch_dtype),
        attn_implementation=model_args.attn_implementation,
        trust_remote_code=model_args.trust_remote_code,  # 🔧 Required for custom model code
    )

    # Ensure config is properly set for quasar model
    if hasattr(model.config, "model_type") and model.config.model_type == "quasar":
        logger.info("✅ Loaded Quasar model with custom configuration")
        # Verify key quasar config params are present
        if hasattr(model.config, "quasar_layers"):
            logger.info(
                f"   Quasar config: quasar_layers={model.config.quasar_layers}, "
                f"gated_layers={model.config.gated_layers}, "
                f"moe_type={getattr(model.config, 'moe_type', 'N/A')}"
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
            use_rslora=True,
        )
        model = get_peft_model(model, lora_config)
        logger.info("✅ LoRA applied successfully")

    return model


def load_dataset_and_tokenizer(
    data_args: DataArguments,
    model_args: ModelArguments,
    train_args: TrainingArgumentsCustom,
    logger: logging.Logger,
):
    logger.info(f"Loading tokenizer from: {model_args.model_path}")

    # 🔧 For custom models, tokenizer loading may need trust_remote_code too
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_path, trust_remote_code=model_args.trust_remote_code
    )

    # Ensure pad_token is set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        logger.info("ℹ️ Set pad_token to eos_token")

    logger.info(f"Loading training dataset: {data_args.data_file}")
    dataset = load_dataset("json", data_files=data_args.data_file)["train"]
    dataset = dataset.shuffle(seed=train_args.seed)
    logger.info(f"Training dataset size: {len(dataset)}")
    return tokenizer, dataset


def resolve_resume_path(args: TrainingArgumentsCustom) -> Optional[str]:
    if args.resume_from_checkpoint and args.resume_from_checkpoint != "auto":
        return args.resume_from_checkpoint
    if args.auto_resume:
        ckpt = get_latest_checkpoint(args.output_dir)
        if ckpt:
            logging.getLogger().info(f"✓ Auto-resume: {ckpt}")
        return ckpt
    return None


def train(
    model_args: ModelArguments,
    data_args: DataArguments,
    lora_args: LoRAArguments,
    train_args: TrainingArgumentsCustom,
):
    train_args = sanitize_training_args(train_args)

    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    logger = setup_logger(train_args.log_dir, rank, local_rank)

    validate_config(data_args, model_args, train_args, logger)

    model = load_model(model_args, lora_args, logger)
    tokenizer, train_dataset = load_dataset_and_tokenizer(
        data_args, model_args, train_args, logger
    )

    # 🆕 Prepare evaluation dataset: shards (preferred) or split from training (fallback)
    eval_dataset = None
    if data_args.eval_shard_dir and os.path.isdir(data_args.eval_shard_dir):
        # Use shard-based evaluation for fair, independent eval set
        eval_seq_len = data_args.eval_shard_seq_len or train_args.max_length
        eval_dataset = load_eval_dataset_from_shards(
            shard_dir=data_args.eval_shard_dir,
            seq_len=eval_seq_len,
            max_per_shard=data_args.eval_shard_max_samples,
            total_max_samples=data_args.eval_shard_total_samples,
            seed=data_args.eval_shard_seed,
            logger=logger,
        )
    elif data_args.eval_split_ratio > 0:
        # Fallback: split from training data (original behavior)
        eval_dataset = prepare_eval_dataset(
            train_dataset, data_args.eval_split_ratio, seed=train_args.seed
        )
    else:
        logger.info(
            "⚠️ No evaluation dataset configured (eval_split_ratio=0 and no eval_shard_dir)"
        )

    collator = TokenIDCollator(pad_token_id=tokenizer.pad_token_id)
    train_args.dataset_kwargs = {"skip_prepare_dataset": data_args.skip_prepare_dataset}

    logger.info("🚀 Initializing trainer...")
    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
        args=train_args,
    )

    resume_path = resolve_resume_path(train_args)
    logger.info("🎯 Starting training loop...")

    # Debug logging
    logger.info(f"🔍 learning_rate = {train_args.learning_rate!r}")
    logger.info(f"🔍 batch_size = {train_args.per_device_train_batch_size!r}")
    logger.info(f"🔍 max_length = {train_args.max_length!r}")

    trainer.train(resume_from_checkpoint=resume_path)

    if rank == 0:
        logger.info("💾 Saving final model...")
        trainer.save_model()
        tokenizer.save_pretrained(train_args.output_dir)
    logger.info("✅ Training completed!")


def main():
    parser = HfArgumentParser(
        (ModelArguments, DataArguments, LoRAArguments, TrainingArgumentsCustom)
    )

    if len(sys.argv) == 2 and sys.argv[1].endswith((".yaml", ".yml")):
        model_args, data_args, lora_args, train_args = parser.parse_yaml_file(
            yaml_file=sys.argv[1]
        )
    else:
        model_args, data_args, lora_args, train_args = (
            parser.parse_args_into_dataclasses()
        )

    train(model_args, data_args, lora_args, train_args)


if __name__ == "__main__":
    main()
