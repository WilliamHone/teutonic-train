#!/usr/bin/env python3
"""
Modular LLM Fine-tuning Script for Teutonic-III
Updated: Support SCALE optimizer + eval data from .npy shard directories
"""

import os
import re
import sys
import logging
import json
from datetime import datetime, timezone
from dataclasses import dataclass, field
from typing import Optional, List, Union, Tuple, Dict, Any
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from datasets import load_dataset, Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    get_scheduler,
)
from trl import SFTTrainer, SFTConfig
from peft import LoraConfig, get_peft_model, PeftModel

# Import SCALE optimizer
try:
    from mem_eff_pt.pt_scale.scale_optimizer import SCALE
    from mem_eff_pt.utils.train_utils import build_optimizer as build_scale_optimizer

    SCALE_AVAILABLE = True
except ImportError:
    SCALE_AVAILABLE = False
    logging.warning(
        "SCALE optimizer not available. Install mem_eff_pt to use --use_scale_optimizer"
    )

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
        default=False, metadata={"help": "Trust remote code when loading model"}
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
        default=20048,
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
    deepspeed: str = field(default="ds_zero3.json")

    # 🆕 SCALE Optimizer Arguments
    use_scale_optimizer: bool = field(
        default=False,
        metadata={"help": "Use SCALE optimizer instead of standard optimizers"},
    )
    scale_momentum: float = field(
        default=0.9, metadata={"help": "Momentum for secondary parameters in SCALE"}
    )
    scale_adam_lr: Optional[float] = field(
        default=None,
        metadata={
            "help": "Learning rate for 1D params in SCALE (defaults to main lr if None)"
        },
    )
    scale_adam_beta1: float = field(
        default=0.9, metadata={"help": "Adam beta1 for 1D params in SCALE"}
    )
    scale_adam_beta2: float = field(
        default=0.999, metadata={"help": "Adam beta2 for 1D params in SCALE"}
    )
    scale_adam_eps: float = field(
        default=1e-8, metadata={"help": "Adam epsilon for 1D params in SCALE"}
    )
    scale_debug: bool = field(
        default=False, metadata={"help": "Enable debug logging in SCALE optimizer"}
    )
    scale_main_modules: str = field(
        default="attn,mlp,attention,embed_tokens",
        metadata={
            "help": "Comma-separated list of module name patterns to treat as 'main' params in SCALE"
        },
    )


# =============================================================================
# SCALE Optimizer Helper Functions
# =============================================================================


def classify_parameters_for_scale(
    model: nn.Module, main_module_patterns: List[str]
) -> Tuple[List[nn.Parameter], List[nn.Parameter], List[nn.Parameter], Dict[int, str]]:
    """
    Classify model parameters into main/secondary/1D categories for SCALE optimizer.
    Returns: (main_params, secondary_params, oned_params, id_to_name_dict)
    """
    main_params = []
    oned_params = []
    secondary_params = []

    id_to_name_main = {}
    id_to_name_oned = {}
    id_to_name_secondary = {}

    # First pass: identify main params from target modules
    for module_name, module in model.named_modules():
        if not (isinstance(module, nn.Linear) or isinstance(module, nn.Embedding)):
            continue
        if not any(
            target_key in module_name.lower() for target_key in main_module_patterns
        ):
            continue

        if hasattr(module, "weight") and module.weight.requires_grad:
            main_params.append(module.weight)
            id_to_name_main[id(module.weight)] = module_name

    # Second pass: classify remaining parameters
    for param_name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        param_id = id(param)

        # Skip if already classified as main
        if param_id in id_to_name_main:
            continue

        # Classify by dimensionality
        if param.ndim == 1:
            oned_params.append(param)
            id_to_name_oned[param_id] = param_name
        else:
            secondary_params.append(param)
            id_to_name_secondary[param_id] = param_name

    # Merge id_to_name mappings
    id_to_name = {**id_to_name_main, **id_to_name_secondary, **id_to_name_oned}

    # Optional debug logging
    logger = logging.getLogger()
    if logger.isEnabledFor(logging.DEBUG):
        for module_name, module in model.named_modules():
            if hasattr(module, "weight"):
                p = module.weight
                pid = id(p)
                if pid in id_to_name_main:
                    logger.debug(f"Main param: {module_name}")
                elif pid in id_to_name_oned:
                    logger.debug(f"1D param: {module_name}")
                elif pid in id_to_name_secondary:
                    logger.debug(f"Secondary param: {module_name}")

    return main_params, secondary_params, oned_params, id_to_name


def build_scale_optimizer_for_model(
    model: nn.Module, args: TrainingArgumentsCustom
) -> torch.optim.Optimizer:
    """Build SCALE optimizer with parameters classified from model."""
    if not SCALE_AVAILABLE:
        raise ImportError(
            "SCALE optimizer not available. Please install mem_eff_pt package."
        )

    # Parse main module patterns
    main_module_patterns = [
        p.strip().lower() for p in args.scale_main_modules.split(",") if p.strip()
    ]

    # Classify parameters
    main_params, secondary_params, oned_params, id_to_name = (
        classify_parameters_for_scale(model, main_module_patterns)
    )

    logger = logging.getLogger()
    logger.info(
        f"SCALE optimizer: {len(main_params)} main params, "
        f"{len(secondary_params)} secondary params, "
        f"{len(oned_params)} 1D params"
    )

    # Determine effective Adam LR for 1D params
    adam_lr = (
        args.scale_adam_lr if args.scale_adam_lr is not None else args.learning_rate
    )

    # Create SCALE optimizer
    optimizer = SCALE(
        lr=args.learning_rate,
        wd=args.weight_decay,
        main_params=main_params,
        secondary_params=secondary_params,
        oned_params=oned_params,
        id_to_name=id_to_name,
        debug=args.scale_debug,
        momentum=args.scale_momentum,
        adam_lr=adam_lr,
        adamw_betas=(args.scale_adam_beta1, args.scale_adam_beta2),
        adamw_eps=args.scale_adam_eps,
    )

    return optimizer


def create_scheduler_for_scale(
    optimizer: torch.optim.Optimizer,
    args: TrainingArgumentsCustom,
    num_training_steps: int,
) -> torch.optim.lr_scheduler.LRScheduler:
    """Create learning rate scheduler compatible with SCALE optimizer."""
    return get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=num_training_steps,
    )


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

    # Validate SCALE optimizer availability if requested
    if train_args.use_scale_optimizer and not SCALE_AVAILABLE:
        errors.append(
            "use_scale_optimizer=True but SCALE optimizer not available (install mem_eff_pt)"
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
        "scale_momentum",
        "scale_adam_lr",
        "scale_adam_beta1",
        "scale_adam_beta2",
        "scale_adam_eps",
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
    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_path,
        torch_dtype=parse_torch_dtype(model_args.torch_dtype),
        attn_implementation=model_args.attn_implementation,
        trust_remote_code=model_args.trust_remote_code,
        use_safetensors=True,
    )
    model.config.use_cache = False
    if lora_args.use_lora:
        logger.info(f"Applying LoRA: r={lora_args.r}, alpha={lora_args.alpha}")
        lora_config = LoraConfig(
            r=lora_args.r,
            lora_alpha=lora_args.alpha,
            target_modules=[
                "q_proj",
                "v_proj",
                # "k_proj",
                # "o_proj",
                # "gate_proj",
                # "up_proj",
                # "down_proj",
            ],
            lora_dropout=lora_args.dropout,
            bias="none",
            task_type="CAUSAL_LM",
            init_lora_weights=lora_args.init_lora_weights,
            use_rslora=True,
        )
        model = get_peft_model(model, lora_config)
    return model


def load_dataset_and_tokenizer(
    data_args: DataArguments,
    model_args: ModelArguments,
    train_args: TrainingArgumentsCustom,
    logger: logging.Logger,
):
    logger.info(f"Loading tokenizer from: {model_args.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_path)
    tokenizer.pad_token = tokenizer.eos_token

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

    # 🆕 Build custom optimizer and scheduler if using SCALE
    optimizer = None
    scheduler = None

    if train_args.use_scale_optimizer:
        logger.info("🔧 Building SCALE optimizer...")
        # Get the underlying model if wrapped by PEFT
        model_for_opt = model
        if isinstance(model, PeftModel):
            model_for_opt = model.base_model

        optimizer = build_scale_optimizer_for_model(model_for_opt, train_args)

        # Calculate total training steps for scheduler
        if train_args.max_steps > 0:
            num_training_steps = train_args.max_steps
        else:
            # Estimate from dataset size
            num_training_steps = (
                len(train_dataset)
                // (
                    train_args.per_device_train_batch_size
                    * max(1, train_args.gradient_accumulation_steps)
                )
                * train_args.num_train_epochs
            )

        scheduler = create_scheduler_for_scale(
            optimizer, train_args, num_training_steps
        )
        logger.info(f"✅ SCALE optimizer ready: {type(optimizer).__name__}")

    logger.info("🚀 Initializing trainer...")
    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
        args=train_args,
        # 🆕 Pass custom optimizer/scheduler if using SCALE
        optimizers=(optimizer, scheduler) if train_args.use_scale_optimizer else None,
    )

    resume_path = resolve_resume_path(train_args)
    logger.info("🎯 Starting training loop...")

    # Debug logging
    logger.info(f"🔍 learning_rate = {train_args.learning_rate!r}")
    logger.info(f"🔍 batch_size = {train_args.per_device_train_batch_size!r}")
    logger.info(f"🔍 max_length = {train_args.max_length!r}")
    if train_args.use_scale_optimizer:
        logger.info(
            f"🔧 Using SCALE optimizer with momentum={train_args.scale_momentum}"
        )

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
