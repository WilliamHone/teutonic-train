#!/usr/bin/env python3
"""
Modular LLM Fine-tuning Script for Teutonic-III
Fixed: Removed argument conflicts with SFTConfig
Added: mu_hat/LCB evaluation with automatic "king-beating" checkpoint detection
"""

import os
import re
import sys
import logging
import ast
import struct
import hashlib
import json
from datetime import datetime, timezone
import numpy as np
import torch
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from pathlib import Path
import io

from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    TrainerCallback,
    TrainerState,
    TrainerControl,
)
from trl import SFTTrainer, SFTConfig
from peft import LoraConfig, get_peft_model, PeftModel

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
    eval_split_ratio: float = field(
        default=0.01,
        metadata={"help": "Fraction of data to use for evaluation (0 = no eval)"}
    )
    skip_prepare_dataset: bool = field(
        default=True,
        metadata={"help": "Skip SFTTrainer's internal dataset preparation"}
    )
    
    # ✅ mu_hat/LCB evaluation arguments
    eval_metrics_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Directory containing .npy shard files for mu_hat/LCB evaluation"}
    )
    eval_king_model: Optional[str] = field(
        default=None,
        metadata={"help": "King model path/ID for mu_hat/LCB comparison (defaults to base model)"}
    )
    eval_n_sequences: int = field(
        default=320,
        metadata={"help": "Number of sequences to sample for mu_hat/LCB evaluation"}
    )
    eval_seq_len: int = field(
        default=2048,
        metadata={"help": "Tokens per sequence for mu_hat/LCB evaluation"}
    )
    eval_batch_size: int = field(
        default=16,
        metadata={"help": "Batch size for mu_hat/LCB loss computation"}
    )
    eval_bootstrap_alpha: float = field(
        default=0.001,
        metadata={"help": "Confidence level for LCB (one-sided), e.g., 0.001 = 99.9% LCB"}
    )
    eval_n_bootstrap: int = field(
        default=10000,
        metadata={"help": "Number of bootstrap replicates for LCB calculation"}
    )
    
    # ✅ NEW: King-beating threshold & behavior
    eval_delta: float = field(
        default=0.011,
        metadata={"help": "LCB threshold to declare 'king-beating' (default: 0.011 nats/token)"}
    )
    save_on_king_beaten: bool = field(
        default=True,
        metadata={"help": "Save special checkpoint when LCB > delta"}
    )
    stop_on_king_beaten: bool = field(
        default=False,
        metadata={"help": "Stop training early when LCB > delta (use with caution)"}
    )
    king_beaten_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Directory to save 'king-beaten' checkpoints (defaults to output_dir/king-beaten)"}
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
    
    # Enable mu_hat/LCB evaluation
    enable_eval_metrics: bool = field(default=True)
    
    # ❌ DO NOT add fields that overlap with SFTConfig!


# =============================================================================
# Helper Functions for mu_hat/LCB Evaluation
# =============================================================================

def validate_config(data_args, model_args, train_args, logger):
    """Validate critical paths and settings before training."""
    errors = []
    
    if data_args.eval_metrics_dir and not os.path.isdir(data_args.eval_metrics_dir):
        errors.append(f"eval_metrics_dir not found: {data_args.eval_metrics_dir}")
    
    if not os.path.isdir(os.path.dirname(train_args.output_dir)):
        errors.append(f"output_dir parent not found: {os.path.dirname(train_args.output_dir)}")
    
    if train_args.report_to == "wandb":
        try:
            import wandb
            if not wandb.login(relogin=False, anonymous="allow"):
                logger.warning("⚠️ WandB login failed, metrics may not sync")
        except ImportError:
            errors.append("report_to='wandb' but wandb not installed")
    
    if errors:
        logger.error("❌ Configuration validation failed:")
        for e in errors:
            logger.error(f"  - {e}")
        sys.exit(1)
    
    logger.info("✅ Configuration validated")

def _read_npy_header(raw: bytes):
    """Return (data_offset, header_dict) for a .npy file header buffer."""
    buf = io.BytesIO(raw)
    buf.read(6)  # magic
    ver = struct.unpack("BB", buf.read(2))
    hl = struct.unpack("<H" if ver[0] == 1 else "<I", buf.read(2 if ver[0] == 1 else 4))[0]
    header = ast.literal_eval(buf.read(hl).decode("latin1").strip())
    return buf.tell(), header


def _load_npy_metadata(shard_path):
    """Load .npy metadata without loading full array."""
    with open(shard_path, "rb") as f:
        prefix = f.read(10)
        if len(prefix) < 10:
            raise ValueError(f"incomplete npy header in {shard_path}")
        major, _minor = struct.unpack("BB", prefix[6:8])
        header_len_size = 2 if major == 1 else 4
        extra = f.read(header_len_size)
        if len(extra) != header_len_size:
            raise ValueError(f"incomplete npy header length in {shard_path}")
        header_len = struct.unpack("<H" if major == 1 else "<I", extra)[0]
        header_buf = prefix + extra + f.read(header_len)
    data_offset, header = _read_npy_header(header_buf)
    return data_offset, header


def _get_shard_capacity(shard_path: str, seq_len: int) -> int:
    """Get number of complete sequences in a shard."""
    _, header = _load_npy_metadata(shard_path)
    n_tokens = 1
    for dim in header["shape"]:
        n_tokens *= dim
    return int(n_tokens) // seq_len


def _sample_sequences_from_shards(
    dataset_dir: str,
    n_sequences: int,
    seq_len: int,
    seed: int,
    max_shards: Optional[int] = None
) -> List[List[int]]:
    """Sample sequences from npy shards for evaluation (FIXED: uniform sampling)."""
    import io
    
    seed_str = f"eval:{seed}"
    seed_material = seed_str.encode()
    seed_int = int.from_bytes(hashlib.blake2b(seed_material, digest_size=8).digest(), "little")
    rng = np.random.Generator(np.random.PCG64(seed_int))
    
    dataset_path = Path(dataset_dir).expanduser()
    shards = sorted([f for f in os.listdir(dataset_path) if f.endswith(".npy")])
    if max_shards:
        shards = shards[:max_shards]
    shard_paths = [str(dataset_path / s) for s in shards]
    
    if not shard_paths:
        raise ValueError(f"No .npy shards found in {dataset_dir}")
    
    # 🔍 Build list of (shard_path, start_idx, end_idx) for all valid sequences
    all_sequence_locations = []  # List of (shard_path, seq_start_token_idx)
    
    for shard_path in shard_paths:
        data_offset, header = _load_npy_metadata(shard_path)
        n_tokens = 1
        for dim in header["shape"]:
            n_tokens *= dim
        
        n_sequences_in_shard = int(n_tokens) // seq_len
        if n_sequences_in_shard <= 0:
            continue
        
        # Add each sequence's starting token index
        for seq_idx in range(n_sequences_in_shard):
            all_sequence_locations.append((shard_path, seq_idx * seq_len))
    
    if not all_sequence_locations:
        raise ValueError(f"No valid sequences found for seq_len={seq_len} in {dataset_dir}")
    
    # 🔍 Sample uniformly from all available sequences
    n_available = len(all_sequence_locations)
    n_to_sample = min(n_sequences, n_available)
    
    if n_to_sample < n_sequences:
        logging.getLogger().warning(
            f"⚠️ Only {n_available} sequences available, sampling {n_to_sample}/{n_sequences}"
        )
    
    # Randomly select which sequences to load
    selected_indices = rng.choice(n_available, size=n_to_sample, replace=False)
    
    sequences = []
    for idx in selected_indices:
        shard_path, start_token = all_sequence_locations[idx]
        
        # Load shard metadata again (could cache, but simplicity first)
        data_offset, header = _load_npy_metadata(shard_path)
        n_tokens = 1
        for dim in header["shape"]:
            n_tokens *= dim
        
        shard_tokens = np.memmap(
            shard_path, dtype="<u4", mode="r",
            offset=data_offset, shape=(int(n_tokens),)
        )
        
        # Extract the sequence
        seq_tokens = shard_tokens[start_token : start_token + seq_len].tolist()
        sequences.append(seq_tokens)
    
    return sequences


@torch.no_grad()
def _compute_chunked_loss(
    model,
    token_batches: List[List[int]],
    device: torch.device,
    chunk_size: int = 128  # Keep small for memory efficiency
) -> List[float]:
    """Compute per-sequence mean cross-entropy loss with chunked lm_head (bfloat16-safe)."""
    if not token_batches:
        return []
    
    input_ids = torch.tensor(token_batches, dtype=torch.long, device=device)
    batch_size, seq_len = input_ids.shape
    
    # ✅ Get hidden states with output_hidden_states=True
    outputs = model.model(
        input_ids=input_ids,
        output_hidden_states=True,
        return_dict=True,
    )
    hidden = outputs.hidden_states[-1]  # Last transformer layer (bfloat16)
    
    # ✅ Get lm_head
    if hasattr(model, 'lm_head'):
        lm_head = model.lm_head
    elif hasattr(model, 'get_output_embeddings'):
        lm_head = model.get_output_embeddings()
    else:
        raise AttributeError("Could not find lm_head or output embeddings")
    
    # ✅ Compute loss with chunked lm_head (memory efficient + dtype-safe)
    n_positions = seq_len - 1
    total_loss = torch.zeros(batch_size, device=device, dtype=torch.float32)  # ← Force float32 accumulator
    
    for i in range(0, n_positions, chunk_size):
        end_pos = min(i + chunk_size, n_positions)
        
        # Get chunk of hidden states (bfloat16)
        chunk_hidden = hidden[:, i:end_pos, :]
        
        # Project to logits via lm_head (still bfloat16)
        chunk_logits = lm_head(chunk_hidden)
        
        # Labels are next tokens (long)
        chunk_labels = input_ids[:, i + 1 : end_pos + 1]
        
        # ✅ CRITICAL FIX: Cast logits to float32 for cross_entropy
        chunk_logits_fp32 = chunk_logits.float()  # ← bfloat16 → float32
        
        # Compute cross-entropy (requires float32)
        loss = F.cross_entropy(
            chunk_logits_fp32.reshape(-1, chunk_logits_fp32.size(-1)),
            chunk_labels.reshape(-1),
            reduction="none",
        )
        
        # Accumulate in float32, then sum
        total_loss += loss.reshape(batch_size, -1).sum(dim=1)
        
        # ✅ Aggressive cleanup to free VRAM
        del chunk_logits, chunk_logits_fp32, loss, chunk_hidden
    
    # ✅ Return mean loss per sequence (as Python floats)
    return (total_loss / n_positions).cpu().tolist()


def _compute_lcb(differences: np.ndarray, alpha: float, n_bootstrap: int, seed: int) -> float:
    """Compute lower confidence bound via non-parametric bootstrap."""
    if len(differences) == 0:
        return 0.0
    
    rng = np.random.default_rng(seed)
    boot_means = np.empty(n_bootstrap)
    for b in range(n_bootstrap):
        idx = rng.integers(0, len(differences), size=len(differences))
        boot_means[b] = differences[idx].mean()
    
    return float(np.quantile(boot_means, alpha))


# =============================================================================
# Custom Callback for mu_hat/LCB + King-Beating Detection
# =============================================================================

class MuHatLCBCallback(TrainerCallback):
    """Callback to compute mu_hat/LCB and detect when model beats the king."""
    
    def __init__(self, data_args, model_args, train_args, logger, tokenizer, trainer):
        self.data_args = data_args
        self.model_args = model_args
        self.train_args = train_args
        self.logger = logger
        self.tokenizer = tokenizer
        self.trainer = trainer  # Reference to trainer for saving
        
        self.king_model = None
        self.king_device = None
        self._eval_step_counter = 0
        self._king_beaten = False
        self._best_lcb = -float('inf')
        self._best_lcb_step = None
        
    def _get_king_model(self):
        """Load king model for comparison (cached, eval mode)."""
        if self.king_model is not None:
            return self.king_model, self.king_device
        
        king_path = self.data_args.eval_king_model or self.model_args.model_path
        self.logger.info(f"🔍 Loading king model for eval: {king_path}")
        
        king_model = AutoModelForCausalLM.from_pretrained(
            king_path,
            torch_dtype=parse_torch_dtype(self.model_args.torch_dtype),
            attn_implementation=self.model_args.attn_implementation,
            trust_remote_code=self.model_args.trust_remote_code,
            # device_map="auto",
            device_map="cpu",  # ← KEY: Keep king model on CPU
            low_cpu_mem_usage=True,
        )
        king_model.eval()
        king_device = torch.device("cpu")
        # king_device = next(king_model.parameters()).device
        self.logger.info(f"✓ King model loaded on {king_device}")
        
        self.king_model = king_model
        self.king_device = king_device
        return king_model, king_device
    
    def _run_eval_metrics(self, challenger_model, global_step: int, seed: int) -> Optional[Dict[str, Any]]:
        """Run mu_hat/LCB evaluation and return metrics dict."""
        if not self.data_args.eval_metrics_dir:
            return None
        
        try:
            sequences = _sample_sequences_from_shards(
                dataset_dir=self.data_args.eval_metrics_dir,
                n_sequences=self.data_args.eval_n_sequences,
                seq_len=self.data_args.eval_seq_len,
                seed=seed,
                max_shards=None
            )
            
            if len(sequences) < self.data_args.eval_n_sequences:
                self.logger.warning(f"⚠️ Only got {len(sequences)}/{self.data_args.eval_n_sequences} sequences")
            
            if not sequences:
                return None
            
            batches = [
                sequences[i:i + self.data_args.eval_batch_size]
                for i in range(0, len(sequences), self.data_args.eval_batch_size)
            ]
            
            king_model, king_device = self._get_king_model()
            challenger_device = next(challenger_model.parameters()).device
            
            all_diffs = []
            king_sum, chall_sum = 0.0, 0.0
            
            for token_batches in batches:
                king_losses = _compute_chunked_loss(king_model, token_batches, king_device)
                chall_losses = _compute_chunked_loss(challenger_model, token_batches, challenger_device)
                
                for k_loss, c_loss in zip(king_losses, chall_losses):
                    if k_loss is not None and c_loss is not None:
                        king_sum += k_loss
                        chall_sum += c_loss
                        all_diffs.append(k_loss - c_loss)
            
            if len(all_diffs) < 2:
                self.logger.warning("⚠️ Too few valid losses for mu_hat/LCB calculation")
                return None
            
            d = np.array(all_diffs)
            mu_hat = float(d.mean())
            lcb = _compute_lcb(
                d,
                alpha=self.data_args.eval_bootstrap_alpha,
                n_bootstrap=self.data_args.eval_n_bootstrap,
                seed=seed
            )
            
            return {
                "eval/mu_hat": mu_hat,
                "eval/lcb": lcb,
                "eval/avg_king_loss": king_sum / len(all_diffs),
                "eval/avg_challenger_loss": chall_sum / len(all_diffs),
                "eval/n_sequences": len(all_diffs),
                "eval/diff_std": float(d.std()),
            }
            
        except Exception as e:
            self.logger.error(f"❌ Error in mu_hat/LCB evaluation: {e}", exc_info=True)
            return None
    
    def _save_king_beaten_checkpoint(self, global_step: int, metrics: Dict[str, float]):
        """Save special checkpoint when LCB > delta."""
        save_dir = self.data_args.king_beaten_dir or os.path.join(self.train_args.output_dir, "king-beaten")
        checkpoint_name = f"king-beaten-step-{global_step}"
        full_path = os.path.join(save_dir, checkpoint_name)
        
        os.makedirs(save_dir, exist_ok=True)
        
        # Save model (handle LoRA vs full model)
        if isinstance(self.trainer.model, PeftModel):
            self.trainer.model.save_pretrained(full_path)
        else:
            self.trainer.model.save_pretrained(full_path)
        
        # Save tokenizer
        self.tokenizer.save_pretrained(full_path)
        
        # Save metadata
        metadata = {
            "global_step": global_step,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "metrics": metrics,
            "delta_threshold": self.data_args.eval_delta,
            "king_model": self.data_args.eval_king_model or self.model_args.model_path,
            "config": {
                "eval_n_sequences": self.data_args.eval_n_sequences,
                "eval_seq_len": self.data_args.eval_seq_len,
                "eval_bootstrap_alpha": self.data_args.eval_bootstrap_alpha,
            }
        }
        with open(os.path.join(full_path, "king_beaten_info.json"), "w") as f:
            json.dump(metadata, f, indent=2)
        
        self.logger.info(f"🏆 Saved king-beaten checkpoint: {full_path}")
        return full_path
    
    def _log_king_beaten_achievement(self, global_step: int, metrics: Dict[str, float], checkpoint_path: str):
        """Log achievement to console and wandb."""
        msg = (
            f"\n{'='*70}\n"
            f"🎉 KING BEATEN! 🎉\n"
            f"{'='*70}\n"
            f"Step: {global_step}\n"
            f"LCB: {metrics['eval/lcb']:.6f} > delta: {self.data_args.eval_delta:.6f} ✓\n"
            f"mu_hat: {metrics['eval/mu_hat']:.6f}\n"
            f"Challenger loss: {metrics['eval/avg_challenger_loss']:.6f}\n"
            f"King loss: {metrics['eval/avg_king_loss']:.6f}\n"
            f"Checkpoint: {checkpoint_path}\n"
            f"{'='*70}\n"
        )
        self.logger.info(msg)
        
        if wandb and wandb.run:
            wandb.log({
                "training/king_beaten": 1,
                "training/king_beaten_step": global_step,
                "training/king_beaten_lcb": metrics["eval/lcb"],
                "training/king_beaten_mu_hat": metrics["eval/mu_hat"],
                **{f"king_beaten/{k}": v for k, v in metrics.items()}
            }, step=global_step)
    
    def on_evaluate(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        """Hook into evaluation step."""
        if not self.train_args.enable_eval_metrics:
            return control
        
        if not state.is_world_process_zero:
            return control
        
        if wandb is None or wandb.run is None:
            self.logger.warning("⚠️ wandb not available, skipping mu_hat/LCB logging")
            return control
        
        challenger_model = kwargs.get('model')
        if challenger_model is None:
            return control
        
        seed = self.train_args.seed + self._eval_step_counter
        self._eval_step_counter += 1
        
        self.logger.info(f"📊 Computing mu_hat/LCB at step {state.global_step}")
        metrics = self._run_eval_metrics(challenger_model, state.global_step, seed)
        
        if metrics:
            # Log standard metrics
            wandb.log(metrics, step=state.global_step)
            self.logger.info(
                f"📈 mu_hat={metrics['eval/mu_hat']:.6f}, "
                f"LCB={metrics['eval/lcb']:.6f} @ step {state.global_step}"
            )
            
            # Track best LCB
            if metrics["eval/lcb"] > self._best_lcb:
                self._best_lcb = metrics["eval/lcb"]
                self._best_lcb_step = state.global_step
                self.logger.info(f"✨ New best LCB: {self._best_lcb:.6f} @ step {self._best_lcb_step}")
                wandb.log({"eval/best_lcb": self._best_lcb, "eval/best_lcb_step": self._best_lcb_step}, step=state.global_step)
            
            # 🏆 Check if king is beaten
            if not self._king_beaten and metrics["eval/lcb"] > self.data_args.eval_delta:
                self._king_beaten = True
                
                # Save checkpoint if enabled
                checkpoint_path = None
                if self.data_args.save_on_king_beaten:
                    checkpoint_path = self._save_king_beaten_checkpoint(state.global_step, metrics)
                
                # Log achievement
                self._log_king_beaten_achievement(state.global_step, metrics, checkpoint_path or "N/A")
                
                # Stop training if enabled
                if self.data_args.stop_on_king_beaten:
                    self.logger.info("🛑 stop_on_king_beaten=True, stopping training...")
                    control.should_training_stop = True
        
        return control
    
    def on_train_end(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        """Log summary at end of training."""
        if not state.is_world_process_zero:
            return
        
        summary = {
            "training/final_lcb": self._best_lcb if self._best_lcb > -float('inf') else None,
            "training/best_lcb_step": self._best_lcb_step,
            "training/king_beaten": self._king_beaten,
            "training/eval_delta": self.data_args.eval_delta,
        }
        
        if wandb and wandb.run:
            wandb.log(summary)
        
        if self._king_beaten:
            self.logger.info("✅ Training completed: KING BEATEN! 🏆")
        else:
            self.logger.info(
                f"⚠️ Training completed: LCB={self._best_lcb:.6f} < delta={self.data_args.eval_delta:.6f}, "
                f"king not yet beaten"
            )


# =============================================================================
# Existing Helper Functions
# =============================================================================

def sanitize_training_args(train_args: TrainingArgumentsCustom) -> TrainingArgumentsCustom:
    """Ensure numeric fields are proper types."""
    logger = logging.getLogger()
    
    float_fields = ["learning_rate", "weight_decay", "adam_beta1", "adam_beta2", "adam_epsilon"]
    for field_name in float_fields:
        value = getattr(train_args, field_name, None)
        if value is not None and isinstance(value, str):
            try:
                setattr(train_args, field_name, float(value))
                logger.warning(f"✓ Converted {field_name}='{value}' → float")
            except ValueError:
                logger.error(f"✗ Failed to convert {field_name}='{value}' to float")
    
    int_fields = [
        "num_train_epochs", "max_steps", "warmup_steps", "logging_steps",
        "eval_steps", "save_steps", "per_device_train_batch_size",
        "gradient_accumulation_steps", "dataloader_num_workers", "max_length"
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
    train_args = sanitize_training_args(train_args)
    
    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    logger = setup_logger(train_args.log_dir, rank, local_rank)
    
    validate_config(data_args, model_args, train_args, logger)  # ← Add this
    
    model = load_model(model_args, lora_args, logger)
    tokenizer, dataset = load_dataset_and_tokenizer(data_args, model_args, train_args, logger)
    eval_dataset = prepare_eval_dataset(dataset, data_args.eval_split_ratio, seed=train_args.seed)
    
    collator = TokenIDCollator(pad_token_id=tokenizer.pad_token_id)
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
    
    # Add mu_hat/LCB callback if enabled
    if train_args.enable_eval_metrics and data_args.eval_metrics_dir:
        logger.info(f"📊 Enabled mu_hat/LCB evaluation (delta={data_args.eval_delta})")
        eval_callback = MuHatLCBCallback(
            data_args=data_args,
            model_args=model_args,
            train_args=train_args,
            logger=logger,
            tokenizer=tokenizer,
            trainer=trainer  # Pass trainer reference for saving
        )
        trainer.add_callback(eval_callback)
    
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
    parser = HfArgumentParser((ModelArguments, DataArguments, LoRAArguments, TrainingArgumentsCustom))
    
    if len(sys.argv) == 2 and sys.argv[1].endswith((".yaml", ".yml")):
        model_args, data_args, lora_args, train_args = parser.parse_yaml_file(yaml_file=sys.argv[1])
    else:
        model_args, data_args, lora_args, train_args = parser.parse_args_into_dataclasses()
        
    train(model_args, data_args, lora_args, train_args)


if __name__ == "__main__":
    main()