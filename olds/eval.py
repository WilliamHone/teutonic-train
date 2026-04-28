import io
import struct
import hashlib

import torch
import numpy as np
from gpu_evaluator import MultiGPUEvaluator

SHARD_NUMBER = 1
DATASET_PATH = f"/dev/shm/teutonic_dataset/shard_{SHARD_NUMBER:06d}.npy"
# MODEL_PATH = "./merged/Teutonic-I-v1/"
MODEL_PATH = "./models/teutonic/"

seed_str = "test:eval"
seq_len = 2048
eval_n = 2000
batch_size = 32

def _parse_npy_header(raw: bytes) -> int:
    """Return the byte offset where data begins in a .npy file."""
    buf = io.BytesIO(raw)
    buf.read(6)  # magic
    ver = struct.unpack("BB", buf.read(2))
    hl = struct.unpack("<H" if ver[0] == 1 else "<I", buf.read(2 if ver[0] == 1 else 4))[0]
    buf.read(hl)
    return buf.tell()

def get_shard(path):
    import pathlib
    shard_path = pathlib.Path(path)
    raw = shard_path.read_bytes()
    data_offset = _parse_npy_header(raw)
    return data_offset, raw

def extract_sequences(shard_data, data_offset, indices, seq_len):
    """Extract sequences from a locally-cached shard."""
    bps = seq_len * 4
    result = {}
    for idx in indices:
        off = data_offset + idx * bps
        result[idx] = np.frombuffer(shard_data[off : off + bps], dtype="<u4").tolist()
    return result


data = np.load(DATASET_PATH)
n_tokens = data.shape[0]
n_sequences = n_tokens // seq_len
actual_N = min(eval_n, n_sequences)

seed_material = seed_str.encode()
seed = int.from_bytes(hashlib.blake2b(seed_material, digest_size=8).digest(), "little")
rng = np.random.Generator(np.random.PCG64(seed))
eval_indices = rng.choice(n_sequences, size=actual_N, replace=False).tolist()

print(f"Evaluating on {actual_N} sequences. \n (seed: {seed_str}, indices: {eval_indices})")

data_offset, shard_data = get_shard(DATASET_PATH)
seq_cache = extract_sequences(shard_data, data_offset, eval_indices, seq_len)

print(f"extracted {len(seq_cache)} sequences")

batches = [
    eval_indices[i : i + batch_size]
    for i in range(0, len(eval_indices), batch_size)
]

gpu_ids = list(range(torch.cuda.device_count()))
king_eval = MultiGPUEvaluator(MODEL_PATH, gpu_ids, label="king")

king_sum, chall_sum = 0.0, 0.0
total_done = 0

for bi, batch_indices in enumerate(batches):
    token_batches = [seq_cache[idx] for idx in batch_indices]
    king_losses = king_eval.compute_losses(token_batches)

    print(f"Batch {bi+1}/{len(batches)}: King avg loss {np.mean(king_losses):.4f}, {len(king_losses)} sequences")
    for k_loss in king_losses:
        total_done += 1
        king_sum += k_loss

print(f"Final King avg loss: {round(king_sum / total_done, 6)} over {total_done} sequences")