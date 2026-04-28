import json
import numpy as np

SHARD_DIR = "../teutonic_dataset_eval"
seq_len = 2048
shard_n = 10
sample_n = 5

shard_indices = list(range(1368, 1369))
shard_paths = [f"{SHARD_DIR}/shard_{i:06d}.npy" for i in shard_indices]

dataset = []

for shard_path in shard_paths:
    try:
        data = np.load(shard_path)
    except Exception as e:
        print(f"Failed to load {shard_path}: {e}")
        continue

    n_tokens = data.shape[0]
    n_sequences = n_tokens // seq_len
    actual_N = min(sample_n, n_sequences)

    indices = np.random.choice(n_sequences, size=actual_N, replace=False).tolist()
    sequences = []
    for idx in indices:
        off = idx * seq_len
        token_ids = data[off : off + seq_len].tolist()
        dataset.append(token_ids)

    print(f"Extracted {len(dataset)} data.")

with open(f"./gen_data.jsonl", "w") as f:
    for token_ids in dataset:
        f.write(json.dumps({"input_ids": token_ids}) + "\n")

print(f"Total sequences saved: {len(dataset)}")