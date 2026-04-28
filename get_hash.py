import hashlib
from pathlib import Path

def sha256_dir(path):
    h = hashlib.sha256()
    for p in sorted(Path(path).glob("*.safetensors")):
        with open(p, "rb") as f:
            while chunk := f.read(1 << 20):
                h.update(chunk)
    return h.hexdigest()

print(sha256_dir("./merged/Teutonic-I-v1/"))