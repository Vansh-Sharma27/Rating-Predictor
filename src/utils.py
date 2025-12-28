import random
import yaml
import numpy as np
import torch

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    print(f"Random seed set to {seed}")

def load_config(path: str = "config.yaml"):
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    print(f"Config loaded from {path}")
    return cfg

def print_gpu_info():
    if torch.cuda.is_available():
        print(f"\nGPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")
        allocated = torch.cuda.memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved() / 1e9
        print(f"Memory: {allocated:.2f} GB allocated, {reserved:.2f} GB reserved")
    else:
        print("No GPU available")
