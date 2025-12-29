import random
import yaml
import numpy as np
import torch
import os

def ensure_artifact_dirs(cfg):
    # Creates dirs if missing; works with real dirs or symlinks (if present).
    os.makedirs(cfg["paths"]["checkpoint_dir"], exist_ok=True)
    os.makedirs(cfg["paths"]["results_dir"], exist_ok=True)
    os.makedirs(cfg["paths"]["logs_dir"], exist_ok=True)
    os.makedirs("data", exist_ok=True)

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

def get_system_stats():
    """
    Returns a dict with CPU/RAM/GPU utilization.
    Safe: returns partial stats if some libs aren't installed.
    """
    stats = {}

    # CPU/RAM
    try:
        import psutil
        stats["cpu_pct"] = psutil.cpu_percent(interval=None)
        vm = psutil.virtual_memory()
        stats["ram_pct"] = vm.percent
        stats["ram_gb_used"] = vm.used / (1024**3)
        stats["ram_gb_total"] = vm.total / (1024**3)
    except Exception:
        pass

    # GPU (NVML)
    try:
        import pynvml
        pynvml.nvmlInit()
        h = pynvml.nvmlDeviceGetHandleByIndex(0)
        util = pynvml.nvmlDeviceGetUtilizationRates(h)
        mem = pynvml.nvmlDeviceGetMemoryInfo(h)
        stats["gpu_util_pct"] = util.gpu
        stats["gpu_mem_gb_used"] = mem.used / (1024**3)
        stats["gpu_mem_gb_total"] = mem.total / (1024**3)
    except Exception:
        pass

    # Torch memory (useful even if NVML fails)
    try:
        import torch
        if torch.cuda.is_available():
            stats["torch_mem_gb_alloc"] = torch.cuda.memory_allocated() / (1024**3)
            stats["torch_mem_gb_reserved"] = torch.cuda.memory_reserved() / (1024**3)
    except Exception:
        pass

    return stats
