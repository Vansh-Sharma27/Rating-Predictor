import os, json, math
import random
from collections import defaultdict
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from datasets import load_dataset

from src.utils import load_config, set_seed, print_gpu_info
from src.preprocessing import ReviewPreprocessor
from src.model import RatingPredictor
from src.balancing import ClassBalancer, HybridLoss
from src.trainer import Trainer

class ReviewDataset(Dataset):
    def __init__(self, data): self.data = data
    def __len__(self): return len(self.data)
    def __getitem__(self, idx): return self.data[idx]

def extract_category(subset_name: str) -> str:
    return subset_name[11:].replace("_", " ") if subset_name.startswith("raw_review_") else subset_name.replace("_", " ")

def pick_stride(n: int) -> int:
    stride = 104729
    if stride >= n:
        stride = 2 * (n // 3) + 1
    while math.gcd(stride, n) != 1:
        stride += 2
    return stride

def iter_pseudorandom_indices(n: int, seed: int):
    rng = random.Random(seed)
    start = rng.randrange(n)
    stride = pick_stride(n)
    for k in range(n):
        yield (start + k * stride) % n

def have_enough(buckets, target, n_classes=5):
    return all(len(buckets[i]) >= target for i in range(n_classes))

def collect_domain(cfg, subset, per_class_target, seed):
    category = extract_category(subset)
    ds = load_dataset(cfg["data"]["dataset_name"], subset)["full"]
    n = len(ds)

    pre = ReviewPreprocessor()
    buckets = defaultdict(list)

    for idx in iter_pseudorandom_indices(n, seed):
        if have_enough(buckets, per_class_target):
            break
        row = dict(ds[idx])
        row["category"] = category
        ex = pre.process_row(row)
        if ex is None:
            continue
        y = ex["label"]
        if len(buckets[y]) < per_class_target:
            buckets[y].append(ex)

    return category, buckets

def load_training_data(cfg):
    subsets = cfg["data"]["subsets"]
    domains = len(subsets)
    classes = 5
    total = cfg["data"]["train_samples"]

    per_class_per_domain = total // (domains * classes)
    if per_class_per_domain * domains * classes != total:
        raise ValueError("train_samples must be divisible by (domains*classes).")

    print(f"\nLoading {total} training samples")
    print(f"  Per class per domain: {per_class_per_domain}")

    all_samples = []
    all_labels = []

    for di, subset in enumerate(subsets):
        cat, buckets = collect_domain(cfg, subset, per_class_per_domain, cfg["seed"] + 1000 * di)
        counts = [len(buckets[i]) for i in range(5)]
        print(f"  {cat}: {counts} total={sum(counts)}")
        for i in range(5):
            all_samples.extend(buckets[i])
            all_labels.extend([i] * len(buckets[i]))

    combined = list(zip(all_samples, all_labels))
    random.Random(cfg["seed"]).shuffle(combined)
    all_samples, all_labels = zip(*combined)

    print(f"\nTotal training samples: {len(all_samples):,}")
    return list(all_samples), list(all_labels)

def main():
    cfg = load_config("config.yaml")
    set_seed(cfg["seed"])
    print_gpu_info()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nUsing device: {device}")

    if not os.path.exists("data/val.json"):
        print("Validation not found. Run: python scripts/prepare_validation_set.py")
        return

    tokenizer = AutoTokenizer.from_pretrained(cfg["model"]["name"])

    train_data, train_labels = load_training_data(cfg)
    train_ds = ReviewDataset(train_data)

    balancer = ClassBalancer(train_labels, strategy=cfg["class_balancing"]["sampling_strategy"], n_classes=5)
    class_weights = balancer.class_weights.to(device)

    loss_fn = HybridLoss(
        class_weights=class_weights,
        alpha=cfg["class_balancing"]["loss_alpha"],
        beta=cfg["class_balancing"]["loss_beta"],
        gamma=cfg["class_balancing"]["loss_gamma"],
        label_smoothing=cfg["training"]["label_smoothing"],
    )

    with open("data/val.json", "r", encoding="utf-8") as f:
        val_data = json.load(f)
    val_ds = ReviewDataset(val_data)

    model = RatingPredictor(
        model_name=cfg["model"]["name"],
        num_labels=cfg["model"]["num_labels"],
        classifier_dropout=cfg["model"]["classifier_dropout"],
        use_mean_pooling=cfg["model"]["use_mean_pooling"],
        gradient_checkpointing=cfg["training"]["gradient_checkpointing"],
    )

    trainer = Trainer(model=model, tokenizer=tokenizer, loss_fn=loss_fn, config=cfg, device=device)

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["training"]["per_device_train_batch_size"],
        shuffle=True,
        num_workers=cfg["data"]["num_workers"],
        pin_memory=True,
        collate_fn=trainer.collate_fn,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg["training"]["per_device_train_batch_size"] * 2,
        shuffle=False,
        num_workers=cfg["data"]["num_workers"],
        pin_memory=True,
        collate_fn=trainer.collate_fn,
    )

    eff_bs = cfg["training"]["per_device_train_batch_size"] * cfg["training"]["gradient_accumulation_steps"]
    print(f"\nTrain batches: {len(train_loader):,}")
    print(f"Val batches: {len(val_loader):,}")
    print(f"Effective batch size: {eff_bs}")

    print("\n" + "=" * 70)
    print("TRAINING V2 - HYBRID LOSS + MEAN POOLING + DUAL BALANCING")
    print("=" * 70)

    best_f1 = trainer.train(train_loader, val_loader)

    print("\n" + "=" * 70)
    print(f"TRAINING COMPLETE - Best F1 Macro: {best_f1:.4f}")
    print("=" * 70)

if __name__ == "__main__":
    main()
