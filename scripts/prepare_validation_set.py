import os, json, math
import random
from collections import defaultdict
from datasets import load_dataset

from src.utils import load_config, set_seed
from src.preprocessing import ReviewPreprocessor

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

def build_split(cfg, split_name, total_samples, seed, out_path):
    subsets = cfg["data"]["subsets"]
    domains = len(subsets)
    classes = 5
    per_class_per_domain = total_samples // (domains * classes)
    if per_class_per_domain * domains * classes != total_samples:
        raise ValueError(f"{split_name} samples must be divisible by (domains*classes).")

    print(f"\nPreparing {split_name}: {total_samples}")
    print(f"Domains: {domains}, per-class-per-domain: {per_class_per_domain}")

    all_samples = []
    for di, subset in enumerate(subsets):
        cat, buckets = collect_domain(cfg, subset, per_class_per_domain, seed + 1000 * di)
        counts = [len(buckets[i]) for i in range(5)]
        print(f"  {cat}: {counts} total={sum(counts)}")
        for i in range(5):
            all_samples.extend(buckets[i])

    random.Random(seed).shuffle(all_samples)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_samples, f, ensure_ascii=False)
    print(f"Saved {len(all_samples)} -> {out_path}")

def main():
    cfg = load_config("config.yaml")
    set_seed(cfg["seed"])
    build_split(cfg, "validation", cfg["data"]["eval_samples"], cfg["seed"], "data/val.json")

    set_seed(cfg["seed"] + 1)
    build_split(cfg, "test", cfg["data"]["test_samples"], cfg["seed"] + 1, "data/test.json")

if __name__ == "__main__":
    main()
