import os, sys, json, math, argparse
import random
from collections import defaultdict
from datasets import load_dataset

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

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

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--total", type=int, default=200000)
    ap.add_argument("--out", default="data/stacker_train.json")
    ap.add_argument("--seed", type=int, default=44)  # different from val/test seeds
    args = ap.parse_args()

    cfg = load_config("config.yaml")
    set_seed(args.seed)

    subsets = cfg["data"]["subsets"]
    domains = len(subsets)
    classes = 5

    if args.total % (domains * classes) != 0:
        raise ValueError(f"--total must be divisible by domains*classes = {domains*classes}")

    per_class_per_domain = args.total // (domains * classes)
    print(f"Building stacker_train: total={args.total}, per_class_per_domain={per_class_per_domain}")

    all_samples = []
    for di, subset in enumerate(subsets):
        cat, buckets = collect_domain(cfg, subset, per_class_per_domain, args.seed + 1000 * di)
        counts = [len(buckets[i]) for i in range(5)]
        print(f"  {cat}: {counts} total={sum(counts)}")
        for i in range(5):
            all_samples.extend(buckets[i])

    random.Random(args.seed).shuffle(all_samples)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(all_samples, f, ensure_ascii=False)

    print(f"Saved {len(all_samples)} -> {args.out}")

if __name__ == "__main__":
    main()