import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import os, json, argparse
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast
from transformers import AutoTokenizer
from tqdm import tqdm
from src.utils import load_config
from src.model import RatingPredictor
from src.metrics import RatingMetrics


class ReviewDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def collate(batch, tokenizer, max_length):
    texts = [b["text"] for b in batch]
    labels = torch.tensor([b["label"] for b in batch], dtype=torch.long)
    enc = tokenizer(texts, padding="longest", truncation=True, max_length=max_length, return_tensors="pt")
    return {"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"], "labels": labels}


@torch.no_grad()
def run_eval(model, loader, device):
    model.eval()
    y_true, y_pred = [], []
    for batch in tqdm(loader, desc="Evaluating"):
        input_ids = batch["input_ids"].to(device)
        attn = batch["attention_mask"].to(device)
        with autocast("cuda") if device == "cuda" else torch.no_grad():
            out = model(input_ids, attn)
        preds = torch.argmax(out["logits"], dim=-1).cpu().numpy()
        y_pred.extend(list(preds))
        y_true.extend(list(batch["labels"].numpy()))
    return np.array(y_true), np.array(y_pred)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--model", default=None, help="Optional path to a checkpoint (.pt). If omitted, loads <checkpoint_dir>/best_model.pt")
    ap.add_argument("--run-name", default=None, help="If set, appends to checkpoint/results/log dirs (must match train.py --run-name)")
    args = ap.parse_args()

    from src.utils import ensure_artifact_dirs
    cfg = load_config(args.config)

    if args.run_name:
        for k in ("checkpoint_dir", "results_dir", "logs_dir"):
            cfg["paths"][k] = os.path.join(cfg["paths"][k], args.run_name)

    ensure_artifact_dirs(cfg)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if not os.path.exists("data/test.json"):
        print("Test set not found. Run: python scripts/prepare_validation_set.py")
        return

    with open("data/test.json", "r", encoding="utf-8") as f:
        test_data = json.load(f)

    tokenizer = AutoTokenizer.from_pretrained(cfg["model"]["name"])
    model_path = args.model or os.path.join(cfg["paths"]["checkpoint_dir"], "best_model.pt")
    print(f"Loading: {model_path}")

    ckpt = torch.load(model_path, map_location=device)
    model = RatingPredictor(
        model_name=cfg["model"]["name"],
        num_labels=cfg["model"]["num_labels"],
        classifier_dropout=cfg["model"]["classifier_dropout"],
        use_mean_pooling=cfg["model"]["use_mean_pooling"],
        gradient_checkpointing=False,
    )

    try:
        model.load_state_dict(ckpt["model_state_dict"], strict=True)
    except RuntimeError as e:
        print(f"Strict load failed: {e}")
        missing, unexpected = model.load_state_dict(ckpt["model_state_dict"], strict=False)
        print(f"Loaded with strict=False. Missing keys: {len(missing)}, unexpected keys: {len(unexpected)}")

    model.to(device)

    ds = ReviewDataset(test_data)
    loader = DataLoader(
        ds,
        batch_size=32,
        shuffle=False,
        num_workers=4,
        collate_fn=lambda b: collate(b, tokenizer, cfg["tokenizer"]["max_length"]),
    )

    y_true, y_pred = run_eval(model, loader, device)
    m = RatingMetrics.compute(y_true, y_pred)

    print("\n" + "=" * 70)
    print("EVALUATION RESULTS")
    print("=" * 70)
    print(m["report"])
    print("-" * 70)
    print(f"Accuracy            {m['accuracy']:.4f}")
    print(f"F1 Macro            {m['f1_macro']:.4f}")
    print(f"F1 Weighted         {m['f1_weighted']:.4f}")
    print(f"MAE                 {m['mae']:.4f}")
    print(f"Off-by-one Accuracy {m['off_by_one_acc']:.4f}")
    print("-" * 70)

    os.makedirs(cfg["paths"]["results_dir"], exist_ok=True)
    out_path = os.path.join(cfg["paths"]["results_dir"], "test_results.json")
    with open(out_path, "w") as f:
        json.dump(m, f, indent=2)
    print(f"Saved metrics to {out_path}")


if __name__ == "__main__":
    main()
