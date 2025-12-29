import os, sys, json, argparse
from dataclasses import dataclass

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast
from transformers import AutoTokenizer

from src.utils import load_config
from src.model import RatingPredictor
from src.metrics import RatingMetrics


class JsonDataset(Dataset):
    def __init__(self, path: str):
        with open(path, "r", encoding="utf-8") as f:
            self.data = json.load(f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def clip_round_rating(x_1to5: np.ndarray) -> np.ndarray:
    # returns LABELS 0..4
    r = np.rint(x_1to5).astype(np.int64)
    r = np.clip(r, 1, 5)
    return r - 1


@torch.no_grad()
def compute_outputs(model, tokenizer, cfg, dataset: Dataset, device: str, batch_size: int = 32):
    # Returns y_true (0..4), probs (N,5), expected (N), reg (N)
    def collate(batch):
        texts = [b["text"] for b in batch]
        labels = torch.tensor([b["label"] for b in batch], dtype=torch.long)
        enc = tokenizer(
            texts,
            padding="longest",
            truncation=True,
            max_length=cfg["tokenizer"]["max_length"],
            return_tensors="pt",
        )
        return enc["input_ids"], enc["attention_mask"], labels

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4, collate_fn=collate)

    probs_list = []
    reg_list = []
    y_list = []

    rating_values = np.array([1, 2, 3, 4, 5], dtype=np.float32)

    for input_ids, attn, labels in loader:
        input_ids = input_ids.to(device)
        attn = attn.to(device)

        if device == "cuda":
            with autocast("cuda"):
                out = model(input_ids, attn)
        else:
            out = model(input_ids, attn)

        logits = out["logits"]
        probs = torch.softmax(logits, dim=-1).float().cpu().numpy()
        reg = out["regression"].float().cpu().numpy()  # continuous (ideally 1..5)

        probs_list.append(probs)
        reg_list.append(reg)
        y_list.append(labels.numpy())

    probs = np.concatenate(probs_list, axis=0)
    reg = np.concatenate(reg_list, axis=0)
    y_true = np.concatenate(y_list, axis=0)

    expected = (probs * rating_values[None, :]).sum(axis=1)  # 1..5

    return y_true, probs, expected, reg


def metrics_for_preds(y_true, y_pred):
    m = RatingMetrics.compute(y_true, y_pred)
    return m


def pred_argmax(probs):
    return np.argmax(probs, axis=1).astype(np.int64)


def pred_expected_round(expected):
    return clip_round_rating(expected)


def pred_reg_round(reg):
    return clip_round_rating(reg)


def pred_gated_expected(probs, expected, tau: float):
    # if model is uncertain (max prob below tau), use expected-round, else argmax
    base = pred_argmax(probs)
    alt = pred_expected_round(expected)
    maxp = probs.max(axis=1)
    use_alt = maxp < tau
    out = base.copy()
    out[use_alt] = alt[use_alt]
    changed_pct = 100.0 * use_alt.mean()
    return out, changed_pct


def pred_margin_expected(probs, expected, margin: float):
    # if top1-top2 margin is small, use expected-round, else argmax
    base = pred_argmax(probs)
    alt = pred_expected_round(expected)
    sorted_p = np.sort(probs, axis=1)
    m = sorted_p[:, -1] - sorted_p[:, -2]
    use_alt = m < margin
    out = base.copy()
    out[use_alt] = alt[use_alt]
    changed_pct = 100.0 * use_alt.mean()
    return out, changed_pct


def pick_best(results, key1="f1_macro", key2="accuracy"):
    # results: list of dicts {name, params..., metrics...}
    def score(r):
        return (r[key1], r[key2])
    return max(results, key=score)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="checkpoints/best_model.pt")
    ap.add_argument("--val", default="data/val.json")
    ap.add_argument("--test", default="data/test.json")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--select-metric", choices=["f1_macro", "accuracy"], default="f1_macro")
    args = ap.parse_args()

    cfg = load_config("config.yaml")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if not os.path.exists(args.model):
        raise FileNotFoundError(f"Model not found: {args.model}")
    if not os.path.exists(args.val) or not os.path.exists(args.test):
        raise FileNotFoundError("val/test json not found. Run prepare_validation_set.py first.")

    tokenizer = AutoTokenizer.from_pretrained(cfg["model"]["name"])
    ckpt = torch.load(args.model, map_location=device)

    model = RatingPredictor(
        model_name=cfg["model"]["name"],
        num_labels=cfg["model"]["num_labels"],
        classifier_dropout=cfg["model"]["classifier_dropout"],
        use_mean_pooling=cfg["model"]["use_mean_pooling"],
        gradient_checkpointing=False,
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()

    print("\n=== Computing VAL model outputs ===")
    yv, pv, ev, rv = compute_outputs(model, tokenizer, cfg, JsonDataset(args.val), device, batch_size=args.batch_size)

    print("\n=== Baselines on VAL ===")
    val_baselines = []
    for name, yhat in [
        ("argmax", pred_argmax(pv)),
        ("round_expected", pred_expected_round(ev)),
        ("round_regression", pred_reg_round(rv)),
    ]:
        m = metrics_for_preds(yv, yhat)
        val_baselines.append((name, m))
        print(f"{name:>15} | acc={m['accuracy']:.4f} f1={m['f1_macro']:.4f} off1={m['off_by_one_acc']:.4f} mae={m['mae']:.4f}")

    # Sweep gating thresholds (VAL only)
    print("\n=== Sweep: gated_expected (tune on VAL) ===")
    gated_results = []
    for tau in np.arange(0.30, 0.86, 0.05):
        yhat, changed = pred_gated_expected(pv, ev, float(tau))
        m = metrics_for_preds(yv, yhat)
        gated_results.append({"name": "gated_expected", "tau": float(tau), "changed_pct": changed, **m})
    best_gated = pick_best(gated_results, key1=args.select_metric, key2="accuracy")

    print(f"Best gated (VAL): tau={best_gated['tau']:.2f} changed={best_gated['changed_pct']:.1f}% "
          f"acc={best_gated['accuracy']:.4f} f1={best_gated['f1_macro']:.4f}")

    print("\n=== Sweep: margin_expected (tune on VAL) ===")
    margin_results = []
    for margin in np.arange(0.02, 0.31, 0.02):
        yhat, changed = pred_margin_expected(pv, ev, float(margin))
        m = metrics_for_preds(yv, yhat)
        margin_results.append({"name": "margin_expected", "margin": float(margin), "changed_pct": changed, **m})
    best_margin = pick_best(margin_results, key1=args.select_metric, key2="accuracy")

    print(f"Best margin (VAL): m={best_margin['margin']:.2f} changed={best_margin['changed_pct']:.1f}% "
          f"acc={best_margin['accuracy']:.4f} f1={best_margin['f1_macro']:.4f}")

    # Pick best overall rule based on VAL selection metric
    candidates = []

    # baselines
    for name, m in val_baselines:
        candidates.append({"name": name, **m})

    candidates.append(best_gated)
    candidates.append(best_margin)

    best_overall = pick_best(candidates, key1=args.select_metric, key2="accuracy")
    print("\n=== Selected best rule (from VAL only) ===")
    print(best_overall)

    print("\n=== Computing TEST model outputs ===")
    yt, pt, et, rt = compute_outputs(model, tokenizer, cfg, JsonDataset(args.test), device, batch_size=args.batch_size)

    print("\n=== Baseline TEST (argmax) ===")
    base_test = metrics_for_preds(yt, pred_argmax(pt))
    print(f"argmax | acc={base_test['accuracy']:.4f} f1={base_test['f1_macro']:.4f} off1={base_test['off_by_one_acc']:.4f} mae={base_test['mae']:.4f}")

    def apply_rule(rule, probs, exp, reg):
        if rule["name"] == "argmax":
            return pred_argmax(probs)
        if rule["name"] == "round_expected":
            return pred_expected_round(exp)
        if rule["name"] == "round_regression":
            return pred_reg_round(reg)
        if rule["name"] == "gated_expected":
            return pred_gated_expected(probs, exp, rule["tau"])[0]
        if rule["name"] == "margin_expected":
            return pred_margin_expected(probs, exp, rule["margin"])[0]
        raise ValueError(f"Unknown rule: {rule['name']}")

    print("\n=== Best rule TEST (single evaluation) ===")
    yhat_best = apply_rule(best_overall, pt, et, rt)
    best_test = metrics_for_preds(yt, yhat_best)

    print(f"{best_overall['name']} | acc={best_test['accuracy']:.4f} f1={best_test['f1_macro']:.4f} "
          f"off1={best_test['off_by_one_acc']:.4f} mae={best_test['mae']:.4f}")

    print("\n=== Delta vs argmax (TEST) ===")
    print(f"Δacc={best_test['accuracy'] - base_test['accuracy']:+.4f} "
          f"Δf1={best_test['f1_macro'] - base_test['f1_macro']:+.4f} "
          f"Δoff1={best_test['off_by_one_acc'] - base_test['off_by_one_acc']:+.4f} "
          f"Δmae={best_test['mae'] - base_test['mae']:+.4f}")


if __name__ == "__main__":
    main()