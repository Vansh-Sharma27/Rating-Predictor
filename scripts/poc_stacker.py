import os, sys, json, argparse, re
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast
from transformers import AutoTokenizer
from tqdm import tqdm

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.utils import load_config
from src.model import RatingPredictor
from src.metrics import RatingMetrics

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
import joblib

CAT_RE = re.compile(r"^\[Category:\s*([^\]]+)\]\s*")

NEG_WORDS = {
    "not","no","never","none","n't","cannot","cant","don't","dont","won't","wont",
    "didn't","didnt","isn't","isnt","wasn't","wasnt","shouldn't","shouldnt",
    "couldn't","couldnt","wouldn't","wouldnt"
}

class JsonDataset(Dataset):
    def __init__(self, path: str):
        with open(path, "r", encoding="utf-8") as f:
            self.data = json.load(f)
    def __len__(self): return len(self.data)
    def __getitem__(self, idx): return self.data[idx]

def parse_category(text: str):
    m = CAT_RE.match(text)
    return m.group(1).strip() if m else None

def text_features(text: str):
    # Basic, cheap signals
    s = text
    chars = len(s)
    words = s.split()
    n_words = len(words)

    exclam = s.count("!")
    quest = s.count("?")
    digits = sum(c.isdigit() for c in s)

    letters = sum(c.isalpha() for c in s)
    uppers = sum(c.isupper() for c in s)
    upper_ratio = (uppers / letters) if letters else 0.0

    toks = re.findall(r"[a-zA-Z']+", s.lower())
    neg = sum(t in NEG_WORDS for t in toks)

    return np.array([chars, n_words, exclam, quest, digits, upper_ratio, neg], dtype=np.float32)

def softmax_entropy(p):
    p = np.clip(p, 1e-9, 1.0)
    return float(-(p * np.log(p)).sum())

@torch.no_grad()
def build_features(cfg, model, tok, json_path: str, cache_dir: str, device: str, batch_size: int):
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, Path(json_path).stem + ".npz")
    if os.path.exists(cache_path):
        z = np.load(cache_path)
        return z["X"], z["y"], z["probs"]

    ds = JsonDataset(json_path)

    # category one-hot mapping from config subsets
    cats = [s[11:].replace("_", " ") if s.startswith("raw_review_") else s.replace("_", " ") for s in cfg["data"]["subsets"]]
    cat_to_idx = {c: i for i, c in enumerate(cats)}
    n_cat = len(cats)

    rating_values = np.array([1,2,3,4,5], dtype=np.float32)

    X_rows = []
    y_rows = []
    probs_rows = []

    def collate(batch):
        texts = [b["text"] for b in batch]
        labels = np.array([b["label"] for b in batch], dtype=np.int64)
        enc = tok(
            texts,
            padding="longest",
            truncation=True,
            max_length=cfg["tokenizer"]["max_length"],
            return_tensors="pt",
        )
        return texts, enc["input_ids"], enc["attention_mask"], labels

    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=4, collate_fn=collate)

    for texts, input_ids, attn, labels in tqdm(loader, desc=f"Featurizing {Path(json_path).name}"):
        input_ids = input_ids.to(device)
        attn = attn.to(device)

        if device == "cuda":
            with autocast("cuda"):
                out = model(input_ids, attn)
        else:
            out = model(input_ids, attn)

        logits = out["logits"]
        probs = torch.softmax(logits, dim=-1).float().cpu().numpy()   # (B,5)
        reg = out["regression"].float().cpu().numpy()                 # (B,)

        expected = (probs * rating_values[None, :]).sum(axis=1)       # (B,)

        # model-derived features
        pmax = probs.max(axis=1)
        sorted_p = np.sort(probs, axis=1)
        margin = sorted_p[:, -1] - sorted_p[:, -2]
        ent = np.array([softmax_entropy(p) for p in probs], dtype=np.float32)
        abs_er = np.abs(expected - reg).astype(np.float32)

        for i, t in enumerate(texts):
            # parse category -> one-hot
            c = parse_category(t)
            cat_oh = np.zeros((n_cat,), dtype=np.float32)
            if c in cat_to_idx:
                cat_oh[cat_to_idx[c]] = 1.0

            tf = text_features(t)  # 7 dims

            # concat features:
            # [probs(5), expected(1), reg(1), entropy(1), margin(1), pmax(1), abs(expected-reg)(1), text_feats(7), cat_onehot(n_cat)]
            feat = np.concatenate([
                probs[i].astype(np.float32),
                np.array([expected[i], reg[i], ent[i], margin[i], pmax[i], abs_er[i]], dtype=np.float32),
                tf,
                cat_oh
            ], axis=0)

            X_rows.append(feat)
            y_rows.append(int(labels[i]))
            probs_rows.append(probs[i])

    X = np.stack(X_rows, axis=0)
    y = np.array(y_rows, dtype=np.int64)
    probs_all = np.stack(probs_rows, axis=0)

    np.savez_compressed(cache_path, X=X, y=y, probs=probs_all)
    return X, y, probs_all

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="checkpoints/best_model.pt")
    ap.add_argument("--train", default="data/stacker_train.json")
    ap.add_argument("--val", default="data/val.json")
    ap.add_argument("--test", default="data/test.json")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--cache-dir", default="results/poc2_cache")
    ap.add_argument("--out-model", default="results/stacker_lr.joblib")
    args = ap.parse_args()

    cfg = load_config("config.yaml")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tok = AutoTokenizer.from_pretrained(cfg["model"]["name"])

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

    # Build features (cached)
    Xtr, ytr, ptr = build_features(cfg, model, tok, args.train, args.cache_dir, device, args.batch_size)
    Xv,  yv,  pv  = build_features(cfg, model, tok, args.val,   args.cache_dir, device, args.batch_size)
    Xt,  yt,  pt  = build_features(cfg, model, tok, args.test,  args.cache_dir, device, args.batch_size)

    # Baseline (argmax of transformer probs)
    print("\n=== Baseline (argmax transformer) ===")
    base_val = RatingMetrics.compute(yv, np.argmax(pv, axis=1))
    base_test = RatingMetrics.compute(yt, np.argmax(pt, axis=1))
    print(f"VAL  acc={base_val['accuracy']:.4f} f1={base_val['f1_macro']:.4f}")
    print(f"TEST acc={base_test['accuracy']:.4f} f1={base_test['f1_macro']:.4f}")

    # Train stacker: tune C on VAL for macro-F1
    Cs = [0.1, 0.3, 1.0, 3.0, 10.0]
    best = None

    for C in Cs:
        clf = Pipeline([
            ("scaler", StandardScaler()),
            ("lr", LogisticRegression(
                C=C,
                solver="saga",
                multi_class="multinomial",
                max_iter=500,
                n_jobs=-1,
                random_state=cfg["seed"],
            ))
        ])
        clf.fit(Xtr, ytr)
        pred_v = clf.predict(Xv)
        f1 = f1_score(yv, pred_v, average="macro")
        acc = (pred_v == yv).mean()
        print(f"C={C:<4} | VAL acc={acc:.4f} f1_macro={f1:.4f}")
        if best is None or f1 > best["f1"]:
            best = {"C": C, "f1": f1, "acc": acc, "clf": clf}

    print("\n=== Best stacker (chosen on VAL only) ===")
    print(best)

    # Evaluate ONCE on TEST using best
    pred_t = best["clf"].predict(Xt)
    m_test = RatingMetrics.compute(yt, pred_t)

    print("\n=== Stacker TEST ===")
    print(f"acc={m_test['accuracy']:.4f} f1_macro={m_test['f1_macro']:.4f} off1={m_test['off_by_one_acc']:.4f} mae={m_test['mae']:.4f}")

    print("\n=== Delta vs baseline (TEST) ===")
    print(f"Δacc={m_test['accuracy'] - base_test['accuracy']:+.4f} "
          f"Δf1={m_test['f1_macro'] - base_test['f1_macro']:+.4f} "
          f"Δoff1={m_test['off_by_one_acc'] - base_test['off_by_one_acc']:+.4f} "
          f"Δmae={m_test['mae'] - base_test['mae']:+.4f}")

    # Save stacker
    os.makedirs(os.path.dirname(args.out_model), exist_ok=True)
    joblib.dump(
        {"pipeline": best["clf"], "C": best["C"], "feature_note": "probs+ordinal+text+domain"},
        args.out_model
    )
    print(f"\nSaved stacker -> {args.out_model}")

if __name__ == "__main__":
    main()