import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import os, json, argparse, warnings
import torch
from torch.amp import autocast
from transformers import AutoTokenizer
from src.utils import load_config
from src.model import RatingPredictor
from src.preprocessing import ReviewPreprocessor, PreprocessorConfig

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

LABELS = {
    0: "1-star (Very Negative)",
    1: "2-star (Negative)",
    2: "3-star (Neutral)",
    3: "4-star (Positive)",
    4: "5-star (Very Positive)",
}

def load_model(cfg, model_path, device):
    ckpt = torch.load(model_path, map_location=device)
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
    return model, ckpt.get("metrics", {})

@torch.no_grad()
def predict(cfg, model, tok, pre, text, title=None, category=None, device="cuda"):
    formatted = pre.format_input(text, title, category)
    enc = tok(formatted, padding="max_length", truncation=True,
              max_length=cfg["tokenizer"]["max_length"], return_tensors="pt")
    input_ids = enc["input_ids"].to(device)
    attn = enc["attention_mask"].to(device)

    with autocast("cuda") if device == "cuda" else torch.no_grad():
        out = model(input_ids, attn)

    logits = out["logits"]
    regression = out["regression"].item()

    probs = torch.softmax(logits, dim=-1)[0]
    pred = int(torch.argmax(probs).item())
    conf = float(probs[pred].item())

    rating_values = torch.tensor([1.,2.,3.,4.,5.], device=device)
    expected = float((probs * rating_values).sum().item())

    return {
        "predicted_rating": pred + 1,
        "predicted_label": LABELS[pred],
        "confidence": round(conf, 4),
        "expected_rating": round(expected, 2),
        "regression_value": round(regression, 2),
        "probabilities": {f"{i+1}-star": round(float(probs[i].item()), 4) for i in range(5)},
    }

def interactive(cfg, model, tok, pre, device):
    print("\nCommands: /category <name>, /title <text>, /clear, /quit, /exit")
    cat, title = None, None
    while True:
        s = input("\n📝 Enter review: ").strip()
        if not s:
            continue
        if s.startswith("/"):
            cmd = s.split()[0].lower()
            arg = s[len(cmd):].strip()
            if cmd in ("/quit", "/exit", "/q"):
                print("Goodbye!")
                break
            if cmd == "/category":
                cat = arg if arg else None
                print(f"Category = {cat}")
                continue
            if cmd == "/title":
                title = arg if arg else None
                print(f"Title = {title}")
                continue
            if cmd == "/clear":
                cat, title = None, None
                print("Cleared.")
                continue
            print("Unknown command.")
            continue

        r = predict(cfg, model, tok, pre, s, title=title, category=cat, device=device)
        print("\n" + "-"*50)
        print(f"⭐ Predicted: {r['predicted_rating']}/5 | {r['predicted_label']}")
        print(f"📈 Expected rating: {r['expected_rating']:.2f}")
        print(f"📊 Regression head: {r['regression_value']:.2f}")
        print(f"🎯 Confidence: {r['confidence']*100:.1f}%")
        for k, v in r["probabilities"].items():
            bar = "█" * int(v * 20)
            print(f"  {k}: {bar} {v*100:.1f}%")
        print("-"*50)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="checkpoints/best_model.pt")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--interactive", "-i", action="store_true")
    ap.add_argument("--text", type=str)
    ap.add_argument("--category", type=str)
    ap.add_argument("--title", type=str)
    args = ap.parse_args()

    cfg = load_config(args.config)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tok = AutoTokenizer.from_pretrained(cfg["model"]["name"])
    pre = ReviewPreprocessor(PreprocessorConfig(min_length=1, max_length=10000, include_category=True, include_title=True))
    model, metrics = load_model(cfg, args.model, device)
    if metrics:
        print(f"Loaded model. Stored val f1_macro={metrics.get('f1_macro','N/A')}")

    if args.interactive:
        interactive(cfg, model, tok, pre, device)
        return

    if not args.text:
        print("Provide --text or --interactive")
        return

    print(json.dumps(predict(cfg, model, tok, pre, args.text, title=args.title, category=args.category, device=device), indent=2))

if __name__ == "__main__":
    main()
