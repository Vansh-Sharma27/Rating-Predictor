#!/usr/bin/env bash
set -euo pipefail

# ---------- config ----------
cat > config.yaml << 'YAML'
model:
  name: "microsoft/deberta-v3-large"
  num_labels: 5
  classifier_dropout: 0.15
  use_mean_pooling: true

tokenizer:
  max_length: 384
  padding: "longest"
  truncation: true

training:
  per_device_train_batch_size: 12
  gradient_accumulation_steps: 4
  learning_rate: 1.5e-5
  weight_decay: 0.01
  warmup_ratio: 0.10
  lr_scheduler_type: "cosine"
  max_steps: 40000
  eval_steps: 1000
  save_steps: 5000
  logging_steps: 100
  early_stopping_patience: 8
  early_stopping_threshold: 0.0005
  fp16: true
  gradient_checkpointing: false
  max_grad_norm: 1.0
  label_smoothing: 0.05

data:
  dataset_name: "McAuley-Lab/Amazon-Reviews-2023"
  subsets:
    - "raw_review_Electronics"
    - "raw_review_Books"
    - "raw_review_Clothing_Shoes_and_Jewelry"
    - "raw_review_Home_and_Kitchen"
  train_samples: 1500000
  eval_samples: 25000
  test_samples: 25000
  num_workers: 4

class_balancing:
  sampling_strategy: "gentle"
  loss_type: "hybrid"
  loss_alpha: 1.0
  loss_beta: 0.4
  loss_gamma: 0.2

paths:
  checkpoint_dir: "checkpoints"
  results_dir: "results"
  logs_dir: "logs"

seed: 42
YAML

# ---------- src ----------
mkdir -p src scripts

cat > src/utils.py << 'PY'
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
PY

cat > src/preprocessing.py << 'PY'
import re
import unicodedata
from dataclasses import dataclass
from typing import Optional, Dict, Any

@dataclass
class PreprocessorConfig:
    min_length: int = 15
    max_length: int = 8000
    include_category: bool = True
    include_title: bool = True

class ReviewPreprocessor:
    URL_PATTERN = re.compile(r'https?://\S+|www\.\S+')
    EMAIL_PATTERN = re.compile(r'\S+@\S+\.\S+')
    WHITESPACE_PATTERN = re.compile(r'\s+')
    HTML_TAG_PATTERN = re.compile(r'<[^>]+>')

    HTML_ENTITIES = {
        '&amp;': '&', '&lt;': '<', '&gt;': '>',
        '&quot;': '"', '&#39;': "'", '&nbsp;': ' ',
        'â€™': "'", 'â€œ': '"', 'â€': '"',
        'â€"': '—', 'â€"': '–', '&ndash;': '–',
        '&mdash;': '—', '&hellip;': '...'
    }

    TEXT_COLUMNS = ['text', 'reviewText', 'review_text', 'body']
    TITLE_COLUMNS = ['title', 'summary', 'review_title', 'headline']
    RATING_COLUMNS = ['rating', 'overall', 'score', 'stars']
    CATEGORY_COLUMNS = ['category', 'main_category', 'product_category']

    def __init__(self, config: Optional[PreprocessorConfig] = None):
        self.config = config or PreprocessorConfig()

    def _get_field(self, row: Dict[str, Any], cols, default=None):
        for c in cols:
            if c in row and row[c] is not None:
                return row[c]
        return default

    def clean_text(self, text: str) -> Optional[str]:
        if not text or not isinstance(text, str):
            return None
        text = text.strip()
        if not text:
            return None

        text = unicodedata.normalize("NFKC", text)
        text = self.HTML_TAG_PATTERN.sub(" ", text)
        for k, v in self.HTML_ENTITIES.items():
            text = text.replace(k, v)

        text = self.URL_PATTERN.sub("[URL]", text)
        text = self.EMAIL_PATTERN.sub("[EMAIL]", text)
        text = self.WHITESPACE_PATTERN.sub(" ", text).strip()

        if len(text) < self.config.min_length:
            return None
        if len(text) > self.config.max_length:
            text = text[: self.config.max_length]
        return text

    def format_input(self, text: str, title: Optional[str] = None, category: Optional[str] = None) -> str:
        parts = []
        if self.config.include_category and category:
            cat = str(category).replace("_", " ").strip()
            if cat:
                parts.append(f"[Category: {cat}]")
        if self.config.include_title and title and isinstance(title, str):
            t = title.strip()
            if len(t) > 2:
                parts.append(t)
        parts.append(text)
        return " ".join(parts)

    def process_row(self, row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        raw_text = self._get_field(row, self.TEXT_COLUMNS, "")
        text = self.clean_text(raw_text)
        if text is None:
            return None

        rating = self._get_field(row, self.RATING_COLUMNS)
        if rating is None:
            return None

        try:
            rating = int(float(rating))
            if not (1 <= rating <= 5):
                return None
        except Exception:
            return None

        title = self._get_field(row, self.TITLE_COLUMNS, "")
        category = self._get_field(row, self.CATEGORY_COLUMNS, "")
        formatted = self.format_input(text, title, category)

        return {"text": formatted, "label": rating - 1}
PY

cat > src/model.py << 'PY'
import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig

class MeanPooling(nn.Module):
    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        mask = attention_mask.unsqueeze(-1).float()
        summed = (hidden_states * mask).sum(dim=1)
        denom = mask.sum(dim=1).clamp(min=1e-9)
        return summed / denom

class RatingPredictor(nn.Module):
    def __init__(self, model_name: str, num_labels: int = 5, classifier_dropout: float = 0.15,
                 use_mean_pooling: bool = True, gradient_checkpointing: bool = False):
        super().__init__()
        self.config = AutoConfig.from_pretrained(model_name)
        self.encoder = AutoModel.from_pretrained(model_name, config=self.config)
        if gradient_checkpointing:
            self.encoder.gradient_checkpointing_enable()

        self.use_mean_pooling = use_mean_pooling
        self.pooler = MeanPooling() if use_mean_pooling else None
        h = self.config.hidden_size

        self.classifier = nn.Sequential(
            nn.Dropout(classifier_dropout),
            nn.Linear(h, h // 2),
            nn.GELU(),
            nn.LayerNorm(h // 2),
            nn.Dropout(classifier_dropout / 2),
            nn.Linear(h // 2, num_labels),
        )

        self.regressor = nn.Sequential(
            nn.Dropout(classifier_dropout),
            nn.Linear(h, h // 4),
            nn.GELU(),
            nn.Linear(h // 4, 1),
        )

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        if self.use_mean_pooling:
            pooled = self.pooler(out.last_hidden_state, attention_mask)
        else:
            pooled = out.last_hidden_state[:, 0, :]
        logits = self.classifier(pooled)
        regression = self.regressor(pooled).squeeze(-1)
        return {"logits": logits, "regression": regression}
PY

cat > src/balancing.py << 'PY'
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import Counter
from typing import Optional, List

class ClassBalancer:
    def __init__(self, labels: List[int], strategy: str = "gentle", n_classes: int = 5):
        counts = Counter(labels)
        for i in range(n_classes):
            counts.setdefault(i, 1)
        freqs = np.array([counts[i] for i in range(n_classes)], dtype=np.float64)
        freqs = freqs / freqs.sum()

        if strategy == "sqrt":
            w = 1.0 / np.sqrt(freqs)
        elif strategy == "gentle":
            w = 1.0 / np.power(freqs, 0.25)
        else:
            w = np.ones(n_classes)

        w = w / w.mean()
        self.class_weights = torch.tensor(w, dtype=torch.float32)
        print(f"Class distribution: {dict(sorted(counts.items()))}")
        print(f"Strategy: {strategy}")
        print(f"Class weights: {self.class_weights.numpy().round(3)}")

class HybridLoss(nn.Module):
    def __init__(self, class_weights: Optional[torch.Tensor] = None,
                 alpha: float = 1.0, beta: float = 0.4, gamma: float = 0.2, label_smoothing: float = 0.05):
        super().__init__()
        self.alpha, self.beta, self.gamma = alpha, beta, gamma
        self.label_smoothing = label_smoothing
        if class_weights is not None:
            self.register_buffer("class_weights", class_weights)
        else:
            self.class_weights = None
        self.register_buffer("rating_values", torch.tensor([1., 2., 3., 4., 5.]))

    def forward(self, logits: torch.Tensor, regression: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(logits, targets, weight=self.class_weights, label_smoothing=self.label_smoothing)
        true_r = targets.float() + 1.0
        reg = F.smooth_l1_loss(regression, true_r)
        probs = F.softmax(logits, dim=-1)
        expected = (probs * self.rating_values.unsqueeze(0)).sum(dim=-1)
        exp = F.smooth_l1_loss(expected, true_r)
        return self.alpha * ce + self.beta * reg + self.gamma * exp
PY

cat > src/trainer.py << 'PY'
import os, json
from datetime import datetime
import numpy as np
import torch
from torch.amp import autocast, GradScaler
from tqdm import tqdm
from transformers import get_scheduler
from .metrics import RatingMetrics

class Trainer:
    def __init__(self, model, tokenizer, loss_fn, config, device="cuda"):
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.loss_fn = loss_fn.to(device)
        self.cfg = config
        self.device = device

        tcfg = config["training"]
        self.grad_accum = tcfg["gradient_accumulation_steps"]
        self.max_steps = tcfg["max_steps"]
        self.eval_steps = tcfg["eval_steps"]
        self.save_steps = tcfg["save_steps"]
        self.log_steps = tcfg["logging_steps"]
        self.max_grad_norm = tcfg["max_grad_norm"]
        self.fp16 = tcfg["fp16"]

        self.patience = tcfg["early_stopping_patience"]
        self.threshold = tcfg["early_stopping_threshold"]

        self.ckpt_dir = config["paths"]["checkpoint_dir"]
        self.res_dir = config["paths"]["results_dir"]
        os.makedirs(self.ckpt_dir, exist_ok=True)
        os.makedirs(self.res_dir, exist_ok=True)

        self.global_step = 0
        self.best_f1 = 0.0
        self.best_acc = 0.0
        self.patience_counter = 0
        self.training_log = []

        self.scaler = GradScaler("cuda") if self.fp16 else None

    def _setup_optim(self):
        tcfg = self.cfg["training"]
        no_decay = ["bias", "LayerNorm.weight", "layer_norm.weight"]
        params = [
            {"params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
             "weight_decay": tcfg["weight_decay"]},
            {"params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
             "weight_decay": 0.0},
        ]
        self.optimizer = torch.optim.AdamW(params, lr=tcfg["learning_rate"], betas=(0.9, 0.999), eps=1e-8)

        warmup = int(self.max_steps * tcfg["warmup_ratio"])
        self.scheduler = get_scheduler(
            name=tcfg["lr_scheduler_type"],
            optimizer=self.optimizer,
            num_warmup_steps=warmup,
            num_training_steps=self.max_steps,
        )
        print(f"Optimizer: {self.max_steps} steps, {warmup} warmup")

    def collate_fn(self, batch):
        texts = [x["text"] for x in batch]
        labels = torch.tensor([x["label"] for x in batch], dtype=torch.long)
        enc = self.tokenizer(
            texts, padding="longest", truncation=True, max_length=self.cfg["tokenizer"]["max_length"], return_tensors="pt"
        )
        return {"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"], "labels": labels}

    def _train_micro_step(self, batch) -> float:
        input_ids = batch["input_ids"].to(self.device)
        attention_mask = batch["attention_mask"].to(self.device)
        labels = batch["labels"].to(self.device)

        if self.fp16:
            with autocast("cuda"):
                out = self.model(input_ids, attention_mask)
                loss = self.loss_fn(out["logits"], out["regression"], labels) / self.grad_accum
            self.scaler.scale(loss).backward()
            return float(loss.item() * self.grad_accum)
        else:
            out = self.model(input_ids, attention_mask)
            loss = self.loss_fn(out["logits"], out["regression"], labels) / self.grad_accum
            loss.backward()
            return float(loss.item() * self.grad_accum)

    def _optim_step(self):
        if self.fp16:
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            self.optimizer.step()

        self.scheduler.step()
        self.optimizer.zero_grad()

    @torch.no_grad()
    def evaluate(self, loader):
        self.model.eval()
        all_preds, all_labels = [], []
        total_loss, n = 0.0, 0

        for batch in tqdm(loader, desc="Evaluating", leave=False):
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels = batch["labels"].to(self.device)

            if self.fp16:
                with autocast("cuda"):
                    out = self.model(input_ids, attention_mask)
                    loss = self.loss_fn(out["logits"], out["regression"], labels)
            else:
                out = self.model(input_ids, attention_mask)
                loss = self.loss_fn(out["logits"], out["regression"], labels)

            total_loss += float(loss.item())
            n += 1

            preds = torch.argmax(out["logits"], dim=-1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

        m = RatingMetrics.compute(np.array(all_labels), np.array(all_preds))
        m["loss"] = total_loss / max(n, 1)
        self.model.train()
        return m

    def _save_checkpoint(self, name: str, metrics=None):
        path = os.path.join(self.ckpt_dir, name)
        ckpt = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "global_step": self.global_step,
            "best_f1": self.best_f1,
            "best_acc": self.best_acc,
            "config": self.cfg,
            "metrics": metrics,
        }
        if self.fp16:
            ckpt["scaler_state_dict"] = self.scaler.state_dict()
        torch.save(ckpt, path)
        print(f"  Saved: {name}")

    def _save_log(self):
        path = os.path.join(self.res_dir, "training_log.json")
        with open(path, "w") as f:
            json.dump(
                {"log": self.training_log, "best_f1": self.best_f1, "best_acc": self.best_acc,
                 "total_steps": self.global_step, "timestamp": datetime.now().isoformat()},
                f, indent=2
            )

    def train(self, train_loader, val_loader):
        self._setup_optim()
        self.model.train()

        it = iter(train_loader)
        running, accum = 0.0, 0
        pbar = tqdm(total=self.max_steps, desc="Training")

        while self.global_step < self.max_steps:
            try:
                batch = next(it)
            except StopIteration:
                it = iter(train_loader)
                batch = next(it)

            running += self._train_micro_step(batch)
            accum += 1

            if accum >= self.grad_accum:
                self._optim_step()
                self.global_step += 1
                accum = 0
                pbar.update(1)

                if self.global_step % self.log_steps == 0:
                    avg = running / self.log_steps
                    lr = self.scheduler.get_last_lr()[0]
                    pbar.set_postfix({"loss": f"{avg:.4f}", "lr": f"{lr:.2e}"})
                    self.training_log.append({"step": self.global_step, "loss": avg, "lr": lr})
                    running = 0.0

                if self.global_step % self.eval_steps == 0:
                    print(f"\nStep {self.global_step}: Evaluation")
                    m = self.evaluate(val_loader)
                    print(f"  Loss: {m['loss']:.4f}")
                    print(f"  Accuracy: {m['accuracy']:.4f}")
                    print(f"  F1 Macro: {m['f1_macro']:.4f}")
                    print(f"  Off-by-one: {m['off_by_one_acc']:.4f}")

                    if m["f1_macro"] > self.best_f1 + self.threshold:
                        self.best_f1 = m["f1_macro"]
                        self.best_acc = m["accuracy"]
                        self.patience_counter = 0
                        self._save_checkpoint("best_model.pt", m)
                        print(f"  ✓ New best! F1={self.best_f1:.4f}, Acc={self.best_acc:.4f}")
                    else:
                        self.patience_counter += 1
                        print(f"  No improvement. Patience: {self.patience_counter}/{self.patience}")

                    if self.patience_counter >= self.patience:
                        print(f"\nEarly stopping at step {self.global_step}")
                        break

                if self.global_step % self.save_steps == 0:
                    self._save_checkpoint(f"checkpoint_{self.global_step}.pt")

        pbar.close()
        self._save_checkpoint("final_model.pt")
        self._save_log()
        print(f"\nTraining complete. Best F1: {self.best_f1:.4f}, Best Acc: {self.best_acc:.4f}")
        return self.best_f1
PY

cat > src/metrics.py << 'PY'
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report, mean_absolute_error

class RatingMetrics:
    NAMES = ["1-star", "2-star", "3-star", "4-star", "5-star"]

    @staticmethod
    def compute(y_true: np.ndarray, y_pred: np.ndarray):
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        return {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
            "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
            "mae": float(mean_absolute_error(y_true, y_pred)),
            "off_by_one_acc": float(np.mean(np.abs(y_true - y_pred) <= 1)),
            "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
            "report": classification_report(y_true, y_pred, target_names=RatingMetrics.NAMES, zero_division=0),
        }
PY

# ---------- scripts ----------
cat > scripts/prepare_validation_set.py << 'PY'
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
PY

cat > scripts/train.py << 'PY'
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
PY

cat > scripts/evaluate.py << 'PY'
import os, json
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
    def __init__(self, data): self.data = data
    def __len__(self): return len(self.data)
    def __getitem__(self, idx): return self.data[idx]

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
    cfg = load_config("config.yaml")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if not os.path.exists("data/test.json"):
        print("Test set not found. Run: python scripts/prepare_validation_set.py")
        return

    with open("data/test.json", "r", encoding="utf-8") as f:
        test_data = json.load(f)

    tokenizer = AutoTokenizer.from_pretrained(cfg["model"]["name"])
    model_path = os.path.join(cfg["paths"]["checkpoint_dir"], "best_model.pt")
    print(f"Loading: {model_path}")

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

    ds = ReviewDataset(test_data)
    loader = DataLoader(ds, batch_size=32, shuffle=False, num_workers=4,
                        collate_fn=lambda b: collate(b, tokenizer, cfg["tokenizer"]["max_length"]))

    y_true, y_pred = run_eval(model, loader, device)
    m = RatingMetrics.compute(y_true, y_pred)

    print("\n" + "="*70)
    print("EVALUATION RESULTS")
    print("="*70)
    print(m["report"])
    print("-"*70)
    print(f"Accuracy            {m['accuracy']:.4f}")
    print(f"F1 Macro            {m['f1_macro']:.4f}")
    print(f"F1 Weighted         {m['f1_weighted']:.4f}")
    print(f"MAE                 {m['mae']:.4f}")
    print(f"Off-by-one Accuracy {m['off_by_one_acc']:.4f}")
    print("-"*70)

    os.makedirs(cfg["paths"]["results_dir"], exist_ok=True)
    out_path = os.path.join(cfg["paths"]["results_dir"], "test_results.json")
    with open(out_path, "w") as f:
        json.dump(m, f, indent=2)
    print(f"Saved metrics to {out_path}")

if __name__ == "__main__":
    main()
PY

cat > scripts/inference.py << 'PY'
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
PY

# README
cat > README.md << 'MD'
# Amazon Reviews Rating Predictor (1–5 Stars)
Fine-tunes DeBERTa-v3-large to predict 1–5 star ratings from review text.

Key ideas:
- Mean pooling over token embeddings
- Classification + regression heads
- Hybrid loss for ordinal learning
- Dual balancing: per-class-per-domain sampling
MD

chmod +x bootstrap.sh
echo "Bootstrap complete."