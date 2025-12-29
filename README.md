# Rating Predictor (Amazon Reviews 2023) — 1 to 5 Stars
A transformer-based model that predicts a discrete star rating (1–5) from customer review text.

## Why this project
Predicting exact 1–5 ratings is harder than binary sentiment because the “middle” classes (2★/3★/4★) are linguistically ambiguous. This project focuses on:
- strong NLP modeling
- reproducible data pipeline
- measurable evaluation

## Approach (high level)
Model: `microsoft/deberta-v3-large`

Key ideas:
- **Mean pooling**: average token embeddings instead of relying only on `[CLS]`
- **Dual heads**:
  - classification head → 5-class probabilities
  - regression head → continuous rating signal (ordinal guidance)
- **Hybrid ordinal loss**:
  - CrossEntropy (exact class)
  - SmoothL1(regression vs true rating)
  - SmoothL1(expected rating from probabilities vs true rating)

## Data pipeline
Dataset: `McAuley-Lab/Amazon-Reviews-2023` (Hugging Face)

To reduce class imbalance and domain bias, training data is sampled with **dual balancing**:
- 4 domains: Electronics, Books, Clothing/Shoes/Jewelry, Home/Kitchen
- 5 rating classes
- equal samples **per-class-per-domain** (config-driven)

Validation/test sets are also generated as balanced splits so that macro-F1 is meaningful and classes are comparable.

Artifacts (data/checkpoints/results/logs) are stored outside git (recommended: on a larger disk for large runs).

## Final Results (Baseline)
**Validation (25k balanced samples):**
- Accuracy: **0.7078**
- Macro-F1: **0.7066**

**Test (25k balanced samples):**
- Accuracy: **0.6847**
- Macro-F1: **0.6835**

Additional quality signals (typical for the baseline):
- Off-by-one accuracy: ~0.97
- MAE: ~0.35 stars

## Experiments tried (no net gain on test)
These were tested with a clean protocol (tune on validation, test once):

1) **Ordinal post-processing / gating** (argmax vs rounded expected rating)
- No meaningful improvement in exact 5-class test accuracy.

2) **Stacker models on transformer outputs** (LogReg / HistGradientBoosting over probs + uncertainty + simple text features)
- Example test: Accuracy **0.6837**, Macro-F1 **0.6820** (worse than baseline).

3) **Ordinal-aware training loss (EMD / Wasserstein term)**
- No validation gain over baseline (best val Macro-F1 ~0.7025 in the run shown).

4) **Leakage-safe metadata (verified_purchase only)**
- Example test: Accuracy **0.6828**, Macro-F1 **0.6838** (no gain).

## Key takeaways
- Most errors are **off-by-one** (e.g., 4★ predicted as 5★), which indicates the model learns strong ordinal structure but exact boundaries between adjacent ratings remain difficult.
- The hardest classes are typically **2★ / 3★ / 4★** due to language ambiguity.

## Repo structure
- `config.yaml` — all hyperparameters and dataset settings
- `src/` — model, loss, trainer, preprocessing
- `scripts/prepare_validation_set.py` — builds balanced val/test splits
- `scripts/train.py` — training entrypoint
- `scripts/evaluate.py` — test evaluation
- `scripts/inference.py` — interactive and CLI inference

## Setup (GPU, Conda)
1) Create env and install deps (example):
- PyTorch GPU build (CUDA-enabled)
- `transformers`, `datasets`, `accelerate`, etc.

2) Important GPU note:
If PyTorch reports no GPU but `nvidia-smi` works, check MIG mode:
`sudo nvidia-smi -i 0 -mig 0`

## Run
From repo root:

1) Create balanced validation/test splits:
`python scripts/prepare_validation_set.py`

2) Train (recommended inside tmux):
`python scripts/train.py`

Optional useful flags for experiments:
- `--run-name <name>` to write artifacts under `checkpoints/<name>/` and `results/<name>/`
- `--init-from <checkpoint.pt>` to initialize weights from a prior run

3) Evaluate:
`python scripts/evaluate.py`

4) Inference:
- Interactive: `python scripts/inference.py --interactive`
- One-off: `python scripts/inference.py --text "..." --category "Electronics"`

## Configuration
Edit `config.yaml` to change:
- training steps, LR schedule, batch size, max_length
- dataset domains
- sample counts (train/val/test)

## Storage (portable by default)
By default, the project writes artifacts to local folders:
- `checkpoints/`, `results/`, `data/`, `logs/`

For large runs on cloud VMs, you can optionally move caches/artifacts to a larger disk by setting:
- `HF_HOME`, `HF_DATASETS_CACHE`, `TORCH_HOME`, `XDG_CACHE_HOME`
or by symlinking `checkpoints/ data/ results/ logs/` to another mount.

None of this is required for small runs.

## License / Notes
This repo does not redistribute the Amazon dataset. It downloads via Hugging Face at runtime.
