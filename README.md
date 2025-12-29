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

To avoid class imbalance and domain bias, training data is sampled with **dual balancing**:
- 4 domains: Electronics, Books, Clothing/Shoes/Jewelry, Home/Kitchen
- 5 rating classes
- equal samples **per-class-per-domain** (config-driven)

Artifacts (data/checkpoints/results/logs) are stored outside git (recommended: on ephemeral disk).

## Results (example run)
Test set: 25k balanced samples (5k per class)
- Accuracy: ~0.685
- Macro-F1: ~0.684
- Off-by-one accuracy: ~0.97
- MAE: ~0.35 stars

(Exact numbers depend on seed/config.)

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
- transformers, datasets, accelerate, etc.

2) Important GPU note:
If PyTorch reports no GPU but `nvidia-smi` works, check MIG mode:
`sudo nvidia-smi -i 0 -mig 0`

## Run
From repo root:

1) Create balanced validation/test splits:
`python scripts/prepare_validation_set.py`

2) Train (recommended inside tmux):
`python scripts/train.py`

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