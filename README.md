# Amazon Reviews Rating Predictor (1–5 Stars)
Fine-tunes DeBERTa-v3-large to predict 1–5 star ratings from review text.

Key ideas:
- Mean pooling over token embeddings
- Classification + regression heads
- Hybrid loss for ordinal learning
- Dual balancing: per-class-per-domain sampling
