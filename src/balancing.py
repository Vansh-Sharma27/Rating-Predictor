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
    """
    Hybrid loss used by the project.

    Baseline terms (v1.x):
      - CE over 5 classes
      - SmoothL1(regression vs true rating)
      - SmoothL1(expected rating vs true rating)

    Optional ordinal term:
      - EMD/Wasserstein on 1D ordered classes via CDF distance
        (disabled by default, enabled with delta>0)
    """

    def __init__(
        self,
        class_weights: Optional[torch.Tensor] = None,
        alpha: float = 1.0,
        beta: float = 0.4,
        gamma: float = 0.2,
        delta: float = 0.0,
        emd_p: int = 1,
        label_smoothing: float = 0.05,
    ):
        super().__init__()
        self.alpha, self.beta, self.gamma, self.delta = alpha, beta, gamma, delta
        self.emd_p = int(emd_p)
        if self.emd_p not in (1, 2):
            raise ValueError("emd_p must be 1 or 2")

        self.label_smoothing = label_smoothing
        if class_weights is not None:
            self.register_buffer("class_weights", class_weights)
        else:
            self.class_weights = None
        self.register_buffer("rating_values", torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0]))

    def forward(self, logits: torch.Tensor, regression: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Standard CE (exact-class)
        ce = F.cross_entropy(
            logits,
            targets,
            weight=self.class_weights,
            label_smoothing=self.label_smoothing,
        )

        # Auxiliary regression heads (optional; set beta/gamma=0.0 to disable)
        true_r = targets.float() + 1.0
        reg = F.smooth_l1_loss(regression, true_r)

        probs = F.softmax(logits, dim=-1)
        expected = (probs * self.rating_values.unsqueeze(0)).sum(dim=-1)
        exp = F.smooth_l1_loss(expected, true_r)

        # Ordinal-aware EMD/Wasserstein on the 1D ordered classes via CDF distance
        if self.delta != 0.0:
            K = logits.size(-1)
            cdf_pred = probs.cumsum(dim=-1)
            cdf_true = F.one_hot(targets, num_classes=K).float().cumsum(dim=-1)
            if self.emd_p == 1:
                emd = (cdf_pred - cdf_true).abs().mean()
            else:
                emd = (cdf_pred - cdf_true).pow(2).mean()
        else:
            emd = logits.new_tensor(0.0)

        return self.alpha * ce + self.beta * reg + self.gamma * exp + self.delta * emd
