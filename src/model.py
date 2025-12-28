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
