import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

# 3. Define a lightweight Pseudotime head: a linear regressor from d_model → 1
class PseudotimeHead(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.fc = nn.Linear(input_dim, 1)

    def forward(self, cls_emb: torch.Tensor) -> torch.Tensor:
        # cls_emb: (B, d_model), returns (B,)
        return self.fc(cls_emb).squeeze(1)
