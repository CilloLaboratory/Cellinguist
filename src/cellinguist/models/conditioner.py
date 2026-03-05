import torch
import torch.nn as nn
import torch.nn.functional as F

class LatentConditioner(nn.Module):
    """
    Combines latent z with condition embeddings.
    mode='concat': return concatenated [z, cond_emb]
    mode='delta':  return z + f([z, cond_emb])
    """
    def __init__(self, d_z: int, d_cond: int,
                 mode: str = "concat", d_hidden: int = 256):
        super().__init__()
        assert mode in ("concat", "delta")
        self.mode = mode
        self.d_z = d_z
        self.d_cond = d_cond
        
        if mode == "concat":
            self.out_dim = d_z + d_cond
            self.net = None  # no transform needed
        else:  # delta
            self.out_dim = d_z
            self.net = nn.Sequential(
                nn.Linear(d_z + d_cond, d_hidden),
                nn.ReLU(),
                nn.Linear(d_hidden, d_z),
            )
    
    def forward(self, z: torch.Tensor,
                cond_emb: torch.Tensor) -> torch.Tensor:
        if self.mode == "concat":
            return torch.cat([z, cond_emb], dim=-1)
        else:
            inp = torch.cat([z, cond_emb], dim=-1)
            delta = self.net(inp)
            return z + delta