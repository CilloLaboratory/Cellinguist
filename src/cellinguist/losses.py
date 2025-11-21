import torch
import torch.nn as nn
import torch.nn.functional as F

def kl_loss(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """
    Standard VAE KL divergence between N(mu, sigma^2) and N(0, I).
    Returns mean KL over batch.
    """
    kl = 0.5 * (torch.exp(logvar) + mu**2 - 1.0 - logvar)
    return kl.sum(dim=1).mean()