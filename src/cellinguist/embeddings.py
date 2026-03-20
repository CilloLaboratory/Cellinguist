from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn


class GeneEmbedding(nn.Module):
    """
    Wrapper around nn.Embedding for gene embeddings.

    Parameters
    ----------
    n_genes : int
        Number of genes (vocabulary size).
    d_gene : int
        Dimensionality of each gene embedding vector.
    pretrained_weight : Optional[torch.Tensor], default None
        If provided, must be of shape (n_genes, d_gene) and is used to
        initialize the embedding weights.
    freeze : bool, default True
        If True, embedding weights are not updated during training.
    """

    def __init__(
        self,
        n_genes: int,
        d_gene: int,
        pretrained_weight: Optional[torch.Tensor] = None,
        freeze: bool = True,
    ) -> None:
        super().__init__()

        self.n_genes = int(n_genes)
        self.d_gene = int(d_gene)

        self.embedding = nn.Embedding(self.n_genes, self.d_gene)

        if pretrained_weight is not None:
            if pretrained_weight.shape != (self.n_genes, self.d_gene):
                raise ValueError(
                    f"pretrained_weight has shape {pretrained_weight.shape}, "
                    f"expected ({self.n_genes}, {self.d_gene})"
                )
            with torch.no_grad():
                self.embedding.weight.copy_(pretrained_weight)

        if freeze:
            self.embedding.weight.requires_grad_(False)

    def forward(self, gene_indices: torch.Tensor) -> torch.Tensor:
        """
        Look up embeddings for the given gene indices.

        Parameters
        ----------
        gene_indices : torch.Tensor
            Integer tensor of gene indices, typically of shape (B, G),
            but any shape is accepted as long as values are in [0, n_genes).

        Returns
        -------
        torch.Tensor
            Gene embeddings with shape (*gene_indices.shape, d_gene),
            e.g., (B, G, d_gene) for a (B, G) input.
        """
        return self.embedding(gene_indices)


def load_gene_embeddings(path: str, map_location: Optional[str] = None) -> torch.Tensor:
    """
    Load a gene embedding weight matrix from disk.
    """
    return torch.load(path, map_location=map_location)


def save_gene_embeddings(path: str, weight: torch.Tensor) -> None:
    """
    Save a gene embedding weight matrix to disk.
    """
    if weight.dim() != 2:
        raise ValueError(
            f"Expected weight to have 2 dimensions (n_genes, d_gene), "
            f"got shape {tuple(weight.shape)}"
        )
    torch.save(weight, path)


def load_token_vocab(path: str) -> Dict[str, int]:
    """Load a token -> id vocabulary mapping from JSON."""
    with Path(path).open('r', encoding='utf-8') as f:
        raw = json.load(f)
    if not isinstance(raw, dict):
        raise ValueError('Token vocabulary JSON must be an object mapping token strings to ids.')

    token_to_id: Dict[str, int] = {}
    for token, idx in raw.items():
        if not isinstance(token, str):
            raise ValueError('Token vocabulary keys must be strings.')
        if not isinstance(idx, int):
            raise ValueError(f"Token vocabulary id for '{token}' must be an integer.")
        token_to_id[token] = idx
    return token_to_id


def save_token_vocab(path: str, token_to_id: Dict[str, int]) -> None:
    """Save a token -> id vocabulary mapping to JSON."""
    normalized = {str(token): int(idx) for token, idx in token_to_id.items()}
    with Path(path).open('w', encoding='utf-8') as f:
        json.dump(normalized, f, indent=2, sort_keys=True)


def initialize_random_embeddings(
    n_genes: int,
    d_gene: int,
    seed: Optional[int] = None,
    dtype: torch.dtype = torch.float32,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """
    Initialize a random gene embedding matrix.
    """
    if seed is not None:
        gen = torch.Generator(device=device)
        gen.manual_seed(seed)
    else:
        gen = None

    weight = torch.randn(
        n_genes,
        d_gene,
        generator=gen,
        dtype=dtype,
        device=device,
    )
    weight = weight / (d_gene**0.5)
    return weight
