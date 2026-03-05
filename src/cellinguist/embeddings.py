from __future__ import annotations

from typing import Optional

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

        # Optionally freeze the embedding weights
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

    Parameters
    ----------
    path : str
        Path to a saved tensor file (typically created by save_gene_embeddings).
    map_location : Optional[str], default None
        Passed to torch.load; e.g. "cpu" to force loading on CPU.

    Returns
    -------
    torch.Tensor
        Tensor of shape (n_genes, d_gene) containing the embedding weights.
    """
    return torch.load(path, map_location=map_location)


def save_gene_embeddings(path: str, weight: torch.Tensor) -> None:
    """
    Save a gene embedding weight matrix to disk.

    Parameters
    ----------
    path : str
        Destination path for the tensor file.
    weight : torch.Tensor
        Tensor of shape (n_genes, d_gene) to save.
    """
    if weight.dim() != 2:
        raise ValueError(
            f"Expected weight to have 2 dimensions (n_genes, d_gene), "
            f"got shape {tuple(weight.shape)}"
        )
    torch.save(weight, path)


def initialize_random_embeddings(
    n_genes: int,
    d_gene: int,
    seed: Optional[int] = None,
    dtype: torch.dtype = torch.float32,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """
    Initialize a random gene embedding matrix.

    Parameters
    ----------
    n_genes : int
        Number of genes (rows of the embedding matrix).
    d_gene : int
        Embedding dimension (columns of the embedding matrix).
    seed : Optional[int], default None
        Random seed for reproducibility. If None, the current RNG state is used.
    dtype : torch.dtype, default torch.float32
        Data type of the returned tensor.
    device : Optional[torch.device], default None
        Device on which to create the tensor. If None, uses the default device.

    Returns
    -------
    torch.Tensor
        Tensor of shape (n_genes, d_gene) with randomly initialized embeddings.
        Uses a standard normal distribution scaled by 1/sqrt(d_gene).
    """
    if seed is not None:
        gen = torch.Generator(device=device)
        gen.manual_seed(seed)
    else:
        gen = None

    # Standard normal scaled by 1/sqrt(d_gene), a common embedding init scheme
    weight = torch.randn(
        n_genes,
        d_gene,
        generator=gen,
        dtype=dtype,
        device=device,
    )
    weight = weight / (d_gene**0.5)
    return weight
