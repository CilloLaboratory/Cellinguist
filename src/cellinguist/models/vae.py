from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Helper MLP
# ---------------------------------------------------------------------------

class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int,
        n_hidden_layers: int,
        activation: nn.Module | None = None,
    ) -> None:
        super().__init__()
        if activation is None:
            activation = nn.ReLU()

        layers = []
        in_dim = input_dim
        for _ in range(n_hidden_layers):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(activation)
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ---------------------------------------------------------------------------
# CBOWCellEncoder: expression-weighted CBOW + MLP to (mu, logvar)
# ---------------------------------------------------------------------------

class CBOWCellEncoder(nn.Module):
    """
    Cell encoder that:
      1) Uses expression-weighted CBOW to get a cell-level representation:
         h_cell = X @ E, where X is (B, G) expression and E is (G, d_gene_emb).
      2) Optionally concatenates a condition embedding.
      3) Passes through an MLP to produce (mu, logvar) for a VAE.

    X should be some transformed expression (e.g. log1p-normalized counts),
    and the decoder's reconstruction target should match that transform.
    """

    def __init__(
        self,
        gene_embeddings: torch.Tensor,
        latent_dim: int,
        hidden_dim: int,
        n_hidden_layers: int,
        n_conditions: Optional[int] = None,
        cond_emb_dim: int = 16,
        freeze_gene_embeddings: bool = True,
    ) -> None:
        super().__init__()

        # gene_embeddings: (n_genes, d_gene)
        n_genes, d_gene = gene_embeddings.shape
        self.n_genes = n_genes
        self.d_gene = d_gene
        self.latent_dim = latent_dim

        # Wrap CBOW gene embeddings in an Embedding-like module
        # We'll use matmul directly, but keep as Parameter/Buffer for convenience.
        self.gene_embedding = nn.Embedding(
            num_embeddings=n_genes,
            embedding_dim=d_gene,
        )
        self.gene_embedding.weight.data.copy_(gene_embeddings)
        if freeze_gene_embeddings:
            self.gene_embedding.weight.requires_grad_(False)

        # Condition embedding (optional)
        if n_conditions is not None:
            self.cond_embedding = nn.Embedding(n_conditions, cond_emb_dim)
            cond_input_dim = cond_emb_dim
        else:
            self.cond_embedding = None
            cond_input_dim = 0

        encoder_input_dim = d_gene + cond_input_dim

        # MLP to latent parameters
        self.mlp_mu = MLP(
            input_dim=encoder_input_dim,
            output_dim=latent_dim,
            hidden_dim=hidden_dim,
            n_hidden_layers=n_hidden_layers,
        )
        self.mlp_logvar = MLP(
            input_dim=encoder_input_dim,
            output_dim=latent_dim,
            hidden_dim=hidden_dim,
            n_hidden_layers=n_hidden_layers,
        )

    def forward(
        self,
        x_expr: torch.Tensor,
        cond_idx: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        x_expr : (B, G) tensor
            Expression matrix (already on device, same order of genes
            as gene_embeddings). Typically log-normalized.
        cond_idx : Optional[(B,) LongTensor]
            Optional condition indices per cell.

        Returns
        -------
        mu : (B, latent_dim)
        logvar : (B, latent_dim)
        """
        B, G = x_expr.shape
        assert G == self.n_genes, (
            f"Input has {G} genes, but encoder expects {self.n_genes}"
        )

        # Expression-weighted CBOW: X @ E
        # E: (G, d_gene)
        E = self.gene_embedding.weight  # (G, d_gene)
        h_cell = x_expr @ E             # (B, d_gene)

        # Optional condition embedding
        if self.cond_embedding is not None and cond_idx is not None:
            c = self.cond_embedding(cond_idx)  # (B, cond_emb_dim)
            h_in = torch.cat([h_cell, c], dim=-1)
        else:
            h_in = h_cell

        mu = self.mlp_mu(h_in)
        logvar = self.mlp_logvar(h_in)
        return mu, logvar


# ---------------------------------------------------------------------------
# ExpressionDecoder: z (+ cond) -> reconstructed expression
# ---------------------------------------------------------------------------

class ExpressionDecoder(nn.Module):
    """
    Simple MLP decoder for expression.

    Given latent z (and optional condition embedding), outputs a reconstruction
    of x_expr. Here we use a Gaussian/MSE-style decoder, so we just output
    a single (B, G) matrix of reconstructed expression.

    If you'd like NB/ZINB later, you can change this to output mu/theta(/pi).
    """

    def __init__(
        self,
        n_genes: int,
        latent_dim: int,
        hidden_dim: int,
        n_hidden_layers: int,
        n_conditions: Optional[int] = None,
        cond_emb_dim: int = 16,
    ) -> None:
        super().__init__()
        self.n_genes = n_genes

        if n_conditions is not None:
            self.cond_embedding = nn.Embedding(n_conditions, cond_emb_dim)
            cond_input_dim = cond_emb_dim
        else:
            self.cond_embedding = None
            cond_input_dim = 0

        decoder_input_dim = latent_dim + cond_input_dim

        self.mlp = MLP(
            input_dim=decoder_input_dim,
            output_dim=n_genes,
            hidden_dim=hidden_dim,
            n_hidden_layers=n_hidden_layers,
        )

    def forward(
        self,
        z: torch.Tensor,
        cond_idx: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        z : (B, latent_dim)
        cond_idx : Optional[(B,) LongTensor]

        Returns
        -------
        recon_x : (B, n_genes)
            Reconstructed expression (same transform as x_expr).
        """
        if self.cond_embedding is not None and cond_idx is not None:
            c = self.cond_embedding(cond_idx)
            h_in = torch.cat([z, c], dim=-1)
        else:
            h_in = z

        recon_x = self.mlp(h_in)
        return recon_x


# ---------------------------------------------------------------------------
# GeneVAE wrapper
# ---------------------------------------------------------------------------

class GeneVAE(nn.Module):
    """
    VAE that uses CBOWCellEncoder + ExpressionDecoder.

    Forward:
        - encode x_expr (+ cond) -> (mu, logvar)
        - reparameterize -> z
        - decode z (+ cond) -> recon_x
    """

    def __init__(
        self,
        encoder: CBOWCellEncoder,
        decoder: ExpressionDecoder,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def encode(
        self,
        x_expr: torch.Tensor,
        cond_idx: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.encoder(x_expr, cond_idx)

    @staticmethod
    def reparameterize(
        mu: torch.Tensor,
        logvar: torch.Tensor,
    ) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(
        self,
        z: torch.Tensor,
        cond_idx: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return self.decoder(z, cond_idx)

    def forward(
        self,
        x_expr: torch.Tensor,
        cond_idx: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns
        -------
        recon_x : (B, n_genes)
        mu : (B, latent_dim)
        logvar : (B, latent_dim)
        """
        mu, logvar = self.encode(x_expr, cond_idx)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z, cond_idx)
        return recon_x, mu, logvar


# ---------------------------------------------------------------------------
# Loss helpers
# ---------------------------------------------------------------------------

def kl_divergence_normal(
    mu: torch.Tensor,
    logvar: torch.Tensor,
    reduction: str = "mean",
) -> torch.Tensor:
    """
    KL(q(z|x) || p(z)) for diagonal Gaussian, p(z)=N(0, I).

    KL = -0.5 * sum(1 + logvar - mu^2 - exp(logvar))
    """
    # (B, D)
    kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
    # (B,)
    kl = kl.sum(dim=-1)

    if reduction == "mean":
        return kl.mean()
    elif reduction == "sum":
        return kl.sum()
    elif reduction == "none":
        return kl
    else:
        raise ValueError(f"Unsupported reduction: {reduction}")


def gaussian_reconstruction_loss(
    recon_x: torch.Tensor,
    x_true: torch.Tensor,
    reduction: str = "mean",
) -> torch.Tensor:
    """
    Simple MSE loss between reconstructed and true expression.

    recon_x and x_true should be on the same scale (e.g. log1p normalized).
    """
    mse = F.mse_loss(recon_x, x_true, reduction="none")  # (B, G)
    mse = mse.sum(dim=-1)  # (B,)
    if reduction == "mean":
        return mse.mean()
    elif reduction == "sum":
        return mse.sum()
    elif reduction == "none":
        return mse
    else:
        raise ValueError(f"Unsupported reduction: {reduction}")
