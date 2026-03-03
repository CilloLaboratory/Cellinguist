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


class FeedForward(nn.Module):
    def __init__(self, d_model: int, ff_mult: int = 4, dropout: float = 0.0) -> None:
        super().__init__()
        d_ff = int(d_model * ff_mult)
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class CrossAttentionBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        ff_mult: int = 4,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.norm_q = nn.LayerNorm(d_model)
        self.norm_kv = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm_ff = nn.LayerNorm(d_model)
        self.ff = FeedForward(d_model=d_model, ff_mult=ff_mult, dropout=dropout)

    def forward(self, latents: torch.Tensor, inputs: torch.Tensor) -> torch.Tensor:
        q = self.norm_q(latents)
        kv = self.norm_kv(inputs)
        attn_out, _ = self.attn(q, kv, kv, need_weights=False)
        latents = latents + attn_out
        latents = latents + self.ff(self.norm_ff(latents))
        return latents


class SelfAttentionBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        ff_mult: int = 4,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.norm_attn = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm_ff = nn.LayerNorm(d_model)
        self.ff = FeedForward(d_model=d_model, ff_mult=ff_mult, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        qkv = self.norm_attn(x)
        attn_out, _ = self.attn(qkv, qkv, qkv, need_weights=False)
        x = x + attn_out
        x = x + self.ff(self.norm_ff(x))
        return x


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
        input_transform: str = "log1p",   # "log1p" or "none"
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

        self.input_transform = input_transform

    def forward(self, x_expr, cond_idx=None):
        B, G = x_expr.shape
        assert G == self.n_genes

        # 1) Check raw input
        if torch.isnan(x_expr).any() or torch.isinf(x_expr).any():
            print("NaNs/Infs in x_expr!")

        # 2) Apply transform for encoder
        if self.input_transform == "log1p":
            x_enc = torch.log1p(x_expr)
        elif self.input_transform == "none":
            x_enc = x_expr
        else:
            raise ValueError(f"Unsupported input_transform: {self.input_transform}")

        if torch.isnan(x_enc).any() or torch.isinf(x_enc).any():
            print("NaNs/Infs after log1p in x_enc")

        # 3) Expression-weighted CBOW
        E = self.gene_embedding.weight  # (G, d_gene)
        if torch.isnan(E).any() or torch.isinf(E).any():
            print("NaNs/Infs in gene_embeddings!")

        h_cell = x_enc @ E  # (B, d_gene)

        if torch.isnan(h_cell).any() or torch.isinf(h_cell).any():
            print("NaNs/Infs after matmul (h_cell)")

        # 4) Optional condition embedding
        if self.cond_embedding is not None and cond_idx is not None:
            c = self.cond_embedding(cond_idx)
            h_in = torch.cat([h_cell, c], dim=-1)
        else:
            h_in = h_cell

        if torch.isnan(h_in).any() or torch.isinf(h_in).any():
            print("NaNs/Infs in h_in BEFORE MLP")

        # 5) Check MLP parameters
        for name, p in self.mlp_mu.named_parameters():
            if torch.isnan(p).any() or torch.isinf(p).any():
                print("NaNs/Infs in mlp_mu parameter:", name)
        for name, p in self.mlp_logvar.named_parameters():
            if torch.isnan(p).any() or torch.isinf(p).any():
                print("NaNs/Infs in mlp_logvar parameter:", name)

        mu = self.mlp_mu(h_in)
        logvar = self.mlp_logvar(h_in)

        if torch.isnan(mu).any() or torch.isnan(logvar).any():
            print("NaNs in mu/logvar AFTER MLP")

        return mu, logvar


class PerceiverCellEncoder(nn.Module):
    """
    Perceiver-style cell encoder:
      1) Builds per-gene input tokens from expression + learned gene-id embeddings.
      2) Uses latent queries with cross-attention to genes.
      3) Applies latent self-attention blocks.
      4) Pools latents and maps to (mu, logvar) for the VAE posterior.
    """

    def __init__(
        self,
        n_genes: int,
        latent_dim: int,
        hidden_dim: int,
        n_hidden_layers: int,
        n_conditions: Optional[int] = None,
        cond_emb_dim: int = 16,
        input_transform: str = "log1p",
        library_norm: str = "size_factor",
        library_norm_target_sum: float = 1e4,
        library_norm_eps: float = 1e-8,
        perceiver_d_model: int = 256,
        perceiver_num_latents: int = 64,
        perceiver_num_cross_attn_heads: int = 8,
        perceiver_num_self_attn_heads: int = 8,
        perceiver_num_self_attn_layers: int = 4,
        perceiver_ff_mult: int = 4,
        perceiver_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.n_genes = int(n_genes)
        self.latent_dim = int(latent_dim)
        self.input_transform = input_transform
        self.library_norm = str(library_norm).lower()
        self.library_norm_target_sum = float(library_norm_target_sum)
        self.library_norm_eps = float(library_norm_eps)
        if self.library_norm not in {"size_factor", "none"}:
            raise ValueError(f"Unsupported library_norm: {library_norm}")
        if self.library_norm_target_sum <= 0:
            raise ValueError("library_norm_target_sum must be > 0.")
        if self.library_norm_eps <= 0:
            raise ValueError("library_norm_eps must be > 0.")

        d_model = int(perceiver_d_model)
        n_latents = int(perceiver_num_latents)

        if d_model % int(perceiver_num_cross_attn_heads) != 0:
            raise ValueError("perceiver_d_model must be divisible by perceiver_num_cross_attn_heads.")
        if d_model % int(perceiver_num_self_attn_heads) != 0:
            raise ValueError("perceiver_d_model must be divisible by perceiver_num_self_attn_heads.")

        self.expr_projection = nn.Sequential(
            nn.Linear(1, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.gene_embedding = nn.Embedding(self.n_genes, d_model)
        self.register_buffer("gene_indices", torch.arange(self.n_genes, dtype=torch.long))

        self.latents = nn.Parameter(torch.randn(n_latents, d_model) * 0.02)
        self.cross_attn = CrossAttentionBlock(
            d_model=d_model,
            n_heads=int(perceiver_num_cross_attn_heads),
            ff_mult=int(perceiver_ff_mult),
            dropout=float(perceiver_dropout),
        )
        self.self_attn_blocks = nn.ModuleList(
            [
                SelfAttentionBlock(
                    d_model=d_model,
                    n_heads=int(perceiver_num_self_attn_heads),
                    ff_mult=int(perceiver_ff_mult),
                    dropout=float(perceiver_dropout),
                )
                for _ in range(int(perceiver_num_self_attn_layers))
            ]
        )
        self.final_norm = nn.LayerNorm(d_model)

        if n_conditions is not None:
            self.cond_embedding = nn.Embedding(n_conditions, cond_emb_dim)
            cond_input_dim = cond_emb_dim
        else:
            self.cond_embedding = None
            cond_input_dim = 0

        encoder_input_dim = d_model + cond_input_dim
        self.mlp_mu = MLP(
            input_dim=encoder_input_dim,
            output_dim=self.latent_dim,
            hidden_dim=hidden_dim,
            n_hidden_layers=n_hidden_layers,
        )
        self.mlp_logvar = MLP(
            input_dim=encoder_input_dim,
            output_dim=self.latent_dim,
            hidden_dim=hidden_dim,
            n_hidden_layers=n_hidden_layers,
        )

    def forward(
        self,
        x_expr: torch.Tensor,
        cond_idx: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        bsz, n_genes = x_expr.shape
        if n_genes != self.n_genes:
            raise ValueError(
                f"Expected {self.n_genes} genes, but got {n_genes}. "
                "Dataset gene order/shape does not match encoder setup."
            )

        if self.library_norm == "size_factor":
            libsize = x_expr.sum(dim=1, keepdim=True).clamp_min(self.library_norm_eps)
            x_norm = x_expr * (self.library_norm_target_sum / libsize)
            x_enc = torch.log1p(x_norm)
        else:
            if self.input_transform == "log1p":
                x_enc = torch.log1p(x_expr)
            elif self.input_transform == "none":
                x_enc = x_expr
            else:
                raise ValueError(f"Unsupported input_transform: {self.input_transform}")

        expr_tokens = self.expr_projection(x_enc.unsqueeze(-1))  # (B, G, D)
        gene_tokens = self.gene_embedding(self.gene_indices).unsqueeze(0)  # (1, G, D)
        inputs = expr_tokens + gene_tokens

        latents = self.latents.unsqueeze(0).expand(bsz, -1, -1)  # (B, L, D)
        latents = self.cross_attn(latents, inputs)
        for block in self.self_attn_blocks:
            latents = block(latents)

        h_cell = self.final_norm(latents.mean(dim=1))  # (B, D)

        if self.cond_embedding is not None and cond_idx is not None:
            c = self.cond_embedding(cond_idx)
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
        encoder: nn.Module,
        decoder: nn.Module,
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
# ZINB decoder
# ---------------------------------------------------------------------------

class ZINBExpressionDecoder(nn.Module):
    """
    ZINB decoder: given z (+ optional cond), outputs
      - mu    : mean counts per gene, shape (B, G), > 0
      - theta : inverse dispersion per gene, shape (B, G) or (1, G), > 0
      - pi    : dropout probability per gene, shape (B, G), in (0, 1)

    For simplicity, we:
      - predict mu and pi via MLP,
      - keep theta as a gene-wise parameter (broadcast across batch).
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

        # Optional condition embedding
        if n_conditions is not None:
            self.cond_embedding = nn.Embedding(n_conditions, cond_emb_dim)
            cond_input_dim = cond_emb_dim
        else:
            self.cond_embedding = None
            cond_input_dim = 0

        decoder_input_dim = latent_dim + cond_input_dim

        # MLP for mu (pre-activation)
        self.mlp_mu = MLP(
            input_dim=decoder_input_dim,
            output_dim=n_genes,
            hidden_dim=hidden_dim,
            n_hidden_layers=n_hidden_layers,
        )

        # MLP for pi (dropout logit)
        self.mlp_pi = MLP(
            input_dim=decoder_input_dim,
            output_dim=n_genes,
            hidden_dim=hidden_dim,
            n_hidden_layers=n_hidden_layers,
        )

        # Gene-wise inverse dispersion parameter (learned)
        # We'll apply softplus to ensure positivity.
        self.log_theta = nn.Parameter(torch.zeros(n_genes))

    def forward(
        self,
        z: torch.Tensor,
        cond_idx: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        z : (B, latent_dim)
        cond_idx : Optional[(B,) LongTensor]

        Returns
        -------
        mu : (B, G)  > 0
        theta : (B, G)  > 0
        pi : (B, G)  in (0, 1)
        """
        if self.cond_embedding is not None and cond_idx is not None:
            c = self.cond_embedding(cond_idx)
            h_in = torch.cat([z, c], dim=-1)
        else:
            h_in = z

        mu_logit = self.mlp_mu(h_in)   # (B, G)
        pi_logit = self.mlp_pi(h_in)   # (B, G)

        # Ensure positivity
        mu = F.softplus(mu_logit) + 1e-8          # mean counts
        theta = F.softplus(self.log_theta) + 1e-8  # (G,)
        theta = theta.unsqueeze(0).expand_as(mu)   # (B, G)

        # Dropout probability
        pi = torch.sigmoid(pi_logit)              # (B, G)

        return mu, theta, pi

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

def zinb_negative_log_likelihood(
    x: torch.Tensor,
    mu: torch.Tensor,
    theta: torch.Tensor,
    pi: torch.Tensor,
    eps: float = 1e-8,
    reduction: str = "mean",
) -> torch.Tensor:
    """
    ZINB negative log-likelihood per cell:

    x    : observed counts,   (B, G)
    mu   : mean parameter,    (B, G)
    theta: inv. dispersion,   (B, G)
    pi   : dropout prob,      (B, G)
    """
    # x, mu, theta, pi must be non-negative / bounded appropriately
    # Ensure numerical stability
    mu = mu.clamp(min=eps)
    theta = theta.clamp(min=eps)
    pi = pi.clamp(min=eps, max=1 - eps)
    x = x.clamp(min=0.0)

    # log NB pmf
    # lgamma(theta + x) - lgamma(theta) - lgamma(x + 1)
    t1 = torch.lgamma(theta + x) - torch.lgamma(theta) - torch.lgamma(x + 1.0)

    log_theta = torch.log(theta + eps)
    log_mu = torch.log(mu + eps)
    log_theta_mu = torch.log(theta + mu + eps)

    t2 = theta * (log_theta - log_theta_mu)
    t3 = x * (log_mu - log_theta_mu)
    log_nb = t1 + t2 + t3            # log NB(x | mu, theta)

    # NB probability of zero
    # when x = 0, NB(0) = (theta / (theta + mu))^theta
    log_nb_zero = theta * (log_theta - log_theta_mu)

    # Mix with zero-inflation
    # For x == 0:
    #   log p(x=0) = log( pi + (1 - pi) * exp(log_nb_zero) )
    # For x > 0:
    #   log p(x)   = log(1 - pi) + log_nb
    is_zero = (x < eps)

    log_prob_zero = torch.log(
        pi + (1.0 - pi) * torch.exp(log_nb_zero) + eps
    )

    log_prob_nonzero = torch.log(1.0 - pi + eps) + log_nb

    log_prob = torch.where(is_zero, log_prob_zero, log_prob_nonzero)

    nll = -log_prob.sum(dim=-1)  # sum over genes -> (B,)

    if reduction == "mean":
        return nll.mean()
    elif reduction == "sum":
        return nll.sum()
    elif reduction == "none":
        return nll
    else:
        raise ValueError(f"Unsupported reduction: {reduction}")
