import torch
import torch.nn as nn
import torch.nn.functional as F

class CBOWCellEncoder(nn.Module):
    """
    VAE-style encoder that:
      - Pools gene embeddings for each cell using expression weights
      - Optionally concatenates a perturbation embedding and library size
      - Outputs mu_z and logvar_z for Gaussian latent z
    """
    def __init__(
        self,
        gene_embedding: GeneEmbedding,
        d_z: int,
        d_hidden: int = 256,
        d_cond: int = 0,
        include_libsize: bool = True,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.gene_embedding = gene_embedding
        self.d_z = d_z
        self.include_libsize = include_libsize
        self.d_cond = d_cond
        
        d_in = gene_embedding.embedding.embedding_dim
        if d_cond > 0:
            d_in += d_cond
        if include_libsize:
            d_in += 1
        
        self.net = nn.Sequential(
            nn.Linear(d_in, d_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_hidden, d_hidden),
            nn.ReLU(),
        )
        self.fc_mu = nn.Linear(d_hidden, d_z)
        self.fc_logvar = nn.Linear(d_hidden, d_z)
    
    def pool_genes(self, x: torch.Tensor,
                   gene_indices: torch.Tensor,
                   eps: float = 1e-8) -> torch.Tensor:
        """
        x: (B, G) expression counts or normalized expression
        gene_indices: (B, G) integer indices (0..n_genes-1)
        returns: h_pool (B, d_gene)
        """
        gene_embs = self.gene_embedding(gene_indices)  # (B, G, d_gene)
        weights = torch.log1p(x)  # (B, G)
        weights = weights / (weights.sum(dim=1, keepdim=True) + eps)
        h_pool = torch.einsum("bg,bgd->bd", weights, gene_embs)
        return h_pool
    
    def forward(
        self,
        x: torch.Tensor,
        gene_indices: torch.Tensor,
        cond_emb: torch.Tensor = None,
        libsize: torch.Tensor = None,
    ):
        """
        x: (B, G) expression
        gene_indices: (B, G)
        cond_emb: (B, d_cond) or None
        libsize: (B,) or None
        returns: mu_z, logvar_z, h_pool
        """
        h_pool = self.pool_genes(x, gene_indices)  # (B, d_gene)
        h = h_pool
        if self.d_cond > 0 and cond_emb is not None:
            h = torch.cat([h, cond_emb], dim=-1)
        if self.include_libsize:
            if libsize is None:
                libsize = x.sum(dim=1)
            log_lib = torch.log1p(libsize).unsqueeze(-1)
            h = torch.cat([h, log_lib], dim=-1)
        
        h_hidden = self.net(h)
        mu_z = self.fc_mu(h_hidden)
        logvar_z = self.fc_logvar(h_hidden)
        return mu_z, logvar_z, h_pool