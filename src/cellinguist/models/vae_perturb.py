import torch
import torch.nn as nn
import torch.nn.functional as F

class CBOWVAEPerturbModel(nn.Module):
    """
    Full model that combines:
      - Gene embeddings from CBOW
      - VAE encoder
      - Latent conditioner with perturbation embeddings
      - NB/ZINB decoder
    """
    def __init__(
        self,
        n_genes: int,
        d_gene: int,
        d_z: int,
        n_conds: int,
        d_cond: int = 32,
        d_hidden_enc: int = 256,
        d_hidden_dec: int = 256,
        use_low_rank_dec: bool = True,
        d_dec: int = 64,
        use_zinb: bool = False,
        pretrained_gene_weight: torch.Tensor = None,
        freeze_gene_emb: bool = True,
        latent_mode: str = "concat",
    ):
        super().__init__()
        self.gene_embedding = GeneEmbedding(
            n_genes=n_genes,
            d_gene=d_gene,
            pretrained_weight=pretrained_gene_weight,
            freeze=freeze_gene_emb,
        )
        self.perturb_embedding = nn.Embedding(n_conds, d_cond)
        self.encoder = CBOWCellEncoder(
            gene_embedding=self.gene_embedding,
            d_z=d_z,
            d_hidden=d_hidden_enc,
            d_cond=0,  # encoder does not need cond by default; set >0 if you want that
            include_libsize=True,
        )
        # Latent conditioner uses perturbation embedding
        self.latent_conditioner = LatentConditioner(
            d_z=d_z,
            d_cond=d_cond,
            mode=latent_mode,
            d_hidden=d_hidden_enc,
        )
        d_in_dec = self.latent_conditioner.out_dim
        self.decoder = NBDecoder(
            d_in=d_in_dec,
            n_genes=n_genes,
            d_hidden=d_hidden_dec,
            use_low_rank=use_low_rank_dec,
            gene_embedding=self.gene_embedding,
            d_dec=d_dec,
            use_zinb=use_zinb,
        )
    
    def reparameterize(self, mu_z: torch.Tensor,
                       logvar_z: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar_z)
        eps = torch.randn_like(std)
        return mu_z + eps * std
    
    def forward(
        self,
        x: torch.Tensor,
        gene_indices: torch.Tensor,
        cond_idx: torch.Tensor,
        libsize: torch.Tensor = None,
        sample_z: bool = True,
    ):
        """
        x: (B, G) expression
        gene_indices: (B, G) integer indices for each gene position
        cond_idx: (B,) integer condition indices
        libsize: (B,) or None
        returns dict with:
          mu: (B, G), theta: (B, G), pi: (B, G) or None
          mu_z, logvar_z, z, h_pool
        """
        mu_z, logvar_z, h_pool = self.encoder(
            x, gene_indices, cond_emb=None, libsize=libsize
        )
        
        z = self.reparameterize(mu_z, logvar_z) if sample_z else mu_z
        
        cond_emb = self.perturb_embedding(cond_idx)  # (B, d_cond)
        z_cond = self.latent_conditioner(z, cond_emb)  # (B, d_in_dec)
        
        if libsize is None:
            libsize = x.sum(dim=1)
        
        mu, theta, pi = self.decoder(z_cond, libsize=libsize)
        
        return {
            "mu": mu,
            "theta": theta,
            "pi": pi,
            "mu_z": mu_z,
            "logvar_z": logvar_z,
            "z": z,
            "h_pool": h_pool,
        }
