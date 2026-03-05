import torch
import torch.nn as nn
import torch.nn.functional as F

class NBDecoder(nn.Module):
    """
    Decoder that maps conditioned latent representation to NB (or ZINB) parameters.
    Optionally uses gene embeddings to implement a low-rank factorization.
    """
    def __init__(
        self,
        d_in: int,
        n_genes: int,
        d_hidden: int = 256,
        use_low_rank: bool = False,
        gene_embedding: GeneEmbedding = None,
        d_dec: int = 64,
        use_zinb: bool = False,
    ):
        super().__init__()
        self.n_genes = n_genes
        self.use_low_rank = use_low_rank
        self.use_zinb = use_zinb
        self.gene_embedding = gene_embedding
        
        if use_low_rank:
            if gene_embedding is None:
                raise ValueError(
                    "gene_embedding must be provided when use_low_rank=True"
                )
            d_gene = gene_embedding.embedding.embedding_dim
            self.net = nn.Sequential(
                nn.Linear(d_in, d_hidden),
                nn.ReLU(),
                nn.Linear(d_hidden, d_dec),
                nn.ReLU(),
            )
            # Project from (d_dec) to gene-embedding space
            self.W_mu = nn.Linear(d_dec, d_gene, bias=False)
            if use_zinb:
                self.W_pi = nn.Linear(d_dec, d_gene, bias=False)
        else:
            self.net = nn.Sequential(
                nn.Linear(d_in, d_hidden),
                nn.ReLU(),
                nn.Linear(d_hidden, d_hidden),
                nn.ReLU(),
            )
            self.fc_mu = nn.Linear(d_hidden, n_genes)
            if use_zinb:
                self.fc_pi = nn.Linear(d_hidden, n_genes)
        
        # Gene-wise dispersion (theta) parameter, shared across cells
        self.log_theta = nn.Parameter(torch.zeros(n_genes))
    
    def forward(self, h: torch.Tensor,
                libsize: torch.Tensor = None):
        """
        h: (B, d_in) conditioned latent representation
        libsize: (B,) or None
        returns:
          mu: (B, n_genes) mean parameter
          theta: (B, n_genes) dispersion parameter
          pi: (B, n_genes) dropout probability (if use_zinb) or None
        """
        if self.use_low_rank:
            B = h.size(0)
            h_dec = self.net(h)  # (B, d_dec)
            # Map to gene-embedding space
            h_proj = self.W_mu(h_dec)  # (B, d_gene)
            gene_embs = self.gene_embedding.embedding.weight  # (n_genes, d_gene)
            log_mu = torch.matmul(h_proj, gene_embs.t())  # (B, n_genes)
            if self.use_zinb:
                h_proj_pi = self.W_pi(h_dec)
                logit_pi = torch.matmul(h_proj_pi, gene_embs.t())
        else:
            h_dec = self.net(h)
            log_mu = self.fc_mu(h_dec)
            if self.use_zinb:
                logit_pi = self.fc_pi(h_dec)
        
        mu = torch.exp(log_mu)
        
        if libsize is not None:
            libsize = libsize.unsqueeze(-1)  # (B, 1)
            mu = mu * (libsize / (mu.sum(dim=1, keepdim=True) + 1e-8))
        
        theta = torch.exp(self.log_theta).unsqueeze(0).expand_as(mu)
        pi = None
        if self.use_zinb:
            pi = torch.sigmoid(logit_pi)
        return mu, theta, pi
