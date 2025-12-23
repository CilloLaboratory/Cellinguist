from torch.utils.data import DataLoader
import torch
import pandas as pd
import anndata as ad
import numpy as np
import gzip

from cellinguist.config import VAEConfig
from cellinguist.data.datasets import SingleCellVAEDataset
from cellinguist.embeddings import load_gene_embeddings
from cellinguist.models.vae import (
    CBOWCellEncoder,
    ZINBExpressionDecoder,
    GeneVAE,
    zinb_negative_log_likelihood,
    kl_divergence_normal,
)

# 1. Load gene-level CBOW embeddings from TSV
gene_emb_tsv = "/home/arc85/Desktop/cellinguist_results_251125/01_cytokine_testing_251125/03_output/cytokines_test_cbow_full_genome_gene_embeddings_251125.tsv.gz"
df_emb = pd.read_csv(gene_emb_tsv,sep="\t")

genes_from_emb = df_emb["gene"].astype(str).tolist()
emb_matrix = df_emb.drop(columns=["gene"]).to_numpy().astype("float32")
gene_embeddings_full = torch.from_numpy(emb_matrix)
n_genes, d_gene = gene_embeddings_full.shape
print(f"Loaded gene embeddings for {n_genes} genes, dim={d_gene}")

# 2. Load data
adata_path = "/home/arc85/Desktop/Cellinguist/cytokine_dict_ser_sub_full_genes_ad_251118.h5ad"
gene_key = "gene" 
adata = ad.read_h5ad(adata_path)
dataset_full = SingleCellVAEDataset(
    adata_or_path=adata[0:1000,:],
    gene_key=gene_key,
    layer=None,
    cond_key=None,       # or e.g. "condition"
    transform="none",
)
genes_expr = dataset_full.gene_order

print(f"Expression dataset: {len(genes_expr)} genes before intersection")

# 3. Computate intersection and alignment
expr_gene_set = set(genes_expr)

# genes_common in the order of the embeddings
genes_common = [g for g in genes_from_emb if g in expr_gene_set]

if len(genes_common) == 0:
    raise ValueError("No overlapping genes between embeddings and expression data!")

print(f"Using {len(genes_common)} genes in common between embeddings and expression")

# Build mask for embeddings
emb_index_map = {g: i for i, g in enumerate(genes_from_emb)}
emb_indices = [emb_index_map[g] for g in genes_common]
gene_embeddings = gene_embeddings_full[emb_indices, :]   # (N_common, d_gene)

# 4. Rebuild VAE dataset with aligned gene order
del(dataset_full)

vae_dataset = SingleCellVAEDataset(
    adata_or_path=adata_path,
    gene_key=gene_key,
    layer=None,
    cond_key=None,            # or some condition column
    gene_order=genes_common,  # <--- align expression to embeddings
    transform="none", # <--- none for ZINB model, log-transform inside encoder
)

n_cells, n_genes = vae_dataset.X.shape
print(f"VAE dataset after alignment: {n_cells} cells, {n_genes} genes")
assert n_genes == gene_embeddings.shape[0], (
    f"Dataset has {n_genes} genes, embeddings have {gene_embeddings.shape[0]}"
)

# 5. Build model and training

cfg = VAEConfig(
    n_genes=n_genes,
    latent_dim=32,
    hidden_dim=256,
    n_hidden_layers=2,
    n_conditions=None,   # or len(dataset.cond_categories)
    cond_emb_dim=16,
    kl_weight=1.0,
    device="cuda",
    epochs=50
)

device = torch.device(cfg.device)

encoder = CBOWCellEncoder(
    gene_embeddings=gene_embeddings,
    latent_dim=cfg.latent_dim,
    hidden_dim=cfg.hidden_dim,
    n_hidden_layers=cfg.n_hidden_layers,
    n_conditions=cfg.n_conditions,
    cond_emb_dim=cfg.cond_emb_dim,
    freeze_gene_embeddings=True,
    input_transform="log1p"
)

decoder = ZINBExpressionDecoder(
    n_genes=n_genes,
    latent_dim=cfg.latent_dim,
    hidden_dim=cfg.hidden_dim,
    n_hidden_layers=cfg.n_hidden_layers,
    n_conditions=cfg.n_conditions,
    cond_emb_dim=cfg.cond_emb_dim,
)

model = GeneVAE(encoder, decoder).to(device)

optimizer = torch.optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=cfg.lr,
    weight_decay=cfg.weight_decay,
)

dataloader = DataLoader(
    vae_dataset,
    batch_size=cfg.batch_size,
    shuffle=True,
    num_workers=4,
    pin_memory=(device.type == "cuda"),
)

# 6. Train a few epochs
model.train()
for epoch in range(cfg.epochs):
    total_loss = 0.0
    total_batches = 0

    for batch in dataloader:
        # x_expr is raw counts now
        x_expr = batch["x_expr"].to(device, non_blocking=True)  # (B, G)
        cond_idx = batch.get("cond_idx", None)
        if cond_idx is not None:
            cond_idx = cond_idx.to(device, non_blocking=True)

        # Encode
        mu_z, logvar_z = model.encode(x_expr, cond_idx)
        # Reparameterize
        z = GeneVAE.reparameterize(mu_z, logvar_z)
        # Decode to ZINB params
        mu, theta, pi = decoder(z, cond_idx)

        # Reconstruction loss in count space
        recon_loss = zinb_negative_log_likelihood(
            x_expr,
            mu,
            theta,
            pi,
            reduction="mean",
        )

        kl = kl_divergence_normal(mu_z, logvar_z, reduction="mean")

        loss = recon_loss + cfg.kl_weight * kl

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += float(loss.item())
        total_batches += 1

    avg_loss = total_loss / max(total_batches, 1)
    print(f"[VAE-ZINB] Epoch {epoch+1}/{cfg.epochs} - loss: {avg_loss:.4f}")

# 7. Extract predicted versus actual counts
model.eval()

cell_indices = range(10000)

all_true_counts = []
all_pred_counts = []
all_cell_ids = []

with torch.no_grad():
    for idx in cell_indices:
        sample = vae_dataset[idx]
        x_expr = sample["x_expr"].unsqueeze(0).to(device)  # (1, G)
        cond_idx = sample.get("cond_idx", None)
        if cond_idx is not None:
            cond_idx = cond_idx.unsqueeze(0).to(device)

        # Using mu directly for deterministic z
        mu_z, logvar_z = model.encode(x_expr, cond_idx)
        recon_mu, recon_theta, recon_pi = decoder(mu_z, cond_idx)  # use mu_z (no sampling)

        # recon_mu is your "predicted mean count" per gene
        pred_counts = recon_mu.cpu().numpy()
        true_counts = x_expr.cpu().numpy()

        # To CPU numpy
        all_true_counts.append(true_counts.squeeze(0))
        all_pred_counts.append(pred_counts.squeeze(0))

        # Optionally track a cell ID (barcode or obs_names)
        cell_id = vae_dataset.adata.obs_names[idx]
        all_cell_ids.append(cell_id)

# Stack into arrays
all_true_counts = np.stack(all_true_counts, axis=0)  # (N_cells, G)
all_pred_counts = np.stack(all_pred_counts, axis=0)  # (N_cells, G)

genes = vae_dataset.gene_order  # genes_common, aligned with both arrays

all_true_counts = pd.DataFrame(all_true_counts, index=all_cell_ids, columns=genes)
all_pred_counts = pd.DataFrame(all_pred_counts, index=all_cell_ids, columns=genes)

out_path_true = "/home/arc85/Desktop/cellinguist_results_251125/01_cytokine_testing_251125/03_output/cytokines_test_true_counts_1000_cells_251204.tsv.gz"
out_path_pred = "/home/arc85/Desktop/cellinguist_results_251125/01_cytokine_testing_251125/03_output/cytokines_test_pred_counts_1000_cells_251204.tsv.gz"

with gzip.open(out_path_true, "wt") as f:
    all_true_counts.to_csv(f, sep="\t", index=False)

with gzip.open(out_path_pred, "wt") as f:
    all_pred_counts.to_csv(f, sep="\t", index=False)