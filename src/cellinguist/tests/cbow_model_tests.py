import torch
from torch.utils.data import DataLoader
from cellinguist.data.datasets import SingleCellDataset, CBOWPairsDataset, CBOWPairsConfig
from cellinguist.models.cbow import (
    CBOWModel,
    CBOWTrainer,
    cbow_negative_sampling_loss
)

# /home/arc85/Desktop/Cellinguist/cytokine_dictionary_2k_pbs_ifng_251108.h5ad
# /home/arc85/Desktop/Cellinguist/cytokine_dict_ser_sub_full_genes_ad_251118.h5ad

# 1. Build SingleCellDataset
ds = SingleCellDataset(
    "/home/arc85/Desktop/Cellinguist/cytokine_dict_ser_sub_full_genes_ad_251118.h5ad",
    gene_key="gene",
    cond_key=None,     # or None
    n_bins=20,
    shuffle_tokens=True,
    min_expr=0.0,
)

# 2. Wrap in CBOWPairsDataser
pairs_cfg = CBOWPairsConfig(
    window_size=5,
    num_negatives=10,
    samples_per_cell=2,   # e.g., 2 random windows per cell per "epoch"
)
cbow_pairs_ds = CBOWPairsDataset(ds, pairs_cfg)

dl = DataLoader(
    cbow_pairs_ds,
    batch_size=512,
    shuffle=False,   # dataset already randomizes; shuffle not essential
    num_workers=0,
)

# 3. Model + trainer
model = CBOWModel(vocab_size=ds.vocab_size, emb_dim=128)
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# 4. Tiny test epoch
for batch in dl:
    target_ids = batch["target_ids"].to(device)       # (B,)
    context_ids = batch["context_ids"].to(device)     # (B, 2w)
    negative_ids = batch["negative_ids"].to(device)   # (B, K)

    pos_logits, neg_logits = model(target_ids, context_ids, negative_ids)
    loss = cbow_negative_sampling_loss(pos_logits, neg_logits)
    opt.zero_grad()
    loss.backward()
    opt.step()
    print("Batch loss:", float(loss.item()))
    break  # just a sanity check

## Stop here

trainer.save_embeddings("/home/arc85/Desktop/Cellinguist/gene_token_embeddings_251121.pt")

# 5. Extract model weights (token embeddings) and write to gzipped tsv
import pandas as pd
import gzip

weight = model.input_emb.weight.detach().cpu()  # (vocab_size, emb_dim)
id_to_token = ds.id_to_token  # {token_id: "GENE__BIN"}

# Convert to numpy
emb_np = weight.numpy()  # (vocab_size, emb_dim)

# Build a list of tokens in index order
tokens = [id_to_token[i] for i in range(emb_np.shape[0])]

# Build DataFrame
df_token = pd.DataFrame(
    emb_np,
    columns=[f"dim_{i+1}" for i in range(emb_np.shape[1])],
)
df_token.insert(0, "token", tokens)

# Write gzipped TSV
out_path = "/home/arc85/Desktop/Cellinguist/cbow_token_embeddings_251121.tsv.gz"
with gzip.open(out_path, "wt") as f:
    df_token.to_csv(f, sep="\t", index=False)

print(f"Saved token embeddings to {out_path}")

# 6. Extract gene embeddings (mean token embeddings for each gene)
import numpy as np
import pandas as pd
import gzip
from collections import defaultdict

emb_np = weight.numpy()
tokens = [id_to_token[i] for i in range(emb_np.shape[0])]

# Collect embeddings per gene
gene_to_vecs = defaultdict(list)

for idx, tok in enumerate(tokens):
    # Token format: "GENE__BIN"
    if "__" in tok:
        gene, bin_str = tok.split("__", 1)
    else:
        gene = tok  # fallback if no bin
    gene_to_vecs[gene].append(emb_np[idx])

# Average vectors per gene
genes = []
gene_embs = []

for gene, vecs in gene_to_vecs.items():
    vecs_arr = np.stack(vecs, axis=0)           # (n_bins_for_gene, emb_dim)
    mean_vec = vecs_arr.mean(axis=0)           # (emb_dim,)
    genes.append(gene)
    gene_embs.append(mean_vec)

gene_embs = np.stack(gene_embs, axis=0)        # (n_genes, emb_dim)

df_gene = pd.DataFrame(
    gene_embs,
    columns=[f"dim_{i+1}" for i in range(gene_embs.shape[1])],
)
df_gene.insert(0, "gene", genes)

out_path_gene = "/home/arc85/Desktop/Cellinguist/cbow_gene_embeddings_251121.tsv.gz"
with gzip.open(out_path_gene, "wt") as f:
    df_gene.to_csv(f, sep="\t", index=False)

print(f"Saved gene embeddings to {out_path_gene}")
