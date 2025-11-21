import torch
from torch.utils.data import DataLoader
from cellinguist.data.datasets import SingleCellDataset
from cellinguist.models.cbow import (
    CBOWModel,
    CBOWTrainer,
    CBOWTrainerConfig,
    build_cbow_training_pairs,
)

# 1. Build SingleCellDataset
ds = SingleCellDataset(
    "/home/arc85/Desktop/Cellinguist/cytokine_dictionary_2k_pbs_ifng_251108.h5ad",
    gene_key="gene",
    cond_key=None,     # or None
    n_bins=20,
    shuffle_tokens=True,
    min_expr=0.0,
)
token_id_seqs = ds._token_id_seqs  # or expose via a property

# 2. Build CBOW triples
target_ids, context_ids, negative_ids = build_cbow_training_pairs(
    token_id_seqs=token_id_seqs,
    vocab_size=ds.vocab_size,
    window_size=5,
    num_negatives=10,
)

# 3. Wrap in a TensorDataset/DataLoader for training
cbow_dataset = torch.utils.data.TensorDataset(target_ids, context_ids, negative_ids)
cbow_loader = DataLoader(cbow_dataset, batch_size=1024, shuffle=True)

# 4. Model + trainer
model = CBOWModel(vocab_size=ds.vocab_size, emb_dim=128)
config = CBOWTrainerConfig(lr=1e-3, epochs=5, device="cuda")
trainer = CBOWTrainer(model, config)

for epoch in range(config.epochs):
    loss = trainer.train_epoch(
        {"target_ids": b[0], "context_ids": b[1], "negative_ids": b[2]}
        for b in cbow_loader
    )
    print(f"Epoch {epoch}: loss={loss:.4f}")

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
