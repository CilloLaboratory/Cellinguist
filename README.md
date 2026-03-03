# Cellinguist
Language of the cell

*This is a work in progress*

## Pretraining
Currently, we can pretrain on an anndata formatted datasets using the following incantation:

``` 
torchrun --nproc-per-node=2 training.py \
    --input_anndata=HD_PBMC_5prime_3prime_1k_hvf_250508.h5ad \
    --domain_for_grad_rev=chemistry \
    --out_model=/HD_PBMC_5prime_3prime_1k_hvf_model.pth
```

For the VAE trainer, use:

```
train-vae --config src/cellinguist/configs/vae_train.yml
```

Multi-GPU (single node):

```
torchrun --standalone --nproc-per-node=4 -m cellinguist.train.train_vae --config src/cellinguist/configs/vae_train.yml
```

Perceiver-native embedding exports from VAE checkpoints:

```
export-perceiver-cell-embeddings --adata /path/to/data.h5ad --checkpoint /path/to/vae_last.ckpt --out /path/to/perceiver_cell_embeddings.tsv.gz
export-perceiver-gene-embeddings --checkpoint /path/to/vae_last.ckpt --out /path/to/perceiver_gene_embeddings.tsv.gz
```

CBOW/MLP VAE cell embedding export:

```
export-cbow-vae-cell-embeddings --adata /path/to/data.h5ad --checkpoint /path/to/vae_last.ckpt --gene-emb-tsv /path/to/gene_embeddings.tsv.gz --out /path/to/cbow_vae_cell_embeddings.tsv.gz
```
