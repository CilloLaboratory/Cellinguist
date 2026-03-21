# Cellinguist
Language of the cell.

Using language inspired deep learning approaches to model cells from single-cell genomic data.

## Installation
Cellinguist can be installed by cloning this repo from github and installing locally into a conda or virtual environment:

```
git clone https://github.com/CilloLaboratory/Cellinguist.git
cd Cellinguist
pip install -e .
```

## Cellinguist workflow
Training of a base model is VAE-based with configurable encoder backbones:
- Transformer encoder (default): gene-id + log1p(expression) token MLP, CLS pooling
- Perceiver encoder (legacy-compatible)
- CBOW encoder with external gene embeddings (legacy-compatible)

The following functionality is currently implemented:
- Learning gene embeddings for co-expression patterns
- Learning cell embeddings from expression profiles
- Whole cell transcriptome modeling
- Data integration / batch correction
- Cell type prediction from reference

## Optional CBOW gene embedding pretraining

CBOW pretraining is optional and only required when `encoder_type: "cbow"`.

```
train-cbow --config cbow_config.yml
```

An example config file can be found in the configs directory of this repository.

## VAE Training
Singe-GPU training:

```
train-vae --config vae_train.yml
```

Multi-GPU training (single node):

```
torchrun --standalone --nproc-per-node=4 -m cellinguist.train.train_vae --config vae_train.yml
```

Example config files can be found in the configs directory of this repository.

## Export gene and cell embeddings
To export gene embeddings from the first phase of training:

```
export-gene-embeddings \
  --adata /home/arc85/Desktop/cellinguist_results_251125/02_hnscc_test_260303/01_input/HNSCC_2k_hvg_annotated_anndata_260303.h5ad \
  --embeddings /home/arc85/Desktop/cellinguist_results_251125/02_hnscc_test_260303/03_output/hnscc_2k_hvg_cbow_gene_embeddings_260303.pth \
  --out /home/arc85/Desktop/cellinguist_results_251125/02_hnscc_test_260303/03_output/hnscc_2k_hvg_cbow_gene_embeddings_260303_440pm.tsv.gz \
  --gene-key gene \
  --n-bins 20 \
  --min-expr 0.0
```

To export cell embeddings from the second phase of training:

```
export-cbow-vae-cell-embeddings \
  --adata /home/arc85/Desktop/cellinguist_results_251125/02_hnscc_test_260303/01_input/HNSCC_2k_hvg_annotated_anndata_260303.h5ad \
  --checkpoint /home/arc85/Desktop/cellinguist_results_251125/02_hnscc_test_260303/03_output/mlp_vae_run_003_last.ckpt \
  --gene-emb-tsv /home/arc85/Desktop/cellinguist_results_251125/02_hnscc_test_260303/03_output/hnscc_2k_hvg_cbow_gene_embeddings_260303_440pm.tsv.gz \
  --batch-size 2048 \
  --no-backed \
  --num-workers 4 \
  --device "cuda" \
  --out /home/arc85/Desktop/cellinguist_results_251125/02_hnscc_test_260303/03_output/hnscc_2k_hvg_cbow_vae_cell_embeddings_260303.tsv.gz
```

## Future directions

Other functionality will be coming soon.