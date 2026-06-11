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

Batch-effect correction can be enabled in `vae_train.yml` with:
- `batch_key`: `adata.obs` column defining batch/sample labels
- `batch_correction_method`: `"none"` or `"mean_scale"`
- `batch_correction_eps`, `batch_correction_clip_min`, `batch_correction_clip_max`

`mean_scale` applies a per-gene multiplicative correction per batch before model input, with clipping to avoid extreme factors.

## Optional CBOW gene embedding pretraining

CBOW pretraining is optional and only required when `encoder_type: "cbow"`.

```
train-cbow --config cbow_config.yml
```

An example config file can be found in the configs directory of this repository.

## Transformer Token Index Cache (Recommended for Large Datasets)

For large transformer runs (especially DDP), precompute token gene indices once on CPU and reuse them across all GPU ranks.

1. Precompute cache:

```
precompute-transformer-token-indices \
  --adata /path/to/data.h5ad \
  --out-dir /path/to/token_cache \
  --gene-key gene \
  --min-expr-for-token 0.0 \
  --max-tokens-per-cell 256 \
  --shard-size-cells 100000 \
  --work-chunk-cells 5000 \
  --num-workers 16
```

2. Train with transformer using the cache:

Set in `vae_train.yml`:

```
encoder_type: "transformer"
token_index_cache_dir: "/path/to/token_cache"
token_index_cache_require: true
```

Then launch training (single GPU or torchrun DDP as usual).

Tip: use fewer, larger shard files for training I/O efficiency, and use `--work-chunk-cells` to increase CPU preprocessing parallelism without increasing shard count.

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

To export gene embeddings directly from a trained Transformer VAE checkpoint:

```
export-transformer-gene-embeddings \
  --checkpoint /path/to/vae_transformer_last.ckpt \
  --out /path/to/transformer_gene_embeddings.tsv.gz
```

(For Perceiver checkpoints, use `export-perceiver-gene-embeddings`.)

To export cell embeddings from a trained Transformer VAE checkpoint:

```
export-transformer-cell-embeddings \
  --adata /path/to/input.h5ad \
  --checkpoint /path/to/vae_transformer_last.ckpt \
  --out /path/to/transformer_cell_embeddings.tsv.gz \
  --device "cuda"
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

## Predict cytokine treatment consequences

For transformer VAE checkpoints trained with `perturbation_mode: "cytokine_vector"`, use:

```
predict-cytokine-treatment /path/to/cytokine_treatment_prediction.yml
```

The override TSV must contain:
- `cell_id`
- one column per checkpoint-resolved cytokine key, in exact order

The command writes:
- `pred_baseline.tsv.gz`
- `pred_treated.tsv.gz`
- `delta.tsv.gz`
- `metadata.json`

Transformer cytokine conditioning in the current implementation does not inject cytokine tokens into self-attention. Instead, the cytokine vector is projected with `PerturbationProjector`, concatenated after CLS pooling in the encoder, and concatenated again in the decoder.

If you trained with a transformer token index cache, set `token_index_cache_dir` in the prediction config to reuse it during inference.

## Future directions

Other functionality will be coming soon.

## GRN construction design spec

An implementation-ready algorithm specification for a new `construct-grn` command is available at:

- `docs/construct_grn_spec.md`

## Construct GRNs from a trained VAE checkpoint

You can construct TF->target GRNs (global and optional context-specific networks) with:

```
construct-grn --config src/cellinguist/configs/grn_construct.yml
```

The config controls:
- data/checkpoint inputs
- TF list and output directory
- perturbation-based decoder scoring parameters
- score fusion weights (`w_a`, `w_e`, `w_d`, `w_p`)
- optional bootstrap confidence settings
