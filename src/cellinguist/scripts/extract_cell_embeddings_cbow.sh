#!/bin/bash

python /home/arc85/Desktop/Cellinguist/src/cellinguist/scripts/extract_cell_embeddings_from_cbow.py \
  --adata /home/arc85/Desktop/Cellinguist/cytokine_dictionary_2k_pbs_ifng_251108.h5ad \
  --embeddings /home/arc85/Desktop/Cellinguist/gene_token_embeddings_251121.pt \
  --out /home/arc85/Desktop/Cellinguist/cbow_cell_embeddings_251121.tsv.gz \
  --gene-key gene \
  --n-bins 20 \
  --min-expr 0.0
