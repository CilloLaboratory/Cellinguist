from __future__ import annotations

import argparse
import gzip
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import torch

from cellinguist.data.datasets import SingleCellDataset
from cellinguist.embeddings import load_gene_embeddings


def compute_gene_embeddings(
    id_to_token: dict[int, str],
    token_embeddings: torch.Tensor,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Aggregate token embeddings into gene embeddings by averaging over bins.

    Parameters
    ----------
    id_to_token : dict
        Mapping from token_id -> token string (e.g. "CD3D__5").
    token_embeddings : torch.Tensor
        Token embedding matrix of shape (vocab_size, emb_dim).

    Returns
    -------
    genes : np.ndarray
        Array of gene names, shape (n_genes,).
    gene_embs : np.ndarray
        Gene embedding matrix, shape (n_genes, emb_dim).
    """
    token_emb_np = token_embeddings.detach().cpu().numpy()
    vocab_size, emb_dim = token_emb_np.shape

    gene_to_vecs: dict[str, list[np.ndarray]] = defaultdict(list)

    for idx in range(vocab_size):
        tok = id_to_token[idx]
        if "__" in tok:
            gene, _bin_str = tok.split("__", 1)
        else:
            gene = tok
        gene_to_vecs[gene].append(token_emb_np[idx])

    genes: list[str] = []
    gene_embs: list[np.ndarray] = []

    for gene, vecs in gene_to_vecs.items():
        arr = np.stack(vecs, axis=0)         # (n_bins_for_gene, emb_dim)
        mean_vec = arr.mean(axis=0)         # (emb_dim,)
        genes.append(gene)
        gene_embs.append(mean_vec)

    genes_arr = np.asarray(genes, dtype=object)
    gene_embs_arr = np.stack(gene_embs, axis=0)
    return genes_arr, gene_embs_arr


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export CBOW-based gene embeddings from token embeddings."
    )
    parser.add_argument(
        "--adata",
        required=True,
        help="Path to input .h5ad file (same data/tokenization used for CBOW training).",
    )
    parser.add_argument(
        "--embeddings",
        required=True,
        help="Path to learned CBOW token embeddings (.pt file).",
    )
    parser.add_argument(
        "--out",
        required=True,
        help="Output path for gzipped TSV of gene embeddings (e.g. gene_embeddings.tsv.gz).",
    )
    parser.add_argument(
        "--gene-key",
        default="gene",
        help="Column in adata.var with gene names (default: gene).",
    )
    parser.add_argument(
        "--cond-key",
        default=None,
        help="Column in adata.obs with condition labels (default: None).",
    )
    parser.add_argument(
        "--layer",
        default=None,
        help="Layer in adata.layers to use instead of X (default: None).",
    )
    parser.add_argument(
        "--n-bins",
        type=int,
        default=20,
        help="Number of expression bins used for tokenization (default: 20).",
    )
    parser.add_argument(
        "--min-expr",
        type=float,
        default=0.0,
        help="Minimum expression threshold for a gene to be considered present (default: 0.0).",
    )
    parser.add_argument(
        "--no-shuffle-tokens",
        action="store_true",
        help="Disable per-cell token shuffling (must match CBOW training if changed).",
    )
    parser.add_argument(
        "--min-token-count",
        type=int,
        default=1,
        help="Minimum token count when building vocab (default: 1; must match CBOW training).",
    )

    args = parser.parse_args()

    adata_path = args.adata
    emb_path = args.embeddings
    out_path = Path(args.out)

    # 1. Load token embeddings
    token_embeddings = load_gene_embeddings(emb_path, map_location="cpu")

    # 2. Rebuild SingleCellDataset to reconstruct the token vocabulary
    ds = SingleCellDataset(
        adata_or_path=adata_path,
        gene_key=args.gene_key,
        cond_key=args.cond_key,
        layer=args.layer,
        n_bins=args.n_bins,
        shuffle_tokens=not args.no_shuffle_tokens,
        min_expr=args.min_expr,
        token_to_id=None,            # build vocab from this dataset
        min_token_count=args.min_token_count,
    )

    if ds.vocab_size != token_embeddings.shape[0]:
        raise ValueError(
            f"Vocab size ({ds.vocab_size}) does not match embedding rows "
            f"({token_embeddings.shape[0]}). Check tokenization and config."
        )

    # 3. Compute gene embeddings
    genes, gene_embs = compute_gene_embeddings(ds.id_to_token, token_embeddings)

    # 4. Build DataFrame and save as gzipped TSV
    df = pd.DataFrame(
        gene_embs,
        columns=[f"dim_{i+1}" for i in range(gene_embs.shape[1])],
    )
    df.insert(0, "gene", genes)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(out_path, "wt") as f:
        df.to_csv(f, sep="\t", index=False)

    print(f"Saved gene embeddings to {out_path}")


if __name__ == "__main__":
    main()
