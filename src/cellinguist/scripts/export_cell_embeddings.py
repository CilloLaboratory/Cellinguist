from __future__ import annotations

import argparse
import gzip
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from cellinguist.data.datasets import SingleCellDataset
from cellinguist.embeddings import load_gene_embeddings


def compute_cell_embeddings(
    dataset: SingleCellDataset,
    token_embeddings: torch.Tensor,
    fill_zero_for_empty: bool = True,
) -> np.ndarray:
    """
    Compute cell embeddings by averaging token embeddings in each cell.

    Parameters
    ----------
    dataset : SingleCellDataset
        Dataset with per-cell token_id sequences.
    token_embeddings : torch.Tensor
        Learned token embedding matrix of shape (vocab_size, emb_dim).
    fill_zero_for_empty : bool, default True
        If a cell has no tokens, return an all-zero vector. If False,
        will raise an error for empty cells.

    Returns
    -------
    cell_embs : np.ndarray
        Array of shape (n_cells, emb_dim) with one embedding per cell.
    """
    token_emb_np = token_embeddings.detach().cpu().numpy()
    vocab_size, emb_dim = token_emb_np.shape

    n_cells = len(dataset)
    cell_embs = np.zeros((n_cells, emb_dim), dtype=np.float32)

    for i, token_ids_np in enumerate(dataset._token_id_seqs):
        if token_ids_np.size == 0:
            if fill_zero_for_empty:
                cell_embs[i, :] = 0.0
            else:
                raise ValueError(f"Cell {i} has no tokens; cannot compute embedding.")
        else:
            if token_ids_np.max() >= vocab_size or token_ids_np.min() < 0:
                raise ValueError(
                    f"Token IDs out of range for cell {i}: "
                    f"min={token_ids_np.min()}, max={token_ids_np.max()}, "
                    f"vocab_size={vocab_size}"
                )
            vecs = token_emb_np[token_ids_np]     # (L_i, emb_dim)
            cell_embs[i, :] = vecs.mean(axis=0)  # (emb_dim,)

    return cell_embs


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute cell embeddings by averaging CBOW token embeddings per cell."
    )
    parser.add_argument(
        "--adata",
        required=True,
        help="Path to input .h5ad file (same one used for CBOW training).",
    )
    parser.add_argument(
        "--embeddings",
        required=True,
        help="Path to learned CBOW token embeddings (.pt file).",
    )
    parser.add_argument(
        "--out",
        required=True,
        help="Output path for gzipped TSV of cell embeddings (e.g. cell_embeddings.tsv.gz).",
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

    # 2. Rebuild SingleCellDataset with same tokenization settings used during CBOW training
    ds = SingleCellDataset(
        adata_or_path=adata_path,
        gene_key=args.gene_key,
        cond_key=args.cond_key,
        layer=args.layer,
        n_bins=args.n_bins,
        shuffle_tokens=not args.no_shuffle_tokens,
        min_expr=args.min_expr,
        token_to_id=None,
        min_token_count=args.min_token_count,
    )

    if ds.vocab_size != token_embeddings.shape[0]:
        raise ValueError(
            f"Vocab size ({ds.vocab_size}) does not match embedding matrix rows "
            f"({token_embeddings.shape[0]}). Make sure you're using the same "
            f"dataset/tokenization settings as during CBOW training."
        )

    # 3. Compute cell embeddings
    cell_embs = compute_cell_embeddings(ds, token_embeddings)

    # 4. Build DataFrame with cell IDs
    cell_ids = np.asarray(ds.adata.obs_names)
    if cell_ids.shape[0] != cell_embs.shape[0]:
        raise ValueError(
            f"Number of cell IDs ({cell_ids.shape[0]}) != number of embeddings "
            f"({cell_embs.shape[0]})."
        )

    df = pd.DataFrame(
        cell_embs,
        columns=[f"dim_{i+1}" for i in range(cell_embs.shape[1])],
    )
    df.insert(0, "cell_id", cell_ids)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(out_path, "wt") as f:
        df.to_csv(f, sep="\t", index=False)

    print(f"Saved cell embeddings to {out_path}")


if __name__ == "__main__":
    main()
