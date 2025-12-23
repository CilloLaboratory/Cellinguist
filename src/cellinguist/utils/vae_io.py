from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch


def set_seed(seed: Optional[int]) -> None:
    if seed is None:
        return
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_gene_embeddings_tsv(gene_emb_tsv: str) -> Tuple[List[str], torch.Tensor]:
    df = pd.read_csv(gene_emb_tsv, sep="\t")
    genes = df["gene"].astype(str).tolist()
    mat = df.drop(columns=["gene"]).to_numpy().astype("float32")
    n_genes, d_mat = mat.shape
    print(f"Loaded gene embeddings for {n_genes} genes and {d_mat} dimensions")
    return genes, torch.from_numpy(mat)


def intersect_genes_in_embedding_order(
    genes_from_emb: List[str],
    genes_expr: List[str],
) -> List[str]:
    expr_set = set(genes_expr)
    genes_common = [g for g in genes_from_emb if g in expr_set]
    if len(genes_common) == 0:
        raise ValueError("No overlapping genes between embeddings and expression data.")
    print(f"Using {len(genes_common)} genes in common between embeddings and expression")
    return genes_common


def subset_embeddings(
    genes_from_emb: List[str],
    emb_matrix: torch.Tensor,
    genes_common: List[str],
) -> torch.Tensor:
    idx_map = {g: i for i, g in enumerate(genes_from_emb)}
    idx = [idx_map[g] for g in genes_common]
    return emb_matrix[idx, :]


def save_vae_checkpoint(
    path: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    genes_common: List[str],
    config_snapshot: Dict[str, Any],
    gene_emb_source: str,
) -> None:
    ckpt = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "genes_common": genes_common,
        "config": config_snapshot,
        "gene_emb_source": gene_emb_source,
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(ckpt, path)


def load_vae_checkpoint(
    path: str,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    map_location: str | torch.device = "cpu",
) -> Dict[str, Any]:
    ckpt = torch.load(path, map_location=map_location)
    model.load_state_dict(ckpt["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    return ckpt
