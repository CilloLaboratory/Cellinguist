from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import anndata as ad
import numpy as np

from cellinguist.utils.token_index_cache import (
    CACHE_SCHEMA,
    CACHE_VERSION,
    compute_adata_fingerprint,
    compute_gene_order_hash,
)

try:
    from scipy import sparse as sp
except ImportError:
    sp = None


def _load_gene_order_source(path: str) -> list[str]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(str(p))
    if p.suffix.lower() in {".txt", ".list"}:
        genes = [ln.strip() for ln in p.read_text().splitlines() if ln.strip()]
        if not genes:
            raise ValueError(f"No gene entries found in gene-order-source: {p}")
        return genes

    import pandas as pd

    sep = "\t" if p.suffix.lower() in {".tsv", ".gz"} else ","
    df = pd.read_csv(p, sep=sep)
    if "gene" in df.columns:
        genes = df["gene"].astype(str).tolist()
    else:
        genes = df.iloc[:, 0].astype(str).tolist()
    genes = [g for g in genes if g]
    if not genes:
        raise ValueError(f"No gene entries found in gene-order-source: {p}")
    return genes


@dataclass
class _ShardTask:
    shard_id: int
    adata_path: str
    out_dir: str
    cell_start: int
    cell_end: int
    layer: Optional[str]
    gene_indices: Optional[np.ndarray]
    min_expr_for_token: float
    max_tokens_per_cell: Optional[int]


def _read_row(adata, layer: Optional[str], idx: int) -> np.ndarray:
    if layer is None:
        row = adata.X[idx]
    else:
        row = adata.layers[layer][idx]
    if sp is not None and sp.issparse(row):
        x = np.asarray(row.toarray()).ravel()
    else:
        x = np.asarray(row).ravel()
    return x.astype(np.float32, copy=False)


def _build_single_shard(task: _ShardTask) -> dict:
    adata = ad.read_h5ad(task.adata_path, backed="r")
    try:
        offsets = [0]
        idx_chunks: list[np.ndarray] = []
        total = 0

        for cell_i in range(task.cell_start, task.cell_end):
            x = _read_row(adata, task.layer, cell_i)
            if task.gene_indices is not None:
                x = x[task.gene_indices]

            idx = np.where(x > task.min_expr_for_token)[0]
            if idx.size > 0 and task.max_tokens_per_cell is not None and idx.size > task.max_tokens_per_cell:
                vals = x[idx]
                topk = np.argpartition(vals, -task.max_tokens_per_cell)[-task.max_tokens_per_cell:]
                idx = idx[topk]
            idx = idx.astype(np.int32, copy=False)
            idx_chunks.append(idx)
            total += int(idx.size)
            offsets.append(total)

        indices = np.concatenate(idx_chunks, axis=0) if idx_chunks else np.zeros((0,), dtype=np.int32)
        offsets_arr = np.asarray(offsets, dtype=np.int64)

        out_dir = Path(task.out_dir)
        idx_name = f"indices_{task.shard_id:05d}.npy"
        off_name = f"offsets_{task.shard_id:05d}.npy"
        np.save(out_dir / idx_name, indices)
        np.save(out_dir / off_name, offsets_arr)

        return {
            "shard_id": int(task.shard_id),
            "cell_start": int(task.cell_start),
            "cell_end": int(task.cell_end),
            "indices_file": idx_name,
            "offsets_file": off_name,
            "n_tokens": int(indices.size),
        }
    finally:
        if getattr(adata, "isbacked", False):
            adata.file.close()


def precompute_token_index_cache(
    *,
    adata_path: str,
    out_dir: str,
    gene_key: str = "gene",
    layer: Optional[str] = None,
    gene_order_source: Optional[str] = None,
    min_expr_for_token: float = 0.0,
    max_tokens_per_cell: Optional[int] = None,
    shard_size_cells: int = 100000,
    num_workers: int = 8,
) -> dict:
    if shard_size_cells <= 0:
        raise ValueError("shard_size_cells must be > 0")
    if max_tokens_per_cell is not None and int(max_tokens_per_cell) <= 0:
        raise ValueError("max_tokens_per_cell must be > 0 when provided")

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    adata_probe = ad.read_h5ad(adata_path, backed="r")
    try:
        if gene_key not in adata_probe.var.columns:
            raise ValueError(
                f"gene_key '{gene_key}' not found in adata.var. "
                f"Available columns: {list(adata_probe.var.columns)}"
            )
        if layer is not None and layer not in adata_probe.layers:
            raise ValueError(
                f"Layer '{layer}' not found in adata.layers. "
                f"Available: {list(adata_probe.layers.keys())}"
            )

        n_cells = int(adata_probe.n_obs)
        var_genes = adata_probe.var[gene_key].astype(str).to_list()

        if gene_order_source:
            gene_order = _load_gene_order_source(gene_order_source)
            idx_map = {g: i for i, g in enumerate(var_genes)}
            missing = [g for g in gene_order if g not in idx_map]
            if missing:
                raise ValueError(
                    f"gene-order-source contains {len(missing)} genes not present in adata.var[{gene_key!r}]. "
                    f"Example missing gene: {missing[0]}"
                )
            gene_indices = np.asarray([idx_map[g] for g in gene_order], dtype=np.int64)
        else:
            gene_order = var_genes
            gene_indices = None

        n_genes = int(len(gene_order))
    finally:
        if getattr(adata_probe, "isbacked", False):
            adata_probe.file.close()

    tasks = []
    shard_id = 0
    for cell_start in range(0, n_cells, shard_size_cells):
        cell_end = min(cell_start + shard_size_cells, n_cells)
        tasks.append(
            _ShardTask(
                shard_id=shard_id,
                adata_path=str(adata_path),
                out_dir=str(out),
                cell_start=int(cell_start),
                cell_end=int(cell_end),
                layer=layer,
                gene_indices=gene_indices,
                min_expr_for_token=float(min_expr_for_token),
                max_tokens_per_cell=(None if max_tokens_per_cell is None else int(max_tokens_per_cell)),
            )
        )
        shard_id += 1

    workers = max(1, int(num_workers))
    if workers == 1:
        shard_meta = [_build_single_shard(t) for t in tasks]
    else:
        shard_meta = []
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futs = [ex.submit(_build_single_shard, t) for t in tasks]
            for f in as_completed(futs):
                shard_meta.append(f.result())

    shard_meta.sort(key=lambda x: int(x["shard_id"]))

    adata_fp = compute_adata_fingerprint(str(adata_path), gene_key=gene_key)
    gene_order_hash = compute_gene_order_hash(gene_order)

    metadata = {
        "schema": CACHE_SCHEMA,
        "version": CACHE_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "adata_path": str(Path(adata_path).resolve()),
        "adata_fingerprint": adata_fp,
        "gene_key": str(gene_key),
        "layer": layer,
        "n_cells": int(n_cells),
        "n_genes": int(n_genes),
        "gene_order_hash": gene_order_hash,
        "min_expr_for_token": float(min_expr_for_token),
        "max_tokens_per_cell": None if max_tokens_per_cell is None else int(max_tokens_per_cell),
        "shard_size_cells": int(shard_size_cells),
        "num_workers": int(workers),
        "gene_order_source": gene_order_source,
        "shards": [
            {
                "shard_id": int(s["shard_id"]),
                "cell_start": int(s["cell_start"]),
                "cell_end": int(s["cell_end"]),
                "indices_file": str(s["indices_file"]),
                "offsets_file": str(s["offsets_file"]),
                "n_tokens": int(s["n_tokens"]),
            }
            for s in shard_meta
        ],
    }

    with (out / "metadata.json").open("w") as f:
        json.dump(metadata, f, indent=2, sort_keys=True)

    # keep explicit gene order for transparency/debugging
    with (out / "gene_order.txt").open("w") as f:
        for g in gene_order:
            f.write(str(g))
            f.write("\n")

    print(
        "[token-cache] wrote cache "
        f"out_dir={out} n_cells={n_cells} n_genes={n_genes} "
        f"n_shards={len(shard_meta)} total_tokens={sum(int(s['n_tokens']) for s in shard_meta)}"
    )
    return metadata


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Precompute sharded transformer token gene index cache for VAE training."
    )
    ap.add_argument("--adata", required=True, help="Path to input .h5ad file.")
    ap.add_argument("--out-dir", required=True, help="Output directory for cache shards and metadata.")
    ap.add_argument("--gene-key", default="gene", help="Gene column in adata.var (default: gene).")
    ap.add_argument("--layer", default=None, help="Expression layer in adata.layers (default: X).")
    ap.add_argument(
        "--gene-order-source",
        default=None,
        help=(
            "Optional path to gene order source (.txt/.tsv/.csv). If provided, cache indices "
            "are generated in this gene order."
        ),
    )
    ap.add_argument(
        "--min-expr-for-token",
        type=float,
        default=0.0,
        help="Expression threshold for including gene token (default: 0.0).",
    )
    ap.add_argument(
        "--max-tokens-per-cell",
        type=int,
        default=None,
        help="Optional cap for number of tokens per cell.",
    )
    ap.add_argument(
        "--shard-size-cells",
        type=int,
        default=100000,
        help="Number of cells per shard file (default: 100000).",
    )
    ap.add_argument(
        "--num-workers",
        type=int,
        default=max(1, (os.cpu_count() or 1) // 2),
        help="Number of CPU worker processes for preprocessing.",
    )
    args = ap.parse_args()

    precompute_token_index_cache(
        adata_path=args.adata,
        out_dir=args.out_dir,
        gene_key=args.gene_key,
        layer=args.layer,
        gene_order_source=args.gene_order_source,
        min_expr_for_token=args.min_expr_for_token,
        max_tokens_per_cell=args.max_tokens_per_cell,
        shard_size_cells=args.shard_size_cells,
        num_workers=args.num_workers,
    )


if __name__ == "__main__":
    main()
