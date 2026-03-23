from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Optional

import anndata as ad

CACHE_SCHEMA = "token_index_cache_v1"
CACHE_VERSION = 1


def _sha256_text(parts: list[str]) -> str:
    h = hashlib.sha256()
    for p in parts:
        h.update(p.encode("utf-8", errors="strict"))
        h.update(b"\n")
    return h.hexdigest()


def compute_gene_order_hash(genes: list[str]) -> str:
    return _sha256_text([str(g) for g in genes])


def compute_adata_fingerprint(
    adata_path: str,
    gene_key: str,
) -> dict[str, Any]:
    p = Path(adata_path).resolve()
    if not p.exists():
        raise FileNotFoundError(str(p))

    st = p.stat()
    adata = ad.read_h5ad(str(p), backed="r")
    try:
        if gene_key not in adata.var.columns:
            raise ValueError(
                f"gene_key '{gene_key}' not found in adata.var for fingerprinting. "
                f"Available columns: {list(adata.var.columns)}"
            )
        var_genes = adata.var[gene_key].astype(str).to_list()
        var_gene_hash = compute_gene_order_hash(var_genes)
        n_cells = int(adata.n_obs)
        n_genes = int(adata.n_vars)
    finally:
        if getattr(adata, "isbacked", False):
            adata.file.close()

    parts = [
        str(p),
        str(int(st.st_size)),
        str(int(st.st_mtime_ns)),
        str(n_cells),
        str(n_genes),
        var_gene_hash,
        str(gene_key),
    ]
    return {
        "path": str(p),
        "size_bytes": int(st.st_size),
        "mtime_ns": int(st.st_mtime_ns),
        "n_cells": n_cells,
        "n_genes": n_genes,
        "gene_key": str(gene_key),
        "var_gene_hash": var_gene_hash,
        "fingerprint_hash": _sha256_text(parts),
    }


def load_cache_metadata(cache_dir: str | Path) -> dict[str, Any]:
    p = Path(cache_dir)
    meta_path = p / "metadata.json"
    if not meta_path.exists():
        raise FileNotFoundError(
            f"Token index cache metadata not found: {meta_path}"
        )
    with meta_path.open("r") as f:
        meta = json.load(f)
    if not isinstance(meta, dict):
        raise ValueError("Token index metadata must be a JSON object.")
    return meta


def _as_int(v: Any, name: str) -> int:
    try:
        return int(v)
    except Exception as e:
        raise ValueError(f"Invalid integer for {name}: {v!r}") from e


def validate_cache_metadata(
    meta: dict[str, Any],
    *,
    n_cells: int,
    n_genes: int,
    gene_order: list[str],
    min_expr_for_token: float,
    max_tokens_per_cell: Optional[int],
    adata_path: Optional[str],
    gene_key: str,
) -> None:
    schema = str(meta.get("schema", ""))
    version = _as_int(meta.get("version", -1), "version")
    if schema != CACHE_SCHEMA or version != CACHE_VERSION:
        raise ValueError(
            f"Unsupported token cache schema/version: schema={schema!r}, version={version}. "
            f"Expected schema={CACHE_SCHEMA!r}, version={CACHE_VERSION}."
        )

    meta_n_cells = _as_int(meta.get("n_cells", -1), "n_cells")
    meta_n_genes = _as_int(meta.get("n_genes", -1), "n_genes")
    if meta_n_cells != int(n_cells):
        raise ValueError(
            f"Token cache n_cells mismatch: cache={meta_n_cells}, data={int(n_cells)}."
        )
    if meta_n_genes != int(n_genes):
        raise ValueError(
            f"Token cache n_genes mismatch: cache={meta_n_genes}, data={int(n_genes)}."
        )

    expected_gene_hash = compute_gene_order_hash([str(g) for g in gene_order])
    got_gene_hash = str(meta.get("gene_order_hash", ""))
    if got_gene_hash != expected_gene_hash:
        raise ValueError("Token cache gene_order_hash mismatch with current dataset gene order.")

    got_min_expr = float(meta.get("min_expr_for_token", 0.0))
    if abs(got_min_expr - float(min_expr_for_token)) > 1e-12:
        raise ValueError(
            "Token cache min_expr_for_token mismatch: "
            f"cache={got_min_expr}, config={float(min_expr_for_token)}."
        )

    meta_max = meta.get("max_tokens_per_cell", None)
    want_max = None if max_tokens_per_cell is None else int(max_tokens_per_cell)
    if meta_max is not None:
        meta_max = int(meta_max)
    if meta_max != want_max:
        raise ValueError(
            "Token cache max_tokens_per_cell mismatch: "
            f"cache={meta_max}, config={want_max}."
        )

    shards = meta.get("shards", None)
    if not isinstance(shards, list) or len(shards) == 0:
        raise ValueError("Token cache metadata has no valid shards list.")

    if adata_path:
        fp_meta = meta.get("adata_fingerprint", None)
        if not isinstance(fp_meta, dict):
            raise ValueError("Token cache metadata missing adata_fingerprint object.")
        fp_now = compute_adata_fingerprint(str(adata_path), gene_key=gene_key)
        if str(fp_meta.get("fingerprint_hash", "")) != str(fp_now.get("fingerprint_hash", "")):
            raise ValueError(
                "Token cache adata_fingerprint mismatch. "
                "Cache likely corresponds to a different dataset file/version."
            )
