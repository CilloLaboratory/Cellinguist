from __future__ import annotations

from typing import Any

import torch


def collate_vae_batch(batch: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
    """
    Collate VAE samples and (optionally) variable-length precomputed token indices.

    Expected per-sample keys:
      - required: x_expr, libsize
      - optional: batch_idx, cond_idx, perturb_vec, token_gene_idx

    Returns padded token tensors when token_gene_idx is present:
      - token_gene_idx: LongTensor[B, Lmax]
      - token_gene_mask: BoolTensor[B, Lmax]  (True = valid token)
    """
    out: dict[str, torch.Tensor] = {}

    out["x_expr"] = torch.stack([item["x_expr"] for item in batch], dim=0)
    out["libsize"] = torch.stack([item["libsize"] for item in batch], dim=0)

    if "batch_idx" in batch[0]:
        out["batch_idx"] = torch.stack([item["batch_idx"] for item in batch], dim=0)
    if "cond_idx" in batch[0]:
        out["cond_idx"] = torch.stack([item["cond_idx"] for item in batch], dim=0)
    if "perturb_vec" in batch[0]:
        out["perturb_vec"] = torch.stack([item["perturb_vec"] for item in batch], dim=0)

    if "token_gene_idx" in batch[0]:
        seqs = [item["token_gene_idx"].to(dtype=torch.long) for item in batch]
        max_len = max((int(s.numel()) for s in seqs), default=0)
        bsz = len(seqs)

        if max_len == 0:
            out["token_gene_idx"] = torch.zeros((bsz, 0), dtype=torch.long)
            out["token_gene_mask"] = torch.zeros((bsz, 0), dtype=torch.bool)
        else:
            tok = torch.zeros((bsz, max_len), dtype=torch.long)
            mask = torch.zeros((bsz, max_len), dtype=torch.bool)
            for i, s in enumerate(seqs):
                n = int(s.numel())
                if n == 0:
                    continue
                tok[i, :n] = s
                mask[i, :n] = True
            out["token_gene_idx"] = tok
            out["token_gene_mask"] = mask

    return out


def collate_sample_phenotype_batch(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Collate variable-length sample bags for phenotype modeling.

    Expected per-sample keys:
      - x_expr: FloatTensor[N_i, G]
      - libsize: FloatTensor[N_i]
      - optional: batch_idx, cond_idx, perturb_vec, token_gene_idx
      - metadata: sample_id, phenotype, cell_ids, cell_types
    """
    out: dict[str, Any] = {}
    bsz = len(batch)
    max_cells = max(int(item["x_expr"].shape[0]) for item in batch)
    n_genes = int(batch[0]["x_expr"].shape[1])

    x = torch.zeros((bsz, max_cells, n_genes), dtype=batch[0]["x_expr"].dtype)
    libsize = torch.zeros((bsz, max_cells), dtype=batch[0]["libsize"].dtype)
    cell_mask = torch.zeros((bsz, max_cells), dtype=torch.bool)

    out["sample_id"] = [str(item["sample_id"]) for item in batch]
    out["phenotype"] = torch.tensor(
        [float(item["phenotype"]) for item in batch],
        dtype=torch.float32,
    )
    out["cell_ids"] = [list(item["cell_ids"]) for item in batch]
    out["cell_types"] = [list(item.get("cell_types", [])) for item in batch]

    has_batch = "batch_idx" in batch[0]
    has_cond = "cond_idx" in batch[0]
    has_perturb = "perturb_vec" in batch[0]
    has_token_idx = "token_gene_idx" in batch[0]

    batch_idx = torch.zeros((bsz, max_cells), dtype=torch.long) if has_batch else None
    cond_idx = torch.zeros((bsz, max_cells), dtype=torch.long) if has_cond else None
    perturb = None
    token_idx = None
    token_mask = None

    if has_perturb:
        p_dim = int(batch[0]["perturb_vec"].shape[1])
        perturb = torch.zeros((bsz, max_cells, p_dim), dtype=batch[0]["perturb_vec"].dtype)

    max_tokens = 0
    if has_token_idx:
        max_tokens = max(
            max((int(t.numel()) for t in item["token_gene_idx"]), default=0)
            for item in batch
        )
        token_idx = torch.zeros((bsz, max_cells, max_tokens), dtype=torch.long)
        token_mask = torch.zeros((bsz, max_cells, max_tokens), dtype=torch.bool)

    for i, item in enumerate(batch):
        n = int(item["x_expr"].shape[0])
        x[i, :n] = item["x_expr"]
        libsize[i, :n] = item["libsize"]
        cell_mask[i, :n] = True

        if has_batch and batch_idx is not None:
            batch_idx[i, :n] = item["batch_idx"]
        if has_cond and cond_idx is not None:
            cond_idx[i, :n] = item["cond_idx"]
        if has_perturb and perturb is not None:
            perturb[i, :n] = item["perturb_vec"]

        if has_token_idx and token_idx is not None and token_mask is not None:
            for j, seq in enumerate(item["token_gene_idx"]):
                seq = seq.to(dtype=torch.long)
                m = int(seq.numel())
                if m <= 0:
                    continue
                token_idx[i, j, :m] = seq
                token_mask[i, j, :m] = True

    out["x_expr"] = x
    out["libsize"] = libsize
    out["cell_mask"] = cell_mask
    if batch_idx is not None:
        out["batch_idx"] = batch_idx
    if cond_idx is not None:
        out["cond_idx"] = cond_idx
    if perturb is not None:
        out["perturb_vec"] = perturb
    if token_idx is not None and token_mask is not None:
        out["token_gene_idx"] = token_idx
        out["token_gene_mask"] = token_mask
    return out
