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
