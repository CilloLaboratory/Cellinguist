from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np


def _signature_from_row(
    row: np.ndarray,
    cytokine_keys: List[str],
    eps: float = 1e-8,
) -> str:
    active = np.where(np.abs(row) > float(eps))[0].tolist()
    if not active:
        return "NONE"
    return "+".join(cytokine_keys[i] for i in active)


def build_cytokine_combo_split(
    perturb_matrix: Optional[np.ndarray],
    cytokine_keys: List[str],
    min_active_for_holdout: int = 2,
    eps: float = 1e-8,
) -> Dict[str, object]:
    """
    Build a default cytokine counterfactual split.

    Cells with >= min_active_for_holdout non-zero cytokines are held out for
    validation. Remaining cells are used for training.
    """
    if perturb_matrix is None:
        return {
            "train_indices": [],
            "val_indices": [],
            "n_train_cells": 0,
            "n_val_cells": 0,
            "n_train_signatures": 0,
            "n_val_signatures": 0,
        }

    if perturb_matrix.ndim != 2:
        raise ValueError("perturb_matrix must have shape (n_cells, n_cytokines).")
    if perturb_matrix.shape[1] != len(cytokine_keys):
        raise ValueError(
            "cytokine_keys length does not match perturb_matrix width: "
            f"{len(cytokine_keys)} vs {perturb_matrix.shape[1]}."
        )

    active_counts = (np.abs(perturb_matrix) > float(eps)).sum(axis=1)
    val_mask = active_counts >= int(min_active_for_holdout)
    train_mask = ~val_mask

    train_indices = np.where(train_mask)[0].astype(np.int64).tolist()
    val_indices = np.where(val_mask)[0].astype(np.int64).tolist()

    # Safety fallback to avoid an empty train set.
    if not train_indices:
        train_indices = np.arange(perturb_matrix.shape[0], dtype=np.int64).tolist()
        val_indices = []
        train_mask = np.ones(perturb_matrix.shape[0], dtype=bool)
        val_mask = ~train_mask

    train_sigs = set()
    val_sigs = set()
    for i in train_indices:
        train_sigs.add(_signature_from_row(perturb_matrix[i], cytokine_keys, eps=eps))
    for i in val_indices:
        val_sigs.add(_signature_from_row(perturb_matrix[i], cytokine_keys, eps=eps))

    return {
        "train_indices": train_indices,
        "val_indices": val_indices,
        "n_train_cells": len(train_indices),
        "n_val_cells": len(val_indices),
        "n_train_signatures": len(train_sigs),
        "n_val_signatures": len(val_sigs),
    }
