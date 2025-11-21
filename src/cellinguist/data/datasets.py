from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Union

import numpy as np
import torch
from torch.utils.data import Dataset

try:
    import anndata as ad
except ImportError as e:
    raise ImportError(
        "anndata is required for SingleCellDataset. "
        "Install with `pip install anndata`."
    ) from e

try:
    from scipy import sparse as sp
except ImportError:
    sp = None

try:
    import pandas as pd
except ImportError as e:
    raise ImportError(
        "pandas is required for SingleCellDataset. "
        "Install with `pip install pandas`."
    ) from e

from collections import Counter


@dataclass
class CellTokens:
    """
    Container for per-cell tokenization results.

    tokens : List[str]
        List of tokens for this cell, e.g. ["CD3D__5", "IFNG__18", ...].
    libsize : float
        Total library size (sum of counts) for the cell.
    cond_idx : Optional[int]
        Integer index of the cell's condition / perturbation, if provided.
    """

    tokens: List[str]
    libsize: float
    cond_idx: Optional[int] = None


class SingleCellDataset(Dataset):
    """
    Single-cell dataset that reads an AnnData object and produces, for each cell,
    a sequence of token IDs suitable for CBOW/word2vec training.

    For each cell, we:
      - take non-zero (or > min_expr) gene expression values,
      - compute per-cell quantile bins (1..n_bins),
      - create tokens "GENE__BIN",
      - optionally scramble order within the cell.

    Then we build (or accept) a vocabulary token -> id, and encode each
    cell's token list into a 1D LongTensor of token IDs.

    Parameters
    ----------
    adata_or_path : AnnData or str
        AnnData object or path to a .h5ad file.
    gene_key : str, default "gene"
        Column in `adata.var` containing gene names used in tokens.
    cond_key : Optional[str], default None
        Column in `adata.obs` containing condition / perturbation labels.
        If provided, these are mapped to integer indices.
    layer : Optional[str], default None
        If not None, use `adata.layers[layer]` instead of `adata.X` for expression.
    n_bins : int, default 20
        Number of quantile bins per cell (labels 1..n_bins).
    shuffle_tokens : bool, default True
        Whether to scramble the order of tokens within each cell (like the R script).
    min_expr : float, default 0.0
        Threshold for considering expression "present" (x > min_expr).
    token_to_id : Optional[Dict[str, int]], default None
        Predefined vocabulary mapping token -> id. If None, a new vocab is built
        from this dataset (respecting min_token_count).
    min_token_count : int, default 1
        Minimum frequency for a token to be included when building a vocab.
        Ignored if `token_to_id` is provided.
    rng : Optional[np.random.Generator], default None
        RNG for token shuffling; if None, a new Generator is created.
    """

    def __init__(
        self,
        adata_or_path: Union["ad.AnnData", str],
        gene_key: str = "gene",
        cond_key: Optional[str] = None,
        layer: Optional[str] = None,
        n_bins: int = 20,
        shuffle_tokens: bool = True,
        min_expr: float = 0.0,
        token_to_id: Optional[Dict[str, int]] = None,
        min_token_count: int = 1,
        rng: Optional[np.random.Generator] = None,
    ) -> None:
        super().__init__()

        # Load AnnData if a path is provided
        if isinstance(adata_or_path, str):
            self.adata = ad.read_h5ad(adata_or_path)
        else:
            self.adata = adata_or_path

        self.n_bins = int(n_bins)
        self.shuffle_tokens = bool(shuffle_tokens)
        self.min_expr = float(min_expr)
        self.rng = rng if rng is not None else np.random.default_rng()

        # --- Expression matrix ---
        if layer is None:
            X = self.adata.X
        else:
            if layer not in self.adata.layers:
                raise ValueError(
                    f"Layer '{layer}' not found in adata.layers. "
                    f"Available: {list(self.adata.layers.keys())}"
                )
            X = self.adata.layers[layer]

        # Support dense or sparse matrices
        if sp is not None and sp.issparse(X):
            self.X = X.tocsr()
        else:
            self.X = np.asarray(X)

        # --- Gene names ---
        if gene_key not in self.adata.var.columns:
            raise ValueError(
                f"gene_key '{gene_key}' not found in adata.var. "
                f"Available columns: {list(self.adata.var.columns)}"
            )
        self.gene_names: np.ndarray = np.asarray(self.adata.var[gene_key].values)
        self.n_genes: int = self.gene_names.shape[0]

        # --- Condition labels (optional) ---
        self.cond_key = cond_key
        self.cond_labels: Optional[np.ndarray]
        self.cond_to_idx: Optional[Dict[str, int]]
        self.idx_to_cond: Optional[Dict[int, str]]

        if cond_key is not None:
            if cond_key not in self.adata.obs.columns:
                raise ValueError(
                    f"cond_key '{cond_key}' not found in adata.obs. "
                    f"Available columns: {list(self.adata.obs.columns)}"
                )
            cond_raw = self.adata.obs[cond_key].astype("str").values
            uniques = sorted(pd.unique(cond_raw))
            self.cond_to_idx = {c: i for i, c in enumerate(uniques)}
            self.idx_to_cond = {i: c for c, i in self.cond_to_idx.items()}
            self.cond_labels = np.array(
                [self.cond_to_idx[c] for c in cond_raw], dtype=int
            )
        else:
            self.cond_labels = None
            self.cond_to_idx = None
            self.idx_to_cond = None

        # --- Per-cell raw tokens ---
        self._cells: List[CellTokens] = self._build_cell_tokens()

        # --- Build or use provided token vocabulary ---
        if token_to_id is None:
            self.token_to_id = self._build_vocab(
                self._cells, min_token_count=min_token_count
            )
        else:
            self.token_to_id = dict(token_to_id)

        # Inverse vocab
        self.id_to_token: Dict[int, str] = {
            idx: tok for tok, idx in self.token_to_id.items()
        }

        # --- Encode per-cell tokens as ID sequences ---
        self._token_id_seqs: List[np.ndarray] = self._encode_cells_to_ids(self._cells)

    # -------------------------------------------------------------------------
    # Internal helpers
    # -------------------------------------------------------------------------

    def _get_row(self, i: int) -> np.ndarray:
        """
        Return expression vector for cell i as a 1D numpy array of shape (n_genes,).
        """
        row = self.X[i]
        if sp is not None and sp.issparse(row):
            return np.asarray(row.toarray()).ravel()
        return np.asarray(row).ravel()

    def _bin_nonzero_values(self, values: np.ndarray) -> np.ndarray:
        """
        Given 1D array of non-zero expression values for a single cell,
        compute per-cell quantile bins and return integer bin labels 1..n_bins.

        Mirrors the R logic:

            u <- unique(x[y])
            set_breaks <- quantile(u, probs=seq(0,1,1/n_bins))
            set_breaks[1] <- 0
            set_breaks[length(set_breaks)] <- max(u)
            binned <- cut(x[y], breaks=set_breaks, labels=1:n_bins)
        """
        if values.size == 0:
            return np.array([], dtype=int)

        unique_vals = np.unique(values)
        # If all non-zero values are identical, put everything in the last bin.
        if unique_vals.size == 1:
            return np.full(values.shape, self.n_bins, dtype=int)

        # Quantile-based breaks
        probs = np.linspace(0.0, 1.0, self.n_bins + 1)
        breaks = np.quantile(unique_vals, probs)
        # Enforce first and last breaks like in the R script
        breaks[0] = 0.0
        breaks[-1] = float(unique_vals.max())

        # Ensure breaks are strictly increasing
        eps = 1e-8
        for j in range(1, len(breaks)):
            if breaks[j] <= breaks[j - 1]:
                breaks[j] = breaks[j - 1] + eps

        # Digitize values into bins 1..n_bins
        bin_idx = np.digitize(values, breaks, right=True)
        bin_idx = np.clip(bin_idx, 1, self.n_bins).astype(int)
        return bin_idx

    def _build_cell_tokens(self) -> List[CellTokens]:
        """
        Construct tokens for each cell, mirroring the R workflow:

          - For each cell i:
            * Extract expression row x_i
            * Find indices with x_i > min_expr
            * Compute quantile bins on non-zero values
            * Generate tokens "GENE__BIN"
            * Scramble within the cell (if shuffle_tokens=True)
        """
        n_cells = self.adata.n_obs
        cells: List[CellTokens] = []

        for i in range(n_cells):
            x = self._get_row(i)  # (n_genes,)
            libsize = float(x.sum())

            # Positions with expression > min_expr
            pos = np.where(x > self.min_expr)[0]
            if pos.size == 0:
                tokens: List[str] = []
            else:
                vals = x[pos]
                gene_names = self.gene_names[pos]
                bin_idx = self._bin_nonzero_values(vals)  # labels 1..n_bins

                df = pd.DataFrame(
                    {"name": gene_names, "value": bin_idx.astype(str)}
                )
                df["token"] = df["name"] + "__" + df["value"]

                if self.shuffle_tokens and len(df) > 1:
                    scramble = self.rng.permutation(len(df))
                    df = df.iloc[scramble]

                tokens = df["token"].tolist()

            cond_idx = None
            if self.cond_labels is not None:
                cond_idx = int(self.cond_labels[i])

            cells.append(CellTokens(tokens=tokens, libsize=libsize, cond_idx=cond_idx))

        return cells

    def _build_vocab(
        self, cells: List[CellTokens], min_token_count: int = 1
    ) -> Dict[str, int]:
        """
        Build a token -> id vocabulary from the list of CellTokens.

        Tokens with frequency < min_token_count are dropped.

        Vocab indices are assigned in a deterministic order:
          - sort tokens by descending frequency, then lexicographically.
        """
        counter = Counter()
        for cell in cells:
            counter.update(cell.tokens)

        # Filter by min_token_count
        items = [
            (tok, cnt) for tok, cnt in counter.items() if cnt >= min_token_count
        ]
        # Sort: most frequent first, then alphabetically
        items.sort(key=lambda x: (-x[1], x[0]))

        token_to_id: Dict[str, int] = {}
        for idx, (tok, _) in enumerate(items):
            token_to_id[tok] = idx

        return token_to_id

    def _encode_cells_to_ids(
        self, cells: List[CellTokens]
    ) -> List[np.ndarray]:
        """
        Convert each cell's token list to an array of token IDs.

        Tokens not found in the vocab are skipped (shouldn't happen if vocab
        was built from the same cells without UNK).
        """
        token_id_seqs: List[np.ndarray] = []
        for cell in cells:
            ids = [
                self.token_to_id[tok]
                for tok in cell.tokens
                if tok in self.token_to_id
            ]
            token_id_seqs.append(np.asarray(ids, dtype=np.int64))
        return token_id_seqs

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._cells)

    def __getitem__(self, idx: int):
        """
        Return per-cell token IDs and some metadata.

        Returns
        -------
        sample : dict
            {
                "token_ids": LongTensor[L_i],
                "tokens": List[str],        # raw string tokens (optional, for inspection)
                "libsize": float,
                "cond_idx": Optional[int],  # if cond_key was provided
                "cell_idx": int,
            }
        """
        cell = self._cells[idx]
        ids_np = self._token_id_seqs[idx]
        token_ids = torch.from_numpy(ids_np)  # (L_i,)

        # Use -1 as a sentinel when there is no condition
        cond_idx = -1 if cell.cond_idx is None else int(cell.cond_idx)

        return {
            "token_ids": token_ids,
            "tokens": cell.tokens,
            "libsize": cell.libsize,
            "cond_idx": cond_idx,
            "cell_idx": idx,
        }

    @property
    def vocab_size(self) -> int:
        return len(self.token_to_id)

    @property
    def input_strings(self) -> List[str]:
        """
        Convenience property to get a vector of space-separated token strings,
        exactly analogous to R's `input_dat`:

            input_dat <- sapply(binned_dat, function(x) paste(x$token, collapse=" "))

        Useful if you still want to feed text into a word2vec tool.
        """
        return [" ".join(c.tokens) if c.tokens else "" for c in self._cells]

    def get_condition_mapping(self) -> Optional[Dict[int, str]]:
        """
        Return mapping from integer condition indices to raw condition labels,
        or None if cond_key was not provided.
        """
        return self.idx_to_cond
