from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Union

import numpy as np
import torch
from torch.utils.data import Dataset

import anndata as ad

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

from typing import Optional, Dict, Any
from dataclasses import dataclass
import numpy as np
import torch
from torch.utils.data import Dataset


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

@dataclass
class CBOWPairsConfig:
    """
    Configuration for CBOWPairsDataset.

    Attributes
    ----------
    window_size : int
        Context window radius. Total context length = 2 * window_size.
    samples_per_cell : int
        Logical number of CBOW samples per cell per "epoch". The dataset
        length will be n_cells * samples_per_cell, and each __getitem__
        will draw one random context window within the chosen cell.
    """

    window_size: int = 5
    samples_per_cell: int = 1

class CBOWPairsDataset(Dataset):
    """
    Dataset that wraps SingleCellDataset and yields CBOW training pairs:
        - target_ids:  LongTensor[1]
        - context_ids: LongTensor[2 * window_size]

    Negative samples are NOT generated here anymore; they are sampled in the
    training loop using torch.multinomial on the device.
    """

    def __init__(
        self,
        sc_dataset: SingleCellDataset,
        config: CBOWPairsConfig,
        rng: Optional[np.random.Generator] = None,
    ) -> None:
        super().__init__()
        self.sc_dataset = sc_dataset
        self.window_size = int(config.window_size)
        self.samples_per_cell = int(config.samples_per_cell)
        self.rng = rng if rng is not None else np.random.default_rng()

        # Convert token_id_seqs to torch.LongTensor once up front
        self.token_id_seqs = [
            torch.from_numpy(seq.astype(np.int64)) for seq in sc_dataset._token_id_seqs
        ]
        self.n_cells = len(self.token_id_seqs)
        self.vocab_size = sc_dataset.vocab_size

        # Precompute which cells have at least one valid CBOW window
        self._valid_cells = self._find_valid_cells()
        if self._valid_cells.size == 0:
            raise ValueError(
                "No cells have enough tokens for the chosen window_size. "
                "Reduce window_size or check tokenization."
            )

    def _find_valid_cells(self) -> np.ndarray:
        """
        Return indices of cells that have at least one valid target position
        with a full context window on both sides.
        """
        valid = []
        w = self.window_size
        min_len = 2 * w + 1
        for i, seq in enumerate(self.token_id_seqs):
            if seq.numel() >= min_len:
                valid.append(i)
        return np.asarray(valid, dtype=np.int64)

    def __len__(self) -> int:
        """
        Logical dataset length is n_cells * samples_per_cell.

        Each __getitem__ draws one random CBOW window from a (random) valid cell.
        """
        return self.n_cells * self.samples_per_cell

    def _sample_cell_and_position(self) -> tuple[int, int]:
        """
        Sample a valid cell and a valid target position within that cell.
        """
        w = self.window_size
        min_len = 2 * w + 1

        # Sample a valid cell index uniformly
        for _ in range(10):
            cell_idx = int(self.rng.choice(self._valid_cells))
            seq = self.token_id_seqs[cell_idx]
            L = seq.numel()
            if L >= min_len:
                pos = int(self.rng.integers(w, L - w))
                return cell_idx, pos

        # Fallback: first valid cell, deterministic safe position
        cell_idx = int(self._valid_cells[0])
        seq = self.token_id_seqs[cell_idx]
        L = seq.numel()
        w = self.window_size
        pos = max(w, min(L - w - 1, w))
        return cell_idx, pos

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Return a single CBOW training pair as a dict:

            {
                "target_ids":  LongTensor[1],
                "context_ids": LongTensor[2 * window_size],
                "cell_idx":    int,
                "position":    int,
            }

        When batched by DataLoader, you'll get:
            - target_ids:  (B,)
            - context_ids: (B, 2 * window_size)
        """
        cell_idx, pos = self._sample_cell_and_position()
        seq = self.token_id_seqs[cell_idx]
        w = self.window_size

        target = seq[pos]                            # scalar LongTensor
        left_context = seq[pos - w : pos]           # (w,)
        right_context = seq[pos + 1 : pos + 1 + w]  # (w,)

        # All torch ops, no numpy
        context = torch.cat([left_context, right_context], dim=0)  # (2w,)

        return {
            "target_ids": target,          # LongTensor[]
            "context_ids": context,        # LongTensor[2w]
            "cell_idx": cell_idx,
            "position": pos,
        }
    
def build_neg_sampling_dist_from_dataset(dataset: SingleCellDataset) -> np.ndarray:
    """
    Build a unigram^0.75 negative sampling distribution from dataset token_id_seqs.
    Returns a 1D numpy array of shape (vocab_size,).
    """
    vocab_size = dataset.vocab_size
    counts = np.zeros(vocab_size, dtype=np.float64)
    for seq in dataset._token_id_seqs:
        if seq.size == 0:
            continue
        local_counts = np.bincount(seq, minlength=vocab_size).astype(np.float64)
        counts += local_counts

    counts[counts <= 0] = 1e-8
    dist = counts ** 0.75
    dist /= dist.sum()
    return dist


class SingleCellVAEDataset(Dataset):
    """
    Simple dataset for VAE training.

    Returns:
      - x_expr: (G,) tensor of expression (e.g., log1p normalized counts)
      - cond_idx: Optional scalar LongTensor (if cond_key is given)
    """

    def __init__(
        self,
        adata_or_path,
        gene_key: str = "gene",
        layer: Optional[str] = None,
        cond_key: Optional[str] = None,
        gene_order: Optional[list[str]] = None,
        transform: str = "log1p",
    ) -> None:
        super().__init__()
        if isinstance(adata_or_path, ad.AnnData):
            adata = adata_or_path
        else:
            adata = ad.read_h5ad(adata_or_path)
        self.adata = adata

        # Choose expression matrix
        if layer is None:
            X = adata.X
        else:
            X = adata.layers[layer]

        X = X.astype(np.float32)
        if transform == "log1p":
            X = np.log1p(X)
        elif transform == "none":
            pass
        else:
            raise ValueError(f"Unsupported transform: {transform}")

        # Ensure dense
        if not isinstance(X, np.ndarray):
            X = X.toarray().astype(np.float32)
        self.X = X  # (N, G)

        # Optionally reorder genes to match a target order (e.g. CBOW vocab order)
        if gene_order is not None:
            # adata.var[gene_key] should list genes
            var_genes = self.adata.var[gene_key].astype(str).to_list()
            idx_map = {g: i for i, g in enumerate(var_genes)}
            indices = [idx_map[g] for g in gene_order]
            self.X = self.X[:, indices]
            self.gene_order = gene_order
        else:
            self.gene_order = self.adata.var[gene_key].astype(str).to_list()

        # Condition
        if cond_key is not None:
            cond_series = self.adata.obs[cond_key].astype("category")
            self.cond_categories = list(cond_series.cat.categories)
            self.cond_idx = cond_series.cat.codes.to_numpy().astype(np.int64)
        else:
            self.cond_categories = None
            self.cond_idx = None

    def __len__(self) -> int:
        return self.X.shape[0]

    def __getitem__(self, idx: int):
        x = torch.from_numpy(self.X[idx])  # (G,)
        if self.cond_idx is not None:
            c = torch.tensor(self.cond_idx[idx], dtype=torch.long)
            return {"x_expr": x, "cond_idx": c}
        else:
            return {"x_expr": x}
