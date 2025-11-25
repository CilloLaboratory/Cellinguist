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
    num_negatives : int
        Number of negative samples per target.
    samples_per_cell : int
        Logical number of CBOW samples per cell per "epoch". The dataset
        length will be n_cells * samples_per_cell, and each __getitem__
        will draw one random context window within the chosen cell.
    """

    window_size: int = 5
    num_negatives: int = 10
    samples_per_cell: int = 1

class CBOWPairsDataset(Dataset):
    """
    Dataset that wraps SingleCellDataset and yields CBOW training triples:
      - target_ids:   LongTensor of shape (1,)
      - context_ids:  LongTensor of shape (2 * window_size,)
      - negative_ids: LongTensor of shape (num_negatives,)

    Each __getitem__:
      - Picks a cell (based on index and samples_per_cell)
      - Randomly selects a valid target position with a full context window
      - Constructs the context window
      - Samples negative tokens from a precomputed negative-sampling distribution
    """

    def __init__(
        self,
        sc_dataset: SingleCellDataset,
        config: CBOWPairsConfig,
        neg_sampling_dist: Optional[np.ndarray] = None,
        rng: Optional[np.random.Generator] = None,
    ) -> None:
        """
        Parameters
        ----------
        sc_dataset : SingleCellDataset
            Underlying single-cell dataset with token_id sequences.
        config : CBOWPairsConfig
            Configuration for window size, negatives, and samples_per_cell.
        neg_sampling_dist : Optional[np.ndarray], default None
            1D array of shape (vocab_size,) with probabilities used for
            negative sampling. If None, a unigram^0.75 distribution is
            estimated from sc_dataset._token_id_seqs.
        rng : Optional[np.random.Generator], default None
            NumPy RNG for sampling target positions and negatives. If None,
            a new default_rng() is created.
        """
        super().__init__()
        self.sc_dataset = sc_dataset
        self.window_size = int(config.window_size)
        self.num_negatives = int(config.num_negatives)
        self.samples_per_cell = int(config.samples_per_cell)
        self.rng = rng if rng is not None else np.random.default_rng()

        self.vocab_size = sc_dataset.vocab_size
        self.token_id_seqs = sc_dataset._token_id_seqs  # List[np.ndarray]
        self.n_cells = len(self.token_id_seqs)

        # Build negative sampling distribution if not provided
        if neg_sampling_dist is None:
            self.neg_sampling_dist = self._build_neg_sampling_dist()
        else:
            if neg_sampling_dist.shape[0] != self.vocab_size:
                raise ValueError(
                    f"neg_sampling_dist length {neg_sampling_dist.shape[0]} "
                    f"does not match vocab_size {self.vocab_size}"
                )
            self.neg_sampling_dist = neg_sampling_dist

        # Precompute which cells have at least one valid CBOW position
        self._valid_cells = self._find_valid_cells()
        if len(self._valid_cells) == 0:
            raise ValueError(
                "No cells have enough tokens for the chosen window_size. "
                "Reduce window_size or check tokenization."
            )

    def _build_neg_sampling_dist(self) -> np.ndarray:
        """
        Build a unigram^0.75 negative sampling distribution from token_id_seqs.
        """
        counts = np.zeros(self.vocab_size, dtype=np.float64)
        for seq in self.token_id_seqs:
            if seq.size == 0:
                continue
            local_counts = np.bincount(seq, minlength=self.vocab_size).astype(
                np.float64
            )
            counts += local_counts

        counts[counts <= 0] = 1e-8
        dist = counts ** 0.75
        dist /= dist.sum()
        return dist

    def _find_valid_cells(self) -> np.ndarray:
        """
        Return indices of cells that have at least one valid target position
        with a full context window on both sides.
        """
        valid = []
        min_len = 2 * self.window_size + 1
        for i, seq in enumerate(self.token_id_seqs):
            if seq.size >= min_len:
                valid.append(i)
        return np.asarray(valid, dtype=np.int64)

    def __len__(self) -> int:
        """
        The logical length is n_cells * samples_per_cell.

        Each __getitem__ will sample one CBOW window from some cell;
        this does NOT enumerate all possible windows in the corpus,
        but provides a stochastic approximation that scales well.
        """
        return self.n_cells * self.samples_per_cell

    def _sample_cell_and_position(self) -> (int, int):
        """
        Sample a valid cell and a valid target position within that cell.
        """
        min_len = 2 * self.window_size + 1

        # Try a few times to find a valid cell; fall back if pathological
        for _ in range(10):
            cell_idx = int(self.rng.choice(self._valid_cells))
            seq = self.token_id_seqs[cell_idx]
            L = seq.size
            if L >= min_len:
                # valid target positions: [window_size, L - window_size - 1]
                pos = self.rng.integers(self.window_size, L - self.window_size)
                return cell_idx, int(pos)

        # If we somehow fail repeatedly, just pick the first valid cell deterministically
        cell_idx = int(self._valid_cells[0])
        seq = self.token_id_seqs[cell_idx]
        L = seq.size
        pos = max(self.window_size, min(L - self.window_size - 1, self.window_size))
        return cell_idx, int(pos)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Return a single CBOW training triple as a dict:

            {
                "target_ids":   LongTensor[1],
                "context_ids":  LongTensor[2 * window_size],
                "negative_ids": LongTensor[num_negatives],
                "cell_idx":     int,
                "position":     int,
            }

        When batched by DataLoader, you'll get:
            - target_ids:   (B,)
            - context_ids:  (B, 2 * window_size)
            - negative_ids: (B, num_negatives)
        """
        # Optionally, map idx -> cell_idx deterministically:
        # cell_idx = idx // self.samples_per_cell
        # but we then must ensure that cell_idx has enough tokens.
        # Here we just sample randomly among valid cells for robustness.
        cell_idx, pos = self._sample_cell_and_position()
        seq = self.token_id_seqs[cell_idx]
        L = seq.size

        w = self.window_size
        target = int(seq[pos])

        left_context = seq[pos - w : pos]
        right_context = seq[pos + 1 : pos + 1 + w]
        assert left_context.size == w, f"Left context size mismatch for cell {cell_idx}"
        assert right_context.size == w, f"Right context size mismatch for cell {cell_idx}"

        context = np.concatenate([left_context, right_context], axis=0)  # (2w,)

        # Sample negatives (independent of target/context)
        neg_ids = self.rng.choice(
            self.vocab_size,
            size=self.num_negatives,
            replace=True,
            p=self.neg_sampling_dist,
        )

        sample: Dict[str, Any] = {
            "target_ids": torch.tensor(target, dtype=torch.long),
            "context_ids": torch.from_numpy(context.astype(np.int64)),
            "negative_ids": torch.from_numpy(neg_ids.astype(np.int64)),
            "cell_idx": cell_idx,
            "position": pos,
        }
        return sample    