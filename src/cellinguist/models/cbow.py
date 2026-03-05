from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class CBOWModel(nn.Module):
    """
    Continuous Bag-of-Words (CBOW) model with negative sampling.

    Holds:
      - input embedding (for context tokens)
      - optionally a separate output embedding (for targets/negatives), as in classic word2vec.

    Forward:
      - given target_ids, context_ids, negative_ids, compute positive and negative logits
        for the negative-sampling objective.

    Typical usage:
        pos_logits, neg_logits = model(target_ids, context_ids, negative_ids)
        loss = cbow_negative_sampling_loss(pos_logits, neg_logits)
    """

    def __init__(
        self,
        vocab_size: int,
        emb_dim: int,
        use_separate_output: bool = True,
    ) -> None:
        super().__init__()

        self.vocab_size = int(vocab_size)
        self.emb_dim = int(emb_dim)
        self.use_separate_output = bool(use_separate_output)

        # Embeddings for input (context) tokens
        self.input_emb = nn.Embedding(self.vocab_size, self.emb_dim)

        # Embeddings for output (target/negative) tokens
        if self.use_separate_output:
            self.output_emb = nn.Embedding(self.vocab_size, self.emb_dim)
        else:
            self.output_emb = self.input_emb  # weight tying

        # Initialize weights (you can later swap in pretrained weights if desired)
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        """
        Initialize embeddings with a small random normal distribution.
        """
        nn.init.normal_(self.input_emb.weight, mean=0.0, std=0.02)
        if self.use_separate_output and self.output_emb is not self.input_emb:
            nn.init.normal_(self.output_emb.weight, mean=0.0, std=0.02)

    def forward(
        self,
        target_ids: torch.Tensor,
        context_ids: torch.Tensor,
        negative_ids: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute positive and negative logits for CBOW with negative sampling.

        Parameters
        ----------
        target_ids : LongTensor of shape (B,)
            Target token IDs.
        context_ids : LongTensor of shape (B, C)
            Context token IDs for each target (fixed context window size C).
        negative_ids : LongTensor of shape (B, K)
            Negative sample token IDs for each target (K negatives per target).

        Returns
        -------
        pos_logits : FloatTensor of shape (B,)
            Logits for positive (target, context) pairs.
        neg_logits : FloatTensor of shape (B, K)
            Logits for negative (target, negative) pairs.
        """
        # (B, C, D)
        context_embs = self.input_emb(context_ids)
        # CBOW: mean over context window dimension
        # (B, D)
        context_vec = context_embs.mean(dim=1)

        # (B, D)
        target_embs = self.output_emb(target_ids)
        # Positive logits: dot product between context vector and target embedding
        # (B,)
        pos_logits = (context_vec * target_embs).sum(dim=-1)

        # Negative embeddings: (B, K, D)
        neg_embs = self.output_emb(negative_ids)
        # Negative logits: dot product between context vector and each negative embedding
        # (B, K)
        neg_logits = torch.einsum("bd,bkd->bk", context_vec, neg_embs)

        return pos_logits, neg_logits


def cbow_negative_sampling_loss(
    pos_logits: torch.Tensor,
    neg_logits: torch.Tensor,
    reduction: str = "mean",
) -> torch.Tensor:
    """
    Standard negative-sampling loss for CBOW / word2vec.

    Loss per example:
        -log sigmoid(pos_logit) - sum_j log sigmoid(-neg_logit_j)

    Parameters
    ----------
    pos_logits : FloatTensor of shape (B,)
        Positive logits from CBOWModel.
    neg_logits : FloatTensor of shape (B, K)
        Negative logits from CBOWModel.
    reduction : {"mean", "sum", "none"}, default "mean"
        How to aggregate loss over the batch.

    Returns
    -------
    loss : FloatTensor
        Scalar (if reduction != "none") or per-example loss (if reduction == "none").
    """
    # (B,)
    pos_loss = F.logsigmoid(pos_logits)
    # (B, K)
    neg_loss = F.logsigmoid(-neg_logits).sum(dim=1)

    # Negative of the sum of log-likelihoods
    loss = -(pos_loss + neg_loss)  # (B,)

    if reduction == "mean":
        return loss.mean()
    elif reduction == "sum":
        return loss.sum()
    elif reduction == "none":
        return loss
    else:
        raise ValueError(f"Unsupported reduction: {reduction}")


def build_cbow_training_pairs(
    token_id_seqs: Sequence[np.ndarray],
    vocab_size: int,
    window_size: int,
    num_negatives: int,
    neg_sampling_dist: Optional[np.ndarray] = None,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Build CBOW training triples (target, context, negatives) from token ID sequences.

    Simplifying assumption:
        - We only use positions with a *full* context window on both sides,
          i.e. indices i in [window_size, L - window_size - 1].
        - This keeps context length fixed at C = 2 * window_size, so no padding is needed.

    Parameters
    ----------
    token_id_seqs : Sequence[np.ndarray]
        List of 1D arrays, one per cell, containing token IDs (e.g. from SingleCellDataset).
    vocab_size : int
        Size of the vocabulary (max token ID + 1).
    window_size : int
        Context window radius. For each target position i, the context is
        tokens in [i-window_size..i-1] and [i+1..i+window_size], total length 2*window_size.
    num_negatives : int
        Number of negative samples per (target, context) pair.
    neg_sampling_dist : Optional[np.ndarray], default None
        1D array of shape (vocab_size,) with probabilities for negative sampling.
        If None, a unigram^0.75 distribution is built from `token_id_seqs`.
    rng : Optional[np.random.Generator], default None
        NumPy random generator. If None, a new default_rng() is created.

    Returns
    -------
    target_ids : LongTensor of shape (N,)
    context_ids : LongTensor of shape (N, 2*window_size)
    negative_ids : LongTensor of shape (N, num_negatives)
    """
    if rng is None:
        rng = np.random.default_rng()

    # Build negative sampling distribution if not provided
    if neg_sampling_dist is None:
        counts = np.zeros(vocab_size, dtype=np.float64)
        for seq in token_id_seqs:
            # assume seq is np.ndarray of ints
            if seq.size > 0:
                # bincount for this sequence
                local_counts = np.bincount(seq, minlength=vocab_size).astype(
                    np.float64
                )
                counts += local_counts
        # Avoid division by zero
        counts[counts <= 0] = 1e-8
        # Mikolov's unigram^0.75 distribution
        neg_sampling_dist = counts**0.75
        neg_sampling_dist /= neg_sampling_dist.sum()

    window_size = int(window_size)
    num_negatives = int(num_negatives)

    targets: List[int] = []
    contexts: List[np.ndarray] = []

    C = 2 * window_size  # context length

    # Build (target, context) pairs
    for seq in token_id_seqs:
        L = seq.size
        if L <= 2 * window_size:
            continue  # no position has full context
        for i in range(window_size, L - window_size):
            target = int(seq[i])
            left_context = seq[i - window_size : i]
            right_context = seq[i + 1 : i + 1 + window_size]
            assert left_context.size == window_size
            assert right_context.size == window_size

            context = np.concatenate([left_context, right_context], axis=0)
            assert context.size == C

            targets.append(target)
            contexts.append(context)

    if len(targets) == 0:
        raise ValueError(
            "No CBOW training pairs could be constructed. "
            "Check token_id_seqs and window_size."
        )

    # Convert to tensors
    target_ids = torch.tensor(targets, dtype=torch.long)
    context_ids_np = np.stack(contexts, axis=0)  # (N, C)
    context_ids = torch.from_numpy(context_ids_np).long()

    # Sample negatives: shape (N, num_negatives)
    N = target_ids.size(0)
    # Sample from neg_sampling_dist (same distribution for all examples)
    neg_ids_np = rng.choice(
        vocab_size, size=(N, num_negatives), replace=True, p=neg_sampling_dist
    )
    negative_ids = torch.from_numpy(neg_ids_np).long()

    return target_ids, context_ids, negative_ids


def subsample_frequent_genes(
    gene_counts: np.ndarray,
    threshold: float = 1e-5,
) -> np.ndarray:
    """
    Compute keep probabilities for subsampling very frequent tokens (genes),
    following the word2vec heuristic.

    Mikolov et al. define:
        P(discard w) = 1 - sqrt(t / f(w))
    where f(w) is the relative frequency of w, and t is a small threshold ~ 1e-5.
    Here we return P(keep w) = 1 - P(discard w) = sqrt(t / f(w)).

    Parameters
    ----------
    gene_counts : np.ndarray
        1D array of raw counts per token (length = vocab_size).
    threshold : float, default 1e-5
        Subsampling threshold t.

    Returns
    -------
    keep_probs : np.ndarray
        1D array of keep probabilities for each token. Values clipped to [0, 1].
    """
    counts = gene_counts.astype(np.float64)
    total = counts.sum()
    if total <= 0:
        raise ValueError("Total gene_counts must be > 0.")
    freqs = counts / total  # relative frequency f(w)

    # Avoid division by zero
    eps = 1e-12
    freqs = np.maximum(freqs, eps)

    # P(keep) = sqrt(t / f(w)), clipped to [0, 1]
    keep_probs = np.sqrt(threshold / freqs)
    keep_probs = np.clip(keep_probs, 0.0, 1.0)
    return keep_probs


@dataclass
class CBOWTrainerConfig:
    """
    Minimal configuration for CBOW training.

    Attributes
    ----------
    lr : float
        Learning rate.
    epochs : int
        Number of training epochs.
    device : str
        Device string, e.g. "cpu" or "cuda".
    """

    lr: float = 1e-3
    epochs: int = 5
    device: str = "cpu"


class CBOWTrainer:
    """
    Minimal trainer for CBOWModel.

    Expects a DataLoader that yields batches with:
      - batch["target_ids"]: LongTensor of shape (B,)
      - batch["context_ids"]: LongTensor of shape (B, C)
      - batch["negative_ids"]: LongTensor of shape (B, K)
    """

    def __init__(
        self,
        model: CBOWModel,
        config: CBOWTrainerConfig,
    ) -> None:
        self.model = model
        self.config = config
        self.device = torch.device(config.device)
        self.model.to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config.lr)

    def train_epoch(self, dataloader: Iterable[Dict[str, torch.Tensor]]) -> float:
        """
        Train the model for one epoch.

        Parameters
        ----------
        dataloader : Iterable of dict
            Each batch dict must contain 'target_ids', 'context_ids', 'negative_ids'.

        Returns
        -------
        avg_loss : float
            Mean training loss for the epoch.
        """
        self.model.train()
        total_loss = 0.0
        total_batches = 0

        for batch in dataloader:
            target_ids = batch["target_ids"].to(self.device)
            context_ids = batch["context_ids"].to(self.device)
            negative_ids = batch["negative_ids"].to(self.device)

            pos_logits, neg_logits = self.model(target_ids, context_ids, negative_ids)
            loss = cbow_negative_sampling_loss(pos_logits, neg_logits, reduction="mean")

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += float(loss.item())
            total_batches += 1

        if total_batches == 0:
            return 0.0
        return total_loss / total_batches

    def save_embeddings(self, path: str) -> None:
        """
        Save the learned input embeddings to disk.

        Parameters
        ----------
        path : str
            Destination path for a tensor file (e.g. "gene_embeddings.pt").
        """
        # We treat the input embeddings as the learned representation
        weight = self.model.input_emb.weight.detach().cpu()
        torch.save(weight, path)
