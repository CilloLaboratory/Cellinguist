# src/cellinguist_cbow/train/train_cbow.py

from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import DataLoader, TensorDataset

from cellinguist_cbow.config import CBOWConfig  # you'll define this
from cellinguist_cbow.data.datasets import SingleCellDataset
from cellinguist_cbow.models.cbow import (
    CBOWModel,
    cbow_negative_sampling_loss,
    build_cbow_training_pairs,
)
from cellinguist_cbow.embeddings import save_gene_embeddings


def train_cbow(
    config: CBOWConfig,
    dataset: SingleCellDataset,
) -> torch.Tensor:
    """
    Train a CBOWModel on token sequences from a SingleCellDataset.

    Parameters
    ----------
    config : CBOWConfig
        Configuration object containing CBOW hyperparameters.
    dataset : SingleCellDataset
        Dataset that provides token_id sequences and vocab size.

    Returns
    -------
    embeddings : torch.Tensor
        Learned input embedding matrix of shape (vocab_size, emb_dim)
        on CPU.
    """
    device = torch.device(config.device)

    # ------------------------------------------------------------------
    # 1. Extract token sequences & vocab info from dataset
    # ------------------------------------------------------------------
    # token_id_seqs: List[np.ndarray], one sequence per cell
    token_id_seqs = dataset._token_id_seqs  # or expose via a property later
    vocab_size = dataset.vocab_size

    # ------------------------------------------------------------------
    # 2. Build CBOW training triples (target, context, negatives)
    # ------------------------------------------------------------------
    target_ids, context_ids, negative_ids = build_cbow_training_pairs(
        token_id_seqs=token_id_seqs,
        vocab_size=vocab_size,
        window_size=config.window_size,
        num_negatives=config.num_negatives,
        neg_sampling_dist=None,  # let function build unigram^0.75 dist
        rng=None,                # let function create its own RNG
    )

    # Wrap into a TensorDataset so DataLoader can batch it
    cbow_dataset = TensorDataset(target_ids, context_ids, negative_ids)

    dataloader = DataLoader(
        cbow_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    # ------------------------------------------------------------------
    # 3. Set up CBOWModel, optimizer, (optional) LR scheduler
    # ------------------------------------------------------------------
    model = CBOWModel(
        vocab_size=vocab_size,
        emb_dim=config.emb_dim,
        use_separate_output=config.use_separate_output,
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.lr,
        weight_decay=config.weight_decay,
    )

    if config.lr_scheduler == "none":
        scheduler = None
    elif config.lr_scheduler == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=config.lr_step_size,
            gamma=config.lr_gamma,
        )
    else:
        raise ValueError(f"Unsupported lr_scheduler: {config.lr_scheduler}")

    # ------------------------------------------------------------------
    # 4. Training loop
    # ------------------------------------------------------------------
    for epoch in range(config.epochs):
        model.train()
        total_loss = 0.0
        total_batches = 0

        for batch in dataloader:
            target_ids_b, context_ids_b, negative_ids_b = batch
            target_ids_b = target_ids_b.to(device)
            context_ids_b = context_ids_b.to(device)
            negative_ids_b = negative_ids_b.to(device)

            pos_logits, neg_logits = model(
                target_ids=target_ids_b,
                context_ids=context_ids_b,
                negative_ids=negative_ids_b,
            )
            loss = cbow_negative_sampling_loss(
                pos_logits,
                neg_logits,
                reduction="mean",
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += float(loss.item())
            total_batches += 1

        avg_loss = total_loss / max(total_batches, 1)
        print(f"[CBOW] Epoch {epoch+1}/{config.epochs} - loss: {avg_loss:.4f}")

        if scheduler is not None:
            scheduler.step()

    # ------------------------------------------------------------------
    # 5. Return learned embeddings (input embeddings)
    # ------------------------------------------------------------------
    embeddings = model.input_emb.weight.detach().cpu()
    return embeddings


def run_cbow_training_from_config(config_path: str) -> None:
    """
    High-level entry point:
      - Load CBOWConfig from a config file
      - Load data into SingleCellDataset
      - Train CBOW model
      - Save embeddings to disk

    Parameters
    ----------
    config_path : str
        Path to a config file (e.g. YAML/JSON) that specifies CBOWConfig and data params.
    """
    from cellinguist_cbow.config import load_config  # you will define this

    # Load a dict-like config and build CBOWConfig
    raw_cfg = load_config(config_path)
    cbow_cfg = CBOWConfig(**raw_cfg["cbow"])

    # Data-related settings (you can structure this however you like)
    data_cfg = raw_cfg["data"]
    adata_path = data_cfg["adata_path"]
    gene_key = data_cfg.get("gene_key", "gene")
    cond_key = data_cfg.get("cond_key", None)
    layer = data_cfg.get("layer", None)
    n_bins = data_cfg.get("n_bins", 20)
    shuffle_tokens = data_cfg.get("shuffle_tokens", True)
    min_expr = data_cfg.get("min_expr", 0.0)
    min_token_count = data_cfg.get("min_token_count", 1)

    # Build dataset
    dataset = SingleCellDataset(
        adata_or_path=adata_path,
        gene_key=gene_key,
        cond_key=cond_key,
        layer=layer,
        n_bins=n_bins,
        shuffle_tokens=shuffle_tokens,
        min_expr=min_expr,
        token_to_id=None,  # build vocab from this dataset
        min_token_count=min_token_count,
    )

    # Train CBOW embeddings
    embeddings = train_cbow(cbow_cfg, dataset)

    # Save embeddings
    out_path = Path(raw_cfg.get("output", {}).get("embeddings_path", "gene_embeddings.pt"))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_gene_embeddings(str(out_path), embeddings)

    print(f"[CBOW] Saved embeddings to: {out_path}")
