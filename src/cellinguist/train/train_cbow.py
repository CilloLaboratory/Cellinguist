from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader

from cellinguist.config import CBOWConfig, load_config
from cellinguist.data.datasets import (
    SingleCellDataset,
    CBOWPairsDataset,
    CBOWPairsConfig,
    build_neg_sampling_dist_from_dataset
)
from cellinguist.models.cbow import (
    CBOWModel,
    cbow_negative_sampling_loss,
)
from cellinguist.embeddings import save_gene_embeddings

def train_cbow(
    config: CBOWConfig,
    dataset: SingleCellDataset,
) -> torch.Tensor:
    """
    Train a CBOWModel on token sequences from a SingleCellDataset,
    using CBOWPairsDataset to generate (target, context, negatives)
    on the fly.

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
    # 1. Wrap SingleCellDataset in CBOWPairsDataset
    # ------------------------------------------------------------------
    pairs_cfg = CBOWPairsConfig(
        window_size=config.window_size,
        samples_per_cell=config.samples_per_cell,
    )
    cbow_pairs_ds = CBOWPairsDataset(
        sc_dataset=dataset,
        config=pairs_cfg,
        rng=None,                # let it create its own RNG
    )

    dataloader = DataLoader(
        cbow_pairs_ds,
        batch_size=config.batch_size,
        shuffle=False,  # dataset is already stochastic; shuffle not essential
        num_workers=config.num_workers,  # adjust as needed
        pin_memory=(device.type == "cuda"),
        persistent_workers=True
    )

    vocab_size = dataset.vocab_size

    # 2. Build negative sampling distribution once, and move to device
    neg_sampling_dist_np = build_neg_sampling_dist_from_dataset(dataset)
    neg_sampling_dist = torch.from_numpy(neg_sampling_dist_np).to(
        device=device, dtype=torch.float32
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
    # 3. Training loop
    # ------------------------------------------------------------------
    for epoch in range(config.epochs):
        model.train()
        total_loss = 0.0
        total_batches = 0

        for batch in dataloader:
            # batch keys: "target_ids", "context_ids", "cell_idx", "position"
            target_ids = batch["target_ids"].to(device, non_blocking=True)       # (B,)
            context_ids = batch["context_ids"].to(device, non_blocking=True)     # (B, 2 * window_size)
            
            B = target_ids.size(0)
            K = config.num_negatives

            # Sample negatives with torch.multinomial on the device
            # neg_sampling_dist is (vocab_size,)
            negative_ids = torch.multinomial(
                neg_sampling_dist,
                num_samples=B * K,
                replacement=True,
            ).view(B, K)   # (B, K

            pos_logits, neg_logits = model(
                target_ids=target_ids,
                context_ids=context_ids,
                negative_ids=negative_ids,
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
    # 4. Return learned embeddings (input embeddings)
    # ------------------------------------------------------------------
    embeddings = model.input_emb.weight.detach().cpu()
    return embeddings


def run_cbow_training_from_config(config_path: str) -> None:
    """
    High-level entry point:
      - Load CBOWConfig from a config file
      - Load data into SingleCellDataset
      - Train CBOW model with CBOWPairsDataset
      - Save embeddings to disk

    Parameters
    ----------
    config_path : str
        Path to a config file (e.g. YAML/JSON) that specifies CBOWConfig and data params.
    """
    # Load a dict-like config and build CBOWConfig
    raw_cfg = load_config(config_path)
    cbow_cfg = CBOWConfig(**raw_cfg["cbow"])

    # Data-related settings
    data_cfg = raw_cfg["data"]
    adata_path = data_cfg["adata_path"]
    gene_key = data_cfg.get("gene_key", "gene")
    cond_key = data_cfg.get("cond_key", None)
    layer = data_cfg.get("layer", None)
    n_bins = data_cfg.get("n_bins", 20)
    shuffle_tokens = data_cfg.get("shuffle_tokens", True)
    min_expr = data_cfg.get("min_expr", 0.0)
    min_token_count = data_cfg.get("min_token_count", 1)

    # Build SingleCellDataset (constructs vocab and token_id_seqs)
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

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Train CBOW embeddings from config file.")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML/JSON config file."
    )
    args = parser.parse_args()
    run_cbow_training_from_config(args.config)

if __name__ == "__main__":
    main()
