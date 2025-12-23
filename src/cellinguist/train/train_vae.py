from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader
import anndata as ad

from cellinguist.config import VAETrainConfig, load_yaml
from cellinguist.data.datasets import SingleCellVAEDataset
from cellinguist.models.vae import (
    CBOWCellEncoder,
    ZINBExpressionDecoder,
    GeneVAE,
    zinb_negative_log_likelihood,
    kl_divergence_normal,
)
from cellinguist.utils.vae_io import (
    set_seed,
    load_gene_embeddings_tsv,
    intersect_genes_in_embedding_order,
    subset_embeddings,
    save_vae_checkpoint,
    load_vae_checkpoint,
)


def train_vae(cfg: VAETrainConfig) -> str:
    set_seed(cfg.seed)
    device = torch.device(cfg.device)

    # --- Load gene embeddings
    genes_from_emb, emb_full = load_gene_embeddings_tsv(cfg.gene_emb_tsv)

    # --- Probe expression genes (lightweight: read once, build a dataset)
    adata = ad.read_h5ad(cfg.adata_path)
    ds_probe = SingleCellVAEDataset(
        adata_or_path=adata,
        gene_key=cfg.gene_key,
        layer=cfg.layer,
        cond_key=cfg.cond_key,
        transform="none",
    )
    genes_expr = ds_probe.gene_order

    print(f"Expression dataset: {len(genes_expr)} genes before intersection")

    # --- Align genes in embedding order
    genes_common = intersect_genes_in_embedding_order(genes_from_emb, genes_expr)
    emb = subset_embeddings(genes_from_emb, emb_full, genes_common)

    # --- Rebuild dataset with aligned gene order
    vae_dataset = SingleCellVAEDataset(
        adata_or_path=adata,
        gene_key=cfg.gene_key,
        layer=cfg.layer,
        cond_key=cfg.cond_key,
        gene_order=genes_common,
        transform="none",
    )
    n_cells, n_genes = vae_dataset.X.shape
    print(f"VAE dataset after alignment: {n_cells} cells, {n_genes} genes")
    assert n_genes == emb.shape[0]

    n_conditions = (
        len(vae_dataset.cond_categories) if vae_dataset.cond_categories is not None else None
    )

    # --- Build model
    encoder = CBOWCellEncoder(
        gene_embeddings=emb,
        latent_dim=cfg.latent_dim,
        hidden_dim=cfg.hidden_dim,
        n_hidden_layers=cfg.n_hidden_layers,
        n_conditions=n_conditions,
        cond_emb_dim=cfg.cond_emb_dim,
        freeze_gene_embeddings=cfg.freeze_gene_embeddings,
        input_transform=cfg.input_transform,
    )
    decoder = ZINBExpressionDecoder(
        n_genes=n_genes,
        latent_dim=cfg.latent_dim,
        hidden_dim=cfg.hidden_dim,
        n_hidden_layers=cfg.n_hidden_layers,
        n_conditions=n_conditions,
        cond_emb_dim=cfg.cond_emb_dim,
    )
    model = GeneVAE(encoder, decoder).to(device)

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )

    # --- Resume if requested
    start_epoch = 0
    if cfg.resume_from:
        ckpt = load_vae_checkpoint(cfg.resume_from, model, optimizer, map_location=device)
        start_epoch = int(ckpt.get("epoch", -1)) + 1

        # Optional: sanity check genes_common match
        ckpt_genes = ckpt.get("genes_common", None)
        if ckpt_genes is not None and ckpt_genes != genes_common:
            raise ValueError("genes_common mismatch between checkpoint and current data/embeddings.")

    # --- DataLoader
    dl = DataLoader(
        vae_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    # --- Checkpoint naming
    ckpt_dir = Path(cfg.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_last = str(ckpt_dir / f"{cfg.run_name}_last.ckpt")

    # --- Train loop
    model.train()
    for epoch in range(start_epoch, cfg.epochs):
        total = 0.0
        nb = 0

        for batch in dl:
            x = batch["x_expr"].to(device, non_blocking=True)
            cond_idx = batch.get("cond_idx", None)
            if cond_idx is not None:
                cond_idx = cond_idx.to(device, non_blocking=True)

            mu_z, logvar_z = model.encode(x, cond_idx)
            z = GeneVAE.reparameterize(mu_z, logvar_z)

            mu, theta, pi = decoder(z, cond_idx)

            recon = zinb_negative_log_likelihood(x, mu, theta, pi, reduction="mean")
            kl = kl_divergence_normal(mu_z, logvar_z, reduction="mean")
            loss = recon + cfg.kl_weight * kl

            optimizer.zero_grad()
            loss.backward()
            if cfg.grad_clip_norm and cfg.grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip_norm)
            optimizer.step()

            total += float(loss.item())
            nb += 1

        print(f"[VAE] Epoch {epoch+1}/{cfg.epochs} loss={total/max(nb,1):.4f}")

        # Save checkpoints
        if cfg.save_every > 0 and ((epoch + 1) % cfg.save_every == 0):
            save_vae_checkpoint(
                ckpt_last,
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                genes_common=genes_common,
                config_snapshot=asdict(cfg),
                gene_emb_source=cfg.gene_emb_tsv,
            )

    # Final save
    save_vae_checkpoint(
        ckpt_last,
        model=model,
        optimizer=optimizer,
        epoch=cfg.epochs - 1,
        genes_common=genes_common,
        config_snapshot=asdict(cfg),
        gene_emb_source=cfg.gene_emb_tsv,
    )
    return ckpt_last


def run_vae_training_from_config(config_path: str) -> None:
    d = load_yaml(config_path)
    cfg = VAETrainConfig(**d)
    train_vae(cfg)


def main() -> None:
    ap = argparse.ArgumentParser(description="Traine VAE from config file.")
    ap.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML/JSON config file."
    )
    args = ap.parse_args()
    run_vae_training_from_config(args.config)
