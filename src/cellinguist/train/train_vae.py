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
    PerceiverCellEncoder,
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
    estimate_gene_means,
    inv_softplus,
    logit,
)

def train_vae(cfg: VAETrainConfig) -> str:
    set_seed(cfg.seed)
    device = torch.device(cfg.device)
    encoder_type = str(cfg.encoder_type).lower()
    if encoder_type not in {"cbow", "perceiver"}:
        raise ValueError(f"Unsupported encoder_type: {cfg.encoder_type}. Use 'cbow' or 'perceiver'.")

    # --- Load expression data once
    adata = ad.read_h5ad(cfg.adata_path)

    if encoder_type == "cbow":
        if not cfg.gene_emb_tsv:
            raise ValueError("gene_emb_tsv must be provided when encoder_type='cbow'.")

        genes_from_emb, emb_full = load_gene_embeddings_tsv(cfg.gene_emb_tsv)
        ds_probe = SingleCellVAEDataset(
            adata_or_path=adata,
            gene_key=cfg.gene_key,
            layer=cfg.layer,
            cond_key=cfg.cond_key,
            transform="none",
        )
        genes_expr = ds_probe.gene_order
        print(f"Expression dataset: {len(genes_expr)} genes before intersection")

        genes_common = intersect_genes_in_embedding_order(genes_from_emb, genes_expr)
        emb = subset_embeddings(genes_from_emb, emb_full, genes_common)

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
        gene_emb_source = cfg.gene_emb_tsv
    else:
        vae_dataset = SingleCellVAEDataset(
            adata_or_path=adata,
            gene_key=cfg.gene_key,
            layer=cfg.layer,
            cond_key=cfg.cond_key,
            transform="none",
        )
        genes_common = vae_dataset.gene_order
        n_cells, n_genes = vae_dataset.X.shape
        print(f"VAE dataset: {n_cells} cells, {n_genes} genes (Perceiver encoder)")
        emb = None
        gene_emb_source = ""

    n_conditions = (
        len(vae_dataset.cond_categories) if vae_dataset.cond_categories is not None else None
    )

    # --- Build model
    if encoder_type == "cbow":
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
    else:
        encoder = PerceiverCellEncoder(
            n_genes=n_genes,
            latent_dim=cfg.latent_dim,
            hidden_dim=cfg.hidden_dim,
            n_hidden_layers=cfg.n_hidden_layers,
            n_conditions=n_conditions,
            cond_emb_dim=cfg.cond_emb_dim,
            input_transform=cfg.input_transform,
            perceiver_d_model=cfg.perceiver_d_model,
            perceiver_num_latents=cfg.perceiver_num_latents,
            perceiver_num_cross_attn_heads=cfg.perceiver_num_cross_attn_heads,
            perceiver_num_self_attn_heads=cfg.perceiver_num_self_attn_heads,
            perceiver_num_self_attn_layers=cfg.perceiver_num_self_attn_layers,
            perceiver_ff_mult=cfg.perceiver_ff_mult,
            perceiver_dropout=cfg.perceiver_dropout,
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

    # --- Decoder initialization (only if NOT resuming)
    if not cfg.resume_from:
        # 1) Initialize theta (gene-wise)
        if hasattr(model.decoder, "log_theta"):
            theta_init = torch.full((n_genes,), float(cfg.decoder_theta_init), dtype=torch.float32)
            model.decoder.log_theta.data = inv_softplus(theta_init).to(model.decoder.log_theta.data.device)

        # 2) Initialize pi head bias to logit(pi_init)
        # Your decoder defines `mlp_pi = nn.Linear(..., n_genes)` so we can set bias.
        if hasattr(model.decoder, "mlp_pi"):
            # Access final Linear layer inside mlp_pi
            pi_linear = model.decoder.mlp_pi.net[-1]

            # Set bias so sigmoid(bias) ≈ pi_init
            pi_bias = logit(cfg.decoder_pi_init)
            pi_linear.bias.data.fill_(pi_bias)

            # Optional but recommended: start with zero weights
            torch.nn.init.zeros_(pi_linear.weight)

        # 3) Initialize mu head bias
        if hasattr(model.decoder, "mlp_mu"):
            mu_linear = model.decoder.mlp_mu.net[-1]

            if cfg.decoder_mu_init == "data_mean":
                mean_x = estimate_gene_means(
                    vae_dataset,
                    max_cells=cfg.decoder_init_n_cells,
                    batch_size=cfg.decoder_init_batch_size,
                    num_workers=cfg.decoder_init_num_workers,  # I recommend 0 here
                    device=None,
                )
                mean_x = torch.clamp(
                    mean_x,
                    min=1e-4,
                    max=cfg.decoder_mu_init_cap,
                )

                mu_bias = inv_softplus(mean_x).to(mu_linear.bias.device)
                mu_linear.bias.data.copy_(mu_bias)

                # Optional but recommended
                torch.nn.init.zeros_(mu_linear.weight)

            elif cfg.decoder_mu_init == "constant":
                mean0 = torch.tensor(cfg.decoder_mu_init_constant)
                mean0 = torch.clamp(mean0,
                    min=cfg.decoder_mu_init_eps,
                    max=cfg.decoder_mu_init_cap)
                b0 = inv_softplus(mean0).item()
                mu_linear.bias.data.fill_(b0)
                torch.nn.init.zeros_(mu_linear.weight)

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
            # if cfg.grad_clip_norm and cfg.grad_clip_norm > 0:
            #     torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip_norm)
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
                gene_emb_source=gene_emb_source,
            )

    # Final save
    save_vae_checkpoint(
        ckpt_last,
        model=model,
        optimizer=optimizer,
        epoch=cfg.epochs - 1,
        genes_common=genes_common,
        config_snapshot=asdict(cfg),
        gene_emb_source=gene_emb_source,
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
