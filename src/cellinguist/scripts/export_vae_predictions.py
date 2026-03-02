from __future__ import annotations

import argparse
import gzip
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
import anndata as ad

from cellinguist.config import VAEExportConfig, load_yaml
from cellinguist.data.datasets import SingleCellVAEDataset
from cellinguist.models.vae import (
    CBOWCellEncoder,
    PerceiverCellEncoder,
    ZINBExpressionDecoder,
    GeneVAE,
)
from cellinguist.utils.vae_io import (
    load_gene_embeddings_tsv,
    subset_embeddings,
    load_vae_checkpoint,
)


def export_predictions(cfg: VAEExportConfig) -> None:
    device = torch.device(cfg.device)

    # Read checkpoint raw for architecture + gene ordering metadata
    ckpt_raw = torch.load(cfg.checkpoint_path, map_location="cpu")
    genes_common = ckpt_raw["genes_common"]
    gene_emb_source = ckpt_raw.get("gene_emb_source", None)
    train_cfg = ckpt_raw.get("config", {})
    encoder_type = str(train_cfg.get("encoder_type", "cbow")).lower()
    if encoder_type not in {"cbow", "perceiver"}:
        raise ValueError(f"Unsupported encoder_type in checkpoint: {encoder_type}")

    # Load adata and align by checkpoint gene order
    adata = ad.read_h5ad(cfg.adata_path)
    ds = SingleCellVAEDataset(
        adata_or_path=adata,
        gene_key=cfg.gene_key,
        layer=cfg.layer,
        cond_key=cfg.cond_key,
        gene_order=genes_common,
        transform="none",
    )
    n_genes = ds.X.shape[1]

    emb = None
    if encoder_type == "cbow":
        emb_path = cfg.gene_emb_tsv or gene_emb_source
        if not emb_path:
            raise ValueError(
                "gene_emb_tsv is required to export with a CBOW checkpoint "
                "when checkpoint does not include a gene_emb_source path."
            )
        genes_from_emb, emb_full = load_gene_embeddings_tsv(emb_path)
        emb = subset_embeddings(genes_from_emb, emb_full, genes_common)
        if emb.shape[0] != n_genes:
            raise ValueError("Checkpoint genes and embedding genes are misaligned.")

    n_conditions = (
        len(ds.cond_categories) if ds.cond_categories is not None else None
    )

    # Rebuild model with same architecture as training (use checkpoint config if present)
    latent_dim = int(train_cfg.get("latent_dim", 32))
    hidden_dim = int(train_cfg.get("hidden_dim", 256))
    n_hidden_layers = int(train_cfg.get("n_hidden_layers", 2))
    cond_emb_dim = int(train_cfg.get("cond_emb_dim", 16))
    input_transform = str(train_cfg.get("input_transform", "log1p"))
    freeze_gene_embeddings = bool(train_cfg.get("freeze_gene_embeddings", True))

    if encoder_type == "cbow":
        encoder = CBOWCellEncoder(
            gene_embeddings=emb,
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            n_hidden_layers=n_hidden_layers,
            n_conditions=n_conditions,
            cond_emb_dim=cond_emb_dim,
            freeze_gene_embeddings=freeze_gene_embeddings,
            input_transform=input_transform,
        )
    else:
        encoder = PerceiverCellEncoder(
            n_genes=n_genes,
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            n_hidden_layers=n_hidden_layers,
            n_conditions=n_conditions,
            cond_emb_dim=cond_emb_dim,
            input_transform=input_transform,
            perceiver_d_model=int(train_cfg.get("perceiver_d_model", 256)),
            perceiver_num_latents=int(train_cfg.get("perceiver_num_latents", 64)),
            perceiver_num_cross_attn_heads=int(train_cfg.get("perceiver_num_cross_attn_heads", 8)),
            perceiver_num_self_attn_heads=int(train_cfg.get("perceiver_num_self_attn_heads", 8)),
            perceiver_num_self_attn_layers=int(train_cfg.get("perceiver_num_self_attn_layers", 4)),
            perceiver_ff_mult=int(train_cfg.get("perceiver_ff_mult", 4)),
            perceiver_dropout=float(train_cfg.get("perceiver_dropout", 0.0)),
        )
    decoder = ZINBExpressionDecoder(
        n_genes=n_genes,
        latent_dim=latent_dim,
        hidden_dim=hidden_dim,
        n_hidden_layers=n_hidden_layers,
        n_conditions=n_conditions,
        cond_emb_dim=cond_emb_dim,
    )
    model = GeneVAE(encoder, decoder).to(device)

    # Load state dicts
    _ = load_vae_checkpoint(cfg.checkpoint_path, model, optimizer=None, map_location=device)

    model.eval()

    max_cells = min(cfg.max_cells, len(ds))
    dl = DataLoader(
        ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    preds = []
    cell_ids = []
    genes = ds.gene_order

    seen = 0
    with torch.no_grad():
        for batch in dl:
            if seen >= max_cells:
                break

            x = batch["x_expr"].to(device, non_blocking=True)
            cond_idx = batch.get("cond_idx", None)
            if cond_idx is not None:
                cond_idx = cond_idx.to(device, non_blocking=True)

            mu_z, logvar_z = model.encode(x, cond_idx)

            # deterministic: z = mu_z
            mu, theta, pi = model.decoder(mu_z, cond_idx)

            mu_np = mu.detach().cpu().numpy()

            # cell ids: infer indices from running position
            b = x.shape[0]
            take = min(b, max_cells - seen)
            preds.append(mu_np[:take])
            # ds.adata.obs_names is aligned to dataset order
            start = seen
            stop = seen + take
            cell_ids.extend(ds.adata.obs_names[start:stop].tolist())

            seen += take

    pred_mat = np.concatenate(preds, axis=0)  # (N, G)

    df = pd.DataFrame(pred_mat, columns=genes)
    df.insert(0, "cell_id", cell_ids)

    out_path = Path(cfg.out_pred_tsv_gz)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(out_path, "wt") as f:
        df.to_csv(f, sep="\t", index=False)

    print(f"[export] Wrote predicted mu to: {out_path}")
    if (
        encoder_type == "cbow"
        and cfg.gene_emb_tsv
        and gene_emb_source is not None
        and gene_emb_source != cfg.gene_emb_tsv
    ):
        print("[export] WARNING: checkpoint gene_emb_source differs from config gene_emb_tsv.")


def run_from_config(config_path: str) -> None:
    d = load_yaml(config_path)
    cfg = VAEExportConfig(**d)
    export_predictions(cfg)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("config", type=str)
    args = ap.parse_args()
    run_from_config(args.config)
