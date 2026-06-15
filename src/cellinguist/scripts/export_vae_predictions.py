from __future__ import annotations

import argparse
import gzip
import json
from dataclasses import fields
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset

from cellinguist.config import VAEExportConfig, load_yaml
from cellinguist.data.dataloaders import collate_vae_batch
from cellinguist.data.datasets import SingleCellVAEDataset
from cellinguist.models.vae import (
    CBOWCellEncoder,
    GeneVAE,
    PerceiverCellEncoder,
    TransformerCellEncoder,
    ZINBExpressionDecoder,
)
from cellinguist.utils.vae_io import (
    load_gene_embeddings_tsv,
    load_vae_checkpoint,
    subset_embeddings,
)


def _resolve_batch_key(batch_key: Optional[str], cond_key: Optional[str]) -> Optional[str]:
    if batch_key is not None and cond_key is not None and batch_key != cond_key:
        raise ValueError(
            f"Both batch_key='{batch_key}' and cond_key='{cond_key}' were provided, but differ."
        )
    return batch_key if batch_key is not None else cond_key


def _load_counterfactual_overrides(
    path: str,
    cytokine_keys: list[str],
) -> Dict[str, np.ndarray]:
    df = pd.read_csv(path, sep="\t")
    expected_cols = ["cell_id"] + list(cytokine_keys)
    if df.columns.tolist() != expected_cols:
        raise ValueError(
            "counterfactual_override_path columns must match exactly: "
            f"{expected_cols}. Got: {df.columns.tolist()}"
        )

    cell_ids = df["cell_id"].astype(str)
    dup_mask = cell_ids.duplicated(keep=False)
    if dup_mask.any():
        dupes = sorted(cell_ids[dup_mask].unique().tolist())
        raise ValueError(
            "counterfactual overrides contain duplicate cell_id values. "
            f"Examples: {dupes[:3]}"
        )

    values = df[cytokine_keys].to_numpy(dtype=np.float32)
    return {cid: values[i] for i, cid in enumerate(cell_ids.tolist())}


def _load_checkpoint_export_context(cfg: VAEExportConfig) -> dict[str, Any]:
    ckpt_raw = torch.load(cfg.checkpoint_path, map_location="cpu")
    genes_common = ckpt_raw["genes_common"]
    gene_emb_source = ckpt_raw.get("gene_emb_source", None)
    train_cfg = ckpt_raw.get("config", {})
    encoder_type = str(train_cfg.get("encoder_type", "cbow")).lower()
    if encoder_type not in {"cbow", "perceiver", "transformer"}:
        raise ValueError(f"Unsupported encoder_type in checkpoint: {encoder_type}")

    perturbation_mode = str(train_cfg.get("perturbation_mode", cfg.perturbation_mode)).lower()
    cytokine_keys = train_cfg.get("cytokine_keys", cfg.cytokine_keys) or []
    cytokine_transform = str(train_cfg.get("cytokine_transform", cfg.cytokine_transform)).lower()
    cytokine_missing_policy = str(
        train_cfg.get("cytokine_missing_policy", cfg.cytokine_missing_policy)
    ).lower()
    batch_correction_method = str(train_cfg.get("batch_correction_method", "none")).lower()
    batch_correction_eps = float(train_cfg.get("batch_correction_eps", 1e-8))
    batch_correction_clip_min = float(train_cfg.get("batch_correction_clip_min", 0.1))
    batch_correction_clip_max = float(train_cfg.get("batch_correction_clip_max", 10.0))
    perturb_emb_dim = int(train_cfg.get("perturb_emb_dim", cfg.perturb_emb_dim))
    perturb_condition_encoder = bool(train_cfg.get("perturb_condition_encoder", cfg.perturb_condition_encoder))
    perturb_condition_decoder = bool(train_cfg.get("perturb_condition_decoder", cfg.perturb_condition_decoder))
    if (
        perturbation_mode == "cytokine_vector"
        and not perturb_condition_encoder
        and not perturb_condition_decoder
    ):
        raise ValueError(
            "cytokine_vector mode requires perturb_condition_encoder or "
            "perturb_condition_decoder to be enabled."
        )

    token_index_cache_dir = ""
    token_index_cache_require = False
    transformer_precompute_token_indices = bool(cfg.transformer_precompute_token_indices)
    if encoder_type == "transformer":
        token_index_cache_dir = str(
            cfg.token_index_cache_dir or train_cfg.get("token_index_cache_dir", "") or ""
        )
        if token_index_cache_dir:
            token_index_cache_require = bool(
                train_cfg.get("token_index_cache_require", cfg.token_index_cache_require)
            )
        transformer_precompute_token_indices = bool(
            train_cfg.get(
                "transformer_precompute_token_indices",
                cfg.transformer_precompute_token_indices,
            )
        )

    return {
        "ckpt_raw": ckpt_raw,
        "train_cfg": train_cfg,
        "genes_common": genes_common,
        "gene_emb_source": gene_emb_source,
        "encoder_type": encoder_type,
        "perturbation_mode": perturbation_mode,
        "cytokine_keys": list(cytokine_keys),
        "cytokine_transform": cytokine_transform,
        "cytokine_missing_policy": cytokine_missing_policy,
        "batch_correction_method": batch_correction_method,
        "batch_correction_eps": batch_correction_eps,
        "batch_correction_clip_min": batch_correction_clip_min,
        "batch_correction_clip_max": batch_correction_clip_max,
        "perturb_emb_dim": perturb_emb_dim,
        "perturb_condition_encoder": perturb_condition_encoder,
        "perturb_condition_decoder": perturb_condition_decoder,
        "token_index_cache_dir": token_index_cache_dir,
        "token_index_cache_require": token_index_cache_require,
        "transformer_precompute_token_indices": transformer_precompute_token_indices,
    }


def _build_export_dataset(
    cfg: VAEExportConfig,
    effective_batch_key: Optional[str],
    ctx: dict[str, Any],
) -> SingleCellVAEDataset:
    encoder_type = str(ctx["encoder_type"])
    use_transformer_cache = encoder_type == "transformer" and bool(ctx["token_index_cache_dir"])
    use_transformer_precompute = (
        encoder_type == "transformer"
        and not use_transformer_cache
        and bool(ctx["transformer_precompute_token_indices"])
    )
    return SingleCellVAEDataset(
        adata_or_path=cfg.adata_path,
        gene_key=cfg.gene_key,
        layer=cfg.layer,
        cond_key=effective_batch_key,
        batch_key=effective_batch_key,
        batch_correction_method=ctx["batch_correction_method"],
        batch_correction_eps=ctx["batch_correction_eps"],
        batch_correction_clip_min=ctx["batch_correction_clip_min"],
        batch_correction_clip_max=ctx["batch_correction_clip_max"],
        perturbation_mode=ctx["perturbation_mode"],
        cytokine_keys=ctx["cytokine_keys"],
        cytokine_transform=ctx["cytokine_transform"],
        cytokine_missing_policy=ctx["cytokine_missing_policy"],
        gene_order=ctx["genes_common"],
        transform="none",
        backed=cfg.backed,
        precompute_token_gene_indices=use_transformer_precompute,
        token_min_expr=float(ctx["train_cfg"].get("min_expr_for_token", 0.0)),
        token_max_genes=ctx["train_cfg"].get("max_tokens_per_cell", None),
        token_index_cache_dir=(ctx["token_index_cache_dir"] if use_transformer_cache else None),
        token_index_cache_require=bool(ctx["token_index_cache_require"] and use_transformer_cache),
    )


def _build_model_for_export(
    cfg: VAEExportConfig,
    ctx: dict[str, Any],
    ds: SingleCellVAEDataset,
    device: torch.device,
) -> GeneVAE:
    train_cfg = ctx["train_cfg"]
    encoder_type = str(ctx["encoder_type"])
    n_genes = ds.n_genes

    emb = None
    if encoder_type == "cbow":
        emb_path = cfg.gene_emb_tsv or ctx["gene_emb_source"]
        if not emb_path:
            raise ValueError(
                "gene_emb_tsv is required to export with a CBOW checkpoint "
                "when checkpoint does not include a gene_emb_source path."
            )
        genes_from_emb, emb_full = load_gene_embeddings_tsv(emb_path)
        emb = subset_embeddings(genes_from_emb, emb_full, ctx["genes_common"])
        if emb.shape[0] != n_genes:
            raise ValueError("Checkpoint genes and embedding genes are misaligned.")

    n_conditions = len(ds.batch_categories) if ds.batch_categories is not None else None
    perturbation_dim = ds.n_perturb_features if ctx["perturbation_mode"] == "cytokine_vector" else None

    latent_dim = int(train_cfg.get("latent_dim", 32))
    hidden_dim = int(train_cfg.get("hidden_dim", 256))
    n_hidden_layers = int(train_cfg.get("n_hidden_layers", 2))
    cond_emb_dim = int(train_cfg.get("cond_emb_dim", 16))
    input_transform = str(train_cfg.get("input_transform", "log1p"))
    library_norm = str(train_cfg.get("library_norm", "size_factor"))
    library_norm_target_sum = float(train_cfg.get("library_norm_target_sum", 1e4))
    library_norm_eps = float(train_cfg.get("library_norm_eps", 1e-8))
    use_library_size_covariate = bool(train_cfg.get("use_library_size_covariate", False))
    library_size_covariate_eps = float(train_cfg.get("library_size_covariate_eps", 1e-8))
    freeze_gene_embeddings = bool(train_cfg.get("freeze_gene_embeddings", True))

    if encoder_type == "cbow":
        encoder = CBOWCellEncoder(
            gene_embeddings=emb,
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            n_hidden_layers=n_hidden_layers,
            n_conditions=n_conditions,
            cond_emb_dim=cond_emb_dim,
            perturbation_dim=perturbation_dim,
            perturb_emb_dim=ctx["perturb_emb_dim"],
            perturb_condition_encoder=ctx["perturb_condition_encoder"],
            freeze_gene_embeddings=freeze_gene_embeddings,
            input_transform=input_transform,
        )
    elif encoder_type == "perceiver":
        encoder = PerceiverCellEncoder(
            n_genes=n_genes,
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            n_hidden_layers=n_hidden_layers,
            n_conditions=n_conditions,
            cond_emb_dim=cond_emb_dim,
            perturbation_dim=perturbation_dim,
            perturb_emb_dim=ctx["perturb_emb_dim"],
            perturb_condition_encoder=ctx["perturb_condition_encoder"],
            input_transform=input_transform,
            library_norm=library_norm,
            library_norm_target_sum=library_norm_target_sum,
            library_norm_eps=library_norm_eps,
            perceiver_d_model=int(train_cfg.get("perceiver_d_model", 256)),
            perceiver_num_latents=int(train_cfg.get("perceiver_num_latents", 64)),
            perceiver_num_cross_attn_heads=int(train_cfg.get("perceiver_num_cross_attn_heads", 8)),
            perceiver_num_self_attn_heads=int(train_cfg.get("perceiver_num_self_attn_heads", 8)),
            perceiver_num_self_attn_layers=int(train_cfg.get("perceiver_num_self_attn_layers", 4)),
            perceiver_ff_mult=int(train_cfg.get("perceiver_ff_mult", 4)),
            perceiver_dropout=float(train_cfg.get("perceiver_dropout", 0.0)),
        )
    else:
        encoder = TransformerCellEncoder(
            n_genes=n_genes,
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            n_hidden_layers=n_hidden_layers,
            n_conditions=n_conditions,
            cond_emb_dim=cond_emb_dim,
            perturbation_dim=perturbation_dim,
            perturb_emb_dim=ctx["perturb_emb_dim"],
            perturb_condition_encoder=ctx["perturb_condition_encoder"],
            input_transform=input_transform,
            transformer_d_model=int(train_cfg.get("transformer_d_model", 256)),
            transformer_n_heads=int(train_cfg.get("transformer_n_heads", 8)),
            transformer_n_layers=int(train_cfg.get("transformer_n_layers", 4)),
            transformer_ff_mult=int(train_cfg.get("transformer_ff_mult", 4)),
            transformer_dropout=float(train_cfg.get("transformer_dropout", 0.0)),
            token_mlp_hidden_dim=int(train_cfg.get("token_mlp_hidden_dim", 256)),
            token_mlp_layers=int(train_cfg.get("token_mlp_layers", 2)),
            max_tokens_per_cell=train_cfg.get("max_tokens_per_cell", None),
            min_expr_for_token=float(train_cfg.get("min_expr_for_token", 0.0)),
        )

    decoder = ZINBExpressionDecoder(
        n_genes=n_genes,
        latent_dim=latent_dim,
        hidden_dim=hidden_dim,
        n_hidden_layers=n_hidden_layers,
        n_conditions=n_conditions,
        cond_emb_dim=cond_emb_dim,
        perturbation_dim=perturbation_dim,
        perturb_emb_dim=ctx["perturb_emb_dim"],
        perturb_condition_decoder=ctx["perturb_condition_decoder"],
        use_library_size_covariate=use_library_size_covariate,
        library_size_covariate_eps=library_size_covariate_eps,
    )
    model = GeneVAE(encoder, decoder).to(device)

    ckpt = load_vae_checkpoint(
        cfg.checkpoint_path,
        model,
        optimizer=None,
        map_location=device,
        strict=False,
    )
    dropped_adv_keys = [
        k for k in ckpt.get("unexpected_keys", []) if str(k).startswith("batch_adversary.")
    ]
    if dropped_adv_keys:
        print(
            f"[export] INFO: ignored {len(dropped_adv_keys)} batch_adversary keys "
            "from checkpoint during export load."
        )
    model.eval()
    return model


def _select_export_indices(
    n_total: int,
    max_cells: Optional[int],
    max_cells_seed: Optional[int],
) -> np.ndarray:
    if max_cells is None:
        return np.arange(n_total, dtype=np.int64)

    n_select = min(int(max_cells), n_total)
    if n_select < 1:
        raise ValueError("max_cells must be >= 1 when provided.")
    if n_select == n_total:
        return np.arange(n_total, dtype=np.int64)

    rng = np.random.default_rng(max_cells_seed)
    return np.sort(rng.choice(n_total, size=n_select, replace=False).astype(np.int64))


def _build_export_dataloader(
    ds: SingleCellVAEDataset,
    selected_indices: np.ndarray,
    batch_size: int,
    num_workers: int,
    device: torch.device,
) -> tuple[Any, DataLoader, np.ndarray]:
    export_ds = ds if selected_indices.shape[0] == len(ds) else Subset(ds, selected_indices.tolist())
    selected_obs_names = ds.obs_names[selected_indices]
    dl = DataLoader(
        export_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        collate_fn=collate_vae_batch,
    )
    return export_ds, dl, selected_obs_names


def _validate_override_cell_ids(
    selected_obs_names: np.ndarray,
    overrides: Dict[str, np.ndarray],
) -> None:
    selected = [str(x) for x in selected_obs_names.tolist()]
    missing = [cid for cid in selected if cid not in overrides]
    if missing:
        raise ValueError(
            "counterfactual overrides are missing cell_id entries for selected cells. "
            f"Example missing id: {missing[0]}"
        )
    extra = sorted(set(overrides.keys()) - set(selected))
    if extra:
        raise ValueError(
            "counterfactual overrides contain cell_id values that are not present in the selected export set. "
            f"Example extra id: {extra[0]}"
        )


def _predict_mu_dataframe(
    *,
    model: GeneVAE,
    dl: DataLoader,
    selected_obs_names: np.ndarray,
    genes: list[str],
    device: torch.device,
    use_library_size_covariate: bool,
    override_map: Optional[Dict[str, np.ndarray]] = None,
) -> pd.DataFrame:
    preds: list[np.ndarray] = []
    cell_ids: list[str] = []
    seen = 0
    max_cells = int(selected_obs_names.shape[0])

    with torch.no_grad():
        for batch in dl:
            if seen >= max_cells:
                break

            x = batch["x_expr"].to(device, non_blocking=True)
            batch_idx = batch.get("batch_idx", None)
            if batch_idx is None:
                batch_idx = batch.get("cond_idx", None)
            if batch_idx is not None:
                batch_idx = batch_idx.to(device, non_blocking=True)

            perturb_vec = batch.get("perturb_vec", None)
            if perturb_vec is not None:
                perturb_vec = perturb_vec.to(device, non_blocking=True)

            token_gene_idx = batch.get("token_gene_idx", None)
            token_gene_mask = batch.get("token_gene_mask", None)
            if token_gene_idx is not None:
                token_gene_idx = token_gene_idx.to(device, non_blocking=True)
            if token_gene_mask is not None:
                token_gene_mask = token_gene_mask.to(device, non_blocking=True)

            libsize = batch.get("libsize", None)
            if libsize is not None:
                libsize = libsize.to(device, non_blocking=True)
            elif use_library_size_covariate:
                libsize = x.sum(dim=1)

            bsz = int(x.shape[0])
            ids_batch = selected_obs_names[seen : seen + bsz].tolist()
            if override_map is not None:
                over = np.stack([override_map[str(cid)] for cid in ids_batch], axis=0).astype(np.float32)
                perturb_vec = torch.from_numpy(over).to(device=device, non_blocking=True)

            mu_z, _ = model.encode(
                x,
                batch_idx,
                perturb_vec=perturb_vec,
                token_gene_idx=token_gene_idx,
                token_gene_mask=token_gene_mask,
            )
            mu, _, _ = model.decoder(
                mu_z,
                batch_idx,
                libsize=libsize,
                perturb_vec=perturb_vec,
            )

            mu_np = mu.detach().cpu().numpy()
            preds.append(mu_np)
            cell_ids.extend([str(x) for x in ids_batch])
            seen += bsz

    pred_mat = np.concatenate(preds, axis=0) if preds else np.zeros((0, len(genes)), dtype=np.float32)
    df = pd.DataFrame(pred_mat, columns=genes)
    df.insert(0, "cell_id", cell_ids)
    return df


def _write_prediction_tsv(df: pd.DataFrame, out_path: str) -> None:
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(out, "wt") as f:
        df.to_csv(f, sep="\t", index=False)


def export_predictions(cfg: VAEExportConfig) -> None:
    device = torch.device(cfg.device)
    effective_batch_key = _resolve_batch_key(cfg.batch_key, cfg.cond_key)
    ctx = _load_checkpoint_export_context(cfg)

    if ctx["perturbation_mode"] == "categorical" and effective_batch_key is None:
        raise ValueError("categorical perturbation export requires batch_key/cond_key.")

    ds = _build_export_dataset(cfg, effective_batch_key, ctx)
    model = _build_model_for_export(cfg, ctx, ds, device)
    selected_indices = _select_export_indices(len(ds), cfg.max_cells, cfg.max_cells_seed)
    _, dl, selected_obs_names = _build_export_dataloader(
        ds,
        selected_indices,
        cfg.batch_size,
        cfg.num_workers,
        device,
    )

    overrides = None
    if cfg.counterfactual_override_path:
        if ctx["perturbation_mode"] != "cytokine_vector":
            raise ValueError(
                "counterfactual_override_path is only supported when perturbation_mode='cytokine_vector'."
            )
        overrides = _load_counterfactual_overrides(
            cfg.counterfactual_override_path,
            list(ctx["cytokine_keys"]),
        )
        _validate_override_cell_ids(selected_obs_names, overrides)

    use_library_size_covariate = bool(ctx["train_cfg"].get("use_library_size_covariate", False))
    df = _predict_mu_dataframe(
        model=model,
        dl=dl,
        selected_obs_names=selected_obs_names,
        genes=ds.gene_order,
        device=device,
        use_library_size_covariate=use_library_size_covariate,
        override_map=overrides,
    )

    _write_prediction_tsv(df, cfg.out_pred_tsv_gz)
    out_path = Path(cfg.out_pred_tsv_gz)
    sidecar = {
        "encoder_type": ctx["encoder_type"],
        "perturbation_mode": ctx["perturbation_mode"],
        "cytokine_keys": list(ctx["cytokine_keys"]),
        "cytokine_transform": ctx["cytokine_transform"],
        "perturb_condition_encoder": ctx["perturb_condition_encoder"],
        "perturb_condition_decoder": ctx["perturb_condition_decoder"],
        "counterfactual_override_path": cfg.counterfactual_override_path,
        "max_cells": cfg.max_cells,
        "max_cells_seed": cfg.max_cells_seed,
        "n_exported_cells": int(selected_obs_names.shape[0]),
        "token_index_cache_dir": ctx["token_index_cache_dir"],
    }
    with out_path.with_suffix(out_path.suffix + ".metadata.json").open("w") as f:
        json.dump(sidecar, f, indent=2, sort_keys=True)

    print(f"[export] Wrote predicted mu to: {out_path}")
    if (
        ctx["encoder_type"] == "cbow"
        and cfg.gene_emb_tsv
        and ctx["gene_emb_source"] is not None
        and ctx["gene_emb_source"] != cfg.gene_emb_tsv
    ):
        print("[export] WARNING: checkpoint gene_emb_source differs from config gene_emb_tsv.")


def run_from_config(config_path: str) -> None:
    d = load_yaml(config_path)
    allowed = {f.name for f in fields(VAEExportConfig)}
    unknown = sorted(k for k in d.keys() if k not in allowed)
    if unknown:
        print(f"[export] WARNING: ignoring unknown config keys: {', '.join(unknown)}")
    cfg = VAEExportConfig(**{k: v for k, v in d.items() if k in allowed})
    export_predictions(cfg)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("config", type=str)
    args = ap.parse_args()
    run_from_config(args.config)
