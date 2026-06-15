from __future__ import annotations

import argparse
import json
from dataclasses import fields
from pathlib import Path

import torch

from cellinguist.config import CytokineTreatmentPredictionConfig, VAEExportConfig, load_yaml
from cellinguist.scripts.export_vae_predictions import (
    _build_export_dataloader,
    _build_export_dataset,
    _build_model_for_export,
    _load_checkpoint_export_context,
    _load_counterfactual_overrides,
    _predict_mu_dataframe,
    _resolve_batch_key,
    _select_export_indices,
    _validate_override_cell_ids,
    _write_prediction_tsv,
)


def _to_export_cfg(cfg: CytokineTreatmentPredictionConfig) -> VAEExportConfig:
    return VAEExportConfig(
        adata_path=cfg.adata_path,
        gene_key=cfg.gene_key,
        layer=cfg.layer,
        cond_key=cfg.cond_key,
        batch_key=cfg.batch_key,
        perturbation_mode=str(cfg.perturbation_mode or "none"),
        cytokine_keys=cfg.cytokine_keys,
        cytokine_transform=str(cfg.cytokine_transform or "log1p"),
        cytokine_missing_policy=str(cfg.cytokine_missing_policy or "error"),
        perturb_emb_dim=int(cfg.perturb_emb_dim or 32),
        perturb_condition_encoder=(
            True if cfg.perturb_condition_encoder is None else bool(cfg.perturb_condition_encoder)
        ),
        perturb_condition_decoder=(
            True if cfg.perturb_condition_decoder is None else bool(cfg.perturb_condition_decoder)
        ),
        counterfactual_override_path=cfg.counterfactual_override_path,
        gene_emb_tsv=cfg.gene_emb_tsv,
        checkpoint_path=cfg.checkpoint_path,
        out_pred_tsv_gz="",
        max_cells=cfg.max_cells,
        max_cells_seed=cfg.max_cells_seed,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        device=cfg.device,
        backed=cfg.backed,
        token_index_cache_dir=cfg.token_index_cache_dir,
        token_index_cache_require=cfg.token_index_cache_require,
        transformer_precompute_token_indices=cfg.transformer_precompute_token_indices,
    )


def predict_cytokine_treatment(cfg: CytokineTreatmentPredictionConfig) -> str:
    export_cfg = _to_export_cfg(cfg)
    device = torch.device(cfg.device)
    effective_batch_key = _resolve_batch_key(cfg.batch_key, cfg.cond_key)
    ctx = _load_checkpoint_export_context(export_cfg)

    if str(ctx["perturbation_mode"]) != "cytokine_vector":
        raise ValueError(
            "predict-cytokine-treatment requires a checkpoint trained with "
            "perturbation_mode='cytokine_vector'."
        )

    ds = _build_export_dataset(export_cfg, effective_batch_key, ctx)
    model = _build_model_for_export(export_cfg, ctx, ds, device)
    selected_indices = _select_export_indices(len(ds), cfg.max_cells, cfg.max_cells_seed)
    _, dl, selected_obs_names = _build_export_dataloader(
        ds,
        selected_indices,
        cfg.batch_size,
        cfg.num_workers,
        device,
    )

    overrides = _load_counterfactual_overrides(
        cfg.counterfactual_override_path,
        list(ctx["cytokine_keys"]),
    )
    _validate_override_cell_ids(selected_obs_names, overrides)

    use_library_size_covariate = bool(ctx["train_cfg"].get("use_library_size_covariate", False))
    baseline_df = _predict_mu_dataframe(
        model=model,
        dl=dl,
        selected_obs_names=selected_obs_names,
        genes=ds.gene_order,
        device=device,
        use_library_size_covariate=use_library_size_covariate,
        override_map=None,
    )
    treated_df = _predict_mu_dataframe(
        model=model,
        dl=dl,
        selected_obs_names=selected_obs_names,
        genes=ds.gene_order,
        device=device,
        use_library_size_covariate=use_library_size_covariate,
        override_map=overrides,
    )

    delta_df = treated_df.copy()
    gene_cols = [c for c in treated_df.columns if c != "cell_id"]
    delta_df.loc[:, gene_cols] = treated_df[gene_cols].to_numpy() - baseline_df[gene_cols].to_numpy()

    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    baseline_path = out_dir / "pred_baseline.tsv.gz"
    treated_path = out_dir / "pred_treated.tsv.gz"
    delta_path = out_dir / "delta.tsv.gz"
    metadata_path = out_dir / "metadata.json"

    _write_prediction_tsv(baseline_df, str(baseline_path))
    _write_prediction_tsv(treated_df, str(treated_path))
    _write_prediction_tsv(delta_df, str(delta_path))

    metadata = {
        "checkpoint_path": cfg.checkpoint_path,
        "encoder_type": ctx["encoder_type"],
        "perturbation_mode": ctx["perturbation_mode"],
        "cytokine_keys": list(ctx["cytokine_keys"]),
        "cytokine_transform": ctx["cytokine_transform"],
        "cytokine_missing_policy": ctx["cytokine_missing_policy"],
        "counterfactual_override_path": cfg.counterfactual_override_path,
        "n_exported_cells": int(selected_obs_names.shape[0]),
        "token_index_cache_dir": ctx["token_index_cache_dir"],
        "conditioning_design": (
            "PerturbationProjector output is applied according to the checkpoint "
            "conditioning flags; cytokines are not injected as transformer tokens."
        ),
        "perturb_condition_encoder": ctx["perturb_condition_encoder"],
        "perturb_condition_decoder": ctx["perturb_condition_decoder"],
        "outputs": {
            "pred_baseline": str(baseline_path),
            "pred_treated": str(treated_path),
            "delta": str(delta_path),
        },
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True))

    print(f"[cytokine] Wrote cytokine treatment predictions to: {out_dir}")
    return str(out_dir)


def run_from_config(config_path: str) -> str:
    d = load_yaml(config_path)
    allowed = {f.name for f in fields(CytokineTreatmentPredictionConfig)}
    unknown = sorted(k for k in d.keys() if k not in allowed)
    if unknown:
        print(f"[cytokine] WARNING: ignoring unknown config keys: {', '.join(unknown)}")
    cfg = CytokineTreatmentPredictionConfig(**{k: v for k, v in d.items() if k in allowed})
    return predict_cytokine_treatment(cfg)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Predict per-cell transcriptomic consequences of cytokine treatment."
    )
    parser.add_argument("config", type=str, help="Path to YAML config.")
    args = parser.parse_args()
    run_from_config(args.config)


if __name__ == "__main__":
    main()
