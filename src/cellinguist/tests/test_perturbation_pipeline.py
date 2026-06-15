from __future__ import annotations

import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import torch

from cellinguist.config import CytokineTreatmentPredictionConfig, VAEExportConfig
from cellinguist.data.datasets import SingleCellVAEDataset
from cellinguist.models.vae import (
    BatchAdversary,
    GeneVAE,
    PerceiverCellEncoder,
    TransformerCellEncoder,
    ZINBExpressionDecoder,
)
from cellinguist.scripts.export_transformer_cell_embeddings import export_transformer_cell_embeddings
from cellinguist.scripts.export_transformer_gene_embeddings import export_transformer_gene_embeddings
from cellinguist.scripts.export_vae_predictions import (
    _load_counterfactual_overrides,
    export_predictions,
)
from cellinguist.scripts.predict_cytokine_treatment import predict_cytokine_treatment
from cellinguist.utils.perturbation_split import build_cytokine_combo_split
from cellinguist.utils.vae_io import load_vae_checkpoint, save_vae_checkpoint


def _write_tiny_h5ad(tmp_path: Path) -> Path:
    x = np.array(
        [
            [1, 0, 3, 2],
            [0, 2, 1, 1],
            [3, 1, 0, 1],
            [2, 2, 2, 0],
        ],
        dtype=np.float32,
    )
    obs = pd.DataFrame(
        {
            "batch": ["a", "a", "b", "b"],
            "IL6": [0.0, 1.0, 0.0, 1.0],
            "IFNG": [0.0, 0.0, 1.0, 1.0],
        },
        index=[f"cell_{i}" for i in range(4)],
    )
    var = pd.DataFrame({"gene": [f"g{i}" for i in range(4)]})
    adata = ad.AnnData(X=x, obs=obs, var=var)
    out = tmp_path / "tiny.h5ad"
    adata.write_h5ad(out)
    return out


def _write_override_tsv(tmp_path: Path, *, include_extra: bool = False) -> Path:
    data = {
        "cell_id": [f"cell_{i}" for i in range(4)],
        "IL6": [1.0, 1.0, 0.0, 0.0],
        "IFNG": [0.0, 0.0, 1.0, 1.0],
    }
    if include_extra:
        data["cell_id"].append("cell_extra")
        data["IL6"].append(0.0)
        data["IFNG"].append(0.0)
    out = tmp_path / "override.tsv"
    pd.DataFrame(data).to_csv(out, sep="\t", index=False)
    return out


def _write_tiny_transformer_cytokine_ckpt(
    tmp_path: Path,
    *,
    perturbation_mode: str = "cytokine_vector",
    perturb_condition_encoder: bool = True,
    perturb_condition_decoder: bool = True,
    include_condition_flags_in_ckpt: bool = True,
) -> Path:
    encoder = TransformerCellEncoder(
        n_genes=4,
        latent_dim=5,
        hidden_dim=8,
        n_hidden_layers=1,
        n_conditions=2,
        cond_emb_dim=4,
        perturbation_dim=(2 if perturbation_mode == "cytokine_vector" else None),
        perturb_emb_dim=6,
        perturb_condition_encoder=perturb_condition_encoder,
        input_transform="none",
        transformer_d_model=8,
        transformer_n_heads=2,
        transformer_n_layers=1,
        transformer_ff_mult=2,
        token_mlp_hidden_dim=8,
        token_mlp_layers=1,
        max_tokens_per_cell=3,
        min_expr_for_token=0.0,
    )
    decoder = ZINBExpressionDecoder(
        n_genes=4,
        latent_dim=5,
        hidden_dim=8,
        n_hidden_layers=1,
        n_conditions=2,
        cond_emb_dim=4,
        perturbation_dim=(2 if perturbation_mode == "cytokine_vector" else None),
        perturb_emb_dim=6,
        perturb_condition_decoder=perturb_condition_decoder,
    )
    model = GeneVAE(encoder, decoder)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    ckpt_path = tmp_path / (
        f"transformer_{perturbation_mode}_enc{int(perturb_condition_encoder)}"
        f"_dec{int(perturb_condition_decoder)}.ckpt"
    )
    config_snapshot = {
        "encoder_type": "transformer",
        "latent_dim": 5,
        "hidden_dim": 8,
        "n_hidden_layers": 1,
        "cond_emb_dim": 4,
        "input_transform": "none",
        "transformer_d_model": 8,
        "transformer_n_heads": 2,
        "transformer_n_layers": 1,
        "transformer_ff_mult": 2,
        "token_mlp_hidden_dim": 8,
        "token_mlp_layers": 1,
        "max_tokens_per_cell": 3,
        "min_expr_for_token": 0.0,
        "transformer_precompute_token_indices": True,
        "perturbation_mode": perturbation_mode,
        "cytokine_keys": ["IL6", "IFNG"] if perturbation_mode == "cytokine_vector" else None,
        "cytokine_transform": "none",
        "cytokine_missing_policy": "error",
        "perturb_emb_dim": 6,
    }
    if include_condition_flags_in_ckpt:
        config_snapshot["perturb_condition_encoder"] = perturb_condition_encoder
        config_snapshot["perturb_condition_decoder"] = perturb_condition_decoder
    save_vae_checkpoint(
        path=str(ckpt_path),
        model=model,
        optimizer=opt,
        epoch=0,
        genes_common=["g0", "g1", "g2", "g3"],
        config_snapshot=config_snapshot,
        gene_emb_source="",
    )
    return ckpt_path


def test_dataset_cytokine_vector_outputs_perturb_vec(tmp_path: Path) -> None:
    h5ad_path = _write_tiny_h5ad(tmp_path)
    ds = SingleCellVAEDataset(
        adata_or_path=str(h5ad_path),
        gene_key="gene",
        batch_key="batch",
        perturbation_mode="cytokine_vector",
        cytokine_keys=["IL6", "IFNG"],
        cytokine_transform="none",
        cytokine_missing_policy="error",
        transform="none",
        backed=False,
    )
    item = ds[0]
    assert "perturb_vec" in item
    assert item["perturb_vec"].shape == (2,)
    assert ds.n_perturb_features == 2


def test_dataset_batch_correction_mean_scale_reduces_batch_shift(tmp_path: Path) -> None:
    x = np.array(
        [
            [10.0, 2.0, 1.0, 1.0],
            [12.0, 1.0, 1.0, 1.0],
            [100.0, 2.0, 1.0, 1.0],
            [120.0, 1.0, 1.0, 1.0],
        ],
        dtype=np.float32,
    )
    obs = pd.DataFrame(
        {
            "batch": ["a", "a", "b", "b"],
            "IL6": [0.0, 0.0, 0.0, 0.0],
            "IFNG": [0.0, 0.0, 0.0, 0.0],
        },
        index=[f"cell_{i}" for i in range(4)],
    )
    var = pd.DataFrame({"gene": [f"g{i}" for i in range(4)]})
    h5ad_path = tmp_path / "batch_shift.h5ad"
    ad.AnnData(X=x, obs=obs, var=var).write_h5ad(h5ad_path)

    ds_none = SingleCellVAEDataset(
        adata_or_path=str(h5ad_path),
        gene_key="gene",
        batch_key="batch",
        transform="none",
        backed=False,
    )
    ds_corr = SingleCellVAEDataset(
        adata_or_path=str(h5ad_path),
        gene_key="gene",
        batch_key="batch",
        batch_correction_method="mean_scale",
        transform="none",
        backed=False,
    )

    g0_none = np.stack([ds_none[i]["x_expr"].numpy() for i in range(len(ds_none))], axis=0)[:, 0]
    g0_corr = np.stack([ds_corr[i]["x_expr"].numpy() for i in range(len(ds_corr))], axis=0)[:, 0]

    mean_a_none = float(g0_none[[0, 1]].mean())
    mean_b_none = float(g0_none[[2, 3]].mean())
    mean_a_corr = float(g0_corr[[0, 1]].mean())
    mean_b_corr = float(g0_corr[[2, 3]].mean())

    assert abs(mean_a_corr - mean_b_corr) < abs(mean_a_none - mean_b_none)
    assert (g0_corr >= 0).all()


def test_dataset_batch_correction_requires_batch_key(tmp_path: Path) -> None:
    h5ad_path = _write_tiny_h5ad(tmp_path)
    try:
        SingleCellVAEDataset(
            adata_or_path=str(h5ad_path),
            gene_key="gene",
            batch_correction_method="mean_scale",
            transform="none",
            backed=False,
        )
        assert False, "Expected ValueError when batch correction is enabled without batch_key/cond_key."
    except ValueError:
        pass


def test_model_forward_with_and_without_perturb() -> None:
    x = torch.rand(3, 4)
    cond = torch.tensor([0, 1, 0], dtype=torch.long)
    perturb = torch.rand(3, 2)

    enc_none = PerceiverCellEncoder(
        n_genes=4,
        latent_dim=5,
        hidden_dim=8,
        n_hidden_layers=1,
        n_conditions=2,
        cond_emb_dim=4,
        input_transform="none",
        library_norm="none",
        perceiver_d_model=8,
        perceiver_num_latents=4,
        perceiver_num_cross_attn_heads=2,
        perceiver_num_self_attn_heads=2,
        perceiver_num_self_attn_layers=1,
    )
    dec_none = ZINBExpressionDecoder(
        n_genes=4,
        latent_dim=5,
        hidden_dim=8,
        n_hidden_layers=1,
        n_conditions=2,
        cond_emb_dim=4,
    )
    model_none = GeneVAE(enc_none, dec_none)
    recon_out, _, _ = model_none(x, cond_idx=cond, libsize=x.sum(dim=1))
    assert recon_out[0].shape == (3, 4)

    enc_pert = PerceiverCellEncoder(
        n_genes=4,
        latent_dim=5,
        hidden_dim=8,
        n_hidden_layers=1,
        n_conditions=2,
        cond_emb_dim=4,
        perturbation_dim=2,
        perturb_emb_dim=6,
        input_transform="none",
        library_norm="none",
        perceiver_d_model=8,
        perceiver_num_latents=4,
        perceiver_num_cross_attn_heads=2,
        perceiver_num_self_attn_heads=2,
        perceiver_num_self_attn_layers=1,
    )
    dec_pert = ZINBExpressionDecoder(
        n_genes=4,
        latent_dim=5,
        hidden_dim=8,
        n_hidden_layers=1,
        n_conditions=2,
        cond_emb_dim=4,
        perturbation_dim=2,
        perturb_emb_dim=6,
    )
    model_pert = GeneVAE(enc_pert, dec_pert)
    recon_out2, _, _ = model_pert(
        x,
        cond_idx=cond,
        libsize=x.sum(dim=1),
        perturb_vec=perturb,
    )
    assert recon_out2[0].shape == (3, 4)


def test_checkpoint_non_strict_load_ignores_adversary(tmp_path: Path) -> None:
    encoder = PerceiverCellEncoder(
        n_genes=4,
        latent_dim=5,
        hidden_dim=8,
        n_hidden_layers=1,
        n_conditions=2,
        cond_emb_dim=4,
        input_transform="none",
        library_norm="none",
        perceiver_d_model=8,
        perceiver_num_latents=4,
        perceiver_num_cross_attn_heads=2,
        perceiver_num_self_attn_heads=2,
        perceiver_num_self_attn_layers=1,
    )
    decoder = ZINBExpressionDecoder(
        n_genes=4,
        latent_dim=5,
        hidden_dim=8,
        n_hidden_layers=1,
        n_conditions=2,
        cond_emb_dim=4,
    )
    model = GeneVAE(
        encoder,
        decoder,
        batch_adversary=BatchAdversary(latent_dim=5, n_batches=2),
    )
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    ckpt = tmp_path / "model.ckpt"
    save_vae_checkpoint(
        path=str(ckpt),
        model=model,
        optimizer=opt,
        epoch=0,
        genes_common=["g0", "g1", "g2", "g3"],
        config_snapshot={},
        gene_emb_source="",
    )

    model_no_adv = GeneVAE(encoder, decoder, batch_adversary=None)
    out = load_vae_checkpoint(str(ckpt), model_no_adv, optimizer=None, strict=False)
    assert any(k.startswith("batch_adversary.") for k in out["unexpected_keys"])


def test_counterfactual_override_order_validation(tmp_path: Path) -> None:
    override_path = tmp_path / "override.tsv"
    pd.DataFrame(
        {
            "cell_id": ["cell_0"],
            "IFNG": [1.0],
            "IL6": [0.0],
        }
    ).to_csv(override_path, sep="\t", index=False)

    try:
        _load_counterfactual_overrides(str(override_path), ["IL6", "IFNG"])
        assert False, "Expected ValueError for wrong column order."
    except ValueError:
        pass


def test_counterfactual_override_duplicate_cell_ids_fail(tmp_path: Path) -> None:
    override_path = tmp_path / "override_dupe.tsv"
    pd.DataFrame(
        {
            "cell_id": ["cell_0", "cell_0"],
            "IL6": [0.0, 1.0],
            "IFNG": [1.0, 0.0],
        }
    ).to_csv(override_path, sep="\t", index=False)

    try:
        _load_counterfactual_overrides(str(override_path), ["IL6", "IFNG"])
        assert False, "Expected ValueError for duplicate cell_id entries."
    except ValueError:
        pass


def test_cytokine_combo_split_builds_holdout() -> None:
    mat = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
        ],
        dtype=np.float32,
    )
    split = build_cytokine_combo_split(mat, ["IL6", "IFNG"], min_active_for_holdout=2)
    assert split["n_train_cells"] == 3
    assert split["n_val_cells"] == 1


def test_minimal_train_step_all_modes(tmp_path: Path) -> None:
    h5ad_path = _write_tiny_h5ad(tmp_path)
    modes = ["none", "categorical", "cytokine_vector"]
    for mode in modes:
        ds = SingleCellVAEDataset(
            adata_or_path=str(h5ad_path),
            gene_key="gene",
            batch_key="batch",
            perturbation_mode=mode,
            cytokine_keys=["IL6", "IFNG"] if mode == "cytokine_vector" else None,
            transform="none",
            backed=False,
        )
        x = torch.stack([ds[i]["x_expr"] for i in range(2)], dim=0)
        cond = torch.tensor([int(ds[i]["batch_idx"]) for i in range(2)], dtype=torch.long)
        perturb = None
        perturbation_dim = None
        if mode == "cytokine_vector":
            perturb = torch.stack([ds[i]["perturb_vec"] for i in range(2)], dim=0)
            perturbation_dim = ds.n_perturb_features

        enc = PerceiverCellEncoder(
            n_genes=ds.n_genes,
            latent_dim=5,
            hidden_dim=8,
            n_hidden_layers=1,
            n_conditions=2,
            cond_emb_dim=4,
            perturbation_dim=perturbation_dim,
            perturb_emb_dim=6,
            input_transform="none",
            library_norm="none",
            perceiver_d_model=8,
            perceiver_num_latents=4,
            perceiver_num_cross_attn_heads=2,
            perceiver_num_self_attn_heads=2,
            perceiver_num_self_attn_layers=1,
        )
        dec = ZINBExpressionDecoder(
            n_genes=ds.n_genes,
            latent_dim=5,
            hidden_dim=8,
            n_hidden_layers=1,
            n_conditions=2,
            cond_emb_dim=4,
            perturbation_dim=perturbation_dim,
            perturb_emb_dim=6,
        )
        model = GeneVAE(enc, dec)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        recon_out, _, _ = model(x, cond_idx=cond, libsize=x.sum(dim=1), perturb_vec=perturb)
        mu, _, _ = recon_out
        loss = mu.mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()


def test_transformer_tokenization_cls_and_padding() -> None:
    x = torch.tensor(
        [
            [0.0, 2.0, 0.0, 5.0],
            [0.0, 0.0, 0.0, 0.0],
        ],
        dtype=torch.float32,
    )
    enc = TransformerCellEncoder(
        n_genes=4,
        latent_dim=5,
        hidden_dim=8,
        n_hidden_layers=1,
        input_transform="log1p",
        transformer_d_model=8,
        transformer_n_heads=2,
        transformer_n_layers=1,
        transformer_ff_mult=2,
        token_mlp_hidden_dim=8,
        token_mlp_layers=1,
        max_tokens_per_cell=None,
        min_expr_for_token=0.0,
    )
    tokens, pad_mask = enc._build_token_batch(x)
    assert tokens.shape[0] == 2
    assert tokens.shape[2] == 8
    assert tokens.shape[1] == 3  # CLS + 2 expressed genes (max in batch)
    assert pad_mask.shape == (2, 3)
    assert bool(pad_mask[0, 0].item()) is False
    assert bool(pad_mask[0, 1].item()) is False
    assert bool(pad_mask[0, 2].item()) is False
    assert bool(pad_mask[1, 0].item()) is False
    assert bool(pad_mask[1, 1].item()) is True
    assert bool(pad_mask[1, 2].item()) is True


def test_transformer_model_forward_with_and_without_perturb() -> None:
    x = torch.rand(3, 4)
    cond = torch.tensor([0, 1, 0], dtype=torch.long)
    perturb = torch.rand(3, 2)

    enc_none = TransformerCellEncoder(
        n_genes=4,
        latent_dim=5,
        hidden_dim=8,
        n_hidden_layers=1,
        n_conditions=2,
        cond_emb_dim=4,
        input_transform="none",
        transformer_d_model=8,
        transformer_n_heads=2,
        transformer_n_layers=1,
        transformer_ff_mult=2,
        token_mlp_hidden_dim=8,
        token_mlp_layers=1,
    )
    dec_none = ZINBExpressionDecoder(
        n_genes=4,
        latent_dim=5,
        hidden_dim=8,
        n_hidden_layers=1,
        n_conditions=2,
        cond_emb_dim=4,
    )
    model_none = GeneVAE(enc_none, dec_none)
    recon_out, _, _ = model_none(x, cond_idx=cond, libsize=x.sum(dim=1))
    assert recon_out[0].shape == (3, 4)

    enc_pert = TransformerCellEncoder(
        n_genes=4,
        latent_dim=5,
        hidden_dim=8,
        n_hidden_layers=1,
        n_conditions=2,
        cond_emb_dim=4,
        perturbation_dim=2,
        perturb_emb_dim=6,
        input_transform="none",
        transformer_d_model=8,
        transformer_n_heads=2,
        transformer_n_layers=1,
        transformer_ff_mult=2,
        token_mlp_hidden_dim=8,
        token_mlp_layers=1,
    )
    dec_pert = ZINBExpressionDecoder(
        n_genes=4,
        latent_dim=5,
        hidden_dim=8,
        n_hidden_layers=1,
        n_conditions=2,
        cond_emb_dim=4,
        perturbation_dim=2,
        perturb_emb_dim=6,
    )
    model_pert = GeneVAE(enc_pert, dec_pert)
    recon_out2, _, _ = model_pert(
        x,
        cond_idx=cond,
        libsize=x.sum(dim=1),
        perturb_vec=perturb,
    )
    assert recon_out2[0].shape == (3, 4)


def test_transformer_decoder_only_ignores_perturb_in_encoder_but_not_decoder() -> None:
    x = torch.rand(3, 4)
    cond = torch.tensor([0, 1, 0], dtype=torch.long)
    perturb_a = torch.zeros(3, 2)
    perturb_b = torch.ones(3, 2)

    enc = TransformerCellEncoder(
        n_genes=4,
        latent_dim=5,
        hidden_dim=8,
        n_hidden_layers=1,
        n_conditions=2,
        cond_emb_dim=4,
        perturbation_dim=2,
        perturb_emb_dim=6,
        perturb_condition_encoder=False,
        input_transform="none",
        transformer_d_model=8,
        transformer_n_heads=2,
        transformer_n_layers=1,
        transformer_ff_mult=2,
        token_mlp_hidden_dim=8,
        token_mlp_layers=1,
    )
    dec = ZINBExpressionDecoder(
        n_genes=4,
        latent_dim=5,
        hidden_dim=8,
        n_hidden_layers=1,
        n_conditions=2,
        cond_emb_dim=4,
        perturbation_dim=2,
        perturb_emb_dim=6,
        perturb_condition_decoder=True,
    )
    model = GeneVAE(enc, dec)

    mu_a, _ = model.encode(x, cond_idx=cond, perturb_vec=perturb_a)
    mu_b, _ = model.encode(x, cond_idx=cond, perturb_vec=perturb_b)
    assert torch.allclose(mu_a, mu_b)

    out_a = model.decode(mu_a, cond_idx=cond, libsize=x.sum(dim=1), perturb_vec=perturb_a)[0]
    out_b = model.decode(mu_a, cond_idx=cond, libsize=x.sum(dim=1), perturb_vec=perturb_b)[0]
    assert not torch.allclose(out_a, out_b)


def test_transformer_dataset_emits_perturb_and_token_indices_together(tmp_path: Path) -> None:
    h5ad_path = _write_tiny_h5ad(tmp_path)
    ds = SingleCellVAEDataset(
        adata_or_path=str(h5ad_path),
        gene_key="gene",
        batch_key="batch",
        perturbation_mode="cytokine_vector",
        cytokine_keys=["IL6", "IFNG"],
        cytokine_transform="none",
        cytokine_missing_policy="error",
        transform="none",
        backed=False,
        precompute_token_gene_indices=True,
        token_min_expr=0.0,
        token_max_genes=3,
    )
    item = ds[0]
    assert "perturb_vec" in item
    assert "token_gene_idx" in item
    assert item["perturb_vec"].shape == (2,)
    assert item["token_gene_idx"].ndim == 1


def test_export_vae_predictions_transformer_counterfactual_uses_checkpoint_cytokine_config(tmp_path: Path) -> None:
    h5ad_path = _write_tiny_h5ad(tmp_path)
    ckpt_path = _write_tiny_transformer_cytokine_ckpt(tmp_path, perturbation_mode="cytokine_vector")
    override_path = _write_override_tsv(tmp_path)
    out_path = tmp_path / "pred_mu.tsv.gz"

    export_predictions(
        VAEExportConfig(
            adata_path=str(h5ad_path),
            checkpoint_path=str(ckpt_path),
            counterfactual_override_path=str(override_path),
            out_pred_tsv_gz=str(out_path),
            gene_key="gene",
            batch_key="batch",
            batch_size=2,
            num_workers=0,
            device="cpu",
            backed=False,
            transformer_precompute_token_indices=True,
        )
    )

    df = pd.read_csv(out_path, sep="\t")
    metadata = json.loads(Path(str(out_path) + ".metadata.json").read_text())
    assert df.shape == (4, 5)
    assert df.columns.tolist() == ["cell_id", "g0", "g1", "g2", "g3"]
    assert metadata["encoder_type"] == "transformer"
    assert metadata["cytokine_keys"] == ["IL6", "IFNG"]
    assert metadata["n_exported_cells"] == 4


def test_export_vae_predictions_legacy_checkpoint_defaults_to_dual_conditioning(tmp_path: Path) -> None:
    h5ad_path = _write_tiny_h5ad(tmp_path)
    ckpt_path = _write_tiny_transformer_cytokine_ckpt(
        tmp_path,
        perturbation_mode="cytokine_vector",
        include_condition_flags_in_ckpt=False,
    )
    override_path = _write_override_tsv(tmp_path)
    out_path = tmp_path / "pred_legacy.tsv.gz"

    export_predictions(
        VAEExportConfig(
            adata_path=str(h5ad_path),
            checkpoint_path=str(ckpt_path),
            counterfactual_override_path=str(override_path),
            out_pred_tsv_gz=str(out_path),
            gene_key="gene",
            batch_key="batch",
            batch_size=2,
            num_workers=0,
            device="cpu",
            backed=False,
        )
    )

    metadata = json.loads(Path(str(out_path) + ".metadata.json").read_text())
    assert metadata["encoder_type"] == "transformer"
    assert metadata["cytokine_keys"] == ["IL6", "IFNG"]
    assert metadata["perturb_condition_encoder"] is True
    assert metadata["perturb_condition_decoder"] is True


def test_export_vae_predictions_rejects_disabled_encoder_and_decoder_conditioning(tmp_path: Path) -> None:
    h5ad_path = _write_tiny_h5ad(tmp_path)
    ckpt_path = _write_tiny_transformer_cytokine_ckpt(
        tmp_path,
        perturbation_mode="cytokine_vector",
        perturb_condition_encoder=False,
        perturb_condition_decoder=False,
    )
    override_path = _write_override_tsv(tmp_path)

    try:
        export_predictions(
            VAEExportConfig(
                adata_path=str(h5ad_path),
                checkpoint_path=str(ckpt_path),
                counterfactual_override_path=str(override_path),
                out_pred_tsv_gz=str(tmp_path / "unused.tsv.gz"),
                gene_key="gene",
                batch_key="batch",
                batch_size=2,
                num_workers=0,
                device="cpu",
                backed=False,
            )
        )
        assert False, "Expected ValueError when both perturb conditioning flags are disabled."
    except ValueError:
        pass


def test_export_vae_predictions_transformer_none_mode_regression(tmp_path: Path) -> None:
    h5ad_path = _write_tiny_h5ad(tmp_path)
    ckpt_path = _write_tiny_transformer_cytokine_ckpt(tmp_path, perturbation_mode="none")
    out_path = tmp_path / "pred_none.tsv.gz"

    export_predictions(
        VAEExportConfig(
            adata_path=str(h5ad_path),
            checkpoint_path=str(ckpt_path),
            out_pred_tsv_gz=str(out_path),
            gene_key="gene",
            batch_key="batch",
            batch_size=2,
            num_workers=0,
            device="cpu",
            backed=False,
            transformer_precompute_token_indices=True,
        )
    )

    df = pd.read_csv(out_path, sep="\t")
    assert df.shape == (4, 5)


def test_predict_cytokine_treatment_outputs_baseline_treated_and_delta(tmp_path: Path) -> None:
    h5ad_path = _write_tiny_h5ad(tmp_path)
    ckpt_path = _write_tiny_transformer_cytokine_ckpt(
        tmp_path,
        perturbation_mode="cytokine_vector",
        perturb_condition_encoder=False,
        perturb_condition_decoder=True,
    )
    override_path = _write_override_tsv(tmp_path)
    out_dir = tmp_path / "cytokine_out"

    predict_cytokine_treatment(
        CytokineTreatmentPredictionConfig(
            adata_path=str(h5ad_path),
            checkpoint_path=str(ckpt_path),
            counterfactual_override_path=str(override_path),
            out_dir=str(out_dir),
            gene_key="gene",
            batch_key="batch",
            batch_size=2,
            num_workers=0,
            device="cpu",
            backed=False,
            transformer_precompute_token_indices=True,
        )
    )

    baseline_df = pd.read_csv(out_dir / "pred_baseline.tsv.gz", sep="\t")
    treated_df = pd.read_csv(out_dir / "pred_treated.tsv.gz", sep="\t")
    delta_df = pd.read_csv(out_dir / "delta.tsv.gz", sep="\t")
    metadata = json.loads((out_dir / "metadata.json").read_text())

    gene_cols = ["g0", "g1", "g2", "g3"]
    expected_delta = treated_df[gene_cols].to_numpy() - baseline_df[gene_cols].to_numpy()
    assert baseline_df["cell_id"].tolist() == ["cell_0", "cell_1", "cell_2", "cell_3"]
    assert treated_df["cell_id"].tolist() == baseline_df["cell_id"].tolist()
    assert delta_df["cell_id"].tolist() == baseline_df["cell_id"].tolist()
    assert np.allclose(delta_df[gene_cols].to_numpy(), expected_delta, atol=1e-6)
    assert metadata["cytokine_keys"] == ["IL6", "IFNG"]
    assert metadata["encoder_type"] == "transformer"
    assert metadata["perturb_condition_encoder"] is False
    assert metadata["perturb_condition_decoder"] is True


def test_predict_cytokine_treatment_rejects_extra_override_rows(tmp_path: Path) -> None:
    h5ad_path = _write_tiny_h5ad(tmp_path)
    ckpt_path = _write_tiny_transformer_cytokine_ckpt(tmp_path, perturbation_mode="cytokine_vector")
    override_path = _write_override_tsv(tmp_path, include_extra=True)

    try:
        predict_cytokine_treatment(
            CytokineTreatmentPredictionConfig(
                adata_path=str(h5ad_path),
                checkpoint_path=str(ckpt_path),
                counterfactual_override_path=str(override_path),
                out_dir=str(tmp_path / "unused"),
                gene_key="gene",
                batch_key="batch",
                batch_size=2,
                num_workers=0,
                device="cpu",
                backed=False,
            )
        )
        assert False, "Expected ValueError for extra override rows."
    except ValueError:
        pass


def test_export_transformer_gene_embeddings_writes_tsv(tmp_path: Path) -> None:
    ckpt_path = tmp_path / "transformer.ckpt"
    out_path = tmp_path / "transformer_gene_emb.tsv.gz"

    torch.save(
        {
            "config": {"encoder_type": "transformer"},
            "genes_common": ["g0", "g1"],
            "model_state_dict": {
                "encoder.gene_embedding.weight": torch.tensor(
                    [[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32
                )
            },
        },
        ckpt_path,
    )

    export_transformer_gene_embeddings(str(ckpt_path), str(out_path))

    df = pd.read_csv(out_path, sep="\t")
    assert list(df.columns) == ["gene", "dim_1", "dim_2"]
    assert df["gene"].tolist() == ["g0", "g1"]
    assert np.allclose(df[["dim_1", "dim_2"]].to_numpy(), np.array([[1.0, 2.0], [3.0, 4.0]]))


def test_export_transformer_gene_embeddings_rejects_non_transformer(tmp_path: Path) -> None:
    ckpt_path = tmp_path / "not_transformer.ckpt"

    torch.save(
        {
            "config": {"encoder_type": "perceiver"},
            "genes_common": ["g0"],
            "model_state_dict": {
                "encoder.gene_embedding.weight": torch.tensor([[1.0, 2.0]], dtype=torch.float32)
            },
        },
        ckpt_path,
    )

    try:
        export_transformer_gene_embeddings(str(ckpt_path), str(tmp_path / "out.tsv.gz"))
        assert False, "Expected ValueError for non-transformer checkpoint."
    except ValueError:
        pass


def test_export_transformer_cell_embeddings_writes_tsv(tmp_path: Path) -> None:
    h5ad_path = _write_tiny_h5ad(tmp_path)

    encoder = TransformerCellEncoder(
        n_genes=4,
        latent_dim=5,
        hidden_dim=8,
        n_hidden_layers=1,
        input_transform="none",
        transformer_d_model=8,
        transformer_n_heads=2,
        transformer_n_layers=1,
        transformer_ff_mult=2,
        token_mlp_hidden_dim=8,
        token_mlp_layers=1,
    )
    decoder = ZINBExpressionDecoder(
        n_genes=4,
        latent_dim=5,
        hidden_dim=8,
        n_hidden_layers=1,
    )
    model = GeneVAE(encoder, decoder)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    ckpt_path = tmp_path / "transformer_cell_export.ckpt"
    save_vae_checkpoint(
        path=str(ckpt_path),
        model=model,
        optimizer=opt,
        epoch=0,
        genes_common=["g0", "g1", "g2", "g3"],
        config_snapshot={
            "encoder_type": "transformer",
            "latent_dim": 5,
            "hidden_dim": 8,
            "n_hidden_layers": 1,
            "input_transform": "none",
            "transformer_d_model": 8,
            "transformer_n_heads": 2,
            "transformer_n_layers": 1,
            "transformer_ff_mult": 2,
            "token_mlp_hidden_dim": 8,
            "token_mlp_layers": 1,
            "perturbation_mode": "none",
        },
        gene_emb_source="",
    )

    out_path = tmp_path / "transformer_cell_emb.tsv.gz"
    export_transformer_cell_embeddings(
        adata_path=str(h5ad_path),
        checkpoint_path=str(ckpt_path),
        out_tsv_gz=str(out_path),
        gene_key="gene",
        batch_size=2,
        num_workers=0,
        device="cpu",
        backed=False,
        max_tokens_per_cell_override=2,
        min_expr_for_token_override=0.0,
    )

    df = pd.read_csv(out_path, sep="\t")
    assert df.shape[0] == 4
    assert df.columns.tolist() == ["cell_id", "dim_1", "dim_2", "dim_3", "dim_4", "dim_5"]
    assert df["cell_id"].tolist() == ["cell_0", "cell_1", "cell_2", "cell_3"]


def test_export_transformer_cell_embeddings_rejects_non_transformer(tmp_path: Path) -> None:
    h5ad_path = _write_tiny_h5ad(tmp_path)
    ckpt_path = tmp_path / "not_transformer_cell_export.ckpt"
    torch.save(
        {
            "config": {"encoder_type": "perceiver"},
            "genes_common": ["g0", "g1", "g2", "g3"],
            "model_state_dict": {},
        },
        ckpt_path,
    )

    try:
        export_transformer_cell_embeddings(
            adata_path=str(h5ad_path),
            checkpoint_path=str(ckpt_path),
            out_tsv_gz=str(tmp_path / "unused.tsv.gz"),
            gene_key="gene",
            batch_size=2,
            num_workers=0,
            device="cpu",
            backed=False,
        )
        assert False, "Expected ValueError for non-transformer checkpoint."
    except ValueError:
        pass
