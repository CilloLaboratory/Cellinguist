from __future__ import annotations

from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import torch

from cellinguist.data.datasets import SingleCellVAEDataset
from cellinguist.models.vae import (
    BatchAdversary,
    GeneVAE,
    PerceiverCellEncoder,
    ZINBExpressionDecoder,
)
from cellinguist.scripts.export_vae_predictions import _load_counterfactual_overrides
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
