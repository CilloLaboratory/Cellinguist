from __future__ import annotations

from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd

from cellinguist.config import VAETrainConfig
from cellinguist.data.datasets import SingleCellVAEDataset
from cellinguist.scripts.precompute_transformer_token_indices import precompute_token_index_cache
from cellinguist.train.train_vae import train_vae


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
    obs = pd.DataFrame(index=[f"cell_{i}" for i in range(4)])
    var = pd.DataFrame({"gene": [f"g{i}" for i in range(4)]})
    adata = ad.AnnData(X=x, obs=obs, var=var)
    out = tmp_path / "tiny.h5ad"
    adata.write_h5ad(out)
    return out


def test_precompute_cache_writes_expected_files(tmp_path: Path) -> None:
    h5ad_path = _write_tiny_h5ad(tmp_path)
    out_dir = tmp_path / "cache"

    meta = precompute_token_index_cache(
        adata_path=str(h5ad_path),
        out_dir=str(out_dir),
        gene_key="gene",
        layer=None,
        min_expr_for_token=0.0,
        max_tokens_per_cell=None,
        shard_size_cells=2,
        num_workers=1,
    )

    assert (out_dir / "metadata.json").exists()
    assert (out_dir / "indices_00000.npy").exists()
    assert (out_dir / "offsets_00000.npy").exists()
    assert (out_dir / "indices_00001.npy").exists()
    assert (out_dir / "offsets_00001.npy").exists()

    assert int(meta["n_cells"]) == 4
    assert int(meta["n_genes"]) == 4
    assert len(meta["shards"]) == 2

    offsets = np.load(out_dir / "offsets_00000.npy")
    indices = np.load(out_dir / "indices_00000.npy")
    assert offsets.shape[0] == 3
    assert np.all(offsets[1:] >= offsets[:-1])
    assert indices.dtype == np.int32
    if indices.size > 0:
        assert int(indices.min()) >= 0
        assert int(indices.max()) < 4


def test_dataset_cache_matches_in_memory_precompute(tmp_path: Path) -> None:
    h5ad_path = _write_tiny_h5ad(tmp_path)
    out_dir = tmp_path / "cache"
    precompute_token_index_cache(
        adata_path=str(h5ad_path),
        out_dir=str(out_dir),
        gene_key="gene",
        min_expr_for_token=0.0,
        max_tokens_per_cell=2,
        shard_size_cells=2,
        num_workers=1,
    )

    ds_mem = SingleCellVAEDataset(
        adata_or_path=str(h5ad_path),
        gene_key="gene",
        transform="none",
        backed=False,
        precompute_token_gene_indices=True,
        token_min_expr=0.0,
        token_max_genes=2,
    )
    ds_cache = SingleCellVAEDataset(
        adata_or_path=str(h5ad_path),
        gene_key="gene",
        transform="none",
        backed=False,
        token_min_expr=0.0,
        token_max_genes=2,
        token_index_cache_dir=str(out_dir),
        token_index_cache_require=True,
    )

    for i in range(len(ds_mem)):
        a = ds_mem[i]["token_gene_idx"].numpy()
        b = ds_cache[i]["token_gene_idx"].numpy()
        assert np.array_equal(a, b)


def test_dataset_cache_mismatch_raises(tmp_path: Path) -> None:
    h5ad_path = _write_tiny_h5ad(tmp_path)
    out_dir = tmp_path / "cache"
    precompute_token_index_cache(
        adata_path=str(h5ad_path),
        out_dir=str(out_dir),
        gene_key="gene",
        min_expr_for_token=0.0,
        max_tokens_per_cell=2,
        shard_size_cells=2,
        num_workers=1,
    )

    try:
        SingleCellVAEDataset(
            adata_or_path=str(h5ad_path),
            gene_key="gene",
            transform="none",
            backed=False,
            token_min_expr=0.0,
            token_max_genes=3,
            token_index_cache_dir=str(out_dir),
            token_index_cache_require=True,
        )
        assert False, "Expected ValueError for max_tokens_per_cell mismatch"
    except ValueError:
        pass


def test_train_vae_transformer_cache_required_and_smoke(tmp_path: Path) -> None:
    h5ad_path = _write_tiny_h5ad(tmp_path)
    cache_dir = tmp_path / "cache"
    precompute_token_index_cache(
        adata_path=str(h5ad_path),
        out_dir=str(cache_dir),
        gene_key="gene",
        min_expr_for_token=0.0,
        max_tokens_per_cell=2,
        shard_size_cells=2,
        num_workers=1,
    )

    cfg_fail = VAETrainConfig(
        adata_path=str(h5ad_path),
        gene_key="gene",
        encoder_type="transformer",
        device="cpu",
        epochs=1,
        batch_size=2,
        num_workers=0,
        transformer_precompute_token_indices=False,
        token_index_cache_dir="",
        token_index_cache_require=True,
        decoder_mu_init="constant",
        decoder_mu_init_constant=0.1,
        checkpoint_dir=str(tmp_path / "ckpt_fail"),
        run_name="fail",
    )
    try:
        train_vae(cfg_fail)
        assert False, "Expected ValueError when cache is required but cache dir is missing"
    except ValueError:
        pass

    cfg_ok = VAETrainConfig(
        adata_path=str(h5ad_path),
        gene_key="gene",
        encoder_type="transformer",
        latent_dim=4,
        hidden_dim=8,
        n_hidden_layers=1,
        cond_emb_dim=4,
        input_transform="none",
        transformer_d_model=8,
        transformer_n_heads=2,
        transformer_n_layers=1,
        transformer_ff_mult=2,
        transformer_dropout=0.0,
        token_mlp_hidden_dim=8,
        token_mlp_layers=1,
        min_expr_for_token=0.0,
        max_tokens_per_cell=2,
        transformer_precompute_token_indices=False,
        token_index_cache_dir=str(cache_dir),
        token_index_cache_require=True,
        perturbation_mode="none",
        lr=1e-3,
        weight_decay=0.0,
        batch_size=2,
        epochs=1,
        num_workers=0,
        device="cpu",
        decoder_mu_init="constant",
        decoder_mu_init_constant=0.1,
        checkpoint_dir=str(tmp_path / "ckpt_ok"),
        run_name="ok",
        save_every=1,
    )
    ckpt_path = train_vae(cfg_ok)
    assert Path(ckpt_path).exists()
