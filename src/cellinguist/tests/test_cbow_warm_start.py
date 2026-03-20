from __future__ import annotations

from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import torch

from cellinguist.config import CBOWConfig
from cellinguist.data.datasets import SingleCellDataset
from cellinguist.embeddings import load_token_vocab, save_gene_embeddings, save_token_vocab
from cellinguist.scripts.export_gene_embeddings import compute_gene_embeddings
from cellinguist.train.train_cbow import initialize_cbow_model


def _write_h5ad(tmp_path: Path, x: np.ndarray, genes: list[str], name: str) -> Path:
    obs = pd.DataFrame(index=[f'cell_{i}' for i in range(x.shape[0])])
    var = pd.DataFrame({'gene': genes})
    adata = ad.AnnData(X=x.astype(np.float32), obs=obs, var=var)
    out = tmp_path / f'{name}.h5ad'
    adata.write_h5ad(out)
    return out


def _dataset(path: Path) -> SingleCellDataset:
    return SingleCellDataset(
        adata_or_path=str(path),
        gene_key='gene',
        cond_key=None,
        layer=None,
        n_bins=3,
        shuffle_tokens=False,
        min_expr=0.0,
        token_to_id=None,
        min_token_count=1,
    )


def _warm_start_artifacts(tmp_path: Path, vocab: dict[str, int], emb_dim: int = 4) -> tuple[Path, Path, torch.Tensor]:
    weight = torch.arange(len(vocab) * emb_dim, dtype=torch.float32).view(len(vocab), emb_dim)
    emb_path = tmp_path / 'warm_start.pt'
    vocab_path = tmp_path / 'warm_start.vocab.json'
    save_gene_embeddings(str(emb_path), weight)
    save_token_vocab(str(vocab_path), vocab)
    return emb_path, vocab_path, weight


def test_token_vocab_round_trip(tmp_path: Path) -> None:
    vocab = {'g1__1': 0, 'g2__2': 1}
    vocab_path = tmp_path / 'tokens.vocab.json'
    save_token_vocab(str(vocab_path), vocab)
    assert load_token_vocab(str(vocab_path)) == vocab


def test_warm_start_requires_both_init_paths(tmp_path: Path) -> None:
    h5ad_path = _write_h5ad(
        tmp_path,
        np.array([[1, 2], [2, 1]], dtype=np.float32),
        ['g1', 'g2'],
        'missing_pair',
    )
    ds = _dataset(h5ad_path)
    emb_path, _, weight = _warm_start_artifacts(tmp_path, ds.token_to_id)

    try:
        initialize_cbow_model(
            CBOWConfig(
                emb_dim=weight.shape[1],
                device='cpu',
                init_embeddings_path=str(emb_path),
                init_vocab_path=None,
                vocab_expansion_mode='strict',
            ),
            ds,
            torch.device('cpu'),
        )
        assert False, 'Expected missing init_vocab_path to fail.'
    except ValueError as exc:
        assert 'provided together' in str(exc)


def test_strict_warm_start_accepts_identical_vocab(tmp_path: Path) -> None:
    h5ad_path = _write_h5ad(
        tmp_path,
        np.array([[1, 2], [2, 1], [3, 1]], dtype=np.float32),
        ['g1', 'g2'],
        'strict_match',
    )
    ds = _dataset(h5ad_path)
    emb_path, vocab_path, weight = _warm_start_artifacts(tmp_path, ds.token_to_id)

    model = initialize_cbow_model(
        CBOWConfig(
            emb_dim=weight.shape[1],
            device='cpu',
            init_embeddings_path=str(emb_path),
            init_vocab_path=str(vocab_path),
            vocab_expansion_mode='strict',
        ),
        ds,
        torch.device('cpu'),
    )

    for token, idx in ds.token_to_id.items():
        expected = weight[idx]
        assert torch.equal(model.input_emb.weight[idx].detach().cpu(), expected)


def test_strict_warm_start_rejects_vocab_mismatch(tmp_path: Path) -> None:
    h5ad_path = _write_h5ad(
        tmp_path,
        np.array([[1, 2, 0], [2, 1, 1], [3, 1, 1]], dtype=np.float32),
        ['g1', 'g2', 'g3'],
        'strict_mismatch',
    )
    ds = _dataset(h5ad_path)
    old_vocab = {'g1__1': 0, 'g2__1': 1}
    emb_path, vocab_path, _ = _warm_start_artifacts(tmp_path, old_vocab)

    try:
        initialize_cbow_model(
            CBOWConfig(
                emb_dim=4,
                device='cpu',
                init_embeddings_path=str(emb_path),
                init_vocab_path=str(vocab_path),
                vocab_expansion_mode='strict',
            ),
            ds,
            torch.device('cpu'),
        )
        assert False, 'Expected strict warm start to fail on mismatched vocab.'
    except ValueError as exc:
        assert 'identical vocabulary' in str(exc)


def test_expand_warm_start_preserves_shared_rows_and_keeps_new_rows(tmp_path: Path) -> None:
    base_path = _write_h5ad(
        tmp_path,
        np.array([[1, 2], [2, 1], [3, 1]], dtype=np.float32),
        ['g1', 'g2'],
        'base',
    )
    expanded_path = _write_h5ad(
        tmp_path,
        np.array([[1, 2, 4], [2, 1, 3], [3, 1, 2]], dtype=np.float32),
        ['g1', 'g2', 'g3'],
        'expanded',
    )
    base_ds = _dataset(base_path)
    expanded_ds = _dataset(expanded_path)
    emb_path, vocab_path, weight = _warm_start_artifacts(tmp_path, base_ds.token_to_id)

    model = initialize_cbow_model(
        CBOWConfig(
            emb_dim=weight.shape[1],
            device='cpu',
            init_embeddings_path=str(emb_path),
            init_vocab_path=str(vocab_path),
            vocab_expansion_mode='expand',
        ),
        expanded_ds,
        torch.device('cpu'),
    )

    shared_tokens = set(base_ds.token_to_id).intersection(expanded_ds.token_to_id)
    new_tokens = set(expanded_ds.token_to_id) - set(base_ds.token_to_id)
    assert shared_tokens
    assert new_tokens

    for token in shared_tokens:
        new_idx = expanded_ds.token_to_id[token]
        old_idx = base_ds.token_to_id[token]
        assert torch.equal(model.input_emb.weight[new_idx].detach().cpu(), weight[old_idx])

    for token in new_tokens:
        new_idx = expanded_ds.token_to_id[token]
        assert torch.isfinite(model.input_emb.weight[new_idx]).all()


def test_warm_start_rejects_embedding_row_count_mismatch(tmp_path: Path) -> None:
    h5ad_path = _write_h5ad(
        tmp_path,
        np.array([[1, 2], [2, 1]], dtype=np.float32),
        ['g1', 'g2'],
        'row_mismatch',
    )
    ds = _dataset(h5ad_path)
    emb_path = tmp_path / 'row_mismatch.pt'
    vocab_path = tmp_path / 'row_mismatch.vocab.json'
    save_gene_embeddings(str(emb_path), torch.arange(4, dtype=torch.float32).view(1, 4))
    save_token_vocab(str(vocab_path), {'g1__1': 0, 'g2__1': 1})

    try:
        initialize_cbow_model(
            CBOWConfig(
                emb_dim=4,
                device='cpu',
                init_embeddings_path=str(emb_path),
                init_vocab_path=str(vocab_path),
                vocab_expansion_mode='expand',
            ),
            ds,
            torch.device('cpu'),
        )
        assert False, 'Expected row-count mismatch to fail.'
    except ValueError as exc:
        assert 'row count' in str(exc)


def test_warm_start_rejects_embedding_dim_mismatch(tmp_path: Path) -> None:
    h5ad_path = _write_h5ad(
        tmp_path,
        np.array([[1, 2], [2, 1]], dtype=np.float32),
        ['g1', 'g2'],
        'dim_mismatch',
    )
    ds = _dataset(h5ad_path)
    emb_path, vocab_path, _ = _warm_start_artifacts(tmp_path, ds.token_to_id, emb_dim=3)

    try:
        initialize_cbow_model(
            CBOWConfig(
                emb_dim=4,
                device='cpu',
                init_embeddings_path=str(emb_path),
                init_vocab_path=str(vocab_path),
                vocab_expansion_mode='strict',
            ),
            ds,
            torch.device('cpu'),
        )
        assert False, 'Expected dimension mismatch to fail.'
    except ValueError as exc:
        assert 'dimension' in str(exc)


def test_export_gene_embeddings_after_vocab_expansion(tmp_path: Path) -> None:
    expanded_path = _write_h5ad(
        tmp_path,
        np.array([[1, 2, 4], [2, 1, 3], [3, 1, 2]], dtype=np.float32),
        ['g1', 'g2', 'g3'],
        'export',
    )
    expanded_ds = _dataset(expanded_path)
    emb_dim = 5
    weights = torch.arange(expanded_ds.vocab_size * emb_dim, dtype=torch.float32).view(
        expanded_ds.vocab_size, emb_dim
    )

    genes, gene_embs = compute_gene_embeddings(expanded_ds.id_to_token, weights)

    assert set(genes.tolist()) == {'g1', 'g2', 'g3'}
    assert gene_embs.shape == (3, emb_dim)
