from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader

from cellinguist.config import CBOWConfig, load_config
from cellinguist.data.datasets import (
    CBOWPairsConfig,
    CBOWPairsDataset,
    SingleCellDataset,
    build_neg_sampling_dist_from_dataset,
)
from cellinguist.embeddings import (
    load_gene_embeddings,
    load_token_vocab,
    save_gene_embeddings,
    save_token_vocab,
)
from cellinguist.models.cbow import CBOWModel, cbow_negative_sampling_loss


def _resolve_vocab_output_path(out_path: Path) -> Path:
    return out_path.with_suffix('.vocab.json')


def _validate_warm_start_config(config: CBOWConfig) -> None:
    has_init_embeddings = config.init_embeddings_path is not None
    has_init_vocab = config.init_vocab_path is not None
    if has_init_embeddings != has_init_vocab:
        raise ValueError('init_embeddings_path and init_vocab_path must be provided together.')
    if config.vocab_expansion_mode not in {'strict', 'expand'}:
        raise ValueError(
            f'Unsupported vocab_expansion_mode: {config.vocab_expansion_mode}'
        )


def initialize_cbow_model(
    config: CBOWConfig,
    dataset: SingleCellDataset,
    device: torch.device,
) -> CBOWModel:
    _validate_warm_start_config(config)

    model = CBOWModel(
        vocab_size=dataset.vocab_size,
        emb_dim=config.emb_dim,
        use_separate_output=config.use_separate_output,
    ).to(device)

    if config.init_embeddings_path is None:
        return model

    old_embeddings = load_gene_embeddings(config.init_embeddings_path, map_location='cpu')
    old_vocab = load_token_vocab(config.init_vocab_path)

    if old_embeddings.dim() != 2:
        raise ValueError(
            f'Warm-start embeddings must be rank-2, got shape {tuple(old_embeddings.shape)}.'
        )
    if old_embeddings.shape[0] != len(old_vocab):
        raise ValueError(
            'Warm-start embeddings row count does not match loaded vocabulary size: '
            f'{old_embeddings.shape[0]} vs {len(old_vocab)}.'
        )
    if old_embeddings.shape[1] != config.emb_dim:
        raise ValueError(
            'Warm-start embeddings dimension does not match config.emb_dim: '
            f'{old_embeddings.shape[1]} vs {config.emb_dim}.'
        )

    new_vocab = dataset.token_to_id
    if config.vocab_expansion_mode == 'strict' and old_vocab != new_vocab:
        old_only = sorted(set(old_vocab) - set(new_vocab))
        new_only = sorted(set(new_vocab) - set(old_vocab))
        raise ValueError(
            'Strict warm start requires an identical vocabulary. '
            f'Tokens only in old vocab: {old_only[:5]}; '
            f'tokens only in new vocab: {new_only[:5]}.'
        )

    shared_tokens = sorted(set(old_vocab).intersection(new_vocab))
    if not shared_tokens:
        raise ValueError('No overlapping tokens found between warm-start vocab and new dataset vocab.')

    with torch.no_grad():
        for token in shared_tokens:
            old_idx = old_vocab[token]
            new_idx = new_vocab[token]
            model.input_emb.weight[new_idx].copy_(old_embeddings[old_idx].to(device))

    print(
        f"[CBOW] Warm-started {len(shared_tokens)} shared tokens "
        f"using mode='{config.vocab_expansion_mode}'."
    )
    if config.use_separate_output:
        print(
            '[CBOW] output_emb kept at random initialization because the warm-start '
            'artifact only stores exported input embeddings.'
        )

    return model


def train_cbow(
    config: CBOWConfig,
    dataset: SingleCellDataset,
) -> torch.Tensor:
    """Train a CBOWModel on token sequences from a SingleCellDataset."""
    device = torch.device(config.device)

    pairs_cfg = CBOWPairsConfig(
        window_size=config.window_size,
        samples_per_cell=config.samples_per_cell,
    )
    cbow_pairs_ds = CBOWPairsDataset(
        sc_dataset=dataset,
        config=pairs_cfg,
        rng=None,
    )

    dataloader = DataLoader(
        cbow_pairs_ds,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=(device.type == 'cuda'),
        persistent_workers=(config.num_workers > 0),
    )

    neg_sampling_dist_np = build_neg_sampling_dist_from_dataset(dataset)
    neg_sampling_dist = torch.from_numpy(neg_sampling_dist_np).to(
        device=device, dtype=torch.float32
    )

    model = initialize_cbow_model(config=config, dataset=dataset, device=device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.lr,
        weight_decay=config.weight_decay,
    )

    if config.lr_scheduler == 'none':
        scheduler = None
    elif config.lr_scheduler == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=config.lr_step_size,
            gamma=config.lr_gamma,
        )
    else:
        raise ValueError(f'Unsupported lr_scheduler: {config.lr_scheduler}')

    for epoch in range(config.epochs):
        model.train()
        total_loss = 0.0
        total_batches = 0

        for batch in dataloader:
            target_ids = batch['target_ids'].to(device, non_blocking=True)
            context_ids = batch['context_ids'].to(device, non_blocking=True)

            batch_size = target_ids.size(0)
            num_negatives = config.num_negatives
            negative_ids = torch.multinomial(
                neg_sampling_dist,
                num_samples=batch_size * num_negatives,
                replacement=True,
            ).view(batch_size, num_negatives)

            pos_logits, neg_logits = model(
                target_ids=target_ids,
                context_ids=context_ids,
                negative_ids=negative_ids,
            )
            loss = cbow_negative_sampling_loss(
                pos_logits,
                neg_logits,
                reduction='mean',
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += float(loss.item())
            total_batches += 1

        avg_loss = total_loss / max(total_batches, 1)
        print(f'[CBOW] Epoch {epoch+1}/{config.epochs} - loss: {avg_loss:.4f}')

        if scheduler is not None:
            scheduler.step()

    return model.input_emb.weight.detach().cpu()


def run_cbow_training_from_config(config_path: str) -> None:
    """Load config, train CBOW embeddings, and save artifacts."""
    raw_cfg = load_config(config_path)
    cbow_cfg = CBOWConfig(**raw_cfg['cbow'])

    data_cfg = raw_cfg['data']
    dataset = SingleCellDataset(
        adata_or_path=data_cfg['adata_path'],
        gene_key=data_cfg.get('gene_key', 'gene'),
        cond_key=data_cfg.get('cond_key', None),
        layer=data_cfg.get('layer', None),
        n_bins=data_cfg.get('n_bins', 20),
        shuffle_tokens=data_cfg.get('shuffle_tokens', True),
        min_expr=data_cfg.get('min_expr', 0.0),
        token_to_id=None,
        min_token_count=data_cfg.get('min_token_count', 1),
    )

    embeddings = train_cbow(cbow_cfg, dataset)

    out_path = Path(raw_cfg.get('output', {}).get('embeddings_path', 'gene_embeddings.pt'))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_gene_embeddings(str(out_path), embeddings)
    vocab_path = _resolve_vocab_output_path(out_path)
    save_token_vocab(str(vocab_path), dataset.token_to_id)

    print(f'[CBOW] Saved embeddings to: {out_path}')
    print(f'[CBOW] Saved token vocabulary to: {vocab_path}')


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Train CBOW embeddings from config file.')
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to YAML/JSON config file.'
    )
    args = parser.parse_args()
    run_cbow_training_from_config(args.config)


if __name__ == '__main__':
    main()
