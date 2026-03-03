from __future__ import annotations

import argparse
import datetime
import os
import traceback
from dataclasses import asdict
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from cellinguist.config import VAETrainConfig, load_yaml
from cellinguist.data.datasets import SingleCellVAEDataset
from cellinguist.models.vae import (
    CBOWCellEncoder,
    PerceiverCellEncoder,
    ZINBExpressionDecoder,
    GeneVAE,
    zinb_negative_log_likelihood,
    kl_divergence_normal,
    expression_triplet_metric_loss,
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


def _is_distributed() -> bool:
    return dist.is_available() and dist.is_initialized()


def _early_log(msg: str) -> None:
    print(f"[ddp-setup] {msg}", flush=True)


def _setup_distributed(cfg_device: str) -> tuple[torch.device, int, int, bool]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size <= 1:
        return torch.device(cfg_device), 0, 1, False

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    rank = int(os.environ.get("RANK", str(local_rank)))
    master_addr = os.environ.get("MASTER_ADDR", "unset")
    master_port = os.environ.get("MASTER_PORT", "unset")
    _early_log(
        f"pre-init rank={rank} local_rank={local_rank} world_size={world_size} "
        f"master={master_addr}:{master_port}"
    )

    if not torch.cuda.is_available():
        _early_log("CUDA unavailable; using gloo backend.")
        backend = "gloo"
        device = torch.device("cpu")
    else:
        torch.cuda.set_device(local_rank)
        backend = os.environ.get("CELLINGUIST_DDP_BACKEND", "nccl").lower()
        device = torch.device("cuda", local_rank)
        _early_log(
            f"cuda_visible={os.environ.get('CUDA_VISIBLE_DEVICES', 'all')} "
            f"device_count={torch.cuda.device_count()} backend={backend}"
        )

    try:
        dist.init_process_group(
            backend=backend,
            init_method="env://",
            timeout=datetime.timedelta(minutes=5),
        )
        _early_log(f"init_process_group ok rank={dist.get_rank()} world_size={dist.get_world_size()}")
    except Exception:
        _early_log("init_process_group failed with exception:")
        traceback.print_exc()
        raise

    return device, dist.get_rank(), dist.get_world_size(), True


def _unwrap(model: torch.nn.Module) -> torch.nn.Module:
    return model.module if isinstance(model, DDP) else model


def _log(rank: int, msg: str) -> None:
    print(f"[rank {rank}] {msg}", flush=True)


def train_vae(cfg: VAETrainConfig) -> str:
    device, rank, world_size, is_ddp = _setup_distributed(cfg.device)

    try:
        _log(
            rank,
            f"startup: ddp={is_ddp} world_size={world_size} device={device} "
            f"cfg_num_workers={cfg.num_workers} resume_from={cfg.resume_from}",
        )
        seed = None if cfg.seed is None else int(cfg.seed) + rank
        set_seed(seed)
        _log(rank, f"seed set to {seed}")

        is_main = rank == 0
        encoder_type = str(cfg.encoder_type).lower()
        if encoder_type not in {"cbow", "perceiver"}:
            raise ValueError(f"Unsupported encoder_type: {cfg.encoder_type}. Use 'cbow' or 'perceiver'.")
        _log(rank, f"encoder_type={encoder_type}")
        if cfg.use_metric_loss:
            if cfg.metric_loss_weight < 0:
                raise ValueError("metric_loss_weight must be >= 0.")
            if cfg.metric_margin < 0:
                raise ValueError("metric_margin must be >= 0.")
            if cfg.metric_k_pos < 1 or cfg.metric_k_neg < 1:
                raise ValueError("metric_k_pos and metric_k_neg must be >= 1.")
            if cfg.metric_expr_transform not in {"log1p", "none"}:
                raise ValueError(
                    f"Unsupported metric_expr_transform: {cfg.metric_expr_transform}"
                )

        if encoder_type == "cbow":
            if not cfg.gene_emb_tsv:
                raise ValueError("gene_emb_tsv must be provided when encoder_type='cbow'.")

            genes_from_emb, emb_full = load_gene_embeddings_tsv(cfg.gene_emb_tsv)
            ds_probe = SingleCellVAEDataset(
                adata_or_path=cfg.adata_path,
                gene_key=cfg.gene_key,
                layer=cfg.layer,
                cond_key=cfg.cond_key,
                transform="none",
                backed=cfg.backed,
            )
            genes_expr = ds_probe.gene_order
            if is_main:
                print(f"Expression dataset: {len(genes_expr)} genes before intersection")

            genes_common = intersect_genes_in_embedding_order(genes_from_emb, genes_expr)
            emb = subset_embeddings(genes_from_emb, emb_full, genes_common)

            vae_dataset = SingleCellVAEDataset(
                adata_or_path=cfg.adata_path,
                gene_key=cfg.gene_key,
                layer=cfg.layer,
                cond_key=cfg.cond_key,
                gene_order=genes_common,
                transform="none",
                backed=cfg.backed,
            )
            n_cells, n_genes = vae_dataset.n_cells, vae_dataset.n_genes
            if is_main:
                print(f"VAE dataset after alignment: {n_cells} cells, {n_genes} genes")
            assert n_genes == emb.shape[0]
            gene_emb_source = cfg.gene_emb_tsv
        else:
            if cfg.use_library_size_covariate and str(cfg.library_norm).lower() != "none" and is_main:
                print(
                    "[VAE] WARNING: use_library_size_covariate=True with library_norm!='none'. "
                    "For a pure covariate strategy, set library_norm='none'."
                )
            vae_dataset = SingleCellVAEDataset(
                adata_or_path=cfg.adata_path,
                gene_key=cfg.gene_key,
                layer=cfg.layer,
                cond_key=cfg.cond_key,
                transform="none",
                backed=cfg.backed,
            )
            genes_common = vae_dataset.gene_order
            n_cells, n_genes = vae_dataset.n_cells, vae_dataset.n_genes
            if is_main:
                print(f"VAE dataset: {n_cells} cells, {n_genes} genes (Perceiver encoder)")
            emb = None
            gene_emb_source = ""
        _log(rank, f"dataset ready: n_cells={n_cells} n_genes={n_genes}")

        n_conditions = (
            len(vae_dataset.cond_categories) if vae_dataset.cond_categories is not None else None
        )

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
                library_norm=cfg.library_norm,
                library_norm_target_sum=cfg.library_norm_target_sum,
                library_norm_eps=cfg.library_norm_eps,
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
            use_library_size_covariate=cfg.use_library_size_covariate,
            library_size_covariate_eps=cfg.library_size_covariate_eps,
        )
        model: torch.nn.Module = GeneVAE(encoder, decoder).to(device)
        _log(rank, "model built")

        if is_ddp:
            model = DDP(model, device_ids=[device.index], output_device=device.index)
            _log(rank, "DDP wrapper initialized")

        raw_model = _unwrap(model)

        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=cfg.lr,
            weight_decay=cfg.weight_decay,
        )

        start_epoch = 0
        if cfg.resume_from:
            ckpt = load_vae_checkpoint(cfg.resume_from, raw_model, optimizer, map_location=device)
            start_epoch = int(ckpt.get("epoch", -1)) + 1

            ckpt_genes = ckpt.get("genes_common", None)
            if ckpt_genes is not None and ckpt_genes != genes_common:
                raise ValueError("genes_common mismatch between checkpoint and current data/embeddings.")
        _log(rank, f"checkpoint state ready: start_epoch={start_epoch}")

        sampler = None
        if is_ddp:
            sampler = DistributedSampler(
                vae_dataset,
                num_replicas=world_size,
                rank=rank,
                shuffle=True,
            )
            _log(rank, "distributed sampler ready")

        # Safer DDP default: avoid worker subprocesses unless explicitly requested >0.
        effective_num_workers = int(cfg.num_workers)
        if is_ddp and effective_num_workers <= 0:
            effective_num_workers = 0

        dl = DataLoader(
            vae_dataset,
            batch_size=cfg.batch_size,
            shuffle=(sampler is None),
            sampler=sampler,
            num_workers=effective_num_workers,
            pin_memory=(device.type == "cuda"),
            persistent_workers=effective_num_workers > 0,
        )
        _log(rank, f"dataloader ready: batch_size={cfg.batch_size} num_workers={effective_num_workers}")

        ckpt_dir = Path(cfg.checkpoint_dir)
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        ckpt_last = str(ckpt_dir / f"{cfg.run_name}_last.ckpt")
        _log(rank, f"checkpoint path={ckpt_last}")

        if not cfg.resume_from:
            if hasattr(raw_model.decoder, "log_theta"):
                theta_init = torch.full((n_genes,), float(cfg.decoder_theta_init), dtype=torch.float32)
                raw_model.decoder.log_theta.data = inv_softplus(theta_init).to(raw_model.decoder.log_theta.data.device)

            if hasattr(raw_model.decoder, "mlp_pi"):
                pi_linear = raw_model.decoder.mlp_pi.net[-1]
                pi_linear.bias.data.fill_(logit(cfg.decoder_pi_init))
                torch.nn.init.zeros_(pi_linear.weight)

            if hasattr(raw_model.decoder, "mlp_mu"):
                mu_linear = raw_model.decoder.mlp_mu.net[-1]

                if cfg.decoder_mu_init == "data_mean":
                    if is_main:
                        mean_x = estimate_gene_means(
                            vae_dataset,
                            max_cells=cfg.decoder_init_n_cells,
                            batch_size=cfg.decoder_init_batch_size,
                            num_workers=cfg.decoder_init_num_workers,
                            device=None,
                        )
                        mean_x = torch.clamp(
                            mean_x,
                            min=cfg.decoder_mu_init_eps,
                            max=cfg.decoder_mu_init_cap,
                        )
                    else:
                        mean_x = torch.empty((n_genes,), dtype=torch.float32, device=device)

                    if is_ddp:
                        # NCCL does not support CPU tensors for collectives.
                        if is_main:
                            mean_x = mean_x.to(device=device, dtype=torch.float32, non_blocking=True)
                        dist.broadcast(mean_x, src=0)
                    elif mean_x.device != device:
                        mean_x = mean_x.to(device=device, dtype=torch.float32, non_blocking=True)

                    mu_bias = inv_softplus(mean_x).to(mu_linear.bias.device)
                    mu_linear.bias.data.copy_(mu_bias)
                    torch.nn.init.zeros_(mu_linear.weight)

                elif cfg.decoder_mu_init == "constant":
                    mean0 = torch.tensor(cfg.decoder_mu_init_constant)
                    mean0 = torch.clamp(mean0, min=cfg.decoder_mu_init_eps, max=cfg.decoder_mu_init_cap)
                    b0 = inv_softplus(mean0).item()
                    mu_linear.bias.data.fill_(b0)
                    torch.nn.init.zeros_(mu_linear.weight)

        if is_ddp:
            for p in raw_model.parameters():
                dist.broadcast(p.data, src=0)
            for b in raw_model.buffers():
                dist.broadcast(b.data, src=0)
            _log(rank, "initial parameter/buffer broadcast complete")

        model.train()
        _log(rank, f"training loop start: epochs={cfg.epochs} start_epoch={start_epoch}")
        for epoch in range(start_epoch, cfg.epochs):
            if sampler is not None:
                sampler.set_epoch(epoch)

            total = 0.0
            total_recon = 0.0
            total_kl = 0.0
            total_metric = 0.0
            nb = 0

            for batch in dl:
                x = batch["x_expr"].to(device, non_blocking=True)
                libsize = x.sum(dim=1)
                cond_idx = batch.get("cond_idx", None)
                if cond_idx is not None:
                    cond_idx = cond_idx.to(device, non_blocking=True)

                recon_out, mu_z, logvar_z = model(x, cond_idx, libsize=libsize)
                mu, theta, pi = recon_out

                recon = zinb_negative_log_likelihood(x, mu, theta, pi, reduction="mean")
                kl = kl_divergence_normal(mu_z, logvar_z, reduction="mean")
                metric = x.new_zeros(())
                if cfg.use_metric_loss and cfg.metric_loss_weight > 0:
                    metric = expression_triplet_metric_loss(
                        x_expr=x,
                        z_latent=mu_z,
                        expr_transform=cfg.metric_expr_transform,
                        margin=cfg.metric_margin,
                        k_pos=cfg.metric_k_pos,
                        k_neg=cfg.metric_k_neg,
                    )
                loss = recon + cfg.kl_weight * kl + cfg.metric_loss_weight * metric

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if cfg.grad_clip_norm and cfg.grad_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip_norm)
                optimizer.step()

                total += float(loss.item())
                total_recon += float(recon.item())
                total_kl += float(kl.item())
                total_metric += float(metric.item())
                nb += 1

            if is_ddp:
                loss_stats = torch.tensor(
                    [total, total_recon, total_kl, total_metric, float(nb)],
                    device=device,
                )
                dist.all_reduce(loss_stats, op=dist.ReduceOp.SUM)
                total = float(loss_stats[0].item())
                total_recon = float(loss_stats[1].item())
                total_kl = float(loss_stats[2].item())
                total_metric = float(loss_stats[3].item())
                nb = int(loss_stats[4].item())

            if is_main:
                denom = max(nb, 1)
                print(
                    f"[VAE] Epoch {epoch+1}/{cfg.epochs} "
                    f"loss={total/denom:.4f} "
                    f"recon={total_recon/denom:.4f} "
                    f"kl={total_kl/denom:.4f} "
                    f"metric={total_metric/denom:.4f}"
                )

                if cfg.save_every > 0 and ((epoch + 1) % cfg.save_every == 0):
                    save_vae_checkpoint(
                        ckpt_last,
                        model=raw_model,
                        optimizer=optimizer,
                        epoch=epoch,
                        genes_common=genes_common,
                        config_snapshot=asdict(cfg),
                        gene_emb_source=gene_emb_source,
                    )

        if is_main:
            save_vae_checkpoint(
                ckpt_last,
                model=raw_model,
                optimizer=optimizer,
                epoch=cfg.epochs - 1,
                genes_common=genes_common,
                config_snapshot=asdict(cfg),
                gene_emb_source=gene_emb_source,
            )

        _log(rank, "train_vae completed")
        return ckpt_last
    except Exception:
        _log(rank, "fatal exception in train_vae")
        traceback.print_exc()
        raise
    finally:
        if _is_distributed():
            _log(rank, "destroying process group")
            dist.destroy_process_group()


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
