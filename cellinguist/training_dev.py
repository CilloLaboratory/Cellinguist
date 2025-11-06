import sys
sys.path.append('/home/arc85/Desktop/Cellinguist/')

import os
import argparse
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from torch.utils.data import DataLoader, DistributedSampler, SubsetRandomSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from scipy.sparse import csr_matrix
import numpy as np
import anndata as ad
import torch.distributed as dist
from cellinguist.models.loss import mse_loss_for_expression, compute_similarity_loss
from cellinguist.data.data_funcs import SingleCellDatasetUnified, collate_fn_unified
from cellinguist.models.base_model import TokenEmbeddingLayer, FlashTransformerEncoderLayer, MaskedGeneExpressionPredictionHead, MaskedGeneIDPredictionHead, WholeGenomeExpressionPredictionHead, DomainClassifier, FullModel, get_random_mask_positions, train_epoch_ddp, CytokineConditioner, PerturbationHead

## Load anndata
dat = ad.read_h5ad('/home/arc85/Desktop/Cellinguist/cytokine_dictionary_pbs_ifng_2k_251103.h5ad')
dense_matrix = dat.X.toarray()

    ## Set gene ids and vocab size
    # gene_ids = dat.var.gene.to_numpy()
    # gene is in var in anndata - sometimes called features 
gene_ids = dat.var.gene.to_numpy()
num_of_genes = len(gene_ids)

    ## Set domains for normalization (optional)
seq_batch_ids = None

    ## Set conditions (optional)
condition_labels = torch.tensor(dat.obs['cytokine_id'].factorize()[0])

num_cytokines = len(set(dat.obs['cytokine_id'].factorize()[0]))

    ## Set other input parameters
CLS_TOKEN_ID = 0
PAD_TOKEN_ID   = 1
MASK_TOKEN_ID  = 2
CYTOKINE_PAD_ID = num_cytokines
reserved_tokens_count = 3

num_expression_bins = 128
expression_vocab_size = num_expression_bins + reserved_tokens_count
gene_vocab_size = num_of_genes + reserved_tokens_count

CONDITION_VOCAB_SIZE = len(np.unique(condition_labels))

NUM_LIBRARY_BINS = 20

num_domains = 1

    ## Create dataset 
dataset = SingleCellDatasetUnified(dense_matrix, condition_labels = condition_labels, domain_labels = seq_batch_ids, num_expression_bins = num_expression_bins, num_library_bins = NUM_LIBRARY_BINS, reserved_tokens_count = 3)
collate = partial(collate_fn_unified, cls_token_id=CLS_TOKEN_ID, pad_token_id=PAD_TOKEN_ID,cytokine_pad_token_id=CYTOKINE_PAD_ID)
dataloader = DataLoader(dataset, batch_size=32, collate_fn=collate)

device = 'cuda'

    ## Token embedding 
token_embedding_layer = TokenEmbeddingLayer(
    gene_vocab_size=gene_vocab_size,
    expression_vocab_size=expression_vocab_size,
    condition_vocab_size=CONDITION_VOCAB_SIZE,
    library_vocab_size=NUM_LIBRARY_BINS,
    embedding_dim=512,
    max_seq_len=1200,
    use_positional=False,
    pad_token_id=PAD_TOKEN_ID
).to(device)

loaded_tensor = torch.load('/home/arc85/Desktop/Cellinguist/cytokine_dict_warm_start_pbs_ifng_gene_embeddings_251103.pth',weights_only=True)

with torch.no_grad():
    token_embedding_layer.gene_embeddings.weight.copy_(loaded_tensor.to(token_embedding_layer.gene_embeddings.weight.device))

## Flash transformer
flash_encoder_layers = nn.ModuleList([
    FlashTransformerEncoderLayer(d_model=512, nhead=8, dropout=0.1, causal=False)
    for _ in range(4)
])
flash_encoder_layers = flash_encoder_layers.to(device)

    ## Prediction heads
masked_head = MaskedGeneExpressionPredictionHead(
    d_model=512,
    expression_vocab_size=expression_vocab_size
).to(device)

masked_gene_head = MaskedGeneIDPredictionHead(
    d_model=512,
    gene_vocab_size=gene_vocab_size
).to(device)

whole_genome_head = WholeGenomeExpressionPredictionHead(
    d_model=512,
    total_gene_count=num_of_genes,
    expression_vocab_size=expression_vocab_size
).to(device)
    
## Domain classifier
domain_classifier = DomainClassifier(input_dim=512, hidden_dim=512, num_domains=num_domains).to(device)

# --- Cytokine conditioner & perturbation head (new) ---
n_cytokines = CONDITION_VOCAB_SIZE  # for v1, reuse your factorized condition labels
conditioner = CytokineConditioner(n_cytokines=n_cytokines,
                                      d_model=512,
                                      receptor_dim=0,
                                      padding_idx=CYTOKINE_PAD_ID).to(device)
perturb_head = PerturbationHead(d_model=512,
                                    hidden=512,
                                    use_film=True).to(device)

    # Combine components into a full model.
    # Assume FullModel is a module that takes the token embedding layer, encoder layers, and prediction heads.
full_model = FullModel(token_embedding_layer,
                        flash_encoder_layers,
                        masked_head,
                        whole_genome_head,
                        domain_classifier,
                        masked_gene_head,
                        lambda_value = 1,
                        conditioner=conditioner,
                        perturb_head=perturb_head).to(device)

optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, full_model.parameters()),
        lr = 1e-4
)

    # Optionally, if you use teacher forcing, define the number of iterations.
num_iterative_steps = 3

    # Main training loop.
NUM_EPOCHS = 1


## To run interactively 
import math, traceback
from contextlib import nullcontext
import torch
import torch.nn.functional as F

def get_random_mask_positions(x_tokens: torch.Tensor, mask_token_id: int, mask_fraction: float) -> torch.Tensor:
    """
    x_tokens: (B, L) integer tokens. Assumes PAD tokens are == pad_token_id (handled outside if needed).
    Returns a boolean mask of same shape selecting ~mask_fraction of non-CLS/PAD positions.
    """
    B, L = x_tokens.shape
    # avoid masking CLS at position 0
    valid = torch.ones_like(x_tokens, dtype=torch.bool)
    valid[:, 0] = False
    # don’t mask PADs if present
    # (callers can also AND with x_tokens != pad_id if they want)
    prob = torch.rand((B, L), device=x_tokens.device)
    sel = (prob < mask_fraction) & valid
    return sel

def to_device_inplace(batch: dict, device: torch.device):
    tensor_keys = []
    for k, v in list(batch.items()):
        if torch.is_tensor(v):
            batch[k] = v.to(device, non_blocking=True)
            tensor_keys.append(k)
        elif isinstance(v, (list, tuple)):
            # try tensor-ify cytokine lists (ids/doses) if they look numeric
            if k in ("cytokine_ids", "doses") and len(v) > 0 and not torch.is_tensor(v):
                try:
                    t = torch.as_tensor(v)
                    batch[k] = t.to(device)
                    tensor_keys.append(k)
                except Exception:
                    pass
    return tensor_keys  # handy if you want to print

def debug_print_batch(batch, keys=("gene_ids","expression_tokens","whole_genome_target",
                                   "cytokine_ids","doses","time_hours","receptor_vec","domain","condition")):
    for k in keys:
        if k in batch:
            v = batch[k]
            if torch.is_tensor(v):
                print(f"[batch] {k}: shape={tuple(v.shape)} dtype={v.dtype} device={v.device}")
            else:
                print(f"[batch] {k}: type={type(v)} example={str(v)[:80]}")

@torch.no_grad()
def _print_conditioner_inputs(batch, model):
    # If you named it CytokineConditioner in model
    cond = getattr(model, "conditioner", None)
    if cond is None:
        return
    print("\n[cond] inspecting conditioner inputs:")
    for k in ("cytokine_ids","doses","time_hours","receptor_vec"):
        if k in batch and torch.is_tensor(batch[k]):
            t = batch[k]
            print(f"  {k}: shape={tuple(t.shape)} dtype={t.dtype} device={t.device}")
    # also print first Linear of dose_mlp if present
    try:
        lin = cond.dose_mlp[0]
        print(f"  dose_mlp in_features={lin.in_features} weight.dtype={lin.weight.dtype} device={lin.weight.device}")
    except Exception:
        pass
    print("")

def run_one_batch_debug(
    ddp_model,
    dataloader,
    optimizer,
    device,
    *,
    mask_token_id: int,
    pad_token_id: int,
    mask_fraction: float = 0.15,
    num_iterative_steps: int = 1,
    amp_enabled: bool = False,  # force False for debugging
    loss_weights = None,
):
    loss_weights = loss_weights or dict(masked=0.005, genome=0.05, presence=1.0, posbins=1.0,
                                        similarity=2.0, domain=5.0, gene_id=5.0)

    # fetch one batch
    it = iter(dataloader)
    batch = next(it)

    # move tensors to device
    to_device_inplace(batch, device)

    # quick visibility
    print("=== One-batch debug ===")
    debug_print_batch(batch)

    # clone originals for teacher forcing
    current_expression_tokens = batch['expression_tokens'].clone()
    current_gene_id_tokens    = batch["gene_ids"].clone()

    ddp_model.train()
    optimizer.zero_grad(set_to_none=True)

    # build mask once per outer step
    mask_positions = get_random_mask_positions(current_expression_tokens, mask_token_id, mask_fraction)
    # don’t mask PADs
    mask_positions &= (current_expression_tokens != pad_token_id)

    # masked inputs
    masked_input = dict(batch)  # shallow copy OK (we replace only 2 tensors)
    input_expression_tokens = current_expression_tokens.clone()
    input_gene_id_tokens    = current_gene_id_tokens.clone()

    masked_input['expression_tokens'] = current_expression_tokens.clone()
    masked_input['gene_ids']          = current_gene_id_tokens.clone()
    masked_input['expression_tokens'][mask_positions] = mask_token_id
    masked_input['gene_ids'][mask_positions]          = mask_token_id

    # helpful extra prints for conditioner
    _print_conditioner_inputs(masked_input, ddp_model.module if hasattr(ddp_model, "module") else ddp_model)

    # choose context (disable AMP for debugging)
    ctx = nullcontext()
    if amp_enabled:
        ctx = torch.cuda.amp.autocast(dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32)

    try:
        with ctx:
            outs = ddp_model(masked_input)

        # Support both forward signatures:
        if isinstance(outs, (list, tuple)) and len(outs) == 5:
            masked_logits, whole_genome_logits, cls_token, domain_preds, gene_id_logits = outs
            p_expr = pos_logits = None
        elif isinstance(outs, (list, tuple)) and len(outs) == 6:
            masked_logits, p_expr, pos_logits, cls_token, domain_preds, gene_id_logits = outs
            whole_genome_logits = None
        else:
            raise RuntimeError(f"Unexpected forward() return of length {len(outs)}")

        # print shapes
        def _sh(x, name):
            if x is None: print(f"[forward] {name}: None")
            elif torch.is_tensor(x): print(f"[forward] {name}: {tuple(x.shape)} {x.dtype} {x.device}")
            else: print(f"[forward] {name}: type={type(x)}")
        _sh(masked_logits, "masked_logits")
        _sh(whole_genome_logits, "whole_genome_logits")
        _sh(p_expr, "p_expr")
        _sh(pos_logits, "pos_logits")
        _sh(cls_token, "cls_token")
        _sh(domain_preds, "domain_preds")
        _sh(gene_id_logits, "gene_id_logits")

        # compute losses that are available
        loss_masked = torch.tensor(0.0, device=device)
        loss_gene_id = torch.tensor(0.0, device=device)
        loss_genome = torch.tensor(0.0, device=device)
        L_presence = torch.tensor(0.0, device=device)
        L_posbins  = torch.tensor(0.0, device=device)

        if mask_positions.any():
            # MSE over masked positions for expression tokens (or adapt to your loss)
            from cellinguist.models.loss import mse_loss_for_expression  # adjust import path if needed
            loss_masked = mse_loss_for_expression(
                masked_logits[mask_positions], input_expression_tokens[mask_positions], ignore_index=pad_token_id
            )
            loss_gene_id = F.cross_entropy(
                gene_id_logits[mask_positions], input_gene_id_tokens[mask_positions], ignore_index=pad_token_id
            )

        # Whole-genome (old path): MSE to binned targets
        if whole_genome_logits is not None and "whole_genome_target" in batch:
            from cellinguist.models.loss import mse_loss_for_expression
            loss_genome = mse_loss_for_expression(whole_genome_logits, batch["whole_genome_target"], ignore_index=pad_token_id)

        # New presence/posbins heads (if available and targets present)
        if (p_expr is not None) and ("wge_presence_target" in batch):
            from cellinguist.models.loss import loss_presence_bce
            L_presence = loss_presence_bce(p_expr, batch["wge_presence_target"], pos_weight=2.0)
        if (pos_logits is not None) and ("wge_posbins_target" in batch):
            from cellinguist.models.loss import loss_posbins_ce
            L_posbins = loss_posbins_ce(pos_logits, batch["wge_posbins_target"], ignore_index=-100)

        # similarity/domain
        from cellinguist.models.loss import compute_similarity_loss
        loss_similarity = compute_similarity_loss(cls_token, threshold=0.9) if torch.is_tensor(cls_token) else torch.tensor(0.0, device=device)
        loss_domain = F.cross_entropy(domain_preds, batch['domain']) if torch.is_tensor(domain_preds) else torch.tensor(0.0, device=device)

        # total (mix old/new heads if both active)
        loss = (loss_weights["masked"] * loss_masked +
                loss_weights["genome"] * loss_genome +
                loss_weights["presence"] * L_presence +
                loss_weights["posbins"] * L_posbins +
                loss_weights["similarity"] * loss_similarity +
                loss_weights["domain"] * loss_domain +
                loss_weights["gene_id"] * loss_gene_id)

        print(f"\n[losses] masked={loss_masked.item():.4f} genome={loss_genome.item():.4f} "
              f"presence={L_presence.item():.4f} posbins={L_posbins.item():.4f} "
              f"similarity={loss_similarity.item():.4f} domain={loss_domain.item():.4f} "
              f"gene_id={loss_gene_id.item():.4f} | total={loss.item():.4f}")

        # backward just to verify graph ok (can comment out while troubleshooting)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        # (optional) teacher forcing update for masked positions
        with torch.no_grad():
            probs = F.softmax(masked_logits, dim=-1)
            conf, _ = probs.max(dim=-1)
            thresh = torch.quantile(conf[mask_positions], 0.8) if mask_positions.any() else 1.0
            update_mask = mask_positions & (conf >= thresh)
            pred_tokens = masked_logits.argmax(dim=-1)
            current_expression_tokens[update_mask] = pred_tokens[update_mask]
            current_expression_tokens = current_expression_tokens.detach()

        return loss.item()

    except Exception as e:
        print("\n[EXCEPTION] during one-batch debug:")
        traceback.print_exc()
        # dump a couple of common culprits
        _print_conditioner_inputs(masked_input, ddp_model.module if hasattr(ddp_model, "module") else ddp_model)
        print("[hint] Try --amp_dtype none, and verify doses/time dtype==float32, ids dtype==long, "
              "and that dose_mlp first linear in_features matches doses last-dim (should be 1).")
        raise

ddp_or_model = full_model # or plain model if not using DDP
loss_val = run_one_batch_debug(
    ddp_or_model,
    dataloader,
    optimizer,
    device,
    mask_token_id=MASK_TOKEN_ID,
    pad_token_id=PAD_TOKEN_ID,
    mask_fraction=0.15,
    num_iterative_steps=1,
    amp_enabled=False,  # keep off while debugging
)
print("one-batch total loss:", loss_val)