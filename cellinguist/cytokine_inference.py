import sys
sys.path.append('/home/arc85/Desktop/Cellinguist/')  # adjust if needed

import argparse, json, os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
import anndata as ad
import torch.nn as nn

# Data + model pieces (same as training/embeddings)
from cellinguist.data.data_funcs import SingleCellDatasetUnified, collate_fn_unified
from cellinguist.models.base_model import (
    TokenEmbeddingLayer, FlashTransformerEncoderLayer,
    MaskedGeneExpressionPredictionHead, WholeGenomeExpressionPredictionHead,
    DomainClassifier, FullModel, MaskedGeneIDPredictionHead
)

# --- If you placed these classes in base_model.py per our wiring:
from cellinguist.models.base_model import CytokineConditioner, PerturbationHead

# ---------------------------
# Helpers
# ---------------------------

@torch.no_grad()
def logits_to_expected(whole_genome_logits, bin_centers=None):
    """
    whole_genome_logits: (B, G, Vexp)
    bin_centers: (Vexp,) torch or numpy; if None, uses 0..Vexp-1
    returns: (B, G)
    """
    probs = torch.softmax(whole_genome_logits, dim=-1)
    Vexp = whole_genome_logits.shape[-1]
    if bin_centers is None:
        centers = torch.arange(Vexp, device=whole_genome_logits.device, dtype=whole_genome_logits.dtype)
    else:
        centers = torch.as_tensor(bin_centers, device=whole_genome_logits.device, dtype=whole_genome_logits.dtype)
        assert centers.shape[0] == Vexp, f"bin_centers length {len(centers)} != Vexp {Vexp}"
    return torch.einsum("bgv,v->bg", probs, centers)

def parse_list_arg(arg):
    """Parse comma-separated list; handles empty/None."""
    if arg is None or str(arg).strip() == "":
        return []
    return [s.strip() for s in str(arg).split(",")]

def build_global_condition(args, n):
    """Build per-batch rich condition tensors for a global condition."""
    # cytokine ids are passed as integers; doses as floats; time hours float
    cids = torch.tensor([[int(x) for x in parse_list_arg(args.cytokine_ids)]], dtype=torch.long)
    doses = torch.tensor([[float(x) for x in parse_list_arg(args.doses)]], dtype=torch.float)
    if cids.numel() == 0:
        raise ValueError("You must provide --cytokine_ids when not using --per_cell_conditions_csv")
    if doses.numel() == 0:
        doses = torch.zeros_like(cids, dtype=torch.float)
    if doses.shape != cids.shape:
        raise ValueError(f"--doses shape must match --cytokine_ids; got {doses.shape} vs {cids.shape}")
    time_hours = torch.tensor([float(args.time_hours)], dtype=torch.float)
    # Repeat to batch length n (we will slice to actual batch size later)
    return cids.repeat(n, 1), doses.repeat(n, 1), time_hours.repeat(n)

def load_bin_centers_from_model(model):
    # If your head stores centers
    for attr in ["bin_centers", "bin_values", "expression_bin_centers"]:
        if hasattr(model.whole_genome_head, attr):
            bc = getattr(model.whole_genome_head, attr)
            if isinstance(bc, torch.Tensor):
                return bc.detach().cpu().numpy()
            try:
                return np.asarray(bc)
            except Exception:
                pass
    return None

def maybe_chunk_write(df, out_csv, mode="w", header=True):
    # Simple wrapper (kept for symmetry if you later want row-wise streaming)
    df.to_csv(out_csv, mode=mode, header=header, index=False)

# ---------------------------
# Main
# ---------------------------

def main():
    parser = argparse.ArgumentParser(description="Cellinguist model for learning the language of cells.")
    parser.add_argument("--input_anndata", type=str, required=True, help="Path to anndata file.")
    parser.add_argument("--in_model", required=True, help="Path to trained model .pth")
    parser.add_argument("--domain_for_grad_rev", type=str, help="Optional column name of metadata containing the domain info (such as sequencing batch) for gradient reversal.")
    parser.add_argument("--condition_data", type=str, help="Optional column name of metadata containing condition info.")
    parser.add_argument("--batch_size", type=int, default=32, help="Optional column name of metadata containing condition info.")
    parser.add_argument("--reserved_cls_token", type=int, default=0, help="Reserved token for CLS.")
    parser.add_argument("--reserved_pad_token", type=int, default=1, help="Reserved token for padding.")
    parser.add_argument("--reserved_mask_token", type=int, default=2, help="Reserved token for masking.")
    parser.add_argument("--num_expression_bins", type=int, default=128, help="Number of bins for expression data.")
    parser.add_argument("--num_library_bins", type=int, default=20, help="Number of bins for library size normalization.")
    parser.add_argument("--max_seq_length", type=int, default=1200, help="Maximum sequence length to consider for input.")
    parser.add_argument("--token_embedding_dim", type=int, default=512, help="Dimensions for gene embedding.")
    parser.add_argument("--flash_encoder_dim", type=int, default=512, help="Dimensions for flash encoder.")
    parser.add_argument("--flash_encoder_layers", type=int, default=4, help="Number of layers for flash encoder.")
    parser.add_argument("--flash_encoder_heads", type=int, default=8, help="Number of heads for flash encoder.")
    parser.add_argument("--prediction_domain_head_dim", type=int, default=512, help="Dimensions for prediction and domain classification heads.")
    parser.add_argument("--num_iterative_steps", type=int, default=3, help="Number of iterative steps within each batch of data.")
    parser.add_argument("--grad_rev_lambda", type=float, default=1.0, help="Lambda value for gradient reversal.")
    parser.add_argument("--out_csv", required=True, help="Where to save predicted expression CSV (wide)")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    # Conditions: either global or per-cell CSV
    parser.add_argument("--cytokine_ids", type=str, default="", help="Comma-separated ints (e.g., '3,7')")
    parser.add_argument("--doses", type=str, default="", help="Comma-separated floats aligned to cytokines (e.g., '0.7,1.0' for log10 doses)")
    parser.add_argument("--time_hours", type=float, default=6.0)
    parser.add_argument("--per_cell_conditions_csv", type=str, default="", help="CSV with columns: cell_id,cytokine_ids,doses,time_hours (ids/doses as ';' separated)")
    parser.add_argument("--bin_centers_npy", type=str, default="", help="Optional .npy of expression-bin centers")
    args = parser.parse_args()

    device = torch.device(args.device)

    # Load data
    dat = ad.read_h5ad(args.input_anndata)
    dense_matrix = dat.X.toarray()
    cell_ids = dat.obs_names.to_list()
    gene_names = dat.var_names.to_list()
    n_cells, n_genes = dat.shape

    ## Set domains for normalization (optional)
    if args.domain_for_grad_rev is not None:
        seq_batch_ids = torch.tensor(dat.obs[args.domain_for_grad_rev].factorize()[0])
    else: 
        seq_batch_ids = None

    ## Set conditions (optional)
    if args.condition_data is not None:
        condition_labels = torch.tensor(dat.obs[args.condition_data].factorize()[0])
    else:
        condition_labels = None

    ## Set other input parameters
    CLS_TOKEN_ID = args.reserved_cls_token
    PAD_TOKEN_ID   = args.reserved_pad_token
    MASK_TOKEN_ID  = args.reserved_mask_token
    reserved_tokens_count = 3

    num_expression_bins = args.num_expression_bins
    expression_vocab_size = num_expression_bins + reserved_tokens_count
    gene_vocab_size = n_genes + reserved_tokens_count

    # if condition_labels is not None:
    #     CONDITION_VOCAB_SIZE = len(np.unique(condition_labels))
    # else:
    #     CONDITION_VOCAB_SIZE = 1
    # Must match training data condition vocab size
    CONDITION_VOCAB_SIZE=2

    n_cytokines = CONDITION_VOCAB_SIZE  # for v1, reuse your factorized condition labels

    NUM_LIBRARY_BINS = args.num_library_bins

    if seq_batch_ids is not None:
        num_domains = len(np.unique(seq_batch_ids))
    else:
        num_domains = 1
        
    NUM_LIBRARY_BINS = args.num_library_bins

    # Dataset (baseline tokens/targets; whole_genome_target is ignored at inference)
    dataset = SingleCellDatasetUnified(expression_matrix = dense_matrix, 
        condition_labels = np.zeros(n_cells, dtype=int),
        domain_labels = np.zeros(n_cells, dtype=int),
        num_expression_bins = num_expression_bins,
        num_library_bins = NUM_LIBRARY_BINS,
        reserved_tokens_count = 3
    )
    collate = lambda samples: collate_fn_unified(samples, cls_token_id=0, pad_token_id=0)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate)

    ## Token embedding 
    token_embedding_layer = TokenEmbeddingLayer(
        gene_vocab_size=gene_vocab_size,
        expression_vocab_size=expression_vocab_size,
        condition_vocab_size=CONDITION_VOCAB_SIZE,
        library_vocab_size=NUM_LIBRARY_BINS,
        embedding_dim=args.token_embedding_dim,
        max_seq_len=args.max_seq_length,
        use_positional=False,
        pad_token_id=PAD_TOKEN_ID
    ).to(device)

    ## Flash transformer
    flash_encoder_layers = nn.ModuleList([
        FlashTransformerEncoderLayer(d_model=args.flash_encoder_dim, nhead=args.flash_encoder_heads, dropout=0.1, causal=False)
        for _ in range(args.flash_encoder_layers)
    ])
    flash_encoder_layers = flash_encoder_layers.to(device)

    ## Prediction heads
    masked_head = MaskedGeneExpressionPredictionHead(
        d_model=args.prediction_domain_head_dim,
        expression_vocab_size=expression_vocab_size
    ).to(device)

    masked_gene_head = MaskedGeneIDPredictionHead(
        d_model=args.prediction_domain_head_dim,
        gene_vocab_size=gene_vocab_size
    ).to(device)

    whole_genome_head = WholeGenomeExpressionPredictionHead(
        d_model=args.prediction_domain_head_dim,
        total_gene_count=n_genes,
        expression_vocab_size=expression_vocab_size
    ).to(device)
    
    ## Domain classifier
    domain_classifier = DomainClassifier(input_dim=args.prediction_domain_head_dim, hidden_dim=256, num_domains=num_domains).to(device)

    # NEW: conditioner + perturbation head
    n_cytokines = CONDITION_VOCAB_SIZE  # for v1, reuse your factorized condition labels
    conditioner = CytokineConditioner(n_cytokines=n_cytokines,
                                      d_model=args.prediction_domain_head_dim,
                                      receptor_dim=0).to(device)
    perturb_head = PerturbationHead(d_model=args.prediction_domain_head_dim,
                                    hidden=512,
                                    use_film=True).to(device)

    model = FullModel(token_embedding_layer, flash_encoder_layers, masked_head, whole_genome_head,
                      domain_classifier, masked_gene_head, lambda_value=0.0,
                      conditioner=conditioner, perturb_head=perturb_head).to(device)

    # Load checkpoint (robust to different save styles)
    ckpt = torch.load(args.in_model, map_location=device)
    state_dict = ckpt.get("state_dict", None)
    if state_dict is None and isinstance(ckpt, dict) and all(k.startswith("token_embedding_layer") or k.startswith("encoder_layers") for k in ckpt.keys()):
        state_dict = ckpt
    if state_dict is None:
        # Might have been saved via torch.save(model.state_dict())
        try:
            model.load_state_dict(ckpt, strict=False)
        except Exception as e:
            raise RuntimeError(f"Could not load checkpoint: {e}")
    else:
        # Remove any 'module.' prefixes from DDP
        state_dict = {k.replace("module.", "", 1): v for k, v in state_dict.items()}
        model.load_state_dict(state_dict, strict=False)
    model.eval()

    # Bin centers (discover -> fallback)
    bin_centers = None
    if args.bin_centers_npy and os.path.exists(args.bin_centers_npy):
        bin_centers = np.load(args.bin_centers_npy)
    if bin_centers is None:
        bin_centers = load_bin_centers_from_model(model)

    # Per-cell condition CSV?
    per_cell_df = None
    if args.per_cell_conditions_csv:
        per_cell_df = pd.read_csv(args.per_cell_conditions_csv)
        if "cell_id" not in per_cell_df.columns:
            raise ValueError("per_cell_conditions_csv must include a 'cell_id' column")
        per_cell_df = per_cell_df.set_index("cell_id")

    # Prepare global condition tensors (oversized; trimmed per batch)
    global_cids, global_doses, global_t = None, None, None
    if per_cell_df is None:
        # Build from CLI
        global_cids, global_doses, global_t = build_global_condition(args, n=args.batch_size)

    # Iterate + predict
    all_chunks = []
    start = 0
    wrote_header = False
    for batch in dataloader:
        B = batch["expression_tokens"].shape[0]

        # Move base tensors to device
        for k in ["gene_ids", "expression_tokens", "whole_genome_target", "library_size", "condition", "domain"]:
            batch[k] = batch[k].to(device)

        # Supply rich condition fields
        if per_cell_df is not None:
            # Build per-item lists aligned to obs_names[start:start+B]
            cid_tensors, dose_tensors, t_list = [], [], []
            cell_batch_ids = cell_ids[start:start+B]
            for cid in cell_batch_ids:
                row = per_cell_df.loc[cid]
                # Expect strings like "3;7" and "0.7;1.0"
                ids = [int(x) for x in str(row["cytokine_ids"]).split(";")]
                ds = [float(x) for x in str(row["doses"]).split(";")]
                if len(ids) != len(ds):
                    raise ValueError(f"Row for {cid} has mismatched cytokine_ids and doses")
                cid_tensors.append(torch.tensor(ids, dtype=torch.long))
                dose_tensors.append(torch.tensor(ds, dtype=torch.float))
                t_list.append(float(row["time_hours"]))
            # Pad to (B, Cmax)
            c_padded = torch.nn.utils.rnn.pad_sequence(cid_tensors, batch_first=True, padding_value=0)
            d_padded = torch.nn.utils.rnn.pad_sequence(dose_tensors, batch_first=True, padding_value=0.0)
            t_tensor  = torch.tensor(t_list, dtype=torch.float)
            batch["cytokine_ids"] = c_padded.to(device)
            batch["doses"]        = d_padded.to(device)
            batch["time_hours"]   = t_tensor.to(device)
        else:
            # Use global condition for whole batch (trim to B)
            batch["cytokine_ids"] = global_cids[:B].to(device)
            batch["doses"]        = global_doses[:B].to(device)
            batch["time_hours"]   = global_t[:B].to(device)

        # Forward
        _, whole_logits, _, _, _ = model(batch)
        x_pred = logits_to_expected(whole_logits, bin_centers=bin_centers)  # (B, G)

        # Build a DataFrame chunk (wide): cell_id + genes
        df = pd.DataFrame(x_pred.cpu().numpy(), columns=gene_names)
        df.insert(0, "cell_id", cell_ids[start:start+B])
        maybe_chunk_write(df, args.out_csv, mode="w" if not wrote_header else "a", header=not wrote_header)
        wrote_header = True

        start += B

    print(f"Done. Wrote predictions to {args.out_csv}")

if __name__ == "__main__":
    main()
