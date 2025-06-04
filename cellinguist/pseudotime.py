import sys
sys.path.append('/home/arc85/Desktop/Cellinguist/')

import os
import argparse
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from torch.utils.data import DataLoader, DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from scipy.sparse import csr_matrix
import numpy as np
import anndata as ad
import torch.distributed as dist
from cellinguist.models.loss import knn_laplacian_loss
from cellinguist.data.data_funcs import SingleCellDatasetUnified, collate_fn_unified
from cellinguist.models.base_model import TokenEmbeddingLayer, FlashTransformerEncoderLayer, MaskedGeneExpressionPredictionHead, WholeGenomeExpressionPredictionHead, MaskedGeneIDPredictionHead, DomainClassifier, FullModel
from cellinguist.models.pseudotime_model import PseudotimeHead

def main():

    ## Parse input arguments
    parser = argparse.ArgumentParser(description="Function to extract cell embedding from a trained model.")
    parser.add_argument("--input_anndata", type=str, required=True, help="Path to anndata file.")
    parser.add_argument("--in_model", type=str, required=True, help="Path to trained model, should end in pth extension.")
    parser.add_argument("--input_pseudotime_model", type=str, required=True, help="Path to trained pseudotime head.")
    parser.add_argument("--domain_for_grad_rev", type=str, help="Optional column name of metadata containing the domain info (such as sequencing batch) for gradient reversal.")
    parser.add_argument("--condition_data", type=str, help="Optional column name of metadata containing condition info.")
    parser.add_argument("--cell_type_input", type=str, required=True, help="Input cell type for pseudotime. Should be from cell_types column in anndata.")
    parser.add_argument("--batch_size", type=int, default=64, help="Optional column name of metadata containing condition info.")
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
    parser.add_argument("--epochs", type=int, default=5, help="Number of epochs for training.")
    parser.add_argument("--grad_rev_lambda", type=float, default=1.0, help="Lambda value for gradient reversal.")
    parser.add_argument("--out_pseudotime", type=str, required=True, help="Path to save output pseudotime as a csv.")
    args = parser.parse_args()

    ## Load anndata
    dat_input = ad.read_h5ad(args.input_anndata)
    ## Subset to specific cell type 
    dat = dat_input[dat_input.obs['cell_types'] == args.cell_type_input].copy()
    del dat_input
    dense_matrix = dat.X.toarray()

    ## Set gene ids and vocab size
    gene_ids = dat.var.gene.to_numpy()
    num_of_genes = len(gene_ids)

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
    gene_vocab_size = num_of_genes + reserved_tokens_count

    if condition_labels is not None:
        CONDITION_VOCAB_SIZE = len(np.unique(condition_labels))
    else:
        CONDITION_VOCAB_SIZE = 1
        
    NUM_LIBRARY_BINS = args.num_library_bins

    if seq_batch_ids is not None:
        num_domains = len(np.unique(seq_batch_ids))
    else:
        num_domains = 1
        
    NUM_LIBRARY_BINS = args.num_library_bins

    # Initialize a device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ## Set gene ids and vocab size
    gene_ids = dat.var.gene.to_numpy()
    num_of_genes = len(gene_ids)

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
    gene_vocab_size = num_of_genes + reserved_tokens_count

    if condition_labels is not None:
        CONDITION_VOCAB_SIZE = len(np.unique(condition_labels))
    else:
        CONDITION_VOCAB_SIZE = 1
        
    NUM_LIBRARY_BINS = args.num_library_bins

    if seq_batch_ids is not None:
        num_domains = len(np.unique(seq_batch_ids))
    else:
        num_domains = 1
        
    NUM_LIBRARY_BINS = args.num_library_bins

    # Initialize a device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ## Create dataset 
    dataset = SingleCellDatasetUnified(dense_matrix, condition_labels = condition_labels, domain_labels = seq_batch_ids, num_expression_bins = num_expression_bins, num_library_bins = NUM_LIBRARY_BINS, reserved_tokens_count = 3)
    collate = partial(collate_fn_unified, cls_token_id=CLS_TOKEN_ID, pad_token_id=PAD_TOKEN_ID)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, collate_fn=collate, shuffle=False)

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

    whole_genome_head = WholeGenomeExpressionPredictionHead(
        d_model=args.prediction_domain_head_dim,
        total_gene_count=num_of_genes,
        expression_vocab_size=expression_vocab_size
    ).to(device)

    masked_gene_head = MaskedGeneIDPredictionHead(
        d_model=args.prediction_domain_head_dim,
        gene_vocab_size=gene_vocab_size
    ).to(device)
    
    ## Domain classifier
    domain_classifier = DomainClassifier(input_dim=args.prediction_domain_head_dim, hidden_dim=256, num_domains=num_domains).to(device)

    # Combine components into a full model.
    # Assume FullModel is a module that takes the token embedding layer, encoder layers, and prediction heads.
    full_model = FullModel(token_embedding_layer, flash_encoder_layers, masked_head, whole_genome_head, domain_classifier, masked_gene_head, lambda_value = args.grad_rev_lambda).to(device)
    
    # Load the full trained model checkpoint
    full_checkpoint = torch.load(args.in_model, map_location=device, weights_only=True)
    full_model.load_state_dict(full_checkpoint['model_state_dict'])
    start_epoch = full_checkpoint['epoch']
    full_model.eval()

    ## Create the pseudotime head 
    pseudotime_head = PseudotimeHead(input_dim=args.token_embedding_dim).to(device)

    ## Load to trained pseudotime head
    pseudotime_checkpoint = torch.load(args.input_pseudotime_model, map_location=device, weights_only=True)
    pseudotime_head.load_state_dict(pseudotime_checkpoint['pseudotime_state'])
    pseudotime_head.eval()

    ## Extract pseudotime
    all_times = []
    all_cell_ids = []  # optional, if you want to keep track of which row corresponds to which cell

    with torch.no_grad():
        for batch in dataloader:
        # Move all tensors to device
            for key, tensor in batch.items():
                if isinstance(tensor, torch.Tensor):
                    batch[key] = tensor.to(device, non_blocking=True)

            # Extract CLS embeddings from frozen backbone
            _, _, cls_embeddings, _, _ = full_model(batch)  
            # cls_embeddings: shape (B, d_model)

            # Compute pseudotime scalars
            t_pred = pseudotime_head(cls_embeddings)  # shape (B,)

            # Move to CPU and collect
            all_times.append(t_pred.cpu())

            # If your Dataset can give you a unique cell index (or barcode) per row:
            if "cell_id" in batch:
                all_cell_ids.append(batch["cell_id"].cpu())

    # 3c) Concatenate everything into one 1-D tensor
    all_times = torch.cat(all_times, dim=0)  # shape (N_filtered,)
    # If you collected IDs:
    # all_cell_ids = torch.cat(all_cell_ids, dim=0)  # shape (N_filtered,)

    # 3d) (Optional) If you want a NumPy array:
    times_np = all_times.numpy()

    ## Save as csv
    times_pd = pd.DataFrame(times_np)
    times_pd.to_csv(args.out_pseudotime)

if __name__ == "__main__":
    main()