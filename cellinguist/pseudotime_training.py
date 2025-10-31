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
from cellinguist.models.pseudotime_model import extract_all_cls_embeddings, compute_global_pseudotime, train_pseudotime_regressor, PseudotimeHead

def main():

    ## Parse input arguments
    parser = argparse.ArgumentParser(description="Function to extract cell embedding from a trained model.")
    parser.add_argument("--input_anndata", type=str, required=True, help="Path to anndata file.")
    parser.add_argument("--in_model", type=str, required=True, help="Path to trained model, should end in pth extension.")
    parser.add_argument("--domain_for_grad_rev", type=str, help="Optional column name of metadata containing the domain info (such as sequencing batch) for gradient reversal.")
    parser.add_argument("--condition_data", type=str, help="Optional column name of metadata containing condition info.")
    parser.add_argument("--k_neighbors", type=int, required=True, help="Number of neighbors to consider for pseudotime.")
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
    parser.add_argument("--out_model", type=str, required=True, help="Path to save output model as pth.")
    parser.add_argument("--out_pseudotime", type=str, required=True, help="Path to save output pseudotime as a csv.")
    args = parser.parse_args()

    ## Load anndata
    dat = ad.read_h5ad(args.input_anndata)
    # sample random cells 
    # my_vec = np.array(range(dat.shape[0]))
    #random_pts = np.random.choice(my_vec,size=5000,replace=F)
    # dat = dat[random_pts,]
    ## Subset to specific cell type 
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
    
    # Load the trained model checkpoint
    checkpoint = torch.load(args.in_model, map_location=device, weights_only=True)
    full_model.load_state_dict(checkpoint['model_state_dict'])
    start_epoch = checkpoint['epoch']

    print(f"Checkpoint loaded for inference. Current epoch is {start_epoch+1}")

     # 1. Extract all CLS embeddings from the pretrained backbone
    cls_emb_np = extract_all_cls_embeddings(
        pretrained_model=full_model,
        dense_matrix=dense_matrix,
        condition_labels=None,
        domain_labels=None,
        num_expression_bins=num_expression_bins,
        NUM_LIBRARY_BINS=NUM_LIBRARY_BINS,
        CLS_TOKEN_ID=CLS_TOKEN_ID,
        PAD_TOKEN_ID=PAD_TOKEN_ID,
        MASK_TOKEN_ID=MASK_TOKEN_ID,
        device=device,
        batch_size= 64
    )  # shape: (N, d_model)

    print(f"Cell embedding extracted. Moving to graph creation.")

    # 2. Compute global pseudotime via K-NN graph and Laplacian eigenmap
    pseudotime_np = compute_global_pseudotime(
        cls_embeddings=cls_emb_np,
        k_neighbors=args.k_neighbors
    )  # shape: (N,)

    print(f"Global pseudotime computed. Moving to training pseudotime regressor.")

    # 3. Train a regressor head to map CLS → pseudotime
    trained_head, optimizer, epoch = train_pseudotime_regressor(
        cls_embeddings=cls_emb_np,
        pseudotime_targets=pseudotime_np,
        d_model=512,
        device=device,
        batch_size=64,
        num_epochs=args.epochs,
        lr=1e-3
    )

    torch.save({
        "epoch": epoch,
        "pseudotime_state": trained_head.state_dict(),
        "optimizer_state": optimizer.state_dict(),
    },  args.out_model)

    # Combine with gene names 
    pseudotime_pd = pd.DataFrame(pseudotime_np)
    # Save output
    pseudotime_pd.to_csv(args.out_pseudotime)

    print(f"Model and pseudotime saved.")

if __name__ == "__main__":
    main()