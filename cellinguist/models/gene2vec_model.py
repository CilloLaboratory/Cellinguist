import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

def freeze_all_except_gene_embeddings(model: torch.nn.Module):
    """
    Set requires_grad=False on every parameter except the raw gene_embeddings.weight.
    We assume gene_embeddings is at model.token_embedding_layer.gene_embeddings.
    """
    for name, p in model.named_parameters():
        p.requires_grad = False
    # Unfreeze only the gene_embeddings table:
    model.token_embedding_layer.gene_embeddings.weight.requires_grad = True

def gene2vec_epoch(
    dataloader: DataLoader,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    reserved_tokens_count: int = 3,
    top_k: int = 20,
    n_neg: int = 10
):
    """
    One epoch of “cell‐to‐gene2vec” training. 
    For each batch:
      1. Forward through model → whole_genome_logits and CLS embedding.
      2. For each cell, take top_k predicted gene indices from whole_genome_logits.
      3. Sample n_neg negative genes per positive.
      4. Compute skip‐gram loss between cell_emb and gene embeddings.
    """
    model.train()
    total_loss = 0.0
    num_batches = 0

    # We assume model.forward(batch) returns:
    #   masked_logits, whole_genome_logits, cls_token, domain_preds, gene_id_logits
    for batch in dataloader:
        # Move everything to device
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(device, non_blocking=True)

        optimizer.zero_grad()

        # 1) Forward to get whole_genome_logits and cell embeddings
        with torch.no_grad():
            # Freeze WGE head and encoder—though they should already be frozen
            # We need gradient only for gene_embeddings
            masked_logits, whole_genome_logits, cls_emb, domain_preds, gene_id_logits = model(batch)
        # At this point:
        #  - whole_genome_logits: (B, G) if the head returns shape (B, G, expr_vocab), 
        #      but typically WGE head will output (B, G, expr_vocab_size). 
        #      We want a “score per gene,” so sum or take max over expr bins.
        #
        # If your WGE head produces (B, G, expr_vocab_size) logits, you can reduce:
        #    gene_logits_score = whole_genome_logits.max(dim=-1).values  → (B, G)
        #
        # For simplicity, assume WGE head returned (B, G) directly. If not, adapt line below:
        gene_logits_score = whole_genome_logits.max(dim=-1).values  # shape (B, G)

        B, G = gene_logits_score.shape
        D = cls_emb.size(-1)

        # 2) For each cell in batch, pick top_k gene indices
        #    topk_vals: (B, top_k), topk_idx: (B, top_k)
        topk_vals, topk_idx = torch.topk(gene_logits_score, k=top_k, dim=1)

        # 3) Build positive‐gene list and negative samples
        #    We’ll flatten pos_u (cell idx) and pos_g (gene idx) for vectorized dot‐prod.
        cell_indices = torch.arange(B, device=device).unsqueeze(1).expand(B, top_k)  # (B, top_k)
        pos_cells = cell_indices.reshape(-1)    # (B * top_k,)
        pos_genes = topk_idx.reshape(-1)        # (B * top_k,)

        # Negative sampling: for each positive, pick n_neg negative gene IDs uniformly
        neg_genes = torch.randint(
            low=0, high=G, size=(pos_genes.numel() * n_neg,),
            device=device
        )  # (B * top_k * n_neg,)

        # 4) Look up embeddings
        #    - cell‐context vectors: cls_emb has shape (B, D). We need to repeat each cell's embedding top_k times.
        pos_cell_emb = cls_emb[pos_cells]             # (B * top_k, D)
        pos_gene_emb = model.token_embedding_layer.gene_embeddings(pos_genes + reserved_tokens_count)
        # Note: if gene IDs are shifted by RESERVED_TOKENS_COUNT in your embedding table, add that offset here.

        # Negative repeats of cell embeddings
        pos_cell_emb_rep = pos_cell_emb.unsqueeze(1).repeat(1, n_neg, 1).view(-1, D)  # (B*top_k*n_neg, D)
        neg_gene_emb = model.token_embedding_layer.gene_embeddings(neg_genes + reserved_tokens_count)  # (B*top_k*n_neg, D)

        # 5) Compute skip‐gram losses
        # Positive loss: -log(sigmoid(c · g))
        pos_score = (pos_cell_emb * pos_gene_emb).sum(dim=-1)         # (B * top_k,)
        loss_pos = -torch.log(torch.sigmoid(pos_score) + 1e-8).mean()

        # Negative loss: -log(sigmoid(-c · g_neg))
        neg_score = (pos_cell_emb_rep * neg_gene_emb).sum(dim=-1)     # (B * top_k * n_neg,)
        loss_neg = -torch.log(torch.sigmoid(-neg_score) + 1e-8).mean()

        loss_2v = loss_pos + loss_neg

        # 6) Backpropagate only into gene_embeddings
        loss_2v.backward()
        optimizer.step()

        total_loss += loss_2v.item()
        num_batches += 1

    return total_loss / num_batches if num_batches > 0 else 0.0

