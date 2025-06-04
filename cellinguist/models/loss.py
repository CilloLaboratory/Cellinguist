import torch
import torch.nn.functional as F

def compute_similarity_loss(cell_embeddings: torch.Tensor, threshold: float = 0.8) -> torch.Tensor:
    """
    Computes a loss that pulls all pairs of cell embeddings with cosine 
    similarity above `threshold` even closer together.

    Loss per pair (i,j):  L_ij = max(0, 1 - cos(c_i, c_j)),  but **only** if cos(c_i,c_j) > threshold.
    We average over those “positive” pairs. If no pair passes the threshold, we return zero.

    Args:
        cell_embeddings: (batch_size, d_model)
        threshold:       float in [-1, +1]. Only pairs with cos > threshold are included.

    Returns:
        a scalar tensor: average of (1 - cos_sim) over all (i<j) such that cos_sim > threshold.
    """
    B, D = cell_embeddings.shape
    if B < 2:
        return torch.tensor(0.0, device=cell_embeddings.device)

    # 1) Normalize so cosine = dot
    normalized = F.normalize(cell_embeddings, p=2, dim=1)  # (B, D)

    # 2) Full pairwise cosine matrix
    cos_sim = normalized @ normalized.t()                   # (B, B)

    # 3) We only care about i < j (upper triangle, no diagonal)
    idx_i, idx_j = torch.triu_indices(B, B, offset=1, device=cell_embeddings.device)
    cos_vals = cos_sim[idx_i, idx_j]                       # (B*(B-1)/2,)

    # 4) Keep only those above threshold
    positive_mask = cos_vals > threshold
    if positive_mask.sum() == 0:
        # no pairs above threshold → zero loss
        return torch.tensor(0.0, device=cell_embeddings.device)

    # 5) Define pairwise loss = (1 - cos), but only for those “positive” pairs.
    selected_cos = cos_vals[positive_mask]                   # (num_pos,)
    loss = torch.mean(1.0 - selected_cos)                    # scalar

    return loss

def mse_loss_for_expression(logits: torch.Tensor, target: torch.Tensor, ignore_index: int):
    """
    Computes an MSE loss for a prediction task where the model outputs logits over discrete bins,
    and we wish to compute the expected value of the distribution.

    Args:
        logits (torch.Tensor): Tensor of shape (B, L, EXPRESSION_VOCAB_SIZE).
        target (torch.Tensor): Tensor of shape (B, L) containing target bin indices.
        ignore_index (int): Token value to ignore (e.g. PAD_TOKEN_ID).
    
    Returns:
        torch.Tensor: Scalar MSE loss.
    """
    # Convert logits to probabilities along the last dimension.
    probs = F.softmax(logits, dim=-1)  # (B, L, EXPRESSION_VOCAB_SIZE)
    # Create a vector of bin indices. These are our "labels" in continuous form.
    # For instance, if EXPRESSION_VOCAB_SIZE is 131, then bins will be [0, 1, 2, ..., 130].
    # (If you reserved certain indices for special tokens, you may want to shift these values accordingly.)
    bins = torch.arange(0, logits.size(-1), device=logits.device).float()  # (EXPRESSION_VOCAB_SIZE,)
    # Compute the expected value: weighted sum of bins using the probabilities.
    # This yields a tensor of shape (B, L).
    expected = torch.sum(probs * bins, dim=-1)
    # Convert target to float.
    target_float = target.float()
    # Create a mask to ignore positions with ignore_index.
    valid_mask = target != ignore_index
    # Compute mean squared error only over valid positions.
    loss = F.mse_loss(expected[valid_mask], target_float[valid_mask])
    return loss

def knn_laplacian_loss(embeddings: torch.Tensor, 
                       times: torch.Tensor, 
                       k: int = 5) -> torch.Tensor:
    """
    embeddings: (B, d_model)     — cell CLS embeddings
    times:      (B,)             — predicted pseudotimes for each cell
    k:          int              — number of nearest neighbors
    
    Returns:
      Scalar: the average of (t_i - t_j)^2 over all i and its k nearest neighbors j.
    """
    B, D = embeddings.shape
    if B <= 1:
        return torch.tensor(0.0, device=embeddings.device)

    # 1) Pairwise Euclidean distances between embeddings
    dist_matrix = torch.cdist(embeddings, embeddings, p=2)  # shape (B, B)

    # 2) For each i, find the indices of the k+1 smallest distances (including self)
    #    Then ignore the first (self), keeping the next k
    knn_vals, knn_idx = torch.topk(dist_matrix, k=k+1, largest=False)  # each row: [i itself, idx1, idx2, ..., idx_k]
    neighbors = knn_idx[:, 1:]  # shape (B, k)

    # 3) Compute squared‐difference in predicted times for each neighbor pair
    t_i = times.unsqueeze(1)                   # (B, 1)
    t_j = times[neighbors]                     # (B, k)
    diff_sq = (t_i - t_j).pow(2)                # (B, k)

    # 4) Average over all (i, j) pairs
    return diff_sq.mean()