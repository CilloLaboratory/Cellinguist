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

# ===== Hurdle NB(+) loss for whole-genome counts =====
import math

def _log1mexp(x: torch.Tensor) -> torch.Tensor:
    """
    Stable log(1 - exp(x)) for x <= 0.
    """
    # Split at log(0.5) for numerical stability
    log_half = -math.log(2.0)
    out = torch.empty_like(x)
    mask = x < log_half  # use log1p(-exp(x)) when far from 0
    out[mask] = torch.log1p(-torch.exp(x[mask]))
    out[~mask] = torch.log(-torch.expm1(x[~mask]))
    return out

def _nb_logpmf(y: torch.Tensor, mu: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
    """
    Negative Binomial log PMF with mean mu and dispersion theta (>0), where:
      Var(Y) = mu + mu^2 / theta
    Shapes:
      y, mu: (B, G)
      theta: (1, G) or (B, G) (broadcastable)
    """
    # Ensure floats
    y = y.to(mu.dtype)
    t = theta
    return (
        torch.lgamma(y + t) - torch.lgamma(t) - torch.lgamma(y + 1.0)
        + t * (torch.log(t) - torch.log(t + mu))
        + y * (torch.log(mu) - torch.log(t + mu))
    )

def hurdle_nb_loss(
    logit_p_nz: torch.Tensor,     # (B, G) logits for P(y>0)
    log_mu: torch.Tensor,         # (B, G) log mean (before size-factor already added)
    log_theta: torch.Tensor,      # (G,)   gene-wise log-dispersion parameters
    y: torch.Tensor,              # (B, G) integer counts
    is_nz: torch.Tensor,          # (B, G) uint8/bool indicators (1 if y>0)
    mask: torch.Tensor | None = None,  # (B, G) 1/0 for valid positions (optional)
    focal_gamma: float = 0.0,     # set >0 (e.g., 1.0) to use focal BCE on zero-vs-nonzero
    pos_weight: torch.Tensor | None = None,  # optional pos_weight for BCE, shape (G,) or scalar
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Two-part (hurdle) loss:
      L = BCE( I[y>0], p_nz ) + I[y>0] * [ -log NB(y; mu, theta) + log(1 - NB(0; mu, theta)) ]
    where NB(·) is the standard NB pmf and the second term is the truncated NB correction.

    Notes:
      - log_mu can include a size-factor offset upstream (recommended).
      - theta = exp(log_theta) is broadcast gene-wise.
      - Set focal_gamma>0 for focal-BCE (helps with heavy zero-imbalance).
    """
    # Types / shapes
    B, G = log_mu.shape
    device = log_mu.device
    dtype = log_mu.dtype

    # Probabilities and parameters
    p_nz = torch.sigmoid(logit_p_nz)                  # (B, G)
    mu   = torch.exp(log_mu).clamp_min(eps)           # (B, G)
    theta= torch.exp(log_theta).reshape(1, -1)        # (1, G) broadcast to (B, G)

    # ----- BCE (optionally focal) for zero vs nonzero -----
    target = is_nz.to(dtype)
    if focal_gamma > 0.0:
        # Focal BCE: -(1-p_t)^γ * log(p_t), with p_t = p if y=1 else (1-p)
        pt = torch.where(target > 0.5, p_nz, 1.0 - p_nz).clamp(min=eps, max=1.0 - eps)
        bce_elem = -((1.0 - pt).pow(focal_gamma)) * torch.log(pt)
        # optional class weighting
        if pos_weight is not None:
            # scale positives by pos_weight
            w = torch.ones_like(bce_elem)
            if pos_weight.numel() == 1:
                w = torch.where(target > 0.5, pos_weight.to(dtype), w)
            else:
                w = torch.where(target > 0.5, pos_weight.to(dtype).reshape(1, -1).expand_as(target), w)
            bce_elem = bce_elem * w
    else:
        # Standard BCE with optional pos_weight
        if pos_weight is not None:
            # F.binary_cross_entropy_with_logits would take logits; we have p already, so do manual BCE:
            p = p_nz.clamp(min=eps, max=1.0 - eps)
            w_pos = pos_weight.to(dtype)
            if w_pos.numel() == 1:
                w = torch.where(target > 0.5, w_pos, torch.ones_like(p))
            else:
                w = torch.where(target > 0.5, w_pos.reshape(1, -1).expand_as(target), torch.ones_like(p))
            bce_elem = -(w * target * torch.log(p) + (1.0 - target) * torch.log(1.0 - p))
        else:
            # Use logits path for best stability
            bce_elem = F.binary_cross_entropy_with_logits(
                logit_p_nz, target, reduction='none'
            )

    # ----- NB+ (truncated) for positive counts -----
    y_float = y.to(dtype)
    log_nb_y = _nb_logpmf(y_float, mu, theta.expand_as(mu))       # (B, G)
    log_nb_0 = _nb_logpmf(torch.zeros_like(mu), mu, theta.expand_as(mu))  # (B, G)
    # -log P(Y=y | Y>0) = -[log NB(y) - log(1 - NB(0))]
    # use stable log(1 - NB(0)) = log1mexp(log_nb_0)
    log_1m_nb0 = _log1mexp(log_nb_0.clamp_max(0.0))
    nb_trunc = -(log_nb_y - log_1m_nb0)

    # apply only to positive observations
    pos_mask = is_nz.bool()
    nb_term = torch.where(pos_mask, nb_trunc, torch.zeros_like(nb_trunc))

    # ----- Combine and reduce -----
    loss_elem = bce_elem + nb_term  # (B, G)

    if mask is not None:
        loss_elem = loss_elem * mask.to(dtype)
        denom = mask.to(dtype).sum().clamp_min(1.0)
    else:
        denom = torch.tensor(loss_elem.numel(), device=device, dtype=dtype).clamp_min(1.0)

    return loss_elem.sum() / denom
    