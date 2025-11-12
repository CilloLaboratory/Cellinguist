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

def mse_loss_for_expression(
    logits: torch.Tensor,
    target: torch.Tensor,
    *,
    ignore_index: int,
    reserved_tokens: int = 3,          # CLS, PAD, MASK count
    bin_values: torch.Tensor | None = None,
    temperature: float = 1.0           # optional softmax temp
) -> torch.Tensor:
    """
    Expected-value MSE over *valid expression bins only* (excludes reserved tokens),
    with bin values mapped to a continuous target space.

    logits: [..., V]
    target: [...] (same leading dims), contains bin ids including reserved tokens
    """
    V = logits.size(-1)
    E = V - reserved_tokens            # number of real expression bins

    # 1) Exclude reserved tokens from the distribution
    logits = logits.clone()
    logits[..., :reserved_tokens] = -1e9

    # 2) Softmax (optionally with temperature)
    probs = F.softmax(logits / temperature, dim=-1)          # [..., V]
    probs_expr = probs[..., reserved_tokens:]                # [..., E]

    # 3) Define numeric bin values (centers), not raw 0..E-1 indices if you have them
    if bin_values is None:
        # simplest: centers at 0..E-1 (float)
        bin_values = torch.arange(E, device=logits.device, dtype=logits.dtype)
    else:
        bin_values = bin_values.to(device=logits.device, dtype=logits.dtype)  # [..., E] or (E,)

    # 4) Expected value over *expression* bins
    expected = (probs_expr * bin_values).sum(dim=-1)         # [...]

    # 5) Targets: shift away reserved tokens and mask invalids
    target_adj = (target - reserved_tokens).to(expected.dtype)     # [...]
    valid_mask = (
        (target != ignore_index) &
        (target_adj >= 0) &
        (target_adj < E)
    )

    if valid_mask.any():
        return F.mse_loss(expected[valid_mask], target_adj[valid_mask], reduction="mean")
    else:
        # return a zero that still participates in the graph
        return expected.sum() * 0.0

import torch
import torch.nn.functional as F

def ce_loss_for_expression(
    logits: torch.Tensor,
    target: torch.Tensor,
    *,
    reserved_tokens: int = 3,          # e.g., [CLS]=0, [PAD]=1, [MASK]=2
    ignore_indices = (0, 1, 2),        # positions to ignore in target
) -> torch.Tensor:
    """
    Cross-entropy over expression bins only (excludes reserved tokens).
    logits: [..., V]
    target: [...] (same leading dims), int64 bin ids including reserved tokens
    """
    V = logits.size(-1)
    E = V - reserved_tokens
    assert target.dtype in (torch.int32, torch.int64), "target must be integer bins"

    # 1) Slice away reserved tokens from logits
    logits_expr = logits[..., reserved_tokens:]                 # [..., E]

    # 2) Shift targets to the expression-bin index space [0..E-1]
    target_adj = target - reserved_tokens                       # [...]

    # 3) Build a boolean mask of positions to ignore
    dev = target.device
    if isinstance(ignore_indices, (list, tuple, set)):
        ignore_tensor = torch.tensor(list(ignore_indices), device=dev, dtype=target.dtype)
        ignore_mask = torch.isin(target, ignore_tensor)         # True where target is a reserved/ignored token
    else:
        # single int
        ignore_mask = (target == ignore_indices)

    # Also ignore anything outside [0, E-1] after shifting
    invalid_range = (target_adj < 0) | (target_adj >= E)

    # 4) Make a CE-friendly target, with ignored spots = -100
    ce_target = target_adj.clone()
    ce_target[ignore_mask | invalid_range] = -100               # CE ignore_index

    # 5) Compute CE (flatten is fine)
    loss = F.cross_entropy(
        logits_expr.reshape(-1, E),
        ce_target.reshape(-1),
        ignore_index=-100,
        reduction="mean",
    )
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
    return_components : bool = False
) -> torch.Tensor | tuple:
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
    # ----- Params with guards -----
    # Keep p_nz as logits for BCE stability; we'll use logits below.
    p_nz = torch.sigmoid(logit_p_nz)

    # Clamp log_mu to avoid exp under/overflow; 11 => exp(11) ~ 6e4 counts
    log_mu = log_mu.clamp(min=-20.0, max=11.0)
    mu = torch.exp(log_mu)  # (B, G)

    # Use softplus to ensure theta > 0 without exploding
    theta = F.softplus(log_theta, beta=1.0) + 1e-4     # (G,)
    theta = theta.reshape(1, -1)                       # (1, G)

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
    th = theta.expand_as(mu)

    log_nb_y = _nb_logpmf(y_float, mu, th)                 # (B, G)
    log_nb_0 = _nb_logpmf(torch.zeros_like(mu), mu, th)    # (B, G)

    # Clamp log NB(0) strictly below 0 so log(1 - NB(0)) stays finite
    tiny = 1e-8
    max_log_nb0 = math.log(1.0 - tiny)  # ~ -1e-8
    log_nb_0 = torch.minimum(
        log_nb_0,
        torch.tensor(max_log_nb0, device=log_nb_0.device, dtype=log_nb_0.dtype)
    )

    # Stable log(1 - NB(0)) using helper
    log_1m_nb0 = _log1mexp(log_nb_0)                       # (B, G)

    # Truncated NB negative log-likelihood, applied only when y>0
    nb_trunc = -(log_nb_y - log_1m_nb0)

    # apply only to positive observations
    pos_mask = is_nz.bool()
    nb_term = torch.where(pos_mask, nb_trunc, torch.zeros_like(nb_trunc))

    # ----- Combine and reduce -----
    loss_elem = bce_elem + nb_term  # (B, G)

    if mask is not None:
        w = mask.to(dtype)
        loss      = (loss_elem * w).sum() / w.sum().clamp_min(1.0)
        loss_bce  = (bce_elem * w).sum() / w.sum().clamp_min(1.0)
        loss_nb   = (nb_term * w).sum() / w.sum().clamp_min(1.0)
    else:
        denom = torch.tensor(loss_elem.numel(), device=loss_elem.device, dtype=loss_elem.dtype).clamp_min(1.0)
        loss      = loss_elem.sum() / denom
        loss_bce  = bce_elem.sum()  / denom
        loss_nb   = nb_term.sum()   / denom

    if torch.isnan(bce_elem).any() or torch.isinf(bce_elem).any() \
        or torch.isnan(nb_term).any() or torch.isinf(nb_term).any():
            raise FloatingPointError("NaN/Inf detected in hurdle loss components")

    if return_components:
        return loss, loss_bce, loss_nb
    return loss