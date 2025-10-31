import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix, diags
from scipy.sparse.linalg import eigsh
from cellinguist.data.data_funcs import SingleCellDatasetUnified, collate_fn_unified
from torch.utils.data import DataLoader
from functools import partial
from scipy.sparse import diags

def extract_all_cls_embeddings(pretrained_model: nn.Module,
                               dense_matrix,
                               condition_labels,
                               domain_labels,
                               num_expression_bins,
                               NUM_LIBRARY_BINS,
                               CLS_TOKEN_ID,
                               PAD_TOKEN_ID,
                               MASK_TOKEN_ID,
                               device: torch.device,
                               batch_size: int = 64) -> np.ndarray:
    """
    Runs every sample in `dataset` through `pretrained_model` to collect
    the [CLS] embeddings. Returns a NumPy array of shape (N, d_model),
    where N = len(dataset).

    Args:
        pretrained_model: Frozen FullModel that outputs (masked_logits, whole_logits, cls_token, domain_preds).
        dataset:          Dataset yielding dict batches; reuses same collate as training.
        device:           torch.device
        batch_size:       batch size for DataLoader.

    Returns:
        cls_embeddings_np: NumPy array of all CLS embeddings (N, d_model).
    """
    pretrained_model.eval()
    dataset = SingleCellDatasetUnified(dense_matrix, condition_labels = condition_labels, domain_labels = domain_labels, num_expression_bins = num_expression_bins, num_library_bins = NUM_LIBRARY_BINS, reserved_tokens_count = 3)
    collate = partial(collate_fn_unified, cls_token_id=CLS_TOKEN_ID, pad_token_id=PAD_TOKEN_ID)
    dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate, shuffle=False)

    all_embeddings = []
    with torch.no_grad():
        for batch in dataloader:
            # Move to device
            for key, tensor in batch.items():
                if isinstance(tensor, torch.Tensor):
                    batch[key] = tensor.to(device, non_blocking=True)

            # Forward to get CLS embedding
            _, _, cls_emb, _, _ = pretrained_model(batch)  # shape: (B, d_model)
            all_embeddings.append(cls_emb.cpu().numpy())

    cls_embeddings_np = np.vstack(all_embeddings)  # (N, d_model)
    return cls_embeddings_np


def compute_global_pseudotime(cls_embeddings: np.ndarray,
                              k_neighbors: int = 10,
                              sigma: float = None) -> np.ndarray:
    """
    Given all CLS embeddings (N, d_model), build a k-NN graph,
    form a weight matrix W, compute graph Laplacian L = D - W, then
    find the second smallest eigenvector of L to serve as pseudotime.

    Args:
        cls_embeddings: (N, d_model) NumPy array.
        k_neighbors:    int number of nearest neighbors for graph.
        sigma:          bandwidth for Gaussian kernel; if None, use mean distance.

    Returns:
        pseudotime: NumPy array of shape (N,) containing the eigenvector
                    corresponding to the second smallest eigenvalue of L.
                    Values may be arbitrary up to sign; can be normalized.
    """
    # 2a. Fit KNN on all embeddings
    nbrs = NearestNeighbors(n_neighbors=k_neighbors + 1, algorithm='auto', metric='euclidean').fit(cls_embeddings)
    distances, indices = nbrs.kneighbors(cls_embeddings)  # both shape (N, k_neighbors+1)

    # distances[:, 0] is zero (each point to itself); skip it
    distances = distances[:, 1:]  # (N, k_neighbors)
    indices = indices[:, 1:]      # (N, k_neighbors)

    N = cls_embeddings.shape[0]

    # 2b. Choose sigma if not provided
    print(f"Mean distance is {np.mean(distances)}")
    sigma = np.mean(distances)*2
    if sigma is None:
        sigma = np.mean(distances)

    # 2c. Build sparse weight matrix W (N x N)
    # For each i, j in indices[i], weight = exp(-dist^2 / sigma^2)
    row_idx = np.repeat(np.arange(N), k_neighbors)
    col_idx = indices.reshape(-1)
    dist_flat = distances.reshape(-1)
    weights = np.exp(- (dist_flat ** 2) / (sigma ** 2))

    # Build symmetric adjacency: add both (i->j) and (j->i)
    all_row = np.concatenate([row_idx, col_idx])
    all_col = np.concatenate([col_idx, row_idx])
    all_weights = np.concatenate([weights, weights])

    W = csr_matrix((all_weights, (all_row, all_col)), shape=(N, N))

    ## Create normalized laplacian
    deg = np.array(W.sum(axis=1)).flatten()
    inv_sqrt = 1.0 / np.sqrt(deg + 1e-12)
    D_inv_sqrt = diags(inv_sqrt)
    L_norm = diags(np.ones(N)) - D_inv_sqrt @ W @ D_inv_sqrt
    eigenvals, eigenvecs = eigsh(L_norm, k=2, which='SM', tol=1e-3)
    fiedler_vec = eigenvecs[:,1]

    # 2g. Normalize Fiedler vector to 0..1
    min_val = fiedler_vec.min()
    max_val = fiedler_vec.max()
    pseudotime = (fiedler_vec - min_val) / (max_val - min_val)

    return pseudotime

class PseudotimeHead(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.fc = nn.Linear(input_dim, 1)

    def forward(self, cls_emb: torch.Tensor) -> torch.Tensor:
        return self.fc(cls_emb).squeeze(1)

def train_pseudotime_regressor(cls_embeddings: np.ndarray,
                               pseudotime_targets: np.ndarray,
                               d_model: int,
                               device: torch.device,
                               batch_size: int = 64,
                               num_epochs: int = 10,
                               lr: float = 1e-3):
    """
    Given all CLS embeddings (N, d_model) and their target pseudotimes (N,),
    train a simple MSE regressor to predict pseudotime from each CLS.

    Returns:
        trained_head: PseudotimeHead on device
    """
    N = cls_embeddings.shape[0]
    # Build a simple TensorDataset and DataLoader
    class RegDataset(torch.utils.data.Dataset):
        def __init__(self, cls_emb_np, pt_np):
            self.cls_t = torch.from_numpy(cls_emb_np).float()
            self.pt_t  = torch.from_numpy(pt_np).float()
        def __len__(self):
            return self.cls_t.size(0)
        def __getitem__(self, idx):
            return self.cls_t[idx], self.pt_t[idx]

    reg_dataset = RegDataset(cls_embeddings, pseudotime_targets)
    reg_loader = DataLoader(reg_dataset, batch_size=batch_size, shuffle=True)

    # Instantiate the head
    head = PseudotimeHead(input_dim=d_model).to(device)
    optimizer = torch.optim.Adam(head.parameters(), lr=lr)
    criterion = nn.MSELoss()

    head.train()
    for epoch in range(num_epochs):
        total_loss = 0.0
        for cls_batch, pt_batch in reg_loader:
            cls_batch = cls_batch.to(device)
            pt_batch  = pt_batch.to(device)

            optimizer.zero_grad()
            pred = head(cls_batch)       # (B,)
            loss = criterion(pred, pt_batch)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(reg_loader)
        print(f"[Regressor] Epoch {epoch+1}/{num_epochs} — MSE Loss = {avg_loss:.5f}")

    return head, optimizer, epoch