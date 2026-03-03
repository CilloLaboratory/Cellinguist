from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Any, Dict

import json

try:
    import yaml  # type: ignore
    _HAS_YAML = True
except ImportError:
    _HAS_YAML = False


# ---------------------------------------------------------------------------
# CBOW config
# ---------------------------------------------------------------------------

@dataclass
class CBOWConfig:
    """
    Configuration for CBOW training.

    Attributes
    ----------
    emb_dim : int
        Embedding dimension for token embeddings.
    window_size : int
        Context window radius. Total context size = 2 * window_size.
    num_negatives : int
        Number of negative samples per (target, context) pair.
    num_workers : int
        Number of cpus for dataloader.
    batch_size : int
        Minibatch size for training.
    epochs : int
        Number of training epochs.
    lr : float
        Learning rate for the optimizer.
    weight_decay : float
        Weight decay (L2 regularization) for the optimizer.
    device : str
        Device string, e.g. "cpu" or "cuda".
    use_separate_output : bool
        If True, use a separate output embedding for targets/negatives
        (classic word2vec). If False, tie input and output embeddings.
    lr_scheduler : str
        Type of LR scheduler: "none" or "step".
    lr_step_size : int
        Step size (in epochs) for StepLR, if used.
    lr_gamma : float
        Multiplicative factor of learning rate decay for StepLR.
    """

    emb_dim: int = 128
    window_size: int = 5
    num_negatives: int = 10
    num_workers: int = 12
    batch_size: int = 1024
    epochs: int = 5
    lr: float = 1e-3
    weight_decay: float = 0.0
    device: str = "cuda"
    use_separate_output: bool = True

    lr_scheduler: str = "none"  # or "step"
    lr_step_size: int = 10
    lr_gamma: float = 0.5

    samples_per_cell: int = 1


# ---------------------------------------------------------------------------
# VAE Config
# ---------------------------------------------------------------------------

@dataclass
class VAETrainConfig:
    adata_path: str
    gene_key: str = "gene"
    layer: Optional[str] = None
    cond_key: Optional[str] = None
    backed: bool = True

    # Used only when encoder_type == "cbow".
    gene_emb_tsv: str = ""
    encoder_type: str = "perceiver"  # "perceiver" or "cbow"

    latent_dim: int = 32
    hidden_dim: int = 256
    n_hidden_layers: int = 2
    cond_emb_dim: int = 16
    input_transform: str = "log1p"
    library_norm: str = "size_factor"  # "size_factor" or "none" (Perceiver encoder)
    library_norm_target_sum: float = 1e4
    library_norm_eps: float = 1e-8
    use_library_size_covariate: bool = False  # Decoder-side log1p(library_size) covariate
    library_size_covariate_eps: float = 1e-8
    freeze_gene_embeddings: bool = True
    perceiver_d_model: int = 256
    perceiver_num_latents: int = 64
    perceiver_num_cross_attn_heads: int = 8
    perceiver_num_self_attn_heads: int = 8
    perceiver_num_self_attn_layers: int = 4
    perceiver_ff_mult: int = 4
    perceiver_dropout: float = 0.0

    kl_weight: float = 1.0
    use_metric_loss: bool = False
    metric_loss_weight: float = 0.1
    metric_expr_transform: str = "log1p"  # "log1p" or "none"
    metric_margin: float = 0.2
    metric_k_pos: int = 5
    metric_k_neg: int = 20

    lr: float = 3e-4
    weight_decay: float = 0.0
    batch_size: int = 256
    epochs: int = 50
    num_workers: int = 4
    device: str = "cuda"
    grad_clip_norm: float = 1.0
    seed: Optional[int] = 0

    checkpoint_dir: str = "checkpoints"
    run_name: str = "vae_run"
    resume_from: Optional[str] = None
    save_every: int = 1

    decoder_theta_init: float = 5.0          # initial theta (dispersion), gene-wise
    decoder_pi_init: float = 0.9             # initial dropout prob pi (ZI prob)
    decoder_mu_init: str = "data_mean"       # "data_mean" or "constant" or "none"
    decoder_mu_init_constant: float = 0.2    # used if decoder_mu_init == "constant"
    decoder_mu_init_cap: float = 10.0        # cap for gene-wise mean init
    decoder_mu_init_eps: float = 1e-4        # lower bound for mean init

    decoder_init_n_cells: int = 5000         # number of cells to estimate gene means from
    decoder_init_batch_size: int = 256       # batch size for mean-estimation pass
    decoder_init_num_workers: int = 0   

@dataclass
class VAEExportConfig:
    adata_path: str
    gene_key: str = "gene"
    layer: Optional[str] = None
    cond_key: Optional[str] = None

    # Used only when checkpoint/config indicates encoder_type == "cbow".
    gene_emb_tsv: str = ""
    checkpoint_path: str = ""
    out_pred_tsv_gz: str = "pred_mu.tsv.gz"

    max_cells: int = 1000
    batch_size: int = 64
    num_workers: int = 4
    device: str = "cuda"


def load_yaml(path: str) -> Dict[str, Any]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(path)
    with p.open("r") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError("Config must be a YAML mapping.")
    return cfg

# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

def load_config(path: str) -> Dict[str, Any]:
    """
    Load a configuration file for CBOW training.

    The file can be YAML (.yml, .yaml) or JSON (.json).
    Returns a dict expected to contain (optionally) the keys:
      - "cbow":   CBOWConfig-related kwargs
      - "data":   data-related settings (paths, gene_key, etc.)
      - "output": output settings (e.g., embeddings_path)

    Any missing sections are filled with empty dicts.

    Parameters
    ----------
    path : str
        Path to a YAML or JSON config file.

    Returns
    -------
    cfg : Dict[str, Any]
        Parsed configuration dictionary with at least the keys:
        "cbow", "data", and "output".
    """
    path_obj = Path(path)
    if not path_obj.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    suffix = path_obj.suffix.lower()

    if suffix in {".yml", ".yaml"}:
        if not _HAS_YAML:
            raise ImportError(
                "PyYAML is required to load YAML config files. "
                "Install with `pip install pyyaml`."
            )
        with path_obj.open("r") as f:
            cfg = yaml.safe_load(f)
    elif suffix == ".json":
        with path_obj.open("r") as f:
            cfg = json.load(f)
    else:
        raise ValueError(
            f"Unsupported config file extension '{suffix}'. "
            "Use .yml, .yaml, or .json."
        )

    if not isinstance(cfg, dict):
        raise ValueError("Top-level config must be a mapping/dict.")

    # Ensure the expected top-level sections exist
    cfg.setdefault("cbow", {})
    cfg.setdefault("data", {})
    cfg.setdefault("output", {})

    return cfg
