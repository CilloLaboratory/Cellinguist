from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

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
    batch_size: int = 1024
    epochs: int = 5
    lr: float = 1e-3
    weight_decay: float = 0.0
    device: str = "cuda"
    use_separate_output: bool = True

    lr_scheduler: str = "none"  # or "step"
    lr_step_size: int = 10
    lr_gamma: float = 0.5


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
