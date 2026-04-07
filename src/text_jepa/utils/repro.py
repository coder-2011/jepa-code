import os
import random

import torch


def _runtime_config(config):
    runtime_config = config.get("runtime") or {}
    if not isinstance(runtime_config, dict):
        raise ValueError("runtime config must be a mapping")
    return runtime_config


def resolve_seed(config, override=None):
    if override is not None:
        seed = override
    else:
        seed = _runtime_config(config).get("seed", 0)
    if not isinstance(seed, int):
        raise ValueError("runtime.seed must be an integer")
    return seed


def resolve_deterministic(config, override=None):
    if override is not None:
        deterministic = override
    else:
        deterministic = _runtime_config(config).get("deterministic", True)
    if not isinstance(deterministic, bool):
        raise ValueError("runtime.deterministic must be a boolean")
    return deterministic


def configure_reproducibility(seed, deterministic=True):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    try:
        import numpy as np
    except ImportError:
        np = None
    if np is not None:
        np.random.seed(seed)

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.cuda.manual_seed_all(seed)

    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = deterministic
        torch.backends.cudnn.benchmark = not deterministic

    if hasattr(torch, "use_deterministic_algorithms"):
        torch.use_deterministic_algorithms(deterministic, warn_only=True)
