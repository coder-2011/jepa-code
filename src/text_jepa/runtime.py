import torch


def default_device() -> str:
    # Keep CLI behavior explicit and stable across machines: implicit runs target CUDA unless overridden.
    return "cuda"


def validate_device(device: str) -> str:
    if device == "cuda":
        if not torch.cuda.is_available():
            raise ValueError(
                "device='cuda' was requested but CUDA is not available. "
                "Use --device mps on Apple Silicon or --device cpu if you want a non-CUDA run."
            )
        return device

    if device == "mps":
        if not hasattr(torch.backends, "mps") or not torch.backends.mps.is_available():
            raise ValueError("device='mps' was requested but MPS is not available on this machine")
        return device

    if device == "cpu":
        return device

    raise ValueError("device must be one of: cuda, mps, cpu")
