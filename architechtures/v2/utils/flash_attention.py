"""Unified flash-attention wrapper with SDPA fallback.

This module exposes a small `flash_attn` namespace with a FA-style API:

    from utils.flash_attention import flash_attn

    y = flash_attn.flash_attn_func(q, k, v, causal=True)

Inputs and outputs use `(B, T, H, D)` layout. On CUDA, we attempt to use
`flash_attn.cute.flash_attn_func` from `flash-attn-4` when it is available and
compatible with the current tensors. Otherwise we fall back to PyTorch SDPA.
"""

from __future__ import annotations

import warnings
from types import SimpleNamespace

import torch
import torch.nn.functional as F


def _normalize_window_size(window_size: tuple[int | None, int | None]) -> tuple[int, int]:
    left, right = window_size
    return (
        -1 if left is None else int(left),
        -1 if right is None else int(right),
    )


def _load_flash_attention():
    if not torch.cuda.is_available():
        return None
    try:
        from flash_attn.cute import flash_attn_func as flash_attn_func_impl
    except Exception:
        return None
    return flash_attn_func_impl


_flash_attn_func_impl = _load_flash_attention()
HAS_FLASH_ATTN = _flash_attn_func_impl is not None
_override_impl: str | None = None
_runtime_disabled_reason: str | None = None


def set_backend_override(impl: str | None) -> None:
    assert impl in {None, "flash", "sdpa"}, "impl must be None, 'flash', or 'sdpa'"
    global _override_impl
    _override_impl = impl


def _disable_flash_runtime(exc: Exception) -> None:
    global _runtime_disabled_reason
    if _runtime_disabled_reason is None:
        _runtime_disabled_reason = f"{type(exc).__name__}: {exc}"
        warnings.warn(
            f"Flash attention backend disabled after runtime failure; falling back to SDPA. {_runtime_disabled_reason}",
            RuntimeWarning,
            stacklevel=2,
        )


def _use_flash(q: torch.Tensor, dropout_p: float) -> bool:
    if _override_impl == "sdpa":
        return False
    if _override_impl == "flash":
        assert HAS_FLASH_ATTN, "Cannot override to flash attention: backend import failed"
    if _flash_attn_func_impl is None or _runtime_disabled_reason is not None:
        return False
    if q.device.type != "cuda":
        return False
    if q.dtype not in {torch.float16, torch.bfloat16}:
        return False
    if dropout_p != 0.0:
        return False
    return True


def _sdpa_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    causal: bool,
    window_size: tuple[int | None, int | None],
    enable_gqa: bool,
    dropout_p: float,
) -> torch.Tensor:
    """SDPA attention with the same `(B, H, T, D)` internal layout as PyTorch."""
    left_window, _ = _normalize_window_size(window_size)
    query_length = q.size(2)
    key_length = k.size(2)

    if left_window < 0 and query_length == key_length:
        return F.scaled_dot_product_attention(
            q,
            k,
            v,
            is_causal=causal,
            attn_mask=None,
            dropout_p=dropout_p,
            enable_gqa=enable_gqa,
        )

    if query_length == 1 and causal:
        if 0 <= left_window < key_length:
            start = max(0, key_length - (left_window + 1))
            k = k[:, :, start:, :]
            v = v[:, :, start:, :]
        return F.scaled_dot_product_attention(
            q,
            k,
            v,
            is_causal=False,
            attn_mask=None,
            dropout_p=dropout_p,
            enable_gqa=enable_gqa,
        )

    if not causal and left_window < 0:
        return F.scaled_dot_product_attention(
            q,
            k,
            v,
            is_causal=False,
            attn_mask=None,
            dropout_p=dropout_p,
            enable_gqa=enable_gqa,
        )

    device = q.device
    row_idx = (key_length - query_length) + torch.arange(query_length, device=device).unsqueeze(1)
    col_idx = torch.arange(key_length, device=device).unsqueeze(0)
    mask = col_idx <= row_idx if causal else torch.ones(query_length, key_length, device=device, dtype=torch.bool)
    if 0 <= left_window < key_length:
        if causal:
            mask = mask & ((row_idx - col_idx) <= left_window)
        else:
            raise AssertionError("Non-causal sliding-window attention is not implemented in this backend")

    return F.scaled_dot_product_attention(
        q,
        k,
        v,
        is_causal=False,
        attn_mask=mask,
        dropout_p=dropout_p,
        enable_gqa=enable_gqa,
    )


def flash_attn_func(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    causal: bool = False,
    window_size: tuple[int | None, int | None] = (-1, -1),
    dropout_p: float = 0.0,
) -> torch.Tensor:
    """Attention on `(B, T, H, D)` tensors."""
    if _use_flash(q, dropout_p):
        fa_window_size = tuple(None if size is None or size < 0 else int(size) for size in window_size)
        try:
            return _flash_attn_func_impl(q, k, v, causal=causal, window_size=fa_window_size)
        except Exception as exc:
            _disable_flash_runtime(exc)

    q_sdpa = q.transpose(1, 2)
    k_sdpa = k.transpose(1, 2)
    v_sdpa = v.transpose(1, 2)
    enable_gqa = q_sdpa.size(1) != k_sdpa.size(1)
    y_sdpa = _sdpa_attention(
        q_sdpa,
        k_sdpa,
        v_sdpa,
        causal=causal,
        window_size=window_size,
        enable_gqa=enable_gqa,
        dropout_p=dropout_p,
    )
    return y_sdpa.transpose(1, 2)


def flash_attn_with_kvcache(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    *,
    k: torch.Tensor | None = None,
    v: torch.Tensor | None = None,
    cache_seqlens: torch.Tensor | None = None,
    causal: bool = False,
    window_size: tuple[int | None, int | None] = (-1, -1),
    dropout_p: float = 0.0,
) -> torch.Tensor:
    """Attention with explicit KV cache management on `(B, T, H, D)` tensors.

    The cache path uses SDPA regardless of flash availability. `flash-attn-4`
    does not expose a matching KV-cache API in this environment.
    """
    batch_size, query_length, _, _ = q.shape
    if cache_seqlens is None:
        cache_seqlens = torch.zeros(batch_size, device=q.device, dtype=torch.int32)
    else:
        cache_seqlens = cache_seqlens.to(device=q.device, dtype=torch.int32)

    if k is not None and v is not None:
        for batch_idx in range(batch_size):
            pos = int(cache_seqlens[batch_idx].item())
            k_cache[batch_idx, pos:pos + query_length, :, :] = k[batch_idx]
            v_cache[batch_idx, pos:pos + query_length, :, :] = v[batch_idx]

    end_positions = cache_seqlens + query_length
    max_end = int(end_positions.max().item())
    k_full = k_cache[:, :max_end, :, :]
    v_full = v_cache[:, :max_end, :, :]

    q_sdpa = q.transpose(1, 2)
    k_sdpa = k_full.transpose(1, 2)
    v_sdpa = v_full.transpose(1, 2)
    enable_gqa = q_sdpa.size(1) != k_sdpa.size(1)

    left_window, _ = _normalize_window_size(window_size)
    key_idx = torch.arange(max_end, device=q.device).view(1, 1, max_end)
    query_abs_idx = cache_seqlens[:, None, None] + torch.arange(query_length, device=q.device).view(1, query_length, 1)
    if causal:
        mask = key_idx <= query_abs_idx
        if 0 <= left_window < max_end:
            mask = mask & ((query_abs_idx - key_idx) <= left_window)
    else:
        mask = key_idx < end_positions[:, None, None]

    y_sdpa = F.scaled_dot_product_attention(
        q_sdpa,
        k_sdpa,
        v_sdpa,
        is_causal=False,
        attn_mask=mask.unsqueeze(1),
        dropout_p=dropout_p,
        enable_gqa=enable_gqa,
    )
    return y_sdpa.transpose(1, 2)


flash_attn = SimpleNamespace(
    flash_attn_func=flash_attn_func,
    flash_attn_with_kvcache=flash_attn_with_kvcache,
)
