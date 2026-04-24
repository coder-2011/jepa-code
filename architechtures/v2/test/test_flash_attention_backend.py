import torch
import torch.nn.functional as F

import utils.flash_attention as flash_attention_backend


def test_flash_attention_sdpa_fallback_matches_reference():
    flash_attention_backend.set_backend_override("sdpa")
    q = torch.randn(2, 5, 3, 4)
    k = torch.randn(2, 5, 3, 4)
    v = torch.randn(2, 5, 3, 4)

    out = flash_attention_backend.flash_attn.flash_attn_func(q, k, v, causal=True)
    ref = F.scaled_dot_product_attention(
        q.transpose(1, 2),
        k.transpose(1, 2),
        v.transpose(1, 2),
        is_causal=True,
    ).transpose(1, 2)

    assert torch.allclose(out, ref, atol=1e-6, rtol=1e-5)
    flash_attention_backend.set_backend_override(None)


def test_flash_attention_kvcache_sdpa_fallback_matches_reference_and_updates_cache():
    flash_attention_backend.set_backend_override("sdpa")
    batch_size, max_cache_len, query_length, num_heads, head_dim = 2, 8, 2, 3, 4
    q = torch.randn(batch_size, query_length, num_heads, head_dim)
    k_new = torch.randn(batch_size, query_length, num_heads, head_dim)
    v_new = torch.randn(batch_size, query_length, num_heads, head_dim)
    k_cache = torch.randn(batch_size, max_cache_len, num_heads, head_dim)
    v_cache = torch.randn(batch_size, max_cache_len, num_heads, head_dim)
    k_cache_ref = k_cache.clone()
    v_cache_ref = v_cache.clone()
    cache_seqlens = torch.tensor([1, 3], dtype=torch.int32)

    out = flash_attention_backend.flash_attn.flash_attn_with_kvcache(
        q,
        k_cache,
        v_cache,
        k=k_new,
        v=v_new,
        cache_seqlens=cache_seqlens,
        causal=True,
    )

    ref_outputs = []
    for batch_idx in range(batch_size):
        pos = int(cache_seqlens[batch_idx].item())
        k_cache_ref[batch_idx, pos:pos + query_length] = k_new[batch_idx]
        v_cache_ref[batch_idx, pos:pos + query_length] = v_new[batch_idx]
        end = pos + query_length
        key_idx = torch.arange(end).view(1, 1, end)
        query_abs_idx = pos + torch.arange(query_length).view(query_length, 1)
        mask = (key_idx <= query_abs_idx.unsqueeze(0)).unsqueeze(0)
        ref = F.scaled_dot_product_attention(
            q[batch_idx:batch_idx + 1].transpose(1, 2),
            k_cache_ref[batch_idx:batch_idx + 1, :end].transpose(1, 2),
            v_cache_ref[batch_idx:batch_idx + 1, :end].transpose(1, 2),
            attn_mask=mask,
        ).transpose(1, 2)
        ref_outputs.append(ref)

    assert torch.equal(k_cache, k_cache_ref)
    assert torch.equal(v_cache, v_cache_ref)
    assert torch.allclose(out, torch.cat(ref_outputs, dim=0), atol=1e-6, rtol=1e-5)
    flash_attention_backend.set_backend_override(None)
