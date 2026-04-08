import random

import torch

from text_jepa.objectives.stp import (
    InclusiveSpan,
    RandomSpanDecomposition,
    SemanticTubeBounds,
    decompose_random_span,
    random_span_batch_loss,
    random_span_loss,
    sample_random_span,
)


def make_hidden_states():
    return torch.tensor(
        [
            [0.0, 0.0],
            [1.0, 2.0],
            [3.0, 5.0],
            [6.0, 9.0],
            [10.0, 14.0],
            [15.0, 20.0],
        ],
        dtype=torch.float32,
    )


def make_bounds():
    return SemanticTubeBounds(
        source=InclusiveSpan(0, 2),
        target=InclusiveSpan(2, 5),
    )


def test_sample_random_span_is_deterministic_with_seed():
    bounds = make_bounds()
    rng_a = random.Random(7)
    rng_b = random.Random(7)

    patch_a = sample_random_span(bounds, rng=rng_a)
    patch_b = sample_random_span(bounds, rng=rng_b)

    assert patch_a == patch_b


def test_sample_random_span_respects_tube_and_non_empty_contract():
    bounds = make_bounds()
    rng = random.Random(3)

    patch = sample_random_span(bounds, rng=rng, min_span_length=1, max_span_length=2)

    assert patch.start >= bounds.source.start
    assert patch.end <= bounds.target.end
    assert patch.start < patch.end
    assert patch != bounds.tube
    assert patch.length <= 2


def test_decompose_random_span_produces_expected_context_patch_and_tube_embeddings():
    hidden_states = make_hidden_states()
    bounds = make_bounds()
    patch = InclusiveSpan(1, 3)

    decomposition = decompose_random_span(hidden_states, bounds, patch)

    assert isinstance(decomposition, RandomSpanDecomposition)
    assert torch.allclose(decomposition.before, torch.tensor([0.0, 0.0]))
    assert torch.allclose(decomposition.patch_embedding, torch.tensor([2.0, 3.5]))
    assert torch.allclose(decomposition.after, torch.tensor([8.0, 11.5]))
    assert torch.allclose(decomposition.context_embedding, torch.tensor([16.0 / 3.0, 23.0 / 3.0]))
    assert torch.allclose(decomposition.tube_embedding, torch.tensor([4.0, 6.0]))


def test_decompose_random_span_handles_tube_boundaries():
    hidden_states = make_hidden_states()
    bounds = make_bounds()
    patch = InclusiveSpan(0, 1)

    decomposition = decompose_random_span(hidden_states, bounds, patch)

    assert torch.allclose(decomposition.before, torch.zeros(2))
    assert torch.allclose(decomposition.patch_embedding, torch.tensor([0.0, 0.0]))
    assert torch.allclose(decomposition.after, torch.tensor([5.0, 7.5]))
    assert torch.allclose(decomposition.context_embedding, decomposition.after)
    assert torch.allclose(decomposition.tube_embedding, torch.tensor([4.0, 6.0]))


def test_decompose_random_span_context_is_mean_of_non_patch_tokens():
    hidden_states = make_hidden_states()
    bounds = make_bounds()
    patch = InclusiveSpan(1, 3)

    decomposition = decompose_random_span(hidden_states, bounds, patch)

    assert torch.allclose(
        decomposition.context_embedding,
        torch.stack([hidden_states[0], hidden_states[3], hidden_states[4]], dim=0).mean(dim=0),
    )
    assert torch.allclose(
        decomposition.patch_embedding,
        torch.stack([hidden_states[1], hidden_states[2]], dim=0).mean(dim=0),
    )


def test_random_span_loss_supports_cosine_mse_and_l2():
    hidden_states = make_hidden_states()
    bounds = make_bounds()
    patch = InclusiveSpan(1, 3)

    cosine_loss = random_span_loss(hidden_states, bounds, patch=patch, metric="cosine")
    mse_loss = random_span_loss(hidden_states, bounds, patch=patch, metric="mse")
    l2_loss = random_span_loss(hidden_states, bounds, patch=patch, metric="l2")

    assert cosine_loss.ndim == 0
    assert mse_loss.ndim == 0
    assert l2_loss.ndim == 0
    assert torch.isfinite(cosine_loss)
    assert torch.isfinite(mse_loss)
    assert torch.isfinite(l2_loss)


def test_random_span_batch_loss_returns_scalar_for_batched_hidden_states():
    hidden_states = torch.stack([make_hidden_states(), make_hidden_states() + 1.0], dim=0)

    loss = random_span_batch_loss(
        hidden_states,
        source_lengths=torch.tensor([2, 2]),
        target_lengths=torch.tensor([3, 3]),
        rng=random.Random(5),
        samples=2,
        metric="cosine",
    )

    assert loss.ndim == 0
    assert torch.isfinite(loss)


def test_random_span_loss_can_return_decomposition():
    hidden_states = make_hidden_states()
    bounds = make_bounds()
    patch = InclusiveSpan(1, 3)

    loss, decomposition = random_span_loss(
        hidden_states,
        bounds,
        patch=patch,
        return_decomposition=True,
    )

    assert loss.ndim == 0
    assert isinstance(decomposition, RandomSpanDecomposition)
    assert decomposition.patch == patch


def test_noncontiguous_semantic_bounds_exclude_gap_tokens_from_context_and_tube():
    hidden_states = make_hidden_states()
    bounds = SemanticTubeBounds(
        source=InclusiveSpan(0, 2),
        target=InclusiveSpan(4, 6),
    )
    patch = InclusiveSpan(4, 5)

    decomposition = decompose_random_span(hidden_states, bounds, patch)

    assert torch.allclose(decomposition.before, torch.tensor([0.5, 1.0]))
    assert torch.allclose(decomposition.patch_embedding, torch.tensor([10.0, 14.0]))
    assert torch.allclose(decomposition.after, torch.tensor([15.0, 20.0]))
    assert torch.allclose(decomposition.context_embedding, torch.tensor([16.0 / 3.0, 22.0 / 3.0]))
    assert torch.allclose(decomposition.tube_embedding, torch.tensor([6.5, 9.0]))


def test_random_span_batch_loss_uses_explicit_noncontiguous_packed_bounds():
    hidden_states = make_hidden_states().unsqueeze(0)

    loss, context_embeddings, patch_embeddings, weights = random_span_batch_loss(
        hidden_states,
        source_lengths=torch.tensor([1]),
        target_lengths=torch.tensor([2]),
        source_span_starts=torch.tensor([0]),
        source_span_ends=torch.tensor([1]),
        target_span_starts=torch.tensor([4]),
        target_span_ends=torch.tensor([6]),
        rng=random.Random(0),
        samples=1,
        min_span_length=2,
        max_span_length=2,
        metric="mse",
        return_embeddings=True,
    )

    assert torch.isfinite(loss)
    assert torch.allclose(context_embeddings, torch.tensor([[0.0, 0.0]]))
    assert torch.allclose(patch_embeddings, torch.tensor([[12.5, 17.0]]))
    assert torch.allclose(weights, torch.tensor([1.0]))
