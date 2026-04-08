from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Literal

import torch
import torch.nn.functional as F


Metric = Literal["cosine", "mse", "l2"]


@dataclass(frozen=True)
class InclusiveSpan:
    """Half-open span [start, end) over a hidden-state sequence."""

    start: int
    end: int

    def __post_init__(self) -> None:
        if self.start < 0:
            raise ValueError("span start must be non-negative")
        if self.end <= self.start:
            raise ValueError("span end must be greater than span start")

    @property
    def length(self) -> int:
        return self.end - self.start


@dataclass(frozen=True)
class SemanticTubeBounds:
    """Ordered source/target semantic spans used by the STP objective."""

    source: InclusiveSpan
    target: InclusiveSpan

    def __post_init__(self) -> None:
        if self.source.end > self.target.start:
            raise ValueError("source and target spans must be ordered and non-overlapping")

    @property
    def tube(self) -> InclusiveSpan:
        return InclusiveSpan(self.source.start, self.target.end)

    @property
    def total_length(self) -> int:
        return self.source.length + self.target.length

    def semantic_token_indices(self) -> list[int]:
        return list(range(self.source.start, self.source.end)) + list(range(self.target.start, self.target.end))

    def contiguous_runs(self) -> list[InclusiveSpan]:
        if self.source.end == self.target.start:
            return [InclusiveSpan(self.source.start, self.target.end)]
        return [self.source, self.target]


@dataclass(frozen=True)
class RandomSpanDecomposition:
    patch: InclusiveSpan
    before: torch.Tensor
    patch_embedding: torch.Tensor
    after: torch.Tensor
    context_embedding: torch.Tensor
    tube_embedding: torch.Tensor


def _validate_hidden_states(hidden_states: torch.Tensor) -> None:
    if hidden_states.ndim != 2:
        raise ValueError("hidden_states must have shape (L, D)")


def _validate_bounds(hidden_states: torch.Tensor, bounds: SemanticTubeBounds) -> None:
    if bounds.source.end > hidden_states.shape[0] or bounds.target.end > hidden_states.shape[0]:
        raise ValueError("semantic tube boundaries must fit inside hidden_states")


def _segment_embedding(hidden_states: torch.Tensor, start: int, end: int) -> torch.Tensor:
    if start > end:
        raise ValueError("span start cannot be greater than span end")
    if start < 0 or end > hidden_states.shape[0]:
        raise ValueError("span must lie inside the hidden-state sequence")
    if start == end:
        return torch.zeros_like(hidden_states[0])
    return hidden_states[start:end].mean(dim=0)


def _context_embedding(hidden_states: torch.Tensor, tube: InclusiveSpan, patch: InclusiveSpan) -> torch.Tensor:
    segments = []
    if tube.start < patch.start:
        segments.append(hidden_states[tube.start:patch.start])
    if patch.end < tube.end:
        segments.append(hidden_states[patch.end:tube.end])
    if not segments:
        return torch.zeros_like(hidden_states[0])
    return torch.cat(segments, dim=0).mean(dim=0)


def _embedding_from_indices(hidden_states: torch.Tensor, indices: list[int]) -> torch.Tensor:
    if not indices:
        return torch.zeros_like(hidden_states[0])
    return hidden_states[torch.tensor(indices, device=hidden_states.device)].mean(dim=0)


def _semantic_index_groups(bounds: SemanticTubeBounds, patch: InclusiveSpan) -> tuple[list[int], list[int], list[int], list[int]]:
    semantic_indices = bounds.semantic_token_indices()
    patch_positions = [
        position
        for position, raw_index in enumerate(semantic_indices)
        if patch.start <= raw_index < patch.end
    ]
    if not patch_positions:
        raise ValueError("patch must cover at least one semantic token")
    if len(patch_positions) != patch.length:
        raise ValueError("patch must not include non-semantic gap tokens")
    first_position = patch_positions[0]
    last_position = patch_positions[-1]
    if patch_positions != list(range(first_position, last_position + 1)):
        raise ValueError("patch must be contiguous in semantic-token order")

    before = semantic_indices[:first_position]
    patch_indices = semantic_indices[first_position : last_position + 1]
    after = semantic_indices[last_position + 1 :]
    return before, patch_indices, after, semantic_indices


def span_weight(total_length: int, patch_length: int, length_adjustment: str | None = None) -> float:
    rest_length = total_length - patch_length
    if length_adjustment is None:
        return 1.0
    if length_adjustment == "cosine_like":
        return 2.0 * rest_length * patch_length / (rest_length * rest_length + patch_length * patch_length)
    if length_adjustment == "jaccard_like":
        return 1.0 - abs(rest_length - patch_length) / (rest_length + patch_length)
    raise ValueError(f"unknown length_adjustment: {length_adjustment}")


def sample_random_span(
    bounds: SemanticTubeBounds,
    *,
    rng: random.Random | None = None,
    min_span_length: int = 1,
    max_span_length: int | None = None,
) -> InclusiveSpan:
    """Sample a non-empty proper patch inside the semantic source/target spans."""

    if rng is None:
        rng = random.Random()
    if min_span_length < 1:
        raise ValueError("min_span_length must be at least 1")
    if max_span_length is not None and max_span_length < min_span_length:
        raise ValueError("max_span_length must be greater than or equal to min_span_length")

    tube_length = bounds.total_length
    if tube_length < 2:
        raise ValueError("semantic tube must contain at least two tokens for random-span STP")

    effective_max = tube_length - 1
    if max_span_length is not None:
        effective_max = min(effective_max, max_span_length)
    if effective_max < min_span_length:
        raise ValueError("no valid span lengths remain after applying max_span_length")

    valid_runs = [run for run in bounds.contiguous_runs() if run.length >= min_span_length]
    if not valid_runs:
        raise ValueError("semantic tube must include at least one contiguous run that can hold a patch")
    while True:
        span_length = rng.randint(min_span_length, effective_max)
        eligible_runs = [run for run in valid_runs if run.length >= span_length]
        if not eligible_runs:
            continue
        run = eligible_runs[rng.randrange(len(eligible_runs))]
        start = rng.randint(run.start, run.end - span_length)
        end = start + span_length
        patch = InclusiveSpan(start, end)
        if patch.length < tube_length:
            return patch


def decompose_random_span(
    hidden_states: torch.Tensor,
    bounds: SemanticTubeBounds,
    patch: InclusiveSpan,
) -> RandomSpanDecomposition:
    """Split a semantic tube into context and patch embeddings."""

    _validate_hidden_states(hidden_states)
    _validate_bounds(hidden_states, bounds)

    tube = bounds.tube
    if patch.start < bounds.source.start or patch.end > bounds.target.end:
        raise ValueError("patch must lie inside the semantic tube")
    before_indices, patch_indices, after_indices, tube_indices = _semantic_index_groups(bounds, patch)
    before = _embedding_from_indices(hidden_states, before_indices)
    patch_embedding = _embedding_from_indices(hidden_states, patch_indices)
    after = _embedding_from_indices(hidden_states, after_indices)
    context_embedding = _embedding_from_indices(hidden_states, before_indices + after_indices)
    tube_embedding = _embedding_from_indices(hidden_states, tube_indices)

    return RandomSpanDecomposition(
        patch=patch,
        before=before,
        patch_embedding=patch_embedding,
        after=after,
        context_embedding=context_embedding,
        tube_embedding=tube_embedding,
    )


def _loss_from_embeddings(
    context_embedding: torch.Tensor,
    patch_embedding: torch.Tensor,
    *,
    metric: Metric,
) -> torch.Tensor:
    if metric == "cosine":
        return 1.0 - F.cosine_similarity(context_embedding.unsqueeze(0), patch_embedding.unsqueeze(0)).mean()
    if metric == "mse":
        return F.mse_loss(context_embedding, patch_embedding)
    if metric == "l2":
        return torch.linalg.vector_norm(context_embedding - patch_embedding, ord=2)
    raise ValueError("metric must be one of: cosine, mse, l2")


def random_span_loss(
    hidden_states: torch.Tensor,
    bounds: SemanticTubeBounds,
    *,
    patch: InclusiveSpan | None = None,
    rng: random.Random | None = None,
    min_span_length: int = 1,
    max_span_length: int | None = None,
    metric: Metric = "cosine",
    return_decomposition: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, RandomSpanDecomposition]:
    """Compute the core STP random-span loss on a contiguous semantic tube."""

    if patch is None:
        patch = sample_random_span(
            bounds,
            rng=rng,
            min_span_length=min_span_length,
            max_span_length=max_span_length,
        )

    decomposition = decompose_random_span(hidden_states, bounds, patch)
    loss = _loss_from_embeddings(
        decomposition.context_embedding,
        decomposition.patch_embedding,
        metric=metric,
    )

    if return_decomposition:
        return loss, decomposition
    return loss


def random_span_batch_loss(
    hidden_states: torch.Tensor,
    source_lengths,
    target_lengths,
    *,
    source_span_starts=None,
    source_span_ends=None,
    target_span_starts=None,
    target_span_ends=None,
    rng: random.Random | None = None,
    samples: int = 1,
    min_span_length: int = 1,
    max_span_length: int | None = None,
    metric: Metric = "cosine",
    length_adjustment: str | None = None,
    predictor=None,
    return_embeddings: bool = False,
):
    if hidden_states.ndim != 3:
        raise ValueError("hidden_states must have shape (B, L, D)")
    if samples <= 0:
        raise ValueError("samples must be positive")

    if isinstance(source_lengths, torch.Tensor):
        source_lengths = source_lengths.tolist()
    if isinstance(target_lengths, torch.Tensor):
        target_lengths = target_lengths.tolist()
    batch_size = hidden_states.shape[0]
    if len(source_lengths) != batch_size or len(target_lengths) != batch_size:
        raise ValueError("source_lengths and target_lengths must match the batch dimension of hidden_states")
    if source_span_starts is None:
        source_span_starts = [0] * batch_size
    elif isinstance(source_span_starts, torch.Tensor):
        source_span_starts = source_span_starts.tolist()
    if source_span_ends is None:
        source_span_ends = source_lengths
    elif isinstance(source_span_ends, torch.Tensor):
        source_span_ends = source_span_ends.tolist()
    if target_span_starts is None:
        target_span_starts = source_lengths
    elif isinstance(target_span_starts, torch.Tensor):
        target_span_starts = target_span_starts.tolist()
    if target_span_ends is None:
        target_span_ends = [int(start) + int(length) for start, length in zip(target_span_starts, target_lengths)]
    elif isinstance(target_span_ends, torch.Tensor):
        target_span_ends = target_span_ends.tolist()
    if not all(
        len(values) == batch_size
        for values in (source_span_starts, source_span_ends, target_span_starts, target_span_ends)
    ):
        raise ValueError("packed span boundary lists must match the batch dimension of hidden_states")

    if rng is None:
        rng = random.Random()

    context_embeddings = []
    patch_embeddings = []
    weights = []

    for index, (source_length, target_length, source_start, source_end, target_start, target_end) in enumerate(
        zip(
            source_lengths,
            target_lengths,
            source_span_starts,
            source_span_ends,
            target_span_starts,
            target_span_ends,
        )
    ):
        source_length = int(source_length)
        target_length = int(target_length)
        source_start = int(source_start)
        source_end = int(source_end)
        target_start = int(target_start)
        target_end = int(target_end)
        total_length = source_length + target_length
        if source_length <= 0 or target_length <= 0 or total_length < 2:
            continue
        if source_end - source_start != source_length:
            raise ValueError("source span boundaries must match source_lengths")
        if target_end - target_start != target_length:
            raise ValueError("target span boundaries must match target_lengths")

        bounds = SemanticTubeBounds(
            source=InclusiveSpan(source_start, source_end),
            target=InclusiveSpan(target_start, target_end),
        )
        row_hidden_states = hidden_states[index]
        for _ in range(samples):
            patch = sample_random_span(
                bounds,
                rng=rng,
                min_span_length=min_span_length,
                max_span_length=max_span_length,
            )
            decomposition = decompose_random_span(row_hidden_states, bounds, patch)
            context_embeddings.append(decomposition.context_embedding)
            patch_embeddings.append(decomposition.patch_embedding)
            weights.append(span_weight(total_length, patch.length, length_adjustment))

    if not context_embeddings:
        zero = hidden_states.sum() * 0.0
        if return_embeddings:
            return zero, None, None, None
        return zero

    context_embeddings = torch.stack(context_embeddings, dim=0)
    patch_embeddings = torch.stack(patch_embeddings, dim=0)
    if predictor is not None:
        context_embeddings = predictor(context_embeddings)

    weights_tensor = torch.tensor(weights, device=hidden_states.device, dtype=hidden_states.dtype)
    if metric == "cosine":
        losses = 1.0 - F.cosine_similarity(context_embeddings, patch_embeddings, dim=-1)
    elif metric == "mse":
        losses = (context_embeddings - patch_embeddings).pow(2).mean(dim=-1)
    elif metric == "l2":
        losses = torch.linalg.vector_norm(context_embeddings - patch_embeddings, dim=-1)
    else:
        raise ValueError("metric must be one of: cosine, mse, l2")

    loss = torch.sum(losses * weights_tensor) / torch.sum(weights_tensor)
    if return_embeddings:
        return loss, context_embeddings, patch_embeddings, weights_tensor
    return loss
