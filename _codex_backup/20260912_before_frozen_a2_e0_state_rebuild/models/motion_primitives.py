"""Reusable motion-primitive codebook and temporal-boundary components.

This module is deliberately independent of the historical HAPPY and
complete-trial pooling implementations.  It contains only the components
shared by the canonical offline and online motion-primitive trajectory path.
"""

from __future__ import annotations

import math
from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def _positive_int(name: str, value: int) -> int:
    if isinstance(value, bool) or int(value) != value or int(value) < 1:
        raise ValueError(f"{name} must be a positive integer, got {value!r}.")
    return int(value)


def _probability(name: str, value: float, *, allow_one: bool = True) -> float:
    value = float(value)
    upper_ok = value <= 1.0 if allow_one else value < 1.0
    if not math.isfinite(value) or value < 0.0 or not upper_ok:
        bracket = "[0,1]" if allow_one else "[0,1)"
        raise ValueError(f"{name} must be finite and in {bracket}, got {value!r}.")
    return value


def _validate_token_mask(
    mask: torch.Tensor, shape: tuple[int, int]
) -> torch.Tensor:
    if not isinstance(mask, torch.Tensor):
        raise TypeError(
            f"token_mask must be a torch.Tensor, got {type(mask).__name__}."
        )
    if tuple(mask.shape) != shape:
        raise ValueError(
            f"token_mask must have shape {shape}, got {tuple(mask.shape)}."
        )
    mask = mask.bool()
    if torch.any(mask.sum(dim=1) < 1):
        raise ValueError("Every trial must contain at least one valid token.")
    if shape[1] > 1 and torch.any((~mask[:, :-1]) & mask[:, 1:]):
        raise ValueError(
            "token_mask must be left aligned (valid tokens, then padding)."
        )
    return mask


def codebook_usage_diagnostics(
    assignments: torch.Tensor,
    token_mask: torch.Tensor,
    *,
    dead_code_fraction: float = 0.001,
) -> dict[str, torch.Tensor]:
    """Pure, differentiable soft-usage and detached hard-usage diagnostics."""

    if not isinstance(assignments, torch.Tensor) or assignments.ndim != 3:
        raise ValueError("assignments must have shape [B,L,K].")
    if not torch.is_floating_point(assignments):
        raise TypeError("assignments must have floating dtype.")
    mask = _validate_token_mask(token_mask, tuple(assignments.shape[:2]))
    if mask.device != assignments.device:
        raise ValueError("assignments and token_mask must be on the same device.")
    threshold = _probability("dead_code_fraction", dead_code_fraction)
    if torch.any(assignments[mask] < 0):
        raise ValueError("valid assignments must be non-negative.")
    sums = assignments[mask].sum(dim=-1)
    if not torch.allclose(sums, torch.ones_like(sums), atol=1e-5, rtol=1e-5):
        raise ValueError("valid assignment rows must sum to one.")

    soft_usage = assignments[mask].mean(dim=0)
    hard_ids = assignments[mask].argmax(dim=-1)
    hard_counts = torch.bincount(hard_ids, minlength=assignments.shape[-1])
    hard_usage = hard_counts.to(assignments.dtype) / max(1, int(mask.sum()))
    dead = hard_usage <= threshold
    entropy = -(
        soft_usage.clamp_min(1e-12) * soft_usage.clamp_min(1e-12).log()
    ).sum()
    assignment_entropy = -(
        assignments[mask].clamp_min(1e-12)
        * assignments[mask].clamp_min(1e-12).log()
    ).sum(dim=-1).mean()
    uniform_kl = (
        soft_usage.clamp_min(1e-12)
        * (soft_usage.clamp_min(1e-12).log() + math.log(assignments.shape[-1]))
    ).sum()
    return {
        "soft_usage": soft_usage,
        "hard_usage": hard_usage,
        "hard_counts": hard_counts,
        "dead_code_mask": dead,
        "dead_code_count": dead.sum(),
        "usage_entropy": entropy,
        "usage_perplexity": entropy.exp(),
        "assignment_entropy": assignment_entropy,
        "uniform_usage_kl": uniform_kl,
    }


class SoftMotionCodebook(nn.Module):
    """K-way learnable normalized prototype head with discrete trajectory export.

    The deterministic straight-through branch uses exact argmax one-hot values
    in its forward pass and the soft posterior in its backward pass.  Gumbel
    sampling remains an explicit ablation only.
    """

    def __init__(
        self,
        embedding_dim: int,
        codebook_size: int = 32,
        temperature: float = 0.25,
        use_gumbel_training: bool = False,
        commitment_beta: float = 0.25,
        dead_code_fraction: float = 0.001,
        epsilon: float = 1e-6,
    ) -> None:
        super().__init__()
        self.embedding_dim = _positive_int("embedding_dim", embedding_dim)
        self.codebook_size = _positive_int("codebook_size", codebook_size)
        if self.codebook_size < 2:
            raise ValueError("codebook_size must be at least two.")
        self.temperature = float(temperature)
        if not math.isfinite(self.temperature) or self.temperature <= 0:
            raise ValueError("temperature must be positive and finite.")
        self.use_gumbel_training = bool(use_gumbel_training)
        self.commitment_beta = float(commitment_beta)
        if not math.isfinite(self.commitment_beta) or self.commitment_beta < 0:
            raise ValueError("commitment_beta must be finite and non-negative.")
        self.dead_code_fraction = _probability(
            "dead_code_fraction", dead_code_fraction
        )
        self.epsilon = float(epsilon)
        if not math.isfinite(self.epsilon) or self.epsilon <= 0:
            raise ValueError("epsilon must be positive and finite.")

        self.vectors = nn.Parameter(
            torch.empty(self.codebook_size, self.embedding_dim)
        )
        nn.init.trunc_normal_(self.vectors, std=0.02)
        self.register_buffer(
            "ema_cluster_size", torch.zeros(self.codebook_size), persistent=True
        )
        self.register_buffer(
            "ema_vector_sum",
            torch.zeros(self.codebook_size, self.embedding_dim),
            persistent=True,
        )
        self.register_buffer(
            "ema_initialized", torch.tensor(False, dtype=torch.bool), persistent=True
        )

    @torch.no_grad()
    def normalize_learnable_prototypes_(self) -> None:
        """Project learnable prototype rows back to the unit sphere."""

        self.vectors.copy_(
            F.normalize(self.vectors, dim=-1, eps=self.epsilon)
        )

    @torch.no_grad()
    def reset_ema_state(
        self, cluster_sizes: Optional[torch.Tensor] = None
    ) -> None:
        """Synchronize EMA sufficient statistics with the current centres."""

        if cluster_sizes is None:
            counts = torch.ones(
                self.codebook_size,
                device=self.vectors.device,
                dtype=self.vectors.dtype,
            )
        else:
            counts = torch.as_tensor(
                cluster_sizes,
                device=self.vectors.device,
                dtype=self.vectors.dtype,
            )
            if tuple(counts.shape) != (self.codebook_size,):
                raise ValueError("cluster_sizes must have shape [K].")
            if not torch.all(torch.isfinite(counts)) or torch.any(counts <= 0.0):
                raise ValueError(
                    "cluster_sizes must be finite and strictly positive."
                )
        directions = F.normalize(self.vectors.detach(), dim=-1, eps=self.epsilon)
        self.ema_cluster_size.copy_(counts)
        self.ema_vector_sum.copy_(directions * counts.unsqueeze(-1))
        self.ema_initialized.fill_(True)

    def forward(
        self,
        embeddings: torch.Tensor,
        token_mask: torch.Tensor,
        *,
        hard: Optional[bool] = None,
        temperature: Optional[float] = None,
    ) -> dict[str, Any]:
        if not isinstance(embeddings, torch.Tensor) or embeddings.ndim != 3:
            raise ValueError("embeddings must have shape [B,L,D].")
        if not torch.is_floating_point(embeddings):
            raise TypeError("embeddings must have floating dtype.")
        if embeddings.shape[-1] != self.embedding_dim:
            raise ValueError(
                f"embedding dimension must be {self.embedding_dim}, "
                f"got {embeddings.shape[-1]}."
            )
        mask = _validate_token_mask(token_mask, tuple(embeddings.shape[:2]))
        if mask.device != embeddings.device:
            raise ValueError("embeddings and token_mask must be on the same device.")
        resolved_temperature = (
            self.temperature if temperature is None else float(temperature)
        )
        if not math.isfinite(resolved_temperature) or resolved_temperature <= 0:
            raise ValueError("temperature must be positive and finite.")
        use_hard = (not self.training) if hard is None else bool(hard)

        normalized_embeddings = F.normalize(
            embeddings, dim=-1, eps=self.epsilon
        )
        normalized_vectors = F.normalize(self.vectors, dim=-1, eps=self.epsilon)
        distances = 2.0 - 2.0 * torch.einsum(
            "bld,kd->blk", normalized_embeddings, normalized_vectors
        )
        unscaled_logits = -distances
        logits = unscaled_logits / resolved_temperature
        posterior = torch.softmax(logits, dim=-1)
        if use_hard:
            hard_ids = posterior.argmax(dim=-1)
            assignments = F.one_hot(
                hard_ids, num_classes=self.codebook_size
            ).to(posterior.dtype)
        elif self.use_gumbel_training:
            assignments = F.gumbel_softmax(
                unscaled_logits,
                tau=resolved_temperature,
                hard=True,
                dim=-1,
            )
            hard_ids = assignments.argmax(dim=-1)
        else:
            hard_ids = posterior.argmax(dim=-1)
            hard_assignments = F.one_hot(
                hard_ids, num_classes=self.codebook_size
            ).to(posterior.dtype)
            assignments = posterior + (hard_assignments - posterior).detach()

        assignments = assignments.masked_fill(~mask.unsqueeze(-1), 0.0)
        posterior = posterior.masked_fill(~mask.unsqueeze(-1), 0.0)
        logits = logits.masked_fill(~mask.unsqueeze(-1), 0.0)
        distances = distances.masked_fill(~mask.unsqueeze(-1), 0.0)
        hard_ids = hard_ids.masked_fill(~mask, -1)
        quantized = torch.einsum("blk,kd->bld", assignments, normalized_vectors)
        quantized = quantized.masked_fill(~mask.unsqueeze(-1), 0.0)
        codebook_fit_quantized = torch.einsum(
            "blk,kd->bld", assignments.detach(), normalized_vectors
        ).masked_fill(~mask.unsqueeze(-1), 0.0)

        commitment_per_token = (
            normalized_embeddings - quantized.detach()
        ).square().mean(dim=-1)
        embedding_per_token = (
            normalized_embeddings.detach() - codebook_fit_quantized
        ).square().mean(dim=-1)
        commitment = commitment_per_token[mask].mean()
        embedding_loss = embedding_per_token[mask].mean()
        objective = embedding_loss + self.commitment_beta * commitment
        diagnostics = codebook_usage_diagnostics(
            assignments,
            mask,
            dead_code_fraction=self.dead_code_fraction,
        )
        return {
            "trajectory_assignment_forward": (
                "hard_one_hot_inference"
                if use_hard
                else (
                    "hard_one_hot_straight_through_gumbel"
                    if self.use_gumbel_training
                    else "hard_one_hot_straight_through_deterministic"
                )
            ),
            "assignment_logits": logits,
            "assignment_probabilities": posterior,
            "trajectory_assignments": assignments,
            "hard_tokens": hard_ids,
            "quantized_embeddings": quantized,
            "codebook_fit_quantized_embeddings": codebook_fit_quantized,
            "normalized_embeddings": normalized_embeddings.masked_fill(
                ~mask.unsqueeze(-1), 0.0
            ),
            "normalized_codebook": normalized_vectors,
            "distances": distances,
            "commitment_loss": commitment,
            "codebook_embedding_loss": embedding_loss,
            "codebook_objective": objective,
            **diagnostics,
        }

    @torch.no_grad()
    def ema_update(
        self,
        embeddings: torch.Tensor,
        token_mask: torch.Tensor,
        hard_ids: torch.Tensor,
        *,
        decay: float,
    ) -> list[int]:
        """Update active centres by trial-balanced exponential moving averages."""

        if embeddings.ndim != 3 or embeddings.shape[-1] != self.embedding_dim:
            raise ValueError("embeddings must have shape [B,L,D].")
        mask = _validate_token_mask(token_mask, tuple(embeddings.shape[:2]))
        if tuple(hard_ids.shape) != tuple(mask.shape):
            raise ValueError("hard_ids must align with token_mask [B,L].")
        if hard_ids.device != embeddings.device or mask.device != embeddings.device:
            raise ValueError("EMA inputs must share a device.")
        decay = float(decay)
        if not math.isfinite(decay) or not 0.0 <= decay < 1.0:
            raise ValueError("decay must lie in [0,1).")
        valid_ids = hard_ids[mask]
        if torch.any(valid_ids < 0) or torch.any(valid_ids >= self.codebook_size):
            raise ValueError("hard_ids contains an invalid code on a valid position.")
        normalized = F.normalize(embeddings.detach(), dim=-1, eps=self.epsilon)
        per_window_weight = mask.to(normalized.dtype) / mask.sum(
            dim=1, keepdim=True
        ).clamp_min(1).to(normalized.dtype)
        if not bool(self.ema_initialized.item()):
            self.reset_ema_state()
        flat_ids = hard_ids[mask]
        flat_weights = per_window_weight[mask]
        flat_embeddings = normalized[mask]
        batch_counts = torch.zeros_like(self.ema_cluster_size)
        batch_counts.scatter_add_(0, flat_ids, flat_weights)
        batch_sums = torch.zeros_like(self.ema_vector_sum)
        batch_sums.index_add_(
            0, flat_ids, flat_embeddings * flat_weights.unsqueeze(-1)
        )
        self.ema_cluster_size.mul_(decay).add_(batch_counts, alpha=1.0 - decay)
        self.ema_vector_sum.mul_(decay).add_(batch_sums, alpha=1.0 - decay)
        typical_norm = self.vectors.detach().norm(dim=-1).median().clamp_min(
            self.epsilon
        )
        supported = self.ema_cluster_size > self.epsilon
        centres = self.ema_vector_sum[supported] / self.ema_cluster_size[
            supported
        ].unsqueeze(-1)
        self.vectors[supported] = F.normalize(
            centres, dim=-1, eps=self.epsilon
        ) * typical_norm
        return [
            int(index)
            for index in torch.nonzero(batch_counts > 0.0, as_tuple=False)
            .flatten()
            .detach()
            .cpu()
            .tolist()
        ]

    @torch.no_grad()
    def revive_unused_codes(
        self,
        usage_counts: torch.Tensor,
        candidate_embeddings: torch.Tensor,
    ) -> list[int]:
        """Reinitialise only exactly unused codes from difficult examples."""

        counts = torch.as_tensor(usage_counts, device=self.vectors.device)
        if tuple(counts.shape) != (self.codebook_size,):
            raise ValueError("usage_counts must have shape [K].")
        candidates = torch.as_tensor(
            candidate_embeddings,
            device=self.vectors.device,
            dtype=self.vectors.dtype,
        )
        if candidates.ndim != 2 or candidates.shape[1] != self.embedding_dim:
            raise ValueError("candidate_embeddings must have shape [N,D].")
        if not len(candidates):
            return []
        candidates = F.normalize(candidates, dim=-1, eps=self.epsilon)
        dead = torch.nonzero(counts <= 0, as_tuple=False).flatten()
        if not len(dead):
            return []
        active = torch.nonzero(counts > 0, as_tuple=False).flatten()
        reference = (
            F.normalize(self.vectors[active], dim=-1, eps=self.epsilon)
            if len(active)
            else candidates[:1]
        )
        selected: list[torch.Tensor] = []
        available = torch.ones(
            len(candidates), dtype=torch.bool, device=candidates.device
        )
        for _ in range(min(len(dead), len(candidates))):
            distance = 1.0 - candidates @ reference.transpose(0, 1)
            nearest = distance.min(dim=1).values.masked_fill(~available, -1.0)
            index = int(nearest.argmax().item())
            chosen = candidates[index : index + 1]
            selected.append(chosen.squeeze(0))
            reference = torch.cat((reference, chosen), dim=0)
            available[index] = False
        if not selected:
            return []
        revived = dead[: len(selected)]
        replacement = torch.stack(selected, dim=0)
        typical_norm = self.vectors.detach().norm(dim=-1).median().clamp_min(
            self.epsilon
        )
        self.vectors[revived] = replacement * typical_norm
        if bool(self.ema_initialized.item()):
            reference_counts = self.ema_cluster_size[active]
            reset_count = (
                reference_counts.median().clamp_min(self.epsilon)
                if len(reference_counts)
                else self.ema_cluster_size.new_tensor(1.0)
            )
            self.ema_cluster_size[revived] = reset_count
            self.ema_vector_sum[revived] = replacement * reset_count
        return [int(index) for index in revived.detach().cpu().tolist()]


class PairwiseBoundaryHead(nn.Module):
    """Predict a boundary between adjacent continuous local embeddings."""

    def __init__(self, embedding_dim: int, hidden_dim: int = 32) -> None:
        super().__init__()
        embedding_dim = _positive_int("embedding_dim", embedding_dim)
        hidden_dim = _positive_int("hidden_dim", hidden_dim)
        self.network = nn.Sequential(
            nn.LayerNorm(2 * embedding_dim),
            nn.Linear(2 * embedding_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self, embeddings: torch.Tensor, token_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if embeddings.ndim != 3:
            raise ValueError("boundary embeddings must have shape [B,L,D].")
        mask = _validate_token_mask(token_mask, tuple(embeddings.shape[:2]))
        pair_mask = mask[:, :-1] & mask[:, 1:]
        if embeddings.shape[1] == 1:
            logits = embeddings.new_zeros((embeddings.shape[0], 0))
        else:
            pair_features = torch.cat(
                (
                    torch.abs(embeddings[:, 1:] - embeddings[:, :-1]),
                    embeddings[:, 1:] * embeddings[:, :-1],
                ),
                dim=-1,
            )
            logits = self.network(pair_features).squeeze(-1)
            logits = logits.masked_fill(~pair_mask, 0.0)
        probabilities = torch.sigmoid(logits).masked_fill(~pair_mask, 0.0)
        return logits, probabilities, pair_mask


def soft_transition_and_duration(
    assignments: torch.Tensor,
    boundary_probabilities: torch.Tensor,
    token_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return differentiable transition probability and causal run duration."""

    if assignments.ndim != 3:
        raise ValueError("assignments must have shape [B,L,K].")
    shape = tuple(assignments.shape[:2])
    mask = _validate_token_mask(token_mask, shape)
    expected_boundary_shape = (shape[0], max(0, shape[1] - 1))
    if tuple(boundary_probabilities.shape) != expected_boundary_shape:
        raise ValueError(
            "boundary_probabilities must have shape "
            f"{expected_boundary_shape}, got {tuple(boundary_probabilities.shape)}."
        )
    if (
        assignments.device != mask.device
        or boundary_probabilities.device != mask.device
    ):
        raise ValueError("transition inputs must share a device.")
    pair_mask = mask[:, :-1] & mask[:, 1:]
    transition = assignments.new_zeros(expected_boundary_shape)
    if shape[1] > 1:
        same = (assignments[:, 1:] * assignments[:, :-1]).sum(dim=-1)
        transition = (1.0 - same).clamp(0.0, 1.0)
    transition = transition.masked_fill(~pair_mask, 0.0)

    duration_values: list[torch.Tensor] = []
    current = assignments.new_ones(shape[0])
    duration_values.append(current)
    for position in range(1, shape[1]):
        continuation = (
            (1.0 - transition[:, position - 1])
            * (1.0 - boundary_probabilities[:, position - 1])
        ).clamp(0.0, 1.0)
        current = 1.0 + continuation * current
        current = torch.where(
            mask[:, position], current, torch.zeros_like(current)
        )
        duration_values.append(current)
    duration = torch.stack(duration_values, dim=1)
    lengths = mask.sum(dim=1, keepdim=True).to(duration.dtype)
    duration = (duration / lengths.clamp_min(1.0)).masked_fill(~mask, 0.0)
    return transition, duration


__all__ = (
    "PairwiseBoundaryHead",
    "SoftMotionCodebook",
    "codebook_usage_diagnostics",
    "soft_transition_and_duration",
)
