"""One-stage motion-primitive trajectory model for complete sensor trials.

The classifier in this module has one deliberately narrow information path::

    raw trial -> local dynamics -> hard primitive ids --+
             -> boundary / duration / transition -------+
                                                       +-> trajectory GRU

In particular, the trajectory classifier never receives the continuous local
embedding, a quantized code vector, or a mean/max pool of either.  The primary
``primitive_only`` mode receives only straight-through discrete primitive
identity, order, duration,
transitions, and boundaries.  Explicit ``state_only`` and
``primitive_plus_state`` modes exist solely as matched attribution controls;
they prevent a raw-state shortcut from being silently folded into the main
motion-primitive CGCD result.

This file defines model components only.  It intentionally contains no window
classification, InfoNCE, SupCon, or activity-level loss implementation.  A
trainer may apply *trial-level* cross entropy to ``trajectory_logits`` and use
the returned codebook/boundary intermediates for the J0-U/J0-T objectives.

Tensor conventions
------------------
``B`` is the trial batch size, ``T`` the padded raw-sample length, ``C`` the
sensor-channel count, ``L`` the number of local windows, ``K`` the codebook
size, and ``D`` a feature dimension.  Both ``[B,T,C]`` and the USC-HAD
collator's ``[B,C,T]`` layout are accepted and canonicalized to ``[B,T,C]``;
``lengths[b]`` gives the number of valid raw samples in trial ``b``.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


_INTEGER_DTYPES = {
    torch.int8,
    torch.int16,
    torch.int32,
    torch.int64,
    torch.uint8,
}

_INPUT_LAYOUTS = {"AUTO", "BTC", "BCT"}
_PERMUTATION_RELATION_POLICIES = {"carry", "recompute"}
TRAJECTORY_INPUT_MODES = (
    "primitive_only",
    "state_only",
    "primitive_plus_state",
)


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


def _canonicalize_trial_layout(
    trials: torch.Tensor,
    expected_channels: int,
    input_layout: str,
    *,
    name: str,
) -> tuple[torch.Tensor, str]:
    """Return a ``[B,T,C]`` view and the resolved source layout.

    Automatic layout detection deliberately fails when both non-batch axes
    equal the channel count.  Guessing in that case could silently exchange
    time and channels while still satisfying every downstream shape check.
    """

    if not isinstance(trials, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor, got {type(trials).__name__}.")
    if trials.ndim != 3:
        raise ValueError(
            f"{name} must have shape [B,T,C] or [B,C,T], got {tuple(trials.shape)}."
        )
    if not isinstance(input_layout, str):
        raise TypeError("input_layout must be one of 'auto', 'BTC', or 'BCT'.")
    requested = input_layout.strip().upper()
    if requested not in _INPUT_LAYOUTS:
        raise ValueError("input_layout must be one of 'auto', 'BTC', or 'BCT'.")

    second_is_channel = trials.shape[1] == int(expected_channels)
    last_is_channel = trials.shape[2] == int(expected_channels)
    if requested == "AUTO":
        if second_is_channel == last_is_channel:
            if second_is_channel:
                raise ValueError(
                    f"{name} layout is ambiguous because both non-batch axes equal "
                    f"C={expected_channels}; pass input_layout='BTC' or 'BCT'."
                )
            raise ValueError(
                f"{name} has no axis matching C={expected_channels}: "
                f"shape={tuple(trials.shape)}."
            )
        resolved = "BCT" if second_is_channel else "BTC"
    else:
        resolved = requested
        channel_axis = 1 if resolved == "BCT" else 2
        if trials.shape[channel_axis] != int(expected_channels):
            raise ValueError(
                f"{name} declared {resolved} but its channel axis is "
                f"{trials.shape[channel_axis]}, expected {expected_channels}."
            )

    canonical = trials.transpose(1, 2) if resolved == "BCT" else trials
    return canonical.contiguous(), resolved


def _validate_trial_inputs(
    trials: torch.Tensor,
    lengths: torch.Tensor,
    expected_channels: int,
) -> torch.Tensor:
    if not isinstance(trials, torch.Tensor):
        raise TypeError(f"trials must be a torch.Tensor, got {type(trials).__name__}.")
    if trials.ndim != 3:
        raise ValueError(f"trials must have shape [B,T,C], got {tuple(trials.shape)}.")
    if not torch.is_floating_point(trials):
        raise TypeError(f"trials must have floating dtype, got {trials.dtype}.")
    batch_size, padded_length, channels = trials.shape
    if batch_size < 1 or padded_length < 1:
        raise ValueError("trials must contain at least one trial and one sample.")
    if channels != int(expected_channels):
        raise ValueError(
            f"trials channel dimension must be {expected_channels}, got {channels}."
        )
    if not isinstance(lengths, torch.Tensor):
        raise TypeError(f"lengths must be a torch.Tensor, got {type(lengths).__name__}.")
    if lengths.ndim != 1 or tuple(lengths.shape) != (batch_size,):
        raise ValueError(
            f"lengths must have shape ({batch_size},), got {tuple(lengths.shape)}."
        )
    if lengths.dtype not in _INTEGER_DTYPES:
        raise TypeError(f"lengths must have integer dtype, got {lengths.dtype}.")
    checked = lengths.to(device=trials.device, dtype=torch.long)
    if torch.any(checked < 1) or torch.any(checked > padded_length):
        raise ValueError(
            "Every trial length must lie in [1,T]; got "
            f"lengths={checked.detach().cpu().tolist()} for T={padded_length}."
        )
    sample_mask = (
        torch.arange(padded_length, device=trials.device).unsqueeze(0)
        < checked.unsqueeze(1)
    )
    if not bool(torch.isfinite(trials[sample_mask]).all().item()):
        raise ValueError("Valid trial samples contain non-finite values.")
    return checked


def _validate_token_mask(mask: torch.Tensor, shape: tuple[int, int]) -> torch.Tensor:
    if not isinstance(mask, torch.Tensor):
        raise TypeError(f"token_mask must be a torch.Tensor, got {type(mask).__name__}.")
    if tuple(mask.shape) != shape:
        raise ValueError(f"token_mask must have shape {shape}, got {tuple(mask.shape)}.")
    mask = mask.bool()
    if torch.any(mask.sum(dim=1) < 1):
        raise ValueError("Every trial must contain at least one valid token.")
    if shape[1] > 1 and torch.any((~mask[:, :-1]) & mask[:, 1:]):
        raise ValueError("token_mask must be left aligned (valid tokens, then padding).")
    return mask


def _validate_assignment_distribution(
    name: str,
    values: torch.Tensor,
    token_mask: torch.Tensor,
) -> None:
    """Fail closed on malformed soft-token distributions."""

    if not isinstance(values, torch.Tensor) or values.ndim != 3:
        raise ValueError(f"{name} must have shape [B,L,K].")
    if not torch.is_floating_point(values):
        raise TypeError(f"{name} must have floating dtype.")
    if tuple(values.shape[:2]) != tuple(token_mask.shape):
        raise ValueError(f"{name} must align with token_mask [B,L].")
    if values.device != token_mask.device:
        raise ValueError(f"{name} and token_mask must share a device.")
    selected = values[token_mask]
    if not bool(torch.isfinite(selected).all().item()):
        raise ValueError(f"Valid rows in {name} contain non-finite values.")
    if torch.any(selected < 0):
        raise ValueError(f"Valid rows in {name} must be non-negative.")
    row_sums = selected.sum(dim=-1)
    if not torch.allclose(
        row_sums, torch.ones_like(row_sums), atol=1e-5, rtol=1e-5
    ):
        raise ValueError(f"Valid rows in {name} must sum to one.")


@dataclass(frozen=True)
class MotionTrajectoryConfig:
    """Architecture and differentiable quantization configuration."""

    in_channels: int = 6
    frame_size: int = 128
    frame_stride: int = 64
    local_hidden_dim: int = 64
    local_embedding_dim: int = 64
    state_hidden_dim: int = 32
    state_dim: int = 12
    codebook_size: int = 32
    codebook_temperature: float = 0.25
    use_gumbel_training: bool = True
    commitment_beta: float = 0.25
    dead_code_fraction: float = 0.001
    trajectory_input_dim: int = 64
    trajectory_hidden_dim: int = 64
    trajectory_layers: int = 1
    trajectory_dropout: float = 0.0
    trajectory_mask_ratio: float = 0.15
    trajectory_input_mode: str = "primitive_only"
    num_classes: int = 6
    unlabelled_index: int = -1
    epsilon: float = 1e-6

    def validated(self) -> "MotionTrajectoryConfig":
        integer_fields = (
            "in_channels",
            "frame_size",
            "frame_stride",
            "local_hidden_dim",
            "local_embedding_dim",
            "state_hidden_dim",
            "state_dim",
            "codebook_size",
            "trajectory_input_dim",
            "trajectory_hidden_dim",
            "trajectory_layers",
            "num_classes",
        )
        for name in integer_fields:
            _positive_int(name, getattr(self, name))
        if int(self.frame_stride) > int(self.frame_size):
            raise ValueError(
                "frame_stride must not exceed frame_size; gapped frame grids are "
                "not supported."
            )
        if self.codebook_size < 2:
            raise ValueError("codebook_size must be at least two.")
        if not math.isfinite(float(self.codebook_temperature)) or self.codebook_temperature <= 0:
            raise ValueError("codebook_temperature must be positive and finite.")
        if not math.isfinite(float(self.commitment_beta)) or self.commitment_beta < 0:
            raise ValueError("commitment_beta must be finite and non-negative.")
        _probability("dead_code_fraction", self.dead_code_fraction)
        _probability("trajectory_dropout", self.trajectory_dropout, allow_one=False)
        _probability("trajectory_mask_ratio", self.trajectory_mask_ratio, allow_one=False)
        if self.trajectory_input_mode not in TRAJECTORY_INPUT_MODES:
            raise ValueError(
                "trajectory_input_mode must be one of "
                f"{TRAJECTORY_INPUT_MODES}; got {self.trajectory_input_mode!r}."
            )
        if (
            isinstance(self.unlabelled_index, bool)
            or int(self.unlabelled_index) != self.unlabelled_index
            or int(self.unlabelled_index) >= 0
        ):
            raise ValueError("unlabelled_index must be a negative integer.")
        if not math.isfinite(float(self.epsilon)) or self.epsilon <= 0:
            raise ValueError("epsilon must be positive and finite.")
        return self


class LocalTemporalEncoder(nn.Module):
    """Encode masked local raw windows without pooling across a complete trial.

    Per-window channel centering/scaling removes static level from this dynamic
    path.  Static/gravity information remains available only through the
    independent :class:`RawKinematicStateEncoder`.
    """

    def __init__(self, in_channels: int, hidden_dim: int, embedding_dim: int) -> None:
        super().__init__()
        self.in_channels = _positive_int("in_channels", in_channels)
        self.embedding_dim = _positive_int("embedding_dim", embedding_dim)
        hidden_dim = _positive_int("hidden_dim", hidden_dim)
        # The explicit validity channel makes a partial final window
        # distinguishable from genuine zero-valued sensor samples.
        self.network = nn.Sequential(
            nn.Conv1d(self.in_channels + 1, hidden_dim, kernel_size=7, padding=3),
            nn.GroupNorm(1, hidden_dim),
            nn.GELU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=5, padding=2),
            nn.GroupNorm(1, hidden_dim),
            nn.GELU(),
            nn.Conv1d(hidden_dim, self.embedding_dim, kernel_size=3, padding=1),
        )
        self.output_norm = nn.LayerNorm(self.embedding_dim)

    def forward(self, windows: torch.Tensor, sample_mask: torch.Tensor) -> torch.Tensor:
        if windows.ndim != 4:
            raise ValueError(
                f"windows must have shape [B,L,W,C], got {tuple(windows.shape)}."
            )
        if windows.shape[-1] != self.in_channels:
            raise ValueError(
                f"windows must have {self.in_channels} channels, got {windows.shape[-1]}."
            )
        if tuple(sample_mask.shape) != tuple(windows.shape[:3]):
            raise ValueError("sample_mask must match windows [B,L,W].")
        if not torch.is_floating_point(windows):
            raise TypeError("windows must have floating dtype.")
        sample_mask = sample_mask.bool()
        weights = sample_mask.to(windows.dtype).unsqueeze(-1)
        counts = weights.sum(dim=2).clamp_min(1.0)
        mean = (windows * weights).sum(dim=2) / counts
        centred = (windows - mean.unsqueeze(2)) * weights
        variance = centred.square().sum(dim=2) / counts
        standard_deviation = torch.sqrt(variance + 1e-5)
        normalized = centred / standard_deviation.unsqueeze(2).clamp_min(1e-4)

        batch_size, token_count, width, channels = normalized.shape
        flat_mask = sample_mask.reshape(batch_size * token_count, width)
        flat = normalized.reshape(batch_size * token_count, width, channels)
        network_input = torch.cat(
            (flat, flat_mask.to(flat.dtype).unsqueeze(-1)), dim=-1
        ).transpose(1, 2)
        encoded = self.network(network_input)
        encoded = encoded * flat_mask.to(encoded.dtype).unsqueeze(1)
        pooled = encoded.sum(dim=-1) / flat_mask.sum(dim=-1, keepdim=True).clamp_min(1)
        return self.output_norm(pooled).reshape(batch_size, token_count, -1)


class RawKinematicStateEncoder(nn.Module):
    """Independent low-dimensional raw-state path for static/postural cues.

    The deterministic descriptor is ``[mean, log(std), log(rms)]`` per sensor
    channel.  It intentionally contains no learned local-encoder feature and
    no max pool.  The learned MLP only compresses these auditable raw moments.
    """

    def __init__(self, in_channels: int, hidden_dim: int, state_dim: int) -> None:
        super().__init__()
        self.in_channels = _positive_int("in_channels", in_channels)
        self.descriptor_dim = 3 * self.in_channels
        self.state_dim = _positive_int("state_dim", state_dim)
        hidden_dim = _positive_int("hidden_dim", hidden_dim)
        self.projector = nn.Sequential(
            nn.LayerNorm(self.descriptor_dim),
            nn.Linear(self.descriptor_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, self.state_dim),
            nn.LayerNorm(self.state_dim),
        )

    def descriptors(
        self, windows: torch.Tensor, sample_mask: torch.Tensor
    ) -> torch.Tensor:
        if windows.ndim != 4 or windows.shape[-1] != self.in_channels:
            raise ValueError(
                "state windows must have shape "
                f"[B,L,W,{self.in_channels}], got {tuple(windows.shape)}."
            )
        if tuple(sample_mask.shape) != tuple(windows.shape[:3]):
            raise ValueError("state sample_mask must match windows [B,L,W].")
        mask = sample_mask.bool().to(windows.dtype).unsqueeze(-1)
        counts = mask.sum(dim=2).clamp_min(1.0)
        mean = (windows * mask).sum(dim=2) / counts
        centred = (windows - mean.unsqueeze(2)) * mask
        variance = centred.square().sum(dim=2) / counts
        mean_square = (windows.square() * mask).sum(dim=2) / counts
        return torch.cat(
            (
                mean,
                0.5 * torch.log(variance + 1e-6),
                0.5 * torch.log(mean_square + 1e-6),
            ),
            dim=-1,
        )

    def forward(
        self, windows: torch.Tensor, sample_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        descriptor = self.descriptors(windows, sample_mask)
        return self.projector(descriptor), descriptor


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
    entropy = -(soft_usage.clamp_min(1e-12) * soft_usage.clamp_min(1e-12).log()).sum()
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
    """K-way differentiable motion codebook with discrete trajectory export.

    During training the default Gumbel branch uses a straight-through hard
    sample: its forward value is exactly one-hot while its backward derivative
    is taken through the soft sample. This prevents the trajectory encoder
    from using the full posterior as a continuous side channel whose values
    may change even when the exported primitive-id sequence is unchanged.
    """

    def __init__(
        self,
        embedding_dim: int,
        codebook_size: int = 32,
        temperature: float = 0.25,
        use_gumbel_training: bool = True,
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

        scale = 1.0 / math.sqrt(self.embedding_dim)
        self.vectors = nn.Parameter(
            torch.randn(self.codebook_size, self.embedding_dim) * scale
        )

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
        resolved_temperature = self.temperature if temperature is None else float(temperature)
        if not math.isfinite(resolved_temperature) or resolved_temperature <= 0:
            raise ValueError("temperature must be positive and finite.")
        use_hard = (not self.training) if hard is None else bool(hard)

        normalized_embeddings = F.normalize(
            embeddings, dim=-1, eps=self.epsilon
        )
        normalized_vectors = F.normalize(self.vectors, dim=-1, eps=self.epsilon)
        # Squared Euclidean distance on the unit sphere: ||z-c||^2=2-2z.c.
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
            assignments = posterior
            hard_ids = posterior.argmax(dim=-1)

        assignments = assignments.masked_fill(~mask.unsqueeze(-1), 0.0)
        posterior = posterior.masked_fill(~mask.unsqueeze(-1), 0.0)
        logits = logits.masked_fill(~mask.unsqueeze(-1), 0.0)
        distances = distances.masked_fill(~mask.unsqueeze(-1), 0.0)
        hard_ids = hard_ids.masked_fill(~mask, -1)
        quantized = torch.einsum("blk,kd->bld", assignments, normalized_vectors)
        quantized = quantized.masked_fill(~mask.unsqueeze(-1), 0.0)
        # The codebook-fitting VQ term must not update the local encoder through
        # the soft assignment weights.  Detaching only the encoded target is
        # insufficient because ``assignments`` itself depends on embeddings.
        # This parallel projection is numerically identical to ``quantized``
        # but routes gradients exclusively to the code vectors.
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
                    else "soft_posterior_ablation"
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
    if assignments.device != mask.device or boundary_probabilities.device != mask.device:
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
        current = torch.where(mask[:, position], current, torch.zeros_like(current))
        duration_values.append(current)
    duration = torch.stack(duration_values, dim=1)
    lengths = mask.sum(dim=1, keepdim=True).to(duration.dtype)
    duration = (duration / lengths.clamp_min(1.0)).masked_fill(~mask, 0.0)
    return transition, duration


def hard_token_runs(
    hard_tokens: torch.Tensor,
    token_lengths: torch.Tensor,
    *,
    boundary_starts: Optional[torch.Tensor] = None,
) -> list[dict[str, Any]]:
    """Pure JSON-friendly run-length export for a padded hard-token batch.

    A new run begins when the token id changes.  If ``boundary_starts`` is
    supplied, a true value additionally starts a run even when the token id is
    unchanged.  The first valid token always begins a run.
    """

    if not isinstance(hard_tokens, torch.Tensor) or hard_tokens.ndim != 2:
        raise ValueError("hard_tokens must have shape [B,L].")
    if hard_tokens.dtype not in _INTEGER_DTYPES:
        raise TypeError("hard_tokens must have integer dtype.")
    if not isinstance(token_lengths, torch.Tensor) or token_lengths.ndim != 1:
        raise ValueError("token_lengths must have shape [B].")
    if token_lengths.dtype not in _INTEGER_DTYPES:
        raise TypeError("token_lengths must have integer dtype.")
    if len(token_lengths) != hard_tokens.shape[0]:
        raise ValueError("token_lengths batch size differs from hard_tokens.")
    lengths = token_lengths.detach().cpu().long()
    if torch.any(lengths < 1) or torch.any(lengths > hard_tokens.shape[1]):
        raise ValueError("token_lengths contains an out-of-range value.")
    if boundary_starts is not None:
        if not isinstance(boundary_starts, torch.Tensor):
            raise TypeError("boundary_starts must be a torch.Tensor.")
        if tuple(boundary_starts.shape) != tuple(hard_tokens.shape):
            raise ValueError("boundary_starts must match hard_tokens [B,L].")
        boundaries = boundary_starts.detach().cpu().bool()
    else:
        boundaries = torch.zeros_like(hard_tokens, dtype=torch.bool, device="cpu")
    tokens_cpu = hard_tokens.detach().cpu().long()

    exported: list[dict[str, Any]] = []
    for row, length_tensor in enumerate(lengths):
        length = int(length_tensor)
        sequence = tokens_cpu[row, :length].tolist()
        if any(token < 0 for token in sequence):
            raise ValueError("A valid hard-token prefix contains a negative token id.")
        starts = [0]
        for position in range(1, length):
            if sequence[position] != sequence[position - 1] or bool(
                boundaries[row, position]
            ):
                starts.append(position)
        ends = starts[1:] + [length]
        runs = [
            {
                "token": int(sequence[start]),
                "start_token_index": int(start),
                "end_token_index_exclusive": int(end),
                "duration_tokens": int(end - start),
            }
            for start, end in zip(starts, ends)
        ]
        exported.append(
            {
                "token_count": length,
                "tokens": [int(token) for token in sequence],
                "run_count": len(runs),
                "runs": runs,
            }
        )
    return exported


class MotionTrajectoryModel(nn.Module):
    """End-to-end local-token/codebook/trajectory classifier.

    ``trajectory_logits`` is suitable for the supervised J0-T arm.  The J0-U
    arm can ignore it and train from the returned codebook, boundary, and
    representation regularizer intermediates.  No class labels enter forward.
    """

    def __init__(
        self,
        config: Optional[MotionTrajectoryConfig] = None,
        **config_overrides: Any,
    ) -> None:
        super().__init__()
        if config is not None and config_overrides:
            raise ValueError("Pass either config or keyword overrides, not both.")
        self.config = (
            MotionTrajectoryConfig(**config_overrides)
            if config is None
            else config
        ).validated()
        cfg = self.config
        self.local_encoder = LocalTemporalEncoder(
            cfg.in_channels, cfg.local_hidden_dim, cfg.local_embedding_dim
        )
        # Boundary learning uses an independent label-free feature extractor.
        # Sharing the content encoder would let trajectory CE alter boundary
        # values indirectly even when boundary probabilities are detached at
        # classifier ingress.
        self.boundary_encoder = LocalTemporalEncoder(
            cfg.in_channels, cfg.local_hidden_dim, cfg.local_embedding_dim
        )
        self.state_encoder = RawKinematicStateEncoder(
            cfg.in_channels, cfg.state_hidden_dim, cfg.state_dim
        )
        self.codebook = SoftMotionCodebook(
            embedding_dim=cfg.local_embedding_dim,
            codebook_size=cfg.codebook_size,
            temperature=cfg.codebook_temperature,
            use_gumbel_training=cfg.use_gumbel_training,
            commitment_beta=cfg.commitment_beta,
            dead_code_fraction=cfg.dead_code_fraction,
            epsilon=cfg.epsilon,
        )
        self.boundary_head = PairwiseBoundaryHead(
            cfg.local_embedding_dim, hidden_dim=max(8, cfg.local_hidden_dim // 2)
        )
        # J0-U reconstruction objectives.  Targets are returned detached by
        # forward; the trainer remains responsible for choosing loss weights.
        self.token_reconstruction_head = nn.Sequential(
            nn.Linear(cfg.local_embedding_dim, cfg.local_embedding_dim),
            nn.GELU(),
            nn.Linear(cfg.local_embedding_dim, cfg.local_embedding_dim),
        )
        self.codebook_state_decoder = nn.Sequential(
            nn.Linear(cfg.local_embedding_dim, cfg.state_hidden_dim),
            nn.GELU(),
            nn.Linear(cfg.state_hidden_dim, 3 * cfg.in_channels),
        )
        self.next_content_predictor = nn.Sequential(
            nn.Linear(cfg.local_embedding_dim, cfg.local_embedding_dim),
            nn.GELU(),
            nn.Linear(cfg.local_embedding_dim, cfg.local_embedding_dim),
        )

        # The default classifier path is deliberately primitive-centred.  Raw
        # state is available only through an explicitly named attribution arm.
        if cfg.trajectory_input_mode == "primitive_only":
            self.trajectory_feature_dim = cfg.codebook_size + 3
        elif cfg.trajectory_input_mode == "state_only":
            self.trajectory_feature_dim = cfg.state_dim
        else:
            self.trajectory_feature_dim = cfg.codebook_size + cfg.state_dim + 3
        self.trajectory_input = nn.Sequential(
            nn.LayerNorm(self.trajectory_feature_dim),
            nn.Linear(self.trajectory_feature_dim, cfg.trajectory_input_dim),
            nn.GELU(),
        )
        recurrent_dropout = (
            cfg.trajectory_dropout if cfg.trajectory_layers > 1 else 0.0
        )
        self.trajectory_encoder = nn.GRU(
            input_size=cfg.trajectory_input_dim,
            hidden_size=cfg.trajectory_hidden_dim,
            num_layers=cfg.trajectory_layers,
            batch_first=True,
            dropout=recurrent_dropout,
            bidirectional=False,
        )
        self.trajectory_norm = nn.LayerNorm(cfg.trajectory_hidden_dim)
        self.trajectory_classifier = nn.Linear(
            cfg.trajectory_hidden_dim, cfg.num_classes
        )
        # Masked trajectory modelling hides the assignment and compressed
        # state input at exactly the selected positions.  The detached raw
        # physical descriptor remains an untouched reconstruction target.
        # Boundary, duration, and transition variables remain observable.
        self.masked_assignment_token = nn.Parameter(
            torch.zeros(cfg.codebook_size)
        )
        self.masked_state_token = nn.Parameter(torch.zeros(cfg.state_dim))
        self.context_token_head = nn.Linear(
            cfg.trajectory_hidden_dim, cfg.codebook_size
        )
        self.context_state_head = nn.Sequential(
            nn.Linear(cfg.trajectory_hidden_dim, cfg.state_hidden_dim),
            nn.GELU(),
            nn.Linear(cfg.state_hidden_dim, 3 * cfg.in_channels),
        )

    @property
    def codebook_size(self) -> int:
        return int(self.config.codebook_size)

    @staticmethod
    def _validated_permutation(
        token_permutation: Optional[torch.Tensor],
        token_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Validate a per-trial permutation and make padded indices harmless."""

        batch_size, token_count = token_mask.shape
        identity = torch.arange(token_count, device=token_mask.device).expand(
            batch_size, -1
        )
        if token_permutation is None:
            return identity
        if not isinstance(token_permutation, torch.Tensor):
            raise TypeError("token_permutation must be a torch.Tensor.")
        if tuple(token_permutation.shape) != (batch_size, token_count):
            raise ValueError(
                "token_permutation must match token_mask [B,L], got "
                f"{tuple(token_permutation.shape)}."
            )
        if token_permutation.dtype not in _INTEGER_DTYPES:
            raise TypeError("token_permutation must have integer dtype.")
        permutation = token_permutation.to(
            device=token_mask.device, dtype=torch.long
        ).clone()
        for row in range(batch_size):
            length = int(token_mask[row].sum())
            observed = permutation[row, :length]
            expected = torch.arange(length, device=token_mask.device)
            if not torch.equal(torch.sort(observed).values, expected):
                raise ValueError(
                    "Each valid token_permutation prefix must contain every "
                    f"index 0..length-1 exactly once; row={row}, length={length}."
                )
            permutation[row, length:] = identity[row, length:]
        return permutation

    @staticmethod
    def _gather_tokens(values: torch.Tensor, permutation: torch.Tensor) -> torch.Tensor:
        if values.ndim == 2:
            return torch.gather(values, 1, permutation)
        if values.ndim == 3:
            return torch.gather(
                values,
                1,
                permutation.unsqueeze(-1).expand(-1, -1, values.shape[-1]),
            )
        raise ValueError("Only [B,L] and [B,L,D] token tensors can be permuted.")

    def encode_trajectory(
        self,
        assignment_probabilities: torch.Tensor,
        token_mask: torch.Tensor,
        state_features: torch.Tensor,
        boundary_probabilities: torch.Tensor,
        transition_probabilities: torch.Tensor,
        duration_proxy: torch.Tensor,
        *,
        trajectory_mask: Optional[torch.Tensor] = None,
        context_target_assignments: Optional[torch.Tensor] = None,
        apply_random_mask: Optional[bool] = None,
        token_permutation: Optional[torch.Tensor] = None,
        permutation_relations: str = "recompute",
    ) -> dict[str, Any]:
        """Encode only the allowed trajectory variables.

        ``token_permutation`` performs an order ablation at token level; raw
        samples are never shuffled.  The default ``recompute`` policy derives
        transitions and durations from the permuted assignments and zeros the
        learned-boundary channel, preventing original-neighbour relations from
        leaking through the shuffled sequence.  ``carry`` moves all original
        relation scalars with each token and is retained only as an explicitly
        named run-level diagnostic.

        Caller-supplied boundary/transition/duration tensors are detached at
        classifier ingress.  Consequently a trajectory activity label cannot
        train the boundary head as a hidden class-communication channel.
        """

        if assignment_probabilities.ndim != 3:
            raise ValueError("assignment_probabilities must have shape [B,L,K].")
        if not torch.is_floating_point(assignment_probabilities):
            raise TypeError("assignment_probabilities must have floating dtype.")
        batch_size, token_count, codebook_size = assignment_probabilities.shape
        if codebook_size != self.config.codebook_size:
            raise ValueError(
                f"Expected K={self.config.codebook_size}, got K={codebook_size}."
            )
        mask = _validate_token_mask(token_mask, (batch_size, token_count))
        if assignment_probabilities.device != mask.device:
            raise ValueError("trajectory inputs and token_mask must share a device.")
        _validate_assignment_distribution(
            "assignment_probabilities", assignment_probabilities, mask
        )
        if tuple(state_features.shape) != (
            batch_size,
            token_count,
            self.config.state_dim,
        ):
            raise ValueError(
                "state_features must have shape "
                f"[B,L,{self.config.state_dim}]."
            )
        if not torch.is_floating_point(state_features):
            raise TypeError("state_features must have floating dtype.")
        if not bool(torch.isfinite(state_features[mask]).all().item()):
            raise ValueError("Valid state_features contain non-finite values.")
        pair_shape = (batch_size, max(0, token_count - 1))
        if tuple(boundary_probabilities.shape) != pair_shape:
            raise ValueError(
                f"boundary_probabilities must have shape {pair_shape}."
            )
        if tuple(transition_probabilities.shape) != pair_shape:
            raise ValueError(
                f"transition_probabilities must have shape {pair_shape}."
            )
        if tuple(duration_proxy.shape) != (batch_size, token_count):
            raise ValueError("duration_proxy must have shape [B,L].")
        if any(
            not torch.is_floating_point(value)
            for value in (
                boundary_probabilities,
                transition_probabilities,
                duration_proxy,
            )
        ):
            raise TypeError("Boundary, transition, and duration inputs must be floating.")
        tensors = (
            state_features,
            boundary_probabilities,
            transition_probabilities,
            duration_proxy,
        )
        if any(value.device != assignment_probabilities.device for value in tensors):
            raise ValueError("Every trajectory input must share one device.")
        if not all(
            bool(torch.isfinite(value).all().item())
            for value in (
                boundary_probabilities,
                transition_probabilities,
                duration_proxy,
            )
        ):
            raise ValueError("Boundary, transition, and duration inputs must be finite.")
        if not isinstance(permutation_relations, str):
            raise TypeError("permutation_relations must be 'recompute' or 'carry'.")
        relation_policy = permutation_relations.strip().lower()
        if relation_policy not in _PERMUTATION_RELATION_POLICIES:
            raise ValueError("permutation_relations must be 'recompute' or 'carry'.")

        # These three caller-supplied relation paths are values available to
        # the classifier, not gradient routes.  Boundary losses train the
        # boundary branch; activity labels train content/codebook via the
        # assignment path and state path only.
        source_boundary_starts = torch.cat(
            (
                boundary_probabilities.new_zeros((batch_size, 1)),
                boundary_probabilities.detach(),
            ),
            dim=1,
        ).masked_fill(~mask, 0.0)
        source_transition_starts = torch.cat(
            (
                transition_probabilities.new_zeros((batch_size, 1)),
                transition_probabilities.detach(),
            ),
            dim=1,
        ).masked_fill(~mask, 0.0)
        source_durations = duration_proxy.detach().masked_fill(~mask, 0.0)
        permutation = self._validated_permutation(token_permutation, mask)
        assignments = self._gather_tokens(assignment_probabilities, permutation)
        states = self._gather_tokens(state_features, permutation)
        if token_permutation is not None and relation_policy == "recompute":
            zero_boundary_pairs = assignment_probabilities.new_zeros(pair_shape)
            recomputed_transition, durations = soft_transition_and_duration(
                assignments,
                zero_boundary_pairs,
                mask,
            )
            boundary_starts = assignment_probabilities.new_zeros(
                (batch_size, token_count)
            )
            transition_starts = torch.cat(
                (
                    recomputed_transition.new_zeros((batch_size, 1)),
                    recomputed_transition,
                ),
                dim=1,
            ).masked_fill(~mask, 0.0)
            applied_relation_policy = "recompute"
            permutation_boundary_policy = "zero_learned_boundary"
        else:
            boundary_starts = self._gather_tokens(
                source_boundary_starts, permutation
            )
            transition_starts = self._gather_tokens(
                source_transition_starts, permutation
            )
            durations = self._gather_tokens(source_durations, permutation)
            applied_relation_policy = (
                "carry" if token_permutation is not None else "original"
            )
            permutation_boundary_policy = (
                "carry_original"
                if token_permutation is not None
                else "original_sequence"
            )

        if trajectory_mask is not None:
            if not isinstance(trajectory_mask, torch.Tensor):
                raise TypeError("trajectory_mask must be a torch.Tensor.")
            if tuple(trajectory_mask.shape) != (batch_size, token_count):
                raise ValueError("trajectory_mask must match token_mask [B,L].")
            if trajectory_mask.device != mask.device:
                raise ValueError("trajectory_mask and token_mask must share a device.")
            if torch.any(trajectory_mask.bool() & ~mask):
                raise ValueError("trajectory_mask selects a padded token position.")
            selected_mask = self._gather_tokens(
                trajectory_mask.bool(), permutation
            )
        else:
            should_mask = self.training if apply_random_mask is None else bool(
                apply_random_mask
            )
            if should_mask and self.config.trajectory_mask_ratio > 0:
                selected_mask = (
                    torch.rand(
                        (batch_size, token_count),
                        device=assignment_probabilities.device,
                    )
                    < float(self.config.trajectory_mask_ratio)
                ) & mask
                # A J0-U batch must not silently produce a vacuous masked-token
                # objective merely because every Bernoulli draw was false.
                for row in range(batch_size):
                    if not bool(selected_mask[row].any().item()):
                        length = int(mask[row].sum())
                        selected = int(
                            torch.randint(length, (1,), device=mask.device).item()
                        )
                        selected_mask[row, selected] = True
            else:
                selected_mask = torch.zeros_like(mask)

        if context_target_assignments is None:
            target_assignments = assignments.detach()
        else:
            if not isinstance(context_target_assignments, torch.Tensor):
                raise TypeError("context_target_assignments must be a torch.Tensor.")
            if tuple(context_target_assignments.shape) != (
                batch_size,
                token_count,
                codebook_size,
            ):
                raise ValueError(
                    "context_target_assignments must have shape [B,L,K]."
                )
            if context_target_assignments.device != mask.device:
                raise ValueError(
                    "context_target_assignments and token_mask must share a device."
                )
            _validate_assignment_distribution(
                "context_target_assignments", context_target_assignments, mask
            )
            target_assignments = self._gather_tokens(
                context_target_assignments.detach(), permutation
            )

        masked_assignments = torch.where(
            selected_mask.unsqueeze(-1),
            self.masked_assignment_token.view(1, 1, -1).to(assignments.dtype),
            assignments,
        )
        masked_states = torch.where(
            selected_mask.unsqueeze(-1),
            self.masked_state_token.view(1, 1, -1).to(states.dtype),
            states,
        )
        # Transition/duration are functions of the hidden token, while the
        # learned boundary is a function of its continuous local embedding.
        # Keeping them visible at a masked position would leak the target back
        # into the masked-token objective.  Only their encoder-input copies are
        # hidden; the unmodified physical/relational values remain in forward's
        # output for separately supervised losses.
        masked_boundary_starts = boundary_starts.masked_fill(selected_mask, 0.0)
        masked_transition_starts = transition_starts.masked_fill(selected_mask, 0.0)
        masked_durations = durations.masked_fill(selected_mask, 0.0)
        # This concatenation is the only classifier/trajectory-encoder ingress.
        # The selected mode is immutable model configuration and is recorded in
        # every run identity, so a state shortcut cannot be enabled implicitly.
        primitive_features = (
            masked_assignments,
            masked_boundary_starts.unsqueeze(-1),
            masked_transition_starts.unsqueeze(-1),
            masked_durations.unsqueeze(-1),
        )
        if self.config.trajectory_input_mode == "primitive_only":
            selected_features = primitive_features
        elif self.config.trajectory_input_mode == "state_only":
            selected_features = (masked_states,)
        else:
            selected_features = (
                masked_assignments,
                masked_states,
                masked_boundary_starts.unsqueeze(-1),
                masked_transition_starts.unsqueeze(-1),
                masked_durations.unsqueeze(-1),
            )
        trajectory_features = torch.cat(selected_features, dim=-1).masked_fill(
            ~mask.unsqueeze(-1), 0.0
        )
        if trajectory_features.shape[-1] != self.trajectory_feature_dim:
            raise RuntimeError("Unexpected trajectory feature width.")
        recurrent_input = self.trajectory_input(trajectory_features)
        recurrent_input = recurrent_input.masked_fill(~mask.unsqueeze(-1), 0.0)
        frame_lengths = mask.sum(dim=1).long()
        packed = pack_padded_sequence(
            recurrent_input,
            frame_lengths.detach().cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        packed_states, hidden = self.trajectory_encoder(packed)
        sequence_states, _ = pad_packed_sequence(
            packed_states, batch_first=True, total_length=token_count
        )
        sequence_states = sequence_states.masked_fill(~mask.unsqueeze(-1), 0.0)
        trajectory_embedding = self.trajectory_norm(hidden[-1])
        trajectory_logits = self.trajectory_classifier(trajectory_embedding)
        context_token_logits = self.context_token_head(sequence_states)
        context_token_logits = context_token_logits.masked_fill(
            ~mask.unsqueeze(-1), 0.0
        )
        context_state_reconstruction = self.context_state_head(sequence_states)
        context_state_reconstruction = context_state_reconstruction.masked_fill(
            ~mask.unsqueeze(-1), 0.0
        )
        full_context_token_targets = target_assignments.argmax(dim=-1)
        context_token_targets = full_context_token_targets.masked_fill(
            ~selected_mask, int(self.config.unlabelled_index)
        )
        return {
            "trajectory_logits": trajectory_logits,
            "trajectory_embedding": trajectory_embedding,
            "trajectory_sequence_states": sequence_states,
            "trajectory_features": trajectory_features,
            "trajectory_input_mode": self.config.trajectory_input_mode,
            "trajectory_mask": selected_mask,
            "context_token_logits": context_token_logits,
            # The loss-facing target is fail-closed: only deliberately masked
            # positions contain an integer pseudo-token.  All other positions
            # carry the sentinel expected by the one-stage loss contract.
            "context_token_targets": context_token_targets,
            "context_token_full_targets": full_context_token_targets,
            "context_token_loss_mask": selected_mask,
            "context_state_reconstruction": context_state_reconstruction,
            "context_state_loss_mask": selected_mask,
            # Explicit aliases used by the generic masked-state loss adapter.
            "masked_state_predictions": context_state_reconstruction,
            "masked_state_mask": selected_mask,
            "applied_token_permutation": permutation,
            "permuted_state_features": states,
            "masked_state_features": masked_states,
            "masked_boundary_starts": masked_boundary_starts,
            "masked_transition_starts": masked_transition_starts,
            "masked_duration_proxy": masked_durations,
            "permuted_boundary_starts": boundary_starts,
            "permuted_transition_starts": transition_starts,
            "permuted_duration_proxy": durations,
            "permutation_relation_policy": applied_relation_policy,
            "permutation_boundary_policy": permutation_boundary_policy,
            "boundary_class_gradient_policy": "detached_at_trajectory_ingress",
        }

    def _materialize_local_windows(
        self, trials: torch.Tensor, lengths: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        cfg = self.config
        window = int(cfg.frame_size)
        stride = int(cfg.frame_stride)
        token_lengths = 1 + torch.div(
            torch.clamp(lengths - window, min=0) + stride - 1,
            stride,
            rounding_mode="floor",
        )
        maximum_tokens = int(token_lengths.max().item())
        required_samples = window + (maximum_tokens - 1) * stride
        sample_positions = torch.arange(
            trials.shape[1], device=trials.device
        ).unsqueeze(0)
        raw_valid = sample_positions < lengths.unsqueeze(1)
        safe_trials = trials.masked_fill(~raw_valid.unsqueeze(-1), 0.0)
        if required_samples > trials.shape[1]:
            safe_trials = F.pad(
                safe_trials, (0, 0, 0, required_samples - trials.shape[1])
            )
        windows = (
            safe_trials.transpose(1, 2)
            .unfold(2, window, stride)
            .permute(0, 2, 3, 1)
            .contiguous()
        )
        if windows.shape[1] != maximum_tokens:
            windows = windows[:, :maximum_tokens]
        starts = torch.arange(maximum_tokens, device=trials.device) * stride
        offsets = torch.arange(window, device=trials.device)
        window_positions = starts.view(1, -1, 1) + offsets.view(1, 1, -1)
        sample_mask = window_positions < lengths.view(-1, 1, 1)
        token_mask = (
            torch.arange(maximum_tokens, device=trials.device).unsqueeze(0)
            < token_lengths.unsqueeze(1)
        )
        sample_mask = sample_mask & token_mask.unsqueeze(-1)
        if torch.any(sample_mask.any(dim=-1) != token_mask):
            raise RuntimeError("Local-window/token masks are internally inconsistent.")
        return windows, sample_mask, token_mask, token_lengths, starts

    def forward(
        self,
        trials: torch.Tensor,
        lengths: torch.Tensor,
        *,
        state_trials: Optional[torch.Tensor] = None,
        input_layout: str = "auto",
        state_input_layout: Optional[str] = None,
        hard_codebook: Optional[bool] = None,
        codebook_temperature: Optional[float] = None,
        trajectory_mask: Optional[torch.Tensor] = None,
        context_target_assignments: Optional[torch.Tensor] = None,
        apply_random_trajectory_mask: Optional[bool] = None,
        token_permutation: Optional[torch.Tensor] = None,
        permutation_relations: str = "recompute",
    ) -> dict[str, Any]:
        """Encode a padded batch of complete trials.

        Parameters
        ----------
        trials:
            ``[B,T,C]`` or ``[B,C,T]`` model input used by the locally
            normalized dynamic encoder.  It may be fold-normalized.
        lengths:
            Valid raw-sample counts ``[B]``.  They are converted to
            ``frame_lengths`` according to ``frame_size/frame_stride``.
        state_trials:
            Optional signal used only by the independent raw-state path.  Pass
            reconstructed, unstandardized physical-unit trials to retain
            gravity/posture magnitude; when omitted, ``trials`` is used and
            the caller must explicitly treat its normalization as part of the
            experiment identity.
        input_layout:
            ``auto`` (default), ``BTC``, or ``BCT``.  Automatic detection is
            strict and rejects an ambiguous axis pair.
        state_input_layout:
            Optional independent layout declaration for ``state_trials``;
            defaults to ``input_layout``.  In ``auto`` mode each input is
            detected independently.
        token_permutation:
            Optional order-ablation permutation over the already extracted
            frames.  Raw samples are never reordered.
        context_target_assignments:
            Optional detached/EMA-teacher distribution ``[B,L,K]`` for masked
            token prediction.  If omitted, the current deterministic softmax
            posterior is stop-gradient; a trainer should prefer an EMA teacher
            once that teacher is available.
        permutation_relations:
            ``recompute`` is the strict order-shuffle ablation; ``carry`` is a
            separately identified diagnostic that preserves original relation
            scalars while moving tokens.
        """

        trials, resolved_input_layout = _canonicalize_trial_layout(
            trials,
            self.config.in_channels,
            input_layout,
            name="trials",
        )
        lengths = _validate_trial_inputs(
            trials, lengths, self.config.in_channels
        )
        windows, sample_mask, token_mask, frame_lengths, frame_starts = (
            self._materialize_local_windows(trials, lengths)
        )
        if state_trials is None:
            state_windows = windows
            state_sample_mask = sample_mask
            state_source = "shared_model_input"
        else:
            resolved_state_request = (
                input_layout if state_input_layout is None else state_input_layout
            )
            state_trials, resolved_state_layout = _canonicalize_trial_layout(
                state_trials,
                self.config.in_channels,
                resolved_state_request,
                name="state_trials",
            )
            state_lengths = _validate_trial_inputs(
                state_trials, lengths, self.config.in_channels
            )
            if state_trials.device != trials.device:
                raise ValueError("state_trials and trials must share a device.")
            if state_trials.dtype != trials.dtype:
                raise TypeError("state_trials and trials must share a dtype.")
            if tuple(state_trials.shape) != tuple(trials.shape):
                raise ValueError("state_trials and trials must have identical [B,T,C].")
            if not torch.equal(state_lengths, lengths):
                raise RuntimeError("Validated state/model lengths differ.")
            (
                state_windows,
                state_sample_mask,
                state_token_mask,
                state_frame_lengths,
                state_frame_starts,
            ) = self._materialize_local_windows(state_trials, lengths)
            if (
                not torch.equal(state_sample_mask, sample_mask)
                or not torch.equal(state_token_mask, token_mask)
                or not torch.equal(state_frame_lengths, frame_lengths)
                or not torch.equal(state_frame_starts, frame_starts)
            ):
                raise RuntimeError("State/model frame grids differ.")
            state_source = "separate_state_trials"
        if state_trials is None:
            resolved_state_layout = resolved_input_layout
        local_embeddings = self.local_encoder(windows, sample_mask)
        local_embeddings = local_embeddings.masked_fill(
            ~token_mask.unsqueeze(-1), 0.0
        )
        boundary_embeddings = self.boundary_encoder(windows, sample_mask)
        boundary_embeddings = boundary_embeddings.masked_fill(
            ~token_mask.unsqueeze(-1), 0.0
        )
        state_features, state_descriptors = self.state_encoder(
            state_windows, state_sample_mask
        )
        state_features = state_features.masked_fill(
            ~token_mask.unsqueeze(-1), 0.0
        )
        state_descriptors = state_descriptors.masked_fill(
            ~token_mask.unsqueeze(-1), 0.0
        )

        codebook = self.codebook(
            local_embeddings,
            token_mask,
            hard=hard_codebook,
            temperature=codebook_temperature,
        )
        boundary_logits, boundary_probabilities, boundary_pair_mask = (
            self.boundary_head(boundary_embeddings, token_mask)
        )
        transition_probabilities, duration_proxy = soft_transition_and_duration(
            codebook["trajectory_assignments"],
            boundary_probabilities,
            token_mask,
        )

        quantized = codebook["quantized_embeddings"]
        token_reconstruction = F.normalize(
            self.token_reconstruction_head(quantized),
            dim=-1,
            eps=self.config.epsilon,
        ).masked_fill(~token_mask.unsqueeze(-1), 0.0)
        codebook_state_reconstruction = self.codebook_state_decoder(
            quantized
        ).masked_fill(~token_mask.unsqueeze(-1), 0.0)
        if local_embeddings.shape[1] > 1:
            next_content_prediction = F.normalize(
                self.next_content_predictor(quantized[:, :-1]),
                dim=-1,
                eps=self.config.epsilon,
            )
            next_content_target = codebook["normalized_embeddings"][:, 1:].detach()
        else:
            next_content_prediction = local_embeddings.new_zeros(
                (local_embeddings.shape[0], 0, local_embeddings.shape[-1])
            )
            next_content_target = next_content_prediction.detach()

        trajectory = self.encode_trajectory(
            codebook["trajectory_assignments"],
            token_mask,
            state_features,
            boundary_probabilities,
            transition_probabilities,
            duration_proxy,
            trajectory_mask=trajectory_mask,
            context_target_assignments=(
                codebook["assignment_probabilities"]
                if context_target_assignments is None
                else context_target_assignments
            ),
            apply_random_mask=apply_random_trajectory_mask,
            token_permutation=token_permutation,
            permutation_relations=permutation_relations,
        )
        permuted_state_descriptors = self._gather_tokens(
            state_descriptors, trajectory["applied_token_permutation"]
        )
        permuted_codebook_state_reconstruction = self._gather_tokens(
            codebook_state_reconstruction,
            trajectory["applied_token_permutation"],
        )
        context_state_target = permuted_state_descriptors.detach()

        return {
            "resolved_input_layout": resolved_input_layout,
            "resolved_state_input_layout": resolved_state_layout,
            "token_mask": token_mask,
            "frame_mask": token_mask,
            "frame_lengths": frame_lengths,
            # Compatibility aliases use "token" because one frame produces
            # exactly one motion-token assignment.
            "token_lengths": frame_lengths,
            "frame_start_samples": frame_starts,
            "token_start_samples": frame_starts,
            "local_sample_mask": sample_mask,
            "state_sample_mask": state_sample_mask,
            "state_source": state_source,
            # Exposed for boundary/noncollapse diagnostics only; this tensor is
            # provably absent from the classifier ingress above.
            "local_embeddings": local_embeddings,
            "boundary_embeddings": boundary_embeddings,
            "state_features": state_features,
            "state_descriptors": state_descriptors,
            "boundary_logits": boundary_logits,
            "boundary_probabilities": boundary_probabilities,
            "boundary_pair_mask": boundary_pair_mask,
            "transition_probabilities": transition_probabilities,
            "duration_proxy": duration_proxy,
            "token_reconstruction": token_reconstruction,
            "token_reconstruction_target": codebook[
                "normalized_embeddings"
            ].detach(),
            "codebook_state_reconstruction": codebook_state_reconstruction,
            "codebook_state_target": state_descriptors.detach(),
            "next_content_prediction": next_content_prediction,
            "next_content_target": next_content_target,
            # The trainer should intersect this with its fixed raw-stable mask;
            # no learned boundary is silently used as its own target.
            "next_content_pair_mask": boundary_pair_mask,
            "context_state_target": context_state_target,
            "state_targets": context_state_target,
            "masked_state_targets": context_state_target,
            "masked_state_loss_mask": trajectory["context_state_loss_mask"],
            "permuted_state_descriptors": permuted_state_descriptors,
            "permuted_codebook_state_reconstruction": (
                permuted_codebook_state_reconstruction
            ),
            # Direct aliases for OneStageLossInputs.  The VQ pair uses
            # unit-normalized encoder states and a codebook projection whose
            # assignment path is detached, so commitment/codebook gradients
            # remain disjoint under the generic loss composer.
            "encoded_states": codebook["normalized_embeddings"],
            "quantized_states": codebook[
                "codebook_fit_quantized_embeddings"
            ],
            "reconstructed_content": token_reconstruction,
            "content_targets": codebook["normalized_embeddings"].detach(),
            "next_content_predictions": next_content_prediction,
            "next_content_targets": next_content_target,
            "valid_window_mask": token_mask,
            "valid_boundary_mask": boundary_pair_mask,
            "reconstructed_states": permuted_codebook_state_reconstruction,
            "masked_token_logits": trajectory["context_token_logits"],
            "pseudo_token_targets": trajectory["context_token_targets"],
            "masked_trajectory_mask": trajectory["context_token_loss_mask"],
            **codebook,
            **trajectory,
        }

    def export_config(self) -> dict[str, Any]:
        """Return a JSON-serializable architecture identity."""

        return asdict(self.config)


class OneStageMotionTrajectoryModel(MotionTrajectoryModel):
    """Descriptive public name used by the one-stage experiment runner."""

    pass


__all__ = [
    "LocalTemporalEncoder",
    "MotionTrajectoryConfig",
    "MotionTrajectoryModel",
    "OneStageMotionTrajectoryModel",
    "PairwiseBoundaryHead",
    "RawKinematicStateEncoder",
    "SoftMotionCodebook",
    "TRAJECTORY_INPUT_MODES",
    "codebook_usage_diagnostics",
    "hard_token_runs",
    "soft_transition_and_duration",
]
