"""Trajectory-only motion-primitive model for wearable-sensor HAR-CGCD.

This canonical module is independent of complete-trial pooling and image
classification heads. A trial remains an ordered bag of local windows until
learned boundaries compress it into a variable-length motion-primitive run
sequence; only that sequence is classified.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Mapping, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence

from models.motion_primitives import (
    PairwiseBoundaryHead,
    SoftMotionCodebook,
    soft_transition_and_duration,
)
from models.resnet1d import ResNet1D



@dataclass(frozen=True)
class MotionPrimitiveConfig:
    """Architecture of the trajectory-only motion-primitive model."""

    in_channels: int = 6
    window_size: int = 256
    window_stride: int = 128
    tail_policy: str = "drop"
    feature_dim: int = 256
    base_channels: int = 64
    backbone_dropout: float = 0.0
    old_class_count: int = 6
    codebook_size: int = 32
    codebook_temperature: float = 0.25
    trajectory_input_dim: int = 128
    trajectory_hidden_dim: int = 128
    trajectory_layers: int = 1
    trajectory_dropout: float = 0.0
    run_state_dim: int = 12
    boundary_threshold: float = 0.50
    boundary_initial_bias: float = -1.50
    epsilon: float = 1.0e-6

    def validated(self) -> "MotionPrimitiveConfig":
        integer_fields = (
            "in_channels",
            "window_size",
            "window_stride",
            "feature_dim",
            "base_channels",
            "old_class_count",
            "codebook_size",
            "trajectory_input_dim",
            "trajectory_hidden_dim",
            "trajectory_layers",
        )
        for name in integer_fields:
            value = getattr(self, name)
            if isinstance(value, bool) or int(value) != value or int(value) < 1:
                raise ValueError(
                    f"{name} must be a positive integer, got {value!r}."
                )
        if int(self.codebook_size) < 2:
            raise ValueError("codebook_size must be at least two.")
        if (
            isinstance(self.run_state_dim, bool)
            or int(self.run_state_dim) != self.run_state_dim
            or int(self.run_state_dim) < 0
        ):
            raise ValueError("run_state_dim must be a non-negative integer.")
        if self.tail_policy not in {"drop", "pad"}:
            raise ValueError("tail_policy must be 'drop' or 'pad'.")
        for name in ("backbone_dropout", "trajectory_dropout"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or not 0.0 <= value < 1.0:
                raise ValueError(f"{name} must lie in [0, 1).")
        if (
            not math.isfinite(float(self.codebook_temperature))
            or float(self.codebook_temperature) <= 0.0
        ):
            raise ValueError("codebook_temperature must be positive and finite.")
        if not 0.0 <= float(self.boundary_threshold) <= 1.0:
            raise ValueError("boundary_threshold must lie in [0, 1].")
        if not math.isfinite(float(self.boundary_initial_bias)):
            raise ValueError("boundary_initial_bias must be finite.")
        if not math.isfinite(float(self.epsilon)) or float(self.epsilon) <= 0.0:
            raise ValueError("epsilon must be positive and finite.")
        return self

    @property
    def run_feature_dim(self) -> int:
        # Fixed-dimensional codebook embedding + boundary/transition/duration +
        # relative/sine/cosine position + projected physical state.
        return int(self.feature_dim) + 6 + int(self.run_state_dim)

    def audit_dict(self) -> dict[str, object]:
        return asdict(self)

def _validate_trial_bag(
    batch: Mapping[str, torch.Tensor], *, expected_channels: int, expected_window: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    required = {"windows", "positions", "mask", "lengths"}
    missing = required - set(batch)
    if missing:
        raise KeyError(f"Trial batch is missing keys {sorted(missing)}.")
    windows = batch["windows"]
    positions = batch["positions"]
    mask = batch["mask"].bool()
    lengths = batch["lengths"].long()
    if windows.ndim != 4:
        raise ValueError(f"windows must be [B,L,C,W], got {tuple(windows.shape)}.")
    if tuple(windows.shape[2:]) != (int(expected_channels), int(expected_window)):
        raise ValueError(
            "Window shape differs from the configured USC-HAD input: "
            f"got {tuple(windows.shape[2:])}, expected "
            f"({expected_channels}, {expected_window})."
        )
    if tuple(mask.shape) != tuple(windows.shape[:2]):
        raise ValueError("mask must match windows [B,L].")
    if tuple(positions.shape[:2]) != tuple(windows.shape[:2]):
        raise ValueError("positions must match windows [B,L].")
    if tuple(lengths.shape) != (windows.shape[0],):
        raise ValueError("lengths must have shape [B].")
    if not torch.equal(mask.sum(dim=1), lengths):
        raise ValueError("lengths and mask contain different valid-window counts.")
    if windows.shape[1] > 1 and torch.any((~mask[:, :-1]) & mask[:, 1:]):
        raise ValueError("mask must be left aligned.")
    if torch.any(lengths <= 0):
        raise ValueError("Every trial must contain at least one window.")
    return windows, positions, mask, lengths

def _forward_local_encoder(
    window_encoder: ResNet1D,
    batch: Mapping[str, torch.Tensor],
    config: MotionPrimitiveConfig,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Encode ordered windows without invoking a complete-trial pool."""

    windows, positions, mask, _ = _validate_trial_bag(
        batch,
        expected_channels=config.in_channels,
        expected_window=config.window_size,
    )
    valid_features = window_encoder(windows[mask])
    if valid_features.ndim != 2 or valid_features.shape[1] != config.feature_dim:
        raise RuntimeError(
            f"Window encoder returned {tuple(valid_features.shape)}, expected [N,D]."
        )
    local_features = valid_features.new_zeros(
        windows.shape[0], windows.shape[1], config.feature_dim
    )
    local_features[mask] = valid_features
    return local_features, positions, mask


def _soft_parent_duration(
    final_boundary_probabilities: torch.Tensor,
    token_mask: torch.Tensor,
) -> torch.Tensor:
    """Causal normalized duration governed only by parent-run boundaries.

    Child-code changes may be merged into one parent primitive, so they must
    not silently reset the duration supplied to the run-level classifier.
    """

    if token_mask.ndim != 2:
        raise ValueError("token_mask must have shape [B,L].")
    expected = (token_mask.shape[0], max(0, token_mask.shape[1] - 1))
    if tuple(final_boundary_probabilities.shape) != expected:
        raise ValueError(
            f"final_boundary_probabilities must have shape {expected}."
        )
    mask = token_mask.bool()
    if torch.any(mask.sum(dim=1) < 1):
        raise ValueError("Every trial must contain at least one valid token.")
    values: list[torch.Tensor] = []
    current = final_boundary_probabilities.new_ones(mask.shape[0])
    values.append(current)
    for position in range(1, mask.shape[1]):
        continuation = 1.0 - final_boundary_probabilities[:, position - 1]
        current = 1.0 + continuation * current
        current = torch.where(mask[:, position], current, torch.zeros_like(current))
        values.append(current)
    duration = torch.stack(values, dim=1)
    lengths = mask.sum(dim=1, keepdim=True).to(duration.dtype)
    return (duration / lengths.clamp_min(1.0)).masked_fill(~mask, 0.0)


def _window_physical_descriptors(
    windows: torch.Tensor,
    token_mask: torch.Tensor,
    *,
    epsilon: float,
) -> torch.Tensor:
    """Return gravity/amplitude-sensitive moments for every ordered window.

    The input has already been normalized with fold-training statistics, not
    independently standardized per window.  Channel means therefore retain
    static posture information needed to distinguish activities such as sit
    and stand.  No activity label enters this computation.
    """

    if windows.ndim != 4:
        raise ValueError("windows must have shape [B,L,C,W].")
    if tuple(token_mask.shape) != tuple(windows.shape[:2]):
        raise ValueError("token_mask must align with windows [B,L].")
    if not math.isfinite(float(epsilon)) or float(epsilon) <= 0.0:
        raise ValueError("epsilon must be positive and finite.")
    mean = windows.mean(dim=-1)
    variance = windows.var(dim=-1, unbiased=False)
    mean_square = windows.square().mean(dim=-1)
    descriptors = torch.cat(
        (
            mean,
            0.5 * torch.log(variance + float(epsilon)),
            0.5 * torch.log(mean_square + float(epsilon)),
        ),
        dim=-1,
    )
    return descriptors.masked_fill(~token_mask.bool().unsqueeze(-1), 0.0)


def _compress_motion_runs(
    assignments: torch.Tensor,
    hard_tokens: torch.Tensor,
    token_mask: torch.Tensor,
    hard_boundary_starts: torch.Tensor,
    learned_boundary_probabilities: torch.Tensor,
    token_change_probabilities: torch.Tensor,
    final_boundary_probabilities: torch.Tensor,
    duration_proxy: torch.Tensor,
    position_features: torch.Tensor,
    physical_state_features: torch.Tensor,
    normalized_codebook: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Compress window tokens into padded, differentiable primitive runs.

    Run membership is discrete in the forward pass: a run starts at the first
    valid window or at a thresholded *fused* boundary.  A hard child-code
    change is evidence for a boundary, but it is not itself a compulsory run
    start; the learned boundary logit may therefore merge several child codes
    into one parent run.  Conversely, the learned logit may split two adjacent
    windows even when their hard child code is unchanged.  The values presented
    to the trajectory encoder remain differentiable:

    * the run code is the mean straight-through assignment inside the run and
      may consequently be a distribution over several child codes;
    * duration has the exact hard run fraction in the forward pass and the
      causal soft-duration gradient in the backward pass;
    * an entering boundary retains the differentiable fused confidence;
    * entering transition is the differentiable code-change probability.

    Consequently the GRU sees a genuinely shorter, variable-length sequence
    while trajectory supervision can still update the local encoder, codebook
    and boundary head.  Only the integer choice of where a run begins is
    non-differentiable, as is standard for straight-through segmentation.
    """

    if assignments.ndim != 3:
        raise ValueError("assignments must have shape [B,L,K].")
    batch_size, token_count, codebook_size = assignments.shape
    expected_token_shape = (batch_size, token_count)
    if tuple(hard_tokens.shape) != expected_token_shape:
        raise ValueError("hard_tokens must align with assignments [B,L].")
    if tuple(token_mask.shape) != expected_token_shape:
        raise ValueError("token_mask must align with assignments [B,L].")
    if tuple(hard_boundary_starts.shape) != expected_token_shape:
        raise ValueError("hard_boundary_starts must align with assignments [B,L].")
    if tuple(duration_proxy.shape) != expected_token_shape:
        raise ValueError("duration_proxy must align with assignments [B,L].")
    if position_features.ndim != 3 or tuple(position_features.shape[:2]) != expected_token_shape:
        raise ValueError("position_features must align with assignments [B,L,P].")
    if physical_state_features.ndim != 3 or tuple(
        physical_state_features.shape[:2]
    ) != expected_token_shape:
        raise ValueError(
            "physical_state_features must align with assignments [B,L,S]."
        )
    expected_pair_shape = (batch_size, max(0, token_count - 1))
    for name, value in (
        ("learned_boundary_probabilities", learned_boundary_probabilities),
        ("token_change_probabilities", token_change_probabilities),
        ("final_boundary_probabilities", final_boundary_probabilities),
    ):
        if tuple(value.shape) != expected_pair_shape:
            raise ValueError(f"{name} must have shape {expected_pair_shape}.")
    if normalized_codebook.ndim != 2:
        raise ValueError("normalized_codebook must have shape [K,D].")
    if normalized_codebook.shape[0] != codebook_size:
        raise ValueError("normalized_codebook K differs from assignments.")
    devices = {
        assignments.device,
        hard_tokens.device,
        token_mask.device,
        hard_boundary_starts.device,
        learned_boundary_probabilities.device,
        token_change_probabilities.device,
        final_boundary_probabilities.device,
        duration_proxy.device,
        position_features.device,
        physical_state_features.device,
        normalized_codebook.device,
    }
    if len(devices) != 1:
        raise ValueError("All run-compression tensors must share one device.")

    mask = token_mask.bool()
    starts_mask = hard_boundary_starts.bool() & mask
    if torch.any(mask.sum(dim=1) < 1):
        raise ValueError("Every trial must contain at least one valid token.")
    if token_count > 1 and torch.any((~mask[:, :-1]) & mask[:, 1:]):
        raise ValueError("token_mask must be a contiguous valid prefix.")
    if not torch.all(starts_mask[:, 0]):
        raise ValueError("Every run sequence must start at its first token.")
    feature_rows: list[torch.Tensor] = []
    assignment_rows: list[torch.Tensor] = []
    codebook_rows: list[torch.Tensor] = []
    duration_rows: list[torch.Tensor] = []
    boundary_rows: list[torch.Tensor] = []
    transition_rows: list[torch.Tensor] = []
    position_rows: list[torch.Tensor] = []
    state_rows: list[torch.Tensor] = []
    run_lengths: list[int] = []

    for batch_index in range(batch_size):
        valid_length = int(mask[batch_index].sum().item())
        starts = torch.nonzero(
            starts_mask[batch_index, :valid_length], as_tuple=False
        ).flatten()
        if len(starts) < 1 or int(starts[0].item()) != 0:
            raise RuntimeError("Invalid hard run starts.")
        ends = torch.cat(
            (
                starts[1:],
                starts.new_tensor([valid_length]),
            )
        )
        row_assignments: list[torch.Tensor] = []
        row_durations: list[torch.Tensor] = []
        row_boundaries: list[torch.Tensor] = []
        row_transitions: list[torch.Tensor] = []
        row_positions: list[torch.Tensor] = []
        row_states: list[torch.Tensor] = []
        for run_index, (start_tensor, end_tensor) in enumerate(zip(starts, ends)):
            start = int(start_tensor.item())
            end = int(end_tensor.item())
            if not 0 <= start < end <= valid_length:
                raise RuntimeError("Run boundaries do not form a valid partition.")

            # ``assignments`` is hard-one-hot in the forward pass and soft in
            # the backward pass during training, so this is both an exact code
            # identity and a differentiable run representation.
            run_assignment = assignments[batch_index, start:end].mean(dim=0)
            row_assignments.append(run_assignment)
            row_positions.append(position_features[batch_index, start:end].mean(dim=0))
            row_states.append(
                physical_state_features[batch_index, start:end].mean(dim=0)
            )

            hard_duration = assignments.new_tensor(
                float(end - start) / float(valid_length)
            )
            soft_duration = duration_proxy[batch_index, end - 1]
            row_durations.append(
                hard_duration + soft_duration - soft_duration.detach()
            )

            if run_index == 0:
                # The first run has no predecessor.  Preserve a graph-zero so
                # tensor assembly stays uniform without inventing an edge.
                graph_zero = run_assignment.sum() * 0.0
                row_boundaries.append(graph_zero)
                row_transitions.append(graph_zero)
            else:
                pair_index = start - 1
                # The fused probability is already differentiable with
                # respect to both the learned boundary head and the child-code
                # assignments.  Do not replace it with a hard OR surrogate:
                # doing so would again make a child-code change compulsory.
                row_boundaries.append(
                    final_boundary_probabilities[batch_index, pair_index]
                )
                row_transitions.append(
                    token_change_probabilities[batch_index, pair_index]
                )

        row_assignment_tensor = torch.stack(row_assignments, dim=0)
        row_duration_tensor = torch.stack(row_durations, dim=0)
        row_boundary_tensor = torch.stack(row_boundaries, dim=0)
        row_transition_tensor = torch.stack(row_transitions, dim=0)
        row_position_tensor = torch.stack(row_positions, dim=0)
        row_state_tensor = torch.stack(row_states, dim=0)
        # Keep the trajectory coordinate system independent of active K.
        # Discrete ids remain available for VQ and diagnostics, while the GRU
        # reads their fixed-dimensional codebook embeddings.
        row_codebook_tensor = row_assignment_tensor @ normalized_codebook
        row_features = torch.cat(
            (
                row_codebook_tensor,
                row_boundary_tensor.unsqueeze(-1),
                row_transition_tensor.unsqueeze(-1),
                row_duration_tensor.unsqueeze(-1),
                row_position_tensor,
                row_state_tensor,
            ),
            dim=-1,
        )
        feature_rows.append(row_features)
        assignment_rows.append(row_assignment_tensor)
        codebook_rows.append(row_codebook_tensor)
        duration_rows.append(row_duration_tensor)
        boundary_rows.append(row_boundary_tensor)
        transition_rows.append(row_transition_tensor)
        position_rows.append(row_position_tensor)
        state_rows.append(row_state_tensor)
        run_lengths.append(len(starts))

    padded_features = pad_sequence(feature_rows, batch_first=True)
    padded_assignments = pad_sequence(assignment_rows, batch_first=True)
    padded_codebook = pad_sequence(codebook_rows, batch_first=True)
    padded_durations = pad_sequence(duration_rows, batch_first=True)
    padded_boundaries = pad_sequence(boundary_rows, batch_first=True)
    padded_transitions = pad_sequence(transition_rows, batch_first=True)
    padded_positions = pad_sequence(position_rows, batch_first=True)
    padded_states = pad_sequence(state_rows, batch_first=True)
    lengths_tensor = torch.as_tensor(
        run_lengths, device=assignments.device, dtype=torch.long
    )
    run_mask = (
        torch.arange(padded_features.shape[1], device=assignments.device)
        .unsqueeze(0)
        .lt(lengths_tensor.unsqueeze(1))
    )
    return {
        "run_features": padded_features.masked_fill(
            ~run_mask.unsqueeze(-1), 0.0
        ),
        "run_assignment_distributions": padded_assignments.masked_fill(
            ~run_mask.unsqueeze(-1), 0.0
        ),
        "run_codebook_embeddings": padded_codebook.masked_fill(
            ~run_mask.unsqueeze(-1), 0.0
        ),
        "run_durations_normalized": padded_durations.masked_fill(~run_mask, 0.0),
        "run_enter_boundary": padded_boundaries.masked_fill(~run_mask, 0.0),
        "run_enter_transition": padded_transitions.masked_fill(~run_mask, 0.0),
        "run_relative_positions": padded_positions.masked_fill(
            ~run_mask.unsqueeze(-1), 0.0
        ),
        "run_physical_states": padded_states.masked_fill(
            ~run_mask.unsqueeze(-1), 0.0
        ),
        "run_mask": run_mask,
        "run_lengths": lengths_tensor,
    }


def _boundary_defined_token_runs(
    hard_tokens: torch.Tensor,
    token_lengths: torch.Tensor,
    hard_boundary_starts: torch.Tensor,
    assignments: torch.Tensor,
) -> list[dict[str, Any]]:
    """Export JSON-friendly runs defined only by fused hard boundaries.

    The historical ``hard_token_runs`` helper always opens a new run when a
    hard token changes.  That contract is intentionally unsuitable here: a
    learned parent primitive must be able to contain multiple child codes.
    ``token`` is therefore the dominant code of the whole run, while
    ``active_tokens`` and ``code_distribution`` retain its child composition.
    """

    if hard_tokens.ndim != 2:
        raise ValueError("hard_tokens must have shape [B,L].")
    if token_lengths.ndim != 1 or token_lengths.shape[0] != hard_tokens.shape[0]:
        raise ValueError("token_lengths must align with hard_tokens [B].")
    if tuple(hard_boundary_starts.shape) != tuple(hard_tokens.shape):
        raise ValueError("hard_boundary_starts must align with hard_tokens [B,L].")
    if assignments.ndim != 3 or tuple(assignments.shape[:2]) != tuple(
        hard_tokens.shape
    ):
        raise ValueError("assignments must align with hard_tokens [B,L,K].")

    tokens_cpu = hard_tokens.detach().cpu().long()
    lengths_cpu = token_lengths.detach().cpu().long()
    starts_cpu = hard_boundary_starts.detach().cpu().bool()
    assignments_cpu = assignments.detach().cpu().float()
    if torch.any(lengths_cpu < 1) or torch.any(
        lengths_cpu > hard_tokens.shape[1]
    ):
        raise ValueError("token_lengths contains an out-of-range value.")

    exported: list[dict[str, Any]] = []
    for row, length_tensor in enumerate(lengths_cpu):
        length = int(length_tensor.item())
        sequence = tokens_cpu[row, :length].tolist()
        if any(int(token) < 0 for token in sequence):
            raise ValueError("A valid hard-token prefix contains a negative id.")
        starts = torch.nonzero(
            starts_cpu[row, :length], as_tuple=False
        ).flatten().tolist()
        if not starts or int(starts[0]) != 0:
            raise ValueError("Every exported run sequence must start at token zero.")
        starts = [int(value) for value in starts]
        ends = starts[1:] + [length]
        runs: list[dict[str, Any]] = []
        for start, end in zip(starts, ends):
            distribution = assignments_cpu[row, start:end].mean(dim=0)
            dominant = int(distribution.argmax().item())
            active = sorted({int(token) for token in sequence[start:end]})
            runs.append(
                {
                    "token": dominant,
                    "dominant_token": dominant,
                    "active_tokens": active,
                    "code_distribution": [
                        float(value) for value in distribution.tolist()
                    ],
                    "start_token_index": int(start),
                    "end_token_index_exclusive": int(end),
                    "duration_tokens": int(end - start),
                }
            )
        exported.append(
            {
                "token_count": length,
                "tokens": [int(token) for token in sequence],
                "run_count": len(runs),
                "runs": runs,
            }
        )
    return exported

class MotionPrimitiveCGCDModel(nn.Module):
    """Pure motion-primitive trajectory model.

    The historical class name is kept temporarily so older result readers can
    locate the symbol.  Unlike the migration build, this model does not own or
    instantiate a complete-trial pooling module or a pooled classification
    head.  Its only classification path is the variable-length primitive
    trajectory readout.
    """

    def __init__(
        self, config: Optional[MotionPrimitiveConfig] = None
    ) -> None:
        super().__init__()
        self.config = (config or MotionPrimitiveConfig()).validated()
        cfg = self.config
        self.window_encoder = ResNet1D(
            in_channels=cfg.in_channels,
            feat_dim=cfg.feature_dim,
            base_channels=cfg.base_channels,
            layers=[2, 2, 2],
            dropout=cfg.backbone_dropout,
        )
        self.primitive_input_norm = nn.LayerNorm(cfg.feature_dim)
        self.codebook = SoftMotionCodebook(
            embedding_dim=cfg.feature_dim,
            codebook_size=cfg.codebook_size,
            temperature=cfg.codebook_temperature,
            use_gumbel_training=False,
            # Internal legacy attribute; the public loss exposes the only
            # effective VQ weights explicitly.
            commitment_beta=0.25,
            dead_code_fraction=0.001,
            epsilon=cfg.epsilon,
        )
        self.boundary_head = PairwiseBoundaryHead(
            cfg.feature_dim, hidden_dim=max(32, cfg.feature_dim // 4)
        )
        nn.init.constant_(
            self.boundary_head.network[-1].bias,
            float(cfg.boundary_initial_bias),
        )
        if cfg.run_state_dim > 0:
            descriptor_dim = 3 * cfg.in_channels
            state_hidden_dim = max(32, 2 * cfg.run_state_dim)
            self.run_state_projector: Optional[nn.Module] = nn.Sequential(
                nn.LayerNorm(descriptor_dim),
                nn.Linear(descriptor_dim, state_hidden_dim),
                nn.GELU(),
                nn.Linear(state_hidden_dim, cfg.run_state_dim),
                nn.LayerNorm(cfg.run_state_dim),
            )
        else:
            self.run_state_projector = None
        self.trajectory_input = nn.Sequential(
            nn.LayerNorm(cfg.run_feature_dim),
            nn.Linear(cfg.run_feature_dim, cfg.trajectory_input_dim),
            nn.GELU(),
        )
        self.trajectory_encoder = nn.GRU(
            input_size=cfg.trajectory_input_dim,
            hidden_size=cfg.trajectory_hidden_dim,
            num_layers=cfg.trajectory_layers,
            batch_first=True,
            dropout=(cfg.trajectory_dropout if cfg.trajectory_layers > 1 else 0.0),
        )
        self.trajectory_norm = nn.LayerNorm(cfg.trajectory_hidden_dim)
        self.trajectory_classifier = nn.Linear(
            cfg.trajectory_hidden_dim, cfg.old_class_count
        )

    @property
    def class_count(self) -> int:
        return int(self.trajectory_classifier.out_features)

    @staticmethod
    def relative_positions(
        starts: torch.Tensor, token_mask: torch.Tensor, window_size: int
    ) -> torch.Tensor:
        """Encode each local window's position without aggregating the trial."""

        del window_size
        if tuple(starts.shape) != tuple(token_mask.shape):
            raise ValueError("starts and token_mask must share shape [B,L].")
        token_mask = token_mask.bool()
        positions = torch.zeros(
            (*starts.shape, 3), dtype=torch.float32, device=starts.device
        )
        for row in range(starts.shape[0]):
            valid_starts = starts[row, token_mask[row]].float()
            if valid_starts.numel() == 0:
                raise ValueError("Every trial must contain at least one valid window.")
            first = valid_starts[0]
            denominator = torch.clamp(valid_starts[-1] - first, min=1.0)
            relative = (valid_starts - first) / denominator
            positions[row, token_mask[row]] = torch.stack(
                (
                    relative,
                    torch.sin(2.0 * math.pi * relative),
                    torch.cos(2.0 * math.pi * relative),
                ),
                dim=1,
            )
        return positions

    def materialize_windows(
        self, trials: torch.Tensor, lengths: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """Convert raw trials to ordered local windows; no trial pooling occurs."""

        cfg = self.config
        if trials.ndim != 3 or trials.shape[1] != cfg.in_channels:
            raise ValueError(
                f"trials must be [B,{cfg.in_channels},T], got {tuple(trials.shape)}."
            )
        if tuple(lengths.shape) != (trials.shape[0],):
            raise ValueError("lengths must have shape [B].")
        lengths = lengths.to(device=trials.device, dtype=torch.long)
        if torch.any(lengths <= 0) or torch.any(lengths > trials.shape[-1]):
            raise ValueError("lengths contains an invalid raw-trial length.")
        window = int(cfg.window_size)
        stride = int(cfg.window_stride)
        if cfg.tail_policy == "drop":
            if torch.any(lengths < window):
                raise ValueError("drop policy requires length >= window_size.")
            token_lengths = 1 + torch.div(
                lengths - window, stride, rounding_mode="floor"
            )
        else:
            token_lengths = 1 + torch.div(
                torch.clamp(lengths - window, min=0) + stride - 1,
                stride,
                rounding_mode="floor",
            )
        maximum = int(token_lengths.max().item())
        required = window + (maximum - 1) * stride
        sample_mask = (
            torch.arange(trials.shape[-1], device=trials.device).unsqueeze(0)
            < lengths.unsqueeze(1)
        )
        safe = trials.masked_fill(~sample_mask.unsqueeze(1), 0.0)
        if required > safe.shape[-1]:
            safe = F.pad(safe, (0, required - safe.shape[-1]))
        windows = safe.unfold(2, window, stride).permute(0, 2, 1, 3).contiguous()
        windows = windows[:, :maximum]
        mask = (
            torch.arange(maximum, device=trials.device).unsqueeze(0)
            < token_lengths.unsqueeze(1)
        )
        starts = (
            torch.arange(maximum, device=trials.device).unsqueeze(0) * stride
        ).expand(trials.shape[0], -1).clone()
        starts.masked_fill_(~mask, -1)
        return {
            "windows": windows,
            "mask": mask,
            "lengths": token_lengths,
            "starts": starts,
            "positions": self.relative_positions(starts, mask, window),
        }

    @property
    def codebook_size(self) -> int:
        return int(self.codebook.codebook_size)

    def _forward_motion(
        self,
        local_features: torch.Tensor,
        token_mask: torch.Tensor,
        windows: torch.Tensor,
        positions: torch.Tensor,
        *,
        hard_codebook: Optional[bool],
        codebook_temperature: Optional[float],
    ) -> dict[str, Any]:
        token_mask = token_mask.bool()
        lengths = token_mask.sum(dim=1).long()
        primitive_features = self.primitive_input_norm(local_features)
        primitive_features = primitive_features.masked_fill(
            ~token_mask.unsqueeze(-1), 0.0
        )
        if windows.ndim != 4 or tuple(windows.shape[:2]) != tuple(token_mask.shape):
            raise ValueError("windows must align with token_mask as [B,L,C,W].")
        if positions.ndim != 3 or tuple(positions.shape[:2]) != tuple(token_mask.shape):
            raise ValueError("positions must align with token_mask as [B,L,P].")
        if positions.shape[-1] != 3:
            raise ValueError("HHR relative positions must have three channels.")
        position_features = positions.to(dtype=local_features.dtype).masked_fill(
            ~token_mask.unsqueeze(-1), 0.0
        )
        physical_descriptors = _window_physical_descriptors(
            windows,
            token_mask,
            epsilon=float(self.config.epsilon),
        )
        if self.run_state_projector is None:
            physical_state_features = physical_descriptors[..., :0]
        else:
            physical_state_features = self.run_state_projector(
                physical_descriptors
            ).masked_fill(~token_mask.unsqueeze(-1), 0.0)
        codebook_output = self.codebook(
            primitive_features,
            token_mask,
            hard=hard_codebook,
            temperature=codebook_temperature,
        )
        learned_logits, learned_boundary, pair_mask = self.boundary_head(
            primitive_features, token_mask
        )
        assignments = codebook_output["trajectory_assignments"]
        token_change = assignments.new_zeros(
            token_mask.shape[0], max(0, token_mask.shape[1] - 1)
        )
        if token_mask.shape[1] > 1:
            token_change = 1.0 - torch.sum(
                assignments[:, :-1] * assignments[:, 1:], dim=-1
            )
            token_change = token_change.clamp(0.0, 1.0).masked_fill(~pair_mask, 0.0)

        # A hard child-code change is only symmetric evidence, not a mandatory
        # cut.  With zero learned logit, changes favour a cut and non-changes
        # favour a merge.  The boundary head can override either prior:
        # learned_logits < -1 suppresses a hard change, while > +1 can split an
        # unchanged code.  This is the smallest learnable fusion that preserves
        # gradients to both the boundary head and straight-through assignments.
        token_change_evidence = 2.0 * token_change - 1.0
        final_boundary_logits = learned_logits + token_change_evidence
        final_boundary = torch.sigmoid(final_boundary_logits)
        final_boundary = final_boundary.masked_fill(~pair_mask, 0.0)
        transitions, window_duration = soft_transition_and_duration(
            assignments, final_boundary, token_mask
        )
        parent_duration = _soft_parent_duration(final_boundary, token_mask)

        # Pair quantities [B,L-1] become token-start quantities [B,L].
        boundary_starts = assignments.new_zeros(token_mask.shape)
        transition_starts = assignments.new_zeros(token_mask.shape)
        # Pairwise relations describe what happened *before* a token.  The
        # first token has no predecessor, so its relation channels are zero.
        if token_mask.shape[1] > 1:
            boundary_starts[:, 1:] = final_boundary
            transition_starts[:, 1:] = transitions
        boundary_starts = boundary_starts.masked_fill(~token_mask, 0.0)
        transition_starts = transition_starts.masked_fill(~token_mask, 0.0)

        # Keep the former window-level representation as a diagnostic.  It is
        # no longer fed directly to the trajectory encoder.
        trajectory_features = torch.cat(
            (
                assignments,
                boundary_starts.unsqueeze(-1),
                transition_starts.unsqueeze(-1),
                parent_duration.unsqueeze(-1),
            ),
            dim=-1,
        ).masked_fill(~token_mask.unsqueeze(-1), 0.0)

        hard_boundary_starts = torch.zeros_like(token_mask)
        hard_boundary_starts[:, 0] = token_mask[:, 0]
        if token_mask.shape[1] > 1:
            hard_boundary_starts[:, 1:] = (
                final_boundary >= float(self.config.boundary_threshold)
            ) & pair_mask
        run_output = _compress_motion_runs(
            assignments,
            codebook_output["hard_tokens"],
            token_mask,
            hard_boundary_starts,
            learned_boundary,
            token_change,
            final_boundary,
            parent_duration,
            position_features,
            physical_state_features,
            codebook_output["normalized_codebook"],
        )
        run_features = run_output["run_features"]
        run_mask = run_output["run_mask"].bool()
        run_lengths = run_output["run_lengths"].long()
        recurrent = self.trajectory_input(run_features)
        recurrent = recurrent.masked_fill(~run_mask.unsqueeze(-1), 0.0)
        packed = pack_padded_sequence(
            recurrent,
            run_lengths.detach().cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        packed_states, hidden = self.trajectory_encoder(packed)
        sequence_states, _ = pad_packed_sequence(
            packed_states,
            batch_first=True,
            total_length=run_mask.shape[1],
        )
        sequence_states = sequence_states.masked_fill(
            ~run_mask.unsqueeze(-1), 0.0
        )
        trajectory_embedding = self.trajectory_norm(hidden[-1])
        trajectory_logits = self.trajectory_classifier(trajectory_embedding)

        runs = _boundary_defined_token_runs(
            codebook_output["hard_tokens"],
            lengths,
            hard_boundary_starts,
            assignments,
        )
        return {
            "primitive_features": primitive_features,
            "window_physical_descriptors": physical_descriptors,
            "window_physical_states": physical_state_features,
            "trajectory_embedding": trajectory_embedding,
            "trajectory_logits": trajectory_logits,
            "trajectory_sequence_states": sequence_states,
            "run_sequence_states": sequence_states,
            # Window-level diagnostic retained for loss terms and plots.  The
            # GRU consumes ``run_features`` instead.
            "trajectory_features": trajectory_features,
            "window_trajectory_features": trajectory_features,
            "learned_boundary_logits": learned_logits,
            "learned_boundary_probabilities": learned_boundary,
            "token_change_probabilities": token_change,
            "token_change_evidence": token_change_evidence,
            "final_boundary_logits": final_boundary_logits,
            "final_boundary_probabilities": final_boundary,
            "boundary_probabilities": final_boundary,
            "boundary_pair_mask": pair_mask,
            "boundary_starts": boundary_starts,
            "hard_boundary_starts": hard_boundary_starts,
            "transition_probabilities": transitions,
            "transition_starts": transition_starts,
            "duration_proxy": parent_duration,
            "window_code_duration_proxy": window_duration,
            "primitive_runs": runs,
            "primitive_classifier_input": (
                "run_level_codebook_embedding_boundary_transition_duration_"
                "position_physical_state"
            ),
            **run_output,
            **codebook_output,
        }

    def forward_trajectory(
        self,
        batch: Mapping[str, torch.Tensor],
        *,
        hard_codebook: Optional[bool] = None,
        codebook_temperature: Optional[float] = None,
    ) -> dict[str, Any]:
        """Run the sole classification path: variable-length primitive trajectory."""

        local_features, _, token_mask = _forward_local_encoder(
            self.window_encoder, batch, self.config
        )
        output: dict[str, Any] = {
            "local_features": local_features,
            "token_mask": token_mask,
            "token_lengths": batch["lengths"].long(),
        }
        output.update(
            self._forward_motion(
                local_features,
                token_mask,
                batch["windows"],
                batch["positions"],
                hard_codebook=hard_codebook,
                codebook_temperature=codebook_temperature,
            )
        )
        return output

    def forward_window_bag(
        self,
        windows: torch.Tensor,
        token_mask: torch.Tensor,
        *,
        starts: Optional[torch.Tensor] = None,
        hard_codebook: Optional[bool] = None,
        codebook_temperature: Optional[float] = None,
    ) -> dict[str, Any]:
        """Legacy-shaped adapter over the primary NPZ trial-bag forward."""

        token_mask = token_mask.bool()
        if starts is None:
            starts = (
                torch.arange(windows.shape[1], device=windows.device).unsqueeze(0)
                * int(self.config.window_stride)
            ).expand(windows.shape[0], -1).clone()
            starts.masked_fill_(~token_mask, -1)
        batch = {
            "windows": windows,
            "mask": token_mask,
            "lengths": token_mask.sum(dim=1),
            "positions": self.relative_positions(
                starts, token_mask, self.config.window_size
            ).to(windows.dtype),
        }
        output = self.forward_trajectory(
            batch,
            hard_codebook=hard_codebook,
            codebook_temperature=codebook_temperature,
        )
        output["window_starts"] = starts
        output["positions"] = batch["positions"]
        return output

    def forward(
        self,
        batch: Mapping[str, torch.Tensor],
        *,
        hard_codebook: Optional[bool] = None,
        codebook_temperature: Optional[float] = None,
    ):
        return self.forward_trajectory(
            batch,
            hard_codebook=hard_codebook,
            codebook_temperature=codebook_temperature,
        )

    @torch.no_grad()
    def normalize_codebook_(self) -> None:
        self.codebook.normalize_learnable_prototypes_()

    def expand_class_heads(
        self,
        new_class_count: int,
        *,
        trajectory_new_centres: Optional[torch.Tensor] = None,
    ) -> None:
        """Expand only the trajectory classifier, preserving seen-class rows."""

        target = int(new_class_count)
        previous = self.class_count
        if target <= previous:
            raise ValueError(
                f"new_class_count must exceed {previous}, got {new_class_count}."
            )
        cfg = self.config
        old_classifier = self.trajectory_classifier
        expanded = nn.Linear(cfg.trajectory_hidden_dim, target).to(
            old_classifier.weight.device
        )
        with torch.no_grad():
            expanded.weight[:previous].copy_(old_classifier.weight)
            expanded.bias[:previous].copy_(old_classifier.bias)
            if trajectory_new_centres is not None:
                centres = trajectory_new_centres
                if tuple(centres.shape) != (
                    target - previous,
                    cfg.trajectory_hidden_dim,
                ):
                    raise ValueError("trajectory_new_centres has an invalid shape.")
                expanded.weight[previous:].copy_(centres)
        self.trajectory_classifier = expanded

def forward_two_views(
    model: MotionPrimitiveCGCDModel,
    views: Sequence[Mapping[str, torch.Tensor]],
    *,
    hard_codebook: Optional[bool] = None,
    codebook_temperature: Optional[float] = None,
) -> list[dict[str, Any]]:
    if len(views) != 2:
        raise ValueError("Motion-primitive training requires exactly two trial views.")
    if not isinstance(model, MotionPrimitiveCGCDModel):
        raise TypeError("forward_two_views requires the motion-primitive model.")
    return [
        model.forward_trajectory(
            view,
            hard_codebook=hard_codebook,
            codebook_temperature=codebook_temperature,
        )
        for view in views
    ]



__all__ = [
    "MotionPrimitiveConfig",
    "MotionPrimitiveCGCDModel",
    "forward_two_views",
]
