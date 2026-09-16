"""Loss contract for the checkpoint-free one-stage motion-primitive experiment.

Only two profiles are admitted:

``J0-U``
    Fully label-free representation, boundary and codebook learning.

``J0-T``
    The same objectives plus trajectory-level cross entropy on explicitly
    marked, labelled old-class trials.  Unlabelled rows must carry the sentinel
    label ``-1``; labels are never consumed by a window-level objective.

The module deliberately contains no Happy checkpoint distillation, window
classification, InfoNCE, activity-level SupCon, DINO or MeMax objective.  Raw
kinematic pseudo-boundaries are supplied by ``one_stage_boundaries.py`` and
their source tag is checked here to prevent a frozen legacy encoder from being
silently reintroduced.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, replace
from typing import Optional

import torch
import torch.nn.functional as F


J0_UNSUPERVISED = "J0-U"
J0_TRAJECTORY = "J0-T"
RAW_KINEMATIC_ONLY = "raw_kinematic_only"
LABEL_FREE_STATE_SOURCES = frozenset(
    {"raw_kinematic_state_descriptor", "raw_sensor_window"}
)


def _finite_nonnegative(name: str, value: float) -> float:
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be finite and non-negative.")
    return result


def _floating(name: str, value: torch.Tensor, ndim: Optional[int] = None) -> None:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor.")
    if ndim is not None and value.ndim != ndim:
        raise ValueError(f"{name} must have {ndim} dimensions, got {value.ndim}.")
    if not torch.is_floating_point(value):
        raise TypeError(f"{name} must have floating dtype.")


def _boolean_mask(name: str, value: torch.Tensor, shape: tuple[int, ...]) -> torch.Tensor:
    if not isinstance(value, torch.Tensor) or value.dtype != torch.bool:
        raise TypeError(f"{name} must be a boolean torch.Tensor.")
    if tuple(value.shape) != tuple(shape):
        raise ValueError(f"{name} must have shape {shape}, got {tuple(value.shape)}.")
    return value


def _graph_zero(*values: torch.Tensor) -> torch.Tensor:
    if not values:
        return torch.tensor(0.0)
    return sum((item.sum() * 0.0 for item in values), values[0].new_zeros(()))


@dataclass(frozen=True)
class OneStageLossConfig:
    """Weights and structural priors for a J0 one-stage loss combination."""

    profile: str = J0_UNSUPERVISED
    raw_boundary_weight: float = 1.0
    vq_commitment_weight: float = 0.25
    vq_codebook_weight: float = 1.0
    content_reconstruction_weight: float = 0.5
    stable_next_content_weight: float = 0.25
    utilization_floor_weight: float = 0.05
    boundary_sparsity_weight: float = 0.02
    minimum_duration_weight: float = 0.05
    state_reconstruction_weight: float = 0.5
    masked_token_prediction_weight: float = 1.0
    masked_state_reconstruction_weight: float = 0.5
    trajectory_ce_weight: float = 0.0
    utilization_entropy_floor: float = 0.35
    utilization_temperature: float = 1.0
    maximum_unanchored_boundary_rate: float = 0.35
    minimum_segment_windows: int = 2
    state_huber_delta: float = 1.0
    old_class_count: int = 6
    unlabelled_index: int = -1

    @classmethod
    def j0_u(cls, **overrides: object) -> "OneStageLossConfig":
        return cls(**overrides).validated()

    @classmethod
    def j0_t(cls, **overrides: object) -> "OneStageLossConfig":
        values = {"profile": J0_TRAJECTORY, "trajectory_ce_weight": 1.0}
        values.update(overrides)
        return cls(**values).validated()

    def validated(self) -> "OneStageLossConfig":
        if self.profile not in {J0_UNSUPERVISED, J0_TRAJECTORY}:
            raise ValueError("profile must be 'J0-U' or 'J0-T'.")
        for name in (
            "raw_boundary_weight",
            "vq_commitment_weight",
            "vq_codebook_weight",
            "content_reconstruction_weight",
            "stable_next_content_weight",
            "utilization_floor_weight",
            "boundary_sparsity_weight",
            "minimum_duration_weight",
            "state_reconstruction_weight",
            "masked_token_prediction_weight",
            "masked_state_reconstruction_weight",
            "trajectory_ce_weight",
        ):
            _finite_nonnegative(name, getattr(self, name))
        if self.profile == J0_UNSUPERVISED and float(self.trajectory_ce_weight) != 0.0:
            raise ValueError("J0-U forbids trajectory supervision and requires its weight to be zero.")
        if self.profile == J0_TRAJECTORY and float(self.trajectory_ce_weight) <= 0.0:
            raise ValueError("J0-T requires a positive trajectory_ce_weight.")
        if (
            float(self.masked_token_prediction_weight) <= 0.0
            and float(self.masked_state_reconstruction_weight) <= 0.0
        ):
            raise ValueError(
                "At least one masked-trajectory objective must remain active so the "
                "trajectory encoder is trainable in J0-U."
            )
        if not 0.0 <= float(self.utilization_entropy_floor) <= 1.0:
            raise ValueError("utilization_entropy_floor must lie in [0,1].")
        if not math.isfinite(float(self.utilization_temperature)) or float(
            self.utilization_temperature
        ) <= 0.0:
            raise ValueError("utilization_temperature must be positive and finite.")
        if not 0.0 <= float(self.maximum_unanchored_boundary_rate) <= 1.0:
            raise ValueError("maximum_unanchored_boundary_rate must lie in [0,1].")
        if int(self.minimum_segment_windows) < 1:
            raise ValueError("minimum_segment_windows must be positive.")
        if not math.isfinite(float(self.state_huber_delta)) or float(
            self.state_huber_delta
        ) <= 0.0:
            raise ValueError("state_huber_delta must be positive and finite.")
        if int(self.old_class_count) < 1:
            raise ValueError("old_class_count must be positive.")
        if int(self.unlabelled_index) >= 0:
            raise ValueError("unlabelled_index must be negative.")
        return self

    def with_ablation(self, **weights: float) -> "OneStageLossConfig":
        """Return a validated copy with explicitly named weight changes."""

        allowed = {
            name
            for name in asdict(self)
            if name.endswith("_weight")
        }
        unknown = sorted(set(weights) - allowed)
        if unknown:
            raise ValueError(f"Unknown ablation weight(s): {unknown}.")
        return replace(self, **weights).validated()

    def to_audit(self) -> dict:
        self.validated()
        result = asdict(self)
        result.update(
            {
                "window_classification_used": False,
                "infonce_used": False,
                "activity_supcon_used": False,
                "dino_used": False,
                "memax_used": False,
                "happy_checkpoint_used": False,
                "boundary_anchor_source": RAW_KINEMATIC_ONLY,
                "label_scope": (
                    "none"
                    if self.profile == J0_UNSUPERVISED
                    else "trajectory_cross_entropy_on_explicitly_labelled_old_trials_only"
                ),
                "trajectory_encoder_label_free_objective": (
                    "masked pseudo-token prediction plus masked physical-state reconstruction"
                ),
                "pseudo_token_target_gradient": "stopped_by_integer_target",
                "utilization_regularizer": (
                    "one-sided_hinge_below_normalized_entropy_floor; "
                    "no pressure toward uniform usage above the floor"
                ),
                "objective_scopes": {
                    "raw_boundary": "valid model-frame boundaries; raw q50/q90 anchors only",
                    "vq_commitment": "valid local window embeddings; encoder gradient only",
                    "vq_codebook": "valid local window embeddings; codebook gradient only",
                    "content_reconstruction": (
                        "valid quantized tokens; detached local-content targets"
                    ),
                    "stable_next_content": (
                        "raw-q50 stable adjacent frames only; detached next-content targets"
                    ),
                    "utilization_floor": "batch aggregate of valid soft VQ assignments",
                    "boundary_sparsity": "valid non-q90 boundaries; one-sided rate budget",
                    "minimum_duration": "valid within-trial boundary probabilities",
                    "state_reconstruction": "valid windows; detached label-free physical targets",
                    "masked_token_prediction": (
                        "masked valid trajectory positions; detached integer VQ targets"
                    ),
                    "masked_state_reconstruction": (
                        "masked valid trajectory positions; detached physical targets"
                    ),
                    "trajectory_ce": (
                        "disabled"
                        if self.profile == J0_UNSUPERVISED
                        else "explicitly labelled old-class trials only"
                    ),
                },
            }
        )
        json.dumps(result, allow_nan=False)
        return result


@dataclass
class OneStageLossInputs:
    """Tensor bundle whose API exposes no window- or activity-pair labels."""

    encoded_states: torch.Tensor
    quantized_states: torch.Tensor
    reconstructed_content: torch.Tensor
    content_targets: torch.Tensor
    next_content_predictions: torch.Tensor
    next_content_targets: torch.Tensor
    assignment_logits: torch.Tensor
    valid_window_mask: torch.Tensor
    boundary_logits: torch.Tensor
    valid_boundary_mask: torch.Tensor
    raw_stable_mask: torch.Tensor
    raw_change_mask: torch.Tensor
    reconstructed_states: torch.Tensor
    state_targets: torch.Tensor
    masked_token_logits: torch.Tensor
    pseudo_token_targets: torch.Tensor
    masked_state_predictions: torch.Tensor
    masked_trajectory_mask: torch.Tensor
    boundary_anchor_source: str = RAW_KINEMATIC_ONLY
    state_target_source: str = "raw_kinematic_state_descriptor"
    trajectory_logits: Optional[torch.Tensor] = None
    trajectory_labels: Optional[torch.Tensor] = None
    labelled_old_trial_mask: Optional[torch.Tensor] = None


@dataclass
class OneStageLossResult:
    total: torch.Tensor
    components: dict[str, torch.Tensor]
    weighted_components: dict[str, torch.Tensor]
    metrics: dict[str, float | int | None]
    config: OneStageLossConfig
    labelled_old_trial_count: int

    def to_audit(self) -> dict:
        result = {
            "config": self.config.to_audit(),
            "total": float(self.total.detach().cpu()),
            "components": {
                name: float(value.detach().cpu()) for name, value in self.components.items()
            },
            "weighted_components": {
                name: float(value.detach().cpu())
                for name, value in self.weighted_components.items()
            },
            "metrics": dict(self.metrics),
            "labelled_old_trial_count": int(self.labelled_old_trial_count),
            "physical_activity_labels_consumed_outside_trajectory_ce": False,
        }
        json.dumps(result, allow_nan=False)
        return result


def raw_kinematic_boundary_supervision_loss(
    boundary_logits: torch.Tensor,
    valid_boundary_mask: torch.Tensor,
    raw_stable_mask: torch.Tensor,
    raw_change_mask: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, int]]:
    """Balanced BCE on raw-only q50 stable and q90 change anchors.

    Uncertain boundaries between q50 and q90 are ignored.  Stable and change
    groups are averaged separately so the more numerous group cannot dominate.
    """

    _floating("boundary_logits", boundary_logits, 2)
    shape = tuple(boundary_logits.shape)
    valid = _boolean_mask("valid_boundary_mask", valid_boundary_mask, shape)
    stable = _boolean_mask("raw_stable_mask", raw_stable_mask, shape)
    change = _boolean_mask("raw_change_mask", raw_change_mask, shape)
    if boundary_logits.device != valid.device or stable.device != valid.device or change.device != valid.device:
        raise ValueError("Boundary logits and masks must be on the same device.")
    if torch.any((stable | change) & ~valid):
        raise ValueError("Raw boundary anchors select invalid/padded positions.")
    if torch.any(stable & change):
        raise ValueError("Raw stable and change masks must be disjoint.")

    terms: list[torch.Tensor] = []
    if torch.any(stable):
        terms.append(
            F.binary_cross_entropy_with_logits(
                boundary_logits[stable], torch.zeros_like(boundary_logits[stable])
            )
        )
    if torch.any(change):
        terms.append(
            F.binary_cross_entropy_with_logits(
                boundary_logits[change], torch.ones_like(boundary_logits[change])
            )
        )
    loss = torch.stack(terms).mean() if terms else _graph_zero(boundary_logits)
    return loss, {
        "valid_boundary_count": int(valid.sum().item()),
        "raw_stable_anchor_count": int(stable.sum().item()),
        "raw_change_anchor_count": int(change.sum().item()),
        "raw_uncertain_boundary_count": int((valid & ~stable & ~change).sum().item()),
    }


def vq_losses(
    encoded_states: torch.Tensor,
    quantized_states: torch.Tensor,
    valid_window_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return encoder commitment and codebook fitting losses.

    Stop-gradient routing follows VQ-VAE: commitment updates the encoder and
    codebook fitting updates the code vectors.  If an EMA codebook is used,
    callers should ablate ``vq_codebook_weight`` to zero.
    """

    _floating("encoded_states", encoded_states, 3)
    _floating("quantized_states", quantized_states, 3)
    if tuple(encoded_states.shape) != tuple(quantized_states.shape):
        raise ValueError("encoded_states and quantized_states must have identical shape.")
    if encoded_states.device != quantized_states.device:
        raise ValueError("encoded_states and quantized_states must share a device.")
    valid = _boolean_mask(
        "valid_window_mask", valid_window_mask, tuple(encoded_states.shape[:2])
    )
    if valid.device != encoded_states.device:
        raise ValueError("valid_window_mask must share the feature device.")
    if not torch.any(valid):
        zero = _graph_zero(encoded_states, quantized_states)
        return zero, zero
    commitment = F.mse_loss(encoded_states[valid], quantized_states[valid].detach())
    codebook = F.mse_loss(quantized_states[valid], encoded_states[valid].detach())
    return commitment, codebook


def codebook_utilization_floor_loss(
    assignment_logits: torch.Tensor,
    valid_window_mask: torch.Tensor,
    *,
    normalized_entropy_floor: float,
    temperature: float = 1.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Penalise only severe codebook collapse, never enforce uniform usage.

    The loss is ``relu(floor - H(mean soft assignments)/log(K))^2``.  Once the
    minimum entropy floor is reached the gradient is exactly zero; this avoids
    erasing semantic frequency imbalance merely to make every code equally common.
    """

    _floating("assignment_logits", assignment_logits, 3)
    valid = _boolean_mask(
        "valid_window_mask", valid_window_mask, tuple(assignment_logits.shape[:2])
    )
    if valid.device != assignment_logits.device:
        raise ValueError("valid_window_mask must share the assignment device.")
    floor = float(normalized_entropy_floor)
    if not 0.0 <= floor <= 1.0:
        raise ValueError("normalized_entropy_floor must lie in [0,1].")
    if not math.isfinite(float(temperature)) or float(temperature) <= 0.0:
        raise ValueError("temperature must be positive and finite.")
    code_count = int(assignment_logits.shape[-1])
    if code_count < 1:
        raise ValueError("assignment_logits must contain at least one code.")
    if not torch.any(valid) or code_count == 1:
        zero = _graph_zero(assignment_logits)
        return zero, {"normalized_codebook_entropy": 1.0, "effective_code_count": 1.0}
    probabilities = F.softmax(assignment_logits[valid] / float(temperature), dim=-1)
    usage = probabilities.mean(dim=0)
    entropy = -(usage * usage.clamp_min(torch.finfo(usage.dtype).tiny).log()).sum()
    normalized = entropy / math.log(code_count)
    loss = F.relu(normalized.new_tensor(floor) - normalized).square()
    return loss, {
        "normalized_codebook_entropy": float(normalized.detach().cpu()),
        "effective_code_count": float(torch.exp(entropy).detach().cpu()),
    }


def boundary_sparsity_budget_loss(
    boundary_logits: torch.Tensor,
    valid_boundary_mask: torch.Tensor,
    raw_change_mask: torch.Tensor,
    *,
    maximum_unanchored_rate: float,
) -> tuple[torch.Tensor, float | None]:
    """One-sided boundary-rate budget excluding raw q90 change anchors."""

    _floating("boundary_logits", boundary_logits, 2)
    shape = tuple(boundary_logits.shape)
    valid = _boolean_mask("valid_boundary_mask", valid_boundary_mask, shape)
    change = _boolean_mask("raw_change_mask", raw_change_mask, shape)
    if torch.any(change & ~valid):
        raise ValueError("raw_change_mask selects invalid boundaries.")
    maximum = float(maximum_unanchored_rate)
    if not 0.0 <= maximum <= 1.0:
        raise ValueError("maximum_unanchored_rate must lie in [0,1].")
    eligible = valid & ~change
    if not torch.any(eligible):
        return _graph_zero(boundary_logits), None
    rate = torch.sigmoid(boundary_logits[eligible]).mean()
    loss = F.relu(rate - rate.new_tensor(maximum)).square()
    return loss, float(rate.detach().cpu())


def boundary_minimum_duration_loss(
    boundary_logits: torch.Tensor,
    valid_boundary_mask: torch.Tensor,
    *,
    minimum_segment_windows: int,
) -> torch.Tensor:
    """Softly discourage edge segments and consecutive segments shorter than a floor."""

    _floating("boundary_logits", boundary_logits, 2)
    valid = _boolean_mask(
        "valid_boundary_mask", valid_boundary_mask, tuple(boundary_logits.shape)
    )
    minimum = int(minimum_segment_windows)
    if minimum < 1:
        raise ValueError("minimum_segment_windows must be positive.")
    if minimum == 1:
        return _graph_zero(boundary_logits)
    probabilities = torch.sigmoid(boundary_logits)
    terms: list[torch.Tensor] = []
    for row in range(boundary_logits.shape[0]):
        indices = torch.nonzero(valid[row], as_tuple=False).flatten()
        if not len(indices):
            continue
        expected = torch.arange(len(indices), device=indices.device)
        if not torch.equal(indices, expected):
            raise ValueError("valid_boundary_mask must be a contiguous prefix per trial.")
        row_probabilities = probabilities[row, : len(indices)]
        window_count = len(indices) + 1
        edge_count = min(minimum - 1, len(indices))
        if edge_count:
            terms.append(row_probabilities[:edge_count].mean())
            terms.append(row_probabilities[-edge_count:].mean())
        for gap in range(1, min(minimum, len(indices))):
            terms.append((row_probabilities[:-gap] * row_probabilities[gap:]).mean())
        # If a trial itself is shorter than the requested duration, no boundary
        # placement can satisfy the prior; fail closed instead of hiding it.
        if window_count < minimum:
            raise ValueError(
                "A trial is shorter than minimum_segment_windows; the duration prior is infeasible."
            )
    return torch.stack(terms).mean() if terms else _graph_zero(boundary_logits)


def state_reconstruction_loss(
    reconstructed_states: torch.Tensor,
    state_targets: torch.Tensor,
    valid_window_mask: torch.Tensor,
    *,
    huber_delta: float = 1.0,
) -> torch.Tensor:
    """Label-free physical-state fidelity for static and low-motion activities."""

    _floating("reconstructed_states", reconstructed_states)
    _floating("state_targets", state_targets)
    if reconstructed_states.ndim < 3:
        raise ValueError("State tensors must have at least [batch,window,feature] dimensions.")
    if tuple(reconstructed_states.shape) != tuple(state_targets.shape):
        raise ValueError("reconstructed_states and state_targets must have identical shape.")
    if reconstructed_states.device != state_targets.device:
        raise ValueError("State reconstruction tensors must share a device.")
    valid = _boolean_mask(
        "valid_window_mask", valid_window_mask, tuple(reconstructed_states.shape[:2])
    )
    if valid.device != reconstructed_states.device:
        raise ValueError("valid_window_mask must share the state tensor device.")
    delta = float(huber_delta)
    if not math.isfinite(delta) or delta <= 0.0:
        raise ValueError("huber_delta must be positive and finite.")
    if not torch.any(valid):
        return _graph_zero(reconstructed_states)
    return F.huber_loss(
        reconstructed_states[valid],
        state_targets.detach()[valid],
        reduction="mean",
        delta=delta,
    )


def stable_next_content_prediction_loss(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    valid_boundary_mask: torch.Tensor,
    raw_stable_mask: torch.Tensor,
    *,
    huber_delta: float = 1.0,
) -> tuple[torch.Tensor, int]:
    """Predict the next local content only across raw-stable boundaries.

    The selection mask is the frozen raw-kinematic q50 anchor.  A learned
    boundary can therefore never declare its own easy targets or smooth a
    representation across a raw q90 change anchor.
    """

    _floating("next_content_predictions", predictions, 3)
    _floating("next_content_targets", targets, 3)
    if tuple(predictions.shape) != tuple(targets.shape):
        raise ValueError("Next-content prediction and target shapes differ.")
    shape = tuple(predictions.shape[:2])
    valid = _boolean_mask("valid_boundary_mask", valid_boundary_mask, shape)
    stable = _boolean_mask("raw_stable_mask", raw_stable_mask, shape)
    if any(item.device != predictions.device for item in (targets, valid, stable)):
        raise ValueError("Stable next-content tensors must share one device.")
    if torch.any(stable & ~valid):
        raise ValueError("raw_stable_mask selects an invalid boundary.")
    selected = stable & valid
    if not torch.any(selected):
        return _graph_zero(predictions), 0
    delta = float(huber_delta)
    if not math.isfinite(delta) or delta <= 0.0:
        raise ValueError("huber_delta must be positive and finite.")
    loss = F.huber_loss(
        predictions[selected],
        targets.detach()[selected],
        reduction="mean",
        delta=delta,
    )
    return loss, int(selected.sum().item())


def masked_trajectory_modeling_losses(
    masked_token_logits: torch.Tensor,
    pseudo_token_targets: torch.Tensor,
    masked_state_predictions: torch.Tensor,
    state_targets: torch.Tensor,
    masked_trajectory_mask: torch.Tensor,
    valid_window_mask: torch.Tensor,
    *,
    unlabelled_index: int = -1,
    huber_delta: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Train the trajectory encoder without activity labels.

    Pseudo-token targets are integer VQ assignments, which are necessarily
    stop-gradient.  Only deliberately masked valid positions are targets; all
    other token entries must be ``unlabelled_index`` so no hidden supervision
    can leak into the objective.  Physical state/descriptor targets are also
    detached before the masked Huber loss.
    """

    _floating("masked_token_logits", masked_token_logits, 3)
    _floating("masked_state_predictions", masked_state_predictions)
    _floating("state_targets", state_targets)
    batch_size, length, code_count = masked_token_logits.shape
    if code_count < 1:
        raise ValueError("masked_token_logits must contain at least one code.")
    if not isinstance(pseudo_token_targets, torch.Tensor) or pseudo_token_targets.dtype != torch.long:
        raise TypeError("pseudo_token_targets must be a torch.long tensor.")
    if tuple(pseudo_token_targets.shape) != (batch_size, length):
        raise ValueError("pseudo_token_targets must have shape [batch,window].")
    if tuple(masked_state_predictions.shape) != tuple(state_targets.shape):
        raise ValueError("Masked state prediction and target shapes differ.")
    if tuple(masked_state_predictions.shape[:2]) != (batch_size, length):
        raise ValueError("Masked state tensors must align with token positions.")
    valid = _boolean_mask("valid_window_mask", valid_window_mask, (batch_size, length))
    selected = _boolean_mask(
        "masked_trajectory_mask", masked_trajectory_mask, (batch_size, length)
    )
    tensors = (
        pseudo_token_targets,
        masked_state_predictions,
        state_targets,
        valid,
        selected,
    )
    if any(item.device != masked_token_logits.device for item in tensors):
        raise ValueError("All masked-trajectory tensors must share a device.")
    if torch.any(selected & ~valid):
        raise ValueError("masked_trajectory_mask selects padded positions.")
    if torch.any(pseudo_token_targets[~selected] != int(unlabelled_index)):
        raise ValueError(
            "Unmasked/padded pseudo-token targets must carry only unlabelled_index."
        )
    targets = pseudo_token_targets[selected]
    if not targets.numel():
        raise ValueError("At least one valid trajectory position must be masked.")
    if torch.any(targets < 0) or torch.any(targets >= code_count):
        raise ValueError("A masked pseudo-token target lies outside the codebook range.")
    delta = float(huber_delta)
    if not math.isfinite(delta) or delta <= 0.0:
        raise ValueError("huber_delta must be positive and finite.")
    token_loss = F.cross_entropy(masked_token_logits[selected], targets.detach())
    state_loss = F.huber_loss(
        masked_state_predictions[selected],
        state_targets.detach()[selected],
        reduction="mean",
        delta=delta,
    )
    return token_loss, state_loss, int(targets.numel())


def trajectory_old_class_cross_entropy(
    trajectory_logits: torch.Tensor,
    trajectory_labels: torch.Tensor,
    labelled_old_trial_mask: torch.Tensor,
    *,
    old_class_count: int,
    unlabelled_index: int = -1,
) -> tuple[torch.Tensor, int]:
    """Cross entropy restricted to explicitly labelled old-class trials."""

    _floating("trajectory_logits", trajectory_logits, 2)
    batch_size, class_count = trajectory_logits.shape
    if class_count != int(old_class_count):
        raise ValueError("Trajectory logit width must equal old_class_count.")
    if not isinstance(trajectory_labels, torch.Tensor) or trajectory_labels.dtype != torch.long:
        raise TypeError("trajectory_labels must be a torch.long tensor.")
    if tuple(trajectory_labels.shape) != (batch_size,):
        raise ValueError("trajectory_labels must have shape [batch].")
    selected = _boolean_mask(
        "labelled_old_trial_mask", labelled_old_trial_mask, (batch_size,)
    )
    if trajectory_labels.device != trajectory_logits.device or selected.device != trajectory_logits.device:
        raise ValueError("Trajectory logits, labels and mask must share a device.")
    if torch.any(trajectory_labels[~selected] != int(unlabelled_index)):
        raise ValueError(
            "Unlabelled trials must carry only unlabelled_index; hidden physical labels are forbidden."
        )
    labelled = trajectory_labels[selected]
    if not labelled.numel():
        raise ValueError("J0-T requires at least one explicitly labelled old-class trial.")
    if torch.any(labelled < 0) or torch.any(labelled >= int(old_class_count)):
        raise ValueError("A labelled trajectory target lies outside the old-class range.")
    return F.cross_entropy(trajectory_logits[selected], labelled), int(labelled.numel())


def compose_one_stage_loss(
    inputs: OneStageLossInputs,
    config: OneStageLossConfig,
) -> OneStageLossResult:
    """Compose a leakage-checked J0-U or J0-T objective."""

    config = config.validated()
    if inputs.boundary_anchor_source != RAW_KINEMATIC_ONLY:
        raise ValueError(
            "J0 boundary supervision must come from raw kinematics only; "
            "frozen legacy features/checkpoints are forbidden."
        )
    if inputs.state_target_source not in LABEL_FREE_STATE_SOURCES:
        raise ValueError("state_target_source must be an approved label-free physical target.")

    _floating("encoded_states", inputs.encoded_states, 3)
    batch_size, window_count, _ = inputs.encoded_states.shape
    expected_boundary_shape = (batch_size, max(0, window_count - 1))
    if tuple(inputs.boundary_logits.shape) != expected_boundary_shape:
        raise ValueError(
            "boundary_logits must be model-frame boundaries with shape [B,L-1]; "
            f"expected {expected_boundary_shape}, got {tuple(inputs.boundary_logits.shape)}."
        )
    if tuple(inputs.assignment_logits.shape[:2]) != (batch_size, window_count):
        raise ValueError("assignment_logits must align with encoded window positions.")
    if tuple(inputs.masked_token_logits.shape[:2]) != (batch_size, window_count):
        raise ValueError("masked_token_logits must align with trajectory positions.")
    if inputs.masked_token_logits.shape[-1] != inputs.assignment_logits.shape[-1]:
        raise ValueError("Masked-token vocabulary must equal the VQ codebook size.")

    # Enforce the label boundary before evaluating even one unsupervised term.
    # This makes J0-U fail closed if a caller accidentally forwards dataset
    # labels, including an apparently harmless all--1 vector.
    trajectory_arguments = (
        inputs.trajectory_logits,
        inputs.trajectory_labels,
        inputs.labelled_old_trial_mask,
    )
    if config.profile == J0_UNSUPERVISED:
        if any(item is not None for item in trajectory_arguments):
            raise ValueError("J0-U forbids trajectory logits, labels and labelled masks.")
    elif any(item is None for item in trajectory_arguments):
        raise ValueError("J0-T requires trajectory logits, labels and labelled-old mask.")

    raw_boundary, boundary_metrics = raw_kinematic_boundary_supervision_loss(
        inputs.boundary_logits,
        inputs.valid_boundary_mask,
        inputs.raw_stable_mask,
        inputs.raw_change_mask,
    )
    commitment, codebook = vq_losses(
        inputs.encoded_states,
        inputs.quantized_states,
        inputs.valid_window_mask,
    )
    content_reconstruction = state_reconstruction_loss(
        inputs.reconstructed_content,
        inputs.content_targets,
        inputs.valid_window_mask,
        huber_delta=float(config.state_huber_delta),
    )
    stable_next_content, stable_next_count = stable_next_content_prediction_loss(
        inputs.next_content_predictions,
        inputs.next_content_targets,
        inputs.valid_boundary_mask,
        inputs.raw_stable_mask,
        huber_delta=float(config.state_huber_delta),
    )
    utilization, utilization_metrics = codebook_utilization_floor_loss(
        inputs.assignment_logits,
        inputs.valid_window_mask,
        normalized_entropy_floor=float(config.utilization_entropy_floor),
        temperature=float(config.utilization_temperature),
    )
    sparsity, unanchored_rate = boundary_sparsity_budget_loss(
        inputs.boundary_logits,
        inputs.valid_boundary_mask,
        inputs.raw_change_mask,
        maximum_unanchored_rate=float(config.maximum_unanchored_boundary_rate),
    )
    minimum_duration = boundary_minimum_duration_loss(
        inputs.boundary_logits,
        inputs.valid_boundary_mask,
        minimum_segment_windows=int(config.minimum_segment_windows),
    )
    reconstruction = state_reconstruction_loss(
        inputs.reconstructed_states,
        inputs.state_targets,
        inputs.valid_window_mask,
        huber_delta=float(config.state_huber_delta),
    )
    masked_token, masked_state, masked_position_count = masked_trajectory_modeling_losses(
        inputs.masked_token_logits,
        inputs.pseudo_token_targets,
        inputs.masked_state_predictions,
        inputs.state_targets,
        inputs.masked_trajectory_mask,
        inputs.valid_window_mask,
        unlabelled_index=int(config.unlabelled_index),
        huber_delta=float(config.state_huber_delta),
    )

    trajectory = _graph_zero(inputs.encoded_states)
    labelled_count = 0
    if config.profile == J0_UNSUPERVISED:
        pass
    else:
        assert inputs.trajectory_logits is not None
        assert inputs.trajectory_labels is not None
        assert inputs.labelled_old_trial_mask is not None
        trajectory, labelled_count = trajectory_old_class_cross_entropy(
            inputs.trajectory_logits,
            inputs.trajectory_labels,
            inputs.labelled_old_trial_mask,
            old_class_count=int(config.old_class_count),
            unlabelled_index=int(config.unlabelled_index),
        )

    components = {
        "raw_boundary": raw_boundary,
        "vq_commitment": commitment,
        "vq_codebook": codebook,
        "content_reconstruction": content_reconstruction,
        "stable_next_content": stable_next_content,
        "utilization_floor": utilization,
        "boundary_sparsity": sparsity,
        "minimum_duration": minimum_duration,
        "state_reconstruction": reconstruction,
        "masked_token_prediction": masked_token,
        "masked_state_reconstruction": masked_state,
        "trajectory_ce": trajectory,
    }
    weights = {
        "raw_boundary": float(config.raw_boundary_weight),
        "vq_commitment": float(config.vq_commitment_weight),
        "vq_codebook": float(config.vq_codebook_weight),
        "content_reconstruction": float(config.content_reconstruction_weight),
        "stable_next_content": float(config.stable_next_content_weight),
        "utilization_floor": float(config.utilization_floor_weight),
        "boundary_sparsity": float(config.boundary_sparsity_weight),
        "minimum_duration": float(config.minimum_duration_weight),
        "state_reconstruction": float(config.state_reconstruction_weight),
        "masked_token_prediction": float(config.masked_token_prediction_weight),
        "masked_state_reconstruction": float(config.masked_state_reconstruction_weight),
        "trajectory_ce": float(config.trajectory_ce_weight),
    }
    weighted = {name: weights[name] * value for name, value in components.items()}
    total = sum(weighted.values(), inputs.encoded_states.new_zeros(()))
    if not bool(torch.isfinite(total).item()):
        raise RuntimeError("One-stage objective produced a non-finite total loss.")
    metrics: dict[str, float | int | None] = {
        **boundary_metrics,
        **utilization_metrics,
        "predicted_unanchored_boundary_rate": unanchored_rate,
        "masked_trajectory_position_count": masked_position_count,
        "stable_next_content_pair_count": stable_next_count,
    }
    return OneStageLossResult(
        total=total,
        components=components,
        weighted_components=weighted,
        metrics=metrics,
        config=config,
        labelled_old_trial_count=labelled_count,
    )


__all__ = [
    "J0_TRAJECTORY",
    "J0_UNSUPERVISED",
    "LABEL_FREE_STATE_SOURCES",
    "RAW_KINEMATIC_ONLY",
    "OneStageLossConfig",
    "OneStageLossInputs",
    "OneStageLossResult",
    "boundary_minimum_duration_loss",
    "boundary_sparsity_budget_loss",
    "codebook_utilization_floor_loss",
    "compose_one_stage_loss",
    "masked_trajectory_modeling_losses",
    "raw_kinematic_boundary_supervision_loss",
    "state_reconstruction_loss",
    "stable_next_content_prediction_loss",
    "trajectory_old_class_cross_entropy",
    "vq_losses",
]
