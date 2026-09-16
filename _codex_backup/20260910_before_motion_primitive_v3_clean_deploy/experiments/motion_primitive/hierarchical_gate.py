"""Label-free residual-to-gravity hierarchical gate utilities.

The gate is intentionally narrower than a general action recognizer.  It
refines one automatically selected coarse motion token as follows:

1. fit a two-child codebook in encoder-residual space;
2. identify the more static residual child using train-only raw-motion energy;
3. define an in-distribution confidence radius from that train child;
4. fit a gravity-only two-child codebook on confident static occurrences;
5. keep dynamic, low-confidence, and non-selected occurrences at the original
   coarse token, while mapping confident static occurrences to two new tokens.

No activity label or name is accepted anywhere in this module.  Ground truth
must therefore be joined only after the returned assignments are frozen.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Mapping, Sequence

import numpy as np

from experiments.motion_primitive.core import EPS
from experiments.motion_primitive.online_secondary_codebook import (
    SecondaryCodebook,
    UnlabelledTokenOccurrence,
    assign_secondary_codebook,
    fit_secondary_codebook,
    occurrence_distance_matrix,
    secondary_fit_diagnostics,
    validate_occurrences,
)


GATE_STATES = {
    0: "coarse_fallback",
    1: "confident_static_gravity_child_0",
    2: "confident_static_gravity_child_1",
}


@dataclass(frozen=True)
class HierarchicalGateModel:
    """Frozen label-free gate fitted only on cumulative online-train trials."""

    coarse_token: int
    residual_codebook: SecondaryCodebook
    gravity_codebook: SecondaryCodebook | None
    gate_enabled: bool
    gate_disable_reason: str | None
    static_residual_child: int
    static_radius: float
    static_radius_quantile: float
    minimum_token_fraction: float
    minimum_motion_energy_ratio: float
    minimum_motion_energy_gap: float
    motion_energy_ratio: float
    motion_energy_gap: float
    motion_component_scales: np.ndarray
    residual_child_motion_component_medians: np.ndarray
    residual_child_motion_medians: tuple[float, float]
    fit_trial_ids: tuple[int, ...]
    confident_static_fit_trial_ids: tuple[int, ...]


@dataclass(frozen=True)
class HierarchicalAssignments:
    """Per-occurrence frozen gate decisions; state 0 preserves the coarse token."""

    trial_ids: np.ndarray
    residual_children: np.ndarray
    assigned_residual_distances: np.ndarray
    confident_static: np.ndarray
    gravity_children: np.ndarray
    gate_states: np.ndarray


def assert_gate_schema_is_label_free() -> None:
    """Fail if a future edit adds activity identity to the fitting schema."""

    forbidden_fragments = ("label", "activity", "class", "name")
    for schema in (HierarchicalGateModel, HierarchicalAssignments):
        names = {item.name.lower() for item in fields(schema)}
        leaked = sorted(
            name
            for name in names
            if any(fragment in name for fragment in forbidden_fragments)
        )
        if leaked:
            raise RuntimeError(f"{schema.__name__} contains identity fields: {leaked}")


def _selected_runs(
    partition_starts: np.ndarray,
    partition_ends: np.ndarray,
    partition_tokens: np.ndarray,
    selected_token: int,
    signal_length: int,
) -> list[tuple[int, int]]:
    starts = np.asarray(partition_starts, dtype=np.int64)
    ends = np.asarray(partition_ends, dtype=np.int64)
    tokens = np.asarray(partition_tokens, dtype=np.int64)
    if starts.ndim != 1 or ends.shape != starts.shape or tokens.shape != starts.shape:
        raise ValueError("Partition starts, ends, and tokens must be equal 1D arrays.")
    if len(starts) == 0 or np.any(starts < 0) or np.any(ends <= starts):
        raise ValueError("Partitions must be non-empty positive half-open intervals.")
    order = np.argsort(starts, kind="stable")
    starts, ends, tokens = starts[order], ends[order], tokens[order]
    if np.any(starts[1:] < ends[:-1]):
        raise ValueError("Partition intervals overlap.")
    if int(ends.max()) > int(signal_length):
        raise ValueError("Partition extends beyond the visible raw trial span.")
    selected = np.flatnonzero(tokens == int(selected_token))
    if len(selected) == 0:
        raise ValueError(f"Selected token {selected_token} is absent from the trial.")

    runs: list[tuple[int, int]] = []
    for position in selected.tolist():
        start, end = int(starts[position]), int(ends[position])
        if runs and start == runs[-1][1]:
            runs[-1] = (runs[-1][0], end)
        else:
            runs.append((start, end))
    return runs


def motion_components_from_token_partitions(
    sensor: np.ndarray,
    partition_starts: np.ndarray,
    partition_ends: np.ndarray,
    partition_tokens: np.ndarray,
    selected_token: int,
) -> np.ndarray:
    """Return robust acceleration-deviation and gyroscope-magnitude features.

    The acceleration component removes each selected run's median vector, so
    different gravity orientations do not by themselves make a static trial
    look dynamic.  Both run summaries are duration weighted.  Their physical
    units are intentionally kept separate and are scaled using train data in
    :func:`fit_hierarchical_gate`.
    """

    signal = np.asarray(sensor, dtype=np.float64)
    if signal.ndim != 2 or signal.shape[0] < 6 or signal.shape[1] < 1:
        raise ValueError(f"sensor must have shape [>=6,T], got {signal.shape}.")
    if not np.all(np.isfinite(signal)):
        raise ValueError("sensor must contain only finite values.")
    runs = _selected_runs(
        partition_starts,
        partition_ends,
        partition_tokens,
        selected_token=int(selected_token),
        signal_length=signal.shape[1],
    )
    components = []
    durations = []
    for start, end in runs:
        acceleration = signal[:3, start:end]
        gyroscope = signal[3:6, start:end]
        gravity = np.median(acceleration, axis=1, keepdims=True)
        acceleration_deviation = np.linalg.norm(acceleration - gravity, axis=0)
        gyroscope_magnitude = np.linalg.norm(gyroscope, axis=0)
        components.append(
            [
                float(np.median(acceleration_deviation)),
                float(np.median(gyroscope_magnitude)),
            ]
        )
        durations.append(end - start)
    result = np.average(
        np.asarray(components, dtype=np.float64),
        axis=0,
        weights=np.asarray(durations, dtype=np.float64),
    )
    if result.shape != (2,) or not np.all(np.isfinite(result)) or np.any(result < 0):
        raise RuntimeError("Motion-energy components are invalid.")
    return result.astype(np.float32)


def _positive_p95(values: np.ndarray) -> np.ndarray:
    matrix = np.asarray(values, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[1] != 2 or len(matrix) == 0:
        raise ValueError("Motion components must have shape [N,2].")
    scales = np.ones(2, dtype=np.float64)
    for column in range(2):
        positive = matrix[:, column][matrix[:, column] > 0]
        if len(positive):
            scales[column] = max(float(np.percentile(positive, 95.0)), EPS)
    return scales


def _residual_assignments_and_distances(
    model: SecondaryCodebook,
    occurrences: Sequence[UnlabelledTokenOccurrence],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    assignments = assign_secondary_codebook(model, occurrences)
    medoids = [
        UnlabelledTokenOccurrence(
            trial_global_id=int(model.medoid_trial_ids[child]),
            coarse_token=int(model.coarse_token),
            token_fraction=1.0,
            duration_samples=1,
            mean_quantization_residual=model.medoid_residuals[child],
            gravity_direction=model.medoid_gravity[child],
        )
        for child in (0, 1)
    ]
    distance = occurrence_distance_matrix(
        occurrences,
        medoids,
        arm="U1_residual",
        residual_scale=float(model.residual_scale),
    )
    assigned = distance[np.arange(len(occurrences)), assignments]
    return assignments, assigned, distance


def fit_hierarchical_gate(
    occurrences: Sequence[UnlabelledTokenOccurrence],
    motion_components_by_trial: Mapping[int, np.ndarray],
    static_radius_quantile: float = 0.95,
    minimum_token_fraction: float = 0.50,
    minimum_motion_energy_ratio: float = 2.0,
    minimum_motion_energy_gap: float = 0.25,
    restarts: int = 50,
    seed: int = 500,
) -> HierarchicalGateModel:
    """Fit the residual gate and gravity child codebook without class labels."""

    assert_gate_schema_is_label_free()
    validate_occurrences(occurrences)
    quantile = float(static_radius_quantile)
    token_fraction_threshold = float(minimum_token_fraction)
    minimum_ratio = float(minimum_motion_energy_ratio)
    minimum_gap = float(minimum_motion_energy_gap)
    if not 0.0 < quantile <= 1.0:
        raise ValueError("static_radius_quantile must lie in (0,1].")
    if not 0.0 < token_fraction_threshold <= 1.0:
        raise ValueError("minimum_token_fraction must lie in (0,1].")
    if not np.isfinite(minimum_ratio) or minimum_ratio < 1.0:
        raise ValueError("minimum_motion_energy_ratio must be finite and >=1.")
    if not np.isfinite(minimum_gap) or minimum_gap < 0.0:
        raise ValueError("minimum_motion_energy_gap must be finite and non-negative.")
    trial_ids = tuple(int(item.trial_global_id) for item in occurrences)
    below_support = [
        int(item.trial_global_id)
        for item in occurrences
        if float(item.token_fraction) + 1e-12 < token_fraction_threshold
    ]
    if below_support:
        raise ValueError(
            "Every gate-fit occurrence must satisfy minimum_token_fraction; "
            f"violations={below_support}."
        )
    if set(int(value) for value in motion_components_by_trial) != set(trial_ids):
        raise ValueError("Motion-component keys must exactly match fit trial ids.")
    motion = np.asarray(
        [motion_components_by_trial[trial_id] for trial_id in trial_ids],
        dtype=np.float64,
    )
    if motion.shape != (len(occurrences), 2) or not np.all(np.isfinite(motion)):
        raise ValueError("Every fit occurrence needs one finite two-component motion vector.")
    if np.any(motion < 0):
        raise ValueError("Motion components must be non-negative.")

    residual_model = fit_secondary_codebook(
        occurrences,
        arm="U1_residual",
        restarts=int(restarts),
        seed=int(seed),
    )
    residual_children, assigned_distances, _ = _residual_assignments_and_distances(
        residual_model, occurrences
    )
    scales = _positive_p95(motion)
    normalized_motion = motion / scales[None, :]
    motion_scores = np.mean(normalized_motion, axis=1)
    child_component_medians = np.asarray(
        [
            np.median(normalized_motion[residual_children == child], axis=0)
            for child in (0, 1)
        ],
        dtype=np.float64,
    )
    child_medians = tuple(
        float(np.median(motion_scores[residual_children == child]))
        for child in (0, 1)
    )
    if np.isclose(child_medians[0], child_medians[1], rtol=0.0, atol=1e-12):
        static_child = 0
    else:
        static_child = int(np.argmin(np.asarray(child_medians, dtype=np.float64)))
    lower_energy = float(min(child_medians))
    higher_energy = float(max(child_medians))
    motion_energy_ratio = float((higher_energy + EPS) / (lower_energy + EPS))
    motion_energy_gap = float(higher_energy - lower_energy)
    static_positions = np.flatnonzero(residual_children == static_child)
    static_radius = float(
        np.quantile(assigned_distances[static_positions], quantile)
    )
    confident_positions = static_positions[
        assigned_distances[static_positions] <= static_radius + 1e-12
    ]
    disable_reasons = []
    if motion_energy_ratio + 1e-12 < minimum_ratio:
        disable_reasons.append("motion_energy_ratio_below_threshold")
    if motion_energy_gap + 1e-12 < minimum_gap:
        disable_reasons.append("motion_energy_gap_below_threshold")
    dynamic_child = 1 - static_child
    if not np.all(
        child_component_medians[static_child]
        <= child_component_medians[dynamic_child] + 1e-12
    ):
        disable_reasons.append("motion_component_directions_disagree")
    if len(static_positions) < 2:
        disable_reasons.append("static_residual_child_has_fewer_than_two_trials")
    if len(confident_positions) < 2:
        disable_reasons.append("confidence_radius_has_fewer_than_two_static_trials")
    gate_enabled = not disable_reasons
    gravity_model = None
    if gate_enabled:
        confident_occurrences = [
            occurrences[int(position)] for position in confident_positions
        ]
        try:
            gravity_model = fit_secondary_codebook(
                confident_occurrences,
                arm="U2_gravity",
                restarts=int(restarts),
                seed=int(seed),
            )
        except RuntimeError as error:
            gate_enabled = False
            disable_reasons.append(f"gravity_codebook_unavailable:{error}")
    active_positions = confident_positions if gate_enabled else np.asarray([], dtype=np.int64)
    return HierarchicalGateModel(
        coarse_token=int(residual_model.coarse_token),
        residual_codebook=residual_model,
        gravity_codebook=gravity_model,
        gate_enabled=bool(gate_enabled),
        gate_disable_reason=(";".join(disable_reasons) if disable_reasons else None),
        static_residual_child=static_child,
        static_radius=static_radius,
        static_radius_quantile=quantile,
        minimum_token_fraction=token_fraction_threshold,
        minimum_motion_energy_ratio=minimum_ratio,
        minimum_motion_energy_gap=minimum_gap,
        motion_energy_ratio=motion_energy_ratio,
        motion_energy_gap=motion_energy_gap,
        motion_component_scales=scales.astype(np.float32),
        residual_child_motion_component_medians=child_component_medians.astype(
            np.float32
        ),
        residual_child_motion_medians=child_medians,
        fit_trial_ids=trial_ids,
        confident_static_fit_trial_ids=tuple(
            int(occurrences[int(position)].trial_global_id)
            for position in active_positions
        ),
    )


def assign_hierarchical_gate(
    model: HierarchicalGateModel,
    occurrences: Sequence[UnlabelledTokenOccurrence],
) -> HierarchicalAssignments:
    """Freeze hierarchical states for arbitrary train or test occurrences."""

    assert_gate_schema_is_label_free()
    validate_occurrences(occurrences, expected_token=model.coarse_token)
    residual_children, assigned_distances, _ = _residual_assignments_and_distances(
        model.residual_codebook, occurrences
    )
    confident_static = (
        (residual_children == int(model.static_residual_child))
        & (assigned_distances <= float(model.static_radius) + 1e-12)
        & (
            np.asarray(
                [float(item.token_fraction) for item in occurrences],
                dtype=np.float64,
            )
            + 1e-12
            >= float(model.minimum_token_fraction)
        )
    )
    if not model.gate_enabled:
        confident_static[:] = False
    gravity_children = np.full(len(occurrences), -1, dtype=np.int64)
    selected_positions = np.flatnonzero(confident_static)
    if len(selected_positions):
        if model.gravity_codebook is None:
            raise RuntimeError("An enabled hierarchy has no gravity codebook.")
        selected_occurrences = [occurrences[int(position)] for position in selected_positions]
        gravity_children[selected_positions] = assign_secondary_codebook(
            model.gravity_codebook, selected_occurrences
        )
    gate_states = np.zeros(len(occurrences), dtype=np.int64)
    gate_states[selected_positions] = 1 + gravity_children[selected_positions]
    if not set(np.unique(gate_states).tolist()).issubset(GATE_STATES):
        raise RuntimeError("Hierarchical gate emitted an invalid state.")
    return HierarchicalAssignments(
        trial_ids=np.asarray(
            [int(item.trial_global_id) for item in occurrences], dtype=np.int64
        ),
        residual_children=residual_children.astype(np.int64),
        assigned_residual_distances=assigned_distances.astype(np.float64),
        confident_static=confident_static.astype(bool),
        gravity_children=gravity_children,
        gate_states=gate_states,
    )


def hierarchical_duration_histograms(
    trial_ids: Sequence[int],
    trial_token_durations: Mapping[int, Mapping[int, float]],
    primitive_num: int,
    selected_token: int,
    gate_state_by_trial: Mapping[int, int],
) -> np.ndarray:
    """Build K+2 histograms with the original token as a safe fallback."""

    dimension = int(primitive_num) + 2
    matrix = np.zeros((len(trial_ids), dimension), dtype=np.float64)
    for row, trial_id_value in enumerate(trial_ids):
        trial_id = int(trial_id_value)
        token_durations = trial_token_durations[trial_id]
        total = float(sum(float(value) for value in token_durations.values()))
        if total <= 0:
            raise ValueError(f"Trial {trial_id} has no positive token duration.")
        for token_value, duration_value in token_durations.items():
            token = int(token_value)
            duration = float(duration_value)
            if token < 0 or token >= int(primitive_num) or duration <= 0:
                raise ValueError(f"Trial {trial_id} has invalid token/duration {token}:{duration}.")
            column = token
            if token == int(selected_token):
                if trial_id not in gate_state_by_trial:
                    raise ValueError(f"Trial {trial_id} lacks a frozen gate state.")
                state = int(gate_state_by_trial[trial_id])
                if state not in GATE_STATES:
                    raise ValueError(f"Trial {trial_id} has invalid gate state {state}.")
                if state > 0:
                    column = int(primitive_num) + state - 1
            matrix[row, column] += duration / total
    if not np.allclose(matrix.sum(axis=1), 1.0, atol=1e-8):
        raise RuntimeError("Hierarchical duration histograms do not sum to one.")
    return matrix.astype(np.float32)


def hierarchical_gate_diagnostics(
    model: HierarchicalGateModel,
    fit_occurrences: Sequence[UnlabelledTokenOccurrence],
) -> dict:
    """Return a JSON-ready train-only audit of the fitted hierarchy."""

    observed_ids = tuple(int(item.trial_global_id) for item in fit_occurrences)
    if observed_ids != model.fit_trial_ids:
        raise ValueError("Gate diagnostics require the exact fitting occurrence order.")
    assignments = assign_hierarchical_gate(model, fit_occurrences)
    state_counts = np.bincount(assignments.gate_states, minlength=3)
    return {
        "coarse_token": int(model.coarse_token),
        "gate_enabled": bool(model.gate_enabled),
        "gate_disable_reason": model.gate_disable_reason,
        "static_residual_child": int(model.static_residual_child),
        "static_radius": float(model.static_radius),
        "static_radius_quantile": float(model.static_radius_quantile),
        "minimum_token_fraction": float(model.minimum_token_fraction),
        "minimum_motion_energy_ratio": float(model.minimum_motion_energy_ratio),
        "minimum_motion_energy_gap": float(model.minimum_motion_energy_gap),
        "motion_energy_ratio": float(model.motion_energy_ratio),
        "motion_energy_gap": float(model.motion_energy_gap),
        "motion_component_names": [
            "median_acceleration_deviation",
            "median_gyroscope_magnitude",
        ],
        "motion_component_p95_scales": model.motion_component_scales.tolist(),
        "residual_child_normalized_motion_component_medians": (
            model.residual_child_motion_component_medians.tolist()
        ),
        "residual_child_motion_score_medians": list(
            model.residual_child_motion_medians
        ),
        "fit_trial_count": len(model.fit_trial_ids),
        "fit_trial_ids": list(model.fit_trial_ids),
        "confident_static_fit_trial_ids": list(
            model.confident_static_fit_trial_ids
        ),
        "confident_static_fit_count": len(model.confident_static_fit_trial_ids),
        "fit_gate_state_counts": state_counts.astype(int).tolist(),
        "gate_state_meanings": {str(key): value for key, value in GATE_STATES.items()},
        "residual_codebook_fit": secondary_fit_diagnostics(
            model.residual_codebook, fit_occurrences
        ),
        "gravity_codebook_fit": (
            secondary_fit_diagnostics(
                model.gravity_codebook,
                [
                    item
                    for item in fit_occurrences
                    if int(item.trial_global_id)
                    in set(model.confident_static_fit_trial_ids)
                ],
            )
            if model.gravity_codebook is not None
            else None
        ),
        "fit_uses_activity_labels": False,
        "static_child_naming_uses_train_only_raw_motion": True,
        "test_motion_energy_used_for_assignment": False,
    }


__all__ = [
    "GATE_STATES",
    "HierarchicalAssignments",
    "HierarchicalGateModel",
    "assert_gate_schema_is_label_free",
    "assign_hierarchical_gate",
    "fit_hierarchical_gate",
    "hierarchical_duration_histograms",
    "hierarchical_gate_diagnostics",
    "motion_components_from_token_partitions",
]
