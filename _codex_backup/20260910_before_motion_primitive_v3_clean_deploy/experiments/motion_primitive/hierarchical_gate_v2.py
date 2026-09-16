"""Conservative train-only stability wrapper for the v1 hierarchical gate.

This module deliberately leaves :mod:`hierarchical_gate` unchanged because the
v1 experiment artifacts fingerprint that implementation.  The v2 wrapper adds
three fail-closed requirements before a fitted v1 hierarchy may route a trial:

* both residual children and both gravity children need robust sample support;
* when train-subject identities are supplied, every child needs support from at
  least two subjects; and
* an optional leave-one-candidate-out audit must show that the train-only
  hierarchy is stable.

Activity labels and activity names are not accepted by any fitting function.
Subject identity, when available, is used only to reject single-subject child
clusters and is never used as a clustering feature.
"""

from __future__ import annotations

from dataclasses import dataclass, fields, replace
import math
from numbers import Integral
from typing import Mapping, Sequence

import numpy as np

from experiments.motion_primitive.hierarchical_gate import (
    GATE_STATES,
    HierarchicalAssignments,
    HierarchicalGateModel,
    assign_hierarchical_gate,
    fit_hierarchical_gate,
)
from experiments.motion_primitive.online_secondary_codebook import (
    UnlabelledTokenOccurrence,
    validate_occurrences,
)


MINIMUM_CHILD_COUNT = 3
MINIMUM_CHILD_FRACTION = 0.15
MINIMUM_SUBJECTS_PER_CHILD = 2
DEFAULT_LOO_ENABLED_FRACTION = 1.0
DEFAULT_LOO_AGREEMENT = 0.90


@dataclass(frozen=True)
class HierarchicalGateV2Model:
    """A v1 model plus conservative support and stability decisions."""

    coarse_token: int
    base_model: HierarchicalGateModel | None
    gate_enabled: bool
    gate_disable_reasons: tuple[str, ...]
    minimum_child_count: int
    minimum_child_fraction: float
    minimum_subjects_per_child: int
    fit_trial_ids: tuple[int, ...]
    support_audit: Mapping[str, object]
    leave_one_out_audit: Mapping[str, object] | None

    @property
    def gate_disable_reason(self) -> str | None:
        return ";".join(self.gate_disable_reasons) if self.gate_disable_reasons else None


def assert_gate_v2_schema_is_label_free() -> None:
    """Reject accidental activity identity in the v2 fitting model."""

    forbidden = ("label", "activity", "class", "name")
    names = {item.name.lower() for item in fields(HierarchicalGateV2Model)}
    leaked = sorted(
        name for name in names if any(fragment in name for fragment in forbidden)
    )
    if leaked:
        raise RuntimeError(
            "HierarchicalGateV2Model contains activity-identity fields: "
            f"{leaked}."
        )


def _validated_thresholds(
    minimum_child_count: int,
    minimum_child_fraction: float,
    minimum_subjects_per_child: int,
) -> tuple[int, float, int]:
    count = int(minimum_child_count)
    fraction = float(minimum_child_fraction)
    subjects = int(minimum_subjects_per_child)
    if count < MINIMUM_CHILD_COUNT:
        raise ValueError(
            f"minimum_child_count must be at least {MINIMUM_CHILD_COUNT}."
        )
    if not np.isfinite(fraction) or not MINIMUM_CHILD_FRACTION <= fraction <= 0.5:
        raise ValueError(
            "minimum_child_fraction must be finite and lie in [0.15,0.5]."
        )
    if subjects < MINIMUM_SUBJECTS_PER_CHILD:
        raise ValueError(
            "minimum_subjects_per_child must be at least two."
        )
    return count, fraction, subjects


def _required_child_count(total: int, count: int, fraction: float) -> int:
    return max(int(count), int(math.ceil(float(fraction) * int(total))))


def _normalize_subjects(
    subjects_by_trial: Mapping[int, int] | None,
    trial_ids: Sequence[int],
) -> dict[int, int] | None:
    if subjects_by_trial is None:
        return None
    expected = {int(value) for value in trial_ids}
    result: dict[int, int] = {}
    for raw_trial_id, raw in subjects_by_trial.items():
        try:
            trial_id = int(raw_trial_id)
        except (TypeError, ValueError) as error:
            raise ValueError("Every subject-map trial id must be integer-like.") from error
        if trial_id in result:
            raise ValueError("subjects_by_trial contains duplicate integer trial ids.")
        if isinstance(raw, bool) or not isinstance(raw, Integral):
            raise ValueError("Every subject id must be an integer, not a boolean.")
        result[trial_id] = int(raw)
    if set(result) != expected:
        raise ValueError(
            "subjects_by_trial keys must exactly match the train-only fit trial ids."
        )
    return result


def _child_subject_counts(
    trial_ids: Sequence[int],
    child_ids: Sequence[int],
    subjects: Mapping[int, int] | None,
) -> list[int] | None:
    if subjects is None:
        return None
    ids = np.asarray(trial_ids, dtype=np.int64)
    children = np.asarray(child_ids, dtype=np.int64)
    if children.shape != ids.shape:
        raise RuntimeError("Trial and child arrays disagree while auditing subjects.")
    return [
        len({int(subjects[int(ids[index])]) for index in np.flatnonzero(children == child)})
        for child in (0, 1)
    ]


def _fit_without_leave_one_out(
    occurrences: Sequence[UnlabelledTokenOccurrence],
    motion_components_by_trial: Mapping[int, np.ndarray],
    subjects_by_trial: Mapping[int, int] | None,
    *,
    static_radius_quantile: float,
    minimum_token_fraction: float,
    minimum_motion_energy_ratio: float,
    minimum_motion_energy_gap: float,
    minimum_child_count: int,
    minimum_child_fraction: float,
    minimum_subjects_per_child: int,
    restarts: int,
    seed: int,
) -> HierarchicalGateV2Model:
    """Fit and audit support without recursively invoking the LOO audit."""

    assert_gate_v2_schema_is_label_free()
    validate_occurrences(occurrences)
    count, fraction, subject_minimum = _validated_thresholds(
        minimum_child_count,
        minimum_child_fraction,
        minimum_subjects_per_child,
    )
    trial_ids = tuple(int(item.trial_global_id) for item in occurrences)
    subjects = _normalize_subjects(subjects_by_trial, trial_ids)
    coarse_tokens = {int(item.coarse_token) for item in occurrences}
    if len(coarse_tokens) != 1:
        raise ValueError("A v2 gate may fit exactly one coarse token.")
    coarse_token = int(next(iter(coarse_tokens)))

    try:
        base = fit_hierarchical_gate(
            occurrences,
            motion_components_by_trial,
            static_radius_quantile=float(static_radius_quantile),
            minimum_token_fraction=float(minimum_token_fraction),
            minimum_motion_energy_ratio=float(minimum_motion_energy_ratio),
            minimum_motion_energy_gap=float(minimum_motion_energy_gap),
            restarts=int(restarts),
            seed=int(seed),
        )
    except RuntimeError as error:
        reason = f"base_fit_failed:{type(error).__name__}"
        audit = {
            "passed": False,
            "failure_reasons": [reason],
            "fit_trial_count": len(trial_ids),
            "subjects_available": subjects is not None,
            "minimum_child_count": count,
            "minimum_child_fraction": fraction,
            "minimum_subjects_per_child": subject_minimum,
            "base_fit_error": str(error),
        }
        return HierarchicalGateV2Model(
            coarse_token=coarse_token,
            base_model=None,
            gate_enabled=False,
            gate_disable_reasons=(reason,),
            minimum_child_count=count,
            minimum_child_fraction=fraction,
            minimum_subjects_per_child=subject_minimum,
            fit_trial_ids=trial_ids,
            support_audit=audit,
            leave_one_out_audit=None,
        )

    reasons: list[str] = []
    if not base.gate_enabled:
        reasons.append(f"base_gate_disabled:{base.gate_disable_reason}")

    residual_children = np.asarray(
        base.residual_codebook.train_child_ids, dtype=np.int64
    )
    residual_counts = np.bincount(residual_children, minlength=2)[:2]
    residual_required = _required_child_count(len(occurrences), count, fraction)
    if np.any(residual_counts < residual_required):
        reasons.append("residual_child_below_minimum_support")
    residual_subject_counts = _child_subject_counts(
        trial_ids, residual_children, subjects
    )
    if residual_subject_counts is not None and min(residual_subject_counts) < subject_minimum:
        reasons.append("residual_child_below_minimum_subjects")

    component_medians = np.asarray(
        base.residual_child_motion_component_medians, dtype=np.float64
    )
    static_child = int(base.static_residual_child)
    dynamic_child = 1 - static_child
    componentwise_lower = bool(
        component_medians.shape == (2, 2)
        and np.all(np.isfinite(component_medians))
        and np.all(
            component_medians[static_child]
            <= component_medians[dynamic_child] + 1e-12
        )
        and np.any(
            component_medians[static_child]
            < component_medians[dynamic_child] - 1e-12
        )
    )
    score_medians = np.asarray(base.residual_child_motion_medians, dtype=np.float64)
    score_lower = bool(
        score_medians.shape == (2,)
        and np.all(np.isfinite(score_medians))
        and score_medians[static_child] < score_medians[dynamic_child] - 1e-12
    )
    physical_consistency = bool(componentwise_lower and score_lower)
    if not physical_consistency:
        reasons.append("residual_static_physical_inconsistency")

    gravity_counts: np.ndarray | None = None
    gravity_subject_counts: list[int] | None = None
    gravity_required: int | None = None
    gravity_fit_trial_ids: tuple[int, ...] = ()
    if base.gravity_codebook is None:
        reasons.append("gravity_codebook_unavailable")
    else:
        gravity_fit_trial_ids = tuple(
            int(value) for value in base.gravity_codebook.train_trial_ids
        )
        if gravity_fit_trial_ids != tuple(base.confident_static_fit_trial_ids):
            reasons.append("gravity_fit_ids_disagree_with_static_gate")
        gravity_children = np.asarray(
            base.gravity_codebook.train_child_ids, dtype=np.int64
        )
        gravity_counts = np.bincount(gravity_children, minlength=2)[:2]
        gravity_required = _required_child_count(
            len(gravity_fit_trial_ids), count, fraction
        )
        if np.any(gravity_counts < gravity_required):
            reasons.append("gravity_child_below_minimum_support")
        gravity_subject_counts = _child_subject_counts(
            gravity_fit_trial_ids, gravity_children, subjects
        )
        if (
            gravity_subject_counts is not None
            and min(gravity_subject_counts) < subject_minimum
        ):
            reasons.append("gravity_child_below_minimum_subjects")

    reasons = list(dict.fromkeys(reasons))
    audit = {
        "passed": not reasons,
        "failure_reasons": reasons,
        "fit_trial_count": len(trial_ids),
        "subjects_available": subjects is not None,
        "threshold_policy": "max(minimum_count, ceil(minimum_fraction * stage_N))",
        "minimum_child_count": count,
        "minimum_child_fraction": fraction,
        "minimum_subjects_per_child": subject_minimum,
        "base_gate_enabled": bool(base.gate_enabled),
        "base_gate_disable_reason": base.gate_disable_reason,
        "residual_required_per_child": residual_required,
        "residual_child_counts": residual_counts.astype(int).tolist(),
        "residual_child_subject_counts": residual_subject_counts,
        "static_residual_child": static_child,
        "physical_consistency": {
            "passed": physical_consistency,
            "componentwise_lower_motion": componentwise_lower,
            "aggregate_score_strictly_lower": score_lower,
            "normalized_component_medians": component_medians.tolist(),
            "motion_score_medians": score_medians.tolist(),
        },
        "confident_static_fit_count": len(gravity_fit_trial_ids),
        "gravity_required_per_child": gravity_required,
        "gravity_child_counts": (
            gravity_counts.astype(int).tolist() if gravity_counts is not None else None
        ),
        "gravity_child_subject_counts": gravity_subject_counts,
        "fit_uses_activity_labels_or_names": False,
        "support_and_physical_thresholds_use_train_only": True,
    }
    return HierarchicalGateV2Model(
        coarse_token=coarse_token,
        base_model=base,
        gate_enabled=not reasons,
        gate_disable_reasons=tuple(reasons),
        minimum_child_count=count,
        minimum_child_fraction=fraction,
        minimum_subjects_per_child=subject_minimum,
        fit_trial_ids=trial_ids,
        support_audit=audit,
        leave_one_out_audit=None,
    )


def _binary_partition_agreement(left: np.ndarray, right: np.ndarray) -> float | None:
    first = np.asarray(left, dtype=np.int64)
    second = np.asarray(right, dtype=np.int64)
    if first.ndim != 1 or second.shape != first.shape or len(first) < 2:
        return None
    if len(np.unique(first)) != 2 or len(np.unique(second)) != 2:
        return None
    direct = float(np.mean(first == second))
    flipped = float(np.mean(first == (1 - second)))
    return max(direct, flipped)


def _gravity_orientation(
    full_assignments: HierarchicalAssignments,
    reduced_assignments: HierarchicalAssignments,
) -> int | None:
    common = np.asarray(full_assignments.confident_static, dtype=bool) & np.asarray(
        reduced_assignments.confident_static, dtype=bool
    )
    first = np.asarray(full_assignments.gravity_children, dtype=np.int64)[common]
    second = np.asarray(reduced_assignments.gravity_children, dtype=np.int64)[common]
    if len(first) < 2 or len(np.unique(first)) != 2 or len(np.unique(second)) != 2:
        return None
    direct = float(np.mean(first == second))
    flipped = float(np.mean(first == (1 - second)))
    return 0 if direct >= flipped else 1


def _leave_one_out_from_full(
    full: HierarchicalGateV2Model,
    occurrences: Sequence[UnlabelledTokenOccurrence],
    motion_components_by_trial: Mapping[int, np.ndarray],
    subjects_by_trial: Mapping[int, int] | None,
    *,
    static_radius_quantile: float,
    minimum_token_fraction: float,
    minimum_motion_energy_ratio: float,
    minimum_motion_energy_gap: float,
    minimum_child_count: int,
    minimum_child_fraction: float,
    minimum_subjects_per_child: int,
    minimum_enabled_fraction: float,
    minimum_partition_agreement: float,
    minimum_heldout_agreement: float,
    restarts: int,
    seed: int,
) -> dict:
    if not full.gate_enabled or full.base_model is None:
        return {
            "passed": False,
            "status": "not_run_full_fit_failed_structural_checks",
            "replicate_count": 0,
            "full_fit_failure_reasons": list(full.gate_disable_reasons),
            "uses_only_train_candidates": True,
        }

    trial_ids = [int(item.trial_global_id) for item in occurrences]
    subjects = _normalize_subjects(subjects_by_trial, trial_ids)
    full_all = assign_hierarchical_gate(full.base_model, occurrences)
    replicas = []
    structural_enabled = []
    residual_agreements: list[float] = []
    gravity_agreements: list[float] = []
    heldout_route_matches: list[bool] = []
    heldout_gravity_matches: list[bool] = []

    for omitted_position, omitted in enumerate(occurrences):
        remaining = [
            item for position, item in enumerate(occurrences) if position != omitted_position
        ]
        remaining_ids = [int(item.trial_global_id) for item in remaining]
        reduced_motion = {
            trial_id: motion_components_by_trial[trial_id]
            for trial_id in remaining_ids
        }
        reduced_subjects = (
            {trial_id: subjects[trial_id] for trial_id in remaining_ids}
            if subjects is not None
            else None
        )
        reduced = _fit_without_leave_one_out(
            remaining,
            reduced_motion,
            reduced_subjects,
            static_radius_quantile=static_radius_quantile,
            minimum_token_fraction=minimum_token_fraction,
            minimum_motion_energy_ratio=minimum_motion_energy_ratio,
            minimum_motion_energy_gap=minimum_motion_energy_gap,
            minimum_child_count=minimum_child_count,
            minimum_child_fraction=minimum_child_fraction,
            minimum_subjects_per_child=minimum_subjects_per_child,
            restarts=restarts,
            seed=seed,
        )
        enabled = bool(reduced.gate_enabled and reduced.base_model is not None)
        structural_enabled.append(enabled)
        replica = {
            "omitted_trial_id": int(omitted.trial_global_id),
            "structural_gate_enabled": enabled,
            "failure_reasons": list(reduced.gate_disable_reasons),
            "residual_static_partition_agreement": None,
            "gravity_partition_agreement": None,
            "heldout_route_matches_full": None,
            "heldout_gravity_matches_full": None,
        }
        if not enabled:
            replicas.append(replica)
            continue

        full_remaining = assign_hierarchical_gate(
            full.base_model, remaining
        )
        reduced_remaining = assign_hierarchical_gate(
            reduced.base_model, remaining
        )
        full_static = (
            full_remaining.residual_children
            == int(full.base_model.static_residual_child)
        ).astype(np.int64)
        reduced_static = (
            reduced_remaining.residual_children
            == int(reduced.base_model.static_residual_child)
        ).astype(np.int64)
        residual_agreement = float(np.mean(full_static == reduced_static))
        residual_agreements.append(residual_agreement)
        replica["residual_static_partition_agreement"] = residual_agreement

        gravity_agreement = _binary_partition_agreement(
            full_remaining.gravity_children[
                full_remaining.confident_static
                & reduced_remaining.confident_static
            ],
            reduced_remaining.gravity_children[
                full_remaining.confident_static
                & reduced_remaining.confident_static
            ],
        )
        if gravity_agreement is not None:
            gravity_agreements.append(float(gravity_agreement))
        replica["gravity_partition_agreement"] = gravity_agreement

        heldout = assign_hierarchical_gate(reduced.base_model, [omitted])
        full_routed = bool(full_all.confident_static[omitted_position])
        reduced_routed = bool(heldout.confident_static[0])
        route_match = full_routed == reduced_routed
        heldout_route_matches.append(route_match)
        replica["heldout_route_matches_full"] = route_match

        gravity_match: bool | None = None
        if full_routed and reduced_routed:
            orientation = _gravity_orientation(full_remaining, reduced_remaining)
            if orientation is not None:
                reduced_child = int(heldout.gravity_children[0])
                if orientation == 1:
                    reduced_child = 1 - reduced_child
                gravity_match = reduced_child == int(
                    full_all.gravity_children[omitted_position]
                )
                heldout_gravity_matches.append(gravity_match)
        replica["heldout_gravity_matches_full"] = gravity_match
        replicas.append(replica)

    enabled_fraction = float(np.mean(structural_enabled)) if replicas else 0.0
    minimum_residual = min(residual_agreements) if residual_agreements else 0.0
    minimum_gravity = min(gravity_agreements) if gravity_agreements else 0.0
    route_agreement = (
        float(np.mean(heldout_route_matches)) if heldout_route_matches else 0.0
    )
    gravity_heldout_agreement = (
        float(np.mean(heldout_gravity_matches))
        if heldout_gravity_matches
        else 0.0
    )
    checks = {
        "all_replicates_structurally_supported": enabled_fraction
        + 1e-12
        >= float(minimum_enabled_fraction),
        "residual_static_partition_stable": minimum_residual
        + 1e-12
        >= float(minimum_partition_agreement),
        "gravity_partition_stable": minimum_gravity
        + 1e-12
        >= float(minimum_partition_agreement),
        "heldout_route_stable": route_agreement
        + 1e-12
        >= float(minimum_heldout_agreement),
        "heldout_gravity_stable": gravity_heldout_agreement
        + 1e-12
        >= float(minimum_heldout_agreement),
    }
    return {
        "passed": bool(all(checks.values())),
        "status": "completed",
        "replicate_count": len(replicas),
        "minimum_enabled_fraction_threshold": float(minimum_enabled_fraction),
        "minimum_partition_agreement_threshold": float(
            minimum_partition_agreement
        ),
        "minimum_heldout_agreement_threshold": float(minimum_heldout_agreement),
        "enabled_fraction": enabled_fraction,
        "minimum_residual_static_partition_agreement": minimum_residual,
        "minimum_gravity_partition_agreement": minimum_gravity,
        "heldout_route_agreement": route_agreement,
        "heldout_gravity_agreement_on_jointly_routed": gravity_heldout_agreement,
        "jointly_routed_heldout_count": len(heldout_gravity_matches),
        "checks": checks,
        "replicates": replicas,
        "uses_only_train_candidates": True,
        "uses_test_occurrences": False,
        "uses_activity_labels_or_names": False,
    }


def _validated_agreement(value: float, name: str, minimum: float) -> float:
    parsed = float(value)
    if not np.isfinite(parsed) or not float(minimum) <= parsed <= 1.0:
        raise ValueError(
            f"{name} must be finite and lie in [{float(minimum):.2f},1]."
        )
    return parsed


def leave_one_candidate_out_stability_audit(
    occurrences: Sequence[UnlabelledTokenOccurrence],
    motion_components_by_trial: Mapping[int, np.ndarray],
    subjects_by_trial: Mapping[int, int] | None = None,
    *,
    static_radius_quantile: float = 0.95,
    minimum_token_fraction: float = 0.50,
    minimum_motion_energy_ratio: float = 2.0,
    minimum_motion_energy_gap: float = 0.25,
    minimum_child_count: int = MINIMUM_CHILD_COUNT,
    minimum_child_fraction: float = MINIMUM_CHILD_FRACTION,
    minimum_subjects_per_child: int = MINIMUM_SUBJECTS_PER_CHILD,
    minimum_enabled_fraction: float = DEFAULT_LOO_ENABLED_FRACTION,
    minimum_partition_agreement: float = DEFAULT_LOO_AGREEMENT,
    minimum_heldout_agreement: float = DEFAULT_LOO_AGREEMENT,
    restarts: int = 50,
    seed: int = 500,
) -> dict:
    """Audit sensitivity to deleting each train candidate exactly once."""

    enabled_threshold = _validated_agreement(
        minimum_enabled_fraction,
        "minimum_enabled_fraction",
        DEFAULT_LOO_ENABLED_FRACTION,
    )
    partition_threshold = _validated_agreement(
        minimum_partition_agreement,
        "minimum_partition_agreement",
        DEFAULT_LOO_AGREEMENT,
    )
    heldout_threshold = _validated_agreement(
        minimum_heldout_agreement,
        "minimum_heldout_agreement",
        DEFAULT_LOO_AGREEMENT,
    )
    full = _fit_without_leave_one_out(
        occurrences,
        motion_components_by_trial,
        subjects_by_trial,
        static_radius_quantile=static_radius_quantile,
        minimum_token_fraction=minimum_token_fraction,
        minimum_motion_energy_ratio=minimum_motion_energy_ratio,
        minimum_motion_energy_gap=minimum_motion_energy_gap,
        minimum_child_count=minimum_child_count,
        minimum_child_fraction=minimum_child_fraction,
        minimum_subjects_per_child=minimum_subjects_per_child,
        restarts=restarts,
        seed=seed,
    )
    return _leave_one_out_from_full(
        full,
        occurrences,
        motion_components_by_trial,
        subjects_by_trial,
        static_radius_quantile=static_radius_quantile,
        minimum_token_fraction=minimum_token_fraction,
        minimum_motion_energy_ratio=minimum_motion_energy_ratio,
        minimum_motion_energy_gap=minimum_motion_energy_gap,
        minimum_child_count=minimum_child_count,
        minimum_child_fraction=minimum_child_fraction,
        minimum_subjects_per_child=minimum_subjects_per_child,
        minimum_enabled_fraction=enabled_threshold,
        minimum_partition_agreement=partition_threshold,
        minimum_heldout_agreement=heldout_threshold,
        restarts=restarts,
        seed=seed,
    )


def fit_hierarchical_gate_v2(
    occurrences: Sequence[UnlabelledTokenOccurrence],
    motion_components_by_trial: Mapping[int, np.ndarray],
    subjects_by_trial: Mapping[int, int] | None = None,
    *,
    static_radius_quantile: float = 0.95,
    minimum_token_fraction: float = 0.50,
    minimum_motion_energy_ratio: float = 2.0,
    minimum_motion_energy_gap: float = 0.25,
    minimum_child_count: int = MINIMUM_CHILD_COUNT,
    minimum_child_fraction: float = MINIMUM_CHILD_FRACTION,
    minimum_subjects_per_child: int = MINIMUM_SUBJECTS_PER_CHILD,
    require_leave_one_out: bool = True,
    minimum_loo_enabled_fraction: float = DEFAULT_LOO_ENABLED_FRACTION,
    minimum_loo_partition_agreement: float = DEFAULT_LOO_AGREEMENT,
    minimum_loo_heldout_agreement: float = DEFAULT_LOO_AGREEMENT,
    restarts: int = 50,
    seed: int = 500,
) -> HierarchicalGateV2Model:
    """Fit a hierarchy and fail closed unless every v2 audit passes."""

    structural = _fit_without_leave_one_out(
        occurrences,
        motion_components_by_trial,
        subjects_by_trial,
        static_radius_quantile=static_radius_quantile,
        minimum_token_fraction=minimum_token_fraction,
        minimum_motion_energy_ratio=minimum_motion_energy_ratio,
        minimum_motion_energy_gap=minimum_motion_energy_gap,
        minimum_child_count=minimum_child_count,
        minimum_child_fraction=minimum_child_fraction,
        minimum_subjects_per_child=minimum_subjects_per_child,
        restarts=restarts,
        seed=seed,
    )
    if not require_leave_one_out:
        return structural
    enabled_threshold = _validated_agreement(
        minimum_loo_enabled_fraction,
        "minimum_loo_enabled_fraction",
        DEFAULT_LOO_ENABLED_FRACTION,
    )
    partition_threshold = _validated_agreement(
        minimum_loo_partition_agreement,
        "minimum_loo_partition_agreement",
        DEFAULT_LOO_AGREEMENT,
    )
    heldout_threshold = _validated_agreement(
        minimum_loo_heldout_agreement,
        "minimum_loo_heldout_agreement",
        DEFAULT_LOO_AGREEMENT,
    )
    audit = _leave_one_out_from_full(
        structural,
        occurrences,
        motion_components_by_trial,
        subjects_by_trial,
        static_radius_quantile=static_radius_quantile,
        minimum_token_fraction=minimum_token_fraction,
        minimum_motion_energy_ratio=minimum_motion_energy_ratio,
        minimum_motion_energy_gap=minimum_motion_energy_gap,
        minimum_child_count=minimum_child_count,
        minimum_child_fraction=minimum_child_fraction,
        minimum_subjects_per_child=minimum_subjects_per_child,
        minimum_enabled_fraction=enabled_threshold,
        minimum_partition_agreement=partition_threshold,
        minimum_heldout_agreement=heldout_threshold,
        restarts=restarts,
        seed=seed,
    )
    reasons = list(structural.gate_disable_reasons)
    if structural.gate_enabled and not bool(audit.get("passed")):
        reasons.append("leave_one_out_stability_failed")
    reasons = list(dict.fromkeys(reasons))
    return replace(
        structural,
        gate_enabled=bool(structural.gate_enabled and audit.get("passed", False)),
        gate_disable_reasons=tuple(reasons),
        leave_one_out_audit=audit,
    )


def assign_hierarchical_gate_v2(
    model: HierarchicalGateV2Model,
    occurrences: Sequence[UnlabelledTokenOccurrence],
) -> HierarchicalAssignments:
    """Assign with exact coarse fallback whenever a v2 audit disabled the gate."""

    validate_occurrences(occurrences, expected_token=model.coarse_token)
    if model.base_model is not None:
        assignments = assign_hierarchical_gate(model.base_model, occurrences)
        if model.gate_enabled:
            return assignments
        count = len(occurrences)
        return HierarchicalAssignments(
            trial_ids=assignments.trial_ids.copy(),
            residual_children=assignments.residual_children.copy(),
            assigned_residual_distances=assignments.assigned_residual_distances.copy(),
            confident_static=np.zeros(count, dtype=bool),
            gravity_children=np.full(count, -1, dtype=np.int64),
            gate_states=np.zeros(count, dtype=np.int64),
        )
    count = len(occurrences)
    return HierarchicalAssignments(
        trial_ids=np.asarray(
            [int(item.trial_global_id) for item in occurrences], dtype=np.int64
        ),
        residual_children=np.full(count, -1, dtype=np.int64),
        assigned_residual_distances=np.full(count, np.inf, dtype=np.float64),
        confident_static=np.zeros(count, dtype=bool),
        gravity_children=np.full(count, -1, dtype=np.int64),
        gate_states=np.zeros(count, dtype=np.int64),
    )


def hierarchical_gate_v2_diagnostics(
    model: HierarchicalGateV2Model,
    fit_occurrences: Sequence[UnlabelledTokenOccurrence],
) -> dict:
    """Return a JSON-ready audit of all train-only v2 decisions."""

    observed = tuple(int(item.trial_global_id) for item in fit_occurrences)
    if observed != model.fit_trial_ids:
        raise ValueError("Diagnostics require the exact v2 fitting occurrence order.")
    assignments = assign_hierarchical_gate_v2(model, fit_occurrences)
    state_counts = np.bincount(assignments.gate_states, minlength=3)[:3]
    return {
        "coarse_token": int(model.coarse_token),
        "gate_enabled": bool(model.gate_enabled),
        "gate_disable_reason": model.gate_disable_reason,
        "gate_disable_reasons": list(model.gate_disable_reasons),
        "fit_trial_count": len(model.fit_trial_ids),
        "fit_trial_ids": list(model.fit_trial_ids),
        "fit_gate_state_counts": state_counts.astype(int).tolist(),
        "gate_state_meanings": {str(key): value for key, value in GATE_STATES.items()},
        "support_audit": dict(model.support_audit),
        "leave_one_out_audit": (
            dict(model.leave_one_out_audit)
            if model.leave_one_out_audit is not None
            else None
        ),
        "fit_uses_activity_labels_or_names": False,
        "all_v2_thresholds_are_train_only": True,
    }


__all__ = [
    "DEFAULT_LOO_AGREEMENT",
    "DEFAULT_LOO_ENABLED_FRACTION",
    "HierarchicalGateV2Model",
    "MINIMUM_CHILD_COUNT",
    "MINIMUM_CHILD_FRACTION",
    "MINIMUM_SUBJECTS_PER_CHILD",
    "assert_gate_v2_schema_is_label_free",
    "assign_hierarchical_gate_v2",
    "fit_hierarchical_gate_v2",
    "hierarchical_gate_v2_diagnostics",
    "leave_one_candidate_out_stability_audit",
]
