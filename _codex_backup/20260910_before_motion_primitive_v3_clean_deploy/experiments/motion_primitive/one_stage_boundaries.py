"""Raw-only pseudo-boundaries for one-stage motion-primitive training.

This module is deliberately independent of a pretrained/frozen feature
encoder.  It fits the robust component scaler and the q50/q90 score
thresholds exclusively on caller-supplied *training-subject, old-class*
trials.  Validation trials are transformed with that frozen calibration and
can never participate in fitting.

The high-level :func:`build_one_stage_raw_boundaries` function accepts the
``TrialExample``-shaped records used by the motion-primitive experiment.  It
assigns ``raw_scores``, ``stable_mask`` and ``change_mask`` tensors in place
and returns a completely JSON-serialisable audit document.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
from typing import Any, Sequence

import numpy as np
import torch

from experiments.motion_primitive.raw_changepoint import (
    boundary_candidate_masks,
    fit_trial_equal_robust_component_scaler,
    fit_trial_equal_score_thresholds,
    parse_raw_scales,
    raw_component_names,
    raw_formula_metadata,
    raw_multiscale_boundary_components,
    transform_raw_boundary_components,
)


SCHEMA_VERSION = 1
CALIBRATION_TYPE = "one_stage_raw_kinematic_boundary_calibration"
ANCHOR_SOURCE = "raw_kinematic_only"
LOW_QUANTILE = 0.50
HIGH_QUANTILE = 0.90


def _canonical_sha256(document: dict) -> str:
    encoded = json.dumps(
        document,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _required_integer(record: Any, name: str) -> int:
    if not hasattr(record, name):
        raise TypeError(f"Every boundary record must define {name!r}.")
    value = getattr(record, name)
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"Boundary record {name!r} must be an integer, not bool.")
    try:
        integer = int(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise TypeError(f"Boundary record {name!r} must be an integer.") from error
    if isinstance(value, (float, np.floating)) and float(value) != float(integer):
        raise TypeError(f"Boundary record {name!r} must be an exact integer.")
    return integer


def _validate_records(
    records: Sequence[Any],
    *,
    split: str,
    old_class_count: int,
    require_nonempty: bool,
) -> list[Any]:
    if split not in {"train", "validation"}:
        raise ValueError("split must be 'train' or 'validation'.")
    materialised = list(records)
    if require_nonempty and not materialised:
        raise ValueError(f"The {split} boundary record set must not be empty.")
    if int(old_class_count) < 1:
        raise ValueError("old_class_count must be positive.")

    trial_ids: list[int] = []
    for record in materialised:
        trial_id = _required_integer(record, "trial_id")
        _required_integer(record, "subject_id")
        label = _required_integer(record, "label")
        if not 0 <= label < int(old_class_count):
            raise ValueError(
                f"{split} trial {trial_id} has label {label}, outside the old-class "
                f"range [0,{int(old_class_count) - 1}]."
            )
        if not hasattr(record, "raw_trial") or not hasattr(record, "starts"):
            raise TypeError("Every boundary record must define raw_trial and starts.")
        trial_ids.append(trial_id)
    if len(set(trial_ids)) != len(trial_ids):
        raise ValueError(f"Duplicate trial_id detected in the {split} record set.")
    return materialised


def _extract_components(
    records: Sequence[Any],
    *,
    window_size_samples: int,
    scales: Sequence[float],
    frequency_bins: int,
    epsilon: float,
) -> list[np.ndarray]:
    output: list[np.ndarray] = []
    for record in records:
        components = raw_multiscale_boundary_components(
            record.raw_trial,
            record.starts,
            int(window_size_samples),
            scales=scales,
            frequency_bins=int(frequency_bins),
            epsilon=float(epsilon),
        )
        expected = max(0, len(record.starts) - 1)
        if len(components) != expected:
            raise RuntimeError(
                f"Raw boundary count mismatch for trial "
                f"{_required_integer(record, 'trial_id')}: expected {expected}, "
                f"got {len(components)}."
            )
        if hasattr(record, "clean_windows"):
            clean = getattr(record, "clean_windows")
            if getattr(clean, "ndim", None) != 3:
                raise ValueError("clean_windows must have shape [windows,channels,time].")
            if int(clean.shape[0]) != len(record.starts):
                raise ValueError("clean_windows and starts disagree on the window count.")
            if int(clean.shape[-1]) != int(window_size_samples):
                raise ValueError(
                    "clean_windows width does not match window_size_samples."
                )
        output.append(components)
    return output


def _fit_from_components(
    records: Sequence[Any],
    components: Sequence[np.ndarray],
    *,
    window_size_samples: int,
    old_class_count: int,
    scales: Sequence[float],
    frequency_bins: int,
    epsilon: float,
    scale_floor: float,
    z_clip: float,
) -> dict:
    names = raw_component_names(scales)
    scaler = fit_trial_equal_robust_component_scaler(
        components,
        names,
        scale_floor=float(scale_floor),
        z_clip=float(z_clip),
    )
    train_scores = [
        transform_raw_boundary_components(item, scaler) for item in components
    ]
    thresholds = fit_trial_equal_score_thresholds(
        train_scores,
        LOW_QUANTILE,
        HIGH_QUANTILE,
    )
    document = {
        "schema_version": SCHEMA_VERSION,
        "calibration_type": CALIBRATION_TYPE,
        "anchor_source": ANCHOR_SOURCE,
        "pseudo_anchor_provider": "raw physical-unit kinematics only",
        "uses_frozen_legacy_encoder": False,
        "requires_source_checkpoint": False,
        "ema_teacher_generates_pseudo_anchors": False,
        "fit_split": "train_subjects_old_classes_only",
        "validation_policy": (
            "transform_only_with_train_fitted_component_scaler_and_q50_q90_thresholds"
        ),
        "old_class_count": int(old_class_count),
        "window_size_samples": int(window_size_samples),
        "low_quantile": LOW_QUANTILE,
        "high_quantile": HIGH_QUANTILE,
        "raw_formula": raw_formula_metadata(
            scales,
            int(frequency_bins),
            float(epsilon),
        ),
        "robust_component_calibration": scaler,
        "score_thresholds": thresholds,
        "train_trial_ids": sorted(
            _required_integer(record, "trial_id") for record in records
        ),
        "train_subject_ids": sorted(
            {_required_integer(record, "subject_id") for record in records}
        ),
        "train_trial_count": int(len(records)),
        "per_trial_change_anchor_is_forced": False,
        "anchor_rules": {
            "stable": "raw_score<=q50_threshold AND NOT change",
            "change": "raw_score>=q90_threshold AND raw_score>q50_threshold",
            "uncertain": "neither stable nor change; excluded from boundary supervision",
        },
    }
    document["calibration_sha256"] = _canonical_sha256(document)
    # Fail at the fitting boundary if an audit field accidentally becomes a
    # NumPy scalar/array or contains NaN/Inf.
    json.dumps(document, allow_nan=False)
    return document


def fit_train_old_raw_boundary_calibration(
    train_records: Sequence[Any],
    *,
    window_size_samples: int,
    old_class_count: int,
    scales: Sequence[float] = (1.0, 2.0, 4.0),
    frequency_bins: int = 16,
    epsilon: float = 1e-8,
    scale_floor: float = 1e-6,
    z_clip: float = 10.0,
) -> dict:
    """Fit raw-only scaler and q50/q90 thresholds on train-old trials.

    There is intentionally no validation argument and no encoder/checkpoint
    argument.  This makes validation fitting and legacy-feature anchoring
    impossible through this interface.
    """

    records = _validate_records(
        train_records,
        split="train",
        old_class_count=int(old_class_count),
        require_nonempty=True,
    )
    if int(window_size_samples) < 1:
        raise ValueError("window_size_samples must be positive.")
    parsed_scales = parse_raw_scales(scales)
    if int(frequency_bins) < 2:
        raise ValueError("frequency_bins must be at least two.")
    if not math.isfinite(float(epsilon)) or float(epsilon) <= 0:
        raise ValueError("epsilon must be positive and finite.")
    components = _extract_components(
        records,
        window_size_samples=int(window_size_samples),
        scales=parsed_scales,
        frequency_bins=int(frequency_bins),
        epsilon=float(epsilon),
    )
    return _fit_from_components(
        records,
        components,
        window_size_samples=int(window_size_samples),
        old_class_count=int(old_class_count),
        scales=parsed_scales,
        frequency_bins=int(frequency_bins),
        epsilon=float(epsilon),
        scale_floor=float(scale_floor),
        z_clip=float(z_clip),
    )


def _validated_calibration(calibration: dict) -> dict:
    if not isinstance(calibration, dict):
        raise TypeError("calibration must be a dictionary.")
    frozen = copy.deepcopy(calibration)
    digest = frozen.pop("calibration_sha256", None)
    if frozen.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Unsupported raw-boundary calibration schema version.")
    if frozen.get("calibration_type") != CALIBRATION_TYPE:
        raise ValueError("Unexpected raw-boundary calibration type.")
    if frozen.get("anchor_source") != ANCHOR_SOURCE:
        raise ValueError("One-stage boundary calibration must be raw-only.")
    if frozen.get("uses_frozen_legacy_encoder") is not False:
        raise ValueError("One-stage boundary calibration cannot use a legacy encoder.")
    if frozen.get("requires_source_checkpoint") is not False:
        raise ValueError("One-stage boundary calibration cannot require a checkpoint.")
    if float(frozen.get("low_quantile", -1.0)) != LOW_QUANTILE or float(
        frozen.get("high_quantile", -1.0)
    ) != HIGH_QUANTILE:
        raise ValueError("One-stage raw boundary thresholds must be q50/q90.")
    if not isinstance(digest, str) or digest != _canonical_sha256(frozen):
        raise ValueError("Raw-boundary calibration fingerprint mismatch.")
    return calibration


def _score_summary(scores: np.ndarray) -> dict:
    if not len(scores):
        return {"minimum": None, "median": None, "maximum": None}
    return {
        "minimum": float(np.min(scores)),
        "median": float(np.median(scores)),
        "maximum": float(np.max(scores)),
    }


def _assign_from_components(
    records: Sequence[Any],
    components_by_trial: Sequence[np.ndarray],
    calibration: dict,
    *,
    split: str,
) -> list[dict]:
    scaler = calibration["robust_component_calibration"]
    thresholds = calibration["score_thresholds"]
    diagnostics: list[dict] = []
    for record, components in zip(records, components_by_trial):
        scores = transform_raw_boundary_components(components, scaler)
        stable, change = boundary_candidate_masks(scores, thresholds)
        if stable.shape != change.shape or np.any(stable & change):
            raise RuntimeError("Raw stable/change boundary masks overlap or differ in shape.")
        # TrialExample is intentionally mutated in the same format consumed by
        # the existing boundary loss.  No source_scores are created or read.
        record.raw_scores = torch.from_numpy(scores.astype(np.float32, copy=False))
        record.stable_mask = torch.from_numpy(stable.astype(np.bool_, copy=False))
        record.change_mask = torch.from_numpy(change.astype(np.bool_, copy=False))
        diagnostics.append(
            {
                "trial_id": _required_integer(record, "trial_id"),
                "subject_id": _required_integer(record, "subject_id"),
                "label": _required_integer(record, "label"),
                "split": split,
                "boundary_count": int(len(scores)),
                "selected_stable_count": int(stable.sum()),
                "selected_change_count": int(change.sum()),
                "selected_stable_boundary_indices": np.flatnonzero(stable).astype(int).tolist(),
                "selected_change_boundary_indices": np.flatnonzero(change).astype(int).tolist(),
                "score_summary": _score_summary(scores),
            }
        )
    return diagnostics


def assign_raw_boundary_anchors(
    records: Sequence[Any],
    calibration: dict,
    *,
    split: str,
) -> list[dict]:
    """Transform records with frozen train calibration and assign masks.

    This function performs no fitting.  In particular, validation values can
    neither alter the component scaler nor the q50/q90 thresholds.
    """

    calibration = _validated_calibration(calibration)
    materialised = _validate_records(
        records,
        split=split,
        old_class_count=int(calibration["old_class_count"]),
        require_nonempty=False,
    )
    raw_formula = calibration["raw_formula"]
    components = _extract_components(
        materialised,
        window_size_samples=int(calibration["window_size_samples"]),
        scales=raw_formula["scale_stride_multipliers"],
        frequency_bins=int(raw_formula["frequency_bins"]),
        epsilon=float(raw_formula["epsilon"]),
    )
    before = copy.deepcopy(calibration)
    diagnostics = _assign_from_components(
        materialised,
        components,
        calibration,
        split=split,
    )
    if calibration != before:
        raise RuntimeError("Transform-only assignment mutated train calibration.")
    json.dumps(diagnostics, allow_nan=False)
    return diagnostics


def _coverage(diagnostics: Sequence[dict]) -> dict:
    boundary_count = int(sum(item["boundary_count"] for item in diagnostics))
    stable_count = int(sum(item["selected_stable_count"] for item in diagnostics))
    change_count = int(sum(item["selected_change_count"] for item in diagnostics))
    return {
        "trial_count": int(len(diagnostics)),
        "boundary_count": boundary_count,
        "boundary_bearing_trial_count": int(
            sum(item["boundary_count"] > 0 for item in diagnostics)
        ),
        "selected_stable_count": stable_count,
        "selected_change_count": change_count,
        "selected_stable_rate": stable_count / boundary_count if boundary_count else None,
        "selected_change_rate": change_count / boundary_count if boundary_count else None,
        "trials_without_selected_stable": int(
            sum(item["selected_stable_count"] == 0 for item in diagnostics)
        ),
        "trials_without_selected_change": int(
            sum(item["selected_change_count"] == 0 for item in diagnostics)
        ),
        "trials_with_selected_anchor_pair": int(
            sum(
                item["selected_stable_count"] > 0
                and item["selected_change_count"] > 0
                for item in diagnostics
            )
        ),
    }


def build_one_stage_raw_boundaries(
    train_records: Sequence[Any],
    validation_records: Sequence[Any],
    *,
    window_size_samples: int,
    old_class_count: int,
    scales: Sequence[float] = (1.0, 2.0, 4.0),
    frequency_bins: int = 16,
    epsilon: float = 1e-8,
    scale_floor: float = 1e-6,
    z_clip: float = 10.0,
    require_subject_disjoint: bool = True,
) -> dict:
    """Fit train-old raw anchors, transform validation, and return an audit.

    Both record sets must contain old-class trials.  By default their subjects
    and trial IDs must be disjoint.  Only ``train_records`` are supplied to the
    fitting helpers; validation is passed solely to the transform-only helper.
    """

    train = _validate_records(
        train_records,
        split="train",
        old_class_count=int(old_class_count),
        require_nonempty=True,
    )
    validation = _validate_records(
        validation_records,
        split="validation",
        old_class_count=int(old_class_count),
        require_nonempty=False,
    )
    train_ids = {_required_integer(item, "trial_id") for item in train}
    validation_ids = {_required_integer(item, "trial_id") for item in validation}
    overlap = sorted(train_ids & validation_ids)
    if overlap:
        raise ValueError(f"Train/validation trial overlap: {overlap}.")
    train_subjects = {_required_integer(item, "subject_id") for item in train}
    validation_subjects = {
        _required_integer(item, "subject_id") for item in validation
    }
    subject_overlap = sorted(train_subjects & validation_subjects)
    if require_subject_disjoint and subject_overlap:
        raise ValueError(f"Train/validation subject overlap: {subject_overlap}.")

    parsed_scales = parse_raw_scales(scales)
    train_components = _extract_components(
        train,
        window_size_samples=int(window_size_samples),
        scales=parsed_scales,
        frequency_bins=int(frequency_bins),
        epsilon=float(epsilon),
    )
    calibration = _fit_from_components(
        train,
        train_components,
        window_size_samples=int(window_size_samples),
        old_class_count=int(old_class_count),
        scales=parsed_scales,
        frequency_bins=int(frequency_bins),
        epsilon=float(epsilon),
        scale_floor=float(scale_floor),
        z_clip=float(z_clip),
    )
    train_diagnostics = _assign_from_components(
        train,
        train_components,
        calibration,
        split="train",
    )
    validation_diagnostics = assign_raw_boundary_anchors(
        validation,
        calibration,
        split="validation",
    )
    audit = {
        "schema_version": SCHEMA_VERSION,
        "protocol": "one_stage_raw_kinematic_boundary_anchors",
        "anchor_source": ANCHOR_SOURCE,
        "pseudo_anchor_provider": "raw physical-unit kinematics only",
        "uses_frozen_legacy_encoder": False,
        "requires_source_checkpoint": False,
        "fit_split": "train_subjects_old_classes_only",
        "validation_policy": (
            "transform_only_with_train_fitted_component_scaler_and_q50_q90_thresholds"
        ),
        "train_validation_trial_overlap": overlap,
        "train_validation_subject_overlap": subject_overlap,
        "subject_disjoint_required": bool(require_subject_disjoint),
        "calibration": calibration,
        "coverage": {
            "train": _coverage(train_diagnostics),
            "validation": _coverage(validation_diagnostics),
        },
        "trial_diagnostics": train_diagnostics + validation_diagnostics,
    }
    json.dumps(audit, allow_nan=False)
    return audit


__all__ = [
    "ANCHOR_SOURCE",
    "CALIBRATION_TYPE",
    "HIGH_QUANTILE",
    "LOW_QUANTILE",
    "SCHEMA_VERSION",
    "assign_raw_boundary_anchors",
    "build_one_stage_raw_boundaries",
    "fit_train_old_raw_boundary_calibration",
]
