"""Strict cross-subject Sitting/Standing representation probe.

This module deliberately stays outside the CGCD trainer.  It uses one held-out
subject's Sitting/Standing labels to build two class prototypes and evaluates
the other held-out subject, then reverses the direction.  Consequently its
outputs are an oracle representation diagnostic, never a CGCD accuracy.

The three pre-registered representations are:

``token_only``
    Expanded-window-occupancy-normalised KMeans token histogram.
``token_residual``
    Token histogram plus the expanded-window-weighted mean quantisation residual.
``token_gravity``
    Token histogram plus a robust raw-acceleration gravity direction.

Every distance scale and prototype is fitted from the source subject only.
Equal class distances receive half credit instead of an arbitrary tie break.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

from experiments.motion_primitive.core import EPS, jensen_shannon_distance


REPRESENTATIONS = ("token_only", "token_residual", "token_gravity")
PASS_CORRECT_PER_DIRECTION = 8
EXPECTED_TRIALS_PER_DIRECTION = 10
EXPECTED_TRIALS_PER_CLASS = 5
TIE_ATOL = 1e-12


@dataclass(frozen=True)
class TrialProbeFeature:
    trial_global_id: int
    subject_id: int
    activity_label: int
    activity_name: str
    trial_number: int
    token_histogram: np.ndarray
    mean_quantization_residual: np.ndarray
    gravity_direction: np.ndarray


def _unit_vector(value: np.ndarray, name: str) -> np.ndarray:
    vector = np.asarray(value, dtype=np.float64)
    if vector.ndim != 1 or not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must be a finite 1D vector, got {vector.shape}.")
    norm = float(np.linalg.norm(vector))
    if norm <= EPS:
        raise ValueError(f"{name} has zero norm.")
    return (vector / norm).astype(np.float32)


def robust_gravity_direction(sensor: np.ndarray, trim_fraction: float = 0.10) -> np.ndarray:
    """Return unit median acceleration over the central visible trial span."""

    signal = np.asarray(sensor, dtype=np.float64)
    if signal.ndim != 2 or signal.shape[0] < 3 or signal.shape[1] < 3:
        raise ValueError(f"sensor must have shape [>=3,T>=3], got {signal.shape}.")
    if not 0.0 <= float(trim_fraction) < 0.5:
        raise ValueError("trim_fraction must lie in [0,0.5).")
    trim = int(math.floor(signal.shape[1] * float(trim_fraction)))
    stop = signal.shape[1] - trim
    central = signal[:3, trim:stop]
    if central.shape[1] == 0:
        raise ValueError("Central gravity span is empty after trimming.")
    return _unit_vector(np.median(central, axis=1), "median acceleration")


def validate_probe_trials(trials: Sequence[TrialProbeFeature]) -> tuple[list[int], list[int]]:
    if not trials:
        raise ValueError("At least one probe trial is required.")
    ids = [int(trial.trial_global_id) for trial in trials]
    if len(ids) != len(set(ids)):
        raise ValueError("Probe trial_global_id values must be unique.")
    subjects = sorted({int(trial.subject_id) for trial in trials})
    labels = sorted({int(trial.activity_label) for trial in trials})
    if len(subjects) != 2:
        raise ValueError(f"Exactly two held-out subjects are required, got {subjects}.")
    if len(labels) != 2:
        raise ValueError(f"Exactly Sitting and Standing labels are required, got {labels}.")
    name_to_labels: dict[str, set[int]] = {}
    for trial in trials:
        name_to_labels.setdefault(trial.activity_name.strip().lower(), set()).add(
            int(trial.activity_label)
        )
        for name, value in (
            ("token_histogram", trial.token_histogram),
            ("mean_quantization_residual", trial.mean_quantization_residual),
            ("gravity_direction", trial.gravity_direction),
        ):
            array = np.asarray(value)
            if array.ndim != 1 or not np.all(np.isfinite(array)):
                raise ValueError(
                    f"Trial {trial.trial_global_id} has invalid {name}: {array.shape}."
                )
    required_names = {"sitting", "standing"}
    if set(name_to_labels) != required_names:
        raise ValueError(
            "Probe activities must be exactly Sitting and Standing; got "
            f"{sorted(name_to_labels)}."
        )
    if any(len(values) != 1 for values in name_to_labels.values()):
        raise ValueError("An activity name maps to multiple labels.")
    if len({next(iter(values)) for values in name_to_labels.values()}) != 2:
        raise ValueError("Sitting and Standing must map to distinct labels.")
    for subject in subjects:
        subject_trials = [trial for trial in trials if trial.subject_id == subject]
        counts = {
            label: sum(trial.activity_label == label for trial in subject_trials)
            for label in labels
        }
        if any(count != EXPECTED_TRIALS_PER_CLASS for count in counts.values()):
            raise ValueError(
                "The registered USC-HAD probe requires exactly "
                f"{EXPECTED_TRIALS_PER_CLASS} trials per class and subject; "
                f"subject {subject} has {counts}."
            )
    return subjects, labels


def _histogram_distance(left: np.ndarray, right: np.ndarray) -> float:
    # Natural-log JSD has maximum sqrt(log(2)); normalise every block to [0,1].
    return float(jensen_shannon_distance(left, right) / math.sqrt(math.log(2.0)))


def _residual_distance(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.linalg.norm(np.asarray(left, dtype=np.float64) - np.asarray(right, dtype=np.float64)))


def _gravity_distance(left: np.ndarray, right: np.ndarray) -> float:
    left_unit = _unit_vector(left, "left gravity")
    right_unit = _unit_vector(right, "right gravity")
    cosine = float(np.clip(np.dot(left_unit, right_unit), -1.0, 1.0))
    return float(math.acos(cosine) / math.pi)


def _block_value(trial: TrialProbeFeature, block: str) -> np.ndarray:
    if block == "token":
        return trial.token_histogram
    if block == "residual":
        return trial.mean_quantization_residual
    if block == "gravity":
        return trial.gravity_direction
    raise KeyError(block)


def _block_distance(block: str, left: np.ndarray, right: np.ndarray) -> float:
    if block == "token":
        return _histogram_distance(left, right)
    if block == "residual":
        return _residual_distance(left, right)
    if block == "gravity":
        return _gravity_distance(left, right)
    raise KeyError(block)


def _prototype(block: str, trials: Sequence[TrialProbeFeature]) -> np.ndarray:
    if not trials:
        raise ValueError("A class prototype cannot be built from zero trials.")
    values = np.asarray([_block_value(trial, block) for trial in trials], dtype=np.float64)
    mean = np.mean(values, axis=0)
    if block == "token":
        total = float(mean.sum())
        if total <= EPS:
            raise ValueError("Token prototype has zero mass.")
        return (mean / total).astype(np.float32)
    if block == "gravity":
        return _unit_vector(mean, "gravity prototype")
    return mean.astype(np.float32)


def _source_block_scale(
    block: str,
    source_trials: Sequence[TrialProbeFeature],
    labels: Sequence[int],
) -> float:
    """Return a source-only block scale with a deterministic zero-spread fallback.

    The primary scale is the median within-class leave-one-trial-out distance.
    If both source classes are internally identical, that value is zero even
    when their prototypes are perfectly separated.  In that case the distance
    between the source-class prototypes is the scale.  A block is disabled only
    when both its within-class spread and its between-class prototype distance
    are numerically zero.
    """

    distances = []
    for label in labels:
        class_trials = [trial for trial in source_trials if trial.activity_label == label]
        if len(class_trials) < 2:
            raise ValueError(
                f"Source class {label} needs at least two trials for LOTO scaling."
            )
        for held_out in class_trials:
            remaining = [trial for trial in class_trials if trial is not held_out]
            prototype = _prototype(block, remaining)
            distances.append(
                _block_distance(block, _block_value(held_out, block), prototype)
            )
    values = np.asarray(distances, dtype=np.float64)
    scale = float(np.median(values))
    if scale > TIE_ATOL:
        return scale
    class_prototypes = [
        _prototype(
            block,
            [trial for trial in source_trials if trial.activity_label == label],
        )
        for label in labels
    ]
    prototype_distance = _block_distance(
        block, class_prototypes[0], class_prototypes[1]
    )
    return float(prototype_distance) if prototype_distance > TIE_ATOL else 0.0


def _representation_blocks(representation: str) -> tuple[str, ...]:
    table = {
        "token_only": ("token",),
        "token_residual": ("token", "residual"),
        "token_gravity": ("token", "gravity"),
    }
    if representation not in table:
        raise ValueError(f"Unknown representation {representation!r}.")
    return table[representation]


def _scaled_distance(
    block: str,
    value: np.ndarray,
    prototype: np.ndarray,
    scale: float,
) -> float | None:
    raw = _block_distance(block, value, prototype)
    if scale <= TIE_ATOL:
        # Both within-class spread and between-class prototype distance are zero.
        # The source subject therefore supplies no calibrated preference.
        return None
    return float(raw / scale)


def evaluate_direction(
    trials: Sequence[TrialProbeFeature],
    source_subject: int,
    target_subject: int,
    representation: str,
) -> dict:
    subjects, labels = validate_probe_trials(trials)
    if sorted([int(source_subject), int(target_subject)]) != subjects:
        raise ValueError("Direction subjects must equal the two probe subjects.")
    if int(source_subject) == int(target_subject):
        raise ValueError("Source and target subjects must differ.")
    source = [trial for trial in trials if trial.subject_id == int(source_subject)]
    target = [trial for trial in trials if trial.subject_id == int(target_subject)]
    blocks = _representation_blocks(representation)
    scales = {block: _source_block_scale(block, source, labels) for block in blocks}
    prototypes = {
        label: {
            block: _prototype(
                block, [trial for trial in source if trial.activity_label == label]
            )
            for block in blocks
        }
        for label in labels
    }
    confusion = np.zeros((2, 2), dtype=np.float64)
    predictions = []
    ties = 0
    label_to_position = {label: position for position, label in enumerate(labels)}
    for trial in target:
        class_distances = {}
        for label in labels:
            block_distances = []
            for block in blocks:
                value = _scaled_distance(
                    block,
                    _block_value(trial, block),
                    prototypes[label][block],
                    scales[block],
                )
                if value is not None:
                    block_distances.append(value)
            class_distances[label] = (
                float(np.mean(block_distances)) if block_distances else 0.0
            )
        minimum = min(class_distances.values())
        tied_labels = [
            label
            for label, value in class_distances.items()
            if math.isclose(value, minimum, rel_tol=0.0, abs_tol=TIE_ATOL)
        ]
        if len(tied_labels) > 1:
            ties += 1
        weight = 1.0 / len(tied_labels)
        true_position = label_to_position[int(trial.activity_label)]
        for predicted in tied_labels:
            confusion[true_position, label_to_position[predicted]] += weight
        predictions.append(
            {
                "trial_global_id": int(trial.trial_global_id),
                "trial_number": int(trial.trial_number),
                "true_label": int(trial.activity_label),
                "true_activity": trial.activity_name,
                "class_distances": {
                    str(label): float(class_distances[label]) for label in labels
                },
                "tied_predicted_labels": [int(value) for value in tied_labels],
                "correct_credit": float(
                    weight if int(trial.activity_label) in tied_labels else 0.0
                ),
            }
        )
    support = confusion.sum(axis=1)
    recalls = np.divide(
        np.diag(confusion),
        support,
        out=np.full(2, np.nan, dtype=np.float64),
        where=support > 0,
    )
    correct_credit = float(np.trace(confusion))
    return {
        "source_subject": int(source_subject),
        "target_subject": int(target_subject),
        "representation": representation,
        "source_trial_count": len(source),
        "target_trial_count": len(target),
        "class_ids": labels,
        "active_blocks": [block for block in blocks if scales[block] > TIE_ATOL],
        "inactive_zero_scale_blocks": [
            block for block in blocks if scales[block] <= TIE_ATOL
        ],
        "source_only_block_scales": scales,
        "confusion_counts_tie_aware": confusion,
        "per_class_recall": {
            str(label): float(recalls[position])
            for position, label in enumerate(labels)
        },
        "correct_credit": correct_credit,
        "accuracy": float(correct_credit / max(len(target), 1)),
        "balanced_accuracy": float(np.nanmean(recalls)),
        "tie_query_count": int(ties),
        "tie_query_ratio": float(ties / max(len(target), 1)),
        "predictions": predictions,
    }


def evaluate_bidirectional_probe(trials: Sequence[TrialProbeFeature]) -> dict:
    subjects, labels = validate_probe_trials(trials)
    counts = {
        subject: sum(trial.subject_id == subject for trial in trials)
        for subject in subjects
    }
    if any(count != EXPECTED_TRIALS_PER_DIRECTION for count in counts.values()):
        raise ValueError(
            "The registered USC-HAD fold06 probe expects exactly 10 trials per "
            f"subject, got {counts}."
        )
    results = {}
    for representation in REPRESENTATIONS:
        directions = [
            evaluate_direction(trials, subjects[0], subjects[1], representation),
            evaluate_direction(trials, subjects[1], subjects[0], representation),
        ]
        correct = [float(item["correct_credit"]) for item in directions]
        passed = all(value >= PASS_CORRECT_PER_DIRECTION for value in correct)
        results[representation] = {
            "directions": directions,
            "macro_direction_accuracy": float(
                np.mean([item["accuracy"] for item in directions])
            ),
            "minimum_direction_correct_credit": float(min(correct)),
            "registered_gate": {
                "rule": "both directions correct_credit >= 8 of 10",
                "threshold_correct_per_direction": PASS_CORRECT_PER_DIRECTION,
                "passed": bool(passed),
            },
        }
    return {
        "scope": "oracle_novel_label_representation_diagnostic_only",
        "is_cgcd_metric": False,
        "subjects": subjects,
        "class_ids": labels,
        "representations": results,
    }


def features_to_rows(trials: Sequence[TrialProbeFeature]) -> list[dict]:
    rows = []
    for trial in sorted(
        trials,
        key=lambda item: (
            item.subject_id,
            item.activity_label,
            item.trial_number,
            item.trial_global_id,
        ),
    ):
        row: dict[str, object] = {
            "trial_global_id": int(trial.trial_global_id),
            "subject_id": int(trial.subject_id),
            "activity_label_0based": int(trial.activity_label),
            "activity_name": trial.activity_name,
            "trial_number": int(trial.trial_number),
        }
        row.update(
            {
                f"token_{index:02d}_fraction": float(value)
                for index, value in enumerate(trial.token_histogram)
            }
        )
        row.update(
            {
                f"residual_{index:02d}": float(value)
                for index, value in enumerate(trial.mean_quantization_residual)
            }
        )
        row.update(
            {
                f"gravity_{axis}": float(value)
                for axis, value in zip("xyz", trial.gravity_direction)
            }
        )
        rows.append(row)
    return rows


__all__ = [
    "EXPECTED_TRIALS_PER_DIRECTION",
    "EXPECTED_TRIALS_PER_CLASS",
    "PASS_CORRECT_PER_DIRECTION",
    "REPRESENTATIONS",
    "TrialProbeFeature",
    "evaluate_bidirectional_probe",
    "evaluate_direction",
    "features_to_rows",
    "robust_gravity_direction",
    "validate_probe_trials",
]
