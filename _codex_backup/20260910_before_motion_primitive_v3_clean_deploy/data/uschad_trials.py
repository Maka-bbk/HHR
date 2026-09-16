"""Leakage-safe, full-trial USC-HAD input pipeline for the one-stage J0 study.

The legacy project stores overlapping windows in an NPZ.  J0 instead consumes
the original MAT file for each physical trial, preserving every sensor sample.
Discovery reads filenames and small MATLAB metadata fields only; sensor values
are loaded lazily by a split-specific dataset.  Fold normalization is fitted by
streaming *train-subject, old-class* trials and therefore never touches
validation or outer-test sensor arrays.

USC-HAD has a handful of inconsistent MATLAB metadata fields.  Filesystem
identity is the canonical physical identity because the published layout is a
complete ``SubjectN/aMtK.mat`` grid.  In particular, ``Subject14/a3t2.mat`` is
kept as physical activity 3 by default while its internal activity 2 is
reported.  ``anomaly_policy="exclude"`` removes that known trial; it is never
silently relabelled.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch
from scipy.io import loadmat
from torch.utils.data import Dataset


PROTOCOL_SCHEMA = "uschad_full_trial_j0_v1"
SAMPLE_RATE_HZ = 100
CHANNEL_NAMES = (
    "acc_x",
    "acc_y",
    "acc_z",
    "gyro_x",
    "gyro_y",
    "gyro_z",
)
ACTIVITY_NAMES = {
    1: "Walking Forward",
    2: "Walking Left",
    3: "Walking Right",
    4: "Walking Upstairs",
    5: "Walking Downstairs",
    6: "Running Forward",
    7: "Jumping Up",
    8: "Sitting",
    9: "Standing",
    10: "Sleeping",
    11: "Elevator Up",
    12: "Elevator Down",
}
SUBJECT_IDS = tuple(range(1, 15))
ACTIVITY_IDS = tuple(range(1, 13))
TRIAL_NUMBERS = tuple(range(1, 6))
DEFAULT_OLD_ACTIVITY_IDS = tuple(range(1, 7))
DEFAULT_NEW_ACTIVITY_IDS = tuple(range(7, 13))
LABEL_REGIMES = ("J0-U", "J0-T")
IGNORE_INDEX = -100
DEFAULT_USCHAD_ROOT = Path(
    r"D:\WorkDir\DataSet\USC-HAD"
    if os.name == "nt"
    else "/mnt/d/WorkDir/DataSet/USC-HAD"
)

# These are the established seven subject-disjoint folds used by the project.
# Validation is the next outer-test pair in the cycle, so every subject occurs
# exactly once in validation and once in outer test.
_TEST_PAIRS = (
    (11, 10),
    (2, 13),
    (3, 9),
    (7, 1),
    (12, 8),
    (5, 4),
    (14, 6),
)
_VALIDATION_PAIRS = (
    (2, 13),
    (3, 9),
    (7, 1),
    (12, 8),
    (5, 4),
    (14, 6),
    (11, 10),
)

KNOWN_LABEL_ANOMALY_KEY = "s14_a03_t02"
_SUBJECT_PATTERN = re.compile(r"^Subject(\d+)$", flags=re.IGNORECASE)
_TRIAL_PATTERN = re.compile(r"^a(\d+)t(\d+)\.mat$", flags=re.IGNORECASE)


class USCHADProtocolError(RuntimeError):
    """Raised when raw data would violate the fixed J0 protocol."""


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _identity_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _normalise_int_set(values: Iterable[int], name: str) -> tuple[int, ...]:
    result: list[int] = []
    for value in values:
        if isinstance(value, (bool, np.bool_)) or not isinstance(
            value, (int, np.integer)
        ):
            raise TypeError(f"{name} must contain integers; got {value!r}.")
        result.append(int(value))
    if not result:
        raise ValueError(f"{name} cannot be empty.")
    if len(result) != len(set(result)):
        raise ValueError(f"{name} contains duplicate values: {result}.")
    return tuple(sorted(result))


@dataclass(frozen=True)
class SubjectSplit:
    """One immutable, canonical subject-disjoint fold."""

    fold: int
    train_subjects: tuple[int, ...]
    validation_subjects: tuple[int, ...]
    test_subjects: tuple[int, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "fold": int(self.fold),
            "train_subjects": list(self.train_subjects),
            "validation_subjects": list(self.validation_subjects),
            "test_subjects": list(self.test_subjects),
        }


def _make_subject_split(fold: int) -> SubjectSplit:
    if isinstance(fold, bool) or int(fold) != fold or not 1 <= int(fold) <= 7:
        raise ValueError(f"fold must be an integer in 1..7; got {fold!r}.")
    index = int(fold) - 1
    validation = tuple(sorted(_VALIDATION_PAIRS[index]))
    test = tuple(sorted(_TEST_PAIRS[index]))
    train = tuple(sorted(set(SUBJECT_IDS) - set(validation) - set(test)))
    split = SubjectSplit(int(fold), train, validation, test)
    validate_subject_split(split, require_canonical=False)
    return split


def subject_split_for_fold(fold: int) -> SubjectSplit:
    """Return the canonical split for a one-based fold id."""

    if isinstance(fold, bool) or not isinstance(fold, (int, np.integer)):
        raise TypeError(f"fold must be an integer in 1..7; got {fold!r}.")
    if not 1 <= int(fold) <= len(SUBJECT_SPLITS):
        raise ValueError(f"fold must be in 1..7; got {fold!r}.")
    return SUBJECT_SPLITS[int(fold) - 1]


def validate_subject_split(
    split: SubjectSplit, *, require_canonical: bool = True
) -> dict[str, Any]:
    """Fail closed on overlap, omissions, extras, or fold identity drift."""

    if not isinstance(split, SubjectSplit):
        raise TypeError("split must be a SubjectSplit.")
    train = _normalise_int_set(split.train_subjects, "train_subjects")
    validation = _normalise_int_set(
        split.validation_subjects, "validation_subjects"
    )
    test = _normalise_int_set(split.test_subjects, "test_subjects")
    groups = {"train": set(train), "validation": set(validation), "test": set(test)}
    overlaps = {
        "train_validation": sorted(groups["train"] & groups["validation"]),
        "train_test": sorted(groups["train"] & groups["test"]),
        "validation_test": sorted(groups["validation"] & groups["test"]),
    }
    if any(overlaps.values()):
        raise USCHADProtocolError(f"Subject leakage detected: {overlaps}.")
    observed = groups["train"] | groups["validation"] | groups["test"]
    if observed != set(SUBJECT_IDS):
        raise USCHADProtocolError(
            "A fold must partition USC-HAD subjects 1..14 exactly; "
            f"missing={sorted(set(SUBJECT_IDS) - observed)}, "
            f"extra={sorted(observed - set(SUBJECT_IDS))}."
        )
    if len(train) != 10 or len(validation) != 2 or len(test) != 2:
        raise USCHADProtocolError(
            "A canonical fold requires 10 train, 2 validation, and 2 test subjects."
        )
    if require_canonical:
        canonical = subject_split_for_fold(int(split.fold))
        if split.as_dict() != canonical.as_dict():
            raise USCHADProtocolError(
                f"Fold {split.fold} differs from the fixed seven-fold manifest."
            )
    return {
        "subject_overlap": overlaps,
        "partition_is_exact": True,
        "counts": {"train": 10, "validation": 2, "test": 2},
    }


# Construct the public constant only after the validator exists.  The helper
# invokes the validator while each canonical fold is built.
SUBJECT_SPLITS = tuple(_make_subject_split(fold) for fold in range(1, 8))


def fixed_subject_manifest() -> dict[str, Any]:
    """Return a JSON-safe fixed-manifest identity and balance audit."""

    validation_counts = {subject: 0 for subject in SUBJECT_IDS}
    test_counts = {subject: 0 for subject in SUBJECT_IDS}
    train_counts = {subject: 0 for subject in SUBJECT_IDS}
    folds = []
    for split in SUBJECT_SPLITS:
        validate_subject_split(split)
        folds.append(split.as_dict())
        for subject in split.train_subjects:
            train_counts[subject] += 1
        for subject in split.validation_subjects:
            validation_counts[subject] += 1
        for subject in split.test_subjects:
            test_counts[subject] += 1
    if set(train_counts.values()) != {5}:
        raise USCHADProtocolError(f"Train-fold subject balance drifted: {train_counts}.")
    if set(validation_counts.values()) != {1} or set(test_counts.values()) != {1}:
        raise USCHADProtocolError(
            "Every subject must occur once in validation and once in outer test."
        )
    identity = {
        "schema": PROTOCOL_SCHEMA,
        "subject_universe": list(SUBJECT_IDS),
        "folds": folds,
        "coverage": {
            "train_folds_per_subject": train_counts,
            "validation_folds_per_subject": validation_counts,
            "test_folds_per_subject": test_counts,
        },
    }
    identity["manifest_sha256"] = _identity_sha256(identity)
    return identity


def resolve_activity_split(
    old_activity_ids: Iterable[int] = DEFAULT_OLD_ACTIVITY_IDS,
    new_activity_ids: Iterable[int] | None = None,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Validate and return a complete, disjoint old/new activity partition."""

    old = _normalise_int_set(old_activity_ids, "old_activity_ids")
    if new_activity_ids is None:
        new = tuple(sorted(set(ACTIVITY_IDS) - set(old)))
    else:
        new = _normalise_int_set(new_activity_ids, "new_activity_ids")
    overlap = sorted(set(old) & set(new))
    if overlap:
        raise ValueError(f"old/new activity overlap: {overlap}.")
    observed = set(old) | set(new)
    if observed != set(ACTIVITY_IDS):
        raise ValueError(
            "old/new activities must partition physical ids 1..12 exactly; "
            f"missing={sorted(set(ACTIVITY_IDS) - observed)}, "
            f"extra={sorted(observed - set(ACTIVITY_IDS))}."
        )
    return old, new


def _stable_trial_id(subject_id: int, activity_id: int, trial_number: int) -> int:
    return (
        (int(subject_id) - 1) * len(ACTIVITY_IDS) * len(TRIAL_NUMBERS)
        + (int(activity_id) - 1) * len(TRIAL_NUMBERS)
        + int(trial_number)
        - 1
    )


def _trial_key(subject_id: int, activity_id: int, trial_number: int) -> str:
    return f"s{int(subject_id):02d}_a{int(activity_id):02d}_t{int(trial_number):02d}"


def _mat_scalar(value: Any) -> str | None:
    if value is None:
        return None
    array = np.asarray(value).squeeze()
    if array.size != 1:
        return None
    try:
        return str(array.item()).strip()
    except ValueError:
        return None


def _mat_integer(value: Any) -> int | None:
    text = _mat_scalar(value)
    if text is None or not text:
        return None
    try:
        number = float(text)
    except ValueError:
        return None
    if not math.isfinite(number) or int(number) != number:
        return None
    return int(number)


@dataclass(frozen=True)
class TrialDescriptor:
    """Sensor-free physical identity for one raw MAT trial."""

    path: Path
    relative_path: str
    subject_id: int
    physical_activity_id: int
    trial_number: int
    trial_id: int
    trial_key: str
    activity_name: str
    file_size_bytes: int
    internal_subject_id: int | None = None
    internal_activity_id: int | None = None
    internal_trial_number: int | None = None
    internal_activity_name: str | None = None
    metadata_issues: tuple[str, ...] = ()
    known_label_anomaly: bool = False

    @property
    def class_index(self) -> int:
        """Contiguous zero-based class index, kept separate from physical id."""

        return int(self.physical_activity_id) - 1

    def identity_dict(self) -> dict[str, Any]:
        return {
            "relative_path": self.relative_path,
            "subject_id": int(self.subject_id),
            "physical_activity_id_1based": int(self.physical_activity_id),
            "trial_number": int(self.trial_number),
            "trial_id_0based": int(self.trial_id),
            "trial_key": self.trial_key,
            "file_size_bytes": int(self.file_size_bytes),
        }


def _inspect_trial_metadata(path: Path, expected: tuple[int, int, int]) -> dict[str, Any]:
    # variable_names excludes sensor_readings: discovery cannot materialize a
    # validation or outer-test sensor array.
    metadata = loadmat(
        str(path),
        variable_names=["subject", "activity_number", "activity", "trial"],
        squeeze_me=True,
    )
    subject_id = _mat_integer(metadata.get("subject"))
    activity_id = _mat_integer(metadata.get("activity_number"))
    trial_number = _mat_integer(metadata.get("trial"))
    activity_name = _mat_scalar(metadata.get("activity"))
    expected_subject, expected_activity, expected_trial = expected
    issues = []
    for name, observed, wanted in (
        ("subject", subject_id, expected_subject),
        ("activity_number", activity_id, expected_activity),
        ("trial", trial_number, expected_trial),
    ):
        if observed is None:
            issues.append(f"missing_or_invalid_internal_{name}")
        elif int(observed) != int(wanted):
            issues.append(f"internal_{name}={observed}_filename_{name}={wanted}")
    if subject_id is not None and subject_id != expected_subject:
        raise USCHADProtocolError(
            f"Internal subject {subject_id} conflicts with path subject "
            f"{expected_subject}: {path}. Subject identity cannot be repaired safely."
        )
    return {
        "internal_subject_id": subject_id,
        "internal_activity_id": activity_id,
        "internal_trial_number": trial_number,
        "internal_activity_name": activity_name,
        "metadata_issues": tuple(issues),
    }


def discover_trial_manifest(
    root: str | Path = DEFAULT_USCHAD_ROOT,
    *,
    anomaly_policy: str = "report",
    require_complete: bool = True,
    inspect_mat_metadata: bool = True,
) -> tuple[list[TrialDescriptor], dict[str, Any]]:
    """Discover raw trials without loading any ``sensor_readings`` arrays.

    ``report`` retains the known Subject14/a3t2 label conflict under its
    filename-defined physical label.  ``exclude`` removes that trial after the
    complete on-disk grid has been validated.
    """

    if anomaly_policy not in {"report", "exclude"}:
        raise ValueError("anomaly_policy must be 'report' or 'exclude'.")
    root_path = Path(root).expanduser().resolve()
    if not root_path.is_dir():
        raise FileNotFoundError(f"USC-HAD root is not a directory: {root_path}.")
    paths = sorted(root_path.rglob("*.mat"))
    if not paths:
        raise FileNotFoundError(f"No MAT trials found under {root_path}.")

    descriptors: list[TrialDescriptor] = []
    seen: set[tuple[int, int, int]] = set()
    for path in paths:
        subject_match = _SUBJECT_PATTERN.fullmatch(path.parent.name)
        trial_match = _TRIAL_PATTERN.fullmatch(path.name)
        if subject_match is None or trial_match is None:
            raise USCHADProtocolError(
                f"Unexpected USC-HAD MAT path; expected SubjectN/aMtK.mat: {path}."
            )
        subject_id = int(subject_match.group(1))
        activity_id = int(trial_match.group(1))
        trial_number = int(trial_match.group(2))
        identity = (subject_id, activity_id, trial_number)
        if identity in seen:
            raise USCHADProtocolError(f"Duplicate physical trial identity {identity}.")
        seen.add(identity)
        if subject_id not in SUBJECT_IDS:
            raise USCHADProtocolError(f"Unexpected subject id {subject_id}: {path}.")
        if activity_id not in ACTIVITY_IDS:
            raise USCHADProtocolError(f"Unexpected activity id {activity_id}: {path}.")
        if trial_number not in TRIAL_NUMBERS:
            raise USCHADProtocolError(f"Unexpected trial number {trial_number}: {path}.")
        metadata = (
            _inspect_trial_metadata(path, identity)
            if inspect_mat_metadata
            else {
                "internal_subject_id": None,
                "internal_activity_id": None,
                "internal_trial_number": None,
                "internal_activity_name": None,
                "metadata_issues": (),
            }
        )
        key = _trial_key(*identity)
        known_anomaly = key == KNOWN_LABEL_ANOMALY_KEY
        descriptors.append(
            TrialDescriptor(
                path=path.resolve(),
                relative_path=path.relative_to(root_path).as_posix(),
                subject_id=subject_id,
                physical_activity_id=activity_id,
                trial_number=trial_number,
                trial_id=_stable_trial_id(*identity),
                trial_key=key,
                activity_name=ACTIVITY_NAMES[activity_id],
                file_size_bytes=int(path.stat().st_size),
                known_label_anomaly=known_anomaly,
                **metadata,
            )
        )

    expected = {
        (subject, activity, trial)
        for subject in SUBJECT_IDS
        for activity in ACTIVITY_IDS
        for trial in TRIAL_NUMBERS
    }
    missing = sorted(expected - seen)
    extra = sorted(seen - expected)
    if require_complete and (missing or extra):
        raise USCHADProtocolError(
            "Raw USC-HAD must contain the complete 14x12x5 trial grid; "
            f"missing_count={len(missing)}, extra_count={len(extra)}, "
            f"first_missing={missing[:10]}, first_extra={extra[:10]}."
        )

    descriptors.sort(key=lambda item: item.trial_id)
    metadata_issue_records = [
        {
            "trial_key": descriptor.trial_key,
            "relative_path": descriptor.relative_path,
            "issues": list(descriptor.metadata_issues),
            "internal_activity_id": descriptor.internal_activity_id,
            "physical_activity_id": descriptor.physical_activity_id,
            "internal_trial_number": descriptor.internal_trial_number,
            "physical_trial_number": descriptor.trial_number,
            "known_label_anomaly": descriptor.known_label_anomaly,
        }
        for descriptor in descriptors
        if descriptor.metadata_issues or descriptor.known_label_anomaly
    ]
    before_policy = len(descriptors)
    if anomaly_policy == "exclude":
        descriptors = [item for item in descriptors if not item.known_label_anomaly]
    manifest_identity = {
        "schema": PROTOCOL_SCHEMA,
        "root": str(root_path),
        "physical_identity_source": "SubjectN/aMtK.mat path",
        "entries": [item.identity_dict() for item in descriptors],
    }
    audit = {
        "schema": "uschad_raw_trial_manifest_audit_v1",
        "root": str(root_path),
        "on_disk_mat_count": len(paths),
        "complete_grid_required": bool(require_complete),
        "complete_grid_observed": not missing and not extra,
        "expected_complete_trial_count": len(expected),
        "trial_count_before_anomaly_policy": before_policy,
        "trial_count_after_anomaly_policy": len(descriptors),
        "anomaly_policy": anomaly_policy,
        "known_label_anomaly": {
            "trial_key": KNOWN_LABEL_ANOMALY_KEY,
            "policy": anomaly_policy,
            "silently_relabelled": False,
            "retained": anomaly_policy == "report",
        },
        "metadata_issue_records": metadata_issue_records,
        "sensor_arrays_loaded_during_discovery": 0,
        "manifest_sha256": _identity_sha256(manifest_identity),
    }
    return descriptors, audit


def _load_sensor_readings(path: Path) -> np.ndarray:
    value = loadmat(str(path), variable_names=["sensor_readings"])
    if "sensor_readings" not in value:
        raise USCHADProtocolError(f"sensor_readings is missing: {path}.")
    signal = np.asarray(value["sensor_readings"])
    if signal.ndim != 2:
        raise USCHADProtocolError(
            f"sensor_readings must be two-dimensional [T,6]: {path} has {signal.shape}."
        )
    if signal.shape[1] != len(CHANNEL_NAMES) and signal.shape[0] == len(CHANNEL_NAMES):
        signal = signal.T
    if signal.shape[1] != len(CHANNEL_NAMES) or signal.shape[0] < 1:
        raise USCHADProtocolError(
            f"sensor_readings must have non-empty shape [T,6]: {path} has {signal.shape}."
        )
    if signal.dtype.kind not in "fiu" or not np.all(np.isfinite(signal)):
        raise USCHADProtocolError(f"sensor_readings is non-numeric or non-finite: {path}.")
    return np.ascontiguousarray(signal, dtype=np.float32)


@dataclass(frozen=True)
class NormalizationStats:
    """Train-old-only per-channel sample-weighted z-score state."""

    mean: np.ndarray
    std: np.ndarray
    epsilon: float
    sample_count: int
    trial_count: int
    fit_subject_ids: tuple[int, ...]
    fit_activity_ids: tuple[int, ...]
    fit_trial_ids_sha256: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "method": "full_raw_sample_weighted_per_channel_zscore",
            "mean": np.asarray(self.mean, dtype=np.float64).tolist(),
            "std": np.asarray(self.std, dtype=np.float64).tolist(),
            "epsilon": float(self.epsilon),
            "sample_count": int(self.sample_count),
            "trial_count": int(self.trial_count),
            "fit_subject_ids": list(self.fit_subject_ids),
            "fit_activity_ids": list(self.fit_activity_ids),
            "fit_trial_ids_sha256": self.fit_trial_ids_sha256,
        }


def fit_train_old_normalization(
    descriptors: Sequence[TrialDescriptor],
    *,
    expected_train_subjects: Iterable[int],
    expected_old_activity_ids: Iterable[int],
    epsilon: float = 1e-6,
) -> NormalizationStats:
    """Stream full train-old trials only and fit per-channel z-score state."""

    train_subjects = _normalise_int_set(expected_train_subjects, "train_subjects")
    old_activities = _normalise_int_set(
        expected_old_activity_ids, "old_activity_ids"
    )
    if not math.isfinite(float(epsilon)) or float(epsilon) <= 0:
        raise ValueError("epsilon must be positive and finite.")
    selected = list(descriptors)
    if not selected:
        raise ValueError("At least one train-old descriptor is required.")
    forbidden = [
        item.trial_key
        for item in selected
        if item.subject_id not in train_subjects
        or item.physical_activity_id not in old_activities
    ]
    if forbidden:
        raise USCHADProtocolError(
            "Normalization received a non-train-old trial; refusing potential leakage: "
            f"{forbidden[:10]}."
        )

    count = 0
    mean = np.zeros(len(CHANNEL_NAMES), dtype=np.float64)
    second_moment = np.zeros(len(CHANNEL_NAMES), dtype=np.float64)
    for descriptor in selected:
        signal = _load_sensor_readings(descriptor.path).astype(np.float64, copy=False)
        batch_count = int(signal.shape[0])
        batch_mean = signal.mean(axis=0, dtype=np.float64)
        centered = signal - batch_mean
        batch_second = np.sum(centered * centered, axis=0, dtype=np.float64)
        if count == 0:
            mean = batch_mean
            second_moment = batch_second
            count = batch_count
            continue
        total = count + batch_count
        delta = batch_mean - mean
        second_moment += (
            batch_second + delta * delta * count * batch_count / float(total)
        )
        mean += delta * batch_count / float(total)
        count = total
    variance = second_moment / float(count)
    std = np.maximum(np.sqrt(np.maximum(variance, 0.0)), float(epsilon))
    if not np.all(np.isfinite(mean)) or not np.all(np.isfinite(std)):
        raise USCHADProtocolError("Train-old normalization produced non-finite state.")
    trial_ids = np.asarray(sorted(item.trial_id for item in selected), dtype=np.int64)
    return NormalizationStats(
        mean=mean.astype(np.float32),
        std=std.astype(np.float32),
        epsilon=float(epsilon),
        sample_count=int(count),
        trial_count=len(selected),
        fit_subject_ids=tuple(sorted({item.subject_id for item in selected})),
        fit_activity_ids=tuple(
            sorted({item.physical_activity_id for item in selected})
        ),
        fit_trial_ids_sha256=hashlib.sha256(trial_ids.tobytes()).hexdigest(),
    )


def _validate_regime(value: str) -> str:
    regime = str(value).upper()
    if regime not in LABEL_REGIMES:
        raise ValueError(f"label_regime must be one of {LABEL_REGIMES}; got {value!r}.")
    return regime


@dataclass(frozen=True)
class StratifiedLabelAssignment:
    """Seeded train-old 4/1 trial assignment for trajectory supervision."""

    labelled_trial_ids: frozenset[int]
    unlabelled_trial_ids: frozenset[int]
    seed: int
    labelled_fraction: float
    strata: tuple[Mapping[str, Any], ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "method": "per_subject_per_old_activity_trial_stratified_v1",
            "seed": int(self.seed),
            "requested_labelled_fraction": float(self.labelled_fraction),
            "labelled_trial_count": len(self.labelled_trial_ids),
            "unlabelled_trial_count": len(self.unlabelled_trial_ids),
            "assignment_overlap": 0,
            "strata": [dict(item) for item in self.strata],
        }


def stratified_train_label_assignment(
    descriptors: Sequence[TrialDescriptor],
    *,
    seed: int,
    labelled_fraction: float = 0.8,
    allow_known_anomaly_exclusion: bool = False,
) -> StratifiedLabelAssignment:
    """Assign four of five trials per ``(subject, old activity)`` as labelled.

    The RNG stream is derived independently from ``seed``, subject, and
    activity, so adding or reordering an unrelated stratum cannot change an
    existing assignment.  If the known Subject14/a3t2 trial was explicitly
    excluded, that sole four-trial stratum receives a deterministic 3/1 split
    and the exception is recorded.
    """

    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)):
        raise TypeError("seed must be an integer.")
    fraction = float(labelled_fraction)
    if not math.isfinite(fraction) or not 0.0 < fraction < 1.0:
        raise ValueError("labelled_fraction must lie strictly between zero and one.")
    records = list(descriptors)
    if not records:
        raise ValueError("Training label assignment requires at least one trial.")
    groups: dict[tuple[int, int], list[TrialDescriptor]] = {}
    for descriptor in records:
        groups.setdefault(
            (descriptor.subject_id, descriptor.physical_activity_id), []
        ).append(descriptor)

    labelled: set[int] = set()
    unlabelled: set[int] = set()
    strata: list[Mapping[str, Any]] = []
    for (subject_id, activity_id), group in sorted(groups.items()):
        ordered = sorted(group, key=lambda item: item.trial_number)
        trial_numbers = [item.trial_number for item in ordered]
        expected_numbers = list(TRIAL_NUMBERS)
        anomaly_shortfall = (
            bool(allow_known_anomaly_exclusion)
            and subject_id == 14
            and activity_id == 3
            and trial_numbers == [1, 3, 4, 5]
        )
        if trial_numbers != expected_numbers and not anomaly_shortfall:
            raise USCHADProtocolError(
                "Each train (subject, old activity) stratum must contain physical "
                f"trials 1..5; subject={subject_id}, activity={activity_id}, "
                f"observed={trial_numbers}."
            )
        # round-half-up gives exactly four of five at 0.8 and three of four for
        # the explicit known-anomaly exclusion. Keep at least one on each side.
        labelled_count = int(math.floor(len(ordered) * fraction + 0.5))
        labelled_count = min(len(ordered) - 1, max(1, labelled_count))
        sequence = np.random.SeedSequence(
            [int(seed) & 0xFFFFFFFF, int(subject_id), int(activity_id), 0x4A30]
        )
        rng = np.random.default_rng(sequence)
        permutation = rng.permutation(len(ordered)).tolist()
        labelled_positions = set(permutation[:labelled_count])
        stratum_labelled = {
            ordered[position].trial_id for position in labelled_positions
        }
        stratum_unlabelled = {
            item.trial_id
            for position, item in enumerate(ordered)
            if position not in labelled_positions
        }
        labelled.update(stratum_labelled)
        unlabelled.update(stratum_unlabelled)
        strata.append(
            {
                "subject_id": int(subject_id),
                "physical_activity_id": int(activity_id),
                "available_trial_count": len(ordered),
                "labelled_trial_numbers": sorted(
                    item.trial_number
                    for item in ordered
                    if item.trial_id in stratum_labelled
                ),
                "unlabelled_trial_numbers": sorted(
                    item.trial_number
                    for item in ordered
                    if item.trial_id in stratum_unlabelled
                ),
                "known_anomaly_exclusion_shortfall": anomaly_shortfall,
            }
        )
    all_ids = {item.trial_id for item in records}
    if labelled & unlabelled or labelled | unlabelled != all_ids:
        raise USCHADProtocolError("Train label assignment is not an exact partition.")
    return StratifiedLabelAssignment(
        labelled_trial_ids=frozenset(labelled),
        unlabelled_trial_ids=frozenset(unlabelled),
        seed=int(seed),
        labelled_fraction=fraction,
        strata=tuple(strata),
    )


class USCHADTrialDataset(Dataset):
    """Lazy full-trial dataset with an explicit supervision visibility mask."""

    def __init__(
        self,
        descriptors: Sequence[TrialDescriptor],
        *,
        split: SubjectSplit,
        partition: str,
        label_regime: str,
        old_activity_ids: Iterable[int],
        new_activity_ids: Iterable[int],
        normalization: NormalizationStats,
        trajectory_label_trial_ids: Iterable[int] = (),
        outer_test_locked: bool = True,
    ) -> None:
        partition = str(partition).lower()
        if partition not in {"train", "validation", "test"}:
            raise ValueError("partition must be train, validation, or test.")
        validate_subject_split(split)
        old, new = resolve_activity_split(old_activity_ids, new_activity_ids)
        regime = _validate_regime(label_regime)
        allowed_subjects = {
            "train": set(split.train_subjects),
            "validation": set(split.validation_subjects),
            "test": set(split.test_subjects),
        }[partition]
        records = list(descriptors)
        if not records:
            raise ValueError(f"{partition} dataset cannot be empty.")
        leaking = [item.trial_key for item in records if item.subject_id not in allowed_subjects]
        if leaking:
            raise USCHADProtocolError(
                f"{partition} dataset contains subjects outside its fold partition: "
                f"{leaking[:10]}."
            )
        if len({item.trial_id for item in records}) != len(records):
            raise USCHADProtocolError(f"{partition} dataset contains duplicate trial ids.")
        self.descriptors = tuple(sorted(records, key=lambda item: item.trial_id))
        self.split = split
        self.partition = partition
        self.label_regime = regime
        self.old_activity_ids = old
        self.new_activity_ids = new
        self.normalization = normalization
        label_ids = frozenset(int(value) for value in trajectory_label_trial_ids)
        record_ids = {item.trial_id for item in self.descriptors}
        unknown_label_ids = sorted(label_ids - record_ids)
        if unknown_label_ids:
            raise USCHADProtocolError(
                "Trajectory-label ids are absent from this dataset: "
                f"{unknown_label_ids[:10]}."
            )
        non_old_label_ids = sorted(
            item.trial_id
            for item in self.descriptors
            if item.trial_id in label_ids
            and item.physical_activity_id not in self.old_activity_ids
        )
        if non_old_label_ids:
            raise USCHADProtocolError(
                "New-class labels cannot be exposed to J0 trajectory CE: "
                f"{non_old_label_ids[:10]}."
            )
        if regime == "J0-U" and label_ids:
            raise USCHADProtocolError("J0-U cannot receive trajectory-label ids.")
        if partition == "test" and label_ids:
            raise USCHADProtocolError("Outer-test labels cannot be model-visible.")
        self.trajectory_label_trial_ids = label_ids
        self._outer_test_unlocked = partition != "test" or not bool(outer_test_locked)
        self._outer_test_checkpoint_identity: str | None = None
        self._sensor_trials_loaded = 0

    def __len__(self) -> int:
        return len(self.descriptors)

    @property
    def outer_test_locked(self) -> bool:
        return self.partition == "test" and not self._outer_test_unlocked

    @property
    def sensor_trials_loaded(self) -> int:
        return int(self._sensor_trials_loaded)

    def unlock_outer_test_for_evaluation(self, checkpoint_identity: str) -> None:
        """Explicitly unlock test sensors only after checkpoint selection."""

        if self.partition != "test":
            raise RuntimeError("Only the outer-test dataset has an access lock.")
        identity = str(checkpoint_identity).strip()
        if not identity:
            raise ValueError("A non-empty selected checkpoint identity is required.")
        self._outer_test_checkpoint_identity = identity
        self._outer_test_unlocked = True

    def _label_visible(self, descriptor: TrialDescriptor) -> bool:
        return descriptor.trial_id in self.trajectory_label_trial_ids

    def __getitem__(self, index: int) -> dict[str, Any]:
        if self.outer_test_locked:
            raise USCHADProtocolError(
                "Outer-test sensors are locked during training/checkpoint selection. "
                "Call unlock_outer_test_for_evaluation(checkpoint_identity) only after "
                "a checkpoint has been selected."
            )
        descriptor = self.descriptors[int(index)]
        signal = _load_sensor_readings(descriptor.path)
        self._sensor_trials_loaded += 1
        model_signal = (
            signal - np.asarray(self.normalization.mean, dtype=np.float32)[None, :]
        ) / np.asarray(self.normalization.std, dtype=np.float32)[None, :]
        if not np.all(np.isfinite(model_signal)):
            raise USCHADProtocolError(
                f"Fold normalization produced non-finite values: {descriptor.path}."
            )
        visible = self._label_visible(descriptor)
        is_old = descriptor.physical_activity_id in self.old_activity_ids
        return {
            # Channel-first tensors are directly compatible with temporal
            # window extraction and Conv1d without dropping the raw trial tail.
            "physical_trial": torch.from_numpy(
                np.ascontiguousarray(signal.T, dtype=np.float32)
            ),
            "model_trial": torch.from_numpy(
                np.ascontiguousarray(model_signal.T, dtype=np.float32)
            ),
            "length": int(signal.shape[0]),
            "physical_activity_id": int(descriptor.physical_activity_id),
            "class_index": int(descriptor.class_index),
            "activity_name": descriptor.activity_name,
            "is_old_class": bool(is_old),
            "label_visible": bool(visible),
            "trajectory_label_mask": bool(visible),
            "supervision_target": int(descriptor.class_index) if visible else IGNORE_INDEX,
            "subject_id": int(descriptor.subject_id),
            "trial_number": int(descriptor.trial_number),
            "trial_id": int(descriptor.trial_id),
            "trial_key": descriptor.trial_key,
            "relative_path": descriptor.relative_path,
            "known_label_anomaly": bool(descriptor.known_label_anomaly),
            "metadata_issues": tuple(descriptor.metadata_issues),
        }

    def access_audit(self) -> dict[str, Any]:
        return {
            "partition": self.partition,
            "sensor_trials_loaded": int(self._sensor_trials_loaded),
            "outer_test_locked": bool(self.outer_test_locked),
            "outer_test_checkpoint_identity": self._outer_test_checkpoint_identity,
            "outer_test_sensor_trials_loaded": (
                int(self._sensor_trials_loaded) if self.partition == "test" else 0
            ),
        }


def pad_trial_batch(
    batch: Sequence[Mapping[str, Any]], *, padding_value: float = 0.0
) -> dict[str, Any]:
    """Right-pad variable raw trials and return a sample-validity mask.

    Both ``physical_trial`` and ``model_trial`` are returned as ``[B,6,Tmax]``;
    ``sample_mask`` is ``[B,Tmax]`` and must gate every temporal operation.
    """

    if not batch:
        raise ValueError("Cannot collate an empty trial batch.")
    if not math.isfinite(float(padding_value)):
        raise ValueError("padding_value must be finite.")
    lengths = torch.as_tensor([int(item["length"]) for item in batch], dtype=torch.long)
    if torch.any(lengths <= 0):
        raise ValueError(f"Every trial length must be positive: {lengths.tolist()}.")
    maximum = int(lengths.max().item())
    batch_size = len(batch)
    physical = torch.full(
        (batch_size, len(CHANNEL_NAMES), maximum),
        float(padding_value),
        dtype=torch.float32,
    )
    model = torch.full_like(physical, float(padding_value))
    mask = torch.zeros((batch_size, maximum), dtype=torch.bool)
    for row, item in enumerate(batch):
        physical_item = torch.as_tensor(item["physical_trial"], dtype=torch.float32)
        model_item = torch.as_tensor(item["model_trial"], dtype=torch.float32)
        length = int(lengths[row].item())
        expected = (len(CHANNEL_NAMES), length)
        if tuple(physical_item.shape) != expected or tuple(model_item.shape) != expected:
            raise ValueError(
                f"Trial tensors must both have shape {expected}; got "
                f"{tuple(physical_item.shape)} and {tuple(model_item.shape)}."
            )
        physical[row, :, :length] = physical_item
        model[row, :, :length] = model_item
        mask[row, :length] = True

    label_visible = torch.as_tensor(
        [bool(item["label_visible"]) for item in batch], dtype=torch.bool
    )
    supervision = torch.as_tensor(
        [int(item["supervision_target"]) for item in batch], dtype=torch.long
    )
    if torch.any((~label_visible) & (supervision != IGNORE_INDEX)):
        raise USCHADProtocolError(
            "A hidden label has a non-ignore supervision target in the collate batch."
        )
    return {
        "physical_trials": physical,
        "model_trials": model,
        "sample_mask": mask,
        "lengths": lengths,
        "physical_activity_ids": torch.as_tensor(
            [int(item["physical_activity_id"]) for item in batch], dtype=torch.long
        ),
        "class_indices": torch.as_tensor(
            [int(item["class_index"]) for item in batch], dtype=torch.long
        ),
        "is_old_class": torch.as_tensor(
            [bool(item["is_old_class"]) for item in batch], dtype=torch.bool
        ),
        "label_visible": label_visible,
        "trajectory_label_mask": label_visible.clone(),
        "supervision_targets": supervision,
        "subject_ids": torch.as_tensor(
            [int(item["subject_id"]) for item in batch], dtype=torch.long
        ),
        "trial_numbers": torch.as_tensor(
            [int(item["trial_number"]) for item in batch], dtype=torch.long
        ),
        "trial_ids": torch.as_tensor(
            [int(item["trial_id"]) for item in batch], dtype=torch.long
        ),
        "trial_keys": tuple(str(item["trial_key"]) for item in batch),
        "activity_names": tuple(str(item["activity_name"]) for item in batch),
        "relative_paths": tuple(str(item["relative_path"]) for item in batch),
        "known_label_anomaly": torch.as_tensor(
            [bool(item["known_label_anomaly"]) for item in batch], dtype=torch.bool
        ),
    }


@dataclass(frozen=True)
class USCHADFoldDatasets:
    """Train/validation/locked-test datasets plus immutable protocol audits."""

    train: USCHADTrialDataset
    validation: USCHADTrialDataset
    outer_test: USCHADTrialDataset
    split: SubjectSplit
    old_activity_ids: tuple[int, ...]
    new_activity_ids: tuple[int, ...]
    normalization: NormalizationStats
    manifest_audit: Mapping[str, Any]
    protocol_audit: Mapping[str, Any]

    def audit_dict(self) -> dict[str, Any]:
        result = copy.deepcopy(dict(self.protocol_audit))
        result["runtime_sensor_access"] = {
            "train": self.train.access_audit(),
            "validation": self.validation.access_audit(),
            "outer_test": self.outer_test.access_audit(),
        }
        return result


def build_uschad_fold_datasets(
    root: str | Path = DEFAULT_USCHAD_ROOT,
    *,
    fold: int,
    label_regime: str,
    old_activity_ids: Iterable[int] = DEFAULT_OLD_ACTIVITY_IDS,
    new_activity_ids: Iterable[int] | None = None,
    anomaly_policy: str = "report",
    normalization_epsilon: float = 1e-6,
    labelled_fraction: float = 0.8,
    label_seed: int = 0,
    require_complete: bool = True,
    inspect_mat_metadata: bool = True,
) -> USCHADFoldDatasets:
    """Build J0 train-old, validation-old, and locked outer-test-all datasets."""

    split = subject_split_for_fold(fold)
    validate_subject_split(split)
    old, new = resolve_activity_split(old_activity_ids, new_activity_ids)
    regime = _validate_regime(label_regime)
    descriptors, manifest_audit = discover_trial_manifest(
        root,
        anomaly_policy=anomaly_policy,
        require_complete=require_complete,
        inspect_mat_metadata=inspect_mat_metadata,
    )

    train = [
        item
        for item in descriptors
        if item.subject_id in split.train_subjects
        and item.physical_activity_id in old
    ]
    validation = [
        item
        for item in descriptors
        if item.subject_id in split.validation_subjects
        and item.physical_activity_id in old
    ]
    outer_test = [
        item for item in descriptors if item.subject_id in split.test_subjects
    ]
    if not train or not validation or not outer_test:
        raise USCHADProtocolError(
            "Fold construction produced an empty train, validation, or outer-test set."
        )
    normalization = fit_train_old_normalization(
        train,
        expected_train_subjects=split.train_subjects,
        expected_old_activity_ids=old,
        epsilon=normalization_epsilon,
    )
    label_assignment = stratified_train_label_assignment(
        train,
        seed=label_seed,
        labelled_fraction=labelled_fraction,
        allow_known_anomaly_exclusion=anomaly_policy == "exclude",
    )
    train_label_ids: frozenset[int] = (
        label_assignment.labelled_trial_ids
        if regime == "J0-T"
        else frozenset()
    )
    # Validation subjects are disjoint and never optimized.  J0-T exposes all
    # validation-old labels solely for validation loss/checkpoint selection;
    # J0-U exposes none. Outer-test truth remains evaluator-only in both arms.
    validation_label_ids: frozenset[int] = (
        frozenset(item.trial_id for item in validation)
        if regime == "J0-T"
        else frozenset()
    )
    train_dataset = USCHADTrialDataset(
        train,
        split=split,
        partition="train",
        label_regime=regime,
        old_activity_ids=old,
        new_activity_ids=new,
        normalization=normalization,
        trajectory_label_trial_ids=train_label_ids,
    )
    validation_dataset = USCHADTrialDataset(
        validation,
        split=split,
        partition="validation",
        label_regime=regime,
        old_activity_ids=old,
        new_activity_ids=new,
        normalization=normalization,
        trajectory_label_trial_ids=validation_label_ids,
    )
    test_dataset = USCHADTrialDataset(
        outer_test,
        split=split,
        partition="test",
        label_regime=regime,
        old_activity_ids=old,
        new_activity_ids=new,
        normalization=normalization,
        trajectory_label_trial_ids=(),
        outer_test_locked=True,
    )

    subject_sets = {
        "train": sorted({item.subject_id for item in train}),
        "validation": sorted({item.subject_id for item in validation}),
        "test": sorted({item.subject_id for item in outer_test}),
    }
    trial_sets = {
        "train": {item.trial_id for item in train},
        "validation": {item.trial_id for item in validation},
        "test": {item.trial_id for item in outer_test},
    }
    overlap = {
        "train_validation": sorted(trial_sets["train"] & trial_sets["validation"]),
        "train_test": sorted(trial_sets["train"] & trial_sets["test"]),
        "validation_test": sorted(
            trial_sets["validation"] & trial_sets["test"]
        ),
    }
    if any(overlap.values()):
        raise USCHADProtocolError(f"Trial identity leakage detected: {overlap}.")
    protocol_identity = {
        "schema": PROTOCOL_SCHEMA,
        "fold": int(fold),
        "subject_split": split.as_dict(),
        "old_activity_ids_1based": list(old),
        "new_activity_ids_1based": list(new),
        "label_regime": regime,
        "label_visibility": {
            "J0-U": "no model-visible activity labels",
            "J0-T": (
                "seeded 4/1 labels per train subject/old activity; all validation-old "
                "labels available only for validation/checkpoint selection"
            ),
            "outer_test_model_visible": False,
        },
        "train_label_assignment": label_assignment.as_dict(),
        "manifest_sha256": manifest_audit["manifest_sha256"],
        "normalization": normalization.as_dict(),
    }
    protocol_audit = {
        **protocol_identity,
        "identity_sha256": _identity_sha256(protocol_identity),
        "subject_sets_observed": subject_sets,
        "trial_id_overlap": overlap,
        "trial_counts": {
            "train_old": len(train),
            "train_old_labelled_assignment": len(
                label_assignment.labelled_trial_ids
            ),
            "train_old_unlabelled_assignment": len(
                label_assignment.unlabelled_trial_ids
            ),
            "train_model_visible_labels": len(train_label_ids),
            "validation_old": len(validation),
            "validation_model_visible_labels": len(validation_label_ids),
            "outer_test_all": len(outer_test),
            "outer_test_model_visible_labels": 0,
        },
        "sensor_access_during_construction": {
            "normalization_train_old_trials_loaded": len(train),
            "validation_sensor_trials_loaded": 0,
            "outer_test_sensor_trials_loaded": 0,
            "outer_test_locked_until_checkpoint_selected": True,
        },
        "full_trial_policy": "all original sensor samples; no crop/window tail drop",
        "physical_label_source": "filename activity id; internal conflicts reported",
    }
    return USCHADFoldDatasets(
        train=train_dataset,
        validation=validation_dataset,
        outer_test=test_dataset,
        split=split,
        old_activity_ids=old,
        new_activity_ids=new,
        normalization=normalization,
        manifest_audit=manifest_audit,
        protocol_audit=protocol_audit,
    )


__all__ = [
    "ACTIVITY_IDS",
    "ACTIVITY_NAMES",
    "CHANNEL_NAMES",
    "DEFAULT_NEW_ACTIVITY_IDS",
    "DEFAULT_OLD_ACTIVITY_IDS",
    "DEFAULT_USCHAD_ROOT",
    "IGNORE_INDEX",
    "KNOWN_LABEL_ANOMALY_KEY",
    "LABEL_REGIMES",
    "NormalizationStats",
    "PROTOCOL_SCHEMA",
    "SAMPLE_RATE_HZ",
    "StratifiedLabelAssignment",
    "SUBJECT_IDS",
    "SUBJECT_SPLITS",
    "SubjectSplit",
    "TRIAL_NUMBERS",
    "TrialDescriptor",
    "USCHADFoldDatasets",
    "USCHADProtocolError",
    "USCHADTrialDataset",
    "build_uschad_fold_datasets",
    "discover_trial_manifest",
    "fit_train_old_normalization",
    "fixed_subject_manifest",
    "pad_trial_batch",
    "resolve_activity_split",
    "stratified_train_label_assignment",
    "subject_split_for_fold",
    "validate_subject_split",
]
