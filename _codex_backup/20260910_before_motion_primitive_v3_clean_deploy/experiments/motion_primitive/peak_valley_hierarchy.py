"""Deterministic sample-level peak/valley motion-primitive discovery.

This module is intentionally independent from the existing window encoder and
segmentation pipeline.  It provides a label-free, fit/transform style API for:

1. detecting sample-level extrema from six smoothed sensor axes;
2. forming a complete, non-overlapping partition of every observed trial;
3. extracting instance-normalised fixed-length shape features while retaining
   physical statistics as a separate side channel;
4. fitting a train-only weighted PCA/KMeans child-primitive codebook; and
5. registering repeated peak/valley parent motifs without replacing children.

Activity labels are deliberately absent from every public input type.  Fitting
functions therefore cannot accidentally use held-out activity labels.  The
caller remains responsible for passing only the intended fit-subject/old-class
trials to ``fit_segmenter``, ``fit_codebook``, and ``fit_parent_catalog``.

The observed trial span ends at ``signal.shape[1]``.  If the source was rebuilt
from complete sliding windows, any final incomplete raw-trial tail is outside
this module's scope and must be disclosed by the caller.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Iterable, Literal, Mapping, Sequence

import numpy as np
import sklearn
from scipy.optimize import linear_sum_assignment
from scipy.signal import find_peaks
from sklearn.cluster import KMeans


SCHEMA_VERSION = 4
ALGORITHM_REVISION = "sample_peak_valley_hierarchy_v4"
DEFAULT_CHANNEL_NAMES = (
    "acc_x",
    "acc_y",
    "acc_z",
    "gyro_x",
    "gyro_y",
    "gyro_z",
)
EVENT_KINDS = ("peak", "valley")
EPS = 1e-12
Identifier = int | str
Weighting = Literal["uniform", "trial_equal", "subject_equal", "subject_trial_equal"]


def _require_finite(name: str, value: float) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite, got {value!r}.")
    return result


def _exact_int_array(name: str, values, *, ndim: int | None = None) -> np.ndarray:
    raw = np.asarray(values)
    if ndim is not None and raw.ndim != int(ndim):
        raise ValueError(f"{name} must be {ndim}D, got shape {raw.shape}.")
    if raw.dtype.kind not in "iuf":
        raise TypeError(f"{name} must contain numeric integer values.")
    if raw.dtype.kind == "f" and not np.all(np.isfinite(raw)):
        raise ValueError(f"{name} contains non-finite values.")
    integers = raw.astype(np.int64)
    if not np.array_equal(raw, integers):
        raise ValueError(f"{name} must contain exact integers without truncation.")
    return integers


def _validate_identifier(name: str, value: Identifier) -> Identifier:
    if isinstance(value, bool) or not isinstance(value, (int, str, np.integer)):
        raise TypeError(f"{name} must be an int or str, got {type(value).__name__}.")
    if isinstance(value, (int, np.integer)):
        return int(value)
    if not value:
        raise ValueError(f"{name} must not be empty.")
    return value


def _identifier_payload(value: Identifier) -> dict:
    value = _validate_identifier("identifier", value)
    return {
        "type": "int" if isinstance(value, int) else "str",
        "value": value,
    }


def _identifier_sort_key(value: Identifier) -> str:
    return json.dumps(_identifier_payload(value), sort_keys=True, separators=(",", ":"))


def _identifier_from_payload(payload: Mapping) -> Identifier:
    if not isinstance(payload, Mapping) or set(payload) != {"type", "value"}:
        raise ValueError("Serialized identifier must contain exactly type/value.")
    kind = payload["type"]
    value = payload["value"]
    if kind == "int" and isinstance(value, int) and not isinstance(value, bool):
        return int(value)
    if kind == "str" and isinstance(value, str) and value:
        return value
    raise ValueError(f"Invalid serialized identifier {payload!r}.")


def _jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    return value


def canonical_sha256(payload: Mapping) -> str:
    """Hash a finite JSON-compatible mapping using one canonical encoding."""

    encoded = json.dumps(
        _jsonable(payload),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _validate_sha256(name: str, value: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase hexadecimal SHA-256 string.")
    return value


def _array_payload(values: np.ndarray) -> dict:
    array = np.ascontiguousarray(np.asarray(values))
    if array.dtype.kind == "f" and not np.all(np.isfinite(array)):
        raise ValueError("State array contains non-finite values.")
    return {
        "dtype": str(array.dtype),
        "shape": list(array.shape),
        "values": array.tolist(),
    }


def _array_from_payload(payload: Mapping, *, name: str, dtype) -> np.ndarray:
    if not isinstance(payload, Mapping):
        raise TypeError(f"{name} payload must be a mapping.")
    if set(payload) != {"dtype", "shape", "values"}:
        raise ValueError(f"{name} array payload has an unexpected schema.")
    array = np.asarray(payload["values"], dtype=dtype)
    expected_shape = tuple(int(item) for item in payload["shape"])
    if array.shape != expected_shape:
        raise ValueError(
            f"{name} shape mismatch: encoded={expected_shape}, values={array.shape}."
        )
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains non-finite values.")
    return array


def _hash_array(digest: "hashlib._Hash", name: str, values: np.ndarray) -> None:
    array = np.asarray(values)
    if array.dtype.kind in "fc" and not np.all(np.isfinite(array)):
        raise ValueError(f"Cannot hash non-finite {name}.")
    if array.dtype.kind == "f":
        canonical = np.ascontiguousarray(array, dtype="<f8")
    elif array.dtype.kind in "iu":
        canonical = np.ascontiguousarray(array, dtype="<i8")
    else:
        raise TypeError(f"Unsupported array dtype for {name}: {array.dtype}.")
    digest.update(name.encode("utf-8"))
    digest.update(str(canonical.shape).encode("ascii"))
    digest.update(canonical.tobytes(order="C"))


@dataclass(frozen=True)
class TrialSignal:
    """One observed six-axis trial and optional time-aligned feature sequence."""

    trial_id: Identifier
    subject_id: Identifier
    signal: np.ndarray
    feature_values: np.ndarray | None = None
    feature_sample_positions: np.ndarray | None = None

    def validated(self, expected_feature_dim: int | None = None) -> "TrialSignal":
        trial_id = _validate_identifier("trial_id", self.trial_id)
        subject_id = _validate_identifier("subject_id", self.subject_id)
        signal = np.asarray(self.signal, dtype=np.float64)
        if signal.ndim != 2 or signal.shape[0] != 6 or signal.shape[1] < 2:
            raise ValueError(f"signal must have shape [6,T>=2], got {signal.shape}.")
        if not np.all(np.isfinite(signal)):
            raise ValueError("signal contains non-finite samples.")
        if (self.feature_values is None) != (self.feature_sample_positions is None):
            raise ValueError(
                "feature_values and feature_sample_positions must be supplied together."
            )
        features = None
        positions = None
        if self.feature_values is not None:
            features = np.asarray(self.feature_values, dtype=np.float64)
            positions = _exact_int_array(
                "feature_sample_positions", self.feature_sample_positions, ndim=1
            )
            if features.ndim != 2 or len(features) < 2 or features.shape[1] < 1:
                raise ValueError("feature_values must have shape [M>=2,D>=1].")
            if positions.shape != (len(features),):
                raise ValueError("feature positions must match feature rows.")
            if not np.all(np.isfinite(features)):
                raise ValueError("feature_values contains non-finite values.")
            if positions[0] < 0 or positions[-1] >= signal.shape[1]:
                raise ValueError("feature positions must lie inside [0,T).")
            if np.any(np.diff(positions) <= 0):
                raise ValueError("feature positions must be strictly increasing.")
            if expected_feature_dim is not None and features.shape[1] != int(
                expected_feature_dim
            ):
                raise ValueError(
                    f"Feature dimension {features.shape[1]} != state dimension "
                    f"{expected_feature_dim}."
                )
        return TrialSignal(trial_id, subject_id, signal, features, positions)


@dataclass(frozen=True)
class PeakValleyConfig:
    """Detection and child-feature parameters expressed in physical time."""

    sample_rate_hz: float = 100.0
    channel_names: tuple[str, ...] = DEFAULT_CHANNEL_NAMES
    smoothing_seconds: float = 0.15
    extrema_min_distance_seconds: float = 0.20
    prominence_mad_multiplier: float = 1.5
    mad_scale_floor: float = 1e-8
    axis_vote_tolerance_seconds: float = 0.10
    min_axis_votes: int = 2
    min_segment_seconds: float = 0.25
    feature_confirmation: bool = False
    feature_context_seconds: float = 0.50
    feature_score_quantile: float = 0.50
    shape_points: int = 64
    shape_scale_floor: float = 1e-6
    statistics_epsilon: float = 1e-8

    def validate(self) -> "PeakValleyConfig":
        positive = {
            "sample_rate_hz": self.sample_rate_hz,
            "smoothing_seconds": self.smoothing_seconds,
            "extrema_min_distance_seconds": self.extrema_min_distance_seconds,
            "prominence_mad_multiplier": self.prominence_mad_multiplier,
            "mad_scale_floor": self.mad_scale_floor,
            "axis_vote_tolerance_seconds": self.axis_vote_tolerance_seconds,
            "min_segment_seconds": self.min_segment_seconds,
            "feature_context_seconds": self.feature_context_seconds,
            "shape_scale_floor": self.shape_scale_floor,
            "statistics_epsilon": self.statistics_epsilon,
        }
        for name, value in positive.items():
            if _require_finite(name, value) <= 0:
                raise ValueError(f"{name} must be positive.")
        if (
            isinstance(self.min_axis_votes, bool)
            or int(self.min_axis_votes) != self.min_axis_votes
            or not 0 <= int(self.min_axis_votes) <= 6
        ):
            raise ValueError("min_axis_votes must be an integer in [0,6].")
        if (
            isinstance(self.shape_points, bool)
            or int(self.shape_points) != self.shape_points
            or int(self.shape_points) < 2
        ):
            raise ValueError("shape_points must be an integer >=2.")
        if not 0.0 <= _require_finite(
            "feature_score_quantile", self.feature_score_quantile
        ) <= 1.0:
            raise ValueError("feature_score_quantile must lie in [0,1].")
        names = tuple(str(item) for item in self.channel_names)
        if len(names) != 6 or len(set(names)) != 6 or any(not item for item in names):
            raise ValueError("channel_names must contain six unique non-empty names.")
        return replace(
            self,
            sample_rate_hz=float(self.sample_rate_hz),
            channel_names=names,
            min_axis_votes=int(self.min_axis_votes),
            shape_points=int(self.shape_points),
            feature_confirmation=bool(self.feature_confirmation),
        )

    def to_dict(self) -> dict:
        return _jsonable(asdict(self))

    @classmethod
    def from_dict(cls, payload: Mapping) -> "PeakValleyConfig":
        values = dict(payload)
        values["channel_names"] = tuple(values["channel_names"])
        return cls(**values).validate()

    @property
    def smoothing_samples(self) -> int:
        count = max(1, int(round(self.sample_rate_hz * self.smoothing_seconds)))
        return count if count % 2 == 1 else count + 1

    @property
    def extrema_min_distance_samples(self) -> int:
        return max(1, int(round(self.sample_rate_hz * self.extrema_min_distance_seconds)))

    @property
    def axis_vote_tolerance_samples(self) -> int:
        return max(0, int(round(self.sample_rate_hz * self.axis_vote_tolerance_seconds)))

    @property
    def min_segment_samples(self) -> int:
        return max(1, int(round(self.sample_rate_hz * self.min_segment_seconds)))

    @property
    def feature_context_samples(self) -> int:
        return max(1, int(round(self.sample_rate_hz * self.feature_context_seconds)))


@dataclass(frozen=True)
class TurningEvent:
    sample_index: int
    kind: Literal["peak", "valley"]
    canonical_prominence: float
    canonical_normalized_prominence: float
    axis_indices: tuple[int, ...]
    axis_sample_indices: tuple[int, ...]
    axis_prominences: tuple[float, ...]
    axis_normalized_prominences: tuple[float, ...]
    feature_change_score: float | None = None
    source: str = "detected"

    @property
    def axis_votes(self) -> int:
        return len(self.axis_indices)

    @property
    def mean_normalized_prominence(self) -> float:
        return (
            float(np.mean(self.axis_normalized_prominences))
            if self.axis_normalized_prominences
            else 0.0
        )

    def to_dict(self) -> dict:
        return _jsonable(asdict(self))


@dataclass(frozen=True)
class SampleSegment:
    segment_index: int
    start_sample: int
    end_sample_exclusive: int
    left_event_kind: str | None
    right_event_kind: str | None

    @property
    def sample_count(self) -> int:
        return int(self.end_sample_exclusive - self.start_sample)

    def to_dict(self) -> dict:
        result = asdict(self)
        result["sample_count"] = self.sample_count
        return result


@dataclass(frozen=True)
class SegmentedTrial:
    trial_id: Identifier
    subject_id: Identifier
    sample_count: int
    boundaries: np.ndarray
    events: tuple[TurningEvent, ...]
    segments: tuple[SampleSegment, ...]
    state_sha256: str
    source: str
    diagnostics: dict

    def validate(self, minimum_samples: int = 1) -> "SegmentedTrial":
        _validate_identifier("trial_id", self.trial_id)
        _validate_identifier("subject_id", self.subject_id)
        if int(self.sample_count) < 2:
            raise ValueError("Segmented trial must contain at least two samples.")
        boundaries = _exact_int_array("boundaries", self.boundaries, ndim=1)
        if boundaries.ndim != 1 or len(boundaries) < 2:
            raise ValueError("boundaries must be a 1D vector of length >=2.")
        if boundaries[0] != 0 or boundaries[-1] != int(self.sample_count):
            raise ValueError("boundaries must exactly cover [0,sample_count].")
        if np.any(np.diff(boundaries) <= 0):
            raise ValueError("boundaries must be strictly increasing.")
        if int(self.sample_count) >= int(minimum_samples) and len(boundaries) > 2:
            if np.any(np.diff(boundaries) < int(minimum_samples)):
                raise ValueError("A segment is shorter than the configured minimum.")
        if len(self.events) != len(boundaries) - 2:
            raise ValueError("Every internal boundary must have exactly one event.")
        if [event.sample_index for event in self.events] != boundaries[1:-1].tolist():
            raise ValueError("Event samples do not match internal boundaries.")
        for event in self.events:
            if event.kind not in EVENT_KINDS:
                raise ValueError("Turning event kind must be peak or valley.")
            for name, value in (
                ("canonical_prominence", event.canonical_prominence),
                (
                    "canonical_normalized_prominence",
                    event.canonical_normalized_prominence,
                ),
            ):
                if _require_finite(name, value) < 0:
                    raise ValueError(f"{name} must be non-negative.")
            lengths = {
                len(event.axis_indices),
                len(event.axis_sample_indices),
                len(event.axis_prominences),
                len(event.axis_normalized_prominences),
            }
            if len(lengths) != 1:
                raise ValueError("Turning event per-axis metadata lengths differ.")
            if len(set(event.axis_indices)) != len(event.axis_indices) or any(
                not 0 <= int(axis) < 6 for axis in event.axis_indices
            ):
                raise ValueError("Turning event axis indices must be unique in [0,6).")
            if any(
                _require_finite("axis prominence", value) < 0
                for value in (*event.axis_prominences, *event.axis_normalized_prominences)
            ):
                raise ValueError("Axis prominences must be non-negative.")
            if event.feature_change_score is not None:
                score = _require_finite(
                    "feature_change_score", event.feature_change_score
                )
                if not 0.0 <= score <= 2.0 + 1e-9:
                    raise ValueError("Cosine feature-change score must lie in [0,2].")
        if len(self.segments) != len(boundaries) - 1:
            raise ValueError("Segment count does not match partition boundaries.")
        for index, segment in enumerate(self.segments):
            if (
                segment.segment_index != index
                or segment.start_sample != int(boundaries[index])
                or segment.end_sample_exclusive != int(boundaries[index + 1])
            ):
                raise ValueError("Segment metadata does not match the partition.")
        _validate_sha256("state_sha256", self.state_sha256)
        return self

    def to_dict(self) -> dict:
        return {
            "trial_id": _identifier_payload(self.trial_id),
            "subject_id": _identifier_payload(self.subject_id),
            "sample_count": int(self.sample_count),
            "boundaries": self.boundaries.astype(int).tolist(),
            "events": [event.to_dict() for event in self.events],
            "segments": [segment.to_dict() for segment in self.segments],
            "state_sha256": self.state_sha256,
            "source": self.source,
            "diagnostics": _jsonable(self.diagnostics),
        }


@dataclass(frozen=True)
class PeakValleyState:
    config: PeakValleyConfig
    feature_threshold: float | None
    feature_dim: int | None
    fit_data_sha256: str
    fit_trial_count: int
    feature_calibration: dict
    algorithm_revision: str = ALGORITHM_REVISION
    schema_version: int = SCHEMA_VERSION

    def validate(self) -> "PeakValleyState":
        config = self.config.validate()
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError(f"Unsupported peak/valley schema {self.schema_version}.")
        if self.algorithm_revision != ALGORITHM_REVISION:
            raise ValueError("Peak/valley algorithm revision mismatch.")
        _validate_sha256("fit_data_sha256", self.fit_data_sha256)
        if int(self.fit_trial_count) < 1:
            raise ValueError("fit_trial_count must be positive.")
        if config.feature_confirmation:
            if self.feature_threshold is None or self.feature_dim is None:
                raise ValueError("Feature-confirmed state lacks threshold/dimension.")
            threshold = _require_finite("feature_threshold", self.feature_threshold)
            if not 0.0 <= threshold <= 2.0 + 1e-9:
                raise ValueError("feature_threshold must lie in cosine-distance range [0,2].")
            if int(self.feature_dim) < 1:
                raise ValueError("feature_dim must be positive.")
        elif self.feature_threshold is not None or self.feature_dim is not None:
            raise ValueError("Feature-disabled state must not contain feature calibration.")
        return self

    def _payload(self) -> dict:
        return {
            "schema_version": int(self.schema_version),
            "algorithm_revision": self.algorithm_revision,
            "config": self.config.to_dict(),
            "feature_threshold": self.feature_threshold,
            "feature_dim": self.feature_dim,
            "fit_data_sha256": self.fit_data_sha256,
            "fit_trial_count": int(self.fit_trial_count),
            "feature_calibration": _jsonable(self.feature_calibration),
        }

    def state_hash(self) -> str:
        self.validate()
        return canonical_sha256(self._payload())

    def to_dict(self) -> dict:
        result = self._payload()
        result["state_sha256"] = self.state_hash()
        return result

    @classmethod
    def from_dict(cls, payload: Mapping) -> "PeakValleyState":
        values = dict(payload)
        expected_hash = values.pop("state_sha256", None)
        if expected_hash is None:
            raise ValueError("Serialized peak/valley state lacks state_sha256.")
        config_payload = values.pop("config")
        state = cls(
            config=PeakValleyConfig.from_dict(config_payload),
            **values,
        ).validate()
        _validate_sha256("state_sha256", expected_hash)
        if expected_hash != state.state_hash():
            raise ValueError("Peak/valley state hash mismatch.")
        return state


@dataclass(frozen=True)
class _AxisEvent:
    sample_index: int
    kind: str
    axis_index: int
    prominence: float
    normalized_prominence: float


@dataclass(frozen=True)
class _CanonicalEvent:
    sample_index: int
    kind: str
    prominence: float
    normalized_prominence: float


def _validate_unique_trials(trials: Sequence[TrialSignal]) -> list[TrialSignal]:
    validated = [trial.validated() for trial in trials]
    if not validated:
        raise ValueError("At least one trial is required.")
    keys = [_identifier_sort_key(trial.trial_id) for trial in validated]
    if len(keys) != len(set(keys)):
        raise ValueError("trial_id values must be unique.")
    return sorted(validated, key=lambda item: _identifier_sort_key(item.trial_id))


def _fit_trial_hash(trials: Sequence[TrialSignal]) -> str:
    digest = hashlib.sha256()
    digest.update(ALGORITHM_REVISION.encode("ascii"))
    for trial in sorted(trials, key=lambda item: _identifier_sort_key(item.trial_id)):
        digest.update(_identifier_sort_key(trial.trial_id).encode("utf-8"))
        digest.update(_identifier_sort_key(trial.subject_id).encode("utf-8"))
        _hash_array(digest, "signal", trial.signal)
        if trial.feature_values is not None:
            _hash_array(digest, "feature_values", trial.feature_values)
            _hash_array(digest, "feature_sample_positions", trial.feature_sample_positions)
    return digest.hexdigest()


def smooth_six_axis(signal: np.ndarray, smoothing_samples: int) -> np.ndarray:
    """Apply the same centred moving-average filter independently to six axes."""

    values = np.asarray(signal, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] != 6 or values.shape[1] < 2:
        raise ValueError("signal must have shape [6,T>=2].")
    if not np.all(np.isfinite(values)):
        raise ValueError("signal contains non-finite values.")
    width = int(smoothing_samples)
    if width < 1:
        raise ValueError("smoothing_samples must be positive.")
    if width % 2 == 0:
        raise ValueError("smoothing_samples must be odd for centred filtering.")
    if width == 1:
        return values.copy()
    radius = width // 2
    mode = "reflect" if values.shape[1] > 2 else "edge"
    padded = np.pad(values, ((0, 0), (radius, radius)), mode=mode)
    kernel = np.full(width, 1.0 / width, dtype=np.float64)
    output = np.stack(
        [np.convolve(row, kernel, mode="valid") for row in padded], axis=0
    )
    if output.shape != values.shape:
        raise RuntimeError("Six-axis smoother changed the signal shape.")
    return output


def reconstruct_overlapping_windows(
    raw_windows: np.ndarray,
    window_start_samples: Sequence[int],
    window_indices: Sequence[int] | None = None,
    *,
    overlap_relative_tolerance: float = 5e-5,
) -> tuple[np.ndarray, dict]:
    """Overlap-add complete raw windows into one observed sample-level trial.

    ``raw_windows`` must already be in physical units.  The function refuses
    gaps, duplicate/non-increasing starts, incomplete window indices, and
    materially disagreeing overlap samples.  Averaging only removes harmless
    floating-point reconstruction error; it must not blend different signals.
    """

    windows = np.asarray(raw_windows, dtype=np.float64)
    if windows.ndim != 3 or windows.shape[1] != 6 or windows.shape[0] < 1:
        raise ValueError("raw_windows must have shape [N>=1,6,W].")
    if windows.shape[2] < 2 or not np.all(np.isfinite(windows)):
        raise ValueError("raw_windows must be finite with W>=2.")
    starts = _exact_int_array(
        "window_start_samples", window_start_samples, ndim=1
    )
    if starts.shape != (len(windows),):
        raise ValueError("Window start count differs from raw_windows.")
    if window_indices is None:
        indices = np.arange(len(windows), dtype=np.int64)
    else:
        indices = _exact_int_array("window_indices", window_indices, ndim=1)
        if indices.shape != starts.shape:
            raise ValueError("window_indices must match window starts.")
    order = np.argsort(starts, kind="stable")
    starts = starts[order]
    windows = windows[order]
    indices = indices[order]
    if starts[0] != 0 or (len(starts) > 1 and np.any(np.diff(starts) <= 0)):
        raise ValueError("Sorted window starts must begin at zero and increase.")
    if not np.array_equal(indices, np.arange(len(indices), dtype=np.int64)):
        raise ValueError("window_indices must be complete and ordered after start sort.")
    tolerance = _require_finite(
        "overlap_relative_tolerance", overlap_relative_tolerance
    )
    if tolerance < 0:
        raise ValueError("overlap_relative_tolerance must be non-negative.")
    window_size = int(windows.shape[2])
    length = int(starts[-1]) + window_size
    accumulator = np.zeros((6, length), dtype=np.float64)
    counts = np.zeros(length, dtype=np.int64)
    maximum_overlap_error = 0.0
    for window, start in zip(windows, starts):
        start = int(start)
        end = start + window_size
        overlap = counts[start:end] > 0
        if np.any(overlap):
            existing = (
                accumulator[:, start:end][:, overlap]
                / counts[start:end][overlap][None, :]
            )
            maximum_overlap_error = max(
                maximum_overlap_error,
                float(np.max(np.abs(existing - window[:, overlap]))),
            )
        accumulator[:, start:end] += window
        counts[start:end] += 1
    if np.any(counts == 0):
        raise ValueError("Sliding windows leave a gap inside the observed trial span.")
    scale = max(1.0, float(np.max(np.abs(windows))))
    if maximum_overlap_error > tolerance * scale:
        raise ValueError(
            "Overlapping windows disagree: "
            f"max_abs_error={maximum_overlap_error:.8g}, scale={scale:.8g}."
        )
    signal = accumulator / counts[None, :]
    return signal, {
        "window_count": int(len(windows)),
        "window_size_samples": window_size,
        "observed_sample_count": int(length),
        "maximum_overlap_error": float(maximum_overlap_error),
        "overlap_relative_tolerance": float(tolerance),
        "complete_partition_scope": "observed_span_through_last_complete_window",
    }


def _mad_prominence_events(
    values: np.ndarray,
    config: PeakValleyConfig,
) -> list[tuple[int, str, float, float]]:
    """Return peak/valley events significant relative to one series' MAD."""

    row = np.asarray(values, dtype=np.float64).reshape(-1)
    median = float(np.median(row))
    mad = float(np.median(np.abs(row - median)))
    robust_scale = 1.4826 * mad
    relative_floor = float(config.mad_scale_floor) * max(
        1.0, float(np.max(np.abs(row)))
    )
    if robust_scale <= relative_floor:
        return []
    threshold = float(config.prominence_mad_multiplier) * robust_scale
    result = []
    for kind, signed in (("peak", row), ("valley", -row)):
        indices, properties = find_peaks(
            signed,
            prominence=threshold,
            distance=int(config.extrema_min_distance_samples),
        )
        prominences = np.asarray(properties.get("prominences", []), dtype=np.float64)
        result.extend(
            (
                int(sample_index),
                kind,
                float(prominence),
                float(prominence / robust_scale),
            )
            for sample_index, prominence in zip(indices, prominences)
        )
    return sorted(result, key=lambda item: (item[1], item[0]))


def _motion_envelopes(
    smoothed: np.ndarray, config: PeakValleyConfig
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Build six non-negative axis envelopes and one canonical scalar.

    Peak/valley semantics cannot be assigned unambiguously to a six-dimensional
    vector because an axis sign flip swaps them.  The canonical signal is
    therefore the RMS of per-axis, MAD-normalised absolute derivatives.  It is
    a motion-intensity envelope: peak means intensity rise->fall and valley
    means fall->rise.  Per-axis envelopes provide independent time-tolerance
    votes; they do not decide the event kind.
    """

    derivative = np.gradient(smoothed, axis=1)
    envelopes = np.zeros_like(derivative, dtype=np.float64)
    active_axes = []
    scales = []
    for axis, row in enumerate(derivative):
        median = float(np.median(row))
        scale = 1.4826 * float(np.median(np.abs(row - median)))
        floor = float(config.mad_scale_floor) * max(1.0, float(np.max(np.abs(row))))
        scales.append(scale)
        if scale > floor:
            envelopes[axis] = np.abs(row) / scale
            active_axes.append(axis)
    envelopes = smooth_six_axis(envelopes, config.smoothing_samples)
    canonical = np.sqrt(np.mean(np.square(envelopes), axis=0, dtype=np.float64))
    return envelopes, canonical, {
        "canonical_scalar": (
            "RMS across six per-axis abs(gradient(smoothed_signal))/"
            "1.4826*MAD(gradient) motion envelopes"
        ),
        "canonical_peak_semantics": "motion intensity rise then fall",
        "canonical_valley_semantics": "motion intensity fall then rise",
        "active_axis_indices": active_axes,
        "derivative_mad_scales": scales,
    }


def _axis_extrema(envelopes: np.ndarray, config: PeakValleyConfig) -> list[_AxisEvent]:
    events: list[_AxisEvent] = []
    for axis_index, row in enumerate(envelopes):
        for sample_index, kind, prominence, normalized in _mad_prominence_events(
            row, config
        ):
            events.append(
                _AxisEvent(
                    sample_index=sample_index,
                    kind=kind,
                    axis_index=int(axis_index),
                    prominence=prominence,
                    normalized_prominence=normalized,
                )
            )
    return sorted(
        events,
        key=lambda item: (item.kind, item.sample_index, item.axis_index),
    )


def _consensus_events(
    canonical_events: Sequence[_CanonicalEvent],
    axis_events: Sequence[_AxisEvent],
    config: PeakValleyConfig,
) -> tuple[list[TurningEvent], dict]:
    if int(config.min_axis_votes) == 0:
        accepted = [
            TurningEvent(
                sample_index=int(item.sample_index),
                kind=item.kind,
                canonical_prominence=float(item.prominence),
                canonical_normalized_prominence=float(item.normalized_prominence),
                axis_indices=(),
                axis_sample_indices=(),
                axis_prominences=(),
                axis_normalized_prominences=(),
                source="canonical_motion_envelope_no_axis_vote",
            )
            for item in sorted(canonical_events, key=lambda value: value.sample_index)
        ]
        return accepted, {
            "canonical_extremum_count": int(len(canonical_events)),
            "axis_extremum_count": int(len(axis_events)),
            # Retain this legacy key for result readers; in this mode it is the
            # accepted canonical count rather than a voted-event count.
            "multi_axis_consensus_count": int(len(accepted)),
            "canonical_candidates_below_min_axis_votes": 0,
            "consensus_mode": "canonical_motion_envelope_without_axis_vote",
            "axis_vote_bypassed": True,
        }
    tolerance = int(config.axis_vote_tolerance_samples)
    accepted: list[TurningEvent] = []
    rejected = 0
    # Strong candidates claim votes first.  The min-distance rule normally
    # prevents overlap; explicit vote consumption removes the remaining edge
    # ambiguity when two canonical events are exactly 2*tolerance apart.
    used_axis_events: set[tuple[str, int, int]] = set()
    ordered = sorted(
        canonical_events,
        key=lambda item: (-item.normalized_prominence, item.sample_index, item.kind),
    )
    for canonical in ordered:
        chosen: dict[int, _AxisEvent] = {}
        for event in axis_events:
            event_key = (event.kind, event.axis_index, event.sample_index)
            if (
                event.kind != canonical.kind
                or event_key in used_axis_events
                or abs(event.sample_index - canonical.sample_index) > tolerance
            ):
                continue
            previous = chosen.get(event.axis_index)
            rank = (
                event.normalized_prominence,
                -abs(event.sample_index - canonical.sample_index),
                -event.sample_index,
            )
            if previous is None or rank > (
                previous.normalized_prominence,
                -abs(previous.sample_index - canonical.sample_index),
                -previous.sample_index,
            ):
                chosen[event.axis_index] = event
        members = sorted(chosen.values(), key=lambda item: item.axis_index)
        if len(members) < int(config.min_axis_votes):
            rejected += 1
            continue
        used_axis_events.update(
            (item.kind, item.axis_index, item.sample_index) for item in members
        )
        accepted.append(
            TurningEvent(
                sample_index=int(canonical.sample_index),
                kind=canonical.kind,
                canonical_prominence=float(canonical.prominence),
                canonical_normalized_prominence=float(canonical.normalized_prominence),
                axis_indices=tuple(item.axis_index for item in members),
                axis_sample_indices=tuple(item.sample_index for item in members),
                axis_prominences=tuple(item.prominence for item in members),
                axis_normalized_prominences=tuple(
                    item.normalized_prominence for item in members
                ),
            )
        )
    result = sorted(accepted, key=lambda item: item.sample_index)
    return result, {
        "canonical_extremum_count": int(len(canonical_events)),
        "axis_extremum_count": int(len(axis_events)),
        "multi_axis_consensus_count": int(len(result)),
        "canonical_candidates_below_min_axis_votes": int(rejected),
        "consensus_mode": "canonical_candidates_with_per_axis_support_vote",
        "axis_vote_bypassed": False,
    }


def _feature_change_score(trial: TrialSignal, sample_index: int, radius: int) -> float | None:
    if trial.feature_values is None or trial.feature_sample_positions is None:
        return None
    positions = trial.feature_sample_positions
    left = (positions < int(sample_index)) & (positions >= int(sample_index) - int(radius))
    right = (positions >= int(sample_index)) & (
        positions <= int(sample_index) + int(radius)
    )
    if not np.any(left) or not np.any(right):
        return None
    left_mean = trial.feature_values[left].mean(axis=0)
    right_mean = trial.feature_values[right].mean(axis=0)
    denominator = float(np.linalg.norm(left_mean) * np.linalg.norm(right_mean))
    if denominator <= EPS:
        return 0.0
    cosine = float(np.dot(left_mean, right_mean) / denominator)
    return float(1.0 - np.clip(cosine, -1.0, 1.0))


def _trial_equal_quantile(values_by_trial: Sequence[np.ndarray], quantile: float) -> float:
    values = []
    weights = []
    for row in values_by_trial:
        array = np.asarray(row, dtype=np.float64).reshape(-1)
        if len(array):
            if not np.all(np.isfinite(array)):
                raise ValueError("Quantile calibration contains non-finite values.")
            values.append(array)
            weights.append(np.full(len(array), 1.0 / len(array), dtype=np.float64))
    if not values:
        raise RuntimeError("No fit trial produced a valid feature-confirmation score.")
    pooled = np.concatenate(values)
    mass = np.concatenate(weights)
    order = np.argsort(pooled, kind="stable")
    pooled = pooled[order]
    cumulative = np.cumsum(mass[order])
    target = float(quantile) * float(cumulative[-1])
    index = int(np.searchsorted(cumulative, target, side="left"))
    return float(pooled[min(index, len(pooled) - 1)])


def _raw_consensus_for_trial(
    trial: TrialSignal, config: PeakValleyConfig
) -> tuple[np.ndarray, list[TurningEvent], dict]:
    smoothed = smooth_six_axis(trial.signal, config.smoothing_samples)
    envelopes, canonical, envelope_diagnostics = _motion_envelopes(smoothed, config)
    canonical_events = [
        _CanonicalEvent(sample, kind, prominence, normalized)
        for sample, kind, prominence, normalized in _mad_prominence_events(
            canonical, config
        )
    ]
    axis_events = (
        [] if int(config.min_axis_votes) == 0 else _axis_extrema(envelopes, config)
    )
    events, diagnostics = _consensus_events(canonical_events, axis_events, config)
    diagnostics.update(envelope_diagnostics)
    diagnostics["axis_extrema_by_channel"] = {
        config.channel_names[axis]: {
            kind: int(
                sum(item.axis_index == axis and item.kind == kind for item in axis_events)
            )
            for kind in EVENT_KINDS
        }
        for axis in range(6)
    }
    return smoothed, events, diagnostics


def fit_segmenter(
    fit_trials: Sequence[TrialSignal], config: PeakValleyConfig = PeakValleyConfig()
) -> PeakValleyState:
    """Fit only optional feature-score calibration on caller-supplied fit trials."""

    config = config.validate()
    trials = _validate_unique_trials(fit_trials)
    feature_threshold = None
    feature_dim = None
    calibration: dict = {
        "fit_scope": "caller_supplied_fit_trials_only",
        "uses_activity_labels": False,
        "prominence_rule": (
            "per-trial canonical motion-envelope and per-axis envelope prominence "
            ">= multiplier*1.4826*MAD"
        ),
        "event_kind_source": (
            "canonical RMS motion-intensity envelope; peak=rise-fall, "
            "valley=fall-rise"
        ),
        "feature_confirmation_enabled": bool(config.feature_confirmation),
    }
    if config.feature_confirmation:
        if any(trial.feature_values is None for trial in trials):
            raise ValueError(
                "Every fit trial must provide features when feature confirmation is enabled."
            )
        dimensions = {int(trial.feature_values.shape[1]) for trial in trials}
        if len(dimensions) != 1:
            raise ValueError("All fit trials must use one feature dimension.")
        feature_dim = dimensions.pop()
        scores_by_trial = []
        candidate_counts = []
        for trial in trials:
            _, events, _ = _raw_consensus_for_trial(trial, config)
            scores = [
                score
                for event in events
                if (
                    score := _feature_change_score(
                        trial, event.sample_index, config.feature_context_samples
                    )
                )
                is not None
            ]
            scores_by_trial.append(np.asarray(scores, dtype=np.float64))
            candidate_counts.append(len(events))
        feature_threshold = _trial_equal_quantile(
            scores_by_trial, config.feature_score_quantile
        )
        calibration.update(
            {
                "feature_dim": int(feature_dim),
                "feature_score": "cosine_distance_between_local_left_right_feature_means",
                "feature_context_samples": int(config.feature_context_samples),
                "feature_score_quantile": float(config.feature_score_quantile),
                "feature_threshold": float(feature_threshold),
                "trial_weighting": "each_candidate_bearing_fit_trial_has_total_mass_one",
                "fit_candidate_counts": candidate_counts,
                "fit_valid_score_counts": [int(len(row)) for row in scores_by_trial],
            }
        )
    else:
        calibration["feature_threshold"] = None
    return PeakValleyState(
        config=config,
        feature_threshold=feature_threshold,
        feature_dim=feature_dim,
        fit_data_sha256=_fit_trial_hash(trials),
        fit_trial_count=len(trials),
        feature_calibration=calibration,
    ).validate()


def _event_quality(event: TurningEvent) -> tuple[float, float, float, float, int]:
    return (
        float(event.axis_votes),
        float(event.canonical_normalized_prominence),
        float(event.mean_normalized_prominence),
        float(event.feature_change_score or 0.0),
        -int(event.sample_index),
    )


def _merge_short_segments(
    events: Sequence[TurningEvent], sample_count: int, minimum: int
) -> tuple[list[TurningEvent], list[dict]]:
    retained = sorted(events, key=lambda item: item.sample_index)
    removed: list[dict] = []
    if int(sample_count) < int(minimum):
        # A short whole trial is still a valid one-segment complete partition.
        return [], [
            {
                "reason": "whole_trial_shorter_than_minimum_single_segment_retained",
                "sample_count": int(sample_count),
                "minimum": int(minimum),
            }
        ]
    while retained:
        boundaries = np.asarray(
            [0] + [item.sample_index for item in retained] + [int(sample_count)],
            dtype=np.int64,
        )
        lengths = np.diff(boundaries)
        short = np.flatnonzero(lengths < int(minimum))
        if not len(short):
            break
        # Resolve the shortest offending segment first, then the earliest.
        segment_index = min(short.tolist(), key=lambda item: (lengths[item], item))
        if segment_index == 0:
            remove_index = 0
        elif segment_index == len(lengths) - 1:
            remove_index = len(retained) - 1
        else:
            left_event_index = segment_index - 1
            right_event_index = segment_index
            left = retained[left_event_index]
            right = retained[right_event_index]
            # Remove the weaker boundary.  Exact ties remove the later boundary
            # so the rule is stable and independent of object identity.
            remove_index = (
                left_event_index
                if _event_quality(left) < _event_quality(right)
                else right_event_index
            )
        event = retained.pop(remove_index)
        removed.append(
            {
                "reason": "minimum_segment_merge",
                "sample_index": int(event.sample_index),
                "kind": event.kind,
                "axis_votes": int(event.axis_votes),
            }
        )
    return retained, removed


def _assemble_segmented_trial(
    trial: TrialSignal,
    state: PeakValleyState,
    events: Sequence[TurningEvent],
    *,
    source: str,
    diagnostics: Mapping,
) -> SegmentedTrial:
    boundaries = np.asarray(
        [0] + [int(event.sample_index) for event in events] + [trial.signal.shape[1]],
        dtype=np.int64,
    )
    segments = []
    for index, (start, end) in enumerate(zip(boundaries[:-1], boundaries[1:])):
        segments.append(
            SampleSegment(
                segment_index=int(index),
                start_sample=int(start),
                end_sample_exclusive=int(end),
                left_event_kind=None if index == 0 else events[index - 1].kind,
                right_event_kind=(
                    None if index == len(boundaries) - 2 else events[index].kind
                ),
            )
        )
    return SegmentedTrial(
        trial_id=trial.trial_id,
        subject_id=trial.subject_id,
        sample_count=int(trial.signal.shape[1]),
        boundaries=boundaries,
        events=tuple(events),
        segments=tuple(segments),
        state_sha256=state.state_hash(),
        source=source,
        diagnostics=dict(diagnostics),
    ).validate(state.config.min_segment_samples)


def segment_trial(trial: TrialSignal, state: PeakValleyState) -> SegmentedTrial:
    """Transform one trial with a frozen peak/valley calibration state."""

    state = state.validate()
    trial = trial.validated(state.feature_dim if state.config.feature_confirmation else None)
    if state.config.feature_confirmation and trial.feature_values is None:
        raise ValueError("Feature-confirmed segmentation requires trial features.")
    _, candidates, diagnostics = _raw_consensus_for_trial(trial, state.config)
    confirmed = []
    unavailable = 0
    rejected = 0
    for event in candidates:
        score = _feature_change_score(
            trial, event.sample_index, state.config.feature_context_samples
        )
        updated = replace(event, feature_change_score=score)
        if state.config.feature_confirmation:
            if score is None:
                unavailable += 1
                continue
            if float(score) < float(state.feature_threshold):
                rejected += 1
                continue
        confirmed.append(updated)
    retained, merge_log = _merge_short_segments(
        confirmed,
        trial.signal.shape[1],
        state.config.min_segment_samples,
    )
    diagnostics.update(
        {
            "feature_confirmation_enabled": bool(state.config.feature_confirmation),
            "feature_threshold": state.feature_threshold,
            "feature_score_unavailable_count": int(unavailable),
            "feature_score_rejected_count": int(rejected),
            "pre_minimum_merge_event_count": int(len(confirmed)),
            "minimum_merge_removed_count": int(
                sum(item["reason"] == "minimum_segment_merge" for item in merge_log)
            ),
            "minimum_merge_log": merge_log,
            "retained_event_count": int(len(retained)),
            "segment_count": int(len(retained) + 1),
            "complete_partition": True,
            "boundary_resolution": "one raw sample",
        }
    )
    return _assemble_segmented_trial(
        trial, state, retained, source="peak_valley_detected", diagnostics=diagnostics
    )


def segment_trials(
    trials: Sequence[TrialSignal], state: PeakValleyState
) -> list[SegmentedTrial]:
    validated = _validate_unique_trials(trials)
    return [segment_trial(trial, state) for trial in validated]


def segmentation_from_boundaries(
    trial: TrialSignal,
    state: PeakValleyState,
    boundaries: Sequence[int],
    event_kinds: Sequence[str] | None = None,
    *,
    source: str = "explicit_boundaries",
) -> SegmentedTrial:
    """Build a strictly validated complete partition for matched controls."""

    state = state.validate()
    trial = trial.validated()
    values = _exact_int_array("explicit boundaries", boundaries, ndim=1)
    if len(values) < 2:
        raise ValueError("Explicit boundaries must be a 1D vector of length >=2.")
    if values[0] != 0 or values[-1] != trial.signal.shape[1]:
        raise ValueError("Explicit boundaries must include exact endpoints 0 and T.")
    if np.any(np.diff(values) <= 0):
        raise ValueError("Explicit boundaries must be strictly increasing.")
    if len(values) > 2 and np.any(np.diff(values) < state.config.min_segment_samples):
        raise ValueError("Explicit partition violates min_segment_samples.")
    internal = values[1:-1]
    if event_kinds is None:
        kinds = [EVENT_KINDS[index % 2] for index in range(len(internal))]
    else:
        kinds = [str(item) for item in event_kinds]
    if len(kinds) != len(internal) or any(item not in EVENT_KINDS for item in kinds):
        raise ValueError("event_kinds must contain one peak/valley per internal boundary.")
    events = [
        TurningEvent(
            sample_index=int(sample),
            kind=kinds[index],
            canonical_prominence=0.0,
            canonical_normalized_prominence=0.0,
            axis_indices=(),
            axis_sample_indices=(),
            axis_prominences=(),
            axis_normalized_prominences=(),
            source=source,
        )
        for index, sample in enumerate(internal)
    ]
    return _assemble_segmented_trial(
        trial,
        state,
        events,
        source=source,
        diagnostics={
            "complete_partition": True,
            "boundary_resolution": "one raw sample",
            "explicit_control": True,
        },
    )


def _uniform_weak_composition(
    total: int, parts: int, rng: np.random.Generator
) -> np.ndarray:
    """Sample uniformly from all ordered non-negative integer compositions."""

    if (
        isinstance(total, bool)
        or int(total) != total
        or int(total) < 0
        or isinstance(parts, bool)
        or int(parts) != parts
        or int(parts) < 1
    ):
        raise ValueError("Composition total must be >=0 and parts must be >=1.")
    total = int(total)
    parts = int(parts)
    if parts == 1:
        return np.asarray([total], dtype=np.int64)
    # Stars and bars: choosing the (parts-1) bar locations uniformly from the
    # total+parts-1 slots is a bijection with ordered weak compositions, so no
    # composition receives the multinomial coefficient bias of rng.multinomial.
    slot_count = total + parts - 1
    bars = np.sort(
        rng.choice(slot_count, size=parts - 1, replace=False).astype(np.int64)
    )
    augmented = np.r_[np.int64(-1), bars, np.int64(slot_count)]
    composition = np.diff(augmented) - 1
    if (
        composition.shape != (parts,)
        or np.any(composition < 0)
        or int(composition.sum()) != total
    ):
        raise RuntimeError("Uniform stars-and-bars composition invariant failed.")
    return composition.astype(np.int64, copy=False)


def matched_random_segmentations(
    trials: Sequence[TrialSignal],
    reference_segmentations: Sequence[SegmentedTrial],
    state: PeakValleyState,
    seed: int,
) -> list[SegmentedTrial]:
    """Random partitions with each trial's reference segment count/minimum length."""

    state = state.validate()
    validated = _validate_unique_trials(trials)
    references = {
        _identifier_sort_key(item.trial_id): item.validate(state.config.min_segment_samples)
        for item in reference_segmentations
    }
    if len(references) != len(reference_segmentations):
        raise ValueError("Reference segmentations contain duplicate trial IDs.")
    if set(references) != {
        _identifier_sort_key(item.trial_id) for item in validated
    }:
        raise ValueError("Trials and reference segmentations do not have identical IDs.")
    output = []
    for trial in validated:
        reference = references[_identifier_sort_key(trial.trial_id)]
        if _identifier_sort_key(reference.subject_id) != _identifier_sort_key(
            trial.subject_id
        ):
            raise ValueError("Reference segmentation subject differs from trial subject.")
        if reference.state_sha256 != state.state_hash():
            raise ValueError("Reference segmentation came from a different state.")
        if reference.sample_count != trial.signal.shape[1]:
            raise ValueError("Reference sample count differs from trial signal length.")
        segment_count = len(reference.segments)
        minimum = int(state.config.min_segment_samples)
        required = segment_count * minimum
        if required > trial.signal.shape[1]:
            raise ValueError("Reference count cannot satisfy the configured minimum length.")
        local_seed = int.from_bytes(
            hashlib.sha256(
                f"{int(seed)}|{_identifier_sort_key(trial.trial_id)}".encode("utf-8")
            ).digest()[:8],
            "little",
        )
        rng = np.random.default_rng(local_seed)
        residual = int(trial.signal.shape[1] - required)
        extras = _uniform_weak_composition(
            residual,
            segment_count,
            rng,
        )
        lengths = extras + minimum
        boundaries = np.r_[0, np.cumsum(lengths)].astype(np.int64)
        kinds = [event.kind for event in reference.events]
        if kinds:
            kinds = [kinds[index] for index in rng.permutation(len(kinds))]
        randomized = segmentation_from_boundaries(
            trial,
            state,
            boundaries,
            kinds,
            source="matched_random_partition",
        )
        event_histogram = {
            kind: int(sum(event.kind == kind for event in reference.events))
            for kind in EVENT_KINDS
        }
        randomized = replace(
            randomized,
            diagnostics={
                **randomized.diagnostics,
                "control_matching": {
                    "reference_segment_count": int(segment_count),
                    "matched_segment_count": True,
                    "minimum_segment_samples": int(minimum),
                    "matched_minimum_segment_length_constraint": True,
                    "reference_event_kind_histogram": event_histogram,
                    "matched_event_kind_histogram": True,
                    "matched_reference_segment_length_distribution": False,
                    "residual_sample_count": int(residual),
                    "length_sampling": (
                        "uniform_over_ordered_weak_integer_compositions_of_"
                        "residual_samples"
                    ),
                    "joint_legal_partition_sampling_is_uniform": True,
                    "individual_boundary_marginals_are_not_claimed_uniform": True,
                },
            },
        ).validate(state.config.min_segment_samples)
        output.append(randomized)
    return output


@dataclass(frozen=True)
class ChildPrimitiveBatch:
    trial_id: Identifier
    subject_id: Identifier
    segment_indices: np.ndarray
    start_samples: np.ndarray
    end_samples_exclusive: np.ndarray
    shape_features: np.ndarray
    statistic_features: np.ndarray
    statistic_names: tuple[str, ...]
    segmentation_state_sha256: str

    def validate(self) -> "ChildPrimitiveBatch":
        _validate_identifier("trial_id", self.trial_id)
        _validate_identifier("subject_id", self.subject_id)
        indices = _exact_int_array("segment_indices", self.segment_indices, ndim=1)
        count = len(indices)
        if count < 1:
            raise ValueError("A child batch must contain at least one segment.")
        if not np.array_equal(indices, np.arange(count)):
            raise ValueError("segment_indices must be contiguous from zero.")
        starts = _exact_int_array("start_samples", self.start_samples, ndim=1)
        ends = _exact_int_array(
            "end_samples_exclusive", self.end_samples_exclusive, ndim=1
        )
        if starts.shape != (count,) or ends.shape != (count,):
            raise ValueError("Child segment spans must match segment count.")
        if np.any(starts < 0) or np.any(ends <= starts):
            raise ValueError("Child segment spans must have positive length.")
        if count > 1 and (
            np.any(np.diff(starts) <= 0) or np.any(np.diff(ends) <= 0)
        ):
            raise ValueError("Child segment spans must be strictly ordered.")
        if self.shape_features.ndim != 2 or len(self.shape_features) != count:
            raise ValueError("shape_features must be [segments,D].")
        if self.statistic_features.shape != (count, len(self.statistic_names)):
            raise ValueError("statistic feature shape/name mismatch.")
        if not np.all(np.isfinite(self.shape_features)) or not np.all(
            np.isfinite(self.statistic_features)
        ):
            raise ValueError("Child features contain non-finite values.")
        _validate_sha256(
            "segmentation_state_sha256", self.segmentation_state_sha256
        )
        return self

    def to_dict(self) -> dict:
        return {
            "trial_id": _identifier_payload(self.trial_id),
            "subject_id": _identifier_payload(self.subject_id),
            "segment_indices": self.segment_indices.astype(int).tolist(),
            "start_samples": self.start_samples.astype(int).tolist(),
            "end_samples_exclusive": self.end_samples_exclusive.astype(int).tolist(),
            "shape_features": self.shape_features.tolist(),
            "statistic_features": self.statistic_features.tolist(),
            "statistic_names": list(self.statistic_names),
            "segmentation_state_sha256": self.segmentation_state_sha256,
        }


def _resample_rows(values: np.ndarray, points: int) -> np.ndarray:
    if values.shape[1] == 1:
        return np.repeat(values, int(points), axis=1)
    old = np.linspace(0.0, 1.0, values.shape[1], dtype=np.float64)
    new = np.linspace(0.0, 1.0, int(points), dtype=np.float64)
    return np.stack([np.interp(new, old, row) for row in values], axis=0)


def _statistic_names(config: PeakValleyConfig) -> tuple[str, ...]:
    result = ["duration_seconds"]
    for prefix in ("mean", "std", "log_mean_square_energy", "end_minus_start"):
        result.extend(f"{prefix}__{name}" for name in config.channel_names)
    return tuple(result)


def extract_child_features(
    trial: TrialSignal, segmented: SegmentedTrial, state: PeakValleyState
) -> ChildPrimitiveBatch:
    """Extract shape-only codebook inputs and a separate physical-statistics path."""

    state = state.validate()
    trial = trial.validated()
    segmented = segmented.validate(state.config.min_segment_samples)
    if _identifier_sort_key(trial.trial_id) != _identifier_sort_key(segmented.trial_id):
        raise ValueError("Trial and segmentation IDs differ.")
    if _identifier_sort_key(trial.subject_id) != _identifier_sort_key(
        segmented.subject_id
    ):
        raise ValueError("Trial and segmentation subject IDs differ.")
    if segmented.sample_count != trial.signal.shape[1]:
        raise ValueError("Trial signal length differs from segmentation.")
    if segmented.state_sha256 != state.state_hash():
        raise ValueError("Segmentation was produced by a different state.")
    smoothed = smooth_six_axis(trial.signal, state.config.smoothing_samples)
    shape_rows = []
    statistic_rows = []
    for segment in segmented.segments:
        start = segment.start_sample
        end = segment.end_sample_exclusive
        shape = _resample_rows(smoothed[:, start:end], state.config.shape_points)
        centre = shape.mean(axis=1, keepdims=True)
        scale = shape.std(axis=1, keepdims=True)
        scale = np.maximum(scale, float(state.config.shape_scale_floor))
        shape = ((shape - centre) / scale).reshape(-1)
        raw = trial.signal[:, start:end]
        means = raw.mean(axis=1)
        stds = raw.std(axis=1)
        energies = np.log(
            np.mean(np.square(raw), axis=1) + float(state.config.statistics_epsilon)
        )
        deltas = raw[:, -1] - raw[:, 0]
        statistic_rows.append(
            np.r_[
                (end - start) / float(state.config.sample_rate_hz),
                means,
                stds,
                energies,
                deltas,
            ]
        )
        shape_rows.append(shape)
    batch = ChildPrimitiveBatch(
        trial_id=trial.trial_id,
        subject_id=trial.subject_id,
        segment_indices=np.arange(len(segmented.segments), dtype=np.int64),
        start_samples=segmented.boundaries[:-1].astype(np.int64, copy=True),
        end_samples_exclusive=segmented.boundaries[1:].astype(np.int64, copy=True),
        shape_features=np.asarray(shape_rows, dtype=np.float64),
        statistic_features=np.asarray(statistic_rows, dtype=np.float64),
        statistic_names=_statistic_names(state.config),
        segmentation_state_sha256=state.state_hash(),
    )
    return batch.validate()


def extract_child_feature_batches(
    trials: Sequence[TrialSignal],
    segmentations: Sequence[SegmentedTrial],
    state: PeakValleyState,
) -> list[ChildPrimitiveBatch]:
    trial_map = {
        _identifier_sort_key(item.trial_id): item for item in _validate_unique_trials(trials)
    }
    segmentation_map = {
        _identifier_sort_key(item.trial_id): item for item in segmentations
    }
    if len(segmentation_map) != len(segmentations) or set(trial_map) != set(segmentation_map):
        raise ValueError("Trials and segmentations must have identical unique trial IDs.")
    return [
        extract_child_features(trial_map[key], segmentation_map[key], state)
        for key in sorted(trial_map)
    ]


@dataclass(frozen=True)
class CodebookConfig:
    primitive_num: int = 32
    pca_dim: int = 64
    weighting: Weighting = "subject_trial_equal"
    l2_normalize: bool = True
    kmeans_n_init: int = 20
    kmeans_max_iter: int = 300
    random_seed: int = 0

    def validate(self) -> "CodebookConfig":
        if (
            isinstance(self.primitive_num, bool)
            or int(self.primitive_num) != self.primitive_num
            or int(self.primitive_num) < 2
        ):
            raise ValueError("primitive_num must be an integer >=2.")
        if (
            isinstance(self.pca_dim, bool)
            or int(self.pca_dim) != self.pca_dim
            or int(self.pca_dim) < 0
        ):
            raise ValueError("pca_dim must be a non-negative integer; zero disables PCA.")
        if self.weighting not in {
            "uniform",
            "trial_equal",
            "subject_equal",
            "subject_trial_equal",
        }:
            raise ValueError(f"Unknown weighting mode {self.weighting!r}.")
        if (
            isinstance(self.kmeans_n_init, bool)
            or int(self.kmeans_n_init) != self.kmeans_n_init
            or isinstance(self.kmeans_max_iter, bool)
            or int(self.kmeans_max_iter) != self.kmeans_max_iter
            or int(self.kmeans_n_init) < 1
            or int(self.kmeans_max_iter) < 1
        ):
            raise ValueError("KMeans n_init/max_iter must be positive.")
        return replace(
            self,
            primitive_num=int(self.primitive_num),
            pca_dim=int(self.pca_dim),
            l2_normalize=bool(self.l2_normalize),
            kmeans_n_init=int(self.kmeans_n_init),
            kmeans_max_iter=int(self.kmeans_max_iter),
            random_seed=int(self.random_seed),
        )

    def to_dict(self) -> dict:
        return _jsonable(asdict(self))

    @classmethod
    def from_dict(cls, payload: Mapping) -> "CodebookConfig":
        return cls(**dict(payload)).validate()


@dataclass(frozen=True)
class CodebookState:
    config: CodebookConfig
    input_dim: int
    output_dim: int
    pca_mean: np.ndarray | None
    pca_components: np.ndarray | None
    cluster_centers: np.ndarray
    segmentation_state_sha256: str
    fit_data_sha256: str
    fit_segment_count: int
    fit_trial_count: int
    fit_subject_count: int
    weight_summary: dict
    kmeans_inertia: float
    kmeans_n_iter: int
    numpy_version: str
    sklearn_version: str
    algorithm_revision: str = ALGORITHM_REVISION
    schema_version: int = SCHEMA_VERSION

    def validate(self) -> "CodebookState":
        config = self.config.validate()
        if self.schema_version != SCHEMA_VERSION or self.algorithm_revision != ALGORITHM_REVISION:
            raise ValueError("Codebook schema/revision mismatch.")
        if int(self.input_dim) < 1 or int(self.output_dim) < 1:
            raise ValueError("Codebook dimensions must be positive.")
        if config.pca_dim > 0:
            if self.pca_mean is None or self.pca_components is None:
                raise ValueError("PCA-enabled codebook lacks PCA state.")
            if np.asarray(self.pca_mean).shape != (int(self.input_dim),):
                raise ValueError("PCA mean shape mismatch.")
            if np.asarray(self.pca_components).shape != (
                int(self.output_dim),
                int(self.input_dim),
            ):
                raise ValueError("PCA component shape mismatch.")
        elif self.pca_mean is not None or self.pca_components is not None:
            raise ValueError("PCA-disabled codebook must not contain PCA arrays.")
        centers = np.asarray(self.cluster_centers)
        if centers.shape != (config.primitive_num, int(self.output_dim)):
            raise ValueError("Cluster center shape mismatch.")
        arrays = [centers]
        if self.pca_mean is not None:
            arrays.extend([self.pca_mean, self.pca_components])
        if any(not np.all(np.isfinite(item)) for item in arrays):
            raise ValueError("Codebook state contains non-finite arrays.")
        if int(self.fit_segment_count) < config.primitive_num:
            raise ValueError("Codebook fit segment count is smaller than K.")
        if int(self.fit_trial_count) < 1 or int(self.fit_subject_count) < 1:
            raise ValueError("Codebook fit trial/subject counts must be positive.")
        _validate_sha256(
            "segmentation_state_sha256", self.segmentation_state_sha256
        )
        _validate_sha256("fit_data_sha256", self.fit_data_sha256)
        _require_finite("kmeans_inertia", self.kmeans_inertia)
        return self

    def _payload(self) -> dict:
        return {
            "schema_version": int(self.schema_version),
            "algorithm_revision": self.algorithm_revision,
            "config": self.config.to_dict(),
            "input_dim": int(self.input_dim),
            "output_dim": int(self.output_dim),
            "pca_mean": None if self.pca_mean is None else _array_payload(self.pca_mean),
            "pca_components": (
                None if self.pca_components is None else _array_payload(self.pca_components)
            ),
            "cluster_centers": _array_payload(self.cluster_centers),
            "segmentation_state_sha256": self.segmentation_state_sha256,
            "fit_data_sha256": self.fit_data_sha256,
            "fit_segment_count": int(self.fit_segment_count),
            "fit_trial_count": int(self.fit_trial_count),
            "fit_subject_count": int(self.fit_subject_count),
            "weight_summary": _jsonable(self.weight_summary),
            "kmeans_inertia": float(self.kmeans_inertia),
            "kmeans_n_iter": int(self.kmeans_n_iter),
            "numpy_version": self.numpy_version,
            "sklearn_version": self.sklearn_version,
        }

    def state_hash(self) -> str:
        self.validate()
        return canonical_sha256(self._payload())

    def to_dict(self) -> dict:
        result = self._payload()
        result["state_sha256"] = self.state_hash()
        return result

    @classmethod
    def from_dict(cls, payload: Mapping) -> "CodebookState":
        values = dict(payload)
        expected_hash = values.pop("state_sha256", None)
        if expected_hash is None:
            raise ValueError("Serialized codebook state lacks state_sha256.")
        config = CodebookConfig.from_dict(values.pop("config"))
        pca_mean_payload = values.pop("pca_mean")
        pca_components_payload = values.pop("pca_components")
        centers_payload = values.pop("cluster_centers")
        input_dim = int(values["input_dim"])
        output_dim = int(values["output_dim"])
        state = cls(
            config=config,
            pca_mean=(
                None
                if pca_mean_payload is None
                else _array_from_payload(pca_mean_payload, name="pca_mean", dtype=np.float64)
            ),
            pca_components=(
                None
                if pca_components_payload is None
                else _array_from_payload(
                    pca_components_payload, name="pca_components", dtype=np.float64
                )
            ),
            cluster_centers=_array_from_payload(
                centers_payload, name="cluster_centers", dtype=np.float64
            ),
            **values,
        ).validate()
        if state.input_dim != input_dim or state.output_dim != output_dim:
            raise ValueError("Serialized codebook dimensions changed during load.")
        _validate_sha256("state_sha256", expected_hash)
        if expected_hash != state.state_hash():
            raise ValueError("Codebook state hash mismatch.")
        return state


def _ordered_batches(batches: Sequence[ChildPrimitiveBatch]) -> list[ChildPrimitiveBatch]:
    validated = [item.validate() for item in batches]
    if not validated:
        raise ValueError("At least one child-feature batch is required.")
    keys = [_identifier_sort_key(item.trial_id) for item in validated]
    if len(keys) != len(set(keys)):
        raise ValueError("Child-feature batches contain duplicate trial IDs.")
    dimensions = {item.shape_features.shape[1] for item in validated}
    statistics = {(item.statistic_features.shape[1], item.statistic_names) for item in validated}
    state_hashes = {item.segmentation_state_sha256 for item in validated}
    if len(dimensions) != 1 or len(statistics) != 1 or len(state_hashes) != 1:
        raise ValueError("Child-feature batches have incompatible schemas/states.")
    return sorted(validated, key=lambda item: _identifier_sort_key(item.trial_id))


def _codebook_fit_hash(batches: Sequence[ChildPrimitiveBatch]) -> str:
    digest = hashlib.sha256()
    digest.update(ALGORITHM_REVISION.encode("ascii"))
    for batch in batches:
        digest.update(_identifier_sort_key(batch.trial_id).encode("utf-8"))
        digest.update(_identifier_sort_key(batch.subject_id).encode("utf-8"))
        digest.update(batch.segmentation_state_sha256.encode("ascii"))
        _hash_array(digest, "segment_indices", batch.segment_indices)
        _hash_array(digest, "start_samples", batch.start_samples)
        _hash_array(digest, "end_samples_exclusive", batch.end_samples_exclusive)
        _hash_array(digest, "shape_features", batch.shape_features)
    return digest.hexdigest()


def _segment_weights(
    batches: Sequence[ChildPrimitiveBatch], weighting: Weighting
) -> np.ndarray:
    counts = np.asarray([len(item.segment_indices) for item in batches], dtype=np.int64)
    subjects = [_identifier_sort_key(item.subject_id) for item in batches]
    if weighting == "uniform":
        trial_mass = counts.astype(np.float64)
        rows = [np.ones(count, dtype=np.float64) for count in counts]
    elif weighting == "trial_equal":
        trial_mass = np.ones(len(batches), dtype=np.float64)
        rows = [np.full(count, 1.0 / count, dtype=np.float64) for count in counts]
    elif weighting == "subject_equal":
        subject_segments = {
            subject: int(sum(count for count, item in zip(counts, subjects) if item == subject))
            for subject in set(subjects)
        }
        rows = [
            np.full(count, 1.0 / subject_segments[subject], dtype=np.float64)
            for count, subject in zip(counts, subjects)
        ]
        trial_mass = np.asarray([row.sum() for row in rows], dtype=np.float64)
    elif weighting == "subject_trial_equal":
        subject_trial_counts = {
            subject: int(sum(item == subject for item in subjects)) for subject in set(subjects)
        }
        rows = [
            np.full(
                count,
                1.0 / (subject_trial_counts[subject] * count),
                dtype=np.float64,
            )
            for count, subject in zip(counts, subjects)
        ]
        trial_mass = np.asarray([row.sum() for row in rows], dtype=np.float64)
    else:
        raise ValueError(f"Unknown weighting {weighting!r}.")
    weights = np.concatenate(rows)
    weights *= len(weights) / float(weights.sum())
    if np.any(weights <= 0) or not np.all(np.isfinite(weights)):
        raise RuntimeError("Codebook weighting produced invalid weights.")
    return weights


def _weighted_pca(
    values: np.ndarray, weights: np.ndarray, output_dim: int
) -> tuple[np.ndarray, np.ndarray]:
    mean = np.average(values, axis=0, weights=weights)
    centred = values - mean
    covariance = (centred.T * weights) @ centred / float(weights.sum())
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues, kind="stable")[::-1][: int(output_dim)]
    components = eigenvectors[:, order].T
    # Remove the arbitrary eigenvector sign so serialized states are stable.
    for row in components:
        pivot = int(np.argmax(np.abs(row)))
        if row[pivot] < 0:
            row *= -1.0
    return mean.astype(np.float64), components.astype(np.float64)


def _transform_codebook_features(values: np.ndarray, state: CodebookState) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != state.input_dim:
        raise ValueError("Shape-feature dimension does not match codebook state.")
    if not np.all(np.isfinite(array)):
        raise ValueError("Shape features contain non-finite values.")
    if state.pca_components is not None:
        array = (array - state.pca_mean) @ state.pca_components.T
    else:
        array = array.copy()
    if state.config.l2_normalize:
        norms = np.linalg.norm(array, axis=1, keepdims=True)
        array = array / np.maximum(norms, EPS)
    return array


def fit_codebook(
    fit_batches: Sequence[ChildPrimitiveBatch],
    config: CodebookConfig = CodebookConfig(),
) -> CodebookState:
    """Fit weighted PCA and KMeans using only caller-supplied child batches."""

    config = config.validate()
    batches = _ordered_batches(fit_batches)
    values = np.concatenate([item.shape_features for item in batches], axis=0)
    if len(values) < config.primitive_num:
        raise ValueError(
            f"K={config.primitive_num} exceeds fit segment count {len(values)}."
        )
    weights = _segment_weights(batches, config.weighting)
    pca_mean = None
    pca_components = None
    if config.pca_dim > 0:
        output_dim = min(config.pca_dim, values.shape[1], len(values) - 1)
        if output_dim < 1:
            raise ValueError("PCA has no valid output dimension.")
        pca_mean, pca_components = _weighted_pca(values, weights, output_dim)
        transformed = (values - pca_mean) @ pca_components.T
    else:
        output_dim = int(values.shape[1])
        transformed = values.copy()
    if config.l2_normalize:
        transformed /= np.maximum(np.linalg.norm(transformed, axis=1, keepdims=True), EPS)
    model = KMeans(
        n_clusters=config.primitive_num,
        random_state=config.random_seed,
        n_init=config.kmeans_n_init,
        max_iter=config.kmeans_max_iter,
        algorithm="lloyd",
    )
    model.fit(transformed, sample_weight=weights)
    subjects = {_identifier_sort_key(item.subject_id) for item in batches}
    subject_mass = {}
    cursor = 0
    for batch in batches:
        count = len(batch.segment_indices)
        key = _identifier_sort_key(batch.subject_id)
        subject_mass[key] = subject_mass.get(key, 0.0) + float(
            weights[cursor : cursor + count].sum()
        )
        cursor += count
    state = CodebookState(
        config=config,
        input_dim=int(values.shape[1]),
        output_dim=int(output_dim),
        pca_mean=pca_mean,
        pca_components=pca_components,
        cluster_centers=np.asarray(model.cluster_centers_, dtype=np.float64),
        segmentation_state_sha256=batches[0].segmentation_state_sha256,
        fit_data_sha256=_codebook_fit_hash(batches),
        fit_segment_count=int(len(values)),
        fit_trial_count=int(len(batches)),
        fit_subject_count=int(len(subjects)),
        weight_summary={
            "mode": config.weighting,
            "normalization": "mean segment weight equals one",
            "minimum": float(weights.min()),
            "maximum": float(weights.max()),
            "subject_total_mass": subject_mass,
        },
        kmeans_inertia=float(model.inertia_),
        kmeans_n_iter=int(model.n_iter_),
        numpy_version=np.__version__,
        sklearn_version=sklearn.__version__,
    )
    return state.validate()


@dataclass(frozen=True)
class TokenizedTrial:
    segmentation: SegmentedTrial
    child_features: ChildPrimitiveBatch
    primitive_tokens: np.ndarray
    nearest_center_distances: np.ndarray
    codebook_embeddings: np.ndarray
    codebook_state_sha256: str
    primitive_num: int

    @property
    def trial_id(self) -> Identifier:
        return self.segmentation.trial_id

    @property
    def subject_id(self) -> Identifier:
        return self.segmentation.subject_id

    def validate(self) -> "TokenizedTrial":
        segmentation = self.segmentation.validate()
        child_features = self.child_features.validate()
        _validate_child_segmentation_alignment(child_features, segmentation)
        if (
            isinstance(self.primitive_num, bool)
            or int(self.primitive_num) != self.primitive_num
            or int(self.primitive_num) < 2
        ):
            raise ValueError("primitive_num must be an integer >=2.")
        _validate_sha256("codebook_state_sha256", self.codebook_state_sha256)
        count = len(segmentation.segments)
        tokens = _exact_int_array("primitive_tokens", self.primitive_tokens, ndim=1)
        if tokens.shape != (count,):
            raise ValueError("primitive token count differs from segment count.")
        if self.nearest_center_distances.shape != (count,):
            raise ValueError("distance count differs from segment count.")
        if self.codebook_embeddings.ndim != 2 or len(self.codebook_embeddings) != count:
            raise ValueError("codebook embedding count differs from segment count.")
        if np.any(tokens < 0) or np.any(tokens >= int(self.primitive_num)):
            raise ValueError("Primitive token lies outside the codebook.")
        if not np.all(np.isfinite(self.nearest_center_distances)) or not np.all(
            np.isfinite(self.codebook_embeddings)
        ):
            raise ValueError("Tokenized trial contains non-finite values.")
        if np.any(np.asarray(self.nearest_center_distances) < 0):
            raise ValueError("Nearest-centre distances must be non-negative.")
        return self

    def to_dict(self) -> dict:
        return {
            "trial_id": _identifier_payload(self.trial_id),
            "subject_id": _identifier_payload(self.subject_id),
            "segment_start_samples": self.child_features.start_samples.astype(int).tolist(),
            "segment_end_samples_exclusive": self.child_features.end_samples_exclusive.astype(
                int
            ).tolist(),
            "primitive_tokens": self.primitive_tokens.astype(int).tolist(),
            "nearest_center_distances": self.nearest_center_distances.tolist(),
            "statistics": self.child_features.statistic_features.tolist(),
            "statistic_names": list(self.child_features.statistic_names),
            "events": [item.to_dict() for item in self.segmentation.events],
            "codebook_state_sha256": self.codebook_state_sha256,
            "segmentation_state_sha256": self.segmentation.state_sha256,
        }


def _validate_child_segmentation_alignment(
    child_features: ChildPrimitiveBatch,
    segmentation: SegmentedTrial,
) -> None:
    """Fail closed when a feature batch is paired with another partition."""

    child = child_features.validate()
    segmented = segmentation.validate()
    if _identifier_sort_key(child.trial_id) != _identifier_sort_key(
        segmented.trial_id
    ):
        raise ValueError("Child features and segmentation trial IDs differ.")
    if _identifier_sort_key(child.subject_id) != _identifier_sort_key(
        segmented.subject_id
    ):
        raise ValueError("Child features and segmentation subject IDs differ.")
    if child.segmentation_state_sha256 != segmented.state_sha256:
        raise ValueError("Child features and segmentation state fingerprints differ.")
    expected_count = len(segmented.segments)
    if len(child.segment_indices) != expected_count:
        raise ValueError("Child feature count differs from segmentation segment count.")
    if not np.array_equal(
        np.asarray(child.start_samples, dtype=np.int64),
        np.asarray(segmented.boundaries[:-1], dtype=np.int64),
    ) or not np.array_equal(
        np.asarray(child.end_samples_exclusive, dtype=np.int64),
        np.asarray(segmented.boundaries[1:], dtype=np.int64),
    ):
        raise ValueError("Child feature spans differ from segmentation boundaries.")


def assign_codebook(
    child_features: ChildPrimitiveBatch,
    segmentation: SegmentedTrial,
    state: CodebookState,
) -> TokenizedTrial:
    state = state.validate()
    child_features = child_features.validate()
    segmentation = segmentation.validate()
    _validate_child_segmentation_alignment(child_features, segmentation)
    if child_features.segmentation_state_sha256 != state.segmentation_state_sha256:
        raise ValueError(
            "Child features were produced by a different segmentation state than "
            "the fitted codebook."
        )
    transformed = _transform_codebook_features(child_features.shape_features, state)
    squared = np.sum(
        np.square(transformed[:, None, :] - state.cluster_centers[None, :, :]), axis=2
    )
    tokens = np.argmin(squared, axis=1).astype(np.int64)
    distances = np.sqrt(squared[np.arange(len(tokens)), tokens])
    return TokenizedTrial(
        segmentation=segmentation,
        child_features=child_features,
        primitive_tokens=tokens,
        nearest_center_distances=distances.astype(np.float64),
        codebook_embeddings=transformed,
        codebook_state_sha256=state.state_hash(),
        primitive_num=state.config.primitive_num,
    ).validate()


def assign_codebook_trials(
    child_batches: Sequence[ChildPrimitiveBatch],
    segmentations: Sequence[SegmentedTrial],
    state: CodebookState,
) -> list[TokenizedTrial]:
    batches = {_identifier_sort_key(item.trial_id): item for item in child_batches}
    segments = {_identifier_sort_key(item.trial_id): item for item in segmentations}
    if len(batches) != len(child_batches) or len(segments) != len(segmentations):
        raise ValueError("Duplicate trial IDs in child batches or segmentations.")
    if set(batches) != set(segments):
        raise ValueError("Child batches and segmentations have different trial IDs.")
    return [assign_codebook(batches[key], segments[key], state) for key in sorted(batches)]


ParentKey = tuple[str, int, int]


@dataclass(frozen=True)
class ParentGateConfig:
    minimum_occurrences: int = 10
    minimum_trials: int = 6
    minimum_subjects: int = 2
    minimum_npmi: float = 0.0
    minimum_mdl_gain_bits: float = 0.0
    minimum_loso_stability: float = 0.80

    def validate(self) -> "ParentGateConfig":
        for name in (
            "minimum_occurrences",
            "minimum_trials",
            "minimum_subjects",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or int(value) != value or int(value) < 1:
                raise ValueError(f"{name} must be a positive integer.")
        if not (
            int(self.minimum_occurrences)
            >= int(self.minimum_trials)
            >= int(self.minimum_subjects)
            >= 2
        ):
            raise ValueError(
                "Parent support gates must satisfy minimum_occurrences >= "
                "minimum_trials >= minimum_subjects >= 2."
            )
        if not -1.0 <= _require_finite("minimum_npmi", self.minimum_npmi) <= 1.0:
            raise ValueError("minimum_npmi must lie in [-1,1].")
        if _require_finite(
            "minimum_mdl_gain_bits", self.minimum_mdl_gain_bits
        ) < 0:
            raise ValueError("minimum_mdl_gain_bits must be non-negative.")
        if not 0.0 <= _require_finite(
            "minimum_loso_stability", self.minimum_loso_stability
        ) <= 1.0:
            raise ValueError("minimum_loso_stability must lie in [0,1].")
        return replace(
            self,
            minimum_occurrences=int(self.minimum_occurrences),
            minimum_trials=int(self.minimum_trials),
            minimum_subjects=int(self.minimum_subjects),
        )

    def to_dict(self) -> dict:
        return _jsonable(asdict(self))

    @classmethod
    def from_dict(cls, payload: Mapping) -> "ParentGateConfig":
        return cls(**dict(payload)).validate()


@dataclass(frozen=True)
class ParentPatternStats:
    kind: str
    left_token: int
    right_token: int
    occurrences: int
    nonoverlap_occurrences: int
    trial_count: int
    subject_count: int
    pmi_bits: float
    npmi: float
    mdl_parent_symbol_count: int
    mdl_child_symbol_bits: int
    mdl_parent_symbol_bits: int
    mdl_event_bits: int
    mdl_savings_per_nonoverlap_bits: float
    mdl_dictionary_cost_bits: float
    mdl_gain_bits: float
    loso_total_fit_subject_count: int
    loso_supporter_stability: float
    loso_supporter_pass_count: int
    loso_supporter_evaluation_count: int
    loso_marginal_only_subject_count: int
    loso_effectful_stability: float
    loso_effectful_pass_count: int
    loso_effectful_evaluation_count: int
    loso_no_kind_noop_subject_count: int
    loso_folds: tuple[dict, ...]
    gate_checks: dict
    parent_id: str | None = None

    @property
    def key(self) -> ParentKey:
        return (self.kind, int(self.left_token), int(self.right_token))

    @property
    def loso_stability(self) -> float:
        """Backward-compatible conservative LOSO value used by CSV exports."""

        return min(
            float(self.loso_supporter_stability),
            float(self.loso_effectful_stability),
        )

    @property
    def loso_supporting_subject_count(self) -> int:
        """Backward-compatible alias for the supporter denominator."""

        return int(self.loso_supporter_evaluation_count)

    def to_dict(self) -> dict:
        return _jsonable(asdict(self))


@dataclass(frozen=True)
class ParentCatalogState:
    config: ParentGateConfig
    patterns: tuple[ParentPatternStats, ...]
    fit_data_sha256: str
    codebook_state_sha256: str
    fit_trial_keys: tuple[str, ...]
    fit_subject_keys: tuple[str, ...]
    primitive_num: int
    catalog_mode: str
    selection_metadata: dict
    algorithm_revision: str = ALGORITHM_REVISION
    schema_version: int = SCHEMA_VERSION

    def validate(self) -> "ParentCatalogState":
        self.config.validate()
        if self.schema_version != SCHEMA_VERSION or self.algorithm_revision != ALGORITHM_REVISION:
            raise ValueError("Parent catalog schema/revision mismatch.")
        if int(self.primitive_num) < 2:
            raise ValueError("Parent catalog primitive_num must be >=2.")
        keys = [item.key for item in self.patterns]
        if len(keys) != len(set(keys)):
            raise ValueError("Parent catalog contains duplicate keys.")
        if keys != sorted(keys):
            raise ValueError("Parent catalog patterns must use deterministic key order.")
        ids = [item.parent_id for item in self.patterns if item.parent_id is not None]
        if len(ids) != len(set(ids)) or any(not item for item in ids):
            raise ValueError("Registered parent IDs must be unique and non-empty.")
        for item in self.patterns:
            if item.kind not in EVENT_KINDS:
                raise ValueError("Parent pattern kind must be peak or valley.")
            if not 0 <= item.left_token < self.primitive_num or not 0 <= item.right_token < self.primitive_num:
                raise ValueError("Parent child token outside codebook.")
            if item.occurrences < 1 or item.nonoverlap_occurrences < 1:
                raise ValueError("Observed parent pattern must have positive support.")
            if (
                item.mdl_parent_symbol_count < 1
                or item.mdl_child_symbol_bits < 1
                or item.mdl_parent_symbol_bits < 1
                or item.mdl_event_bits < 1
            ):
                raise ValueError("Parent MDL alphabet sizes/bits must be positive.")
            if not -1.0 - 1e-9 <= item.npmi <= 1.0 + 1e-9:
                raise ValueError("Parent NPMI is outside [-1,1].")
            for value in (
                item.pmi_bits,
                item.npmi,
                item.mdl_savings_per_nonoverlap_bits,
                item.mdl_dictionary_cost_bits,
                item.mdl_gain_bits,
                item.loso_supporter_stability,
                item.loso_effectful_stability,
            ):
                _require_finite("parent statistic", value)
            loso_counts = {
                "loso_total_fit_subject_count": (
                    item.loso_total_fit_subject_count
                ),
                "loso_supporter_pass_count": item.loso_supporter_pass_count,
                "loso_supporter_evaluation_count": (
                    item.loso_supporter_evaluation_count
                ),
                "loso_marginal_only_subject_count": (
                    item.loso_marginal_only_subject_count
                ),
                "loso_effectful_pass_count": item.loso_effectful_pass_count,
                "loso_effectful_evaluation_count": (
                    item.loso_effectful_evaluation_count
                ),
                "loso_no_kind_noop_subject_count": (
                    item.loso_no_kind_noop_subject_count
                ),
            }
            for name, value in loso_counts.items():
                if (
                    isinstance(value, bool)
                    or int(value) != value
                    or int(value) < 0
                ):
                    raise ValueError(f"{name} must be a non-negative integer.")
            if item.loso_supporter_evaluation_count < 1:
                raise ValueError("Observed parent keys require a LOSO supporter.")
            if (
                item.loso_supporter_pass_count
                > item.loso_supporter_evaluation_count
                or item.loso_effectful_pass_count
                > item.loso_effectful_evaluation_count
            ):
                raise ValueError("LOSO pass count exceeds its evaluation count.")
            if item.loso_effectful_evaluation_count != (
                item.loso_supporter_evaluation_count
                + item.loso_marginal_only_subject_count
            ):
                raise ValueError(
                    "Effectful LOSO count must equal supporter plus marginal-only "
                    "subject counts."
                )
            if item.loso_total_fit_subject_count != (
                item.loso_effectful_evaluation_count
                + item.loso_no_kind_noop_subject_count
            ):
                raise ValueError(
                    "Effectful and no-kind LOSO counts do not sum to all fit subjects."
                )
            if len(item.loso_folds) != item.loso_total_fit_subject_count:
                raise ValueError(
                    "LOSO fold table length differs from total fit-subject count."
                )
            for name, value in (
                ("supporter", item.loso_supporter_stability),
                ("effectful", item.loso_effectful_stability),
            ):
                if not 0.0 <= value <= 1.0:
                    raise ValueError(f"LOSO {name} stability must lie in [0,1].")
            expected_supporter_stability = (
                item.loso_supporter_pass_count
                / item.loso_supporter_evaluation_count
            )
            expected_effectful_stability = (
                item.loso_effectful_pass_count
                / item.loso_effectful_evaluation_count
            )
            if not math.isclose(
                item.loso_supporter_stability,
                expected_supporter_stability,
                rel_tol=1e-12,
                abs_tol=1e-12,
            ):
                raise ValueError(
                    "Supporter LOSO stability disagrees with pass/evaluation counts."
                )
            if not math.isclose(
                item.loso_effectful_stability,
                expected_effectful_stability,
                rel_tol=1e-12,
                abs_tol=1e-12,
            ):
                raise ValueError(
                    "Effectful LOSO stability disagrees with pass/evaluation counts."
                )
        _validate_sha256("fit_data_sha256", self.fit_data_sha256)
        _validate_sha256("codebook_state_sha256", self.codebook_state_sha256)
        if (
            tuple(sorted(set(self.fit_trial_keys))) != tuple(self.fit_trial_keys)
            or tuple(sorted(set(self.fit_subject_keys))) != tuple(self.fit_subject_keys)
            or not self.fit_trial_keys
            or not self.fit_subject_keys
        ):
            raise ValueError("Fit trial/subject keys must be non-empty sorted unique tuples.")
        expected_fold_subjects = tuple(self.fit_subject_keys)
        allowed_effects = {
            "support_and_kind_marginals",
            "kind_marginals_only",
            "no_kind_noop",
        }
        for item in self.patterns:
            if item.loso_total_fit_subject_count != len(expected_fold_subjects):
                raise ValueError(
                    "LOSO total count must equal the complete fit-subject count."
                )
            fold_subjects = tuple(
                fold.get("omitted_subject_key") for fold in item.loso_folds
            )
            if fold_subjects != expected_fold_subjects:
                raise ValueError(
                    "LOSO folds must cover every fit subject exactly once in "
                    "deterministic order."
                )
            observed_effect_counts = {
                "support_and_kind_marginals": 0,
                "kind_marginals_only": 0,
                "no_kind_noop": 0,
            }
            observed_supporter_pass_count = 0
            observed_effectful_pass_count = 0
            for fold in item.loso_folds:
                supports_key = fold.get("subject_supports_key")
                has_event_kind = fold.get("subject_has_event_kind")
                passed = fold.get("passed")
                in_supporter = fold.get("included_in_supporter_stability")
                in_effectful = fold.get("included_in_effectful_stability")
                if not all(
                    isinstance(value, bool)
                    for value in (
                        supports_key,
                        has_event_kind,
                        passed,
                        in_supporter,
                        in_effectful,
                    )
                ):
                    raise ValueError(
                        "LOSO fold support/kind/inclusion/pass flags must be booleans."
                    )
                effect = fold.get("omission_effect")
                if effect not in allowed_effects:
                    raise ValueError("Unknown LOSO omission effect.")
                expected_effect = (
                    "support_and_kind_marginals"
                    if supports_key
                    else "kind_marginals_only"
                    if has_event_kind
                    else "no_kind_noop"
                )
                if supports_key and not has_event_kind:
                    raise ValueError(
                        "A subject cannot support a key without its event kind."
                    )
                if effect != expected_effect:
                    raise ValueError(
                        "LOSO omission effect disagrees with support/kind flags."
                    )
                if in_supporter != supports_key or in_effectful != has_event_kind:
                    raise ValueError(
                        "LOSO stability inclusion flags disagree with omission role."
                    )
                observed_effect_counts[effect] += 1
                observed_supporter_pass_count += int(passed and in_supporter)
                observed_effectful_pass_count += int(passed and in_effectful)
            if (
                observed_supporter_pass_count
                != item.loso_supporter_pass_count
                or observed_effectful_pass_count
                != item.loso_effectful_pass_count
            ):
                raise ValueError("LOSO fold pass flags disagree with pass counts.")
            if (
                observed_effect_counts["support_and_kind_marginals"]
                != item.loso_supporter_evaluation_count
                or observed_effect_counts["kind_marginals_only"]
                != item.loso_marginal_only_subject_count
                or observed_effect_counts["no_kind_noop"]
                != item.loso_no_kind_noop_subject_count
            ):
                raise ValueError(
                    "LOSO fold omission effects disagree with category counts."
                )
            expected_loso_gates = {
                "loso_supporter_stability": (
                    item.loso_supporter_stability
                    >= self.config.minimum_loso_stability
                ),
                "loso_effectful_stability": (
                    item.loso_effectful_stability
                    >= self.config.minimum_loso_stability
                ),
            }
            if "loso_stability" in item.gate_checks:
                raise ValueError(
                    "Ambiguous legacy loso_stability gate is not valid in schema v4."
                )
            for name, expected in expected_loso_gates.items():
                observed = item.gate_checks.get(name)
                if not isinstance(observed, bool) or observed != expected:
                    raise ValueError(f"{name} gate disagrees with stored stability.")
        return self

    @property
    def registered_patterns(self) -> tuple[ParentPatternStats, ...]:
        return tuple(item for item in self.patterns if item.parent_id is not None)

    def _payload(self) -> dict:
        return {
            "schema_version": int(self.schema_version),
            "algorithm_revision": self.algorithm_revision,
            "config": self.config.to_dict(),
            "patterns": [item.to_dict() for item in self.patterns],
            "fit_data_sha256": self.fit_data_sha256,
            "codebook_state_sha256": self.codebook_state_sha256,
            "fit_trial_keys": list(self.fit_trial_keys),
            "fit_subject_keys": list(self.fit_subject_keys),
            "primitive_num": int(self.primitive_num),
            "catalog_mode": self.catalog_mode,
            "selection_metadata": _jsonable(self.selection_metadata),
        }

    def state_hash(self) -> str:
        self.validate()
        return canonical_sha256(self._payload())

    def to_dict(self) -> dict:
        result = self._payload()
        result["state_sha256"] = self.state_hash()
        return result

    @classmethod
    def from_dict(cls, payload: Mapping) -> "ParentCatalogState":
        values = dict(payload)
        expected_hash = values.pop("state_sha256", None)
        if expected_hash is None:
            raise ValueError("Serialized parent catalog lacks state_sha256.")
        if (
            values.get("schema_version") != SCHEMA_VERSION
            or values.get("algorithm_revision") != ALGORITHM_REVISION
        ):
            raise ValueError("Parent catalog schema/revision mismatch.")
        config = ParentGateConfig.from_dict(values.pop("config"))
        patterns = tuple(
            ParentPatternStats(
                **{
                    **item,
                    "loso_folds": tuple(item.get("loso_folds", ())),
                }
            )
            for item in values.pop("patterns")
        )
        values["fit_trial_keys"] = tuple(values["fit_trial_keys"])
        values["fit_subject_keys"] = tuple(values["fit_subject_keys"])
        state = cls(config=config, patterns=patterns, **values).validate()
        _validate_sha256("state_sha256", expected_hash)
        if expected_hash != state.state_hash():
            raise ValueError("Parent catalog state hash mismatch.")
        return state


@dataclass(frozen=True)
class _ParentEventRow:
    key: ParentKey
    trial_id: Identifier
    subject_id: Identifier
    boundary_offset: int


def _ordered_tokenized(trials: Sequence[TokenizedTrial]) -> list[TokenizedTrial]:
    validated = [item.validate() for item in trials]
    if not validated:
        raise ValueError("At least one tokenized trial is required.")
    keys = [_identifier_sort_key(item.trial_id) for item in validated]
    if len(keys) != len(set(keys)):
        raise ValueError("Tokenized trials contain duplicate IDs.")
    hashes = {item.codebook_state_sha256 for item in validated}
    primitive_nums = {item.primitive_num for item in validated}
    if len(hashes) != 1 or len(primitive_nums) != 1:
        raise ValueError("Tokenized trials came from different codebooks.")
    return sorted(validated, key=lambda item: _identifier_sort_key(item.trial_id))


def _parent_rows(trials: Sequence[TokenizedTrial]) -> list[_ParentEventRow]:
    rows = []
    for trial in trials:
        if len(trial.segmentation.events) != len(trial.primitive_tokens) - 1:
            raise ValueError("Each adjacent child pair must have one retained turning event.")
        for offset, event in enumerate(trial.segmentation.events):
            rows.append(
                _ParentEventRow(
                    key=(
                        event.kind,
                        int(trial.primitive_tokens[offset]),
                        int(trial.primitive_tokens[offset + 1]),
                    ),
                    trial_id=trial.trial_id,
                    subject_id=trial.subject_id,
                    boundary_offset=int(offset),
                )
            )
    return rows


def _nonoverlap_count(rows: Sequence[_ParentEventRow]) -> int:
    by_trial: dict[str, list[int]] = {}
    for row in rows:
        by_trial.setdefault(_identifier_sort_key(row.trial_id), []).append(
            row.boundary_offset
        )
    count = 0
    for offsets in by_trial.values():
        previous_end = -1
        for start in sorted(offsets):
            if start > previous_end:
                count += 1
                previous_end = start + 1
    return count


def _association_for_key(
    rows: Sequence[_ParentEventRow],
    key: ParentKey,
    primitive_num: int,
    parent_symbol_count: int = 1,
) -> dict:
    if (
        isinstance(parent_symbol_count, bool)
        or int(parent_symbol_count) != parent_symbol_count
        or int(parent_symbol_count) < 1
    ):
        raise ValueError("parent_symbol_count must be an integer >=1.")
    parent_symbol_count = int(parent_symbol_count)
    child_bits = math.ceil(math.log2(max(2, primitive_num)))
    parent_symbol_bits = math.ceil(
        math.log2(max(2, primitive_num + parent_symbol_count))
    )
    event_bits = math.ceil(math.log2(max(2, len(EVENT_KINDS))))
    dictionary_cost = 2 * child_bits + event_bits
    savings_per_occurrence = (
        2 * child_bits + event_bits - parent_symbol_bits
    )
    kind_rows = [row for row in rows if row.key[0] == key[0]]
    matching = [row for row in kind_rows if row.key == key]
    if not matching or not kind_rows:
        return {
            "occurrences": 0,
            "nonoverlap_occurrences": 0,
            "trial_count": 0,
            "subject_count": 0,
            "pmi_bits": float("-inf"),
            "npmi": -1.0,
            "mdl_parent_symbol_count": parent_symbol_count,
            "mdl_child_symbol_bits": child_bits,
            "mdl_parent_symbol_bits": parent_symbol_bits,
            "mdl_event_bits": event_bits,
            "mdl_savings_per_nonoverlap_bits": float(savings_per_occurrence),
            "dictionary_cost_bits": float(dictionary_cost),
            "mdl_gain_bits": float("-inf"),
        }
    count = len(matching)

    # Subject -> trial -> boundary equal weighting.  Raw occurrence counts are
    # still reported separately for the support gates; only probability mass is
    # balanced, preventing a long trial or prolific subject from dominating
    # the NPMI marginals.
    subject_groups: dict[str, list[_ParentEventRow]] = {}
    for row in kind_rows:
        subject_groups.setdefault(_identifier_sort_key(row.subject_id), []).append(row)
    weights_by_identity: dict[int, float] = {}
    subject_count_for_kind = len(subject_groups)
    for subject_rows in subject_groups.values():
        trial_groups: dict[str, list[_ParentEventRow]] = {}
        for row in subject_rows:
            trial_groups.setdefault(_identifier_sort_key(row.trial_id), []).append(row)
        for trial_rows in trial_groups.values():
            weight = 1.0 / (
                subject_count_for_kind * len(trial_groups) * len(trial_rows)
            )
            for row in trial_rows:
                weights_by_identity[id(row)] = float(weight)
    total_mass = float(sum(weights_by_identity.values()))
    if not math.isclose(total_mass, 1.0, rel_tol=1e-10, abs_tol=1e-10):
        raise RuntimeError(f"Balanced NPMI weights sum to {total_mass}, expected one.")
    pxy = float(sum(weights_by_identity[id(row)] for row in matching))
    px = float(
        sum(
            weights_by_identity[id(row)]
            for row in kind_rows
            if row.key[1] == key[1]
        )
    )
    py = float(
        sum(
            weights_by_identity[id(row)]
            for row in kind_rows
            if row.key[2] == key[2]
        )
    )
    pmi_bits = math.log2(pxy / (px * py))
    denominator = -math.log(pxy)
    # A corpus containing only one pair has no evidence beyond its marginals;
    # define the otherwise 0/0 NPMI as zero, not perfect association.
    npmi = 0.0 if denominator <= EPS else math.log(pxy / (px * py)) / denominator
    nonoverlap = _nonoverlap_count(matching)
    return {
        "occurrences": int(count),
        "nonoverlap_occurrences": int(nonoverlap),
        "trial_count": int(
            len({_identifier_sort_key(row.trial_id) for row in matching})
        ),
        "subject_count": int(
            len({_identifier_sort_key(row.subject_id) for row in matching})
        ),
        "pmi_bits": float(pmi_bits),
        "npmi": float(np.clip(npmi, -1.0, 1.0)),
        "mdl_parent_symbol_count": parent_symbol_count,
        "mdl_child_symbol_bits": child_bits,
        "mdl_parent_symbol_bits": parent_symbol_bits,
        "mdl_event_bits": event_bits,
        "mdl_savings_per_nonoverlap_bits": float(savings_per_occurrence),
        "dictionary_cost_bits": float(dictionary_cost),
        "mdl_gain_bits": float(
            nonoverlap * savings_per_occurrence - dictionary_cost
        ),
    }


class _ParentAssociationIndex:
    """Cache all parent-association statistics needed by one catalog pass.

    ``_association_for_key`` is intentionally retained as a small, transparent
    reference implementation. Calling it from ``_build_parent_patterns`` used
    to rescan the complete event corpus once per key and once again for every
    leave-one-subject-out (LOSO) fold. A real USC-HAD fold contains thousands
    of events and up to ``2*K*K`` keys, so that implementation made the formal
    grid unnecessarily quadratic in the number of observed motifs.

    This index performs the same subject-equal -> trial-equal -> boundary-equal
    weighting in one stable row-order pass per event kind and omitted subject.
    Raw support and non-overlap counts are cached by key/subject. No activity
    label enters the index.  The index also distinguishes subjects that affect
    only the event-kind probability marginals from subjects for which omitting
    that event kind is an exact no-op.
    """

    def __init__(
        self,
        rows: Sequence[_ParentEventRow],
        primitive_num: int,
        parent_symbol_count: int,
    ) -> None:
        if (
            isinstance(parent_symbol_count, bool)
            or int(parent_symbol_count) != parent_symbol_count
            or int(parent_symbol_count) < 1
        ):
            raise ValueError("parent_symbol_count must be an integer >=1.")
        if (
            isinstance(primitive_num, bool)
            or int(primitive_num) != primitive_num
            or int(primitive_num) < 2
        ):
            raise ValueError("primitive_num must be an integer >=2.")
        self.primitive_num = int(primitive_num)
        self.parent_symbol_count = int(parent_symbol_count)
        self._rows = list(rows)
        self._rows_by_key: dict[ParentKey, list[_ParentEventRow]] = {}
        self._rows_by_kind: dict[str, list[_ParentEventRow]] = {
            kind: [] for kind in EVENT_KINDS
        }
        for row in self._rows:
            self._rows_by_key.setdefault(row.key, []).append(row)
            self._rows_by_kind.setdefault(row.key[0], []).append(row)

        self._support: dict[ParentKey, dict] = {}
        for key, matching in self._rows_by_key.items():
            subject_groups: dict[str, list[_ParentEventRow]] = {}
            for row in matching:
                subject_groups.setdefault(
                    _identifier_sort_key(row.subject_id), []
                ).append(row)
            subject_support = {}
            for subject_key, subject_rows in subject_groups.items():
                subject_support[subject_key] = {
                    "occurrences": int(len(subject_rows)),
                    "trial_count": int(
                        len(
                            {
                                _identifier_sort_key(row.trial_id)
                                for row in subject_rows
                            }
                        )
                    ),
                    "nonoverlap_occurrences": int(
                        _nonoverlap_count(subject_rows)
                    ),
                }
            self._support[key] = {
                "occurrences": int(len(matching)),
                "nonoverlap_occurrences": int(_nonoverlap_count(matching)),
                "trial_count": int(
                    len(
                        {
                            _identifier_sort_key(row.trial_id)
                            for row in matching
                        }
                    )
                ),
                "subject_keys": tuple(sorted(subject_groups)),
                "by_subject": subject_support,
            }

        # (event kind, omitted subject key or None) -> probability arrays.
        # Arrays are accumulated in original row order. The denominator is
        # written exactly as in _association_for_key to keep floating-point
        # results equivalent up to machine precision.
        self._mass_tables: dict[tuple[str, str | None], dict] = {}
        self._subject_keys_by_kind: dict[str, frozenset[str]] = {}
        for kind in EVENT_KINDS:
            kind_rows = self._rows_by_kind.get(kind, [])
            subject_trial_rows: dict[
                str, dict[str, list[_ParentEventRow]]
            ] = {}
            for row in kind_rows:
                subject_key = _identifier_sort_key(row.subject_id)
                trial_key = _identifier_sort_key(row.trial_id)
                subject_trial_rows.setdefault(subject_key, {}).setdefault(
                    trial_key, []
                ).append(row)
            kind_subjects = tuple(subject_trial_rows)
            self._subject_keys_by_kind[kind] = frozenset(kind_subjects)
            exclusions: tuple[str | None, ...] = (None, *kind_subjects)
            for omitted_subject in exclusions:
                active_subject_count = len(kind_subjects) - int(
                    omitted_subject in subject_trial_rows
                )
                pair_mass = np.zeros(
                    (self.primitive_num, self.primitive_num), dtype=np.float64
                )
                left_mass = np.zeros(self.primitive_num, dtype=np.float64)
                right_mass = np.zeros(self.primitive_num, dtype=np.float64)
                total_mass = 0.0
                if active_subject_count:
                    for row in kind_rows:
                        subject_key = _identifier_sort_key(row.subject_id)
                        if subject_key == omitted_subject:
                            continue
                        trial_key = _identifier_sort_key(row.trial_id)
                        trial_groups = subject_trial_rows[subject_key]
                        weight = 1.0 / (
                            active_subject_count
                            * len(trial_groups)
                            * len(trial_groups[trial_key])
                        )
                        left_token = int(row.key[1])
                        right_token = int(row.key[2])
                        pair_mass[left_token, right_token] += weight
                        left_mass[left_token] += weight
                        right_mass[right_token] += weight
                        total_mass += weight
                    if not math.isclose(
                        total_mass, 1.0, rel_tol=1e-10, abs_tol=1e-10
                    ):
                        raise RuntimeError(
                            "Cached balanced NPMI weights sum to "
                            f"{total_mass}, expected one."
                        )
                self._mass_tables[(kind, omitted_subject)] = {
                    "kind_row_count": int(
                        sum(
                            _identifier_sort_key(row.subject_id)
                            != omitted_subject
                            for row in kind_rows
                        )
                    ),
                    "pair_mass": pair_mass,
                    "left_mass": left_mass,
                    "right_mass": right_mass,
                }

    def supporting_subjects(self, key: ParentKey) -> tuple[str, ...]:
        support = self._support.get(key)
        return () if support is None else tuple(support["subject_keys"])

    def subject_has_event_kind(self, kind: str, subject_key: str) -> bool:
        return subject_key in self._subject_keys_by_kind.get(kind, frozenset())

    def association(
        self, key: ParentKey, omitted_subject: str | None = None
    ) -> dict:
        kind = str(key[0])
        support = self._support.get(key)
        subject_support = None
        if support is not None and omitted_subject is not None:
            subject_support = support["by_subject"].get(omitted_subject)

        occurrences = 0 if support is None else int(support["occurrences"])
        nonoverlap = (
            0 if support is None else int(support["nonoverlap_occurrences"])
        )
        trial_count = 0 if support is None else int(support["trial_count"])
        subject_count = (
            0 if support is None else int(len(support["subject_keys"]))
        )
        if subject_support is not None:
            occurrences -= int(subject_support["occurrences"])
            nonoverlap -= int(subject_support["nonoverlap_occurrences"])
            trial_count -= int(subject_support["trial_count"])
            subject_count -= 1

        child_bits = math.ceil(math.log2(max(2, self.primitive_num)))
        parent_symbol_bits = math.ceil(
            math.log2(
                max(2, self.primitive_num + self.parent_symbol_count)
            )
        )
        event_bits = math.ceil(math.log2(max(2, len(EVENT_KINDS))))
        dictionary_cost = 2 * child_bits + event_bits
        savings_per_occurrence = (
            2 * child_bits + event_bits - parent_symbol_bits
        )

        # Omitting a subject with no event of this kind is a no-op, matching
        # the reference implementation's subject universe.
        table_key = (kind, omitted_subject)
        if table_key not in self._mass_tables:
            table_key = (kind, None)
        table = self._mass_tables.get(table_key)
        if (
            occurrences < 1
            or table is None
            or int(table["kind_row_count"]) < 1
        ):
            return {
                "occurrences": 0,
                "nonoverlap_occurrences": 0,
                "trial_count": 0,
                "subject_count": 0,
                "pmi_bits": float("-inf"),
                "npmi": -1.0,
                "mdl_parent_symbol_count": self.parent_symbol_count,
                "mdl_child_symbol_bits": child_bits,
                "mdl_parent_symbol_bits": parent_symbol_bits,
                "mdl_event_bits": event_bits,
                "mdl_savings_per_nonoverlap_bits": float(
                    savings_per_occurrence
                ),
                "dictionary_cost_bits": float(dictionary_cost),
                "mdl_gain_bits": float("-inf"),
            }

        left_token = int(key[1])
        right_token = int(key[2])
        pxy = float(table["pair_mass"][left_token, right_token])
        px = float(table["left_mass"][left_token])
        py = float(table["right_mass"][right_token])
        if pxy <= 0.0 or px <= 0.0 or py <= 0.0:
            raise RuntimeError(
                "Cached parent probabilities disagree with positive raw support."
            )
        pmi_bits = math.log2(pxy / (px * py))
        denominator = -math.log(pxy)
        npmi = (
            0.0
            if denominator <= EPS
            else math.log(pxy / (px * py)) / denominator
        )
        return {
            "occurrences": int(occurrences),
            "nonoverlap_occurrences": int(nonoverlap),
            "trial_count": int(trial_count),
            "subject_count": int(subject_count),
            "pmi_bits": float(pmi_bits),
            "npmi": float(np.clip(npmi, -1.0, 1.0)),
            "mdl_parent_symbol_count": self.parent_symbol_count,
            "mdl_child_symbol_bits": child_bits,
            "mdl_parent_symbol_bits": parent_symbol_bits,
            "mdl_event_bits": event_bits,
            "mdl_savings_per_nonoverlap_bits": float(
                savings_per_occurrence
            ),
            "dictionary_cost_bits": float(dictionary_cost),
            "mdl_gain_bits": float(
                nonoverlap * savings_per_occurrence - dictionary_cost
            ),
        }


def _tokenized_fit_hash(trials: Sequence[TokenizedTrial]) -> str:
    digest = hashlib.sha256()
    digest.update(ALGORITHM_REVISION.encode("ascii"))
    for trial in trials:
        digest.update(_identifier_sort_key(trial.trial_id).encode("utf-8"))
        digest.update(_identifier_sort_key(trial.subject_id).encode("utf-8"))
        _hash_array(digest, "boundaries", trial.segmentation.boundaries)
        _hash_array(digest, "primitive_tokens", trial.primitive_tokens)
        digest.update("|".join(event.kind for event in trial.segmentation.events).encode("ascii"))
    return digest.hexdigest()


def _loso_selection_metadata(trials: Sequence[TokenizedTrial]) -> dict:
    """Describe dual-denominator LOSO diagnostics used by parent catalogs."""

    return {
        "loso_definition": (
            "dual LOSO gates over the complete caller-supplied offline-fit subject "
            "fold table: supporter-conditioned stability prevents non-supporters "
            "from diluting key-support failures, while effectful stability also "
            "tests same-kind non-supporters that change NPMI marginals; no-kind "
            "no-op folds are audited but excluded from both denominators; LOSO "
            "is not recursively applied"
        ),
        "loso_supporter_definition": (
            "fraction of subjects supporting the candidate key whose removal "
            "leaves every absolute occurrence/trial/subject/NPMI/MDL base gate "
            "passing"
        ),
        "loso_effectful_definition": (
            "fraction of subjects containing the candidate event kind whose "
            "removal leaves every base gate passing; includes key supporters and "
            "same-kind marginal-only subjects"
        ),
        "loso_threshold_diagnostics": (
            "supporter_stability and effectful_stability are each compared with "
            "minimum_loso_stability; catalog_mode determines whether those checks "
            "can veto registration"
        ),
        "loso_subject_universe": (
            "all caller-supplied offline-fit subjects"
        ),
        "fit_subject_count": int(
            len({_identifier_sort_key(trial.subject_id) for trial in trials})
        ),
        "no_kind_subjects_are_explicit_noops": True,
        "no_kind_subjects_in_stability_denominators": False,
    }


def _build_parent_patterns(
    trials: Sequence[TokenizedTrial],
    config: ParentGateConfig,
    *,
    parent_symbol_count: int = 1,
) -> tuple[list[ParentPatternStats], list[_ParentEventRow]]:
    rows = _parent_rows(trials)
    if not rows:
        return [], rows
    keys = sorted({row.key for row in rows})
    primitive_num = trials[0].primitive_num
    association_index = _ParentAssociationIndex(
        rows,
        primitive_num,
        parent_symbol_count,
    )
    all_fit_subjects = tuple(
        sorted({_identifier_sort_key(trial.subject_id) for trial in trials})
    )
    patterns = []
    for key in keys:
        metrics = association_index.association(key)
        supporting_subjects = set(association_index.supporting_subjects(key))
        loso_supporter_pass = 0
        loso_effectful_pass = 0
        loso_folds = []
        omission_counts = {
            "support_and_kind_marginals": 0,
            "kind_marginals_only": 0,
            "no_kind_noop": 0,
        }
        for subject in all_fit_subjects:
            subject_supports_key = subject in supporting_subjects
            subject_has_event_kind = association_index.subject_has_event_kind(
                key[0], subject
            )
            omission_effect = (
                "support_and_kind_marginals"
                if subject_supports_key
                else "kind_marginals_only"
                if subject_has_event_kind
                else "no_kind_noop"
            )
            omission_counts[omission_effect] += 1
            reduced_metrics = association_index.association(
                key,
                omitted_subject=subject,
            )
            base_checks = {
                "occurrences": (
                    reduced_metrics["occurrences"] >= config.minimum_occurrences
                ),
                "trials": reduced_metrics["trial_count"] >= config.minimum_trials,
                "subjects": (
                    reduced_metrics["subject_count"] >= config.minimum_subjects
                ),
                "npmi": reduced_metrics["npmi"] >= config.minimum_npmi,
                "mdl_gain": (
                    reduced_metrics["mdl_gain_bits"]
                    >= config.minimum_mdl_gain_bits
                ),
            }
            passed = all(base_checks.values())
            if passed and subject_supports_key:
                loso_supporter_pass += 1
            if passed and subject_has_event_kind:
                loso_effectful_pass += 1
            loso_folds.append(
                {
                    "omitted_subject_key": subject,
                    "subject_supports_key": bool(subject_supports_key),
                    "subject_has_event_kind": bool(subject_has_event_kind),
                    "omission_effect": omission_effect,
                    "included_in_supporter_stability": bool(
                        subject_supports_key
                    ),
                    "included_in_effectful_stability": bool(
                        subject_has_event_kind
                    ),
                    "occurrences": reduced_metrics["occurrences"],
                    "trial_count": reduced_metrics["trial_count"],
                    "subject_count": reduced_metrics["subject_count"],
                    "npmi": (
                        None
                        if not math.isfinite(reduced_metrics["npmi"])
                        else reduced_metrics["npmi"]
                    ),
                    "mdl_gain_bits": (
                        None
                        if not math.isfinite(reduced_metrics["mdl_gain_bits"])
                        else reduced_metrics["mdl_gain_bits"]
                    ),
                    "base_gate_checks": base_checks,
                    "passed": bool(passed),
                }
            )
        loso_supporter_total = omission_counts[
            "support_and_kind_marginals"
        ]
        loso_effectful_total = (
            loso_supporter_total + omission_counts["kind_marginals_only"]
        )
        loso_supporter_stability = (
            loso_supporter_pass / loso_supporter_total
        )
        loso_effectful_stability = (
            loso_effectful_pass / loso_effectful_total
        )
        checks = {
            "occurrences": metrics["occurrences"] >= config.minimum_occurrences,
            "trials": metrics["trial_count"] >= config.minimum_trials,
            "subjects": metrics["subject_count"] >= config.minimum_subjects,
            "npmi": metrics["npmi"] >= config.minimum_npmi,
            "mdl_gain": metrics["mdl_gain_bits"] >= config.minimum_mdl_gain_bits,
            "loso_supporter_stability": (
                loso_supporter_stability >= config.minimum_loso_stability
            ),
            "loso_effectful_stability": (
                loso_effectful_stability >= config.minimum_loso_stability
            ),
        }
        patterns.append(
            ParentPatternStats(
                kind=key[0],
                left_token=key[1],
                right_token=key[2],
                occurrences=metrics["occurrences"],
                nonoverlap_occurrences=metrics["nonoverlap_occurrences"],
                trial_count=metrics["trial_count"],
                subject_count=metrics["subject_count"],
                pmi_bits=metrics["pmi_bits"],
                npmi=metrics["npmi"],
                mdl_parent_symbol_count=metrics["mdl_parent_symbol_count"],
                mdl_child_symbol_bits=metrics["mdl_child_symbol_bits"],
                mdl_parent_symbol_bits=metrics["mdl_parent_symbol_bits"],
                mdl_event_bits=metrics["mdl_event_bits"],
                mdl_savings_per_nonoverlap_bits=metrics[
                    "mdl_savings_per_nonoverlap_bits"
                ],
                mdl_dictionary_cost_bits=metrics["dictionary_cost_bits"],
                mdl_gain_bits=metrics["mdl_gain_bits"],
                loso_total_fit_subject_count=int(len(all_fit_subjects)),
                loso_supporter_stability=float(loso_supporter_stability),
                loso_supporter_pass_count=int(loso_supporter_pass),
                loso_supporter_evaluation_count=int(loso_supporter_total),
                loso_marginal_only_subject_count=int(
                    omission_counts["kind_marginals_only"]
                ),
                loso_effectful_stability=float(loso_effectful_stability),
                loso_effectful_pass_count=int(loso_effectful_pass),
                loso_effectful_evaluation_count=int(loso_effectful_total),
                loso_no_kind_noop_subject_count=int(
                    omission_counts["no_kind_noop"]
                ),
                loso_folds=tuple(loso_folds),
                gate_checks=checks,
            )
        )
    return patterns, rows


def fit_parent_catalog(
    fit_tokenized_trials: Sequence[TokenizedTrial],
    config: ParentGateConfig = ParentGateConfig(),
) -> ParentCatalogState:
    """Register fit-only parent motifs passing support, NPMI, MDL, and LOSO gates."""

    config = config.validate()
    trials = _ordered_tokenized(fit_tokenized_trials)
    # First screen each candidate with a one-parent alphabet.  The final pass
    # conservatively reserves one symbol for every optimistic candidate, so a
    # power-of-two codebook boundary cannot make the MDL gain look better merely
    # because other registered parents were ignored.  The final registered set
    # may be smaller than this capacity, never larger.
    optimistic_patterns, _ = _build_parent_patterns(
        trials, config, parent_symbol_count=1
    )
    parent_symbol_capacity = max(
        1,
        sum(
            all(bool(value) for value in item.gate_checks.values())
            for item in optimistic_patterns
        ),
    )
    patterns, rows = _build_parent_patterns(
        trials, config, parent_symbol_count=parent_symbol_capacity
    )
    registered = []
    parent_index = 0
    for item in patterns:
        if all(bool(value) for value in item.gate_checks.values()):
            registered.append(replace(item, parent_id=f"H{parent_index:03d}"))
            parent_index += 1
        else:
            registered.append(item)
    state = ParentCatalogState(
        config=config,
        patterns=tuple(registered),
        fit_data_sha256=_tokenized_fit_hash(trials),
        codebook_state_sha256=trials[0].codebook_state_sha256,
        fit_trial_keys=tuple(
            sorted(_identifier_sort_key(item.trial_id) for item in trials)
        ),
        fit_subject_keys=tuple(
            sorted({_identifier_sort_key(item.subject_id) for item in trials})
        ),
        primitive_num=trials[0].primitive_num,
        catalog_mode="gated_fit_only",
        selection_metadata={
            "uses_activity_labels": False,
            **_loso_selection_metadata(trials),
            "loso_registration_rule": (
                "supporter_stability and effectful_stability must each be at least "
                "minimum_loso_stability"
            ),
            "event_count": int(len(rows)),
            "observed_pattern_count": int(len(patterns)),
            "registered_parent_count": int(parent_index),
            "mdl_parent_symbol_capacity": int(parent_symbol_capacity),
            "npmi_definition": (
                "kind-conditioned subject-equal, then trial-equal, then boundary-equal "
                "PMI(left,right)/-ln(p(pair)); single-pattern 0/0 is zero"
            ),
            "mdl_definition": (
                "conservative fixed-width latent compression proxy: each "
                "nonoverlapping 2-child+event occurrence is replaced by one parent "
                "token from an alphabet of K+P symbols; gain is "
                "m*(2*ceil(log2(K))+ceil(log2(2))-ceil(log2(K+P))) minus "
                "the 2-child+event dictionary entry cost. P reserves every parent "
                "passing the optimistic one-parent screen; overlays remain "
                "non-destructive, so this is not physical output-file compression"
            ),
        },
    )
    return state.validate()


def fit_frequency_parent_catalog(
    fit_tokenized_trials: Sequence[TokenizedTrial],
    minimum_occurrences: int,
) -> ParentCatalogState:
    """Register the E3 frequency-only parent baseline.

    Only the raw segment-level occurrence count decides registration.  Trial
    support, subject support, balanced NPMI, fixed-width MDL, and LOSO are still
    computed and serialized for an honest post-hoc audit, but they cannot veto
    an E3 parent.  E4 must continue to use ``fit_parent_catalog``.
    """

    if (
        isinstance(minimum_occurrences, bool)
        or int(minimum_occurrences) != minimum_occurrences
        or int(minimum_occurrences) < 2
    ):
        raise ValueError("Frequency-only minimum_occurrences must be an integer >=2.")
    threshold = int(minimum_occurrences)
    # This valid E4-shaped configuration is used only to compute the complete
    # diagnostic table.  catalog_mode and selection_metadata explicitly record
    # that registration itself uses occurrence count alone.
    audit_config = ParentGateConfig(
        minimum_occurrences=threshold,
        minimum_trials=2,
        minimum_subjects=2,
        minimum_npmi=-1.0,
        minimum_mdl_gain_bits=0.0,
        minimum_loso_stability=0.0,
    ).validate()
    trials = _ordered_tokenized(fit_tokenized_trials)
    optimistic_patterns, _ = _build_parent_patterns(
        trials, audit_config, parent_symbol_count=1
    )
    parent_symbol_capacity = max(
        1,
        sum(item.occurrences >= threshold for item in optimistic_patterns),
    )
    patterns, rows = _build_parent_patterns(
        trials,
        audit_config,
        parent_symbol_count=parent_symbol_capacity,
    )
    selected = []
    parent_index = 0
    for item in patterns:
        frequency_passed = item.occurrences >= threshold
        item = replace(
            item,
            gate_checks={
                **item.gate_checks,
                "frequency_only_registration": bool(frequency_passed),
            },
            parent_id=f"F{parent_index:03d}" if frequency_passed else None,
        )
        selected.append(item)
        if frequency_passed:
            parent_index += 1
    return ParentCatalogState(
        config=audit_config,
        patterns=tuple(selected),
        fit_data_sha256=_tokenized_fit_hash(trials),
        codebook_state_sha256=trials[0].codebook_state_sha256,
        fit_trial_keys=tuple(
            sorted(_identifier_sort_key(item.trial_id) for item in trials)
        ),
        fit_subject_keys=tuple(
            sorted({_identifier_sort_key(item.subject_id) for item in trials})
        ),
        primitive_num=trials[0].primitive_num,
        catalog_mode="frequency_only_fit_baseline",
        selection_metadata={
            "uses_activity_labels": False,
            **_loso_selection_metadata(trials),
            "registration_gate": "occurrences >= minimum_occurrences only",
            "minimum_occurrences": threshold,
            "other_statistics_can_veto_registration": False,
            "event_count": int(len(rows)),
            "observed_pattern_count": int(len(patterns)),
            "registered_parent_count": int(parent_index),
            "mdl_parent_symbol_capacity": int(parent_symbol_capacity),
            "mdl_role": (
                "diagnostic conservative latent-compression proxy only; MDL cannot "
                "veto E3 frequency registration"
            ),
        },
    ).validate()


@dataclass(frozen=True)
class ParentOccurrence:
    parent_id: str
    kind: str
    boundary_sample: int
    left_segment_index: int
    right_segment_index: int
    span_start_sample: int
    span_end_sample_exclusive: int
    left_token: int
    right_token: int

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class ParentOverlay:
    trial_id: Identifier
    subject_id: Identifier
    child_tokens: np.ndarray
    parent_occurrences: tuple[ParentOccurrence, ...]
    parent_event_sequence: tuple[str, ...]
    catalog_state_sha256: str
    non_destructive: bool = True

    def validate(self, expected_child_tokens: np.ndarray | None = None) -> "ParentOverlay":
        _validate_identifier("trial_id", self.trial_id)
        _validate_identifier("subject_id", self.subject_id)
        tokens = np.asarray(self.child_tokens, dtype=np.int64)
        if tokens.ndim != 1 or len(tokens) < 1:
            raise ValueError("Parent overlay child_tokens must be non-empty 1D.")
        if expected_child_tokens is not None and not np.array_equal(
            tokens, np.asarray(expected_child_tokens, dtype=np.int64)
        ):
            raise ValueError("Parent overlay modified the child token sequence.")
        if not self.non_destructive:
            raise ValueError("Parent overlays must be explicitly non-destructive.")
        if tuple(item.parent_id for item in self.parent_occurrences) != tuple(
            self.parent_event_sequence
        ):
            raise ValueError("Parent event sequence differs from ordered occurrences.")
        return self

    def to_dict(self) -> dict:
        return {
            "trial_id": _identifier_payload(self.trial_id),
            "subject_id": _identifier_payload(self.subject_id),
            "child_tokens": self.child_tokens.astype(int).tolist(),
            "parent_occurrences": [item.to_dict() for item in self.parent_occurrences],
            "parent_event_sequence": list(self.parent_event_sequence),
            "catalog_state_sha256": self.catalog_state_sha256,
            "non_destructive": True,
        }


def overlay_parents(
    tokenized_trial: TokenizedTrial,
    catalog: ParentCatalogState,
    *,
    role: Literal["fit_replay", "held_out"] = "held_out",
    require_subject_disjoint: bool = True,
) -> ParentOverlay:
    """Add registered parent spans while preserving every child token verbatim."""

    trial = tokenized_trial.validate()
    catalog = catalog.validate()
    if role not in {"fit_replay", "held_out"}:
        raise ValueError("role must be fit_replay or held_out.")
    trial_key = _identifier_sort_key(trial.trial_id)
    subject_key = _identifier_sort_key(trial.subject_id)
    if role == "held_out" and trial_key in catalog.fit_trial_keys:
        raise ValueError("Held-out parent application overlaps a fit trial ID.")
    if (
        role == "held_out"
        and bool(require_subject_disjoint)
        and subject_key in catalog.fit_subject_keys
    ):
        raise ValueError("Held-out parent application overlaps a fit subject ID.")
    if trial.codebook_state_sha256 != catalog.codebook_state_sha256:
        raise ValueError("Parent catalog and tokenized trial use different codebooks.")
    registered = {item.key: item for item in catalog.registered_patterns}
    occurrences = []
    for offset, event in enumerate(trial.segmentation.events):
        key = (
            event.kind,
            int(trial.primitive_tokens[offset]),
            int(trial.primitive_tokens[offset + 1]),
        )
        pattern = registered.get(key)
        if pattern is None:
            continue
        occurrences.append(
            ParentOccurrence(
                parent_id=str(pattern.parent_id),
                kind=event.kind,
                boundary_sample=int(event.sample_index),
                left_segment_index=int(offset),
                right_segment_index=int(offset + 1),
                span_start_sample=int(trial.segmentation.boundaries[offset]),
                span_end_sample_exclusive=int(
                    trial.segmentation.boundaries[offset + 2]
                ),
                left_token=key[1],
                right_token=key[2],
            )
        )
    overlay = ParentOverlay(
        trial_id=trial.trial_id,
        subject_id=trial.subject_id,
        child_tokens=trial.primitive_tokens.copy(),
        parent_occurrences=tuple(occurrences),
        parent_event_sequence=tuple(item.parent_id for item in occurrences),
        catalog_state_sha256=catalog.state_hash(),
    )
    return overlay.validate(trial.primitive_tokens)


def overlay_parent_trials(
    tokenized_trials: Sequence[TokenizedTrial],
    catalog: ParentCatalogState,
    *,
    role: Literal["fit_replay", "held_out"] = "held_out",
    require_subject_disjoint: bool = True,
) -> list[ParentOverlay]:
    return [
        overlay_parents(
            item,
            catalog,
            role=role,
            require_subject_disjoint=require_subject_disjoint,
        )
        for item in _ordered_tokenized(tokenized_trials)
    ]


def fit_matched_random_parent_catalog(
    fit_tokenized_trials: Sequence[TokenizedTrial],
    reference_catalog: ParentCatalogState,
    config: ParentGateConfig,
    seed: int,
) -> ParentCatalogState:
    """Build a label-free support-matched negative-motif control catalog.

    This is deliberately *not* a uniformly random catalog.  Within each event
    kind, global linear-sum assignment matches reference parents to distinct
    non-reference motifs by occurrence/trial/subject support.  Seeded epsilon
    perturbations only randomise exact or numerically indistinguishable optima.
    """

    config = config.validate()
    reference_catalog = reference_catalog.validate()
    trials = _ordered_tokenized(fit_tokenized_trials)
    if trials[0].codebook_state_sha256 != reference_catalog.codebook_state_sha256:
        raise ValueError("Reference catalog and fit trials use different codebooks.")
    fit_hash = _tokenized_fit_hash(trials)
    fit_trial_keys = tuple(
        sorted(_identifier_sort_key(item.trial_id) for item in trials)
    )
    fit_subject_keys = tuple(
        sorted({_identifier_sort_key(item.subject_id) for item in trials})
    )
    if reference_catalog.fit_data_sha256 != fit_hash:
        raise ValueError("Reference catalog was fitted on different tokenized trials.")
    if (
        reference_catalog.fit_trial_keys != fit_trial_keys
        or reference_catalog.fit_subject_keys != fit_subject_keys
    ):
        raise ValueError("Reference catalog fit trial/subject identities differ.")
    if reference_catalog.primitive_num != trials[0].primitive_num:
        raise ValueError("Reference catalog primitive count differs from fit trials.")
    references = sorted(
        reference_catalog.registered_patterns, key=lambda item: str(item.parent_id)
    )
    patterns, rows = _build_parent_patterns(
        trials,
        config,
        parent_symbol_count=max(1, len(references)),
    )
    excluded = {item.key for item in references}
    candidates = [item for item in patterns if item.key not in excluded]
    rng = np.random.default_rng(int(seed))
    selected: list[ParentPatternStats] = []
    matching: list[dict] = []

    def support_distance(
        reference: ParentPatternStats, candidate: ParentPatternStats
    ) -> float:
        return float(
            abs(math.log1p(candidate.occurrences) - math.log1p(reference.occurrences))
            + abs(candidate.trial_count - reference.trial_count)
            / max(1, reference.trial_count)
            + abs(candidate.subject_count - reference.subject_count)
            / max(1, reference.subject_count)
        )

    for kind in EVENT_KINDS:
        kind_references = [
            (index, item)
            for index, item in enumerate(references)
            if item.kind == kind
        ]
        if not kind_references:
            continue
        kind_candidates = [item for item in candidates if item.kind == kind]
        if len(kind_candidates) < len(kind_references):
            raise RuntimeError(
                "Not enough distinct same-event-kind non-reference motifs for "
                f"support matching: kind={kind!r}, references={len(kind_references)}, "
                f"candidates={len(kind_candidates)}."
            )
        base_cost = np.asarray(
            [
                [support_distance(reference, candidate) for candidate in kind_candidates]
                for _, reference in kind_references
            ],
            dtype=np.float64,
        )
        jitter_scale = np.finfo(np.float64).eps * max(
            1.0, float(np.max(np.abs(base_cost)))
        ) * 32.0
        perturbed_cost = base_cost + rng.random(base_cost.shape) * jitter_scale
        row_indices, column_indices = linear_sum_assignment(perturbed_cost)
        if len(row_indices) != len(kind_references):
            raise RuntimeError("Global support matching left a reference parent unmatched.")
        for row_index, column_index in zip(row_indices, column_indices):
            reference_index, reference = kind_references[int(row_index)]
            chosen = kind_candidates[int(column_index)]
            parent_id = f"R{reference_index:03d}"
            selected.append(replace(chosen, parent_id=parent_id))
            matching.append(
                {
                    "reference_parent_id": reference.parent_id,
                    "reference_key": list(reference.key),
                    "reference_event_kind": reference.kind,
                    "reference_occurrences": int(reference.occurrences),
                    "reference_trial_count": int(reference.trial_count),
                    "reference_subject_count": int(reference.subject_count),
                    # Retain the old field names for readers of schema-v1 result
                    # tables while explicitly identifying this as a negative motif.
                    "random_parent_id": parent_id,
                    "random_key": list(chosen.key),
                    "negative_parent_id": parent_id,
                    "negative_key": list(chosen.key),
                    "negative_event_kind": chosen.kind,
                    "negative_occurrences": int(chosen.occurrences),
                    "negative_trial_count": int(chosen.trial_count),
                    "negative_subject_count": int(chosen.subject_count),
                    "support_distance": float(base_cost[row_index, column_index]),
                    "seeded_perturbation": float(
                        perturbed_cost[row_index, column_index]
                        - base_cost[row_index, column_index]
                    ),
                }
            )
    matching.sort(key=lambda item: str(item["reference_parent_id"]))
    selected_keys = {item.key: item for item in selected}
    final_patterns = tuple(selected_keys.get(item.key, item) for item in patterns)
    support_distances = [float(item["support_distance"]) for item in matching]
    return ParentCatalogState(
        config=config,
        patterns=final_patterns,
        fit_data_sha256=fit_hash,
        codebook_state_sha256=trials[0].codebook_state_sha256,
        fit_trial_keys=fit_trial_keys,
        fit_subject_keys=fit_subject_keys,
        primitive_num=trials[0].primitive_num,
        catalog_mode="matched_random_control",
        selection_metadata={
            "uses_activity_labels": False,
            **_loso_selection_metadata(trials),
            "seed": int(seed),
            "reference_catalog_sha256": reference_catalog.state_hash(),
            "reference_parent_count": len(references),
            "selected_parent_count": len(selected),
            "control_definition": (
                "support_matched_negative_motif_control_not_uniform_random"
            ),
            "uniform_random_catalog": False,
            "same_event_kind_required": True,
            "selection_rule": (
                "same-event-kind global minimum-cost linear-sum assignment on "
                "log-occurrence/trial/subject support with seeded epsilon perturbation"
            ),
            "assignment_algorithm": "scipy.optimize.linear_sum_assignment",
            "seed_role": "epsilon perturbation of tied/nearly tied global assignments",
            "reference_fit_data_verified": True,
            "maximum_support_distance": (
                max(support_distances) if support_distances else 0.0
            ),
            "mean_support_distance": (
                float(np.mean(support_distances)) if support_distances else 0.0
            ),
            "matching": matching,
            "fit_event_count": len(rows),
        },
    ).validate()


def save_state_json(path: str | Path, state: PeakValleyState | CodebookState | ParentCatalogState) -> None:
    """Serialize one validated learned state as canonical, hash-bearing JSON."""

    payload = state.to_dict()
    Path(path).write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False),
        encoding="utf-8",
    )


def load_peak_valley_state(path: str | Path) -> PeakValleyState:
    return PeakValleyState.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def load_codebook_state(path: str | Path) -> CodebookState:
    return CodebookState.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def load_parent_catalog_state(path: str | Path) -> ParentCatalogState:
    return ParentCatalogState.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


__all__ = [
    "ALGORITHM_REVISION",
    "ChildPrimitiveBatch",
    "CodebookConfig",
    "CodebookState",
    "DEFAULT_CHANNEL_NAMES",
    "ParentCatalogState",
    "ParentGateConfig",
    "ParentOccurrence",
    "ParentOverlay",
    "ParentPatternStats",
    "PeakValleyConfig",
    "PeakValleyState",
    "SampleSegment",
    "SegmentedTrial",
    "TokenizedTrial",
    "TrialSignal",
    "TurningEvent",
    "assign_codebook",
    "assign_codebook_trials",
    "canonical_sha256",
    "extract_child_feature_batches",
    "extract_child_features",
    "fit_codebook",
    "fit_frequency_parent_catalog",
    "fit_matched_random_parent_catalog",
    "fit_parent_catalog",
    "fit_segmenter",
    "load_codebook_state",
    "load_parent_catalog_state",
    "load_peak_valley_state",
    "matched_random_segmentations",
    "overlay_parent_trials",
    "overlay_parents",
    "reconstruct_overlapping_windows",
    "save_state_json",
    "segment_trial",
    "segment_trials",
    "segmentation_from_boundaries",
    "smooth_six_axis",
]
