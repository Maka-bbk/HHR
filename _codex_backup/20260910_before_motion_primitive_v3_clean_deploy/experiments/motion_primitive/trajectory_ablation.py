"""Fixed-trajectory readout ablations for the motion-primitive experiment.

The four readouts in this module share exactly the same change-point segments,
KMeans codebook, RLE token sequences, and outer-subject split.  Only the local
edit cost changes:

``g1_hard_rle``
    Unit substitution cost for unequal token ids.
``g2_dynamic_soft``
    Continuous substitution cost from frozen codebook-center cosine distance.
``g3_dynamic_state``
    G2 plus a run-local raw-sensor state residual.
``g4_dynamic_state_duration_context``
    G3 plus run duration and immediate predecessor/successor context.

All feature scaling is fitted on split-role 0 trajectories only.  Evaluation
labels are read solely after distance matrices have been constructed.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable, Iterable, Optional, Sequence

import numpy as np

from experiments.motion_primitive.core import (
    EPS,
    activity_distance_matrix,
    association_summary,
    normalized_weighted_levenshtein,
    shuffle_valid_rle_tokens,
)


GROUP_NAMES = (
    "g1_hard_rle",
    "g2_dynamic_soft",
    "g3_dynamic_state",
    "g4_dynamic_state_duration_context",
)

STATE_FEATURE_NAMES = (
    "mean_acc_x",
    "mean_acc_y",
    "mean_acc_z",
    "gravity_dir_x",
    "gravity_dir_y",
    "gravity_dir_z",
    "log1p_std_acc_x",
    "log1p_std_acc_y",
    "log1p_std_acc_z",
    "log1p_std_gyro_x",
    "log1p_std_gyro_y",
    "log1p_std_gyro_z",
)


def jsonable(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    return value


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(jsonable(value), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def translate_recorded_path(value: str | Path) -> Path:
    """Translate a WSL /mnt/<drive>/ path for native Windows Python."""
    text = str(value).strip()
    match = re.match(r"^/mnt/([a-zA-Z])/(.*)$", text)
    if match and os.name == "nt":
        return Path(f"{match.group(1).upper()}:/{match.group(2)}")
    return Path(text).expanduser()


def stable_array_hash(*arrays: np.ndarray) -> str:
    digest = hashlib.sha256()
    for value in arrays:
        array = np.ascontiguousarray(np.asarray(value))
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(str(tuple(array.shape)).encode("ascii"))
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def stable_text_hash(values: Iterable[str]) -> str:
    digest = hashlib.sha256()
    for value in values:
        encoded = str(value).encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "little"))
        digest.update(encoded)
    return digest.hexdigest()


def sha256_file(path: Path, chunk_size: int = 8 << 20) -> str:
    """Return a streaming SHA-256 fingerprint without loading a file at once."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(int(chunk_size)), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_source_npz(
    run_dir: Path,
    project_root: Path,
    explicit_npz: str = "",
) -> Path:
    candidates: list[Path] = []
    if explicit_npz:
        candidates.append(translate_recorded_path(explicit_npz))
    config_path = run_dir / "experiment_config.json"
    if config_path.exists():
        config = json.loads(config_path.read_text(encoding="utf-8"))
        recorded = str(config.get("npz_path", "")).strip()
        if recorded:
            candidates.append(translate_recorded_path(recorded))
        metadata_path = str(
            config.get("checkpoint_metadata", {}).get("uschad_npz_path", "")
        ).strip()
        if metadata_path:
            metadata_candidate = translate_recorded_path(metadata_path)
            if not metadata_candidate.is_absolute():
                metadata_candidate = project_root / metadata_candidate
            candidates.append(metadata_candidate)
    for candidate in candidates:
        candidate = candidate.resolve()
        if candidate.exists():
            return candidate
    rendered = "\n".join(f"  - {candidate}" for candidate in candidates)
    raise FileNotFoundError(
        "Could not resolve the source USC-HAD NPZ. Checked:\n" + rendered
    )


class SourceSignalRepository:
    """Read raw sensor samples for physical trials without overlap duplication."""

    def __init__(self, npz_path: Path, source_consistency_atol: float = 1e-3):
        self.npz_path = Path(npz_path).resolve()
        self.source_consistency_atol = float(source_consistency_atol)
        if self.source_consistency_atol < 0:
            raise ValueError("source_consistency_atol must be non-negative.")
        required = {
            "windows",
            "labels",
            "subject_ids",
            "trial_numbers",
            "trial_global_ids",
            "window_start_indices",
            "file_paths",
            "activity_names",
            "mean",
            "std",
        }
        with np.load(self.npz_path, allow_pickle=True) as data:
            missing = required - set(data.files)
            if missing:
                raise RuntimeError(f"Source NPZ is missing fields: {sorted(missing)}")
            self.windows = np.asarray(data["windows"], dtype=np.float32)
            self.labels = np.asarray(data["labels"], dtype=np.int64)
            self.subject_ids = np.asarray(data["subject_ids"], dtype=np.int64)
            self.trial_numbers = np.asarray(data["trial_numbers"], dtype=np.int64)
            self.trial_ids = np.asarray(data["trial_global_ids"], dtype=np.int64)
            self.window_starts = np.asarray(
                data["window_start_indices"], dtype=np.int64
            )
            self.file_paths = np.asarray(data["file_paths"], dtype=object)
            self.activity_names_raw = np.asarray(
                data["activity_names"], dtype=object
            )
            self.stored_mean = np.asarray(data["mean"], dtype=np.float32)
            self.stored_std = np.asarray(data["std"], dtype=np.float32)
            self.channel_names = (
                np.asarray(data["channel_names"], dtype=object)
                if "channel_names" in data.files
                else np.asarray(
                    [
                        "acc_x",
                        "acc_y",
                        "acc_z",
                        "gyro_x",
                        "gyro_y",
                        "gyro_z",
                    ],
                    dtype=object,
                )
            )
        if self.windows.ndim != 3 or self.windows.shape[1] < 6:
            raise RuntimeError(
                f"Expected source windows [N,>=6,T], got {self.windows.shape}."
            )
        if self.channel_names[:6].tolist() != [
            "acc_x",
            "acc_y",
            "acc_z",
            "gyro_x",
            "gyro_y",
            "gyro_z",
        ]:
            raise RuntimeError(
                "Unexpected USC-HAD channel order: "
                f"{self.channel_names[:6].tolist()}"
            )
        window_aligned = {
            "labels": self.labels,
            "subject_ids": self.subject_ids,
            "trial_numbers": self.trial_numbers,
            "trial_global_ids": self.trial_ids,
            "window_start_indices": self.window_starts,
            "file_paths": self.file_paths,
            "activity_names": self.activity_names_raw,
        }
        invalid_lengths = {
            name: tuple(np.asarray(values).shape)
            for name, values in window_aligned.items()
            if np.asarray(values).ndim != 1 or len(values) != len(self.windows)
        }
        if invalid_lengths:
            raise RuntimeError(
                "Source NPZ window-aligned fields have invalid shapes: "
                f"{invalid_lengths}."
            )
        self.window_size = int(self.windows.shape[-1])
        self.npz_sha256 = sha256_file(self.npz_path)
        self.dataset_semantic_hash = stable_text_hash(
            [
                stable_array_hash(
                    self.labels,
                    self.subject_ids,
                    self.trial_numbers,
                    self.trial_ids,
                    self.window_starts,
                    self.stored_mean,
                    self.stored_std,
                ),
                stable_text_hash(str(value) for value in self.activity_names_raw),
                stable_text_hash(str(value) for value in self.channel_names),
            ]
        )
        self._trial_indices = {
            int(trial_id): np.flatnonzero(self.trial_ids == trial_id)
            for trial_id in np.unique(self.trial_ids)
        }
        self._sensor_cache: dict[int, np.ndarray] = {}
        self._sensor_source_mode: dict[int, str] = {}
        self._sensor_consistency_max_abs: dict[int, Optional[float]] = {}
        self._sensor_consistency_mean_abs: dict[int, Optional[float]] = {}
        self._sensor_hash: dict[int, str] = {}

    def metadata(self, trial_id: int) -> dict:
        indices = self._trial_indices.get(int(trial_id))
        if indices is None or len(indices) == 0:
            raise KeyError(f"Unknown trial_global_id {trial_id}.")
        result = {}
        for key, values in [
            ("label", self.labels),
            ("subject_id", self.subject_ids),
            ("trial_number", self.trial_numbers),
        ]:
            unique = np.unique(values[indices])
            if len(unique) != 1:
                raise RuntimeError(
                    f"Trial {trial_id} has inconsistent {key}: {unique.tolist()}."
                )
            result[key] = int(unique[0])
        names = sorted(set(str(value) for value in self.activity_names_raw[indices]))
        if len(names) != 1:
            raise RuntimeError(
                f"Trial {trial_id} has inconsistent activity names: {names}."
            )
        paths = sorted(set(str(value) for value in self.file_paths[indices]))
        if len(paths) != 1:
            raise RuntimeError(f"Trial {trial_id} has inconsistent paths: {paths}.")
        result["activity_name"] = names[0]
        result["source_path"] = paths[0]
        result["visible_end_sample"] = int(
            np.max(self.window_starts[indices]) + self.window_size
        )
        return result

    def trial_window_starts(self, trial_id: int) -> tuple[int, ...]:
        indices = self._trial_indices.get(int(trial_id))
        if indices is None or len(indices) == 0:
            raise KeyError(f"Unknown trial_global_id {trial_id}.")
        starts = np.sort(self.window_starts[indices], kind="stable")
        return tuple(int(value) for value in starts)

    def trial_window_global_indices(self, trial_id: int) -> tuple[int, ...]:
        indices = self._trial_indices.get(int(trial_id))
        if indices is None or len(indices) == 0:
            raise KeyError(f"Unknown trial_global_id {trial_id}.")
        order = np.argsort(self.window_starts[indices], kind="stable")
        return tuple(int(value) for value in indices[order])

    def _load_mat_signal(self, source_path: str, visible_end: int) -> Optional[np.ndarray]:
        path = translate_recorded_path(source_path)
        if not path.exists():
            return None
        try:
            from scipy.io import loadmat
        except ImportError:
            return None
        payload = loadmat(path, variable_names=["sensor_readings"])
        if "sensor_readings" not in payload:
            raise RuntimeError(f"MAT file has no sensor_readings: {path}")
        sensor = np.asarray(payload["sensor_readings"], dtype=np.float32)
        if sensor.ndim != 2 or sensor.shape[1] < 6:
            raise RuntimeError(f"Invalid sensor_readings shape in {path}: {sensor.shape}")
        if len(sensor) < int(visible_end):
            raise RuntimeError(
                f"Raw trial {path} is shorter than its NPZ window coverage: "
                f"{len(sensor)} < {visible_end}."
            )
        return sensor[: int(visible_end), :6].T.copy()

    def _reconstruct_from_windows(self, trial_id: int) -> np.ndarray:
        indices = self._trial_indices[int(trial_id)]
        order = np.argsort(self.window_starts[indices], kind="stable")
        indices = indices[order]
        starts = self.window_starts[indices]
        raw = (
            self.windows[indices] * self.stored_std + self.stored_mean
        ).astype(np.float32)
        end = int(starts[-1] + self.window_size)
        accumulator = np.zeros((raw.shape[1], end), dtype=np.float64)
        counts = np.zeros(end, dtype=np.int32)
        for window, start in zip(raw, starts):
            begin = int(start)
            finish = begin + self.window_size
            accumulator[:, begin:finish] += window
            counts[begin:finish] += 1
        if np.any(counts == 0):
            raise RuntimeError(
                f"Trial {trial_id} has gaps in reconstructed NPZ coverage."
            )
        return (accumulator / counts[None, :]).astype(np.float32)

    def sensor(self, trial_id: int) -> np.ndarray:
        trial_id = int(trial_id)
        cached = self._sensor_cache.get(trial_id)
        if cached is not None:
            return cached
        metadata = self.metadata(trial_id)
        mat_sensor = self._load_mat_signal(
            metadata["source_path"], metadata["visible_end_sample"]
        )
        reconstructed = self._reconstruct_from_windows(trial_id)
        if mat_sensor is None:
            sensor = reconstructed
            source_mode = "npz_inverse_overlap_deduplicated"
            maximum_error = None
            mean_error = None
        else:
            if mat_sensor.shape != reconstructed.shape:
                raise RuntimeError(
                    f"MAT/NPZ visible-span shapes differ for trial {trial_id}: "
                    f"{mat_sensor.shape} != {reconstructed.shape}."
                )
            absolute_error = np.abs(
                mat_sensor.astype(np.float64) - reconstructed.astype(np.float64)
            )
            maximum_error = float(np.max(absolute_error))
            mean_error = float(np.mean(absolute_error))
            if maximum_error > self.source_consistency_atol:
                raise RuntimeError(
                    f"MAT/NPZ source mismatch for trial {trial_id}: max_abs="
                    f"{maximum_error:.9g} exceeds tolerance "
                    f"{self.source_consistency_atol:.9g}."
                )
            sensor = mat_sensor
            source_mode = "original_mat_visible_span"
        if not np.all(np.isfinite(sensor)):
            raise RuntimeError(f"Trial {trial_id} raw sensor signal is non-finite.")
        self._sensor_cache[trial_id] = sensor
        self._sensor_source_mode[trial_id] = source_mode
        self._sensor_consistency_max_abs[trial_id] = maximum_error
        self._sensor_consistency_mean_abs[trial_id] = mean_error
        self._sensor_hash[trial_id] = stable_array_hash(sensor)
        return sensor

    def source_mode_counts(self, trial_ids: Iterable[int]) -> dict[str, int]:
        counts: dict[str, int] = {}
        for trial_id in trial_ids:
            self.sensor(int(trial_id))
            mode = self._sensor_source_mode[int(trial_id)]
            counts[mode] = counts.get(mode, 0) + 1
        return dict(sorted(counts.items()))

    def source_audit(self, trial_ids: Iterable[int]) -> dict:
        """Describe the exact raw source used and its MAT/NPZ agreement."""
        ids = sorted(set(int(value) for value in trial_ids))
        counts = self.source_mode_counts(ids)
        maximum_errors = [
            self._sensor_consistency_max_abs[trial_id]
            for trial_id in ids
            if self._sensor_consistency_max_abs[trial_id] is not None
        ]
        mean_errors = [
            self._sensor_consistency_mean_abs[trial_id]
            for trial_id in ids
            if self._sensor_consistency_mean_abs[trial_id] is not None
        ]
        return {
            "source_npz": str(self.npz_path),
            "source_npz_size_bytes": int(self.npz_path.stat().st_size),
            "source_npz_sha256": self.npz_sha256,
            "dataset_semantic_hash": self.dataset_semantic_hash,
            "trial_count": len(ids),
            "source_mode_counts": counts,
            "uniform_source_mode": len(counts) == 1,
            "mat_npz_compared_trial_count": len(maximum_errors),
            "mat_npz_max_abs_error": (
                float(max(maximum_errors)) if maximum_errors else None
            ),
            "mat_npz_mean_abs_error": (
                float(np.mean(mean_errors)) if mean_errors else None
            ),
            "mat_npz_consistency_atol": self.source_consistency_atol,
            "raw_signal_manifest_hash": stable_text_hash(
                f"{trial_id}|{self._sensor_hash[trial_id]}" for trial_id in ids
            ),
        }


@dataclass(frozen=True)
class TrajectoryRun:
    token: int
    start_sample: int
    end_sample_exclusive: int
    duration_seconds: float
    state: np.ndarray


@dataclass(frozen=True)
class TrialTrajectory:
    trial_global_id: int
    trial_key: str
    split_role: int
    subject_id: int
    activity_label: int
    activity_name: str
    trial_number: int
    runs: tuple[TrajectoryRun, ...]

    @property
    def tokens(self) -> tuple[int, ...]:
        return tuple(int(run.token) for run in self.runs)


@dataclass(frozen=True)
class RobustScaler:
    center: np.ndarray
    scale: np.ndarray

    def transform(self, value: np.ndarray) -> np.ndarray:
        array = np.asarray(value, dtype=np.float32)
        return ((array - self.center) / self.scale).astype(np.float32)


def state_features(sensor: np.ndarray) -> np.ndarray:
    sensor = np.asarray(sensor, dtype=np.float64)
    if sensor.ndim != 2 or sensor.shape[0] < 6 or sensor.shape[1] == 0:
        raise ValueError(f"Expected non-empty sensor [>=6,T], got {sensor.shape}.")
    acceleration = sensor[:3]
    gyroscope = sensor[3:6]
    mean_acceleration = np.mean(acceleration, axis=1)
    gravity_norm = max(float(np.linalg.norm(mean_acceleration)), EPS)
    gravity_direction = mean_acceleration / gravity_norm
    std_acceleration = np.std(acceleration, axis=1)
    std_gyroscope = np.std(gyroscope, axis=1)
    result = np.concatenate(
        [
            mean_acceleration,
            gravity_direction,
            np.log1p(np.maximum(std_acceleration, 0.0)),
            np.log1p(np.maximum(std_gyroscope, 0.0)),
        ]
    ).astype(np.float32)
    if result.shape != (len(STATE_FEATURE_NAMES),) or not np.all(np.isfinite(result)):
        raise RuntimeError("State feature extraction produced an invalid vector.")
    return result


def _weighted_quantile(
    values: np.ndarray, weights: np.ndarray, quantile: float
) -> float:
    order = np.argsort(values, kind="stable")
    ordered_values = np.asarray(values, dtype=np.float64)[order]
    ordered_weights = np.asarray(weights, dtype=np.float64)[order]
    total = float(np.sum(ordered_weights))
    if total <= 0:
        raise ValueError("Weighted quantile needs positive total weight.")
    cumulative = np.cumsum(ordered_weights) - 0.5 * ordered_weights
    cumulative /= total
    return float(
        np.interp(float(quantile), cumulative, ordered_values, left=ordered_values[0], right=ordered_values[-1])
    )


def fit_state_scaler(trials: Sequence[TrialTrajectory]) -> RobustScaler:
    values = []
    weights = []
    for trial in trials:
        if not trial.runs:
            continue
        weight = 1.0 / len(trial.runs)
        for run in trial.runs:
            values.append(np.asarray(run.state, dtype=np.float64))
            weights.append(weight)
    matrix = np.asarray(values, dtype=np.float64)
    sample_weights = np.asarray(weights, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[1] != len(STATE_FEATURE_NAMES):
        raise ValueError("No valid fit-run state features were available.")
    median = np.asarray(
        [
            _weighted_quantile(matrix[:, column], sample_weights, 0.5)
            for column in range(matrix.shape[1])
        ],
        dtype=np.float32,
    )
    lower = np.asarray(
        [
            _weighted_quantile(matrix[:, column], sample_weights, 0.25)
            for column in range(matrix.shape[1])
        ],
        dtype=np.float32,
    )
    upper = np.asarray(
        [
            _weighted_quantile(matrix[:, column], sample_weights, 0.75)
            for column in range(matrix.shape[1])
        ],
        dtype=np.float32,
    )
    scale = np.maximum(upper - lower, np.float32(1e-6))
    return RobustScaler(center=median, scale=scale)


def transform_trial_states(
    trials: Sequence[TrialTrajectory], scaler: RobustScaler
) -> list[TrialTrajectory]:
    transformed = []
    for trial in trials:
        runs = tuple(
            replace(run, state=scaler.transform(run.state)) for run in trial.runs
        )
        transformed.append(replace(trial, runs=runs))
    return transformed


def _pairwise_p95(
    matrix: np.ndarray,
    sample_weights: Optional[np.ndarray] = None,
    default: float = 1.0,
) -> float:
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.ndim != 2 or len(matrix) < 2:
        return float(default)
    differences = matrix[:, None, :] - matrix[None, :, :]
    distances = np.linalg.norm(differences, axis=2) / math.sqrt(matrix.shape[1])
    pair_indices = np.triu_indices(len(matrix), k=1)
    values = distances[pair_indices]
    valid = np.isfinite(values) & (values > 0)
    positive = values[valid]
    if len(positive) == 0:
        return float(default)
    if sample_weights is None:
        percentile = float(np.percentile(positive, 95.0))
    else:
        weights = np.asarray(sample_weights, dtype=np.float64)
        if weights.shape != (len(matrix),):
            raise ValueError("Pairwise scale weights do not match the matrix.")
        pair_weights = (weights[pair_indices[0]] * weights[pair_indices[1]])[valid]
        percentile = _weighted_quantile(positive, pair_weights, 0.95)
    return max(percentile, 1e-6)


def fit_local_scales(trials: Sequence[TrialTrajectory]) -> dict[str, float]:
    states = []
    duration_values = []
    run_weights = []
    for trial in trials:
        if not trial.runs:
            continue
        weight = 1.0 / len(trial.runs)
        for run in trial.runs:
            states.append(run.state)
            duration_values.append([math.log1p(max(run.duration_seconds, 0.0))])
            run_weights.append(weight)
    states = np.asarray(states, dtype=np.float32)
    duration_values = np.asarray(duration_values, dtype=np.float32)
    run_weights = np.asarray(run_weights, dtype=np.float64)
    return {
        "state_distance_p95": _pairwise_p95(states, run_weights),
        "log_duration_difference_p95": _pairwise_p95(
            duration_values, run_weights
        ),
        "scale_weighting": "each fit trial contributes total run weight 1",
    }


def codebook_dynamic_cost(centers: np.ndarray) -> np.ndarray:
    centers = np.asarray(centers, dtype=np.float64)
    if centers.ndim != 2 or len(centers) < 2:
        raise ValueError(f"Codebook centers must be [K,D], got {centers.shape}.")
    norms = np.linalg.norm(centers, axis=1, keepdims=True)
    normalized = centers / np.maximum(norms, EPS)
    similarity = np.clip(normalized @ normalized.T, -1.0, 1.0)
    cost = np.clip((1.0 - similarity) / 2.0, 0.0, 1.0)
    cost = 0.5 * (cost + cost.T)
    np.fill_diagonal(cost, 0.0)
    return cost.astype(np.float32)


def _read_expected_eval_tokens(run_dir: Path) -> dict[int, tuple[int, ...]]:
    path = run_dir / "trial_primitive_sequences.jsonl"
    if not path.exists():
        raise FileNotFoundError(path)
    result = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            trial_id = int(record["trial_global_id_within_npz"])
            tokens = tuple(
                int(value)
                for value in record["sequence_variants"]["full"]["rle_tokens"]
            )
            result[trial_id] = tokens
    return result


def build_trajectories(
    run_dir: Path,
    source: SourceSignalRepository,
    sample_rate_hz: float,
    old_class_count: int,
) -> tuple[list[TrialTrajectory], list[TrialTrajectory], dict[str, str]]:
    segment_path = run_dir / "segment_embeddings_and_tokens.npz"
    codebook_path = run_dir / "primitive_codebook.npz"
    with np.load(segment_path, allow_pickle=False) as data:
        required = {
            "split_role",
            "trial_global_ids",
            "partition_start_samples",
            "partition_end_samples_exclusive",
            "primitive_tokens",
        }
        missing = required - set(data.files)
        if missing:
            raise RuntimeError(f"Segment NPZ is missing fields: {sorted(missing)}")
        split_role = np.asarray(data["split_role"], dtype=np.int8)
        trial_ids = np.asarray(data["trial_global_ids"], dtype=np.int64)
        starts = np.asarray(data["partition_start_samples"], dtype=np.int64)
        ends = np.asarray(data["partition_end_samples_exclusive"], dtype=np.int64)
        tokens = np.asarray(data["primitive_tokens"], dtype=np.int64)
        row_arrays = {
            name: np.asarray(data[name])
            for name in data.files
            if np.asarray(data[name]).ndim > 0
        }
    with np.load(codebook_path, allow_pickle=False) as data:
        centers = np.asarray(data["centers"], dtype=np.float32)

    core_arrays = {
        "split_role": split_role,
        "trial_global_ids": trial_ids,
        "partition_start_samples": starts,
        "partition_end_samples_exclusive": ends,
        "primitive_tokens": tokens,
    }
    core_lengths = {name: len(value) for name, value in core_arrays.items()}
    if any(np.asarray(value).ndim != 1 for value in core_arrays.values()):
        raise RuntimeError("Segment core arrays must all be one-dimensional.")
    if len(set(core_lengths.values())) != 1 or not next(iter(core_lengths.values())):
        raise RuntimeError(
            f"Segment core arrays must have the same non-zero length: {core_lengths}."
        )
    row_count = len(trial_ids)
    inconsistent_rows = {
        name: tuple(value.shape)
        for name, value in row_arrays.items()
        if len(value) != row_count
    }
    if inconsistent_rows:
        raise RuntimeError(
            "Segment row arrays disagree on their first dimension: "
            f"{inconsistent_rows}."
        )
    if centers.ndim != 2 or not np.all(np.isfinite(centers)):
        raise RuntimeError(f"Invalid codebook centers: shape={centers.shape}.")
    if np.any(tokens < 0) or np.any(tokens >= len(centers)):
        raise RuntimeError("Segment primitive token lies outside the codebook.")
    if np.any(~np.isin(split_role, np.asarray([0, 1], dtype=np.int8))):
        raise RuntimeError("Segment split_role contains a value outside {0,1}.")
    if np.any(starts < 0):
        raise RuntimeError("Segment partitions contain a negative start sample.")
    optional_partition_fields = {
        "segment_ids",
        "window_counts",
        "first_global_window_indices",
        "last_global_window_indices",
        "support_start_samples",
        "support_end_samples_exclusive",
    }
    has_partition_metadata = optional_partition_fields <= set(row_arrays)
    if "segment_ids" in row_arrays and not np.array_equal(
        np.asarray(row_arrays["segment_ids"], dtype=np.int64),
        np.arange(row_count, dtype=np.int64),
    ):
        raise RuntimeError("Segment ids are not the canonical contiguous row ids.")

    trajectories: list[TrialTrajectory] = []
    boundary_rows = []
    for trial_id in np.unique(trial_ids):
        indices = np.flatnonzero(trial_ids == trial_id)
        order = np.argsort(starts[indices], kind="stable")
        indices = indices[order]
        trial_starts = starts[indices]
        trial_ends = ends[indices]
        trial_tokens = tokens[indices]
        roles = np.unique(split_role[indices])
        if len(roles) != 1 or int(roles[0]) not in (0, 1):
            raise RuntimeError(f"Trial {trial_id} has invalid split roles {roles}.")
        if np.any(trial_ends <= trial_starts):
            raise RuntimeError(f"Trial {trial_id} has empty segment partitions.")
        if int(trial_starts[0]) != 0 or np.any(trial_starts[1:] != trial_ends[:-1]):
            raise RuntimeError(f"Trial {trial_id} partitions are not contiguous.")
        sensor = source.sensor(int(trial_id))
        if int(trial_ends[-1]) != sensor.shape[1]:
            raise RuntimeError(
                f"Trial {trial_id} partition does not cover the complete visible "
                f"raw signal: last_end={trial_ends[-1]}, "
                f"visible_samples={sensor.shape[1]}."
            )
        if has_partition_metadata:
            window_counts = np.asarray(
                row_arrays["window_counts"], dtype=np.int64
            )[indices]
            first_global = np.asarray(
                row_arrays["first_global_window_indices"], dtype=np.int64
            )[indices]
            last_global = np.asarray(
                row_arrays["last_global_window_indices"], dtype=np.int64
            )[indices]
            support_starts = np.asarray(
                row_arrays["support_start_samples"], dtype=np.int64
            )[indices]
            support_ends = np.asarray(
                row_arrays["support_end_samples_exclusive"], dtype=np.int64
            )[indices]
            global_windows = source.trial_window_global_indices(int(trial_id))
            local_position = {
                global_index: position
                for position, global_index in enumerate(global_windows)
            }
            if any(
                int(value) not in local_position
                for value in np.r_[first_global, last_global]
            ):
                raise RuntimeError(
                    f"Trial {trial_id} segment window index crosses trials."
                )
            first_local = np.asarray(
                [local_position[int(value)] for value in first_global],
                dtype=np.int64,
            )
            last_local = np.asarray(
                [local_position[int(value)] for value in last_global],
                dtype=np.int64,
            )
            if (
                int(first_local[0]) != 0
                or int(last_local[-1]) != len(global_windows) - 1
                or np.any(first_local[1:] != last_local[:-1] + 1)
                or np.any(window_counts != last_local - first_local + 1)
            ):
                raise RuntimeError(
                    f"Trial {trial_id} segment window membership has a gap, "
                    "overlap, or invalid count."
                )
            trial_window_starts = np.asarray(
                source.trial_window_starts(int(trial_id)), dtype=np.int64
            )
            expected_support_starts = trial_window_starts[first_local]
            expected_support_ends = (
                trial_window_starts[last_local] + int(source.window_size)
            )
            if not np.array_equal(support_starts, expected_support_starts) or not np.array_equal(
                support_ends, expected_support_ends
            ):
                raise RuntimeError(
                    f"Trial {trial_id} segment support spans disagree with the "
                    "source window grid."
                )
            if np.any(trial_starts < support_starts) or np.any(
                trial_ends > support_ends
            ):
                raise RuntimeError(
                    f"Trial {trial_id} partition lies outside segment support."
                )
            window_centers = trial_window_starts.astype(np.float64) + float(
                source.window_size
            ) / 2.0
            expected_boundaries = [int(trial_window_starts[0])]
            for boundary in first_local[1:]:
                midpoint = 0.5 * (
                    window_centers[int(boundary) - 1]
                    + window_centers[int(boundary)]
                )
                expected_boundaries.append(int(round(midpoint)))
            expected_boundaries.append(
                int(trial_window_starts[-1] + int(source.window_size))
            )
            expected_boundaries = np.asarray(expected_boundaries, dtype=np.int64)
            actual_boundaries = np.r_[trial_starts, trial_ends[-1]]
            if not np.array_equal(actual_boundaries, expected_boundaries):
                raise RuntimeError(
                    f"Trial {trial_id} partition boundaries disagree with the "
                    "window-center midpoint rule."
                )
        merged: list[tuple[int, int, int]] = []
        for token, begin, end in zip(trial_tokens, trial_starts, trial_ends):
            token = int(token)
            begin = int(begin)
            end = int(end)
            boundary_rows.append((int(trial_id), int(roles[0]), token, begin, end))
            if merged and merged[-1][0] == token:
                previous = merged[-1]
                if previous[2] != begin:
                    raise RuntimeError("Merged equal-token segments are not adjacent.")
                merged[-1] = (token, previous[1], end)
            else:
                merged.append((token, begin, end))
        runs = tuple(
            TrajectoryRun(
                token=token,
                start_sample=begin,
                end_sample_exclusive=end,
                duration_seconds=float((end - begin) / float(sample_rate_hz)),
                state=state_features(sensor[:, begin:end]),
            )
            for token, begin, end in merged
        )
        metadata = source.metadata(int(trial_id))
        subject = int(metadata["subject_id"])
        label = int(metadata["label"])
        trial_number = int(metadata["trial_number"])
        role = int(roles[0])
        if role == 0 and label >= int(old_class_count):
            raise RuntimeError(
                f"Fit trajectory {trial_id} contains non-old class label {label}."
            )
        trajectories.append(
            TrialTrajectory(
                trial_global_id=int(trial_id),
                trial_key=f"S{subject:02d}-A{label + 1:02d}-T{trial_number}",
                split_role=role,
                subject_id=subject,
                activity_label=label,
                activity_name=str(metadata["activity_name"]),
                trial_number=trial_number,
                runs=runs,
            )
        )
    trajectories.sort(
        key=lambda trial: (
            trial.split_role,
            trial.subject_id,
            trial.activity_label,
            trial.trial_number,
        )
    )
    fit_trials = [trial for trial in trajectories if trial.split_role == 0]
    eval_trials = [trial for trial in trajectories if trial.split_role == 1]
    expected_tokens = _read_expected_eval_tokens(run_dir)
    actual_ids = {trial.trial_global_id for trial in eval_trials}
    if actual_ids != set(expected_tokens):
        raise RuntimeError("Eval trial ids differ between segment NPZ and JSONL.")
    mismatches = [
        trial.trial_key
        for trial in eval_trials
        if trial.tokens != expected_tokens[trial.trial_global_id]
    ]
    if mismatches:
        raise RuntimeError(
            "Segment-derived RLE tokens differ from saved hard baseline: "
            f"{mismatches[:5]}"
        )
    hashes = {
        "trial_grid_hash": stable_text_hash(
            (
                f"{trial.trial_global_id}|{trial.subject_id}|"
                f"{trial.activity_label}|{trial.trial_number}|"
                f"{','.join(map(str, source.trial_window_starts(trial.trial_global_id)))}"
            )
            for trial in eval_trials
        ),
        "rle_token_hash": stable_text_hash(
            f"{trial.trial_global_id}|{','.join(map(str, trial.tokens))}"
            for trial in eval_trials
        ),
        "codebook_center_hash": stable_array_hash(centers),
        "segment_boundary_hash": stable_array_hash(
            np.asarray(boundary_rows, dtype=np.int64)
        ),
    }
    return fit_trials, eval_trials, hashes


def _neighbor_cost(
    left: Sequence[TrajectoryRun],
    left_index: int,
    right: Sequence[TrajectoryRun],
    right_index: int,
    offset: int,
    dynamic_cost: np.ndarray,
) -> float:
    left_position = left_index + int(offset)
    right_position = right_index + int(offset)
    left_valid = 0 <= left_position < len(left)
    right_valid = 0 <= right_position < len(right)
    if not left_valid and not right_valid:
        return 0.0
    if left_valid != right_valid:
        return 1.0
    return float(
        dynamic_cost[left[left_position].token, right[right_position].token]
    )


def local_substitution_cost(
    group: str,
    left_runs: Sequence[TrajectoryRun],
    left_index: int,
    right_runs: Sequence[TrajectoryRun],
    right_index: int,
    dynamic_cost: np.ndarray,
    scales: dict[str, float],
    state_weight: float,
    context_weight: float,
    duration_weight: float,
) -> float:
    left = left_runs[left_index]
    right = right_runs[right_index]
    if group == "g1_hard_rle":
        return float(left.token != right.token)
    dynamic = float(dynamic_cost[left.token, right.token])
    if group == "g2_dynamic_soft":
        return dynamic
    state = float(
        np.linalg.norm(left.state - right.state)
        / math.sqrt(len(STATE_FEATURE_NAMES))
    )
    state /= max(float(scales["state_distance_p95"]), 1e-6)
    state = float(np.clip(state, 0.0, 1.0))
    base = (1.0 - float(state_weight)) * dynamic + float(state_weight) * state
    if group == "g3_dynamic_state":
        return float(np.clip(base, 0.0, 1.0))
    if group != "g4_dynamic_state_duration_context":
        raise ValueError(f"Unknown ablation group {group!r}.")
    duration = abs(
        math.log1p(max(left.duration_seconds, 0.0))
        - math.log1p(max(right.duration_seconds, 0.0))
    )
    duration /= max(float(scales["log_duration_difference_p95"]), 1e-6)
    duration = float(np.clip(duration, 0.0, 1.0))
    predecessor = _neighbor_cost(
        left_runs, left_index, right_runs, right_index, -1, dynamic_cost
    )
    successor = _neighbor_cost(
        left_runs, left_index, right_runs, right_index, 1, dynamic_cost
    )
    context = 0.5 * (predecessor + successor)
    remaining = 1.0 - float(context_weight) - float(duration_weight)
    if remaining < 0:
        raise ValueError("context_weight + duration_weight must not exceed 1.")
    result = remaining * base
    result += float(context_weight) * context
    result += float(duration_weight) * duration
    return float(np.clip(result, 0.0, 1.0))


def trajectory_distance_matrix(
    trials: Sequence[TrialTrajectory],
    group: str,
    dynamic_cost: np.ndarray,
    scales: dict[str, float],
    state_weight: float,
    context_weight: float,
    duration_weight: float,
) -> np.ndarray:
    size = len(trials)
    matrix = np.zeros((size, size), dtype=np.float32)
    for left_index in range(size):
        left_runs = trials[left_index].runs
        for right_index in range(left_index + 1, size):
            right_runs = trials[right_index].runs
            value = normalized_weighted_levenshtein(
                [run.token for run in left_runs],
                [run.token for run in right_runs],
                lambda _left_token, _right_token, left_position, right_position: local_substitution_cost(
                    group,
                    left_runs,
                    left_position,
                    right_runs,
                    right_position,
                    dynamic_cost,
                    scales,
                    state_weight,
                    context_weight,
                    duration_weight,
                ),
            )
            matrix[left_index, right_index] = value
            matrix[right_index, left_index] = value
    return matrix


def total_duration_distance_matrix(
    trials: Sequence[TrialTrajectory],
) -> np.ndarray:
    """Order-free shortcut control using visible total trial duration only."""
    values = np.log1p(
        np.asarray(
            [
                sum(run.duration_seconds for run in trial.runs)
                for trial in trials
            ],
            dtype=np.float64,
        )
    )
    scale = max(float(np.max(values) - np.min(values)), EPS)
    return (np.abs(values[:, None] - values[None, :]) / scale).astype(np.float32)


def tie_aware_confusion(
    distance_matrix: np.ndarray,
    labels: Sequence[int],
    subjects: Sequence[int],
    class_ids: Optional[Sequence[int]] = None,
) -> dict:
    matrix = np.asarray(distance_matrix, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    subjects = np.asarray(subjects, dtype=np.int64)
    if matrix.shape != (len(labels), len(labels)):
        raise ValueError("Distance matrix and label vector sizes differ.")
    if class_ids is None:
        classes = sorted(int(value) for value in np.unique(labels))
    else:
        classes = sorted(int(value) for value in class_ids)
    class_to_position = {label: position for position, label in enumerate(classes)}
    selected = np.isin(labels, np.asarray(classes, dtype=np.int64))
    confusion = np.zeros((len(classes), len(classes)), dtype=np.float64)
    tie_counts = []
    minimum_distances = []
    query_indices = []
    prediction_probabilities = np.full(
        (len(labels), len(classes)), np.nan, dtype=np.float64
    )
    for query in np.flatnonzero(selected):
        candidates = np.flatnonzero(selected & (subjects != subjects[query]))
        if len(candidates) == 0:
            continue
        distances = matrix[query, candidates]
        minimum = float(np.min(distances))
        tied = candidates[np.isclose(distances, minimum, rtol=0.0, atol=1e-12)]
        weights = np.full(len(tied), 1.0 / len(tied), dtype=np.float64)
        true_position = class_to_position[int(labels[query])]
        probabilities = np.zeros(len(classes), dtype=np.float64)
        for candidate, weight in zip(tied, weights):
            predicted_position = class_to_position[int(labels[candidate])]
            confusion[true_position, predicted_position] += float(weight)
            probabilities[predicted_position] += float(weight)
        prediction_probabilities[query] = probabilities
        query_indices.append(int(query))
        tie_counts.append(int(len(tied)))
        minimum_distances.append(minimum)
    row_support = confusion.sum(axis=1)
    column_support = confusion.sum(axis=0)
    recall = np.divide(
        np.diag(confusion),
        row_support,
        out=np.full(len(classes), np.nan),
        where=row_support > 0,
    )
    precision = np.divide(
        np.diag(confusion),
        column_support,
        out=np.full(len(classes), np.nan),
        where=column_support > 0,
    )
    f1 = np.divide(
        2.0 * precision * recall,
        precision + recall,
        out=np.zeros(len(classes), dtype=np.float64),
        where=(precision + recall) > 0,
    )
    minimum_array = np.asarray(minimum_distances, dtype=np.float64)
    return {
        "class_ids": classes,
        "query_indices": query_indices,
        "confusion_counts": confusion,
        "confusion_row_normalized": np.divide(
            confusion,
            row_support[:, None],
            out=np.zeros_like(confusion),
            where=row_support[:, None] > 0,
        ),
        "per_class_recall": recall,
        "per_class_precision": precision,
        "per_class_f1": f1,
        "accuracy": float(np.trace(confusion) / max(float(confusion.sum()), EPS)),
        "macro_recall": float(np.nanmean(recall)),
        "macro_f1": float(np.nanmean(f1)),
        "mean_tied_nearest_count": float(np.mean(tie_counts)),
        "median_tied_nearest_count": float(np.median(tie_counts)),
        "zero_min_distance_query_ratio": float(
            np.mean(np.isclose(minimum_array, 0.0, rtol=0.0, atol=1e-12))
        ),
        "mean_minimum_distance": float(np.mean(minimum_array)),
        "prediction_probabilities": prediction_probabilities,
    }


def evaluate_distance_matrix(
    distance_matrix: np.ndarray,
    trials: Sequence[TrialTrajectory],
    old_class_count: int,
) -> dict:
    labels = np.asarray([trial.activity_label for trial in trials], dtype=np.int64)
    subjects = np.asarray([trial.subject_id for trial in trials], dtype=np.int64)
    class_ids = sorted(int(value) for value in np.unique(labels))
    all_association = association_summary(distance_matrix, labels, subjects, class_ids)
    old_ids = [value for value in class_ids if value < int(old_class_count)]
    novel_ids = [value for value in class_ids if value >= int(old_class_count)]
    all_confusion = tie_aware_confusion(distance_matrix, labels, subjects, class_ids)
    old_confusion = tie_aware_confusion(distance_matrix, labels, subjects, old_ids)
    novel_confusion = tie_aware_confusion(distance_matrix, labels, subjects, novel_ids)

    def all_candidate_query_summary(query_class_ids: Sequence[int]) -> dict:
        requested = sorted(int(value) for value in query_class_ids)
        class_positions = {
            int(value): position
            for position, value in enumerate(all_confusion["class_ids"])
        }
        probabilities = np.asarray(
            all_confusion["prediction_probabilities"], dtype=np.float64
        )
        correct_probabilities = []
        per_class = {}
        for class_id in requested:
            indices = np.flatnonzero(labels == class_id)
            position = class_positions[class_id]
            values = probabilities[indices, position]
            values = values[np.isfinite(values)]
            if len(values) == 0:
                continue
            per_class[class_id] = float(np.mean(values))
            correct_probabilities.extend(values.tolist())
        if not correct_probabilities:
            raise ValueError("No all-candidate queries were available for the subset.")
        return {
            "query_class_ids": requested,
            "candidate_class_ids": list(all_confusion["class_ids"]),
            "query_count": int(len(correct_probabilities)),
            "accuracy": float(np.mean(correct_probabilities)),
            "macro_recall": float(np.mean(list(per_class.values()))),
            "per_class_recall": per_class,
        }

    name_to_id = {
        trial.activity_name.strip().lower(): trial.activity_label for trial in trials
    }
    low_dynamic_names = ["sitting", "standing", "elevator up", "elevator down"]
    low_dynamic_ids = [
        name_to_id[name] for name in low_dynamic_names if name in name_to_id
    ]
    sit_stand_ids = [
        name_to_id[name] for name in ["sitting", "standing"] if name in name_to_id
    ]
    low_dynamic = tie_aware_confusion(
        distance_matrix, labels, subjects, low_dynamic_ids
    )
    sit_stand = tie_aware_confusion(
        distance_matrix, labels, subjects, sit_stand_ids
    )
    activity_matrix = activity_distance_matrix(
        distance_matrix, labels, subjects, class_ids
    )
    return {
        "all_association": all_association,
        "all_confusion": all_confusion,
        "old_confusion": old_confusion,
        "novel_confusion": novel_confusion,
        "old_queries_all_candidates": all_candidate_query_summary(old_ids),
        "novel_queries_all_candidates": all_candidate_query_summary(novel_ids),
        "low_dynamic_confusion": low_dynamic,
        "sit_stand_binary_confusion": sit_stand,
        "activity_distance_matrix": activity_matrix,
    }


def shuffled_trajectories(
    trials: Sequence[TrialTrajectory],
    rng: np.random.Generator,
    mode: str,
) -> list[TrialTrajectory]:
    if mode == "total_duration":
        totals = np.asarray(
            [sum(run.duration_seconds for run in trial.runs) for trial in trials],
            dtype=np.float64,
        )
        shuffled_totals = totals.copy()
        subjects = np.asarray(
            [trial.subject_id for trial in trials], dtype=np.int64
        )
        for subject_id in np.unique(subjects):
            indices = np.flatnonzero(subjects == subject_id)
            shuffled_totals[indices] = rng.permutation(totals[indices])
        shuffled = []
        for trial, source_total, target_total in zip(
            trials, totals, shuffled_totals
        ):
            scale = float(target_total / max(float(source_total), EPS))
            runs = tuple(
                replace(run, duration_seconds=float(run.duration_seconds * scale))
                for run in trial.runs
            )
            shuffled.append(replace(trial, runs=runs))
        return shuffled

    shuffled = []
    for trial in trials:
        runs = list(trial.runs)
        if mode == "order":
            if len(runs) > 1:
                shuffled_tokens = shuffle_valid_rle_tokens(
                    [run.token for run in runs], rng
                )
                token_runs = {}
                for token in sorted(set(shuffled_tokens)):
                    candidates = [run for run in runs if run.token == token]
                    permutation = rng.permutation(len(candidates))
                    token_runs[token] = [
                        candidates[int(index)] for index in permutation
                    ]
                runs = [token_runs[int(token)].pop() for token in shuffled_tokens]
                if any(
                    left.token == right.token
                    for left, right in zip(runs, runs[1:])
                ):
                    raise RuntimeError("Order control produced an invalid RLE.")
        elif mode == "duration_alignment":
            if len(runs) > 1:
                durations = [run.duration_seconds for run in runs]
                permuted = rng.permutation(durations)
                runs = [
                    replace(run, duration_seconds=float(duration))
                    for run, duration in zip(runs, permuted)
                ]
        else:
            raise ValueError(f"Unknown shuffle mode {mode!r}.")
        shuffled.append(replace(trial, runs=tuple(runs)))
    return shuffled


def run_controls(
    trials: Sequence[TrialTrajectory],
    observed_accuracy: float,
    dynamic_cost: np.ndarray,
    scales: dict[str, float],
    state_weight: float,
    context_weight: float,
    duration_weight: float,
    shuffles: int,
    seed: int,
) -> dict:
    if int(shuffles) <= 0:
        return {"shuffles": 0}
    labels = np.asarray([trial.activity_label for trial in trials], dtype=np.int64)
    subjects = np.asarray([trial.subject_id for trial in trials], dtype=np.int64)
    rng = np.random.default_rng(int(seed))
    result = {"shuffles": int(shuffles)}
    for mode in ["order", "duration_alignment", "total_duration"]:
        accuracies = []
        for _ in range(int(shuffles)):
            shuffled = shuffled_trajectories(trials, rng, mode)
            matrix = trajectory_distance_matrix(
                shuffled,
                "g4_dynamic_state_duration_context",
                dynamic_cost,
                scales,
                state_weight,
                context_weight,
                duration_weight,
            )
            summary = tie_aware_confusion(matrix, labels, subjects)
            accuracies.append(float(summary["accuracy"]))
        values = np.asarray(accuracies, dtype=np.float64)
        result[mode] = {
            "accuracy_mean": float(np.mean(values)),
            "accuracy_std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
            "observed_minus_shuffled_accuracy": float(
                observed_accuracy - np.mean(values)
            ),
            "accuracy_values": values,
        }
    return result
