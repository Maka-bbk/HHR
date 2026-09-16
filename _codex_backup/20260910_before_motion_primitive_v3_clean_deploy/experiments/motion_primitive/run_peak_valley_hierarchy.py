"""One-fold sample-level peak/valley motion-primitive hierarchy experiment.

This is an isolated feasibility experiment, not a replacement for the Happy-CGCD
learner.  Primitive discovery is fitted on outer-fold training subjects and old
classes only.  The label-aware benchmark protocol builder fixes the Session-2
48/52 trial manifest, but exposes only trial IDs and subject IDs to this
predictor.  The trial readout is fitted on the 48 unlabelled cumulative-online-
train trials and predicts the disjoint 52 test trials.  Scoring activity
identity is joined only after raw predictions are frozen.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import sklearn
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score

from experiments.motion_primitive.core import (  # noqa: E402
    assign_to_codebook,
    inverse_trial_frequency_weights,
    l2_normalize,
)
from experiments.motion_primitive.frozen_hierarchical_readout import (  # noqa: E402
    fit_frozen_coarse_readout,
)
from experiments.motion_primitive.peak_valley_hierarchy import (  # noqa: E402
    ChildPrimitiveBatch,
    CodebookConfig,
    CodebookState,
    ParentCatalogState,
    ParentGateConfig,
    ParentOverlay,
    PeakValleyConfig,
    SegmentedTrial,
    TokenizedTrial,
    TrialSignal,
    _codebook_fit_hash,
    assign_codebook_trials,
    extract_child_feature_batches,
    fit_codebook,
    fit_frequency_parent_catalog,
    fit_matched_random_parent_catalog,
    fit_parent_catalog,
    fit_segmenter,
    matched_random_segmentations,
    overlay_parent_trials,
    save_state_json,
    segment_trials,
)
from experiments.motion_primitive.run_experiment import (  # noqa: E402
    build_frozen_motion_encoder,
    choose_device,
    encode_motion_windows,
    load_torch_checkpoint,
    prepare_primitive_features,
)
from experiments.motion_primitive.run_online_hierarchical_gate_v2 import (  # noqa: E402
    build_session2_manifest_v2,
    _complete_clustering_metrics,
)
from experiments.motion_primitive.run_online_secondary_codebook import (  # noqa: E402
    LabelFreeSourceSignalRepository,
    _truth_maps,
)
from experiments.motion_primitive.trajectory_ablation import (  # noqa: E402
    SourceSignalRepository,
    sha256_file,
)


SCHEMA = "peak_valley_hierarchy_run_v1"
ALLOWED_ARMS = ("E0", "E1", "E2", "E3", "E4")
READOUT_VARIANTS = ("state", "no_state")
CONTROL_SPECS = {
    "C1": {"base_arm": "E2", "control_type": "matched_random_boundaries"},
    "C2": {"base_arm": "E4", "control_type": "matched_random_parents"},
}
CHANNEL_NAMES = ("acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z")
CROSS_SUBJECT_DIAGNOSTIC_SCHEMA = "cross_subject_trajectory_diagnostics_v1"
CROSS_SUBJECT_DIAGNOSTIC_FILES = (
    "cross_subject_trajectory_diagnostics.json",
    "cross_subject_trajectory_diagnostics.csv",
)


@dataclass(frozen=True)
class WindowGrid:
    windows: np.ndarray
    labels: np.ndarray
    subject_ids: np.ndarray
    trial_ids: np.ndarray
    window_indices: np.ndarray
    starts: np.ndarray
    stored_mean: np.ndarray
    stored_std: np.ndarray
    window_size_samples: int = 256
    stride_samples: int = 128


@dataclass(frozen=True)
class PrimitiveTrial:
    """Label-free trajectory passed to all fitting/readout functions."""

    trial_id: int
    subject_id: int
    starts: np.ndarray
    ends: np.ndarray
    child_tokens: np.ndarray
    child_distances: np.ndarray
    child_embeddings: np.ndarray
    child_statistics: np.ndarray
    statistic_names: tuple[str, ...]
    event_kinds: tuple[str, ...]
    parent_overlay: ParentOverlay | None = None

    def validate(self, primitive_num: int) -> "PrimitiveTrial":
        count = len(self.child_tokens)
        if count < 1:
            raise ValueError("A primitive trial must contain at least one child.")
        if self.starts.shape != (count,) or self.ends.shape != (count,):
            raise ValueError("Primitive spans do not match token count.")
        if self.child_distances.shape != (count,):
            raise ValueError("Primitive distances do not match token count.")
        if self.child_embeddings.ndim != 2 or len(self.child_embeddings) != count:
            raise ValueError("Primitive embeddings do not match token count.")
        if self.child_statistics.ndim != 2 or len(self.child_statistics) != count:
            raise ValueError("Primitive statistics do not match token count.")
        if len(self.statistic_names) != self.child_statistics.shape[1]:
            raise ValueError("Primitive statistic names do not match columns.")
        if len(self.event_kinds) not in {0, count - 1}:
            raise ValueError("Every internal boundary needs one event kind.")
        if np.any(self.starts < 0) or np.any(self.ends <= self.starts):
            raise ValueError("Primitive spans must be positive.")
        if np.any(self.starts[1:] != self.ends[:-1]):
            raise ValueError("Primitive spans must be a contiguous partition.")
        if np.any(self.child_tokens < 0) or np.any(self.child_tokens >= primitive_num):
            raise ValueError("A primitive token lies outside the child codebook.")
        numeric = (
            np.all(np.isfinite(self.child_distances))
            and np.all(np.isfinite(self.child_embeddings))
            and np.all(np.isfinite(self.child_statistics))
        )
        if not numeric:
            raise ValueError("Primitive trajectory contains non-finite values.")
        if self.parent_overlay is not None:
            self.parent_overlay.validate(self.child_tokens)
        return self


@dataclass(frozen=True)
class FeatureTransform:
    keep_columns: np.ndarray
    mean: np.ndarray
    scale: np.ndarray
    pca_mean: np.ndarray | None
    pca_components: np.ndarray | None

    def transform(self, values: np.ndarray) -> np.ndarray:
        matrix = np.asarray(values, dtype=np.float64)[:, self.keep_columns]
        matrix = (matrix - self.mean) / self.scale
        if self.pca_components is not None:
            matrix = (matrix - self.pca_mean) @ self.pca_components.T
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        return (matrix / np.maximum(norms, 1e-12)).astype(np.float32)


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    return value


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_jsonable(value), ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Cannot write empty CSV {path}.")
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(str(key))
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: (
                        json.dumps(_jsonable(value), ensure_ascii=False, separators=(",", ":"))
                        if isinstance(value, (dict, tuple, list, np.ndarray))
                        else value
                    )
                    for key, value in row.items()
                }
            )


def _generated_file_manifest(
    output_dir: Path, relative_paths: Sequence[str]
) -> list[dict[str, Any]]:
    """Build a traversal-safe, content-addressed artifact manifest.

    ``experiment_result.json`` is intentionally not included because it owns
    this manifest and therefore cannot contain a stable hash of itself.
    """

    root = output_dir.resolve()
    records: list[dict[str, Any]] = []
    seen: set[str] = set()
    for value in relative_paths:
        relative = Path(str(value))
        if relative.is_absolute() or ".." in relative.parts:
            raise RuntimeError(f"Generated artifact path escapes output root: {value!r}.")
        normalized = relative.as_posix()
        if not normalized or normalized == "." or normalized in seen:
            raise RuntimeError(f"Invalid or duplicate generated artifact: {value!r}.")
        candidate = (root / relative).resolve()
        try:
            candidate.relative_to(root)
        except ValueError as error:
            raise RuntimeError(
                f"Generated artifact resolves outside output root: {value!r}."
            ) from error
        if not candidate.is_file():
            raise RuntimeError(f"Generated artifact is missing: {candidate}.")
        size = int(candidate.stat().st_size)
        if size < 1:
            raise RuntimeError(f"Generated artifact is empty: {candidate}.")
        records.append(
            {
                "relative_path": normalized,
                "size_bytes": size,
                "sha256": sha256_file(candidate),
            }
        )
        seen.add(normalized)
    return records


def _parse_arms(value: str) -> tuple[str, ...]:
    arms: list[str] = []
    for token in str(value).split(","):
        arm = token.strip().upper()
        if not arm:
            continue
        if arm not in ALLOWED_ARMS:
            raise argparse.ArgumentTypeError(
                f"Unknown arm {arm!r}; allowed={list(ALLOWED_ARMS)}."
            )
        if arm not in arms:
            arms.append(arm)
    if not arms:
        raise argparse.ArgumentTypeError("At least one experiment arm is required.")
    return tuple(arms)


REGISTERED_WINDOW_GRIDS = frozenset({(256, 128), (128, 64)})


def _validate_registered_window_grid(
    window_size_samples: int, stride_samples: int
) -> tuple[int, int]:
    identity = (int(window_size_samples), int(stride_samples))
    if identity not in REGISTERED_WINDOW_GRIDS:
        raise ValueError(
            "The registered window grid must be one of "
            f"{sorted(REGISTERED_WINDOW_GRIDS)}; got {identity}."
        )
    return identity


def _load_numeric_grid(
    path: Path,
    *,
    expected_window_size_samples: int = 256,
    expected_stride_samples: int = 128,
) -> WindowGrid:
    expected_window_size, expected_stride = _validate_registered_window_grid(
        expected_window_size_samples, expected_stride_samples
    )
    required = {
        "windows",
        "labels",
        "subject_ids",
        "trial_global_ids",
        "window_indices",
        "window_start_indices",
        "mean",
        "std",
    }
    with np.load(path, allow_pickle=False) as data:
        missing = required - set(data.files)
        if missing:
            raise RuntimeError(f"USC-HAD NPZ lacks fields {sorted(missing)}.")
        grid = WindowGrid(
            windows=np.asarray(data["windows"], dtype=np.float32),
            labels=np.asarray(data["labels"], dtype=np.int64),
            subject_ids=np.asarray(data["subject_ids"], dtype=np.int64),
            trial_ids=np.asarray(data["trial_global_ids"], dtype=np.int64),
            window_indices=np.asarray(data["window_indices"], dtype=np.int64),
            starts=np.asarray(data["window_start_indices"], dtype=np.int64),
            stored_mean=np.asarray(data["mean"], dtype=np.float32),
            stored_std=np.asarray(data["std"], dtype=np.float32),
            window_size_samples=expected_window_size,
            stride_samples=expected_stride,
        )
    row_count = len(grid.windows)
    expected_window_shape = (6, expected_window_size)
    if grid.windows.ndim != 3 or grid.windows.shape[1:] != expected_window_shape:
        raise RuntimeError(
            "The requested registered comparison requires USC-HAD windows "
            f"[N,6,{expected_window_size}]; observed {grid.windows.shape}."
        )
    for name in ("labels", "subject_ids", "trial_ids", "window_indices", "starts"):
        if np.asarray(getattr(grid, name)).shape != (row_count,):
            raise RuntimeError(f"Window-grid field {name} has an invalid shape.")
    expected_stats = (1, 6, 1)
    if grid.stored_mean.shape != expected_stats or grid.stored_std.shape != expected_stats:
        raise RuntimeError("USC-HAD stored normalization has an unexpected shape.")
    if not np.all(np.isfinite(grid.windows)) or np.any(grid.stored_std <= 0):
        raise RuntimeError("USC-HAD numeric grid is invalid.")
    differences: list[int] = []
    for trial_id in np.unique(grid.trial_ids):
        starts = np.sort(grid.starts[grid.trial_ids == trial_id])
        differences.extend(int(value) for value in np.diff(starts))
    if not differences or set(differences) != {expected_stride}:
        raise RuntimeError(
            f"The requested w{expected_window_size}/s{expected_stride} grid "
            f"does not match observed strides {sorted(set(differences))}."
        )
    return grid


def _metadata_subjects(metadata: Mapping[str, Any], key: str) -> list[int]:
    values = metadata.get(key)
    if not isinstance(values, list) or not values:
        raise RuntimeError(f"Checkpoint metadata lacks non-empty {key}.")
    result = sorted(int(value) for value in values)
    if len(result) != len(set(result)):
        raise RuntimeError(f"Checkpoint metadata {key} contains duplicates.")
    return result


def _array_sha256(*arrays: np.ndarray) -> str:
    digest = hashlib.sha256()
    for value in arrays:
        array = np.ascontiguousarray(np.asarray(value))
        digest.update(str(array.shape).encode("ascii"))
        digest.update(array.dtype.str.encode("ascii"))
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _trial_rows(grid: WindowGrid, trial_id: int) -> np.ndarray:
    rows = np.flatnonzero(grid.trial_ids == int(trial_id))
    if not len(rows):
        raise KeyError(f"Unknown trial_global_id={trial_id}.")
    order = np.argsort(grid.starts[rows], kind="stable")
    rows = rows[order]
    if not np.array_equal(grid.window_indices[rows], np.arange(len(rows))):
        raise RuntimeError(f"Trial {trial_id} has incomplete window indices.")
    if int(grid.windows.shape[2]) != int(grid.window_size_samples):
        raise RuntimeError(
            f"Trial grid window metadata disagrees with data shape: "
            f"{grid.window_size_samples} vs {grid.windows.shape[2]}."
        )
    if int(grid.starts[rows[0]]) != 0 or np.any(
        np.diff(grid.starts[rows]) != int(grid.stride_samples)
    ):
        raise RuntimeError(
            f"Trial {trial_id} violates the requested "
            f"w{grid.window_size_samples}/s{grid.stride_samples} grid."
        )
    return rows


def _fold_normalization(
    grid: WindowGrid, fit_window_mask: np.ndarray, epsilon: float
) -> tuple[np.ndarray, np.ndarray, dict]:
    if fit_window_mask.shape != (len(grid.windows),) or not np.any(fit_window_mask):
        raise ValueError("The offline fit-window mask is empty or malformed.")
    raw = (
        grid.windows[fit_window_mask] * grid.stored_std + grid.stored_mean
    ).astype(np.float64)
    mean = raw.mean(axis=(0, 2), keepdims=True)
    std = np.maximum(raw.std(axis=(0, 2), keepdims=True), float(epsilon))
    return mean.astype(np.float32), std.astype(np.float32), {
        "mode": "fold_train_subjects_old_classes",
        "fit_window_count": int(np.sum(fit_window_mask)),
        "mean": mean,
        "std": std,
        "fit_window_mask_sha256": _array_sha256(fit_window_mask.astype(np.uint8)),
    }


def _normalized_windows(
    grid: WindowGrid,
    rows: np.ndarray,
    fold_mean: np.ndarray,
    fold_std: np.ndarray,
) -> np.ndarray:
    raw = grid.windows[rows] * grid.stored_std + grid.stored_mean
    normalized = (raw - fold_mean) / fold_std
    if not np.all(np.isfinite(normalized)):
        raise RuntimeError("Checkpoint-normalized windows contain non-finite values.")
    return normalized.astype(np.float32)


def _window_partition(starts: np.ndarray, window_size: int) -> tuple[np.ndarray, np.ndarray]:
    starts = np.asarray(starts, dtype=np.int64)
    if starts.ndim != 1 or len(starts) < 1 or starts[0] != 0:
        raise ValueError("Window starts must be a non-empty vector beginning at zero.")
    centers = starts.astype(np.float64) + float(window_size) / 2.0
    boundaries = [0]
    boundaries.extend(
        int(round(0.5 * (left + right)))
        for left, right in zip(centers[:-1], centers[1:])
    )
    boundaries.append(int(starts[-1]) + int(window_size))
    values = np.asarray(boundaries, dtype=np.int64)
    if np.any(np.diff(values) <= 0):
        raise RuntimeError("Window-centre Voronoi partition is not strictly increasing.")
    return values[:-1], values[1:]


def _statistic_names() -> tuple[str, ...]:
    result = ["duration_seconds"]
    for prefix in ("mean", "std", "log_mean_square_energy", "end_minus_start"):
        result.extend(f"{prefix}__{name}" for name in CHANNEL_NAMES)
    return tuple(result)


def _segment_statistics(
    signal: np.ndarray,
    starts: np.ndarray,
    ends: np.ndarray,
    sample_rate_hz: float,
) -> np.ndarray:
    rows = []
    for start, end in zip(starts.tolist(), ends.tolist()):
        raw = np.asarray(signal[:, int(start) : int(end)], dtype=np.float64)
        if raw.shape[1] < 1:
            raise RuntimeError("An empty sensor partition reached the statistic path.")
        rows.append(
            np.r_[
                (int(end) - int(start)) / float(sample_rate_hz),
                raw.mean(axis=1),
                raw.std(axis=1),
                np.log(np.mean(np.square(raw), axis=1) + 1e-8),
                raw[:, -1] - raw[:, 0],
            ]
        )
    return np.asarray(rows, dtype=np.float64)


def _codebook_transform(values: np.ndarray, state: CodebookState) -> np.ndarray:
    matrix = np.asarray(values, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[1] != state.input_dim:
        raise ValueError("Codebook input dimension mismatch.")
    if state.pca_components is not None:
        matrix = (matrix - state.pca_mean) @ state.pca_components.T
    else:
        matrix = matrix.copy()
    if state.config.l2_normalize:
        matrix /= np.maximum(np.linalg.norm(matrix, axis=1, keepdims=True), 1e-12)
    return matrix


def _codebook_assign(values: np.ndarray, state: CodebookState) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    embedded = _codebook_transform(values, state)
    squared = np.sum(
        np.square(embedded[:, None, :] - state.cluster_centers[None, :, :]), axis=2
    )
    tokens = np.argmin(squared, axis=1).astype(np.int64)
    distances = np.sqrt(squared[np.arange(len(tokens)), tokens])
    return tokens, distances.astype(np.float64), embedded.astype(np.float64)


def _e0_codebook_assign(
    values: np.ndarray, state: CodebookState
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Reproduce the historical fixed-window cosine assignment exactly.

    The legacy experiment L2-normalized both PCA features and fitted KMeans
    centres before assignment, then stored ``1 - cosine_similarity`` as its
    quantization distance.  The variable-segment arms deliberately retain the
    generic Euclidean assignment above.
    """

    base = l2_normalize(np.asarray(values, dtype=np.float32))
    if state.pca_components is not None:
        embedded = (
            (base - np.asarray(state.pca_mean, dtype=np.float32))
            @ np.asarray(state.pca_components, dtype=np.float32).T
        ).astype(np.float32)
    else:
        embedded = base
    embedded = l2_normalize(embedded)
    tokens, distances, _ = assign_to_codebook(
        embedded,
        np.asarray(state.cluster_centers, dtype=np.float32),
        "cosine",
    )
    return (
        tokens.astype(np.int64),
        distances.astype(np.float64),
        embedded.astype(np.float64),
    )


def _resample_signal(signal: np.ndarray, points: int) -> np.ndarray:
    source = np.asarray(signal, dtype=np.float64)
    if source.ndim != 2 or source.shape[0] != 6 or source.shape[1] < 1:
        raise ValueError("A segment signal must be [6,T>=1].")
    if source.shape[1] == 1:
        return np.repeat(source, int(points), axis=1)
    old = np.linspace(0.0, 1.0, source.shape[1])
    new = np.linspace(0.0, 1.0, int(points))
    return np.stack([np.interp(new, old, row) for row in source], axis=0)


def _replace_shape_features(
    batches: Sequence[ChildPrimitiveBatch], features: Sequence[np.ndarray]
) -> list[ChildPrimitiveBatch]:
    if len(batches) != len(features):
        raise ValueError("Child batches/features differ in trial count.")
    result = []
    for batch, values in zip(batches, features):
        matrix = np.asarray(values, dtype=np.float64)
        if matrix.ndim != 2 or len(matrix) != len(batch.segment_indices):
            raise ValueError("Replacement child features have an invalid shape.")
        result.append(replace(batch, shape_features=matrix).validate())
    return result


def _encode_segment_content(
    trials: Sequence[TrialSignal],
    segmentations: Sequence[SegmentedTrial],
    encoder,
    device,
    batch_size: int,
    fold_mean: np.ndarray,
    fold_std: np.ndarray,
    window_size: int,
) -> list[np.ndarray]:
    trial_map = {int(item.trial_id): item for item in trials}
    rows: list[np.ndarray] = []
    counts: list[int] = []
    for segmented in segmentations:
        signal = trial_map[int(segmented.trial_id)].signal
        current = [
            _resample_signal(signal[:, item.start_sample : item.end_sample_exclusive], window_size)
            for item in segmented.segments
        ]
        rows.extend(current)
        counts.append(len(current))
    raw = np.asarray(rows, dtype=np.float32)
    normalized = ((raw - fold_mean) / fold_std).astype(np.float32)
    encoded = encode_motion_windows(encoder, normalized, device, int(batch_size))["content"]
    output: list[np.ndarray] = []
    cursor = 0
    for count in counts:
        output.append(np.asarray(encoded[cursor : cursor + count], dtype=np.float64))
        cursor += count
    if cursor != len(encoded):
        raise RuntimeError("Segment-content batching lost rows.")
    return output


def _select_child_features(
    raw_batches: Sequence[ChildPrimitiveBatch],
    content_rows: Sequence[np.ndarray],
    source: str,
) -> list[ChildPrimitiveBatch]:
    if source == "raw_shape":
        return [item.validate() for item in raw_batches]
    if source == "content_embedding":
        return _replace_shape_features(raw_batches, content_rows)
    if source != "hybrid":
        raise ValueError(f"Unknown child feature source {source!r}.")
    hybrid = []
    for raw, content in zip(raw_batches, content_rows):
        shape = np.asarray(raw.shape_features, dtype=np.float64)
        encoded = np.asarray(content, dtype=np.float64)
        shape /= np.maximum(np.linalg.norm(shape, axis=1, keepdims=True), 1e-12)
        encoded /= np.maximum(np.linalg.norm(encoded, axis=1, keepdims=True), 1e-12)
        hybrid.append(np.concatenate([encoded, shape], axis=1) / math.sqrt(2.0))
    return _replace_shape_features(raw_batches, hybrid)


def _primitive_trials_from_tokenized(
    tokenized: Sequence[TokenizedTrial],
    primitive_num: int,
    overlays: Sequence[ParentOverlay] | None = None,
) -> list[PrimitiveTrial]:
    overlay_map = (
        {} if overlays is None else {int(item.trial_id): item for item in overlays}
    )
    result = []
    for item in tokenized:
        trial = PrimitiveTrial(
            trial_id=int(item.trial_id),
            subject_id=int(item.subject_id),
            starts=item.child_features.start_samples.astype(np.int64, copy=True),
            ends=item.child_features.end_samples_exclusive.astype(np.int64, copy=True),
            child_tokens=item.primitive_tokens.astype(np.int64, copy=True),
            child_distances=item.nearest_center_distances.astype(np.float64, copy=True),
            child_embeddings=item.codebook_embeddings.astype(np.float64, copy=True),
            child_statistics=item.child_features.statistic_features.astype(np.float64, copy=True),
            statistic_names=tuple(item.child_features.statistic_names),
            event_kinds=tuple(event.kind for event in item.segmentation.events),
            parent_overlay=overlay_map.get(int(item.trial_id)),
        ).validate(int(primitive_num))
        result.append(trial)
    return sorted(result, key=lambda item: item.trial_id)


def _primitive_trials_e0(
    trial_ids: Sequence[int],
    subject_by_trial: Mapping[int, int],
    grid: WindowGrid,
    window_embeddings: Mapping[int, np.ndarray],
    codebook: CodebookState,
    source: LabelFreeSourceSignalRepository,
    sample_rate_hz: float,
) -> list[PrimitiveTrial]:
    result = []
    for trial_id_value in sorted(int(value) for value in trial_ids):
        trial_id = int(trial_id_value)
        rows = _trial_rows(grid, trial_id)
        window_starts = np.asarray(grid.starts[rows], dtype=np.int64)
        window_size = int(grid.windows.shape[2])
        starts, ends = _window_partition(window_starts, window_size)
        tokens, distances, embedded = _e0_codebook_assign(
            window_embeddings[trial_id], codebook
        )
        signal = source.sensor(trial_id)
        window_ends = window_starts + window_size
        if np.any(window_starts < 0) or np.any(window_ends > signal.shape[1]):
            raise RuntimeError(
                "An E0 encoder window falls outside its reconstructed trial signal."
            )
        statistics = _segment_statistics(
            signal, window_starts, window_ends, sample_rate_hz
        )
        # Duration is an ownership quantity, not a signal-state statistic.
        # Keep it non-overlapping even though every remaining statistic uses
        # the token's original overlapping encoder window.
        statistics[:, 0] = (ends - starts) / float(sample_rate_hz)
        trial = PrimitiveTrial(
            trial_id=trial_id,
            subject_id=int(subject_by_trial[trial_id]),
            starts=starts,
            ends=ends,
            child_tokens=tokens,
            child_distances=distances,
            child_embeddings=embedded,
            # Token-conditioned physical state uses the same overlapping raw
            # window as the encoded token.  Starts/ends remain non-overlapping
            # ownership cells so duration fractions never double-count time.
            child_statistics=statistics,
            statistic_names=_statistic_names(),
            event_kinds=(),
        ).validate(codebook.config.primitive_num)
        result.append(trial)
    return result


def _fit_descriptor_transform(values: np.ndarray, maximum_components: int = 32) -> FeatureTransform:
    matrix = np.asarray(values, dtype=np.float64)
    if matrix.ndim != 2 or len(matrix) < 2 or matrix.shape[1] < 1:
        raise ValueError("Descriptor fit matrix must be [N>=2,D>=1].")
    variation = matrix.std(axis=0)
    keep = np.flatnonzero(variation > 1e-10)
    if not len(keep):
        raise RuntimeError("Every trajectory descriptor column is constant.")
    selected = matrix[:, keep]
    mean = selected.mean(axis=0)
    scale = np.maximum(selected.std(axis=0), 1e-8)
    standardized = (selected - mean) / scale
    output_dim = min(int(maximum_components), len(standardized) - 1, standardized.shape[1])
    pca_mean = None
    components = None
    if output_dim < standardized.shape[1]:
        pca_mean = standardized.mean(axis=0)
        _, _, right = np.linalg.svd(standardized - pca_mean, full_matrices=False)
        components = right[:output_dim].copy()
        for row in components:
            pivot = int(np.argmax(np.abs(row)))
            if row[pivot] < 0:
                row *= -1.0
    return FeatureTransform(
        keep_columns=keep.astype(np.int64),
        mean=mean,
        scale=scale,
        pca_mean=pca_mean,
        pca_components=components,
    )


def _trajectory_descriptor(
    trial: PrimitiveTrial,
    primitive_num: int,
    parent_ids: Sequence[str],
    *,
    include_state: bool,
) -> tuple[np.ndarray, tuple[str, ...]]:
    trial.validate(int(primitive_num))
    tokens = trial.child_tokens.astype(np.int64)
    durations = (trial.ends - trial.starts).astype(np.float64)
    total_duration = float(durations.sum())
    count_hist = np.bincount(tokens, minlength=primitive_num).astype(np.float64)
    count_hist /= max(float(len(tokens)), 1.0)
    duration_hist = np.bincount(tokens, weights=durations, minlength=primitive_num)
    duration_hist /= max(total_duration, 1.0)
    values: list[float] = []
    names: list[str] = []

    def extend(prefix: str, vector: np.ndarray) -> None:
        flat = np.asarray(vector, dtype=np.float64).reshape(-1)
        values.extend(float(value) for value in flat)
        names.extend(f"{prefix}{index}" for index in range(len(flat)))

    extend("child_count_fraction__", count_hist)
    extend("child_duration_fraction__", duration_hist)

    midpoints = 0.5 * (trial.starts.astype(np.float64) + trial.ends.astype(np.float64))
    relative = np.clip(midpoints / max(float(trial.ends[-1]), 1.0), 0.0, 1.0 - 1e-12)
    temporal = np.zeros((4, primitive_num), dtype=np.float64)
    for token, duration, position in zip(tokens, durations, relative):
        temporal[min(3, int(position * 4.0)), int(token)] += float(duration)
    temporal /= max(total_duration, 1.0)
    extend("child_temporal_quartile_duration__", temporal)

    any_transition = np.zeros((primitive_num, primitive_num), dtype=np.float64)
    peak_transition = np.zeros_like(any_transition)
    valley_transition = np.zeros_like(any_transition)
    if len(tokens) > 1:
        for index, (left, right) in enumerate(zip(tokens[:-1], tokens[1:])):
            any_transition[int(left), int(right)] += 1.0
            kind = trial.event_kinds[index] if trial.event_kinds else "none"
            if kind == "peak":
                peak_transition[int(left), int(right)] += 1.0
            elif kind == "valley":
                valley_transition[int(left), int(right)] += 1.0
        any_transition /= float(len(tokens) - 1)
        peak_transition /= float(len(tokens) - 1)
        valley_transition /= float(len(tokens) - 1)
    extend("transition_any__", any_transition)
    extend("transition_peak__", peak_transition)
    extend("transition_valley__", valley_transition)

    quality = np.asarray(trial.child_distances, dtype=np.float64)
    duration_seconds = trial.child_statistics[:, 0]
    scalar = np.asarray(
        [
            math.log1p(len(tokens)),
            math.log1p(total_duration),
            float(np.mean(duration_seconds)),
            float(np.std(duration_seconds)),
            float(np.max(duration_seconds)),
            float(np.mean(quality)),
            float(np.std(quality)),
            float(np.max(quality)),
            float(sum(kind == "peak" for kind in trial.event_kinds))
            / max(1, len(trial.event_kinds)),
            float(sum(kind == "valley" for kind in trial.event_kinds))
            / max(1, len(trial.event_kinds)),
        ],
        dtype=np.float64,
    )
    scalar_names = (
        "log_child_count",
        "log_duration_samples",
        "child_duration_seconds_mean",
        "child_duration_seconds_std",
        "child_duration_seconds_max",
        "quantization_distance_mean",
        "quantization_distance_std",
        "quantization_distance_max",
        "peak_boundary_fraction",
        "valley_boundary_fraction",
    )
    values.extend(scalar.tolist())
    names.extend(scalar_names)

    parent_position = {str(parent_id): index for index, parent_id in enumerate(parent_ids)}
    parent_count = np.zeros(len(parent_ids), dtype=np.float64)
    parent_duration = np.zeros(len(parent_ids), dtype=np.float64)
    parent_temporal = np.zeros((4, len(parent_ids)), dtype=np.float64)
    occurrences = (
        () if trial.parent_overlay is None else trial.parent_overlay.parent_occurrences
    )
    for occurrence in occurrences:
        index = parent_position.get(str(occurrence.parent_id))
        if index is None:
            raise RuntimeError("Overlay contains an unknown registered parent id.")
        span = occurrence.span_end_sample_exclusive - occurrence.span_start_sample
        centre = 0.5 * (
            occurrence.span_start_sample + occurrence.span_end_sample_exclusive
        )
        quartile = min(3, int(4.0 * centre / max(float(trial.ends[-1]), 1.0)))
        parent_count[index] += 1.0
        parent_duration[index] += float(span)
        parent_temporal[quartile, index] += float(span)
    if len(occurrences):
        parent_count /= float(len(occurrences))
    parent_duration /= max(total_duration, 1.0)
    parent_temporal /= max(total_duration, 1.0)
    extend("parent_count_fraction__", parent_count)
    extend("parent_span_fraction__", parent_duration)
    extend("parent_temporal_quartile_span__", parent_temporal)
    values.append(math.log1p(len(occurrences)))
    names.append("log_parent_occurrence_count")

    if include_state:
        statistics = np.asarray(trial.child_statistics[:, 1:], dtype=np.float64)
        weights = durations / max(total_duration, 1.0)
        global_mean = np.sum(statistics * weights[:, None], axis=0)
        global_std = np.sqrt(
            np.sum(np.square(statistics - global_mean) * weights[:, None], axis=0)
        )
        statistic_names = trial.statistic_names[1:]
        values.extend(global_mean.tolist())
        names.extend(f"state_global_mean__{name}" for name in statistic_names)
        values.extend(global_std.tolist())
        names.extend(f"state_global_std__{name}" for name in statistic_names)

        selected_columns = [
            index
            for index, name in enumerate(statistic_names)
            if name.startswith("mean__") or name.startswith("log_mean_square_energy__")
        ]
        token_state = np.zeros((primitive_num, len(selected_columns)), dtype=np.float64)
        token_presence = np.zeros(primitive_num, dtype=np.float64)
        for token in range(primitive_num):
            selected = tokens == token
            if not np.any(selected):
                continue
            local_weights = durations[selected]
            token_state[token] = np.average(
                statistics[selected][:, selected_columns], axis=0, weights=local_weights
            )
            token_presence[token] = 1.0
        extend("state_by_child_token__", token_state)
        extend("state_child_token_presence__", token_presence)

    vector = np.asarray(values, dtype=np.float64)
    if not np.all(np.isfinite(vector)) or len(vector) != len(names):
        raise RuntimeError("Trajectory descriptor construction failed.")
    return vector, tuple(names)


def _descriptor_matrices(
    train_trials: Sequence[PrimitiveTrial],
    test_trials: Sequence[PrimitiveTrial],
    primitive_num: int,
    parent_ids: Sequence[str],
    *,
    include_state: bool,
) -> tuple[np.ndarray, np.ndarray, tuple[str, ...]]:
    rows = [
        _trajectory_descriptor(
            item, primitive_num, parent_ids, include_state=include_state
        )
        for item in (*train_trials, *test_trials)
    ]
    schemas = {names for _, names in rows}
    if len(schemas) != 1:
        raise RuntimeError("Trajectory descriptor schemas differ across trials.")
    matrix = np.stack([values for values, _ in rows], axis=0)
    return matrix[: len(train_trials)], matrix[len(train_trials) :], rows[0][1]


def _make_trial_signals(
    trial_ids: Sequence[int],
    subject_by_trial: Mapping[int, int],
    source: LabelFreeSourceSignalRepository,
    grid: WindowGrid,
    boundary_embeddings: Mapping[int, np.ndarray] | None,
    feature_interpolation_step_samples: int = 25,
) -> list[TrialSignal]:
    result = []
    for trial_id_value in sorted(int(value) for value in trial_ids):
        trial_id = int(trial_id_value)
        rows = _trial_rows(grid, trial_id)
        features = None
        positions = None
        if boundary_embeddings is not None:
            sparse_features = np.asarray(
                boundary_embeddings[trial_id], dtype=np.float64
            )
            sparse_positions = grid.starts[rows] + grid.windows.shape[2] // 2
            signal_length = source.sensor(trial_id).shape[1]
            step = max(1, int(feature_interpolation_step_samples))
            positions = np.arange(0, signal_length, step, dtype=np.int64)
            if positions[-1] != signal_length - 1:
                positions = np.r_[positions, signal_length - 1]
            # Window features live on the explicitly validated NPZ stride grid.
            # Linear interpolation creates a continuous frozen-feature trajectory
            # so a physical 0.5-second left/right confirmation context is not
            # spuriously empty.
            features = np.stack(
                [
                    np.interp(
                        positions,
                        sparse_positions,
                        sparse_features[:, column],
                    )
                    for column in range(sparse_features.shape[1])
                ],
                axis=1,
            )
        result.append(
            TrialSignal(
                trial_id=trial_id,
                subject_id=int(subject_by_trial[trial_id]),
                signal=source.sensor(trial_id).astype(np.float64),
                feature_values=features,
                feature_sample_positions=positions,
            ).validated()
        )
    return result


def _child_pipeline(
    fit_trials: Sequence[TrialSignal],
    online_trials: Sequence[TrialSignal],
    config: PeakValleyConfig,
    child_feature_source: str,
    encoder,
    device,
    batch_size: int,
    fold_mean: np.ndarray,
    fold_std: np.ndarray,
    encoder_window_size_samples: int,
    primitive_num: int,
    pca_dim: int,
    seed: int,
) -> tuple[Any, CodebookState, list[TokenizedTrial], list[TokenizedTrial], list[SegmentedTrial], list[SegmentedTrial]]:
    segmenter = fit_segmenter(fit_trials, config)
    fit_segmentations = segment_trials(fit_trials, segmenter)
    online_segmentations = segment_trials(online_trials, segmenter)
    fit_raw = extract_child_feature_batches(fit_trials, fit_segmentations, segmenter)
    online_raw = extract_child_feature_batches(
        online_trials, online_segmentations, segmenter
    )
    if child_feature_source == "raw_shape":
        fit_content: list[np.ndarray] = []
        online_content: list[np.ndarray] = []
    else:
        fit_content = _encode_segment_content(
            fit_trials,
            fit_segmentations,
            encoder,
            device,
            batch_size,
            fold_mean,
            fold_std,
            int(encoder_window_size_samples),
        )
        online_content = _encode_segment_content(
            online_trials,
            online_segmentations,
            encoder,
            device,
            batch_size,
            fold_mean,
            fold_std,
            int(encoder_window_size_samples),
        )
    fit_batches = _select_child_features(
        fit_raw, fit_content, child_feature_source
    )
    online_batches = _select_child_features(
        online_raw, online_content, child_feature_source
    )
    codebook = fit_codebook(
        fit_batches,
        CodebookConfig(
            primitive_num=int(primitive_num),
            pca_dim=int(pca_dim),
            weighting="subject_trial_equal",
            l2_normalize=True,
            kmeans_n_init=20,
            kmeans_max_iter=300,
            random_seed=int(seed),
        ),
    )
    fit_tokenized = assign_codebook_trials(
        fit_batches, fit_segmentations, codebook
    )
    online_tokenized = assign_codebook_trials(
        online_batches, online_segmentations, codebook
    )
    return (
        segmenter,
        codebook,
        fit_tokenized,
        online_tokenized,
        fit_segmentations,
        online_segmentations,
    )


def _e0_codebook_batches(
    trial_ids: Sequence[int],
    subject_by_trial: Mapping[int, int],
    grid: WindowGrid,
    embeddings: Mapping[int, np.ndarray],
) -> list[ChildPrimitiveBatch]:
    protocol_hash = hashlib.sha256(
        (
            "legacy_npz_overlapping_windows_"
            f"w{grid.window_size_samples}_s{grid.stride_samples}_v1"
        ).encode("ascii")
    ).hexdigest()
    batches = []
    for trial_id in sorted(int(value) for value in trial_ids):
        rows = _trial_rows(grid, trial_id)
        starts = grid.starts[rows].astype(np.int64)
        count = len(rows)
        batches.append(
            ChildPrimitiveBatch(
                trial_id=trial_id,
                subject_id=int(subject_by_trial[trial_id]),
                segment_indices=np.arange(count, dtype=np.int64),
                start_samples=starts,
                end_samples_exclusive=starts + grid.windows.shape[2],
                shape_features=np.asarray(embeddings[trial_id], dtype=np.float64),
                statistic_features=np.zeros((count, 1), dtype=np.float64),
                statistic_names=("unused",),
                segmentation_state_sha256=protocol_hash,
            ).validate()
        )
    return batches


def _e0_pre_pca_l2(
    embeddings: Mapping[int, np.ndarray],
) -> dict[int, np.ndarray]:
    """Apply the historical per-window L2 step before weighted PCA."""

    result: dict[int, np.ndarray] = {}
    for trial_id, values in embeddings.items():
        matrix = np.asarray(values, dtype=np.float32)
        if matrix.ndim != 2 or not len(matrix) or not np.all(np.isfinite(matrix)):
            raise ValueError("E0 content embeddings must be a finite non-empty matrix.")
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        if np.any(norms <= 1e-12):
            raise RuntimeError("E0 content embedding contains a zero-norm row.")
        result[int(trial_id)] = (
            matrix / np.maximum(norms, np.float32(1e-12))
        ).astype(np.float32)
    return result


def _fit_e0_historical_codebook(
    fit_batches: Sequence[ChildPrimitiveBatch],
    *,
    primitive_num: int,
    pca_dim: int,
    seed: int,
) -> CodebookState:
    """Fit E0 through the exact historical float32 feature/KMeans path."""

    batches = sorted(
        (item.validate() for item in fit_batches), key=lambda item: int(item.trial_id)
    )
    if not batches:
        raise ValueError("E0 requires at least one fit trial.")
    if int(primitive_num) != 32 or int(pca_dim) != 64:
        raise ValueError("Historical E0 is pinned to KMeans32/PCA64.")
    state_hashes = {item.segmentation_state_sha256 for item in batches}
    if len(state_hashes) != 1:
        raise ValueError("E0 batches do not share one window-grid state.")
    raw_values = np.concatenate(
        [np.asarray(item.shape_features, dtype=np.float32) for item in batches], axis=0
    )
    trial_ids = np.concatenate(
        [
            np.full(len(item.segment_indices), int(item.trial_id), dtype=np.int64)
            for item in batches
        ]
    )
    weights = inverse_trial_frequency_weights(trial_ids)
    transformed, _, pca = prepare_primitive_features(
        raw_values,
        raw_values[:1],
        weights,
        int(pca_dim),
        "l2",
    )
    if pca is None or transformed.dtype != np.float32:
        raise RuntimeError("Historical E0 PCA64 did not produce float32 features.")
    model = KMeans(
        n_clusters=int(primitive_num),
        random_state=int(seed),
        n_init=20,
        max_iter=300,
        algorithm="lloyd",
    )
    model.fit(transformed, sample_weight=weights)
    subject_mass: dict[str, float] = {}
    cursor = 0
    for item in batches:
        count = len(item.segment_indices)
        key = str(int(item.subject_id))
        subject_mass[key] = subject_mass.get(key, 0.0) + float(
            weights[cursor : cursor + count].sum()
        )
        cursor += count
    config = CodebookConfig(
        primitive_num=int(primitive_num),
        pca_dim=int(pca_dim),
        weighting="trial_equal",
        l2_normalize=True,
        kmeans_n_init=20,
        kmeans_max_iter=300,
        random_seed=int(seed),
    )
    return CodebookState(
        config=config,
        input_dim=int(raw_values.shape[1]),
        output_dim=int(transformed.shape[1]),
        # Store float32-trained values exactly inside the float64-compatible
        # state schema; E0 inference casts them back to float32 above.
        pca_mean=np.asarray(pca.mean, dtype=np.float64),
        pca_components=np.asarray(pca.components, dtype=np.float64),
        cluster_centers=np.asarray(model.cluster_centers_, dtype=np.float64),
        segmentation_state_sha256=batches[0].segmentation_state_sha256,
        fit_data_sha256=_codebook_fit_hash(batches),
        fit_segment_count=int(len(raw_values)),
        fit_trial_count=int(len(batches)),
        fit_subject_count=len({int(item.subject_id) for item in batches}),
        weight_summary={
            "mode": "trial_equal",
            "normalization": "mean window weight equals one",
            "historical_float32_pipeline": True,
            "minimum": float(weights.min()),
            "maximum": float(weights.max()),
            "subject_total_mass": subject_mass,
        },
        kmeans_inertia=float(model.inertia_),
        kmeans_n_iter=int(model.n_iter_),
        numpy_version=np.__version__,
        sklearn_version=sklearn.__version__,
    ).validate()


def _registered_parent_ids(catalog: ParentCatalogState | None) -> tuple[str, ...]:
    if catalog is None:
        return ()
    return tuple(
        str(item.parent_id)
        for item in sorted(catalog.registered_patterns, key=lambda item: str(item.parent_id))
    )


def _overlay_trials(
    fit_tokenized: Sequence[TokenizedTrial],
    online_tokenized: Sequence[TokenizedTrial],
    catalog: ParentCatalogState,
    primitive_num: int,
) -> tuple[list[PrimitiveTrial], list[PrimitiveTrial]]:
    fit_overlays = overlay_parent_trials(fit_tokenized, catalog, role="fit_replay")
    online_overlays = overlay_parent_trials(
        online_tokenized,
        catalog,
        role="held_out",
        require_subject_disjoint=True,
    )
    return (
        _primitive_trials_from_tokenized(fit_tokenized, primitive_num, fit_overlays),
        _primitive_trials_from_tokenized(
            online_tokenized, primitive_num, online_overlays
        ),
    )


def _run_readout(
    fit_trials: Sequence[PrimitiveTrial],
    online_trials: Sequence[PrimitiveTrial],
    train_ids: Sequence[int],
    test_ids: Sequence[int],
    primitive_num: int,
    parent_ids: Sequence[str],
    variant: str,
    cluster_count: int,
    seed: int,
) -> dict:
    fit_map = {item.trial_id: item for item in fit_trials}
    online_map = {item.trial_id: item for item in online_trials}
    # Stage-0 trajectories are deliberately not part of the Session-2 K=10
    # readout.  This explicit argument remains in the signature for audit and
    # makes accidental mixing detectable below.
    if set(fit_map) & set(online_map):
        raise RuntimeError("Offline-fit and held-out-subject trajectory ids overlap.")
    train_trials = [online_map[int(value)] for value in train_ids]
    test_trials = [online_map[int(value)] for value in test_ids]
    train_raw, test_raw, names = _descriptor_matrices(
        train_trials,
        test_trials,
        primitive_num,
        parent_ids,
        include_state=(variant == "state"),
    )
    transform = _fit_descriptor_transform(train_raw)
    train_features = transform.transform(train_raw)
    test_features = transform.transform(test_raw)
    readout = fit_frozen_coarse_readout(
        train_features,
        test_features,
        train_trial_ids=train_ids,
        test_trial_ids=test_ids,
        cluster_count=int(cluster_count),
        seed=int(seed),
        n_init=50,
        max_iter=300,
    )
    return {
        "readout": readout,
        "raw_train_descriptors": train_raw,
        "raw_test_descriptors": test_raw,
        "train_features": train_features,
        "test_features": test_features,
        "descriptor_names": names,
        "transform": transform,
        "leakage_audit": {
            "offline_stage0_trial_count_seen_by_readout": 0,
            "online_train_trial_count": len(train_ids),
            "online_test_trial_count": len(test_ids),
            "train_test_trial_overlap": sorted(set(train_ids) & set(test_ids)),
            "fit_uses_activity_labels": False,
            "transform_fit_scope": "cumulative_online_train_only",
            "test_used_to_fit_transform_or_kmeans": False,
        },
    }


def _tokenize_partitions(
    fit_trials: Sequence[TrialSignal],
    online_trials: Sequence[TrialSignal],
    fit_segmentations: Sequence[SegmentedTrial],
    online_segmentations: Sequence[SegmentedTrial],
    segmenter,
    child_feature_source: str,
    encoder,
    device,
    batch_size: int,
    fold_mean: np.ndarray,
    fold_std: np.ndarray,
    encoder_window_size_samples: int,
    primitive_num: int,
    pca_dim: int,
    seed: int,
) -> tuple[CodebookState, list[TokenizedTrial], list[TokenizedTrial]]:
    fit_raw = extract_child_feature_batches(fit_trials, fit_segmentations, segmenter)
    online_raw = extract_child_feature_batches(
        online_trials, online_segmentations, segmenter
    )
    if child_feature_source == "raw_shape":
        fit_content: list[np.ndarray] = []
        online_content: list[np.ndarray] = []
    else:
        fit_content = _encode_segment_content(
            fit_trials,
            fit_segmentations,
            encoder,
            device,
            batch_size,
            fold_mean,
            fold_std,
            int(encoder_window_size_samples),
        )
        online_content = _encode_segment_content(
            online_trials,
            online_segmentations,
            encoder,
            device,
            batch_size,
            fold_mean,
            fold_std,
            int(encoder_window_size_samples),
        )
    fit_batches = _select_child_features(
        fit_raw, fit_content, child_feature_source
    )
    online_batches = _select_child_features(
        online_raw, online_content, child_feature_source
    )
    codebook = fit_codebook(
        fit_batches,
        CodebookConfig(
            primitive_num=int(primitive_num),
            pca_dim=int(pca_dim),
            weighting="subject_trial_equal",
            l2_normalize=True,
            kmeans_n_init=20,
            kmeans_max_iter=300,
            random_seed=int(seed),
        ),
    )
    return (
        codebook,
        assign_codebook_trials(fit_batches, fit_segmentations, codebook),
        assign_codebook_trials(online_batches, online_segmentations, codebook),
    )


def _class_distance_matrix(features: np.ndarray, labels: np.ndarray) -> np.ndarray:
    classes = sorted(set(np.asarray(labels, dtype=np.int64).tolist()))
    means = np.stack(
        [np.asarray(features)[np.asarray(labels) == value].mean(axis=0) for value in classes]
    )
    distances = np.linalg.norm(means[:, None, :] - means[None, :, :], axis=2)
    maximum = float(np.max(distances))
    if maximum > 0:
        distances /= maximum
    return distances.astype(np.float64)


def _cross_subject_trajectory_diagnostics(
    features: np.ndarray,
    activity_labels: Sequence[int],
    subject_ids: Sequence[int],
    raw_cluster_predictions: Sequence[int],
) -> dict[str, Any]:
    """Post-truth diagnostics for cross-subject trajectory consistency.

    ``features`` and ``raw_cluster_predictions`` must already be frozen by the
    label-free readout.  Activity labels are used only to score whether trials
    from different subjects but the same activity are closer than trials from
    different activities.  Nothing returned here is consumed by a predictor.
    """

    matrix = np.asarray(features, dtype=np.float64)
    labels = np.asarray(activity_labels)
    subjects = np.asarray(subject_ids)
    clusters = np.asarray(raw_cluster_predictions)
    if matrix.ndim != 2:
        raise ValueError("Cross-subject diagnostic features must be a 2D matrix.")
    row_count = int(matrix.shape[0])
    expected_shape = (row_count,)
    for name, values in (
        ("activity_labels", labels),
        ("subject_ids", subjects),
        ("raw_cluster_predictions", clusters),
    ):
        if values.shape != expected_shape:
            raise ValueError(
                f"Cross-subject diagnostic {name} must have shape "
                f"{expected_shape}, got {values.shape}."
            )
    if row_count and matrix.shape[1] < 1:
        raise ValueError("Cross-subject diagnostic features have zero columns.")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("Cross-subject diagnostic features contain non-finite values.")
    for name, values in (
        ("activity_labels", labels),
        ("subject_ids", subjects),
        ("raw_cluster_predictions", clusters),
    ):
        if not np.issubdtype(values.dtype, np.integer):
            raise ValueError(f"Cross-subject diagnostic {name} must be integer-valued.")
        if np.any(values < 0):
            raise ValueError(f"Cross-subject diagnostic {name} contains a negative ID.")
    labels = labels.astype(np.int64, copy=False)
    subjects = subjects.astype(np.int64, copy=False)
    clusters = clusters.astype(np.int64, copy=False)

    class_count = int(len(np.unique(labels))) if row_count else 0
    subject_count = int(len(np.unique(subjects))) if row_count else 0
    cluster_count = int(len(np.unique(clusters))) if row_count else 0
    common = {
        "trial_count": row_count,
        "feature_dim": int(matrix.shape[1]),
        "activity_class_count": class_count,
        "subject_count": subject_count,
        "raw_cluster_count": cluster_count,
        "test_feature_sha256": _array_sha256(matrix.astype(np.float32)),
        "raw_cluster_prediction_sha256": _array_sha256(clusters),
    }

    unavailable_reason = (
        "no_test_trials"
        if row_count == 0
        else "requires_at_least_two_subjects"
        if subject_count < 2
        else None
    )
    distance_effect: dict[str, Any] = {
        "available": False,
        "reason": unavailable_reason,
        "distance": "euclidean_on_frozen_train_fitted_l2_readout_features",
        "cross_subject_pair_count": 0,
        "same_activity_pair_count": 0,
        "different_activity_pair_count": 0,
    }
    nearest_neighbour: dict[str, Any] = {
        "available": False,
        "reason": unavailable_reason,
        "tie_policy": (
            "mean correctness across every equally nearest other-subject trial"
        ),
        "tie_absolute_tolerance": 1e-12,
        "query_count": 0,
    }
    cluster_subject_nmi: dict[str, Any] = {
        "available": False,
        "reason": unavailable_reason,
        "average_method": "arithmetic",
        "interpretation": (
            "higher values mean raw trial clusters encode more subject identity"
        ),
    }
    if unavailable_reason is not None:
        return {
            **common,
            "cross_subject_distance_effect": distance_effect,
            "tie_aware_cross_subject_1nn": nearest_neighbour,
            "cluster_subject_nmi": cluster_subject_nmi,
            "post_truth_join_diagnostic_only": True,
            "used_to_fit_or_modify_predictions": False,
        }

    squared = (
        np.sum(matrix * matrix, axis=1, keepdims=True)
        + np.sum(matrix * matrix, axis=1)[None, :]
        - 2.0 * matrix @ matrix.T
    )
    distances = np.sqrt(np.maximum(squared, 0.0))
    left, right = np.triu_indices(row_count, k=1)
    cross_mask = subjects[left] != subjects[right]
    left = left[cross_mask]
    right = right[cross_mask]
    cross_distances = distances[left, right]
    same_mask = labels[left] == labels[right]
    same = cross_distances[same_mask]
    different = cross_distances[~same_mask]
    distance_effect.update(
        {
            "cross_subject_pair_count": int(len(cross_distances)),
            "same_activity_pair_count": int(len(same)),
            "different_activity_pair_count": int(len(different)),
        }
    )
    if len(same) and len(different):
        comparison = same[:, None] - different[None, :]
        tolerance = 1e-12
        probability = float(
            np.mean(comparison < -tolerance)
            + 0.5 * np.mean(np.abs(comparison) <= tolerance)
        )
        same_mean = float(np.mean(same))
        different_mean = float(np.mean(different))
        distance_effect.update(
            {
                "available": True,
                "reason": None,
                "same_activity_distance_mean": same_mean,
                "same_activity_distance_median": float(np.median(same)),
                "different_activity_distance_mean": different_mean,
                "different_activity_distance_median": float(
                    np.median(different)
                ),
                "mean_margin_different_minus_same": different_mean - same_mean,
                "separation_ratio_different_over_same": (
                    different_mean / max(same_mean, 1e-12)
                ),
                "probability_same_distance_is_smaller": probability,
                "rank_separation_effect": 2.0 * probability - 1.0,
            }
        )
    else:
        distance_effect["reason"] = (
            "requires_both_same_and_different_activity_cross_subject_pairs"
        )

    query_scores: list[float] = []
    tied_counts: list[int] = []
    for index in range(row_count):
        candidates = np.flatnonzero(subjects != subjects[index])
        if not len(candidates):
            continue
        candidate_distances = distances[index, candidates]
        minimum = float(np.min(candidate_distances))
        tied = candidates[
            np.isclose(candidate_distances, minimum, rtol=0.0, atol=1e-12)
        ]
        tied_counts.append(int(len(tied)))
        query_scores.append(float(np.mean(labels[tied] == labels[index])))
    if query_scores:
        nearest_neighbour.update(
            {
                "available": True,
                "reason": None,
                "accuracy": float(np.mean(query_scores)),
                "query_count": int(len(query_scores)),
                "mean_tied_neighbour_count": float(np.mean(tied_counts)),
                "maximum_tied_neighbour_count": int(max(tied_counts)),
                "query_count_with_ties": int(
                    np.sum(np.asarray(tied_counts, dtype=np.int64) > 1)
                ),
            }
        )
    else:
        nearest_neighbour["reason"] = "no_cross_subject_neighbour_for_any_query"

    cluster_subject_nmi.update(
        {
            "available": True,
            "reason": None,
            "normalized_mutual_information": float(
                normalized_mutual_info_score(
                    subjects, clusters, average_method="arithmetic"
                )
            ),
        }
    )
    return {
        **common,
        "cross_subject_distance_effect": distance_effect,
        "tie_aware_cross_subject_1nn": nearest_neighbour,
        "cluster_subject_nmi": cluster_subject_nmi,
        "post_truth_join_diagnostic_only": True,
        "used_to_fit_or_modify_predictions": False,
    }


def _cross_subject_diagnostic_csv_row(
    arm_id: str, readout_variant: str, diagnostic: Mapping[str, Any]
) -> dict[str, Any]:
    """Flatten one diagnostic record into a stable one-row CSV schema."""

    effect = diagnostic["cross_subject_distance_effect"]
    nearest = diagnostic["tie_aware_cross_subject_1nn"]
    nmi = diagnostic["cluster_subject_nmi"]
    return {
        "arm_id": str(arm_id),
        "readout_variant": str(readout_variant),
        "trial_count": int(diagnostic["trial_count"]),
        "feature_dim": int(diagnostic["feature_dim"]),
        "activity_class_count": int(diagnostic["activity_class_count"]),
        "subject_count": int(diagnostic["subject_count"]),
        "raw_cluster_count": int(diagnostic["raw_cluster_count"]),
        "distance_effect_available": bool(effect["available"]),
        "distance_effect_reason": effect.get("reason"),
        "cross_subject_pair_count": int(effect["cross_subject_pair_count"]),
        "same_activity_pair_count": int(effect["same_activity_pair_count"]),
        "different_activity_pair_count": int(
            effect["different_activity_pair_count"]
        ),
        "same_activity_distance_mean": effect.get(
            "same_activity_distance_mean"
        ),
        "different_activity_distance_mean": effect.get(
            "different_activity_distance_mean"
        ),
        "mean_margin_different_minus_same": effect.get(
            "mean_margin_different_minus_same"
        ),
        "separation_ratio_different_over_same": effect.get(
            "separation_ratio_different_over_same"
        ),
        "probability_same_distance_is_smaller": effect.get(
            "probability_same_distance_is_smaller"
        ),
        "rank_separation_effect": effect.get("rank_separation_effect"),
        "cross_subject_1nn_available": bool(nearest["available"]),
        "cross_subject_1nn_reason": nearest.get("reason"),
        "cross_subject_1nn_accuracy": nearest.get("accuracy"),
        "cross_subject_1nn_query_count": int(nearest["query_count"]),
        "cross_subject_1nn_query_count_with_ties": nearest.get(
            "query_count_with_ties"
        ),
        "cluster_subject_nmi_available": bool(nmi["available"]),
        "cluster_subject_nmi_reason": nmi.get("reason"),
        "cluster_subject_nmi": nmi.get("normalized_mutual_information"),
        "test_feature_sha256": str(diagnostic["test_feature_sha256"]),
        "raw_cluster_prediction_sha256": str(
            diagnostic["raw_cluster_prediction_sha256"]
        ),
        "post_truth_join_diagnostic_only": bool(
            diagnostic["post_truth_join_diagnostic_only"]
        ),
        "used_to_fit_or_modify_predictions": bool(
            diagnostic["used_to_fit_or_modify_predictions"]
        ),
    }


def _arm_metadata(
    arm_id: str,
    fit_trials: Sequence[PrimitiveTrial],
    online_trials: Sequence[PrimitiveTrial],
    catalog: ParentCatalogState | None,
) -> dict:
    fit_counts = np.asarray([len(item.child_tokens) for item in fit_trials], dtype=np.int64)
    online_counts = np.asarray(
        [len(item.child_tokens) for item in online_trials], dtype=np.int64
    )
    fit_durations = np.concatenate(
        [(item.ends - item.starts).astype(np.int64) for item in fit_trials]
    )
    online_durations = np.concatenate(
        [(item.ends - item.starts).astype(np.int64) for item in online_trials]
    )
    fit_token_counts = np.bincount(
        np.concatenate([item.child_tokens for item in fit_trials]), minlength=32
    ).astype(np.int64)
    online_token_counts = np.bincount(
        np.concatenate([item.child_tokens for item in online_trials]), minlength=32
    ).astype(np.int64)
    if len(fit_token_counts) != 32 or len(online_token_counts) != 32:
        raise RuntimeError(f"Arm {arm_id} produced a child token outside K=32.")
    return {
        "arm_id": arm_id,
        "segmentation": {
            "fit_trial_count": len(fit_trials),
            "eval_trial_count": len(online_trials),
            "fit_segment_count": int(fit_counts.sum()),
            "eval_segment_count": int(online_counts.sum()),
            "fit_segments_per_trial_mean": float(fit_counts.mean()),
            "eval_segments_per_trial_mean": float(online_counts.mean()),
            "fit_segment_duration_samples_median": float(np.median(fit_durations)),
            "eval_segment_duration_samples_median": float(
                np.median(online_durations)
            ),
        },
        "child_codebook_usage": {
            "K": 32,
            "fit_used_token_count": int(np.count_nonzero(fit_token_counts)),
            "eval_used_token_count": int(np.count_nonzero(online_token_counts)),
            "fit_token_counts": fit_token_counts.tolist(),
            "eval_token_counts": online_token_counts.tolist(),
        },
        "parent_catalog": {
            "parent_count": (
                0 if catalog is None else len(catalog.registered_patterns)
            ),
            "catalog_mode": None if catalog is None else catalog.catalog_mode,
            "catalog_state_sha256": None if catalog is None else catalog.state_hash(),
        },
    }


def _raw_freeze_payload(
    readouts: Mapping[tuple[str, str], Mapping[str, Any]],
    train_ids: Sequence[int],
    test_ids: Sequence[int],
) -> dict[str, np.ndarray]:
    payload: dict[str, np.ndarray] = {
        "train_trial_ids": np.asarray(train_ids, dtype=np.int64),
        "test_trial_ids": np.asarray(test_ids, dtype=np.int64),
    }
    for (arm, variant), item in sorted(readouts.items()):
        readout = item["readout"]
        payload[f"train_raw__{arm}__{variant}"] = np.asarray(
            readout.train_predictions, dtype=np.int64
        )
        payload[f"test_raw__{arm}__{variant}"] = np.asarray(
            readout.test_predictions, dtype=np.int64
        )
        payload[f"train_features__{arm}__{variant}"] = np.asarray(
            item["train_features"], dtype=np.float32
        )
        payload[f"test_features__{arm}__{variant}"] = np.asarray(
            item["test_features"], dtype=np.float32
        )
    return payload


def _export_parent_catalogs(
    path: Path, catalogs: Mapping[str, ParentCatalogState | None]
) -> None:
    rows = []
    for arm, catalog in sorted(catalogs.items()):
        if catalog is None:
            continue
        for pattern in catalog.patterns:
            rows.append(
                {
                    "arm_id": arm,
                    "catalog_mode": catalog.catalog_mode,
                    "parent_id": pattern.parent_id or "",
                    "registered": pattern.parent_id is not None,
                    "event_kind": pattern.kind,
                    "left_child": pattern.left_token,
                    "right_child": pattern.right_token,
                    "occurrences": pattern.occurrences,
                    "nonoverlap_occurrences": pattern.nonoverlap_occurrences,
                    "trial_count": pattern.trial_count,
                    "subject_count": pattern.subject_count,
                    "npmi": pattern.npmi,
                    "mdl_gain_bits": pattern.mdl_gain_bits,
                    "loso_stability": pattern.loso_stability,
                    "gate_checks": pattern.gate_checks,
                }
            )
    if rows:
        _write_csv(path, rows)
    else:
        _write_csv(
            path,
            [
                {
                    "arm_id": "none",
                    "catalog_mode": "no_parent_catalog_requested",
                    "parent_id": "",
                    "registered": False,
                }
            ],
        )


def _export_trajectories(
    path: Path,
    arm_fit: Mapping[str, Sequence[PrimitiveTrial]],
    arm_online: Mapping[str, Sequence[PrimitiveTrial]],
    train_ids: Sequence[int],
    test_ids: Sequence[int],
    truth_source: SourceSignalRepository,
) -> None:
    train_set = set(int(value) for value in train_ids)
    test_set = set(int(value) for value in test_ids)
    arm_maps = {
        arm: {
            item.trial_id: ("offline_fit", item) for item in arm_fit[arm]
        }
        | {
            item.trial_id: (
                "cumulative_online_train"
                if item.trial_id in train_set
                else "session_2_test",
                item,
            )
            for item in arm_online[arm]
        }
        for arm in arm_fit
    }
    all_ids = sorted(set().union(*(set(value) for value in arm_maps.values())))
    with path.open("w", encoding="utf-8") as handle:
        for trial_id in all_ids:
            metadata = truth_source.metadata(trial_id)
            arms = {}
            role = None
            for arm, mapping in sorted(arm_maps.items()):
                if trial_id not in mapping:
                    continue
                current_role, item = mapping[trial_id]
                role = current_role if role is None else role
                overlay = item.parent_overlay
                arms[arm] = {
                    "boundaries_samples": np.r_[item.starts, item.ends[-1]].tolist(),
                    "durations_samples": (item.ends - item.starts).tolist(),
                    "child_sequence": item.child_tokens.tolist(),
                    "child_quantization_distances": item.child_distances.tolist(),
                    "parent_sequence": (
                        [] if overlay is None else list(overlay.parent_event_sequence)
                    ),
                    "parent_occurrences": (
                        []
                        if overlay is None
                        else [value.to_dict() for value in overlay.parent_occurrences]
                    ),
                    "parent_overlay_preserved_children": (
                        True
                        if overlay is None
                        else bool(np.array_equal(overlay.child_tokens, item.child_tokens))
                    ),
                }
            record = {
                "trial_global_id": trial_id,
                "split_role": role,
                "subject_id": int(metadata["subject_id"]),
                "activity_label_0based": int(metadata["label"]),
                "activity_name": str(metadata["activity_name"]),
                "trial_number": int(metadata["trial_number"]),
                "arms": arms,
            }
            handle.write(json.dumps(_jsonable(record), ensure_ascii=False) + "\n")


def _export_representatives(
    jsonl_path: Path,
    fit_tokenized: Sequence[TokenizedTrial],
    signal_by_trial: Mapping[int, TrialSignal],
    primitive_num: int,
) -> list[dict]:
    candidates: dict[int, tuple[float, TokenizedTrial, int]] = {}
    for trial in fit_tokenized:
        for index, (token, distance) in enumerate(
            zip(trial.primitive_tokens.tolist(), trial.nearest_center_distances.tolist())
        ):
            value = (float(distance), trial, int(index))
            if int(token) not in candidates or value[0] < candidates[int(token)][0]:
                candidates[int(token)] = value
    if set(candidates) != set(range(int(primitive_num))):
        raise RuntimeError("Every child codebook centre needs one observed representative.")
    records = []
    with jsonl_path.open("w", encoding="utf-8") as handle:
        for token in range(int(primitive_num)):
            distance, trial, index = candidates[token]
            start = int(trial.child_features.start_samples[index])
            end = int(trial.child_features.end_samples_exclusive[index])
            raw = signal_by_trial[int(trial.trial_id)].signal[:, start:end]
            record = {
                "child_token": token,
                "trial_global_id": int(trial.trial_id),
                "subject_id": int(trial.subject_id),
                "start_sample": start,
                "end_sample_exclusive": end,
                "sample_count": end - start,
                "nearest_center_distance": distance,
                "channel_names": list(CHANNEL_NAMES),
                "raw_physical_sequence_6xT": raw.tolist(),
            }
            records.append(record)
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    return records


def _plot_representatives(path: Path, records: Sequence[Mapping[str, Any]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    columns = 4
    rows = int(math.ceil(len(records) / columns))
    fig, axes = plt.subplots(
        rows,
        columns,
        figsize=(4.4 * columns, 2.8 * rows),
        squeeze=False,
        constrained_layout=True,
    )
    colors = plt.get_cmap("tab10")
    for axis, record in zip(axes.ravel(), records):
        raw = np.asarray(record["raw_physical_sequence_6xT"], dtype=np.float64)
        shape = _resample_signal(raw, 64)
        shape = (shape - shape.mean(axis=1, keepdims=True)) / np.maximum(
            shape.std(axis=1, keepdims=True), 1e-6
        )
        x = np.linspace(0.0, 1.0, shape.shape[1])
        for channel in range(6):
            axis.plot(
                x,
                shape[channel] + 4.0 * channel,
                color=colors(channel),
                linewidth=0.9,
            )
        axis.set_title(
            f"child {record['child_token']} | T{record['trial_global_id']} | "
            f"n={record['sample_count']} | d={record['nearest_center_distance']:.3f}",
            fontsize=8,
        )
        axis.set_yticks(4.0 * np.arange(6), CHANNEL_NAMES, fontsize=6)
        axis.set_xticks([0.0, 0.5, 1.0])
        axis.grid(alpha=0.15)
    for axis in axes.ravel()[len(records) :]:
        axis.axis("off")
    fig.suptitle(
        "Nearest-centre child representatives (shape-normalized; raw values in JSONL)"
    )
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_trajectories(
    path: Path,
    arm_online: Mapping[str, Sequence[PrimitiveTrial]],
    test_ids: Sequence[int],
    labels: Mapping[int, int],
    subjects: Mapping[int, int],
    names: Mapping[int, str],
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ordered = sorted(
        (int(value) for value in test_ids),
        key=lambda value: (labels[value], subjects[value], value),
    )
    arms = list(arm_online)
    fig, axes = plt.subplots(
        len(arms),
        1,
        figsize=(20, max(7.0, 0.19 * len(ordered) * len(arms))),
        squeeze=False,
        constrained_layout=True,
    )
    palette = plt.get_cmap("tab20")
    for axis, arm in zip(axes.ravel(), arms):
        mapping = {item.trial_id: item for item in arm_online[arm]}
        for row, trial_id in enumerate(ordered):
            item = mapping[trial_id]
            for token, start, end in zip(item.child_tokens, item.starts, item.ends):
                axis.barh(
                    row,
                    (int(end) - int(start)) / 100.0,
                    left=int(start) / 100.0,
                    height=0.78,
                    color=palette((int(token) % 20) / 19.0),
                    linewidth=0,
                )
            if item.parent_overlay is not None:
                for occurrence in item.parent_overlay.parent_occurrences:
                    axis.plot(
                        [
                            occurrence.span_start_sample / 100.0,
                            occurrence.span_end_sample_exclusive / 100.0,
                        ],
                        [row, row],
                        color="black",
                        linewidth=1.6,
                        solid_capstyle="butt",
                    )
        parent_count = sum(
            0
            if item.parent_overlay is None
            else len(item.parent_overlay.parent_occurrences)
            for item in mapping.values()
            if item.trial_id in set(ordered)
        )
        axis.set_title(f"{arm}: child sequence; black overlay=parent spans ({parent_count})")
        axis.set_yticks(
            np.arange(len(ordered)),
            [f"{names[value]} | S{subjects[value]} | T{value}" for value in ordered],
            fontsize=5,
        )
        axis.invert_yaxis()
        axis.set_ylabel("Session-2 test trial")
    axes.ravel()[-1].set_xlabel("visible trial time (seconds)")
    fig.suptitle("Motion-primitive trajectories: variable child durations and parent overlays")
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_waveform_examples(
    path: Path,
    e2_trials: Sequence[PrimitiveTrial],
    signal_by_id: Mapping[int, TrialSignal],
    test_ids: Sequence[int],
    labels: Mapping[int, int],
    names: Mapping[int, str],
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    mapping = {item.trial_id: item for item in e2_trials}
    representatives = [
        min(value for value in test_ids if labels[int(value)] == label)
        for label in sorted(set(labels[int(value)] for value in test_ids))
    ]
    fig, axes = plt.subplots(
        len(representatives),
        1,
        figsize=(20, 2.6 * len(representatives)),
        squeeze=False,
        constrained_layout=True,
    )
    colors = plt.get_cmap("tab10")
    for axis, trial_id in zip(axes.ravel(), representatives):
        item = mapping[int(trial_id)]
        signal = np.asarray(signal_by_id[int(trial_id)].signal, dtype=np.float64)
        normalized = (signal - signal.mean(axis=1, keepdims=True)) / np.maximum(
            signal.std(axis=1, keepdims=True), 1e-6
        )
        time = np.arange(signal.shape[1]) / 100.0
        for channel in range(6):
            axis.plot(
                time,
                normalized[channel] + 5.0 * channel,
                color=colors(channel),
                linewidth=0.7,
                label=CHANNEL_NAMES[channel] if trial_id == representatives[0] else None,
            )
        for index, boundary in enumerate(item.starts[1:]):
            kind = item.event_kinds[index] if item.event_kinds else "none"
            axis.axvline(
                int(boundary) / 100.0,
                color="#d62728" if kind == "peak" else "#1f77b4",
                alpha=0.55,
                linewidth=0.8,
            )
        for token, start, end in zip(item.child_tokens, item.starts, item.ends):
            if end - start >= 20:
                axis.text(
                    (start + end) / 200.0,
                    27.0,
                    str(int(token)),
                    fontsize=5,
                    ha="center",
                    va="top",
                )
        axis.set_title(
            f"{names[int(trial_id)]} | trial {trial_id} | {len(item.child_tokens)} children; "
            "red=peak, blue=valley",
            fontsize=9,
        )
        axis.set_yticks(5.0 * np.arange(6), CHANNEL_NAMES, fontsize=6)
        axis.set_xlabel("seconds")
        axis.grid(alpha=0.1)
    fig.suptitle("E2 sample-level waveform decomposition (one test trial per activity)")
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_heatmaps(
    path: Path,
    matrices: Mapping[tuple[str, str], np.ndarray],
    activity_names: Sequence[str],
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    keys = sorted(matrices)
    columns = min(4, len(keys))
    rows = int(math.ceil(len(keys) / columns))
    fig, axes = plt.subplots(
        rows,
        columns,
        figsize=(5.2 * columns, 4.8 * rows),
        squeeze=False,
        constrained_layout=True,
    )
    for axis, key in zip(axes.ravel(), keys):
        image = axis.imshow(matrices[key], vmin=0.0, vmax=1.0, cmap="magma")
        axis.set_title(f"{key[0]} / {key[1]}")
        axis.set_xticks(
            range(len(activity_names)), activity_names, rotation=65, ha="right", fontsize=6
        )
        axis.set_yticks(range(len(activity_names)), activity_names, fontsize=6)
        fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    for axis in axes.ravel()[len(keys) :]:
        axis.axis("off")
    fig.suptitle("Session-2 activity centroid distances in frozen trajectory-feature space")
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_confusions(
    path: Path,
    metrics_by_arm: Mapping[str, Mapping[str, Any]],
    activity_names: Sequence[str],
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    arms = sorted(metrics_by_arm)
    columns = min(4, len(arms))
    rows = int(math.ceil(len(arms) / columns))
    fig, axes = plt.subplots(
        rows,
        columns,
        figsize=(5.2 * columns, 4.8 * rows),
        squeeze=False,
        constrained_layout=True,
    )
    for axis, arm in zip(axes.ravel(), arms):
        metrics = metrics_by_arm[arm]
        matrix = np.asarray(metrics["confusion_counts"], dtype=np.float64)
        matrix = np.divide(
            matrix,
            matrix.sum(axis=1, keepdims=True),
            out=np.zeros_like(matrix),
            where=matrix.sum(axis=1, keepdims=True) > 0,
        )
        image = axis.imshow(matrix, vmin=0.0, vmax=1.0, cmap="Blues")
        axis.set_title(
            f"{arm}/state\nAll={metrics['all_accuracy']:.3f}, "
            f"Old={metrics['old_accuracy']:.3f}, New={metrics['new_accuracy']:.3f}"
        )
        axis.set_xticks(
            range(len(activity_names)), activity_names, rotation=65, ha="right", fontsize=6
        )
        axis.set_yticks(range(len(activity_names)), activity_names, fontsize=6)
        fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    for axis in axes.ravel()[len(arms) :]:
        axis.axis("off")
    fig.suptitle("Session-2 globally aligned confusion matrices (primary state descriptor)")
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_segment_distributions(
    path: Path,
    arm_fit: Mapping[str, Sequence[PrimitiveTrial]],
    arm_online: Mapping[str, Sequence[PrimitiveTrial]],
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    arms = list(arm_fit)
    fig, axes = plt.subplots(
        len(arms), 2, figsize=(12, 3.0 * len(arms)), squeeze=False, constrained_layout=True
    )
    for row, arm in enumerate(arms):
        for split, trials, color in (
            ("offline fit", arm_fit[arm], "#3182ce"),
            ("held-out subject", arm_online[arm], "#dd6b20"),
        ):
            counts = [len(item.child_tokens) for item in trials]
            durations = np.concatenate(
                [(item.ends - item.starts).astype(np.float64) / 100.0 for item in trials]
            )
            axes[row, 0].hist(counts, bins=20, alpha=0.5, label=split, color=color)
            axes[row, 1].hist(durations, bins=30, alpha=0.5, label=split, color=color)
        axes[row, 0].set_title(f"{arm}: children per trial")
        axes[row, 1].set_title(f"{arm}: child duration (seconds)")
        axes[row, 0].legend(fontsize=7)
        axes[row, 1].legend(fontsize=7)
        for axis in axes[row]:
            axis.grid(alpha=0.15)
    fig.suptitle("Segmentation count and variable-duration distributions")
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="One-fold peak/valley child-and-parent trajectory experiment."
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--npz-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--expected-profile", choices=("A0", "A2", "A3"), required=True)
    parser.add_argument("--orchestrator-wiring-sha256", default="")
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument(
        "--window-size-samples",
        type=int,
        choices=(128, 256),
        default=256,
        help="Explicit temporal length of the registered NPZ/encoder window grid.",
    )
    parser.add_argument(
        "--window-stride-samples",
        type=int,
        choices=(64, 128),
        default=128,
        help="Explicit stride of the registered NPZ window grid.",
    )
    parser.add_argument("--arms", default="E0,E1,E2,E3,E4")
    parser.add_argument("--old-class-count", type=int, default=6)
    parser.add_argument("--primitive-num", type=int, default=32)
    parser.add_argument("--pca-dim", type=int, default=64)
    parser.add_argument(
        "--child-feature-source",
        choices=("content_embedding", "hybrid", "raw_shape"),
        default="content_embedding",
    )
    parser.add_argument("--trial-cluster-count", type=int, default=10)
    parser.add_argument("--sample-rate-hz", type=float, default=100.0)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--smooth-seconds", type=float, default=0.15)
    parser.add_argument("--prominence-mad", type=float, default=1.5)
    parser.add_argument("--extrema-distance-seconds", type=float, default=0.20)
    parser.add_argument("--vote-tolerance-seconds", type=float, default=0.10)
    parser.add_argument("--minimum-axes", type=int, default=2)
    parser.add_argument("--minimum-segment-seconds", type=float, default=0.25)
    parser.add_argument("--feature-confirm-quantile", type=float, default=0.50)
    parser.add_argument("--feature-context-seconds", type=float, default=0.50)
    parser.add_argument("--shape-points", type=int, default=64)
    parser.add_argument("--parent-min-occurrences", type=int, default=10)
    parser.add_argument("--parent-min-trials", type=int, default=6)
    parser.add_argument("--parent-min-subjects", type=int, default=2)
    parser.add_argument("--parent-min-npmi", type=float, default=0.0)
    parser.add_argument("--parent-min-mdl-gain", type=float, default=0.0)
    parser.add_argument("--parent-min-loso-stability", type=float, default=0.8)
    parser.add_argument(
        "--matched-random-controls", action=argparse.BooleanOptionalAction, default=True
    )
    return parser


def main() -> None:
    result = run(build_parser().parse_args())
    print(json.dumps(_jsonable(result), ensure_ascii=False, indent=2), flush=True)


def run(args: argparse.Namespace) -> dict:
    arms = _parse_arms(args.arms)
    requested_window_size, requested_stride = _validate_registered_window_grid(
        args.window_size_samples, args.window_stride_samples
    )
    if int(args.fold) not in range(1, 8):
        raise ValueError("--fold must be in 1..7.")
    if int(args.old_class_count) != 6:
        raise ValueError("This registered experiment is pinned to six old classes.")
    if int(args.primitive_num) != 32:
        raise ValueError("This registered experiment is pinned to K=32 children.")
    if int(args.pca_dim) != 64:
        raise ValueError("The historical comparison is pinned to PCA64.")
    if int(args.trial_cluster_count) != 10:
        raise ValueError("The Session-2 readout is pinned to K=10 trial clusters.")
    if not math.isclose(float(args.sample_rate_hz), 100.0, abs_tol=1e-12):
        raise ValueError("The USC-HAD experiment is pinned to 100 Hz.")
    if int(args.batch_size) < 1:
        raise ValueError("Batch size must be positive.")
    if not 1 <= int(args.minimum_axes) <= 6:
        raise ValueError("E2 --minimum-axes must lie in 1..6; E1 internally uses zero.")
    if not 0.0 <= float(args.feature_confirm_quantile) <= 1.0:
        raise ValueError("Feature-confirm quantile must lie in [0,1].")
    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    npz_path = Path(args.npz_path).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    if not checkpoint_path.is_file() or not npz_path.is_file():
        raise FileNotFoundError(
            f"Missing checkpoint or NPZ: {checkpoint_path}, {npz_path}."
        )
    if output_dir.exists():
        raise FileExistsError(f"Output directory already exists: {output_dir}")

    from experiments.motion_primitive.run_peak_valley_hierarchy_cv import (
        _member_source_fingerprints,
        _run_identity,
        _runner_cli_wiring_sha256,
        _source_fingerprints,
        _validate_profile_semantics,
    )

    expected_wiring = _runner_cli_wiring_sha256()
    if (
        str(args.orchestrator_wiring_sha256).strip()
        and str(args.orchestrator_wiring_sha256).strip() != expected_wiring
    ):
        raise RuntimeError(
            "CV/member CLI wiring fingerprint differs; do not mix a stale wrapper "
            "with this single-run implementation."
        )
    npz_hash = sha256_file(npz_path)
    profile_audit = _validate_profile_semantics(
        checkpoint_path,
        str(args.expected_profile).upper(),
        int(args.fold),
        int(args.seed),
        npz_hash,
    )
    checkpoint = load_torch_checkpoint(checkpoint_path)
    metadata = checkpoint.get("experiment_metadata")
    if not isinstance(metadata, Mapping):
        raise RuntimeError("Motion checkpoint lacks experiment_metadata.")
    checkpoint_window_size = metadata.get("uschad_window_size")
    if checkpoint_window_size is None:
        raise RuntimeError("Motion checkpoint lacks uschad_window_size metadata.")
    if int(checkpoint_window_size) != requested_window_size:
        raise RuntimeError(
            "Explicit window size disagrees with motion checkpoint metadata: "
            f"requested={requested_window_size}, checkpoint={checkpoint_window_size}."
        )
    fit_subjects = _metadata_subjects(metadata, "uschad_train_subjects")
    validation_subjects = _metadata_subjects(metadata, "offline_val_subjects")
    eval_subjects = _metadata_subjects(metadata, "uschad_test_subjects")
    if (
        set(fit_subjects) & set(validation_subjects)
        or set(fit_subjects) & set(eval_subjects)
        or set(validation_subjects) & set(eval_subjects)
    ):
        raise RuntimeError("Checkpoint fit/validation/evaluation subjects overlap.")
    if metadata.get("uschad_recompute_norm_from_train_subjects") is not True:
        raise RuntimeError("Fold-train-only normalization is required.")
    if metadata.get("outer_test_used_during_encoder_training") is not False:
        raise RuntimeError("Checkpoint records outer-test access during training.")

    grid = _load_numeric_grid(
        npz_path,
        expected_window_size_samples=requested_window_size,
        expected_stride_samples=requested_stride,
    )
    fit_subject_mask = np.isin(grid.subject_ids, fit_subjects)
    fit_window_mask = fit_subject_mask & (grid.labels < int(args.old_class_count))
    if not np.any(fit_window_mask):
        raise RuntimeError("No train-subject old-class windows were selected.")
    fit_trial_ids = sorted(
        set(grid.trial_ids[fit_window_mask].astype(int).tolist())
    )
    # Labels are permitted only to define the supervised old-class Stage-0 set.
    # Remove every held-out-subject label from the predictor-facing grid before
    # constructing signals, codebooks, parents or trial descriptors.
    sanitized_labels = grid.labels.copy()
    sanitized_labels[~fit_subject_mask] = -1
    grid = replace(grid, labels=sanitized_labels)

    manifest = build_session2_manifest_v2(
        npz_path,
        fold=int(args.fold),
        fit_subjects=fit_subjects,
        eval_subjects=eval_subjects,
        validation_subjects=validation_subjects,
        seed=int(args.seed),
        window_size_samples=requested_window_size,
    )
    online_train_ids = [int(value) for value in manifest["cumulative_train_trial_ids"]]
    test_ids = [int(value) for value in manifest["session_2_test_trial_ids"]]
    online_ids = online_train_ids + test_ids
    if len(online_train_ids) != 48 or len(test_ids) != 52:
        raise RuntimeError("Registered Session-2 48/52 trial counts changed.")
    if set(online_train_ids) & set(test_ids):
        raise RuntimeError("Session-2 online train/test trial ids overlap.")
    if set(fit_trial_ids) & set(online_ids):
        raise RuntimeError("Offline-fit and held-out-subject trial ids overlap.")
    online_rows = np.isin(grid.trial_ids, online_ids)
    if not np.all(grid.labels[online_rows] == -1):
        raise RuntimeError("Held-out activity labels survived predictor-grid sanitization.")
    online_subject_by_trial = {
        int(key): int(value)
        for key, value in manifest["subject_by_feature_trial_protocol_only"].items()
    }
    if set(online_subject_by_trial) != set(online_ids):
        raise RuntimeError("Session manifest subject map differs from online trial ids.")
    fit_subject_by_trial: dict[int, int] = {}
    for trial_id in fit_trial_ids:
        subjects = np.unique(grid.subject_ids[_trial_rows(grid, trial_id)])
        if len(subjects) != 1 or int(subjects[0]) not in fit_subjects:
            raise RuntimeError(f"Offline fit trial {trial_id} has invalid subject metadata.")
        fit_subject_by_trial[trial_id] = int(subjects[0])
    for trial_id, expected_subject in online_subject_by_trial.items():
        subjects = np.unique(grid.subject_ids[_trial_rows(grid, trial_id)])
        if len(subjects) != 1 or int(subjects[0]) != int(expected_subject):
            raise RuntimeError("Protocol-only subject map disagrees with numeric grid.")

    fold_mean, fold_std, normalization_audit = _fold_normalization(
        grid,
        fit_window_mask,
        float(metadata.get("uschad_norm_eps", 1e-6)),
    )
    label_free_source = LabelFreeSourceSignalRepository(npz_path)
    all_predictor_ids = fit_trial_ids + online_ids
    selected_rows = np.concatenate(
        [_trial_rows(grid, trial_id) for trial_id in all_predictor_ids]
    )
    normalized = _normalized_windows(
        grid, selected_rows, fold_mean, fold_std
    )
    device = choose_device(str(args.device))
    encoder = build_frozen_motion_encoder(checkpoint, dict(metadata))
    encoded = encode_motion_windows(
        encoder, normalized, device, int(args.batch_size)
    )
    window_content: dict[int, np.ndarray] = {}
    window_boundary: dict[int, np.ndarray] = {}
    cursor = 0
    for trial_id in all_predictor_ids:
        count = len(_trial_rows(grid, trial_id))
        window_content[trial_id] = encoded["content"][cursor : cursor + count].copy()
        window_boundary[trial_id] = encoded["segmentation"][cursor : cursor + count].copy()
        cursor += count
    if cursor != len(selected_rows):
        raise RuntimeError("Window embedding split lost rows.")
    e0_window_content = window_content

    fit_trials_e1 = _make_trial_signals(
        fit_trial_ids,
        fit_subject_by_trial,
        label_free_source,
        grid,
        boundary_embeddings=None,
    )
    online_trials_e1 = _make_trial_signals(
        online_ids,
        online_subject_by_trial,
        label_free_source,
        grid,
        boundary_embeddings=None,
    )
    fit_trials_e2 = _make_trial_signals(
        fit_trial_ids,
        fit_subject_by_trial,
        label_free_source,
        grid,
        boundary_embeddings=window_boundary,
        feature_interpolation_step_samples=max(
            1,
            int(
                round(
                    float(args.sample_rate_hz)
                    * float(args.feature_context_seconds)
                    / 2.0
                )
            ),
        ),
    )
    online_trials_e2 = _make_trial_signals(
        online_ids,
        online_subject_by_trial,
        label_free_source,
        grid,
        boundary_embeddings=window_boundary,
        feature_interpolation_step_samples=max(
            1,
            int(
                round(
                    float(args.sample_rate_hz)
                    * float(args.feature_context_seconds)
                    / 2.0
                )
            ),
        ),
    )

    output_dir.mkdir(parents=True, exist_ok=False)
    print(
        f"[peak-valley] fold={args.fold} seed={args.seed} profile={args.expected_profile} "
        f"offline_trials={len(fit_trial_ids)} online=48 test=52",
        flush=True,
    )
    arm_fit: dict[str, list[PrimitiveTrial]] = {}
    arm_online: dict[str, list[PrimitiveTrial]] = {}
    arm_catalog: dict[str, ParentCatalogState | None] = {}
    state_files: list[str] = []
    codebooks: dict[str, CodebookState] = {}
    tokenized_fit_by_arm: dict[str, list[TokenizedTrial]] = {}
    tokenized_online_by_arm: dict[str, list[TokenizedTrial]] = {}
    segmentations_fit: dict[str, list[SegmentedTrial]] = {}
    segmentations_online: dict[str, list[SegmentedTrial]] = {}

    if "E0" in arms:
        e0_batches = _e0_codebook_batches(
            fit_trial_ids, fit_subject_by_trial, grid, e0_window_content
        )
        e0_codebook = _fit_e0_historical_codebook(
            e0_batches,
            primitive_num=32,
            pca_dim=int(args.pca_dim),
            seed=int(args.seed),
        )
        codebooks["E0"] = e0_codebook
        arm_fit["E0"] = _primitive_trials_e0(
            fit_trial_ids,
            fit_subject_by_trial,
            grid,
            e0_window_content,
            e0_codebook,
            label_free_source,
            float(args.sample_rate_hz),
        )
        arm_online["E0"] = _primitive_trials_e0(
            online_ids,
            online_subject_by_trial,
            grid,
            e0_window_content,
            e0_codebook,
            label_free_source,
            float(args.sample_rate_hz),
        )
        arm_catalog["E0"] = None
        save_state_json(output_dir / "e0_legacy_window_codebook_state.json", e0_codebook)
        state_files.append("e0_legacy_window_codebook_state.json")

    need_e1 = "E1" in arms
    need_e2_family = bool(set(arms) & {"E2", "E3", "E4"}) or (
        bool(args.matched_random_controls) and bool(set(arms) & {"E2", "E4"})
    )
    if need_e1:
        e1_config = PeakValleyConfig(
            sample_rate_hz=float(args.sample_rate_hz),
            smoothing_seconds=float(args.smooth_seconds),
            extrema_min_distance_seconds=float(args.extrema_distance_seconds),
            prominence_mad_multiplier=float(args.prominence_mad),
            axis_vote_tolerance_seconds=float(args.vote_tolerance_seconds),
            min_axis_votes=0,
            min_segment_seconds=float(args.minimum_segment_seconds),
            feature_confirmation=False,
            feature_context_seconds=float(args.feature_context_seconds),
            feature_score_quantile=float(args.feature_confirm_quantile),
            shape_points=int(args.shape_points),
        )
        (
            e1_segmenter,
            e1_codebook,
            e1_fit_tokens,
            e1_online_tokens,
            e1_fit_segments,
            e1_online_segments,
        ) = _child_pipeline(
            fit_trials_e1,
            online_trials_e1,
            e1_config,
            str(args.child_feature_source),
            encoder,
            device,
            int(args.batch_size),
            fold_mean,
            fold_std,
            int(args.window_size_samples),
            32,
            int(args.pca_dim),
            int(args.seed),
        )
        codebooks["E1"] = e1_codebook
        tokenized_fit_by_arm["E1"] = e1_fit_tokens
        tokenized_online_by_arm["E1"] = e1_online_tokens
        segmentations_fit["E1"] = e1_fit_segments
        segmentations_online["E1"] = e1_online_segments
        arm_fit["E1"] = _primitive_trials_from_tokenized(e1_fit_tokens, 32)
        arm_online["E1"] = _primitive_trials_from_tokenized(e1_online_tokens, 32)
        arm_catalog["E1"] = None
        save_state_json(output_dir / "e1_peak_valley_segmenter_state.json", e1_segmenter)
        save_state_json(output_dir / "e1_child_codebook_state.json", e1_codebook)
        state_files.extend(
            ["e1_peak_valley_segmenter_state.json", "e1_child_codebook_state.json"]
        )

    if need_e2_family:
        e2_config = PeakValleyConfig(
            sample_rate_hz=float(args.sample_rate_hz),
            smoothing_seconds=float(args.smooth_seconds),
            extrema_min_distance_seconds=float(args.extrema_distance_seconds),
            prominence_mad_multiplier=float(args.prominence_mad),
            axis_vote_tolerance_seconds=float(args.vote_tolerance_seconds),
            min_axis_votes=int(args.minimum_axes),
            min_segment_seconds=float(args.minimum_segment_seconds),
            feature_confirmation=True,
            feature_context_seconds=float(args.feature_context_seconds),
            feature_score_quantile=float(args.feature_confirm_quantile),
            shape_points=int(args.shape_points),
        )
        (
            e2_segmenter,
            e2_codebook,
            e2_fit_tokens,
            e2_online_tokens,
            e2_fit_segments,
            e2_online_segments,
        ) = _child_pipeline(
            fit_trials_e2,
            online_trials_e2,
            e2_config,
            str(args.child_feature_source),
            encoder,
            device,
            int(args.batch_size),
            fold_mean,
            fold_std,
            int(args.window_size_samples),
            32,
            int(args.pca_dim),
            int(args.seed),
        )
        codebooks["E2"] = e2_codebook
        tokenized_fit_by_arm["E2"] = e2_fit_tokens
        tokenized_online_by_arm["E2"] = e2_online_tokens
        segmentations_fit["E2"] = e2_fit_segments
        segmentations_online["E2"] = e2_online_segments
        if "E2" in arms:
            arm_fit["E2"] = _primitive_trials_from_tokenized(e2_fit_tokens, 32)
            arm_online["E2"] = _primitive_trials_from_tokenized(e2_online_tokens, 32)
            arm_catalog["E2"] = None
        save_state_json(output_dir / "e2_peak_valley_feature_segmenter_state.json", e2_segmenter)
        save_state_json(output_dir / "e2_shared_child_codebook_state.json", e2_codebook)
        state_files.extend(
            [
                "e2_peak_valley_feature_segmenter_state.json",
                "e2_shared_child_codebook_state.json",
            ]
        )

        parent_config = ParentGateConfig(
            minimum_occurrences=int(args.parent_min_occurrences),
            minimum_trials=int(args.parent_min_trials),
            minimum_subjects=int(args.parent_min_subjects),
            minimum_npmi=float(args.parent_min_npmi),
            minimum_mdl_gain_bits=float(args.parent_min_mdl_gain),
            minimum_loso_stability=float(args.parent_min_loso_stability),
        ).validate()
        e4_catalog = None
        if "E3" in arms:
            e3_catalog = fit_frequency_parent_catalog(
                e2_fit_tokens, int(args.parent_min_occurrences)
            )
            arm_fit["E3"], arm_online["E3"] = _overlay_trials(
                e2_fit_tokens, e2_online_tokens, e3_catalog, 32
            )
            arm_catalog["E3"] = e3_catalog
            save_state_json(output_dir / "e3_frequency_parent_catalog_state.json", e3_catalog)
            state_files.append("e3_frequency_parent_catalog_state.json")
        if "E4" in arms or (
            bool(args.matched_random_controls) and "E4" in arms
        ):
            e4_catalog = fit_parent_catalog(e2_fit_tokens, parent_config)
            if "E4" in arms:
                arm_fit["E4"], arm_online["E4"] = _overlay_trials(
                    e2_fit_tokens, e2_online_tokens, e4_catalog, 32
                )
                arm_catalog["E4"] = e4_catalog
            save_state_json(output_dir / "e4_strict_parent_catalog_state.json", e4_catalog)
            state_files.append("e4_strict_parent_catalog_state.json")

        if bool(args.matched_random_controls) and "E2" in arms:
            random_fit_segments = matched_random_segmentations(
                fit_trials_e2,
                e2_fit_segments,
                e2_segmenter,
                seed=int(args.seed) + 10_001,
            )
            random_online_segments = matched_random_segmentations(
                online_trials_e2,
                e2_online_segments,
                e2_segmenter,
                seed=int(args.seed) + 10_001,
            )
            c1_codebook, c1_fit_tokens, c1_online_tokens = _tokenize_partitions(
                fit_trials_e2,
                online_trials_e2,
                random_fit_segments,
                random_online_segments,
                e2_segmenter,
                str(args.child_feature_source),
                encoder,
                device,
                int(args.batch_size),
                fold_mean,
                fold_std,
                int(args.window_size_samples),
                32,
                int(args.pca_dim),
                int(args.seed),
            )
            arm_fit["C1"] = _primitive_trials_from_tokenized(c1_fit_tokens, 32)
            arm_online["C1"] = _primitive_trials_from_tokenized(c1_online_tokens, 32)
            arm_catalog["C1"] = None
            codebooks["C1"] = c1_codebook
            tokenized_fit_by_arm["C1"] = c1_fit_tokens
            tokenized_online_by_arm["C1"] = c1_online_tokens
            segmentations_fit["C1"] = random_fit_segments
            segmentations_online["C1"] = random_online_segments
            save_state_json(output_dir / "c1_matched_random_codebook_state.json", c1_codebook)
            state_files.append("c1_matched_random_codebook_state.json")
        if bool(args.matched_random_controls) and "E4" in arms:
            if e4_catalog is None:
                raise RuntimeError("C2 requires the E4 reference catalog.")
            c2_catalog = fit_matched_random_parent_catalog(
                e2_fit_tokens,
                e4_catalog,
                parent_config,
                seed=int(args.seed) + 20_003,
            )
            arm_fit["C2"], arm_online["C2"] = _overlay_trials(
                e2_fit_tokens, e2_online_tokens, c2_catalog, 32
            )
            arm_catalog["C2"] = c2_catalog
            save_state_json(output_dir / "c2_matched_negative_parent_catalog_state.json", c2_catalog)
            state_files.append("c2_matched_negative_parent_catalog_state.json")

    expected_output_arms = list(arms)
    if bool(args.matched_random_controls) and "E2" in arms:
        expected_output_arms.append("C1")
    if bool(args.matched_random_controls) and "E4" in arms:
        expected_output_arms.append("C2")
    if set(arm_fit) != set(expected_output_arms) or set(arm_online) != set(
        expected_output_arms
    ):
        raise RuntimeError(
            f"Built arm set differs from protocol: {sorted(arm_fit)} != "
            f"{sorted(expected_output_arms)}."
        )

    readouts: dict[tuple[str, str], dict] = {}
    for arm in expected_output_arms:
        parent_ids = _registered_parent_ids(arm_catalog.get(arm))
        for variant in READOUT_VARIANTS:
            readouts[(arm, variant)] = _run_readout(
                arm_fit[arm],
                arm_online[arm],
                online_train_ids,
                test_ids,
                32,
                parent_ids,
                variant,
                int(args.trial_cluster_count),
                int(args.seed),
            )

    raw_path = output_dir / "raw_cluster_predictions.npz"
    np.savez_compressed(
        raw_path,
        **_raw_freeze_payload(readouts, online_train_ids, test_ids),
    )
    raw_prediction_sha256 = sha256_file(raw_path)
    # Only now may the scoring/reporting phase open activity identity.
    truth_source = SourceSignalRepository(npz_path)
    labels, subjects, activity_name_by_trial = _truth_maps(truth_source, test_ids)
    y_true = np.asarray([labels[value] for value in test_ids], dtype=np.int64)
    if sorted(set(y_true.tolist())) != list(range(10)):
        raise RuntimeError("Session-2 scoring truth must contain activities 0..9.")
    class_names = [
        next(
            activity_name_by_trial[trial_id]
            for trial_id in test_ids
            if labels[trial_id] == class_id
        )
        for class_id in range(10)
    ]
    test_subject_vector = np.asarray(
        [subjects[trial_id] for trial_id in test_ids], dtype=np.int64
    )

    arm_results: dict[str, dict] = {}
    distance_matrices: dict[tuple[str, str], np.ndarray] = {}
    aligned_by_key: dict[tuple[str, str], np.ndarray] = {}
    cross_subject_records: list[dict[str, Any]] = []
    cross_subject_csv_rows: list[dict[str, Any]] = []
    for arm in expected_output_arms:
        base = _arm_metadata(
            arm, arm_fit[arm], arm_online[arm], arm_catalog.get(arm)
        )
        is_control = arm in CONTROL_SPECS
        base_arm = CONTROL_SPECS[arm]["base_arm"] if is_control else arm
        control_type = CONTROL_SPECS[arm]["control_type"] if is_control else None
        variant_results = {}
        for variant in READOUT_VARIANTS:
            item = readouts[(arm, variant)]
            raw_predictions = np.asarray(
                item["readout"].test_predictions, dtype=np.int64
            )
            metrics = _complete_clustering_metrics(y_true, raw_predictions)
            aligned_by_key[(arm, variant)] = np.asarray(
                metrics["aligned_predictions"], dtype=np.int64
            )
            distance_matrices[(arm, variant)] = _class_distance_matrix(
                item["test_features"], y_true
            )
            cross_subject_diagnostic = _cross_subject_trajectory_diagnostics(
                item["test_features"],
                y_true,
                test_subject_vector,
                raw_predictions,
            )
            cross_subject_records.append(
                {
                    "arm_id": arm,
                    "readout_variant": variant,
                    **cross_subject_diagnostic,
                }
            )
            cross_subject_csv_rows.append(
                _cross_subject_diagnostic_csv_row(
                    arm, variant, cross_subject_diagnostic
                )
            )
            transform = item["transform"]
            variant_results[variant] = {
                "metrics": metrics,
                "cross_subject_diagnostics": cross_subject_diagnostic,
                "readout": {
                    "fit_scope": "Session-2 cumulative online train only",
                    "train_trial_count": len(online_train_ids),
                    "test_trial_count": len(test_ids),
                    "cluster_count": int(item["readout"].cluster_count),
                    "inertia": float(item["readout"].inertia),
                    "iterations": int(item["readout"].iterations),
                    "raw_descriptor_dim": int(
                        item["raw_train_descriptors"].shape[1]
                    ),
                    "kept_descriptor_dim": int(len(transform.keep_columns)),
                    "readout_feature_dim": int(item["train_features"].shape[1]),
                    "descriptor_schema_sha256": hashlib.sha256(
                        "\n".join(item["descriptor_names"]).encode("utf-8")
                    ).hexdigest(),
                    "raw_predictions_frozen_before_truth_join": True,
                    "raw_prediction_artifact_sha256": raw_prediction_sha256,
                    **item["leakage_audit"],
                },
            }
        arm_results[arm] = {
            "base_arm": base_arm,
            "is_control": is_control,
            "control_type": control_type,
            "segmentation": base["segmentation"],
            "child_codebook_usage": base["child_codebook_usage"],
            "parent_catalog": base["parent_catalog"],
            "readout_variants": variant_results,
        }

    cross_subject_records.sort(
        key=lambda value: (str(value["arm_id"]), str(value["readout_variant"]))
    )
    cross_subject_csv_rows.sort(
        key=lambda value: (str(value["arm_id"]), str(value["readout_variant"]))
    )
    cross_subject_report = {
        "schema": CROSS_SUBJECT_DIAGNOSTIC_SCHEMA,
        "scope": "post_truth_join_descriptive_diagnostic_only",
        "raw_prediction_artifact": raw_path.name,
        "raw_prediction_artifact_sha256": raw_prediction_sha256,
        "test_trial_count": len(test_ids),
        "test_trial_ids_sha256": _array_sha256(
            np.asarray(test_ids, dtype=np.int64)
        ),
        "subject_ids": sorted(set(test_subject_vector.tolist())),
        "activity_class_ids": sorted(set(y_true.tolist())),
        "records": cross_subject_records,
        "causal_role": (
            "diagnostics are computed after raw predictions are persisted and "
            "cannot alter segmentation, codebooks, descriptors or predictions"
        ),
    }
    _write_json(
        output_dir / CROSS_SUBJECT_DIAGNOSTIC_FILES[0],
        cross_subject_report,
    )
    _write_csv(
        output_dir / CROSS_SUBJECT_DIAGNOSTIC_FILES[1],
        cross_subject_csv_rows,
    )

    prediction_rows = []
    for row_index, trial_id in enumerate(test_ids):
        row: dict[str, Any] = {
            "trial_global_id": int(trial_id),
            "subject_id": int(subjects[trial_id]),
            "activity_label_0based": int(labels[trial_id]),
            "activity_name": activity_name_by_trial[trial_id],
        }
        for arm in expected_output_arms:
            for variant in READOUT_VARIANTS:
                row[f"{arm}_{variant}_raw_cluster"] = int(
                    readouts[(arm, variant)]["readout"].test_predictions[row_index]
                )
                row[f"{arm}_{variant}_aligned_prediction"] = int(
                    aligned_by_key[(arm, variant)][row_index]
                )
        prediction_rows.append(row)
    _write_csv(output_dir / "session2_predictions.csv", prediction_rows)
    _write_csv(
        output_dir / "session_manifest.csv",
        [
            {"split": "online_train", "session": 1, "trial_global_id": int(value)}
            for value in manifest["session_1_train_trial_ids"]
        ]
        + [
            {"split": "online_train", "session": 2, "trial_global_id": int(value)}
            for value in manifest["session_2_train_trial_ids"]
        ]
        + [
            {"split": "online_test", "session": 2, "trial_global_id": int(value)}
            for value in test_ids
        ],
    )
    _export_parent_catalogs(output_dir / "parent_catalog.csv", arm_catalog)
    _export_trajectories(
        output_dir / "trial_trajectories.jsonl",
        arm_fit,
        arm_online,
        online_train_ids,
        test_ids,
        truth_source,
    )

    matrix_payload: dict[str, np.ndarray] = {
        "activity_names": np.asarray(class_names, dtype=object),
        "profile": np.asarray(str(args.expected_profile).upper()),
    }
    for (arm, variant), matrix in sorted(distance_matrices.items()):
        matrix_payload[f"matrix__{arm}__{variant}"] = matrix
    np.savez_compressed(output_dir / "activity_distance_matrices.npz", **matrix_payload)

    descriptor_payload: dict[str, np.ndarray] = {}
    for (arm, variant), item in sorted(readouts.items()):
        transform = item["transform"]
        prefix = f"{arm}__{variant}"
        descriptor_payload[f"keep_columns__{prefix}"] = transform.keep_columns
        descriptor_payload[f"mean__{prefix}"] = transform.mean
        descriptor_payload[f"scale__{prefix}"] = transform.scale
        descriptor_payload[f"pca_mean__{prefix}"] = (
            np.asarray([], dtype=np.float64)
            if transform.pca_mean is None
            else transform.pca_mean
        )
        descriptor_payload[f"pca_components__{prefix}"] = (
            np.empty((0, 0), dtype=np.float64)
            if transform.pca_components is None
            else transform.pca_components
        )
    np.savez_compressed(output_dir / "descriptor_transforms.npz", **descriptor_payload)

    representative_files: list[str] = []
    primary_fit_tokens = tokenized_fit_by_arm.get("E2")
    if primary_fit_tokens is not None:
        signal_by_fit = {int(item.trial_id): item for item in fit_trials_e2}
        representative_records = _export_representatives(
            output_dir / "primitive_representatives.jsonl",
            primary_fit_tokens,
            signal_by_fit,
            32,
        )
        _plot_representatives(
            output_dir / "primitive_representatives.png", representative_records
        )
        representative_files = [
            "primitive_representatives.jsonl",
            "primitive_representatives.png",
        ]

    _plot_trajectories(
        output_dir / "trajectory_sequences.png",
        arm_online,
        test_ids,
        labels,
        subjects,
        activity_name_by_trial,
    )
    waveform_files: list[str] = []
    if "E2" in tokenized_online_by_arm:
        e2_primitive_online = _primitive_trials_from_tokenized(
            tokenized_online_by_arm["E2"], 32
        )
        signal_by_online = {int(item.trial_id): item for item in online_trials_e2}
        _plot_waveform_examples(
            output_dir / "peak_valley_waveform_examples.png",
            e2_primitive_online,
            signal_by_online,
            test_ids,
            labels,
            activity_name_by_trial,
        )
        waveform_files = ["peak_valley_waveform_examples.png"]
    _plot_heatmaps(
        output_dir / "activity_distance_heatmaps.png",
        distance_matrices,
        class_names,
    )
    _plot_confusions(
        output_dir / "confusion_matrices.png",
        {
            arm: arm_results[arm]["readout_variants"]["state"]["metrics"]
            for arm in expected_output_arms
        },
        class_names,
    )
    _plot_segment_distributions(
        output_dir / "segment_duration_and_count_distributions.png",
        arm_fit,
        arm_online,
    )

    c1_audit = None
    if "C1" in arm_fit:
        reference = {
            int(item.trial_id): item
            for item in (*segmentations_fit["E2"], *segmentations_online["E2"])
        }
        randomized = {
            int(item.trial_id): item
            for item in (*segmentations_fit["C1"], *segmentations_online["C1"])
        }
        mismatches = []
        for trial_id, original in reference.items():
            control = randomized[trial_id]
            original_hist = {
                kind: sum(event.kind == kind for event in original.events)
                for kind in ("peak", "valley")
            }
            control_hist = {
                kind: sum(event.kind == kind for event in control.events)
                for kind in ("peak", "valley")
            }
            if len(original.segments) != len(control.segments) or original_hist != control_hist:
                mismatches.append(trial_id)
        if mismatches:
            raise RuntimeError(f"C1 matching failed for trials {mismatches[:5]}.")
        c1_audit = {
            "trial_count": len(reference),
            "segment_count_matched_per_trial": True,
            "event_kind_histogram_matched_per_trial": True,
            "minimum_segment_length_respected": True,
            "boundary_locations_are_uniform": False,
            "control_definition": (
                "uniform weak-composition lengths conditional on segment count, "
                "minimum length and peak/valley histogram"
            ),
        }
    c2_audit = None
    if "C2" in arm_catalog:
        catalog = arm_catalog["C2"]
        reference = arm_catalog["E4"]
        matching_rows = list(catalog.selection_metadata.get("matching", []))
        reference_kind_counts = {
            kind: int(
                sum(item.kind == kind for item in reference.registered_patterns)
            )
            for kind in ("peak", "valley")
        }
        negative_kind_counts = {
            kind: int(sum(item.kind == kind for item in catalog.registered_patterns))
            for kind in ("peak", "valley")
        }
        matched_event_kinds = all(
            row.get("reference_event_kind") == row.get("negative_event_kind")
            for row in matching_rows
        )
        c2_audit = {
            "reference_parent_count": len(reference.registered_patterns),
            "negative_parent_count": len(catalog.registered_patterns),
            "counts_match": len(reference.registered_patterns)
            == len(catalog.registered_patterns),
            "same_fit_data_sha256": catalog.fit_data_sha256
            == reference.fit_data_sha256,
            "reference_event_kind_counts": reference_kind_counts,
            "negative_event_kind_counts": negative_kind_counts,
            "event_kind_counts_match": reference_kind_counts
            == negative_kind_counts,
            "every_matched_pair_has_same_event_kind": matched_event_kinds,
            "matching": matching_rows,
            "selection_is_uniform_random": False,
            "selection_rule": catalog.selection_metadata.get("selection_rule"),
        }
        if not all(
            c2_audit[key]
            for key in (
                "counts_match",
                "same_fit_data_sha256",
                "event_kind_counts_match",
                "every_matched_pair_has_same_event_kind",
            )
        ):
            raise RuntimeError(
                "C2 failed parent-count, event-kind, or fit-corpus matching."
            )

    e2_family_codebook_audit = None
    e2_family_arms = [
        arm for arm in ("E2", "E3", "E4") if arm in arm_fit
    ]
    if e2_family_arms:
        if "E2" not in codebooks:
            raise RuntimeError("E2-family outputs lack the shared child codebook.")
        shared_hash = codebooks["E2"].state_hash()
        reference_fit = {
            int(item.trial_id): item for item in _primitive_trials_from_tokenized(
                e2_fit_tokens, 32
            )
        }
        reference_online = {
            int(item.trial_id): item for item in _primitive_trials_from_tokenized(
                e2_online_tokens, 32
            )
        }
        child_sequences_match = True
        for arm in e2_family_arms:
            for observed_trials, reference_trials in (
                (arm_fit[arm], reference_fit),
                (arm_online[arm], reference_online),
            ):
                if {int(item.trial_id) for item in observed_trials} != set(
                    reference_trials
                ):
                    child_sequences_match = False
                    break
                for item in observed_trials:
                    reference_item = reference_trials[int(item.trial_id)]
                    if not (
                        np.array_equal(item.starts, reference_item.starts)
                        and np.array_equal(item.ends, reference_item.ends)
                        and np.array_equal(
                            item.child_tokens, reference_item.child_tokens
                        )
                    ):
                        child_sequences_match = False
                        break
                if not child_sequences_match:
                    break
            if not child_sequences_match:
                break
        if not child_sequences_match:
            raise RuntimeError(
                "E2/E3/E4 parent overlays changed a shared child trajectory."
            )
        e2_family_codebook_audit = {
            "consumer_arms": e2_family_arms,
            "codebook_state_sha256_by_arm": {
                arm: shared_hash for arm in e2_family_arms
            },
            "unique_codebook_state_sha256": [shared_hash],
            "all_arms_share_one_child_codebook": True,
            "fit_and_online_child_spans_and_tokens_match_e2": True,
            "parent_overlays_are_non_destructive": True,
        }

    split_audit = {
        "fold": int(args.fold),
        "fit_subjects": fit_subjects,
        "validation_subjects": validation_subjects,
        "eval_subjects": eval_subjects,
        "subject_sets_pairwise_disjoint": True,
        "offline_fit_trial_count": len(fit_trial_ids),
        "offline_fit_scope": "checkpoint train subjects and activity ids 0..5 only",
        "session_2_cumulative_online_train_count": len(online_train_ids),
        "session_2_test_count": len(test_ids),
        "offline_online_trial_overlap": [],
        "online_train_test_trial_overlap": [],
        "predictor_grid_eval_labels_all_removed": True,
        "predictor_grid_eval_label_sentinel": -1,
        "protocol_manifest_builder_is_label_aware": True,
        "pre_freeze_label_access_scope": (
            "closed benchmark split construction/validation only; returned "
            "predictor manifest contains trial IDs and protocol-only subject IDs, "
            "not activity labels"
        ),
        "predictor_received_test_activity_labels_before_freeze": False,
        "raw_prediction_artifact": raw_path.name,
        "raw_prediction_artifact_sha256": raw_prediction_sha256,
        "truth_joined_after_raw_prediction_write": True,
        "global_hungarian_calls_per_arm_variant": 1,
        "e0_historical_feature_order": (
            "content_embedding -> per-window L2 -> trial-equal weighted PCA64 "
            "-> L2 -> KMeans32 -> normalized-centre cosine assignment"
        ),
        "e0_pre_pca_window_l2_verified": True,
        "e0_support_interval_policy": {
            "token_and_token_conditioned_signal_state": (
                "original_overlapping_npz_window"
            ),
            "all_duration_features_and_weights": (
                "non_overlapping_window_centre_voronoi_cells"
            ),
            "reason": (
                "align physical state with each encoded token while avoiding "
                "double-counted duration"
            ),
            "historical_scope": (
                "strict tokenization reproduction only; trajectory readout and "
                "Hungarian-scored endpoint are this experiment's unified proxy"
            ),
        },
        "normalization": normalization_audit,
        "label_free_signal_source": label_free_source.source_audit(all_predictor_ids),
        "session_manifest": manifest,
        "c1_matched_random_boundary_audit": c1_audit,
        "c2_matched_negative_parent_audit": c2_audit,
        "e2_family_shared_child_codebook_audit": e2_family_codebook_audit,
    }
    _write_json(output_dir / "split_audit.json", split_audit)

    source_fingerprints = _source_fingerprints()
    request_identity = _run_identity(
        args=args,
        profile=str(args.expected_profile).upper(),
        fold=int(args.fold),
        seed=int(args.seed),
        checkpoint=checkpoint_path,
        arms=arms,
        source_fingerprints=_member_source_fingerprints(source_fingerprints),
    )
    generated_file_names = [
        "raw_cluster_predictions.npz",
        "session_manifest.csv",
        "session2_predictions.csv",
        "trial_trajectories.jsonl",
        "parent_catalog.csv",
        "activity_distance_matrices.npz",
        "descriptor_transforms.npz",
        "trajectory_sequences.png",
        "activity_distance_heatmaps.png",
        "confusion_matrices.png",
        "segment_duration_and_count_distributions.png",
        "split_audit.json",
        *CROSS_SUBJECT_DIAGNOSTIC_FILES,
        *state_files,
        *representative_files,
        *waveform_files,
    ]
    if sha256_file(raw_path) != raw_prediction_sha256:
        raise RuntimeError(
            "Raw cluster predictions changed after the pre-truth freeze point."
        )
    generated_files = _generated_file_manifest(output_dir, generated_file_names)
    result = {
        "schema": SCHEMA,
        "request_identity": request_identity,
        "scope": "isolated_cgcd_session2_clustering_proxy_not_formal_happy_learner",
        "profile_semantics_audit": profile_audit,
        "e2_family_shared_child_codebook_audit": e2_family_codebook_audit,
        "arm_definitions": {
            "E0": (
                f"registered w{requested_window_size}/s{requested_stride} "
                "content-embedding KMeans32 tokenization plus the current "
                "unified trajectory readout"
            ),
            "E1": (
                "canonical motion-envelope peak/valley boundaries without axis "
                "vote or encoder feature confirmation"
            ),
            "E2": (
                "E1 candidates plus multi-axis support and fit-only calibrated "
                "frozen-encoder feature-change confirmation"
            ),
            "E3": "E2 children plus occurrence-frequency-only parent overlays",
            "E4": "E2 children plus support/NPMI/MDL/LOSO-gated parent overlays",
            "C1": "E2-matched random legal partitions",
            "C2": "E4-count and event-kind support-matched negative parent motifs",
        },
        "descriptor_ablation": {
            "shared": (
                "child count/duration/temporal-position/transition, quantization "
                "distance, event-kind and parent count/span/position features"
            ),
            "state_only": (
                "duration-weighted physical-statistic global mean/std and "
                "child-token-conditioned mean/log-energy statistics"
            ),
        },
        "cross_subject_trajectory_diagnostics": cross_subject_report,
        "arm_results": arm_results,
        "split_audit": split_audit,
        "generated_files": generated_files,
    }
    _write_json(output_dir / "experiment_result.json", result)
    return result


if __name__ == "__main__":
    main()
