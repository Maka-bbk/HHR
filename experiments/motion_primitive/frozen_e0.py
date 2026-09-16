"""Frozen E0 motion primitives and the full physical-state trajectory descriptor.

The historical E0 arm has a precise meaning: every 256-sample/128-stride
encoder window is one token; A2 ``content`` features are L2-normalised, reduced
by trial-equal weighted PCA64, L2-normalised again, and assigned to KMeans32 by
cosine distance.  No run-length compression is performed.

Overlapping windows retain their original signal when physical statistics are
computed.  Duration is the only exception: it uses window-centre Voronoi
ownership cells so overlapping windows never double-count trial time.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import sklearn
from sklearn.cluster import KMeans

from experiments.motion_primitive.core import (
    assign_to_codebook,
    fit_weighted_pca,
    inverse_trial_frequency_weights,
    l2_normalize,
)


SCHEMA = "hhr_frozen_a2_e0_state_v3"
PREVIOUS_SCHEMA = "hhr_frozen_a2_e0_state_v2"
LEGACY_SCHEMA = "hhr_frozen_a2_e0_state_v1"
CHANNEL_NAMES = (
    "acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z",
)

LEGACY_DESCRIPTOR_PROFILE = "legacy_state_v1"
GRAVITY_DESCRIPTOR_PROFILE = "gravity_signed_v1"
DURATION_INVARIANT_DESCRIPTOR_PROFILE = "duration_invariant_v1"
SUBJECT_DEBIASED_DESCRIPTOR_PROFILE = "subject_debiased_v1"
FULL_DEBIASED_DESCRIPTOR_PROFILE = "gravity_duration_subject_v1"
DURATION_SOFT_SUBJECT_A025_PROFILE = "duration_soft_subject_a025_v1"
DURATION_SOFT_SUBJECT_A050_PROFILE = "duration_soft_subject_a050_v1"
DURATION_SOFT_SUBJECT_A075_PROFILE = "duration_soft_subject_a075_v1"


@dataclass(frozen=True)
class DescriptorProfileSpec:
    """Immutable trajectory-descriptor contract used by experiment identity."""

    name: str
    signed_vertical: bool
    duration_invariant: bool
    subject_debias: bool
    subject_nuisance_projection_strength: float = 0.0


DESCRIPTOR_PROFILES: dict[str, DescriptorProfileSpec] = {
    LEGACY_DESCRIPTOR_PROFILE: DescriptorProfileSpec(
        LEGACY_DESCRIPTOR_PROFILE, False, False, False
    ),
    GRAVITY_DESCRIPTOR_PROFILE: DescriptorProfileSpec(
        GRAVITY_DESCRIPTOR_PROFILE, True, False, False
    ),
    DURATION_INVARIANT_DESCRIPTOR_PROFILE: DescriptorProfileSpec(
        DURATION_INVARIANT_DESCRIPTOR_PROFILE, False, True, False
    ),
    SUBJECT_DEBIASED_DESCRIPTOR_PROFILE: DescriptorProfileSpec(
        SUBJECT_DEBIASED_DESCRIPTOR_PROFILE, False, False, True, 1.0
    ),
    FULL_DEBIASED_DESCRIPTOR_PROFILE: DescriptorProfileSpec(
        FULL_DEBIASED_DESCRIPTOR_PROFILE, True, True, True, 1.0
    ),
    DURATION_SOFT_SUBJECT_A025_PROFILE: DescriptorProfileSpec(
        DURATION_SOFT_SUBJECT_A025_PROFILE, False, True, True, 0.25
    ),
    DURATION_SOFT_SUBJECT_A050_PROFILE: DescriptorProfileSpec(
        DURATION_SOFT_SUBJECT_A050_PROFILE, False, True, True, 0.50
    ),
    DURATION_SOFT_SUBJECT_A075_PROFILE: DescriptorProfileSpec(
        DURATION_SOFT_SUBJECT_A075_PROFILE, False, True, True, 0.75
    ),
}

ABSOLUTE_DURATION_DESCRIPTOR_NAMES = frozenset(
    {
        "log_child_count",
        "log_duration_samples",
        "child_duration_seconds_mean",
        "child_duration_seconds_std",
        "child_duration_seconds_max",
    }
)


def descriptor_profile_spec(name: str) -> DescriptorProfileSpec:
    try:
        spec = DESCRIPTOR_PROFILES[str(name)]
    except KeyError as error:
        raise ValueError(
            f"Unknown descriptor profile {name!r}; expected {sorted(DESCRIPTOR_PROFILES)}."
        ) from error
    strength = float(spec.subject_nuisance_projection_strength)
    if spec.subject_debias:
        if not 0.0 < strength <= 1.0:
            raise RuntimeError(
                f"Subject-debiased profile {spec.name!r} must use strength in (0,1]."
            )
    elif strength != 0.0:
        raise RuntimeError(
            f"Non-debiased profile {spec.name!r} cannot carry projection strength."
        )
    return spec


def _array_hash(*arrays: np.ndarray) -> str:
    digest = hashlib.sha256()
    for value in arrays:
        array = np.ascontiguousarray(np.asarray(value))
        digest.update(str(array.shape).encode("ascii"))
        digest.update(array.dtype.str.encode("ascii"))
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


@dataclass(frozen=True)
class WindowGrid:
    windows: np.ndarray
    labels: np.ndarray
    labels_1based: np.ndarray
    subject_ids: np.ndarray
    trial_ids: np.ndarray
    trial_numbers: np.ndarray
    window_indices: np.ndarray
    starts: np.ndarray
    stored_mean: np.ndarray
    stored_std: np.ndarray
    window_size_samples: int
    stride_samples: int

    def rows_for_trial(self, trial_id: int) -> np.ndarray:
        rows = np.flatnonzero(self.trial_ids == int(trial_id))
        if not len(rows):
            raise KeyError(f"Unknown trial_global_id={trial_id}.")
        rows = rows[np.argsort(self.starts[rows], kind="stable")]
        if not np.array_equal(
            self.window_indices[rows], np.arange(len(rows), dtype=np.int64)
        ):
            raise RuntimeError(f"Trial {trial_id} has incomplete window indices.")
        if int(self.starts[rows[0]]) != 0:
            raise RuntimeError(f"Trial {trial_id} does not begin at sample zero.")
        if len(rows) > 1 and np.any(np.diff(self.starts[rows]) != self.stride_samples):
            raise RuntimeError(f"Trial {trial_id} violates the registered stride.")
        return rows


def load_window_grid(
    path: Path,
    *,
    expected_window_size: int = 256,
    expected_stride: int = 128,
) -> WindowGrid:
    required = {
        "windows", "labels", "labels_1based", "subject_ids",
        "trial_global_ids", "trial_numbers", "window_indices",
        "window_start_indices", "mean", "std",
    }
    with np.load(Path(path), allow_pickle=False) as data:
        missing = required - set(data.files)
        if missing:
            raise RuntimeError(f"USC-HAD NPZ lacks fields {sorted(missing)}.")
        grid = WindowGrid(
            windows=np.asarray(data["windows"], dtype=np.float32),
            labels=np.asarray(data["labels"], dtype=np.int64),
            labels_1based=np.asarray(data["labels_1based"], dtype=np.int64),
            subject_ids=np.asarray(data["subject_ids"], dtype=np.int64),
            trial_ids=np.asarray(data["trial_global_ids"], dtype=np.int64),
            trial_numbers=np.asarray(data["trial_numbers"], dtype=np.int64),
            window_indices=np.asarray(data["window_indices"], dtype=np.int64),
            starts=np.asarray(data["window_start_indices"], dtype=np.int64),
            stored_mean=np.asarray(data["mean"], dtype=np.float32),
            stored_std=np.asarray(data["std"], dtype=np.float32),
            window_size_samples=int(expected_window_size),
            stride_samples=int(expected_stride),
        )
    count = len(grid.windows)
    if grid.windows.shape != (count, len(CHANNEL_NAMES), int(expected_window_size)):
        raise RuntimeError(
            "Frozen E0 requires windows shaped "
            f"[N,{len(CHANNEL_NAMES)},{expected_window_size}], got {grid.windows.shape}."
        )
    for name in (
        "labels", "labels_1based", "subject_ids", "trial_ids", "trial_numbers",
        "window_indices", "starts",
    ):
        if np.asarray(getattr(grid, name)).shape != (count,):
            raise RuntimeError(f"Window-grid field {name!r} has an invalid shape.")
    if not np.array_equal(grid.labels_1based, grid.labels + 1):
        raise RuntimeError("labels and labels_1based disagree.")
    if grid.stored_mean.shape != (1, 6, 1) or grid.stored_std.shape != (1, 6, 1):
        raise RuntimeError("Stored USC-HAD normalization has an unexpected shape.")
    if not np.all(np.isfinite(grid.windows)) or np.any(grid.stored_std <= 0):
        raise RuntimeError("USC-HAD numeric grid contains invalid values.")
    observed_differences: list[int] = []
    for trial_id in np.unique(grid.trial_ids):
        starts = np.sort(grid.starts[grid.trial_ids == trial_id])
        observed_differences.extend(int(value) for value in np.diff(starts))
    if observed_differences and set(observed_differences) != {int(expected_stride)}:
        raise RuntimeError(
            f"Expected stride {expected_stride}, observed {sorted(set(observed_differences))}."
        )
    return grid


def fold_normalization(
    grid: WindowGrid,
    fit_window_mask: np.ndarray,
    *,
    epsilon: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    mask = np.asarray(fit_window_mask, dtype=bool)
    if mask.shape != (len(grid.windows),) or not np.any(mask):
        raise ValueError("fit_window_mask is empty or malformed.")
    raw = (grid.windows[mask] * grid.stored_std + grid.stored_mean).astype(np.float64)
    mean = raw.mean(axis=(0, 2), keepdims=True)
    std = np.maximum(raw.std(axis=(0, 2), keepdims=True), float(epsilon))
    return mean.astype(np.float32), std.astype(np.float32), {
        "mode": "offline_train_subjects_old_classes_only",
        "fit_window_count": int(mask.sum()),
        "fit_window_mask_sha256": _array_hash(mask.astype(np.uint8)),
        "mean": mean.reshape(-1).tolist(),
        "std": std.reshape(-1).tolist(),
    }


def normalize_windows(
    grid: WindowGrid,
    rows: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
) -> np.ndarray:
    raw = grid.windows[np.asarray(rows, dtype=np.int64)] * grid.stored_std + grid.stored_mean
    values = (raw - np.asarray(mean, dtype=np.float32)) / np.asarray(std, dtype=np.float32)
    if not np.all(np.isfinite(values)):
        raise RuntimeError("Fold-normalized windows contain non-finite values.")
    return values.astype(np.float32)


@dataclass(frozen=True)
class FrozenE0Codebook:
    primitive_num: int
    pca_dim: int
    input_dim: int
    pca_mean: np.ndarray
    pca_components: np.ndarray
    cluster_centers: np.ndarray
    seed: int
    fit_window_count: int
    fit_trial_count: int
    fit_subject_count: int
    fit_data_sha256: str
    kmeans_inertia: float
    kmeans_n_iter: int

    def validate(self, *, strict_historical: bool = True) -> "FrozenE0Codebook":
        if strict_historical and (self.primitive_num, self.pca_dim) != (32, 64):
            raise RuntimeError("Historical E0 is fixed to PCA64/KMeans32.")
        if self.primitive_num < 2 or self.pca_dim < 1 or self.input_dim < self.pca_dim:
            raise ValueError("Invalid E0 dimensions.")
        if self.pca_mean.shape != (self.input_dim,):
            raise ValueError("E0 PCA mean shape mismatch.")
        if self.pca_components.shape != (self.pca_dim, self.input_dim):
            raise ValueError("E0 PCA component shape mismatch.")
        if self.cluster_centers.shape != (self.primitive_num, self.pca_dim):
            raise ValueError("E0 cluster-centre shape mismatch.")
        arrays = (self.pca_mean, self.pca_components, self.cluster_centers)
        if any(not np.all(np.isfinite(array)) for array in arrays):
            raise ValueError("E0 state contains non-finite values.")
        if self.fit_window_count < self.primitive_num or self.fit_trial_count < 1:
            raise ValueError("E0 fit counts are invalid.")
        return self

    def transform(self, content: np.ndarray) -> np.ndarray:
        values = l2_normalize(np.asarray(content, dtype=np.float32))
        embedded = (
            (values - np.asarray(self.pca_mean, dtype=np.float32))
            @ np.asarray(self.pca_components, dtype=np.float32).T
        ).astype(np.float32)
        return l2_normalize(embedded)

    def assign(self, content: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        embedded = self.transform(content)
        tokens, distances, _ = assign_to_codebook(
            embedded,
            np.asarray(self.cluster_centers, dtype=np.float32),
            "cosine",
        )
        return tokens.astype(np.int64), distances.astype(np.float32), embedded


def fit_frozen_e0_codebook(
    content: np.ndarray,
    trial_ids: np.ndarray,
    subject_ids: np.ndarray,
    *,
    primitive_num: int = 32,
    pca_dim: int = 64,
    seed: int = 0,
    strict_historical: bool = True,
) -> FrozenE0Codebook:
    values = np.asarray(content, dtype=np.float32)
    trials = np.asarray(trial_ids, dtype=np.int64)
    subjects = np.asarray(subject_ids, dtype=np.int64)
    if values.ndim != 2 or len(values) < int(primitive_num):
        raise ValueError("E0 content must be [N,D] with N >= primitive_num.")
    if trials.shape != (len(values),) or subjects.shape != (len(values),):
        raise ValueError("E0 trial/subject ids must have one value per window.")
    if strict_historical and (int(primitive_num), int(pca_dim)) != (32, 64):
        raise ValueError("Historical E0 is pinned to PCA64/KMeans32.")
    base = l2_normalize(values)
    weights = inverse_trial_frequency_weights(trials)
    pca = fit_weighted_pca(base, int(pca_dim), weights)
    embedded = l2_normalize(pca.transform(base))
    estimator = KMeans(
        n_clusters=int(primitive_num),
        random_state=int(seed),
        n_init=20,
        max_iter=300,
        algorithm="lloyd",
    ).fit(embedded, sample_weight=weights)
    return FrozenE0Codebook(
        primitive_num=int(primitive_num),
        pca_dim=int(pca_dim),
        input_dim=int(values.shape[1]),
        pca_mean=np.asarray(pca.mean, dtype=np.float32),
        pca_components=np.asarray(pca.components, dtype=np.float32),
        cluster_centers=np.asarray(estimator.cluster_centers_, dtype=np.float32),
        seed=int(seed),
        fit_window_count=int(len(values)),
        fit_trial_count=int(len(np.unique(trials))),
        fit_subject_count=int(len(np.unique(subjects))),
        fit_data_sha256=_array_hash(values, trials, subjects, weights),
        kmeans_inertia=float(estimator.inertia_),
        kmeans_n_iter=int(estimator.n_iter_),
    ).validate(strict_historical=strict_historical)


def window_ownership_partition(
    starts: np.ndarray,
    window_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    starts = np.asarray(starts, dtype=np.int64)
    if starts.ndim != 1 or len(starts) < 1 or int(starts[0]) != 0:
        raise ValueError("Window starts must be non-empty and begin at zero.")
    centers = starts.astype(np.float64) + float(window_size) / 2.0
    boundaries = [0]
    boundaries.extend(
        int(round(0.5 * (left + right)))
        for left, right in zip(centers[:-1], centers[1:])
    )
    boundaries.append(int(starts[-1]) + int(window_size))
    edges = np.asarray(boundaries, dtype=np.int64)
    if np.any(np.diff(edges) <= 0):
        raise RuntimeError("Window ownership cells are not strictly positive.")
    return edges[:-1], edges[1:]


def statistic_names() -> tuple[str, ...]:
    names = ["duration_seconds"]
    for prefix in ("mean", "std", "log_mean_square_energy", "end_minus_start"):
        names.extend(f"{prefix}__{channel}" for channel in CHANNEL_NAMES)
    return tuple(names)


def window_statistics(
    raw_windows: np.ndarray,
    ownership_starts: np.ndarray,
    ownership_ends: np.ndarray,
    *,
    sample_rate_hz: float = 100.0,
) -> np.ndarray:
    raw = np.asarray(raw_windows, dtype=np.float64)
    if raw.ndim != 3 or raw.shape[1] != 6 or raw.shape[2] < 1:
        raise ValueError("raw_windows must be [L,6,T].")
    durations = (
        np.asarray(ownership_ends, dtype=np.int64)
        - np.asarray(ownership_starts, dtype=np.int64)
    ) / float(sample_rate_hz)
    values = np.c_[
        durations,
        raw.mean(axis=2),
        raw.std(axis=2),
        np.log(np.mean(np.square(raw), axis=2) + 1e-8),
        raw[:, :, -1] - raw[:, :, 0],
    ]
    if values.shape != (len(raw), 25) or not np.all(np.isfinite(values)):
        raise RuntimeError("Physical-state statistic construction failed.")
    return values.astype(np.float64)


SIGNED_VERTICAL_NAMES = tuple(
    [f"signed_vertical_phase_mean__{index}" for index in range(8)]
    + ["signed_vertical_temporal_moment"]
)


def reconstruct_trial_channels(
    raw_windows: np.ndarray,
    window_starts: np.ndarray,
    *,
    overlap_tolerance: float = 1e-5,
) -> np.ndarray:
    """Reconstruct the observed trial prefix by overlap-averaging raw windows.

    The preprocessor never pads incomplete tails.  Consequently the final
    reconstructed sample is exactly ``last_start + window_size`` and contains
    no invented sensor values.  Overlap disagreement is rejected because it
    indicates that windows no longer came from one immutable trial signal.
    """

    windows = np.asarray(raw_windows, dtype=np.float64)
    starts = np.asarray(window_starts, dtype=np.int64)
    if windows.ndim != 3 or windows.shape[1] != len(CHANNEL_NAMES):
        raise ValueError("raw_windows must be [L,6,T].")
    if starts.shape != (len(windows),) or not len(starts):
        raise ValueError("window_starts must contain one value per raw window.")
    if starts[0] != 0 or np.any(np.diff(starts) <= 0):
        raise ValueError("Raw windows must start at zero and be strictly ordered.")
    window_size = int(windows.shape[2])
    trial_length = int(starts[-1]) + window_size
    accumulated = np.zeros((len(CHANNEL_NAMES), trial_length), dtype=np.float64)
    counts = np.zeros(trial_length, dtype=np.int64)
    for window, start in zip(windows, starts):
        begin = int(start)
        end = begin + window_size
        accumulated[:, begin:end] += window
        counts[begin:end] += 1
    if np.any(counts == 0):
        raise RuntimeError("Raw window grid leaves uncovered samples in the trial.")
    reconstructed = accumulated / counts[None, :]
    scale = max(1.0, float(np.max(np.abs(windows))))
    maximum_disagreement = max(
        float(np.max(np.abs(window - reconstructed[:, int(start) : int(start) + window_size])))
        for window, start in zip(windows, starts)
    )
    if maximum_disagreement > float(overlap_tolerance) * scale:
        raise RuntimeError(
            "Overlapping raw windows disagree; a gravity trajectory cannot be "
            f"reconstructed safely (max error={maximum_disagreement:.6g})."
        )
    if not np.all(np.isfinite(reconstructed)):
        raise RuntimeError("Reconstructed trial contains non-finite values.")
    return reconstructed


@dataclass(frozen=True)
class GravityAlignedState:
    """Signed gravity-axis dynamics plus learner-safe quality diagnostics."""

    descriptor: np.ndarray
    names: tuple[str, ...]
    gravity_norm_g: float
    edge_angle_degrees: float
    reliable: bool
    sample_count: int

    def validate(self) -> "GravityAlignedState":
        if self.descriptor.shape != (len(SIGNED_VERTICAL_NAMES),):
            raise ValueError("Signed vertical descriptor has an invalid shape.")
        if self.names != SIGNED_VERTICAL_NAMES:
            raise ValueError("Signed vertical descriptor schema differs.")
        numeric = np.r_[
            self.descriptor,
            float(self.gravity_norm_g),
            float(self.edge_angle_degrees),
        ]
        if not np.all(np.isfinite(numeric)) or int(self.sample_count) < 1:
            raise ValueError("Gravity-aligned state contains invalid values.")
        return self


def gravity_aligned_trial_state(
    raw_windows: np.ndarray,
    window_starts: np.ndarray,
    *,
    temporal_bins: int = 8,
    gravity_norm_bounds: tuple[float, float] = (0.7, 1.3),
    maximum_edge_angle_degrees: float = 15.0,
) -> GravityAlignedState:
    """Build a rotation-equivariant, signed vertical trend on normalized time.

    USC-HAD acceleration is measured in ``g``.  The full-trial mean acceleration
    estimates the gravity vector robustly even for periodic activities.  Its
    unit vector defines positive gravity direction, while subtracting the mean
    projected acceleration removes sensor bias.  Phase-bin means and the signed
    temporal moment use normalized time, so no integral scales with trial length.
    Neither the gravity-vector components nor subject identity enter the output.
    """

    if int(temporal_bins) != 8:
        raise ValueError("The registered signed-vertical profile is fixed to 8 bins.")
    signal = reconstruct_trial_channels(raw_windows, window_starts)
    acceleration = signal[:3].T
    gravity = acceleration.mean(axis=0)
    gravity_norm = float(np.linalg.norm(gravity))
    valid_gravity = bool(np.isfinite(gravity_norm) and gravity_norm > 1e-8)
    unit = gravity / gravity_norm if valid_gravity else np.asarray([1.0, 0.0, 0.0])
    projected = acceleration @ unit
    vertical = projected - float(projected.mean())

    pieces = np.array_split(vertical, int(temporal_bins))
    phase_means = np.asarray([float(piece.mean()) for piece in pieces], dtype=np.float64)
    normalized_time = (np.arange(len(vertical), dtype=np.float64) + 0.5) / len(vertical)
    temporal_moment = float(np.mean(vertical * (0.5 - normalized_time)))
    descriptor = np.r_[phase_means, temporal_moment].astype(np.float64)

    edge_count = max(1, min(len(acceleration) // 2, int(round(0.1 * len(acceleration)))))
    first = acceleration[:edge_count].mean(axis=0)
    last = acceleration[-edge_count:].mean(axis=0)
    denominator = float(np.linalg.norm(first) * np.linalg.norm(last))
    if denominator <= 1e-12:
        edge_angle = 180.0
    else:
        cosine = float(np.clip(np.dot(first, last) / denominator, -1.0, 1.0))
        edge_angle = float(np.degrees(np.arccos(cosine)))
    lower, upper = map(float, gravity_norm_bounds)
    reliable = bool(
        valid_gravity
        and lower <= gravity_norm <= upper
        and edge_angle <= float(maximum_edge_angle_degrees)
    )
    return GravityAlignedState(
        descriptor=descriptor,
        names=SIGNED_VERTICAL_NAMES,
        gravity_norm_g=gravity_norm if valid_gravity else 0.0,
        edge_angle_degrees=edge_angle,
        reliable=reliable,
        sample_count=int(signal.shape[1]),
    ).validate()


@dataclass(frozen=True)
class PrimitiveTrajectory:
    trial_id: int
    subject_id: int
    starts: np.ndarray
    ends: np.ndarray
    tokens: np.ndarray
    distances: np.ndarray
    embeddings: np.ndarray
    statistics: np.ndarray
    statistic_names: tuple[str, ...]
    gravity_aligned: GravityAlignedState | None = None

    def validate(self, primitive_num: int) -> "PrimitiveTrajectory":
        count = len(self.tokens)
        if count < 1:
            raise ValueError("A trajectory must contain at least one E0 token.")
        if self.starts.shape != (count,) or self.ends.shape != (count,):
            raise ValueError("Trajectory span count mismatch.")
        if np.any(self.starts[1:] != self.ends[:-1]) or np.any(self.ends <= self.starts):
            raise ValueError("Trajectory ownership spans must form a contiguous partition.")
        if self.distances.shape != (count,) or self.embeddings.shape[0] != count:
            raise ValueError("Trajectory codebook fields have inconsistent lengths.")
        if self.statistics.shape != (count, 25):
            raise ValueError("Trajectory physical-state statistics must be [L,25].")
        if self.statistic_names != statistic_names():
            raise ValueError("Trajectory statistic schema mismatch.")
        if np.any(self.tokens < 0) or np.any(self.tokens >= int(primitive_num)):
            raise ValueError("Trajectory contains an out-of-range token.")
        numeric = (self.distances, self.embeddings, self.statistics)
        if any(not np.all(np.isfinite(value)) for value in numeric):
            raise ValueError("Trajectory contains non-finite values.")
        if self.gravity_aligned is not None:
            self.gravity_aligned.validate()
            if int(self.gravity_aligned.sample_count) != int(self.ends[-1]):
                raise ValueError(
                    "Gravity-aligned signal length differs from trajectory ownership."
                )
        return self


def build_e0_trajectories(
    grid: WindowGrid,
    content: np.ndarray,
    codebook: FrozenE0Codebook,
    *,
    trial_ids: Sequence[int] | None = None,
    sample_rate_hz: float = 100.0,
    include_gravity: bool = False,
) -> list[PrimitiveTrajectory]:
    features = np.asarray(content, dtype=np.float32)
    if features.shape != (len(grid.windows), codebook.input_dim):
        raise ValueError(
            f"Expected content shape {(len(grid.windows), codebook.input_dim)}, got {features.shape}."
        )
    selected_ids = (
        np.unique(grid.trial_ids).astype(np.int64).tolist()
        if trial_ids is None
        else sorted({int(value) for value in trial_ids})
    )
    result: list[PrimitiveTrajectory] = []
    for trial_id in selected_ids:
        rows = grid.rows_for_trial(trial_id)
        subject_values = np.unique(grid.subject_ids[rows])
        label_values = np.unique(grid.labels[rows])
        if len(subject_values) != 1 or len(label_values) != 1:
            raise RuntimeError(f"Trial {trial_id} has inconsistent metadata.")
        starts, ends = window_ownership_partition(
            grid.starts[rows], grid.window_size_samples
        )
        tokens, distances, embedded = codebook.assign(features[rows])
        raw_windows = (
            grid.windows[rows] * grid.stored_std + grid.stored_mean
        ).astype(np.float32)
        trajectory = PrimitiveTrajectory(
            trial_id=int(trial_id),
            subject_id=int(subject_values[0]),
            starts=starts,
            ends=ends,
            tokens=tokens,
            distances=distances,
            embeddings=embedded,
            statistics=window_statistics(
                raw_windows, starts, ends, sample_rate_hz=sample_rate_hz
            ),
            statistic_names=statistic_names(),
            gravity_aligned=(
                gravity_aligned_trial_state(raw_windows, grid.starts[rows])
                if include_gravity else None
            ),
        ).validate(codebook.primitive_num)
        result.append(trajectory)
    return result


def raw_descriptor_dimension(
    primitive_num: int,
    *,
    include_state: bool = True,
    descriptor_profile: str = LEGACY_DESCRIPTOR_PROFILE,
) -> int:
    k = int(primitive_num)
    spec = descriptor_profile_spec(descriptor_profile)
    if spec.duration_invariant:
        # Three normalised transition blocks are retained for schema parity;
        # only transition_any can be non-zero for fixed-window E0.
        base = k + k + 4 * k + 3 * k * k + 5
        state = 48 + 12 * k if include_state else 0
    else:
        base = k + k + 4 * k + 3 * k * k + 10 + 1
        state = 48 + 12 * k + k if include_state else 0
    vertical = len(SIGNED_VERTICAL_NAMES) if spec.signed_vertical else 0
    return base + state + vertical


def trajectory_descriptor(
    trial: PrimitiveTrajectory,
    primitive_num: int,
    *,
    include_state: bool = True,
    descriptor_profile: str = LEGACY_DESCRIPTOR_PROFILE,
) -> tuple[np.ndarray, tuple[str, ...]]:
    trial.validate(int(primitive_num))
    spec = descriptor_profile_spec(descriptor_profile)
    tokens = trial.tokens.astype(np.int64)
    durations = (trial.ends - trial.starts).astype(np.float64)
    total_duration = float(durations.sum())
    values: list[float] = []
    names: list[str] = []

    def extend(prefix: str, vector: np.ndarray) -> None:
        flat = np.asarray(vector, dtype=np.float64).reshape(-1)
        values.extend(float(value) for value in flat)
        names.extend(f"{prefix}{index}" for index in range(len(flat)))

    count_hist = np.bincount(tokens, minlength=primitive_num).astype(np.float64)
    count_hist /= max(float(len(tokens)), 1.0)
    duration_hist = np.bincount(
        tokens, weights=durations, minlength=primitive_num
    ).astype(np.float64)
    duration_hist /= max(total_duration, 1.0)
    extend("child_count_fraction__", count_hist)
    extend("child_duration_fraction__", duration_hist)

    midpoints = 0.5 * (trial.starts.astype(np.float64) + trial.ends.astype(np.float64))
    positions = np.clip(midpoints / max(float(trial.ends[-1]), 1.0), 0.0, 1.0 - 1e-12)
    temporal = np.zeros((4, primitive_num), dtype=np.float64)
    for token, duration, position in zip(tokens, durations, positions):
        temporal[min(3, int(position * 4.0)), int(token)] += float(duration)
    temporal /= max(total_duration, 1.0)
    extend("child_temporal_quartile_duration__", temporal)

    transition_tokens = tokens
    if spec.duration_invariant and len(tokens) > 1:
        transition_tokens = tokens[
            np.r_[True, tokens[1:] != tokens[:-1]]
        ]
    transition = np.zeros((primitive_num, primitive_num), dtype=np.float64)
    if len(transition_tokens) > 1:
        for left, right in zip(transition_tokens[:-1], transition_tokens[1:]):
            transition[int(left), int(right)] += 1.0
        transition /= float(len(transition_tokens) - 1)
    extend(
        "transition_run_length__" if spec.duration_invariant else "transition_any__",
        transition,
    )
    # E0 has no peak/valley boundary type.  The zero columns are retained here
    # for exact schema parity, then removed by the train-only constant filter.
    extend("transition_peak__", np.zeros_like(transition))
    extend("transition_valley__", np.zeros_like(transition))

    quality = np.asarray(trial.distances, dtype=np.float64)
    if spec.duration_invariant:
        scalar = np.asarray(
            [
                float(quality.mean()),
                float(quality.std()),
                float(np.quantile(quality, 0.90)),
                0.0,
                0.0,
            ],
            dtype=np.float64,
        )
        values.extend(scalar.tolist())
        names.extend(
            (
                "quantization_distance_mean",
                "quantization_distance_std",
                "quantization_distance_q90",
                "peak_boundary_fraction",
                "valley_boundary_fraction",
            )
        )
    else:
        duration_seconds = trial.statistics[:, 0]
        scalar = np.asarray(
            [
                math.log1p(len(tokens)), math.log1p(total_duration),
                float(duration_seconds.mean()), float(duration_seconds.std()),
                float(duration_seconds.max()), float(quality.mean()),
                float(quality.std()), float(quality.max()), 0.0, 0.0,
            ],
            dtype=np.float64,
        )
        values.extend(scalar.tolist())
        names.extend(
            (
                "log_child_count", "log_duration_samples",
                "child_duration_seconds_mean", "child_duration_seconds_std",
                "child_duration_seconds_max", "quantization_distance_mean",
                "quantization_distance_std", "quantization_distance_max",
                "peak_boundary_fraction", "valley_boundary_fraction",
            )
        )
        # E0 has no hierarchical parents, but the historical schema contains it.
        values.append(0.0)
        names.append("log_parent_occurrence_count")

    if include_state:
        statistics = np.asarray(trial.statistics[:, 1:], dtype=np.float64)
        weights = durations / max(total_duration, 1.0)
        global_mean = np.sum(statistics * weights[:, None], axis=0)
        global_std = np.sqrt(
            np.sum(np.square(statistics - global_mean) * weights[:, None], axis=0)
        )
        physical_names = trial.statistic_names[1:]
        values.extend(global_mean.tolist())
        names.extend(f"state_global_mean__{name}" for name in physical_names)
        values.extend(global_std.tolist())
        names.extend(f"state_global_std__{name}" for name in physical_names)

        selected_columns = [
            index
            for index, name in enumerate(physical_names)
            if name.startswith("mean__")
            or name.startswith("log_mean_square_energy__")
        ]
        if len(selected_columns) != 12:
            raise RuntimeError("E0 token-conditioned state must select exactly 12 columns.")
        token_state = np.zeros((primitive_num, 12), dtype=np.float64)
        token_presence = np.zeros(primitive_num, dtype=np.float64)
        for token in range(int(primitive_num)):
            selected = tokens == token
            if not np.any(selected):
                continue
            token_state[token] = np.average(
                statistics[selected][:, selected_columns],
                axis=0,
                weights=durations[selected],
            )
            token_presence[token] = 1.0
        extend("state_by_child_token__", token_state)
        if not spec.duration_invariant:
            extend("state_child_token_presence__", token_presence)

    if spec.signed_vertical:
        if trial.gravity_aligned is None:
            raise RuntimeError(
                "The selected descriptor profile requires raw-trial gravity features."
            )
        gravity = trial.gravity_aligned.validate()
        values.extend(gravity.descriptor.astype(np.float64).tolist())
        names.extend(gravity.names)

    vector = np.asarray(values, dtype=np.float64)
    expected = raw_descriptor_dimension(
        primitive_num,
        include_state=include_state,
        descriptor_profile=descriptor_profile,
    )
    if vector.shape != (expected,) or len(names) != expected:
        raise RuntimeError(
            f"Descriptor schema mismatch: expected {expected}, got {vector.shape}."
        )
    if not np.all(np.isfinite(vector)):
        raise RuntimeError("Trajectory descriptor contains non-finite values.")
    return vector, tuple(names)


@dataclass(frozen=True)
class DescriptorTransform:
    keep_columns: np.ndarray
    mean: np.ndarray
    scale: np.ndarray
    pca_mean: np.ndarray | None
    pca_components: np.ndarray | None
    protected_columns: np.ndarray | None = None
    protected_mean: np.ndarray | None = None
    protected_scale: np.ndarray | None = None
    protected_distance_weight: float = 0.0
    nuisance_basis: np.ndarray | None = None
    nuisance_singular_values: np.ndarray | None = None
    nuisance_explained_fraction: float = 0.0
    nuisance_fit_subject_count: int = 0
    nuisance_fit_trial_count: int = 0
    nuisance_projection_strength: float = 0.0

    @property
    def primary_output_dim(self) -> int:
        return (
            int(self.pca_components.shape[0])
            if self.pca_components is not None
            else int(len(self.keep_columns))
        )

    @property
    def output_dim(self) -> int:
        protected = (
            0 if self.protected_columns is None else int(len(self.protected_columns))
        )
        return self.primary_output_dim + protected

    def _normalized_primary(self, matrix: np.ndarray) -> np.ndarray:
        primary = (matrix[:, self.keep_columns] - self.mean) / self.scale
        if self.pca_components is not None:
            primary = (primary - self.pca_mean) @ self.pca_components.T
        norms = np.linalg.norm(primary, axis=1, keepdims=True)
        if np.any(norms <= 1e-12):
            raise RuntimeError("Descriptor structural block collapsed before projection.")
        return primary / norms

    def transform(self, values: np.ndarray) -> np.ndarray:
        matrix = np.asarray(values, dtype=np.float64)
        required_columns = [int(self.keep_columns.max())]
        if self.protected_columns is not None and len(self.protected_columns):
            required_columns.append(int(self.protected_columns.max()))
        if matrix.ndim != 2 or matrix.shape[1] <= max(required_columns):
            raise ValueError("Descriptor transform input shape mismatch.")
        primary = self._normalized_primary(matrix)

        # New artifacts store a structural-only nuisance basis.  Re-normalising
        # the projected structural block preserves its distance budget, so the
        # protected signed-vertical block has exactly the same values and
        # direction with or without subject debiasing.  A full-width basis is
        # accepted only to keep historical v2 artifacts loadable; those old
        # artifacts retain their original combined-space behaviour.
        legacy_combined_basis = None
        if self.nuisance_basis is not None:
            basis = np.asarray(self.nuisance_basis, dtype=np.float64)
            strength = float(self.nuisance_projection_strength)
            if basis.ndim != 2:
                raise RuntimeError("Subject-nuisance basis must be two-dimensional.")
            if not 0.0 < strength <= 1.0:
                raise RuntimeError(
                    "Subject-nuisance projection strength must lie in (0,1]."
                )
            if basis.shape[1] == primary.shape[1]:
                primary = primary - strength * (primary @ basis.T) @ basis
                primary_norms = np.linalg.norm(primary, axis=1, keepdims=True)
                if np.any(primary_norms <= 1e-12):
                    raise RuntimeError(
                        "Subject-nuisance projection collapsed a structural trajectory."
                    )
                primary /= primary_norms
            elif (
                self.protected_columns is not None
                and len(self.protected_columns)
                and basis.shape[1] == self.output_dim
            ):
                legacy_combined_basis = basis
            else:
                raise RuntimeError("Subject-nuisance basis shape differs from features.")

        if self.protected_columns is None or not len(self.protected_columns):
            combined = primary
        else:
            if self.protected_mean is None or self.protected_scale is None:
                raise RuntimeError("Protected descriptor block lacks scaling state.")
            weight = float(self.protected_distance_weight)
            if not 0.0 < weight < 1.0:
                raise RuntimeError("Protected-block distance weight must lie in (0,1).")
            protected = (
                matrix[:, self.protected_columns] - self.protected_mean
            ) / self.protected_scale
            # Squared Euclidean distance receives approximately ``weight`` of
            # its energy from this block without normalising quiet trials into
            # artificial full-strength vertical evidence.
            protected *= math.sqrt(weight / len(self.protected_columns))
            primary *= math.sqrt(1.0 - weight)
            combined = np.c_[primary, protected]

        if legacy_combined_basis is not None:
            combined = (
                combined
                - float(self.nuisance_projection_strength)
                * (combined @ legacy_combined_basis.T)
                @ legacy_combined_basis
            )
        norms = np.linalg.norm(combined, axis=1, keepdims=True)
        if np.any(norms <= 1e-12):
            raise RuntimeError("Descriptor projection collapsed at least one trajectory.")
        return (combined / norms).astype(np.float32)


def fit_descriptor_transform(
    values: np.ndarray,
    *,
    maximum_components: int = 32,
    protected_columns: Sequence[int] | np.ndarray | None = None,
    protected_distance_weight: float = 0.15,
) -> DescriptorTransform:
    matrix = np.asarray(values, dtype=np.float64)
    if matrix.ndim != 2 or len(matrix) < 2 or matrix.shape[1] < 1:
        raise ValueError("Descriptor fit matrix must be [N>=2,D>=1].")
    protected = np.asarray(
        [] if protected_columns is None else protected_columns, dtype=np.int64
    )
    if protected.ndim != 1 or len(np.unique(protected)) != len(protected):
        raise ValueError("Protected descriptor columns must be one-dimensional and unique.")
    if len(protected) and (
        np.any(protected < 0) or np.any(protected >= matrix.shape[1])
    ):
        raise ValueError("Protected descriptor column lies outside the input matrix.")
    if len(protected) and not 0.0 < float(protected_distance_weight) < 1.0:
        raise ValueError("protected_distance_weight must lie in (0,1).")
    eligible = np.ones(matrix.shape[1], dtype=bool)
    eligible[protected] = False
    keep = np.flatnonzero((matrix.std(axis=0) > 1e-10) & eligible)
    if not len(keep):
        raise RuntimeError("Every descriptor column is constant.")
    selected = matrix[:, keep]
    mean = selected.mean(axis=0)
    scale = np.maximum(selected.std(axis=0), 1e-8)
    standardized = (selected - mean) / scale
    output_dim = min(
        int(maximum_components), len(standardized) - 1, standardized.shape[1]
    )
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
    return DescriptorTransform(
        keep_columns=keep.astype(np.int64),
        mean=mean.astype(np.float64),
        scale=scale.astype(np.float64),
        pca_mean=None if pca_mean is None else pca_mean.astype(np.float64),
        pca_components=None if components is None else components.astype(np.float64),
        protected_columns=(protected if len(protected) else None),
        protected_mean=(
            matrix[:, protected].mean(axis=0).astype(np.float64)
            if len(protected) else None
        ),
        protected_scale=(
            np.maximum(matrix[:, protected].std(axis=0), 1e-8).astype(np.float64)
            if len(protected) else None
        ),
        protected_distance_weight=(
            float(protected_distance_weight) if len(protected) else 0.0
        ),
    )


def fit_subject_nuisance_projection(
    transform: DescriptorTransform,
    values: np.ndarray,
    subject_ids: Sequence[int] | np.ndarray,
    *,
    maximum_rank: int = 4,
    explained_variance: float = 0.90,
    projection_strength: float = 1.0,
) -> tuple[DescriptorTransform, dict[str, Any]]:
    """Fit a source-subject nuisance basis in the chosen descriptor coordinates.

    The source protocol contains the same five trials from each of six old
    activities for every training subject.  Equal-weight subject centroids
    therefore cancel the common activity mixture without opening activity
    labels.  Only source subject metadata are used for this basis; target
    subject IDs are not required by the returned transform.  The surrounding
    descriptor coordinate transform may still have been fitted transductively
    from outer unlabeled descriptors, so the complete transform is not
    accurately described as source-only.
    """

    if transform.nuisance_basis is not None:
        raise ValueError("Descriptor transform already contains a nuisance basis.")
    matrix = np.asarray(values, dtype=np.float64)
    subjects = np.asarray(subject_ids, dtype=np.int64)
    if matrix.ndim != 2 or subjects.shape != (len(matrix),):
        raise ValueError("Nuisance-fit values and subject IDs are not row aligned.")
    unique = np.unique(subjects)
    if len(unique) < 2 or int(maximum_rank) < 1:
        raise ValueError("Subject nuisance fitting requires at least two subjects and rank >= 1.")
    if not 0.0 < float(explained_variance) <= 1.0:
        raise ValueError("explained_variance must lie in (0,1].")
    if not 0.0 < float(projection_strength) <= 1.0:
        raise ValueError("projection_strength must lie in (0,1].")
    # Subject nuisance is learned solely in the structural primary block.
    # Protected signed-vertical coordinates never enter this SVD and therefore
    # cannot be removed by the fitted projection.
    base = transform._normalized_primary(matrix).astype(np.float64)
    centroids = np.stack([base[subjects == subject].mean(axis=0) for subject in unique])
    effects = centroids - centroids.mean(axis=0, keepdims=True)
    _, singular, right = np.linalg.svd(effects, full_matrices=False)
    nonzero = int(np.count_nonzero(singular > max(float(singular[0]), 1.0) * 1e-10))
    if nonzero < 1:
        raise RuntimeError("Source subject centroids contain no measurable nuisance direction.")
    energy = np.square(singular[:nonzero])
    required = int(np.searchsorted(np.cumsum(energy) / energy.sum(), explained_variance) + 1)
    rank = min(int(maximum_rank), nonzero, required, base.shape[1] - 1)
    if rank < 1:
        raise RuntimeError("Subject nuisance rank collapsed to zero.")
    basis = right[:rank].copy()
    for row in basis:
        pivot = int(np.argmax(np.abs(row)))
        if row[pivot] < 0:
            row *= -1.0
    explained = float(energy[:rank].sum() / energy.sum())
    projected = base - float(projection_strength) * (base @ basis.T) @ basis
    before = float(np.mean(np.linalg.norm(effects, axis=1)))
    after_centroids = np.stack(
        [projected[subjects == subject].mean(axis=0) for subject in unique]
    )
    after_effects = after_centroids - after_centroids.mean(axis=0, keepdims=True)
    after = float(np.mean(np.linalg.norm(after_effects, axis=1)))
    fitted = replace(
        transform,
        nuisance_basis=basis.astype(np.float64),
        nuisance_singular_values=singular[:rank].astype(np.float64),
        nuisance_explained_fraction=explained,
        nuisance_fit_subject_count=int(len(unique)),
        nuisance_fit_trial_count=int(len(matrix)),
        nuisance_projection_strength=float(projection_strength),
    )
    return fitted, {
        "mode": (
            "offline_old6_balanced_subject_centroid_svd_hard"
            if float(projection_strength) == 1.0
            else "offline_old6_balanced_subject_centroid_svd_soft"
        ),
        "target_subject_ids_required": False,
        "activity_labels_used": False,
        "fit_subject_count": int(len(unique)),
        "fit_trial_count": int(len(matrix)),
        "maximum_rank": int(maximum_rank),
        "selected_rank": int(rank),
        "requested_explained_variance": float(explained_variance),
        "selected_explained_fraction": explained,
        "projection_strength": float(projection_strength),
        "source_subject_centroid_norm_before": before,
        "source_subject_centroid_norm_after_linear_projection": after,
    }


def descriptor_matrix(
    trials: Sequence[PrimitiveTrajectory],
    primitive_num: int,
    *,
    include_state: bool = True,
    descriptor_profile: str = LEGACY_DESCRIPTOR_PROFILE,
) -> tuple[np.ndarray, tuple[str, ...]]:
    if not trials:
        raise ValueError("At least one trajectory is required.")
    rows = [
        trajectory_descriptor(
            trial,
            primitive_num,
            include_state=include_state,
            descriptor_profile=descriptor_profile,
        )
        for trial in trials
    ]
    schemas = {names for _, names in rows}
    if len(schemas) != 1:
        raise RuntimeError("Trajectory descriptor schemas disagree.")
    return np.stack([vector for vector, _ in rows]), rows[0][1]


def save_frozen_artifacts(
    path: Path,
    codebook: FrozenE0Codebook,
    descriptor_transform: DescriptorTransform,
    descriptor_names: Sequence[str],
    metadata: Mapping[str, Any],
) -> Path:
    codebook.validate(strict_historical=(codebook.primitive_num, codebook.pca_dim) == (32, 64))
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": np.asarray(SCHEMA),
        "metadata_json": np.asarray(
            json.dumps(dict(metadata), ensure_ascii=False, sort_keys=True)
        ),
        "descriptor_names_json": np.asarray(
            json.dumps(list(descriptor_names), ensure_ascii=False)
        ),
        "primitive_num": np.asarray(codebook.primitive_num, dtype=np.int64),
        "pca_dim": np.asarray(codebook.pca_dim, dtype=np.int64),
        "input_dim": np.asarray(codebook.input_dim, dtype=np.int64),
        "codebook_pca_mean": codebook.pca_mean,
        "codebook_pca_components": codebook.pca_components,
        "cluster_centers": codebook.cluster_centers,
        "seed": np.asarray(codebook.seed, dtype=np.int64),
        "fit_window_count": np.asarray(codebook.fit_window_count, dtype=np.int64),
        "fit_trial_count": np.asarray(codebook.fit_trial_count, dtype=np.int64),
        "fit_subject_count": np.asarray(codebook.fit_subject_count, dtype=np.int64),
        "fit_data_sha256": np.asarray(codebook.fit_data_sha256),
        "kmeans_inertia": np.asarray(codebook.kmeans_inertia, dtype=np.float64),
        "kmeans_n_iter": np.asarray(codebook.kmeans_n_iter, dtype=np.int64),
        "descriptor_keep_columns": descriptor_transform.keep_columns,
        "descriptor_mean": descriptor_transform.mean,
        "descriptor_scale": descriptor_transform.scale,
        "descriptor_has_pca": np.asarray(
            descriptor_transform.pca_components is not None, dtype=np.bool_
        ),
        "descriptor_pca_mean": (
            np.asarray([], dtype=np.float64)
            if descriptor_transform.pca_mean is None
            else descriptor_transform.pca_mean
        ),
        "descriptor_pca_components": (
            np.empty((0, 0), dtype=np.float64)
            if descriptor_transform.pca_components is None
            else descriptor_transform.pca_components
        ),
        "descriptor_protected_columns": (
            np.asarray([], dtype=np.int64)
            if descriptor_transform.protected_columns is None
            else descriptor_transform.protected_columns
        ),
        "descriptor_protected_mean": (
            np.asarray([], dtype=np.float64)
            if descriptor_transform.protected_mean is None
            else descriptor_transform.protected_mean
        ),
        "descriptor_protected_scale": (
            np.asarray([], dtype=np.float64)
            if descriptor_transform.protected_scale is None
            else descriptor_transform.protected_scale
        ),
        "descriptor_protected_distance_weight": np.asarray(
            descriptor_transform.protected_distance_weight, dtype=np.float64
        ),
        "descriptor_nuisance_basis": (
            np.empty((0, descriptor_transform.output_dim), dtype=np.float64)
            if descriptor_transform.nuisance_basis is None
            else descriptor_transform.nuisance_basis
        ),
        "descriptor_nuisance_singular_values": (
            np.asarray([], dtype=np.float64)
            if descriptor_transform.nuisance_singular_values is None
            else descriptor_transform.nuisance_singular_values
        ),
        "descriptor_nuisance_explained_fraction": np.asarray(
            descriptor_transform.nuisance_explained_fraction, dtype=np.float64
        ),
        "descriptor_nuisance_fit_subject_count": np.asarray(
            descriptor_transform.nuisance_fit_subject_count, dtype=np.int64
        ),
        "descriptor_nuisance_fit_trial_count": np.asarray(
            descriptor_transform.nuisance_fit_trial_count, dtype=np.int64
        ),
        "descriptor_nuisance_projection_strength": np.asarray(
            descriptor_transform.nuisance_projection_strength, dtype=np.float64
        ),
        "numpy_version": np.asarray(np.__version__),
        "sklearn_version": np.asarray(sklearn.__version__),
    }
    temporary = target.with_suffix(target.suffix + ".tmp.npz")
    np.savez_compressed(temporary, **payload)
    temporary.replace(target)
    return target


def load_frozen_artifacts(
    path: Path,
) -> tuple[FrozenE0Codebook, DescriptorTransform, tuple[str, ...], dict[str, Any]]:
    with np.load(Path(path), allow_pickle=False) as data:
        artifact_schema = str(data["schema"].item())
        if artifact_schema not in {SCHEMA, PREVIOUS_SCHEMA, LEGACY_SCHEMA}:
            raise RuntimeError(f"Unexpected frozen-artifact schema {data['schema'].item()!r}.")
        codebook = FrozenE0Codebook(
            primitive_num=int(data["primitive_num"]),
            pca_dim=int(data["pca_dim"]),
            input_dim=int(data["input_dim"]),
            pca_mean=np.asarray(data["codebook_pca_mean"], dtype=np.float32),
            pca_components=np.asarray(data["codebook_pca_components"], dtype=np.float32),
            cluster_centers=np.asarray(data["cluster_centers"], dtype=np.float32),
            seed=int(data["seed"]),
            fit_window_count=int(data["fit_window_count"]),
            fit_trial_count=int(data["fit_trial_count"]),
            fit_subject_count=int(data["fit_subject_count"]),
            fit_data_sha256=str(data["fit_data_sha256"].item()),
            kmeans_inertia=float(data["kmeans_inertia"]),
            kmeans_n_iter=int(data["kmeans_n_iter"]),
        ).validate(strict_historical=(int(data["primitive_num"]) == 32 and int(data["pca_dim"]) == 64))
        has_pca = bool(data["descriptor_has_pca"])
        extended = artifact_schema in {SCHEMA, PREVIOUS_SCHEMA}
        protected_columns = (
            np.asarray(data["descriptor_protected_columns"], dtype=np.int64)
            if extended and len(data["descriptor_protected_columns"])
            else None
        )
        nuisance_basis = (
            np.asarray(data["descriptor_nuisance_basis"], dtype=np.float64)
            if extended and len(data["descriptor_nuisance_basis"])
            else None
        )
        transform = DescriptorTransform(
            keep_columns=np.asarray(data["descriptor_keep_columns"], dtype=np.int64),
            mean=np.asarray(data["descriptor_mean"], dtype=np.float64),
            scale=np.asarray(data["descriptor_scale"], dtype=np.float64),
            pca_mean=(
                np.asarray(data["descriptor_pca_mean"], dtype=np.float64)
                if has_pca else None
            ),
            pca_components=(
                np.asarray(data["descriptor_pca_components"], dtype=np.float64)
                if has_pca else None
            ),
            protected_columns=protected_columns,
            protected_mean=(
                np.asarray(data["descriptor_protected_mean"], dtype=np.float64)
                if protected_columns is not None else None
            ),
            protected_scale=(
                np.asarray(data["descriptor_protected_scale"], dtype=np.float64)
                if protected_columns is not None else None
            ),
            protected_distance_weight=(
                float(data["descriptor_protected_distance_weight"])
                if extended else 0.0
            ),
            nuisance_basis=nuisance_basis,
            nuisance_singular_values=(
                np.asarray(data["descriptor_nuisance_singular_values"], dtype=np.float64)
                if nuisance_basis is not None else None
            ),
            nuisance_explained_fraction=(
                float(data["descriptor_nuisance_explained_fraction"])
                if extended else 0.0
            ),
            nuisance_fit_subject_count=(
                int(data["descriptor_nuisance_fit_subject_count"])
                if extended else 0
            ),
            nuisance_fit_trial_count=(
                int(data["descriptor_nuisance_fit_trial_count"])
                if extended else 0
            ),
            nuisance_projection_strength=(
                float(data["descriptor_nuisance_projection_strength"])
                if extended
                and "descriptor_nuisance_projection_strength" in data.files
                else (1.0 if nuisance_basis is not None else 0.0)
            ),
        )
        names = tuple(json.loads(str(data["descriptor_names_json"].item())))
        metadata = dict(json.loads(str(data["metadata_json"].item())))
    return codebook, transform, names, metadata


__all__ = [
    "ABSOLUTE_DURATION_DESCRIPTOR_NAMES", "CHANNEL_NAMES", "DESCRIPTOR_PROFILES",
    "DURATION_INVARIANT_DESCRIPTOR_PROFILE", "DURATION_SOFT_SUBJECT_A025_PROFILE",
    "DURATION_SOFT_SUBJECT_A050_PROFILE", "DURATION_SOFT_SUBJECT_A075_PROFILE",
    "DescriptorProfileSpec",
    "DescriptorTransform", "FULL_DEBIASED_DESCRIPTOR_PROFILE",
    "FrozenE0Codebook", "GRAVITY_DESCRIPTOR_PROFILE", "GravityAlignedState",
    "LEGACY_DESCRIPTOR_PROFILE", "LEGACY_SCHEMA", "PrimitiveTrajectory", "SCHEMA",
    "SIGNED_VERTICAL_NAMES", "SUBJECT_DEBIASED_DESCRIPTOR_PROFILE", "WindowGrid",
    "build_e0_trajectories", "descriptor_matrix", "descriptor_profile_spec",
    "fit_descriptor_transform", "fit_frozen_e0_codebook",
    "fit_subject_nuisance_projection", "gravity_aligned_trial_state",
    "fold_normalization", "load_frozen_artifacts", "load_window_grid",
    "normalize_windows", "raw_descriptor_dimension", "save_frozen_artifacts",
    "statistic_names", "trajectory_descriptor", "reconstruct_trial_channels",
    "window_ownership_partition", "window_statistics",
]
