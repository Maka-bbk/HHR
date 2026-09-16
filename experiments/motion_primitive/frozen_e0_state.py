"""Frozen A2 -> E0/K32 -> state trajectory representation.

This module is the numerical centre of the default HHR route.  It deliberately
contains no learnable codebook, learned boundary head, recurrent readout, class
label, or online update.  Every transform is fitted on the registered offline
old-class training split and can then be applied unchanged to validation and
all online sessions.

The implementation keeps the historical E0 floating-point order:

``A2 content -> row L2 -> trial-equal weighted PCA64 -> row L2 ->
KMeans32 -> cosine hard assignment``.

The raw state descriptor is intentionally schema-compatible with the validated
experiment, including the all-zero E0 peak/valley transition blocks.  Those
constant columns are removed only by :class:`DescriptorTransform`.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import sklearn
from sklearn.cluster import KMeans


CHANNEL_NAMES = (
    "acc_x",
    "acc_y",
    "acc_z",
    "gyro_x",
    "gyro_y",
    "gyro_z",
)
STATE_DESCRIPTOR_RAW_DIM = 3739
NO_STATE_DESCRIPTOR_RAW_DIM = 3275
HISTORICAL_STATE_SCHEMA_SHA256 = (
    "dbdceb949b66e900400dc9908286f5ac371db4d04b56aaf6e90f8d206c7e70cc"
)


def canonical_json_sha256(value: Mapping[str, Any]) -> str:
    text = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def array_sha256(*arrays: np.ndarray) -> str:
    digest = hashlib.sha256()
    for index, value in enumerate(arrays):
        array = np.ascontiguousarray(np.asarray(value))
        if array.dtype.kind == "f" and not np.all(np.isfinite(array)):
            raise ValueError("Cannot hash a non-finite array.")
        digest.update(str(index).encode("ascii"))
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(str(array.shape).encode("ascii"))
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def l2_normalize(values: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    matrix = np.asarray(values, dtype=np.float32)
    if matrix.ndim != 2 or not len(matrix):
        raise ValueError(f"Expected a non-empty [N,D] matrix, got {matrix.shape}.")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("L2 input contains non-finite values.")
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    if np.any(norms <= float(eps)):
        raise ValueError("L2 input contains a zero-norm row.")
    return (matrix / np.maximum(norms, np.float32(eps))).astype(np.float32)


def inverse_trial_frequency_weights(trial_ids: np.ndarray) -> np.ndarray:
    trial_ids = np.asarray(trial_ids, dtype=np.int64)
    if trial_ids.ndim != 1 or not len(trial_ids):
        raise ValueError("trial_ids must be a non-empty vector.")
    _, inverse, counts = np.unique(trial_ids, return_inverse=True, return_counts=True)
    weights = 1.0 / counts[inverse].astype(np.float64)
    return weights * (len(weights) / weights.sum())


@dataclass(frozen=True)
class WeightedPCA:
    mean: np.ndarray
    components: np.ndarray
    explained_variance: np.ndarray
    explained_variance_ratio: np.ndarray

    def transform(self, values: np.ndarray) -> np.ndarray:
        matrix = np.asarray(values, dtype=np.float32)
        if matrix.ndim != 2 or matrix.shape[1] != len(self.mean):
            raise ValueError("WeightedPCA input dimension mismatch.")
        return ((matrix - self.mean) @ self.components.T).astype(np.float32)


def fit_weighted_pca(
    values: np.ndarray,
    n_components: int,
    sample_weights: np.ndarray,
) -> WeightedPCA:
    matrix = np.asarray(values, dtype=np.float64)
    weights = np.asarray(sample_weights, dtype=np.float64)
    if matrix.ndim != 2 or len(matrix) < 2:
        raise ValueError("Weighted PCA requires [N>=2,D].")
    if weights.shape != (len(matrix),) or np.any(weights < 0) or not np.all(np.isfinite(weights)):
        raise ValueError("Weighted PCA received invalid sample weights.")
    maximum = min(len(matrix) - 1, matrix.shape[1])
    if not 1 <= int(n_components) <= maximum:
        raise ValueError(f"PCA components must lie in [1,{maximum}].")
    weights = weights / weights.sum()
    mean = np.sum(matrix * weights[:, None], axis=0)
    centered = matrix - mean
    covariance = centered.T @ (centered * weights[:, None])
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = np.maximum(eigenvalues[order], 0.0)
    components = eigenvectors[:, order[: int(n_components)]].T
    selected = eigenvalues[: int(n_components)]
    return WeightedPCA(
        mean=mean.astype(np.float32),
        components=components.astype(np.float32),
        explained_variance=selected.astype(np.float32),
        explained_variance_ratio=(selected / max(float(eigenvalues.sum()), 1e-12)).astype(np.float32),
    )


@dataclass(frozen=True)
class WindowTrial:
    """One complete ordered set of saved windows from a USC-HAD trial."""

    trial_id: int
    subject_id: int
    window_starts: np.ndarray
    raw_windows: np.ndarray
    content_embeddings: np.ndarray

    def validate(self, *, window_size: int | None = None) -> "WindowTrial":
        starts = np.asarray(self.window_starts, dtype=np.int64)
        raw = np.asarray(self.raw_windows, dtype=np.float32)
        content = np.asarray(self.content_embeddings, dtype=np.float32)
        if starts.ndim != 1 or not len(starts) or starts[0] != 0 or np.any(np.diff(starts) <= 0):
            raise ValueError("A WindowTrial must start at zero and be strictly time ordered.")
        if raw.ndim != 3 or raw.shape[:2] != (len(starts), 6):
            raise ValueError(f"raw_windows must be [L,6,T], got {raw.shape}.")
        if window_size is not None and raw.shape[2] != int(window_size):
            raise ValueError("WindowTrial window size differs from the registered grid.")
        if content.ndim != 2 or len(content) != len(starts) or content.shape[1] < 1:
            raise ValueError("content_embeddings must be [L,D].")
        if not np.all(np.isfinite(raw)) or not np.all(np.isfinite(content)):
            raise ValueError("WindowTrial contains non-finite values.")
        return self


@dataclass(frozen=True)
class FrozenE0Codebook:
    primitive_num: int
    pca_dim: int
    input_dim: int
    pca_mean: np.ndarray
    pca_components: np.ndarray
    cluster_centers: np.ndarray
    seed: int
    fit_trial_count: int
    fit_window_count: int
    fit_subject_count: int
    fit_trial_ids_sha256: str
    fit_used_k: int
    inertia: float
    iterations: int
    sklearn_version: str

    def validate(self) -> "FrozenE0Codebook":
        if int(self.primitive_num) != 32 or int(self.pca_dim) != 64:
            raise ValueError("The registered E0 route is fixed to PCA64/KMeans32.")
        if np.asarray(self.pca_mean).shape != (int(self.input_dim),):
            raise ValueError("Codebook PCA mean shape mismatch.")
        if np.asarray(self.pca_components).shape != (64, int(self.input_dim)):
            raise ValueError("Codebook PCA component shape mismatch.")
        if np.asarray(self.cluster_centers).shape != (32, 64):
            raise ValueError("Codebook centre shape mismatch.")
        for value in (self.pca_mean, self.pca_components, self.cluster_centers):
            if not np.all(np.isfinite(value)):
                raise ValueError("Codebook contains non-finite state.")
        if int(self.fit_used_k) != 32:
            raise RuntimeError("Registered offline KMeans fit did not use all 32 codes.")
        return self

    @property
    def state_sha256(self) -> str:
        metadata = np.asarray(
            [self.primitive_num, self.pca_dim, self.input_dim, self.seed, self.fit_trial_count,
             self.fit_window_count, self.fit_subject_count, self.fit_used_k],
            dtype=np.int64,
        )
        return array_sha256(metadata, self.pca_mean, self.pca_components, self.cluster_centers)

    def transform(self, content_embeddings: np.ndarray) -> np.ndarray:
        base = l2_normalize(content_embeddings)
        projected = ((base - np.asarray(self.pca_mean, dtype=np.float32)) @ np.asarray(self.pca_components, dtype=np.float32).T).astype(np.float32)
        return l2_normalize(projected)

    def assign(self, content_embeddings: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        embedded = self.transform(content_embeddings)
        centres = l2_normalize(np.asarray(self.cluster_centers, dtype=np.float32))
        similarity = embedded @ centres.T
        tokens = np.argmax(similarity, axis=1).astype(np.int64)
        distances = (1.0 - similarity[np.arange(len(tokens)), tokens]).astype(np.float32)
        return tokens, distances, embedded


def fit_e0_codebook(
    trials: Sequence[WindowTrial],
    *,
    seed: int,
    primitive_num: int = 32,
    pca_dim: int = 64,
) -> FrozenE0Codebook:
    ordered = sorted((item.validate() for item in trials), key=lambda item: int(item.trial_id))
    if len(ordered) < 2 or len({item.trial_id for item in ordered}) != len(ordered):
        raise ValueError("E0 fit needs at least two uniquely identified trials.")
    if int(primitive_num) != 32 or int(pca_dim) != 64:
        raise ValueError("The registered E0 route is fixed to PCA64/KMeans32.")
    dimensions = {item.content_embeddings.shape[1] for item in ordered}
    if len(dimensions) != 1:
        raise ValueError("E0 trials do not share a content dimension.")
    raw = np.concatenate([np.asarray(item.content_embeddings, dtype=np.float32) for item in ordered])
    trial_ids = np.concatenate([
        np.full(len(item.window_starts), int(item.trial_id), dtype=np.int64) for item in ordered
    ])
    base = l2_normalize(raw)
    weights = inverse_trial_frequency_weights(trial_ids)
    pca = fit_weighted_pca(base, 64, weights)
    embedded = l2_normalize(pca.transform(base))
    model = KMeans(
        n_clusters=32,
        random_state=int(seed),
        n_init=20,
        max_iter=300,
        algorithm="lloyd",
    )
    fit_tokens = model.fit_predict(embedded, sample_weight=weights).astype(np.int64)
    used_k = int(len(np.unique(fit_tokens)))
    result = FrozenE0Codebook(
        primitive_num=32,
        pca_dim=64,
        input_dim=int(raw.shape[1]),
        pca_mean=pca.mean.copy(),
        pca_components=pca.components.copy(),
        cluster_centers=np.asarray(model.cluster_centers_, dtype=np.float32).copy(),
        seed=int(seed),
        fit_trial_count=len(ordered),
        fit_window_count=len(raw),
        fit_subject_count=len({int(item.subject_id) for item in ordered}),
        fit_trial_ids_sha256=array_sha256(np.asarray([item.trial_id for item in ordered], dtype=np.int64)),
        fit_used_k=used_k,
        inertia=float(model.inertia_),
        iterations=int(model.n_iter_),
        sklearn_version=str(sklearn.__version__),
    )
    return result.validate()


def _window_partition(starts: np.ndarray, window_size: int) -> tuple[np.ndarray, np.ndarray]:
    starts = np.asarray(starts, dtype=np.int64)
    if starts.ndim != 1 or not len(starts) or starts[0] != 0:
        raise ValueError("Window starts must be non-empty and begin at zero.")
    centres = starts.astype(np.float64) + float(window_size) / 2.0
    boundaries = [0]
    boundaries.extend(int(round(0.5 * (left + right))) for left, right in zip(centres[:-1], centres[1:]))
    boundaries.append(int(starts[-1]) + int(window_size))
    values = np.asarray(boundaries, dtype=np.int64)
    if np.any(np.diff(values) <= 0):
        raise RuntimeError("Window-centre Voronoi ownership is not strictly increasing.")
    return values[:-1], values[1:]


def statistic_names() -> tuple[str, ...]:
    names = ["duration_seconds"]
    for prefix in ("mean", "std", "log_mean_square_energy", "end_minus_start"):
        names.extend(f"{prefix}__{name}" for name in CHANNEL_NAMES)
    return tuple(names)


def _window_statistics(raw_windows: np.ndarray, durations: np.ndarray, sample_rate_hz: float) -> np.ndarray:
    windows = np.asarray(raw_windows, dtype=np.float64)
    result = []
    for window, duration in zip(windows, np.asarray(durations, dtype=np.float64)):
        result.append(np.r_[
            float(duration) / float(sample_rate_hz),
            window.mean(axis=1),
            window.std(axis=1),
            np.log(np.mean(np.square(window), axis=1) + 1e-8),
            window[:, -1] - window[:, 0],
        ])
    return np.asarray(result, dtype=np.float64)


@dataclass(frozen=True)
class PrimitiveTrial:
    """A label-free E0 motion-primitive trajectory."""

    trial_id: int
    subject_id: int
    starts: np.ndarray
    ends: np.ndarray
    child_tokens: np.ndarray
    child_distances: np.ndarray
    child_embeddings: np.ndarray
    child_statistics: np.ndarray
    statistic_names: tuple[str, ...]
    event_kinds: tuple[str, ...] = ()

    def validate(self, primitive_num: int = 32) -> "PrimitiveTrial":
        count = len(self.child_tokens)
        if count < 1:
            raise ValueError("A primitive trajectory cannot be empty.")
        if np.asarray(self.starts).shape != (count,) or np.asarray(self.ends).shape != (count,):
            raise ValueError("Primitive ownership spans differ from token count.")
        if self.starts[0] != 0 or np.any(self.starts[1:] != self.ends[:-1]) or np.any(self.ends <= self.starts):
            raise ValueError("Primitive ownership spans must be a contiguous positive partition.")
        if np.asarray(self.child_distances).shape != (count,):
            raise ValueError("Primitive distances differ from token count.")
        if np.asarray(self.child_embeddings).ndim != 2 or len(self.child_embeddings) != count:
            raise ValueError("Primitive embeddings differ from token count.")
        if np.asarray(self.child_statistics).ndim != 2 or len(self.child_statistics) != count:
            raise ValueError("Primitive statistics differ from token count.")
        if len(self.statistic_names) != self.child_statistics.shape[1]:
            raise ValueError("Primitive statistic schema differs from values.")
        if len(self.event_kinds) not in (0, count - 1):
            raise ValueError("E0 event_kinds must be empty; variable segmentation needs N-1 kinds.")
        tokens = np.asarray(self.child_tokens, dtype=np.int64)
        if np.any(tokens < 0) or np.any(tokens >= int(primitive_num)):
            raise ValueError("A primitive token lies outside the codebook.")
        for value in (self.child_distances, self.child_embeddings, self.child_statistics):
            if not np.all(np.isfinite(value)):
                raise ValueError("Primitive trajectory contains non-finite state.")
        return self


def tokenize_e0_trial(
    trial: WindowTrial,
    codebook: FrozenE0Codebook,
    *,
    sample_rate_hz: float = 100.0,
) -> PrimitiveTrial:
    trial.validate()
    codebook.validate()
    window_size = int(trial.raw_windows.shape[2])
    starts, ends = _window_partition(trial.window_starts, window_size)
    tokens, distances, embedded = codebook.assign(trial.content_embeddings)
    statistics = _window_statistics(trial.raw_windows, ends - starts, float(sample_rate_hz))
    return PrimitiveTrial(
        trial_id=int(trial.trial_id),
        subject_id=int(trial.subject_id),
        starts=starts,
        ends=ends,
        child_tokens=tokens,
        child_distances=distances.astype(np.float64),
        child_embeddings=embedded.astype(np.float64),
        child_statistics=statistics,
        statistic_names=statistic_names(),
        event_kinds=(),
    ).validate(32)


def codebook_usage(trials: Sequence[PrimitiveTrial], primitive_num: int = 32) -> dict[str, Any]:
    tokens = np.concatenate([np.asarray(item.child_tokens, dtype=np.int64) for item in trials])
    counts = np.bincount(tokens, minlength=int(primitive_num)).astype(np.int64)
    probabilities = counts / max(float(counts.sum()), 1.0)
    positive = probabilities > 0
    entropy = float(-np.sum(probabilities[positive] * np.log(probabilities[positive])))
    used = int(np.count_nonzero(counts))
    return {
        "capacity_k": int(primitive_num),
        "used_k": used,
        "dead_k": int(primitive_num) - used,
        "dead_fraction": float((int(primitive_num) - used) / int(primitive_num)),
        "effective_k": float(np.exp(entropy)),
        "counts": counts.tolist(),
        "fractions": probabilities.tolist(),
    }


def trajectory_descriptor(
    trial: PrimitiveTrial,
    primitive_num: int = 32,
    *,
    include_state: bool = True,
) -> tuple[np.ndarray, tuple[str, ...]]:
    """Build the exact E0 state descriptor without complete-trial mean/max pooling."""

    trial.validate(int(primitive_num))
    tokens = np.asarray(trial.child_tokens, dtype=np.int64)
    durations = (np.asarray(trial.ends) - np.asarray(trial.starts)).astype(np.float64)
    total_duration = float(durations.sum())
    values: list[float] = []
    names: list[str] = []

    def extend(prefix: str, vector: np.ndarray) -> None:
        flat = np.asarray(vector, dtype=np.float64).reshape(-1)
        values.extend(float(item) for item in flat)
        names.extend(f"{prefix}{index}" for index in range(len(flat)))

    count_hist = np.bincount(tokens, minlength=int(primitive_num)).astype(np.float64)
    count_hist /= max(float(len(tokens)), 1.0)
    duration_hist = np.bincount(tokens, weights=durations, minlength=int(primitive_num))
    duration_hist /= max(total_duration, 1.0)
    extend("child_count_fraction__", count_hist)
    extend("child_duration_fraction__", duration_hist)

    midpoints = 0.5 * (trial.starts.astype(np.float64) + trial.ends.astype(np.float64))
    relative = np.clip(midpoints / max(float(trial.ends[-1]), 1.0), 0.0, 1.0 - 1e-12)
    temporal = np.zeros((4, int(primitive_num)), dtype=np.float64)
    for token, duration, position in zip(tokens, durations, relative):
        temporal[min(3, int(position * 4.0)), int(token)] += float(duration)
    temporal /= max(total_duration, 1.0)
    extend("child_temporal_quartile_duration__", temporal)

    any_transition = np.zeros((int(primitive_num), int(primitive_num)), dtype=np.float64)
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
    duration_seconds = np.asarray(trial.child_statistics[:, 0], dtype=np.float64)
    scalar = np.asarray([
        math.log1p(len(tokens)),
        math.log1p(total_duration),
        float(np.mean(duration_seconds)),
        float(np.std(duration_seconds)),
        float(np.max(duration_seconds)),
        float(np.mean(quality)),
        float(np.std(quality)),
        float(np.max(quality)),
        float(sum(kind == "peak" for kind in trial.event_kinds)) / max(1, len(trial.event_kinds)),
        float(sum(kind == "valley" for kind in trial.event_kinds)) / max(1, len(trial.event_kinds)),
    ])
    values.extend(scalar.tolist())
    names.extend((
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
    ))

    # E0 has no learned parent primitives.  Keep the historical empty parent
    # blocks and its scalar count so the raw schema remains comparable.
    values.append(0.0)
    names.append("log_parent_occurrence_count")

    if include_state:
        statistics = np.asarray(trial.child_statistics[:, 1:], dtype=np.float64)
        weights = durations / max(total_duration, 1.0)
        global_mean = np.sum(statistics * weights[:, None], axis=0)
        global_std = np.sqrt(np.sum(np.square(statistics - global_mean) * weights[:, None], axis=0))
        state_names = trial.statistic_names[1:]
        values.extend(global_mean.tolist())
        names.extend(f"state_global_mean__{name}" for name in state_names)
        values.extend(global_std.tolist())
        names.extend(f"state_global_std__{name}" for name in state_names)
        selected_columns = [
            index for index, name in enumerate(state_names)
            if name.startswith("mean__") or name.startswith("log_mean_square_energy__")
        ]
        token_state = np.zeros((int(primitive_num), len(selected_columns)), dtype=np.float64)
        token_presence = np.zeros(int(primitive_num), dtype=np.float64)
        for token in range(int(primitive_num)):
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
    expected = STATE_DESCRIPTOR_RAW_DIM if include_state else NO_STATE_DESCRIPTOR_RAW_DIM
    if vector.shape != (expected,) or len(names) != expected or not np.all(np.isfinite(vector)):
        raise RuntimeError(
            f"Trajectory descriptor schema mismatch: values={vector.shape}, names={len(names)}, expected={expected}."
        )
    if include_state:
        observed_schema = hashlib.sha256("\n".join(names).encode("utf-8")).hexdigest()
        if observed_schema != HISTORICAL_STATE_SCHEMA_SHA256:
            raise RuntimeError(
                "State descriptor schema drifted from the validated historical E0 route: "
                f"{observed_schema}."
            )
    return vector, tuple(names)


@dataclass(frozen=True)
class DescriptorTransform:
    keep_columns: np.ndarray
    mean: np.ndarray
    scale: np.ndarray
    pca_mean: np.ndarray | None
    pca_components: np.ndarray | None
    schema_names: tuple[str, ...]
    fit_trial_ids_sha256: str

    def validate(self) -> "DescriptorTransform":
        keep = np.asarray(self.keep_columns, dtype=np.int64)
        if keep.ndim != 1 or not len(keep) or np.any(np.diff(keep) <= 0):
            raise ValueError("Descriptor keep_columns must be sorted and unique.")
        if np.asarray(self.mean).shape != (len(keep),) or np.asarray(self.scale).shape != (len(keep),):
            raise ValueError("Descriptor normalization state shape mismatch.")
        if np.any(np.asarray(self.scale) <= 0):
            raise ValueError("Descriptor scale must be positive.")
        if self.pca_components is not None:
            if self.pca_mean is None or np.asarray(self.pca_mean).shape != (len(keep),):
                raise ValueError("Descriptor PCA mean shape mismatch.")
            if np.asarray(self.pca_components).ndim != 2 or self.pca_components.shape[1] != len(keep):
                raise ValueError("Descriptor PCA component shape mismatch.")
        if len(self.schema_names) <= int(keep[-1]):
            raise ValueError("Descriptor schema is shorter than keep_columns.")
        return self

    @property
    def output_dim(self) -> int:
        return int(self.pca_components.shape[0]) if self.pca_components is not None else len(self.keep_columns)

    @property
    def schema_sha256(self) -> str:
        return hashlib.sha256("\n".join(self.schema_names).encode("utf-8")).hexdigest()

    @property
    def state_sha256(self) -> str:
        pca_mean = np.asarray([], dtype=np.float64) if self.pca_mean is None else self.pca_mean
        pca_components = np.empty((0, 0), dtype=np.float64) if self.pca_components is None else self.pca_components
        return array_sha256(self.keep_columns, self.mean, self.scale, pca_mean, pca_components)

    def transform(self, values: np.ndarray) -> np.ndarray:
        matrix = np.asarray(values, dtype=np.float64)
        if matrix.ndim == 1:
            matrix = matrix[None, :]
        if matrix.ndim != 2 or matrix.shape[1] != len(self.schema_names):
            raise ValueError("Raw descriptor dimension differs from the frozen schema.")
        matrix = (matrix[:, self.keep_columns] - self.mean) / self.scale
        if self.pca_components is not None:
            matrix = (matrix - self.pca_mean) @ self.pca_components.T
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        if np.any(norms <= 1e-12):
            raise RuntimeError("A transformed trajectory descriptor has zero norm.")
        return (matrix / norms).astype(np.float32)


def fit_descriptor_transform(
    trials: Sequence[PrimitiveTrial],
    *,
    maximum_components: int = 32,
    include_state: bool = True,
) -> tuple[DescriptorTransform, np.ndarray]:
    if len(trials) < 2:
        raise ValueError("Descriptor transform needs at least two offline fit trials.")
    rows = [trajectory_descriptor(item, 32, include_state=include_state) for item in trials]
    schemas = {names for _, names in rows}
    if len(schemas) != 1:
        raise RuntimeError("Offline trajectory descriptors do not share one schema.")
    matrix = np.stack([values for values, _ in rows])
    variation = matrix.std(axis=0)
    keep = np.flatnonzero(variation > 1e-10)
    if not len(keep):
        raise RuntimeError("Every trajectory descriptor column is constant.")
    selected = matrix[:, keep]
    mean = selected.mean(axis=0)
    scale = np.maximum(selected.std(axis=0), 1e-8)
    standardized = (selected - mean) / scale
    output_dim = min(int(maximum_components), len(standardized) - 1, standardized.shape[1])
    pca_mean: np.ndarray | None = None
    components: np.ndarray | None = None
    if output_dim < standardized.shape[1]:
        pca_mean = standardized.mean(axis=0)
        _, _, right = np.linalg.svd(standardized - pca_mean, full_matrices=False)
        components = right[:output_dim].copy()
        for row in components:
            pivot = int(np.argmax(np.abs(row)))
            if row[pivot] < 0:
                row *= -1.0
    trial_ids = np.asarray([int(item.trial_id) for item in trials], dtype=np.int64)
    transform = DescriptorTransform(
        keep_columns=keep.astype(np.int64),
        mean=mean.astype(np.float64),
        scale=scale.astype(np.float64),
        pca_mean=None if pca_mean is None else pca_mean.astype(np.float64),
        pca_components=None if components is None else components.astype(np.float64),
        schema_names=rows[0][1],
        fit_trial_ids_sha256=array_sha256(trial_ids),
    ).validate()
    return transform, transform.transform(matrix)


def descriptor_matrix(
    trials: Sequence[PrimitiveTrial],
    transform: DescriptorTransform,
    *,
    include_state: bool = True,
) -> np.ndarray:
    rows = [trajectory_descriptor(item, 32, include_state=include_state) for item in trials]
    if any(names != transform.schema_names for _, names in rows):
        raise RuntimeError("Evaluation descriptor schema differs from the offline fit schema.")
    return transform.transform(np.stack([values for values, _ in rows]))


__all__ = [
    "CHANNEL_NAMES",
    "DescriptorTransform",
    "FrozenE0Codebook",
    "HISTORICAL_STATE_SCHEMA_SHA256",
    "NO_STATE_DESCRIPTOR_RAW_DIM",
    "PrimitiveTrial",
    "STATE_DESCRIPTOR_RAW_DIM",
    "WeightedPCA",
    "WindowTrial",
    "array_sha256",
    "canonical_json_sha256",
    "codebook_usage",
    "descriptor_matrix",
    "fit_descriptor_transform",
    "fit_e0_codebook",
    "fit_weighted_pca",
    "inverse_trial_frequency_weights",
    "l2_normalize",
    "statistic_names",
    "tokenize_e0_trial",
    "trajectory_descriptor",
]
