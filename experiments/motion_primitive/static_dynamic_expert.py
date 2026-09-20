"""Label-free body-motion gate and branch experts for fixed-split HAR-GCD.

The gate is fitted only from trial-level inertial energy. It deliberately does
not receive activity labels or subject identities. The dynamic branch keeps
the duration-invariant motion-primitive trajectory representation. The static
branch combines posture, energy, signed gravity-axis dynamics, and a compact
motion-primitive block with explicit distance weights.

This module contains no scorer and no truth-store access. Keeping that API
boundary small makes accidental pre-truth leakage harder.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture

from experiments.motion_primitive.frozen_e0 import (
    DURATION_INVARIANT_DESCRIPTOR_PROFILE,
    DURATION_SOFT_SUBJECT_A025_PROFILE,
    PrimitiveTrajectory,
    descriptor_matrix,
    descriptor_profile_spec,
    fit_descriptor_transform,
    fit_subject_nuisance_projection,
    gravity_aligned_trial_state,
    reconstruct_trial_channels,
)


SAMPLE_RATE_HZ = 100.0
GATE_FEATURE_NAMES = (
    "log_acc_dynamic_rms",
    "log_gyro_rms",
    "log_acc_norm_std",
    "log_gravity_axis_dynamic_rms",
    "log_acc_jerk_rms",
)
POSTURE_FEATURE_NAMES = (
    "gravity_unit_x",
    "gravity_unit_y",
    "gravity_unit_z",
    "gravity_norm_g",
    "acc_mean_x",
    "acc_mean_y",
    "acc_mean_z",
    "edge_gravity_cosine",
)
ENERGY_FEATURE_NAMES = GATE_FEATURE_NAMES + (
    "log_acc_residual_rms_x",
    "log_acc_residual_rms_y",
    "log_acc_residual_rms_z",
    "log_gyro_rms_x",
    "log_gyro_rms_y",
    "log_gyro_rms_z",
)
STATIC_BLOCK_ORDER = ("posture", "signed_gravity", "energy", "motion_primitive")
DEFAULT_STATIC_BLOCK_WEIGHTS: Mapping[str, float] = {
    "posture": 0.35,
    "signed_gravity": 0.35,
    "energy": 0.20,
    "motion_primitive": 0.10,
}


def _array_sha256(*arrays: np.ndarray) -> str:
    digest = hashlib.sha256()
    for index, value in enumerate(arrays):
        array = np.ascontiguousarray(np.asarray(value))
        if array.dtype.kind == "f" and not np.all(np.isfinite(array)):
            raise ValueError("Cannot hash non-finite state.")
        digest.update(str(index).encode("ascii"))
        digest.update(array.dtype.str.encode("ascii"))
        digest.update(str(array.shape).encode("ascii"))
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _derived_seed(seed: int, offset: int) -> int:
    return int((int(seed) * 1_000_003 + int(offset)) % (2**32 - 1))


def _log_positive(value: np.ndarray | float, epsilon: float = 1e-8) -> np.ndarray:
    return np.log(np.maximum(np.asarray(value, dtype=np.float64), 0.0) + epsilon)


def trial_physical_blocks(
    raw_windows: np.ndarray,
    window_starts: np.ndarray,
) -> dict[str, np.ndarray]:
    """Return duration-free physical blocks from one complete raw trial."""

    signal = reconstruct_trial_channels(raw_windows, window_starts)
    acceleration = np.asarray(signal[:3].T, dtype=np.float64)
    gyroscope = np.asarray(signal[3:].T, dtype=np.float64)
    gravity = acceleration.mean(axis=0)
    gravity_norm = float(np.linalg.norm(gravity))
    if not np.isfinite(gravity_norm) or gravity_norm <= 1e-10:
        gravity_unit = np.asarray([1.0, 0.0, 0.0], dtype=np.float64)
        gravity_norm = 0.0
    else:
        gravity_unit = gravity / gravity_norm
    residual = acceleration - gravity[None, :]
    projected = residual @ gravity_unit
    acceleration_norm = np.linalg.norm(acceleration, axis=1)
    acc_jerk = np.diff(acceleration, axis=0) * SAMPLE_RATE_HZ
    if not len(acc_jerk):
        acc_jerk = np.zeros((1, 3), dtype=np.float64)

    gate = _log_positive(
        np.asarray(
            [
                np.sqrt(np.mean(np.sum(np.square(residual), axis=1))),
                np.sqrt(np.mean(np.sum(np.square(gyroscope), axis=1))),
                np.std(acceleration_norm),
                np.sqrt(np.mean(np.square(projected))),
                np.sqrt(np.mean(np.sum(np.square(acc_jerk), axis=1))),
            ],
            dtype=np.float64,
        )
    )

    edge_count = max(1, min(len(acceleration) // 2, int(round(0.1 * len(acceleration)))))
    first = acceleration[:edge_count].mean(axis=0)
    last = acceleration[-edge_count:].mean(axis=0)
    denominator = float(np.linalg.norm(first) * np.linalg.norm(last))
    edge_cosine = (
        float(np.clip(np.dot(first, last) / denominator, -1.0, 1.0))
        if denominator > 1e-12
        else -1.0
    )
    posture = np.r_[gravity_unit, gravity_norm, gravity, edge_cosine].astype(np.float64)
    energy = np.r_[
        gate,
        _log_positive(np.sqrt(np.mean(np.square(residual), axis=0))),
        _log_positive(np.sqrt(np.mean(np.square(gyroscope), axis=0))),
    ].astype(np.float64)
    signed = gravity_aligned_trial_state(raw_windows, window_starts).descriptor.astype(
        np.float64
    )
    result = {"gate": gate, "posture": posture, "energy": energy, "signed_gravity": signed}
    expected = {
        "gate": len(GATE_FEATURE_NAMES),
        "posture": len(POSTURE_FEATURE_NAMES),
        "energy": len(ENERGY_FEATURE_NAMES),
        "signed_gravity": 9,
    }
    for name, width in expected.items():
        if result[name].shape != (width,) or not np.all(np.isfinite(result[name])):
            raise RuntimeError(f"Physical feature block {name!r} is malformed.")
    return result


def physical_block_matrices(
    raw_windows: Sequence[np.ndarray],
    window_starts: Sequence[np.ndarray],
) -> dict[str, np.ndarray]:
    if len(raw_windows) != len(window_starts) or not len(raw_windows):
        raise ValueError("Physical block inputs must be non-empty and row aligned.")
    rows = [trial_physical_blocks(raw, starts) for raw, starts in zip(raw_windows, window_starts)]
    names = ("gate", "posture", "energy", "signed_gravity")
    return {name: np.stack([row[name] for row in rows]) for name in names}


@dataclass(frozen=True)
class GateState:
    median: np.ndarray
    scale: np.ndarray
    mixture_weights: np.ndarray
    mixture_means: np.ndarray
    mixture_covariances: np.ndarray
    static_component: int
    n_iter: int
    lower_bound: float

    @property
    def state_sha256(self) -> str:
        return _array_sha256(
            self.median,
            self.scale,
            self.mixture_weights,
            self.mixture_means,
            self.mixture_covariances,
            np.asarray([self.static_component, self.n_iter], dtype="<i8"),
            np.asarray([self.lower_bound], dtype="<f8"),
        )


@dataclass(frozen=True)
class GateResult:
    is_static: np.ndarray
    static_probability: np.ndarray
    standardized_features: np.ndarray
    state: GateState


def fit_static_dynamic_gate(
    gate_features: np.ndarray,
    *,
    seed: int,
    n_init: int = 20,
    reg_covar: float = 1e-6,
) -> GateResult:
    """Fit a two-component unlabeled GMM and name its low-energy component."""

    values = np.asarray(gate_features, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != len(GATE_FEATURE_NAMES) or len(values) < 4:
        raise ValueError("Gate features must be [N>=4,5].")
    if not np.all(np.isfinite(values)):
        raise ValueError("Gate features contain non-finite values.")
    median = np.median(values, axis=0)
    q25, q75 = np.quantile(values, [0.25, 0.75], axis=0)
    scale = np.maximum((q75 - q25) / 1.349, 1e-6)
    standardized = np.clip((values - median) / scale, -12.0, 12.0)
    model = GaussianMixture(
        n_components=2,
        covariance_type="diag",
        n_init=int(n_init),
        reg_covar=float(reg_covar),
        random_state=_derived_seed(seed, 101),
        init_params="kmeans",
        max_iter=500,
    )
    component = model.fit_predict(standardized)
    probabilities = model.predict_proba(standardized)
    # All registered dimensions are log energy/variability measurements, so
    # the lower summed centroid has a physical, label-free static meaning.
    static_component = int(np.argmin(model.means_.sum(axis=1)))
    is_static = component == static_component
    if int(is_static.sum()) < 2 or int((~is_static).sum()) < 2:
        raise RuntimeError("The unlabeled gate collapsed one branch below two trials.")
    state = GateState(
        median=median.astype(np.float64),
        scale=scale.astype(np.float64),
        mixture_weights=np.asarray(model.weights_, dtype=np.float64),
        mixture_means=np.asarray(model.means_, dtype=np.float64),
        mixture_covariances=np.asarray(model.covariances_, dtype=np.float64),
        static_component=static_component,
        n_iter=int(model.n_iter_),
        lower_bound=float(model.lower_bound_),
    )
    return GateResult(
        is_static=is_static.astype(bool),
        static_probability=probabilities[:, static_component].astype(np.float64),
        standardized_features=standardized.astype(np.float64),
        state=state,
    )


@dataclass(frozen=True)
class RobustBlockTransform:
    keep_columns: np.ndarray
    median: np.ndarray
    scale: np.ndarray
    pca_center: np.ndarray | None
    pca_components: np.ndarray | None
    normalization: str

    @property
    def output_dim(self) -> int:
        return int(self.pca_components.shape[0]) if self.pca_components is not None else int(len(self.keep_columns))

    @property
    def state_sha256(self) -> str:
        return _array_sha256(
            self.keep_columns.astype("<i8"),
            self.median.astype("<f8"),
            self.scale.astype("<f8"),
            np.asarray([], dtype="<f8") if self.pca_center is None else self.pca_center,
            np.empty((0, 0), dtype="<f8") if self.pca_components is None else self.pca_components,
            np.frombuffer(self.normalization.encode("utf-8"), dtype=np.uint8),
        )

    def transform(self, values: np.ndarray) -> np.ndarray:
        matrix = np.asarray(values, dtype=np.float64)
        if matrix.ndim != 2 or not len(self.keep_columns):
            raise ValueError("Block-transform input is malformed.")
        if matrix.shape[1] <= int(self.keep_columns.max()):
            raise ValueError("Block-transform input width differs from fit data.")
        transformed = np.clip((matrix[:, self.keep_columns] - self.median) / self.scale, -8.0, 8.0)
        if self.pca_components is not None:
            transformed = (transformed - self.pca_center) @ self.pca_components.T
        norms = np.linalg.norm(transformed, axis=1, keepdims=True)
        if self.normalization == "unit":
            transformed = transformed / np.maximum(norms, 1e-8)
        elif self.normalization == "confidence_preserving":
            # Do not amplify tiny vertical noise to the same length as a real
            # elevator acceleration trajectory.
            transformed = transformed / np.maximum(norms, math.sqrt(transformed.shape[1]))
        else:
            raise RuntimeError(f"Unknown block normalization {self.normalization!r}.")
        if not np.all(np.isfinite(transformed)):
            raise RuntimeError("Static expert block produced non-finite features.")
        return transformed.astype(np.float32)


def fit_robust_block_transform(
    values: np.ndarray,
    *,
    maximum_components: int | None,
    normalization: str,
) -> RobustBlockTransform:
    matrix = np.asarray(values, dtype=np.float64)
    if matrix.ndim != 2 or len(matrix) < 2 or matrix.shape[1] < 1:
        raise ValueError("Static expert block must be [N>=2,D>=1].")
    if normalization not in {"unit", "confidence_preserving"}:
        raise ValueError("Unsupported block normalization.")
    if maximum_components is not None and int(maximum_components) < 1:
        raise ValueError("maximum_components must be positive when provided.")
    keep = np.flatnonzero(np.ptp(matrix, axis=0) > 1e-10)
    if not len(keep):
        raise RuntimeError("Every feature in a static expert block is constant.")
    selected = matrix[:, keep]
    median = np.median(selected, axis=0)
    q25, q75 = np.quantile(selected, [0.25, 0.75], axis=0)
    robust = (q75 - q25) / 1.349
    fallback = np.std(selected, axis=0)
    scale = np.maximum(np.where(robust > 1e-8, robust, fallback), 1e-8)
    standardized = np.clip((selected - median) / scale, -8.0, 8.0)
    pca_center = None
    components = None
    if maximum_components is not None:
        output_dim = min(int(maximum_components), len(matrix) - 1, standardized.shape[1])
        if output_dim < standardized.shape[1]:
            pca_center = standardized.mean(axis=0)
            _, _, right = np.linalg.svd(standardized - pca_center, full_matrices=False)
            components = right[:output_dim].copy()
            for row in components:
                pivot = int(np.argmax(np.abs(row)))
                if row[pivot] < 0:
                    row *= -1.0
    return RobustBlockTransform(
        keep_columns=keep.astype(np.int64),
        median=median.astype(np.float64),
        scale=scale.astype(np.float64),
        pca_center=None if pca_center is None else pca_center.astype(np.float64),
        pca_components=None if components is None else components.astype(np.float64),
        normalization=normalization,
    )


def motion_primitive_static_matrix(
    trajectories: Sequence[PrimitiveTrajectory],
    primitive_num: int,
) -> tuple[np.ndarray, tuple[str, ...]]:
    rows: list[np.ndarray] = []
    names = tuple(
        [f"token_count_fraction__{index}" for index in range(int(primitive_num))]
        + [f"token_duration_fraction__{index}" for index in range(int(primitive_num))]
        + ["quantization_distance_mean", "quantization_distance_std", "quantization_distance_q90"]
    )
    for trajectory in trajectories:
        trajectory.validate(int(primitive_num))
        tokens = trajectory.tokens.astype(np.int64)
        durations = (trajectory.ends - trajectory.starts).astype(np.float64)
        counts = np.bincount(tokens, minlength=int(primitive_num)).astype(np.float64)
        counts /= max(float(len(tokens)), 1.0)
        duration = np.bincount(tokens, weights=durations, minlength=int(primitive_num)).astype(np.float64)
        duration /= max(float(durations.sum()), 1.0)
        distances = np.asarray(trajectory.distances, dtype=np.float64)
        rows.append(np.r_[counts, duration, distances.mean(), distances.std(), np.quantile(distances, 0.90)])
    matrix = np.stack(rows)
    if matrix.shape != (len(trajectories), len(names)) or not np.all(np.isfinite(matrix)):
        raise RuntimeError("Static motion-primitive block is malformed.")
    return matrix, names


@dataclass(frozen=True)
class StaticExpertTransform:
    posture: RobustBlockTransform
    signed_gravity: RobustBlockTransform
    energy: RobustBlockTransform
    motion_primitive: RobustBlockTransform
    weights: np.ndarray

    @property
    def output_dim(self) -> int:
        return sum(getattr(self, name).output_dim for name in STATIC_BLOCK_ORDER)

    @property
    def state_sha256(self) -> str:
        hashes = np.frombuffer(
            "".join(getattr(self, name).state_sha256 for name in STATIC_BLOCK_ORDER).encode("ascii"),
            dtype=np.uint8,
        )
        return _array_sha256(self.weights.astype("<f8"), hashes)

    def transform(self, blocks: Mapping[str, np.ndarray]) -> np.ndarray:
        pieces = []
        for index, name in enumerate(STATIC_BLOCK_ORDER):
            if name not in blocks:
                raise KeyError(f"Static expert lacks block {name!r}.")
            piece = getattr(self, name).transform(blocks[name])
            pieces.append(piece * math.sqrt(float(self.weights[index])))
        combined = np.concatenate(pieces, axis=1)
        if not np.all(np.isfinite(combined)):
            raise RuntimeError("Static expert produced non-finite features.")
        return combined.astype(np.float32)


def fit_static_expert(
    blocks: Mapping[str, np.ndarray],
    *,
    weights: Mapping[str, float] = DEFAULT_STATIC_BLOCK_WEIGHTS,
    motion_primitive_pca_dim: int = 8,
) -> tuple[StaticExpertTransform, np.ndarray]:
    if set(weights) != set(STATIC_BLOCK_ORDER):
        raise ValueError("Static expert weights must name all and only registered blocks.")
    weight_values = np.asarray([float(weights[name]) for name in STATIC_BLOCK_ORDER])
    if np.any(weight_values <= 0.0) or not np.isclose(weight_values.sum(), 1.0):
        raise ValueError("Static expert weights must be positive and sum to one.")
    row_counts = {len(np.asarray(blocks[name])) for name in STATIC_BLOCK_ORDER}
    if len(row_counts) != 1 or next(iter(row_counts)) < 2:
        raise ValueError("Static expert blocks must contain aligned rows.")
    model = StaticExpertTransform(
        posture=fit_robust_block_transform(blocks["posture"], maximum_components=None, normalization="unit"),
        signed_gravity=fit_robust_block_transform(
            blocks["signed_gravity"], maximum_components=None, normalization="confidence_preserving"
        ),
        energy=fit_robust_block_transform(blocks["energy"], maximum_components=None, normalization="unit"),
        motion_primitive=fit_robust_block_transform(
            blocks["motion_primitive"],
            maximum_components=int(motion_primitive_pca_dim),
            normalization="unit",
        ),
        weights=weight_values.astype(np.float64),
    )
    return model, model.transform(blocks)


def fit_duration_invariant_trajectory_features(
    trajectories: Sequence[PrimitiveTrajectory],
    primitive_num: int,
    *,
    maximum_components: int = 32,
) -> tuple[np.ndarray, Any, tuple[str, ...]]:
    raw, names = descriptor_matrix(
        trajectories,
        int(primitive_num),
        include_state=True,
        descriptor_profile=DURATION_INVARIANT_DESCRIPTOR_PROFILE,
    )
    transform = fit_descriptor_transform(raw, maximum_components=int(maximum_components))
    return transform.transform(raw), transform, names


def fit_duration_soft_subject_trajectory_features(
    target_trajectories: Sequence[PrimitiveTrajectory],
    nuisance_source_trajectories: Sequence[PrimitiveTrajectory],
    primitive_num: int,
    *,
    maximum_components: int = 32,
    maximum_rank: int = 4,
    explained_variance: float = 0.90,
    projection_strength: float = 0.25,
) -> tuple[np.ndarray, Any, tuple[str, ...], dict[str, Any]]:
    """Fit duration-free coordinates plus source-only soft subject debiasing.

    The coordinate transform is fitted transductively from the unlabeled target
    trajectories.  Only the nuisance basis uses subject metadata, and that
    metadata comes exclusively from the balanced offline-old6 source trials.
    Target subject identities are neither accepted nor required by this API.
    """

    profile = descriptor_profile_spec(DURATION_SOFT_SUBJECT_A025_PROFILE)
    if not profile.duration_invariant or not profile.subject_debias:
        raise RuntimeError("The registered A025 profile contract changed.")
    if not np.isclose(
        float(projection_strength),
        float(profile.subject_nuisance_projection_strength),
        rtol=0.0,
        atol=1e-12,
    ):
        raise ValueError(
            "The combined experiment locks soft subject projection strength to 0.25."
        )

    target_raw, target_names = descriptor_matrix(
        target_trajectories,
        int(primitive_num),
        include_state=True,
        descriptor_profile=profile.name,
    )
    source_raw, source_names = descriptor_matrix(
        nuisance_source_trajectories,
        int(primitive_num),
        include_state=True,
        descriptor_profile=profile.name,
    )
    if target_names != source_names:
        raise RuntimeError("Target and source duration-invariant descriptor schemas differ.")

    transform = fit_descriptor_transform(
        target_raw, maximum_components=int(maximum_components)
    )
    source_subjects = np.asarray(
        [int(item.subject_id) for item in nuisance_source_trajectories],
        dtype=np.int64,
    )
    unique_subjects, source_counts = np.unique(source_subjects, return_counts=True)
    if len(unique_subjects) < 2 or len(set(source_counts.astype(int).tolist())) != 1:
        raise RuntimeError(
            "Soft subject debiasing requires a balanced offline-old6 source mixture."
        )
    transform, audit = fit_subject_nuisance_projection(
        transform,
        source_raw,
        source_subjects,
        maximum_rank=int(maximum_rank),
        explained_variance=float(explained_variance),
        projection_strength=float(profile.subject_nuisance_projection_strength),
    )
    features = transform.transform(target_raw)
    if not np.all(np.isfinite(features)) or np.any(
        np.linalg.norm(features, axis=1) <= 1e-8
    ):
        raise RuntimeError("Soft subject projection produced invalid trajectory features.")
    return features, transform, target_names, {
        **audit,
        "descriptor_profile": profile.name,
        "coordinate_transform_fit_scope": "gate_selected_outer_dynamic_unlabeled",
        "nuisance_basis_fit_scope": (
            "offline_train_old6_balanced_source_subject_centroids"
        ),
        "outer_subject_ids_used": False,
        "outer_subject_ids_required_at_inference": False,
        "source_subject_count": int(len(unique_subjects)),
        "source_trials_per_subject": int(source_counts[0]),
    }


def _fit_kmeans(
    features: np.ndarray,
    clusters: int,
    *,
    seed: int,
    n_init: int,
    max_iter: int,
) -> tuple[KMeans, np.ndarray]:
    matrix = np.asarray(features, dtype=np.float64)
    if matrix.ndim != 2 or not 1 <= int(clusters) < len(matrix):
        raise ValueError("KMeans cluster count must lie in [1,N-1].")
    model = KMeans(
        n_clusters=int(clusters),
        random_state=int(seed),
        n_init=int(n_init),
        max_iter=int(max_iter),
        algorithm="lloyd",
    ).fit(matrix)
    labels = np.asarray(model.labels_, dtype=np.int64)
    if len(np.unique(labels)) != int(clusters):
        raise RuntimeError("KMeans did not use every requested branch cluster.")
    return model, labels


@dataclass(frozen=True)
class BranchClustering:
    dynamic_k: int
    static_k: int
    dynamic_labels: np.ndarray
    static_labels: np.ndarray
    dynamic_centers: np.ndarray
    static_centers: np.ndarray
    candidate_table: tuple[dict[str, Any], ...]

    @property
    def state_sha256(self) -> str:
        return _array_sha256(
            np.asarray([self.dynamic_k, self.static_k], dtype="<i8"),
            self.dynamic_centers.astype("<f8"),
            self.static_centers.astype("<f8"),
            self.dynamic_labels.astype("<i8"),
            self.static_labels.astype("<i8"),
        )


def fit_count_proportional_branch_clusters(
    dynamic_features: np.ndarray,
    static_features: np.ndarray,
    *,
    total_clusters: int,
    minimum_dynamic_clusters: int,
    seed: int,
    n_init: int = 50,
    max_iter: int = 300,
) -> BranchClustering:
    """Allocate known total K from unlabeled gate counts, then fit both branches.

    USC-HAD contributes the same number of outer trials per activity. Therefore
    ``round(K_total * N_branch/N_total)`` is a label-free class-count estimate in
    this registered batch. This assumption is explicit and must not be carried
    silently to an imbalanced stream.
    """

    dynamic = np.asarray(dynamic_features, dtype=np.float64)
    static = np.asarray(static_features, dtype=np.float64)
    maximum_static = int(total_clusters) - int(minimum_dynamic_clusters)
    if dynamic.ndim != 2 or static.ndim != 2 or len(dynamic) < 2 or len(static) < 2:
        raise ValueError("Both gate branches require two-dimensional non-empty features.")
    raw_static_k = int(np.rint(int(total_clusters) * len(static) / (len(dynamic) + len(static))))
    static_k = int(np.clip(raw_static_k, 1, maximum_static))
    dynamic_k = int(total_clusters) - static_k
    if dynamic_k >= len(dynamic) or static_k >= len(static):
        raise RuntimeError("Count-proportional branch K is infeasible for observed membership.")
    dynamic_model, dynamic_labels = _fit_kmeans(
        dynamic,
        dynamic_k,
        seed=_derived_seed(seed, 1_000 + static_k),
        n_init=n_init,
        max_iter=max_iter,
    )
    static_model, static_labels = _fit_kmeans(
        static,
        static_k,
        seed=_derived_seed(seed, 2_000 + static_k),
        n_init=n_init,
        max_iter=max_iter,
    )
    table = [
        {
            "strategy": "count_proportional_under_balanced_trial_protocol",
            "observed_dynamic_trials": int(len(dynamic)),
            "observed_static_trials": int(len(static)),
            "observed_static_fraction": float(len(static) / (len(dynamic) + len(static))),
            "unclipped_static_k": int(raw_static_k),
            "dynamic_k": int(dynamic_k),
            "static_k": int(static_k),
            "dynamic_inertia": float(dynamic_model.inertia_),
            "static_inertia": float(static_model.inertia_),
        }
    ]
    return BranchClustering(
        dynamic_k=int(dynamic_k),
        static_k=int(static_k),
        dynamic_labels=dynamic_labels,
        static_labels=static_labels,
        dynamic_centers=np.asarray(dynamic_model.cluster_centers_, dtype=np.float64),
        static_centers=np.asarray(static_model.cluster_centers_, dtype=np.float64),
        candidate_table=tuple(table),
    )


def fit_fixed_k_branch(
    features: np.ndarray,
    clusters: int,
    *,
    seed: int,
    n_init: int = 50,
    max_iter: int = 300,
) -> tuple[np.ndarray, np.ndarray, float]:
    model, labels = _fit_kmeans(
        features,
        clusters,
        seed=int(seed),
        n_init=int(n_init),
        max_iter=int(max_iter),
    )
    return labels, np.asarray(model.cluster_centers_, dtype=np.float64), float(model.inertia_)


def merge_branch_predictions(
    is_static: np.ndarray,
    dynamic_labels: np.ndarray,
    static_labels: np.ndarray,
    *,
    dynamic_k: int,
    total_clusters: int,
) -> np.ndarray:
    mask = np.asarray(is_static, dtype=bool)
    dynamic = np.asarray(dynamic_labels, dtype=np.int64)
    static = np.asarray(static_labels, dtype=np.int64)
    if dynamic.shape != (int((~mask).sum()),) or static.shape != (int(mask.sum()),):
        raise ValueError("Branch predictions are not aligned with the gate mask.")
    merged = np.empty(len(mask), dtype=np.int64)
    merged[~mask] = dynamic
    merged[mask] = static + int(dynamic_k)
    if set(np.unique(merged).tolist()) != set(range(int(total_clusters))):
        raise RuntimeError("Merged branch IDs do not form a complete disjoint range.")
    return merged


__all__ = [
    "DEFAULT_STATIC_BLOCK_WEIGHTS",
    "ENERGY_FEATURE_NAMES",
    "GATE_FEATURE_NAMES",
    "POSTURE_FEATURE_NAMES",
    "STATIC_BLOCK_ORDER",
    "BranchClustering",
    "GateResult",
    "GateState",
    "RobustBlockTransform",
    "StaticExpertTransform",
    "fit_duration_invariant_trajectory_features",
    "fit_duration_soft_subject_trajectory_features",
    "fit_fixed_k_branch",
    "fit_robust_block_transform",
    "fit_static_dynamic_gate",
    "fit_static_expert",
    "merge_branch_predictions",
    "motion_primitive_static_matrix",
    "physical_block_matrices",
    "fit_count_proportional_branch_clusters",
    "trial_physical_blocks",
]
