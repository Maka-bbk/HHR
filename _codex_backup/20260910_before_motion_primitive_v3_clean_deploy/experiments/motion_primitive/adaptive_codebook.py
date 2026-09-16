"""Train-only adaptive hierarchical expansion of a frozen K=32 codebook.

This module deliberately implements a *parent-conditioned* vocabulary
expansion instead of appending flat embedding centres that compete with every
old primitive.  The original 32 centres and token ids are immutable.  At the
end of one online session, at most one eligible parent token may receive two
new child tokens.  Low-confidence occurrences always keep their parent token.

The fitting schema contains trial and subject provenance, but no activity
identity.  Subject ids are used only for equal weighting, support checks and a
subject-confound rejection diagnostic; they never enter the descriptor.
Ground truth must be joined only after the returned model and assignments have
been frozen by the caller.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
import hashlib
import json
from typing import Sequence

import numpy as np
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    silhouette_samples,
)


EPS = 1e-12
EXPECTED_BASE_TOKEN_COUNT = 32
CHILDREN_PER_EXPANSION = 2
DESCRIPTOR_RECIPE = (
    "mean_quantization_residual + unit_gravity_direction + "
    "optional_log1p_motion_components; train-only coordinate robust scaling; "
    "each block weighted by inverse_sqrt(block_dimension)"
)


@dataclass(frozen=True)
class AdaptiveOccurrence:
    """One train or inference trial-parent occurrence.

    There must be at most one occurrence for a ``(trial_global_id,
    parent_token)`` pair.  ``descriptor`` is a caller-defined, label-free
    motion descriptor (for example a residual/gravity descriptor) and must use
    the same coordinates for fitting and assignment.
    """

    trial_global_id: int
    subject_id: int
    parent_token: int
    token_fraction: float
    duration_samples: int
    descriptor: np.ndarray
    descriptor_block_sizes: tuple[int, ...] = ()


@dataclass(frozen=True)
class AdaptiveCodebookConfig:
    """Pre-registered split/no-split and capacity controls."""

    base_token_count: int = EXPECTED_BASE_TOKEN_COUNT
    minimum_token_fraction: float = 0.50
    minimum_parent_trials: int = 6
    minimum_parent_subjects: int = 2
    minimum_parent_trials_per_subject: int = 1
    minimum_child_trials: int = 3
    minimum_child_fraction: float = 0.15
    minimum_child_subjects: int = 2
    minimum_silhouette: float = 0.25
    minimum_loo_distortion_reduction: float = 0.15
    minimum_loo_stability: float = 0.80
    minimum_loso_stability: float = 0.80
    maximum_subject_nmi: float = 0.25
    confidence_radius_quantile: float = 0.95
    max_expanded_parents_per_session: int = 1
    max_new_children_per_session: int = CHILDREN_PER_EXPANSION


@dataclass(frozen=True)
class ParentCandidateAudit:
    """Train-only evidence and the deterministic split/no-split decision."""

    parent_token: int
    trial_count: int
    subject_count: int
    subject_trial_counts: tuple[tuple[int, int], ...]
    subject_weight_totals: tuple[tuple[int, float], ...]
    child_trial_counts: tuple[int, int] | None
    required_child_trial_count: int | None
    child_subject_counts: tuple[int, int] | None
    one_medoid_trial_id: int | None
    two_medoid_trial_ids: tuple[int, int] | None
    in_sample_distortion_one: float | None
    in_sample_distortion_two: float | None
    in_sample_distortion_reduction: float | None
    loo_distortion_reduction: float | None
    silhouette_subject_balanced: float | None
    loo_stability_subject_balanced: float | None
    loso_stability_subject_balanced: float | None
    loso_evaluable_holdout_count: int
    loso_stability_by_heldout_subject: tuple[tuple[int, float], ...]
    child_subject_nmi: float | None
    accepted: bool
    rejection_reasons: tuple[str, ...]


@dataclass(frozen=True)
class AdaptiveExpansion:
    """Two append-only children of one frozen parent token."""

    session_index: int
    parent_token: int
    child_token_ids: tuple[int, int]
    descriptor_location: np.ndarray
    descriptor_scale: np.ndarray
    descriptor_block_sizes: tuple[int, ...]
    child_medoids: np.ndarray
    child_medoid_trial_ids: tuple[int, int]
    child_confidence_radii: np.ndarray
    fit_trial_ids: tuple[int, ...]
    fit_subject_ids: tuple[int, ...]
    fit_child_ids: tuple[int, ...]
    candidate_audit: ParentCandidateAudit


@dataclass(frozen=True)
class AdaptiveCodebookModel:
    """Frozen base centres plus zero or one session-local hierarchy."""

    session_index: int
    config: AdaptiveCodebookConfig
    base_centers: np.ndarray
    base_center_hash: str
    train_trial_ids: tuple[int, ...]
    candidate_audits: tuple[ParentCandidateAudit, ...]
    expansions: tuple[AdaptiveExpansion, ...]

    @property
    def active_token_count(self) -> int:
        return int(self.config.base_token_count) + sum(
            len(item.child_token_ids) for item in self.expansions
        )

    @property
    def K_total(self) -> int:
        """Compatibility alias used by experiment runners and manifests."""

        return self.active_token_count

    @property
    def gate_enabled(self) -> bool:
        return bool(self.expansions)

    @property
    def selected_parent_token(self) -> int | None:
        return int(self.expansions[0].parent_token) if self.expansions else None


@dataclass(frozen=True)
class AdaptiveAssignments:
    """Frozen occurrence-level routing decisions."""

    trial_ids: np.ndarray
    parent_tokens: np.ndarray
    output_tokens: np.ndarray
    child_ids: np.ndarray
    assigned_child_distances: np.ndarray
    routed_to_child: np.ndarray


def assert_adaptive_schema_is_label_free() -> None:
    """Fail if activity identity is ever added to a fitting schema."""

    forbidden = ("label", "activity", "class", "name")
    for schema in (
        AdaptiveOccurrence,
        AdaptiveCodebookConfig,
        ParentCandidateAudit,
        AdaptiveExpansion,
        AdaptiveCodebookModel,
        AdaptiveAssignments,
    ):
        names = {item.name.lower() for item in fields(schema)}
        leaked = sorted(
            name
            for name in names
            if any(fragment in name for fragment in forbidden)
        )
        if leaked:
            raise RuntimeError(f"{schema.__name__} contains identity fields: {leaked}")


def codebook_center_hash(centers: np.ndarray) -> str:
    """Return a shape/dtype-aware SHA-256 hash for a centre matrix."""

    matrix = np.asarray(centers)
    if matrix.ndim != 2 or not np.all(np.isfinite(matrix)):
        raise ValueError("Codebook centres must be a finite 2D matrix.")
    contiguous = np.ascontiguousarray(matrix)
    digest = hashlib.sha256()
    digest.update(
        json.dumps(
            {"shape": list(contiguous.shape), "dtype": contiguous.dtype.str},
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    )
    digest.update(contiguous.tobytes(order="C"))
    return digest.hexdigest()


def build_adaptive_occurrence(
    unlabelled_occurrence,
    subject_id: int,
    motion_components: np.ndarray | None = None,
) -> AdaptiveOccurrence:
    """Build the registered residual/gravity/(optional) motion descriptor.

    ``unlabelled_occurrence`` is intentionally duck-typed so the helper can
    consume :class:`online_secondary_codebook.UnlabelledTokenOccurrence`
    without introducing a reverse dependency.  It must expose only the
    label-free fields used below.  Data-dependent scaling is *not* performed
    here; :func:`fit_adaptive_codebook` estimates it from train occurrences
    and :func:`assign_adaptive_codebook` reuses the frozen train scaler.
    """

    required = (
        "trial_global_id",
        "coarse_token",
        "token_fraction",
        "duration_samples",
        "mean_quantization_residual",
        "gravity_direction",
    )
    missing = [field for field in required if not hasattr(unlabelled_occurrence, field)]
    if missing:
        raise TypeError(f"Label-free occurrence lacks fields: {missing}.")
    residual = np.asarray(
        unlabelled_occurrence.mean_quantization_residual, dtype=np.float64
    )
    gravity = np.asarray(unlabelled_occurrence.gravity_direction, dtype=np.float64)
    if residual.ndim != 1 or len(residual) == 0 or not np.all(np.isfinite(residual)):
        raise ValueError("Quantization residual must be a non-empty finite vector.")
    if gravity.shape != (3,) or not np.all(np.isfinite(gravity)):
        raise ValueError("Gravity direction must be a finite three-vector.")
    gravity_norm = float(np.linalg.norm(gravity))
    if gravity_norm <= EPS:
        raise ValueError("Gravity direction has zero norm.")
    blocks = [residual, gravity / gravity_norm]
    block_sizes = [len(residual), 3]
    if motion_components is not None:
        motion = np.asarray(motion_components, dtype=np.float64)
        if motion.shape != (2,) or not np.all(np.isfinite(motion)) or np.any(motion < 0):
            raise ValueError("Motion components must be a finite non-negative two-vector.")
        blocks.append(np.log1p(motion))
        block_sizes.append(2)
    descriptor = np.concatenate(blocks).astype(np.float32)
    return AdaptiveOccurrence(
        trial_global_id=int(unlabelled_occurrence.trial_global_id),
        subject_id=int(subject_id),
        parent_token=int(unlabelled_occurrence.coarse_token),
        token_fraction=float(unlabelled_occurrence.token_fraction),
        duration_samples=int(unlabelled_occurrence.duration_samples),
        descriptor=_frozen_copy(descriptor),
        descriptor_block_sizes=tuple(int(value) for value in block_sizes),
    )


def _frozen_copy(value: np.ndarray, dtype=None) -> np.ndarray:
    result = np.asarray(value, dtype=dtype).copy()
    result.setflags(write=False)
    return result


def _validate_config(config: AdaptiveCodebookConfig) -> None:
    if int(config.base_token_count) != EXPECTED_BASE_TOKEN_COUNT:
        raise ValueError(
            f"The registered first experiment requires K={EXPECTED_BASE_TOKEN_COUNT}."
        )
    if not 0.0 < float(config.minimum_token_fraction) <= 1.0:
        raise ValueError("minimum_token_fraction must lie in (0,1].")
    integer_fields = {
        "minimum_parent_trials": config.minimum_parent_trials,
        "minimum_parent_subjects": config.minimum_parent_subjects,
        "minimum_parent_trials_per_subject": config.minimum_parent_trials_per_subject,
        "minimum_child_trials": config.minimum_child_trials,
        "minimum_child_subjects": config.minimum_child_subjects,
        "max_expanded_parents_per_session": config.max_expanded_parents_per_session,
        "max_new_children_per_session": config.max_new_children_per_session,
    }
    for name, value in integer_fields.items():
        if int(value) != value or int(value) < 1:
            raise ValueError(f"{name} must be a positive integer.")
    if int(config.max_expanded_parents_per_session) != 1:
        raise ValueError("The first experiment permits at most one expanded parent.")
    if int(config.max_new_children_per_session) != CHILDREN_PER_EXPANSION:
        raise ValueError("The first experiment permits exactly a two-child budget.")
    for name, value in (
        ("minimum_silhouette", config.minimum_silhouette),
        ("minimum_loo_distortion_reduction", config.minimum_loo_distortion_reduction),
        ("minimum_loo_stability", config.minimum_loo_stability),
        ("minimum_loso_stability", config.minimum_loso_stability),
        ("maximum_subject_nmi", config.maximum_subject_nmi),
    ):
        if not np.isfinite(value) or not 0.0 <= float(value) <= 1.0:
            raise ValueError(f"{name} must lie in [0,1].")
    if not 0.0 < float(config.minimum_child_fraction) <= 0.5:
        raise ValueError("minimum_child_fraction must lie in (0,0.5].")
    if not 0.0 < float(config.confidence_radius_quantile) <= 1.0:
        raise ValueError("confidence_radius_quantile must lie in (0,1].")


def _validate_occurrences(
    occurrences: Sequence[AdaptiveOccurrence],
    base_token_count: int,
    *,
    allow_empty: bool = False,
) -> tuple[AdaptiveOccurrence, ...]:
    assert_adaptive_schema_is_label_free()
    if not occurrences and not allow_empty:
        raise ValueError("At least one occurrence is required.")
    ordered = tuple(
        sorted(
            occurrences,
            key=lambda item: (
                int(item.parent_token),
                int(item.trial_global_id),
                int(item.subject_id),
            ),
        )
    )
    keys: set[tuple[int, int]] = set()
    descriptor_dim = None
    block_sizes = None
    for item in ordered:
        key = (int(item.trial_global_id), int(item.parent_token))
        if key in keys:
            raise ValueError("There may be at most one occurrence per trial-parent pair.")
        keys.add(key)
        if int(item.trial_global_id) < 0 or int(item.subject_id) < 0:
            raise ValueError("Trial and subject ids must be non-negative.")
        if not 0 <= int(item.parent_token) < int(base_token_count):
            raise ValueError("An occurrence parent token lies outside the frozen base codebook.")
        if not 0.0 < float(item.token_fraction) <= 1.0:
            raise ValueError("token_fraction must lie in (0,1].")
        if int(item.duration_samples) <= 0:
            raise ValueError("duration_samples must be positive.")
        descriptor = np.asarray(item.descriptor, dtype=np.float64)
        if descriptor.ndim != 1 or len(descriptor) == 0 or not np.all(np.isfinite(descriptor)):
            raise ValueError("Every occurrence descriptor must be a non-empty finite vector.")
        descriptor_dim = len(descriptor) if descriptor_dim is None else descriptor_dim
        if len(descriptor) != descriptor_dim:
            raise ValueError("All occurrence descriptors must share a dimension.")
        observed_blocks = tuple(int(value) for value in item.descriptor_block_sizes)
        if not observed_blocks:
            observed_blocks = (len(descriptor),)
        if any(value < 1 for value in observed_blocks) or sum(observed_blocks) != len(
            descriptor
        ):
            raise ValueError("Descriptor block sizes must be positive and cover the descriptor.")
        block_sizes = observed_blocks if block_sizes is None else block_sizes
        if observed_blocks != block_sizes:
            raise ValueError("All occurrence descriptors must share one block layout.")
    return ordered


def _subject_balanced_weights(subjects: np.ndarray) -> np.ndarray:
    values = np.asarray(subjects, dtype=np.int64)
    if values.ndim != 1 or len(values) == 0:
        raise ValueError("Subjects must be a non-empty 1D array.")
    unique, counts = np.unique(values, return_counts=True)
    weights = np.zeros(len(values), dtype=np.float64)
    for subject, count in zip(unique.tolist(), counts.tolist()):
        weights[values == int(subject)] = 1.0 / (len(unique) * int(count))
    if not np.isclose(weights.sum(), 1.0, atol=1e-12):
        raise RuntimeError("Subject-balanced weights do not sum to one.")
    return weights


def _weighted_quantile(values: np.ndarray, weights: np.ndarray, quantile: float) -> float:
    samples = np.asarray(values, dtype=np.float64)
    sample_weights = np.asarray(weights, dtype=np.float64)
    if samples.ndim != 1 or sample_weights.shape != samples.shape or len(samples) == 0:
        raise ValueError("Weighted quantile inputs must be equal non-empty vectors.")
    order = np.argsort(samples, kind="stable")
    sorted_values = samples[order]
    sorted_weights = sample_weights[order]
    total = float(sorted_weights.sum())
    if total <= 0 or np.any(sorted_weights < 0):
        raise ValueError("Weighted quantile needs non-negative positive-total weights.")
    threshold = float(quantile) * total
    position = int(np.searchsorted(np.cumsum(sorted_weights), threshold, side="left"))
    return float(sorted_values[min(position, len(sorted_values) - 1)])


def _robust_scale(descriptors: np.ndarray, weights: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    matrix = np.asarray(descriptors, dtype=np.float64)
    location = np.asarray(
        [_weighted_quantile(matrix[:, col], weights, 0.50) for col in range(matrix.shape[1])],
        dtype=np.float64,
    )
    scale = np.asarray(
        [
            _weighted_quantile(np.abs(matrix[:, col] - location[col]), weights, 0.95)
            for col in range(matrix.shape[1])
        ],
        dtype=np.float64,
    )
    positive = scale[scale > EPS]
    fallback = float(np.median(positive)) if len(positive) else 1.0
    # A nearly constant coordinate must not be magnified into the dominant
    # split direction.  The floor is relative to the median non-zero robust
    # scale and is itself derived only from the fitting occurrences.
    scale_floor = max(0.05 * fallback, 1e-6)
    scale[scale < scale_floor] = scale_floor
    return location, scale


def _balance_descriptor_blocks(
    standardized: np.ndarray, block_sizes: tuple[int, ...]
) -> np.ndarray:
    result = np.asarray(standardized, dtype=np.float64).copy()
    offset = 0
    for size in block_sizes:
        end = offset + int(size)
        result[..., offset:end] /= np.sqrt(float(size))
        offset = end
    if offset != result.shape[-1]:
        raise RuntimeError("Descriptor block layout does not cover its final dimension.")
    return result


def _fit_transform_fold_descriptors(
    train_descriptors: np.ndarray,
    held_descriptors: np.ndarray,
    train_subjects: np.ndarray,
    block_sizes: tuple[int, ...],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Fit preprocessing on a fold complement and transform both partitions.

    ``held_descriptors`` is never consulted while estimating ``location`` or
    ``scale``.  Keeping this boundary explicit prevents a leave-one-out fold
    from inheriting optimistic full-candidate preprocessing.
    """

    train = np.asarray(train_descriptors, dtype=np.float64)
    held = np.asarray(held_descriptors, dtype=np.float64)
    subjects = np.asarray(train_subjects, dtype=np.int64)
    if train.ndim != 2 or len(train) == 0:
        raise ValueError("A fold complement must be a non-empty 2D matrix.")
    if held.ndim != 2 or held.shape[1] != train.shape[1]:
        raise ValueError("Held descriptors must share the complement dimension.")
    if subjects.shape != (len(train),):
        raise ValueError("Every fold-complement descriptor needs one subject id.")
    if sum(int(value) for value in block_sizes) != train.shape[1]:
        raise ValueError("Fold descriptor blocks do not cover the descriptor.")
    weights = _subject_balanced_weights(subjects)
    location, scale = _robust_scale(train, weights)
    transformed_train = _balance_descriptor_blocks(
        (train - location[None, :]) / scale[None, :], block_sizes
    )
    transformed_held = _balance_descriptor_blocks(
        (held - location[None, :]) / scale[None, :], block_sizes
    )
    return transformed_train, transformed_held, location, scale


def _euclidean_distance_matrix(values: np.ndarray) -> np.ndarray:
    matrix = np.asarray(values, dtype=np.float64)
    squared = (
        np.sum(matrix * matrix, axis=1, keepdims=True)
        + np.sum(matrix * matrix, axis=1)[None, :]
        - 2.0 * matrix @ matrix.T
    )
    result = np.sqrt(np.maximum(squared, 0.0))
    np.fill_diagonal(result, 0.0)
    return result


def _cross_euclidean_distance_matrix(
    left: np.ndarray, right: np.ndarray
) -> np.ndarray:
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    if left.ndim != 2 or right.ndim != 2 or left.shape[1] != right.shape[1]:
        raise ValueError("Cross-distance inputs must be compatible 2D matrices.")
    squared = (
        np.sum(left * left, axis=1, keepdims=True)
        + np.sum(right * right, axis=1)[None, :]
        - 2.0 * left @ right.T
    )
    return np.sqrt(np.maximum(squared, 0.0))


def _fit_one_medoid(
    distance: np.ndarray, weights: np.ndarray, trial_ids: np.ndarray
) -> tuple[int, float]:
    costs = np.asarray(weights, dtype=np.float64) @ np.asarray(distance, dtype=np.float64)
    minimum = float(np.min(costs))
    candidates = np.flatnonzero(np.isclose(costs, minimum, rtol=0.0, atol=1e-12))
    selected = min(candidates.tolist(), key=lambda index: int(trial_ids[int(index)]))
    return int(selected), float(costs[int(selected)])


def _fit_two_medoids(
    distance: np.ndarray, weights: np.ndarray, trial_ids: np.ndarray
) -> tuple[np.ndarray, np.ndarray, float]:
    matrix = np.asarray(distance, dtype=np.float64)
    sample_weights = np.asarray(weights, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape != (len(matrix), len(matrix)) or len(matrix) < 2:
        raise ValueError("Two-medoids fitting needs a square matrix with at least two rows.")
    best = None
    for left in range(len(matrix) - 1):
        for right in range(left + 1, len(matrix)):
            if float(matrix[left, right]) <= EPS:
                continue
            pair = np.asarray([left, right], dtype=np.int64)
            labels = np.argmin(matrix[:, pair], axis=1).astype(np.int64)
            labels[left] = 0
            labels[right] = 1
            if len(np.unique(labels)) != 2:
                continue
            objective = float(
                np.sum(sample_weights * matrix[np.arange(len(matrix)), pair[labels]])
            )
            key = (
                objective,
                int(trial_ids[left]),
                int(trial_ids[right]),
            )
            if best is None or key < best[0]:
                best = (key, pair, labels, objective)
    if best is None:
        raise RuntimeError("Distinct two-child medoids could not be formed.")
    return best[1], best[2], best[3]


def _loo_metrics(
    raw_descriptors: np.ndarray,
    block_sizes: tuple[int, ...],
    subjects: np.ndarray,
    trial_ids: np.ndarray,
    full_labels: np.ndarray,
) -> tuple[float, float]:
    """Return subject-balanced held-out distortion gain and ARI stability."""

    descriptors = np.asarray(raw_descriptors, dtype=np.float64)
    subjects = np.asarray(subjects, dtype=np.int64)
    trial_ids = np.asarray(trial_ids, dtype=np.int64)
    evaluation_weights = _subject_balanced_weights(subjects)
    one_errors = np.zeros(len(descriptors), dtype=np.float64)
    two_errors = np.zeros(len(descriptors), dtype=np.float64)
    stability = np.zeros(len(descriptors), dtype=np.float64)
    all_positions = np.arange(len(descriptors), dtype=np.int64)
    for held_out in range(len(descriptors)):
        kept = all_positions[all_positions != held_out]
        if len(kept) < 2:
            continue
        transformed_train, transformed_held, _, _ = _fit_transform_fold_descriptors(
            descriptors[kept],
            descriptors[[held_out]],
            subjects[kept],
            block_sizes,
        )
        train_weights = _subject_balanced_weights(subjects[kept])
        submatrix = _euclidean_distance_matrix(transformed_train)
        one_local, _ = _fit_one_medoid(submatrix, train_weights, trial_ids[kept])
        held_to_train = _cross_euclidean_distance_matrix(
            transformed_held, transformed_train
        )[0]
        one_errors[held_out] = float(held_to_train[one_local])
        try:
            two_local, _, _ = _fit_two_medoids(
                submatrix, train_weights, trial_ids[kept]
            )
        except RuntimeError:
            two_errors[held_out] = one_errors[held_out]
            stability[held_out] = 0.0
            continue
        two_errors[held_out] = float(np.min(held_to_train[two_local]))
        refit_labels = np.argmin(submatrix[:, two_local], axis=1).astype(
            np.int64
        )
        stability[held_out] = float(
            adjusted_rand_score(full_labels[kept], refit_labels)
        )
    one_total = float(np.sum(evaluation_weights * one_errors))
    two_total = float(np.sum(evaluation_weights * two_errors))
    reduction = float((one_total - two_total) / max(one_total, EPS))
    stability_score = float(np.sum(evaluation_weights * stability))
    return reduction, stability_score


def _leave_one_subject_out_stability(
    raw_descriptors: np.ndarray,
    block_sizes: tuple[int, ...],
    subjects: np.ndarray,
    trial_ids: np.ndarray,
    full_labels: np.ndarray,
) -> tuple[float | None, tuple[tuple[int, float], ...]]:
    """Test whether a split learned without one subject transfers to it.

    A holdout is evaluable only when both its fitting complement and the held
    subject contain both full-fit modes.  ARI is permutation invariant, so
    child ids need not be manually aligned.
    """

    descriptors = np.asarray(raw_descriptors, dtype=np.float64)
    subjects = np.asarray(subjects, dtype=np.int64)
    trial_ids = np.asarray(trial_ids, dtype=np.int64)
    full_labels = np.asarray(full_labels, dtype=np.int64)
    scores: list[tuple[int, float]] = []
    all_positions = np.arange(len(descriptors), dtype=np.int64)
    for held_subject in sorted(np.unique(subjects).tolist()):
        train = all_positions[subjects != int(held_subject)]
        held = all_positions[subjects == int(held_subject)]
        train_counts = np.bincount(full_labels[train], minlength=2)
        held_counts = np.bincount(full_labels[held], minlength=2)
        if (
            len(train) < 4
            or len(held) < 4
            or np.any(train_counts < 2)
            or np.any(held_counts < 2)
        ):
            continue
        transformed_train, transformed_held, _, _ = _fit_transform_fold_descriptors(
            descriptors[train],
            descriptors[held],
            subjects[train],
            block_sizes,
        )
        train_weights = _subject_balanced_weights(subjects[train])
        train_distance = _euclidean_distance_matrix(transformed_train)
        try:
            medoids_local, _, _ = _fit_two_medoids(
                train_distance,
                train_weights,
                trial_ids[train],
            )
        except RuntimeError:
            continue
        held_to_medoids = _cross_euclidean_distance_matrix(
            transformed_held, transformed_train[medoids_local]
        )
        held_predictions = np.argmin(
            held_to_medoids, axis=1
        ).astype(np.int64)
        score = float(adjusted_rand_score(full_labels[held], held_predictions))
        scores.append((int(held_subject), score))
    if not scores:
        return None, ()
    return float(np.mean([score for _, score in scores])), tuple(scores)


def _empty_candidate_audit(
    parent_token: int,
    trial_count: int,
    subjects: np.ndarray,
    reasons: Sequence[str],
) -> ParentCandidateAudit:
    subject_ids, counts = np.unique(subjects, return_counts=True)
    weights = _subject_balanced_weights(subjects) if len(subjects) else np.asarray([])
    weight_totals = tuple(
        (int(subject), float(weights[subjects == int(subject)].sum()))
        for subject in subject_ids.tolist()
    )
    return ParentCandidateAudit(
        parent_token=int(parent_token),
        trial_count=int(trial_count),
        subject_count=int(len(subject_ids)),
        subject_trial_counts=tuple(
            (int(subject), int(count))
            for subject, count in zip(subject_ids.tolist(), counts.tolist())
        ),
        subject_weight_totals=weight_totals,
        child_trial_counts=None,
        required_child_trial_count=None,
        child_subject_counts=None,
        one_medoid_trial_id=None,
        two_medoid_trial_ids=None,
        in_sample_distortion_one=None,
        in_sample_distortion_two=None,
        in_sample_distortion_reduction=None,
        loo_distortion_reduction=None,
        silhouette_subject_balanced=None,
        loo_stability_subject_balanced=None,
        loso_stability_subject_balanced=None,
        loso_evaluable_holdout_count=0,
        loso_stability_by_heldout_subject=(),
        child_subject_nmi=None,
        accepted=False,
        rejection_reasons=tuple(reasons),
    )


def _evaluate_parent_candidate(
    parent_token: int,
    occurrences: Sequence[AdaptiveOccurrence],
    config: AdaptiveCodebookConfig,
) -> tuple[ParentCandidateAudit, dict | None]:
    eligible = tuple(
        item
        for item in occurrences
        if float(item.token_fraction) + EPS >= float(config.minimum_token_fraction)
    )
    subjects = np.asarray([int(item.subject_id) for item in eligible], dtype=np.int64)
    reasons = []
    if len(eligible) < int(config.minimum_parent_trials):
        reasons.append("parent_trial_support_below_minimum")
    unique_subjects, subject_counts = np.unique(subjects, return_counts=True)
    if len(unique_subjects) < int(config.minimum_parent_subjects):
        reasons.append("parent_subject_support_below_minimum")
    if len(subject_counts) and int(subject_counts.min()) < int(
        config.minimum_parent_trials_per_subject
    ):
        reasons.append("parent_per_subject_trial_support_below_minimum")
    if reasons:
        return (
            _empty_candidate_audit(
                parent_token, len(eligible), subjects, reasons
            ),
            None,
        )

    trial_ids = np.asarray(
        [int(item.trial_global_id) for item in eligible], dtype=np.int64
    )
    descriptors = np.asarray([item.descriptor for item in eligible], dtype=np.float64)
    block_sizes = tuple(int(value) for value in eligible[0].descriptor_block_sizes)
    if not block_sizes:
        block_sizes = (descriptors.shape[1],)
    weights = _subject_balanced_weights(subjects)
    location, scale = _robust_scale(descriptors, weights)
    standardized = _balance_descriptor_blocks(
        (descriptors - location[None, :]) / scale[None, :], block_sizes
    )
    distance = _euclidean_distance_matrix(standardized)
    one_medoid, distortion_one = _fit_one_medoid(distance, weights, trial_ids)
    try:
        two_medoids, labels, distortion_two = _fit_two_medoids(
            distance, weights, trial_ids
        )
    except RuntimeError as error:
        audit = _empty_candidate_audit(
            parent_token,
            len(eligible),
            subjects,
            [f"two_child_fit_unavailable:{error}"],
        )
        return audit, None

    child_counts = np.bincount(labels, minlength=2)
    required_child_trials = max(
        int(config.minimum_child_trials),
        int(np.ceil(float(config.minimum_child_fraction) * len(eligible))),
    )
    child_subject_counts = np.asarray(
        [len(np.unique(subjects[labels == child])) for child in (0, 1)],
        dtype=np.int64,
    )
    if np.any(child_counts < required_child_trials):
        reasons.append("child_trial_support_below_minimum")
    if np.any(child_subject_counts < int(config.minimum_child_subjects)):
        reasons.append("child_subject_support_below_minimum")

    silhouette = None
    if np.all(child_counts >= 2):
        sample_scores = silhouette_samples(distance, labels, metric="precomputed")
        silhouette = float(np.sum(weights * sample_scores))
        if silhouette + EPS < float(config.minimum_silhouette):
            reasons.append("silhouette_below_minimum")
    else:
        reasons.append("silhouette_unavailable")

    in_sample_reduction = float(
        (distortion_one - distortion_two) / max(distortion_one, EPS)
    )
    loo_reduction, loo_stability = _loo_metrics(
        descriptors, block_sizes, subjects, trial_ids, labels
    )
    loso_stability, loso_scores = _leave_one_subject_out_stability(
        descriptors, block_sizes, subjects, trial_ids, labels
    )
    if loo_reduction + EPS < float(config.minimum_loo_distortion_reduction):
        reasons.append("loo_distortion_reduction_below_minimum")
    if loo_stability + EPS < float(config.minimum_loo_stability):
        reasons.append("loo_stability_below_minimum")
    if len(loso_scores) < 2:
        reasons.append("fewer_than_two_evaluable_loso_holdouts")
    elif loso_stability + EPS < float(config.minimum_loso_stability):
        reasons.append("loso_stability_below_minimum")

    subject_nmi = float(normalized_mutual_info_score(subjects, labels))
    if subject_nmi > float(config.maximum_subject_nmi) + EPS:
        reasons.append("child_partition_is_subject_confounded")

    audit = ParentCandidateAudit(
        parent_token=int(parent_token),
        trial_count=int(len(eligible)),
        subject_count=int(len(unique_subjects)),
        subject_trial_counts=tuple(
            (int(subject), int(count))
            for subject, count in zip(unique_subjects.tolist(), subject_counts.tolist())
        ),
        subject_weight_totals=tuple(
            (int(subject), float(weights[subjects == int(subject)].sum()))
            for subject in unique_subjects.tolist()
        ),
        child_trial_counts=tuple(int(value) for value in child_counts.tolist()),
        required_child_trial_count=int(required_child_trials),
        child_subject_counts=tuple(
            int(value) for value in child_subject_counts.tolist()
        ),
        one_medoid_trial_id=int(trial_ids[one_medoid]),
        two_medoid_trial_ids=tuple(
            int(trial_ids[index]) for index in two_medoids.tolist()
        ),
        in_sample_distortion_one=float(distortion_one),
        in_sample_distortion_two=float(distortion_two),
        in_sample_distortion_reduction=in_sample_reduction,
        loo_distortion_reduction=float(loo_reduction),
        silhouette_subject_balanced=silhouette,
        loo_stability_subject_balanced=float(loo_stability),
        loso_stability_subject_balanced=(
            float(loso_stability) if loso_stability is not None else None
        ),
        loso_evaluable_holdout_count=len(loso_scores),
        loso_stability_by_heldout_subject=tuple(
            (int(subject), float(score)) for subject, score in loso_scores
        ),
        child_subject_nmi=subject_nmi,
        accepted=not reasons,
        rejection_reasons=tuple(reasons),
    )
    if reasons:
        return audit, None

    assigned_distances = distance[np.arange(len(distance)), two_medoids[labels]]
    radii = np.zeros(2, dtype=np.float64)
    for child in (0, 1):
        selected = labels == child
        child_weights = _subject_balanced_weights(subjects[selected])
        radii[child] = max(
            _weighted_quantile(
                assigned_distances[selected],
                child_weights,
                float(config.confidence_radius_quantile),
            ),
            1e-6,
        )
    payload = {
        "eligible": eligible,
        "trial_ids": trial_ids,
        "subjects": subjects,
        "location": location,
        "scale": scale,
        "block_sizes": block_sizes,
        "standardized": standardized,
        "two_medoids": two_medoids,
        "labels": labels,
        "radii": radii,
    }
    return audit, payload


def _candidate_rank(audit: ParentCandidateAudit) -> tuple[float, float, float, int, int]:
    """Higher evidence wins; the final tie-break is the smaller parent id."""

    return (
        -float(audit.loo_distortion_reduction),
        -float(audit.silhouette_subject_balanced),
        -float(audit.loo_stability_subject_balanced),
        -int(audit.trial_count),
        int(audit.parent_token),
    )


def fit_adaptive_codebook(
    base_centers: np.ndarray,
    train_occurrences: Sequence[AdaptiveOccurrence],
    *,
    session_index: int,
    config: AdaptiveCodebookConfig | None = None,
) -> AdaptiveCodebookModel:
    """Fit a deterministic K=32-or-34 model using train occurrences only."""

    assert_adaptive_schema_is_label_free()
    resolved = config or AdaptiveCodebookConfig()
    _validate_config(resolved)
    centers = np.asarray(base_centers)
    if centers.ndim != 2 or centers.shape[0] != int(resolved.base_token_count):
        raise ValueError(
            f"Expected frozen base centres with shape [32,D], got {centers.shape}."
        )
    if not np.all(np.isfinite(centers)):
        raise ValueError("Base centres must be finite.")
    if int(session_index) < 1:
        raise ValueError("session_index must be positive.")
    ordered = _validate_occurrences(
        train_occurrences, int(resolved.base_token_count)
    )
    frozen_centers = _frozen_copy(centers)
    center_hash = codebook_center_hash(frozen_centers)

    grouped: dict[int, list[AdaptiveOccurrence]] = {}
    for item in ordered:
        grouped.setdefault(int(item.parent_token), []).append(item)
    audits: list[ParentCandidateAudit] = []
    payload_by_parent: dict[int, dict] = {}
    for parent_token in sorted(grouped):
        audit, payload = _evaluate_parent_candidate(
            parent_token, grouped[parent_token], resolved
        )
        audits.append(audit)
        if audit.accepted:
            if payload is None:
                raise RuntimeError("Accepted parent candidate has no fitted payload.")
            payload_by_parent[int(parent_token)] = payload

    accepted = sorted(
        (item for item in audits if item.accepted), key=_candidate_rank
    )
    selected = accepted[: int(resolved.max_expanded_parents_per_session)]
    expansions: list[AdaptiveExpansion] = []
    next_child_token = int(resolved.base_token_count)
    for audit in selected:
        if next_child_token + CHILDREN_PER_EXPANSION > int(
            resolved.base_token_count + resolved.max_new_children_per_session
        ):
            break
        payload = payload_by_parent[int(audit.parent_token)]
        medoid_positions = np.asarray(payload["two_medoids"], dtype=np.int64)
        child_ids = tuple(
            range(next_child_token, next_child_token + CHILDREN_PER_EXPANSION)
        )
        expansions.append(
            AdaptiveExpansion(
                session_index=int(session_index),
                parent_token=int(audit.parent_token),
                child_token_ids=tuple(int(value) for value in child_ids),
                descriptor_location=_frozen_copy(payload["location"], np.float64),
                descriptor_scale=_frozen_copy(payload["scale"], np.float64),
                descriptor_block_sizes=tuple(
                    int(value) for value in payload["block_sizes"]
                ),
                child_medoids=_frozen_copy(
                    payload["standardized"][medoid_positions], np.float64
                ),
                child_medoid_trial_ids=tuple(
                    int(payload["trial_ids"][position])
                    for position in medoid_positions.tolist()
                ),
                child_confidence_radii=_frozen_copy(payload["radii"], np.float64),
                fit_trial_ids=tuple(
                    int(value) for value in payload["trial_ids"].tolist()
                ),
                fit_subject_ids=tuple(
                    int(value) for value in payload["subjects"].tolist()
                ),
                fit_child_ids=tuple(
                    int(value) for value in payload["labels"].tolist()
                ),
                candidate_audit=audit,
            )
        )
        next_child_token += CHILDREN_PER_EXPANSION

    model = AdaptiveCodebookModel(
        session_index=int(session_index),
        config=resolved,
        base_centers=frozen_centers,
        base_center_hash=center_hash,
        train_trial_ids=tuple(
            sorted({int(item.trial_global_id) for item in ordered})
        ),
        candidate_audits=tuple(audits),
        expansions=tuple(expansions),
    )
    audit = audit_codebook_invariants(model, current_base_centers=centers)
    if not audit["all_invariants_passed"]:
        raise RuntimeError(f"Adaptive codebook invariant failure: {audit}")
    return model


def assign_adaptive_codebook(
    model: AdaptiveCodebookModel,
    occurrences: Sequence[AdaptiveOccurrence],
) -> AdaptiveAssignments:
    """Route high-confidence expanded-parent occurrences, else keep parent."""

    ordered = _validate_occurrences(
        occurrences, int(model.config.base_token_count), allow_empty=True
    )
    trial_ids = np.asarray(
        [int(item.trial_global_id) for item in ordered], dtype=np.int64
    )
    parents = np.asarray([int(item.parent_token) for item in ordered], dtype=np.int64)
    output = parents.copy()
    child_ids = np.full(len(ordered), -1, dtype=np.int64)
    assigned_distances = np.full(len(ordered), np.inf, dtype=np.float64)
    routed = np.zeros(len(ordered), dtype=bool)
    expansion_by_parent = {
        int(item.parent_token): item for item in model.expansions
    }
    for position, item in enumerate(ordered):
        expansion = expansion_by_parent.get(int(item.parent_token))
        if expansion is None:
            continue
        if float(item.token_fraction) + EPS < float(
            model.config.minimum_token_fraction
        ):
            continue
        descriptor = np.asarray(item.descriptor, dtype=np.float64)
        if descriptor.shape != expansion.descriptor_location.shape:
            raise ValueError("Inference descriptor dimension differs from its child codebook.")
        standardized = (
            descriptor - expansion.descriptor_location
        ) / expansion.descriptor_scale
        standardized = _balance_descriptor_blocks(
            standardized, expansion.descriptor_block_sizes
        )
        distances = np.linalg.norm(
            expansion.child_medoids - standardized[None, :], axis=1
        )
        child = int(np.argmin(distances))
        distance = float(distances[child])
        assigned_distances[position] = distance
        if distance <= float(expansion.child_confidence_radii[child]) + EPS:
            child_ids[position] = child
            output[position] = int(expansion.child_token_ids[child])
            routed[position] = True
    return AdaptiveAssignments(
        trial_ids=_frozen_copy(trial_ids),
        parent_tokens=_frozen_copy(parents),
        output_tokens=_frozen_copy(output),
        child_ids=_frozen_copy(child_ids),
        assigned_child_distances=_frozen_copy(assigned_distances),
        routed_to_child=_frozen_copy(routed),
    )


def adaptive_state_by_trial(
    model: AdaptiveCodebookModel,
    assignments: AdaptiveAssignments,
) -> dict[int, int]:
    """Return ``0=fallback, 1/2=child`` for the selected-parent occurrences.

    The first registered experiment can expand only one parent.  Trials that
    do not contain that parent are intentionally absent from the mapping and
    do not need a state in :func:`adaptive_duration_histograms`.
    """

    if not model.gate_enabled:
        return {}
    parent = int(model.selected_parent_token)
    selected = np.flatnonzero(assignments.parent_tokens == parent)
    result: dict[int, int] = {}
    for position in selected.tolist():
        trial_id = int(assignments.trial_ids[position])
        if trial_id in result:
            raise ValueError("Selected-parent assignments contain a duplicate trial.")
        child = int(assignments.child_ids[position])
        result[trial_id] = 0 if child < 0 else child + 1
    return result


def adaptive_duration_histograms(
    trial_ids: Sequence[int],
    trial_token_durations: dict[int, dict[int, float]],
    model: AdaptiveCodebookModel,
    gate_state_by_trial: dict[int, int] | None = None,
) -> np.ndarray:
    """Build K=32-or-34 duration histograms with exact parent fallback."""

    dimension = int(model.active_token_count)
    matrix = np.zeros((len(trial_ids), dimension), dtype=np.float64)
    parent = model.selected_parent_token
    states = gate_state_by_trial or {}
    valid_states = {0, 1, 2}
    child_tokens = (
        model.expansions[0].child_token_ids if model.gate_enabled else ()
    )
    for row, trial_id_value in enumerate(trial_ids):
        trial_id = int(trial_id_value)
        if trial_id not in trial_token_durations:
            raise ValueError(f"Trial {trial_id} lacks token-duration data.")
        token_durations = trial_token_durations[trial_id]
        total = float(sum(float(value) for value in token_durations.values()))
        if not np.isfinite(total) or total <= 0:
            raise ValueError(f"Trial {trial_id} has no positive finite duration.")
        for token_value, duration_value in token_durations.items():
            token = int(token_value)
            duration = float(duration_value)
            if not 0 <= token < int(model.config.base_token_count):
                raise ValueError(f"Token {token} lies outside the frozen base K=32.")
            if not np.isfinite(duration) or duration <= 0:
                raise ValueError(f"Trial {trial_id} has invalid duration {duration}.")
            output_token = token
            if parent is not None and token == int(parent):
                if trial_id not in states:
                    raise ValueError(
                        f"Trial {trial_id} contains expanded parent {parent} but lacks a gate state."
                    )
                state = int(states[trial_id])
                if state not in valid_states:
                    raise ValueError(f"Trial {trial_id} has invalid gate state {state}.")
                if state > 0:
                    output_token = int(child_tokens[state - 1])
            matrix[row, output_token] += duration / total
    if not np.allclose(matrix.sum(axis=1), 1.0, atol=1e-8):
        raise RuntimeError("Adaptive duration histograms do not sum to one.")
    return matrix.astype(np.float32)


def audit_codebook_invariants(
    model: AdaptiveCodebookModel,
    current_base_centers: np.ndarray | None = None,
) -> dict:
    """Return a JSON-ready immutable-base/append-only capacity audit."""

    expected_base = int(model.config.base_token_count)
    snapshot_hash = codebook_center_hash(model.base_centers)
    current = model.base_centers if current_base_centers is None else np.asarray(
        current_base_centers
    )
    current_hash = codebook_center_hash(current)
    child_ids = [
        int(token)
        for expansion in model.expansions
        for token in expansion.child_token_ids
    ]
    expected_child_ids = list(range(expected_base, expected_base + len(child_ids)))
    parent_ids = [int(item.parent_token) for item in model.expansions]
    checks = {
        "base_shape_is_registered_k32": bool(
            model.base_centers.ndim == 2
            and model.base_centers.shape[0] == EXPECTED_BASE_TOKEN_COUNT
        ),
        "stored_base_hash_matches_snapshot": snapshot_hash == model.base_center_hash,
        "current_base_hash_matches_frozen": current_hash == model.base_center_hash,
        "current_base_centers_exactly_equal": bool(
            current.shape == model.base_centers.shape
            and np.array_equal(current, model.base_centers)
        ),
        "base_snapshot_is_read_only": not bool(model.base_centers.flags.writeable),
        "expanded_parent_budget_respected": len(model.expansions)
        <= int(model.config.max_expanded_parents_per_session),
        "new_child_budget_respected": len(child_ids)
        <= int(model.config.max_new_children_per_session),
        "exactly_two_children_per_expansion": all(
            len(item.child_token_ids) == CHILDREN_PER_EXPANSION
            for item in model.expansions
        ),
        "child_ids_are_contiguous_append_only": child_ids == expected_child_ids,
        "parent_ids_are_unique": len(parent_ids) == len(set(parent_ids)),
        "parent_ids_remain_in_base_range": all(
            0 <= value < expected_base for value in parent_ids
        ),
        "active_size_is_k32_or_k34": model.active_token_count
        in (EXPECTED_BASE_TOKEN_COUNT, EXPECTED_BASE_TOKEN_COUNT + 2),
    }
    return {
        "base_center_hash": model.base_center_hash,
        "snapshot_center_hash": snapshot_hash,
        "current_center_hash": current_hash,
        "base_center_shape": list(model.base_centers.shape),
        "expanded_parent_tokens": parent_ids,
        "appended_child_token_ids": child_ids,
        "active_token_count": int(model.active_token_count),
        "checks": checks,
        "all_invariants_passed": bool(all(checks.values())),
    }


def adaptive_codebook_diagnostics(model: AdaptiveCodebookModel) -> dict:
    """Return compact JSON-ready train-only model diagnostics."""

    def candidate_dict(item: ParentCandidateAudit) -> dict:
        return {
            field.name: getattr(item, field.name) for field in fields(item)
        }

    configuration = {
        field.name: getattr(model.config, field.name) for field in fields(model.config)
    }
    expansion_rows = [
        {
            "session_index": int(item.session_index),
            "parent_token": int(item.parent_token),
            "child_token_ids": list(item.child_token_ids),
            "child_medoid_trial_ids": list(item.child_medoid_trial_ids),
            "child_confidence_radii": item.child_confidence_radii.tolist(),
            "descriptor_block_sizes": list(item.descriptor_block_sizes),
            "fit_trial_ids": list(item.fit_trial_ids),
            "fit_subject_ids": list(item.fit_subject_ids),
            "fit_child_ids": list(item.fit_child_ids),
        }
        for item in model.expansions
    ]

    return {
        "session_index": int(model.session_index),
        "fit_trial_count": len(model.train_trial_ids),
        "fit_trial_ids": list(model.train_trial_ids),
        "base_token_count": int(model.config.base_token_count),
        "active_token_count": int(model.active_token_count),
        "K_total": int(model.K_total),
        "gate_enabled": bool(model.gate_enabled),
        "selected_parent_token": model.selected_parent_token,
        "selected_parent_tokens": [
            int(item.parent_token) for item in model.expansions
        ],
        "configuration": configuration,
        "expansions": expansion_rows,
        "candidate_audits": [candidate_dict(item) for item in model.candidate_audits],
        "invariant_audit": audit_codebook_invariants(model),
        "fit_uses_activity_labels_or_names": False,
        "subject_id_used_as_descriptor": False,
        "subject_id_uses": [
            "equal_weighting",
            "minimum_support",
            "subject_confound_rejection",
        ],
        "test_occurrences_used_for_fit": False,
        "descriptor_recipe": DESCRIPTOR_RECIPE,
        "leave_one_trial_preprocessing": (
            "location/scale fitted independently on each trial-fold complement"
        ),
        "leave_one_subject_preprocessing": (
            "location/scale fitted independently on each subject-fold complement"
        ),
        "held_out_descriptor_used_to_fit_fold_scaler": False,
    }


__all__ = [
    "AdaptiveAssignments",
    "AdaptiveCodebookConfig",
    "AdaptiveCodebookModel",
    "AdaptiveExpansion",
    "AdaptiveOccurrence",
    "CHILDREN_PER_EXPANSION",
    "DESCRIPTOR_RECIPE",
    "EXPECTED_BASE_TOKEN_COUNT",
    "ParentCandidateAudit",
    "adaptive_codebook_diagnostics",
    "adaptive_duration_histograms",
    "adaptive_state_by_trial",
    "assert_adaptive_schema_is_label_free",
    "assign_adaptive_codebook",
    "audit_codebook_invariants",
    "build_adaptive_occurrence",
    "codebook_center_hash",
    "fit_adaptive_codebook",
]
