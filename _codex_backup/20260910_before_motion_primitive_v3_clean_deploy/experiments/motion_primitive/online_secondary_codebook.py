"""Unlabelled within-token secondary codebook utilities.

The fitting path in this module deliberately has no activity label/name field.
Ground truth is accepted only by the post-hoc evaluation functions after child
assignments and downstream cluster predictions have already been frozen.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, fields, replace
from typing import Mapping, Sequence

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import (
    adjusted_rand_score,
    confusion_matrix,
    f1_score,
    normalized_mutual_info_score,
    recall_score,
    silhouette_score,
)

from experiments.motion_primitive.core import EPS, jensen_shannon_distance


ARMS = ("U0_coarse", "U1_residual", "U2_gravity", "U3_joint")
SPLIT_ARMS = ARMS[1:]


@dataclass(frozen=True)
class UnlabelledTokenOccurrence:
    """One trial-token occurrence used by the unlabelled fitter.

    Keeping labels and activity names out of the type makes accidental fitting
    leakage substantially harder than relying on a convention at call sites.
    """

    trial_global_id: int
    coarse_token: int
    token_fraction: float
    duration_samples: int
    mean_quantization_residual: np.ndarray
    gravity_direction: np.ndarray


@dataclass(frozen=True)
class SecondaryCodebook:
    arm: str
    coarse_token: int
    residual_scale: float
    medoid_trial_ids: tuple[int, int]
    medoid_residuals: np.ndarray
    medoid_gravity: np.ndarray
    objective: float
    restarts: int
    train_trial_ids: tuple[int, ...]
    train_child_ids: tuple[int, ...]


def assert_unlabelled_occurrence_schema() -> None:
    names = {item.name.lower() for item in fields(UnlabelledTokenOccurrence)}
    forbidden = {"label", "labels", "name", "activity", "activity_name"}
    if names & forbidden or any("label" in name or "name" in name for name in names):
        raise RuntimeError(
            "UnlabelledTokenOccurrence must not contain label/name fields: "
            f"{sorted(names)}"
        )


def validate_session_manifest(
    session_one_train_ids: Sequence[int],
    session_two_train_ids: Sequence[int],
    session_two_test_ids: Sequence[int],
) -> dict:
    """Validate the cumulative Session-2 train/test trial boundary."""

    first = tuple(sorted({int(value) for value in session_one_train_ids}))
    second = tuple(sorted({int(value) for value in session_two_train_ids}))
    test = tuple(sorted({int(value) for value in session_two_test_ids}))
    if not first or not second or not test:
        raise ValueError("Session-1 train, Session-2 train, and Session-2 test must be non-empty.")
    repeated_train = set(first) & set(second)
    cumulative = tuple(sorted(set(first) | set(second)))
    overlap = set(cumulative) & set(test)
    if repeated_train:
        raise RuntimeError(
            f"A trial was reused by Session-1 and Session-2 train: {sorted(repeated_train)}"
        )
    if overlap:
        raise RuntimeError(
            f"Cumulative online-train/session-2-test trial leakage: {sorted(overlap)}"
        )
    return {
        "session_1_train_trial_ids": list(first),
        "session_2_train_trial_ids": list(second),
        "cumulative_train_trial_ids": list(cumulative),
        "session_2_test_trial_ids": list(test),
        "session_1_session_2_train_overlap": 0,
        "cumulative_train_test_overlap": 0,
    }


def validate_trial_token_durations(
    trial_token_durations: Mapping[int, Mapping[int, float]],
) -> None:
    if not trial_token_durations:
        raise ValueError("At least one trial token-duration mapping is required.")
    for trial_id, token_durations in trial_token_durations.items():
        if not token_durations:
            raise ValueError(f"Trial {trial_id} has no token duration.")
        for token, duration in token_durations.items():
            if int(token) < 0 or not np.isfinite(duration) or float(duration) <= 0:
                raise ValueError(
                    f"Trial {trial_id} has invalid token/duration {token}:{duration}."
                )


def select_dominant_token(
    trial_token_durations: Mapping[int, Mapping[int, float]],
    minimum_fraction: float = 0.50,
    minimum_support: int = 10,
) -> dict:
    """Choose the token dominating the most trials, without class metadata."""

    validate_trial_token_durations(trial_token_durations)
    if not 0.0 < float(minimum_fraction) <= 1.0:
        raise ValueError("minimum_fraction must lie in (0,1].")
    if int(minimum_support) < 1:
        raise ValueError("minimum_support must be positive.")
    support: dict[int, int] = {}
    dominant_trials: dict[int, list[int]] = {}
    for trial_id, token_durations in trial_token_durations.items():
        total = float(sum(float(value) for value in token_durations.values()))
        for token, duration in token_durations.items():
            fraction = float(duration) / total
            if fraction + 1e-12 >= float(minimum_fraction):
                token = int(token)
                support[token] = support.get(token, 0) + 1
                dominant_trials.setdefault(token, []).append(int(trial_id))
    eligible = [token for token, count in support.items() if count >= int(minimum_support)]
    if not eligible:
        raise RuntimeError(
            "No coarse token reached the registered dominant-trial support: "
            f"minimum_support={minimum_support}, observed={dict(sorted(support.items()))}."
        )
    selected = min(eligible, key=lambda token: (-support[token], token))
    return {
        "selected_token": int(selected),
        "selected_support": int(support[selected]),
        "minimum_fraction": float(minimum_fraction),
        "minimum_support": int(minimum_support),
        "support_by_token": {str(key): int(value) for key, value in sorted(support.items())},
        "dominant_trial_ids": sorted(dominant_trials[selected]),
        "selection_uses_labels": False,
    }


def duration_weighted_residual(
    segment_embeddings: np.ndarray,
    assigned_center: np.ndarray,
    duration_samples: np.ndarray,
) -> np.ndarray:
    embeddings = np.asarray(segment_embeddings, dtype=np.float64)
    center = np.asarray(assigned_center, dtype=np.float64)
    durations = np.asarray(duration_samples, dtype=np.float64)
    if embeddings.ndim != 2 or center.shape != (embeddings.shape[1],):
        raise ValueError(
            f"Embedding/center shapes disagree: {embeddings.shape}, {center.shape}."
        )
    if durations.shape != (len(embeddings),) or np.any(durations <= 0):
        raise ValueError("Every segment must have one positive partition duration.")
    residuals = embeddings - center[None, :]
    result = np.average(residuals, axis=0, weights=durations)
    if not np.all(np.isfinite(result)):
        raise RuntimeError("Duration-weighted residual is non-finite.")
    return result.astype(np.float32)


def _unit_vector(value: np.ndarray, name: str) -> np.ndarray:
    vector = np.asarray(value, dtype=np.float64)
    if vector.ndim != 1 or not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must be a finite vector, got {vector.shape}.")
    norm = float(np.linalg.norm(vector))
    if norm <= EPS:
        raise ValueError(f"{name} has zero norm.")
    return (vector / norm).astype(np.float32)


def gravity_from_token_partitions(
    sensor: np.ndarray,
    partition_starts: np.ndarray,
    partition_ends: np.ndarray,
    partition_tokens: np.ndarray,
    selected_token: int,
) -> np.ndarray:
    """Aggregate robust gravity over contiguous non-overlapping token runs."""

    signal = np.asarray(sensor, dtype=np.float64)
    starts = np.asarray(partition_starts, dtype=np.int64)
    ends = np.asarray(partition_ends, dtype=np.int64)
    tokens = np.asarray(partition_tokens, dtype=np.int64)
    if signal.ndim != 2 or signal.shape[0] < 3 or signal.shape[1] < 1:
        raise ValueError(f"sensor must have shape [>=3,T], got {signal.shape}.")
    if starts.ndim != 1 or ends.shape != starts.shape or tokens.shape != starts.shape:
        raise ValueError("Partition starts, ends, and tokens must be equal 1D arrays.")
    if len(starts) == 0 or np.any(starts < 0) or np.any(ends <= starts):
        raise ValueError("Partitions must be non-empty positive half-open intervals.")
    order = np.argsort(starts, kind="stable")
    starts, ends, tokens = starts[order], ends[order], tokens[order]
    if np.any(starts[1:] < ends[:-1]):
        raise ValueError("Partition intervals overlap.")
    if int(ends.max()) > signal.shape[1]:
        raise ValueError("Partition extends beyond the visible raw trial span.")
    selected = np.flatnonzero(tokens == int(selected_token))
    if len(selected) == 0:
        raise ValueError(f"Selected token {selected_token} is absent from the trial.")

    runs: list[tuple[int, int]] = []
    for position in selected.tolist():
        start, end = int(starts[position]), int(ends[position])
        if runs and start == runs[-1][1]:
            runs[-1] = (runs[-1][0], end)
        else:
            runs.append((start, end))
    directions = []
    durations = []
    for start, end in runs:
        median_acceleration = np.median(signal[:3, start:end], axis=1)
        directions.append(_unit_vector(median_acceleration, "run median acceleration"))
        durations.append(end - start)
    spherical_mean = np.average(
        np.asarray(directions, dtype=np.float64),
        axis=0,
        weights=np.asarray(durations, dtype=np.float64),
    )
    return _unit_vector(spherical_mean, "duration-weighted gravity direction")


def validate_occurrences(
    occurrences: Sequence[UnlabelledTokenOccurrence],
    expected_token: int | None = None,
) -> None:
    assert_unlabelled_occurrence_schema()
    if not occurrences:
        raise ValueError("At least one unlabelled occurrence is required.")
    trial_ids = [int(item.trial_global_id) for item in occurrences]
    if len(trial_ids) != len(set(trial_ids)):
        raise ValueError("There must be at most one occurrence per trial-token.")
    residual_dim = None
    for item in occurrences:
        if expected_token is not None and int(item.coarse_token) != int(expected_token):
            raise ValueError("Occurrence coarse token differs from the selected token.")
        if not 0.0 < float(item.token_fraction) <= 1.0:
            raise ValueError("token_fraction must lie in (0,1].")
        if int(item.duration_samples) <= 0:
            raise ValueError("duration_samples must be positive.")
        residual = np.asarray(item.mean_quantization_residual)
        gravity = np.asarray(item.gravity_direction)
        if residual.ndim != 1 or not np.all(np.isfinite(residual)):
            raise ValueError("Occurrence residual must be a finite vector.")
        residual_dim = len(residual) if residual_dim is None else residual_dim
        if len(residual) != residual_dim:
            raise ValueError("All occurrence residuals must share a dimension.")
        _unit_vector(gravity, "occurrence gravity")


def _angular_distance_matrix(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    left = left / np.maximum(np.linalg.norm(left, axis=1, keepdims=True), EPS)
    right = right / np.maximum(np.linalg.norm(right, axis=1, keepdims=True), EPS)
    cosine = np.clip(left @ right.T, -1.0, 1.0)
    return np.arccos(cosine) / math.pi


def _euclidean_distance_matrix(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    squared = (
        np.sum(left * left, axis=1, keepdims=True)
        + np.sum(right * right, axis=1)[None, :]
        - 2.0 * left @ right.T
    )
    return np.sqrt(np.maximum(squared, 0.0))


def _positive_p95(pairwise: np.ndarray) -> float:
    matrix = np.asarray(pairwise, dtype=np.float64)
    values = matrix[np.triu_indices(len(matrix), k=1)]
    values = values[np.isfinite(values) & (values > 0)]
    return max(float(np.percentile(values, 95.0)), 1e-6) if len(values) else 1.0


def occurrence_distance_matrix(
    left: Sequence[UnlabelledTokenOccurrence],
    right: Sequence[UnlabelledTokenOccurrence],
    arm: str,
    residual_scale: float,
) -> np.ndarray:
    if arm not in SPLIT_ARMS:
        raise ValueError(f"Distance matrix requires a split arm, got {arm!r}.")
    left_residual = np.asarray(
        [item.mean_quantization_residual for item in left], dtype=np.float64
    )
    right_residual = np.asarray(
        [item.mean_quantization_residual for item in right], dtype=np.float64
    )
    left_gravity = np.asarray([item.gravity_direction for item in left], dtype=np.float64)
    right_gravity = np.asarray([item.gravity_direction for item in right], dtype=np.float64)
    residual = _euclidean_distance_matrix(left_residual, right_residual) / max(
        float(residual_scale), 1e-6
    )
    gravity = _angular_distance_matrix(left_gravity, right_gravity)
    if arm == "U1_residual":
        return residual
    if arm == "U2_gravity":
        return gravity
    return 0.5 * residual + 0.5 * gravity


def _pam_two_clusters(distance: np.ndarray, initial_medoids: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    matrix = np.asarray(distance, dtype=np.float64)
    medoids = np.sort(np.asarray(initial_medoids, dtype=np.int64))
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] or len(matrix) < 2:
        raise ValueError("PAM needs a square distance matrix with at least two rows.")
    for _ in range(100):
        labels = np.argmin(matrix[:, medoids], axis=1).astype(np.int64)
        # A medoid always belongs to itself, but exact duplicate points can tie.
        labels[medoids[0]] = 0
        labels[medoids[1]] = 1
        new_medoids = []
        for child in (0, 1):
            members = np.flatnonzero(labels == child)
            if len(members) == 0:
                raise RuntimeError("K-medoids produced an empty child cluster.")
            costs = np.sum(matrix[np.ix_(members, members)], axis=1)
            minimum = float(np.min(costs))
            candidates = members[np.isclose(costs, minimum, rtol=0.0, atol=1e-12)]
            new_medoids.append(int(candidates.min()))
        updated = np.sort(np.asarray(new_medoids, dtype=np.int64))
        if np.array_equal(updated, medoids):
            break
        if updated[0] == updated[1]:
            raise RuntimeError("K-medoids collapsed to one medoid.")
        medoids = updated
    labels = np.argmin(matrix[:, medoids], axis=1).astype(np.int64)
    labels[medoids[0]] = 0
    labels[medoids[1]] = 1
    objective = float(np.sum(matrix[np.arange(len(matrix)), medoids[labels]]))
    return medoids, labels, objective


def fit_secondary_codebook(
    occurrences: Sequence[UnlabelledTokenOccurrence],
    arm: str,
    restarts: int = 50,
    seed: int = 500,
) -> SecondaryCodebook:
    validate_occurrences(occurrences)
    if arm not in SPLIT_ARMS:
        raise ValueError(f"Unknown split arm {arm!r}.")
    if len(occurrences) < 2 or int(restarts) < 1:
        raise ValueError("Two occurrences and at least one restart are required.")
    coarse_tokens = {int(item.coarse_token) for item in occurrences}
    if len(coarse_tokens) != 1:
        raise ValueError("A secondary codebook may fit exactly one coarse token.")
    residuals = np.asarray(
        [item.mean_quantization_residual for item in occurrences], dtype=np.float64
    )
    residual_scale = _positive_p95(_euclidean_distance_matrix(residuals, residuals))
    distance = occurrence_distance_matrix(
        occurrences, occurrences, arm=arm, residual_scale=residual_scale
    )
    rng = np.random.default_rng(int(seed))
    best = None
    degenerate_restart_count = 0
    for _ in range(int(restarts)):
        initial = rng.choice(len(occurrences), size=2, replace=False)
        try:
            candidate = _pam_two_clusters(distance, initial)
        except RuntimeError:
            continue
        medoids = candidate[0]
        if float(distance[int(medoids[0]), int(medoids[1])]) <= 1e-12:
            # Two feature-identical medoids do not define two motion
            # primitives.  In particular, forcing each medoid to own itself
            # would fabricate a second child that frozen argmin assignment
            # cannot reproduce.
            degenerate_restart_count += 1
            continue
        key = (candidate[2], tuple(candidate[0].tolist()))
        if best is None or key < best[0]:
            best = (key, candidate)
    if best is None:
        if degenerate_restart_count:
            raise RuntimeError(
                "All registered K-medoids restarts failed; distinct child "
                "medoids could not be formed."
            )
        raise RuntimeError("All registered K-medoids restarts failed.")
    medoids, labels, objective = best[1]
    medoid_items = [occurrences[int(index)] for index in medoids]
    model = SecondaryCodebook(
        arm=arm,
        coarse_token=int(next(iter(coarse_tokens))),
        residual_scale=float(residual_scale),
        medoid_trial_ids=tuple(int(item.trial_global_id) for item in medoid_items),
        medoid_residuals=np.asarray(
            [item.mean_quantization_residual for item in medoid_items], dtype=np.float32
        ),
        medoid_gravity=np.asarray(
            [item.gravity_direction for item in medoid_items], dtype=np.float32
        ),
        objective=float(objective),
        restarts=int(restarts),
        train_trial_ids=tuple(int(item.trial_global_id) for item in occurrences),
        train_child_ids=tuple(int(value) for value in labels.tolist()),
    )
    reassigned = assign_secondary_codebook(model, occurrences)
    if not np.array_equal(reassigned, labels) or len(np.unique(reassigned)) != 2:
        raise RuntimeError(
            "Frozen secondary-codebook assignment does not reproduce its "
            "two non-empty training children."
        )
    return model


def assign_secondary_codebook(
    model: SecondaryCodebook,
    occurrences: Sequence[UnlabelledTokenOccurrence],
) -> np.ndarray:
    validate_occurrences(occurrences, expected_token=model.coarse_token)
    medoids = [
        UnlabelledTokenOccurrence(
            trial_global_id=int(model.medoid_trial_ids[index]),
            coarse_token=int(model.coarse_token),
            token_fraction=1.0,
            duration_samples=1,
            mean_quantization_residual=model.medoid_residuals[index],
            gravity_direction=model.medoid_gravity[index],
        )
        for index in range(2)
    ]
    distance = occurrence_distance_matrix(
        occurrences, medoids, arm=model.arm, residual_scale=model.residual_scale
    )
    assignments = np.argmin(distance, axis=1).astype(np.int64)
    # Match the fitted PAM convention for exact-distance ties: each medoid
    # remains a member of its own child.  Trial ids are globally unique and
    # contain no activity identity.
    trial_ids = np.asarray(
        [int(item.trial_global_id) for item in occurrences], dtype=np.int64
    )
    for child, medoid_trial_id in enumerate(model.medoid_trial_ids):
        assignments[trial_ids == int(medoid_trial_id)] = int(child)
    return assignments


def secondary_fit_diagnostics(
    model: SecondaryCodebook,
    occurrences: Sequence[UnlabelledTokenOccurrence],
) -> dict:
    observed_trial_ids = tuple(
        int(item.trial_global_id) for item in occurrences
    )
    if observed_trial_ids != model.train_trial_ids:
        raise ValueError(
            "Secondary fit diagnostics require occurrences in the exact "
            "training order used by the fitted model."
        )
    labels = np.asarray(model.train_child_ids, dtype=np.int64)
    distance = occurrence_distance_matrix(
        occurrences, occurrences, arm=model.arm, residual_scale=model.residual_scale
    )
    # Numerical round-off in Euclidean/angular distances can leave an
    # O(1e-8) self-distance.  sklearn requires the precomputed diagonal to be
    # exactly zero even though that noise has no clustering meaning.
    np.fill_diagonal(distance, 0.0)
    counts = np.bincount(labels, minlength=2)
    silhouette = (
        float(silhouette_score(distance, labels, metric="precomputed"))
        if np.all(counts >= 2)
        else None
    )
    return {
        "objective": float(model.objective),
        "child_counts": counts.astype(int).tolist(),
        "minimum_child_fraction": float(counts.min() / len(labels)),
        "silhouette_precomputed": silhouette,
        "residual_distance_p95_scale": float(model.residual_scale),
        "medoid_trial_ids": list(model.medoid_trial_ids),
        "restarts": int(model.restarts),
        "fit_uses_activity_labels": False,
    }


def duration_histograms(
    trial_ids: Sequence[int],
    trial_token_durations: Mapping[int, Mapping[int, float]],
    primitive_num: int,
    selected_token: int | None = None,
    child_by_trial: Mapping[int, int] | None = None,
) -> np.ndarray:
    split = selected_token is not None
    dimension = int(primitive_num) + (1 if split else 0)
    matrix = np.zeros((len(trial_ids), dimension), dtype=np.float64)
    for row, trial_id_value in enumerate(trial_ids):
        trial_id = int(trial_id_value)
        token_durations = trial_token_durations[trial_id]
        total = float(sum(float(value) for value in token_durations.values()))
        for token_value, duration in token_durations.items():
            token = int(token_value)
            if token < 0 or token >= int(primitive_num):
                raise ValueError(f"Token {token} lies outside K={primitive_num}.")
            column = token
            if split and token == int(selected_token):
                if child_by_trial is None or trial_id not in child_by_trial:
                    raise ValueError(f"Trial {trial_id} lacks a frozen child assignment.")
                child = int(child_by_trial[trial_id])
                if child not in (0, 1):
                    raise ValueError(f"Invalid child id {child} for trial {trial_id}.")
                column = token if child == 0 else int(primitive_num)
            matrix[row, column] += float(duration) / total
    if not np.allclose(matrix.sum(axis=1), 1.0, atol=1e-8):
        raise RuntimeError("Duration histograms do not sum to one.")
    return matrix.astype(np.float32)


def shuffle_gravity_within_subject_split(
    occurrences: Sequence[UnlabelledTokenOccurrence],
    subjects_by_trial: Mapping[int, int],
    split_roles_by_trial: Mapping[int, str],
    rng: np.random.Generator,
) -> tuple[list[UnlabelledTokenOccurrence], dict]:
    """Break trajectory/gravity pairing within subject and train/test role.

    Activity labels are intentionally absent.  Each descriptor donor and its
    recipient must share both subject id and split role, so the subject-level
    gravity marginal is preserved without moving test information into the
    visible cumulative training pool.
    """

    validate_occurrences(occurrences)
    if not isinstance(rng, np.random.Generator):
        raise TypeError("rng must be a numpy.random.Generator.")
    groups: dict[tuple[int, str], list[int]] = {}
    for position, item in enumerate(occurrences):
        trial_id = int(item.trial_global_id)
        if trial_id not in subjects_by_trial or trial_id not in split_roles_by_trial:
            raise ValueError(
                f"Trial {trial_id} lacks a subject or split-role shuffle key."
            )
        role = str(split_roles_by_trial[trial_id])
        if role not in {"cumulative_online_train", "session_2_test"}:
            raise ValueError(f"Trial {trial_id} has invalid shuffle role {role!r}.")
        key = (int(subjects_by_trial[trial_id]), role)
        groups.setdefault(key, []).append(position)

    shuffled = list(occurrences)
    moved = 0
    group_sizes = {}
    for (subject_id, role), positions_list in sorted(groups.items()):
        positions = np.asarray(positions_list, dtype=np.int64)
        donors = positions[rng.permutation(len(positions))]
        group_sizes[f"subject={subject_id}|role={role}"] = int(len(positions))
        for recipient, donor in zip(positions.tolist(), donors.tolist()):
            recipient_item = occurrences[int(recipient)]
            donor_item = occurrences[int(donor)]
            recipient_trial = int(recipient_item.trial_global_id)
            donor_trial = int(donor_item.trial_global_id)
            if (
                int(subjects_by_trial[recipient_trial])
                != int(subjects_by_trial[donor_trial])
                or str(split_roles_by_trial[recipient_trial])
                != str(split_roles_by_trial[donor_trial])
            ):
                raise RuntimeError("Gravity shuffle crossed a subject/split boundary.")
            shuffled[int(recipient)] = replace(
                recipient_item,
                gravity_direction=np.asarray(
                    donor_item.gravity_direction, dtype=np.float32
                ).copy(),
            )
            moved += int(recipient != donor)

    validate_occurrences(shuffled)
    return shuffled, {
        "group_sizes": group_sizes,
        "occurrence_count": len(occurrences),
        "moved_occurrence_count": int(moved),
        "moved_occurrence_fraction": float(moved / len(occurrences)),
        "preserves_subject_gravity_marginal": True,
        "crosses_train_test_boundary": False,
        "uses_activity_labels": False,
    }


def monte_carlo_upper_tail_summary(
    observed: float,
    null_values: Sequence[float],
) -> dict:
    values = np.asarray(null_values, dtype=np.float64)
    if values.ndim != 1 or len(values) < 1 or not np.all(np.isfinite(values)):
        raise ValueError("Monte Carlo null values must be a finite non-empty vector.")
    observed = float(observed)
    if not np.isfinite(observed):
        raise ValueError("Observed Monte Carlo statistic must be finite.")
    quantiles = np.quantile(values, [0.025, 0.50, 0.975])
    return {
        "observed": observed,
        "null_mean": float(np.mean(values)),
        "observed_minus_null_mean": float(observed - np.mean(values)),
        "monte_carlo_upper_tail_p": float(
            (1 + np.sum(values >= observed)) / (len(values) + 1)
        ),
        "null_quantiles": {
            "q025": float(quantiles[0]),
            "q500": float(quantiles[1]),
            "q975": float(quantiles[2]),
        },
        "permutation_count": int(len(values)),
    }


def stratified_paired_accuracy_bootstrap(
    trial_ids: Sequence[int],
    y_true: Sequence[int],
    subjects: Sequence[int],
    reference_aligned: Sequence[int],
    challenger_aligned: Sequence[int],
    subset_mask: Sequence[bool],
    resamples: int = 5000,
    seed: int = 500,
    confidence_level: float = 0.95,
) -> dict:
    """Bootstrap paired correctness with frozen full-test-set alignments."""

    trial_ids_array = np.asarray(trial_ids, dtype=np.int64)
    truth = np.asarray(y_true, dtype=np.int64)
    subject_values = np.asarray(subjects, dtype=np.int64)
    reference = np.asarray(reference_aligned, dtype=np.int64)
    challenger = np.asarray(challenger_aligned, dtype=np.int64)
    mask = np.asarray(subset_mask, dtype=bool)
    expected = (len(trial_ids_array),)
    for name, values in (
        ("truth", truth),
        ("subjects", subject_values),
        ("reference", reference),
        ("challenger", challenger),
        ("subset_mask", mask),
    ):
        if values.shape != expected:
            raise ValueError(f"{name} shape {values.shape} != {expected}.")
    if len(trial_ids_array) == 0 or len(set(trial_ids_array.tolist())) != len(
        trial_ids_array
    ):
        raise ValueError("Bootstrap trial ids must be non-empty and unique.")
    if int(resamples) < 1:
        raise ValueError("Bootstrap resamples must be positive.")
    if not 0.0 < float(confidence_level) < 1.0:
        raise ValueError("confidence_level must lie in (0,1).")
    selected = np.flatnonzero(mask)
    if len(selected) == 0:
        raise ValueError("Bootstrap subset is empty.")

    strata: dict[tuple[int, int], np.ndarray] = {}
    for subject_id in sorted(set(subject_values[selected].tolist())):
        for activity_id in sorted(set(truth[selected].tolist())):
            positions = selected[
                (subject_values[selected] == int(subject_id))
                & (truth[selected] == int(activity_id))
            ]
            if len(positions):
                strata[(int(subject_id), int(activity_id))] = positions
    if sum(len(value) for value in strata.values()) != len(selected):
        raise RuntimeError("Bootstrap strata do not partition the selected trials.")

    delta = (challenger == truth).astype(np.float64) - (
        reference == truth
    ).astype(np.float64)
    rng = np.random.default_rng(int(seed))
    bootstrap_sums = np.zeros(int(resamples), dtype=np.float64)
    for positions in strata.values():
        draws = rng.choice(
            positions,
            size=(int(resamples), len(positions)),
            replace=True,
        )
        bootstrap_sums += np.sum(delta[draws], axis=1)
    bootstrap_values = bootstrap_sums / len(selected)
    alpha = (1.0 - float(confidence_level)) / 2.0
    lower, median, upper = np.quantile(
        bootstrap_values, [alpha, 0.5, 1.0 - alpha]
    )
    return {
        "difference_direction": "challenger_minus_reference_accuracy",
        "observed_difference": float(np.mean(delta[selected])),
        "bootstrap_mean_difference": float(np.mean(bootstrap_values)),
        "confidence_level": float(confidence_level),
        "ci_lower": float(lower),
        "ci_median": float(median),
        "ci_upper": float(upper),
        "bootstrap_probability_gt_zero": float(np.mean(bootstrap_values > 0.0)),
        "resamples": int(resamples),
        "trial_count": int(len(selected)),
        "resampling_unit": "trial",
        "stratification": "subject_x_activity",
        "hungarian_mapping_policy": "frozen_on_complete_test_set_before_bootstrap",
        "strata_counts": {
            f"subject={subject}|activity={activity}": int(len(positions))
            for (subject, activity), positions in sorted(strata.items())
        },
    }


def global_hungarian_metrics(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    old_class_count: int = 6,
) -> dict:
    """Use exactly one global assignment for All, Old, New, F1 and recall."""

    truth = np.asarray(y_true, dtype=np.int64)
    predicted = np.asarray(y_pred, dtype=np.int64)
    if truth.ndim != 1 or predicted.shape != truth.shape or len(truth) == 0:
        raise ValueError("y_true/y_pred must be equal non-empty 1D arrays.")
    if np.any(truth < 0) or np.any(predicted < 0):
        raise ValueError("Cluster and activity ids must be non-negative.")
    size = int(max(truth.max(initial=0), predicted.max(initial=0)) + 1)
    contingency = np.zeros((size, size), dtype=np.int64)
    for cluster, label in zip(predicted.tolist(), truth.tolist()):
        contingency[int(cluster), int(label)] += 1
    rows, columns = linear_sum_assignment(contingency.max() - contingency)
    mapping = {int(row): int(column) for row, column in zip(rows, columns)}
    aligned = np.asarray([mapping[int(value)] for value in predicted], dtype=np.int64)
    correct = aligned == truth
    old_mask = truth < int(old_class_count)
    new_mask = ~old_mask
    class_ids = sorted(set(truth.tolist()))
    recalls = recall_score(
        truth, aligned, labels=class_ids, average=None, zero_division=0
    )
    confusion = confusion_matrix(truth, aligned, labels=class_ids)
    return {
        "alignment_scope": "single_global_hungarian",
        "hungarian_call_count": 1,
        "mapping_pred_to_true": {str(key): int(value) for key, value in sorted(mapping.items())},
        "all_accuracy": float(np.mean(correct)),
        "old_accuracy": float(np.mean(correct[old_mask])) if np.any(old_mask) else None,
        "new_accuracy": float(np.mean(correct[new_mask])) if np.any(new_mask) else None,
        "macro_f1": float(
            f1_score(truth, aligned, labels=class_ids, average="macro", zero_division=0)
        ),
        "class_ids": class_ids,
        "per_class_recall": {
            str(label): float(value) for label, value in zip(class_ids, recalls.tolist())
        },
        "confusion_counts": confusion.astype(int),
        "aligned_predictions": aligned.astype(int),
    }


def posthoc_secondary_metrics(
    child_by_trial: Mapping[int, int],
    labels_by_trial: Mapping[int, int],
    subjects_by_trial: Mapping[int, int],
    sitting_label: int = 7,
    standing_label: int = 8,
) -> dict:
    """Evaluate frozen children; none of these values feed back into fitting."""

    common = sorted(
        set(int(value) for value in child_by_trial)
        & set(int(value) for value in labels_by_trial)
        & set(int(value) for value in subjects_by_trial)
    )
    all_children = np.asarray([child_by_trial[key] for key in common], dtype=np.int64)
    all_subjects = np.asarray([subjects_by_trial[key] for key in common], dtype=np.int64)
    all_labels = np.asarray([labels_by_trial[key] for key in common], dtype=np.int64)
    selected = np.isin(all_labels, [int(sitting_label), int(standing_label)])
    if not np.any(selected):
        raise RuntimeError("No Sitting/Standing trials reached post-hoc evaluation.")
    binary_truth = all_labels[selected]
    binary_children = all_children[selected]
    unique_children = sorted(set(binary_children.tolist()))
    unique_truth = sorted(set(binary_truth.tolist()))
    contingency = np.zeros((2, 2), dtype=np.int64)
    child_position = {value: index for index, value in enumerate([0, 1])}
    truth_position = {
        int(sitting_label): 0,
        int(standing_label): 1,
    }
    for child, label in zip(binary_children.tolist(), binary_truth.tolist()):
        contingency[child_position[int(child)], truth_position[int(label)]] += 1
    rows, columns = linear_sum_assignment(contingency.max() - contingency)
    mapping = {
        int(row): [int(sitting_label), int(standing_label)][int(column)]
        for row, column in zip(rows, columns)
    }
    aligned = np.asarray([mapping[int(value)] for value in binary_children], dtype=np.int64)
    class_recalls = recall_score(
        binary_truth,
        aligned,
        labels=[int(sitting_label), int(standing_label)],
        average=None,
        zero_division=0,
    )
    by_subject = {}
    selected_subjects = all_subjects[selected]
    for subject_id in sorted(set(selected_subjects.tolist())):
        subject_mask = selected_subjects == int(subject_id)
        subject_truth = binary_truth[subject_mask]
        subject_aligned = aligned[subject_mask]
        observed_classes = sorted(set(subject_truth.tolist()))
        subject_recalls = recall_score(
            subject_truth,
            subject_aligned,
            labels=[int(sitting_label), int(standing_label)],
            average=None,
            zero_division=0,
        )
        complete_support = observed_classes == [
            int(sitting_label),
            int(standing_label),
        ]
        by_subject[str(int(subject_id))] = {
            "trial_count": int(np.sum(subject_mask)),
            "observed_activity_ids": observed_classes,
            "complete_binary_support": bool(complete_support),
            "accuracy_using_global_binary_mapping": float(
                np.mean(subject_aligned == subject_truth)
            ),
            "balanced_accuracy_using_global_binary_mapping": (
                float(np.mean(subject_recalls)) if complete_support else None
            ),
            "per_class_recall_using_global_binary_mapping": {
                str(label): float(value)
                for label, value in zip(
                    [int(sitting_label), int(standing_label)],
                    subject_recalls.tolist(),
                )
            },
            "confusion_counts_using_global_binary_mapping": confusion_matrix(
                subject_truth,
                subject_aligned,
                labels=[int(sitting_label), int(standing_label)],
            ).astype(int),
        }
    return {
        "evaluation_is_posthoc": True,
        "fit_used_labels_or_names": False,
        "evaluated_trial_ids": [int(common[index]) for index in np.flatnonzero(selected)],
        "binary_trial_count": int(np.sum(selected)),
        "observed_child_ids": unique_children,
        "observed_activity_ids": unique_truth,
        "binary_hungarian_accuracy": float(np.mean(aligned == binary_truth)),
        "binary_balanced_accuracy": float(np.mean(class_recalls)),
        "binary_nmi": float(normalized_mutual_info_score(binary_truth, binary_children)),
        "binary_ari": float(adjusted_rand_score(binary_truth, binary_children)),
        "child_subject_nmi_all_selected_token_trials": float(
            normalized_mutual_info_score(all_subjects, all_children)
        ),
        "child_activity_nmi_all_selected_token_trials": float(
            normalized_mutual_info_score(all_labels, all_children)
        ),
        "binary_mapping_child_to_activity": {
            str(key): int(value) for key, value in sorted(mapping.items())
        },
        "binary_by_subject_using_same_global_mapping": by_subject,
        "binary_per_class_recall": {
            str(label): float(value)
            for label, value in zip(
                [int(sitting_label), int(standing_label)], class_recalls.tolist()
            )
        },
        "binary_confusion_counts": confusion_matrix(
            binary_truth,
            aligned,
            labels=[int(sitting_label), int(standing_label)],
        ).astype(int),
    }


def class_mean_histogram_distances(
    histograms: np.ndarray,
    labels: Sequence[int],
    class_ids: Sequence[int],
) -> np.ndarray:
    values = np.asarray(histograms, dtype=np.float64)
    truth = np.asarray(labels, dtype=np.int64)
    if values.ndim != 2 or truth.shape != (len(values),):
        raise ValueError("Histogram matrix and labels disagree.")
    means = []
    for class_id in class_ids:
        selected = values[truth == int(class_id)]
        if len(selected) == 0:
            raise ValueError(f"Class {class_id} is absent from the heatmap input.")
        mean = np.mean(selected, axis=0)
        means.append(mean / max(float(mean.sum()), EPS))
    matrix = np.zeros((len(means), len(means)), dtype=np.float64)
    for left in range(len(means)):
        for right in range(left + 1, len(means)):
            distance = float(jensen_shannon_distance(means[left], means[right]))
            matrix[left, right] = matrix[right, left] = distance
    return matrix.astype(np.float32)


__all__ = [
    "ARMS",
    "SPLIT_ARMS",
    "SecondaryCodebook",
    "UnlabelledTokenOccurrence",
    "assert_unlabelled_occurrence_schema",
    "assign_secondary_codebook",
    "class_mean_histogram_distances",
    "duration_histograms",
    "duration_weighted_residual",
    "fit_secondary_codebook",
    "global_hungarian_metrics",
    "gravity_from_token_partitions",
    "monte_carlo_upper_tail_summary",
    "posthoc_secondary_metrics",
    "secondary_fit_diagnostics",
    "shuffle_gravity_within_subject_split",
    "stratified_paired_accuracy_bootstrap",
    "select_dominant_token",
    "validate_occurrences",
    "validate_session_manifest",
]
