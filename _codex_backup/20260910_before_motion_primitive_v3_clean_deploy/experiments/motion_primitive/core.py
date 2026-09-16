"""Pure numerical helpers for the motion-primitive feasibility experiment.

The functions in this module deliberately avoid Happy's training loop.  They
operate on frozen window embeddings and ordered per-trial metadata so that the
experiment can be tested without changing continual-training behavior.
"""

from __future__ import annotations

import heapq
from collections import Counter
from dataclasses import dataclass
from typing import Iterable, Optional, Sequence

import numpy as np


EPS = 1e-12


def l2_normalize(values: np.ndarray, eps: float = EPS) -> np.ndarray:
    """Return row-wise L2-normalized float32 values."""
    values = np.asarray(values, dtype=np.float32)
    if values.ndim != 2:
        raise ValueError(f"Expected a 2D array, got {values.shape}.")
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    return (values / np.maximum(norms, float(eps))).astype(np.float32)


@dataclass(frozen=True)
class WeightedPCA:
    """Small deterministic PCA transform fitted with explicit sample weights."""

    mean: np.ndarray
    components: np.ndarray
    explained_variance: np.ndarray
    explained_variance_ratio: np.ndarray

    def transform(self, values: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=np.float32)
        if values.ndim != 2 or values.shape[1] != self.mean.shape[0]:
            raise ValueError(
                f"PCA input must be [N,{self.mean.shape[0]}], got {values.shape}."
            )
        return ((values - self.mean) @ self.components.T).astype(np.float32)


def fit_weighted_pca(
    values: np.ndarray,
    n_components: int,
    sample_weights: Optional[np.ndarray] = None,
) -> WeightedPCA:
    """Fit PCA without letting long trials dominate the covariance estimate.

    ``sample_weights`` are normalized to sum to one.  The returned components
    are ordered by decreasing weighted covariance eigenvalue.
    """
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] < 2:
        raise ValueError(f"PCA requires a 2D array with at least 2 rows: {values.shape}.")
    n_components = int(n_components)
    max_components = min(values.shape[0] - 1, values.shape[1])
    if not 1 <= n_components <= max_components:
        raise ValueError(
            f"n_components must be in [1,{max_components}], got {n_components}."
        )

    if sample_weights is None:
        weights = np.full(values.shape[0], 1.0 / values.shape[0], dtype=np.float64)
    else:
        weights = np.asarray(sample_weights, dtype=np.float64)
        if weights.shape != (values.shape[0],):
            raise ValueError(
                f"PCA weights must have shape {(values.shape[0],)}, got {weights.shape}."
            )
        if not np.all(np.isfinite(weights)) or np.any(weights < 0):
            raise ValueError("PCA weights must be finite and non-negative.")
        total = float(weights.sum())
        if total <= 0:
            raise ValueError("PCA weights must have positive total mass.")
        weights = weights / total

    mean = np.sum(values * weights[:, None], axis=0)
    centered = values - mean
    covariance = centered.T @ (centered * weights[:, None])
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = np.maximum(eigenvalues[order], 0.0)
    components = eigenvectors[:, order[:n_components]].T
    selected = eigenvalues[:n_components]
    total_variance = float(eigenvalues.sum())
    ratios = selected / max(total_variance, EPS)
    return WeightedPCA(
        mean=mean.astype(np.float32),
        components=components.astype(np.float32),
        explained_variance=selected.astype(np.float32),
        explained_variance_ratio=ratios.astype(np.float32),
    )


def inverse_trial_frequency_weights(trial_ids: np.ndarray) -> np.ndarray:
    """Give every trial equal total mass regardless of its window count."""
    trial_ids = np.asarray(trial_ids)
    if trial_ids.ndim != 1 or len(trial_ids) == 0:
        raise ValueError("trial_ids must be a non-empty 1D array.")
    _, inverse, counts = np.unique(trial_ids, return_inverse=True, return_counts=True)
    weights = 1.0 / counts[inverse].astype(np.float64)
    # Scale to mean one for estimators whose tolerance depends on total weight.
    return weights * (len(weights) / weights.sum())


def assign_to_codebook(
    values: np.ndarray,
    centers: np.ndarray,
    metric: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Assign embeddings to centers and return token, distance, stored centers."""
    values = np.asarray(values, dtype=np.float32)
    centers = np.asarray(centers, dtype=np.float32)
    if values.ndim != 2 or centers.ndim != 2 or values.shape[1] != centers.shape[1]:
        raise ValueError(
            f"Incompatible values/centers shapes: {values.shape}, {centers.shape}."
        )
    if metric == "cosine":
        normalized_values = l2_normalize(values)
        stored_centers = l2_normalize(centers)
        similarity = normalized_values @ stored_centers.T
        tokens = np.argmax(similarity, axis=1).astype(np.int64)
        distances = 1.0 - similarity[np.arange(len(values)), tokens]
    elif metric == "euclidean":
        stored_centers = centers.copy()
        squared = (
            np.sum(values * values, axis=1, keepdims=True)
            + np.sum(stored_centers * stored_centers, axis=1)[None, :]
            - 2.0 * values @ stored_centers.T
        )
        squared = np.maximum(squared, 0.0)
        tokens = np.argmin(squared, axis=1).astype(np.int64)
        distances = np.sqrt(squared[np.arange(len(values)), tokens])
    else:
        raise ValueError(f"Unknown assignment metric {metric!r}.")
    return tokens, distances.astype(np.float32), stored_centers.astype(np.float32)


def run_length_encode(tokens: Sequence[int]) -> tuple[np.ndarray, np.ndarray]:
    """Return run token ids and run lengths for a non-empty token sequence."""
    tokens = np.asarray(tokens, dtype=np.int64)
    if tokens.ndim != 1 or len(tokens) == 0:
        raise ValueError("tokens must be a non-empty 1D sequence.")
    boundaries = np.flatnonzero(np.r_[True, tokens[1:] != tokens[:-1], True])
    starts = boundaries[:-1]
    ends = boundaries[1:]
    return tokens[starts].copy(), (ends - starts).astype(np.int64)


def shuffle_valid_rle_tokens(
    tokens: Sequence[int],
    rng: np.random.Generator,
    rejection_attempts: int = 256,
) -> list[int]:
    """Shuffle an RLE token multiset without creating equal neighbors.

    Directly permuting RLE tokens can put two runs with the same token next to
    each other. Such a sequence is not valid RLE because those runs would
    merge. Valid permutations are rejection-sampled first; a randomized
    max-count scheduler provides a guaranteed fallback.
    """
    values = np.asarray(tokens, dtype=np.int64)
    if values.ndim != 1 or len(values) == 0:
        raise ValueError("RLE tokens must be a non-empty 1D sequence.")
    if np.any(values[1:] == values[:-1]):
        raise ValueError("Input contains equal adjacent tokens and is not valid RLE.")
    if len(values) == 1:
        return values.astype(int).tolist()

    for _ in range(max(0, int(rejection_attempts))):
        candidate = rng.permutation(values)
        if np.all(candidate[1:] != candidate[:-1]):
            return candidate.astype(int).tolist()

    counts = Counter(int(value) for value in values.tolist())
    heap = [
        (-count, float(rng.random()), token)
        for token, count in counts.items()
    ]
    heapq.heapify(heap)
    result = []
    held_count = 0
    held_token = None
    while heap:
        count, _, token = heapq.heappop(heap)
        result.append(int(token))
        count += 1
        if held_count < 0:
            heapq.heappush(
                heap,
                (held_count, float(rng.random()), int(held_token)),
            )
        held_count = count
        held_token = int(token)
    if held_count < 0 or len(result) != len(values):
        raise RuntimeError("Could not construct a valid shuffled RLE sequence.")
    if any(left == right for left, right in zip(result, result[1:])):
        raise RuntimeError("RLE shuffle produced equal adjacent tokens.")
    if Counter(result) != counts:
        raise RuntimeError("RLE shuffle did not preserve the token multiset.")
    return result


def build_trial_sequence(
    tokens: Sequence[int],
    window_starts: Sequence[int],
    window_size: int,
    sample_rate_hz: float,
) -> dict:
    """Build raw and run-length sequence metadata for one ordered trial.

    Durations are explicitly named *observed spans*.  The NPZ omits the final
    incomplete tail of each raw trial, so these values must not be interpreted
    as exact activity durations.
    """
    tokens = np.asarray(tokens, dtype=np.int64)
    starts = np.asarray(window_starts, dtype=np.int64)
    if tokens.ndim != 1 or starts.ndim != 1 or len(tokens) != len(starts):
        raise ValueError("tokens and window_starts must be equally sized 1D arrays.")
    if len(tokens) == 0:
        raise ValueError("A trial sequence cannot be empty.")
    if np.any(np.diff(starts) <= 0):
        raise ValueError("window_starts must be strictly increasing.")
    if int(window_size) <= 0 or float(sample_rate_hz) <= 0:
        raise ValueError("window_size and sample_rate_hz must be positive.")

    run_tokens, run_lengths = run_length_encode(tokens)
    run_starts = np.r_[0, np.cumsum(run_lengths)[:-1]].astype(np.int64)
    runs = []
    for primitive_id, run_length, begin in zip(run_tokens, run_lengths, run_starts):
        end = int(begin + run_length)
        start_sample = int(starts[begin])
        end_sample_exclusive = int(starts[end - 1] + int(window_size))
        runs.append(
            {
                "primitive_id": int(primitive_id),
                "run_length_windows": int(run_length),
                "first_window_offset": int(begin),
                "last_window_offset": int(end - 1),
                "start_sample": start_sample,
                "end_sample_exclusive": end_sample_exclusive,
                "observed_span_seconds": float(
                    (end_sample_exclusive - start_sample) / float(sample_rate_hz)
                ),
            }
        )
    trial_end = int(starts[-1] + int(window_size))
    return {
        "raw_tokens": tokens.tolist(),
        "rle_tokens": run_tokens.tolist(),
        "run_lengths": run_lengths.tolist(),
        "runs": runs,
        "window_count": int(len(tokens)),
        "run_count": int(len(run_tokens)),
        "unique_primitive_count": int(len(np.unique(tokens))),
        "observed_trial_span_seconds": float(
            (trial_end - int(starts[0])) / float(sample_rate_hz)
        ),
    }


def primitive_histogram(tokens: Sequence[int], primitive_num: int) -> np.ndarray:
    tokens = np.asarray(tokens, dtype=np.int64)
    if tokens.ndim != 1 or len(tokens) == 0:
        raise ValueError("tokens must be a non-empty 1D sequence.")
    if np.any(tokens < 0) or np.any(tokens >= int(primitive_num)):
        raise ValueError("Token id is outside the codebook range.")
    counts = np.bincount(tokens, minlength=int(primitive_num)).astype(np.float64)
    return (counts / counts.sum()).astype(np.float32)


def normalized_levenshtein(left: Sequence[int], right: Sequence[int]) -> float:
    """Levenshtein distance divided by max sequence length."""
    left = tuple(int(value) for value in left)
    right = tuple(int(value) for value in right)
    if len(left) == 0 and len(right) == 0:
        return 0.0
    if len(left) > len(right):
        left, right = right, left
    previous = np.arange(len(left) + 1, dtype=np.int32)
    for row, right_value in enumerate(right, start=1):
        current = np.empty(len(left) + 1, dtype=np.int32)
        current[0] = row
        for column, left_value in enumerate(left, start=1):
            current[column] = min(
                current[column - 1] + 1,
                previous[column] + 1,
                previous[column - 1] + (left_value != right_value),
            )
        previous = current
    return float(previous[-1] / max(len(left), len(right)))


def normalized_weighted_levenshtein(
    left: Sequence[int],
    right: Sequence[int],
    substitution_cost,
) -> float:
    """Weighted edit distance divided by the maximum sequence length.

    Insertion and deletion each have fixed cost ``1``. ``substitution_cost``
    must be either a finite, non-negative square ``[K,K]`` matrix indexed by
    token id, or a callback with signature
    ``(left_token, right_token, left_index, right_index) -> cost``. Indices
    passed to the callback are zero-based positions in the original input
    sequences, which permits position-dependent substitution costs.

    Substitution costs greater than one are accepted. Consequently, length
    normalization alone does not guarantee a result in ``[0,1]``; callers
    that require that range must supply costs in ``[0,1]``.
    """
    left = tuple(int(value) for value in left)
    right = tuple(int(value) for value in right)

    if callable(substitution_cost):

        def resolve_cost(
            left_token: int,
            right_token: int,
            left_index: int,
            right_index: int,
        ) -> float:
            raw_cost = substitution_cost(
                left_token, right_token, left_index, right_index
            )
            value = np.asarray(raw_cost)
            if value.ndim != 0:
                raise ValueError(
                    "Substitution-cost callback must return a scalar value."
                )
            try:
                cost = float(value)
            except (TypeError, ValueError) as error:
                raise ValueError(
                    "Substitution-cost callback must return a real scalar."
                ) from error
            if not np.isfinite(cost) or cost < 0.0:
                raise ValueError(
                    "Substitution costs must be finite and non-negative."
                )
            return cost

    else:
        try:
            costs = np.asarray(substitution_cost, dtype=np.float64)
        except (TypeError, ValueError) as error:
            raise ValueError(
                "substitution_cost must be a callback or numeric square matrix."
            ) from error
        if (
            costs.ndim != 2
            or costs.shape[0] != costs.shape[1]
            or costs.shape[0] == 0
        ):
            raise ValueError("Substitution-cost matrix must have shape [K,K].")
        if not np.all(np.isfinite(costs)) or np.any(costs < 0.0):
            raise ValueError("Substitution costs must be finite and non-negative.")
        primitive_num = int(costs.shape[0])
        tokens = left + right
        if any(token < 0 or token >= primitive_num for token in tokens):
            raise ValueError("Token id is outside the substitution-cost matrix range.")

        def resolve_cost(
            left_token: int,
            right_token: int,
            left_index: int,
            right_index: int,
        ) -> float:
            del left_index, right_index
            return float(costs[left_token, right_token])

    if len(left) == 0 and len(right) == 0:
        return 0.0

    # Do not swap the inputs to reduce memory: callback indices must continue
    # to refer to the original left/right sequences.
    previous = np.arange(len(right) + 1, dtype=np.float64)
    for left_offset, left_token in enumerate(left):
        current = np.empty(len(right) + 1, dtype=np.float64)
        current[0] = float(left_offset + 1)
        for right_offset, right_token in enumerate(right):
            substitution = resolve_cost(
                left_token,
                right_token,
                left_offset,
                right_offset,
            )
            current[right_offset + 1] = min(
                current[right_offset] + 1.0,
                previous[right_offset + 1] + 1.0,
                previous[right_offset] + substitution,
            )
        previous = current
    return float(previous[-1] / max(len(left), len(right)))


def jensen_shannon_distance(left: np.ndarray, right: np.ndarray) -> float:
    """Stable square-root Jensen-Shannon divergence using natural logarithms."""
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    if left.shape != right.shape or left.ndim != 1:
        raise ValueError(f"JSD inputs must be equal 1D shapes: {left.shape}, {right.shape}.")
    left = np.maximum(left, 0.0)
    right = np.maximum(right, 0.0)
    left = left / max(float(left.sum()), EPS)
    right = right / max(float(right.sum()), EPS)
    middle = 0.5 * (left + right)

    def kl_divergence(source: np.ndarray, target: np.ndarray) -> float:
        valid = source > 0
        return float(np.sum(source[valid] * np.log(source[valid] / target[valid])))

    divergence = 0.5 * kl_divergence(left, middle)
    divergence += 0.5 * kl_divergence(right, middle)
    return float(np.sqrt(max(divergence, 0.0)))


def _symmetric_distance_matrix(size: int, distance_function) -> np.ndarray:
    matrix = np.zeros((int(size), int(size)), dtype=np.float32)
    for left in range(int(size)):
        for right in range(left + 1, int(size)):
            value = float(distance_function(left, right))
            matrix[left, right] = value
            matrix[right, left] = value
    return matrix


def distance_matrix_from_sequences(sequences: Sequence[Sequence[int]]) -> np.ndarray:
    return _symmetric_distance_matrix(
        len(sequences),
        lambda left, right: normalized_levenshtein(
            sequences[left], sequences[right]
        ),
    )


def distance_matrix_from_histograms(histograms: np.ndarray) -> np.ndarray:
    histograms = np.asarray(histograms, dtype=np.float32)
    if histograms.ndim != 2:
        raise ValueError(f"histograms must be 2D, got {histograms.shape}.")
    return _symmetric_distance_matrix(
        len(histograms),
        lambda left, right: jensen_shannon_distance(
            histograms[left], histograms[right]
        ),
    )


def length_distance_matrix(lengths: Sequence[int]) -> np.ndarray:
    """Order-free duration shortcut baseline on log observed window count."""
    values = np.log1p(np.asarray(lengths, dtype=np.float64))
    if values.ndim != 1 or len(values) == 0:
        raise ValueError("lengths must be a non-empty 1D sequence.")
    scale = max(float(values.max() - values.min()), EPS)
    return (np.abs(values[:, None] - values[None, :]) / scale).astype(np.float32)


def cross_subject_pair_indices(
    labels: Sequence[int],
    subject_ids: Sequence[int],
    class_ids: Optional[Iterable[int]] = None,
) -> tuple[np.ndarray, np.ndarray]:
    labels = np.asarray(labels, dtype=np.int64)
    subject_ids = np.asarray(subject_ids, dtype=np.int64)
    if labels.shape != subject_ids.shape or labels.ndim != 1:
        raise ValueError("labels and subject_ids must be equally sized 1D arrays.")
    selected = np.ones(len(labels), dtype=bool)
    if class_ids is not None:
        selected = np.isin(labels, np.asarray(list(class_ids), dtype=np.int64))
    left, right = np.triu_indices(len(labels), k=1)
    valid = selected[left] & selected[right] & (subject_ids[left] != subject_ids[right])
    return left[valid].astype(np.int64), right[valid].astype(np.int64)


def cross_subject_nearest_neighbor_accuracy(
    distance_matrix: np.ndarray,
    labels: Sequence[int],
    subject_ids: Sequence[int],
    class_ids: Optional[Iterable[int]] = None,
) -> tuple[float, np.ndarray]:
    """Return tie-aware expected 1-NN accuracy across other subjects.

    Discrete edit distances often produce several equally near trials.  Each
    tied candidate receives equal weight instead of using record order as an
    implicit tie breaker. ``nearest`` retains the first tie only as a compact
    query-availability marker for existing callers.
    """
    distance_matrix = np.asarray(distance_matrix, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    subject_ids = np.asarray(subject_ids, dtype=np.int64)
    if distance_matrix.shape != (len(labels), len(labels)):
        raise ValueError("distance_matrix shape does not match labels.")
    selected = np.ones(len(labels), dtype=bool)
    if class_ids is not None:
        selected = np.isin(labels, np.asarray(list(class_ids), dtype=np.int64))
    nearest = np.full(len(labels), -1, dtype=np.int64)
    correct = []
    for index in np.flatnonzero(selected):
        candidates = np.flatnonzero(selected & (subject_ids != subject_ids[index]))
        if len(candidates) == 0:
            continue
        candidate_distances = distance_matrix[index, candidates]
        minimum = float(np.min(candidate_distances))
        tied = candidates[
            np.isclose(candidate_distances, minimum, rtol=0.0, atol=1e-12)
        ]
        nearest[index] = int(tied[0])
        correct.append(float(np.mean(labels[tied] == labels[index])))
    if not correct:
        raise ValueError("No cross-subject nearest-neighbor queries were available.")
    return float(np.mean(correct)), nearest


def association_summary(
    distance_matrix: np.ndarray,
    labels: Sequence[int],
    subject_ids: Sequence[int],
    class_ids: Optional[Iterable[int]] = None,
) -> dict:
    """Summarize same-activity versus different-activity cross-subject distance."""
    distance_matrix = np.asarray(distance_matrix, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    subject_ids = np.asarray(subject_ids, dtype=np.int64)
    class_ids_list = None if class_ids is None else sorted(int(x) for x in class_ids)
    left, right = cross_subject_pair_indices(labels, subject_ids, class_ids_list)
    if len(left) == 0:
        raise ValueError("No cross-subject pairs are available.")
    distances = distance_matrix[left, right]
    same_mask = labels[left] == labels[right]
    same = distances[same_mask]
    different = distances[~same_mask]
    if len(same) == 0 or len(different) == 0:
        raise ValueError("Both same- and different-activity pairs are required.")
    probability = np.mean(same[:, None] < different[None, :])
    probability += 0.5 * np.mean(same[:, None] == different[None, :])
    nn_accuracy, nearest = cross_subject_nearest_neighbor_accuracy(
        distance_matrix, labels, subject_ids, class_ids_list
    )
    return {
        "class_ids": class_ids_list,
        "same_pair_count": int(len(same)),
        "different_pair_count": int(len(different)),
        "same_mean": float(np.mean(same)),
        "same_median": float(np.median(same)),
        "different_mean": float(np.mean(different)),
        "different_median": float(np.median(different)),
        "separation_ratio_different_over_same": float(
            np.mean(different) / max(float(np.mean(same)), EPS)
        ),
        "mean_margin_different_minus_same": float(
            np.mean(different) - np.mean(same)
        ),
        "probability_same_distance_is_smaller": float(probability),
        "cross_subject_1nn_activity_accuracy": float(nn_accuracy),
        "cross_subject_1nn_query_count": int(np.sum(nearest >= 0)),
    }


def permute_labels_within_subjects(
    labels: Sequence[int], subject_ids: Sequence[int], rng: np.random.Generator
) -> np.ndarray:
    labels = np.asarray(labels, dtype=np.int64)
    subject_ids = np.asarray(subject_ids, dtype=np.int64)
    permuted = labels.copy()
    for subject_id in np.unique(subject_ids):
        indices = np.flatnonzero(subject_ids == subject_id)
        permuted[indices] = rng.permutation(permuted[indices])
    return permuted


def label_permutation_test(
    distance_matrix: np.ndarray,
    labels: Sequence[int],
    subject_ids: Sequence[int],
    class_ids: Optional[Iterable[int]],
    permutations: int,
    seed: int,
) -> dict:
    """Blocked label-permutation test that preserves each subject's class counts."""
    permutations = int(permutations)
    distance_matrix = np.asarray(distance_matrix, dtype=np.float64)
    labels_array = np.asarray(labels, dtype=np.int64)
    subjects_array = np.asarray(subject_ids, dtype=np.int64)
    class_ids_list = None if class_ids is None else sorted(int(x) for x in class_ids)
    observed = association_summary(
        distance_matrix, labels_array, subjects_array, class_ids_list
    )
    if permutations <= 0:
        return {"permutations": 0, "observed": observed}

    selected = np.ones(len(labels_array), dtype=bool)
    if class_ids_list is not None:
        selected = np.isin(labels_array, np.asarray(class_ids_list, dtype=np.int64))
    left, right = cross_subject_pair_indices(
        labels_array, subjects_array, class_ids_list
    )
    pair_distances = distance_matrix[left, right]
    query_indices = np.flatnonzero(selected)
    tied_candidates = {}
    for query_index in query_indices:
        candidates = np.flatnonzero(
            selected & (subjects_array != subjects_array[query_index])
        )
        if len(candidates) == 0:
            continue
        candidate_distances = distance_matrix[query_index, candidates]
        minimum = float(np.min(candidate_distances))
        tied_candidates[int(query_index)] = candidates[
            np.isclose(candidate_distances, minimum, rtol=0.0, atol=1e-12)
        ]
    query_indices = np.asarray(sorted(tied_candidates), dtype=np.int64)

    rng = np.random.default_rng(int(seed))
    margins = np.empty(permutations, dtype=np.float64)
    accuracies = np.empty(permutations, dtype=np.float64)
    for iteration in range(permutations):
        # When a class subset is requested, shuffle only inside that subset so
        # old/novel membership remains fixed while semantic labels are broken.
        permuted = labels_array.copy()
        for subject_id in np.unique(subjects_array[selected]):
            indices = np.flatnonzero(selected & (subjects_array == subject_id))
            permuted[indices] = rng.permutation(permuted[indices])
        same = permuted[left] == permuted[right]
        if not np.any(same) or np.all(same):
            raise RuntimeError("A label permutation removed a required pair group.")
        margins[iteration] = float(
            np.mean(pair_distances[~same]) - np.mean(pair_distances[same])
        )
        accuracies[iteration] = float(
            np.mean(
                [
                    np.mean(
                        permuted[tied_candidates[int(query_index)]]
                        == permuted[query_index]
                    )
                    for query_index in query_indices
                ]
            )
        )
    observed_margin = observed["mean_margin_different_minus_same"]
    observed_accuracy = observed["cross_subject_1nn_activity_accuracy"]
    return {
        "permutations": permutations,
        "observed": observed,
        "margin_null_mean": float(np.mean(margins)),
        "margin_p_value_greater": float(
            (1 + np.sum(margins >= observed_margin)) / (permutations + 1)
        ),
        "one_nn_null_mean": float(np.mean(accuracies)),
        "one_nn_p_value_greater": float(
            (1 + np.sum(accuracies >= observed_accuracy)) / (permutations + 1)
        ),
    }


def activity_distance_matrix(
    distance_matrix: np.ndarray,
    labels: Sequence[int],
    subject_ids: Sequence[int],
    class_ids: Sequence[int],
) -> np.ndarray:
    """Mean cross-subject trial distance for every ordered activity pair."""
    distance_matrix = np.asarray(distance_matrix, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    subject_ids = np.asarray(subject_ids, dtype=np.int64)
    class_ids = [int(value) for value in class_ids]
    result = np.full((len(class_ids), len(class_ids)), np.nan, dtype=np.float32)
    for row, left_label in enumerate(class_ids):
        left_indices = np.flatnonzero(labels == left_label)
        for column, right_label in enumerate(class_ids):
            right_indices = np.flatnonzero(labels == right_label)
            values = []
            for left in left_indices:
                for right in right_indices:
                    if subject_ids[left] == subject_ids[right]:
                        continue
                    if left_label == right_label and left >= right:
                        continue
                    values.append(distance_matrix[left, right])
            if values:
                result[row, column] = float(np.mean(values))
    return result
