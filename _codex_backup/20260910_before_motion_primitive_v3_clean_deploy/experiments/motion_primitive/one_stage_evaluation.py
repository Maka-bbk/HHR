"""CGCD-oriented evaluation for one-stage motion-primitive trajectories.

The routines in this module never use evaluation labels while fitting the
semi-supervised clusters.  Old-class train trials are fixed labelled anchors;
all outer-test trials are treated as unlabeled during clustering.  Ground
truth is accepted only by :func:`score_cgcd_clusters`, after assignments are
complete, to name the novel clusters and compute research metrics.

This is a transductive evaluation: outer-test *features* participate in the
clustering fit.  It is therefore a representation/CGCD screening protocol,
not a deployable inductive classifier.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import confusion_matrix, f1_score


EPSILON = 1.0e-12


def _as_finite_matrix(name: str, value: np.ndarray) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 2 or array.shape[0] < 1 or array.shape[1] < 1:
        raise ValueError(f"{name} must be a non-empty [samples,features] matrix.")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains non-finite values.")
    return array


def _as_integer_vector(name: str, value: np.ndarray, size: int) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim != 1 or len(array) != int(size):
        raise ValueError(f"{name} must be a vector of length {size}.")
    if not np.issubdtype(array.dtype, np.integer):
        if not np.all(np.isfinite(array)) or not np.all(array == np.floor(array)):
            raise TypeError(f"{name} must contain exact integers.")
    return array.astype(np.int64, copy=False)


def l2_normalize_rows(features: np.ndarray) -> np.ndarray:
    values = _as_finite_matrix("features", features)
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    return values / np.maximum(norms, EPSILON)


def _squared_distances(features: np.ndarray, centres: np.ndarray) -> np.ndarray:
    distances = (
        np.sum(features * features, axis=1, keepdims=True)
        + np.sum(centres * centres, axis=1)[None, :]
        - 2.0 * features @ centres.T
    )
    return np.maximum(distances, 0.0)


def _kmeans_plus_plus_new_centres(
    features: np.ndarray,
    fixed_centres: np.ndarray,
    count: int,
    rng: np.random.Generator,
) -> np.ndarray:
    selected: list[np.ndarray] = []
    all_centres = fixed_centres.copy()
    for _ in range(int(count)):
        minimum = _squared_distances(features, all_centres).min(axis=1)
        total = float(minimum.sum())
        if total <= EPSILON:
            # Deterministic under ``rng`` and safe when all points coincide.
            index = int(rng.integers(0, len(features)))
        else:
            index = int(rng.choice(len(features), p=minimum / total))
        centre = features[index].copy()
        selected.append(centre)
        all_centres = np.concatenate([all_centres, centre[None, :]], axis=0)
    return np.stack(selected, axis=0)


@dataclass(frozen=True)
class SemiSupervisedKMeansResult:
    assignments: np.ndarray
    centres: np.ndarray
    inertia: float
    iterations: int
    seed: int
    restart: int


def semi_supervised_kmeans(
    labelled_features: np.ndarray,
    labelled_targets: np.ndarray,
    unlabelled_features: np.ndarray,
    *,
    num_classes: int = 12,
    num_old_classes: int = 6,
    seed: int = 0,
    n_init: int = 10,
    max_iterations: int = 100,
    tolerance: float = 1.0e-6,
    normalize: bool = True,
) -> SemiSupervisedKMeansResult:
    """Fit constrained K-means without reading unlabelled ground truth.

    The first ``num_old_classes`` cluster identities are fixed by the labelled
    old-class trials.  Labelled assignments never change.  Remaining centres
    are initialised from the unlabeled population and every unlabeled trial may
    select any old or new centre.
    """

    labelled = _as_finite_matrix("labelled_features", labelled_features)
    unlabelled = _as_finite_matrix("unlabelled_features", unlabelled_features)
    if labelled.shape[1] != unlabelled.shape[1]:
        raise ValueError("Labelled and unlabelled feature widths differ.")
    targets = _as_integer_vector("labelled_targets", labelled_targets, len(labelled))
    num_classes = int(num_classes)
    num_old_classes = int(num_old_classes)
    if not 1 <= num_old_classes < num_classes:
        raise ValueError("Expected 1 <= num_old_classes < num_classes.")
    if int(n_init) < 1 or int(max_iterations) < 1:
        raise ValueError("n_init and max_iterations must be positive.")
    if not np.isfinite(tolerance) or float(tolerance) < 0:
        raise ValueError("tolerance must be finite and non-negative.")
    if np.any(targets < 0) or np.any(targets >= num_old_classes):
        raise ValueError("Labelled targets must contain old-class ids only.")
    observed = set(targets.tolist())
    expected = set(range(num_old_classes))
    if observed != expected:
        raise ValueError(
            f"Every old class needs a labelled anchor; observed={sorted(observed)}."
        )
    if len(unlabelled) < num_classes - num_old_classes:
        raise ValueError("Too few unlabeled trials to initialise novel centres.")

    if normalize:
        labelled = l2_normalize_rows(labelled)
        unlabelled = l2_normalize_rows(unlabelled)

    old_centres = np.stack(
        [labelled[targets == class_id].mean(axis=0) for class_id in range(num_old_classes)],
        axis=0,
    )
    if normalize:
        old_centres = l2_normalize_rows(old_centres)

    best: Optional[SemiSupervisedKMeansResult] = None
    master = np.random.SeedSequence(int(seed))
    restart_sequences = master.spawn(int(n_init))
    for restart, sequence in enumerate(restart_sequences):
        rng = np.random.default_rng(sequence)
        novel_centres = _kmeans_plus_plus_new_centres(
            unlabelled,
            old_centres,
            num_classes - num_old_classes,
            rng,
        )
        centres = np.concatenate([old_centres, novel_centres], axis=0)
        previous_assignments: Optional[np.ndarray] = None
        iterations = 0
        for iteration in range(1, int(max_iterations) + 1):
            iterations = iteration
            assignments = np.argmin(
                _squared_distances(unlabelled, centres), axis=1
            ).astype(np.int64)
            updated = np.empty_like(centres)
            distances_to_assigned = _squared_distances(unlabelled, centres)[
                np.arange(len(unlabelled)), assignments
            ]
            for cluster_id in range(num_classes):
                pieces = []
                if cluster_id < num_old_classes:
                    pieces.append(labelled[targets == cluster_id])
                members = unlabelled[assignments == cluster_id]
                if len(members):
                    pieces.append(members)
                if pieces:
                    updated[cluster_id] = np.concatenate(pieces, axis=0).mean(axis=0)
                else:
                    # Re-seed only an empty novel centre with the currently
                    # worst represented unlabeled trial.
                    farthest = int(np.argmax(distances_to_assigned))
                    updated[cluster_id] = unlabelled[farthest]
                    distances_to_assigned[farthest] = -np.inf
            if normalize:
                updated = l2_normalize_rows(updated)
            shift = float(np.max(np.linalg.norm(updated - centres, axis=1)))
            converged = previous_assignments is not None and np.array_equal(
                assignments, previous_assignments
            )
            centres = updated
            previous_assignments = assignments.copy()
            if converged or shift <= float(tolerance):
                break

        final_distances = _squared_distances(unlabelled, centres)
        assignments = np.argmin(final_distances, axis=1).astype(np.int64)
        labelled_inertia = float(
            _squared_distances(labelled, centres)[np.arange(len(labelled)), targets].sum()
        )
        unlabelled_inertia = float(
            final_distances[np.arange(len(unlabelled)), assignments].sum()
        )
        candidate = SemiSupervisedKMeansResult(
            assignments=assignments,
            centres=centres,
            inertia=labelled_inertia + unlabelled_inertia,
            iterations=iterations,
            seed=int(seed),
            restart=int(restart),
        )
        if best is None or candidate.inertia < best.inertia:
            best = candidate
    if best is None:  # pragma: no cover - guarded by n_init validation.
        raise RuntimeError("Semi-supervised K-means produced no candidate.")
    return best


def _novel_hungarian_mapping(
    cluster_ids: np.ndarray,
    targets: np.ndarray,
    num_classes: int,
    num_old_classes: int,
) -> Dict[int, int]:
    novel_clusters = np.arange(num_old_classes, num_classes, dtype=np.int64)
    novel_targets = np.arange(num_old_classes, num_classes, dtype=np.int64)
    matrix = np.zeros((len(novel_clusters), len(novel_targets)), dtype=np.int64)
    # Ground truth is used here only to name already-fitted novel clusters.
    novel_truth_mask = targets >= num_old_classes
    for row, cluster_id in enumerate(novel_clusters):
        selected = novel_truth_mask & (cluster_ids == cluster_id)
        if np.any(selected):
            matrix[row] = np.bincount(
                targets[selected] - num_old_classes,
                minlength=len(novel_targets),
            )[: len(novel_targets)]
    rows, columns = linear_sum_assignment(-matrix)
    return {
        int(novel_clusters[row]): int(novel_targets[column])
        for row, column in zip(rows, columns)
    }


def score_cgcd_clusters(
    assignments: np.ndarray,
    targets: np.ndarray,
    *,
    num_classes: int = 12,
    num_old_classes: int = 6,
    activity_names: Optional[Dict[int, str]] = None,
) -> Dict[str, Any]:
    """Map novel clusters for scoring and report old/new/H-score metrics."""

    clusters = _as_integer_vector("assignments", assignments, len(assignments))
    truth = _as_integer_vector("targets", targets, len(clusters))
    num_classes = int(num_classes)
    num_old_classes = int(num_old_classes)
    if np.any(clusters < 0) or np.any(clusters >= num_classes):
        raise ValueError("assignments contains an out-of-range cluster id.")
    if np.any(truth < 0) or np.any(truth >= num_classes):
        raise ValueError("targets contains an out-of-range class id.")
    mapping = {class_id: class_id for class_id in range(num_old_classes)}
    mapping.update(
        _novel_hungarian_mapping(clusters, truth, num_classes, num_old_classes)
    )
    predictions = np.asarray([mapping[int(item)] for item in clusters], dtype=np.int64)
    old_mask = truth < num_old_classes
    new_mask = ~old_mask

    def accuracy(mask: np.ndarray) -> float:
        return float(np.mean(predictions[mask] == truth[mask])) if np.any(mask) else float("nan")

    old_accuracy = accuracy(old_mask)
    new_accuracy = accuracy(new_mask)
    denominator = old_accuracy + new_accuracy
    h_score = (
        float(2.0 * old_accuracy * new_accuracy / denominator)
        if denominator > 0.0
        else 0.0
    )
    recalls = {}
    for class_id in range(num_classes):
        mask = truth == class_id
        name = activity_names.get(class_id, str(class_id)) if activity_names else str(class_id)
        recalls[name] = accuracy(mask)
    return {
        "all_accuracy": accuracy(np.ones(len(truth), dtype=bool)),
        "old_accuracy": old_accuracy,
        "new_accuracy": new_accuracy,
        "h_score": h_score,
        "macro_f1": float(
            f1_score(
                truth,
                predictions,
                labels=np.arange(num_classes),
                average="macro",
                zero_division=0,
            )
        ),
        "per_class_recall": recalls,
        "novel_cluster_mapping": {
            str(key): int(value)
            for key, value in mapping.items()
            if key >= num_old_classes
        },
        "confusion_matrix": confusion_matrix(
            truth, predictions, labels=np.arange(num_classes)
        ).astype(int).tolist(),
        "predictions": predictions,
    }


def codebook_diagnostics(
    hard_tokens: np.ndarray,
    subject_ids: np.ndarray,
    *,
    num_codes: int,
) -> Dict[str, float]:
    """Report occupancy/effective count and token-subject dependence."""

    tokens = np.asarray(hard_tokens, dtype=np.int64).reshape(-1)
    subjects = np.asarray(subject_ids, dtype=np.int64).reshape(-1)
    if len(tokens) != len(subjects) or not len(tokens):
        raise ValueError("hard_tokens and subject_ids must be equally sized and non-empty.")
    if np.any(tokens < 0) or np.any(tokens >= int(num_codes)):
        raise ValueError("hard_tokens contains an out-of-range code id.")
    counts = np.bincount(tokens, minlength=int(num_codes)).astype(np.float64)
    probabilities = counts / counts.sum()
    nonzero = probabilities > 0
    entropy = float(-np.sum(probabilities[nonzero] * np.log(probabilities[nonzero])))
    # Implement NMI locally to keep this diagnostic explicit.
    subject_values, subject_inverse = np.unique(subjects, return_inverse=True)
    joint = np.zeros((int(num_codes), len(subject_values)), dtype=np.float64)
    np.add.at(joint, (tokens, subject_inverse), 1.0)
    joint /= joint.sum()
    token_marginal = joint.sum(axis=1, keepdims=True)
    subject_marginal = joint.sum(axis=0, keepdims=True)
    expected = token_marginal @ subject_marginal
    valid = joint > 0
    mutual_information = float(np.sum(joint[valid] * np.log(joint[valid] / expected[valid])))
    subject_probs = subject_marginal.reshape(-1)
    subject_entropy = float(-np.sum(subject_probs[subject_probs > 0] * np.log(subject_probs[subject_probs > 0])))
    normalizer = max(EPSILON, float(np.sqrt(entropy * subject_entropy)))
    return {
        "occupied_codes": int(np.count_nonzero(counts)),
        "occupancy_fraction": float(np.count_nonzero(counts) / int(num_codes)),
        "effective_code_count": float(np.exp(entropy)),
        "token_entropy": entropy,
        "token_subject_nmi": float(mutual_information / normalizer),
    }


__all__ = [
    "SemiSupervisedKMeansResult",
    "codebook_diagnostics",
    "l2_normalize_rows",
    "score_cgcd_clusters",
    "semi_supervised_kmeans",
]
