"""Pure scoring for strict motion-primitive CGCD predictions.

This module is intentionally independent from ``strict_registry``.  It may
read benchmark truth after a runner has frozen raw predictions, but it returns
new values only: it cannot update registry state, thresholds or discovery
decisions.
"""

from __future__ import annotations

from typing import Any, Mapping, Optional, Sequence

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import f1_score


UNKNOWN_REGISTRY_ID = -1


def _validated_vectors(
    targets: Sequence[int],
    predictions: Sequence[int],
    *,
    class_count: int,
    registered_class_ids: Optional[Sequence[int]],
) -> tuple[np.ndarray, np.ndarray, tuple[int, ...]]:
    truth = np.asarray(targets, dtype=np.int64).reshape(-1).copy()
    raw = np.asarray(predictions, dtype=np.int64).reshape(-1).copy()
    count = int(class_count)
    if count < 2 or truth.shape != raw.shape or not len(truth):
        raise ValueError("targets/predictions must be equal non-empty vectors.")
    if np.any(truth < 0) or np.any(truth >= count):
        raise ValueError("A target lies outside [0,class_count).")
    registered = (
        tuple(range(count))
        if registered_class_ids is None
        else tuple(sorted(set(int(value) for value in registered_class_ids)))
    )
    if any(value < 0 or value >= count for value in registered):
        raise ValueError("registered_class_ids contains an invalid class ID.")
    permitted = np.isin(raw, np.asarray(registered, dtype=np.int64)) | (
        raw == UNKNOWN_REGISTRY_ID
    )
    if not np.all(permitted):
        raise ValueError("Predictions contain an unregistered ID other than unknown=-1.")
    return truth, raw, registered


def _hungarian_mapping(
    truth: np.ndarray,
    raw: np.ndarray,
    raw_ids: Sequence[int],
    truth_ids: Sequence[int],
) -> dict[int, int]:
    rows = tuple(int(value) for value in raw_ids)
    columns = tuple(int(value) for value in truth_ids)
    if not rows or not columns:
        return {}
    contingency = np.zeros((len(rows), len(columns)), dtype=np.int64)
    row_index = {value: index for index, value in enumerate(rows)}
    column_index = {value: index for index, value in enumerate(columns)}
    for target, prediction in zip(truth.tolist(), raw.tolist()):
        if prediction in row_index and target in column_index:
            contingency[row_index[prediction], column_index[target]] += 1
    selected_rows, selected_columns = linear_sum_assignment(
        contingency.max(initial=0) - contingency
    )
    return {
        rows[int(row)]: columns[int(column)]
        for row, column in zip(selected_rows, selected_columns)
    }


def _apply_mapping(raw: np.ndarray, mapping: Mapping[int, int]) -> np.ndarray:
    return np.asarray(
        [
            UNKNOWN_REGISTRY_ID
            if value == UNKNOWN_REGISTRY_ID
            else int(mapping.get(int(value), int(value)))
            for value in raw.tolist()
        ],
        dtype=np.int64,
    )


def _harmonic(left: float, right: float) -> float:
    return 0.0 if left + right <= 0.0 else 2.0 * left * right / (left + right)


def _augmented_confusion(
    truth: np.ndarray, predictions: np.ndarray, class_count: int
) -> np.ndarray:
    matrix = np.zeros((class_count, class_count + 1), dtype=np.int64)
    for target, prediction in zip(truth.tolist(), predictions.tolist()):
        column = class_count if prediction == UNKNOWN_REGISTRY_ID else int(prediction)
        matrix[int(target), column] += 1
    return matrix


def _layer_metrics(
    truth: np.ndarray,
    predictions: np.ndarray,
    *,
    class_count: int,
    old_class_count: int,
    seen_class_count_before_session: int,
    mapping: Mapping[int, int],
    name: str,
    alignment_uses_test_labels: bool,
) -> dict[str, Any]:
    old_mask = truth < old_class_count
    novel_mask = ~old_mask
    seen_mask = truth < seen_class_count_before_session
    current_new_mask = ~seen_mask

    def accuracy(mask: np.ndarray) -> float:
        return float(np.mean(predictions[mask] == truth[mask])) if np.any(mask) else 0.0

    old_accuracy = accuracy(old_mask)
    novel_accuracy = accuracy(novel_mask)
    confusion = _augmented_confusion(truth, predictions, class_count)
    return {
        "prediction_alignment": name,
        "alignment_uses_test_labels": bool(alignment_uses_test_labels),
        "scoring_only_no_registry_writeback": True,
        "all_accuracy": float(np.mean(predictions == truth)),
        "old_accuracy": old_accuracy,
        "new_accuracy": novel_accuracy,
        "h_score": _harmonic(old_accuracy, novel_accuracy),
        "seen_accuracy": accuracy(seen_mask),
        "unseen_accuracy": accuracy(current_new_mask),
        "macro_f1": float(
            f1_score(
                truth,
                predictions,
                labels=np.arange(class_count),
                average="macro",
                zero_division=0,
            )
        ),
        "unknown_prediction_count": int(np.sum(predictions == UNKNOWN_REGISTRY_ID)),
        "unknown_prediction_fraction": float(
            np.mean(predictions == UNKNOWN_REGISTRY_ID)
        ),
        "confusion_matrix_with_unknown_column": confusion.tolist(),
        "confusion_columns": list(range(class_count)) + ["unknown"],
        "assignment_pred_to_true": [
            [int(source), int(target)] for source, target in sorted(mapping.items())
        ],
        "aligned_predictions": predictions.tolist(),
        "sample_count": int(len(truth)),
    }


def strict_three_layer_metrics(
    targets: Sequence[int],
    raw_predictions: Sequence[int],
    *,
    class_count: int,
    old_class_count: int,
    seen_class_count_before_session: int,
    registered_class_ids: Optional[Sequence[int]] = None,
) -> dict[str, Any]:
    """Score frozen predictions under three explicitly separated alignments.

    ``old_fixed_novel_hungarian`` is the primary CGCD layer.  It cannot rename
    old rows or repair old/novel cross-partition errors.  Both Hungarian layers
    use test identity for scoring only.  ``direct_registry`` performs no
    matching and is the only deployable-ID layer.
    """

    count = int(class_count)
    old_count = int(old_class_count)
    seen_count = int(seen_class_count_before_session)
    if not 1 <= old_count < count:
        raise ValueError("old_class_count must split the active class range.")
    if not old_count <= seen_count <= count:
        raise ValueError("seen_class_count_before_session is outside the active range.")
    truth, raw, registered = _validated_vectors(
        targets,
        raw_predictions,
        class_count=count,
        registered_class_ids=registered_class_ids,
    )

    identity = {class_id: class_id for class_id in registered}
    direct_predictions = _apply_mapping(raw, identity)
    direct = _layer_metrics(
        truth,
        direct_predictions,
        class_count=count,
        old_class_count=old_count,
        seen_class_count_before_session=seen_count,
        mapping=identity,
        name="direct_registry",
        alignment_uses_test_labels=False,
    )

    registered_novel = [value for value in registered if value >= old_count]
    novel_mapping = _hungarian_mapping(
        truth,
        raw,
        registered_novel,
        range(old_count, count),
    )
    old_fixed_mapping = {
        **{value: value for value in registered if value < old_count},
        **novel_mapping,
    }
    old_fixed_predictions = _apply_mapping(raw, old_fixed_mapping)
    old_fixed = _layer_metrics(
        truth,
        old_fixed_predictions,
        class_count=count,
        old_class_count=old_count,
        seen_class_count_before_session=seen_count,
        mapping=old_fixed_mapping,
        name="old_fixed_novel_hungarian",
        alignment_uses_test_labels=True,
    )

    global_mapping = _hungarian_mapping(
        truth, raw, registered, range(count)
    )
    global_predictions = _apply_mapping(raw, global_mapping)
    global_layer = _layer_metrics(
        truth,
        global_predictions,
        class_count=count,
        old_class_count=old_count,
        seen_class_count_before_session=seen_count,
        mapping=global_mapping,
        name="global_hungarian_upper_bound",
        alignment_uses_test_labels=True,
    )
    return {
        "schema": "hhr_strict_cgcd_metrics_v1",
        "primary_layer": "old_fixed_novel_hungarian",
        "raw_predictions_frozen_before_scoring_required": True,
        "scoring_only_no_registry_writeback": True,
        "registered_class_ids": list(registered),
        "layers": {
            "direct_registry": direct,
            "old_fixed_novel_hungarian": old_fixed,
            "global_hungarian_upper_bound": global_layer,
        },
    }


__all__ = ["UNKNOWN_REGISTRY_ID", "strict_three_layer_metrics"]
