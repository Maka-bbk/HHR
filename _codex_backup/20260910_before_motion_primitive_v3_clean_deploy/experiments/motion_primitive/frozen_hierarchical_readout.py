"""Frozen coarse readout with local hierarchical leaf overrides.

This module isolates the effect of an online motion-primitive expansion from
the trial-level KMeans readout.  The coarse KMeans is fitted exactly once.  A
refined arm starts from the frozen coarse predictions and may replace only
explicitly routed trials with newly appended leaf cluster ids.

Activity labels are deliberately absent from every fitting and routing API.
They may be joined later for post-hoc clustering evaluation.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Mapping, Sequence

import numpy as np
from sklearn.cluster import KMeans


@dataclass(frozen=True)
class FrozenCoarseReadout:
    """One train-only KMeans readout and its frozen trial predictions."""

    cluster_count: int
    seed: int
    centers: np.ndarray
    train_trial_ids: tuple[int, ...]
    test_trial_ids: tuple[int, ...]
    train_predictions: np.ndarray
    test_predictions: np.ndarray
    inertia: float
    iterations: int


@dataclass(frozen=True)
class FrozenLeafPredictions:
    """Local leaf overrides applied to a frozen coarse prediction vector."""

    trial_ids: np.ndarray
    coarse_predictions: np.ndarray
    refined_predictions: np.ndarray
    gate_states: np.ndarray
    routed_mask: np.ndarray
    leaf_ids_by_gate_state: Mapping[int, int]


def assert_frozen_readout_schema_is_label_free() -> None:
    """Reject accidental activity identity in the train-time schemas."""

    forbidden = ("label", "activity", "class", "name")
    for schema in (FrozenCoarseReadout, FrozenLeafPredictions):
        leaked = [
            item.name
            for item in fields(schema)
            if any(fragment in item.name.lower() for fragment in forbidden)
        ]
        if leaked:
            raise RuntimeError(f"{schema.__name__} contains identity fields: {leaked}")


def _finite_matrix(value: np.ndarray, name: str) -> np.ndarray:
    matrix = np.asarray(value, dtype=np.float32)
    if matrix.ndim != 2 or len(matrix) == 0 or matrix.shape[1] == 0:
        raise ValueError(f"{name} must be a non-empty 2D matrix.")
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name} must contain only finite values.")
    return matrix


def _unique_trial_ids(values: Sequence[int], expected_length: int, name: str) -> tuple[int, ...]:
    result = tuple(int(value) for value in values)
    if len(result) != int(expected_length) or len(set(result)) != len(result):
        raise ValueError(f"{name} must contain one unique id per feature row.")
    return result


def fit_frozen_coarse_readout(
    train_features: np.ndarray,
    test_features: np.ndarray,
    train_trial_ids: Sequence[int],
    test_trial_ids: Sequence[int],
    *,
    cluster_count: int = 10,
    seed: int = 500,
    n_init: int = 50,
    max_iter: int = 300,
) -> FrozenCoarseReadout:
    """Fit the shared coarse readout exactly once using online-train trials."""

    assert_frozen_readout_schema_is_label_free()
    train = _finite_matrix(train_features, "train_features")
    test = _finite_matrix(test_features, "test_features")
    if train.shape[1] != test.shape[1]:
        raise ValueError("Train and test features must share a dimension.")
    count = int(cluster_count)
    if count < 2 or len(train) < count:
        raise ValueError("cluster_count must be >=2 and no larger than train size.")
    if int(n_init) < 1 or int(max_iter) < 1:
        raise ValueError("n_init and max_iter must be positive.")
    train_ids = _unique_trial_ids(train_trial_ids, len(train), "train_trial_ids")
    test_ids = _unique_trial_ids(test_trial_ids, len(test), "test_trial_ids")
    if set(train_ids) & set(test_ids):
        raise RuntimeError("Frozen readout train/test trial ids overlap.")

    model = KMeans(
        n_clusters=count,
        n_init=int(n_init),
        max_iter=int(max_iter),
        random_state=int(seed),
        algorithm="lloyd",
    )
    train_predictions = model.fit_predict(train).astype(np.int64)
    test_predictions = model.predict(test).astype(np.int64)
    return FrozenCoarseReadout(
        cluster_count=count,
        seed=int(seed),
        centers=np.asarray(model.cluster_centers_, dtype=np.float32).copy(),
        train_trial_ids=train_ids,
        test_trial_ids=test_ids,
        train_predictions=train_predictions,
        test_predictions=test_predictions,
        inertia=float(model.inertia_),
        iterations=int(model.n_iter_),
    )


def apply_frozen_leaf_overrides(
    readout: FrozenCoarseReadout,
    gate_state_by_trial: Mapping[int, int],
    *,
    leaf_ids_by_gate_state: Mapping[int, int] | None = None,
) -> FrozenLeafPredictions:
    """Replace only routed test predictions; every fallback remains bitwise equal.

    Gate state ``0`` is the mandatory coarse fallback.  Positive states receive
    appended raw cluster ids.  The default uses ``K`` and ``K+1`` for states 1
    and 2 respectively, so it cannot collide with the frozen coarse ids.
    """

    assert_frozen_readout_schema_is_label_free()
    default_mapping = {1: int(readout.cluster_count), 2: int(readout.cluster_count) + 1}
    mapping = {
        int(key): int(value)
        for key, value in (leaf_ids_by_gate_state or default_mapping).items()
    }
    if set(mapping) != {1, 2}:
        raise ValueError("Exactly gate states 1 and 2 need leaf ids.")
    if len(set(mapping.values())) != 2 or any(
        value < int(readout.cluster_count) for value in mapping.values()
    ):
        raise ValueError("Leaf ids must be distinct and outside coarse cluster ids.")
    expected_ids = set(readout.test_trial_ids)
    observed_ids = {int(value) for value in gate_state_by_trial}
    if not observed_ids.issubset(expected_ids):
        raise ValueError("gate_state_by_trial contains a non-test trial id.")

    states = np.asarray(
        [int(gate_state_by_trial.get(trial_id, 0)) for trial_id in readout.test_trial_ids],
        dtype=np.int64,
    )
    if not set(np.unique(states).tolist()).issubset({0, 1, 2}):
        raise ValueError("Gate states must be 0, 1, or 2.")
    coarse = np.asarray(readout.test_predictions, dtype=np.int64).copy()
    refined = coarse.copy()
    for state, leaf_id in mapping.items():
        refined[states == int(state)] = int(leaf_id)
    routed = states > 0
    if not np.array_equal(refined[~routed], coarse[~routed]):
        raise RuntimeError("A fallback trial changed under a frozen leaf override.")
    changed = refined != coarse
    if np.any(changed & ~routed):
        raise RuntimeError("Only routed trials may change raw prediction.")
    return FrozenLeafPredictions(
        trial_ids=np.asarray(readout.test_trial_ids, dtype=np.int64),
        coarse_predictions=coarse,
        refined_predictions=refined,
        gate_states=states,
        routed_mask=routed,
        leaf_ids_by_gate_state=dict(sorted(mapping.items())),
    )


def frozen_readout_diagnostics(
    readout: FrozenCoarseReadout,
    leaf_predictions: FrozenLeafPredictions,
) -> dict:
    """Return JSON-ready mechanical invariants for the local intervention."""

    fallback = ~np.asarray(leaf_predictions.routed_mask, dtype=bool)
    changed = (
        np.asarray(leaf_predictions.refined_predictions)
        != np.asarray(leaf_predictions.coarse_predictions)
    )
    return {
        "coarse_fit_call_count": 1,
        "cluster_count": int(readout.cluster_count),
        "coarse_center_shape": list(readout.centers.shape),
        "coarse_inertia": float(readout.inertia),
        "coarse_iterations": int(readout.iterations),
        "test_trial_count": len(readout.test_trial_ids),
        "routed_trial_count": int(np.sum(~fallback)),
        "fallback_trial_count": int(np.sum(fallback)),
        "changed_raw_prediction_count": int(np.sum(changed)),
        "fallback_raw_mismatch_count": int(
            np.sum(
                np.asarray(leaf_predictions.refined_predictions)[fallback]
                != np.asarray(leaf_predictions.coarse_predictions)[fallback]
            )
        ),
        "changed_trial_ids": np.asarray(leaf_predictions.trial_ids)[changed]
        .astype(int)
        .tolist(),
        "routed_trial_ids": np.asarray(leaf_predictions.trial_ids)[~fallback]
        .astype(int)
        .tolist(),
        "leaf_ids_by_gate_state": {
            str(key): int(value)
            for key, value in leaf_predictions.leaf_ids_by_gate_state.items()
        },
        "fit_uses_activity_labels": False,
        "routing_uses_activity_labels": False,
    }


__all__ = [
    "FrozenCoarseReadout",
    "FrozenLeafPredictions",
    "apply_frozen_leaf_overrides",
    "assert_frozen_readout_schema_is_label_free",
    "fit_frozen_coarse_readout",
    "frozen_readout_diagnostics",
]
