"""Label-free online registry for the frozen A2/E0/state CGCD route.

The module deliberately contains no neural optimiser and no activity-label
argument.  Offline supervision reaches it only as already grouped
``OldClassReference`` objects.  Once the old registry has been created, every
online operation receives only descriptor vectors and protocol provenance
(trial, subject and session identifiers).

All representation components are external and immutable.  Their combined
SHA256 is carried by every registry state, while a second hash anchors the old
semantic rows.  Successor states are append-only and form a hash chain.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Optional, Sequence

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_samples


SCHEMA = "hhr_strict_registry_state_v1"
UNKNOWN_REGISTRY_ID = -1
_EPSILON = 1.0e-12


def _positive_integer(name: str, value: int, *, allow_zero: bool = False) -> int:
    minimum = 0 if allow_zero else 1
    if isinstance(value, bool) or int(value) != value or int(value) < minimum:
        qualifier = "non-negative" if allow_zero else "positive"
        raise ValueError(f"{name} must be a {qualifier} integer, got {value!r}.")
    return int(value)


def _probability(name: str, value: float, *, allow_zero: bool = False) -> float:
    number = float(value)
    lower_ok = number >= 0.0 if allow_zero else number > 0.0
    if not math.isfinite(number) or not lower_ok or number >= 1.0:
        interval = "[0,1)" if allow_zero else "(0,1)"
        raise ValueError(f"{name} must lie in {interval}, got {value!r}.")
    return number


def _finite_non_negative(name: str, value: float) -> float:
    number = float(value)
    if not math.isfinite(number) or number < 0.0:
        raise ValueError(f"{name} must be finite and non-negative, got {value!r}.")
    return number


def _validated_sha256(name: str, value: str) -> str:
    text = str(value).lower()
    if len(text) != 64 or any(character not in "0123456789abcdef" for character in text):
        raise ValueError(f"{name} must be a lowercase hexadecimal SHA256 digest.")
    return text


def _frozen_vector(name: str, value: Any, *, unit: bool) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 1 or not len(array) or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be a non-empty finite rank-1 vector.")
    result = np.ascontiguousarray(array).copy()
    norm = float(np.linalg.norm(result))
    if unit:
        if norm <= _EPSILON:
            raise ValueError(f"{name} must have non-zero norm.")
        result /= norm
    result.setflags(write=False)
    return result


def _frozen_matrix(name: str, value: Any, *, unit_rows: bool) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 2 or not len(array) or array.shape[1] < 1:
        raise ValueError(f"{name} must have shape [N>=1,D>=1].")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains a non-finite value.")
    result = np.ascontiguousarray(array).copy()
    if unit_rows:
        norms = np.linalg.norm(result, axis=1, keepdims=True)
        if np.any(norms <= _EPSILON):
            raise ValueError(f"{name} contains a zero-norm row.")
        result /= norms
    result.setflags(write=False)
    return result


def _unit_rows(value: Any) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    if np.any(norms <= _EPSILON):
        raise ValueError("Spherical features contain a zero-norm row.")
    return array / norms


def _cosine_distances(rows: np.ndarray, centres: np.ndarray) -> np.ndarray:
    left = _unit_rows(rows)
    right = _unit_rows(centres)
    return np.clip(1.0 - left @ right.T, 0.0, 2.0)


def finite_sample_conformal_quantile(scores: Sequence[float], alpha: float) -> float:
    """Return the split-conformal ``1-alpha`` upper order statistic.

    For ``n`` calibration examples the selected one-based order is
    ``ceil((n + 1) * (1 - alpha))`` and is capped at ``n``.  The cap makes the
    usual small-sample conservative behaviour explicit (for example, alpha
    0.05 and n=10 select the maximum score).
    """

    alpha = _probability("alpha", alpha)
    values = np.asarray(scores, dtype=np.float64).reshape(-1)
    if not len(values) or not np.all(np.isfinite(values)):
        raise ValueError("Conformal scores must be a non-empty finite vector.")
    rank = min(len(values), int(math.ceil((len(values) + 1) * (1.0 - alpha))))
    return float(np.sort(values, kind="stable")[rank - 1])


def _array_digest(value: np.ndarray) -> str:
    array = np.ascontiguousarray(np.asarray(value, dtype=np.float64))
    digest = hashlib.sha256()
    digest.update(json.dumps(list(array.shape), separators=(",", ":")).encode("ascii"))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


@dataclass(frozen=True)
class LabelFreeTrial:
    """One frozen trajectory descriptor with non-label provenance only."""

    trial_id: int
    subject_id: int
    session_id: int
    descriptor: np.ndarray = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "trial_id", _positive_integer("trial_id", self.trial_id, allow_zero=True))
        object.__setattr__(self, "subject_id", _positive_integer("subject_id", self.subject_id, allow_zero=True))
        object.__setattr__(self, "session_id", _positive_integer("session_id", self.session_id))
        object.__setattr__(self, "descriptor", _frozen_vector("descriptor", self.descriptor, unit=True))

    def audit_dict(self) -> dict[str, Any]:
        return {
            "trial_id": self.trial_id,
            "subject_id": self.subject_id,
            "session_id": self.session_id,
            "descriptor_sha256": _array_digest(self.descriptor),
        }


@dataclass(frozen=True)
class OldClassReference:
    """Offline-labelled data already partitioned into one old semantic class.

    The API intentionally accepts no label vector.  ``registry_id`` is the
    immutable old semantic identifier, and the caller performs the supervised
    grouping before crossing this boundary.
    """

    registry_id: int
    fit_descriptors: np.ndarray = field(repr=False, compare=False)
    calibration_descriptors: np.ndarray = field(repr=False, compare=False)
    fit_trial_ids: tuple[int, ...] = ()
    fit_subject_ids: tuple[int, ...] = ()
    calibration_trial_ids: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        registry_id = _positive_integer("registry_id", self.registry_id, allow_zero=True)
        fit = _frozen_matrix("fit_descriptors", self.fit_descriptors, unit_rows=True)
        calibration = _frozen_matrix(
            "calibration_descriptors", self.calibration_descriptors, unit_rows=True
        )
        if fit.shape[1] != calibration.shape[1]:
            raise ValueError("Fit and calibration descriptor dimensions differ.")
        fit_trial_ids = tuple(int(value) for value in self.fit_trial_ids)
        fit_subject_ids = tuple(int(value) for value in self.fit_subject_ids)
        calibration_trial_ids = tuple(int(value) for value in self.calibration_trial_ids)
        for name, values, expected in (
            ("fit_trial_ids", fit_trial_ids, len(fit)),
            ("fit_subject_ids", fit_subject_ids, len(fit)),
            ("calibration_trial_ids", calibration_trial_ids, len(calibration)),
        ):
            if values and len(values) != expected:
                raise ValueError(f"{name} must be empty or have length {expected}.")
            if len(values) != len(set(values)) and name != "fit_subject_ids":
                raise ValueError(f"{name} contains duplicates.")
        object.__setattr__(self, "registry_id", registry_id)
        object.__setattr__(self, "fit_descriptors", fit)
        object.__setattr__(self, "calibration_descriptors", calibration)
        object.__setattr__(self, "fit_trial_ids", fit_trial_ids)
        object.__setattr__(self, "fit_subject_ids", fit_subject_ids)
        object.__setattr__(self, "calibration_trial_ids", calibration_trial_ids)


@dataclass(frozen=True)
class RegistryEntry:
    registry_id: int
    kind: str
    created_session: int
    prototype: np.ndarray = field(repr=False, compare=False)
    distance_threshold: float
    ratio_threshold: float
    support_trial_ids: tuple[int, ...]
    support_subject_ids: tuple[int, ...]

    def __post_init__(self) -> None:
        registry_id = _positive_integer("registry_id", self.registry_id, allow_zero=True)
        if self.kind not in {"old", "novel"}:
            raise ValueError("Registry entry kind must be 'old' or 'novel'.")
        created = _positive_integer(
            "created_session", self.created_session, allow_zero=self.kind == "old"
        )
        if self.kind == "old" and created != 0:
            raise ValueError("Old semantic rows must be created at session zero.")
        if self.kind == "novel" and created < 1:
            raise ValueError("Novel rows must be created during an online session.")
        distance = _finite_non_negative("distance_threshold", self.distance_threshold)
        ratio = _finite_non_negative("ratio_threshold", self.ratio_threshold)
        if ratio > 1.0 + 1.0e-12:
            raise ValueError("ratio_threshold cannot exceed one for nearest-centre routing.")
        trials = tuple(sorted(set(int(value) for value in self.support_trial_ids)))
        subjects = tuple(sorted(set(int(value) for value in self.support_subject_ids)))
        if not trials or not subjects:
            raise ValueError("Every registry entry requires trial and subject support.")
        object.__setattr__(self, "registry_id", registry_id)
        object.__setattr__(self, "created_session", created)
        object.__setattr__(self, "prototype", _frozen_vector("prototype", self.prototype, unit=True))
        object.__setattr__(self, "distance_threshold", distance)
        object.__setattr__(self, "ratio_threshold", min(1.0, ratio))
        object.__setattr__(self, "support_trial_ids", trials)
        object.__setattr__(self, "support_subject_ids", subjects)

    def audit_dict(self) -> dict[str, Any]:
        return {
            "registry_id": self.registry_id,
            "kind": self.kind,
            "created_session": self.created_session,
            "prototype_sha256": _array_digest(self.prototype),
            "distance_threshold": self.distance_threshold,
            "ratio_threshold": self.ratio_threshold,
            "support_trial_ids": list(self.support_trial_ids),
            "support_subject_ids": list(self.support_subject_ids),
        }


@dataclass(frozen=True)
class StrictRegistryConfig:
    old_distance_alpha: float = 0.05
    old_ratio_alpha: float = 0.05
    novel_distance_alpha: float = 0.05
    max_new_classes_per_session: int = 2
    minimum_cluster_trials: int = 3
    minimum_cluster_subjects: int = 2
    minimum_cluster_silhouette: float = 0.20
    bootstrap_replicates: int = 20
    minimum_bootstrap_stability: float = 0.80
    minimum_registry_separation: float = 0.10
    minimum_candidate_separation: float = 0.10
    minimum_novel_radius: float = 1.0e-4
    minimum_novel_ratio: float = 0.05
    kmeans_n_init: int = 20
    kmeans_max_iter: int = 300
    random_seed: int = 0

    def validated(self) -> "StrictRegistryConfig":
        _probability("old_distance_alpha", self.old_distance_alpha)
        _probability("old_ratio_alpha", self.old_ratio_alpha)
        _probability("novel_distance_alpha", self.novel_distance_alpha)
        for name in (
            "max_new_classes_per_session",
            "minimum_cluster_trials",
            "minimum_cluster_subjects",
            "bootstrap_replicates",
            "kmeans_n_init",
            "kmeans_max_iter",
        ):
            _positive_integer(name, getattr(self, name))
        _positive_integer("random_seed", self.random_seed, allow_zero=True)
        silhouette = float(self.minimum_cluster_silhouette)
        if not math.isfinite(silhouette) or not -1.0 <= silhouette <= 1.0:
            raise ValueError("minimum_cluster_silhouette must lie in [-1,1].")
        _probability(
            "minimum_bootstrap_stability",
            self.minimum_bootstrap_stability,
            allow_zero=True,
        )
        for name in (
            "minimum_registry_separation",
            "minimum_candidate_separation",
            "minimum_novel_radius",
            "minimum_novel_ratio",
        ):
            value = _finite_non_negative(name, getattr(self, name))
            if name == "minimum_novel_ratio" and value > 1.0:
                raise ValueError("minimum_novel_ratio cannot exceed one.")
            if "separation" in name and value > 2.0:
                raise ValueError(f"{name} cannot exceed cosine distance two.")
        return self

    def audit_dict(self) -> dict[str, Any]:
        return {
            key: getattr(self, key)
            for key in self.__dataclass_fields__
        }


def _entry_equal(left: RegistryEntry, right: RegistryEntry) -> bool:
    return (
        left.registry_id == right.registry_id
        and left.kind == right.kind
        and left.created_session == right.created_session
        and left.distance_threshold == right.distance_threshold
        and left.ratio_threshold == right.ratio_threshold
        and left.support_trial_ids == right.support_trial_ids
        and left.support_subject_ids == right.support_subject_ids
        and np.array_equal(left.prototype, right.prototype)
    )


def _entries_digest(entries: Sequence[RegistryEntry]) -> str:
    payload = [entry.audit_dict() for entry in entries]
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _state_digest(
    *,
    representation_sha256: str,
    old_anchor_sha256: str,
    previous_state_sha256: Optional[str],
    session_completed: int,
    entries: Sequence[RegistryEntry],
    unknown_buffer: Sequence[LabelFreeTrial],
    seen_trial_ids: Sequence[int],
) -> str:
    payload = {
        "schema": SCHEMA,
        "representation_sha256": representation_sha256,
        "old_anchor_sha256": old_anchor_sha256,
        "previous_state_sha256": previous_state_sha256,
        "session_completed": int(session_completed),
        "entries": [entry.audit_dict() for entry in entries],
        "unknown_buffer": [record.audit_dict() for record in unknown_buffer],
        "seen_trial_ids": [int(value) for value in seen_trial_ids],
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


@dataclass(frozen=True)
class RegistryState:
    """Immutable append-only online registry state."""

    old_class_count: int
    representation_sha256: str
    entries: tuple[RegistryEntry, ...]
    session_completed: int = 0
    unknown_buffer: tuple[LabelFreeTrial, ...] = ()
    seen_trial_ids: tuple[int, ...] = ()
    previous_state_sha256: Optional[str] = None
    old_anchor_sha256: str = ""
    state_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        old_count = _positive_integer("old_class_count", self.old_class_count)
        representation = _validated_sha256(
            "representation_sha256", self.representation_sha256
        )
        entries = tuple(self.entries)
        if len(entries) < old_count:
            raise ValueError("Registry has fewer entries than old classes.")
        expected_ids = list(range(len(entries)))
        if [entry.registry_id for entry in entries] != expected_ids:
            raise ValueError("Registry IDs must be contiguous and append-only from zero.")
        if any(entry.kind != "old" for entry in entries[:old_count]):
            raise ValueError("The registry prefix must contain only old semantic rows.")
        if any(entry.kind != "novel" for entry in entries[old_count:]):
            raise ValueError("Every row appended after the old prefix must be novel.")
        dimensions = {len(entry.prototype) for entry in entries}
        if len(dimensions) != 1:
            raise ValueError("All registry prototypes must share one descriptor dimension.")
        session = _positive_integer(
            "session_completed", self.session_completed, allow_zero=True
        )
        if any(entry.created_session > session for entry in entries):
            raise ValueError("A registry entry cannot be created in a future session.")
        buffer = tuple(sorted(self.unknown_buffer, key=lambda item: item.trial_id))
        if len({item.trial_id for item in buffer}) != len(buffer):
            raise ValueError("Unknown buffer contains duplicate trial IDs.")
        if any(len(item.descriptor) not in dimensions for item in buffer):
            raise ValueError("Unknown descriptor dimension differs from registry entries.")
        seen = tuple(sorted(set(int(value) for value in self.seen_trial_ids)))
        if not set(item.trial_id for item in buffer).issubset(seen):
            raise ValueError("Every buffered trial must be present in seen_trial_ids.")
        previous = self.previous_state_sha256
        if previous is not None:
            previous = _validated_sha256("previous_state_sha256", previous)
        computed_old_anchor = _entries_digest(entries[:old_count])
        old_anchor = self.old_anchor_sha256 or computed_old_anchor
        old_anchor = _validated_sha256("old_anchor_sha256", old_anchor)
        if old_anchor != computed_old_anchor:
            raise ValueError("old_anchor_sha256 differs from the immutable old rows.")
        object.__setattr__(self, "old_class_count", old_count)
        object.__setattr__(self, "representation_sha256", representation)
        object.__setattr__(self, "entries", entries)
        object.__setattr__(self, "session_completed", session)
        object.__setattr__(self, "unknown_buffer", buffer)
        object.__setattr__(self, "seen_trial_ids", seen)
        object.__setattr__(self, "previous_state_sha256", previous)
        object.__setattr__(self, "old_anchor_sha256", old_anchor)
        object.__setattr__(
            self,
            "state_sha256",
            _state_digest(
                representation_sha256=representation,
                old_anchor_sha256=old_anchor,
                previous_state_sha256=previous,
                session_completed=session,
                entries=entries,
                unknown_buffer=buffer,
                seen_trial_ids=seen,
            ),
        )

    @property
    def descriptor_dim(self) -> int:
        return len(self.entries[0].prototype)

    @property
    def next_registry_id(self) -> int:
        return len(self.entries)

    def audit_dict(self) -> dict[str, Any]:
        return {
            "schema": SCHEMA,
            "old_class_count": self.old_class_count,
            "representation_sha256": self.representation_sha256,
            "old_anchor_sha256": self.old_anchor_sha256,
            "previous_state_sha256": self.previous_state_sha256,
            "state_sha256": self.state_sha256,
            "session_completed": self.session_completed,
            "entries": [entry.audit_dict() for entry in self.entries],
            "unknown_buffer": [record.audit_dict() for record in self.unknown_buffer],
            "seen_trial_ids": list(self.seen_trial_ids),
            "online_activity_labels_used": False,
        }


def validate_successor(previous: RegistryState, current: RegistryState) -> None:
    """Fail closed if an online transition altered frozen state."""

    if current.session_completed != previous.session_completed + 1:
        raise RuntimeError("Registry successor must advance exactly one session.")
    if current.previous_state_sha256 != previous.state_sha256:
        raise RuntimeError("Registry hash-chain predecessor does not match.")
    if current.representation_sha256 != previous.representation_sha256:
        raise RuntimeError("Frozen representation SHA256 changed online.")
    if current.old_anchor_sha256 != previous.old_anchor_sha256:
        raise RuntimeError("Old semantic anchor changed online.")
    if len(current.entries) < len(previous.entries):
        raise RuntimeError("A successor registry cannot remove rows.")
    for index, entry in enumerate(previous.entries):
        if not _entry_equal(entry, current.entries[index]):
            raise RuntimeError(f"Registry row {index} was modified instead of preserved.")
    if not set(previous.seen_trial_ids).issubset(current.seen_trial_ids):
        raise RuntimeError("A successor registry forgot previously observed trial IDs.")


def fit_old_registry(
    references: Sequence[OldClassReference],
    *,
    representation_sha256: str,
    config: StrictRegistryConfig = StrictRegistryConfig(),
) -> RegistryState:
    """Build frozen old rows from pre-grouped offline references.

    There is intentionally no ``labels`` parameter.  Old-class supervision is
    represented only by the caller's grouping into contiguous registry IDs.
    """

    cfg = config.validated()
    grouped = tuple(sorted(references, key=lambda item: item.registry_id))
    if not grouped:
        raise ValueError("At least one old-class reference is required.")
    if [item.registry_id for item in grouped] != list(range(len(grouped))):
        raise ValueError("Old registry IDs must be contiguous from zero.")
    dimensions = {item.fit_descriptors.shape[1] for item in grouped}
    dimensions.update(item.calibration_descriptors.shape[1] for item in grouped)
    if len(dimensions) != 1:
        raise ValueError("Old class references have different descriptor dimensions.")
    prototypes = np.stack(
        [
            _unit_rows(reference.fit_descriptors).mean(axis=0)
            for reference in grouped
        ]
    )
    prototypes = _unit_rows(prototypes)
    if len(prototypes) < 2:
        raise ValueError("Conformal ratio gating requires at least two old classes.")

    entries: list[RegistryEntry] = []
    for reference in grouped:
        calibration = _unit_rows(reference.calibration_descriptors)
        distances = _cosine_distances(calibration, prototypes)
        own = distances[:, reference.registry_id]
        competitors = np.delete(distances, reference.registry_id, axis=1).min(axis=1)
        ratios = own / np.maximum(competitors, _EPSILON)
        radius = max(
            _EPSILON,
            finite_sample_conformal_quantile(own, cfg.old_distance_alpha),
        )
        ratio = min(
            1.0,
            finite_sample_conformal_quantile(ratios, cfg.old_ratio_alpha),
        )
        fit_trials = reference.fit_trial_ids or tuple(range(len(reference.fit_descriptors)))
        fit_subjects = reference.fit_subject_ids or (reference.registry_id,)
        entries.append(
            RegistryEntry(
                registry_id=reference.registry_id,
                kind="old",
                created_session=0,
                prototype=prototypes[reference.registry_id],
                distance_threshold=radius,
                ratio_threshold=ratio,
                support_trial_ids=fit_trials,
                support_subject_ids=fit_subjects,
            )
        )
    return RegistryState(
        old_class_count=len(entries),
        representation_sha256=representation_sha256,
        entries=tuple(entries),
    )


@dataclass(frozen=True)
class RoutingDecision:
    trial_id: int
    subject_id: int
    session_id: int
    registry_id: int
    accepted_known: bool
    nearest_distance: float
    second_distance: float
    distance_ratio: float
    distance_threshold: float
    ratio_threshold: float

    def audit_dict(self) -> dict[str, Any]:
        return {key: getattr(self, key) for key in self.__dataclass_fields__}


def route_label_free_trials(
    state: RegistryState,
    records: Sequence[LabelFreeTrial],
) -> tuple[RoutingDecision, ...]:
    """Route records without mutating the registry or reading activity labels."""

    values = tuple(records)
    if not values:
        return ()
    if len({record.trial_id for record in values}) != len(values):
        raise ValueError("Routing input contains duplicate trial IDs.")
    if any(len(record.descriptor) != state.descriptor_dim for record in values):
        raise ValueError("Routing descriptor dimension differs from registry state.")
    descriptors = np.stack([record.descriptor for record in values])
    prototypes = np.stack([entry.prototype for entry in state.entries])
    distances = _cosine_distances(descriptors, prototypes)
    nearest = distances.argmin(axis=1)
    ordered = np.sort(distances, axis=1)
    decisions: list[RoutingDecision] = []
    for index, (record, registry_id) in enumerate(zip(values, nearest.tolist())):
        entry = state.entries[int(registry_id)]
        first = float(distances[index, registry_id])
        second = float(ordered[index, 1]) if distances.shape[1] > 1 else 2.0
        ratio = first / max(second, _EPSILON)
        accepted = first <= entry.distance_threshold and ratio <= entry.ratio_threshold
        decisions.append(
            RoutingDecision(
                trial_id=record.trial_id,
                subject_id=record.subject_id,
                session_id=record.session_id,
                registry_id=int(registry_id) if accepted else UNKNOWN_REGISTRY_ID,
                accepted_known=bool(accepted),
                nearest_distance=first,
                second_distance=second,
                distance_ratio=ratio,
                distance_threshold=entry.distance_threshold,
                ratio_threshold=entry.ratio_threshold,
            )
        )
    return tuple(decisions)


def _spherical_kmeans(
    features: np.ndarray,
    cluster_count: int,
    *,
    random_state: int,
    n_init: int,
    max_iter: int,
) -> tuple[np.ndarray, np.ndarray]:
    values = _unit_rows(features)
    estimator = KMeans(
        n_clusters=int(cluster_count),
        random_state=int(random_state),
        n_init=int(n_init),
        max_iter=int(max_iter),
        algorithm="lloyd",
    ).fit(values)
    centres = _unit_rows(estimator.cluster_centers_)
    labels = np.argmax(values @ centres.T, axis=1).astype(np.int64)
    # Two deterministic spherical refinement passes remove the Euclidean-centre
    # ambiguity while keeping sklearn's robust multi-start initialisation.
    for _ in range(2):
        for cluster_id in range(int(cluster_count)):
            selected = values[labels == cluster_id]
            if len(selected):
                centres[cluster_id] = _unit_rows(selected.mean(axis=0, keepdims=True))[0]
        labels = np.argmax(values @ centres.T, axis=1).astype(np.int64)
    return labels, centres


def _bootstrap_stability(
    features: np.ndarray,
    reference_labels: np.ndarray,
    cluster_count: int,
    cfg: StrictRegistryConfig,
    *,
    session_id: int,
) -> tuple[float, tuple[float, ...]]:
    generator = np.random.default_rng(cfg.random_seed + 1009 * int(session_id))
    values = _unit_rows(features)
    scores: list[float] = []
    for replicate in range(cfg.bootstrap_replicates):
        indices = generator.integers(0, len(values), size=len(values))
        sampled = values[indices]
        if len(np.unique(sampled, axis=0)) < cluster_count:
            scores.append(0.0)
            continue
        try:
            _, centres = _spherical_kmeans(
                sampled,
                cluster_count,
                random_state=cfg.random_seed + 1009 * int(session_id) + replicate + 1,
                n_init=max(1, min(cfg.kmeans_n_init, 5)),
                max_iter=cfg.kmeans_max_iter,
            )
            predicted = np.argmax(values @ centres.T, axis=1)
            scores.append(float(adjusted_rand_score(reference_labels, predicted)))
        except ValueError:
            scores.append(0.0)
    return float(np.median(scores)), tuple(scores)


def _leave_one_out_radius(members: np.ndarray, alpha: float) -> float:
    values = _unit_rows(members)
    scores = []
    for index in range(len(values)):
        complement = np.delete(values, index, axis=0)
        if not len(complement):
            scores.append(2.0)
            continue
        centre = _unit_rows(complement.mean(axis=0, keepdims=True))[0]
        scores.append(float(np.clip(1.0 - values[index] @ centre, 0.0, 2.0)))
    return finite_sample_conformal_quantile(scores, alpha)


@dataclass(frozen=True)
class CandidateAudit:
    source_cluster_id: int
    member_trial_ids: tuple[int, ...]
    member_subject_ids: tuple[int, ...]
    trial_support: int
    subject_support: int
    mean_silhouette: float
    bootstrap_stability: float
    separation_to_registry: float
    separation_to_other_candidate: float
    proposed_radius: float
    accepted: bool
    rejection_reasons: tuple[str, ...]
    centre: np.ndarray = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "centre", _frozen_vector("candidate centre", self.centre, unit=True))

    def audit_dict(self) -> dict[str, Any]:
        return {
            "source_cluster_id": self.source_cluster_id,
            "member_trial_ids": list(self.member_trial_ids),
            "member_subject_ids": list(self.member_subject_ids),
            "trial_support": self.trial_support,
            "subject_support": self.subject_support,
            "mean_silhouette": self.mean_silhouette,
            "bootstrap_stability": self.bootstrap_stability,
            "separation_to_registry": self.separation_to_registry,
            "separation_to_other_candidate": self.separation_to_other_candidate,
            "proposed_radius": self.proposed_radius,
            "accepted": self.accepted,
            "rejection_reasons": list(self.rejection_reasons),
            "centre_sha256": _array_digest(self.centre),
        }


@dataclass(frozen=True)
class DiscoveryResult:
    session_id: int
    unknown_trial_ids: tuple[int, ...]
    requested_cluster_count: int
    fitted_cluster_count: int
    bootstrap_stability: float
    bootstrap_scores: tuple[float, ...]
    candidates: tuple[CandidateAudit, ...]
    activity_labels_used: bool = False

    @property
    def accepted_candidates(self) -> tuple[CandidateAudit, ...]:
        return tuple(candidate for candidate in self.candidates if candidate.accepted)

    def audit_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "unknown_trial_ids": list(self.unknown_trial_ids),
            "requested_cluster_count": self.requested_cluster_count,
            "fitted_cluster_count": self.fitted_cluster_count,
            "bootstrap_stability": self.bootstrap_stability,
            "bootstrap_scores": list(self.bootstrap_scores),
            "candidates": [candidate.audit_dict() for candidate in self.candidates],
            "activity_labels_used": False,
        }


def discover_unknown_candidates(
    state: RegistryState,
    unknown_records: Sequence[LabelFreeTrial],
    *,
    session_id: int,
    config: StrictRegistryConfig = StrictRegistryConfig(),
) -> DiscoveryResult:
    """Fit candidates only on trials already rejected by the registry."""

    cfg = config.validated()
    records = tuple(sorted(unknown_records, key=lambda item: item.trial_id))
    requested = cfg.max_new_classes_per_session
    if not records:
        return DiscoveryResult(session_id, (), requested, 0, 0.0, (), ())
    if len({record.trial_id for record in records}) != len(records):
        raise ValueError("Unknown candidate input contains duplicate trial IDs.")
    if any(len(record.descriptor) != state.descriptor_dim for record in records):
        raise ValueError("Unknown candidate descriptor dimension differs from registry.")
    if len(records) < requested * cfg.minimum_cluster_trials:
        return DiscoveryResult(
            session_id,
            tuple(record.trial_id for record in records),
            requested,
            0,
            0.0,
            (),
            (),
        )

    features = np.stack([record.descriptor for record in records])
    labels, centres = _spherical_kmeans(
        features,
        requested,
        random_state=cfg.random_seed + 97 * int(session_id),
        n_init=cfg.kmeans_n_init,
        max_iter=cfg.kmeans_max_iter,
    )
    supports = np.bincount(labels, minlength=requested)
    if np.any(supports < 2):
        silhouettes = np.full(len(records), -1.0, dtype=np.float64)
    else:
        silhouettes = silhouette_samples(_unit_rows(features), labels, metric="cosine")
    stability, bootstrap_scores = _bootstrap_stability(
        features, labels, requested, cfg, session_id=session_id
    )
    registry_centres = np.stack([entry.prototype for entry in state.entries])
    registry_separations = _cosine_distances(centres, registry_centres).min(axis=1)
    candidate_pair_distances = _cosine_distances(centres, centres)

    candidates: list[CandidateAudit] = []
    for cluster_id in range(requested):
        mask = labels == cluster_id
        members = features[mask]
        member_records = [record for record, selected in zip(records, mask) if selected]
        support = len(member_records)
        subjects = tuple(sorted({record.subject_id for record in member_records}))
        other_distances = np.delete(candidate_pair_distances[cluster_id], cluster_id)
        other_separation = float(other_distances.min()) if len(other_distances) else 2.0
        silhouette = float(np.mean(silhouettes[mask])) if support else -1.0
        radius = (
            max(cfg.minimum_novel_radius, _leave_one_out_radius(members, cfg.novel_distance_alpha))
            if support >= 2
            else 2.0
        )
        reasons: list[str] = []
        if support < cfg.minimum_cluster_trials:
            reasons.append("minimum_cluster_trials_not_met")
        if len(subjects) < cfg.minimum_cluster_subjects:
            reasons.append("minimum_cluster_subjects_not_met")
        if silhouette < cfg.minimum_cluster_silhouette:
            reasons.append("minimum_cluster_silhouette_not_met")
        if stability < cfg.minimum_bootstrap_stability:
            reasons.append("minimum_bootstrap_stability_not_met")
        if float(registry_separations[cluster_id]) < cfg.minimum_registry_separation:
            reasons.append("minimum_registry_separation_not_met")
        if other_separation < cfg.minimum_candidate_separation:
            reasons.append("minimum_candidate_separation_not_met")
        candidates.append(
            CandidateAudit(
                source_cluster_id=cluster_id,
                member_trial_ids=tuple(sorted(record.trial_id for record in member_records)),
                member_subject_ids=subjects,
                trial_support=support,
                subject_support=len(subjects),
                mean_silhouette=silhouette,
                bootstrap_stability=stability,
                separation_to_registry=float(registry_separations[cluster_id]),
                separation_to_other_candidate=other_separation,
                proposed_radius=radius,
                accepted=not reasons,
                rejection_reasons=tuple(reasons),
                centre=centres[cluster_id],
            )
        )
    # KMeans cluster numbers have no semantics.  Trial-based ordering makes
    # appended IDs deterministic without consulting activity identity.
    candidates.sort(key=lambda item: (min(item.member_trial_ids), item.source_cluster_id))
    return DiscoveryResult(
        session_id=session_id,
        unknown_trial_ids=tuple(record.trial_id for record in records),
        requested_cluster_count=requested,
        fitted_cluster_count=requested,
        bootstrap_stability=stability,
        bootstrap_scores=bootstrap_scores,
        candidates=tuple(candidates),
    )


@dataclass(frozen=True)
class SessionUpdate:
    state: RegistryState
    initial_routing: tuple[RoutingDecision, ...]
    discovery: DiscoveryResult
    post_registration_routing: tuple[RoutingDecision, ...]
    registered_ids: tuple[int, ...]
    accepted_trial_ids: tuple[int, ...]
    unresolved_trial_ids: tuple[int, ...]
    activity_labels_used: bool = False

    def audit_dict(self) -> dict[str, Any]:
        return {
            "state_sha256": self.state.state_sha256,
            "previous_state_sha256": self.state.previous_state_sha256,
            "initial_routing": [item.audit_dict() for item in self.initial_routing],
            "discovery": self.discovery.audit_dict(),
            "post_registration_routing": [
                item.audit_dict() for item in self.post_registration_routing
            ],
            "registered_ids": list(self.registered_ids),
            "accepted_trial_ids": list(self.accepted_trial_ids),
            "unresolved_trial_ids": list(self.unresolved_trial_ids),
            "activity_labels_used": False,
        }


def _build_novel_entries(
    state: RegistryState,
    discovery: DiscoveryResult,
    records_by_id: Mapping[int, LabelFreeTrial],
    cfg: StrictRegistryConfig,
) -> tuple[RegistryEntry, ...]:
    accepted = discovery.accepted_candidates
    if not accepted:
        return ()
    accepted_centres = np.stack([item.centre for item in accepted])
    existing_centres = np.stack([entry.prototype for entry in state.entries])
    all_centres = np.concatenate((existing_centres, accepted_centres), axis=0)
    new_entries = []
    for offset, candidate in enumerate(accepted):
        new_id = state.next_registry_id + offset
        members = np.stack(
            [records_by_id[trial_id].descriptor for trial_id in candidate.member_trial_ids]
        )
        distances = _cosine_distances(members, all_centres)
        own_column = len(existing_centres) + offset
        own = distances[:, own_column]
        competitors = np.delete(distances, own_column, axis=1).min(axis=1)
        ratios = own / np.maximum(competitors, _EPSILON)
        ratio_threshold = max(
            cfg.minimum_novel_ratio,
            finite_sample_conformal_quantile(ratios, cfg.novel_distance_alpha),
        )
        new_entries.append(
            RegistryEntry(
                registry_id=new_id,
                kind="novel",
                created_session=discovery.session_id,
                prototype=candidate.centre,
                distance_threshold=candidate.proposed_radius,
                ratio_threshold=min(1.0, ratio_threshold),
                support_trial_ids=candidate.member_trial_ids,
                support_subject_ids=candidate.member_subject_ids,
            )
        )
    return tuple(new_entries)


def advance_registry_session(
    state: RegistryState,
    incoming_records: Sequence[LabelFreeTrial],
    *,
    config: StrictRegistryConfig = StrictRegistryConfig(),
) -> SessionUpdate:
    """Route one mixed unlabelled session, discover unknowns and append rows."""

    cfg = config.validated()
    incoming = tuple(sorted(incoming_records, key=lambda item: item.trial_id))
    session_id = state.session_completed + 1
    if not incoming:
        raise ValueError("An online session must contain at least one incoming trial.")
    if any(record.session_id != session_id for record in incoming):
        raise ValueError(
            f"Every incoming record must have session_id={session_id}."
        )
    incoming_ids = {record.trial_id for record in incoming}
    if len(incoming_ids) != len(incoming):
        raise ValueError("Incoming session contains duplicate trial IDs.")
    overlap = incoming_ids & set(state.seen_trial_ids)
    if overlap:
        raise ValueError(f"Incoming session reuses observed trial IDs: {sorted(overlap)}.")

    initial = route_label_free_trials(state, incoming)
    incoming_by_id = {record.trial_id: record for record in incoming}
    rejected = [
        incoming_by_id[decision.trial_id]
        for decision in initial
        if not decision.accepted_known
    ]
    pool_by_id = {record.trial_id: record for record in state.unknown_buffer}
    pool_by_id.update({record.trial_id: record for record in rejected})
    pool = tuple(sorted(pool_by_id.values(), key=lambda item: item.trial_id))
    discovery = discover_unknown_candidates(
        state, pool, session_id=session_id, config=cfg
    )
    appended = _build_novel_entries(state, discovery, pool_by_id, cfg)
    provisional = RegistryState(
        old_class_count=state.old_class_count,
        representation_sha256=state.representation_sha256,
        entries=state.entries + appended,
        session_completed=session_id,
        unknown_buffer=pool,
        seen_trial_ids=tuple(sorted(set(state.seen_trial_ids) | incoming_ids)),
        previous_state_sha256=state.state_sha256,
        old_anchor_sha256=state.old_anchor_sha256,
    )
    post = route_label_free_trials(provisional, pool)
    unresolved_ids = {
        decision.trial_id for decision in post if not decision.accepted_known
    }
    unresolved = tuple(record for record in pool if record.trial_id in unresolved_ids)
    successor = RegistryState(
        old_class_count=state.old_class_count,
        representation_sha256=state.representation_sha256,
        entries=provisional.entries,
        session_completed=session_id,
        unknown_buffer=unresolved,
        seen_trial_ids=provisional.seen_trial_ids,
        previous_state_sha256=state.state_sha256,
        old_anchor_sha256=state.old_anchor_sha256,
    )
    validate_successor(state, successor)
    initially_accepted = {
        decision.trial_id for decision in initial if decision.accepted_known
    }
    newly_accepted = {
        decision.trial_id for decision in post if decision.accepted_known
    }
    return SessionUpdate(
        state=successor,
        initial_routing=initial,
        discovery=discovery,
        post_registration_routing=post,
        registered_ids=tuple(entry.registry_id for entry in appended),
        accepted_trial_ids=tuple(sorted(initially_accepted | newly_accepted)),
        unresolved_trial_ids=tuple(sorted(unresolved_ids)),
    )


def serialize_registry_state(state: RegistryState) -> dict[str, Any]:
    """Return an auditable JSON-safe state description.

    Numeric prototype arrays are represented by values as well as hashes so a
    runner can persist and reconstruct the state.  Serialization itself does
    not expose or accept activity labels.
    """

    payload = state.audit_dict()
    for output, entry in zip(payload["entries"], state.entries):
        output["prototype"] = entry.prototype.tolist()
    for output, record in zip(payload["unknown_buffer"], state.unknown_buffer):
        output["descriptor"] = record.descriptor.tolist()
    return payload


def deserialize_registry_state(payload: Mapping[str, Any]) -> RegistryState:
    if payload.get("schema") != SCHEMA:
        raise ValueError(f"Unexpected registry schema {payload.get('schema')!r}.")
    entries = tuple(
        RegistryEntry(
            registry_id=item["registry_id"],
            kind=item["kind"],
            created_session=item["created_session"],
            prototype=item["prototype"],
            distance_threshold=item["distance_threshold"],
            ratio_threshold=item["ratio_threshold"],
            support_trial_ids=tuple(item["support_trial_ids"]),
            support_subject_ids=tuple(item["support_subject_ids"]),
        )
        for item in payload["entries"]
    )
    buffer = tuple(
        LabelFreeTrial(
            trial_id=item["trial_id"],
            subject_id=item["subject_id"],
            session_id=item["session_id"],
            descriptor=item["descriptor"],
        )
        for item in payload["unknown_buffer"]
    )
    state = RegistryState(
        old_class_count=payload["old_class_count"],
        representation_sha256=payload["representation_sha256"],
        entries=entries,
        session_completed=payload["session_completed"],
        unknown_buffer=buffer,
        seen_trial_ids=tuple(payload["seen_trial_ids"]),
        previous_state_sha256=payload.get("previous_state_sha256"),
        old_anchor_sha256=payload["old_anchor_sha256"],
    )
    if state.state_sha256 != payload.get("state_sha256"):
        raise RuntimeError("Serialized registry state SHA256 does not verify.")
    return state


__all__ = [
    "CandidateAudit",
    "DiscoveryResult",
    "LabelFreeTrial",
    "OldClassReference",
    "RegistryEntry",
    "RegistryState",
    "RoutingDecision",
    "SCHEMA",
    "SessionUpdate",
    "StrictRegistryConfig",
    "UNKNOWN_REGISTRY_ID",
    "advance_registry_session",
    "deserialize_registry_state",
    "discover_unknown_candidates",
    "finite_sample_conformal_quantile",
    "fit_old_registry",
    "route_label_free_trials",
    "serialize_registry_state",
    "validate_successor",
]
