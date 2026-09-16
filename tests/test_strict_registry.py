from __future__ import annotations

import dataclasses
import hashlib
import inspect
import json
import unittest

import numpy as np

from experiments.motion_primitive.strict_metrics import strict_three_layer_metrics
from experiments.motion_primitive.strict_registry import (
    UNKNOWN_REGISTRY_ID,
    LabelFreeTrial,
    OldClassReference,
    RegistryState,
    StrictRegistryConfig,
    advance_registry_session,
    deserialize_registry_state,
    finite_sample_conformal_quantile,
    fit_old_registry,
    route_label_free_trials,
    serialize_registry_state,
    validate_successor,
)


REPRESENTATION_SHA256 = hashlib.sha256(
    b"frozen-resnet1d-a2-e0-k32-state-descriptor"
).hexdigest()


def _old_state(*, config: StrictRegistryConfig | None = None) -> RegistryState:
    references = (
        OldClassReference(
            registry_id=0,
            fit_descriptors=np.asarray(
                [[1.0, 0.02, 0.0, 0.0], [1.0, -0.02, 0.0, 0.0], [1.0, 0.0, 0.02, 0.0]]
            ),
            calibration_descriptors=np.asarray(
                [[1.0, 0.01, 0.0, 0.0], [1.0, -0.01, 0.0, 0.0], [1.0, 0.0, 0.01, 0.0]]
            ),
            fit_trial_ids=(100, 101, 102),
            fit_subject_ids=(1, 2, 3),
            calibration_trial_ids=(110, 111, 112),
        ),
        OldClassReference(
            registry_id=1,
            fit_descriptors=np.asarray(
                [[0.02, 1.0, 0.0, 0.0], [-0.02, 1.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.02]]
            ),
            calibration_descriptors=np.asarray(
                [[0.01, 1.0, 0.0, 0.0], [-0.01, 1.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.01]]
            ),
            fit_trial_ids=(200, 201, 202),
            fit_subject_ids=(1, 2, 3),
            calibration_trial_ids=(210, 211, 212),
        ),
    )
    return fit_old_registry(
        references,
        representation_sha256=REPRESENTATION_SHA256,
        config=config or StrictRegistryConfig(),
    )


def _config(**updates: object) -> StrictRegistryConfig:
    values = {
        "max_new_classes_per_session": 2,
        "minimum_cluster_trials": 4,
        "minimum_cluster_subjects": 2,
        "minimum_cluster_silhouette": 0.50,
        "bootstrap_replicates": 11,
        "minimum_bootstrap_stability": 0.70,
        "minimum_registry_separation": 0.20,
        "minimum_candidate_separation": 0.20,
        "kmeans_n_init": 20,
        "random_seed": 23,
    }
    values.update(updates)
    return StrictRegistryConfig(**values)


def _trial(
    trial_id: int,
    subject_id: int,
    session_id: int,
    descriptor: list[float],
) -> LabelFreeTrial:
    return LabelFreeTrial(
        trial_id=trial_id,
        subject_id=subject_id,
        session_id=session_id,
        descriptor=np.asarray(descriptor, dtype=np.float64),
    )


def _two_novel_clusters(
    *,
    session_id: int,
    first_trial_id: int = 0,
    subjects: tuple[int, ...] = (10, 11),
) -> tuple[LabelFreeTrial, ...]:
    # Each cluster contains small deterministic perturbations around an axis.
    # Interleaved trial IDs make the registration order independent of the
    # arbitrary KMeans cluster numbering.
    rows = (
        [0.010, 0.000, 1.000, 0.000],
        [0.000, 0.010, 0.000, 1.000],
        [-0.010, 0.000, 1.000, 0.000],
        [0.000, -0.010, 0.000, 1.000],
        [0.000, 0.010, 1.000, 0.000],
        [0.010, 0.000, 0.000, 1.000],
        [0.000, -0.010, 1.000, 0.000],
        [-0.010, 0.000, 0.000, 1.000],
    )
    return tuple(
        _trial(
            first_trial_id + offset,
            subjects[(offset // 2) % len(subjects)],
            session_id,
            row,
        )
        for offset, row in enumerate(rows)
    )


class StrictRegistryProtocolTests(unittest.TestCase):
    def test_online_trial_and_api_have_no_activity_label_argument(self) -> None:
        fields = {item.name for item in dataclasses.fields(LabelFreeTrial)}
        self.assertEqual(fields, {"trial_id", "subject_id", "session_id", "descriptor"})
        for callable_object in (route_label_free_trials, advance_registry_session):
            parameter_names = set(inspect.signature(callable_object).parameters)
            self.assertFalse(
                {"label", "labels", "target", "targets", "activity_label"}
                & parameter_names
            )

    def test_descriptors_and_registry_prototypes_are_read_only_copies(self) -> None:
        source = np.asarray([1.0, 2.0, 3.0, 4.0])
        trial = _trial(1, 1, 1, source.tolist())
        source[:] = 0.0
        self.assertGreater(float(np.linalg.norm(trial.descriptor)), 0.99)
        self.assertFalse(trial.descriptor.flags.writeable)
        with self.assertRaises(ValueError):
            trial.descriptor[0] = 7.0

        state = _old_state()
        self.assertTrue(all(not entry.prototype.flags.writeable for entry in state.entries))
        with self.assertRaises(ValueError):
            state.entries[0].prototype[0] = 0.0

    def test_finite_sample_conformal_quantile_and_known_unknown_routing(self) -> None:
        # ceil((3+1)*(1-.25)) == 3, so the maximum is selected.
        self.assertEqual(finite_sample_conformal_quantile([0.1, 0.3, 0.2], 0.25), 0.3)
        state = _old_state()
        known = _trial(1, 7, 1, state.entries[0].prototype.tolist())
        unknown = _trial(2, 8, 1, [0.0, 0.0, 1.0, 0.0])
        decisions = route_label_free_trials(state, (known, unknown))
        self.assertTrue(decisions[0].accepted_known)
        self.assertEqual(decisions[0].registry_id, 0)
        self.assertFalse(decisions[1].accepted_known)
        self.assertEqual(decisions[1].registry_id, UNKNOWN_REGISTRY_ID)

    def test_only_rejected_trials_enter_discovery_and_two_clusters_append(self) -> None:
        state = _old_state()
        old_rows = tuple(entry.prototype.tobytes() for entry in state.entries)
        known = _trial(90, 20, 1, state.entries[0].prototype.tolist())
        update = advance_registry_session(
            state,
            (known,) + _two_novel_clusters(session_id=1, first_trial_id=10),
            config=_config(),
        )
        self.assertNotIn(known.trial_id, update.discovery.unknown_trial_ids)
        self.assertEqual(update.registered_ids, (2, 3))
        self.assertEqual([entry.registry_id for entry in update.state.entries], [0, 1, 2, 3])
        self.assertEqual([entry.kind for entry in update.state.entries], ["old", "old", "novel", "novel"])
        self.assertTrue(all(candidate.accepted for candidate in update.discovery.candidates))
        self.assertEqual(update.unresolved_trial_ids, ())
        self.assertEqual(tuple(entry.prototype.tobytes() for entry in update.state.entries[:2]), old_rows)

    def test_old_rows_thresholds_ids_and_anchor_are_byte_stable_across_session(self) -> None:
        state = _old_state()
        before = tuple(
            (
                entry.registry_id,
                entry.prototype.tobytes(),
                np.float64(entry.distance_threshold).tobytes(),
                np.float64(entry.ratio_threshold).tobytes(),
            )
            for entry in state.entries
        )
        update = advance_registry_session(
            state,
            _two_novel_clusters(session_id=1),
            config=_config(),
        )
        after = tuple(
            (
                entry.registry_id,
                entry.prototype.tobytes(),
                np.float64(entry.distance_threshold).tobytes(),
                np.float64(entry.ratio_threshold).tobytes(),
            )
            for entry in update.state.entries[: state.old_class_count]
        )
        self.assertEqual(before, after)
        self.assertEqual(state.old_anchor_sha256, update.state.old_anchor_sha256)
        self.assertEqual(update.state.previous_state_sha256, state.state_sha256)
        validate_successor(state, update.state)

    def test_single_subject_candidates_are_rejected_and_carried_to_next_session(self) -> None:
        state = _old_state()
        first = advance_registry_session(
            state,
            _two_novel_clusters(session_id=1, subjects=(30,)),
            config=_config(),
        )
        self.assertEqual(first.registered_ids, ())
        self.assertEqual(len(first.state.unknown_buffer), 8)
        self.assertTrue(
            all(
                "minimum_cluster_subjects_not_met" in candidate.rejection_reasons
                for candidate in first.discovery.candidates
            )
        )

        # Two observations per cluster from a second subject are enough only
        # because the first session's unknown buffer is retained.
        second_subject_rows = (
            _trial(100, 31, 2, [0.005, 0.0, 1.0, 0.0]),
            _trial(101, 31, 2, [0.0, 0.005, 0.0, 1.0]),
            _trial(102, 31, 2, [-0.005, 0.0, 1.0, 0.0]),
            _trial(103, 31, 2, [0.0, -0.005, 0.0, 1.0]),
        )
        second = advance_registry_session(first.state, second_subject_rows, config=_config())
        self.assertEqual(second.registered_ids, (2, 3))
        self.assertEqual(second.state.unknown_buffer, ())
        self.assertTrue(set(first.unresolved_trial_ids).issubset(second.accepted_trial_ids))

    def test_insufficient_support_keeps_buffer_and_duplicate_trial_is_rejected(self) -> None:
        state = _old_state()
        sparse = (
            _trial(1, 1, 1, [0.0, 0.0, 1.0, 0.0]),
            _trial(2, 2, 1, [0.0, 0.0, 0.0, 1.0]),
        )
        first = advance_registry_session(state, sparse, config=_config())
        self.assertEqual(first.discovery.fitted_cluster_count, 0)
        self.assertEqual(first.registered_ids, ())
        self.assertEqual(tuple(item.trial_id for item in first.state.unknown_buffer), (1, 2))
        with self.assertRaisesRegex(ValueError, "reuses observed trial IDs"):
            advance_registry_session(
                first.state,
                (_trial(2, 3, 2, [0.0, 0.0, 1.0, 0.0]),),
                config=_config(),
            )

    def test_hash_chain_and_json_serialization_round_trip(self) -> None:
        state = _old_state()
        update = advance_registry_session(
            state,
            _two_novel_clusters(session_id=1),
            config=_config(),
        )
        payload = serialize_registry_state(update.state)
        # Prove that this is an actual JSON-safe persistence payload.
        decoded = json.loads(json.dumps(payload, sort_keys=True))
        restored = deserialize_registry_state(decoded)
        self.assertEqual(restored.state_sha256, update.state.state_sha256)
        self.assertEqual(restored.previous_state_sha256, state.state_sha256)
        self.assertEqual(restored.old_anchor_sha256, state.old_anchor_sha256)
        self.assertTrue(
            all(
                np.array_equal(left.prototype, right.prototype)
                for left, right in zip(restored.entries, update.state.entries)
            )
        )

        tampered = json.loads(json.dumps(payload))
        tampered["entries"][0]["distance_threshold"] += 0.01
        with self.assertRaises((RuntimeError, ValueError)):
            deserialize_registry_state(tampered)


class StrictMetricLayerTests(unittest.TestCase):
    def test_unknown_column_and_novel_only_alignment(self) -> None:
        truth = np.asarray([0, 1, 2, 2, 3, 3], dtype=np.int64)
        raw = np.asarray([0, UNKNOWN_REGISTRY_ID, 3, 3, 2, 2], dtype=np.int64)
        result = strict_three_layer_metrics(
            truth,
            raw,
            class_count=4,
            old_class_count=2,
            seen_class_count_before_session=2,
            registered_class_ids=(0, 1, 2, 3),
        )
        primary = result["layers"]["old_fixed_novel_hungarian"]
        self.assertEqual(result["primary_layer"], "old_fixed_novel_hungarian")
        self.assertEqual(primary["unknown_prediction_count"], 1)
        self.assertEqual(primary["confusion_columns"][-1], "unknown")
        self.assertEqual(primary["new_accuracy"], 1.0)
        self.assertEqual(primary["old_accuracy"], 0.5)
        self.assertTrue(primary["alignment_uses_test_labels"])
        self.assertFalse(result["layers"]["direct_registry"]["alignment_uses_test_labels"])

    def test_constrained_layer_cannot_hide_old_swap_but_global_layer_can(self) -> None:
        truth = np.asarray([0, 0, 1, 1, 2, 2, 3, 3], dtype=np.int64)
        raw = np.asarray([1, 1, 0, 0, 3, 3, 2, 2], dtype=np.int64)
        result = strict_three_layer_metrics(
            truth,
            raw,
            class_count=4,
            old_class_count=2,
            seen_class_count_before_session=2,
        )
        constrained = result["layers"]["old_fixed_novel_hungarian"]
        optimistic = result["layers"]["global_hungarian_upper_bound"]
        self.assertEqual(constrained["old_accuracy"], 0.0)
        self.assertEqual(constrained["new_accuracy"], 1.0)
        self.assertEqual(constrained["h_score"], 0.0)
        self.assertEqual(optimistic["all_accuracy"], 1.0)

    def test_old_novel_cross_partition_errors_cannot_be_repaired(self) -> None:
        truth = np.asarray([0, 0, 2, 2], dtype=np.int64)
        raw = np.asarray([2, 2, 0, 0], dtype=np.int64)
        result = strict_three_layer_metrics(
            truth,
            raw,
            class_count=4,
            old_class_count=2,
            seen_class_count_before_session=2,
        )
        constrained = result["layers"]["old_fixed_novel_hungarian"]
        optimistic = result["layers"]["global_hungarian_upper_bound"]
        self.assertEqual(constrained["all_accuracy"], 0.0)
        self.assertEqual(optimistic["all_accuracy"], 1.0)

    def test_scoring_does_not_mutate_arrays_or_registry_state(self) -> None:
        truth = np.asarray([0, 1, 2, 3], dtype=np.int64)
        raw = np.asarray([0, 1, 3, UNKNOWN_REGISTRY_ID], dtype=np.int64)
        truth_before = truth.copy()
        raw_before = raw.copy()
        state = _old_state()
        state_hash = state.state_sha256
        entry_bytes = tuple(entry.prototype.tobytes() for entry in state.entries)
        strict_three_layer_metrics(
            truth,
            raw,
            class_count=4,
            old_class_count=2,
            seen_class_count_before_session=2,
            registered_class_ids=(0, 1, 2, 3),
        )
        self.assertTrue(np.array_equal(truth, truth_before))
        self.assertTrue(np.array_equal(raw, raw_before))
        self.assertEqual(state.state_sha256, state_hash)
        self.assertEqual(tuple(entry.prototype.tobytes() for entry in state.entries), entry_bytes)


if __name__ == "__main__":
    unittest.main()
