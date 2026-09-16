from __future__ import annotations

import argparse
from dataclasses import fields
from pathlib import Path
import tempfile
from unittest import mock
import unittest

import numpy as np

import experiments.motion_primitive.online_secondary_codebook as secondary_module
import experiments.motion_primitive.run_online_secondary_codebook as runner_module
from experiments.motion_primitive.online_secondary_codebook import (
    UnlabelledTokenOccurrence,
    assign_secondary_codebook,
    duration_weighted_residual,
    fit_secondary_codebook,
    global_hungarian_metrics,
    gravity_from_token_partitions,
    monte_carlo_upper_tail_summary,
    posthoc_secondary_metrics,
    secondary_fit_diagnostics,
    select_dominant_token,
    shuffle_gravity_within_subject_split,
    stratified_paired_accuracy_bootstrap,
    validate_session_manifest,
)
from experiments.motion_primitive.run_online_secondary_codebook import (
    LabelFreeSourceSignalRepository,
    SegmentArtifacts,
    _gravity_shuffle_null_predictions,
    _positive_int,
    _validate_registered_session_protocol,
    build_trial_token_durations,
)


def make_occurrence(
    trial_id: int,
    residual: tuple[float, float],
    gravity: tuple[float, float, float],
    token: int = 7,
) -> UnlabelledTokenOccurrence:
    return UnlabelledTokenOccurrence(
        trial_global_id=int(trial_id),
        coarse_token=int(token),
        token_fraction=0.75,
        duration_samples=30,
        mean_quantization_residual=np.asarray(residual, dtype=np.float32),
        gravity_direction=np.asarray(gravity, dtype=np.float32),
    )


class OnlineSecondaryCodebookTests(unittest.TestCase):
    def test_occurrence_schema_cannot_carry_activity_identity(self) -> None:
        names = {field.name for field in fields(UnlabelledTokenOccurrence)}
        self.assertEqual(
            names,
            {
                "trial_global_id",
                "coarse_token",
                "token_fraction",
                "duration_samples",
                "mean_quantization_residual",
                "gravity_direction",
            },
        )
        self.assertFalse(any("label" in name or "name" in name for name in names))

    def test_dominant_token_uses_partition_duration_support_and_token_tie_break(self) -> None:
        durations = {}
        # Tokens 3 and 5 each dominate ten trials.  The deterministic tie rule
        # must select token 3, irrespective of mapping insertion order.
        for trial_id in range(10):
            durations[trial_id] = {9: 49.0, 5: 51.0}
        for trial_id in range(10, 20):
            durations[trial_id] = {3: 50.0, 8: 50.0}
        result = select_dominant_token(
            durations, minimum_fraction=0.50, minimum_support=10
        )
        self.assertEqual(result["selected_token"], 3)
        self.assertEqual(result["selected_support"], 10)
        self.assertEqual(result["dominant_trial_ids"], list(range(10, 20)))
        self.assertEqual(result["support_by_token"], {"3": 10, "5": 10, "8": 10})
        self.assertIs(result["selection_uses_labels"], False)

    def test_residual_is_weighted_by_nonoverlapping_partition_duration(self) -> None:
        embeddings = np.asarray([[2.0, 0.0], [0.0, 4.0]], dtype=np.float32)
        center = np.asarray([1.0, 1.0], dtype=np.float32)
        result = duration_weighted_residual(
            embeddings,
            center,
            duration_samples=np.asarray([1, 3], dtype=np.int64),
        )
        # (([1,-1] * 1) + ([-1,3] * 3)) / 4 = [-0.5, 2.0].
        np.testing.assert_allclose(result, np.asarray([-0.5, 2.0]), atol=1e-7)

    def test_gravity_aggregates_contiguous_token_runs_by_run_duration(self) -> None:
        sensor = np.zeros((6, 12), dtype=np.float32)
        # Two adjacent selected partitions form one six-sample run whose
        # robust direction is +x.  A later two-sample run points +y.
        sensor[0, 0:6] = 2.0
        sensor[2, 3] = 1000.0  # Median makes this impulse irrelevant.
        sensor[1, 8:10] = 3.0
        result = gravity_from_token_partitions(
            sensor=sensor,
            partition_starts=np.asarray([0, 3, 6, 8, 10]),
            partition_ends=np.asarray([3, 6, 8, 10, 12]),
            partition_tokens=np.asarray([7, 7, 2, 7, 2]),
            selected_token=7,
        )
        expected = np.asarray([6.0, 2.0, 0.0], dtype=np.float64)
        expected /= np.linalg.norm(expected)
        np.testing.assert_allclose(result, expected, atol=1e-7)

    def test_session_manifest_rejects_all_train_reuse_and_test_leakage(self) -> None:
        clean = validate_session_manifest([1, 2], [3, 4], [5, 6])
        self.assertEqual(clean["cumulative_train_trial_ids"], [1, 2, 3, 4])
        self.assertEqual(clean["cumulative_train_test_overlap"], 0)
        with self.assertRaisesRegex(RuntimeError, "reused"):
            validate_session_manifest([1, 2], [2, 3], [4, 5])
        with self.assertRaisesRegex(RuntimeError, "leakage"):
            validate_session_manifest([1, 2], [3, 4], [4, 5])

    def test_registered_session_rejects_count_order_and_future_class_drift(self) -> None:
        audit = {
            "session_1_incremental_train_count": 22,
            "session_2_incremental_train_count": 26,
            "session_2_cumulative_train_count": 48,
            "session_2_test_count": 52,
            "cumulative_train_trial_ids": list(range(48)),
            "session_2_test_trial_ids": list(range(48, 100)),
        }
        labels = {trial_id: trial_id % 10 for trial_id in range(100)}
        subjects = {
            trial_id: 4 if trial_id < 50 else 5 for trial_id in range(100)
        }
        clean = _validate_registered_session_protocol(
            dict(audit), list(range(6, 12)), labels, subjects
        )
        self.assertIs(clean["registered_protocol_verified"], True)
        self.assertEqual(clean["feature_trial_count"], 100)
        self.assertEqual(clean["future_feature_trial_count"], 0)
        with self.assertRaisesRegex(RuntimeError, "novel-class order"):
            _validate_registered_session_protocol(
                dict(audit), [7, 6, 8, 9, 10, 11], labels, subjects
            )
        changed_count = dict(audit)
        changed_count["session_2_test_count"] = 51
        with self.assertRaisesRegex(RuntimeError, "trial counts"):
            _validate_registered_session_protocol(
                changed_count, list(range(6, 12)), labels, subjects
            )
        future = dict(labels)
        future[0] = 10
        with self.assertRaisesRegex(RuntimeError, "Future or missing activity"):
            _validate_registered_session_protocol(
                dict(audit), list(range(6, 12)), future, subjects
            )
        wrong_subject = dict(subjects)
        wrong_subject[0] = 6
        with self.assertRaisesRegex(RuntimeError, "feature subjects drifted"):
            _validate_registered_session_protocol(
                dict(audit), list(range(6, 12)), labels, wrong_subject
            )

    def test_partition_duration_requires_complete_contiguous_visible_span(self) -> None:
        def artifacts(starts: list[int], ends: list[int]) -> SegmentArtifacts:
            count = len(starts)
            return SegmentArtifacts(
                split_role=np.ones(count, dtype=np.int8),
                trial_ids=np.full(count, 42, dtype=np.int64),
                starts=np.asarray(starts, dtype=np.int64),
                ends=np.asarray(ends, dtype=np.int64),
                tokens=np.arange(count, dtype=np.int64),
                embeddings=np.zeros((count, 2), dtype=np.float32),
                centers=np.zeros((max(count, 1), 2), dtype=np.float32),
            )

        clean = build_trial_token_durations(
            artifacts([0, 2], [2, 6]), [42], visible_end_by_trial={42: 6}
        )
        self.assertEqual(clean, {42: {0: 2.0, 1: 4.0}})
        for starts, ends, message in (
            ([1, 2], [2, 6], "incomplete"),
            ([0, 3], [2, 6], "overlap or gap"),
            ([0, 2], [2, 5], "visible end"),
        ):
            with self.subTest(starts=starts, ends=ends):
                with self.assertRaisesRegex(RuntimeError, message):
                    build_trial_token_durations(
                        artifacts(starts, ends),
                        [42],
                        visible_end_by_trial={42: 6},
                    )

    def test_label_free_sensor_accessor_never_calls_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "signals_without_metadata.npz"
            windows = np.stack(
                [
                    np.ones((6, 4), dtype=np.float32),
                    np.full((6, 4), 3.0, dtype=np.float32),
                ]
            )
            # Deliberately omit labels, activity_names, subjects and paths.
            np.savez(
                path,
                windows=windows,
                trial_global_ids=np.asarray([7, 7], dtype=np.int64),
                window_start_indices=np.asarray([0, 2], dtype=np.int64),
                mean=np.zeros((1, 6, 1), dtype=np.float32),
                std=np.ones((1, 6, 1), dtype=np.float32),
            )
            repository = LabelFreeSourceSignalRepository(path)
            self.assertFalse(hasattr(repository, "metadata"))
            self.assertFalse(
                {
                    "labels",
                    "activity_names_raw",
                    "subject_ids",
                    "file_paths",
                }
                & set(vars(repository))
            )
            result = repository.sensor(7)
            expected_row = np.asarray([1, 1, 2, 2, 3, 3], dtype=np.float32)
            np.testing.assert_array_equal(
                result, np.tile(expected_row[None, :], (6, 1))
            )
            audit = repository.source_audit([7])
            self.assertIs(audit["label_name_subject_or_path_fields_loaded"], False)
            self.assertEqual(
                audit["loaded_npz_fields"],
                [
                    "mean",
                    "std",
                    "trial_global_ids",
                    "window_start_indices",
                    "windows",
                ],
            )

    def test_kmedoids_fit_and_assignment_are_deterministic(self) -> None:
        occurrences = [
            make_occurrence(10, (0.00, 0.00), (1.0, 0.0, 0.0)),
            make_occurrence(11, (0.05, 0.00), (1.0, 0.0, 0.0)),
            make_occurrence(12, (0.00, 0.05), (1.0, 0.0, 0.0)),
            make_occurrence(20, (4.00, 4.00), (0.0, 1.0, 0.0)),
            make_occurrence(21, (4.05, 4.00), (0.0, 1.0, 0.0)),
            make_occurrence(22, (4.00, 4.05), (0.0, 1.0, 0.0)),
        ]
        first = fit_secondary_codebook(
            occurrences, arm="U3_joint", restarts=50, seed=500
        )
        second = fit_secondary_codebook(
            occurrences, arm="U3_joint", restarts=50, seed=500
        )
        self.assertEqual(first.medoid_trial_ids, second.medoid_trial_ids)
        self.assertEqual(first.train_child_ids, second.train_child_ids)
        self.assertEqual(first.objective, second.objective)
        np.testing.assert_array_equal(
            assign_secondary_codebook(first, occurrences),
            np.asarray(first.train_child_ids, dtype=np.int64),
        )
        diagnostics = secondary_fit_diagnostics(first, occurrences)
        self.assertIsNotNone(diagnostics["silhouette_precomputed"])
        with self.assertRaisesRegex(ValueError, "exact training order"):
            secondary_fit_diagnostics(first, list(reversed(occurrences)))
        self.assertEqual(
            [set(first.train_child_ids[:3]), set(first.train_child_ids[3:])],
            [{0}, {1}],
        )

    def test_kmedoids_rejects_feature_identical_fabricated_children(self) -> None:
        identical = [
            make_occurrence(
                trial_id,
                (0.0, 0.0),
                (1.0, 0.0, 0.0),
            )
            for trial_id in range(4)
        ]
        with self.assertRaisesRegex(RuntimeError, "distinct child medoids"):
            fit_secondary_codebook(
                identical, arm="U3_joint", restarts=10, seed=500
            )

    def test_gravity_shuffle_preserves_each_subject_split_multiset(self) -> None:
        occurrences = []
        subjects = {}
        roles = {}
        for trial_id in range(12):
            occurrences.append(
                make_occurrence(
                    trial_id,
                    residual=(float(trial_id), float(trial_id % 3)),
                    gravity=(1.0, float(trial_id + 1), float((trial_id % 4) + 1)),
                )
            )
            subjects[trial_id] = 4 if trial_id < 6 else 5
            roles[trial_id] = (
                "cumulative_online_train"
                if trial_id % 6 < 3
                else "session_2_test"
            )

        shuffled, audit = shuffle_gravity_within_subject_split(
            occurrences,
            subjects_by_trial=subjects,
            split_roles_by_trial=roles,
            rng=np.random.default_rng(19),
        )

        def grouped_gravity(items):
            result = {}
            for item in items:
                trial_id = int(item.trial_global_id)
                key = (subjects[trial_id], roles[trial_id])
                result.setdefault(key, []).append(
                    tuple(np.asarray(item.gravity_direction).tolist())
                )
            return {key: sorted(values) for key, values in result.items()}

        self.assertEqual(grouped_gravity(shuffled), grouped_gravity(occurrences))
        self.assertGreater(audit["moved_occurrence_count"], 0)
        self.assertIs(audit["crosses_train_test_boundary"], False)
        self.assertIs(audit["uses_activity_labels"], False)
        for original, permuted in zip(occurrences, shuffled):
            self.assertEqual(original.trial_global_id, permuted.trial_global_id)
            np.testing.assert_array_equal(
                original.mean_quantization_residual,
                permuted.mean_quantization_residual,
            )

    def test_monte_carlo_upper_tail_uses_plus_one_and_reports_quantiles(self) -> None:
        null = np.asarray([0.50, 0.70, 0.80], dtype=np.float64)
        result = monte_carlo_upper_tail_summary(0.70, null)
        self.assertEqual(result["permutation_count"], 3)
        self.assertAlmostEqual(result["null_mean"], float(np.mean(null)))
        self.assertAlmostEqual(
            result["observed_minus_null_mean"], 0.70 - float(np.mean(null))
        )
        self.assertEqual(result["monte_carlo_upper_tail_p"], 0.75)
        expected = np.quantile(null, [0.025, 0.50, 0.975])
        self.assertEqual(
            result["null_quantiles"],
            {
                "q025": float(expected[0]),
                "q500": float(expected[1]),
                "q975": float(expected[2]),
            },
        )

    def test_paired_bootstrap_is_deterministic_stratified_and_antisymmetric(self) -> None:
        trial_ids = np.arange(8, dtype=np.int64)
        truth = np.asarray([0, 0, 6, 6, 0, 0, 6, 6], dtype=np.int64)
        subjects = np.asarray([4, 4, 4, 4, 5, 5, 5, 5], dtype=np.int64)
        reference = np.asarray([1, 0, 7, 6, 1, 0, 7, 6], dtype=np.int64)
        challenger = truth.copy()
        kwargs = {
            "trial_ids": trial_ids,
            "y_true": truth,
            "subjects": subjects,
            "subset_mask": np.ones(8, dtype=bool),
            "resamples": 400,
            "seed": 31,
        }
        first = stratified_paired_accuracy_bootstrap(
            reference_aligned=reference,
            challenger_aligned=challenger,
            **kwargs,
        )
        second = stratified_paired_accuracy_bootstrap(
            reference_aligned=reference,
            challenger_aligned=challenger,
            **kwargs,
        )
        reverse = stratified_paired_accuracy_bootstrap(
            reference_aligned=challenger,
            challenger_aligned=reference,
            **kwargs,
        )
        self.assertEqual(first, second)
        self.assertEqual(first["observed_difference"], 0.5)
        self.assertEqual(
            first["strata_counts"],
            {
                "subject=4|activity=0": 2,
                "subject=4|activity=6": 2,
                "subject=5|activity=0": 2,
                "subject=5|activity=6": 2,
            },
        )
        self.assertAlmostEqual(
            reverse["observed_difference"], -first["observed_difference"]
        )
        self.assertAlmostEqual(reverse["ci_lower"], -first["ci_upper"])
        self.assertAlmostEqual(reverse["ci_upper"], -first["ci_lower"])

    def test_bootstrap_holds_hungarian_mapping_frozen(self) -> None:
        truth = np.asarray([0, 0, 6, 6], dtype=np.int64)
        with mock.patch.object(
            secondary_module,
            "linear_sum_assignment",
            side_effect=AssertionError("Hungarian must stay outside bootstrap"),
        ):
            result = stratified_paired_accuracy_bootstrap(
                trial_ids=np.arange(4),
                y_true=truth,
                subjects=np.asarray([4, 4, 5, 5]),
                reference_aligned=np.asarray([0, 1, 6, 7]),
                challenger_aligned=truth,
                subset_mask=np.ones(4, dtype=bool),
                resamples=50,
                seed=7,
            )
        self.assertEqual(
            result["hungarian_mapping_policy"],
            "frozen_on_complete_test_set_before_bootstrap",
        )

    def test_gravity_null_runner_helper_is_label_free_and_has_expected_shape(self) -> None:
        train_ids = list(range(12))
        test_ids = list(range(12, 20))
        all_ids = train_ids + test_ids
        occurrences = {
            trial_id: make_occurrence(
                trial_id,
                residual=(float(trial_id), float((trial_id * 3) % 7)),
                gravity=(
                    1.0,
                    float((trial_id % 5) + 1),
                    float((trial_id % 7) + 1),
                ),
                token=1,
            )
            for trial_id in all_ids
        }
        subjects = {trial_id: 4 + (trial_id % 2) for trial_id in all_ids}
        durations = {trial_id: {0: 10.0, 1: 90.0} for trial_id in all_ids}
        artifacts = SegmentArtifacts(
            split_role=np.empty(0, dtype=np.int8),
            trial_ids=np.empty(0, dtype=np.int64),
            starts=np.empty(0, dtype=np.int64),
            ends=np.empty(0, dtype=np.int64),
            tokens=np.empty(0, dtype=np.int64),
            embeddings=np.empty((0, 2), dtype=np.float32),
            centers=np.zeros((2, 2), dtype=np.float32),
        )
        readout_calls = []

        def fake_readout(train_features, test_features, seed):
            readout_calls.append((train_features.shape, test_features.shape, seed))
            return np.arange(len(test_features), dtype=np.int64) % 10, {
                "fit_uses_activity_labels": False
            }

        with mock.patch.object(
            runner_module, "_fit_trial_clusterer", side_effect=fake_readout
        ):
            predictions, test_children, audit = _gravity_shuffle_null_predictions(
                artifacts=artifacts,
                occurrences=occurrences,
                fit_trial_ids=train_ids,
                trial_token_durations=durations,
                train_ids=train_ids,
                test_ids=test_ids,
                subjects_by_trial=subjects,
                selected_token=1,
                shuffles=3,
                seed=500,
            )
        self.assertEqual(predictions["U2_gravity"].shape, (3, 8))
        self.assertEqual(predictions["U3_joint"].shape, (3, 8))
        for arm in ("U2_gravity", "U3_joint"):
            np.testing.assert_array_equal(
                test_children[arm]["trial_ids"], np.asarray(test_ids)
            )
            self.assertEqual(test_children[arm]["child_ids"].shape, (3, 8))
            self.assertTrue(
                np.all(np.isin(test_children[arm]["child_ids"], [0, 1]))
            )
        self.assertEqual(len(readout_calls), 6)
        self.assertIs(audit["uses_activity_labels_for_permutation_or_fitting"], False)
        self.assertIs(audit["secondary_fit_trial_ids_are_train_only"], True)
        self.assertNotIn("label", vars(next(iter(occurrences.values()))))
        self.assertNotIn("activity_name", vars(next(iter(occurrences.values()))))
        with self.assertRaisesRegex(RuntimeError, "non-train trial"):
            _gravity_shuffle_null_predictions(
                artifacts=artifacts,
                occurrences=occurrences,
                fit_trial_ids=[test_ids[0]],
                trial_token_durations=durations,
                train_ids=train_ids,
                test_ids=test_ids,
                subjects_by_trial=subjects,
                selected_token=1,
                shuffles=1,
                seed=500,
            )

    def test_fixed_u0_shift_preserves_gravity_null_effect_and_p_value(self) -> None:
        observed_u2 = 0.75
        fixed_u0 = 0.60
        null_u2 = np.asarray([0.55, 0.65, 0.75, 0.80], dtype=np.float64)
        absolute = monte_carlo_upper_tail_summary(observed_u2, null_u2)
        contrast = monte_carlo_upper_tail_summary(
            observed_u2 - fixed_u0, null_u2 - fixed_u0
        )
        self.assertAlmostEqual(
            absolute["observed_minus_null_mean"],
            contrast["observed_minus_null_mean"],
        )
        self.assertEqual(
            absolute["monte_carlo_upper_tail_p"],
            contrast["monte_carlo_upper_tail_p"],
        )
        for key in ("q025", "q500", "q975"):
            self.assertAlmostEqual(
                absolute["null_quantiles"][key] - fixed_u0,
                contrast["null_quantiles"][key],
            )

    def test_positive_cli_count_rejects_zero(self) -> None:
        self.assertEqual(_positive_int("3"), 3)
        with self.assertRaisesRegex(argparse.ArgumentTypeError, "positive integer"):
            _positive_int("0")

    def test_posthoc_sit_stand_reports_each_subject_with_one_mapping(self) -> None:
        labels = {
            0: 7,
            1: 7,
            2: 8,
            3: 8,
            4: 7,
            5: 7,
            6: 8,
            7: 8,
        }
        subjects = {trial_id: 4 if trial_id < 4 else 5 for trial_id in labels}
        children = {0: 0, 1: 0, 2: 1, 3: 1, 4: 0, 5: 1, 6: 1, 7: 1}
        result = posthoc_secondary_metrics(children, labels, subjects)
        self.assertEqual(result["binary_hungarian_accuracy"], 0.875)
        by_subject = result["binary_by_subject_using_same_global_mapping"]
        self.assertEqual(by_subject["4"]["accuracy_using_global_binary_mapping"], 1.0)
        self.assertEqual(
            by_subject["4"]["balanced_accuracy_using_global_binary_mapping"],
            1.0,
        )
        self.assertEqual(by_subject["5"]["accuracy_using_global_binary_mapping"], 0.75)
        self.assertEqual(
            by_subject["5"]["balanced_accuracy_using_global_binary_mapping"],
            0.75,
        )

    def test_old_and_new_metrics_reuse_one_global_hungarian_alignment(self) -> None:
        truth = np.asarray([0, 0, 1, 1, 2, 2, 3, 3], dtype=np.int64)
        predicted = np.asarray([2, 2, 3, 3, 0, 0, 1, 1], dtype=np.int64)
        real_assignment = secondary_module.linear_sum_assignment
        with mock.patch.object(
            secondary_module,
            "linear_sum_assignment",
            wraps=real_assignment,
        ) as assignment:
            result = global_hungarian_metrics(
                truth, predicted, old_class_count=2
            )
        assignment.assert_called_once()
        self.assertEqual(result["hungarian_call_count"], 1)
        self.assertEqual(result["alignment_scope"], "single_global_hungarian")
        self.assertEqual(result["mapping_pred_to_true"], {"0": 2, "1": 3, "2": 0, "3": 1})
        self.assertEqual(result["all_accuracy"], 1.0)
        self.assertEqual(result["old_accuracy"], 1.0)
        self.assertEqual(result["new_accuracy"], 1.0)
        aligned = np.asarray(result["aligned_predictions"], dtype=np.int64)
        correct = aligned == truth
        self.assertEqual(result["old_accuracy"], float(np.mean(correct[truth < 2])))
        self.assertEqual(result["new_accuracy"], float(np.mean(correct[truth >= 2])))
        np.testing.assert_array_equal(result["aligned_predictions"], truth)


if __name__ == "__main__":
    unittest.main()
