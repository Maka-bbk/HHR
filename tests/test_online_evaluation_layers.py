from __future__ import annotations

import unittest
from unittest import mock

import numpy as np

from experiments.motion_primitive import motion_online
from experiments.motion_primitive.motion_online import (
    aligned_online_metrics,
    constrained_old_fixed_alignment,
    constrained_old_fixed_online_metrics,
    direct_head_online_metrics,
    stratified_online_metrics,
)
from experiments.motion_primitive.online_runner import online_fieldnames


class ConstrainedOldFixedAlignmentTests(unittest.TestCase):
    def setUp(self) -> None:
        # Both supervised old rows and unsupervised novel rows are swapped.
        self.truth = np.asarray([0, 0, 1, 1, 2, 2, 3, 3], dtype=np.int64)
        self.raw = np.asarray([1, 1, 0, 0, 3, 3, 2, 2], dtype=np.int64)

    def test_global_hungarian_can_hide_complete_old_class_forgetting(self):
        standard = aligned_online_metrics(
            self.truth,
            self.raw,
            class_count=4,
            old_class_count=2,
            seen_class_count_before_session=2,
        )
        constrained = constrained_old_fixed_online_metrics(
            self.truth,
            self.raw,
            class_count=4,
            old_class_count=2,
            seen_class_count_before_session=2,
        )
        self.assertEqual(standard["old_accuracy"], 1.0)
        self.assertEqual(standard["new_accuracy"], 1.0)
        self.assertEqual(constrained["old_accuracy"], 0.0)
        self.assertEqual(constrained["new_accuracy"], 1.0)
        self.assertEqual(constrained["h_score"], 0.0)

    def test_old_rows_are_identity_and_only_novel_rows_are_permuted(self):
        aligned, pairs = constrained_old_fixed_alignment(
            self.truth, self.raw, class_count=4, old_class_count=2
        )
        self.assertEqual(pairs[:2], [[0, 0], [1, 1]])
        self.assertEqual(pairs[2:], [[2, 3], [3, 2]])
        self.assertTrue(np.array_equal(aligned[:4], self.raw[:4]))
        self.assertTrue(np.array_equal(aligned[4:], self.truth[4:]))

    def test_cross_partition_errors_cannot_be_relabelled_away(self):
        truth = np.asarray([0, 0, 2, 2], dtype=np.int64)
        raw = np.asarray([2, 2, 0, 0], dtype=np.int64)
        aligned, _ = constrained_old_fixed_alignment(
            truth, raw, class_count=4, old_class_count=2
        )
        self.assertTrue(np.array_equal(aligned[:2], np.asarray([2, 2])))
        self.assertTrue(np.array_equal(aligned[2:], np.asarray([0, 0])))


class DirectHeadEvaluationTests(unittest.TestCase):
    def test_direct_head_does_not_call_hungarian(self):
        truth = np.asarray([0, 1, 2, 3], dtype=np.int64)
        raw = np.asarray([0, 1, 3, 2], dtype=np.int64)
        with mock.patch.object(
            motion_online,
            "linear_sum_assignment",
            side_effect=AssertionError("direct_head must not fit a label map"),
        ):
            metrics = direct_head_online_metrics(
                truth,
                raw,
                class_count=4,
                old_class_count=2,
                seen_class_count_before_session=2,
            )
        self.assertFalse(metrics["alignment_uses_test_labels"])
        self.assertEqual(metrics["prediction_alignment"], "direct_head")
        self.assertEqual(
            metrics["assignment_pred_to_true"],
            [[0, 0], [1, 1], [2, 2], [3, 3]],
        )
        self.assertEqual(metrics["old_accuracy"], 1.0)
        self.assertEqual(metrics["new_accuracy"], 0.0)

    def test_stratified_result_contains_all_three_declared_layers(self):
        truth = np.asarray([0, 1, 2, 3], dtype=np.int64)
        raw = np.asarray([1, 0, 3, 2], dtype=np.int64)
        layers = stratified_online_metrics(
            truth,
            raw,
            class_count=4,
            old_class_count=2,
            seen_class_count_before_session=2,
        )
        self.assertEqual(
            set(layers),
            {
                "standard_global_hungarian",
                "constrained_old_fixed",
                "direct_head",
            },
        )
        self.assertTrue(
            layers["standard_global_hungarian"]["alignment_uses_test_labels"]
        )
        self.assertTrue(
            layers["constrained_old_fixed"]["alignment_uses_test_labels"]
        )
        self.assertFalse(layers["direct_head"]["alignment_uses_test_labels"])

    def test_runner_csv_contract_has_explicit_layer_columns(self):
        row = {
            "schema": "test",
            "primary_evaluation_layer": "constrained_old_fixed",
            "eval_standard_global_hungarian_old_accuracy": 1.0,
            "eval_constrained_old_fixed_old_accuracy": 0.0,
            "eval_direct_head_old_accuracy": 0.0,
        }
        fields = online_fieldnames([row])
        self.assertIn("primary_evaluation_layer", fields)
        self.assertIn("eval_standard_global_hungarian_old_accuracy", fields)
        self.assertIn("eval_constrained_old_fixed_old_accuracy", fields)
        self.assertIn("eval_direct_head_old_accuracy", fields)


if __name__ == "__main__":
    unittest.main()
