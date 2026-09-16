from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from experiments.motion_primitive.run_peak_valley_hierarchy import (
    CROSS_SUBJECT_DIAGNOSTIC_FILES,
    CROSS_SUBJECT_DIAGNOSTIC_SCHEMA,
    _cross_subject_diagnostic_csv_row,
    _cross_subject_trajectory_diagnostics,
    _generated_file_manifest,
    _write_csv,
    _write_json,
)


class CrossSubjectTrajectoryDiagnosticTests(unittest.TestCase):
    def test_same_activity_geometry_and_subject_independent_clusters(self) -> None:
        features = np.asarray([[0.0], [10.0], [0.0], [10.0]], dtype=np.float32)
        labels = np.asarray([0, 1, 0, 1], dtype=np.int64)
        subjects = np.asarray([1, 1, 2, 2], dtype=np.int64)
        raw_clusters = np.asarray([0, 1, 0, 1], dtype=np.int64)
        frozen_features = features.copy()
        frozen_clusters = raw_clusters.copy()

        result = _cross_subject_trajectory_diagnostics(
            features, labels, subjects, raw_clusters
        )

        effect = result["cross_subject_distance_effect"]
        nearest = result["tie_aware_cross_subject_1nn"]
        nmi = result["cluster_subject_nmi"]
        self.assertTrue(effect["available"])
        self.assertEqual(effect["same_activity_pair_count"], 2)
        self.assertEqual(effect["different_activity_pair_count"], 2)
        self.assertAlmostEqual(effect["same_activity_distance_mean"], 0.0)
        self.assertAlmostEqual(effect["different_activity_distance_mean"], 10.0)
        self.assertAlmostEqual(effect["probability_same_distance_is_smaller"], 1.0)
        self.assertAlmostEqual(effect["rank_separation_effect"], 1.0)
        self.assertTrue(nearest["available"])
        self.assertAlmostEqual(nearest["accuracy"], 1.0)
        self.assertEqual(nearest["query_count"], 4)
        self.assertTrue(nmi["available"])
        self.assertAlmostEqual(nmi["normalized_mutual_information"], 0.0)
        self.assertTrue(result["post_truth_join_diagnostic_only"])
        self.assertFalse(result["used_to_fit_or_modify_predictions"])
        np.testing.assert_array_equal(features, frozen_features)
        np.testing.assert_array_equal(raw_clusters, frozen_clusters)

        subject_coded = _cross_subject_trajectory_diagnostics(
            features,
            labels,
            subjects,
            np.asarray([0, 0, 1, 1], dtype=np.int64),
        )
        self.assertAlmostEqual(
            subject_coded["cluster_subject_nmi"]["normalized_mutual_information"],
            1.0,
        )

    def test_cross_subject_1nn_averages_all_tied_neighbours(self) -> None:
        # The subject-1 query is equally close to a correct and an incorrect
        # subject-2 neighbour, so its contribution must be 0.5 rather than
        # depending on input order.  The other two query contributions are 1/0.
        result = _cross_subject_trajectory_diagnostics(
            np.asarray([[0.0], [1.0], [-1.0]], dtype=np.float32),
            np.asarray([0, 0, 1], dtype=np.int64),
            np.asarray([1, 2, 2], dtype=np.int64),
            np.asarray([0, 0, 1], dtype=np.int64),
        )

        nearest = result["tie_aware_cross_subject_1nn"]
        effect = result["cross_subject_distance_effect"]
        self.assertTrue(nearest["available"])
        self.assertAlmostEqual(nearest["accuracy"], 0.5)
        self.assertEqual(nearest["query_count"], 3)
        self.assertEqual(nearest["query_count_with_ties"], 1)
        self.assertEqual(nearest["maximum_tied_neighbour_count"], 2)
        self.assertTrue(effect["available"])
        self.assertAlmostEqual(effect["probability_same_distance_is_smaller"], 0.5)

    def test_single_subject_is_explicitly_unavailable(self) -> None:
        result = _cross_subject_trajectory_diagnostics(
            np.asarray([[0.0], [1.0]], dtype=np.float32),
            np.asarray([0, 1], dtype=np.int64),
            np.asarray([7, 7], dtype=np.int64),
            np.asarray([0, 1], dtype=np.int64),
        )

        for key in (
            "cross_subject_distance_effect",
            "tie_aware_cross_subject_1nn",
            "cluster_subject_nmi",
        ):
            self.assertFalse(result[key]["available"])
            self.assertEqual(result[key]["reason"], "requires_at_least_two_subjects")

    def test_one_class_keeps_1nn_but_marks_distance_effect_unavailable(self) -> None:
        result = _cross_subject_trajectory_diagnostics(
            np.asarray([[0.0], [0.2]], dtype=np.float32),
            np.asarray([0, 0], dtype=np.int64),
            np.asarray([1, 2], dtype=np.int64),
            np.asarray([0, 0], dtype=np.int64),
        )

        effect = result["cross_subject_distance_effect"]
        self.assertFalse(effect["available"])
        self.assertEqual(
            effect["reason"],
            "requires_both_same_and_different_activity_cross_subject_pairs",
        )
        self.assertEqual(effect["same_activity_pair_count"], 1)
        self.assertEqual(effect["different_activity_pair_count"], 0)
        self.assertTrue(result["tie_aware_cross_subject_1nn"]["available"])
        self.assertAlmostEqual(
            result["tie_aware_cross_subject_1nn"]["accuracy"], 1.0
        )
        self.assertTrue(result["cluster_subject_nmi"]["available"])

    def test_empty_input_and_malformed_metadata_have_clear_behaviour(self) -> None:
        empty = _cross_subject_trajectory_diagnostics(
            np.empty((0, 3), dtype=np.float32),
            np.asarray([], dtype=np.int64),
            np.asarray([], dtype=np.int64),
            np.asarray([], dtype=np.int64),
        )
        self.assertEqual(empty["trial_count"], 0)
        for key in (
            "cross_subject_distance_effect",
            "tie_aware_cross_subject_1nn",
            "cluster_subject_nmi",
        ):
            self.assertFalse(empty[key]["available"])
            self.assertEqual(empty[key]["reason"], "no_test_trials")

        with self.assertRaisesRegex(ValueError, "activity_labels must have shape"):
            _cross_subject_trajectory_diagnostics(
                np.zeros((2, 2), dtype=np.float32),
                np.asarray([0], dtype=np.int64),
                np.asarray([1, 2], dtype=np.int64),
                np.asarray([0, 1], dtype=np.int64),
            )
        with self.assertRaisesRegex(ValueError, "subject_ids must be integer-valued"):
            _cross_subject_trajectory_diagnostics(
                np.zeros((2, 2), dtype=np.float32),
                np.asarray([0, 1], dtype=np.int64),
                np.asarray([1.0, 2.0], dtype=np.float64),
                np.asarray([0, 1], dtype=np.int64),
            )

    def test_json_csv_and_content_manifest_are_stable(self) -> None:
        diagnostic = _cross_subject_trajectory_diagnostics(
            np.asarray([[0.0], [2.0]], dtype=np.float32),
            np.asarray([0, 1], dtype=np.int64),
            np.asarray([1, 2], dtype=np.int64),
            np.asarray([0, 1], dtype=np.int64),
        )
        report = {
            "schema": CROSS_SUBJECT_DIAGNOSTIC_SCHEMA,
            "records": [
                {"arm_id": "E2", "readout_variant": "state", **diagnostic}
            ],
        }
        row = _cross_subject_diagnostic_csv_row("E2", "state", diagnostic)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _write_json(root / CROSS_SUBJECT_DIAGNOSTIC_FILES[0], report)
            _write_csv(root / CROSS_SUBJECT_DIAGNOSTIC_FILES[1], [row])
            manifest = _generated_file_manifest(
                root, list(CROSS_SUBJECT_DIAGNOSTIC_FILES)
            )

            self.assertEqual(
                [item["relative_path"] for item in manifest],
                list(CROSS_SUBJECT_DIAGNOSTIC_FILES),
            )
            self.assertTrue(all(len(item["sha256"]) == 64 for item in manifest))
            saved = json.loads(
                (root / CROSS_SUBJECT_DIAGNOSTIC_FILES[0]).read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(saved["schema"], CROSS_SUBJECT_DIAGNOSTIC_SCHEMA)
            self.assertEqual(saved["records"][0]["arm_id"], "E2")
            csv_text = (root / CROSS_SUBJECT_DIAGNOSTIC_FILES[1]).read_text(
                encoding="utf-8-sig"
            )
            self.assertIn("cross_subject_1nn_accuracy", csv_text.splitlines()[0])


if __name__ == "__main__":
    unittest.main()
