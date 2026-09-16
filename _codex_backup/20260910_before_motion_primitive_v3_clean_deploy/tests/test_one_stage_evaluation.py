import unittest

import numpy as np

from experiments.motion_primitive.one_stage_evaluation import (
    codebook_diagnostics,
    score_cgcd_clusters,
    semi_supervised_kmeans,
)


class SemiSupervisedKMeansTests(unittest.TestCase):
    def test_separated_old_and_new_classes_are_recovered(self):
        labelled = np.asarray(
            [[1.0, 0.0], [0.9, 0.1], [-1.0, 0.0], [-0.9, -0.1]],
            dtype=np.float64,
        )
        labelled_targets = np.asarray([0, 0, 1, 1], dtype=np.int64)
        unlabelled = np.asarray(
            [
                [1.0, 0.02],
                [-1.0, -0.02],
                [0.0, 1.0],
                [0.02, 1.0],
                [0.0, -1.0],
                [-0.02, -1.0],
            ],
            dtype=np.float64,
        )
        truth = np.asarray([0, 1, 2, 2, 3, 3], dtype=np.int64)
        result = semi_supervised_kmeans(
            labelled,
            labelled_targets,
            unlabelled,
            num_classes=4,
            num_old_classes=2,
            seed=17,
            n_init=8,
        )
        scores = score_cgcd_clusters(
            result.assignments,
            truth,
            num_classes=4,
            num_old_classes=2,
        )
        self.assertEqual(scores["all_accuracy"], 1.0)
        self.assertEqual(scores["old_accuracy"], 1.0)
        self.assertEqual(scores["new_accuracy"], 1.0)
        self.assertEqual(scores["h_score"], 1.0)

    def test_unlabelled_truth_cannot_be_passed_to_fit(self):
        # The API intentionally has no unlabelled-target argument.  Supplying
        # one must fail rather than silently introduce evaluation leakage.
        with self.assertRaises(TypeError):
            semi_supervised_kmeans(
                np.eye(2),
                np.asarray([0, 1]),
                np.eye(2),
                unlabelled_targets=np.asarray([0, 1]),
                num_classes=3,
                num_old_classes=2,
            )

    def test_every_old_class_requires_an_anchor(self):
        with self.assertRaisesRegex(ValueError, "Every old class"):
            semi_supervised_kmeans(
                np.asarray([[1.0, 0.0], [0.9, 0.1]]),
                np.asarray([0, 0]),
                np.asarray([[0.0, 1.0], [0.0, -1.0]]),
                num_classes=3,
                num_old_classes=2,
            )


class CodebookDiagnosticTests(unittest.TestCase):
    def test_effective_count_and_subject_nmi(self):
        result = codebook_diagnostics(
            np.asarray([0, 1, 0, 1]),
            np.asarray([1, 1, 2, 2]),
            num_codes=4,
        )
        self.assertEqual(result["occupied_codes"], 2)
        self.assertAlmostEqual(result["effective_code_count"], 2.0)
        self.assertAlmostEqual(result["token_subject_nmi"], 0.0)


if __name__ == "__main__":
    unittest.main()
