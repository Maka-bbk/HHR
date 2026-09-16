from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import numpy as np

from experiments.motion_primitive.analyze_online_secondary_confirmations import (
    EXPECTED_RUNS,
    FrozenRun,
    paired_bootstrap_effect,
    stratified_bootstrap_indices,
    validate_paired_runs,
)


class OnlineSecondaryConfirmationTests(unittest.TestCase):
    def test_stratified_bootstrap_preserves_subject_activity_counts(self) -> None:
        subjects = np.asarray([4, 4, 4, 4, 5, 5, 5, 5], dtype=np.int64)
        labels = np.asarray([0, 0, 1, 1, 0, 0, 1, 1], dtype=np.int64)
        samples = stratified_bootstrap_indices(
            subjects, labels, replicates=100, seed=17
        )
        self.assertEqual(samples.shape, (100, 8))
        for row in samples:
            sampled_pairs = list(zip(subjects[row].tolist(), labels[row].tolist()))
            for pair in ((4, 0), (4, 1), (5, 0), (5, 1)):
                self.assertEqual(sampled_pairs.count(pair), 2)

    def test_paired_effect_holds_mapping_fixed_and_reuses_trial_indices(self) -> None:
        truth = np.asarray([0, 0, 1, 1, 6, 6, 7, 7], dtype=np.int64)
        left = truth.copy()
        right = np.asarray([1, 0, 0, 1, 7, 6, 6, 7], dtype=np.int64)
        subjects = np.asarray([4, 4, 5, 5, 4, 4, 5, 5], dtype=np.int64)
        samples = stratified_bootstrap_indices(
            subjects, truth, replicates=200, seed=29
        )
        all_effect = paired_bootstrap_effect(
            truth, left, right, samples, "all_accuracy"
        )
        old_effect = paired_bootstrap_effect(
            truth, left, right, samples, "old_accuracy"
        )
        new_effect = paired_bootstrap_effect(
            truth, left, right, samples, "new_accuracy"
        )
        self.assertEqual(all_effect["left_minus_right"], 0.5)
        self.assertEqual(old_effect["left_minus_right"], 0.5)
        self.assertEqual(new_effect["left_minus_right"], 0.5)
        self.assertEqual(all_effect["evaluated_trial_count"], 8)
        self.assertEqual(old_effect["evaluated_trial_count"], 4)
        self.assertEqual(new_effect["evaluated_trial_count"], 4)
        self.assertIn("held fixed", all_effect["mapping_policy"])

    def test_four_run_pairing_rejects_truth_drift(self) -> None:
        trial_ids = np.arange(20, dtype=np.int64)
        subjects = np.repeat(np.asarray([4, 5], dtype=np.int64), 10)
        truth = np.tile(np.arange(10, dtype=np.int64), 2)
        predictions = {arm: truth.copy() for arm in (
            "U0_coarse",
            "U1_residual",
            "U2_gravity",
            "U3_joint",
        )}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runs = {}
            for key in EXPECTED_RUNS:
                run_dir = root / key
                run_dir.mkdir()
                (run_dir / "session_manifest.csv").write_text(
                    "split,session,trial_global_id\n", encoding="utf-8"
                )
                (run_dir / "online_secondary_codebook_results.json").write_text(
                    "{}\n", encoding="utf-8"
                )
                runs[key] = FrozenRun(
                    key=key,
                    directory=run_dir,
                    result={
                        "input_audit": {
                            "npz_sha256": "same",
                            "experiment_implementation_fingerprint": {
                                "runner_sha256": "runner",
                                "secondary_codebook_helper_sha256": "helper",
                            },
                        },
                    },
                    trial_ids=trial_ids.copy(),
                    subjects=subjects.copy(),
                    truth=truth.copy(),
                    aligned_predictions=predictions,
                )
            audit = validate_paired_runs(runs)
            self.assertEqual(audit["paired_test_trial_count"], 20)
            changed = dict(runs)
            changed_truth = truth.copy()
            changed_truth[0] = 9
            changed["A3_fixed"] = FrozenRun(
                key="A3_fixed",
                directory=runs["A3_fixed"].directory,
                result=runs["A3_fixed"].result,
                trial_ids=trial_ids.copy(),
                subjects=subjects.copy(),
                truth=changed_truth,
                aligned_predictions=predictions,
            )
            with self.assertRaisesRegex(RuntimeError, "ground truth is not paired"):
                validate_paired_runs(changed)


if __name__ == "__main__":
    unittest.main()
