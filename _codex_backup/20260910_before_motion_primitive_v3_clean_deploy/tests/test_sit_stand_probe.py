from __future__ import annotations

import csv
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from experiments.motion_primitive.sit_stand_probe import (
    TrialProbeFeature,
    evaluate_bidirectional_probe,
    evaluate_direction,
    robust_gravity_direction,
    validate_probe_trials,
)
from experiments.motion_primitive.run_sit_stand_probe import (
    _validate_global_window_indices,
    _validate_registered_run_metadata,
    _validate_trial_window_coverage,
    _write_csv,
)


def synthetic_trials() -> list[TrialProbeFeature]:
    trials = []
    trial_id = 0
    for subject in (4, 5):
        subject_shift = 0.03 * (subject - 4)
        for label, name in ((7, "Sitting"), (8, "Standing")):
            for trial_number in range(1, 6):
                jitter = 0.01 * trial_number
                residual_base = 0.0 if label == 7 else 2.0
                gravity = (
                    np.asarray([1.0, jitter, 0.0], dtype=np.float32)
                    if label == 7
                    else np.asarray([jitter, 1.0, 0.0], dtype=np.float32)
                )
                gravity /= np.linalg.norm(gravity)
                trials.append(
                    TrialProbeFeature(
                        trial_global_id=trial_id,
                        subject_id=subject,
                        activity_label=label,
                        activity_name=name,
                        trial_number=trial_number,
                        token_histogram=np.asarray([1.0, 0.0], dtype=np.float32),
                        mean_quantization_residual=np.asarray(
                            [residual_base + jitter + subject_shift, jitter],
                            dtype=np.float32,
                        ),
                        gravity_direction=gravity,
                    )
                )
                trial_id += 1
    return trials


class SitStandProbeTests(unittest.TestCase):
    def test_identical_tokens_are_tie_aware_chance(self) -> None:
        result = evaluate_bidirectional_probe(synthetic_trials())
        token = result["representations"]["token_only"]
        self.assertEqual(
            [item["accuracy"] for item in token["directions"]], [0.5, 0.5]
        )
        self.assertEqual(
            [item["tie_query_ratio"] for item in token["directions"]], [1.0, 1.0]
        )
        self.assertIs(token["registered_gate"]["passed"], False)

    def test_synthetic_residual_and_gravity_pass_both_directions(self) -> None:
        result = evaluate_bidirectional_probe(synthetic_trials())
        for representation in ("token_residual", "token_gravity"):
            values = result["representations"][representation]
            self.assertEqual(
                [item["accuracy"] for item in values["directions"]], [1.0, 1.0]
            )
            self.assertIs(values["registered_gate"]["passed"], True)

    def test_zero_within_class_scale_falls_back_to_interclass_distance(self) -> None:
        stable = [
            replace(
                trial,
                mean_quantization_residual=(
                    np.asarray([0.0, 0.0], dtype=np.float32)
                    if trial.activity_label == 7
                    else np.asarray([2.0, 0.0], dtype=np.float32)
                ),
                gravity_direction=(
                    np.asarray([1.0, 0.0, 0.0], dtype=np.float32)
                    if trial.activity_label == 7
                    else np.asarray([0.0, 1.0, 0.0], dtype=np.float32)
                ),
            )
            for trial in synthetic_trials()
        ]
        result = evaluate_bidirectional_probe(stable)
        for representation, block, expected_scale in (
            ("token_residual", "residual", 2.0),
            ("token_gravity", "gravity", 0.5),
        ):
            directions = result["representations"][representation]["directions"]
            self.assertEqual([item["accuracy"] for item in directions], [1.0, 1.0])
            for direction in directions:
                self.assertIn(block, direction["active_blocks"])
                self.assertAlmostEqual(
                    direction["source_only_block_scales"][block], expected_scale
                )

    def test_source_scale_does_not_depend_on_target_features(self) -> None:
        trials = synthetic_trials()
        original = evaluate_direction(trials, 4, 5, "token_residual")
        changed = [
            replace(
                trial,
                mean_quantization_residual=(
                    trial.mean_quantization_residual * 1000.0
                    if trial.subject_id == 5
                    else trial.mean_quantization_residual
                ),
            )
            for trial in trials
        ]
        mutated = evaluate_direction(changed, 4, 5, "token_residual")
        self.assertEqual(
            original["source_only_block_scales"],
            mutated["source_only_block_scales"],
        )

    def test_validation_fails_closed_for_missing_class_or_subject(self) -> None:
        trials = synthetic_trials()
        with self.assertRaisesRegex(ValueError, "Exactly Sitting and Standing"):
            validate_probe_trials(
                [trial for trial in trials if trial.activity_label == 7]
            )
        with self.assertRaisesRegex(ValueError, "Exactly two held-out subjects"):
            validate_probe_trials([trial for trial in trials if trial.subject_id == 4])

    def test_validation_fails_closed_for_name_label_alias(self) -> None:
        trials = synthetic_trials()
        changed = [
            replace(trial, activity_name="Sitting")
            if trial.activity_label == 8
            else trial
            for trial in trials
        ]
        with self.assertRaisesRegex(ValueError, "exactly Sitting and Standing"):
            validate_probe_trials(changed)

    def test_validation_requires_five_trials_per_class_and_subject(self) -> None:
        trials = synthetic_trials()
        changed = [
            replace(trial, activity_label=8, activity_name="Standing")
            if trial.subject_id == 4
            and trial.activity_label == 7
            and trial.trial_number == 1
            else trial
            for trial in trials
        ]
        with self.assertRaisesRegex(ValueError, "exactly 5 trials per class"):
            validate_probe_trials(changed)

    def test_robust_gravity_uses_central_span(self) -> None:
        sensor = np.zeros((6, 100), dtype=np.float32)
        sensor[0, :10] = 1000.0
        sensor[2, 10:90] = 9.81
        sensor[1, 90:] = -1000.0
        gravity = robust_gravity_direction(sensor, trim_fraction=0.10)
        np.testing.assert_allclose(
            gravity, np.asarray([0.0, 0.0, 1.0]), atol=1e-7
        )

    def test_registered_run_metadata_fails_closed(self) -> None:
        fit_subjects = [1, 2, 3, 7, 8, 9, 10, 11, 12, 13]
        config = {
            "checkpoint_metadata": {
                "outer_test_used_during_encoder_training": False,
                "smoke_test": False,
                "uschad_cv_fold": 6,
                "old_class_count": 6,
                "uschad_train_subjects": fit_subjects,
                "uschad_test_subjects": [4, 5],
            },
            "arguments": {"old_class_count": 6},
        }
        split = {
            "fit_subjects": fit_subjects,
            "eval_subjects": [4, 5],
            "old_class_ids_0based": [0, 1, 2, 3, 4, 5],
        }
        self.assertEqual(
            _validate_registered_run_metadata(config, split),
            (fit_subjects, [4, 5]),
        )

        bad_outer = {
            **config,
            "checkpoint_metadata": {
                **config["checkpoint_metadata"],
                "outer_test_used_during_encoder_training": True,
            },
        }
        with self.assertRaisesRegex(RuntimeError, "outer_test_used"):
            _validate_registered_run_metadata(bad_outer, split)

        bad_eval = {**split, "eval_subjects": [4, 6]}
        with self.assertRaisesRegex(RuntimeError, "eval subjects"):
            _validate_registered_run_metadata(config, bad_eval)

        bad_fit = {**split, "fit_subjects": fit_subjects[:-1]}
        with self.assertRaisesRegex(RuntimeError, "fit subjects"):
            _validate_registered_run_metadata(config, bad_fit)

        for metadata_update, message in (
            ({"uschad_cv_fold": 5}, "fold 6"),
            ({"old_class_count": 5}, "old_class_count=6"),
            ({"smoke_test": True}, "smoke-test"),
        ):
            with self.subTest(metadata_update=metadata_update):
                changed_config = {
                    **config,
                    "checkpoint_metadata": {
                        **config["checkpoint_metadata"],
                        **metadata_update,
                    },
                }
                with self.assertRaisesRegex(RuntimeError, message):
                    _validate_registered_run_metadata(changed_config, split)

    def test_window_indices_reject_duplicates_and_require_exact_coverage(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "duplicate"):
            _validate_global_window_indices(np.asarray([10, 11, 11]))
        clean = _validate_global_window_indices(np.asarray([10, 11, 12]))
        np.testing.assert_array_equal(clean, np.asarray([10, 11, 12]))
        _validate_trial_window_coverage(
            99, np.asarray([12, 10, 11]), np.asarray([10, 11, 12])
        )
        with self.assertRaisesRegex(RuntimeError, "source NPZ coverage"):
            _validate_trial_window_coverage(
                99, np.asarray([10, 11, 13]), np.asarray([10, 11, 12])
            )

    def test_csv_serializes_composite_fields_as_json(self) -> None:
        rows = [{"id": 1, "labels": [7, 8], "distances": {"8": 0.2, "7": 0.1}}]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rows.csv"
            _write_csv(path, rows)
            with path.open("r", encoding="utf-8-sig", newline="") as handle:
                row = next(csv.DictReader(handle))
        self.assertEqual(json.loads(row["labels"]), [7, 8])
        self.assertEqual(json.loads(row["distances"]), {"7": 0.1, "8": 0.2})


if __name__ == "__main__":
    unittest.main()
