from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from experiments.motion_primitive.export_trajectory_visuals import (
    _enrich_runs,
    _overlap_average,
    align_online_predictions,
    plot_activity_codebook_heatmap,
    plot_confusion,
    plot_trajectory_panel,
    segmentation_summary,
    write_codebook_usage,
    write_transition_counts,
)


def _records():
    values = []
    for label, activity in ((0, "Walking Forward"), (1, "Walking Left")):
        for subject in (10, 11):
            tokens = [label, label, 2, 2]
            values.append(
                {
                    "trial_id": label * 10 + subject,
                    "subject_id": subject,
                    "trial_number": 1,
                    "activity": activity,
                    "model_label": label,
                    "window_count": 4,
                    "primitive_count": 2,
                    "run_count": 2,
                    "boundary_count": 1,
                    "boundary_rate": 1 / 3,
                    "used_code_ids": sorted(set(tokens)),
                    "used_code_count": len(set(tokens)),
                    "primitive_ids_per_window": tokens,
                    "window_token_sequence": tokens,
                    "primitive_sequence": [label, 2],
                    "primitive_transitions": [
                        {
                            "transition_index": 0,
                            "source_primitive_id": label,
                            "target_primitive_id": 2,
                        }
                    ],
                    "primitive_runs": [
                        {
                            "primitive_id": label,
                            "dominant_primitive_id": label,
                            "start_window_index": 0,
                            "end_window_index_exclusive": 2,
                            "duration_windows": 2,
                        },
                        {
                            "primitive_id": 2,
                            "dominant_primitive_id": 2,
                            "start_window_index": 2,
                            "end_window_index_exclusive": 4,
                            "duration_windows": 2,
                        },
                    ],
                    "trajectory_prediction_raw": label,
                    "trajectory_confidence": 0.9,
                }
            )
    return values


class TrajectoryVisualExportTests(unittest.TestCase):
    def test_runs_receive_window_and_sample_boundaries(self):
        result = _enrich_runs(
            {
                "runs": [
                    {
                        "token": 3,
                        "start_token_index": 0,
                        "end_token_index_exclusive": 2,
                    },
                    {
                        "token": 4,
                        "start_token_index": 2,
                        "end_token_index_exclusive": 3,
                    },
                ]
            },
            np.asarray([0, 128, 256]),
            256,
        )
        self.assertEqual(result[0]["duration_windows"], 2)
        self.assertEqual(result[0]["end_sample_exclusive"], 256)
        self.assertEqual(result[1]["start_sample"], 256)
        self.assertEqual(result[1]["end_sample_exclusive"], 512)

    def test_overlap_average_reconstructs_shared_samples(self):
        windows = np.asarray([[[1, 1, 1]], [[3, 3, 3]]], dtype=float)
        signal, origin = _overlap_average(windows, [10, 12])
        self.assertEqual(origin, 10)
        np.testing.assert_allclose(signal, [[1, 1, 2, 3, 3]])

    def test_online_alignment_keeps_old_rows_fixed(self):
        records = _records()
        # Old class 0 stays fixed; novel classifier rows 1 and 2 are swapped.
        for item in records:
            item["model_label"] += 1
            item["trajectory_prediction_raw"] = 3 - item["model_label"]
        mappings = align_online_predictions(
            records, class_count=3, old_class_count=1
        )
        self.assertEqual(sorted(mappings), [[1, 2], [2, 1]])
        self.assertTrue(
            all(
                item["trajectory_prediction"] == item["model_label"]
                for item in records
            )
        )

    def test_segmentation_diagnostics_report_one_run_boundary_and_usage(self):
        records = _records()
        diagnostics = segmentation_summary(records, codebook_size=4)["overall"]
        self.assertEqual(diagnostics["motion_primitive_run_count"], 8)
        self.assertEqual(diagnostics["one_run_fraction"], 0.0)
        self.assertAlmostEqual(diagnostics["boundary_rate"], 1 / 3)
        self.assertEqual(diagnostics["used_code_ids"], [0, 1, 2])
        self.assertFalse(diagnostics["requires_segmentation_review"])

    def test_collapsed_segmentation_is_flagged_without_rejecting_one_run_trials(self):
        records = _records()
        for item in records:
            item["window_token_sequence"] = [3, 3, 3, 3]
            item["used_code_ids"] = [3]
            item["run_count"] = 1
            item["primitive_count"] = 1
            item["primitive_runs"] = [
                {
                    "primitive_id": 3,
                    "dominant_primitive_id": 3,
                    "duration_windows": 4,
                }
            ]
        diagnostics = segmentation_summary(records, codebook_size=4)["overall"]
        self.assertTrue(diagnostics["diagnostic_flags"]["single_code_used"])
        self.assertTrue(diagnostics["diagnostic_flags"]["all_trials_one_run"])
        self.assertTrue(diagnostics["requires_segmentation_review"])

    def test_three_core_figures_are_rendered(self):
        records = _records()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plot_trajectory_panel(
                records,
                root / "trajectory.png",
                codebook_size=4,
                per_activity=2,
            )
            plot_activity_codebook_heatmap(
                records,
                root / "codebook.png",
                root / "codebook.csv",
                codebook_size=4,
            )
            plot_confusion(records, root / "confusion.png")
            write_codebook_usage(records, root / "usage.csv", 4)
            write_transition_counts(records, root / "transitions.csv")
            for name in (
                "trajectory.png",
                "codebook.png",
                "codebook.csv",
                "confusion.png",
                "usage.csv",
                "transitions.csv",
            ):
                self.assertTrue((root / name).is_file())
                self.assertGreater((root / name).stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
