"""Canonical cross-validation launcher contracts for offline A2-MP."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

from experiments.motion_primitive.offline_cv_runner import (
    CV_SCHEMA,
    PROFILE_JOINT,
    _command_value,
    build_member_command,
    find_window_pretrain_checkpoint,
    registered_subject_split,
    selection_heads_for_profile,
    validate_args,
)
from experiments.motion_primitive.offline_cv_runner import build_parser


class OfflineCVRunnerTests(unittest.TestCase):
    def test_registered_split_is_subject_disjoint_and_trajectory_selected(self) -> None:
        self.assertEqual(CV_SCHEMA, "hhr_motion_primitive_offline_subject_cv_v3")
        split = registered_subject_split(1)
        self.assertEqual(
            (len(split.train), len(split.validation), len(split.outer_test)),
            (10, 2, 2),
        )
        self.assertFalse(set(split.train) & set(split.validation))
        self.assertFalse(set(split.train) & set(split.outer_test))
        self.assertFalse(set(split.validation) & set(split.outer_test))
        self.assertEqual(selection_heads_for_profile(PROFILE_JOINT), ("trajectory",))

    def test_launcher_requires_initialization_and_forwards_registered_route(self) -> None:
        parser = build_parser()
        self.assertTrue(parser._option_string_actions["--encoder-initialization"].required)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            npz = root / "data.npz"
            npz.touch()
            args = parser.parse_args(
                [
                    "--npz-path", str(npz),
                    "--output-root", str(root / "out"),
                    "--python-executable", sys.executable,
                    "--encoder-initialization", "random",
                ]
            )
            command = build_member_command(
                args,
                profile=PROFILE_JOINT,
                split=registered_subject_split(1),
                seed=0,
                checkpoint=None,
                output_dir=root / "member",
            )
        expected = {
            "--window-size": "256",
            "--window-stride": "128",
            "--selection-head": "trajectory",
            "--changepoint-weight": "1.0",
            "--content-boundary-alignment-weight": "0.1",
            "--noncollapse-weight": "0.05",
            "--temporal-prediction-weight": "0.5",
            "--changepoint-absolute-floor": "0.01",
            "--changepoint-null-mad-multiplier": "3.0",
            "--utilization-weight": "0.0",
            "--assignment-confidence-weight": "0.0",
            "--codebook-diversity-weight": "0.0",
        }
        for option, value in expected.items():
            self.assertEqual(_command_value(command, option), value)
        joined = " ".join(command).lower()
        for forbidden in ("pooled", "fusion", "infonce"):
            self.assertNotIn(forbidden, joined)

    def test_cv_rejects_dirty_anchor_motion_ramp_and_random_lr_drift(self) -> None:
        parser = build_parser()
        with self.assertRaises(SystemExit):
            parser.parse_args(
                [
                    "--npz-path", __file__,
                    "--output-root", "out",
                    "--encoder-initialization", "random",
                    "--har-weak-jitter-std", "0.01",
                ]
            )
        for extra in (
            ["--motion-ramp-end-epoch", "1"],
            ["--encoder-lr-scale", "0.1"],
        ):
            args = parser.parse_args(
                [
                    "--npz-path", __file__,
                    "--output-root", "out",
                    "--encoder-initialization", "random",
                    *extra,
                ]
            )
            with self.assertRaises(ValueError):
                validate_args(args)

    def test_historical_a2_backbone_checkpoint_layout_is_discovered(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            expected = root / "fold_01_seed_5_A2_formal_v1" / "motion_encoder_best.pt"
            expected.parent.mkdir(parents=True)
            expected.touch()
            self.assertEqual(
                find_window_pretrain_checkpoint(root, fold=1, seed=5),
                expected.resolve(),
            )


if __name__ == "__main__":
    unittest.main()
