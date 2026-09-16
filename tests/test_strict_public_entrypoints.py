from __future__ import annotations

import contextlib
import ast
import io
import subprocess
import sys
import unittest
from pathlib import Path

from experiments.motion_primitive import (
    legacy_profiles,
    profiles,
    run_offline_cv,
    run_online,
    run_online_cv,
    train_offline,
)
from experiments.motion_primitive import offline_cv_runner as legacy_offline_cv
from experiments.motion_primitive import online_cv_runner as legacy_online_cv
from experiments.motion_primitive.strict_offline_runner import PROFILE


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class PublicProfileTests(unittest.TestCase):
    def test_only_frozen_a2_e0_state_k32_is_public(self) -> None:
        self.assertEqual(PROFILE, "frozen_a2_e0_state_k32")
        self.assertEqual(profiles.PROFILE, PROFILE)
        self.assertEqual(profiles.PROFILES, (PROFILE,))
        self.assertEqual(profiles.normalize_profile(PROFILE), PROFILE)
        self.assertFalse(hasattr(profiles, "PROFILE_JOINT"))
        with self.assertRaisesRegex(ValueError, "Unknown profile"):
            profiles.normalize_profile("motion_primitive_joint")

    def test_joint_profile_remains_explicitly_legacy(self) -> None:
        self.assertEqual(legacy_profiles.PROFILES, ("motion_primitive_joint",))
        self.assertEqual(
            legacy_profiles.normalize_profile("motion_primitive_joint"),
            "motion_primitive_joint",
        )
        self.assertEqual(legacy_offline_cv.TRAIN_SCRIPT.name, "offline_trainer.py")
        self.assertEqual(legacy_online_cv.ONLINE_SCRIPT.name, "online_runner.py")
        legacy_sources = (
            "offline_trainer.py",
            "offline_cv_runner.py",
            "online_runner.py",
            "online_cv_runner.py",
            "export_trajectory_visuals.py",
        )
        source_root = PROJECT_ROOT / "experiments" / "motion_primitive"
        for filename in legacy_sources:
            text = (source_root / filename).read_text(encoding="utf-8")
            self.assertIn("Legacy", ast.get_docstring(ast.parse(text)) or "", filename)
            self.assertIn("experiments.motion_primitive.legacy_profiles", text, filename)
            self.assertNotIn("from experiments.motion_primitive.profiles import", text, filename)
        exporter = (source_root / "export_trajectory_visuals.py").read_text(encoding="utf-8")
        self.assertIn("from experiments.motion_primitive.offline_trainer import", exporter)
        self.assertNotIn("from experiments.motion_primitive.train_offline import", exporter)


class PublicBindingTests(unittest.TestCase):
    def test_four_public_wrappers_bind_only_to_strict_runners(self) -> None:
        expected = {
            train_offline: "experiments.motion_primitive.strict_offline_runner",
            run_offline_cv: "experiments.motion_primitive.strict_offline_cv_runner",
            run_online: "experiments.motion_primitive.strict_online_runner",
            run_online_cv: "experiments.motion_primitive.strict_online_cv_runner",
        }
        for wrapper, module_name in expected.items():
            with self.subTest(wrapper=wrapper.__name__):
                self.assertEqual(wrapper.main.__module__, module_name)
                self.assertEqual(wrapper.run.__module__, module_name)
                self.assertEqual(wrapper.build_parser.__module__, module_name)
        self.assertFalse(hasattr(train_offline, "build_model"))
        self.assertFalse(hasattr(train_offline, "training_step"))

    def test_all_public_scripts_have_working_strict_help(self) -> None:
        scripts = {
            "train_offline.py": ("frozen_a2_e0_state_k32", "Build frozen A2/E0/state/K32"),
            "run_offline_cv.py": ("--a2-root", "Frozen A2/E0/state offline subject CV"),
            "run_online.py": ("frozen_a2_e0_state_k32", "Strict three-session"),
            "run_online_cv.py": ("--offline-cv-root", "canonical 7x4x3 CV"),
        }
        source = PROJECT_ROOT / "experiments" / "motion_primitive"
        for filename, required in scripts.items():
            with self.subTest(script=filename):
                result = subprocess.run(
                    [sys.executable, str(source / filename), "--help"],
                    cwd=PROJECT_ROOT,
                    capture_output=True,
                    text=True,
                    check=False,
                )
                self.assertEqual(result.returncode, 0, result.stderr)
                for token in required:
                    self.assertIn(token, result.stdout)
                self.assertNotIn("motion_primitive_joint", result.stdout)
                self.assertNotIn("--motion-weight", result.stdout)
                self.assertNotIn("--codebook-expansion", result.stdout)


class LegacyArgumentRejectionTests(unittest.TestCase):
    def _assert_parse_rejected(self, parser, arguments: list[str]) -> None:
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as captured:
                parser.parse_args(arguments)
        self.assertEqual(captured.exception.code, 2)

    def test_old_profile_is_rejected_by_single_member_public_entries(self) -> None:
        self._assert_parse_rejected(
            train_offline.build_parser(),
            [
                "--a2-checkpoint", "a.pt",
                "--npz-path", "data.npz",
                "--output-dir", "out",
                "--fold", "1",
                "--seed", "0",
                "--profile", "motion_primitive_joint",
            ],
        )
        self._assert_parse_rejected(
            run_online.build_parser(),
            [
                "--offline-run-dir", "offline",
                "--output-dir", "online",
                "--fold", "1",
                "--seed", "0",
                "--profile", "motion_primitive_joint",
            ],
        )

    def test_old_training_and_grid_arguments_are_not_silently_accepted(self) -> None:
        self._assert_parse_rejected(
            train_offline.build_parser(),
            [
                "--a2-checkpoint", "a.pt",
                "--npz-path", "data.npz",
                "--output-dir", "out",
                "--fold", "1",
                "--seed", "0",
                "--motion-weight", "1.0",
            ],
        )
        self._assert_parse_rejected(
            run_offline_cv.build_parser(),
            [
                "--a2-root", "a2",
                "--npz-path", "data.npz",
                "--output-root", "out",
                "--profiles", "motion_primitive_joint",
            ],
        )
        self._assert_parse_rejected(
            run_online.build_parser(),
            [
                "--offline-run-dir", "offline",
                "--output-dir", "online",
                "--fold", "1",
                "--seed", "0",
                "--codebook-expansion", "residual_adaptive",
            ],
        )
        self._assert_parse_rejected(
            run_online_cv.build_parser(),
            [
                "--offline-cv-root", "offline",
                "--output-root", "online",
                "--profiles", "motion_primitive_joint",
            ],
        )

    def test_strict_fixed_route_overrides_are_explicitly_rejected(self) -> None:
        parsed = train_offline.build_parser().parse_args(
            [
                "--a2-checkpoint", "a.pt",
                "--npz-path", "data.npz",
                "--output-dir", "out",
                "--fold", "1",
                "--seed", "0",
                "--primitive-num", "64",
            ]
        )
        with self.assertRaisesRegex(ValueError, "pinned to K32"):
            train_offline.validate_args(parsed)


if __name__ == "__main__":
    unittest.main()
