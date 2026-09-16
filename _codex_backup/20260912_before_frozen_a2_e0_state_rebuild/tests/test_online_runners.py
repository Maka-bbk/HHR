"""Canonical single-run and CV contracts for online trajectory CGCD."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import torch

from experiments.motion_primitive import online_cv_runner as cv
from experiments.motion_primitive import online_runner as single
from models.motion_primitive_cgcd import MotionPrimitiveCGCDModel, MotionPrimitiveConfig


def _architecture() -> MotionPrimitiveConfig:
    return MotionPrimitiveConfig(
        in_channels=2,
        window_size=32,
        window_stride=16,
        feature_dim=8,
        base_channels=4,
        old_class_count=6,
        codebook_size=32,
        trajectory_input_dim=6,
        trajectory_hidden_dim=7,
    )


class OnlineSingleRunnerTests(unittest.TestCase):
    def test_public_launchers_import_only_neutral_runners(self) -> None:
        source_root = Path(single.__file__).parent
        canonical_single = (source_root / "run_online.py").read_text(encoding="utf-8")
        canonical_cv = (source_root / "run_online_cv.py").read_text(encoding="utf-8")
        self.assertIn("experiments.motion_primitive.online_runner", canonical_single)
        self.assertIn("experiments.motion_primitive.online_cv_runner", canonical_cv)
        for source in (
            canonical_single,
            canonical_cv,
            Path(single.__file__).read_text(encoding="utf-8"),
            Path(cv.__file__).read_text(encoding="utf-8"),
        ):
            self.assertNotIn("run_happy_online", source)
            self.assertNotIn("happy_online", source)
            self.assertNotIn("train_happy", source)
        self.assertEqual(cv.ONLINE_SCRIPT.name, "run_online.py")

    def _offline_artifacts(self, root: Path, mode: str) -> Path:
        config = _architecture()
        model = MotionPrimitiveCGCDModel(config)
        manifest = {
            "schema": single.OFFLINE_MANIFEST_SCHEMA,
            "profile": single.PROFILE_JOINT,
            "architecture": config.audit_dict(),
            "old_classes_physical": list(range(6)),
            "encoder_initialization": {
                "mode": mode,
                "checkpoint": "source.pt" if mode == "warmstart" else None,
            },
        }
        checkpoint = {
            "schema": single.OFFLINE_CHECKPOINT_SCHEMA,
            "profile": single.PROFILE_JOINT,
            "architecture": config.audit_dict(),
            "old_classes_physical": list(range(6)),
            "encoder_initialization": mode,
            "selection_head": "trajectory",
            "selection_metric": "macro_f1",
            "test_metrics_used_for_selection": False,
            "model": model.state_dict(),
        }
        (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        path = root / "checkpoint_best_trajectory.pt"
        torch.save(checkpoint, path)
        return path

    def test_loader_accepts_only_explicit_random_or_warmstart(self) -> None:
        for mode in ("random", "warmstart"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                checkpoint = self._offline_artifacts(root, mode)
                model, manifest, payload, _ = single.load_offline_model(
                    root, checkpoint, device=torch.device("cpu")
                )
                self.assertEqual(model.class_count, 6)
                self.assertEqual(manifest["encoder_initialization"]["mode"], mode)
                self.assertEqual(payload["encoder_initialization"], mode)

                manifest.pop("encoder_initialization")
                (root / "manifest.json").write_text(
                    json.dumps(manifest), encoding="utf-8"
                )
                with self.assertRaisesRegex(RuntimeError, "explicitly record"):
                    single.load_offline_model(
                        root, checkpoint, device=torch.device("cpu")
                    )

    def test_fixed_delta_is_available_only_as_explicit_ablation(self) -> None:
        args = single.build_parser().parse_args(
            [
                "--offline-run-dir", "offline",
                "--checkpoint", "checkpoint.pt",
                "--output-dir", "online",
                "--codebook-expansion", "fixed_delta",
            ]
        )
        self.assertIsNotNone(
            single.build_codebook_expansion_policy(args, single.PROFILE_JOINT)
        )


class OnlineCVRunnerTests(unittest.TestCase):
    def test_member_command_is_trajectory_only_and_forwards_adaptive_policy(self) -> None:
        args = cv.build_parser().parse_args(
            [
                "--offline-cv-root", "offline",
                "--output-root", "online",
                "--python-executable", sys.executable,
            ]
        )
        command = cv.build_member_command(
            args,
            profile=cv.PROFILE_JOINT,
            fold=1,
            seed=0,
            offline_run_dir=Path("offline-member"),
            checkpoint=Path("checkpoint_best_trajectory.pt"),
            output_dir=Path("online-member"),
        )
        joined = " ".join(command)
        required = (
            "--codebook-expansion residual_adaptive",
            "--expected-offline-codebook-size 32",
            "--primitive-feature-distillation-weight 1.0",
            "--old-codebook-anchor-weight 1.0",
            "--codebook-minimum-cluster-trials 3",
            "--codebook-minimum-cluster-subjects 2",
        )
        for value in required:
            self.assertIn(value, joined)
        for forbidden in ("pooled", "proto-aug", "fusion", "happy-distillation"):
            self.assertNotIn(forbidden, joined)

    def test_single_and_cv_defaults_match_retention_and_support_gates(self) -> None:
        single_args = single.build_parser().parse_args(
            [
                "--offline-run-dir", "offline",
                "--checkpoint", "checkpoint.pt",
                "--output-dir", "online",
            ]
        )
        cv_args = cv.build_parser().parse_args(
            ["--offline-cv-root", "offline", "--output-root", "online"]
        )
        for name in (
            "primitive_feature_distillation_weight",
            "old_codebook_anchor_weight",
            "codebook_minimum_cluster_trials",
            "codebook_minimum_cluster_subjects",
        ):
            self.assertEqual(getattr(single_args, name), getattr(cv_args, name))

    def test_grid_validation_requires_exactly_seven_by_four_by_three(self) -> None:
        rows = [
            {"fold": fold, "seed": seed, "session": session}
            for fold in range(1, 8)
            for seed in (0, 5, 50, 500)
            for session in range(1, 4)
        ]
        cv.validate_online_grid(rows, tuple(range(1, 8)), (0, 5, 50, 500))
        with self.assertRaises(RuntimeError):
            cv.validate_online_grid(
                rows + [rows[0]], tuple(range(1, 8)), (0, 5, 50, 500)
            )


if __name__ == "__main__":
    unittest.main()
