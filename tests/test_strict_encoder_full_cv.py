from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import torch

from experiments.motion_primitive import pretrain_window_encoder as window_stage
from experiments.motion_primitive import run_full_cv
from experiments.motion_primitive import strict_encoder_cv_runner as encoder_cv
from experiments.motion_primitive import strict_offline_cv_runner
from experiments.motion_primitive import strict_online_cv_runner
from experiments.motion_primitive import strict_visualization
from experiments.motion_primitive import train_motion_encoder as a2_stage
from experiments.motion_primitive.strict_cv_common import (
    CANONICAL_FOLDS,
    CANONICAL_SEEDS,
    validate_or_create_grid_manifest,
)
from experiments.motion_primitive.strict_protocol import sha256_file


def _encoder_args(npz: Path, output: Path, **updates: object) -> argparse.Namespace:
    values = vars(encoder_cv.build_parser().parse_args([
        "--npz-path", str(npz),
        "--output-root", str(output),
        "--dry-run",
    ]))
    values.update(updates)
    return argparse.Namespace(**values)


class EncoderCommandContractTests(unittest.TestCase):
    def test_formal_dry_run_emits_56_parseable_child_commands(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            npz = base / "windows.npz"
            npz.write_bytes(b"identity-only-test")
            result = encoder_cv.run(_encoder_args(npz, base / "out"))
            commands = result["commands"]
            self.assertEqual(len(commands), 56)
            window_count = 0
            a2_count = 0
            for command in commands:
                script = Path(command[1]).name
                if script == "pretrain_window_encoder.py":
                    window_stage.build_parser().parse_args(command[2:])
                    window_count += 1
                elif script == "train_motion_encoder.py":
                    a2_stage.build_parser().parse_args(command[2:])
                    a2_count += 1
                else:
                    self.fail(f"Unexpected encoder child script: {script}")
            self.assertEqual((window_count, a2_count), (28, 28))

    def test_noncanonical_encoder_grid_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            npz = base / "windows.npz"
            npz.write_bytes(b"x")
            with self.assertRaisesRegex(ValueError, "fixed to folds"):
                encoder_cv.run(
                    _encoder_args(npz, base / "out", folds="1", seeds="0")
                )

    def test_w128_confirmation_dry_run_is_exactly_fourteen_encoders(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            npz = base / "windows.npz"
            npz.write_bytes(b"identity-only-test")
            result = encoder_cv.run(
                _encoder_args(
                    npz,
                    base / "out",
                    protocol=encoder_cv.W128_CONFIRMATION_PROTOCOL,
                    folds="1,2,3,4,5,6,7",
                    seeds="0,5",
                    window_epochs=60,
                    window_batch_size=256,
                    window_eval_batch_size=1024,
                    a2_epochs=30,
                    a2_trial_batch_size=8,
                    a2_source_encode_batch_size=1024,
                )
            )
            self.assertEqual(result["member_count"], 14)
            self.assertEqual(len(result["commands"]), 28)
            self.assertEqual((result["window_size"], result["window_stride"]), (128, 64))
            pairs: set[tuple[int, int]] = set()
            for command in result["commands"]:
                script = Path(command[1]).name
                if script == "pretrain_window_encoder.py":
                    child = window_stage.build_parser().parse_args(command[2:])
                    self.assertEqual((child.window_size, child.window_stride), (128, 64))
                    self.assertEqual((child.epochs, child.batch_size, child.eval_batch_size), (60, 256, 1024))
                    pairs.add((int(child.fold), int(child.seed)))
                elif script == "train_motion_encoder.py":
                    child = a2_stage.build_parser().parse_args(command[2:])
                    self.assertEqual((child.epochs, child.trial_batch_size, child.source_encode_batch_size), (30, 8, 1024))
                else:
                    self.fail(f"Unexpected encoder child script: {script}")
            self.assertEqual(
                pairs,
                {(fold, seed) for fold in range(1, 8) for seed in (0, 5)},
            )

    def test_w128_confirmation_rejects_wrong_grid_and_training_recipe(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            npz = base / "windows.npz"
            npz.write_bytes(b"x")
            common = {
                "protocol": encoder_cv.W128_CONFIRMATION_PROTOCOL,
                "folds": "1,2,3,4,5,6,7",
                "seeds": "0,5",
                "window_epochs": 60,
                "window_batch_size": 256,
                "window_eval_batch_size": 1024,
                "a2_epochs": 30,
                "a2_trial_batch_size": 8,
                "a2_source_encode_batch_size": 1024,
            }
            with self.assertRaisesRegex(ValueError, "fixed to folds"):
                encoder_cv.run(
                    _encoder_args(npz, base / "wrong-grid", **{**common, "folds": "1,2,3"})
                )
            with self.assertRaisesRegex(ValueError, "locked to the original screening"):
                encoder_cv.run(
                    _encoder_args(npz, base / "wrong-batch", **{**common, "window_batch_size": 128})
                )

    def test_nonempty_unidentified_root_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "out"
            root.mkdir()
            (root / "orphan.txt").write_text("unknown", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "no grid_manifest"):
                validate_or_create_grid_manifest(root, {"schema": "test"})


class MemberIdentityTests(unittest.TestCase):
    def test_window_validator_binds_full_identity_and_checkpoint_hash(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            npz = base / "windows.npz"
            npz.write_bytes(b"npz")
            target = base / "member"
            target.mkdir()
            arguments = window_stage.build_parser().parse_args([
                "--npz-path", str(npz),
                "--output-dir", str(target),
                "--fold", "1",
                "--seed", "0",
                "--device", "cpu",
            ])
            identity = window_stage._run_identity(arguments)
            checkpoint = target / "model_best.pt"
            torch.save({"run_identity": identity}, checkpoint)
            complete = {
                "schema": encoder_cv.WINDOW_SCHEMA,
                "identity": identity,
                "fold": 1,
                "seed": 0,
                "checkpoint": checkpoint.name,
                "checkpoint_sha256": sha256_file(checkpoint),
                "complete": True,
            }
            (target / "complete.json").write_text(json.dumps(complete), encoding="utf-8")
            self.assertIsNotNone(encoder_cv._validate_window_member(
                target,
                fold=1,
                seed=0,
                npz_sha256=sha256_file(npz),
                expected_identity=identity,
            ))
            changed = json.loads(json.dumps(identity))
            changed["arguments"]["learning_rate"] = 0.2
            with self.assertRaisesRegex(RuntimeError, "another complete run identity"):
                encoder_cv._validate_window_member(
                    target,
                    fold=1,
                    seed=0,
                    npz_sha256=sha256_file(npz),
                    expected_identity=changed,
                )
            checkpoint.write_bytes(b"changed")
            with self.assertRaisesRegex(RuntimeError, "absent or has changed"):
                encoder_cv._validate_window_member(
                    target,
                    fold=1,
                    seed=0,
                    npz_sha256=sha256_file(npz),
                    expected_identity=identity,
                )

    def test_a2_validator_binds_full_identity_and_checkpoint_hash(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            target = Path(temporary)
            identity = {
                "schema": a2_stage.RUN_IDENTITY_SCHEMA,
                "arguments": {"learning_rate": 1e-4, "epochs": 30},
                "resolved_training_config": {"loss_weights": {"changepoint": 1.0}},
                "resolved_device": "cpu",
            }
            checkpoint = target / "motion_encoder_final.pt"
            torch.save({"run_identity": identity}, checkpoint)
            complete = {
                "schema": encoder_cv.A2_COMPLETE_SCHEMA,
                "identity": identity,
                "fold": 2,
                "seed": 5,
                "npz_sha256": "n" * 64,
                "source_checkpoint_sha256": "s" * 64,
                "ablation_profile": "A2",
                "selection_policy": "final_epoch",
                "final_checkpoint_sha256": sha256_file(checkpoint),
                "complete": True,
            }
            (target / "complete.json").write_text(json.dumps(complete), encoding="utf-8")
            self.assertIsNotNone(encoder_cv._validate_a2_member(
                target,
                fold=2,
                seed=5,
                npz_sha256="n" * 64,
                source_checkpoint_sha256="s" * 64,
                expected_identity=identity,
            ))
            changed = json.loads(json.dumps(identity))
            changed["arguments"]["epochs"] = 31
            with self.assertRaisesRegex(RuntimeError, "another complete run identity"):
                encoder_cv._validate_a2_member(
                    target,
                    fold=2,
                    seed=5,
                    npz_sha256="n" * 64,
                    source_checkpoint_sha256="s" * 64,
                    expected_identity=changed,
                )
            checkpoint.write_bytes(b"changed")
            with self.assertRaisesRegex(RuntimeError, "absent or has changed"):
                encoder_cv._validate_a2_member(
                    target,
                    fold=2,
                    seed=5,
                    npz_sha256="n" * 64,
                    source_checkpoint_sha256="s" * 64,
                    expected_identity=identity,
                )

    def test_a2_identity_changes_with_training_parameters_and_resolved_device(self) -> None:
        parser = a2_stage.build_parser()
        first = parser.parse_args([
            "--source-checkpoint", "source.pt",
            "--npz-path", "windows.npz",
            "--learning-rate", "0.0001",
            "--device", "cpu",
        ])
        second = parser.parse_args([
            "--source-checkpoint", "source.pt",
            "--npz-path", "windows.npz",
            "--learning-rate", "0.0002",
            "--device", "cpu",
        ])
        common = {
            "source_path": Path("source.pt").resolve(),
            "source_sha256": "s" * 64,
            "npz_path": Path("windows.npz").resolve(),
            "npz_sha256": "n" * 64,
            "fold": 1,
            "resolved_config": {"profile": "A2"},
            "resolved_device": torch.device("cpu"),
        }
        self.assertNotEqual(
            a2_stage._build_run_identity(first, **common),
            a2_stage._build_run_identity(second, **common),
        )

    def test_member_set_rejects_extra_malformed_and_missing_directories(self) -> None:
        expected = {(1, 0), (1, 5)}
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "fold_01_seed_0").mkdir()
            encoder_cv._validate_member_set(root, expected, allow_missing=True)
            with self.assertRaisesRegex(RuntimeError, "missing"):
                encoder_cv._validate_member_set(root, expected, allow_missing=False)
            (root / "fold_bad").mkdir()
            with self.assertRaisesRegex(RuntimeError, "malformed"):
                encoder_cv._validate_member_set(root, expected, allow_missing=True)
            (root / "fold_bad").rmdir()
            (root / "fold_07_seed_500").mkdir()
            with self.assertRaisesRegex(RuntimeError, "extra"):
                encoder_cv._validate_member_set(root, expected, allow_missing=True)


class FullDriverContractTests(unittest.TestCase):
    def test_full_dry_run_commands_are_accepted_by_target_parsers(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            npz = base / "windows.npz"
            npz.write_bytes(b"identity-only-test")
            args = run_full_cv.build_parser().parse_args([
                "--npz-path", str(npz),
                "--output-root", str(base / "full"),
                "--device", "cpu",
                "--dry-run",
            ])
            commands = run_full_cv.run(args)["commands"]
            self.assertEqual(len(commands), 4)
            parsers = {
                "strict_encoder_cv_runner.py": encoder_cv.build_parser,
                "strict_offline_cv_runner.py": strict_offline_cv_runner.build_parser,
                "strict_online_cv_runner.py": strict_online_cv_runner.build_parser,
                "strict_visualization.py": strict_visualization.build_parser,
            }
            for command in commands:
                script = Path(command[1]).name
                self.assertIn(script, parsers)
                parsers[script]().parse_args(command[2:])

    def test_source_hash_coverage_includes_every_executed_stage_and_core(self) -> None:
        full = set(run_full_cv._implementation_hashes())
        self.assertTrue({
            "experiments/motion_primitive/run_full_cv.py",
            "experiments/motion_primitive/strict_encoder_cv_runner.py",
            "experiments/motion_primitive/pretrain_window_encoder.py",
            "experiments/motion_primitive/train_motion_encoder.py",
            "experiments/motion_primitive/strict_offline_cv_runner.py",
            "experiments/motion_primitive/strict_offline_runner.py",
            "experiments/motion_primitive/strict_online_cv_runner.py",
            "experiments/motion_primitive/strict_online_runner.py",
            "experiments/motion_primitive/strict_artifacts.py",
            "experiments/motion_primitive/frozen_e0_state.py",
            "experiments/motion_primitive/motion_encoder.py",
            "models/resnet1d.py",
        }.issubset(full))
        online = set(strict_online_cv_runner._implementation_hashes())
        self.assertTrue({
            "experiments/motion_primitive/strict_online_cv_runner.py",
            "experiments/motion_primitive/strict_online_runner.py",
            "experiments/motion_primitive/strict_registry.py",
            "experiments/motion_primitive/strict_metrics.py",
            "experiments/motion_primitive/strict_artifacts.py",
            "experiments/motion_primitive/frozen_e0_state.py",
        }.issubset(online))

    def test_run_full_help_succeeds(self) -> None:
        script = Path(run_full_cv.__file__).resolve()
        completed = subprocess.run(
            [sys.executable, str(script), "--help"],
            cwd=run_full_cv.PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("--npz-path", completed.stdout)


if __name__ == "__main__":
    unittest.main()
