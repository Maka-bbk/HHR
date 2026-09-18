from __future__ import annotations

import ast
import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import torch

from experiments.motion_primitive import pretrain_window_encoder, train_motion_encoder
from experiments.motion_primitive import run_window_codebook_batch_proxy_cv as batch_cv
from experiments.motion_primitive import run_trajectory_descriptor_ablation_cv as descriptor_cv
from experiments.motion_primitive import strict_protocol
from experiments.motion_primitive import window_codebook_batch_proxy as batch_member
from experiments.motion_primitive.frozen_e0 import (
    DURATION_INVARIANT_DESCRIPTOR_PROFILE,
    DURATION_SOFT_SUBJECT_A025_PROFILE,
    DURATION_SOFT_SUBJECT_A050_PROFILE,
    DURATION_SOFT_SUBJECT_A075_PROFILE,
    FULL_DEBIASED_DESCRIPTOR_PROFILE,
    GRAVITY_DESCRIPTOR_PROFILE,
    LEGACY_DESCRIPTOR_PROFILE,
    SUBJECT_DEBIASED_DESCRIPTOR_PROFILE,
    DescriptorTransform,
    PrimitiveTrajectory,
    fit_frozen_e0_codebook,
    load_frozen_artifacts,
    raw_descriptor_dimension,
    save_frozen_artifacts,
    statistic_names,
    trajectory_descriptor,
)
from experiments.motion_primitive.frozen_e0_state import (
    WindowTrial as RegisteredWindowTrial,
    fit_e0_codebook as fit_registered_e0_codebook,
)
from experiments.motion_primitive.strict_artifacts import configure_deterministic_runtime
from experiments.motion_primitive.strict_protocol import SensorTrial
from experiments.motion_primitive.strict_cv_common import canonical_hash


EXPECTED_ARMS = (
    (64, 32, 128, "w64_s32_k128"),
    (128, 64, 64, "w128_s64_k64"),
    (128, 64, 128, "w128_s64_k128"),
)
EXPECTED_STATE_DIMS = {32: 3_739, 64: 13_563, 128: 51_643}


def _sensor_trial(trial_id: int, *, window_size: int = 64) -> SensorTrial:
    windows = np.zeros((1, 6, int(window_size)), dtype=np.float32)
    return SensorTrial(
        trial_id=int(trial_id),
        subject_id=10 + int(trial_id) % 2,
        windows=windows.copy(),
        raw_windows=windows.copy(),
        window_starts=np.asarray([0], dtype=np.int64),
    ).validate(int(window_size))


def _batch_protocol(*, duplicate: bool = False, missing: bool = False) -> SimpleNamespace:
    trials = [_sensor_trial(index) for index in range(120)]
    first = tuple(trials[:22])
    second = tuple(trials[22:48])
    third = tuple(trials[48:78])
    evaluation = tuple(trials[78:120])
    if duplicate:
        evaluation = (trials[0],) + evaluation[1:]
    if missing:
        evaluation = evaluation[:-1]
    sessions = (
        SimpleNamespace(session=1, incoming=first, evaluation=()),
        SimpleNamespace(session=2, incoming=second, evaluation=()),
        SimpleNamespace(session=3, incoming=third, evaluation=evaluation),
    )
    return SimpleNamespace(
        sessions=sessions,
        split=SimpleNamespace(outer_test=(10, 11)),
    )


class ExperimentGridContractTests(unittest.TestCase):
    def test_default_three_arms_are_exactly_the_requested_scale_capacity_configs(self) -> None:
        arms = batch_cv.parse_arms(batch_cv.DEFAULT_ARMS)
        observed = tuple(
            (
                int(arm.window_size),
                int(arm.window_stride),
                int(arm.primitive_num),
                str(arm.config_id),
            )
            for arm in arms
        )
        self.assertEqual(observed, EXPECTED_ARMS)
        self.assertTrue(all(isinstance(arm, batch_cv.ExperimentArm) for arm in arms))
        self.assertEqual(
            set(batch_member.ARM_BY_PAIR),
            {(64, 128), (128, 64), (128, 128), (256, 32)},
        )
        self.assertTrue(
            all(arm.window_stride * 2 == arm.window_size for arm in arms)
        )
        # W128/K64 versus W128/K128 is the one paired capacity-only contrast.
        w128 = [arm for arm in arms if int(arm.window_size) == 128]
        self.assertEqual(
            {(int(arm.window_stride), int(arm.primitive_num)) for arm in w128},
            {(64, 64), (64, 128)},
        )

    def test_three_folds_two_seeds_expand_to_exactly_eighteen_unique_members(self) -> None:
        arms = batch_cv.parse_arms(batch_cv.DEFAULT_ARMS)
        normalized_arms, folds, seeds = batch_cv.validate_experiment_grid(
            arms, (1, 2, 3), (0, 5)
        )
        self.assertEqual(tuple(folds), (1, 2, 3))
        self.assertEqual(tuple(seeds), (0, 5))
        self.assertEqual(tuple(normalized_arms), tuple(arms))
        keys = batch_cv.expected_member_keys(normalized_arms, folds, seeds)
        expected = {
            (config_id, fold, seed)
            for _, _, _, config_id in EXPECTED_ARMS
            for fold in (1, 2, 3)
            for seed in (0, 5)
        }
        self.assertEqual(keys, expected)
        self.assertEqual(len(keys), 18)

    def test_arm_and_grid_validation_rejects_silent_protocol_drift(self) -> None:
        arms = batch_cv.parse_arms(batch_cv.DEFAULT_ARMS)
        with self.assertRaises(ValueError):
            batch_cv.parse_arms(
                "64:32:128,64:32:128,128:64:64,128:64:128"
            )
        with self.assertRaises(ValueError):
            batch_cv.parse_arms(
                "64:64:128,128:64:64,128:64:128"
            )
        unauthorized = batch_cv.parse_arms(
            "64:32:64,128:64:64,128:64:128"
        )
        with self.assertRaisesRegex(ValueError, "three registered arms"):
            batch_cv.validate_experiment_grid(unauthorized, (1, 2, 3), (0, 5))
        with self.assertRaises(ValueError):
            batch_cv.validate_experiment_grid(arms, (1, 2), (0, 5))
        with self.assertRaises(ValueError):
            batch_cv.validate_experiment_grid(arms, (1, 2, 3), (0,))

    def test_public_parsers_expose_w64_s32_and_default_three_by_three_by_two_grid(self) -> None:
        window_args = pretrain_window_encoder.build_parser().parse_args(
            [
                "--npz-path",
                "windows.npz",
                "--output-dir",
                "out",
                "--fold",
                "1",
                "--seed",
                "0",
                "--window-size",
                "64",
                "--window-stride",
                "32",
            ]
        )
        self.assertEqual((window_args.window_size, window_args.window_stride), (64, 32))
        cv_args = batch_cv.build_parser().parse_args(
            ["--project-root", ".", "--output-root", "out"]
        )
        self.assertEqual(cv_args.folds, "1,2,3")
        self.assertEqual(cv_args.seeds, "0,5")
        self.assertEqual(cv_args.arms, batch_cv.DEFAULT_ARMS)
        self.assertEqual(cv_args.window_selection_policy, "best_val_macro_f1")
        self.assertTrue(window_args.deterministic)
        self.assertEqual(window_args.selection_policy, "best_val_macro_f1")
        self.assertEqual(cv_args.descriptor_profile, LEGACY_DESCRIPTOR_PROFILE)
        self.assertEqual(batch_cv.MEMBER_SCHEMA, batch_member.SCHEMA)

        cv_args.descriptor_profile = FULL_DEBIASED_DESCRIPTOR_PROFILE
        with self.assertRaisesRegex(ValueError, "locked to legacy_state_v1"):
            batch_cv.validate_args(cv_args)

        member_args = batch_member.build_parser().parse_args(
            [
                "--a2-checkpoint", "encoder.pt",
                "--npz-path", "windows.npz",
                "--output-dir", "out",
                "--fold", "1",
                "--seed", "0",
                "--window-size", "128",
                "--window-stride", "64",
                "--primitive-num", "64",
                "--descriptor-profile", FULL_DEBIASED_DESCRIPTOR_PROFILE,
            ]
        )
        self.assertEqual(
            member_args.descriptor_profile, FULL_DEBIASED_DESCRIPTOR_PROFILE
        )
        with self.assertRaisesRegex(ValueError, "reuse"):
            batch_member.validate_args(member_args)
        member_args.reuse_codebook_run_dir = "legacy-member"
        batch_member.validate_args(member_args)

    def test_descriptor_ablation_parts_are_versioned_and_factor_isolated(self) -> None:
        gravity = descriptor_cv.profiles_for_part(
            descriptor_cv.GRAVITY_EXPERIMENT_PART
        )
        subject = descriptor_cv.profiles_for_part(
            descriptor_cv.SUBJECT_DEBIAS_EXPERIMENT_PART
        )
        duration = descriptor_cv.profiles_for_part(
            descriptor_cv.DURATION_EXPERIMENT_PART
        )
        duration_soft = descriptor_cv.profiles_for_part(
            descriptor_cv.DURATION_SOFT_SUBJECT_EXPERIMENT_PART
        )
        self.assertEqual(
            gravity,
            (LEGACY_DESCRIPTOR_PROFILE, GRAVITY_DESCRIPTOR_PROFILE),
        )
        self.assertEqual(
            subject,
            (
                LEGACY_DESCRIPTOR_PROFILE,
                SUBJECT_DEBIASED_DESCRIPTOR_PROFILE,
            ),
        )
        self.assertEqual(
            duration,
            (LEGACY_DESCRIPTOR_PROFILE, DURATION_INVARIANT_DESCRIPTOR_PROFILE),
        )
        self.assertEqual(
            duration_soft,
            (
                LEGACY_DESCRIPTOR_PROFILE,
                DURATION_INVARIANT_DESCRIPTOR_PROFILE,
                DURATION_SOFT_SUBJECT_A025_PROFILE,
                DURATION_SOFT_SUBJECT_A050_PROFILE,
                DURATION_SOFT_SUBJECT_A075_PROFILE,
            ),
        )
        self.assertNotIn(
            FULL_DEBIASED_DESCRIPTOR_PROFILE,
            gravity + subject + duration + duration_soft,
        )
        with self.assertRaises(ValueError):
            descriptor_cv.profiles_for_part("combined")

        subject_contrasts = dict(
            descriptor_cv.PART_CONTRASTS[
                descriptor_cv.SUBJECT_DEBIAS_EXPERIMENT_PART
            ]
        )
        self.assertEqual(
            subject_contrasts["subject_debias_minus_legacy"],
            {
                SUBJECT_DEBIASED_DESCRIPTOR_PROFILE: 1.0,
                LEGACY_DESCRIPTOR_PROFILE: -1.0,
            },
        )
        duration_contrasts = dict(
            descriptor_cv.PART_CONTRASTS[
                descriptor_cv.DURATION_EXPERIMENT_PART
            ]
        )
        self.assertEqual(
            duration_contrasts["duration_invariant_minus_legacy"],
            {
                DURATION_INVARIANT_DESCRIPTOR_PROFILE: 1.0,
                LEGACY_DESCRIPTOR_PROFILE: -1.0,
            },
        )
        duration_soft_contrasts = dict(
            descriptor_cv.PART_CONTRASTS[
                descriptor_cv.DURATION_SOFT_SUBJECT_EXPERIMENT_PART
            ]
        )
        self.assertEqual(
            duration_soft_contrasts["soft_a025_minus_duration_invariant"],
            {
                DURATION_SOFT_SUBJECT_A025_PROFILE: 1.0,
                DURATION_INVARIANT_DESCRIPTOR_PROFILE: -1.0,
            },
        )
        self.assertEqual(
            duration_soft_contrasts["soft_a050_minus_duration_invariant"],
            {
                DURATION_SOFT_SUBJECT_A050_PROFILE: 1.0,
                DURATION_INVARIANT_DESCRIPTOR_PROFILE: -1.0,
            },
        )
        self.assertEqual(
            duration_soft_contrasts["soft_a075_minus_duration_invariant"],
            {
                DURATION_SOFT_SUBJECT_A075_PROFILE: 1.0,
                DURATION_INVARIANT_DESCRIPTOR_PROFILE: -1.0,
            },
        )
        self.assertNotIn(
            descriptor_cv.DURATION_SOFT_SUBJECT_EXPERIMENT_PART,
            descriptor_cv.REGISTERED_EXPERIMENT_PARTS,
        )

    def test_descriptor_dry_runs_emit_the_registered_member_counts(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            npz = root / "windows.npz"
            npz.write_bytes(b"identity-only-npz")
            encoder_root = root / "encoders" / "w128_s64"
            for fold in range(1, 8):
                for seed in (0, 5):
                    checkpoint = (
                        encoder_root
                        / "a2"
                        / f"fold_{fold:02d}_seed_{seed}"
                        / "motion_encoder_final.pt"
                    )
                    checkpoint.parent.mkdir(parents=True, exist_ok=True)
                    checkpoint.write_bytes(f"fold={fold},seed={seed}".encode("ascii"))
            for part, expected_profiles in descriptor_cv.EXPERIMENT_PART_PROFILES.items():
                with self.subTest(part=part):
                    part_folds = (
                        descriptor_cv.CONFIRMATION_FOLDS
                        if part
                        == descriptor_cv.DURATION_SOFT_SUBJECT_CONFIRM_7FOLD_EXPERIMENT_PART
                        else descriptor_cv.DEFAULT_FOLDS
                    )
                    output = root / f"output_{part}"
                    args = descriptor_cv.build_parser().parse_args(
                        [
                            "--npz-path", str(npz),
                            "--encoder-source-root", str(encoder_root),
                            "--output-root", str(output),
                            "--experiment-part", part,
                            "--folds", ",".join(map(str, part_folds)),
                            "--device", "cpu",
                            "--dry-run",
                        ]
                    )
                    result = descriptor_cv.run(args)
                    expected_count = len(expected_profiles) * len(part_folds) * 2
                    self.assertEqual(result["member_count"], expected_count)
                    self.assertEqual(len(result["commands"]), expected_count)

                    observed = set()
                    output_directories: list[str] = []
                    checkpoints_by_pair: dict[tuple[int, int], set[str]] = {}
                    for command in result["commands"]:
                        options = {
                            command[index]: command[index + 1]
                            for index in range(2, len(command), 2)
                        }
                        profile = options["--descriptor-profile"]
                        fold = int(options["--fold"])
                        seed = int(options["--seed"])
                        observed.add((profile, fold, seed))
                        output_directories.append(options["--output-dir"])
                        checkpoints_by_pair.setdefault((fold, seed), set()).add(
                            options["--a2-checkpoint"]
                        )
                        expected_legacy = descriptor_cv.member_directory(
                            output / "members" / LEGACY_DESCRIPTOR_PROFILE,
                            fold,
                            seed,
                        ).resolve()
                        if profile == LEGACY_DESCRIPTOR_PROFILE:
                            self.assertNotIn("--reuse-codebook-run-dir", options)
                        else:
                            self.assertEqual(
                                Path(options["--reuse-codebook-run-dir"]),
                                expected_legacy,
                            )
                        self.assertEqual(
                            (
                                int(options["--window-size"]),
                                int(options["--window-stride"]),
                                int(options["--primitive-num"]),
                            ),
                            (128, 64, 64),
                        )
                        expected_strength = {
                            DURATION_SOFT_SUBJECT_A025_PROFILE: 0.25,
                            DURATION_SOFT_SUBJECT_A050_PROFILE: 0.50,
                            DURATION_SOFT_SUBJECT_A075_PROFILE: 0.75,
                        }.get(profile, 0.0 if profile != SUBJECT_DEBIASED_DESCRIPTOR_PROFILE else 1.0)
                        self.assertEqual(
                            descriptor_cv.descriptor_profile_spec(
                                profile
                            ).subject_nuisance_projection_strength,
                            expected_strength,
                        )
                    self.assertEqual(len(observed), expected_count)
                    self.assertEqual(
                        len(output_directories), len(set(output_directories))
                    )
                    if part == descriptor_cv.DURATION_SOFT_SUBJECT_EXPERIMENT_PART:
                        self.assertEqual(expected_count, 30)
                    if part == descriptor_cv.DURATION_SOFT_SUBJECT_CONFIRM_7FOLD_EXPERIMENT_PART:
                        self.assertEqual(expected_count, 56)
                        self.assertNotIn(
                            DURATION_SOFT_SUBJECT_A075_PROFILE,
                            {profile for profile, _, _ in observed},
                        )
                    self.assertTrue(
                        all(len(value) == 1 for value in checkpoints_by_pair.values())
                    )
                    self.assertFalse(output.exists())

            suite_output = root / "output_all"
            suite_args = descriptor_cv.build_parser().parse_args(
                [
                    "--npz-path", str(npz),
                    "--encoder-source-root", str(encoder_root),
                    "--output-root", str(suite_output),
                    "--experiment-part", descriptor_cv.ALL_EXPERIMENT_PARTS,
                    "--device", "cpu",
                    "--dry-run",
                ]
            )
            suite = descriptor_cv.run(suite_args)
            self.assertEqual(suite["schema"], descriptor_cv.SUITE_SCHEMA)
            self.assertEqual(suite["member_count"], 36)
            self.assertEqual(len(suite["commands"]), 36)
            self.assertEqual(
                set(suite["children"]), set(descriptor_cv.REGISTERED_EXPERIMENT_PARTS)
            )
            self.assertTrue(
                all(child["member_count"] == 12 for child in suite["children"].values())
            )
            outputs = [
                command[command.index("--output-dir") + 1]
                for command in suite["commands"]
            ]
            self.assertEqual(len(outputs), len(set(outputs)))
            self.assertFalse(suite_output.exists())

    def test_descriptor_suite_writes_three_independent_child_completions(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            npz = root / "windows.npz"
            npz.write_bytes(b"identity-only-npz")
            encoder_root = root / "encoders" / "w128_s64"
            for fold in (1, 2, 3):
                for seed in (0, 5):
                    checkpoint = (
                        encoder_root
                        / "a2"
                        / f"fold_{fold:02d}_seed_{seed}"
                        / "motion_encoder_final.pt"
                    )
                    checkpoint.parent.mkdir(parents=True, exist_ok=True)
                    checkpoint.write_bytes(f"fold={fold},seed={seed}".encode("ascii"))
            output = root / "suite"
            args = descriptor_cv.build_parser().parse_args(
                [
                    "--npz-path", str(npz),
                    "--encoder-source-root", str(encoder_root),
                    "--output-root", str(output),
                    "--experiment-part", descriptor_cv.ALL_EXPERIMENT_PARTS,
                    "--device", "cpu",
                ]
            )

            original_run = descriptor_cv.run
            calls = []

            def fake_child_run(child_args):
                self.assertIn(
                    child_args.experiment_part,
                    descriptor_cv.REGISTERED_EXPERIMENT_PARTS,
                )
                _, part, profiles, folds, seeds = descriptor_cv.validate_args(
                    child_args
                )
                child_identity = descriptor_cv._identity(
                    child_args, part, profiles, folds, seeds
                )
                result = {
                    "schema": descriptor_cv.SCHEMA,
                    "complete": True,
                    "member_count": 12,
                    "grid_identity_sha256": canonical_hash(child_identity),
                }
                child_output = Path(child_args.output_root)
                child_output.mkdir(parents=True, exist_ok=True)
                (child_output / "complete.json").write_text(
                    json.dumps(result, sort_keys=True) + "\n", encoding="utf-8"
                )
                calls.append(part)
                return result

            descriptor_cv.run = fake_child_run
            try:
                result = descriptor_cv._run_suite(args)
            finally:
                descriptor_cv.run = original_run

            self.assertEqual(calls, list(descriptor_cv.REGISTERED_EXPERIMENT_PARTS))
            self.assertEqual(result["member_count"], 36)
            self.assertEqual(result["member_count_per_part"], 12)
            self.assertTrue(result["independent_single_factor_children"])
            validated = descriptor_cv._validate_suite_complete(
                output,
                identity_sha256=result["grid_identity_sha256"],
            )
            self.assertEqual(validated, result)

    def test_confirmation_and_historical_parts_reject_each_others_grids(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            npz = root / "windows.npz"
            npz.write_bytes(b"identity-only-npz")
            common = [
                "--npz-path", str(npz),
                "--encoder-source-root", str(root / "encoders"),
                "--output-root", str(root / "output"),
            ]
            confirmation = descriptor_cv.build_parser().parse_args(
                [
                    *common,
                    "--experiment-part",
                    descriptor_cv.DURATION_SOFT_SUBJECT_CONFIRM_7FOLD_EXPERIMENT_PART,
                ]
            )
            with self.assertRaisesRegex(ValueError, "fixed to folds"):
                descriptor_cv.validate_args(confirmation)
            historical = descriptor_cv.build_parser().parse_args(
                [
                    *common,
                    "--experiment-part",
                    descriptor_cv.DURATION_SOFT_SUBJECT_EXPERIMENT_PART,
                    "--folds",
                    "1,2,3,4,5,6,7",
                ]
            )
            with self.assertRaisesRegex(ValueError, "fixed to folds"):
                descriptor_cv.validate_args(historical)

    def test_dry_run_with_two_missing_window_grids_emits_exactly_forty_four_commands(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            args = batch_cv.build_parser().parse_args(
                [
                    "--project-root",
                    str(batch_cv.PROJECT_ROOT),
                    "--dataset-root",
                    str(root / "dataset"),
                    "--processed-root",
                    str(root / "processed"),
                    "--output-root",
                    str(root / "output"),
                    "--dry-run",
                ]
            )
            result = batch_cv.run(args)
            commands = result["commands"]
            command_counts: dict[str, int] = {}
            for command in commands:
                script = Path(command[1]).name
                command_counts[script] = command_counts.get(script, 0) + 1
                if script == "pretrain_window_encoder.py":
                    self.assertIn("--deterministic", command)
                    self.assertEqual(
                        command[command.index("--selection-policy") + 1],
                        "best_val_macro_f1",
                    )
                elif script == "train_motion_encoder.py":
                    self.assertIn("--deterministic", command)
                    self.assertEqual(
                        command[command.index("--selection-policy") + 1],
                        "final_epoch",
                    )

            self.assertEqual(result["member_count"], 18)
            self.assertEqual(len(commands), 44)
            self.assertEqual(
                command_counts,
                {
                    "preprocess.py": 2,
                    "pretrain_window_encoder.py": 12,
                    "train_motion_encoder.py": 12,
                    "window_codebook_batch_proxy.py": 18,
                },
            )

    def test_batch_runner_overrides_conflicting_child_reproducibility_environment(self) -> None:
        captured: dict[str, str] = {}

        def fake_run(*_args, **kwargs):
            captured.update(kwargs["env"])
            return subprocess.CompletedProcess(args=["python"], returncode=0)

        with mock.patch.dict(
            os.environ,
            {"CUBLAS_WORKSPACE_CONFIG": "conflicting", "PYTHONHASHSEED": "123"},
            clear=False,
        ), mock.patch.object(batch_cv.subprocess, "run", side_effect=fake_run):
            batch_cv._run_command(["python", "child.py"], stage="probe")

        self.assertEqual(captured["CUBLAS_WORKSPACE_CONFIG"], ":4096:8")
        self.assertEqual(captured["PYTHONHASHSEED"], "0")
        for name in (
            "OMP_NUM_THREADS",
            "MKL_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "NUMEXPR_NUM_THREADS",
            "VECLIB_MAXIMUM_THREADS",
        ):
            self.assertEqual(captured[name], "1")

    def test_descriptor_runner_pins_child_blas_and_hash_environment(self) -> None:
        captured: dict[str, str] = {}

        def fake_run(*_args, **kwargs):
            captured.update(kwargs["env"])
            return subprocess.CompletedProcess(args=["python"], returncode=0)

        with mock.patch.object(
            descriptor_cv.subprocess, "run", side_effect=fake_run
        ):
            descriptor_cv._run_command(["python", "child.py"], stage="probe")

        self.assertEqual(captured["CUBLAS_WORKSPACE_CONFIG"], ":4096:8")
        self.assertEqual(captured["PYTHONHASHSEED"], "0")
        for name in (
            "OMP_NUM_THREADS",
            "MKL_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "NUMEXPR_NUM_THREADS",
            "VECLIB_MAXIMUM_THREADS",
        ):
            self.assertEqual(captured[name], "1")

    def test_frozen_runtime_enables_strict_torch_and_single_thread_environment(self) -> None:
        runtime = configure_deterministic_runtime(seed=19)
        self.assertTrue(runtime["torch_deterministic_algorithms"])
        self.assertTrue(runtime["cudnn_deterministic"])
        self.assertFalse(runtime["cudnn_benchmark"])
        self.assertTrue(runtime["cudnn_allow_tf32"])
        self.assertFalse(runtime["cuda_matmul_allow_tf32"])
        self.assertEqual(runtime["float32_matmul_precision"], "highest")
        self.assertEqual(runtime["environment"]["CUBLAS_WORKSPACE_CONFIG"], ":4096:8")
        for name in (
            "OMP_NUM_THREADS",
            "MKL_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "NUMEXPR_NUM_THREADS",
            "VECLIB_MAXIMUM_THREADS",
        ):
            self.assertEqual(runtime["environment"][name], "1")

    def test_batch_a2_validator_requires_runtime_and_tensor_hashes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            checkpoint = output / "motion_encoder_final.pt"
            runtime = {
                "enabled": True,
                "torch_deterministic_algorithms": True,
                "cudnn_benchmark": False,
                "cublas_workspace_config": ":4096:8",
                "python_hash_seed": "0",
                "cuda_device_name": None,
                "blas_thread_environment": {
                    "OMP_NUM_THREADS": "1",
                    "MKL_NUM_THREADS": "1",
                    "OPENBLAS_NUM_THREADS": "1",
                    "NUMEXPR_NUM_THREADS": "1",
                    "VECLIB_MAXIMUM_THREADS": "1",
                },
            }
            model_sha = "m" * 64
            teacher_sha = "t" * 64
            torch.save({
                "determinism": runtime,
                "model_state_dict_sha256": model_sha,
                "ema_teacher_state_dict_sha256": teacher_sha,
            }, checkpoint)
            complete = {
                "schema": train_motion_encoder.COMPLETE_SCHEMA,
                "fold": 1,
                "seed": 0,
                "npz_sha256": "n" * 64,
                "source_checkpoint_sha256": "s" * 64,
                "ablation_profile": "A2",
                "selection_policy": "final_epoch",
                "final_checkpoint_sha256": strict_protocol.sha256_file(checkpoint),
                "final_model_state_dict_sha256": model_sha,
                "final_ema_teacher_state_dict_sha256": teacher_sha,
                "determinism": runtime,
                "complete": True,
            }
            (output / "complete.json").write_text(
                json.dumps(complete), encoding="utf-8"
            )
            with mock.patch.object(
                train_motion_encoder, "_validate_output_checkpoint"
            ):
                self.assertEqual(
                    batch_cv._validated_a2_checkpoint(
                        output,
                        fold=1,
                        seed=0,
                        npz_sha256="n" * 64,
                        source_checkpoint_sha256="s" * 64,
                    ),
                    checkpoint,
                )
                complete["final_model_state_dict_sha256"] = "x" * 64
                (output / "complete.json").write_text(
                    json.dumps(complete), encoding="utf-8"
                )
                with self.assertRaisesRegex(RuntimeError, "model tensor-state"):
                    batch_cv._validated_a2_checkpoint(
                        output,
                        fold=1,
                        seed=0,
                        npz_sha256="n" * 64,
                        source_checkpoint_sha256="s" * 64,
                    )

    def test_batch_window_validator_binds_selected_tensor_and_backbone_hashes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            npz_sha256 = "n" * 64
            arm = batch_cv.parse_arms(batch_cv.DEFAULT_ARMS)[0]
            identity = {
                "npz_sha256": npz_sha256,
                "arguments": {
                    "window_size": int(arm.window_size),
                    "window_stride": int(arm.window_stride),
                    "fold": 1,
                    "seed": 0,
                    "deterministic": True,
                    "selection_policy": "best_val_macro_f1",
                },
            }
            state = {"0.weight": torch.tensor([1.0, 2.0], dtype=torch.float32)}
            model_sha = pretrain_window_encoder.motion_state_dict_sha256(state)
            backbone_sha = pretrain_window_encoder._backbone_state_dict_sha256(state)
            runtime = {
                "enabled": True,
                "torch_deterministic_algorithms": True,
                "cudnn_benchmark": False,
                "cublas_workspace_config": ":4096:8",
                "python_hash_seed": "0",
                "cuda_device_name": None,
            }
            checkpoint = output / "model_best.pt"
            torch.save(
                {
                    "run_identity": identity,
                    "model": state,
                    "model_state_dict": state,
                    "model_state_dict_sha256": model_sha,
                    "backbone_state_dict_sha256": backbone_sha,
                    "determinism": runtime,
                },
                checkpoint,
            )
            complete = {
                "schema": batch_cv.WINDOW_SCHEMA,
                "complete": True,
                "fold": 1,
                "seed": 0,
                "identity": identity,
                "checkpoint": checkpoint.name,
                "checkpoint_sha256": batch_cv.sha256_file(checkpoint),
                "model_state_dict_sha256": model_sha,
                "backbone_state_dict_sha256": backbone_sha,
                "determinism": runtime,
            }
            (output / "complete.json").write_text(
                json.dumps(complete), encoding="utf-8"
            )
            self.assertEqual(
                batch_cv._validated_window_checkpoint(
                    output,
                    arm=arm,
                    fold=1,
                    seed=0,
                    npz_sha256=npz_sha256,
                    selection_policy="best_val_macro_f1",
                ),
                checkpoint,
            )

            complete["backbone_state_dict_sha256"] = "0" * 64
            (output / "complete.json").write_text(
                json.dumps(complete), encoding="utf-8"
            )
            with self.assertRaisesRegex(RuntimeError, "backbone SHA256 mismatch"):
                batch_cv._validated_window_checkpoint(
                    output,
                    arm=arm,
                    fold=1,
                    seed=0,
                    npz_sha256=npz_sha256,
                    selection_policy="best_val_macro_f1",
                )

    def test_descriptor_summary_rejects_any_cross_profile_codebook_drift(self) -> None:
        profiles = descriptor_cv.profiles_for_part(
            descriptor_cv.SUBJECT_DEBIAS_EXPERIMENT_PART
        )
        legacy_identity = "1" * 64
        rows = []
        for profile in profiles:
            rows.append(
                {
                    "descriptor_profile": profile,
                    "fold": 1,
                    "seed": 0,
                    "codebook_state_sha256": "a" * 64,
                    "member_run_identity_sha256": (
                        legacy_identity if profile == LEGACY_DESCRIPTOR_PROFILE else "2" * 64
                    ),
                    "codebook_origin_mode": (
                        "fit_offline_old6_current_legacy_member"
                        if profile == LEGACY_DESCRIPTOR_PROFILE
                        else "reuse_completed_same_fold_seed_legacy_member"
                    ),
                    "codebook_source_run_identity_sha256": (
                        "" if profile == LEGACY_DESCRIPTOR_PROFILE else legacy_identity
                    ),
                }
            )
        audit = descriptor_cv._validate_shared_codebooks(
            rows,
            profiles=profiles,
            folds=(1,),
            seeds=(0,),
        )
        self.assertTrue(audit["one_fit_per_fold_seed"])
        self.assertTrue(audit["all_profiles_share_exact_state_sha256"])
        self.assertEqual(audit["fit_count"], 1)

        rows[-1] = {**rows[-1], "codebook_state_sha256": "b" * 64}
        with self.assertRaisesRegex(RuntimeError, "one exact codebook"):
            descriptor_cv._validate_shared_codebooks(
                rows,
                profiles=profiles,
                folds=(1,),
                seeds=(0,),
            )

    def test_duration_soft_coordinate_audit_requires_exact_preprojection_state(self) -> None:
        profiles = descriptor_cv.profiles_for_part(
            descriptor_cv.DURATION_SOFT_SUBJECT_EXPERIMENT_PART
        )
        compared = profiles[1:]
        rows = [
            {
                "descriptor_profile": profile,
                "fold": 1,
                "seed": 0,
                "pre_subject_debias_transform_state_sha256": "a" * 64,
            }
            for profile in profiles
        ]
        audit = descriptor_cv._validate_duration_soft_subject_coordinates(
            rows,
            experiment_part=descriptor_cv.DURATION_SOFT_SUBJECT_EXPERIMENT_PART,
            folds=(1,),
            seeds=(0,),
        )
        self.assertTrue(audit["required"])
        self.assertTrue(audit["verified"])
        self.assertEqual(audit["profiles"], list(compared))
        self.assertEqual(len(audit["pairs"]), 1)

        rows[-1] = {
            **rows[-1],
            "pre_subject_debias_transform_state_sha256": "b" * 64,
        }
        with self.assertRaisesRegex(RuntimeError, "pre-debias descriptor coordinates"):
            descriptor_cv._validate_duration_soft_subject_coordinates(
                rows,
                experiment_part=descriptor_cv.DURATION_SOFT_SUBJECT_EXPERIMENT_PART,
                folds=(1,),
                seeds=(0,),
            )

        confirmation_profiles = descriptor_cv.profiles_for_part(
            descriptor_cv.DURATION_SOFT_SUBJECT_CONFIRM_7FOLD_EXPERIMENT_PART
        )
        confirmation_rows = [
            {
                "descriptor_profile": profile,
                "fold": 1,
                "seed": 0,
                "pre_subject_debias_transform_state_sha256": "c" * 64,
            }
            for profile in confirmation_profiles
        ]
        confirmation_audit = descriptor_cv._validate_duration_soft_subject_coordinates(
            confirmation_rows,
            experiment_part=descriptor_cv.DURATION_SOFT_SUBJECT_CONFIRM_7FOLD_EXPERIMENT_PART,
            folds=(1,),
            seeds=(0,),
        )
        self.assertEqual(
            confirmation_audit["profiles"],
            [
                DURATION_INVARIANT_DESCRIPTOR_PROFILE,
                DURATION_SOFT_SUBJECT_A025_PROFILE,
                DURATION_SOFT_SUBJECT_A050_PROFILE,
            ],
        )

    def test_subject_report_recovers_registered_paired_effect(self) -> None:
        profiles = descriptor_cv.profiles_for_part(
            descriptor_cv.SUBJECT_DEBIAS_EXPERIMENT_PART
        )
        profile_effect = {
            LEGACY_DESCRIPTOR_PROFILE: 0.0,
            SUBJECT_DEBIASED_DESCRIPTOR_PROFILE: 2.0,
        }
        rows = []
        for fold in (1, 2, 3):
            for seed in (0, 5):
                legacy_identity = f"{fold}{seed}".ljust(64, "1")
                for profile in profiles:
                    value = 10.0 * fold + 0.01 * seed + profile_effect[profile]
                    row = {
                        "descriptor_profile": profile,
                        "fold": fold,
                        "seed": seed,
                        "codebook_state_sha256": f"{fold}{seed}".ljust(64, "a"),
                        "member_run_identity_sha256": (
                            legacy_identity if profile == LEGACY_DESCRIPTOR_PROFILE
                            else f"{profile}{fold}{seed}".ljust(64, "2")
                        ),
                        "codebook_origin_mode": (
                            "fit_offline_old6_current_legacy_member"
                            if profile == LEGACY_DESCRIPTOR_PROFILE
                            else "reuse_completed_same_fold_seed_legacy_member"
                        ),
                        "codebook_source_run_identity_sha256": (
                            "" if profile == LEGACY_DESCRIPTOR_PROFILE else legacy_identity
                        ),
                    }
                    for metric in (
                        *descriptor_cv.PERFORMANCE_METRICS,
                        *descriptor_cv.BIAS_DIAGNOSTICS,
                    ):
                        row[metric] = value
                    rows.append(row)

        with tempfile.TemporaryDirectory() as temporary:
            (Path(temporary) / "grid_manifest.json").write_text(
                "{}\n", encoding="utf-8"
            )
            result = descriptor_cv._reports(
                Path(temporary),
                rows,
                experiment_part=descriptor_cv.SUBJECT_DEBIAS_EXPERIMENT_PART,
                profiles=profiles,
                folds=(1, 2, 3),
                seeds=(0, 5),
                identity_sha256="f" * 64,
                bootstrap_seed=20260916,
                bootstrap_replicates=100,
            )
            effect = result["targeted_contrasts"][
                "subject_debias_minus_legacy"
            ]["h_score"]
            self.assertAlmostEqual(effect["mean"], 2.0)
            np.testing.assert_allclose(
                effect["fold_mean_values_after_averaging_seeds"],
                [2.0, 2.0, 2.0],
                atol=1e-12,
            )
            self.assertTrue(
                (Path(temporary) / "descriptor_targeted_contrasts.json").is_file()
            )

    def test_duration_soft_report_uses_duration_as_each_direct_baseline(self) -> None:
        profiles = descriptor_cv.profiles_for_part(
            descriptor_cv.DURATION_SOFT_SUBJECT_EXPERIMENT_PART
        )
        profile_effect = {
            LEGACY_DESCRIPTOR_PROFILE: -9.0,
            DURATION_INVARIANT_DESCRIPTOR_PROFILE: 0.0,
            DURATION_SOFT_SUBJECT_A025_PROFILE: 0.25,
            DURATION_SOFT_SUBJECT_A050_PROFILE: 0.50,
            DURATION_SOFT_SUBJECT_A075_PROFILE: 0.75,
        }
        rows = []
        for fold in (1, 2, 3):
            for seed in (0, 5):
                legacy_identity = f"legacy-{fold}-{seed}".ljust(64, "1")
                shared_codebook = f"codebook-{fold}-{seed}".ljust(64, "a")
                pre_transform = f"transform-{fold}-{seed}".ljust(64, "b")
                for profile in profiles:
                    value = 10.0 * fold + 0.01 * seed + profile_effect[profile]
                    row = {
                        "descriptor_profile": profile,
                        "fold": fold,
                        "seed": seed,
                        "codebook_state_sha256": shared_codebook,
                        "member_run_identity_sha256": (
                            legacy_identity
                            if profile == LEGACY_DESCRIPTOR_PROFILE
                            else f"{profile}-{fold}-{seed}".ljust(64, "2")
                        ),
                        "codebook_origin_mode": (
                            "fit_offline_old6_current_legacy_member"
                            if profile == LEGACY_DESCRIPTOR_PROFILE
                            else "reuse_completed_same_fold_seed_legacy_member"
                        ),
                        "codebook_source_run_identity_sha256": (
                            "" if profile == LEGACY_DESCRIPTOR_PROFILE else legacy_identity
                        ),
                        "pre_subject_debias_transform_state_sha256": pre_transform,
                    }
                    for metric in (
                        *descriptor_cv.PERFORMANCE_METRICS,
                        *descriptor_cv.BIAS_DIAGNOSTICS,
                    ):
                        row[metric] = value
                    rows.append(row)

        with tempfile.TemporaryDirectory() as temporary:
            (Path(temporary) / "grid_manifest.json").write_text(
                "{}\n", encoding="utf-8"
            )
            result = descriptor_cv._reports(
                Path(temporary),
                rows,
                experiment_part=descriptor_cv.DURATION_SOFT_SUBJECT_EXPERIMENT_PART,
                profiles=profiles,
                folds=(1, 2, 3),
                seeds=(0, 5),
                identity_sha256="f" * 64,
                bootstrap_seed=20260916,
                bootstrap_replicates=100,
            )
            expected = {
                "soft_a025_minus_duration_invariant": 0.25,
                "soft_a050_minus_duration_invariant": 0.50,
                "soft_a075_minus_duration_invariant": 0.75,
            }
            for contrast, delta in expected.items():
                with self.subTest(contrast=contrast):
                    effect = result["targeted_contrasts"][contrast]["h_score"]
                    self.assertAlmostEqual(effect["mean"], delta)
                    np.testing.assert_allclose(
                        effect["fold_mean_values_after_averaging_seeds"],
                        [delta, delta, delta],
                        atol=1e-12,
                    )
            self.assertTrue(
                result["duration_soft_pre_debias_coordinate_audit"]["verified"]
            )

    def test_seven_fold_confirmation_reports_screening_and_holdout_separately(self) -> None:
        profiles = descriptor_cv.profiles_for_part(
            descriptor_cv.DURATION_SOFT_SUBJECT_CONFIRM_7FOLD_EXPERIMENT_PART
        )
        profile_effect = {
            LEGACY_DESCRIPTOR_PROFILE: -0.10,
            DURATION_INVARIANT_DESCRIPTOR_PROFILE: 0.0,
            DURATION_SOFT_SUBJECT_A025_PROFILE: 0.25,
            DURATION_SOFT_SUBJECT_A050_PROFILE: 0.50,
        }
        rows = []
        for fold in descriptor_cv.CONFIRMATION_FOLDS:
            for seed in descriptor_cv.CONFIRMATION_SEEDS:
                legacy_identity = f"legacy-{fold}-{seed}".ljust(64, "1")
                shared_codebook = f"codebook-{fold}-{seed}".ljust(64, "a")
                pre_transform = f"transform-{fold}-{seed}".ljust(64, "b")
                for profile in profiles:
                    value = 10.0 * fold + 0.01 * seed + profile_effect[profile]
                    row = {
                        "descriptor_profile": profile,
                        "fold": fold,
                        "seed": seed,
                        "codebook_state_sha256": shared_codebook,
                        "member_run_identity_sha256": (
                            legacy_identity
                            if profile == LEGACY_DESCRIPTOR_PROFILE
                            else f"{profile}-{fold}-{seed}".ljust(64, "2")
                        ),
                        "codebook_origin_mode": (
                            "fit_offline_old6_current_legacy_member"
                            if profile == LEGACY_DESCRIPTOR_PROFILE
                            else "reuse_completed_same_fold_seed_legacy_member"
                        ),
                        "codebook_source_run_identity_sha256": (
                            "" if profile == LEGACY_DESCRIPTOR_PROFILE else legacy_identity
                        ),
                        "pre_subject_debias_transform_state_sha256": pre_transform,
                    }
                    for metric in (
                        *descriptor_cv.PERFORMANCE_METRICS,
                        *descriptor_cv.BIAS_DIAGNOSTICS,
                    ):
                        row[metric] = value
                    rows.append(row)

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "grid_manifest.json").write_text("{}\n", encoding="utf-8")
            result = descriptor_cv._reports(
                root,
                rows,
                experiment_part=descriptor_cv.DURATION_SOFT_SUBJECT_CONFIRM_7FOLD_EXPERIMENT_PART,
                profiles=profiles,
                folds=descriptor_cv.CONFIRMATION_FOLDS,
                seeds=descriptor_cv.CONFIRMATION_SEEDS,
                identity_sha256="f" * 64,
                bootstrap_seed=20260916,
                bootstrap_replicates=100,
            )
            self.assertEqual(result["fold_count"], 7)
            self.assertFalse(result["three_fold_confidence_intervals_are_exploratory"])
            self.assertEqual(
                set(result["targeted_contrasts"]),
                {
                    "duration_invariant_minus_legacy",
                    "soft_a025_minus_duration_invariant",
                    "soft_a050_minus_duration_invariant",
                },
            )
            partitions = result["confirmation_partitions"]
            self.assertEqual(
                partitions["confirmation_holdout_folds_4_7"]["folds"],
                [4, 5, 6, 7],
            )
            holdout = partitions["confirmation_holdout_folds_4_7"]["contrasts"]
            self.assertAlmostEqual(
                holdout["soft_a025_minus_duration_invariant"]["h_score"]["mean"],
                0.25,
            )
            self.assertAlmostEqual(
                holdout["soft_a050_minus_duration_invariant"]["h_score"]["mean"],
                0.50,
            )
            self.assertTrue((root / "descriptor_confirmation_partitions.json").is_file())


class DynamicDescriptorContractTests(unittest.TestCase):
    @staticmethod
    def _trajectory(primitive_num: int) -> PrimitiveTrajectory:
        tokens = np.asarray([0, int(primitive_num) - 1, 1], dtype=np.int64)
        statistics = np.zeros((3, len(statistic_names())), dtype=np.float64)
        statistics[:, 0] = np.asarray([0.48, 0.32, 0.48], dtype=np.float64)
        statistics[:, 1:] = np.arange(1, len(statistic_names()), dtype=np.float64)
        return PrimitiveTrajectory(
            trial_id=7,
            subject_id=10,
            starts=np.asarray([0, 48, 80], dtype=np.int64),
            ends=np.asarray([48, 80, 128], dtype=np.int64),
            tokens=tokens,
            distances=np.asarray([0.1, 0.2, 0.3], dtype=np.float32),
            embeddings=np.eye(3, 64, dtype=np.float32),
            statistics=statistics,
            statistic_names=statistic_names(),
        ).validate(int(primitive_num))

    def test_state_descriptor_dimensions_are_dynamic_for_k32_k64_k128(self) -> None:
        for primitive_num, expected_dim in EXPECTED_STATE_DIMS.items():
            with self.subTest(primitive_num=primitive_num):
                self.assertEqual(
                    raw_descriptor_dimension(primitive_num, include_state=True),
                    expected_dim,
                )
                values, names = trajectory_descriptor(
                    self._trajectory(primitive_num),
                    primitive_num,
                    include_state=True,
                )
                self.assertEqual(values.shape, (expected_dim,))
                self.assertEqual(len(names), expected_dim)
                self.assertEqual(len(set(names)), expected_dim)
                self.assertTrue(np.all(np.isfinite(values)))

    def test_generic_k32_codebook_is_numerically_equivalent_to_registered_k32(self) -> None:
        rng = np.random.default_rng(20260914)
        directions = rng.normal(size=(32, 80))
        directions /= np.linalg.norm(directions, axis=1, keepdims=True)
        registered_trials: list[RegisteredWindowTrial] = []
        content_rows: list[np.ndarray] = []
        trial_id_rows: list[np.ndarray] = []
        subject_id_rows: list[np.ndarray] = []
        for offset in range(4):
            trial_id = 100 + offset
            subject_id = 1 + offset % 2
            content = (
                directions + rng.normal(scale=0.003, size=directions.shape)
            ).astype(np.float32)
            registered_trials.append(
                RegisteredWindowTrial(
                    trial_id=trial_id,
                    subject_id=subject_id,
                    window_starts=np.arange(32, dtype=np.int64) * 8,
                    raw_windows=rng.normal(size=(32, 6, 16)).astype(np.float32),
                    content_embeddings=content,
                ).validate(window_size=16)
            )
            content_rows.append(content)
            trial_id_rows.append(np.full(32, trial_id, dtype=np.int64))
            subject_id_rows.append(np.full(32, subject_id, dtype=np.int64))

        registered = fit_registered_e0_codebook(
            tuple(registered_trials), seed=5, primitive_num=32, pca_dim=64
        )
        generic = fit_frozen_e0_codebook(
            np.concatenate(content_rows),
            np.concatenate(trial_id_rows),
            np.concatenate(subject_id_rows),
            seed=5,
            primitive_num=32,
            pca_dim=64,
            strict_historical=True,
        )
        np.testing.assert_allclose(generic.pca_mean, registered.pca_mean, atol=1e-7)

        # Eigenvector signs are algebraically arbitrary.  Align each generic
        # component before comparing both PCA bases and their KMeans centres.
        agreement = np.sum(
            registered.pca_components * generic.pca_components, axis=1
        )
        self.assertTrue(np.all(np.abs(agreement) > 0.99))
        signs = np.where(agreement < 0.0, -1.0, 1.0).astype(np.float32)
        np.testing.assert_allclose(
            generic.pca_components * signs[:, None],
            registered.pca_components,
            atol=1e-6,
        )
        np.testing.assert_allclose(
            generic.cluster_centers * signs[None, :],
            registered.cluster_centers,
            atol=1e-5,
        )

        probe = np.concatenate(content_rows)[:24]
        generic_tokens, generic_distances, generic_embeddings = generic.assign(probe)
        registered_tokens, registered_distances, registered_embeddings = registered.assign(probe)
        np.testing.assert_array_equal(generic_tokens, registered_tokens)
        np.testing.assert_allclose(generic_distances, registered_distances, atol=1e-6)
        np.testing.assert_allclose(
            generic_embeddings @ generic_embeddings.T,
            registered_embeddings @ registered_embeddings.T,
            atol=1e-6,
        )

    def test_k64_and_k128_codebooks_fit_assign_and_round_trip(self) -> None:
        rng = np.random.default_rng(2026091401)
        content = rng.normal(size=(256, 80)).astype(np.float32)
        trial_ids = np.repeat(np.arange(16, dtype=np.int64), 16)
        subject_ids = np.repeat(np.arange(1, 9, dtype=np.int64), 32)
        descriptor_transform = DescriptorTransform(
            keep_columns=np.asarray([0], dtype=np.int64),
            mean=np.asarray([0.0], dtype=np.float64),
            scale=np.asarray([1.0], dtype=np.float64),
            pca_mean=None,
            pca_components=None,
        )
        for primitive_num in (64, 128):
            with self.subTest(primitive_num=primitive_num), tempfile.TemporaryDirectory() as temporary:
                codebook = fit_frozen_e0_codebook(
                    content,
                    trial_ids,
                    subject_ids,
                    primitive_num=primitive_num,
                    pca_dim=64,
                    seed=5,
                    strict_historical=False,
                )
                tokens, distances, embeddings = codebook.assign(content[:32])
                self.assertEqual(tokens.shape, (32,))
                self.assertTrue(np.all((0 <= tokens) & (tokens < primitive_num)))
                self.assertTrue(np.all(np.isfinite(distances)))
                self.assertEqual(embeddings.shape, (32, 64))

                path = Path(temporary) / f"k{primitive_num}.npz"
                save_frozen_artifacts(
                    path,
                    codebook,
                    descriptor_transform,
                    ("probe",),
                    {"primitive_num": primitive_num},
                )
                restored, transform, names, metadata = load_frozen_artifacts(path)
                self.assertEqual(restored.primitive_num, primitive_num)
                np.testing.assert_array_equal(restored.pca_mean, codebook.pca_mean)
                np.testing.assert_array_equal(
                    restored.pca_components, codebook.pca_components
                )
                np.testing.assert_array_equal(
                    restored.cluster_centers, codebook.cluster_centers
                )
                restored_tokens, restored_distances, restored_embeddings = restored.assign(
                    content[:32]
                )
                np.testing.assert_array_equal(restored_tokens, tokens)
                np.testing.assert_array_equal(restored_distances, distances)
                np.testing.assert_array_equal(restored_embeddings, embeddings)
                np.testing.assert_array_equal(
                    transform.keep_columns, descriptor_transform.keep_columns
                )
                self.assertEqual(names, ("probe",))
                self.assertEqual(metadata, {"primitive_num": primitive_num})


class TimeRasterContractTests(unittest.TestCase):
    def test_ownership_spans_are_rasterized_on_a_seconds_axis(self) -> None:
        first = DynamicDescriptorContractTests._trajectory(8)
        second = PrimitiveTrajectory(
            trial_id=8,
            subject_id=11,
            starts=np.asarray([0, 50], dtype=np.int64),
            ends=np.asarray([50, 100], dtype=np.int64),
            tokens=np.asarray([3, 4], dtype=np.int64),
            distances=np.asarray([0.1, 0.2], dtype=np.float32),
            embeddings=np.eye(2, 64, dtype=np.float32),
            statistics=np.zeros((2, len(statistic_names())), dtype=np.float64),
            statistic_names=statistic_names(),
        ).validate(8)

        raster, time_edges_seconds = batch_member.trajectory_time_raster_seconds(
            (first, second), sample_rate_hz=100.0
        )
        self.assertEqual(raster.shape, (2, 128))
        self.assertEqual(time_edges_seconds.shape, (129,))
        np.testing.assert_allclose(np.diff(time_edges_seconds), 0.01)
        self.assertAlmostEqual(float(time_edges_seconds[48]), 0.48)
        self.assertAlmostEqual(float(time_edges_seconds[-1]), 1.28)

        np.testing.assert_array_equal(raster[0, :48], np.zeros(48))
        np.testing.assert_array_equal(raster[0, 48:80], np.full(32, 7.0))
        np.testing.assert_array_equal(raster[0, 80:128], np.ones(48))
        np.testing.assert_array_equal(raster[1, :50], np.full(50, 3.0))
        np.testing.assert_array_equal(raster[1, 50:100], np.full(50, 4.0))
        self.assertFalse(np.any(np.isnan(raster[0])))
        self.assertFalse(np.any(np.isnan(raster[1, :100])))
        self.assertTrue(np.all(np.isnan(raster[1, 100:])))


class W64ProtocolContractTests(unittest.TestCase):
    class _GridWasReached(RuntimeError):
        pass

    def test_registered_protocol_accepts_w64_s32_before_loading_the_grid(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            npz = Path(temporary) / "windows.npz"
            npz.write_bytes(b"parameter-validation-only")
            original = strict_protocol._load_grid

            def reached(*_args, **_kwargs):
                raise self._GridWasReached

            strict_protocol._load_grid = reached
            try:
                with self.assertRaises(self._GridWasReached):
                    strict_protocol.build_registered_protocol(
                        npz,
                        fold=1,
                        seed=0,
                        window_size=64,
                        stride=32,
                    )
                with self.assertRaises(ValueError):
                    strict_protocol.build_registered_protocol(
                        npz,
                        fold=1,
                        seed=0,
                        window_size=64,
                        stride=64,
                    )
            finally:
                strict_protocol._load_grid = original

    def test_protocol_rejects_reordered_sensor_channels(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "reordered.npz"
            np.savez_compressed(
                path,
                windows=np.zeros((2, 6, 64), dtype=np.float32),
                labels=np.asarray([0, 0], dtype=np.int64),
                subject_ids=np.asarray([1, 1], dtype=np.int64),
                trial_global_ids=np.asarray([0, 0], dtype=np.int64),
                window_indices=np.asarray([0, 1], dtype=np.int64),
                window_start_indices=np.asarray([0, 32], dtype=np.int64),
                mean=np.zeros((1, 6, 1), dtype=np.float32),
                std=np.ones((1, 6, 1), dtype=np.float32),
                activity_names=np.asarray(["Walking Forward", "Walking Forward"]),
                channel_names=np.asarray(
                    ["acc_y", "acc_x", "acc_z", "gyro_x", "gyro_y", "gyro_z"]
                ),
            )
            with self.assertRaisesRegex(RuntimeError, "channel order"):
                strict_protocol._load_grid(path, 64, 32)


class BatchAssignmentContractTests(unittest.TestCase):
    def test_outer_batch_is_one_sorted_unique_set_of_all_120_trials(self) -> None:
        trials = batch_member.outer_batch_trials(_batch_protocol())
        ids = [int(trial.trial_id) for trial in trials]
        self.assertEqual(len(ids), 120)
        self.assertEqual(ids, list(range(120)))
        self.assertEqual(len(set(ids)), 120)

    def test_outer_batch_rejects_duplicate_or_incomplete_trial_sets(self) -> None:
        with self.assertRaises(RuntimeError):
            batch_member.outer_batch_trials(_batch_protocol(duplicate=True))
        with self.assertRaises(RuntimeError):
            batch_member.outer_batch_trials(_batch_protocol(missing=True))

    def test_pretruth_trajectory_row_does_not_serialize_outer_subject_identity(self) -> None:
        trajectory = DynamicDescriptorContractTests._trajectory(8)
        sensor = SimpleNamespace(
            trial_id=trajectory.trial_id,
            subject_id=99,
            window_starts=np.asarray([0, 48, 80], dtype=np.int64),
        )
        row = batch_member._trajectory_rows((sensor,), (trajectory,))[0]
        self.assertNotIn("subject_id", row)
        self.assertEqual(row["trial_id"], trajectory.trial_id)

    def test_pretruth_subject_identity_scan_catches_aliases_and_nested_fields(self) -> None:
        self.assertEqual(
            batch_member._subject_identity_paths(
                {"nested": {"outer-subject-identity": [10, 11]}}
            ),
            ["nested.outer-subject-identity"],
        )
        self.assertEqual(
            batch_member._subject_identity_paths(
                {
                    "offline_codebook_fit_subject_count": 10,
                    "source_run_identity_sha256": "a" * 64,
                }
            ),
            [],
        )

    @staticmethod
    def _balanced_truth_rows() -> tuple[list[SimpleNamespace], np.ndarray, np.ndarray]:
        targets = np.tile(
            np.repeat(np.arange(batch_member.TOTAL_CLASS_COUNT), 5), 2
        ).astype(np.int64)
        subjects = np.repeat(np.asarray([10, 11], dtype=np.int64), 60)
        trials = [
            SimpleNamespace(trial_id=index, subject_id=int(subject))
            for index, subject in enumerate(subjects.tolist())
        ]
        return trials, targets, subjects

    def test_scorer_truth_requires_row_subject_and_balanced_class_alignment(self) -> None:
        trials, targets, subjects = self._balanced_truth_rows()
        audit = batch_member.validate_scorer_truth(
            trials,
            targets,
            subjects,
            expected_outer_subjects=(10, 11),
        )
        self.assertTrue(audit["row_subject_ids_match_learner_trials"])
        self.assertEqual(audit["class_counts"], [10] * 12)
        self.assertEqual(audit["subject_class_counts"], {"10": [5] * 12, "11": [5] * 12})

        row_misaligned = subjects.copy()
        row_misaligned[[0, 60]] = row_misaligned[[60, 0]]
        with self.assertRaisesRegex(RuntimeError, "row-aligned"):
            batch_member.validate_scorer_truth(
                trials,
                targets,
                row_misaligned,
                expected_outer_subjects=(10, 11),
            )

        globally_unbalanced = targets.copy()
        globally_unbalanced[0] = 1
        with self.assertRaisesRegex(RuntimeError, "ten trials per activity"):
            batch_member.validate_scorer_truth(
                trials,
                globally_unbalanced,
                subjects,
                expected_outer_subjects=(10, 11),
            )

        per_subject_unbalanced = targets.copy()
        per_subject_unbalanced[[0, 65]] = per_subject_unbalanced[[65, 0]]
        with self.assertRaisesRegex(RuntimeError, "five trials per activity"):
            batch_member.validate_scorer_truth(
                trials,
                per_subject_unbalanced,
                subjects,
                expected_outer_subjects=(10, 11),
            )


class ResumeIntegrityContractTests(unittest.TestCase):
    @staticmethod
    def _write_json(path: Path, value: object) -> None:
        path.write_text(
            json.dumps(value, ensure_ascii=False, allow_nan=False),
            encoding="utf-8",
        )

    def test_completed_member_rejects_tampered_completion_identity(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            identity_body = {"schema": "test_identity", "fold": 1, "seed": 0}
            identity = {
                **identity_body,
                "identity_sha256": canonical_hash(identity_body),
            }
            self._write_json(output / "run_identity.json", identity)

            rng = np.random.default_rng(20260915)
            content = rng.normal(size=(16, 4)).astype(np.float32)
            codebook = fit_frozen_e0_codebook(
                content,
                np.repeat(np.arange(4, dtype=np.int64), 4),
                np.repeat(np.arange(2, dtype=np.int64), 8),
                primitive_num=2,
                pca_dim=2,
                seed=0,
                strict_historical=False,
            )
            transform = DescriptorTransform(
                keep_columns=np.asarray([0], dtype=np.int64),
                mean=np.asarray([0.0], dtype=np.float64),
                scale=np.asarray([1.0], dtype=np.float64),
                pca_mean=None,
                pca_components=None,
            )
            codebook_sha = batch_member._codebook_state_sha256(codebook)
            transform_sha = batch_member._transform_state_sha256(transform)
            pre_transform_sha = transform_sha
            save_frozen_artifacts(
                output / "frozen_representation.npz",
                codebook,
                transform,
                ("probe",),
                {"descriptor_profile": LEGACY_DESCRIPTOR_PROFILE},
            )
            np.savez_compressed(
                output / "raw_predictions.npz",
                codebook_state_sha256=np.asarray(codebook_sha),
                transform_state_sha256=np.asarray(transform_sha),
                pre_subject_debias_transform_state_sha256=np.asarray(
                    pre_transform_sha
                ),
                subject_nuisance_projection_strength=np.asarray(
                    0.0, dtype=np.float64
                ),
            )
            raw_prediction_sha = batch_member.sha256_file(
                output / "raw_predictions.npz"
            )
            self._write_json(
                output / "fit_manifest.json",
                {
                    "descriptor_transform_state_sha256": transform_sha,
                    "pre_subject_debias_transform_state_sha256": pre_transform_sha,
                },
            )
            (output / "trajectories_label_free.jsonl").write_text(
                "{}\n", encoding="utf-8"
            )
            self._write_json(
                output / "descriptor_bias_audit.json",
                {
                    "raw_predictions_sha256_before_subject_diagnostic": (
                        raw_prediction_sha
                    )
                },
            )
            self._write_json(
                output / "leakage_audit.json",
                {"raw_predictions_sha256": raw_prediction_sha},
            )
            summary = {
                "schema": batch_member.SCHEMA,
                "outer_trial_count": batch_member.OUTER_TRIAL_COUNT,
                "codebook_state_sha256": codebook_sha,
                "descriptor_transform_state_sha256": transform_sha,
                "pre_subject_debias_transform_state_sha256": pre_transform_sha,
                "subject_nuisance_projection_strength": 0.0,
                "raw_predictions_sha256": raw_prediction_sha,
            }
            self._write_json(output / "summary.json", summary)
            artifact_hashes = {}
            for name in batch_member._output_artifacts():
                artifact = output / name
                if not artifact.exists():
                    artifact.parent.mkdir(parents=True, exist_ok=True)
                    artifact.write_bytes(("artifact:" + name).encode("utf-8"))
                artifact_hashes[name] = batch_member.sha256_file(artifact)
            complete = {
                **summary,
                "run_identity_sha256": identity["identity_sha256"],
                "summary_sha256": batch_member.sha256_file(output / "summary.json"),
                "artifact_sha256": artifact_hashes,
                "complete": True,
            }
            self._write_json(output / "complete.json", complete)
            batch_member.validate_completed_output(
                output, expected_identity=identity
            )

            complete["run_identity_sha256"] = "0" * 64
            self._write_json(output / "complete.json", complete)
            with self.assertRaisesRegex(RuntimeError, "completion identity"):
                batch_member.validate_completed_output(
                    output, expected_identity=identity
                )


class TargetedContrastContractTests(unittest.TestCase):
    @staticmethod
    def _rows() -> list[dict[str, float | int | str]]:
        rows: list[dict[str, float | int | str]] = []
        arms = batch_cv.parse_arms(batch_cv.DEFAULT_ARMS)
        for arm in arms:
            for fold in (1, 2, 3):
                for seed in (0, 5):
                    base = 0.20 + 0.01 * fold + 0.001 * seed
                    if arm.config_id == batch_cv.W128_K64_CONFIG_ID:
                        value = base
                    elif arm.config_id == batch_cv.W128_K128_CONFIG_ID:
                        value = base + 0.01 * fold + 0.001 * seed
                    elif arm.config_id == batch_cv.W64_K128_CONFIG_ID:
                        value = (
                            base
                            + 0.01 * fold
                            + 0.001 * seed
                            - 0.002 * fold
                            - 0.0002 * seed
                        )
                    else:
                        value = base - 0.02
                    row: dict[str, float | int | str] = {
                        "config_id": arm.config_id,
                        "window_size": int(arm.window_size),
                        "window_stride": int(arm.window_stride),
                        "primitive_num": int(arm.primitive_num),
                        "fold": fold,
                        "seed": seed,
                    }
                    row.update({metric: value for metric in batch_cv.PERFORMANCE_METRICS})
                    row.update(
                        {
                            "used_primitive_k": float(arm.primitive_num),
                            "effective_primitive_k": float(arm.primitive_num) / 2.0,
                            "dead_primitive_fraction": 0.0,
                        }
                    )
                    rows.append(row)
        return rows

    def test_two_contrast_keys_directions_and_same_member_pairing_are_fixed(self) -> None:
        definitions = {
            contrast.contrast_id: (
                contrast.minuend_config_id,
                contrast.subtrahend_config_id,
            )
            for contrast in batch_cv.ISOLATED_PAIRED_CONTRASTS
        }
        self.assertEqual(
            definitions,
            {
                "w128_k128_minus_w128_k64": (
                    batch_cv.W128_K128_CONFIG_ID,
                    batch_cv.W128_K64_CONFIG_ID,
                ),
                "w128_k128_minus_w64_k128": (
                    batch_cv.W128_K128_CONFIG_ID,
                    batch_cv.W64_K128_CONFIG_ID,
                ),
            },
        )
        result, flat = batch_cv._isolated_paired_comparisons(
            self._rows(),
            folds=(1, 2, 3),
            seeds=(0, 5),
            bootstrap_seed=20260914,
            bootstrap_replicates=100,
        )
        self.assertEqual(result["contrast_count"], 2)
        self.assertEqual(len(flat), 2 * len(batch_cv.PERFORMANCE_METRICS))

        capacity = result["contrasts"]["w128_k128_minus_w128_k64"]
        capacity_h = capacity["metrics"]["h_score"]
        self.assertEqual(
            capacity_h["same_fold_same_seed_deltas"].keys(),
            {"1", "2", "3"},
        )
        np.testing.assert_allclose(
            capacity_h["same_fold_same_seed_deltas"]["1"], [0.01, 0.015]
        )
        np.testing.assert_allclose(
            capacity_h["fold_mean_paired_deltas_after_averaging_seeds"],
            [0.0125, 0.0225, 0.0325],
        )

        resolution = result["contrasts"]["w128_k128_minus_w64_k128"]
        resolution_h = resolution["metrics"]["h_score"]
        np.testing.assert_allclose(
            resolution_h["same_fold_same_seed_deltas"]["1"], [0.002, 0.003]
        )
        np.testing.assert_allclose(
            resolution_h["fold_mean_paired_deltas_after_averaging_seeds"],
            [0.0025, 0.0045, 0.0065],
        )
        self.assertEqual(
            resolution["pairing_unit"],
            "same_fold_same_seed_difference_then_average_seeds_within_fold",
        )

    def test_isolated_contrast_outputs_are_in_completion_artifact_inventory(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            (output / "grid_manifest.json").write_text("{}", encoding="utf-8")
            complete = batch_cv._write_reports(
                output,
                self._rows(),
                arms=batch_cv.parse_arms(batch_cv.DEFAULT_ARMS),
                folds=(1, 2, 3),
                seeds=(0, 5),
                grid_identity_sha256="test_grid_identity",
                bootstrap_seed=20260914,
                bootstrap_replicates=100,
            )
            expected = {
                "grid_manifest.json",
                "batch_proxy_runs.csv",
                "batch_proxy_summary.csv",
                "batch_proxy_summary.json",
                "isolated_paired_contrasts.csv",
                "isolated_paired_contrasts.json",
            }
            self.assertEqual(set(complete["artifact_sha256"]), expected)
            for name, digest in complete["artifact_sha256"].items():
                self.assertTrue((output / name).is_file())
                self.assertEqual(batch_cv.sha256_file(output / name), digest)
            self.assertEqual(
                set(complete["isolated_paired_contrasts"]["contrasts"]),
                {
                    "w128_k128_minus_w128_k64",
                    "w128_k128_minus_w64_k128",
                },
            )


class SourceIsolationTests(unittest.TestCase):
    @staticmethod
    def _imports(module) -> set[str]:
        tree = ast.parse(Path(module.__file__).read_text(encoding="utf-8"))
        names: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                names.add(node.module)
        return names

    def test_batch_proxy_has_no_gate_or_online_runner_dependency(self) -> None:
        forbidden = (
            "experiments.motion_primitive.strict_registry",
            "experiments.motion_primitive.strict_online_runner",
            "experiments.motion_primitive.strict_online_cv_runner",
        )
        for module in (batch_member, batch_cv):
            with self.subTest(module=module.__name__):
                imports = self._imports(module)
                offending = sorted(
                    name
                    for name in imports
                    if any(
                        name == prefix or name.startswith(prefix + ".")
                        for prefix in forbidden
                    )
                )
                self.assertEqual(offending, [])


if __name__ == "__main__":
    unittest.main()
