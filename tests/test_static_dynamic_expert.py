from __future__ import annotations

import argparse
import inspect
import json
import tempfile
import unittest
from unittest import mock
from pathlib import Path

import numpy as np

from experiments.motion_primitive import run_static_dynamic_expert_fixedsplit01 as suite
from experiments.motion_primitive import run_dual_encoder_static_dynamic_fixedsplit01 as dual
from experiments.motion_primitive import static_dynamic_batch_proxy as member
from experiments.motion_primitive.frozen_e0 import DescriptorTransform
from experiments.motion_primitive.static_dynamic_expert import (
    DEFAULT_STATIC_BLOCK_WEIGHTS,
    STATIC_BLOCK_ORDER,
    fit_robust_block_transform,
    fit_count_proportional_branch_clusters,
    fit_static_dynamic_gate,
    fit_static_expert,
    merge_branch_predictions,
    physical_block_matrices,
)


def _windows(signal: np.ndarray, window_size: int = 128, stride: int = 64) -> tuple[np.ndarray, np.ndarray]:
    starts = np.arange(0, signal.shape[1] - window_size + 1, stride, dtype=np.int64)
    return np.stack([signal[:, start : start + window_size] for start in starts]), starts


def _signal(scale: float, phase: float, gravity: np.ndarray | None = None) -> np.ndarray:
    time = np.arange(512, dtype=np.float64) / 100.0
    direction = np.asarray([0.0, 0.0, 1.0]) if gravity is None else np.asarray(gravity)
    direction = direction / np.linalg.norm(direction)
    acc = direction[:, None] + scale * np.vstack(
        [
            np.sin(2.0 * np.pi * 1.4 * time + phase),
            np.cos(2.0 * np.pi * 0.9 * time + phase),
            np.sin(2.0 * np.pi * 2.1 * time + phase),
        ]
    )
    gyro = (30.0 * scale) * np.vstack(
        [
            np.cos(2.0 * np.pi * 1.2 * time + phase),
            np.sin(2.0 * np.pi * 0.7 * time + phase),
            np.cos(2.0 * np.pi * 1.8 * time + phase),
        ]
    )
    return np.vstack([acc, gyro]).astype(np.float32)


class StaticDynamicExpertTests(unittest.TestCase):
    def test_alpha_a025_projection_matches_registered_formula(self) -> None:
        transform = DescriptorTransform(
            keep_columns=np.arange(3, dtype=np.int64),
            mean=np.zeros(3, dtype=np.float64),
            scale=np.ones(3, dtype=np.float64),
            pca_mean=None,
            pca_components=None,
            nuisance_basis=np.asarray([[1.0, 0.0, 0.0]], dtype=np.float64),
            nuisance_singular_values=np.asarray([1.0], dtype=np.float64),
            nuisance_explained_fraction=1.0,
            nuisance_fit_subject_count=2,
            nuisance_fit_trial_count=4,
            nuisance_projection_strength=0.25,
        )
        values = np.asarray([[1.0, 2.0, 3.0], [-3.0, 1.0, 2.0]])
        base = values / np.linalg.norm(values, axis=1, keepdims=True)
        basis = np.asarray([[1.0, 0.0, 0.0]], dtype=np.float64)
        expected = base - 0.25 * (base @ basis.T) @ basis
        expected /= np.linalg.norm(expected, axis=1, keepdims=True)
        np.testing.assert_allclose(transform.transform(values), expected, atol=1e-7)

    def test_gate_uses_low_energy_component_without_truth(self) -> None:
        raw_windows = []
        starts = []
        for index in range(8):
            window, local = _windows(_signal(0.002 + index * 0.0002, index * 0.1))
            raw_windows.append(window)
            starts.append(local)
        for index in range(8):
            window, local = _windows(_signal(0.15 + index * 0.005, index * 0.1))
            raw_windows.append(window)
            starts.append(local)
        blocks = physical_block_matrices(raw_windows, starts)
        result = fit_static_dynamic_gate(blocks["gate"], seed=5, n_init=10)
        self.assertGreaterEqual(int(result.is_static[:8].sum()), 7)
        self.assertLessEqual(int(result.is_static[8:].sum()), 1)
        self.assertEqual(result.static_probability.shape, (16,))
        self.assertEqual(len(result.state.state_sha256), 64)

    def test_gate_and_expert_fit_apis_do_not_accept_truth_or_subjects(self) -> None:
        for function in (
            fit_static_dynamic_gate,
            fit_static_expert,
            fit_count_proportional_branch_clusters,
        ):
            parameters = set(inspect.signature(function).parameters)
            self.assertFalse(parameters & {"labels", "targets", "subject_ids", "subjects"})

    def test_confidence_preserving_block_does_not_amplify_zero_signal(self) -> None:
        values = np.asarray(
            [[-2.0, -1.0], [-1.0, -0.5], [1.0, 0.5], [2.0, 1.0]],
            dtype=np.float64,
        )
        transform = fit_robust_block_transform(
            values, maximum_components=None, normalization="confidence_preserving"
        )
        at_median = transform.transform(transform.median[None, :])
        strong = transform.transform(np.asarray([[4.0, 2.0]], dtype=np.float64))
        self.assertAlmostEqual(float(np.linalg.norm(at_median)), 0.0, places=7)
        self.assertGreater(float(np.linalg.norm(strong)), 0.9)
        self.assertLessEqual(float(np.linalg.norm(strong)), 1.000001)
        with self.assertRaisesRegex(ValueError, "maximum_components"):
            fit_robust_block_transform(
                values, maximum_components=0, normalization="unit"
            )

    def test_static_expert_has_explicit_four_block_distance_budget(self) -> None:
        rng = np.random.default_rng(17)
        blocks = {
            "posture": rng.normal(size=(24, 8)),
            "signed_gravity": rng.normal(size=(24, 9)),
            "energy": rng.normal(size=(24, 11)),
            "motion_primitive": rng.normal(size=(24, 131)),
        }
        model, features = fit_static_expert(blocks)
        self.assertEqual(features.shape[0], 24)
        cursor = 0
        for index, name in enumerate(STATIC_BLOCK_ORDER):
            width = getattr(model, name).output_dim
            squared_norm = np.square(features[:, cursor : cursor + width]).sum(axis=1)
            self.assertTrue(
                np.all(squared_norm <= DEFAULT_STATIC_BLOCK_WEIGHTS[name] + 1e-5)
            )
            cursor += width
        self.assertEqual(cursor, features.shape[1])

    def test_unlabeled_count_proportional_allocation_is_7_plus_5(self) -> None:
        rng = np.random.default_rng(23)
        dynamic = rng.normal(size=(70, 6))
        static = rng.normal(size=(50, 5))
        selection = fit_count_proportional_branch_clusters(
            dynamic,
            static,
            total_clusters=12,
            minimum_dynamic_clusters=6,
            seed=50,
            n_init=3,
            max_iter=50,
        )
        mask = np.zeros(120, dtype=bool)
        mask[70:] = True
        merged = merge_branch_predictions(
            mask,
            selection.dynamic_labels,
            selection.static_labels,
            dynamic_k=selection.dynamic_k,
            total_clusters=12,
        )
        self.assertEqual(set(np.unique(merged).tolist()), set(range(12)))
        self.assertEqual((selection.dynamic_k, selection.static_k), (7, 5))
        self.assertEqual(selection.dynamic_k + selection.static_k, 12)


class FixedSplitSuiteContractTests(unittest.TestCase):
    def test_suite_has_exactly_three_reported_arms(self) -> None:
        self.assertEqual(
            member.ARMS,
            (
                "B0_global_trajectory",
                "E1_gate_static_expert",
                "E2_gate_static_expert_soft_a025",
            ),
        )

    def test_raw_predictions_are_frozen_before_truth_join(self) -> None:
        source = inspect.getsource(member._run_deterministic)
        freeze_position = source.index('output / "raw_predictions.npz"')
        hash_position = source.index('sha256_file(output / "raw_predictions.npz")')
        truth_position = source.index("protocol.truth.join(outer_ids)")
        self.assertLess(freeze_position, hash_position)
        self.assertLess(hash_position, truth_position)
        pretruth = source[:truth_position]
        self.assertNotIn("protocol.truth", pretruth)
        self.assertNotIn("subject_values", pretruth)
        self.assertIn("offline_subjects", pretruth)
        self.assertIn("E1/E2 static predictions must be bitwise identical", pretruth)
        self.assertEqual(pretruth.count("item.subject_id"), 1)

    def test_checkpoint_resolution_uses_nested_w128_fold1_path(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkpoint = (
                root
                / "encoders"
                / "w128_s64"
                / "a2"
                / "fold_01_seed_0"
                / "motion_encoder_final.pt"
            )
            checkpoint.parent.mkdir(parents=True)
            checkpoint.write_bytes(b"test")
            self.assertEqual(suite.resolve_a2_checkpoint(root, 0), checkpoint.resolve())

    def test_four_commands_share_encoder_but_have_distinct_run_seeds(self) -> None:
        args = argparse.Namespace(
            npz_path="dataset.npz",
            output_root="outputs",
            encoder_seed=0,
            encode_batch_size=1024,
            gate_n_init=20,
            kmeans_n_init=50,
            kmeans_max_iter=300,
            subject_nuisance_max_rank=4,
            subject_nuisance_explained_variance=0.90,
            subject_nuisance_projection_strength=0.25,
            static_motion_primitive_pca_dim=8,
            static_posture_weight=0.35,
            static_gravity_weight=0.35,
            static_energy_weight=0.20,
            static_motion_primitive_weight=0.10,
            device="cuda",
            allow_smoke_a2=False,
            resume=True,
        )
        checkpoint = Path("encoder.pt").resolve()
        commands = [
            suite.build_member_command(args, checkpoint=checkpoint, run_seed=seed)
            for seed in suite.DEFAULT_RUN_SEEDS
        ]
        self.assertEqual(
            {command[command.index("--a2-checkpoint") + 1] for command in commands},
            {str(checkpoint)},
        )
        self.assertEqual(
            [int(command[command.index("--run-seed") + 1]) for command in commands],
            list(suite.DEFAULT_RUN_SEEDS),
        )
        self.assertTrue(all("--fold" not in command and "--folds" not in command for command in commands))
        self.assertTrue(
            all(
                command[command.index("--subject-nuisance-projection-strength") + 1]
                == "0.25"
                for command in commands
            )
        )

    def test_expert_state_serialization_includes_complete_soft_projection(self) -> None:
        transform = DescriptorTransform(
            keep_columns=np.arange(3, dtype=np.int64),
            mean=np.zeros(3),
            scale=np.ones(3),
            pca_mean=None,
            pca_components=None,
            nuisance_basis=np.asarray([[1.0, 0.0, 0.0]]),
            nuisance_singular_values=np.asarray([2.0]),
            nuisance_explained_fraction=0.8,
            nuisance_fit_subject_count=10,
            nuisance_fit_trial_count=300,
            nuisance_projection_strength=0.25,
        )
        arrays = member._transform_arrays("soft_dynamic", transform)
        self.assertEqual(
            set(arrays),
            {
                "soft_dynamic_keep_columns",
                "soft_dynamic_mean",
                "soft_dynamic_scale",
                "soft_dynamic_pca_mean",
                "soft_dynamic_pca_components",
                "soft_dynamic_nuisance_basis",
                "soft_dynamic_nuisance_singular_values",
                "soft_dynamic_nuisance_explained_fraction",
                "soft_dynamic_nuisance_fit_subject_count",
                "soft_dynamic_nuisance_fit_trial_count",
                "soft_dynamic_nuisance_projection_strength",
            },
        )
        self.assertEqual(float(arrays["soft_dynamic_nuisance_projection_strength"]), 0.25)
        self.assertEqual(int(arrays["soft_dynamic_nuisance_fit_trial_count"]), 300)

    def test_run_seed_grid_is_exactly_four_unique_values(self) -> None:
        self.assertEqual(suite.parse_integer_list("0,5,50,500"), (0, 5, 50, 500))
        with self.assertRaises(ValueError):
            suite.parse_integer_list("0,5")
        with self.assertRaises(ValueError):
            suite.parse_integer_list("0,5,50,501")

    def test_suite_reports_soft_subject_delta_against_shared_E1(self) -> None:
        members = []
        for index, run_seed in enumerate(suite.DEFAULT_RUN_SEEDS):
            arms = {}
            for arm, offset in (
                ("B0_global_trajectory", 0.0),
                ("E1_gate_static_expert", 0.1),
                ("E2_gate_static_expert_soft_a025", 0.15),
            ):
                arms[arm] = {
                    metric: float(index + offset) for metric in suite.METRICS
                }
            members.append({"run_seed": run_seed, "arms": arms})
        aggregate = suite._aggregate(members)
        for metric in suite.METRICS:
            self.assertAlmostEqual(
                aggregate["paired_contrast"][metric]["mean_delta_E1_minus_B0"],
                0.1,
            )
            self.assertAlmostEqual(
                aggregate["soft_subject_paired_contrast"][metric][
                    "mean_delta_E2_minus_E1"
                ],
                0.05,
            )

    def test_member_completion_is_bound_to_identity_summary_and_exact_inventory(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            identity_body = {"schema": member.IDENTITY_SCHEMA, "run_seed": 0}
            identity = {
                **identity_body,
                "identity_sha256": member.canonical_hash(identity_body),
            }
            (root / "run_identity.json").write_text(
                json.dumps(identity), encoding="utf-8"
            )
            summary = {
                "schema": member.SCHEMA,
                "raw_predictions_sha256": "pending",
            }
            for name in member.ARTIFACT_NAMES:
                path = root / name
                if name == "summary.json":
                    continue
                path.write_bytes(("artifact:" + name).encode("utf-8"))
            raw_sha = member.sha256_file(root / "raw_predictions.npz")
            summary["raw_predictions_sha256"] = raw_sha
            (root / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
            complete = {
                **summary,
                "run_identity_sha256": identity["identity_sha256"],
                "summary_sha256": member.sha256_file(root / "summary.json"),
                "artifact_sha256": {
                    name: member.sha256_file(root / name) for name in member.ARTIFACT_NAMES
                },
                "complete": True,
            }
            (root / "complete.json").write_text(json.dumps(complete), encoding="utf-8")
            member.validate_completed_output(root, expected_identity=identity)
            del complete["artifact_sha256"]["metrics.json"]
            (root / "complete.json").write_text(json.dumps(complete), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "exact registered artifact"):
                member.validate_completed_output(root, expected_identity=identity)

    def test_member_output_lock_rejects_a_second_writer(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with member._exclusive_output_lock(root):
                with self.assertRaisesRegex(RuntimeError, "already locked"):
                    with member._exclusive_output_lock(root):
                        self.fail("A second writer acquired the member lock.")

    def test_suite_identity_allows_only_its_lock_before_identity(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with member._exclusive_output_lock(root, lock_name=".suite.lock"):
                identity = suite._prepare_identity(root, {"schema": "test"})
            self.assertTrue((root / "suite_identity.json").is_file())
            self.assertEqual(identity["identity_sha256"], suite.canonical_hash({"schema": "test"}))


class DualEncoderSuiteContractTests(unittest.TestCase):
    def test_dry_run_builds_fresh_training_then_two_downstream_branches(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            npz_path = root / "uschad_windows.npz"
            npz_path.write_bytes(b"npz")
            reused = root / "motion_encoder_final.pt"
            reused.write_bytes(b"encoder")
            output = root / "output"
            args = dual.build_parser().parse_args(
                [
                    "--npz-path",
                    str(npz_path),
                    "--reused-encoder-root",
                    str(reused),
                    "--output-root",
                    str(output),
                    "--device",
                    "cpu",
                    "--resume",
                    "--dry-run",
                ]
            )
            with (
                mock.patch.object(
                    dual, "EXPECTED_NPZ_SHA256", dual.sha256_file(npz_path)
                ),
                mock.patch.object(
                    dual, "EXPECTED_REUSED_A2_SHA256", dual.sha256_file(reused)
                ),
            ):
                result = dual.run(args)
            self.assertTrue(result["dry_run"])
            self.assertEqual(len(result["commands"]), 4)
            self.assertIn("pretrain_window_encoder.py", result["commands"][0][1])
            self.assertIn("train_motion_encoder.py", result["commands"][1][1])
            self.assertTrue(
                all(
                    "run_static_dynamic_expert_fixedsplit01.py" in command[1]
                    for command in result["commands"][2:]
                )
            )
            downstream_outputs = {
                command[command.index("--output-root") + 1]
                for command in result["commands"][2:]
            }
            self.assertEqual(
                downstream_outputs,
                {
                    str((output / "branches" / branch).resolve())
                    for branch in dual.ENCODER_BRANCHES
                },
            )
            fresh_checkpoint = result["commands"][1][
                result["commands"][1].index("--output-dir") + 1
            ]
            self.assertIn("R1_retrained_current", fresh_checkpoint)

    def test_encoder_comparison_is_paired_by_identical_run_seed(self) -> None:
        results = {}
        for branch_index, branch in enumerate(dual.ENCODER_BRANCHES):
            arms = {}
            for arm in member.ARMS:
                metrics = {}
                for metric in suite.METRICS:
                    values = {
                        str(seed): float(seed + branch_index)
                        for seed in suite.DEFAULT_RUN_SEEDS
                    }
                    metrics[metric] = {
                        "mean": float(np.mean(list(values.values()))),
                        "values_by_run_seed": values,
                    }
                arms[arm] = metrics
            results[branch] = {
                "arms": arms,
                "paired_contrast": {},
                "soft_subject_paired_contrast": {},
            }
        comparison = dual._comparison(results)
        for arm in member.ARMS:
            for metric in suite.METRICS:
                record = comparison["arms"][arm][metric]
                self.assertAlmostEqual(record["mean_delta_retrained_minus_reuse"], 1.0)
                self.assertEqual(
                    set(record["paired_delta_by_run_seed"]),
                    {str(seed) for seed in suite.DEFAULT_RUN_SEEDS},
                )


if __name__ == "__main__":
    unittest.main()
