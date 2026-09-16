import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from experiments.motion_primitive.run_one_stage_cv import (
    RUN_CORE_SOURCE_RELATIVE_PATHS,
    _aggregate,
    _canonical_hash,
    _load_rows,
    build_parser,
    run,
)
from experiments.motion_primitive.train_one_stage import (
    CORE_SOURCE_RELATIVE_PATHS,
    REQUIRED_COMPLETION_ARTIFACTS,
)


class OneStageCVRunnerTests(unittest.TestCase):
    def test_member_source_hash_file_set_is_shared_with_train_runner(self):
        self.assertEqual(
            RUN_CORE_SOURCE_RELATIVE_PATHS, tuple(CORE_SOURCE_RELATIVE_PATHS)
        )
        self.assertIn(
            "experiments/motion_primitive/raw_changepoint.py",
            RUN_CORE_SOURCE_RELATIVE_PATHS,
        )
        self.assertIn("requirements-one-stage.txt", RUN_CORE_SOURCE_RELATIVE_PATHS)

    def test_dry_run_builds_exact_seven_fold_two_profile_grid(self):
        with tempfile.TemporaryDirectory() as temporary:
            args = build_parser().parse_args(
                [
                    "--output-root",
                    str(Path(temporary) / "grid"),
                    "--data-root",
                    str(Path(temporary) / "data"),
                    "--profiles",
                    "J0-U,J0-T",
                    "--folds",
                    "1,2,3,4,5,6,7",
                    "--seeds",
                    "50",
                    "--dry-run",
                ]
            )
            result = run(args)
            self.assertTrue(result["dry_run"])
            self.assertEqual(len(result["commands"]), 14)
            joined = [" ".join(command) for command in result["commands"]]
            self.assertEqual(sum("--profile J0-U" in item for item in joined), 7)
            self.assertEqual(sum("--profile J0-T" in item for item in joined), 7)
            self.assertTrue(
                all("--trajectory-input-mode primitive_only" in item for item in joined)
            )
            self.assertTrue(
                all(
                    "--checkpoint-selection common_unsupervised" in item
                    for item in joined
                )
            )
            self.assertTrue(all("--codebook-init learnable" in item for item in joined))
            self.assertTrue(all("--codebook-update gradient" in item for item in joined))
            self.assertTrue(all("--loss-ramp-epochs 10" in item for item in joined))
            self.assertTrue(all("--no-revive-unused-codes" in item for item in joined))
            self.assertTrue(all("--local-encoder resnet1d" in item for item in joined))
            self.assertTrue(
                all("--minimum-hard-effective-code-fraction 0.2" in item for item in joined)
            )
            self.assertTrue(all("--resume" not in item for item in joined))
            self.assertEqual(
                result["identity"]["schema"], "one_stage_motion_trajectory_cv_v5"
            )
            self.assertTrue(
                result["identity"]["primary_motion_primitive_cgcd_arm"]
            )
            self.assertEqual(
                result["identity"]["trajectory_assignment_forward"],
                "hard_one_hot_straight_through_deterministic",
            )

    def test_output_root_rejects_changed_grid_identity(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "grid"
            first = build_parser().parse_args(
                ["--output-root", str(root), "--folds", "1", "--dry-run"]
            )
            run(first)
            changed = build_parser().parse_args(
                [
                    "--output-root",
                    str(root),
                    "--folds",
                    "1",
                    "--codebook-size",
                    "16",
                    "--dry-run",
                ]
            )
            with self.assertRaisesRegex(RuntimeError, "different CV identity"):
                run(changed)

    def test_protocol_and_aggregation_options_are_cv_identity_fields(self):
        changes = (
            ("--trajectory-input-mode", "state_only"),
            ("--checkpoint-selection", "profile_default"),
            ("--codebook-init", "kmeans++"),
            ("--codebook-ema-decay", "0.9"),
            ("--minimum-hard-effective-code-fraction", "0.3"),
            ("--bootstrap-replicates", "101"),
            ("--aggregate-seed", "9"),
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "grid"
            baseline = build_parser().parse_args(
                ["--output-root", str(root), "--folds", "1", "--dry-run"]
            )
            baseline_result = run(baseline)
            protocol = baseline_result["identity"]["aggregation_protocol"]
            self.assertEqual(protocol["bootstrap_replicates"], 10000)
            self.assertEqual(protocol["aggregate_seed"], 20260908)
            for option, value in changes:
                with self.subTest(option=option):
                    changed = build_parser().parse_args(
                        [
                            "--output-root",
                            str(root),
                            "--folds",
                            "1",
                            option,
                            value,
                            "--dry-run",
                        ]
                    )
                    with self.assertRaisesRegex(RuntimeError, "different CV identity"):
                        run(changed)

    def test_learnable_codebook_rejects_ema_before_grid_creation(self):
        with tempfile.TemporaryDirectory() as temporary:
            args = build_parser().parse_args(
                [
                    "--output-root",
                    str(Path(temporary) / "grid"),
                    "--folds",
                    "1",
                    "--codebook-update",
                    "ema",
                    "--dry-run",
                ]
            )
            with self.assertRaisesRegex(ValueError, "learnable requires"):
                run(args)

    def test_fold_block_aggregate_averages_seeds_before_statistics(self):
        rows = []
        for profile, offset in (("J0-U", 0.0), ("J0-T", 0.1)):
            for fold in (1, 2):
                for seed in (5, 50):
                    row = {
                        "profile": profile,
                        "fold": fold,
                        "seed": seed,
                        "order_h_score_drop": 0.02 + offset,
                        "sitting_recall": 0.5 + offset,
                        "standing_recall": 0.6 + offset,
                        "sit_stand_balanced_accuracy": 0.55 + offset,
                        "occupancy_fraction": 0.75,
                        "effective_code_count": 8.0,
                        "validation_hard_occupancy_fraction": 0.75,
                        "validation_hard_effective_code_count": 8.0,
                        "validation_hard_effective_code_fraction": 0.25,
                        "validation_hard_max_code_share": 0.2,
                        "validation_assignment_margin": 0.1,
                        "validation_local_feature_effective_rank": 6.0,
                        "validation_local_feature_centroid_norm": 0.5,
                        "token_subject_nmi": 0.1,
                        "primitive_segment_start_repeatability_margin": 0.2,
                        "mean_primitive_segment_count": 12.0,
                    }
                    row.update(
                        {
                            metric: 0.4 + 0.01 * fold + offset
                            for metric in (
                                "all_accuracy",
                                "old_accuracy",
                                "new_accuracy",
                                "h_score",
                                "macro_f1",
                            )
                        }
                    )
                    rows.append(row)
        result = _aggregate(
            rows,
            ["J0-U", "J0-T"],
            [1, 2],
            [5, 50],
            100,
            7,
        )
        self.assertAlmostEqual(
            result["profiles"]["J0-T"]["h_score"]["mean"]
            - result["profiles"]["J0-U"]["h_score"]["mean"],
            0.1,
        )
        self.assertEqual(
            result["paired_J0_T_vs_J0_U"]["h_score"]["permutation_count"],
            4,
        )
        self.assertEqual(
            result["aggregation_protocol"]["bootstrap_replicates"], 100
        )
        self.assertEqual(result["aggregation_protocol"]["aggregate_seed"], 7)
        self.assertEqual(
            result["aggregation_protocol"]["prespecified_primary_endpoint"],
            "h_score",
        )
        self.assertEqual(
            result["paired_J0_T_vs_J0_U"]["h_score"]["inference_role"],
            "prespecified_primary",
        )
        self.assertEqual(
            result["paired_J0_T_vs_J0_U"]["all_accuracy"]["inference_role"],
            "exploratory_unadjusted",
        )

    def test_member_artifact_manifest_is_exhaustively_verified(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run_dir = root / "profile_J0-U" / "fold_01_seed_50"
            run_dir.mkdir(parents=True)
            source_identity = "source-identity"
            runtime_payload = {"python": "test-runtime"}
            runtime_identity = _canonical_hash(runtime_payload)
            data_protocol_identity = "data-protocol-identity"
            raw_manifest_identity = "raw-manifest-identity"
            identity_payload = {
                "schema": "one_stage_motion_trajectory_run_v5",
                "profile": "J0-U",
                "fold": 1,
                "seed": 50,
                "data_protocol_identity": data_protocol_identity,
                "model_config": {
                    "trajectory_input_mode": "primitive_only",
                    "local_encoder_type": "resnet1d",
                    "use_gumbel_training": False,
                },
                "training_parameters": {
                    "checkpoint_selection": "common_unsupervised",
                    "codebook_init": "learnable",
                    "codebook_update": "gradient",
                    "revive_unused_codes": False,
                },
                "source_identity": {"identity_sha256": source_identity},
                "runtime_environment": {
                    **runtime_payload,
                    "identity_sha256": runtime_identity,
                },
            }
            run_identity = _canonical_hash(identity_payload)
            manifest = {
                **identity_payload,
                "run_identity_sha256": run_identity,
                "data_audit": {
                    "identity_sha256": data_protocol_identity,
                    "manifest_sha256": raw_manifest_identity,
                },
            }
            (run_dir / "run_manifest.json").write_text(
                json.dumps(manifest), encoding="utf-8"
            )
            for relative in REQUIRED_COMPLETION_ARTIFACTS:
                path = run_dir / relative
                if not path.exists():
                    path.write_bytes((relative + "\n").encode("utf-8"))
            checkpoint_path = run_dir / "checkpoint_best.pt"
            checkpoint_hash = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
            summary = {
                "schema": "one_stage_motion_trajectory_run_v5",
                "run_identity_sha256": run_identity,
                "profile": "J0-U",
                "fold": 1,
                "seed": 50,
                "trajectory_input_mode": "primitive_only",
                "local_encoder": "resnet1d",
                "trajectory_assignment_forward": (
                    "hard_one_hot_straight_through_deterministic"
                ),
                "primary_motion_primitive_cgcd_arm": True,
                "checkpoint_selection": "common_unsupervised",
                "codebook_init": "learnable",
                "codebook_update": "gradient",
                "revive_unused_codes": False,
                "selected_epoch": 1,
                "selected_checkpoint_sha256": checkpoint_hash,
                "selection_validation": {
                    "checkpoint_eligible": True,
                    "hard_occupancy_fraction": 0.75,
                    "hard_effective_code_count": 9.0,
                    "hard_effective_code_fraction": 9.0 / 32.0,
                    "hard_max_code_share": 0.2,
                    "mean_assignment_margin": 0.1,
                    "local_feature_effective_rank": 6.0,
                    "local_feature_centroid_norm": 0.5,
                },
                "evaluation": {
                    "cgcd_metrics": {
                        "all_accuracy": 0.5,
                        "old_accuracy": 0.6,
                        "new_accuracy": 0.4,
                        "h_score": 0.48,
                        "macro_f1": 0.45,
                        "per_class_recall": {"Sitting": 0.7, "Standing": 0.8},
                    },
                    "order_shuffle_control": {
                        "identity_control_minus_shuffled": {"h_score": 0.03}
                    },
                    "codebook_diagnostics": {
                        "effective_code_count": 9.0,
                        "token_subject_nmi": 0.1,
                        "occupancy_fraction": 0.75,
                    },
                    "primitive_segment_start_repeatability": {
                        "same_minus_different_margin": 0.2
                    },
                    "primitive_segment_counts": {"mean": 11.0},
                },
            }
            summary_path = run_dir / "summary.json"
            summary_path.write_text(json.dumps(summary), encoding="utf-8")
            artifact_hashes = {
                relative: hashlib.sha256((run_dir / relative).read_bytes()).hexdigest()
                for relative in REQUIRED_COMPLETION_ARTIFACTS
            }
            complete = {
                "run_identity_sha256": run_identity,
                "summary_sha256": artifact_hashes["summary.json"],
                "selected_checkpoint_sha256": checkpoint_hash,
                "artifact_sha256": artifact_hashes,
            }
            complete_path = run_dir / "complete.json"
            complete_path.write_text(json.dumps(complete), encoding="utf-8")

            rows = _load_rows(
                root,
                ["J0-U"],
                [1],
                [50],
                expected_member_source_identity=source_identity,
                expected_trajectory_input_mode="primitive_only",
                expected_checkpoint_selection="common_unsupervised",
                expected_codebook_init="learnable",
                expected_codebook_update="gradient",
            )
            self.assertEqual(len(rows), 1)
            self.assertTrue(rows[0]["primary_motion_primitive_cgcd_arm"])
            self.assertEqual(rows[0]["runtime_identity_sha256"], runtime_identity)
            self.assertEqual(rows[0]["raw_manifest_sha256"], raw_manifest_identity)

            missing = dict(complete)
            missing["artifact_sha256"] = dict(artifact_hashes)
            missing["artifact_sha256"].pop("activity_codebook_heatmap.png")
            complete_path.write_text(json.dumps(missing), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "missing or unexpected"):
                _load_rows(root, ["J0-U"], [1], [50])

            complete_path.write_text(json.dumps(complete), encoding="utf-8")
            trajectory_plot = run_dir / "fixed_trajectories_and_predictions.png"
            trajectory_plot.write_bytes(b"tampered")
            with self.assertRaisesRegex(RuntimeError, "SHA256 check"):
                _load_rows(root, ["J0-U"], [1], [50])

    def test_invalid_zero_mask_ratio_is_rejected_before_grid_creation(self):
        with tempfile.TemporaryDirectory() as temporary:
            args = build_parser().parse_args(
                [
                    "--output-root",
                    str(Path(temporary) / "grid"),
                    "--folds",
                    "1",
                    "--trajectory-mask-ratio",
                    "0",
                    "--dry-run",
                ]
            )
            with self.assertRaisesRegex(ValueError, "strictly between zero and one"):
                run(args)


if __name__ == "__main__":
    unittest.main()
