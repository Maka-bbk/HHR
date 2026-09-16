import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from experiments.motion_primitive.trajectory_ablation import (
    RobustScaler,
    SourceSignalRepository,
    TrajectoryRun,
    TrialTrajectory,
    build_trajectories,
    codebook_dynamic_cost,
    evaluate_distance_matrix,
    fit_state_scaler,
    local_substitution_cost,
    tie_aware_confusion,
    trajectory_distance_matrix,
    transform_trial_states,
    shuffled_trajectories,
)
from experiments.motion_primitive.run_trajectory_ablation import (
    GROUP_NAMES,
    build_gate_report,
    build_paired_summary,
    load_fixed_window_baselines,
    trial_grid_hash_from_jsonl,
)


def make_run(token, state, duration=1.0, start=0):
    return TrajectoryRun(
        token=int(token),
        start_sample=int(start),
        end_sample_exclusive=int(start + round(duration * 100)),
        duration_seconds=float(duration),
        state=np.asarray(state, dtype=np.float32),
    )


def make_trial(trial_id, subject, label, runs, role=1, activity_name=None):
    return TrialTrajectory(
        trial_global_id=int(trial_id),
        trial_key=f"T{trial_id}",
        split_role=int(role),
        subject_id=int(subject),
        activity_label=int(label),
        activity_name=activity_name or f"A{label}",
        trial_number=1,
        runs=tuple(runs),
    )


class TrajectoryAblationCoreTests(unittest.TestCase):
    def test_dynamic_codebook_cost_is_symmetric_bounded_and_zero_diagonal(self):
        centers = np.asarray([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])
        cost = codebook_dynamic_cost(centers)
        self.assertTrue(np.allclose(cost, cost.T))
        self.assertTrue(np.allclose(np.diag(cost), 0.0))
        self.assertTrue(np.all((0.0 <= cost) & (cost <= 1.0)))
        self.assertAlmostEqual(float(cost[0, 1]), 0.5)
        self.assertAlmostEqual(float(cost[0, 2]), 1.0)

    def test_four_groups_keep_tokens_but_change_local_cost(self):
        zeros = np.zeros(12, dtype=np.float32)
        ones = np.ones(12, dtype=np.float32)
        left = [make_run(0, zeros, duration=1.0)]
        right = [make_run(1, ones, duration=2.0)]
        dynamic = np.asarray([[0.0, 0.2], [0.2, 0.0]], dtype=np.float32)
        scales = {"state_distance_p95": 1.0, "log_duration_difference_p95": 1.0}
        costs = [
            local_substitution_cost(
                group,
                left,
                0,
                right,
                0,
                dynamic,
                scales,
                0.25,
                0.15,
                0.15,
            )
            for group in [
                "g1_hard_rle",
                "g2_dynamic_soft",
                "g3_dynamic_state",
                "g4_dynamic_state_duration_context",
            ]
        ]
        self.assertAlmostEqual(costs[0], 1.0)
        self.assertAlmostEqual(costs[1], 0.2, places=6)
        self.assertGreater(costs[2], costs[1])
        self.assertTrue(all(0.0 <= value <= 1.0 for value in costs))
        self.assertEqual(tuple(run.token for run in left), (0,))
        self.assertEqual(tuple(run.token for run in right), (1,))

    def test_state_scaler_is_fitted_from_caller_supplied_trials_only(self):
        fit = [
            make_trial(0, 1, 0, [make_run(0, np.zeros(12))], role=0),
            make_trial(1, 2, 0, [make_run(0, np.full(12, 2.0))], role=0),
        ]
        evaluation = [
            make_trial(2, 3, 1, [make_run(0, np.full(12, 1e6))], role=1)
        ]
        scaler = fit_state_scaler(fit)
        self.assertTrue(np.all(scaler.center < 10.0))
        transformed_fit = transform_trial_states(fit, scaler)
        transformed_eval = transform_trial_states(evaluation, scaler)
        self.assertLess(
            float(np.max(np.abs(transformed_fit[0].runs[0].state))), 10.0
        )
        self.assertGreater(
            float(np.min(transformed_eval[0].runs[0].state)), 1e4
        )

    def test_state_residual_breaks_identical_token_zero_distance(self):
        dynamic = np.asarray([[0.0]], dtype=np.float32)
        scales = {"state_distance_p95": 1.0, "log_duration_difference_p95": 1.0}
        trials = [
            make_trial(0, 1, 0, [make_run(0, np.zeros(12))]),
            make_trial(1, 2, 1, [make_run(0, np.ones(12))]),
        ]
        hard = trajectory_distance_matrix(
            trials, "g1_hard_rle", dynamic, scales, 0.25, 0.15, 0.15
        )
        state = trajectory_distance_matrix(
            trials, "g3_dynamic_state", dynamic, scales, 0.25, 0.15, 0.15
        )
        self.assertEqual(float(hard[0, 1]), 0.0)
        self.assertGreater(float(state[0, 1]), 0.0)

    def test_tie_aware_confusion_splits_prediction_mass(self):
        distances = np.asarray(
            [
                [0.0, 0.8, 0.4, 0.4],
                [0.8, 0.0, 0.4, 0.4],
                [0.4, 0.4, 0.0, 0.8],
                [0.4, 0.4, 0.8, 0.0],
            ],
            dtype=np.float32,
        )
        labels = [0, 1, 0, 1]
        subjects = [1, 1, 2, 2]
        result = tie_aware_confusion(distances, labels, subjects)
        self.assertEqual(result["mean_tied_nearest_count"], 2.0)
        self.assertAlmostEqual(float(result["confusion_counts"].sum()), 4.0)
        self.assertAlmostEqual(result["accuracy"], 0.5)

    def test_old_and_novel_query_metrics_keep_all_class_candidates(self):
        trials = [
            make_trial(0, 1, 0, [make_run(0, np.zeros(12))], activity_name="Sitting"),
            make_trial(1, 1, 1, [make_run(0, np.zeros(12))], activity_name="Standing"),
            make_trial(2, 2, 0, [make_run(0, np.zeros(12))], activity_name="Sitting"),
            make_trial(3, 2, 1, [make_run(0, np.zeros(12))], activity_name="Standing"),
        ]
        distances = np.asarray(
            [
                [0.0, 1.0, 0.9, 0.1],
                [1.0, 0.0, 0.1, 0.9],
                [0.9, 0.1, 0.0, 1.0],
                [0.1, 0.9, 1.0, 0.0],
            ],
            dtype=np.float32,
        )
        result = evaluate_distance_matrix(distances, trials, old_class_count=1)
        self.assertEqual(result["old_confusion"]["accuracy"], 1.0)
        self.assertEqual(result["novel_confusion"]["accuracy"], 1.0)
        self.assertEqual(result["old_queries_all_candidates"]["accuracy"], 0.0)
        self.assertEqual(
            result["novel_queries_all_candidates"]["accuracy"], 0.0
        )

    def test_shuffle_controls_preserve_valid_rle_and_subject_duration_blocks(self):
        state = np.zeros(12, dtype=np.float32)
        trials = [
            make_trial(
                0,
                1,
                0,
                [make_run(token, state, duration=value) for token, value in [(0, 1), (1, 2), (0, 3), (2, 4)]],
            ),
            make_trial(1, 1, 1, [make_run(0, state, duration=10)]),
            make_trial(2, 2, 0, [make_run(0, state, duration=100)]),
            make_trial(3, 2, 1, [make_run(0, state, duration=200)]),
        ]
        rng = np.random.default_rng(123)
        for _ in range(100):
            shuffled = shuffled_trajectories(trials[:1], rng, "order")[0]
            tokens = [run.token for run in shuffled.runs]
            self.assertFalse(any(a == b for a, b in zip(tokens, tokens[1:])))
            self.assertEqual(sorted(tokens), [0, 0, 1, 2])
        shuffled = shuffled_trajectories(trials, rng, "total_duration")
        before = {
            subject: sorted(
                sum(run.duration_seconds for run in trial.runs)
                for trial in trials
                if trial.subject_id == subject
            )
            for subject in [1, 2]
        }
        after = {
            subject: sorted(
                sum(run.duration_seconds for run in trial.runs)
                for trial in shuffled
                if trial.subject_id == subject
            )
            for subject in [1, 2]
        }
        self.assertEqual(before, after)


class TrajectoryAblationDataTests(unittest.TestCase):
    def test_npz_fallback_deduplicates_overlap_and_segment_rle_matches_json(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            npz_path = root / "source.npz"
            # Two six-sample physical trials represented by windows [0:4] and [2:6].
            raw_trials = []
            windows = []
            labels = []
            subjects = []
            trial_numbers = []
            trial_ids = []
            starts = []
            paths = []
            activity_names = []
            for trial_id, (subject, label) in enumerate([(1, 0), (2, 1)]):
                raw = np.vstack(
                    [np.arange(6, dtype=np.float32) + channel + 10 * trial_id for channel in range(6)]
                )
                raw_trials.append(raw)
                for start in [0, 2]:
                    windows.append(raw[:, start : start + 4])
                    labels.append(label)
                    subjects.append(subject)
                    trial_numbers.append(1)
                    trial_ids.append(trial_id)
                    starts.append(start)
                    paths.append(str(root / f"missing_{trial_id}.mat"))
                    activity_names.append(f"A{label}")
            np.savez_compressed(
                npz_path,
                windows=np.asarray(windows, dtype=np.float32),
                labels=np.asarray(labels),
                subject_ids=np.asarray(subjects),
                trial_numbers=np.asarray(trial_numbers),
                trial_global_ids=np.asarray(trial_ids),
                window_start_indices=np.asarray(starts),
                file_paths=np.asarray(paths, dtype=object),
                activity_names=np.asarray(activity_names, dtype=object),
                channel_names=np.asarray(
                    ["acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"],
                    dtype=object,
                ),
                mean=np.zeros((1, 6, 1), dtype=np.float32),
                std=np.ones((1, 6, 1), dtype=np.float32),
            )
            source = SourceSignalRepository(npz_path)
            self.assertTrue(np.allclose(source.sensor(0), raw_trials[0]))

            run_dir = root / "run"
            run_dir.mkdir()
            np.savez_compressed(
                run_dir / "segment_embeddings_and_tokens.npz",
                split_role=np.asarray([0, 0, 1, 1], dtype=np.int8),
                trial_global_ids=np.asarray([0, 0, 1, 1]),
                partition_start_samples=np.asarray([0, 3, 0, 3]),
                partition_end_samples_exclusive=np.asarray([3, 6, 3, 6]),
                primitive_tokens=np.asarray([0, 1, 1, 1]),
            )
            np.savez_compressed(
                run_dir / "primitive_codebook.npz",
                centers=np.eye(2, dtype=np.float32),
            )
            records = [
                {
                    "trial_global_id_within_npz": 1,
                    "sequence_variants": {"full": {"rle_tokens": [1]}},
                }
            ]
            with (run_dir / "trial_primitive_sequences.jsonl").open(
                "w", encoding="utf-8"
            ) as handle:
                for record in records:
                    handle.write(json.dumps(record) + "\n")
            fit, evaluation, hashes = build_trajectories(
                run_dir, source, sample_rate_hz=100.0, old_class_count=1
            )
            self.assertEqual(fit[0].tokens, (0, 1))
            self.assertEqual(evaluation[0].tokens, (1,))
            self.assertEqual(len(evaluation[0].runs), 1)
            self.assertEqual(evaluation[0].runs[0].start_sample, 0)
            self.assertEqual(evaluation[0].runs[0].end_sample_exclusive, 6)
            self.assertEqual(
                set(hashes),
                {
                    "trial_grid_hash",
                    "rle_token_hash",
                    "codebook_center_hash",
                    "segment_boundary_hash",
                },
            )
            np.savez_compressed(
                run_dir / "segment_embeddings_and_tokens.npz",
                split_role=np.asarray([0, 0, 1, 1], dtype=np.int8),
                trial_global_ids=np.asarray([0, 0, 1, 1]),
                partition_start_samples=np.asarray([0, 3, 0, 3]),
                partition_end_samples_exclusive=np.asarray([3, 6, 3, 5]),
                primitive_tokens=np.asarray([0, 1, 1, 1]),
            )
            with self.assertRaisesRegex(RuntimeError, "complete visible raw signal"):
                build_trajectories(
                    run_dir, source, sample_rate_hz=100.0, old_class_count=1
                )


def make_gate_rows(folds, seeds):
    rows = []
    for fold in folds:
        for seed in seeds:
            for position, group in enumerate(GROUP_NAMES):
                value = 0.4 + 0.05 * position
                rows.append(
                    {
                        "fold": fold,
                        "seed": seed,
                        "group": group,
                        "all_1nn_accuracy": value,
                        "all_macro_f1": value,
                        "pairwise_auc": value,
                        "zero_min_distance_query_ratio": 0.4 - 0.05 * position,
                        "mean_tied_nearest_count": 4.0 - 0.5 * position,
                        "old_query_12class_1nn_accuracy": value,
                        "novel_query_12class_1nn_accuracy_diagnostic": value,
                        "old_restricted_candidate_1nn_accuracy": value,
                        "novel_restricted_candidate_1nn_accuracy_diagnostic": value,
                        "sit_stand_12class_macro_recall": value,
                        "low_dynamic_12class_macro_recall": value,
                        "motion_sensitive_12class_macro_recall": value,
                        "sit_stand_binary_1nn_accuracy": value,
                        "sit_stand_zero_min_distance_query_ratio": 0.4
                        - 0.05 * position,
                        "order_control_accuracy_gain": 0.05 if position == 3 else np.nan,
                        "duration_alignment_control_accuracy_gain": 0.05
                        if position == 3
                        else np.nan,
                        "total_duration_control_accuracy_gain": 0.01
                        if position == 3
                        else np.nan,
                    }
                )
    return rows


class TrajectoryAblationProtocolTests(unittest.TestCase):
    @staticmethod
    def _write_fixed_protocol_fixture(root: Path) -> tuple[Path, dict, list[dict]]:
        run_dir = root / "fold_01_seed_50_k32"
        run_dir.mkdir()
        checkpoint = (
            "X/fold_01/window_pretrain/seed_50_offline/run/"
            "checkpoints/model_best.pt"
        )
        config = {
            "arguments": {
                "seed": 50,
                "primitive_num": 32,
                "primitive_segmentation": "fixed_window",
                "pca_dim": 64,
                "embedding_normalization": "l2",
                "codebook_weighting": "per_trial",
                "kmeans_n_init": 20,
                "kmeans_max_iter": 300,
                "fit_subjects": "",
                "eval_subjects": "",
                "allow_split_override": False,
                "allow_unverified_npz_normalization": False,
                "old_class_count": 6,
                "anomaly_policy": "report",
                "sample_rate_hz": 100.0,
            },
            "checkpoint": checkpoint,
            "checkpoint_metadata": {
                "uschad_cv_fold": 1,
                "uschad_train_subjects": [3, 4],
                "offline_val_subjects": [5],
                "uschad_test_subjects": [1, 2],
                "uschad_window_size": 256,
                "uschad_recompute_norm_from_train_subjects": True,
            },
            "segmentation": {"method": "fixed_window"},
            "codebook": {
                "primitive_num": 32,
                "pca_dim": 64,
                "embedding_normalization": "l2",
                "assignment_metric": "cosine",
                "weighting": "per_trial",
            },
            "data": {"sample_rate_hz": 100.0, "window_size_samples": 256},
            "npz_sha256": "canonical-test-npz",
            "normalization": {
                "mode": "fold_train_subjects_old_classes",
                "raw_source": "reconstructed_from_npz_windows_mean_std",
                "stat_window_count": 20,
            },
            "postprocessing": {
                "metric_revision": "tie_aware_1nn_and_valid_rle_shuffle_v2"
            },
        }
        split = {
            "fit_subjects": [3, 4],
            "eval_subjects": [1, 2],
            "subject_overlap": [],
            "old_class_ids_0based": list(range(6)),
            "anomaly_policy": "report",
            "checkpoint_split_override_used": False,
            "checkpoint_split_override_explicitly_allowed": False,
            "unverified_npz_normalization_explicitly_allowed": False,
        }
        (run_dir / "experiment_config.json").write_text(
            json.dumps(config), encoding="utf-8"
        )
        (run_dir / "split_audit.json").write_text(
            json.dumps(split), encoding="utf-8"
        )
        metrics = {
            "rle_sequence_full": {
                "all_classes": {
                    "observed": {"cross_subject_1nn_activity_accuracy": 0.5}
                }
            }
        }
        (run_dir / "sequence_association_metrics.json").write_text(
            json.dumps(metrics), encoding="utf-8"
        )
        record = {
            "trial_global_id_within_npz": 0,
            "subject_id": 1,
            "activity_label_0based": 0,
            "trial_number": 1,
            "window_start_indices": [0, 128],
        }
        (run_dir / "trial_primitive_sequences.jsonl").write_text(
            json.dumps(record) + "\n", encoding="utf-8"
        )
        (run_dir / "sequence_metric_revision.json").write_text(
            "{}", encoding="utf-8"
        )
        expected_runs = [
            {
                "fold": 1,
                "seed": 50,
                "config_train_subjects": [3, 4],
                "config_val_subjects": [5],
                "config_eval_subjects": [1, 2],
                "checkpoint": checkpoint,
                "source_npz_sha256": "canonical-test-npz",
            }
        ]
        return run_dir, config, expected_runs

    def test_fixed_window_data_protocol_tampering_is_rejected(self):
        mutations = {
            "fit_subject_override": ("config", "arguments", "fit_subjects", "3"),
            "split_override": ("config", "arguments", "allow_split_override", True),
            "old_class_count": ("config", "arguments", "old_class_count", 5),
            "anomaly_policy": ("split", "anomaly_policy", None, "exclude"),
            "sample_rate": ("config", "data", "sample_rate_hz", 50.0),
            "npz_sha256": ("config", None, "npz_sha256", "different"),
            "normalization": ("config", "normalization", "mode", "npz_stored"),
            "fit_subject_audit": ("split", "fit_subjects", None, [4]),
        }
        for name, (document_name, section, field, value) in mutations.items():
            with self.subTest(name=name), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                run_dir, config, expected_runs = self._write_fixed_protocol_fixture(root)
                config_path = run_dir / "experiment_config.json"
                split_path = run_dir / "split_audit.json"
                document_path = config_path if document_name == "config" else split_path
                document = json.loads(document_path.read_text(encoding="utf-8"))
                if document_name == "split":
                    document[section] = value
                elif section is None:
                    document[field] = value
                else:
                    document[section][field] = value
                document_path.write_text(json.dumps(document), encoding="utf-8")
                grid_hash = trial_grid_hash_from_jsonl(
                    run_dir / "trial_primitive_sequences.jsonl"
                )
                with self.assertRaisesRegex(RuntimeError, "identity audit failed"):
                    load_fixed_window_baselines(
                        root, {(1, 50): grid_hash}, expected_runs
                    )

    def _fixed_comparison(self):
        return {
            "available": True,
            "raw_delta_test": {"mean_delta": 0.0},
            "margin_shifted_test": {"one_sided_p_greater": 0.01},
            "passed": True,
        }

    def test_smoke_grid_never_becomes_promotion_eligible(self):
        rows = make_gate_rows([1], [50])
        gate = build_gate_report(
            rows,
            build_paired_summary(rows),
            {"passed": True},
            self._fixed_comparison(),
        )
        self.assertFalse(gate["promotion_eligible"])
        self.assertFalse(gate["final_g4"]["passed"])

    def test_noncanonical_parameter_audit_disables_full_grid_promotion(self):
        rows = make_gate_rows(range(1, 8), [0, 5, 50, 500])
        gate = build_gate_report(
            rows,
            build_paired_summary(rows),
            {"passed": False, "reason": "state_weight differs"},
            self._fixed_comparison(),
        )
        self.assertFalse(gate["promotion_eligible"])
        for key in [
            "g2_dynamic_soft",
            "g3_state_residual",
            "g4_duration_context",
            "final_g4",
        ]:
            self.assertFalse(gate[key]["passed"])

    def test_fixed_window_trial_hash_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run_dir = root / "fold_01_seed_50_k32"
            run_dir.mkdir()
            config = {
                "arguments": {
                    "seed": 50,
                    "primitive_num": 32,
                    "primitive_segmentation": "fixed_window",
                    "pca_dim": 64,
                    "embedding_normalization": "l2",
                    "codebook_weighting": "per_trial",
                    "kmeans_n_init": 20,
                    "kmeans_max_iter": 300,
                    "fit_subjects": "",
                    "eval_subjects": "",
                    "allow_split_override": False,
                    "allow_unverified_npz_normalization": False,
                    "old_class_count": 6,
                    "anomaly_policy": "report",
                    "sample_rate_hz": 100.0,
                },
                "checkpoint": "X/fold_01/window_pretrain/seed_50_offline/run/checkpoints/model_best.pt",
                "checkpoint_metadata": {
                    "uschad_cv_fold": 1,
                    "uschad_train_subjects": [3, 4],
                    "offline_val_subjects": [5],
                    "uschad_test_subjects": [1, 2],
                    "uschad_window_size": 256,
                    "uschad_recompute_norm_from_train_subjects": True,
                },
                "segmentation": {"method": "fixed_window"},
                "codebook": {
                    "primitive_num": 32,
                    "pca_dim": 64,
                    "embedding_normalization": "l2",
                    "assignment_metric": "cosine",
                    "weighting": "per_trial",
                },
                "data": {
                    "sample_rate_hz": 100.0,
                    "window_size_samples": 256,
                },
                "npz_sha256": "canonical-test-npz",
                "normalization": {
                    "mode": "fold_train_subjects_old_classes",
                    "raw_source": "reconstructed_from_npz_windows_mean_std",
                    "stat_window_count": 20,
                },
                "postprocessing": {
                    "metric_revision": "tie_aware_1nn_and_valid_rle_shuffle_v2"
                },
            }
            (run_dir / "experiment_config.json").write_text(
                json.dumps(config), encoding="utf-8"
            )
            metrics = {
                "rle_sequence_full": {
                    "all_classes": {
                        "observed": {"cross_subject_1nn_activity_accuracy": 0.5}
                    }
                }
            }
            (run_dir / "sequence_association_metrics.json").write_text(
                json.dumps(metrics), encoding="utf-8"
            )
            record = {
                "trial_global_id_within_npz": 0,
                "subject_id": 1,
                "activity_label_0based": 0,
                "trial_number": 1,
                "window_start_indices": [0, 128],
            }
            (run_dir / "trial_primitive_sequences.jsonl").write_text(
                json.dumps(record) + "\n", encoding="utf-8"
            )
            (run_dir / "sequence_metric_revision.json").write_text(
                "{}", encoding="utf-8"
            )
            (run_dir / "split_audit.json").write_text(
                json.dumps(
                    {
                        "fit_subjects": [3, 4],
                        "eval_subjects": [1, 2],
                        "subject_overlap": [],
                        "old_class_ids_0based": list(range(6)),
                        "anomaly_policy": "report",
                        "checkpoint_split_override_used": False,
                        "checkpoint_split_override_explicitly_allowed": False,
                        "unverified_npz_normalization_explicitly_allowed": False,
                    }
                ),
                encoding="utf-8",
            )
            expected_runs = [
                {
                    "fold": 1,
                    "seed": 50,
                    "config_train_subjects": [3, 4],
                    "config_val_subjects": [5],
                    "config_eval_subjects": [1, 2],
                    "checkpoint": config["checkpoint"],
                    "source_npz_sha256": "canonical-test-npz",
                }
            ]
            with self.assertRaisesRegex(RuntimeError, "trial-grid hash mismatch"):
                load_fixed_window_baselines(
                    root, {(1, 50): "definitely-wrong"}, expected_runs
                )

            actual_hash = trial_grid_hash_from_jsonl(
                run_dir / "trial_primitive_sequences.jsonl"
            )
            config["arguments"]["fit_subjects"] = "99"
            config["arguments"]["allow_split_override"] = True
            config["arguments"]["old_class_count"] = 5
            config["arguments"]["anomaly_policy"] = "exclude"
            config["arguments"]["sample_rate_hz"] = 50.0
            config["npz_sha256"] = "different-npz"
            (run_dir / "experiment_config.json").write_text(
                json.dumps(config), encoding="utf-8"
            )
            with self.assertRaisesRegex(RuntimeError, "identity audit failed"):
                load_fixed_window_baselines(
                    root, {(1, 50): actual_hash}, expected_runs
                )


if __name__ == "__main__":
    unittest.main()
