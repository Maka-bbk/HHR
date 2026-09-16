from __future__ import annotations

import hashlib
import json
import subprocess
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

import experiments.motion_primitive.run_peak_valley_hierarchy_cv as cv_module
import experiments.motion_primitive.run_peak_valley_hierarchy as runner_module
from experiments.motion_primitive.core import (
    assign_to_codebook,
    inverse_trial_frequency_weights,
)
from experiments.motion_primitive.peak_valley_hierarchy import (
    ChildPrimitiveBatch,
    CodebookConfig,
    fit_codebook,
)
from experiments.motion_primitive.run_experiment import prepare_primitive_features
from experiments.motion_primitive.run_peak_valley_hierarchy_cv import (
    METRICS,
    _aggregate_rows,
    _exact_sign_flip_pvalue,
    _grid_protocol_settings,
    _holm_adjusted_pvalues,
    _member_source_fingerprints,
    _metric_rows,
    _minimum_two_sided_exact_sign_flip_pvalue,
    _paired_hscore_rows,
    _protocol_settings,
    _required_generated_files,
    _runner_cli_wiring_sha256,
    _validate_completed_run,
    _validate_profile_semantics,
    _validate_metric_grid,
)


def _protocol_namespace() -> Namespace:
    return Namespace(
        old_class_count=6,
        primitive_num=32,
        pca_dim=64,
        child_feature_source="content_embedding",
        trial_cluster_count=10,
        window_size_samples=256,
        window_stride_samples=128,
        batch_size=512,
        device="cuda",
        sample_rate_hz=100.0,
        smooth_seconds=0.15,
        prominence_mad=1.5,
        extrema_distance_seconds=0.2,
        vote_tolerance_seconds=0.1,
        minimum_axes=2,
        minimum_segment_seconds=0.25,
        feature_confirm_quantile=0.5,
        feature_context_seconds=0.5,
        shape_points=64,
        parent_min_occurrences=10,
        parent_min_trials=6,
        parent_min_subjects=2,
        parent_min_npmi=0.0,
        parent_min_mdl_gain=0.0,
        parent_min_loso_stability=0.8,
        matched_random_controls=True,
        bootstrap_replicates=10000,
        analysis_seed=20260906,
    )


def _metrics(value: float) -> dict[str, float]:
    return {metric: value for metric in METRICS}


def _cross_subject_diagnostics(
    *,
    margin: float = 0.2,
    probability: float = 0.65,
    one_nn_accuracy: float = 0.6,
    subject_nmi: float = 0.1,
) -> dict:
    return {
        "cross_subject_distance_effect": {
            "available": True,
            "reason": None,
            "mean_margin_different_minus_same": margin,
            "probability_same_distance_is_smaller": probability,
            "rank_separation_effect": 2.0 * probability - 1.0,
        },
        "tie_aware_cross_subject_1nn": {
            "available": True,
            "reason": None,
            "accuracy": one_nn_accuracy,
        },
        "cluster_subject_nmi": {
            "available": True,
            "reason": None,
            "normalized_mutual_information": subject_nmi,
        },
        "post_truth_join_diagnostic_only": True,
        "used_to_fit_or_modify_predictions": False,
    }


def _single_result() -> dict:
    arm_results = {}
    for arm_id, base_arm, is_control, control_type in (
        ("E0", "E0", False, None),
        ("E2", "E2", False, None),
        ("E4", "E4", False, None),
        ("C1", "E2", True, "matched_random_boundaries"),
        ("C2", "E4", True, "matched_random_parents"),
    ):
        arm_results[arm_id] = {
            "base_arm": base_arm,
            "is_control": is_control,
            "control_type": control_type,
            "segmentation": {"fit_segment_count": 12, "eval_segment_count": 8},
            "parent_catalog": {"parent_count": 2 if arm_id in {"E4", "C2"} else 0},
            "readout_variants": {
                variant: {
                    "metrics": _metrics(0.6 if variant == "state" else 0.5),
                    "cross_subject_diagnostics": _cross_subject_diagnostics(),
                    "readout": {"test_trial_count": 10},
                }
                for variant in ("state", "no_state")
            },
        }
    return {
        "request_identity": {"profile": "A2", "fold": 1, "seed": 0},
        "arm_results": arm_results,
    }


def _materialize_fake_completed_run(directory: Path, identity: dict) -> dict:
    directory.mkdir(parents=True, exist_ok=False)
    manifest = []
    for name in sorted(_required_generated_files(identity)):
        payload = f"artifact:{name}".encode("utf-8")
        (directory / name).write_bytes(payload)
        manifest.append(
            {
                "relative_path": name,
                "size_bytes": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
        )
    result = {"request_identity": identity, "generated_files": manifest}
    (directory / "experiment_result.json").write_text(
        json.dumps(result), encoding="utf-8"
    )
    return result


class PeakValleyHierarchyCvTests(unittest.TestCase):
    def test_single_runner_rejects_noncanonical_pca_dimension(self) -> None:
        args = runner_module.build_parser().parse_args(
            [
                "--checkpoint",
                "missing-checkpoint.pt",
                "--npz-path",
                "missing-windows.npz",
                "--output-dir",
                "unused-output",
                "--expected-profile",
                "A2",
                "--fold",
                "1",
                "--seed",
                "0",
                "--pca-dim",
                "32",
            ]
        )
        with self.assertRaisesRegex(ValueError, "PCA64"):
            runner_module.run(args)

    def test_registered_w128_s64_grid_is_explicit_and_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "windows.npz"
            np.savez_compressed(
                path,
                windows=np.zeros((4, 6, 128), dtype=np.float32),
                labels=np.asarray([0, 0, 1, 1], dtype=np.int64),
                subject_ids=np.asarray([1, 1, 2, 2], dtype=np.int64),
                trial_global_ids=np.asarray([10, 10, 20, 20], dtype=np.int64),
                window_indices=np.asarray([0, 1, 0, 1], dtype=np.int64),
                window_start_indices=np.asarray([0, 64, 0, 64], dtype=np.int64),
                mean=np.zeros((1, 6, 1), dtype=np.float32),
                std=np.ones((1, 6, 1), dtype=np.float32),
            )

            grid = runner_module._load_numeric_grid(
                path,
                expected_window_size_samples=128,
                expected_stride_samples=64,
            )
            self.assertEqual(grid.window_size_samples, 128)
            self.assertEqual(grid.stride_samples, 64)
            np.testing.assert_array_equal(
                runner_module._trial_rows(grid, 10), np.asarray([0, 1])
            )

            with self.assertRaisesRegex(RuntimeError, r"\[N,6,256\]"):
                runner_module._load_numeric_grid(path)
            with self.assertRaisesRegex(ValueError, "registered window grid"):
                runner_module._load_numeric_grid(
                    path,
                    expected_window_size_samples=128,
                    expected_stride_samples=128,
                )

    def test_cv_wires_explicit_w128_s64_identity_to_member(self) -> None:
        args = cv_module.build_parser().parse_args(
            [
                "--cv-root",
                "cv",
                "--npz-path",
                "windows.npz",
                "--encoder-root",
                "encoders",
                "--output-root",
                "output",
                "--window-size-samples",
                "128",
                "--window-stride-samples",
                "64",
            ]
        )
        command = cv_module._runner_command(
            args,
            Path("encoder.pt"),
            Path("member-output"),
            "A2",
            1,
            0,
            ("E0",),
        )
        size_index = command.index("--window-size-samples")
        stride_index = command.index("--window-stride-samples")
        self.assertEqual(command[size_index + 1], "128")
        self.assertEqual(command[stride_index + 1], "64")
        protocol = cv_module._protocol_settings(args)
        self.assertEqual(protocol["window_size_samples"], 128)
        self.assertEqual(protocol["window_stride_samples"], 64)

    def test_e0_matches_historical_pca_and_cosine_assignment(self) -> None:
        rng = np.random.default_rng(20260906)
        trial_counts = (28, 31, 37)
        fit_raw = rng.normal(size=(sum(trial_counts), 72)).astype(np.float32)
        # Make the pre-PCA normalization observable rather than an identity.
        fit_raw *= rng.lognormal(mean=0.0, sigma=1.0, size=(len(fit_raw), 1))
        eval_raw = rng.normal(size=(19, 72)).astype(np.float32)
        eval_raw *= rng.lognormal(mean=0.0, sigma=1.0, size=(len(eval_raw), 1))
        trial_ids = np.repeat(np.arange(len(trial_counts)), trial_counts)
        weights = inverse_trial_frequency_weights(trial_ids)
        historical_fit, historical_eval, _ = prepare_primitive_features(
            fit_raw,
            eval_raw,
            weights,
            pca_dim=64,
            normalization="l2",
        )

        split = np.cumsum((0, *trial_counts))
        batches = []
        for trial_id, count in enumerate(trial_counts):
            starts = np.arange(count, dtype=np.int64) * 128
            batches.append(
                ChildPrimitiveBatch(
                    trial_id=trial_id,
                    subject_id=trial_id,
                    segment_indices=np.arange(count, dtype=np.int64),
                    start_samples=starts,
                    end_samples_exclusive=starts + 256,
                    shape_features=fit_raw[
                        split[trial_id] : split[trial_id + 1]
                    ],
                    statistic_features=np.zeros((count, 1), dtype=np.float64),
                    statistic_names=("unused",),
                    segmentation_state_sha256="a" * 64,
                ).validate()
            )
        state = runner_module._fit_e0_historical_codebook(
            batches,
            primitive_num=32,
            pca_dim=64,
            seed=7,
        )
        _, _, new_fit = runner_module._e0_codebook_assign(
            fit_raw, state
        )
        _, _, new_eval = runner_module._e0_codebook_assign(eval_raw, state)
        np.testing.assert_allclose(new_fit, historical_fit, rtol=0.0, atol=0.0)
        np.testing.assert_allclose(new_eval, historical_eval, rtol=0.0, atol=0.0)

        expected_tokens, expected_distances, _ = assign_to_codebook(
            new_eval.astype(np.float32),
            state.cluster_centers.astype(np.float32),
            "cosine",
        )
        tokens, distances, embedded = runner_module._e0_codebook_assign(
            eval_raw, state
        )
        np.testing.assert_array_equal(tokens, expected_tokens)
        np.testing.assert_allclose(distances, expected_distances, rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(embedded, new_eval, rtol=0.0, atol=0.0)

    def test_e0_state_uses_encoder_windows_but_duration_uses_ownership_cells(self) -> None:
        signal = np.repeat(np.arange(384, dtype=np.float64)[None, :], 6, axis=0)
        grid = runner_module.WindowGrid(
            windows=np.zeros((2, 6, 256), dtype=np.float32),
            labels=np.full(2, -1, dtype=np.int64),
            subject_ids=np.ones(2, dtype=np.int64),
            trial_ids=np.full(2, 7, dtype=np.int64),
            window_indices=np.arange(2, dtype=np.int64),
            starts=np.asarray([0, 128], dtype=np.int64),
            stored_mean=np.zeros((1, 6, 1), dtype=np.float32),
            stored_std=np.ones((1, 6, 1), dtype=np.float32),
        )
        source = SimpleNamespace(sensor=lambda trial_id: signal)
        codebook = SimpleNamespace(
            config=SimpleNamespace(primitive_num=2),
            pca_components=None,
            cluster_centers=np.eye(2, dtype=np.float32),
        )
        trials = runner_module._primitive_trials_e0(
            [7],
            {7: 1},
            grid,
            {7: np.eye(2, dtype=np.float32)},
            codebook,
            source,
            100.0,
        )
        trial = trials[0]
        np.testing.assert_array_equal(trial.starts, [0, 192])
        np.testing.assert_array_equal(trial.ends, [192, 384])
        # Duration remains the 192-sample ownership quantity, while physical
        # signal summaries below use each token's full 256-sample support.
        np.testing.assert_allclose(trial.child_statistics[:, 0], [1.92, 1.92])
        np.testing.assert_allclose(trial.child_statistics[:, 1], [127.5, 255.5])

    def test_protocol_pins_legacy_e0_and_content_features(self) -> None:
        protocol = _protocol_settings(_protocol_namespace())
        self.assertEqual(protocol["e0_segmentation"], "legacy_npz_windows")
        self.assertEqual(protocol["child_feature_source"], "content_embedding")
        self.assertNotIn("fixed_segment_seconds", protocol)
        self.assertEqual(protocol["readout_variants"], ["state", "no_state"])
        self.assertEqual(protocol["window_size_samples"], 256)
        self.assertEqual(protocol["window_stride_samples"], 128)
        self.assertEqual(protocol["batch_size"], 512)
        self.assertEqual(protocol["device"], "cuda")
        label_boundary = protocol["label_access_boundary"]
        self.assertEqual(
            label_boundary["benchmark_split_builder"],
            "label_aware_before_raw_prediction_freeze",
        )
        self.assertEqual(
            label_boundary["predictor_facing_builder_output"],
            "trial_ids_and_subject_ids_only",
        )
        self.assertIs(
            label_boundary["predictor_receives_held_out_activity_labels"], False
        )
        self.assertEqual(
            label_boundary["scoring_truth_join"],
            "after_raw_prediction_artifact_is_written_and_hashed",
        )
        grid = _grid_protocol_settings(_protocol_namespace())
        self.assertEqual(grid["aggregate"]["bootstrap_replicates"], 10000)
        self.assertEqual(grid["aggregate"]["analysis_seed"], 20260906)
        inference = grid["aggregate"]
        self.assertEqual(
            inference["prespecified_followup_primary_hscore_comparison"]["name"],
            "A2_E2_state_minus_A0_E0_state",
        )
        self.assertEqual(inference["secondary_holm_family"]["planned_family_size"], 8)
        secondary_names = inference["secondary_holm_family"]["comparison_names"]
        self.assertIn("A2_minus_A3_E2", secondary_names)
        self.assertNotIn("A3_minus_A2_E2", secondary_names)
        exploratory_simple = inference["exploratory_unadjusted_simple_comparisons"]
        self.assertEqual(
            [item["name"] for item in exploratory_simple],
            ["A2_minus_A3_E0_fixed_window"],
        )
        interaction_names = [
            item["name"] for item in inference["exploratory_factorial_interactions"]
        ]
        self.assertIn(
            "A2_minus_A3_by_E2_state_difference_in_differences",
            interaction_names,
        )
        self.assertAlmostEqual(
            inference["two_sided_exact_sign_flip_minimum_p_at_seven_folds"],
            0.015625,
        )
        self.assertAlmostEqual(
            inference["holm_eight_minimum_attainable_adjusted_p_at_seven_folds"],
            0.125,
        )
        self.assertIn("descriptive_only", inference["bootstrap_confidence_intervals"])
        diagnostic_protocol = inference["cross_subject_trajectory_diagnostics"]
        self.assertEqual(
            diagnostic_protocol["aggregation_order"],
            "mean_seeds_within_held_out_subject_fold_then_summarize_folds",
        )
        self.assertIs(
            diagnostic_protocol["independent_sample_size_is_not_fold_times_seed"],
            True,
        )
        self.assertIs(
            diagnostic_protocol["not_part_of_primary_classification_test"], True
        )
        self.assertEqual(
            diagnostic_protocol["favourable_directions"]["cluster_subject_nmi"],
            "lower_means_less_subject_identity_leakage",
        )
        staging = grid["staging_publication"]
        self.assertEqual(
            staging["publication"], "same_filesystem_atomic_directory_rename"
        )
        self.assertEqual(
            staging["stale_staging_policy"],
            "preserve_and_ignore_on_retry_never_auto_delete",
        )
        self.assertEqual(len(_runner_cli_wiring_sha256()), 64)

    def test_member_identity_excludes_cv_wrapper(self) -> None:
        observed = _member_source_fingerprints(
            {
                "runner": "r",
                "algorithm": "a",
                "core": "c",
                "motion_encoder": "m",
                "cv_wrapper": "cv",
            }
        )
        self.assertEqual(
            observed,
            {"runner": "r", "algorithm": "a", "core": "c", "motion_encoder": "m"},
        )

    def test_metric_rows_require_both_readout_variants_and_controls(self) -> None:
        rows = _metric_rows(
            _single_result(),
            requested_arms=("E0", "E2", "E4"),
            matched_random_controls=True,
        )
        self.assertEqual(len(rows), 10)
        self.assertEqual(
            {row["arm_id"] for row in rows}, {"E0", "E2", "E4", "C1", "C2"}
        )
        self.assertEqual(
            {row["readout_variant"] for row in rows}, {"state", "no_state"}
        )

        malformed = _single_result()
        del malformed["arm_results"]["E2"]["readout_variants"]["no_state"]
        with self.assertRaisesRegex(RuntimeError, "readout variants differ"):
            _metric_rows(
                malformed,
                requested_arms=("E0", "E2", "E4"),
                matched_random_controls=True,
            )

    def test_metric_grid_detects_a_missing_seed_member(self) -> None:
        rows = _metric_rows(
            _single_result(),
            requested_arms=("E0", "E2", "E4"),
            matched_random_controls=True,
        )
        _validate_metric_grid(
            rows,
            profiles=("A2",),
            folds=(1,),
            seeds=(0,),
            requested_arms=("E0", "E2", "E4"),
            matched_random_controls=True,
        )
        with self.assertRaisesRegex(RuntimeError, "Metric grid"):
            _validate_metric_grid(
                rows[:-1],
                profiles=("A2",),
                folds=(1,),
                seeds=(0,),
                requested_arms=("E0", "E2", "E4"),
                matched_random_controls=True,
            )

    def test_cross_subject_diagnostic_rows_are_strict_and_complete(self) -> None:
        rows = cv_module._cross_subject_diagnostic_rows(
            _single_result(),
            requested_arms=("E0", "E2", "E4"),
            matched_random_controls=True,
        )
        self.assertEqual(len(rows), 10)
        self.assertEqual(
            {(row["arm_id"], row["readout_variant"]) for row in rows},
            {
                (arm, variant)
                for arm in ("E0", "E2", "E4", "C1", "C2")
                for variant in ("state", "no_state")
            },
        )
        self.assertTrue(
            all(row["post_truth_join_diagnostic_only"] is True for row in rows)
        )
        self.assertTrue(
            all(row["used_to_fit_or_modify_predictions"] is False for row in rows)
        )
        self.assertTrue(
            all(
                abs(
                    row["rank_separation_effect"]
                    - (2.0 * row["probability_same_distance_is_smaller"] - 1.0)
                )
                <= 1e-12
                for row in rows
            )
        )
        cv_module._validate_cross_subject_diagnostic_grid(
            rows,
            profiles=("A2",),
            folds=(1,),
            seeds=(0,),
            requested_arms=("E0", "E2", "E4"),
            matched_random_controls=True,
        )
        with self.assertRaisesRegex(RuntimeError, "diagnostic grid"):
            cv_module._validate_cross_subject_diagnostic_grid(
                rows[:-1],
                profiles=("A2",),
                folds=(1,),
                seeds=(0,),
                requested_arms=("E0", "E2", "E4"),
                matched_random_controls=True,
            )
        with self.assertRaisesRegex(RuntimeError, "duplicated"):
            cv_module._validate_cross_subject_diagnostic_grid(
                [*rows, dict(rows[0])],
                profiles=("A2",),
                folds=(1,),
                seeds=(0,),
                requested_arms=("E0", "E2", "E4"),
                matched_random_controls=True,
            )

    def test_cross_subject_diagnostic_missing_or_unavailable_fails_closed(self) -> None:
        missing = _single_result()
        del missing["arm_results"]["E0"]["readout_variants"]["state"][
            "cross_subject_diagnostics"
        ]
        with self.assertRaisesRegex(RuntimeError, "lacks cross_subject_diagnostics"):
            cv_module._cross_subject_diagnostic_rows(
                missing,
                requested_arms=("E0", "E2", "E4"),
                matched_random_controls=True,
            )

        missing_metric = _single_result()
        del missing_metric["arm_results"]["E0"]["readout_variants"]["state"][
            "cross_subject_diagnostics"
        ]["cross_subject_distance_effect"]["mean_margin_different_minus_same"]
        with self.assertRaisesRegex(
            RuntimeError, "lack mean_margin_different_minus_same"
        ):
            cv_module._cross_subject_diagnostic_rows(
                missing_metric,
                requested_arms=("E0", "E2", "E4"),
                matched_random_controls=True,
            )

        for block_name in (
            "cross_subject_distance_effect",
            "tie_aware_cross_subject_1nn",
            "cluster_subject_nmi",
        ):
            with self.subTest(block_name=block_name):
                unavailable = _single_result()
                block = unavailable["arm_results"]["E0"]["readout_variants"][
                    "state"
                ]["cross_subject_diagnostics"][block_name]
                block["available"] = False
                block["reason"] = "synthetic_unavailable"
                with self.assertRaisesRegex(RuntimeError, "is unavailable"):
                    cv_module._cross_subject_diagnostic_rows(
                        unavailable,
                        requested_arms=("E0", "E2", "E4"),
                        matched_random_controls=True,
                    )

        inconsistent = _single_result()
        inconsistent["arm_results"]["E0"]["readout_variants"]["state"][
            "cross_subject_diagnostics"
        ]["cross_subject_distance_effect"]["rank_separation_effect"] = 0.9
        with self.assertRaisesRegex(RuntimeError, "inconsistent probability/rank"):
            cv_module._cross_subject_diagnostic_rows(
                inconsistent,
                requested_arms=("E0", "E2", "E4"),
                matched_random_controls=True,
            )

    def test_cross_subject_diagnostics_average_seeds_before_seven_folds(self) -> None:
        rows = []
        for fold in range(1, 8):
            for seed in (0, 5):
                result = _single_result()
                result["request_identity"].update(
                    {"profile": "A2", "fold": fold, "seed": seed}
                )
                probability = 0.4 + 0.05 * fold + (0.02 if seed == 5 else 0.0)
                for arm_result in result["arm_results"].values():
                    for variant_result in arm_result["readout_variants"].values():
                        variant_result["cross_subject_diagnostics"] = (
                            _cross_subject_diagnostics(
                                margin=probability - 0.5,
                                probability=probability,
                                one_nn_accuracy=probability,
                                subject_nmi=1.0 - probability,
                            )
                        )
                rows.extend(
                    cv_module._cross_subject_diagnostic_rows(
                        result,
                        requested_arms=("E0", "E2", "E4"),
                        matched_random_controls=True,
                    )
                )
        cv_module._validate_cross_subject_diagnostic_grid(
            rows,
            profiles=("A2",),
            folds=range(1, 8),
            seeds=(0, 5),
            requested_arms=("E0", "E2", "E4"),
            matched_random_controls=True,
        )
        fold_rows, aggregate = cv_module._aggregate_cross_subject_diagnostic_rows(
            rows, replicates=50, seed=88
        )
        self.assertEqual(len(rows), 140)
        self.assertEqual(len(fold_rows), 70)
        self.assertEqual(len(aggregate), 50)
        first_fold = next(
            row
            for row in fold_rows
            if row["arm_id"] == "E0"
            and row["readout_variant"] == "state"
            and row["fold"] == 1
        )
        self.assertEqual(first_fold["seeds"], [0, 5])
        self.assertAlmostEqual(
            first_fold["probability_same_distance_is_smaller"], 0.46
        )
        across_folds = next(
            row
            for row in aggregate
            if row["arm_id"] == "E0"
            and row["readout_variant"] == "state"
            and row["metric"] == "probability_same_distance_is_smaller"
        )
        self.assertEqual(across_folds["fold_count"], 7)
        self.assertEqual(across_folds["seed_count_per_fold"], [2])
        self.assertAlmostEqual(across_folds["fold_mean"], 0.61)
        self.assertEqual(
            across_folds["inference_unit"],
            "held_out_subject_fold_not_fold_times_seed_run",
        )
        self.assertEqual(
            across_folds["bootstrap_ci_interpretation"],
            "descriptive_only_not_confirmatory",
        )

    def test_prespecified_followup_primary_and_exploratory_contrast_formulas(self) -> None:
        endpoint_values = {
            ("A2", "E2", "state"): 0.80,
            ("A2", "E0", "state"): 0.60,
            ("A3", "E2", "state"): 0.70,
            ("A3", "E0", "state"): 0.65,
            ("A0", "E2", "state"): 0.55,
            ("A0", "E0", "state"): 0.50,
        }
        fold_rows = [
            {
                "profile": profile,
                "arm_id": arm,
                "readout_variant": variant,
                "fold": fold,
                "h_score": value,
            }
            for fold in range(1, 8)
            for (profile, arm, variant), value in endpoint_values.items()
        ]
        paired, _ = _paired_hscore_rows(
            fold_rows,
            expected_folds=range(1, 8),
            bootstrap_replicates=20,
            seed=101,
        )
        primary = next(
            row
            for row in paired
            if row["comparison"] == "A2_E2_state_minus_A0_E0_state"
        )
        interaction = next(
            row
            for row in paired
            if row["comparison"] == "A2_by_E2_state_difference_in_differences"
        )
        infonce_e2 = next(
            row for row in paired if row["comparison"] == "A2_minus_A3_E2"
        )
        fixed_window = next(
            row
            for row in paired
            if row["comparison"] == "A2_minus_A3_E0_fixed_window"
        )
        infonce_interaction = next(
            row
            for row in paired
            if row["comparison"]
            == "A2_minus_A3_by_E2_state_difference_in_differences"
        )
        self.assertEqual(primary["analysis_role"], "prespecified_followup_primary")
        self.assertEqual(primary["comparison_kind"], "simple_difference")
        self.assertAlmostEqual(primary["mean_paired_delta"], 0.30)
        self.assertEqual(
            [item["coefficient"] for item in primary["contrast_terms"]], [1.0, -1.0]
        )
        self.assertAlmostEqual(primary["exact_sign_flip_p_two_sided"], 0.015625)
        self.assertIs(primary["unadjusted_reject_at_alpha_0_05"], True)

        self.assertEqual(
            interaction["analysis_role"], "exploratory_factorial_interaction"
        )
        self.assertEqual(interaction["comparison_kind"], "difference_in_differences")
        self.assertEqual(
            interaction["contrast_formula"],
            "(A2/E2-A2/E0)-(A0/E2-A0/E0)",
        )
        self.assertEqual(
            [item["coefficient"] for item in interaction["contrast_terms"]],
            [1.0, -1.0, -1.0, 1.0],
        )
        self.assertAlmostEqual(interaction["mean_paired_delta"], 0.15)
        self.assertEqual(
            interaction["bootstrap_ci_interpretation"],
            "descriptive_only_not_confirmatory",
        )
        self.assertIsNone(interaction["holm_adjusted_p_two_sided"])

        self.assertEqual(infonce_e2["analysis_role"], "exploratory_secondary")
        self.assertAlmostEqual(infonce_e2["mean_paired_delta"], 0.10)
        self.assertEqual(
            [item["coefficient"] for item in infonce_e2["contrast_terms"]],
            [1.0, -1.0],
        )
        self.assertEqual(
            (infonce_e2["lhs_profile"], infonce_e2["rhs_profile"]),
            ("A2", "A3"),
        )

        self.assertEqual(
            fixed_window["analysis_role"], "exploratory_fixed_window_replication"
        )
        self.assertAlmostEqual(fixed_window["mean_paired_delta"], -0.05)
        self.assertIsNone(fixed_window["multiplicity_family"])
        self.assertIsNone(fixed_window["holm_adjusted_p_two_sided"])
        self.assertEqual(
            fixed_window["holm_family_status"],
            "not_applicable_unadjusted_exploratory",
        )

        self.assertEqual(
            infonce_interaction["analysis_role"],
            "exploratory_factorial_interaction",
        )
        self.assertEqual(
            infonce_interaction["contrast_formula"],
            "(A2/E2-A3/E2)-(A2/E0-A3/E0)",
        )
        self.assertEqual(
            [item["coefficient"] for item in infonce_interaction["contrast_terms"]],
            [1.0, -1.0, -1.0, 1.0],
        )
        self.assertAlmostEqual(infonce_interaction["mean_paired_delta"], 0.15)
        self.assertIsNone(infonce_interaction["holm_adjusted_p_two_sided"])

    def test_seed_average_precedes_fold_level_sign_flip(self) -> None:
        rows = []
        for profile in ("A0", "A2", "A3"):
            for fold in range(1, 8):
                for seed in (0, 5):
                    for arm in ("E0", "E2", "E4", "C1", "C2"):
                        for variant in ("state", "no_state"):
                            value = 0.4
                            if profile == "A2" and arm == "E2" and variant == "state":
                                value = 0.6 + 0.01 * (seed == 5)
                            rows.append(
                                {
                                    "profile": profile,
                                    "fold": fold,
                                    "seed": seed,
                                    "arm_id": arm,
                                    "readout_variant": variant,
                                    "is_primary_readout": variant == "state",
                                    "base_arm": {"C1": "E2", "C2": "E4"}.get(
                                        arm, arm
                                    ),
                                    "is_control": arm in {"C1", "C2"},
                                    "control_type": {
                                        "C1": "matched_random_boundaries",
                                        "C2": "matched_random_parents",
                                    }.get(arm),
                                    **_metrics(value),
                                }
                            )
        fold_rows, aggregate = _aggregate_rows(rows, replicates=20, seed=9)
        self.assertTrue(
            all(
                row["bootstrap_ci_interpretation"]
                == "descriptive_only_not_confirmatory"
                for row in aggregate
            )
        )
        paired, _ = _paired_hscore_rows(
            fold_rows,
            expected_folds=range(1, 8),
            bootstrap_replicates=20,
            seed=10,
        )
        comparison = next(
            row for row in paired if row["comparison"] == "A2_E2_minus_E0"
        )
        self.assertEqual(comparison["fold_count"], 7)
        self.assertAlmostEqual(comparison["mean_paired_delta"], 0.205)
        self.assertAlmostEqual(comparison["exact_sign_flip_p_two_sided"], 2 / 128)
        primary = next(
            row
            for row in paired
            if row["comparison"] == "A2_E2_state_minus_A0_E0_state"
        )
        interaction = next(
            row
            for row in paired
            if row["comparison"] == "A2_by_E2_state_difference_in_differences"
        )
        infonce_e2 = next(
            row for row in paired if row["comparison"] == "A2_minus_A3_E2"
        )
        fixed_window = next(
            row
            for row in paired
            if row["comparison"] == "A2_minus_A3_E0_fixed_window"
        )
        infonce_interaction = next(
            row
            for row in paired
            if row["comparison"]
            == "A2_minus_A3_by_E2_state_difference_in_differences"
        )
        self.assertAlmostEqual(primary["mean_paired_delta"], 0.205)
        self.assertAlmostEqual(interaction["mean_paired_delta"], 0.205)
        self.assertAlmostEqual(infonce_e2["mean_paired_delta"], 0.205)
        self.assertAlmostEqual(fixed_window["mean_paired_delta"], 0.0)
        self.assertAlmostEqual(infonce_interaction["mean_paired_delta"], 0.205)
        secondary = [
            row for row in paired if row["analysis_role"] == "exploratory_secondary"
        ]
        self.assertEqual(len(secondary), 8)
        self.assertTrue(
            all(row["holm_family_status"] == "complete_eight_item_family" for row in secondary)
        )
        self.assertAlmostEqual(
            min(float(row["holm_adjusted_p_two_sided"]) for row in secondary),
            0.125,
        )
        self.assertTrue(
            all(row["holm_reject_at_alpha_0_05"] is False for row in secondary)
        )

    def test_exact_sign_flip_known_seven_fold_case(self) -> None:
        self.assertAlmostEqual(_exact_sign_flip_pvalue([1.0] * 7), 2 / 128)
        self.assertAlmostEqual(
            _minimum_two_sided_exact_sign_flip_pvalue(7), 0.015625
        )
        self.assertEqual(_holm_adjusted_pvalues([0.015625] * 8), [0.125] * 8)

    def test_completed_run_manifest_is_content_addressed_and_arm_complete(self) -> None:
        identity = {
            "arms": ["E0"],
            "protocol": {"matched_random_controls": False},
        }
        required = _required_generated_files(identity)
        self.assertIn("raw_cluster_predictions.npz", required)
        self.assertIn("e0_legacy_window_codebook_state.json", required)
        e4_control_required = _required_generated_files(
            {
                "arms": ["E4"],
                "protocol": {"matched_random_controls": True},
            }
        )
        self.assertIn("e2_shared_child_codebook_state.json", e4_control_required)
        self.assertIn("e4_strict_parent_catalog_state.json", e4_control_required)
        self.assertIn(
            "c2_matched_negative_parent_catalog_state.json", e4_control_required
        )
        self.assertNotIn("c1_matched_random_codebook_state.json", e4_control_required)

        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory) / "run"
            run_dir.mkdir()
            manifest = []
            payloads: dict[str, bytes] = {}
            for name in sorted(required):
                payload = f"artifact:{name}".encode("utf-8")
                payloads[name] = payload
                (run_dir / name).write_bytes(payload)
                manifest.append(
                    {
                        "relative_path": name,
                        "size_bytes": len(payload),
                        "sha256": hashlib.sha256(payload).hexdigest(),
                    }
                )

            def write_result(records: list[dict]) -> None:
                (run_dir / "experiment_result.json").write_text(
                    json.dumps(
                        {"request_identity": identity, "generated_files": records}
                    ),
                    encoding="utf-8",
                )

            write_result(manifest)
            observed = _validate_completed_run(run_dir, identity)
            self.assertEqual(observed["request_identity"], identity)

            absolute = json.loads(json.dumps(manifest))
            absolute[0]["relative_path"] = str((run_dir / "outside.json").resolve())
            write_result(absolute)
            with self.assertRaisesRegex(RuntimeError, "Unsafe.*relative_path"):
                _validate_completed_run(run_dir, identity)

            traversal = json.loads(json.dumps(manifest))
            traversal[0]["relative_path"] = "../escape.json"
            write_result(traversal)
            with self.assertRaisesRegex(RuntimeError, "Unsafe.*relative_path"):
                _validate_completed_run(run_dir, identity)

            wrong_size = json.loads(json.dumps(manifest))
            wrong_size[0]["size_bytes"] += 1
            write_result(wrong_size)
            with self.assertRaisesRegex(RuntimeError, "size mismatch"):
                _validate_completed_run(run_dir, identity)

            wrong_hash = json.loads(json.dumps(manifest))
            wrong_hash[0]["sha256"] = "0" * 64
            write_result(wrong_hash)
            with self.assertRaisesRegex(RuntimeError, "SHA256 mismatch"):
                _validate_completed_run(run_dir, identity)

            missing_arm_state = [
                record
                for record in manifest
                if record["relative_path"] != "e0_legacy_window_codebook_state.json"
            ]
            write_result(missing_arm_state)
            with self.assertRaisesRegex(RuntimeError, "arm-required artifacts"):
                _validate_completed_run(run_dir, identity)

    def test_member_uses_unique_staging_and_residual_does_not_block_resume(self) -> None:
        identity = {
            "arms": ["E0"],
            "protocol": {"matched_random_controls": False},
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            final = root / "runs" / "A2" / "fold_01_seed_0_peak_valley_hierarchy_v1"
            final.parent.mkdir(parents=True)
            residual = final.parent / f".{final.name}.staging-interrupted"
            residual.mkdir()
            (residual / "partial.txt").write_text("keep me", encoding="utf-8")
            observed_staging: list[Path] = []

            def fake_command(
                args, checkpoint, output_dir, profile, fold, seed, arms
            ) -> list[str]:
                observed_staging.append(Path(output_dir))
                return ["fake-runner", "--output-dir", str(output_dir)]

            def fake_subprocess(command, cwd, check):
                self.assertEqual(command[-1], str(observed_staging[-1]))
                self.assertEqual(cwd, cv_module.PROJECT_ROOT)
                self.assertIs(check, True)
                _materialize_fake_completed_run(observed_staging[-1], identity)

            args = Namespace(skip_existing=True)
            with patch.object(
                cv_module, "_runner_command", side_effect=fake_command
            ), patch.object(cv_module.subprocess, "run", side_effect=fake_subprocess):
                result = cv_module._run_or_resume_member(
                    args=args,
                    checkpoint=Path("unused.pt"),
                    final_directory=final,
                    profile="A2",
                    fold=1,
                    seed=0,
                    arms=("E0",),
                    request_identity=identity,
                )

            self.assertEqual(result["request_identity"], identity)
            self.assertEqual(len(observed_staging), 1)
            staging = observed_staging[0]
            self.assertEqual(staging.parent, final.parent)
            self.assertTrue(staging.name.startswith(f".{final.name}.staging-"))
            self.assertFalse(staging.exists())
            self.assertTrue(final.is_dir())
            self.assertTrue(residual.is_dir())
            self.assertEqual(
                (residual / "partial.txt").read_text(encoding="utf-8"), "keep me"
            )

            with patch.object(cv_module.subprocess, "run") as rerun:
                resumed = cv_module._run_or_resume_member(
                    args=args,
                    checkpoint=Path("unused.pt"),
                    final_directory=final,
                    profile="A2",
                    fold=1,
                    seed=0,
                    arms=("E0",),
                    request_identity=identity,
                )
            self.assertEqual(resumed["request_identity"], identity)
            rerun.assert_not_called()
            self.assertTrue(residual.exists())

    def test_failed_member_keeps_staging_and_never_publishes_partial_output(self) -> None:
        identity = {
            "arms": ["E0"],
            "protocol": {"matched_random_controls": False},
        }
        with tempfile.TemporaryDirectory() as directory:
            final = Path(directory) / "fold_01_seed_0_peak_valley_hierarchy_v1"
            observed_staging: list[Path] = []

            def fake_command(
                args, checkpoint, output_dir, profile, fold, seed, arms
            ) -> list[str]:
                observed_staging.append(Path(output_dir))
                return ["fake-runner"]

            def fail_after_partial_write(command, cwd, check):
                observed_staging[-1].mkdir()
                (observed_staging[-1] / "partial.txt").write_text(
                    "interrupted", encoding="utf-8"
                )
                raise subprocess.CalledProcessError(9, command)

            with patch.object(
                cv_module, "_runner_command", side_effect=fake_command
            ), patch.object(
                cv_module.subprocess, "run", side_effect=fail_after_partial_write
            ):
                with self.assertRaises(subprocess.CalledProcessError):
                    cv_module._run_or_resume_member(
                        args=Namespace(skip_existing=True),
                        checkpoint=Path("unused.pt"),
                        final_directory=final,
                        profile="A2",
                        fold=1,
                        seed=0,
                        arms=("E0",),
                        request_identity=identity,
                    )
            self.assertFalse(final.exists())
            self.assertEqual(len(observed_staging), 1)
            self.assertTrue(observed_staging[0].is_dir())
            self.assertTrue((observed_staging[0] / "partial.txt").is_file())

    def test_atomic_publish_refuses_to_replace_even_an_empty_final_directory(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            staging = root / ".member.staging-fixed"
            final = root / "member"
            staging.mkdir()
            final.mkdir()
            (staging / "artifact.bin").write_bytes(b"complete")
            with self.assertRaisesRegex(FileExistsError, "Refusing to replace"):
                cv_module._atomic_publish_directory(staging, final)
            self.assertTrue(staging.is_dir())
            self.assertTrue((staging / "artifact.bin").is_file())
            self.assertTrue(final.is_dir())
            self.assertEqual(list(final.iterdir()), [])

    def test_missing_encoder_is_validated_in_staging_then_atomically_published(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            encoder_root = root / "encoders"
            cv_root = root / "cv"
            cv_root.mkdir()
            npz_path = root / "windows.npz"
            npz_path.write_bytes(b"registered npz")
            source = root / "source.pt"
            source.write_bytes(b"source")
            observed_staging: list[Path] = []

            def fake_trainer(
                python, source_path, npz, output_dir, profile, seed, device
            ) -> list[str]:
                observed_staging.append(Path(output_dir))
                return ["fake-trainer", str(output_dir)]

            def write_checkpoint(command, cwd, check):
                staging = observed_staging[-1]
                staging.mkdir()
                (staging / "motion_encoder_final.pt").write_bytes(b"checkpoint")

            no_match = RuntimeError(
                "Expected exactly one schema-v1 motion checkpoint; matched 0"
            )
            with patch.object(
                cv_module, "find_motion_encoder_checkpoint", side_effect=no_match
            ), patch.object(cv_module, "find_checkpoint", return_value=source), patch.object(
                cv_module, "_trainer_command", side_effect=fake_trainer
            ), patch.object(
                cv_module.subprocess, "run", side_effect=write_checkpoint
            ), patch.object(
                cv_module, "_validate_profile_semantics", return_value={"ok": True}
            ) as semantics, patch.object(
                cv_module,
                "validate_motion_encoder_grid_identity",
                return_value={"training_identity_sha256": "x"},
            ) as integrity:
                published = cv_module._resolve_checkpoint(
                    encoder_root=encoder_root,
                    cv_root=cv_root,
                    npz_path=npz_path,
                    profile="A2",
                    fold=1,
                    seed=0,
                    python="python",
                    device="cpu",
                    train_missing=True,
                    dry_run=False,
                )

            expected_directory = cv_module._encoder_output_dir(
                encoder_root, "A2", 1, 0
            )
            self.assertEqual(published, expected_directory / "motion_encoder_final.pt")
            self.assertTrue(published.is_file())
            self.assertEqual(published.read_bytes(), b"checkpoint")
            self.assertEqual(len(observed_staging), 1)
            self.assertEqual(observed_staging[0].parent, encoder_root.parent)
            self.assertFalse(observed_staging[0].exists())
            self.assertEqual(semantics.call_count, 2)
            integrity.assert_called_once()

    def test_invalid_staged_encoder_is_preserved_without_canonical_publish(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            encoder_root = root / "encoders"
            cv_root = root / "cv"
            cv_root.mkdir()
            npz_path = root / "windows.npz"
            npz_path.write_bytes(b"registered npz")
            source = root / "source.pt"
            source.write_bytes(b"source")
            observed_staging: list[Path] = []

            def fake_trainer(
                python, source_path, npz, output_dir, profile, seed, device
            ) -> list[str]:
                observed_staging.append(Path(output_dir))
                return ["fake-trainer"]

            def write_invalid_checkpoint(command, cwd, check):
                observed_staging[-1].mkdir()
                (observed_staging[-1] / "motion_encoder_final.pt").write_bytes(
                    b"invalid checkpoint"
                )

            with patch.object(
                cv_module,
                "find_motion_encoder_checkpoint",
                side_effect=RuntimeError("matched 0"),
            ), patch.object(cv_module, "find_checkpoint", return_value=source), patch.object(
                cv_module, "_trainer_command", side_effect=fake_trainer
            ), patch.object(
                cv_module.subprocess, "run", side_effect=write_invalid_checkpoint
            ), patch.object(
                cv_module,
                "_validate_profile_semantics",
                side_effect=RuntimeError("invalid staged encoder"),
            ):
                with self.assertRaisesRegex(RuntimeError, "invalid staged encoder"):
                    cv_module._resolve_checkpoint(
                        encoder_root=encoder_root,
                        cv_root=cv_root,
                        npz_path=npz_path,
                        profile="A2",
                        fold=1,
                        seed=0,
                        python="python",
                        device="cpu",
                        train_missing=True,
                        dry_run=False,
                    )

            canonical = cv_module._encoder_output_dir(encoder_root, "A2", 1, 0)
            self.assertFalse(canonical.exists())
            self.assertEqual(len(observed_staging), 1)
            self.assertTrue(observed_staging[0].is_dir())
            self.assertTrue(
                (observed_staging[0] / "motion_encoder_final.pt").is_file()
            )

    def test_dry_run_does_not_hash_a_checkpoint_that_will_be_trained(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cv_root = root / "cv"
            cv_root.mkdir()
            npz_path = root / "windows.npz"
            npz_path.write_bytes(b"dry-run identity probe")
            missing_checkpoint = (
                root / "encoders" / "A2" / "motion_encoder_final.pt"
            )
            args = cv_module.build_parser().parse_args(
                [
                    "--cv-root",
                    str(cv_root),
                    "--npz-path",
                    str(npz_path),
                    "--encoder-root",
                    str(root / "encoders"),
                    "--output-root",
                    str(root / "results"),
                    "--profiles",
                    "A2",
                    "--folds",
                    "1",
                    "--seeds",
                    "0",
                    "--dry-run",
                ]
            )
            fingerprints = {
                "runner": "runner-hash",
                "algorithm": "algorithm-hash",
                "cv_wrapper": "cv-hash",
            }
            with patch.object(
                cv_module, "_resolve_checkpoint", return_value=missing_checkpoint
            ), patch.object(
                cv_module, "_source_fingerprints", return_value=fingerprints
            ):
                result = cv_module.run(args)

            self.assertIs(result["dry_run"], True)
            self.assertEqual(result["run_count"], 1)
            self.assertFalse(missing_checkpoint.exists())

    def test_a2_semantics_and_npz_hash_are_fail_closed(self) -> None:
        npz_hash = "a" * 64
        payload = {
            "npz_sha256": npz_hash,
            "data": {"npz_sha256": npz_hash},
            "split_audit": {"npz_sha256": npz_hash},
            "experiment_metadata": {
                "uschad_cv_fold": 3,
                "motion_encoder_seed": 50,
                "smoke_test": False,
                "uschad_window_size": 128,
            },
            "selection": {"policy": "final_epoch"},
            "resolved_training_config": {
                "ablation_profile": "A2",
                "backbone_bn_policy": "frozen",
                "window_aug_consistency": "none",
                "window_aug_profile": "basic",
                "window_aug_one_window_per_trial": True,
                "cp_anchor": {"source": "raw_frozen_consensus"},
                "architecture": {"content_residual": True},
                "loss_weights": {
                    "window_augmentation": 0.0,
                    "noncollapse": 0.05,
                    "changepoint": 1.0,
                    "content_boundary_alignment": 0.1,
                    "temporal_prediction": 0.5,
                    "trial_auxiliary": 0.1,
                    "cross_subject": 0.0,
                },
            },
            "command_arguments": {
                "ablation_profile": "A2",
                "window_aug_weight": 1.0,
                "noncollapse_weight": 0.05,
                "prediction_weight": 0.5,
                "trial_weight": 0.1,
                "cross_subject_weight": 0.0,
            },
        }
        with patch.object(cv_module, "_load_torch", return_value=payload):
            checks = _validate_profile_semantics(
                Path("unused.pt"), "A2", 3, 50, npz_hash, 128
            )
        self.assertTrue(all(checks.values()))

        with patch.object(cv_module, "_load_torch", return_value=payload):
            with self.assertRaisesRegex(RuntimeError, "violates A2 semantics"):
                _validate_profile_semantics(
                    Path("unused.pt"), "A2", 3, 50, npz_hash, 256
                )

        payload["resolved_training_config"]["loss_weights"]["cross_subject"] = 0.1
        with patch.object(cv_module, "_load_torch", return_value=payload):
            with self.assertRaisesRegex(RuntimeError, "violates A2 semantics"):
                _validate_profile_semantics(
                    Path("unused.pt"), "A2", 3, 50, npz_hash
                )
        payload["resolved_training_config"]["loss_weights"]["cross_subject"] = 0.0
        payload["data"]["npz_sha256"] = "b" * 64
        with patch.object(cv_module, "_load_torch", return_value=payload):
            with self.assertRaisesRegex(RuntimeError, "violates A2 semantics"):
                _validate_profile_semantics(
                    Path("unused.pt"), "A2", 3, 50, npz_hash
                )


if __name__ == "__main__":
    unittest.main()
