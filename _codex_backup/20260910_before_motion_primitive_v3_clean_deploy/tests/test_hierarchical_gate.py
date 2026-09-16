from __future__ import annotations

from copy import deepcopy
from dataclasses import fields, replace
import csv
import inspect
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

import experiments.motion_primitive.run_online_hierarchical_gate as gate_runner

from experiments.motion_primitive.hierarchical_gate import (
    GATE_STATES,
    HierarchicalAssignments,
    HierarchicalGateModel,
    assert_gate_schema_is_label_free,
    assign_hierarchical_gate,
    fit_hierarchical_gate,
    hierarchical_duration_histograms,
    hierarchical_gate_diagnostics,
    motion_components_from_token_partitions,
)
from experiments.motion_primitive.online_secondary_codebook import (
    UnlabelledTokenOccurrence,
)
from experiments.motion_primitive.run_online_hierarchical_gate import (
    ARMS,
    REGISTERED_SEGMENTATION_PARAMETERS,
    REGISTERED_UPSTREAM_CODEBOOK,
    _posthoc_gate_sit_stand,
    _validate_registered_upstream,
)
from experiments.motion_primitive.analyze_online_hierarchical_gate import (
    CURRENT_IMPLEMENTATION_PATHS,
    EXPECTED_RUNS,
    FrozenGateRun,
    analyze_gate_runs,
    load_frozen_gate_run,
    validate_paired_gate_runs,
)
from experiments.motion_primitive.trajectory_ablation import sha256_file


VALID_DISABLE_REASON = (
    "motion_energy_ratio_below_threshold;motion_energy_gap_below_threshold"
)


def occurrence(
    trial_id: int,
    residual: tuple[float, float],
    gravity: tuple[float, float, float],
    token: int = 7,
) -> UnlabelledTokenOccurrence:
    vector = np.asarray(gravity, dtype=np.float32)
    vector /= np.linalg.norm(vector)
    return UnlabelledTokenOccurrence(
        trial_global_id=int(trial_id),
        coarse_token=int(token),
        token_fraction=0.75,
        duration_samples=30,
        mean_quantization_residual=np.asarray(residual, dtype=np.float32),
        gravity_direction=vector,
    )


def registered_upstream_config(
    *,
    profile: str = "A0",
    checkpoint_sha256: str = "c" * 64,
) -> dict:
    arguments = {
        "primitive_num": 32,
        "pca_dim": 64,
        "embedding_normalization": "l2",
        "codebook_weighting": "per_trial",
        "kmeans_n_init": 20,
        "kmeans_max_iter": 300,
        "old_class_count": 6,
        **REGISTERED_SEGMENTATION_PARAMETERS,
    }
    return {
        "arguments": arguments,
        "codebook": {
            "primitive_num": 32,
            "pca_dim": 64,
            "embedding_normalization": "l2",
            "assignment_metric": "cosine",
            "weighting": "per_trial",
        },
        "checkpoint_metadata": {
            "smoke_test": False,
            "outer_test_used_during_encoder_training": False,
            "offline_val_subjects": [6, 14],
        },
        "checkpoint_sha256": checkpoint_sha256,
        "encoder_implementation_fingerprint": {"combined_sha256": "a" * 64},
        "encoder_source_checkpoint": {"sha256": "b" * 64},
        "encoder_training": {"ablation_profile": profile},
    }


def registered_upstream_artifacts() -> SimpleNamespace:
    return SimpleNamespace(
        centers=np.zeros((32, 64), dtype=np.float32),
        embeddings=np.zeros((3, 64), dtype=np.float32),
    )


def write_registered_frozen_run(
    run_dir: Path,
    *,
    profile: str = "A0",
    segmentation: str = "fixed_window",
    gate_enabled: object = True,
    gate_disable_reason: object = None,
    aligned_predictions: dict[str, list[int]] | None = None,
    raw_predictions: dict[str, list[int]] | None = None,
) -> tuple[dict, list[dict[str, object]]]:
    run_dir.mkdir(parents=True, exist_ok=True)
    trial_ids = list(range(20))
    truth = [value % 10 for value in trial_ids]
    subjects = [4] * 10 + [5] * 10
    checkpoint_sha256 = ("c" if profile == "A0" else "d") * 64
    upstream = _validate_registered_upstream(
        registered_upstream_config(
            profile=profile,
            checkpoint_sha256=checkpoint_sha256,
        ),
        registered_upstream_artifacts(),
    )
    implementation = {
        field: sha256_file(path)
        for field, path in CURRENT_IMPLEMENTATION_PATHS.items()
    }
    aligned_by_arm = {
        arm: list((aligned_predictions or {}).get(arm, truth)) for arm in ARMS
    }
    raw_by_arm = {
        arm: list((raw_predictions or {}).get(arm, truth)) for arm in ARMS
    }
    gate_fit = {
        "fit_trial_ids": [100, 101],
        "gate_enabled": gate_enabled,
        "gate_disable_reason": gate_disable_reason,
        "motion_energy_ratio": 1.2 if gate_enabled is False else 3.0,
        "motion_energy_gap": 0.1 if gate_enabled is False else 0.5,
        "residual_child_normalized_motion_component_medians": [
            [0.1, 0.2],
            [1.5, 2.0],
        ],
        "static_residual_child": 0,
        "fit_trial_count": 2,
        "confident_static_fit_count": 0 if gate_enabled is False else 2,
        "static_radius_quantile": 0.95,
        "minimum_token_fraction": 0.50,
        "minimum_motion_energy_ratio": 2.0,
        "minimum_motion_energy_gap": 0.25,
    }
    result = {
        "arguments": {
            "seed": 500,
            "static_radius_quantile": 0.95,
            "minimum_token_fraction": 0.50,
            "minimum_motion_energy_ratio": 2.0,
            "minimum_motion_energy_gap": 0.25,
        },
        "input_audit": {
            "encoder_ablation_profile": profile,
            "primitive_segmentation": segmentation,
            "checkpoint_sha256": checkpoint_sha256,
            "npz_sha256": "e" * 64,
            "registered_upstream": upstream,
            "implementation_fingerprint": implementation,
        },
        "session_manifest_audit": {
            "registered_protocol_verified": True,
            "future_feature_trial_count": 0,
            "seed": 500,
            "session_1_train_trial_ids": [100],
            "session_2_train_trial_ids": [101],
            "session_2_test_trial_ids": trial_ids,
            "cumulative_train_trial_ids": [100, 101],
        },
        "label_firewall": {
            "metadata_aware_repository_created_after_predictions": True,
            "gate_fit_uses_activity_labels_or_names": False,
            "test_motion_energy_used_for_gate_assignment": False,
        },
        "candidate": {"dominant_trial_ids": [100, 101]},
        "gate_fit": gate_fit,
        "arms": {},
    }
    truth_array = np.asarray(truth, dtype=np.int64)
    for arm in ARMS:
        values = np.asarray(aligned_by_arm[arm], dtype=np.int64)
        correct = values == truth_array
        result["arms"][arm] = {
            "global_metrics": {
                "all_accuracy": float(np.mean(correct)),
                "old_accuracy": float(np.mean(correct[truth_array < 6])),
                "new_accuracy": float(np.mean(correct[truth_array >= 6])),
                "aligned_predictions": values.tolist(),
            }
        }
    rows: list[dict[str, object]] = []
    for position, (trial_id, subject, label) in enumerate(
        zip(trial_ids, subjects, truth)
    ):
        row: dict[str, object] = {
            "trial_global_id": trial_id,
            "subject_id": subject,
            "activity_label_0based": label,
        }
        for arm in ARMS:
            row[f"{arm}_raw_cluster"] = raw_by_arm[arm][position]
            row[f"{arm}_aligned_prediction"] = aligned_by_arm[arm][position]
        rows.append(row)
    manifest_rows = [
        {"split": "online_train", "session": 1, "trial_global_id": 100},
        {"split": "online_train", "session": 2, "trial_global_id": 101},
        *[
            {"split": "online_test", "session": 2, "trial_global_id": trial_id}
            for trial_id in trial_ids
        ],
    ]
    with (run_dir / "session_manifest.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(manifest_rows[0]))
        writer.writeheader()
        writer.writerows(manifest_rows)
    with (run_dir / "session2_predictions.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (run_dir / "online_hierarchical_gate_results.json").write_text(
        json.dumps(result), encoding="utf-8"
    )
    return result, rows


class HierarchicalGateTests(unittest.TestCase):
    @staticmethod
    def _fit_fixture() -> tuple[HierarchicalGateModel, list[UnlabelledTokenOccurrence]]:
        fit = [
            occurrence(10, (0.00, 0.00), (1.0, 0.0, 0.0)),
            occurrence(11, (0.04, 0.00), (0.99, 0.05, 0.0)),
            occurrence(12, (0.00, 0.04), (0.0, 1.0, 0.0)),
            occurrence(13, (0.04, 0.04), (0.05, 0.99, 0.0)),
            occurrence(20, (4.00, 4.00), (0.0, 0.0, 1.0)),
            occurrence(21, (4.05, 4.00), (0.0, 0.0, 1.0)),
            occurrence(22, (4.00, 4.05), (0.0, 0.0, 1.0)),
        ]
        motion = {
            10: np.asarray([0.01, 0.01]),
            11: np.asarray([0.02, 0.01]),
            12: np.asarray([0.01, 0.02]),
            13: np.asarray([0.02, 0.02]),
            20: np.asarray([2.0, 4.0]),
            21: np.asarray([2.2, 4.2]),
            22: np.asarray([2.1, 4.4]),
        }
        model = fit_hierarchical_gate(
            fit,
            motion,
            static_radius_quantile=1.0,
            restarts=30,
            seed=17,
        )
        return model, fit

    def test_registered_upstream_validator_accepts_only_frozen_k32_pca64(self) -> None:
        config = registered_upstream_config()
        artifacts = registered_upstream_artifacts()
        audit = _validate_registered_upstream(config, artifacts)
        self.assertEqual(
            audit["registered_upstream_codebook"], REGISTERED_UPSTREAM_CODEBOOK
        )
        self.assertEqual(audit["artifact_center_shape"], [32, 64])
        self.assertEqual(audit["artifact_embedding_dim"], 64)
        self.assertEqual(
            audit["observed_segmentation_parameters"],
            REGISTERED_SEGMENTATION_PARAMETERS,
        )

        for field, value in (("primitive_num", 31), ("pca_dim", 63)):
            with self.subTest(arguments_field=field):
                drifted = deepcopy(config)
                drifted["arguments"][field] = value
                with self.assertRaisesRegex(RuntimeError, "protocol drifted"):
                    _validate_registered_upstream(drifted, artifacts)

        for field, value in (("primitive_num", 32.5), ("kmeans_n_init", 20.7)):
            with self.subTest(fractional_integer_field=field):
                drifted = deepcopy(config)
                drifted["arguments"][field] = value
                with self.assertRaisesRegex(RuntimeError, "must be an integer"):
                    _validate_registered_upstream(drifted, artifacts)

        for shape in ((31, 64), (32, 63)):
            with self.subTest(artifact_shape=shape):
                drifted_artifacts = SimpleNamespace(
                    centers=np.zeros(shape, dtype=np.float32),
                    embeddings=np.zeros((3, shape[1]), dtype=np.float32),
                )
                with self.assertRaisesRegex(RuntimeError, "not the registered"):
                    _validate_registered_upstream(config, drifted_artifacts)

    def test_registered_upstream_validator_rejects_unsafe_or_invalid_hashes(self) -> None:
        artifacts = registered_upstream_artifacts()
        mutations = {
            "missing implementation hash": lambda item: item[
                "encoder_implementation_fingerprint"
            ].pop("combined_sha256"),
            "short implementation hash": lambda item: item[
                "encoder_implementation_fingerprint"
            ].__setitem__("combined_sha256", "a" * 63),
            "nonhex source hash": lambda item: item[
                "encoder_source_checkpoint"
            ].__setitem__("sha256", "z" * 64),
            "signed hex source hash": lambda item: item[
                "encoder_source_checkpoint"
            ].__setitem__("sha256", "+" + "a" * 63),
            "missing motion checkpoint hash": lambda item: item.pop(
                "checkpoint_sha256"
            ),
            "smoke checkpoint": lambda item: item["checkpoint_metadata"].__setitem__(
                "smoke_test", True
            ),
            "outer-test checkpoint": lambda item: item[
                "checkpoint_metadata"
            ].__setitem__("outer_test_used_during_encoder_training", True),
        }
        for name, mutate in mutations.items():
            with self.subTest(name=name):
                config = registered_upstream_config()
                mutate(config)
                with self.assertRaises(RuntimeError):
                    _validate_registered_upstream(config, artifacts)

    def test_runner_calls_upstream_validator_immediately_after_artifact_load(self) -> None:
        class UpstreamValidationSentinel(RuntimeError):
            pass

        config = registered_upstream_config()
        artifacts = registered_upstream_artifacts()
        events: list[str] = []

        def load_artifacts(*_args, **_kwargs):
            events.append("artifacts_loaded")
            return artifacts

        def validate_upstream(observed_config, observed_artifacts):
            self.assertIs(observed_config, config)
            self.assertIs(observed_artifacts, artifacts)
            self.assertEqual(events, ["artifacts_loaded"])
            events.append("upstream_validated")
            raise UpstreamValidationSentinel("registered-upstream-sentinel")

        args = SimpleNamespace(
            bootstrap_resamples=10,
            static_radius_quantile=0.95,
            minimum_motion_energy_ratio=2.0,
            minimum_motion_energy_gap=0.25,
            minimum_token_fraction=0.50,
            run_dir="upstream-run",
            npz_path="source.npz",
            output_dir="unused-output",
            seed=500,
        )
        split = {"fit_subjects": [1, 2], "eval_subjects": [4, 5]}
        manifest = {
            "cumulative_train_trial_ids": [100, 101],
            "session_2_test_trial_ids": list(range(20)),
        }
        with (
            patch.object(gate_runner, "_require_new_output_dir", return_value=Path("out")),
            patch.object(
                gate_runner,
                "_validate_fold06_run",
                return_value=(config, split, Path("window"), Path("codebook")),
            ),
            patch.object(
                gate_runner,
                "build_fold06_session_manifest",
                return_value=(manifest, None),
            ),
            patch.object(gate_runner, "load_segment_artifacts", side_effect=load_artifacts),
            patch.object(
                gate_runner,
                "_validate_registered_upstream",
                side_effect=validate_upstream,
            ),
            patch.object(
                gate_runner,
                "LabelFreeSourceSignalRepository",
                side_effect=AssertionError("validator was skipped"),
            ),
        ):
            with self.assertRaisesRegex(
                UpstreamValidationSentinel, "registered-upstream-sentinel"
            ):
                gate_runner.run(args)
        self.assertEqual(events, ["artifacts_loaded", "upstream_validated"])

    def test_fitting_schema_has_no_activity_identity(self) -> None:
        assert_gate_schema_is_label_free()
        for schema in (HierarchicalGateModel, HierarchicalAssignments):
            names = {item.name.lower() for item in fields(schema)}
            self.assertFalse(
                any(
                    fragment in name
                    for name in names
                    for fragment in ("label", "activity", "class", "name")
                )
            )
        fit_parameters = set(inspect.signature(fit_hierarchical_gate).parameters)
        self.assertFalse(
            fit_parameters
            & {"labels", "activity_labels", "activity_names", "class_ids"}
        )

    def test_motion_components_remove_gravity_orientation_but_detect_motion(self) -> None:
        static = np.zeros((6, 12), dtype=np.float32)
        static[0] = 9.81
        dynamic = static.copy()
        dynamic[0, 0:8:2] += 3.0
        dynamic[0, 1:8:2] -= 3.0
        dynamic[4, 0:8:2] = 5.0
        dynamic[4, 1:8:2] = -5.0
        kwargs = {
            "partition_starts": np.asarray([0, 4, 8]),
            "partition_ends": np.asarray([4, 8, 12]),
            "partition_tokens": np.asarray([7, 7, 3]),
            "selected_token": 7,
        }
        static_components = motion_components_from_token_partitions(static, **kwargs)
        dynamic_components = motion_components_from_token_partitions(dynamic, **kwargs)
        np.testing.assert_allclose(static_components, np.zeros(2), atol=1e-8)
        self.assertGreater(dynamic_components[0], static_components[0])
        self.assertGreater(dynamic_components[1], static_components[1])

    def test_hierarchy_splits_only_in_distribution_static_occurrences(self) -> None:
        model, fit = self._fit_fixture()
        assignments = assign_hierarchical_gate(model, fit)
        static_ids = {10, 11, 12, 13}
        dynamic_ids = {20, 21, 22}
        observed_static = {
            int(trial_id)
            for trial_id, state in zip(assignments.trial_ids, assignments.gate_states)
            if int(state) > 0
        }
        self.assertEqual(observed_static, static_ids)
        for trial_id, state in zip(assignments.trial_ids, assignments.gate_states):
            if int(trial_id) in dynamic_ids:
                self.assertEqual(int(state), 0)
        self.assertEqual(set(assignments.gate_states[:4].tolist()), {1, 2})

        # This point remains closer to the static residual medoid but lies far
        # outside the train-only static radius, so it must fall back to coarse.
        outlier = occurrence(30, (1.50, 1.50), (1.0, 0.0, 0.0))
        outlier_assignment = assign_hierarchical_gate(model, [outlier])
        self.assertEqual(
            int(outlier_assignment.residual_children[0]),
            int(model.static_residual_child),
        )
        self.assertFalse(bool(outlier_assignment.confident_static[0]))
        self.assertEqual(int(outlier_assignment.gravity_children[0]), -1)
        self.assertEqual(int(outlier_assignment.gate_states[0]), 0)

        low_fraction = replace(fit[0], trial_global_id=31, token_fraction=0.20)
        low_fraction_assignment = assign_hierarchical_gate(model, [low_fraction])
        self.assertEqual(
            int(low_fraction_assignment.residual_children[0]),
            int(model.static_residual_child),
        )
        self.assertFalse(bool(low_fraction_assignment.confident_static[0]))
        self.assertEqual(int(low_fraction_assignment.gate_states[0]), 0)

    def test_fit_is_deterministic_and_diagnostics_close_the_training_loop(self) -> None:
        first, fit = self._fit_fixture()
        motion = {
            10: np.asarray([0.01, 0.01]),
            11: np.asarray([0.02, 0.01]),
            12: np.asarray([0.01, 0.02]),
            13: np.asarray([0.02, 0.02]),
            20: np.asarray([2.0, 4.0]),
            21: np.asarray([2.2, 4.2]),
            22: np.asarray([2.1, 4.4]),
        }
        second = fit_hierarchical_gate(
            fit, motion, static_radius_quantile=1.0, restarts=30, seed=17
        )
        self.assertEqual(first.static_residual_child, second.static_residual_child)
        self.assertEqual(first.static_radius, second.static_radius)
        self.assertEqual(
            first.confident_static_fit_trial_ids,
            second.confident_static_fit_trial_ids,
        )
        self.assertEqual(
            first.residual_codebook.medoid_trial_ids,
            second.residual_codebook.medoid_trial_ids,
        )
        self.assertEqual(
            first.gravity_codebook.medoid_trial_ids,
            second.gravity_codebook.medoid_trial_ids,
        )
        diagnostics = hierarchical_gate_diagnostics(first, fit)
        self.assertEqual(diagnostics["fit_trial_count"], 7)
        self.assertEqual(diagnostics["confident_static_fit_count"], 4)
        self.assertEqual(sum(diagnostics["fit_gate_state_counts"]), 7)
        self.assertIs(diagnostics["fit_uses_activity_labels"], False)
        self.assertIs(diagnostics["test_motion_energy_used_for_assignment"], False)

    def test_hierarchical_histogram_preserves_mass_and_fallback_token(self) -> None:
        durations = {
            1: {2: 4.0, 7: 6.0},
            2: {7: 10.0},
            3: {1: 3.0, 7: 7.0},
        }
        matrix = hierarchical_duration_histograms(
            [1, 2, 3],
            durations,
            primitive_num=8,
            selected_token=7,
            gate_state_by_trial={1: 0, 2: 1, 3: 2},
        )
        self.assertEqual(matrix.shape, (3, 10))
        np.testing.assert_allclose(matrix.sum(axis=1), np.ones(3), atol=1e-8)
        self.assertAlmostEqual(float(matrix[0, 7]), 0.6)
        self.assertAlmostEqual(float(matrix[1, 8]), 1.0)
        self.assertAlmostEqual(float(matrix[2, 9]), 0.7)
        self.assertEqual(GATE_STATES[0], "coarse_fallback")

    def test_fit_rejects_motion_keys_or_energy_tie_that_cannot_name_static(self) -> None:
        fit = [
            occurrence(1, (0.0, 0.0), (1.0, 0.0, 0.0)),
            occurrence(2, (0.1, 0.0), (0.0, 1.0, 0.0)),
            occurrence(3, (4.0, 4.0), (1.0, 0.0, 0.0)),
            occurrence(4, (4.1, 4.0), (0.0, 1.0, 0.0)),
        ]
        with self.assertRaisesRegex(ValueError, "exactly match"):
            fit_hierarchical_gate(
                fit,
                {trial_id: np.ones(2) for trial_id in (1, 2, 3)},
                static_radius_quantile=1.0,
            )
        disabled = fit_hierarchical_gate(
            fit,
            {trial_id: np.ones(2) for trial_id in (1, 2, 3, 4)},
            static_radius_quantile=1.0,
        )
        self.assertIs(disabled.gate_enabled, False)
        self.assertIn("motion_energy_ratio_below_threshold", disabled.gate_disable_reason)
        self.assertIn("motion_energy_gap_below_threshold", disabled.gate_disable_reason)
        self.assertIsNone(disabled.gravity_codebook)
        assignments = assign_hierarchical_gate(disabled, fit)
        np.testing.assert_array_equal(assignments.gate_states, np.zeros(4, dtype=np.int64))
        self.assertFalse(np.any(assignments.confident_static))

    def test_gate_sit_stand_diagnostic_counts_fallback_as_an_error(self) -> None:
        states = {1: 1, 2: 1, 3: 2, 4: 0}
        labels = {1: 7, 2: 7, 3: 8, 4: 8}
        subjects = {1: 4, 2: 5, 3: 4, 4: 5}
        result = _posthoc_gate_sit_stand([1, 2, 3, 4], states, labels, subjects)
        self.assertEqual(result["binary_trial_count"], 4)
        self.assertEqual(result["observed_gate_states"], [0, 1, 2])
        self.assertEqual(result["unmapped_fallback_or_child_count"], 1)
        self.assertAlmostEqual(result["binary_hungarian_accuracy"], 0.75)
        self.assertAlmostEqual(result["binary_balanced_accuracy"], 0.75)
        self.assertIs(result["fallback_is_counted_not_dropped"], True)
        self.assertEqual(result["routed_trial_count"], 3)
        self.assertAlmostEqual(result["conditional_accuracy_on_routed_trials"], 1.0)
        disabled = _posthoc_gate_sit_stand(
            [1, 2, 3, 4], {trial_id: 0 for trial_id in (1, 2, 3, 4)}, labels, subjects
        )
        self.assertEqual(disabled["binary_hungarian_accuracy"], 0.0)
        self.assertEqual(disabled["binary_balanced_accuracy"], 0.0)
        self.assertEqual(disabled["routed_trial_count"], 0)
        self.assertIsNone(disabled["conditional_accuracy_on_routed_trials"])

    def test_four_run_analysis_is_paired_and_detects_encoder_effect(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runs = {}
            for key, (profile, segmentation) in EXPECTED_RUNS.items():
                run_dir = root / key
                g3 = list(range(10)) * 2
                if profile == "A0":
                    g3[0] = 9
                    g3[10] = 9
                write_registered_frozen_run(
                    run_dir,
                    profile=profile,
                    segmentation=segmentation,
                    aligned_predictions={"G3_hierarchical": g3},
                )
                runs[key] = load_frozen_gate_run(key, run_dir)
            audit = validate_paired_gate_runs(runs)
            self.assertEqual(audit["paired_test_trial_count"], 20)
            self.assertIs(audit["upstream_codebook_protocol_identical"], True)
            self.assertIs(audit["segmentation_parameters_identical"], True)
            self.assertIs(audit["encoder_implementation_sha256_identical"], True)
            self.assertIs(
                audit["encoder_source_checkpoint_sha256_identical"], True
            )
            self.assertIs(
                audit[
                    "fixed_changepoint_checkpoint_sha256_identical_within_profile"
                ],
                True,
            )
            result = analyze_gate_runs(runs, replicates=200, seed=31)
            effect = next(
                item
                for item in result["effects"]
                if item["comparison"]
                == "A3_minus_A0/changepoint/G3_hierarchical"
                and item["metric"] == "all_accuracy"
            )
            self.assertAlmostEqual(effect["left_minus_right"], 0.1)

            changed = dict(runs)
            original = runs["A3_fixed"]
            changed_truth = original.truth.copy()
            changed_truth[0] = 9
            changed["A3_fixed"] = replace(original, truth=changed_truth)
            with self.assertRaisesRegex(RuntimeError, "ground truth"):
                validate_paired_gate_runs(changed)

    def test_paired_audit_rejects_upstream_encoder_or_checkpoint_drift(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runs = {}
            for key, (profile, segmentation) in EXPECTED_RUNS.items():
                run_dir = root / key
                write_registered_frozen_run(
                    run_dir,
                    profile=profile,
                    segmentation=segmentation,
                )
                runs[key] = load_frozen_gate_run(key, run_dir)

            mutations = {
                "upstream codebook": lambda result: result["input_audit"][
                    "registered_upstream"
                ]["registered_upstream_codebook"].__setitem__("primitive_num", 31),
                "segmentation parameters": lambda result: result["input_audit"][
                    "registered_upstream"
                ]["observed_segmentation_parameters"].__setitem__(
                    "changepoint_context_windows", 3
                ),
                "encoder implementation": lambda result: result["input_audit"][
                    "registered_upstream"
                ].__setitem__("encoder_implementation_combined_sha256", "1" * 64),
                "source checkpoint": lambda result: result["input_audit"][
                    "registered_upstream"
                ].__setitem__("encoder_source_checkpoint_sha256", "2" * 64),
                "profile checkpoint": lambda result: (
                    result["input_audit"].__setitem__("checkpoint_sha256", "3" * 64),
                    result["input_audit"]["registered_upstream"].__setitem__(
                        "motion_encoder_checkpoint_sha256", "3" * 64
                    ),
                ),
            }
            for name, mutate in mutations.items():
                with self.subTest(name=name):
                    changed = dict(runs)
                    original = runs["A3_changepoint"]
                    changed_result = deepcopy(original.result)
                    mutate(changed_result)
                    changed["A3_changepoint"] = replace(
                        original, result=changed_result
                    )
                    with self.assertRaises(RuntimeError):
                        validate_paired_gate_runs(changed)

    def test_load_rejects_stale_code_and_registered_parameter_drift(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            result_path = run_dir / "online_hierarchical_gate_results.json"
            result, _ = write_registered_frozen_run(run_dir)
            loaded = load_frozen_gate_run("A0_fixed", run_dir)
            self.assertEqual(len(loaded.trial_ids), 20)

            stale = deepcopy(result)
            stale["input_audit"]["implementation_fingerprint"]["runner_sha256"] = "stale"
            result_path.write_text(json.dumps(stale), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "current frozen implementation"):
                load_frozen_gate_run("A0_fixed", run_dir)

            drift = deepcopy(result)
            drift["arguments"]["minimum_motion_energy_ratio"] = 1.9
            result_path.write_text(json.dumps(drift), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "expected registered"):
                load_frozen_gate_run("A0_fixed", run_dir)

            checkpoint_mismatch = deepcopy(result)
            checkpoint_mismatch["input_audit"]["registered_upstream"][
                "motion_encoder_checkpoint_sha256"
            ] = "f" * 64
            result_path.write_text(
                json.dumps(checkpoint_mismatch), encoding="utf-8"
            )
            with self.assertRaisesRegex(RuntimeError, "checkpoint hashes disagree"):
                load_frozen_gate_run("A0_fixed", run_dir)

    def test_load_rejects_missing_drifted_or_nonfinite_gate_parameters(self) -> None:
        cases = (
            ("arguments missing", "arguments", "minimum_token_fraction", "missing"),
            ("arguments drift", "arguments", "static_radius_quantile", 0.90),
            ("arguments nan", "arguments", "minimum_motion_energy_gap", np.nan),
            ("arguments inf", "arguments", "minimum_motion_energy_ratio", np.inf),
            ("gate fit missing", "gate_fit", "minimum_token_fraction", "missing"),
            ("gate fit drift", "gate_fit", "static_radius_quantile", 0.90),
            ("gate fit nan", "gate_fit", "minimum_motion_energy_gap", np.nan),
            ("gate fit inf", "gate_fit", "minimum_motion_energy_ratio", np.inf),
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for index, (name, section, field, value) in enumerate(cases):
                with self.subTest(name=name):
                    run_dir = root / str(index)
                    result, _ = write_registered_frozen_run(run_dir)
                    if value == "missing":
                        result[section].pop(field)
                    else:
                        result[section][field] = value
                    (run_dir / "online_hierarchical_gate_results.json").write_text(
                        json.dumps(result), encoding="utf-8"
                    )
                    with self.assertRaises(RuntimeError):
                        load_frozen_gate_run("A0_fixed", run_dir)

    def test_load_requires_strict_boolean_and_consistent_gate_status(self) -> None:
        invalid_statuses = (
            ("integer enabled", 1, None),
            ("string disabled", "false", "energy threshold failed"),
            ("enabled with reason", True, "energy threshold failed"),
            ("disabled without reason", False, None),
            ("disabled empty reason", False, "  "),
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for index, (name, enabled, reason) in enumerate(invalid_statuses):
                with self.subTest(name=name):
                    run_dir = root / str(index)
                    write_registered_frozen_run(
                        run_dir,
                        gate_enabled=enabled,
                        gate_disable_reason=reason,
                    )
                    with self.assertRaises(RuntimeError):
                        load_frozen_gate_run("A0_fixed", run_dir)

            missing_dir = root / "missing"
            result, _ = write_registered_frozen_run(missing_dir)
            result["gate_fit"].pop("gate_enabled")
            (missing_dir / "online_hierarchical_gate_results.json").write_text(
                json.dumps(result), encoding="utf-8"
            )
            with self.assertRaises(RuntimeError):
                load_frozen_gate_run("A0_fixed", missing_dir)

    def test_load_rejects_nonfinite_or_malformed_gate_diagnostics(self) -> None:
        mutations = {
            "missing ratio": lambda gate: gate.pop("motion_energy_ratio"),
            "nan ratio": lambda gate: gate.__setitem__("motion_energy_ratio", np.nan),
            "infinite gap": lambda gate: gate.__setitem__("motion_energy_gap", np.inf),
            "missing medians": lambda gate: gate.pop(
                "residual_child_normalized_motion_component_medians"
            ),
            "wrong median shape": lambda gate: gate.__setitem__(
                "residual_child_normalized_motion_component_medians", [[0.1, 0.2]]
            ),
            "nonfinite medians": lambda gate: gate.__setitem__(
                "residual_child_normalized_motion_component_medians",
                [[0.1, np.nan], [1.5, 2.0]],
            ),
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for index, (name, mutate) in enumerate(mutations.items()):
                with self.subTest(name=name):
                    run_dir = root / str(index)
                    result, _ = write_registered_frozen_run(run_dir)
                    mutate(result["gate_fit"])
                    (run_dir / "online_hierarchical_gate_results.json").write_text(
                        json.dumps(result), encoding="utf-8"
                    )
                    with self.assertRaises(RuntimeError):
                        load_frozen_gate_run("A0_fixed", run_dir)

    def test_disabled_gate_must_reproduce_g0_raw_and_aligned_predictions(self) -> None:
        truth = list(range(10)) * 2
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            valid_dir = root / "valid"
            write_registered_frozen_run(
                valid_dir,
                gate_enabled=False,
                gate_disable_reason=VALID_DISABLE_REASON,
            )
            loaded = load_frozen_gate_run("A0_fixed", valid_dir)
            self.assertIs(loaded.result["gate_fit"]["gate_enabled"], False)

            raw_mismatch = truth.copy()
            raw_mismatch[0] = 9
            raw_dir = root / "raw"
            write_registered_frozen_run(
                raw_dir,
                gate_enabled=False,
                gate_disable_reason=VALID_DISABLE_REASON,
                raw_predictions={"G3_hierarchical": raw_mismatch},
            )
            with self.assertRaisesRegex(RuntimeError, "G0 raw_cluster"):
                load_frozen_gate_run("A0_fixed", raw_dir)

            aligned_mismatch = truth.copy()
            aligned_mismatch[0] = 9
            aligned_dir = root / "aligned"
            write_registered_frozen_run(
                aligned_dir,
                gate_enabled=False,
                gate_disable_reason=VALID_DISABLE_REASON,
                aligned_predictions={"G3_hierarchical": aligned_mismatch},
            )
            with self.assertRaisesRegex(RuntimeError, "G0 aligned_prediction"):
                load_frozen_gate_run("A0_fixed", aligned_dir)


if __name__ == "__main__":
    unittest.main()
