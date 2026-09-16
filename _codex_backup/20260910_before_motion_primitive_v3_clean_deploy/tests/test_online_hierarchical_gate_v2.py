from __future__ import annotations

import ast
import inspect
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

import experiments.motion_primitive.analyze_online_hierarchical_gate_v2 as analyzer
import experiments.motion_primitive.run_online_hierarchical_gate_v2 as runner


class OnlineHierarchicalGateV2DiscoveryTests(unittest.TestCase):
    def test_recursive_discovery_ignores_incomplete_quarantine(self) -> None:
        with tempfile.TemporaryDirectory(prefix="online_v2_discovery_") as temporary:
            root = Path(temporary)
            active = root / "fold_01_seed_0" / analyzer.RESULT_FILENAMES[0]
            replaced = (
                root
                / "_incomplete"
                / "fold_01_seed_0.old"
                / analyzer.RESULT_FILENAMES[0]
            )
            similarly_named_active = (
                root
                / "not_incomplete_backup"
                / "fold_02_seed_0"
                / analyzer.RESULT_FILENAMES[0]
            )
            active.parent.mkdir(parents=True)
            replaced.parent.mkdir(parents=True)
            similarly_named_active.parent.mkdir(parents=True)
            active.write_text("{}", encoding="utf-8")
            replaced.write_text("{}", encoding="utf-8")
            similarly_named_active.write_text("{}", encoding="utf-8")

            self.assertEqual(
                analyzer.discover_result_paths([root]),
                sorted(
                    [active.resolve(), similarly_named_active.resolve()],
                    key=lambda item: str(item).lower(),
                ),
            )
            self.assertEqual(
                analyzer.discover_result_paths([replaced]),
                [replaced.resolve()],
            )


class OnlineHierarchicalGateV2RunnerTests(unittest.TestCase):
    def _formal_protocol_fixture(self, fold: int, seed: int = 500):
        canonical = runner._canonical_partition(fold)
        metadata = {
            "uschad_cv_fold": int(fold),
            "motion_encoder_seed": int(seed),
            "seed": int(seed),
            "old_class_count": 6,
            "uschad_train_subjects": canonical["fit"],
            "offline_val_subjects": canonical["validation"],
            "uschad_test_subjects": canonical["evaluation"],
            "uschad_recompute_norm_from_train_subjects": True,
            "smoke_test": False,
            "outer_test_used_during_encoder_training": False,
        }
        arguments = {
            "seed": int(seed),
            "old_class_count": 6,
            "anomaly_policy": "report",
            "allow_split_override": False,
            "allow_unverified_npz_normalization": False,
        }
        config = {
            "checkpoint_type": "motion_primitive_encoder",
            "checkpoint_schema_version": 1,
            "checkpoint_metadata": metadata,
            "arguments": arguments,
        }
        split = {
            "fit_subjects": canonical["fit"],
            "eval_subjects": canonical["evaluation"],
            "old_class_ids_0based": list(range(6)),
            "anomaly_policy": "report",
            "checkpoint_split_override_used": False,
            "checkpoint_split_override_explicitly_allowed": False,
            "unverified_npz_normalization_explicitly_allowed": False,
        }
        return config, split

    def _artifacts(self):
        return SimpleNamespace(
            trial_ids=np.asarray([10, 10, 10, 11, 11], dtype=np.int64),
            starts=np.asarray([0, 4, 8, 0, 5], dtype=np.int64),
            ends=np.asarray([4, 8, 12, 5, 10], dtype=np.int64),
            tokens=np.asarray([2, 7, 3, 7, 4], dtype=np.int64),
            centers=np.zeros((32, 4), dtype=np.float32),
        )

    def test_session2_manifest_propagates_explicit_window_size(self) -> None:
        observed = {}

        def stop_after_loader_arguments(**kwargs):
            observed["window_size"] = kwargs["args"].uschad_window_size
            raise RuntimeError("loader argument captured")

        with patch.object(
            runner,
            "get_uschad_datasets",
            side_effect=stop_after_loader_arguments,
        ):
            with self.assertRaisesRegex(RuntimeError, "loader argument captured"):
                runner.build_session2_manifest_v2(
                    Path("windows.npz"),
                    fold=1,
                    fit_subjects=[1, 2],
                    eval_subjects=[3, 4],
                    validation_subjects=[5, 6],
                    seed=0,
                    window_size_samples=128,
                )

        self.assertEqual(observed["window_size"], 128)

    def test_run_fits_the_shared_coarse_readout_exactly_once(self) -> None:
        tree = ast.parse(inspect.getsource(runner.run))
        calls = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "fit_frozen_coarse_readout"
        ]
        self.assertEqual(len(calls), 1)
        self.assertNotIn("_fit_trial_clusterer", inspect.getsource(runner.run))

    def test_registered_parent_absence_is_audited_k32_fallback(self) -> None:
        durations = {
            trial_id: {
                trial_id % 4: 6.0,
                31: 4.0,
            }
            for trial_id in range(12)
        }
        candidate = runner._select_registered_candidate_or_fallback(
            durations,
            minimum_fraction=0.5,
            minimum_support=10,
        )
        self.assertIs(candidate["eligible"], False)
        self.assertIsNone(candidate["selected_token"])
        self.assertEqual(candidate["maximum_observed_support"], 3)
        self.assertIs(candidate["threshold_was_lowered"], False)
        self.assertEqual(
            candidate["fallback_reason"], runner.REGISTERED_NO_PARENT_REASON
        )

    def test_registered_parent_eligible_path_preserves_selector_rule(self) -> None:
        durations = {
            trial_id: {7: 7.0, 3: 3.0}
            for trial_id in range(12)
        }
        candidate = runner._select_registered_candidate_or_fallback(
            durations,
            minimum_fraction=0.5,
            minimum_support=10,
        )
        self.assertIs(candidate["eligible"], True)
        self.assertEqual(candidate["selected_token"], 7)
        self.assertEqual(candidate["selected_support"], 12)
        self.assertEqual(candidate["status"], "eligible_parent_selected")

    def test_fold06_metadata_remains_compatible(self) -> None:
        config, split = self._formal_protocol_fixture(6)
        identity = runner._validate_v2_protocol_metadata(
            config, split, seed=500
        )
        self.assertEqual(identity["fold"], 6)
        self.assertEqual(identity["eval_subjects"], [4, 5])
        self.assertTrue(identity["canonical_partition_verified"])

    def test_non_fold06_metadata_is_accepted(self) -> None:
        config, split = self._formal_protocol_fixture(1, seed=50)
        identity = runner._validate_v2_protocol_metadata(
            config, split, seed=50, expected_fold=1
        )
        self.assertEqual(identity["fold"], 1)
        self.assertEqual(identity["eval_subjects"], [10, 11])
        self.assertEqual(identity["validation_subjects"], [2, 13])

    def test_fold_seed_and_subject_mismatches_are_rejected(self) -> None:
        config, split = self._formal_protocol_fixture(3, seed=5)
        with self.assertRaisesRegex(RuntimeError, "Explicit --fold"):
            runner._validate_v2_protocol_metadata(
                config, split, seed=5, expected_fold=2
            )
        with self.assertRaisesRegex(RuntimeError, "sampling seed"):
            runner._validate_v2_protocol_metadata(config, split, seed=50)
        split["eval_subjects"] = [4, 5]
        with self.assertRaisesRegex(RuntimeError, "disagree"):
            runner._validate_v2_protocol_metadata(config, split, seed=5)

    def test_only_selected_parent_positions_change(self) -> None:
        artifacts = self._artifacts()
        tokens, _, _ = runner._actual_segment_tokens(
            artifacts, 10, selected_parent=7, state_by_trial={10: 2}
        )
        np.testing.assert_array_equal(tokens, np.asarray([2, 33, 3]))
        audit = runner._trajectory_invariant_audit(
            artifacts,
            trial_ids=[10, 11],
            selected_parent=7,
            state_by_trial={10: 2, 11: 0},
        )
        self.assertEqual(audit["routed_trial_ids"], [10])
        self.assertTrue(audit["fallback_trajectories_exactly_equal"])
        self.assertTrue(audit["only_selected_parent_positions_changed"])

    def test_attribution_rejects_a_fallback_raw_change(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "fallback raw"):
            runner._attribution_audit(
                trial_ids=[10, 11],
                y_true=np.asarray([0, 1]),
                coarse_raw=np.asarray([0, 1]),
                refined_raw=np.asarray([2, 1]),
                coarse_aligned=np.asarray([0, 1]),
                refined_aligned=np.asarray([0, 1]),
                state_by_trial={10: 0, 11: 0},
            )

    def test_alignment_only_change_is_reported_separately(self) -> None:
        audit = runner._attribution_audit(
            trial_ids=[10, 11],
            y_true=np.asarray([0, 1]),
            coarse_raw=np.asarray([0, 1]),
            refined_raw=np.asarray([0, 1]),
            coarse_aligned=np.asarray([0, 1]),
            refined_aligned=np.asarray([1, 1]),
            state_by_trial={10: 0, 11: 0},
        )
        self.assertEqual(audit["fallback_raw_mismatch_count"], 0)
        self.assertEqual(audit["fallback_alignment_only_changed_count"], 1)
        self.assertEqual(audit["fallback_alignment_only_changed_trial_ids"], [10])


class OnlineHierarchicalGateV2AnalyzerPairingTests(unittest.TestCase):
    def _run(self, profile: str) -> analyzer.AuditedRun:
        implementation_fingerprint = {
            name: {"sha256": character * 64}
            for name, character in zip(
                (
                    "runner",
                    "adaptive_codebook",
                    "frozen_readout",
                    "hierarchical_gate_v2",
                    "historical_hierarchical_gate_v1",
                ),
                "12345",
            )
        }
        result = {
            "arguments": {
                "run_dir": f"/{profile}/input",
                "output_dir": f"/{profile}/output",
                "fold": 6,
                "seed": 500,
                "minimum_silhouette": 0.25,
            },
            "session_manifest_audit": {
                "cumulative_train_trial_ids": [100, 101, 102],
            },
            "input_audit": {
                "implementation_fingerprint": implementation_fingerprint,
            },
            "protocol": {
                "name": "motion_primitive_adaptive_codebook_frozen_readout_v2",
                "scope": f"fold06_seed500_{profile}",
                "outer_fold": 6,
                "run_seed": 500,
            },
        }
        return analyzer.AuditedRun(
            directory=Path(f"/{profile}"),
            result_path=Path(f"/{profile}/result.json"),
            result=result,
            fold=6,
            seed=500,
            profile=profile,
            segmentation="fixed_window",
            session=2,
            trial_ids=np.asarray([1, 2], dtype=np.int64),
            subjects=np.asarray([4, 5], dtype=np.int64),
            truth=np.asarray([0, 6], dtype=np.int64),
            actual_arm_names={},
            raw_predictions={},
            aligned_predictions={},
            routed_masks={},
            metrics={},
            arm_audits={},
            intervention_audits={},
        )

    def test_a0_a3_pair_accepts_only_identical_protocol_inputs(self) -> None:
        pairs, audit = analyzer._paired_encoder_runs(
            [self._run("A0"), self._run("A3")]
        )
        self.assertEqual(len(pairs), 1)
        self.assertEqual(audit["pair_count"], 1)

    def test_a0_a3_pair_rejects_cumulative_train_manifest_mismatch(self) -> None:
        a0, a3 = self._run("A0"), self._run("A3")
        a3.result["session_manifest_audit"]["cumulative_train_trial_ids"].pop()
        with self.assertRaisesRegex(RuntimeError, "cumulative online-train"):
            analyzer._paired_encoder_runs([a0, a3])

    def test_a0_a3_pair_rejects_implementation_fingerprint_mismatch(self) -> None:
        a0, a3 = self._run("A0"), self._run("A3")
        a3.result["input_audit"]["implementation_fingerprint"]["runner"][
            "sha256"
        ] = "0" * 64
        with self.assertRaisesRegex(RuntimeError, "implementation or protocol"):
            analyzer._paired_encoder_runs([a0, a3])

    def test_a0_a3_pair_rejects_fitting_argument_mismatch(self) -> None:
        a0, a3 = self._run("A0"), self._run("A3")
        a3.result["arguments"]["minimum_silhouette"] = 0.251
        with self.assertRaisesRegex(RuntimeError, "fitting arguments differ"):
            analyzer._paired_encoder_runs([a0, a3])


if __name__ == "__main__":
    unittest.main()
