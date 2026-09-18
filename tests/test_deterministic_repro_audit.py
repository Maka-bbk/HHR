from __future__ import annotations

import argparse
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from experiments.motion_primitive import run_deterministic_repro_audit as repro
from experiments.motion_primitive import pretrain_window_encoder, train_motion_encoder
from experiments.motion_primitive import strict_encoder_cv_runner as encoder_cv
from experiments.motion_primitive import window_codebook_batch_proxy
from experiments.motion_primitive.frozen_e0 import (
    DURATION_SOFT_SUBJECT_A025_PROFILE,
    LEGACY_DESCRIPTOR_PROFILE,
)


def _metrics(value: float = 0.5) -> dict[str, float]:
    return {name: float(value) for name in repro.METRICS}


def _repeat_record(repeat: int, *, prediction_sha: str = "5" * 64) -> dict:
    descriptors = {}
    for index, profile in enumerate(repro.PROFILES):
        values = _metrics(0.5 + index * 0.1)
        descriptors[profile] = {
            "prediction_array_sha256": prediction_sha,
            "trajectory_feature_array_sha256": "6" * 64,
            "metric_signature_sha256": repro._metric_signature(values),
            "codebook_state_sha256": "7" * 64,
            "descriptor_transform_state_sha256": str(index + 8) * 64,
            "activity_cluster_state_sha256": "a" * 64,
            "metrics": values,
            "member_dir": f"repeat_{repeat:02d}/{profile}",
        }
    return {
        "repeat": int(repeat),
        "fold": repro.FOLD,
        "seed": repro.SEED,
        "repeat_root": f"repeat_{repeat:02d}",
        "warmup": {
            "selected_epoch": 3,
            "selected_validation_macro_f1": 0.9,
            "history_canonical_sha256": "b" * 64,
            "model_tensor_sha256": "1" * 64,
            "backbone_tensor_sha256": "2" * 64,
        },
        "a2": {
            "selected_epoch": 30,
            "selected_validation_total_loss": 0.4,
            "history_canonical_sha256": "c" * 64,
            "model_tensor_sha256": "3" * 64,
            "ema_teacher_tensor_sha256": "4" * 64,
        },
        "descriptors": descriptors,
    }


def _args(npz: Path, output: Path, reference: Path | None = None) -> argparse.Namespace:
    return repro.build_parser().parse_args(
        [
            "--npz-path",
            str(npz),
            "--output-root",
            str(output),
            *([] if reference is None else ["--reference-summary", str(reference)]),
        ]
    )


class StableContentHashTests(unittest.TestCase):
    def test_canonical_history_hash_ignores_json_formatting(self) -> None:
        rows = [{"epoch": 1, "validation": {"loss": 0.25, "score": 0.75}}]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            json_path = root / "history.json"
            jsonl_path = root / "history.jsonl"
            json_path.write_text(
                json.dumps(rows, indent=4, sort_keys=False), encoding="utf-8"
            )
            jsonl_path.write_text(
                "  " + json.dumps(rows[0], sort_keys=True) + "  \n\n",
                encoding="utf-8",
            )
            self.assertEqual(
                repro._canonical_history_sha256(json_path, json_lines=False),
                repro._canonical_history_sha256(jsonl_path, json_lines=True),
            )

    def test_raw_prediction_hashes_array_content_not_npz_container(self) -> None:
        trial_ids = np.asarray([7, 9], dtype=np.int64)
        features = np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        clusters = np.asarray([1, 0], dtype=np.int64)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = root / "first.npz"
            second = root / "second.npz"
            np.savez(
                first,
                trial_ids=trial_ids,
                trajectory_features=features,
                raw_activity_cluster_ids=clusters,
                unrelated=np.asarray("first"),
            )
            np.savez(
                second,
                unrelated=np.asarray("second"),
                raw_activity_cluster_ids=clusters,
                trajectory_features=features,
                trial_ids=trial_ids,
            )
            self.assertEqual(
                repro._raw_prediction_content(first),
                repro._raw_prediction_content(second),
            )

    def test_consistency_uses_tensor_and_array_content_not_pt_file_hash(self) -> None:
        rows = [_repeat_record(index) for index in range(1, 6)]
        for index, row in enumerate(rows):
            row["warmup"]["checkpoint_file_sha256"] = str(index) * 64
            row["a2"]["checkpoint_file_sha256"] = str(index + 5) * 64
        report = repro._consistency_report(rows)
        self.assertTrue(report["all_checks_equal"])
        self.assertTrue(report["pt_file_sha256_used_for_within_run_integrity_validation"])
        self.assertFalse(report["pt_file_sha256_used_as_cross_repeat_pass_criterion"])
        self.assertTrue(
            all("checkpoint_file" not in name for name in report["checks"])
        )

    def test_prediction_divergence_fails_consistency(self) -> None:
        rows = [_repeat_record(index) for index in range(1, 6)]
        rows[-1]["descriptors"][LEGACY_DESCRIPTOR_PROFILE][
            "prediction_array_sha256"
        ] = "f" * 64
        report = repro._consistency_report(rows)
        self.assertFalse(report["all_checks_equal"])
        check = report["checks"][
            f"descriptors.{LEGACY_DESCRIPTOR_PROFILE}.prediction_array_sha256"
        ]
        self.assertEqual(check["unique_value_count"], 2)

    def test_epoch_history_and_cluster_state_are_pass_criteria(self) -> None:
        mutations = (
            ("warmup", "selected_epoch", 4),
            ("warmup", "selected_validation_macro_f1", 0.91),
            ("warmup", "history_canonical_sha256", "d" * 64),
            ("a2", "selected_epoch", 29),
            ("a2", "selected_validation_total_loss", 0.41),
            ("a2", "history_canonical_sha256", "e" * 64),
        )
        for section, field, replacement in mutations:
            with self.subTest(field=f"{section}.{field}"):
                rows = [_repeat_record(index) for index in range(1, 6)]
                rows[-1][section][field] = replacement
                self.assertFalse(repro._consistency_report(rows)["all_checks_equal"])
        rows = [_repeat_record(index) for index in range(1, 6)]
        rows[-1]["descriptors"][DURATION_SOFT_SUBJECT_A025_PROFILE][
            "activity_cluster_state_sha256"
        ] = "f" * 64
        self.assertFalse(repro._consistency_report(rows)["all_checks_equal"])


class AuditOrchestrationTests(unittest.TestCase):
    def test_manifest_hashes_cover_all_registered_child_dependencies(self) -> None:
        observed = repro._implementation_hashes()
        expected = {
            **encoder_cv._implementation_hashes(),
            **window_codebook_batch_proxy._implementation_hashes(),
            **repro.descriptor_cv._implementation_hashes(),
        }
        self.assertTrue(set(expected) <= set(observed))
        self.assertEqual(
            {name: observed[name] for name in expected},
            expected,
        )
        self.assertIn(
            "experiments/motion_primitive/run_deterministic_repro_audit.py",
            observed,
        )
        self.assertIn("experiments/motion_primitive/strict_artifacts.py", observed)
        self.assertIn("models/resnet1d.py", observed)

    def test_every_generated_child_command_is_accepted_by_target_parser(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            npz = root / "windows.npz"
            npz.write_bytes(b"fixture")
            args = _args(npz, root / "audit")
            encoder_args = repro._encoder_namespace(args)
            window_dir = root / "repeat_01" / "window"
            window_command = encoder_cv._window_command(
                encoder_args, repro.FOLD, repro.SEED, window_dir
            )
            window_parsed = pretrain_window_encoder.build_parser().parse_args(
                window_command[2:]
            )
            self.assertTrue(window_parsed.deterministic)
            a2_command = encoder_cv._a2_command(
                encoder_args,
                repro.FOLD,
                repro.SEED,
                window_dir / "model_best.pt",
                root / "repeat_01" / "a2",
            )
            a2_parsed = train_motion_encoder.build_parser().parse_args(a2_command[2:])
            self.assertTrue(a2_parsed.deterministic)
            descriptor_args = repro._descriptor_namespace(args)
            descriptor_command = repro.descriptor_cv._member_command(
                descriptor_args,
                profile=LEGACY_DESCRIPTOR_PROFILE,
                fold=repro.FOLD,
                seed=repro.SEED,
                checkpoint=root / "repeat_01" / "a2" / "motion_encoder_final.pt",
                output=root / "repeat_01" / "legacy",
                reuse_codebook_run_dir=None,
            )
            parsed = window_codebook_batch_proxy.build_parser().parse_args(
                descriptor_command[2:]
            )
            self.assertEqual(parsed.primitive_num, 64)
            self.assertEqual(parsed.window_size, 128)
            self.assertEqual(parsed.window_stride, 64)

    def test_five_repeat_audit_and_reference_delta_are_separate(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            npz = root / "windows.npz"
            npz.write_bytes(b"fixture")
            reference = root / "reference.json"
            reference.write_text(
                json.dumps(
                    {
                        "descriptor_profile": DURATION_SOFT_SUBJECT_A025_PROFILE,
                        "fold": 1,
                        "seed": 0,
                        **_metrics(0.55),
                    }
                ),
                encoding="utf-8",
            )
            output = root / "audit"

            def fake_repeat(args, *, repeat, root):
                return _repeat_record(repeat)

            with mock.patch.object(repro, "_run_repeat", side_effect=fake_repeat) as child:
                result = repro.run(_args(npz, output, reference))

            self.assertEqual(child.call_count, 5)
            self.assertEqual(
                [call.kwargs["repeat"] for call in child.call_args_list],
                [1, 2, 3, 4, 5],
            )
            self.assertTrue(result["passed"])
            self.assertTrue(result["consistency"]["all_checks_equal"])
            comparison = result["reference_comparison"]
            self.assertFalse(comparison["affects_reproducibility_pass"])
            self.assertTrue(comparison["identity_matches_expected_fold_seed_profile"])
            self.assertAlmostEqual(
                comparison["repeat_deltas"][0]["delta_current_minus_reference"][
                    "all_accuracy"
                ],
                0.05,
            )
            complete = json.loads((output / "complete.json").read_text(encoding="utf-8"))
            self.assertTrue(complete["passed"])
            self.assertTrue((output / "reproducibility_audit.json").is_file())
            self.assertTrue((output / "repeat_results.csv").is_file())

    def test_failed_consistency_writes_machine_readable_audit_and_no_complete(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            npz = root / "windows.npz"
            npz.write_bytes(b"fixture")
            output = root / "audit"

            def fake_repeat(args, *, repeat, root):
                prediction = "f" * 64 if repeat == 5 else "5" * 64
                return _repeat_record(repeat, prediction_sha=prediction)

            with mock.patch.object(repro, "_run_repeat", side_effect=fake_repeat):
                with self.assertRaisesRegex(RuntimeError, "audit failed"):
                    repro.run(_args(npz, output))

            audit = json.loads(
                (output / "reproducibility_audit.json").read_text(encoding="utf-8")
            )
            self.assertFalse(audit["passed"])
            self.assertFalse((output / "complete.json").exists())


if __name__ == "__main__":
    unittest.main()
