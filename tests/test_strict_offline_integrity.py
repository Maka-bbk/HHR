from __future__ import annotations

import argparse
import json
import tempfile
import unittest
from pathlib import Path

from experiments.motion_primitive import strict_offline_cv_runner as cv
from experiments.motion_primitive import strict_offline_runner as single
from experiments.motion_primitive.strict_artifacts import write_json
from experiments.motion_primitive.strict_cv_common import canonical_hash, find_a2_checkpoint
from experiments.motion_primitive.strict_protocol import sha256_file


def _identity_payload() -> dict[str, object]:
    base: dict[str, object] = {
        "schema": single.RUN_IDENTITY_SCHEMA,
        "profile": single.PROFILE,
        "fold": 1,
        "seed": 0,
        "npz_path": "/data/uschad.npz",
        "npz_sha256": "a" * 64,
        "a2_checkpoint_path": "/models/motion_encoder_final.pt",
        "a2_checkpoint_sha256": "b" * 64,
        "route": {"test": True},
        "fixed_parameters": {"primitive_num": 32},
        "registry_config": {"old_distance_alpha": 0.05},
        "runtime": {
            "encode_batch_size": 512,
            "requested_device": "cpu",
            "resolved_device": "cpu",
            "allow_smoke_a2": False,
        },
        "implementation_sha256": {"test": "c" * 64},
    }
    return {**base, "identity_sha256": canonical_hash(base)}


def _write_valid_output(target: Path) -> dict[str, object]:
    target.mkdir(parents=True, exist_ok=True)
    identity = _identity_payload()
    write_json(target / "run_identity.json", identity)
    for name in single._OUTPUT_ARTIFACTS:
        path = target / name
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.suffix == ".json":
            write_json(path, {})
        else:
            path.write_bytes((name + "\n").encode("utf-8"))
    validation_metrics = {"old_accuracy": 0.5, "macro_f1": 0.4}
    outer_metrics = {"old_accuracy": 0.6, "macro_f1": 0.5}
    write_json(target / "metrics_validation.json", {"metrics": validation_metrics})
    write_json(target / "metrics_outer_test.json", {"metrics": outer_metrics})
    artifact_hashes = {
        name: sha256_file(target / name)
        for name in single._OUTPUT_ARTIFACTS
    }
    manifest = {
        "schema": single.OFFLINE_SCHEMA,
        "profile": single.PROFILE,
        "fold": 1,
        "seed": 0,
        "run_identity_sha256": identity["identity_sha256"],
        "npz_sha256": identity["npz_sha256"],
        "a2_encoder": {"checkpoint_sha256": identity["a2_checkpoint_sha256"]},
        "representation_sha256": "d" * 64,
        "old_registry": {"state_sha256": "e" * 64},
        "artifact_sha256": artifact_hashes,
    }
    write_json(target / "manifest.json", manifest)
    complete = {
        "schema": single.OFFLINE_SCHEMA,
        "profile": single.PROFILE,
        "fold": 1,
        "seed": 0,
        "run_identity_sha256": identity["identity_sha256"],
        "manifest_sha256": sha256_file(target / "manifest.json"),
        "artifact_sha256": artifact_hashes,
        "representation_sha256": manifest["representation_sha256"],
        "old_registry_state_sha256": manifest["old_registry"]["state_sha256"],
        "validation": validation_metrics,
        "outer_test": outer_metrics,
        "complete": True,
    }
    write_json(target / "complete.json", complete)
    return identity


class FinalCheckpointResolutionTests(unittest.TestCase):
    def test_strict_resolver_never_falls_back_to_validation_best(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            member = root / "fold_01_seed_0"
            member.mkdir()
            best = member / "motion_encoder_best.pt"
            best.touch()
            with self.assertRaisesRegex(RuntimeError, "requires motion_encoder_final"):
                find_a2_checkpoint(root, 1, 0)
            final = member / "motion_encoder_final.pt"
            final.touch()
            self.assertEqual(find_a2_checkpoint(root, 1, 0), final)


class SingleMemberIntegrityTests(unittest.TestCase):
    def test_every_recorded_artifact_is_rehashed_before_resume(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            target = Path(temporary)
            identity = _write_valid_output(target)
            complete = single.validate_completed_output(target, expected_identity=identity)
            self.assertTrue(complete["complete"])
            (target / "trajectories_label_free.jsonl").write_text("tampered\n", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "artifact SHA256 mismatch"):
                single.validate_completed_output(target, expected_identity=identity)

    def test_identity_change_is_rejected_before_result_reuse(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            target = Path(temporary)
            identity = _write_valid_output(target)
            changed = dict(identity)
            changed["seed"] = 5
            with self.assertRaisesRegex(RuntimeError, "identity changed"):
                single.validate_completed_output(target, expected_identity=changed)


class OfflineCVMemberTests(unittest.TestCase):
    def test_cv_member_binds_grid_data_checkpoint_registry_and_runtime(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            target = Path(temporary)
            run_identity = _write_valid_output(target)
            member_identity = {
                "schema": cv.MEMBER_SCHEMA,
                "grid_identity_sha256": "f" * 64,
                "profile": single.PROFILE,
                "fold": 1,
                "seed": 0,
                "npz_path": run_identity["npz_path"],
                "npz_sha256": run_identity["npz_sha256"],
                "a2_checkpoint_path": run_identity["a2_checkpoint_path"],
                "a2_checkpoint_sha256": run_identity["a2_checkpoint_sha256"],
                "registry": run_identity["registry_config"],
                "encode_batch_size": 512,
                "requested_device": "cpu",
                "resolved_device": "cpu",
                "allow_smoke_a2": False,
            }
            write_json(
                target / "cv_member_manifest.json",
                {
                    **member_identity,
                    "identity_sha256": canonical_hash(member_identity),
                },
            )
            complete = cv._validate_member(
                target,
                1,
                0,
                expected_cv_identity=member_identity,
            )
            self.assertTrue(complete["complete"])
            changed = {**member_identity, "registry": {"old_distance_alpha": 0.10}}
            with self.assertRaisesRegex(RuntimeError, "CV identity changed"):
                cv._validate_member(target, 1, 0, expected_cv_identity=changed)

    def test_partial_member_needs_exact_durable_identity(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            target = Path(temporary) / "fold_01_seed_0"
            identity = {"schema": cv.MEMBER_SCHEMA, "fold": 1, "seed": 0}
            cv._ensure_member_identity(target, identity, resume=False)
            (target / "partial.log").write_text("interrupted", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                cv._ensure_member_identity(target, identity, resume=False)
            cv._ensure_member_identity(target, identity, resume=True)
            with self.assertRaisesRegex(RuntimeError, "another identity"):
                cv._ensure_member_identity(
                    target,
                    {**identity, "seed": 5},
                    resume=True,
                )


if __name__ == "__main__":
    unittest.main()
