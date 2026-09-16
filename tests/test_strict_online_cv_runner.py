from __future__ import annotations

import argparse
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from experiments.motion_primitive import strict_online_cv_runner as cv
from experiments.motion_primitive.strict_artifacts import write_csv, write_json
from experiments.motion_primitive.strict_cv_common import (
    CANONICAL_FOLDS,
    CANONICAL_SEEDS,
    METRICS,
    canonical_hash,
)
from experiments.motion_primitive.strict_offline_runner import PROFILE
from experiments.motion_primitive.strict_online_runner import (
    ONLINE_SCHEMA,
    RUN_IDENTITY_SCHEMA as ONLINE_RUN_IDENTITY_SCHEMA,
    _output_artifacts,
)
from experiments.motion_primitive.strict_protocol import sha256_file


def _args(offline: Path, output: Path, **updates: object) -> argparse.Namespace:
    values = {
        "offline_cv_root": str(offline),
        "output_root": str(output),
        "folds": "1,2,3,4,5,6,7",
        "seeds": "0,5,50,500",
        "encode_batch_size": 512,
        "device": "cpu",
        "allow_smoke_a2": False,
        "bootstrap_seed": 20260912,
        "resume": False,
        "dry_run": False,
    }
    values.update(updates)
    return argparse.Namespace(**values)


def _metric_value(fold: int, seed: int, session: int) -> float:
    seed_index = CANONICAL_SEEDS.index(int(seed))
    return float(fold) / 100.0 + float(seed_index) / 1000.0 + float(session) / 10000.0


def _raw_rows() -> list[dict[str, object]]:
    rows = []
    for fold in CANONICAL_FOLDS:
        for seed in CANONICAL_SEEDS:
            for session in (1, 2, 3):
                value = _metric_value(fold, seed, session)
                rows.append(
                    {
                        "profile": PROFILE,
                        "fold": fold,
                        "seed": seed,
                        "session": session,
                        **{metric: value for metric in METRICS},
                    }
                )
    return rows


def _write_fake_online_member(target: Path, fold: int, seed: int) -> None:
    target.mkdir(parents=True, exist_ok=True)
    representation = "a" * 64
    cv_path = target / "cv_member_manifest.json"
    if cv_path.is_file():
        cv_identity = json.loads(cv_path.read_text(encoding="utf-8"))
    else:
        base_cv_identity = {
            "schema": cv.MEMBER_SCHEMA,
            "grid_identity_sha256": "0" * 64,
            "profile": PROFILE,
            "fold": fold,
            "seed": seed,
            "session_count": 3,
            "offline_run_dir": str((target.parent / "offline").resolve()),
            "offline_manifest_sha256": "1" * 64,
            "offline_complete_sha256": "2" * 64,
            "offline_representation_sha256": representation,
            "offline_old_registry_state_sha256": "4" * 64,
            "encode_batch_size": 512,
            "requested_device": "cpu",
            "resolved_device": "cpu",
            "allow_smoke_a2": False,
        }
        cv_identity = {
            **base_cv_identity,
            "identity_sha256": canonical_hash(base_cv_identity),
        }
        write_json(cv_path, cv_identity)
    representation = str(cv_identity["offline_representation_sha256"])
    run_identity_base = {
        "schema": ONLINE_RUN_IDENTITY_SCHEMA,
        "profile": PROFILE,
        "fold": fold,
        "seed": seed,
        "offline_run_dir": cv_identity["offline_run_dir"],
        "offline_manifest_sha256": cv_identity["offline_manifest_sha256"],
        "offline_complete_sha256": cv_identity["offline_complete_sha256"],
        "offline_representation_sha256": cv_identity["offline_representation_sha256"],
        "offline_old_registry_state_sha256": cv_identity["offline_old_registry_state_sha256"],
        "runtime": {
            "encode_batch_size": cv_identity["encode_batch_size"],
            "requested_device": cv_identity["requested_device"],
            "resolved_device": cv_identity["resolved_device"],
            "allow_smoke_a2": cv_identity["allow_smoke_a2"],
        },
        "protocol": {
            "sessions": [1, 2, 3],
            "online_activity_labels_used_by_learner": False,
            "test_labels_used_only_for_scoring": True,
            "frozen_representation": True,
            "append_only_activity_registry": True,
        },
        "implementation_sha256": {"test": "5" * 64},
    }
    run_identity = {
        **run_identity_base,
        "identity_sha256": canonical_hash(run_identity_base),
    }
    write_json(target / "run_identity.json", run_identity)
    initial_frozen = {
        "representation_sha256": representation,
        "encoder_state_sha256": "b" * 64,
        "codebook_state_sha256": "c" * 64,
        "descriptor_state_sha256": "d" * 64,
        "old_anchor_sha256": "e" * 64,
    }
    session_rows = []
    for session in (1, 2, 3):
        value = _metric_value(fold, seed, session)
        registry_hash = str(session) * 64
        raw_path = target / f"raw_predictions_session_{session}.npz"
        np.savez_compressed(raw_path, prediction=np.asarray([0, -1], dtype=np.int64))
        primary = {
            **{metric: value for metric in METRICS},
            "unknown_prediction_fraction": 0.25,
        }
        metrics = {
            "schema": "hhr_strict_cgcd_metrics_v1",
            "session": session,
            "raw_predictions_path": raw_path.name,
            "raw_predictions_sha256": sha256_file(raw_path),
            "raw_predictions_frozen_before_truth_join": True,
            "representation_sha256": representation,
            "registry_state_sha256": registry_hash,
            "layers": {"old_fixed_novel_hungarian": primary},
        }
        write_json(target / f"metrics_session_{session}.json", metrics)
        write_json(target / f"registry_session_{session}.json", {"state_sha256": registry_hash})
        write_json(target / f"discovery_session_{session}.json", {"session": session})
        write_json(
            target / f"incoming_label_free_manifest_session_{session}.json",
            {"session": session, "activity_labels_present": False},
        )
        write_csv(
            target / f"predictions_session_{session}.csv",
            [{"trial_id": 1, "raw_registry_prediction": 0}],
        )
        session_rows.append(
            {
                "profile": PROFILE,
                "fold": fold,
                "seed": seed,
                "session": session,
                **{metric: value for metric in METRICS},
                "unknown_fraction": 0.25,
                "registry_k": 6 + 2 * session,
                "registered_this_session": 2,
                "unknown_buffer_size": 0,
                "used_primitive_k": 16 + session,
                "representation_sha256": representation,
                "registry_state_sha256": registry_hash,
            }
        )
    (target / "trajectories_label_free.jsonl").write_text("{}\n", encoding="utf-8")
    write_csv(target / "online_runs.csv", session_rows)
    summary = {
        "schema": ONLINE_SCHEMA,
        "profile": PROFILE,
        "fold": fold,
        "seed": seed,
        "session_count": 3,
        "primary_layer": "old_fixed_novel_hungarian",
        "final_session": session_rows[-1],
        "all_sessions": session_rows,
        "frozen_hashes_initial": initial_frozen,
        "frozen_hashes_final": initial_frozen,
        "final_registry_state_sha256": session_rows[-1]["registry_state_sha256"],
        "final_registry_k": session_rows[-1]["registry_k"],
        "unresolved_trial_count": 0,
        "online_activity_labels_used_by_learner": False,
        "test_labels_used_only_for_scoring": True,
    }
    write_json(target / "online_summary.json", summary)
    artifact_hashes = {
        name: sha256_file(target / name)
        for name in _output_artifacts()
    }
    write_json(
        target / "complete.json",
        {
            **summary,
            "run_identity_sha256": run_identity["identity_sha256"],
            "online_summary_sha256": sha256_file(target / "online_summary.json"),
            "artifact_sha256": artifact_hashes,
            "complete": True,
        },
    )


def _fake_offline_members(root: Path) -> tuple[dict[str, object], dict[tuple[int, int], cv.OfflineMember]]:
    root.mkdir(parents=True, exist_ok=True)
    grid = {"identity_sha256": "f" * 64}
    write_json(root / "grid_manifest.json", grid)
    members = {}
    for fold in CANONICAL_FOLDS:
        for seed in CANONICAL_SEEDS:
            members[(fold, seed)] = cv.OfflineMember(
                fold=fold,
                seed=seed,
                path=root / f"offline_{fold}_{seed}",
                manifest_sha256="1" * 64,
                complete_sha256="2" * 64,
                representation_sha256="3" * 64,
                old_registry_state_sha256="4" * 64,
            )
    return grid, members


class CanonicalGridTests(unittest.TestCase):
    def test_formal_grid_is_exactly_seven_by_four_by_three(self) -> None:
        folds, seeds = cv.validate_canonical_grid(CANONICAL_FOLDS, CANONICAL_SEEDS)
        self.assertEqual(len(folds) * len(seeds) * cv.SESSION_COUNT, 84)
        with self.assertRaisesRegex(ValueError, "exactly"):
            cv.validate_canonical_grid((1,), CANONICAL_SEEDS)
        with self.assertRaisesRegex(ValueError, "exactly"):
            cv.validate_canonical_grid(CANONICAL_FOLDS, (0,))

    def test_member_directory_set_rejects_missing_extra_and_malformed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for fold in CANONICAL_FOLDS:
                for seed in CANONICAL_SEEDS:
                    (root / f"fold_{fold:02d}_seed_{seed}").mkdir()
            cv.validate_member_set(root, CANONICAL_FOLDS, CANONICAL_SEEDS, allow_missing=False)
            (root / "fold_08_seed_0").mkdir()
            with self.assertRaisesRegex(RuntimeError, "extra"):
                cv.validate_member_set(root, CANONICAL_FOLDS, CANONICAL_SEEDS, allow_missing=False)
            (root / "fold_08_seed_0").rmdir()
            (root / "fold_bad").mkdir()
            with self.assertRaisesRegex(RuntimeError, "malformed"):
                cv.validate_member_set(root, CANONICAL_FOLDS, CANONICAL_SEEDS, allow_missing=False)

    def test_offline_grid_identity_and_all_28_members_are_verified(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            identity = {
                "schema": cv.OFFLINE_CV_SCHEMA,
                "profile": PROFILE,
                "folds": list(CANONICAL_FOLDS),
                "seeds": list(CANONICAL_SEEDS),
            }
            grid = {**identity, "identity_sha256": canonical_hash(identity)}
            write_json(root / "grid_manifest.json", grid)
            write_json(root / "offline_summary.json", {"member_count": 28})
            write_csv(root / "offline_runs.csv", [{"fold": 1, "seed": 0}])
            for fold in CANONICAL_FOLDS:
                for seed in CANONICAL_SEEDS:
                    member = root / f"fold_{fold:02d}_seed_{seed}"
                    common = {
                        "profile": PROFILE,
                        "fold": fold,
                        "seed": seed,
                        "npz_path": str((root / "data.npz").resolve()),
                        "npz_sha256": "c" * 64,
                        "a2_checkpoint_path": str((root / f"a2_{fold}_{seed}.pt").resolve()),
                        "a2_checkpoint_sha256": "d" * 64,
                    }
                    cv_identity_base = {
                        "schema": cv.OFFLINE_CV_MEMBER_SCHEMA,
                        "grid_identity_sha256": grid["identity_sha256"],
                        **common,
                    }
                    run_identity_base = {
                        "schema": cv.OFFLINE_RUN_IDENTITY_SCHEMA,
                        **common,
                    }
                    write_json(
                        member / "cv_member_manifest.json",
                        {
                            **cv_identity_base,
                            "identity_sha256": canonical_hash(cv_identity_base),
                        },
                    )
                    write_json(
                        member / "run_identity.json",
                        {
                            **run_identity_base,
                            "identity_sha256": canonical_hash(run_identity_base),
                        },
                    )
                    write_json(
                        member / "manifest.json",
                        {"schema": cv.OFFLINE_SCHEMA, "profile": PROFILE, "fold": fold, "seed": seed},
                    )
                    write_json(
                        member / "complete.json",
                        {
                            "schema": cv.OFFLINE_SCHEMA,
                            "profile": PROFILE,
                            "fold": fold,
                            "seed": seed,
                            "manifest_sha256": sha256_file(member / "manifest.json"),
                            "representation_sha256": "a" * 64,
                            "old_registry_state_sha256": "b" * 64,
                            "complete": True,
                        },
                    )
            write_json(
                root / "complete.json",
                {
                    "schema": cv.OFFLINE_CV_SCHEMA,
                    "profile": PROFILE,
                    "member_count": 28,
                    "grid_identity_sha256": grid["identity_sha256"],
                    "grid_manifest_sha256": sha256_file(root / "grid_manifest.json"),
                    "offline_summary_sha256": sha256_file(root / "offline_summary.json"),
                    "offline_runs_sha256": sha256_file(root / "offline_runs.csv"),
                    "complete": True,
                },
            )
            with mock.patch.object(cv, "validate_completed_offline_output"):
                _, members = cv.validate_offline_grid(root)
            self.assertEqual(len(members), 28)

            payload = json.loads((root / "grid_manifest.json").read_text(encoding="utf-8"))
            payload["seeds"] = [0]
            write_json(root / "grid_manifest.json", payload)
            with self.assertRaisesRegex(RuntimeError, "Identity SHA256"):
                with mock.patch.object(cv, "validate_completed_offline_output"):
                    cv.validate_offline_grid(root)


class ResumeIdentityTests(unittest.TestCase):
    def test_partial_member_can_resume_only_with_exact_identity(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            target = Path(temporary) / "fold_01_seed_0"
            identity = {"schema": cv.MEMBER_SCHEMA, "fold": 1, "seed": 0, "setting": "A"}
            cv.ensure_member_identity(target, identity, resume=False)
            (target / "partial.log").write_text("interrupted", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                cv.ensure_member_identity(target, identity, resume=False)
            cv.ensure_member_identity(target, identity, resume=True)
            with self.assertRaisesRegex(RuntimeError, "another configuration"):
                cv.ensure_member_identity(
                    target,
                    {**identity, "setting": "B"},
                    resume=True,
                )

    def test_output_root_refuses_changed_grid_identity(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "output"
            cv._validate_or_create_output_manifest(root, {"schema": cv.CV_SCHEMA, "batch": 512})
            cv._validate_or_create_output_manifest(root, {"schema": cv.CV_SCHEMA, "batch": 512})
            with self.assertRaisesRegex(RuntimeError, "another experiment identity"):
                cv._validate_or_create_output_manifest(root, {"schema": cv.CV_SCHEMA, "batch": 256})


class CompletedMemberTests(unittest.TestCase):
    def test_completed_member_is_recomputed_from_hashed_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            target = Path(temporary) / "member"
            _write_fake_online_member(target, fold=2, seed=5)
            rows = cv.validate_completed_member(target, fold=2, seed=5)
            self.assertEqual(len(rows), 3)
            self.assertEqual([row["session"] for row in rows], [1, 2, 3])
            self.assertTrue(all(row["member_complete_sha256"] for row in rows))

            metric_path = target / "metrics_session_2.json"
            metric = json.loads(metric_path.read_text(encoding="utf-8"))
            metric["layers"]["old_fixed_novel_hungarian"]["h_score"] += 0.1
            write_json(metric_path, metric)
            with self.assertRaisesRegex(RuntimeError, "artifact SHA256 mismatch"):
                cv.validate_completed_member(target, fold=2, seed=5)


class AggregationAndReportTests(unittest.TestCase):
    def test_reports_have_84_rows_and_seed_mean_precedes_fold_summary(self) -> None:
        rows = _raw_rows()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            complete = cv.write_reports(
                root,
                rows,
                folds=CANONICAL_FOLDS,
                seeds=CANONICAL_SEEDS,
                bootstrap_seed=17,
                grid_identity_sha256="a" * 64,
            )
            self.assertEqual(complete["session_row_count"], 84)
            self.assertEqual(complete["member_count"], 28)
            self.assertTrue(complete["complete"])
            for filename in (
                "online_runs.csv",
                "online_summary.json",
                "online_summary.csv",
                "online_summary.md",
                "complete.json",
            ):
                self.assertTrue((root / filename).is_file(), filename)
            metric = complete["aggregate"]["sessions"]["1"]["all_accuracy"]
            expected_first_fold = np.mean(
                [_metric_value(1, seed, 1) for seed in CANONICAL_SEEDS]
            )
            self.assertAlmostEqual(metric["fold_means"][0], expected_first_fold)
            self.assertEqual(metric["fold_count"], 7)
            markdown = (root / "online_summary.md").read_text(encoding="utf-8")
            self.assertIn("fold after averaging four seeds", markdown)
            self.assertIn("84 session rows", markdown)

    def test_missing_or_duplicate_session_row_is_rejected(self) -> None:
        rows = _raw_rows()
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(RuntimeError, "incomplete"):
                cv.write_reports(
                    Path(temporary),
                    rows[:-1],
                    folds=CANONICAL_FOLDS,
                    seeds=CANONICAL_SEEDS,
                    bootstrap_seed=17,
                    grid_identity_sha256="a" * 64,
                )
            with self.assertRaisesRegex(RuntimeError, "duplicate"):
                cv.write_reports(
                    Path(temporary),
                    rows[:-1] + [rows[0]],
                    folds=CANONICAL_FOLDS,
                    seeds=CANONICAL_SEEDS,
                    bootstrap_seed=17,
                    grid_identity_sha256="a" * 64,
                )


class RunnerIntegrationTests(unittest.TestCase):
    def test_run_launches_28_members_resume_launches_zero_and_change_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            offline = base / "offline"
            output = base / "online"
            grid, members = _fake_offline_members(offline)
            launched: list[tuple[int, int]] = []

            def fake_subprocess(command, **_kwargs):
                output_dir = Path(command[command.index("--output-dir") + 1])
                fold = int(command[command.index("--fold") + 1])
                seed = int(command[command.index("--seed") + 1])
                launched.append((fold, seed))
                _write_fake_online_member(output_dir, fold, seed)
                return mock.Mock(returncode=0)

            captured_counts: list[int] = []

            def fake_reports(_root, rows, **_kwargs):
                captured_counts.append(len(rows))
                return {"schema": cv.CV_SCHEMA, "session_row_count": len(rows), "complete": True}

            with mock.patch.object(cv, "validate_offline_grid", return_value=(grid, members)), mock.patch.object(
                cv.subprocess, "run", side_effect=fake_subprocess
            ), mock.patch.object(cv, "write_reports", side_effect=fake_reports):
                with mock.patch("builtins.print"):
                    result = cv.run(_args(offline, output))
                    self.assertEqual(result["session_row_count"], 84)
                    self.assertEqual(len(launched), 28)
                    self.assertEqual(captured_counts, [84])

                    launched.clear()
                    resumed = cv.run(_args(offline, output, resume=True))
                    self.assertEqual(resumed["session_row_count"], 84)
                    self.assertEqual(launched, [])

                    with self.assertRaisesRegex(RuntimeError, "another experiment identity"):
                        cv.run(_args(offline, output, resume=True, encode_batch_size=256))
                    self.assertEqual(launched, [])


if __name__ == "__main__":
    unittest.main()
