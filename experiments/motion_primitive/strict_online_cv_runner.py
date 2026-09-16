"""Canonical 7-fold x 4-seed x 3-session strict Online CGCD runner.

The runner treats every completed single-member directory as an immutable
artifact.  ``--resume`` may reuse a verified member or restart an interrupted
member only when its pre-written identity manifest exactly matches the current
grid.  It never silently adopts an unidentifiable directory or overwrites a
corrupt complete marker.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.motion_primitive.strict_artifacts import write_csv, write_json
from experiments.motion_primitive.strict_cv_common import (
    CANONICAL_FOLDS,
    CANONICAL_SEEDS,
    METRICS,
    aggregate_online_rows,
    canonical_hash,
    member_directory,
    parse_integer_grid,
)
from experiments.motion_primitive.strict_offline_cv_runner import (
    CV_SCHEMA as OFFLINE_CV_SCHEMA,
    MEMBER_SCHEMA as OFFLINE_CV_MEMBER_SCHEMA,
)
from experiments.motion_primitive.strict_offline_runner import (
    OFFLINE_SCHEMA,
    PROFILE,
    RUN_IDENTITY_SCHEMA as OFFLINE_RUN_IDENTITY_SCHEMA,
    validate_completed_output as validate_completed_offline_output,
)
from experiments.motion_primitive.strict_online_runner import (
    ONLINE_SCHEMA,
    RUN_IDENTITY_SCHEMA as ONLINE_RUN_IDENTITY_SCHEMA,
    validate_completed_output as validate_completed_online_output,
)
from experiments.motion_primitive.strict_protocol import sha256_file

import torch


CV_SCHEMA = "hhr_frozen_a2_e0_state_online_cv_v1"
MEMBER_SCHEMA = "hhr_frozen_a2_e0_state_online_cv_member_v1"
SESSION_COUNT = 3
EXPECTED_MEMBER_COUNT = 28
EXPECTED_ROW_COUNT = 84
BOOTSTRAP_REPLICATES = 10_000
_MEMBER_PATTERN = re.compile(r"^fold_(\d{2})_seed_(\d+)$")


def _implementation_hashes() -> dict[str, str]:
    """Fingerprint every project source that can change online predictions."""

    names = (
        "experiments/motion_primitive/strict_online_cv_runner.py",
        "experiments/motion_primitive/strict_online_runner.py",
        "experiments/motion_primitive/strict_offline_cv_runner.py",
        "experiments/motion_primitive/strict_offline_runner.py",
        "experiments/motion_primitive/strict_artifacts.py",
        "experiments/motion_primitive/strict_cv_common.py",
        "experiments/motion_primitive/strict_protocol.py",
        "experiments/motion_primitive/strict_registry.py",
        "experiments/motion_primitive/strict_metrics.py",
        "experiments/motion_primitive/frozen_e0_state.py",
        "experiments/motion_primitive/motion_checkpoint.py",
        "experiments/motion_primitive/motion_encoder.py",
        "models/resnet1d.py",
    )
    return {name: sha256_file(PROJECT_ROOT / name) for name in names}


@dataclass(frozen=True)
class OfflineMember:
    fold: int
    seed: int
    path: Path
    manifest_sha256: str
    complete_sha256: str
    representation_sha256: str
    old_registry_state_sha256: str


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"Cannot read verified JSON artifact {path}.") from error
    if not isinstance(payload, dict):
        raise RuntimeError(f"JSON artifact {path} is not an object.")
    return payload


def _verified_identity_payload(path: Path) -> dict[str, Any]:
    payload = _read_json(path)
    recorded = payload.get("identity_sha256")
    identity = {key: value for key, value in payload.items() if key != "identity_sha256"}
    if not isinstance(recorded, str) or canonical_hash(identity) != recorded:
        raise RuntimeError(f"Identity SHA256 does not verify in {path}.")
    return payload


def validate_canonical_grid(
    folds: Sequence[int], seeds: Sequence[int]
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Reject reduced grids: the formal runner always represents 84 sessions."""

    fold_grid = tuple(sorted(int(value) for value in folds))
    seed_grid = tuple(sorted(int(value) for value in seeds))
    if fold_grid != CANONICAL_FOLDS or seed_grid != CANONICAL_SEEDS:
        raise ValueError(
            "Strict Online CV requires exactly folds=1..7 and seeds=0,5,50,500; "
            "use the single-member runner for smoke checks."
        )
    return fold_grid, seed_grid


def _resolved_device(value: str) -> str:
    requested = str(value).lower()
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested for Online CV but is unavailable.")
    return str(device)


def _member_keys_below(root: Path) -> tuple[set[tuple[int, int]], list[str]]:
    keys: set[tuple[int, int]] = set()
    malformed: list[str] = []
    if not root.is_dir():
        return keys, malformed
    for child in root.iterdir():
        if not child.is_dir() or not child.name.startswith("fold_"):
            continue
        matched = _MEMBER_PATTERN.fullmatch(child.name)
        if matched is None:
            malformed.append(child.name)
            continue
        key = (int(matched.group(1)), int(matched.group(2)))
        if key in keys:
            # This is mostly defensive on case-sensitive filesystems; the
            # directory spelling is canonical and duplicates are forbidden.
            malformed.append(child.name)
        keys.add(key)
    return keys, sorted(malformed)


def validate_member_set(
    root: Path,
    folds: Sequence[int],
    seeds: Sequence[int],
    *,
    allow_missing: bool,
) -> None:
    expected = {(int(fold), int(seed)) for fold in folds for seed in seeds}
    observed, malformed = _member_keys_below(root)
    extra = sorted(observed - expected)
    missing = sorted(expected - observed)
    if malformed or extra or (missing and not allow_missing):
        raise RuntimeError(
            "Online member directory set is ambiguous: "
            f"malformed={malformed}, missing={missing}, extra={extra}."
        )


def validate_offline_grid(
    root: str | Path,
    folds: Sequence[int] = CANONICAL_FOLDS,
    seeds: Sequence[int] = CANONICAL_SEEDS,
) -> tuple[dict[str, Any], dict[tuple[int, int], OfflineMember]]:
    base = Path(root).expanduser().resolve()
    manifest_path = base / "grid_manifest.json"
    if not manifest_path.is_file():
        raise RuntimeError(f"Offline CV grid manifest is missing: {manifest_path}.")
    grid = _verified_identity_payload(manifest_path)
    if grid.get("schema") != OFFLINE_CV_SCHEMA or grid.get("profile") != PROFILE:
        raise RuntimeError("Offline CV grid belongs to another schema/profile.")
    if tuple(grid.get("folds", ())) != tuple(folds) or tuple(grid.get("seeds", ())) != tuple(seeds):
        raise RuntimeError("Offline CV member grid differs from the canonical online grid.")
    root_complete_path = base / "complete.json"
    root_summary_path = base / "offline_summary.json"
    root_runs_path = base / "offline_runs.csv"
    for required in (root_complete_path, root_summary_path, root_runs_path):
        if not required.is_file():
            raise RuntimeError(f"Offline CV root is incomplete: {required}.")
    root_complete = _read_json(root_complete_path)
    if root_complete.get("schema") != OFFLINE_CV_SCHEMA:
        raise RuntimeError("Offline CV completion belongs to another schema.")
    if root_complete.get("profile") != PROFILE or root_complete.get("complete") is not True:
        raise RuntimeError("Offline CV completion has another profile/status.")
    if int(root_complete.get("member_count", -1)) != EXPECTED_MEMBER_COUNT:
        raise RuntimeError("Offline CV completion does not contain exactly 28 members.")
    if root_complete.get("grid_identity_sha256") != grid.get("identity_sha256"):
        raise RuntimeError("Offline CV completion is not bound to its grid identity.")
    root_hashes = {
        "grid_manifest_sha256": sha256_file(manifest_path),
        "offline_summary_sha256": sha256_file(root_summary_path),
        "offline_runs_sha256": sha256_file(root_runs_path),
    }
    for key, expected in root_hashes.items():
        if root_complete.get(key) != expected:
            raise RuntimeError(f"Offline CV root {key} mismatch.")
    validate_member_set(base, folds, seeds, allow_missing=False)

    members: dict[tuple[int, int], OfflineMember] = {}
    for fold in folds:
        for seed in seeds:
            path = member_directory(base, fold, seed)
            cv_identity_path = path / "cv_member_manifest.json"
            run_identity_path = path / "run_identity.json"
            manifest = path / "manifest.json"
            complete_path = path / "complete.json"
            if not all(
                item.is_file()
                for item in (cv_identity_path, run_identity_path, manifest, complete_path)
            ):
                raise RuntimeError(f"Offline member is incomplete: {path}.")
            cv_identity = _verified_identity_payload(cv_identity_path)
            run_identity = _verified_identity_payload(run_identity_path)
            if cv_identity.get("schema") != OFFLINE_CV_MEMBER_SCHEMA:
                raise RuntimeError(f"Offline member has another CV identity schema: {path}.")
            if run_identity.get("schema") != OFFLINE_RUN_IDENTITY_SCHEMA:
                raise RuntimeError(f"Offline member has another run identity schema: {path}.")
            if cv_identity.get("grid_identity_sha256") != grid.get("identity_sha256"):
                raise RuntimeError(f"Offline member is not bound to this grid: {path}.")
            for identity in (cv_identity, run_identity):
                if identity.get("profile") != PROFILE:
                    raise RuntimeError(f"Offline member identity has another profile: {path}.")
                if (int(identity.get("fold", -1)), int(identity.get("seed", -1))) != (
                    int(fold), int(seed)
                ):
                    raise RuntimeError(f"Offline member identity fold/seed mismatch: {path}.")
            for key in (
                "npz_path",
                "npz_sha256",
                "a2_checkpoint_path",
                "a2_checkpoint_sha256",
            ):
                if cv_identity.get(key) != run_identity.get(key):
                    raise RuntimeError(
                        f"Offline CV/run identity disagreement for {key!r}: {path}."
                    )
            validate_completed_offline_output(path, expected_identity=run_identity)
            complete = _read_json(complete_path)
            if complete.get("schema") != OFFLINE_SCHEMA or complete.get("profile") != PROFILE:
                raise RuntimeError(f"Offline member has another schema/profile: {path}.")
            if complete.get("complete") is not True:
                raise RuntimeError(f"Offline member complete marker is false: {path}.")
            if (int(complete.get("fold", -1)), int(complete.get("seed", -1))) != (
                int(fold), int(seed)
            ):
                raise RuntimeError(f"Offline member fold/seed mismatch: {path}.")
            manifest_hash = sha256_file(manifest)
            if complete.get("manifest_sha256") != manifest_hash:
                raise RuntimeError(f"Offline member manifest SHA256 mismatch: {path}.")
            representation = str(complete.get("representation_sha256", ""))
            registry = str(complete.get("old_registry_state_sha256", ""))
            if len(representation) != 64 or len(registry) != 64:
                raise RuntimeError(f"Offline member lacks frozen-state SHA256 values: {path}.")
            members[(int(fold), int(seed))] = OfflineMember(
                fold=int(fold),
                seed=int(seed),
                path=path,
                manifest_sha256=manifest_hash,
                complete_sha256=sha256_file(complete_path),
                representation_sha256=representation,
                old_registry_state_sha256=registry,
            )
    if len(members) != EXPECTED_MEMBER_COUNT:
        raise RuntimeError("Offline CV validation did not resolve exactly 28 members.")
    return grid, members


def _grid_identity(
    args: argparse.Namespace,
    *,
    offline_root: Path,
    offline_grid_manifest: Mapping[str, Any],
    folds: Sequence[int],
    seeds: Sequence[int],
) -> dict[str, Any]:
    return {
        "schema": CV_SCHEMA,
        "profile": PROFILE,
        "offline_cv_root": str(offline_root),
        "offline_grid_manifest_sha256": sha256_file(offline_root / "grid_manifest.json"),
        "offline_grid_identity_sha256": offline_grid_manifest["identity_sha256"],
        "folds": list(folds),
        "seeds": list(seeds),
        "sessions": [1, 2, 3],
        "expected_member_count": EXPECTED_MEMBER_COUNT,
        "expected_session_row_count": EXPECTED_ROW_COUNT,
        "aggregation_unit": "fold_after_averaging_seeds_within_fold",
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "bootstrap_seed": int(args.bootstrap_seed),
        "online_config": {
            "encode_batch_size": int(args.encode_batch_size),
            "requested_device": str(args.device),
            "resolved_device": _resolved_device(args.device),
            "allow_smoke_a2": bool(args.allow_smoke_a2),
            "learner_activity_labels": False,
            "encoder_codebook_descriptor_frozen": True,
        },
        "implementation_sha256": _implementation_hashes(),
    }


def _validate_or_create_output_manifest(root: Path, identity: Mapping[str, Any]) -> None:
    path = root / "grid_manifest.json"
    expected = {**dict(identity), "identity_sha256": canonical_hash(identity)}
    if root.exists() and any(root.iterdir()) and not path.is_file():
        raise RuntimeError(
            f"Non-empty output root {root} has no grid identity; refusing adoption."
        )
    if path.is_file():
        observed = _verified_identity_payload(path)
        if observed != expected:
            raise RuntimeError(
                f"Output root {root} records another experiment identity; use a new directory."
            )
        return
    root.mkdir(parents=True, exist_ok=True)
    write_json(path, expected)


def _member_identity(
    *,
    grid_identity_sha256: str,
    offline: OfflineMember,
    args: argparse.Namespace,
) -> dict[str, Any]:
    return {
        "schema": MEMBER_SCHEMA,
        "grid_identity_sha256": str(grid_identity_sha256),
        "profile": PROFILE,
        "fold": int(offline.fold),
        "seed": int(offline.seed),
        "session_count": SESSION_COUNT,
        "offline_run_dir": str(offline.path.resolve()),
        "offline_manifest_sha256": offline.manifest_sha256,
        "offline_complete_sha256": offline.complete_sha256,
        "offline_representation_sha256": offline.representation_sha256,
        "offline_old_registry_state_sha256": offline.old_registry_state_sha256,
        "encode_batch_size": int(args.encode_batch_size),
        "requested_device": str(args.device),
        "resolved_device": _resolved_device(args.device),
        "allow_smoke_a2": bool(args.allow_smoke_a2),
    }


def ensure_member_identity(
    target: Path,
    identity: Mapping[str, Any],
    *,
    resume: bool,
) -> None:
    """Create/verify the durable identity that makes partial resume safe."""

    path = target / "cv_member_manifest.json"
    expected = {**dict(identity), "identity_sha256": canonical_hash(identity)}
    if target.exists() and any(target.iterdir()) and not path.is_file():
        raise RuntimeError(f"Existing member has no CV identity and cannot be resumed: {target}.")
    if path.is_file():
        observed = _verified_identity_payload(path)
        if observed != expected:
            raise RuntimeError(f"Member directory contains another configuration: {target}.")
    else:
        target.mkdir(parents=True, exist_ok=True)
        write_json(path, expected)
    other_artifacts = [item for item in target.iterdir() if item.name != path.name]
    if other_artifacts and not resume:
        raise FileExistsError(
            f"Member already contains run artifacts: {target}; use --resume or a new output root."
        )


def _member_command(
    args: argparse.Namespace,
    offline: OfflineMember,
    output: Path,
) -> list[str]:
    # The CV identity manifest is created before execution.  Consequently the
    # child always receives --resume so that an identified directory is valid
    # both on its first run and after an interruption.
    command = [
        sys.executable,
        str(PROJECT_ROOT / "experiments" / "motion_primitive" / "strict_online_runner.py"),
        "--offline-run-dir", str(offline.path),
        "--output-dir", str(output),
        "--profile", PROFILE,
        "--fold", str(int(offline.fold)),
        "--seed", str(int(offline.seed)),
        "--encode-batch-size", str(int(args.encode_batch_size)),
        "--device", str(args.device),
        "--resume",
    ]
    if bool(args.allow_smoke_a2):
        command.append("--allow-smoke-a2")
    return command


def _equal_number(left: Any, right: Any) -> bool:
    try:
        return float(left) == float(right)
    except (TypeError, ValueError):
        return False


def validate_completed_member(
    target: Path,
    *,
    fold: int,
    seed: int,
) -> list[dict[str, Any]]:
    cv_identity_path = target / "cv_member_manifest.json"
    run_identity_path = target / "run_identity.json"
    complete_path = target / "complete.json"
    summary_path = target / "online_summary.json"
    runs_path = target / "online_runs.csv"
    if not complete_path.is_file():
        raise FileNotFoundError(f"Online member has no complete marker: {target}.")
    required = [
        cv_identity_path,
        run_identity_path,
        summary_path,
        runs_path,
        target / "trajectories_label_free.jsonl",
    ]
    for session in range(1, SESSION_COUNT + 1):
        required.extend(
            [
                target / f"metrics_session_{session}.json",
                target / f"raw_predictions_session_{session}.npz",
                target / f"registry_session_{session}.json",
                target / f"discovery_session_{session}.json",
            ]
        )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise RuntimeError(f"Completed online member lacks artifacts: {missing}.")

    cv_identity = _verified_identity_payload(cv_identity_path)
    run_identity = _verified_identity_payload(run_identity_path)
    if cv_identity.get("schema") != MEMBER_SCHEMA:
        raise RuntimeError(f"Online member has another CV member schema: {target}.")
    if run_identity.get("schema") != ONLINE_RUN_IDENTITY_SCHEMA:
        raise RuntimeError(f"Online member has another run identity schema: {target}.")
    for identity in (cv_identity, run_identity):
        if identity.get("profile") != PROFILE:
            raise RuntimeError(f"Online member identity has another profile: {target}.")
        if (int(identity.get("fold", -1)), int(identity.get("seed", -1))) != (
            int(fold), int(seed)
        ):
            raise RuntimeError(f"Online member identity fold/seed mismatch: {target}.")
    identity_pairs = (
        ("offline_run_dir", "offline_run_dir"),
        ("offline_manifest_sha256", "offline_manifest_sha256"),
        ("offline_complete_sha256", "offline_complete_sha256"),
        ("offline_representation_sha256", "offline_representation_sha256"),
        ("offline_old_registry_state_sha256", "offline_old_registry_state_sha256"),
    )
    for cv_key, run_key in identity_pairs:
        if cv_identity.get(cv_key) != run_identity.get(run_key):
            raise RuntimeError(
                f"Online CV/run identity disagreement for {cv_key!r}: {target}."
            )
    runtime = run_identity.get("runtime")
    if not isinstance(runtime, Mapping):
        raise RuntimeError(f"Online member run identity lacks runtime: {target}.")
    for key in ("encode_batch_size", "requested_device", "resolved_device", "allow_smoke_a2"):
        if runtime.get(key) != cv_identity.get(key):
            raise RuntimeError(
                f"Online CV/run runtime disagreement for {key!r}: {target}."
            )
    validate_completed_online_output(target, expected_identity=run_identity)

    complete = _read_json(complete_path)
    summary = _read_json(summary_path)
    if complete.get("schema") != ONLINE_SCHEMA or summary.get("schema") != ONLINE_SCHEMA:
        raise RuntimeError(f"Online member has another schema: {target}.")
    if complete.get("profile") != PROFILE or summary.get("profile") != PROFILE:
        raise RuntimeError(f"Online member has another profile: {target}.")
    if complete.get("complete") is not True:
        raise RuntimeError(f"Online member complete marker is false: {target}.")
    if (int(complete.get("fold", -1)), int(complete.get("seed", -1))) != (int(fold), int(seed)):
        raise RuntimeError(f"Online member fold/seed mismatch: {target}.")
    if int(complete.get("session_count", -1)) != SESSION_COUNT:
        raise RuntimeError(f"Online member does not contain three sessions: {target}.")
    if complete.get("online_summary_sha256") != sha256_file(summary_path):
        raise RuntimeError(f"Online summary SHA256 mismatch: {target}.")
    for key, value in summary.items():
        if complete.get(key) != value:
            raise RuntimeError(f"Online complete/summary disagreement for {key!r}: {target}.")
    if complete.get("online_activity_labels_used_by_learner") is not False:
        raise RuntimeError("A strict online member reports learner access to activity labels.")
    if complete.get("test_labels_used_only_for_scoring") is not True:
        raise RuntimeError("A strict online member does not isolate test labels to scoring.")
    if complete.get("frozen_hashes_initial") != complete.get("frozen_hashes_final"):
        raise RuntimeError("Frozen component hashes changed inside an online member.")

    session_rows = complete.get("all_sessions")
    if not isinstance(session_rows, list) or len(session_rows) != SESSION_COUNT:
        raise RuntimeError(f"Online summary has an invalid session row list: {target}.")
    if [int(row.get("session", -1)) for row in session_rows] != [1, 2, 3]:
        raise RuntimeError(f"Online summary session keys are not exactly 1,2,3: {target}.")

    with runs_path.open("r", encoding="utf-8", newline="") as handle:
        csv_rows = list(csv.DictReader(handle))
    if len(csv_rows) != SESSION_COUNT:
        raise RuntimeError(f"Online member CSV does not contain three rows: {target}.")

    rows: list[dict[str, Any]] = []
    for session, source_row, csv_row in zip(range(1, 4), session_rows, csv_rows):
        metrics_path = target / f"metrics_session_{session}.json"
        metrics = _read_json(metrics_path)
        if metrics.get("schema") != "hhr_strict_cgcd_metrics_v1":
            raise RuntimeError(f"Unexpected strict metric schema in {metrics_path}.")
        if int(metrics.get("session", -1)) != session:
            raise RuntimeError(f"Metric session mismatch in {metrics_path}.")
        if metrics.get("raw_predictions_frozen_before_truth_join") is not True:
            raise RuntimeError(f"Predictions were not frozen before scoring in {metrics_path}.")
        raw_path = target / str(metrics.get("raw_predictions_path", ""))
        if raw_path != target / f"raw_predictions_session_{session}.npz":
            raise RuntimeError(f"Raw prediction path mismatch in {metrics_path}.")
        if metrics.get("raw_predictions_sha256") != sha256_file(raw_path):
            raise RuntimeError(f"Raw prediction SHA256 mismatch in {metrics_path}.")
        layers = metrics.get("layers")
        if not isinstance(layers, Mapping) or "old_fixed_novel_hungarian" not in layers:
            raise RuntimeError(f"Primary metric layer is missing in {metrics_path}.")
        primary = layers["old_fixed_novel_hungarian"]
        if not isinstance(primary, Mapping):
            raise RuntimeError(f"Primary metric layer is malformed in {metrics_path}.")
        for metric in METRICS:
            if not _equal_number(source_row.get(metric), primary.get(metric)):
                raise RuntimeError(
                    f"Summary/metric disagreement for {metric}, session={session}: {target}."
                )
        for field in ("profile", "fold", "seed", "session"):
            expected = source_row.get(field)
            observed: Any = csv_row.get(field)
            if field != "profile":
                try:
                    observed = int(observed)
                except (TypeError, ValueError) as error:
                    raise RuntimeError(f"Invalid {field} in {runs_path}.") from error
            if observed != expected:
                raise RuntimeError(f"CSV/summary disagreement for {field}: {target}.")
        for metric in METRICS:
            if not _equal_number(csv_row.get(metric), source_row.get(metric)):
                raise RuntimeError(f"CSV/summary disagreement for {metric}: {target}.")
        if str(source_row.get("representation_sha256")) != str(metrics.get("representation_sha256")):
            raise RuntimeError(f"Representation SHA256 disagreement: {target}, session={session}.")
        if str(source_row.get("registry_state_sha256")) != str(metrics.get("registry_state_sha256")):
            raise RuntimeError(f"Registry SHA256 disagreement: {target}, session={session}.")
        rows.append(
            {
                "profile": PROFILE,
                "fold": int(fold),
                "seed": int(seed),
                "session": int(session),
                **{metric: float(primary[metric]) for metric in METRICS},
                "unknown_fraction": float(primary["unknown_prediction_fraction"]),
                "registry_k": int(source_row["registry_k"]),
                "registered_this_session": int(source_row["registered_this_session"]),
                "unknown_buffer_size": int(source_row["unknown_buffer_size"]),
                "used_primitive_k": int(source_row["used_primitive_k"]),
                "representation_sha256": str(source_row["representation_sha256"]),
                "registry_state_sha256": str(source_row["registry_state_sha256"]),
                "member_complete_sha256": sha256_file(complete_path),
                "metrics_sha256": sha256_file(metrics_path),
            }
        )
    return rows


def _summary_rows(aggregate: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = []
    sessions = aggregate.get("sessions", {})
    for session in range(1, SESSION_COUNT + 1):
        for metric in METRICS:
            values = sessions[str(session)][metric]
            rows.append(
                {
                    "session": session,
                    "metric": metric,
                    "mean": values["mean"],
                    "std_across_folds": values["std_across_folds"],
                    "ci95_low": values["ci95_low"],
                    "ci95_high": values["ci95_high"],
                    "fold_count": values["fold_count"],
                    "fold_means": values["fold_means"],
                }
            )
    return rows


def _write_markdown(path: Path, aggregate: Mapping[str, Any]) -> None:
    lines = [
        "# Strict Online CGCD CV Summary",
        "",
        "Statistical unit: each held-out-subject fold after averaging four seeds within that fold.",
        "",
        "| Session | Metric | Mean | SD across folds | 95% bootstrap CI |",
        "|---:|---|---:|---:|---:|",
    ]
    for row in _summary_rows(aggregate):
        lines.append(
            f"| {row['session']} | {row['metric']} | {float(row['mean']):.6f} | "
            f"{float(row['std_across_folds']):.6f} | "
            f"[{float(row['ci95_low']):.6f}, {float(row['ci95_high']):.6f}] |"
        )
    lines.extend(
        [
            "",
            f"Complete grid: {EXPECTED_MEMBER_COUNT} fold/seed members and {EXPECTED_ROW_COUNT} session rows.",
            "Primary score: old-fixed, novel-only Hungarian alignment; global alignment is not aggregated here.",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text("\n".join(lines), encoding="utf-8")
    temporary.replace(path)


def write_reports(
    root: Path,
    rows: Sequence[Mapping[str, Any]],
    *,
    folds: Sequence[int],
    seeds: Sequence[int],
    bootstrap_seed: int,
    grid_identity_sha256: str,
) -> dict[str, Any]:
    aggregate = aggregate_online_rows(
        rows,
        folds=folds,
        seeds=seeds,
        bootstrap_seed=int(bootstrap_seed),
    )
    if int(aggregate["row_count"]) != EXPECTED_ROW_COUNT:
        raise RuntimeError("Strict Online CV must aggregate exactly 84 session rows.")
    ordered = sorted(
        (dict(row) for row in rows),
        key=lambda row: (int(row["fold"]), int(row["seed"]), int(row["session"])),
    )
    write_csv(root / "online_runs.csv", ordered)
    summary = {
        "schema": CV_SCHEMA,
        "profile": PROFILE,
        "grid_identity_sha256": str(grid_identity_sha256),
        "folds": list(folds),
        "seeds": list(seeds),
        "sessions": [1, 2, 3],
        "member_count": EXPECTED_MEMBER_COUNT,
        "session_row_count": EXPECTED_ROW_COUNT,
        "primary_layer": "old_fixed_novel_hungarian",
        "statistical_unit": "held-out subject fold after averaging seeds within fold",
        "aggregate": aggregate,
    }
    write_json(root / "online_summary.json", summary)
    write_csv(root / "online_summary.csv", _summary_rows(aggregate))
    _write_markdown(root / "online_summary.md", aggregate)
    report_hashes = {
        filename: sha256_file(root / filename)
        for filename in (
            "online_runs.csv",
            "online_summary.json",
            "online_summary.csv",
            "online_summary.md",
        )
    }
    complete = {**summary, "report_sha256": report_hashes, "complete": True}
    write_json(root / "complete.json", complete)
    return complete


def validate_args(args: argparse.Namespace) -> argparse.Namespace:
    if int(args.encode_batch_size) < 1:
        raise ValueError("encode-batch-size must be positive.")
    if int(args.bootstrap_seed) < 0:
        raise ValueError("bootstrap-seed must be non-negative.")
    _resolved_device(args.device)
    folds = parse_integer_grid(args.folds, minimum=1, maximum=7)
    seeds = parse_integer_grid(args.seeds, minimum=0)
    validate_canonical_grid(folds, seeds)
    return args


def run(args: argparse.Namespace) -> dict[str, Any]:
    args = validate_args(args)
    folds, seeds = validate_canonical_grid(
        parse_integer_grid(args.folds, minimum=1, maximum=7),
        parse_integer_grid(args.seeds, minimum=0),
    )
    offline_root = Path(args.offline_cv_root).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    offline_grid, offline_members = validate_offline_grid(offline_root, folds, seeds)
    identity = _grid_identity(
        args,
        offline_root=offline_root,
        offline_grid_manifest=offline_grid,
        folds=folds,
        seeds=seeds,
    )
    _validate_or_create_output_manifest(output_root, identity)
    grid_identity_sha256 = canonical_hash(identity)
    validate_member_set(output_root, folds, seeds, allow_missing=True)

    if bool(args.dry_run):
        commands = [
            _member_command(
                args,
                offline_members[(int(fold), int(seed))],
                member_directory(output_root, fold, seed),
            )
            for fold in folds
            for seed in seeds
        ]
        return {
            "schema": CV_SCHEMA,
            "dry_run": True,
            "member_count": len(commands),
            "expected_session_row_count": EXPECTED_ROW_COUNT,
            "commands": commands,
        }

    collected: list[dict[str, Any]] = []
    for fold in folds:
        for seed in seeds:
            offline = offline_members[(int(fold), int(seed))]
            target = member_directory(output_root, fold, seed)
            member_identity = _member_identity(
                grid_identity_sha256=grid_identity_sha256,
                offline=offline,
                args=args,
            )
            ensure_member_identity(target, member_identity, resume=bool(args.resume))
            complete_path = target / "complete.json"
            if complete_path.is_file():
                # Corrupt completed members are never overwritten, even under
                # --resume; validation must succeed before reuse.
                member_rows = validate_completed_member(target, fold=fold, seed=seed)
                if not bool(args.resume):
                    raise FileExistsError(
                        f"Completed member already exists: {target}; use --resume."
                    )
            else:
                command = _member_command(args, offline, target)
                print("[command] " + " ".join(f'\"{item}\"' for item in command), flush=True)
                try:
                    subprocess.run(command, cwd=PROJECT_ROOT, check=True)
                except subprocess.CalledProcessError as error:
                    raise RuntimeError(
                        f"Online member fold={fold} seed={seed} failed with exit code "
                        f"{error.returncode}; its identity-locked partial directory may be resumed."
                    ) from error
                member_rows = validate_completed_member(target, fold=fold, seed=seed)
            collected.extend(member_rows)
            write_csv(
                output_root / "online_runs.partial.csv",
                sorted(
                    collected,
                    key=lambda row: (int(row["fold"]), int(row["seed"]), int(row["session"])),
                ),
            )

    validate_member_set(output_root, folds, seeds, allow_missing=False)
    # Re-read every authoritative member after orchestration; this prevents an
    # incremental in-memory row from concealing an altered or duplicate member.
    rows = []
    for fold in folds:
        for seed in seeds:
            rows.extend(
                validate_completed_member(
                    member_directory(output_root, fold, seed), fold=fold, seed=seed
                )
            )
    if len(rows) != EXPECTED_ROW_COUNT:
        raise RuntimeError(f"Expected 84 session records, observed {len(rows)}.")
    return write_reports(
        output_root,
        rows,
        folds=folds,
        seeds=seeds,
        bootstrap_seed=int(args.bootstrap_seed),
        grid_identity_sha256=grid_identity_sha256,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Strict frozen A2/E0/state Online CGCD over canonical 7x4x3 CV."
    )
    parser.add_argument("--offline-cv-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--folds", default=",".join(map(str, CANONICAL_FOLDS)))
    parser.add_argument("--seeds", default=",".join(map(str, CANONICAL_SEEDS)))
    parser.add_argument("--encode-batch-size", type=int, default=512)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--allow-smoke-a2", action="store_true")
    parser.add_argument("--bootstrap-seed", type=int, default=20260912)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> dict[str, Any]:
    result = run(build_parser().parse_args(argv))
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return result


if __name__ == "__main__":
    main()


__all__ = [
    "BOOTSTRAP_REPLICATES",
    "CV_SCHEMA",
    "EXPECTED_MEMBER_COUNT",
    "EXPECTED_ROW_COUNT",
    "MEMBER_SCHEMA",
    "OfflineMember",
    "SESSION_COUNT",
    "build_parser",
    "ensure_member_identity",
    "main",
    "run",
    "validate_args",
    "validate_canonical_grid",
    "validate_completed_member",
    "validate_member_set",
    "validate_offline_grid",
    "write_reports",
]
