"""Legacy joint VQ/GRU trajectory-only online subject-CV orchestrator.

This implementation remains available for historical reproduction but is no
longer the public HHR launcher. Each member consumes the validation-selected offline trajectory checkpoint.
Resume is fail-closed and never overwrites partial runs.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.motion_primitive.legacy_profiles import (  # noqa: E402
    PROFILE_JOINT,
    PROFILES,
    normalize_profile,
    normalize_profile_grid,
)

ONLINE_SCRIPT = Path(__file__).with_name("online_runner.py")
CV_SCHEMA = "hhr_motion_primitive_online_subject_cv_v3"
OFFLINE_CV_SCHEMA = "hhr_motion_primitive_offline_subject_cv_v3"
SESSION_COUNT = 3


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_hash(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"Cannot read JSON metadata {path}: {error}.") from error
    if not isinstance(payload, dict):
        raise RuntimeError(f"JSON metadata must contain an object: {path}.")
    return payload


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, path)


def _write_csv(
    path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames), extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def parse_integer_grid(
    value: str, *, minimum: int, maximum: Optional[int] = None
) -> tuple[int, ...]:
    tokens = tuple(token.strip() for token in str(value).split(",") if token.strip())
    if not tokens:
        raise ValueError("An experiment grid cannot be empty.")
    values = tuple(int(token) for token in tokens)
    if len(values) != len(set(values)):
        raise ValueError(f"Duplicate experiment-grid values are forbidden: {values}.")
    for item in values:
        if item < minimum or (maximum is not None and item > maximum):
            raise ValueError(f"Experiment-grid value {item} is outside the registered range.")
    return tuple(sorted(values))


def parse_profiles(value: str) -> tuple[str, ...]:
    raw = tuple(token.strip() for token in str(value).split(",") if token.strip())
    if not raw:
        raise ValueError("--profiles cannot be empty.")
    return normalize_profile_grid(raw)


def offline_member_directory(root: Path, profile: str, fold: int, seed: int) -> Path:
    canonical = normalize_profile(profile)
    return Path(root) / f"profile_{canonical}" / f"fold_{int(fold):02d}_seed_{int(seed)}"


def online_member_directory(root: Path, profile: str, fold: int, seed: int) -> Path:
    return offline_member_directory(root, profile, fold, seed)


def selected_checkpoint_name(profile: str) -> str:
    profile = normalize_profile(profile)
    if profile == PROFILE_JOINT:
        return "checkpoint_best_trajectory.pt"
    raise ValueError(f"Unknown profile {profile!r}.")


def _boolean_option(name: str, enabled: bool) -> str:
    return f"--{name}" if enabled else f"--no-{name}"


def build_member_command(
    args: argparse.Namespace,
    *,
    profile: str,
    fold: int,
    seed: int,
    offline_run_dir: Path,
    checkpoint: Path,
    output_dir: Path,
) -> list[str]:
    """Build an explicit command so formal protocol values are auditable."""

    policy = args.codebook_expansion
    command = [
        str(Path(args.python_executable).expanduser().resolve()),
        str(ONLINE_SCRIPT.resolve()),
        "--offline-run-dir", str(Path(offline_run_dir).resolve()),
        "--checkpoint", str(Path(checkpoint).resolve()),
        "--output-dir", str(Path(output_dir).resolve()),
        "--device", str(args.device),
        "--epochs-per-session", str(args.epochs_per_session),
        "--batch-size", str(args.batch_size),
        "--eval-batch-size", str(args.eval_batch_size),
        "--num-workers", str(args.num_workers),
        "--eval-num-workers", str(args.eval_num_workers),
        "--lr", str(args.lr),
        "--encoder-lr-scale", str(args.encoder_lr_scale),
        "--momentum", str(args.momentum),
        "--weight-decay", str(args.weight_decay),
        "--cosine-minimum-ratio", str(args.cosine_minimum_ratio),
        "--student-temperature", str(args.student_temperature),
        "--warmup-teacher-temperature", str(args.warmup_teacher_temperature),
        "--teacher-temperature", str(args.teacher_temperature),
        "--warmup-teacher-epochs", str(args.warmup_teacher_epochs),
        "--memax-old-new-weight", str(args.memax_old_new_weight),
        "--memax-old-in-weight", str(args.memax_old_in_weight),
        "--memax-new-in-weight", str(args.memax_new_in_weight),
        _boolean_option(
            "initialize-new-trajectory-head-with-kmeans",
            bool(args.initialize_new_trajectory_head_with_kmeans),
        ),
        "--kmeans-random-state", str(args.kmeans_random_state),
        "--trajectory-cluster-weight", str(args.trajectory_cluster_weight),
        "--trajectory-logit-distillation-weight",
        str(args.trajectory_logit_distillation_weight),
        "--trajectory-feature-distillation-weight",
        str(args.trajectory_feature_distillation_weight),
        "--primitive-feature-distillation-weight",
        str(args.primitive_feature_distillation_weight),
        "--old-codebook-anchor-weight", str(args.old_codebook_anchor_weight),
        "--trajectory-view-consistency-weight", str(args.trajectory_view_consistency_weight),
        "--vq-commitment-weight", str(args.vq_commitment_weight),
        "--vq-codebook-weight", str(args.vq_codebook_weight),
        "--codebook-expansion", str(policy),
        "--codebook-fixed-delta", str(args.codebook_fixed_delta),
        "--codebook-kmeans-random-state", str(args.codebook_kmeans_random_state),
        "--expected-offline-codebook-size", str(args.expected_offline_codebook_size),
        "--codebook-adaptive-max-delta", str(args.codebook_adaptive_max_delta),
        "--codebook-residual-quantile", str(args.codebook_residual_quantile),
        "--codebook-minimum-residual-support",
        str(args.codebook_minimum_residual_support),
        "--codebook-minimum-cluster-support",
        str(args.codebook_minimum_cluster_support),
        "--codebook-minimum-cluster-trials",
        str(args.codebook_minimum_cluster_trials),
        "--codebook-minimum-cluster-subjects",
        str(args.codebook_minimum_cluster_subjects),
        "--codebook-minimum-relative-improvement",
        str(args.codebook_minimum_relative_improvement),
        "--codebook-complexity-penalty", str(args.codebook_complexity_penalty),
    ]
    return command


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run trajectory-only online CGCD over an offline CV grid",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--offline-cv-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--profiles", default=PROFILE_JOINT)
    parser.add_argument("--folds", default="1,2,3,4,5,6,7")
    parser.add_argument("--seeds", default="0,5,50,500")
    parser.add_argument("--python-executable", default=sys.executable)
    parser.add_argument("--device", default="cuda")

    # Registered trajectory-only online defaults.
    parser.add_argument("--epochs-per-session", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--eval-num-workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--encoder-lr-scale", type=float, default=0.1)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=5.0e-4)
    parser.add_argument("--cosine-minimum-ratio", type=float, default=1.0e-3)
    parser.add_argument("--student-temperature", type=float, default=0.10)
    parser.add_argument("--warmup-teacher-temperature", type=float, default=0.05)
    parser.add_argument("--teacher-temperature", type=float, default=0.05)
    parser.add_argument("--warmup-teacher-epochs", type=int, default=10)
    parser.add_argument("--memax-old-new-weight", type=float, default=1.0)
    parser.add_argument("--memax-old-in-weight", type=float, default=1.0)
    parser.add_argument("--memax-new-in-weight", type=float, default=1.0)
    parser.add_argument(
        "--initialize-new-trajectory-head-with-kmeans",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--kmeans-random-state", type=int, default=0)

    parser.add_argument("--trajectory-cluster-weight", type=float, default=1.0)
    parser.add_argument("--trajectory-logit-distillation-weight", type=float, default=1.0)
    parser.add_argument("--trajectory-feature-distillation-weight", type=float, default=1.0)
    parser.add_argument("--primitive-feature-distillation-weight", type=float, default=1.0)
    parser.add_argument("--old-codebook-anchor-weight", type=float, default=1.0)
    parser.add_argument("--trajectory-view-consistency-weight", type=float, default=0.0)
    parser.add_argument("--vq-commitment-weight", type=float, default=0.25)
    parser.add_argument("--vq-codebook-weight", type=float, default=0.25)
    parser.add_argument(
        "--codebook-expansion",
        choices=("none", "fixed_delta", "residual_adaptive"),
        default="residual_adaptive",
    )
    parser.add_argument("--codebook-fixed-delta", type=int, default=2)
    parser.add_argument("--codebook-kmeans-random-state", type=int, default=0)
    parser.add_argument("--expected-offline-codebook-size", type=int, default=32)
    parser.add_argument("--codebook-adaptive-max-delta", type=int, default=4)
    parser.add_argument("--codebook-residual-quantile", type=float, default=0.90)
    parser.add_argument("--codebook-minimum-residual-support", type=int, default=32)
    parser.add_argument("--codebook-minimum-cluster-support", type=int, default=8)
    parser.add_argument("--codebook-minimum-cluster-trials", type=int, default=3)
    parser.add_argument("--codebook-minimum-cluster-subjects", type=int, default=2)
    parser.add_argument(
        "--codebook-minimum-relative-improvement", type=float, default=0.10
    )
    parser.add_argument("--codebook-complexity-penalty", type=float, default=0.01)

    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def validate_args(args: argparse.Namespace) -> argparse.Namespace:
    if args.resume and args.skip_existing:
        raise ValueError("--resume and --skip-existing are aliases; choose one.")
    for name in ("epochs_per_session", "batch_size", "eval_batch_size"):
        if int(getattr(args, name)) < 1:
            raise ValueError(f"--{name.replace('_', '-')} must be positive.")
    for name in ("num_workers", "eval_num_workers", "warmup_teacher_epochs"):
        if int(getattr(args, name)) < 0:
            raise ValueError(f"--{name.replace('_', '-')} cannot be negative.")
    if int(args.warmup_teacher_epochs) > int(args.epochs_per_session):
        raise ValueError("Teacher-temperature warmup cannot exceed session epochs.")
    for name in (
        "lr", "encoder_lr_scale", "student_temperature",
        "warmup_teacher_temperature", "teacher_temperature",
    ):
        value = float(getattr(args, name))
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive and finite.")
    for name in (
        "momentum", "weight_decay", "memax_old_new_weight", "memax_old_in_weight",
        "memax_new_in_weight", "trajectory_cluster_weight",
        "trajectory_logit_distillation_weight",
        "trajectory_feature_distillation_weight",
        "primitive_feature_distillation_weight", "old_codebook_anchor_weight",
        "trajectory_view_consistency_weight", "vq_commitment_weight",
        "vq_codebook_weight",
    ):
        value = float(getattr(args, name))
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(f"--{name.replace('_', '-')} must be finite and non-negative.")
    if not 0.0 <= float(args.cosine_minimum_ratio) <= 1.0:
        raise ValueError("--cosine-minimum-ratio must lie in [0,1].")
    if int(args.codebook_fixed_delta) < 1:
        raise ValueError("--codebook-fixed-delta must be positive.")
    if int(args.expected_offline_codebook_size) != 32:
        raise ValueError("The registered residual-adaptive experiment requires offline K=32.")
    for name in (
        "codebook_adaptive_max_delta",
        "codebook_minimum_residual_support",
        "codebook_minimum_cluster_support",
        "codebook_minimum_cluster_trials",
        "codebook_minimum_cluster_subjects",
    ):
        value = getattr(args, name)
        if isinstance(value, bool) or int(value) != value or int(value) < 1:
            raise ValueError(f"--{name.replace('_', '-')} must be a positive integer.")
    if not 0.0 <= float(args.codebook_residual_quantile) < 1.0:
        raise ValueError("--codebook-residual-quantile must lie in [0,1).")
    if not 0.0 <= float(args.codebook_minimum_relative_improvement) <= 1.0:
        raise ValueError(
            "--codebook-minimum-relative-improvement must lie in [0,1]."
        )
    if (
        not math.isfinite(float(args.codebook_complexity_penalty))
        or float(args.codebook_complexity_penalty) < 0.0
    ):
        raise ValueError(
            "--codebook-complexity-penalty must be finite and non-negative."
        )
    python = Path(args.python_executable).expanduser()
    if not python.is_file():
        raise FileNotFoundError(f"Python executable does not exist: {python}.")
    return args


def _validate_hashed_manifest(
    path: Path, *, expected_schema: str, allow_created_utc: bool = False
) -> dict[str, Any]:
    payload = _load_json(path)
    if payload.get("schema") != expected_schema:
        raise RuntimeError(
            f"Unexpected manifest schema in {path}: {payload.get('schema')!r}."
        )
    claimed = payload.get("identity_sha256")
    unsigned = {
        key: value
        for key, value in payload.items()
        if key != "identity_sha256" and (not allow_created_utc or key != "created_utc")
    }
    if not claimed or claimed != _canonical_hash(unsigned):
        raise RuntimeError(f"Manifest failed its integrity hash: {path}.")
    return payload


def resolve_offline_grid(
    offline_root: Path,
    profiles: Sequence[str],
    folds: Sequence[int],
    seeds: Sequence[int],
) -> tuple[dict[str, Any], dict[tuple[str, int, int], tuple[Path, Path]]]:
    root = Path(offline_root).expanduser().resolve()
    manifest_path = root / "cv_manifest.json"
    if not root.is_dir() or not manifest_path.is_file():
        raise FileNotFoundError(f"Completed offline CV manifest not found: {manifest_path}.")
    manifest = _validate_hashed_manifest(
        manifest_path, expected_schema=OFFLINE_CV_SCHEMA
    )
    recorded_profile_names: dict[str, str] = {}
    for raw_value in manifest.get("profiles", []):
        raw_profile = str(raw_value)
        canonical_profile = normalize_profile(raw_profile)
        if canonical_profile in recorded_profile_names:
            raise RuntimeError(
                "Offline CV manifest contains duplicate profiles after legacy-name "
                f"normalisation: {canonical_profile}."
            )
        recorded_profile_names[canonical_profile] = raw_profile
    recorded_folds = set(int(value) for value in manifest.get("folds", []))
    recorded_seeds = set(int(value) for value in manifest.get("seeds", []))
    if not set(profiles).issubset(recorded_profile_names):
        raise RuntimeError("Requested online profiles are absent from the offline CV manifest.")
    if not set(folds).issubset(recorded_folds) or not set(seeds).issubset(recorded_seeds):
        raise RuntimeError("Requested online fold/seed grid is absent from the offline CV manifest.")

    members: dict[tuple[str, int, int], tuple[Path, Path]] = {}
    for profile in profiles:
        canonical_profile = normalize_profile(profile)
        recorded_profile = recorded_profile_names[canonical_profile]
        for fold in folds:
            for seed in seeds:
                canonical_dir = offline_member_directory(
                    root, canonical_profile, fold, seed
                ).resolve()
                recorded_dir = (
                    root
                    / f"profile_{recorded_profile}"
                    / f"fold_{int(fold):02d}_seed_{int(seed)}"
                ).resolve()
                run_dir = canonical_dir if canonical_dir.is_dir() else recorded_dir
                checkpoint = (
                    run_dir / selected_checkpoint_name(canonical_profile)
                ).resolve()
                required = (run_dir / "manifest.json", run_dir / "summary.json", checkpoint)
                missing = [str(path) for path in required if not path.is_file()]
                if missing:
                    raise RuntimeError(
                        f"Offline member {canonical_profile}/{fold}/{seed} is incomplete: {missing}."
                    )
                member_manifest = _load_json(run_dir / "manifest.json")
                if normalize_profile(str(member_manifest.get("profile", ""))) != canonical_profile:
                    raise RuntimeError(f"Offline member profile mismatch in {run_dir}.")
                arguments = member_manifest.get("arguments", {})
                if int(arguments.get("uschad_cv_fold", -1)) != int(fold):
                    raise RuntimeError(f"Offline member fold mismatch in {run_dir}.")
                if int(arguments.get("seed", -1)) != int(seed):
                    raise RuntimeError(f"Offline member seed mismatch in {run_dir}.")
                members[(canonical_profile, int(fold), int(seed))] = (
                    run_dir,
                    checkpoint,
                )
    return manifest, members


def _identity(
    args: argparse.Namespace,
    *,
    profiles: Sequence[str],
    folds: Sequence[int],
    seeds: Sequence[int],
    offline_manifest: Path,
    members: Mapping[tuple[str, int, int], tuple[Path, Path]],
) -> dict[str, Any]:
    parameters = {
        key: value
        for key, value in vars(args).items()
        if key not in {"resume", "skip_existing", "dry_run"}
    }
    parameters["offline_cv_root"] = str(Path(args.offline_cv_root).expanduser().resolve())
    parameters["output_root"] = str(Path(args.output_root).expanduser().resolve())
    parameters["python_executable"] = str(
        Path(args.python_executable).expanduser().resolve()
    )
    parameters["profiles"] = ",".join(profiles)
    payload: dict[str, Any] = {
        "schema": CV_SCHEMA,
        "profiles": list(profiles),
        "folds": list(folds),
        "seeds": list(seeds),
        "session_count": SESSION_COUNT,
        "expected_rows_per_profile": len(folds) * len(seeds) * SESSION_COUNT,
        "parameters": parameters,
        "offline_cv_manifest": str(Path(offline_manifest).resolve()),
        "offline_cv_manifest_sha256": _sha256_file(offline_manifest),
        "offline_checkpoints": [
            {
                "profile": profile,
                "fold": fold,
                "seed": seed,
                "offline_run_dir": str(run_dir),
                "path": str(checkpoint),
                "sha256": _sha256_file(checkpoint),
            }
            for (profile, fold, seed), (run_dir, checkpoint) in sorted(members.items())
        ],
        "online_cv_runner": str(Path(__file__).resolve()),
        "online_cv_runner_sha256": _sha256_file(Path(__file__)),
        "online_script": str(ONLINE_SCRIPT.resolve()),
        "online_script_sha256": _sha256_file(ONLINE_SCRIPT),
        "online_runner": str(ONLINE_SCRIPT.with_name("online_runner.py").resolve()),
        "online_runner_sha256": _sha256_file(
            ONLINE_SCRIPT.with_name("online_runner.py")
        ),
    }
    payload["identity_sha256"] = _canonical_hash(payload)
    return payload


def _member_state(run_dir: Path) -> str:
    if not run_dir.exists():
        return "missing"
    if not run_dir.is_dir():
        return "invalid"
    if not any(run_dir.iterdir()):
        return "empty"
    required = [
        "manifest.json",
        "online_protocol.json",
        "online_summary.json",
        "online_runs.csv",
    ]
    for session in range(1, SESSION_COUNT + 1):
        required.extend(
            (f"checkpoint_session_{session}.pt", f"metrics_session_{session}.json")
        )
    return "complete" if all((run_dir / name).is_file() for name in required) else "incomplete"


def _expected_cli_arguments(command: Sequence[str]) -> dict[str, Any]:
    # Parse with the single-run parser to avoid a second interpretation of
    # booleans, None defaults, and numeric types.
    from experiments.motion_primitive.online_runner import build_parser

    parsed = build_parser().parse_args(list(command[2:]))
    return {key: value for key, value in vars(parsed).items() if key != "output_dir"}


def _read_member_rows(
    run_dir: Path,
    *,
    command: Sequence[str],
    profile: str,
    fold: int,
    seed: int,
) -> list[dict[str, str]]:
    manifest = _validate_hashed_manifest(
        run_dir / "manifest.json",
        expected_schema="hhr_motion_primitive_online_run_manifest_v3",
        allow_created_utc=True,
    )
    expected_arguments = _expected_cli_arguments(command)
    if manifest.get("arguments") != expected_arguments:
        raise RuntimeError(f"Existing online member arguments differ in {run_dir}.")
    expected_checkpoint = Path(
        command[command.index("--checkpoint") + 1]
    ).resolve()
    if manifest.get("offline_checkpoint") != str(expected_checkpoint):
        raise RuntimeError(f"Existing online member checkpoint differs in {run_dir}.")
    if manifest.get("offline_checkpoint_sha256") != _sha256_file(expected_checkpoint):
        raise RuntimeError(f"Existing online member checkpoint hash differs in {run_dir}.")

    with (run_dir / "online_runs.csv").open(
        "r", encoding="utf-8-sig", newline=""
    ) as handle:
        rows = [dict(row) for row in csv.DictReader(handle)]
    observed = [
        (str(row.get("profile")), int(row["fold"]), int(row["seed"]), int(row["session"]))
        for row in rows
    ]
    expected = [(profile, int(fold), int(seed), session) for session in range(1, 4)]
    if observed != expected or len(set(observed)) != SESSION_COUNT:
        raise RuntimeError(
            f"Online member has incomplete/duplicate session rows in {run_dir}: {observed}."
        )
    # Recompute every CSV row from the authoritative per-session JSON plus the
    # exact offline manifest/checkpoint.  Merely checking that files exist is
    # insufficient: a stale or hand-edited CSV must never enter a formal CV
    # aggregate.
    from experiments.motion_primitive import online_runner as single

    offline_run_dir = Path(command[command.index("--offline-run-dir") + 1]).resolve()
    offline_manifest = _load_json(offline_run_dir / "manifest.json")
    checkpoint_payload = single._torch_load(expected_checkpoint, map_location="cpu")
    checkpoint_sha256 = _sha256_file(expected_checkpoint)
    policy = str(manifest.get("codebook_expansion", {}).get("policy", ""))
    if not policy:
        raise RuntimeError(f"Online member lacks a codebook policy in {run_dir}.")

    records: list[dict[str, Any]] = []
    for row, session in zip(rows, range(1, SESSION_COUNT + 1)):
        metrics_path = run_dir / f"metrics_session_{session}.json"
        record = _load_json(metrics_path)
        records.append(record)
        checkpoint_path = run_dir / f"checkpoint_session_{session}.pt"
        checkpoint_state = single._torch_load(checkpoint_path, map_location="cpu")
        if checkpoint_state.get("schema") != "hhr_motion_primitive_online_checkpoint_v3":
            raise RuntimeError(f"Unexpected online checkpoint schema in {checkpoint_path}.")
        if int(checkpoint_state.get("session", -1)) != session:
            raise RuntimeError(f"Online checkpoint session mismatch in {checkpoint_path}.")
        if checkpoint_state.get("metrics") != record.get("metrics"):
            raise RuntimeError(
                f"Checkpoint/metrics JSON disagreement for session {session} in {run_dir}."
            )
        if checkpoint_state.get("retention_audit") != record.get("retention_audit"):
            raise RuntimeError(
                "Checkpoint/retention-audit JSON disagreement for session "
                f"{session} in {run_dir}."
            )
        architecture = checkpoint_state.get("architecture")
        state = checkpoint_state.get("model")
        if not isinstance(architecture, Mapping) or not isinstance(state, Mapping):
            raise RuntimeError(
                f"Online checkpoint is not self-describing in {checkpoint_path}."
            )
        from models.motion_primitive_cgcd import (
            MotionPrimitiveCGCDModel,
            MotionPrimitiveConfig,
        )

        try:
            model_config = MotionPrimitiveConfig(**dict(architecture)).validated()
            if int(model_config.old_class_count) != int(
                checkpoint_state.get("class_count", -1)
            ):
                raise RuntimeError("architecture class count disagrees with checkpoint")
            if int(model_config.codebook_size) != int(
                checkpoint_state.get("codebook_size", -1)
            ):
                raise RuntimeError("architecture codebook size disagrees with checkpoint")
            reconstructed = MotionPrimitiveCGCDModel(model_config)
            reconstructed.load_state_dict(dict(state), strict=True)
        except (TypeError, ValueError, RuntimeError) as error:
            raise RuntimeError(
                f"Online checkpoint cannot be strictly reconstructed: "
                f"{checkpoint_path}: {error}."
            ) from error

        expected_row = single.session_metric_row(
            record,
            report_head=str(manifest["report_head"]),
            profile=profile,
            fold=fold,
            seed=seed,
            manifest={
                **offline_manifest,
                "online_config": manifest["online_config"],
            },
            offline_checkpoint_payload=checkpoint_payload,
            offline_run_dir=offline_run_dir,
            offline_checkpoint=expected_checkpoint,
            offline_checkpoint_sha256=checkpoint_sha256,
            metrics_path=metrics_path,
            codebook_policy=policy,
        )
        for field, value in expected_row.items():
            expected_text = "" if value is None else str(value)
            if row.get(field) != expected_text:
                raise RuntimeError(
                    f"CSV/artifact disagreement in {run_dir}, session={session}, "
                    f"field={field}: csv={row.get(field)!r}, expected={expected_text!r}."
                )

    summary = _load_json(run_dir / "online_summary.json")
    if summary.get("schema") != "hhr_motion_primitive_online_summary_v3":
        raise RuntimeError(f"Unexpected online summary schema in {run_dir}.")
    if summary.get("config") != manifest.get("online_config"):
        raise RuntimeError(f"Online summary/config disagreement in {run_dir}.")
    if summary.get("sessions") != records:
        raise RuntimeError(f"Online summary/session JSON disagreement in {run_dir}.")
    return rows


def validate_online_grid(
    rows: Sequence[Mapping[str, Any]],
    folds: Sequence[int],
    seeds: Sequence[int],
) -> None:
    expected = {
        (int(fold), int(seed), int(session))
        for fold in folds
        for seed in seeds
        for session in range(1, SESSION_COUNT + 1)
    }
    observed_list = [
        (int(row["fold"]), int(row["seed"]), int(row["session"])) for row in rows
    ]
    observed = set(observed_list)
    duplicates = sorted(key for key in observed if observed_list.count(key) != 1)
    missing = sorted(expected - observed)
    unexpected = sorted(observed - expected)
    if duplicates or missing or unexpected:
        raise RuntimeError(
            "Incomplete or ambiguous online result grid: "
            f"duplicates={duplicates}, missing={missing}, unexpected={unexpected}."
        )


def _fieldnames(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    from experiments.motion_primitive.online_runner import online_fieldnames

    return online_fieldnames(rows)


def aggregate_summary(
    rows: Sequence[Mapping[str, Any]],
    folds: Sequence[int],
    seeds: Sequence[int],
) -> dict[str, Any]:
    validate_online_grid(rows, folds, seeds)
    summary: dict[str, Any] = {
        "schema": "hhr_motion_primitive_online_subject_cv_summary_v3",
        "statistical_unit": "held-out subject fold after averaging seeds",
        "fold_count": len(folds),
        "seed_count": len(seeds),
        "row_count": len(rows),
        "sessions": {},
    }
    for session in range(1, SESSION_COUNT + 1):
        session_rows = [row for row in rows if int(row["session"]) == session]
        metrics: dict[str, Any] = {}
        for metric in (
            "gcd_all_accuracy", "gcd_old_accuracy", "gcd_new_accuracy", "h_score"
        ):
            fold_means = {
                int(fold): sum(
                    float(row[metric])
                    for row in session_rows
                    if int(row["fold"]) == int(fold)
                )
                / len(seeds)
                for fold in folds
            }
            metrics[metric] = {
                "mean": sum(fold_means.values()) / len(fold_means),
                "fold_means": {
                    str(key): value for key, value in sorted(fold_means.items())
                },
            }
        summary["sessions"][str(session)] = metrics
    return summary


def collect_profile_rows(
    output_root: Path,
    profile: str,
    folds: Sequence[int],
    seeds: Sequence[int],
    commands: Mapping[tuple[str, int, int], Sequence[str]],
) -> list[dict[str, str]]:
    profile_root = Path(output_root) / f"profile_{profile}"
    expected_csvs = {
        (
            online_member_directory(output_root, profile, fold, seed)
            / "online_runs.csv"
        ).resolve()
        for fold in folds
        for seed in seeds
    }
    observed_csvs = {
        path.resolve() for path in profile_root.rglob("online_runs.csv") if path.is_file()
    }
    unexpected = sorted(observed_csvs - expected_csvs)
    if unexpected:
        raise RuntimeError(
            f"Unexpected online member result files below {profile_root}: "
            f"{[str(path) for path in unexpected]}."
        )
    rows: list[dict[str, str]] = []
    for fold in folds:
        for seed in seeds:
            key = (profile, int(fold), int(seed))
            rows.extend(
                _read_member_rows(
                    online_member_directory(output_root, profile, fold, seed),
                    command=commands[key],
                    profile=profile,
                    fold=fold,
                    seed=seed,
                )
            )
    validate_online_grid(rows, folds, seeds)
    rows.sort(key=lambda row: (int(row["fold"]), int(row["seed"]), int(row["session"])))
    return rows


def _paths_overlap(left: Path, right: Path) -> bool:
    left = left.resolve()
    right = right.resolve()
    return left == right or left in right.parents or right in left.parents


def run(args: argparse.Namespace) -> dict[str, Any]:
    args = validate_args(args)
    profiles = parse_profiles(args.profiles)
    folds = parse_integer_grid(args.folds, minimum=1, maximum=7)
    seeds = parse_integer_grid(args.seeds, minimum=0)
    offline_root = Path(args.offline_cv_root).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    if _paths_overlap(offline_root, output_root):
        raise RuntimeError(
            "--output-root must be separate from, and not nested inside, --offline-cv-root."
        )

    _offline_identity, members = resolve_offline_grid(
        offline_root, profiles, folds, seeds
    )
    cv_identity = _identity(
        args,
        profiles=profiles,
        folds=folds,
        seeds=seeds,
        offline_manifest=offline_root / "cv_manifest.json",
        members=members,
    )
    cv_manifest_path = output_root / "cv_manifest.json"
    if output_root.exists() and not output_root.is_dir():
        raise RuntimeError(f"Online output root is not a directory: {output_root}.")
    if output_root.exists() and any(output_root.iterdir()):
        if not cv_manifest_path.is_file():
            raise RuntimeError(
                f"Non-empty online output root has no CV manifest: {output_root}."
            )
        recorded = _validate_hashed_manifest(
            cv_manifest_path, expected_schema=CV_SCHEMA
        )
        if recorded.get("identity_sha256") != cv_identity.get("identity_sha256"):
            raise RuntimeError(
                "Online output root records a different profile/fold/seed/protocol/"
                "checkpoint identity; use a new --output-root."
            )

    commands: dict[tuple[str, int, int], list[str]] = {}
    statuses: dict[tuple[str, int, int], str] = {}
    for key, (offline_dir, checkpoint) in sorted(members.items()):
        profile, fold, seed = key
        output_dir = online_member_directory(output_root, profile, fold, seed)
        commands[key] = build_member_command(
            args,
            profile=profile,
            fold=fold,
            seed=seed,
            offline_run_dir=offline_dir,
            checkpoint=checkpoint,
            output_dir=output_dir,
        )
        statuses[key] = _member_state(output_dir)

    reuse = bool(args.resume or args.skip_existing)
    for key, state in statuses.items():
        if state == "complete":
            _read_member_rows(
                online_member_directory(output_root, *key),
                command=commands[key],
                profile=key[0],
                fold=key[1],
                seed=key[2],
            )
            if not reuse:
                raise FileExistsError(
                    f"Completed online member already exists for {key}; use --resume."
                )
        elif state in {"incomplete", "invalid"}:
            raise RuntimeError(
                f"Cannot safely resume {state} online member {key}; preserve it for "
                "diagnosis and use a clean member/output root."
            )

    for key in sorted(commands):
        action = "skip" if statuses[key] == "complete" else "run"
        print(f"[{action}] {subprocess.list2cmdline(commands[key])}", flush=True)
    if args.dry_run:
        return {
            "dry_run": True,
            "identity": cv_identity,
            "commands": [commands[key] for key in sorted(commands)],
            "statuses": {"|".join(map(str, key)): value for key, value in statuses.items()},
        }

    if not cv_manifest_path.exists():
        output_root.mkdir(parents=True, exist_ok=True)
        _write_json(cv_manifest_path, cv_identity)
    for key in sorted(commands):
        if statuses[key] == "complete":
            continue
        try:
            subprocess.run(commands[key], cwd=PROJECT_ROOT, check=True)
        except subprocess.CalledProcessError as error:
            raise RuntimeError(
                f"Online member profile={key[0]} fold={key[1]} seed={key[2]} "
                f"failed with exit code {error.returncode}; child traceback is above."
            ) from error

    output_tables: dict[str, str] = {}
    summaries: dict[str, str] = {}
    for profile in profiles:
        rows = collect_profile_rows(output_root, profile, folds, seeds, commands)
        profile_root = output_root / f"profile_{profile}"
        csv_path = profile_root / "subject_cv_online_runs.csv"
        summary_path = profile_root / "subject_cv_online_summary.json"
        _write_csv(csv_path, rows, _fieldnames(rows))
        _write_json(summary_path, aggregate_summary(rows, folds, seeds))
        output_tables[profile] = str(csv_path.resolve())
        summaries[profile] = str(summary_path.resolve())
    return {
        "dry_run": False,
        "identity": cv_identity,
        "subject_cv_online_runs": output_tables,
        "summaries": summaries,
    }


def main(argv: Optional[Sequence[str]] = None) -> dict[str, Any]:
    result = run(build_parser().parse_args(argv))
    if not result["dry_run"]:
        for profile, path in result["subject_cv_online_runs"].items():
            print(f"[completed] profile={profile} table={path}", flush=True)
    return result


if __name__ == "__main__":
    main()
