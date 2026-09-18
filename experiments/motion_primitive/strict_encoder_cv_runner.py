"""Resume-safe window warm-up and A2 encoder grid orchestration.

This is the only formal encoder preparation route used by the frozen
``A2 + E0 + state + K32`` experiment.  It deliberately keeps the historical
two optimisation objectives as two *checkpoints* (HAR window warm-up followed
by A2 local-motion training), while every downstream CGCD component consumes
only the frozen A2 ``content`` representation.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.motion_primitive.strict_artifacts import write_csv, write_json
from experiments.motion_primitive.strict_cv_common import (
    CANONICAL_FOLDS,
    CANONICAL_SEEDS,
    member_directory,
    parse_integer_grid,
    validate_or_create_grid_manifest,
)
from experiments.motion_primitive.strict_protocol import sha256_file
from experiments.motion_primitive import pretrain_window_encoder, train_motion_encoder


CV_SCHEMA = "hhr_frozen_a2_encoder_cv_v2"
WINDOW_SCHEMA = "hhr_har_window_pretrain_v2"
A2_COMPLETE_SCHEMA = train_motion_encoder.COMPLETE_SCHEMA
_MEMBER_PATTERN = re.compile(r"fold_(\d{2})_seed_(\d+)")

FORMAL_W256_PROTOCOL = "formal_w256_7fold4seed"
W128_CONFIRMATION_PROTOCOL = "w128_confirmation_7fold2seed"
DEFAULT_PROTOCOL = FORMAL_W256_PROTOCOL
W128_CONFIRMATION_FOLDS = tuple(range(1, 8))
W128_CONFIRMATION_SEEDS = (0, 5)
PROTOCOL_SPECS: dict[str, dict[str, Any]] = {
    FORMAL_W256_PROTOCOL: {
        "folds": CANONICAL_FOLDS,
        "seeds": CANONICAL_SEEDS,
        "window_size": 256,
        "window_stride": 128,
    },
    W128_CONFIRMATION_PROTOCOL: {
        "folds": W128_CONFIRMATION_FOLDS,
        "seeds": W128_CONFIRMATION_SEEDS,
        "window_size": 128,
        "window_stride": 64,
    },
}


def _protocol_spec(value: str) -> dict[str, Any]:
    try:
        return dict(PROTOCOL_SPECS[str(value)])
    except KeyError as error:
        raise ValueError(
            f"Unknown encoder protocol {value!r}; expected {list(PROTOCOL_SPECS)}."
        ) from error


def _validate_confirmation_hyperparameters(args: argparse.Namespace) -> None:
    """Lock the W128 confirmation to the exact three-fold screening recipe."""

    if str(args.protocol) != W128_CONFIRMATION_PROTOCOL:
        return
    expected = {
        "window_epochs": 60,
        "window_batch_size": 256,
        "window_eval_batch_size": 1024,
        "a2_epochs": 30,
        "a2_trial_batch_size": 8,
        "a2_source_encode_batch_size": 1024,
    }
    observed = {name: int(getattr(args, name)) for name in expected}
    if observed != expected:
        raise ValueError(
            "The W128 confirmation protocol is locked to the original screening "
            f"training recipe: expected={expected}, observed={observed}."
        )


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"JSON artifact is not an object: {path}.")
    return value


def _implementation_hashes() -> dict[str, str]:
    relative = (
        "experiments/motion_primitive/strict_encoder_cv_runner.py",
        "experiments/motion_primitive/strict_artifacts.py",
        "experiments/motion_primitive/strict_cv_common.py",
        "models/resnet1d.py",
        "models/window_pretrain.py",
        "experiments/motion_primitive/pretrain_window_encoder.py",
        "experiments/motion_primitive/motion_encoder.py",
        "experiments/motion_primitive/motion_checkpoint.py",
        "experiments/motion_primitive/motion_augmentation.py",
        "experiments/motion_primitive/raw_changepoint.py",
        "experiments/motion_primitive/train_motion_encoder.py",
        "experiments/motion_primitive/strict_protocol.py",
    )
    return {name: sha256_file(PROJECT_ROOT / name) for name in relative}


def _child_arguments(
    command: Sequence[str], parser: argparse.ArgumentParser
) -> argparse.Namespace:
    if len(command) < 2:
        raise ValueError("A child command must contain an executable and script path.")
    return parser.parse_args(list(command[2:]))


def _validate_member_set(
    stage_root: Path,
    expected: set[tuple[int, int]],
    *,
    allow_missing: bool,
) -> None:
    """Reject extra/malformed member directories and optionally missing ones."""

    if not stage_root.exists():
        if allow_missing:
            return
        raise RuntimeError(f"Encoder stage root is absent: {stage_root}.")
    if not stage_root.is_dir():
        raise RuntimeError(f"Encoder stage root is not a directory: {stage_root}.")
    observed: set[tuple[int, int]] = set()
    malformed: list[str] = []
    for child in stage_root.iterdir():
        if not child.name.startswith("fold_"):
            continue
        if not child.is_dir():
            malformed.append(child.name)
            continue
        match = _MEMBER_PATTERN.fullmatch(child.name)
        if match is None:
            malformed.append(child.name)
            continue
        observed.add((int(match.group(1)), int(match.group(2))))
    extra = sorted(observed - expected)
    missing = sorted(expected - observed)
    if malformed or extra or (missing and not allow_missing):
        raise RuntimeError(
            f"Invalid encoder member set below {stage_root}: malformed={sorted(malformed)}, "
            f"extra={extra}, missing={missing if not allow_missing else []}."
        )


def _window_command(args: argparse.Namespace, fold: int, seed: int, output: Path) -> list[str]:
    protocol = _protocol_spec(args.protocol)
    command = [
        sys.executable,
        str(PROJECT_ROOT / "experiments" / "motion_primitive" / "pretrain_window_encoder.py"),
        "--npz-path", str(Path(args.npz_path).expanduser().resolve()),
        "--output-dir", str(output),
        "--fold", str(int(fold)),
        "--seed", str(int(seed)),
        "--window-size", str(int(protocol["window_size"])),
        "--window-stride", str(int(protocol["window_stride"])),
        "--epochs", str(int(args.window_epochs)),
        "--batch-size", str(int(args.window_batch_size)),
        "--eval-batch-size", str(int(args.window_eval_batch_size)),
        "--learning-rate", str(float(args.window_learning_rate)),
        "--weight-decay", str(float(args.window_weight_decay)),
        "--weak-scale-std", str(float(args.window_weak_scale_std)),
        "--strong-scale-std", str(float(args.window_strong_scale_std)),
        "--num-workers", str(int(args.num_workers)),
        "--smoke-max-windows", str(int(args.smoke_max_windows)),
        "--device", str(args.device),
        "--deterministic",
        "--selection-policy", str(args.window_selection_policy),
    ]
    if bool(args.resume):
        command.append("--resume")
    return command


def _a2_command(
    args: argparse.Namespace,
    fold: int,
    seed: int,
    source_checkpoint: Path,
    output: Path,
) -> list[str]:
    command = [
        sys.executable,
        str(PROJECT_ROOT / "experiments" / "motion_primitive" / "train_motion_encoder.py"),
        "--source-checkpoint", str(source_checkpoint),
        "--npz-path", str(Path(args.npz_path).expanduser().resolve()),
        "--output-dir", str(output),
        "--ablation-profile", "A2",
        "--window-aug-consistency", "none",
        "--window-aug-profile", "basic",
        "--window-aug-weight", "1.0",
        "--rotation-max-degrees", "0.0",
        "--cp-weight", "1.0",
        "--content-boundary-alignment-weight", "0.1",
        "--cp-anchor-source", "raw_frozen_consensus",
        "--cp-context-windows", "2",
        "--cp-low-quantile", "0.50",
        "--cp-high-quantile", "0.90",
        "--cp-raw-scales", "1,2,4",
        "--cp-raw-frequency-bins", "16",
        "--cp-rank-margin", "0.20",
        "--cp-equivariance-weight", "0.50",
        "--cp-equivariance-delta", "1.0",
        "--noncollapse-weight", "0.05",
        "--noncollapse-target-std", "1.0",
        "--noncollapse-variance-weight", "1.0",
        "--noncollapse-covariance-weight", "1.0",
        "--noncollapse-windows-per-trial", "4",
        "--prediction-weight", "0.5",
        "--prediction-mask-ratio", "0.20",
        "--prediction-loss", "cosine",
        "--trial-weight", "0.1",
        "--cross-subject-weight", "0.0",
        "--segmentation-dim", "0",
        "--content-dim", "256",
        "--content-residual",
        "--augmentation-dim", "128",
        "--projection-hidden-dim", "256",
        "--trial-hidden-dim", "128",
        "--trial-peak-quantile", "0.90",
        "--trial-dropout", "0.0",
        "--predictor-hidden-dim", "0",
        "--backbone-layers", "2,2,2",
        "--old-class-count", "6",
        "--epochs", str(int(args.a2_epochs)),
        "--trial-batch-size", str(int(args.a2_trial_batch_size)),
        "--source-encode-batch-size", str(int(args.a2_source_encode_batch_size)),
        "--learning-rate", str(float(args.a2_learning_rate)),
        "--minimum-learning-rate", str(float(args.a2_minimum_learning_rate)),
        "--weight-decay", str(float(args.a2_weight_decay)),
        "--gradient-clip-norm", "5.0",
        "--ema-momentum", "0.99",
        "--freeze-backbone-epochs", "0",
        "--backbone-bn-policy", "frozen",
        "--early-stopping-patience", "0",
        "--minimum-improvement", "0.0",
        "--selection-policy", "final_epoch",
        "--normalization-eps", "1e-6",
        "--known-anomaly-policy", "report",
        "--smoke-max-train-trials", str(int(args.smoke_max_train_trials)),
        "--smoke-max-val-trials", str(int(args.smoke_max_val_trials)),
        "--seed", str(int(seed)),
        "--device", str(args.device),
        "--deterministic",
    ]
    if bool(args.resume):
        command.append("--resume")
    return command


def _validate_window_member(
    path: Path,
    *,
    fold: int,
    seed: int,
    npz_sha256: str,
    expected_identity: Mapping[str, Any],
) -> dict[str, Any] | None:
    complete_path = path / "complete.json"
    if not complete_path.is_file():
        return None
    complete = _read_json(complete_path)
    if complete.get("schema") != WINDOW_SCHEMA or complete.get("complete") is not True:
        raise RuntimeError(f"Invalid window-pretrain completion marker: {path}.")
    if (int(complete.get("fold", -1)), int(complete.get("seed", -1))) != (fold, seed):
        raise RuntimeError("Window-pretrain fold/seed identity mismatch.")
    identity = complete.get("identity")
    if not isinstance(identity, Mapping) or dict(identity) != dict(expected_identity):
        raise RuntimeError("Window-pretrain member records another complete run identity.")
    if identity.get("npz_sha256") != npz_sha256:
        raise RuntimeError("Window-pretrain member is bound to another NPZ.")
    checkpoint = path / str(complete.get("checkpoint", "model_best.pt"))
    if not checkpoint.is_file() or sha256_file(checkpoint) != complete.get("checkpoint_sha256"):
        raise RuntimeError("Window-pretrain checkpoint is absent or has changed.")
    checkpoint_payload = pretrain_window_encoder._load_checkpoint(checkpoint)
    if checkpoint_payload.get("run_identity") != dict(expected_identity):
        raise RuntimeError("Window-pretrain checkpoint records another run identity.")
    pretrain_window_encoder._validate_checkpoint_state_hashes(checkpoint_payload)
    runtime = complete.get("determinism")
    if not isinstance(runtime, Mapping):
        raise RuntimeError("Window-pretrain completion lacks deterministic runtime.")
    pretrain_window_encoder._validate_determinism_record(
        runtime, require_python_hash_seed=True
    )
    checkpoint_runtime = checkpoint_payload.get("determinism")
    if checkpoint_runtime != runtime:
        raise RuntimeError(
            "Window-pretrain checkpoint/completion deterministic runtime differs."
        )
    if checkpoint_payload.get("model_state_dict_sha256") != complete.get(
        "model_state_dict_sha256"
    ):
        raise RuntimeError("Window-pretrain selected tensor-state SHA256 mismatch.")
    if checkpoint_payload.get("backbone_state_dict_sha256") != complete.get(
        "backbone_state_dict_sha256"
    ):
        raise RuntimeError("Window-pretrain selected backbone SHA256 mismatch.")
    return complete


def _validate_a2_member(
    path: Path,
    *,
    fold: int,
    seed: int,
    npz_sha256: str,
    source_checkpoint_sha256: str,
    expected_identity: Mapping[str, Any],
) -> dict[str, Any] | None:
    complete_path = path / "complete.json"
    if not complete_path.is_file():
        return None
    complete = _read_json(complete_path)
    expected = {
        "schema": A2_COMPLETE_SCHEMA,
        "fold": int(fold),
        "seed": int(seed),
        "npz_sha256": str(npz_sha256),
        "source_checkpoint_sha256": str(source_checkpoint_sha256),
        "ablation_profile": "A2",
        "selection_policy": "final_epoch",
        "complete": True,
    }
    observed = {key: complete.get(key) for key in expected}
    if observed != expected:
        raise RuntimeError(
            f"A2 member identity/status differs for fold={fold} seed={seed}: {observed}."
        )
    identity = complete.get("identity")
    if not isinstance(identity, Mapping) or dict(identity) != dict(expected_identity):
        raise RuntimeError(
            f"A2 member records another complete run identity for fold={fold} seed={seed}."
        )
    final_path = path / "motion_encoder_final.pt"
    if not final_path.is_file() or sha256_file(final_path) != complete.get("final_checkpoint_sha256"):
        raise RuntimeError("A2 final checkpoint is absent or has changed.")
    checkpoint_payload = train_motion_encoder._load_checkpoint(final_path)
    train_motion_encoder._validate_output_checkpoint(checkpoint_payload)
    if checkpoint_payload.get("run_identity") != dict(expected_identity):
        raise RuntimeError("A2 checkpoint records another run identity.")
    runtime = complete.get("determinism")
    if not isinstance(runtime, Mapping):
        raise RuntimeError("A2 completion lacks deterministic runtime metadata.")
    train_motion_encoder._validate_determinism_record(
        dict(runtime), require_python_hash_seed=True
    )
    if checkpoint_payload.get("determinism") != runtime:
        raise RuntimeError("A2 checkpoint/completion deterministic runtime differs.")
    if checkpoint_payload.get("model_state_dict_sha256") != complete.get(
        "final_model_state_dict_sha256"
    ):
        raise RuntimeError("A2 final model tensor-state SHA256 mismatch.")
    if checkpoint_payload.get("ema_teacher_state_dict_sha256") != complete.get(
        "final_ema_teacher_state_dict_sha256"
    ):
        raise RuntimeError("A2 final EMA-teacher tensor-state SHA256 mismatch.")
    return complete


def _archive_incomplete(path: Path, *, root: Path, stage: str) -> Path:
    """Move, never delete, an interrupted exact member before a clean restart."""

    resolved_path = path.resolve()
    resolved_root = root.resolve()
    if resolved_root not in resolved_path.parents:
        raise RuntimeError("Refusing to archive a directory outside this encoder-CV root.")
    archive_root = resolved_root / "_interrupted"
    archive_root.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    target = archive_root / f"{stage}_{path.name}_{stamp}_{uuid.uuid4().hex[:8]}"
    shutil.move(str(resolved_path), str(target))
    return target


def _ensure_member(
    *,
    args: argparse.Namespace,
    stage: str,
    target: Path,
    validator: Callable[[], dict[str, Any] | None],
    command: Sequence[str],
    root: Path,
) -> dict[str, Any]:
    complete = validator() if target.is_dir() else None
    if complete is not None:
        if not bool(args.resume):
            raise FileExistsError(f"Completed {stage} member already exists: {target}.")
        return complete
    if target.exists():
        if not bool(args.resume):
            raise FileExistsError(f"Incomplete {stage} member exists: {target}.")
        archived = _archive_incomplete(target, root=root, stage=stage)
        print(f"[resume] preserved interrupted {stage} member at {archived}", flush=True)
    print("[command] " + " ".join(f'"{item}"' for item in command), flush=True)
    try:
        environment = os.environ.copy()
        environment.update({
            "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
            "PYTHONHASHSEED": "0",
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
            "VECLIB_MAXIMUM_THREADS": "1",
        })
        subprocess.run(
            list(command), cwd=PROJECT_ROOT, check=True, env=environment
        )
    except subprocess.CalledProcessError as error:
        raise RuntimeError(
            f"{stage} member {target.name} failed with exit code {error.returncode}."
        ) from error
    complete = validator()
    if complete is None:
        raise RuntimeError(f"{stage} subprocess exited successfully without complete artifacts.")
    return complete


def run(args: argparse.Namespace) -> dict[str, Any]:
    protocol = _protocol_spec(args.protocol)
    folds = parse_integer_grid(args.folds, minimum=1, maximum=7)
    seeds = parse_integer_grid(args.seeds, minimum=0)
    expected_folds = tuple(int(value) for value in protocol["folds"])
    expected_seeds = tuple(int(value) for value in protocol["seeds"])
    if folds != expected_folds or seeds != expected_seeds:
        raise ValueError(
            f"Encoder protocol {args.protocol!r} is fixed to folds "
            f"{expected_folds} and seeds {expected_seeds}."
        )
    _validate_confirmation_hyperparameters(args)
    if int(args.window_epochs) < 1 or int(args.a2_epochs) < 1:
        raise ValueError("Encoder epoch counts must be positive.")
    root = Path(args.output_root).expanduser().resolve()
    npz = Path(args.npz_path).expanduser().resolve()
    if not npz.is_file():
        raise FileNotFoundError(npz)
    npz_hash = sha256_file(npz)
    identity = {
        "schema": CV_SCHEMA,
        "protocol": str(args.protocol),
        "route": "ResNet1D_window_warmup_then_A2_final",
        "folds": list(folds),
        "seeds": list(seeds),
        "npz_path": str(npz),
        "npz_sha256": npz_hash,
        "window": {
            "window_size": int(protocol["window_size"]),
            "window_stride": int(protocol["window_stride"]),
            "epochs": int(args.window_epochs),
            "batch_size": int(args.window_batch_size),
            "eval_batch_size": int(args.window_eval_batch_size),
            "learning_rate": float(args.window_learning_rate),
            "weight_decay": float(args.window_weight_decay),
            "weak_scale_std": float(args.window_weak_scale_std),
            "strong_scale_std": float(args.window_strong_scale_std),
            "deterministic": True,
            "selection_policy": str(args.window_selection_policy),
            "smoke_max_windows": int(args.smoke_max_windows),
        },
        "a2": {
            "profile": "A2",
            "epochs": int(args.a2_epochs),
            "trial_batch_size": int(args.a2_trial_batch_size),
            "source_encode_batch_size": int(args.a2_source_encode_batch_size),
            "learning_rate": float(args.a2_learning_rate),
            "minimum_learning_rate": float(args.a2_minimum_learning_rate),
            "weight_decay": float(args.a2_weight_decay),
            "loss_weights": {
                "changepoint": 1.0,
                "content_boundary_alignment": 0.1,
                "temporal_prediction": 0.5,
                "noncollapse": 0.05,
                "trial_auxiliary": 0.1,
                "window_augmentation_InfoNCE": 0.0,
            },
            "selection_policy": "final_epoch",
            "batchnorm_running_statistics": "frozen",
            "smoke_max_train_trials": int(args.smoke_max_train_trials),
            "smoke_max_val_trials": int(args.smoke_max_val_trials),
        },
        "device": str(args.device),
        "num_workers": int(args.num_workers),
        "implementation_sha256": _implementation_hashes(),
    }
    validate_or_create_grid_manifest(root, identity)
    commands: list[list[str]] = []
    rows: list[dict[str, Any]] = []
    window_root = root / "window_pretrain"
    a2_root = root / "a2"
    expected_members = {(fold, seed) for fold in folds for seed in seeds}
    _validate_member_set(window_root, expected_members, allow_missing=True)
    _validate_member_set(a2_root, expected_members, allow_missing=True)
    for fold in folds:
        for seed in seeds:
            window_dir = member_directory(window_root, fold, seed)
            window_command = _window_command(args, fold, seed, window_dir)
            commands.append(window_command)
            if bool(args.dry_run):
                source = window_dir / (
                    "model_best.pt"
                    if str(args.window_selection_policy) == "best_val_macro_f1"
                    else "model_last.pt"
                )
                commands.append(_a2_command(args, fold, seed, source, member_directory(a2_root, fold, seed)))
                continue
            window_arguments = _child_arguments(
                window_command, pretrain_window_encoder.build_parser()
            )
            window_identity = pretrain_window_encoder._run_identity(window_arguments)
            window_validator = lambda d=window_dir, f=fold, s=seed: _validate_window_member(
                d,
                fold=f,
                seed=s,
                npz_sha256=npz_hash,
                expected_identity=window_identity,
            )
            window_complete = _ensure_member(
                args=args,
                stage="window_pretrain",
                target=window_dir,
                validator=window_validator,
                command=window_command,
                root=root,
            )
            source = window_dir / str(window_complete["checkpoint"])
            source_hash = sha256_file(source)
            a2_dir = member_directory(a2_root, fold, seed)
            a2_command = _a2_command(args, fold, seed, source, a2_dir)
            commands.append(a2_command)
            a2_arguments = _child_arguments(
                a2_command, train_motion_encoder.build_parser()
            )
            a2_identity = train_motion_encoder.resolve_run_identity(a2_arguments)
            a2_validator = lambda d=a2_dir, f=fold, s=seed, h=source_hash: _validate_a2_member(
                d,
                fold=f,
                seed=s,
                npz_sha256=npz_hash,
                source_checkpoint_sha256=h,
                expected_identity=a2_identity,
            )
            a2_complete = _ensure_member(
                args=args,
                stage="a2",
                target=a2_dir,
                validator=a2_validator,
                command=a2_command,
                root=root,
            )
            rows.append({
                "fold": int(fold),
                "seed": int(seed),
                "window_best_epoch": int(window_complete["best_epoch"]),
                "window_validation_macro_f1": float(window_complete["best_validation_macro_f1"]),
                "window_selected_epoch": int(window_complete["selected_epoch"]),
                "window_selected_validation_macro_f1": float(
                    window_complete["selected_validation_macro_f1"]
                ),
                "window_model_state_dict_sha256": str(
                    window_complete["model_state_dict_sha256"]
                ),
                "window_backbone_state_dict_sha256": str(
                    window_complete["backbone_state_dict_sha256"]
                ),
                "window_checkpoint_sha256": str(window_complete["checkpoint_sha256"]),
                "a2_selected_epoch": int(a2_complete["selected_epoch"]),
                "a2_validation_total_loss": float(a2_complete["selected_validation_total_loss"]),
                "a2_checkpoint_sha256": str(a2_complete["final_checkpoint_sha256"]),
                "a2_model_state_dict_sha256": str(
                    a2_complete["final_model_state_dict_sha256"]
                ),
                "a2_ema_teacher_state_dict_sha256": str(
                    a2_complete["final_ema_teacher_state_dict_sha256"]
                ),
                "smoke_test": bool(window_complete.get("smoke_test") or a2_complete.get("smoke_test")),
            })
            write_csv(root / "encoder_runs.csv", rows)
    if bool(args.dry_run):
        return {
            "schema": CV_SCHEMA,
            "protocol": str(args.protocol),
            "folds": list(folds),
            "seeds": list(seeds),
            "window_size": int(protocol["window_size"]),
            "window_stride": int(protocol["window_stride"]),
            "member_count": len(folds) * len(seeds),
            "dry_run": True,
            "commands": commands,
        }
    _validate_member_set(window_root, expected_members, allow_missing=False)
    _validate_member_set(a2_root, expected_members, allow_missing=False)
    expected = expected_members
    observed = {(int(row["fold"]), int(row["seed"])) for row in rows}
    if len(rows) != len(expected) or observed != expected:
        raise RuntimeError("Encoder CV did not produce the exact requested grid.")
    summary = {
        "schema": CV_SCHEMA,
        "protocol": str(args.protocol),
        "route": "ResNet1D_window_warmup_then_A2_final",
        "folds": list(folds),
        "seeds": list(seeds),
        "member_count": len(rows),
        "window_size": int(protocol["window_size"]),
        "window_stride": int(protocol["window_stride"]),
        "window_pretrain_root": str(window_root),
        "a2_root": str(a2_root),
        "smoke_test": bool(
            int(args.smoke_max_windows) > 0
            or int(args.smoke_max_train_trials) > 0
            or int(args.smoke_max_val_trials) > 0
        ),
        "complete": True,
    }
    write_json(root / "encoder_summary.json", summary)
    write_json(root / "complete.json", summary)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare a registered ResNet1D warm-up and A2 encoder CV grid, "
            "including the isolated W128/S64 seven-fold confirmation grid."
        )
    )
    parser.add_argument(
        "--protocol",
        choices=tuple(PROTOCOL_SPECS),
        default=DEFAULT_PROTOCOL,
    )
    parser.add_argument("--npz-path", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--folds", default=",".join(map(str, CANONICAL_FOLDS)))
    parser.add_argument("--seeds", default=",".join(map(str, CANONICAL_SEEDS)))
    parser.add_argument("--window-epochs", type=int, default=60)
    parser.add_argument("--window-batch-size", type=int, default=64)
    parser.add_argument("--window-eval-batch-size", type=int, default=256)
    parser.add_argument("--window-learning-rate", type=float, default=0.1)
    parser.add_argument("--window-weight-decay", type=float, default=5e-4)
    parser.add_argument("--window-weak-scale-std", type=float, default=0.1)
    parser.add_argument("--window-strong-scale-std", type=float, default=0.2)
    parser.add_argument(
        "--window-selection-policy",
        choices=("best_val_macro_f1", "final_epoch"),
        default="best_val_macro_f1",
    )
    parser.add_argument("--a2-epochs", type=int, default=30)
    parser.add_argument("--a2-trial-batch-size", type=int, default=8)
    parser.add_argument("--a2-source-encode-batch-size", type=int, default=512)
    parser.add_argument("--a2-learning-rate", type=float, default=1e-4)
    parser.add_argument("--a2-minimum-learning-rate", type=float, default=1e-6)
    parser.add_argument("--a2-weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--smoke-max-windows", type=int, default=0)
    parser.add_argument("--smoke-max-train-trials", type=int, default=0)
    parser.add_argument("--smoke-max-val-trials", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> dict[str, Any]:
    result = run(build_parser().parse_args(argv))
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return result


if __name__ == "__main__":
    main()
