"""Five-run bitwise reproducibility audit for the W128/A2/K64 route.

This module deliberately contains no training, encoding, codebook, or scoring
algorithm.  It orchestrates the registered child entry points and compares
their tensor/array contents.  Serialized ``.pt`` file hashes are integrity
metadata only and are never a reproducibility pass criterion because identical
tensor states need not have identical container bytes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.motion_primitive import run_trajectory_descriptor_ablation_cv as descriptor_cv
from experiments.motion_primitive import strict_encoder_cv_runner as encoder_cv
from experiments.motion_primitive import pretrain_window_encoder, train_motion_encoder
from experiments.motion_primitive import window_codebook_batch_proxy
from experiments.motion_primitive.frozen_e0 import (
    DURATION_SOFT_SUBJECT_A025_PROFILE,
    LEGACY_DESCRIPTOR_PROFILE,
)
from experiments.motion_primitive.strict_artifacts import write_csv, write_json
from experiments.motion_primitive.strict_cv_common import (
    canonical_hash,
    validate_or_create_grid_manifest,
)
from experiments.motion_primitive.strict_protocol import sha256_file


SCHEMA = "hhr_deterministic_repro_audit_v1"
FOLD = 1
SEED = 0
REPEAT_COUNT = 5
PROFILES = (LEGACY_DESCRIPTOR_PROFILE, DURATION_SOFT_SUBJECT_A025_PROFILE)
METRICS = tuple(descriptor_cv.PERFORMANCE_METRICS)
_HEX_DIGEST_LENGTH = 64


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"JSON artifact is not an object: {path}.")
    return value


def _implementation_hashes() -> dict[str, str]:
    """Bind resume identity to every dependency registered by both child routes."""

    hashes = dict(encoder_cv._implementation_hashes())
    hashes.update(window_codebook_batch_proxy._implementation_hashes())
    hashes.update(descriptor_cv._implementation_hashes())
    own_name = "experiments/motion_primitive/run_deterministic_repro_audit.py"
    hashes[own_name] = sha256_file(PROJECT_ROOT / own_name)
    return dict(sorted(hashes.items()))


def _canonical_json_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _canonical_history_sha256(path: Path, *, json_lines: bool) -> str:
    """Hash parsed history content so whitespace/newline formatting is irrelevant."""

    if not path.is_file():
        raise FileNotFoundError(path)
    if json_lines:
        values = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    else:
        values = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(values, list) or not values:
        raise RuntimeError(f"Training history must be a non-empty list: {path}.")
    if not all(isinstance(value, Mapping) for value in values):
        raise RuntimeError(f"Training history contains a non-object entry: {path}.")
    return _canonical_json_sha256(values)


def _stable_array_sha256(named_arrays: Mapping[str, np.ndarray]) -> str:
    """Hash array semantics, not NPZ/ZIP container bytes."""

    digest = hashlib.sha256()
    for name in sorted(named_arrays):
        value = np.ascontiguousarray(np.asarray(named_arrays[name]))
        digest.update(str(name).encode("utf-8"))
        digest.update(b"\0")
        digest.update(value.dtype.str.encode("ascii"))
        digest.update(b"\0")
        digest.update(json.dumps(list(value.shape), separators=(",", ":")).encode("ascii"))
        digest.update(b"\0")
        digest.update(value.tobytes(order="C"))
        digest.update(b"\0")
    return digest.hexdigest()


def _raw_prediction_content(path: Path) -> dict[str, str]:
    """Return stable hashes of learner outputs without hashing the NPZ file."""

    with np.load(path, allow_pickle=False) as payload:
        required = {"trial_ids", "trajectory_features", "raw_activity_cluster_ids"}
        missing = sorted(required - set(payload.files))
        if missing:
            raise RuntimeError(f"Raw-prediction artifact lacks arrays: {missing}.")
        trial_ids = np.asarray(payload["trial_ids"], dtype=np.int64)
        clusters = np.asarray(payload["raw_activity_cluster_ids"], dtype=np.int64)
        features = np.asarray(payload["trajectory_features"], dtype=np.float32)
    if trial_ids.ndim != 1 or clusters.shape != trial_ids.shape:
        raise RuntimeError("Raw prediction trial/cluster arrays have incompatible shapes.")
    if features.ndim != 2 or features.shape[0] != len(trial_ids):
        raise RuntimeError("Raw prediction trajectory-feature rows do not match trial IDs.")
    return {
        "prediction_array_sha256": _stable_array_sha256(
            {"trial_ids": trial_ids, "raw_activity_cluster_ids": clusters}
        ),
        "trajectory_feature_array_sha256": _stable_array_sha256(
            {"trial_ids": trial_ids, "trajectory_features": features}
        ),
    }


def _metric_values(summary: Mapping[str, Any]) -> dict[str, float]:
    values: dict[str, float] = {}
    for name in METRICS:
        value = float(summary.get(name, float("nan")))
        if not math.isfinite(value):
            raise RuntimeError(f"Descriptor metric {name!r} is absent or non-finite.")
        values[name] = value
    return values


def _metric_signature(values: Mapping[str, float]) -> str:
    return canonical_hash({name: float(values[name]) for name in METRICS})


def _reference_record(path_value: str | None) -> dict[str, Any] | None:
    if not path_value:
        return None
    path = Path(path_value).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    summary = _read_json(path)
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "descriptor_profile": str(
            summary.get("descriptor_profile", summary.get("profile", ""))
        ),
        "fold": int(summary.get("fold", -1)),
        "seed": int(summary.get("seed", -1)),
        "metrics": _metric_values(summary),
    }


def _reference_deltas(
    reference: Mapping[str, Any] | None,
    repeat_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    if reference is None:
        return {"provided": False, "affects_reproducibility_pass": False}
    reference_metrics = reference["metrics"]
    rows = []
    for repeat in repeat_rows:
        current = repeat["descriptors"][DURATION_SOFT_SUBJECT_A025_PROFILE]["metrics"]
        rows.append(
            {
                "repeat": int(repeat["repeat"]),
                "current": dict(current),
                "reference": dict(reference_metrics),
                "delta_current_minus_reference": {
                    name: float(current[name]) - float(reference_metrics[name])
                    for name in METRICS
                },
            }
        )
    identity_matches = (
        reference.get("descriptor_profile") == DURATION_SOFT_SUBJECT_A025_PROFILE
        and int(reference.get("fold", -1)) == FOLD
        and int(reference.get("seed", -1)) == SEED
    )
    return {
        "provided": True,
        "affects_reproducibility_pass": False,
        "identity_matches_expected_fold_seed_profile": bool(identity_matches),
        "reference": dict(reference),
        "repeat_deltas": rows,
    }


def _encoder_namespace(args: argparse.Namespace) -> argparse.Namespace:
    """Adapt this audit CLI to the registered child-command builders."""

    return argparse.Namespace(
        protocol=encoder_cv.W128_CONFIRMATION_PROTOCOL,
        npz_path=str(args.npz_path),
        window_epochs=int(args.window_epochs),
        window_batch_size=int(args.window_batch_size),
        window_eval_batch_size=int(args.window_eval_batch_size),
        window_learning_rate=float(args.window_learning_rate),
        window_weight_decay=float(args.window_weight_decay),
        window_weak_scale_std=float(args.window_weak_scale_std),
        window_strong_scale_std=float(args.window_strong_scale_std),
        window_selection_policy="best_val_macro_f1",
        a2_epochs=int(args.a2_epochs),
        a2_trial_batch_size=int(args.a2_trial_batch_size),
        a2_source_encode_batch_size=int(args.a2_source_encode_batch_size),
        a2_learning_rate=float(args.a2_learning_rate),
        a2_minimum_learning_rate=float(args.a2_minimum_learning_rate),
        a2_weight_decay=float(args.a2_weight_decay),
        num_workers=int(args.num_workers),
        device=str(args.device),
        smoke_max_windows=0,
        smoke_max_train_trials=0,
        smoke_max_val_trials=0,
        resume=bool(args.resume),
    )


def _descriptor_namespace(args: argparse.Namespace) -> argparse.Namespace:
    return argparse.Namespace(
        npz_path=str(args.npz_path),
        signed_vertical_distance_weight=float(args.signed_vertical_distance_weight),
        subject_nuisance_max_rank=int(args.subject_nuisance_max_rank),
        subject_nuisance_explained_variance=float(
            args.subject_nuisance_explained_variance
        ),
        kmeans_n_init=int(args.kmeans_n_init),
        kmeans_max_iter=int(args.kmeans_max_iter),
        encode_batch_size=int(args.encode_batch_size),
        device=str(args.device),
        resume=bool(args.resume),
    )


def _run_descriptor_child(command: Sequence[str], *, stage: str) -> None:
    print("[command] " + " ".join(f'"{item}"' for item in command), flush=True)
    environment = os.environ.copy()
    environment["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    environment["PYTHONHASHSEED"] = "0"
    try:
        subprocess.run(
            list(command), cwd=PROJECT_ROOT, check=True, env=environment
        )
    except subprocess.CalledProcessError as error:
        raise RuntimeError(
            f"Reproducibility descriptor stage {stage!r} failed with exit code "
            f"{error.returncode}; the child traceback is printed above."
        ) from error


def _run_repeat(
    args: argparse.Namespace,
    *,
    repeat: int,
    root: Path,
) -> dict[str, Any]:
    repeat_root = root / "repeats" / f"repeat_{repeat:02d}"
    encoder_root = repeat_root / "encoder"
    window_dir = encoder_root / "window_pretrain" / f"fold_{FOLD:02d}_seed_{SEED}"
    a2_dir = encoder_root / "a2" / f"fold_{FOLD:02d}_seed_{SEED}"
    encoder_args = _encoder_namespace(args)
    window_command = encoder_cv._window_command(
        encoder_args, FOLD, SEED, window_dir
    )
    window_arguments = encoder_cv._child_arguments(
        window_command, pretrain_window_encoder.build_parser()
    )
    window_identity = pretrain_window_encoder._run_identity(window_arguments)
    npz_hash = sha256_file(Path(args.npz_path).expanduser().resolve())
    window_complete = encoder_cv._ensure_member(
        args=encoder_args,
        stage=f"repeat_{repeat:02d}_window_pretrain",
        target=window_dir,
        validator=lambda: encoder_cv._validate_window_member(
            window_dir,
            fold=FOLD,
            seed=SEED,
            npz_sha256=npz_hash,
            expected_identity=window_identity,
        ),
        command=window_command,
        root=encoder_root,
    )
    source_checkpoint = window_dir / str(window_complete["checkpoint"])
    a2_command = encoder_cv._a2_command(
        encoder_args, FOLD, SEED, source_checkpoint, a2_dir
    )
    a2_arguments = encoder_cv._child_arguments(
        a2_command, train_motion_encoder.build_parser()
    )
    a2_identity = train_motion_encoder.resolve_run_identity(a2_arguments)
    source_file_sha = sha256_file(source_checkpoint)
    a2_complete = encoder_cv._ensure_member(
        args=encoder_args,
        stage=f"repeat_{repeat:02d}_a2",
        target=a2_dir,
        validator=lambda: encoder_cv._validate_a2_member(
            a2_dir,
            fold=FOLD,
            seed=SEED,
            npz_sha256=npz_hash,
            source_checkpoint_sha256=source_file_sha,
            expected_identity=a2_identity,
        ),
        command=a2_command,
        root=encoder_root,
    )

    descriptor_args = _descriptor_namespace(args)
    descriptor_root = repeat_root / "descriptors"
    checkpoint = a2_dir / "motion_encoder_final.pt"
    descriptor_records: dict[str, Any] = {}
    legacy_dir = descriptor_root / LEGACY_DESCRIPTOR_PROFILE / f"fold_{FOLD:02d}_seed_{SEED}"
    for profile in PROFILES:
        output = descriptor_root / profile / f"fold_{FOLD:02d}_seed_{SEED}"
        reuse = None if profile == LEGACY_DESCRIPTOR_PROFILE else legacy_dir
        command = descriptor_cv._member_command(
            descriptor_args,
            profile=profile,
            fold=FOLD,
            seed=SEED,
            checkpoint=checkpoint,
            output=output,
            reuse_codebook_run_dir=reuse,
        )
        _run_descriptor_child(command, stage=f"repeat_{repeat:02d}_{profile}")
        row = descriptor_cv._validated_member(
            output, profile=profile, fold=FOLD, seed=SEED
        )
        summary = _read_json(output / "summary.json")
        cluster = _read_json(output / "activity_cluster.json")
        cluster_state_sha = str(cluster.get("state_sha256", ""))
        if len(cluster_state_sha) != _HEX_DIGEST_LENGTH:
            raise RuntimeError("Descriptor activity cluster lacks a valid state SHA256.")
        raw_content = _raw_prediction_content(output / "raw_predictions.npz")
        metrics = _metric_values(summary)
        descriptor_records[profile] = {
            **raw_content,
            "metrics": metrics,
            "metric_signature_sha256": _metric_signature(metrics),
            "codebook_state_sha256": str(row["codebook_state_sha256"]),
            "descriptor_transform_state_sha256": str(
                summary["descriptor_transform_state_sha256"]
            ),
            "activity_cluster_state_sha256": cluster_state_sha,
            "member_dir": str(output),
        }
    if (
        descriptor_records[LEGACY_DESCRIPTOR_PROFILE]["codebook_state_sha256"]
        != descriptor_records[DURATION_SOFT_SUBJECT_A025_PROFILE][
            "codebook_state_sha256"
        ]
    ):
        raise RuntimeError("The alpha=0.25 arm did not reuse its repeat-local legacy codebook.")
    return {
        "repeat": int(repeat),
        "fold": FOLD,
        "seed": SEED,
        "repeat_root": str(repeat_root),
        "warmup": {
            "selected_epoch": int(window_complete["selected_epoch"]),
            "selected_validation_macro_f1": float(
                window_complete["selected_validation_macro_f1"]
            ),
            "history_canonical_sha256": _canonical_history_sha256(
                window_dir / "history.jsonl", json_lines=True
            ),
            "model_tensor_sha256": str(window_complete["model_state_dict_sha256"]),
            "backbone_tensor_sha256": str(
                window_complete["backbone_state_dict_sha256"]
            ),
        },
        "a2": {
            "selected_epoch": int(a2_complete["selected_epoch"]),
            "selected_validation_total_loss": float(
                a2_complete["selected_validation_total_loss"]
            ),
            "history_canonical_sha256": _canonical_history_sha256(
                a2_dir / "history.json", json_lines=False
            ),
            "model_tensor_sha256": str(
                a2_complete["final_model_state_dict_sha256"]
            ),
            "ema_teacher_tensor_sha256": str(
                a2_complete["final_ema_teacher_state_dict_sha256"]
            ),
        },
        "descriptors": descriptor_records,
    }


def _path_value(value: Mapping[str, Any], path: str) -> Any:
    current: Any = value
    for part in path.split("."):
        if not isinstance(current, Mapping) or part not in current:
            raise KeyError(path)
        current = current[part]
    return current


def _consistency_report(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    checks = (
        "warmup.selected_epoch",
        "warmup.selected_validation_macro_f1",
        "warmup.history_canonical_sha256",
        "warmup.model_tensor_sha256",
        "warmup.backbone_tensor_sha256",
        "a2.selected_epoch",
        "a2.selected_validation_total_loss",
        "a2.history_canonical_sha256",
        "a2.model_tensor_sha256",
        "a2.ema_teacher_tensor_sha256",
        *tuple(
            f"descriptors.{profile}.{field}"
            for profile in PROFILES
            for field in (
                "prediction_array_sha256",
                "trajectory_feature_array_sha256",
                "metric_signature_sha256",
                "codebook_state_sha256",
                "descriptor_transform_state_sha256",
                "activity_cluster_state_sha256",
            )
        ),
    )
    results: dict[str, Any] = {}
    for path in checks:
        values = [str(_path_value(row, path)) for row in rows]
        unique = sorted(set(values))
        results[path] = {
            "equal_across_all_repeats": len(unique) == 1,
            "unique_value_count": len(unique),
            "values_by_repeat": {
                str(int(row["repeat"])): values[index]
                for index, row in enumerate(rows)
            },
        }
    return {
        "comparison_basis": (
            "exact tensor-state SHA256, canonical ndarray-content SHA256, and exact "
            "metric-vector SHA256"
        ),
        "pt_file_sha256_used_for_within_run_integrity_validation": True,
        "pt_file_sha256_used_as_cross_repeat_pass_criterion": False,
        "all_checks_equal": all(
            bool(value["equal_across_all_repeats"]) for value in results.values()
        ),
        "checks": results,
    }


def _csv_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    result = []
    for item in rows:
        row: dict[str, Any] = {
            "repeat": int(item["repeat"]),
            "fold": FOLD,
            "seed": SEED,
            "warmup_selected_epoch": item["warmup"]["selected_epoch"],
            "warmup_selected_validation_macro_f1": item["warmup"][
                "selected_validation_macro_f1"
            ],
            "warmup_history_canonical_sha256": item["warmup"][
                "history_canonical_sha256"
            ],
            "warmup_model_tensor_sha256": item["warmup"]["model_tensor_sha256"],
            "warmup_backbone_tensor_sha256": item["warmup"][
                "backbone_tensor_sha256"
            ],
            "a2_model_tensor_sha256": item["a2"]["model_tensor_sha256"],
            "a2_selected_epoch": item["a2"]["selected_epoch"],
            "a2_selected_validation_total_loss": item["a2"][
                "selected_validation_total_loss"
            ],
            "a2_history_canonical_sha256": item["a2"][
                "history_canonical_sha256"
            ],
            "a2_ema_teacher_tensor_sha256": item["a2"][
                "ema_teacher_tensor_sha256"
            ],
        }
        for profile in PROFILES:
            profile_record = item["descriptors"][profile]
            prefix = "legacy" if profile == LEGACY_DESCRIPTOR_PROFILE else "a025"
            row[f"{prefix}_prediction_array_sha256"] = profile_record[
                "prediction_array_sha256"
            ]
            row[f"{prefix}_trajectory_feature_array_sha256"] = profile_record[
                "trajectory_feature_array_sha256"
            ]
            row[f"{prefix}_activity_cluster_state_sha256"] = profile_record[
                "activity_cluster_state_sha256"
            ]
            for metric in METRICS:
                row[f"{prefix}_{metric}"] = profile_record["metrics"][metric]
        result.append(row)
    return result


def _identity(args: argparse.Namespace, reference: Mapping[str, Any] | None) -> dict[str, Any]:
    npz = Path(args.npz_path).expanduser().resolve()
    return {
        "schema": SCHEMA,
        "purpose": "same_fold_seed_fresh_initialization_bitwise_reproducibility",
        "fold": FOLD,
        "seed": SEED,
        "repeat_count": REPEAT_COUNT,
        "npz_path": str(npz),
        "npz_sha256": sha256_file(npz),
        "route": "W128_S64_ResNet1D_window_warmup_then_A2_final_then_E0_K64",
        "profiles": list(PROFILES),
        "training": {
            "window_epochs": int(args.window_epochs),
            "window_batch_size": int(args.window_batch_size),
            "window_eval_batch_size": int(args.window_eval_batch_size),
            "window_learning_rate": float(args.window_learning_rate),
            "window_weight_decay": float(args.window_weight_decay),
            "window_weak_scale_std": float(args.window_weak_scale_std),
            "window_strong_scale_std": float(args.window_strong_scale_std),
            "window_selection_policy": "best_val_macro_f1",
            "a2_epochs": int(args.a2_epochs),
            "a2_trial_batch_size": int(args.a2_trial_batch_size),
            "a2_source_encode_batch_size": int(args.a2_source_encode_batch_size),
            "a2_learning_rate": float(args.a2_learning_rate),
            "a2_minimum_learning_rate": float(args.a2_minimum_learning_rate),
            "a2_weight_decay": float(args.a2_weight_decay),
            "fresh_random_initialization_per_repeat": True,
            "deterministic": True,
        },
        "descriptor": {
            "window_size": descriptor_cv.WINDOW_SIZE,
            "window_stride": descriptor_cv.WINDOW_STRIDE,
            "primitive_num": descriptor_cv.PRIMITIVE_NUM,
            "signed_vertical_distance_weight": float(
                args.signed_vertical_distance_weight
            ),
            "subject_nuisance_max_rank": int(args.subject_nuisance_max_rank),
            "subject_nuisance_explained_variance": float(
                args.subject_nuisance_explained_variance
            ),
            "kmeans_n_init": int(args.kmeans_n_init),
            "kmeans_max_iter": int(args.kmeans_max_iter),
            "encode_batch_size": int(args.encode_batch_size),
        },
        "device": str(args.device),
        "num_workers": int(args.num_workers),
        "reference_summary": (
            None
            if reference is None
            else {"path": reference["path"], "sha256": reference["sha256"]}
        ),
        "reproducibility_pass_policy": {
            "requires_all_five_selection_history_tensor_cluster_prediction_"
            "feature_metric_hashes_equal": True,
            "pt_file_sha256_used_as_cross_repeat_criterion": False,
            "reference_performance_used": False,
        },
        "implementation_sha256": _implementation_hashes(),
    }


def _validate_completed(root: Path, expected_identity_sha256: str) -> dict[str, Any]:
    complete_path = root / "complete.json"
    audit_path = root / "reproducibility_audit.json"
    csv_path = root / "repeat_results.csv"
    for path in (complete_path, audit_path, csv_path):
        if not path.is_file():
            raise RuntimeError(f"Completed reproducibility root lacks {path.name}.")
    complete = _read_json(complete_path)
    if (
        complete.get("schema") != SCHEMA
        or complete.get("complete") is not True
        or complete.get("passed") is not True
        or complete.get("identity_sha256") != expected_identity_sha256
    ):
        raise RuntimeError("Reproducibility completion marker is invalid.")
    expected = complete.get("artifact_sha256")
    if not isinstance(expected, Mapping):
        raise RuntimeError("Reproducibility completion lacks artifact hashes.")
    for name, value in expected.items():
        path = root / str(name)
        if not path.is_file() or sha256_file(path) != value:
            raise RuntimeError(f"Reproducibility artifact changed: {name}.")
    return _read_json(audit_path)


def run(args: argparse.Namespace) -> dict[str, Any]:
    npz = Path(args.npz_path).expanduser().resolve()
    if not npz.is_file():
        raise FileNotFoundError(npz)
    reference = _reference_record(args.reference_summary)
    root = Path(args.output_root).expanduser().resolve()
    had_artifacts = root.exists() and any(root.iterdir())
    if had_artifacts and not bool(args.resume):
        raise FileExistsError(
            f"Reproducibility output root is non-empty: {root}; use --resume or a new root."
        )
    identity = _identity(args, reference)
    validate_or_create_grid_manifest(root, identity)
    identity_sha = canonical_hash(identity)
    complete_path = root / "complete.json"
    if complete_path.is_file() and bool(args.resume):
        return _validate_completed(root, identity_sha)

    encoder_cv._validate_confirmation_hyperparameters(_encoder_namespace(args))
    repeats: list[dict[str, Any]] = []
    for repeat in range(1, REPEAT_COUNT + 1):
        repeats.append(_run_repeat(args, repeat=repeat, root=root))
        write_json(root / "partial_repeat_results.json", {"repeats": repeats})
        write_csv(root / "repeat_results.csv", _csv_rows(repeats))

    consistency = _consistency_report(repeats)
    audit = {
        "schema": SCHEMA,
        "identity_sha256": identity_sha,
        "fold": FOLD,
        "seed": SEED,
        "repeat_count": REPEAT_COUNT,
        "independent_repeat_directory_count": REPEAT_COUNT,
        "same_configuration_for_every_repeat": True,
        "passed": bool(consistency["all_checks_equal"]),
        "consistency": consistency,
        "reference_comparison": _reference_deltas(reference, repeats),
        "repeats": repeats,
    }
    write_json(root / "reproducibility_audit.json", audit)
    write_csv(root / "repeat_results.csv", _csv_rows(repeats))
    if not audit["passed"]:
        raise RuntimeError(
            "Deterministic reproducibility audit failed; machine-readable details were "
            f"written to {root / 'reproducibility_audit.json'}."
        )
    complete = {
        "schema": SCHEMA,
        "identity_sha256": identity_sha,
        "passed": True,
        "complete": True,
        "artifact_sha256": {
            "reproducibility_audit.json": sha256_file(
                root / "reproducibility_audit.json"
            ),
            "repeat_results.csv": sha256_file(root / "repeat_results.csv"),
        },
    }
    write_json(complete_path, complete)
    return audit


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Train the exact fold-1/seed-0 W128 warm-up+A2 route five times in "
            "independent directories and audit tensor/prediction/metric equality."
        )
    )
    parser.add_argument("--npz-path", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument(
        "--reference-summary",
        default=None,
        help=(
            "Optional historical alpha=0.25 summary. It is reported only as a "
            "performance delta and never affects reproducibility pass/fail."
        ),
    )
    parser.add_argument("--window-epochs", type=int, default=60)
    parser.add_argument("--window-batch-size", type=int, default=256)
    parser.add_argument("--window-eval-batch-size", type=int, default=1024)
    parser.add_argument("--window-learning-rate", type=float, default=0.1)
    parser.add_argument("--window-weight-decay", type=float, default=5e-4)
    parser.add_argument("--window-weak-scale-std", type=float, default=0.1)
    parser.add_argument("--window-strong-scale-std", type=float, default=0.2)
    parser.add_argument("--a2-epochs", type=int, default=30)
    parser.add_argument("--a2-trial-batch-size", type=int, default=8)
    parser.add_argument("--a2-source-encode-batch-size", type=int, default=1024)
    parser.add_argument("--a2-learning-rate", type=float, default=1e-4)
    parser.add_argument("--a2-minimum-learning-rate", type=float, default=1e-6)
    parser.add_argument("--a2-weight-decay", type=float, default=1e-4)
    parser.add_argument("--signed-vertical-distance-weight", type=float, default=0.15)
    parser.add_argument("--subject-nuisance-max-rank", type=int, default=4)
    parser.add_argument(
        "--subject-nuisance-explained-variance", type=float, default=0.90
    )
    parser.add_argument("--kmeans-n-init", type=int, default=50)
    parser.add_argument("--kmeans-max-iter", type=int, default=300)
    parser.add_argument("--encode-batch-size", type=int, default=1024)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resume", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> dict[str, Any]:
    result = run(build_parser().parse_args(argv))
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return result


if __name__ == "__main__":
    main()


__all__ = [
    "FOLD",
    "METRICS",
    "PROFILES",
    "REPEAT_COUNT",
    "SCHEMA",
    "SEED",
    "_consistency_report",
    "_canonical_history_sha256",
    "_raw_prediction_content",
    "_reference_deltas",
    "_stable_array_sha256",
    "build_parser",
    "run",
]
