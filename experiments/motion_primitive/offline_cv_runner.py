"""Legacy joint VQ/GRU USC-HAD offline subject-CV orchestrator.

This implementation remains available for historical reproduction but is no
longer the public HHR launcher. It is HAR-only and trajectory-only. Every run
must explicitly choose random initialization or a per-fold ResNet1D warm-start;
the project does not silently decide that unresolved experimental variable.

``--resume`` resumes the *grid*: completed members are integrity checked and
reused, while missing members are launched.  The single-run trainer does not
support unsafe mid-epoch restoration, so a non-empty incomplete member always
fails closed with an actionable error.
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
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.motion_primitive.legacy_profiles import (  # noqa: E402
    PROFILE_JOINT,
    normalize_profile_grid,
)

TRAIN_SCRIPT = Path(__file__).with_name("offline_trainer.py")
EVALUATION_SPLITS = ("validation", "outer_test")

# Frozen subject registration inherited from the audited 2026-08-30 HAR split.
# Test pairs partition subjects 1..14; validation for fold i is the next test
# pair, cyclically.  This is deliberately duplicated as immutable experiment
# metadata rather than inferred from arbitrary directory names.
REGISTERED_TEST_SUBJECTS: tuple[tuple[int, int], ...] = (
    (11, 10),
    (2, 13),
    (3, 9),
    (7, 1),
    (12, 8),
    (5, 4),
    (14, 6),
)
REGISTERED_VALIDATION_SUBJECTS: tuple[tuple[int, int], ...] = (
    (2, 13),
    (3, 9),
    (7, 1),
    (12, 8),
    (5, 4),
    (14, 6),
    (11, 10),
)

CV_SCHEMA = "hhr_motion_primitive_offline_subject_cv_v3"


@dataclass(frozen=True)
class SubjectSplit:
    fold: int
    train: tuple[int, ...]
    validation: tuple[int, ...]
    outer_test: tuple[int, ...]

    def audit_dict(self) -> dict[str, Any]:
        return {
            "fold": self.fold,
            "train_subjects": list(self.train),
            "validation_subjects": list(self.validation),
            "outer_test_subjects": list(self.outer_test),
        }


def registered_subject_split(fold: int) -> SubjectSplit:
    """Return and independently validate one registered 10/2/2 split."""

    fold = int(fold)
    if not 1 <= fold <= 7:
        raise ValueError(f"USC-HAD fold must lie in [1,7], got {fold}.")
    outer_test = tuple(sorted(REGISTERED_TEST_SUBJECTS[fold - 1]))
    validation = tuple(sorted(REGISTERED_VALIDATION_SUBJECTS[fold - 1]))
    excluded = set(outer_test) | set(validation)
    train = tuple(subject for subject in range(1, 15) if subject not in excluded)
    groups = {"train": set(train), "validation": set(validation), "outer_test": set(outer_test)}
    if any(groups[left] & groups[right] for left, right in (
        ("train", "validation"), ("train", "outer_test"), ("validation", "outer_test")
    )):
        raise RuntimeError(f"Registered fold {fold} contains subject overlap: {groups}.")
    if set().union(*groups.values()) != set(range(1, 15)):
        raise RuntimeError(f"Registered fold {fold} does not partition subjects 1..14.")
    if (len(train), len(validation), len(outer_test)) != (10, 2, 2):
        raise RuntimeError(f"Registered fold {fold} is not a 10/2/2 split.")
    return SubjectSplit(fold, train, validation, outer_test)


def parse_integer_grid(value: str, *, minimum: int, maximum: Optional[int] = None) -> tuple[int, ...]:
    raw = [token.strip() for token in str(value).split(",") if token.strip()]
    if not raw:
        raise ValueError("An experiment grid cannot be empty.")
    values = tuple(int(token) for token in raw)
    if len(values) != len(set(values)):
        raise ValueError(f"Duplicate values are forbidden in experiment grids: {values}.")
    for item in values:
        if item < minimum or (maximum is not None and item > maximum):
            interval = f"[{minimum},{maximum}]" if maximum is not None else f"[{minimum},infinity)"
            raise ValueError(f"Grid value {item} lies outside {interval}.")
    return tuple(sorted(values))


def parse_profiles(value: str) -> tuple[str, ...]:
    raw = tuple(token.strip() for token in str(value).split(",") if token.strip())
    if not raw:
        raise ValueError("--profiles cannot be empty.")
    return normalize_profile_grid(raw)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_hash(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, path)


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames), extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def find_window_pretrain_checkpoint(root: Path, fold: int, seed: int) -> Path:
    """Resolve one fold/seed ResNet1D warm-start, including historical A2."""

    root = Path(root).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Window-pretrain root does not exist: {root}.")
    patterns = (
        f"fold_{int(fold):02d}_seed_{int(seed)}_A2*/motion_encoder_best.pt",
        f"fold_{int(fold):02d}/window_pretrain/seed_{int(seed)}_offline/uschad/"
        "*/checkpoints/model_best.pt",
    )
    candidates = sorted(
        {
            path.resolve()
            for pattern in patterns
            for path in root.glob(pattern)
            if path.is_file()
        }
    )
    if len(candidates) != 1:
        raise RuntimeError(
            "Expected exactly one historical window-pretrain checkpoint for "
            f"fold={fold}, seed={seed} below {root}; found {len(candidates)} with "
            f"patterns {patterns!r}: {[str(path) for path in candidates]}."
        )
    return candidates[0]


def _csv_subjects(values: Iterable[int]) -> str:
    return ",".join(str(int(value)) for value in values)


def selection_heads_for_profile(
    profile: str, joint_head: str = "trajectory"
) -> tuple[str, ...]:
    """Return the sole task-facing validation-selection checkpoint."""

    if profile != PROFILE_JOINT:
        raise ValueError(f"Unknown profile {profile!r}.")
    if str(joint_head) != "trajectory":
        raise ValueError("HHR only selects the motion-primitive trajectory head.")
    return ("trajectory",)


def member_directory(
    output_root: Path,
    profile: str,
    fold: int,
    seed: int,
    *,
    selection_head: Optional[str] = None,
) -> Path:
    if profile != PROFILE_JOINT or selection_head not in {None, "trajectory"}:
        raise ValueError("HHR members must use motion_primitive_joint/trajectory.")
    return Path(output_root) / f"profile_{profile}" / f"fold_{int(fold):02d}_seed_{int(seed)}"


def build_member_command(
    args: argparse.Namespace,
    *,
    profile: str,
    split: SubjectSplit,
    seed: int,
    checkpoint: Optional[Path],
    output_dir: Path,
    selection_head: Optional[str] = None,
) -> list[str]:
    """Build an explicit command; no important protocol default is implicit."""

    if profile != PROFILE_JOINT:
        raise ValueError("The HHR launcher only accepts motion_primitive_joint.")
    selection_head = str(selection_head or "trajectory")
    if selection_head != "trajectory":
        raise ValueError("The HHR launcher only selects the trajectory head.")
    command = [
        str(Path(args.python_executable).expanduser()),
        str(TRAIN_SCRIPT),
        "--profile", profile,
        "--npz-path", str(Path(args.npz_path).expanduser().resolve()),
        "--output-dir", str(Path(output_dir).resolve()),
        "--train-subjects", _csv_subjects(split.train),
        "--val-subjects", _csv_subjects(split.validation),
        "--test-subjects", _csv_subjects(split.outer_test),
        "--old-classes", str(args.old_classes),
        "--total-classes", str(args.total_classes),
        "--novel-classes-per-session", str(args.novel_classes_per_session),
        "--seed", str(seed),
        "--cv-fold", str(split.fold),
        "--device", str(args.device),
        "--epochs", str(args.epochs),
        "--batch-size", str(args.batch_size),
        "--eval-batch-size", str(args.eval_batch_size),
        "--num-workers", str(args.num_workers),
        "--eval-num-workers", str(args.eval_num_workers),
        "--print-frequency", str(args.print_frequency),
        "--lr", str(args.lr),
        "--encoder-lr-scale", str(args.encoder_lr_scale),
        "--momentum", str(args.momentum),
        "--weight-decay", str(args.weight_decay),
        "--window-size", str(args.window_size),
        "--window-stride", str(args.window_stride),
        "--trial-view-mode", str(args.trial_view_mode),
        "--trial-crop-ratio", str(args.trial_crop_ratio),
        "--trial-min-windows", str(args.trial_min_windows),
        "--har-aug-mode", str(args.har_aug_mode),
        "--har-weak-jitter-std", str(args.har_weak_jitter_std),
        "--har-weak-scale-std", str(args.har_weak_scale_std),
        "--har-strong-jitter-std", str(args.har_strong_jitter_std),
        "--har-strong-scale-std", str(args.har_strong_scale_std),
        "--har-time-mask-ratio", str(args.har_time_mask_ratio),
        "--encoder-initialization", str(args.encoder_initialization),
        "--encoder-freeze-epochs", str(args.encoder_freeze_epochs),
        "--selection-head", selection_head,
        "--selection-metric", str(args.selection_metric),
        "--normalization-epsilon", str(args.normalization_epsilon),
        "--recompute-normalization",
    ]
    if args.encoder_initialization == "warmstart":
        if checkpoint is None:
            raise ValueError("Checkpoint warm-start requested without a checkpoint.")
        command.extend(
            ["--trial-encoder-checkpoint", str(Path(checkpoint).resolve())]
        )
    elif checkpoint is not None:
        raise ValueError("Random initialization must not receive a checkpoint.")
    command.extend([
        "--motion-weight", str(args.motion_weight),
        "--motion-ramp-start-epoch", str(args.motion_ramp_start_epoch),
        "--motion-ramp-end-epoch", str(args.motion_ramp_end_epoch),
        "--codebook-size", str(args.codebook_size),
        "--run-state-dim", str(args.run_state_dim),
        "--boundary-initial-bias", str(args.boundary_initial_bias),
        "--trajectory-supcon-weight", str(args.trajectory_supcon_weight),
        "--changepoint-weight", str(args.changepoint_weight),
        "--content-boundary-alignment-weight", str(
            args.content_boundary_alignment_weight
        ),
        "--noncollapse-weight", str(args.noncollapse_weight),
        "--temporal-prediction-weight", str(args.temporal_prediction_weight),
        "--temporal-prediction-mask-ratio", str(
            args.temporal_prediction_mask_ratio
        ),
        "--temporal-predictor-hidden-dim", str(
            args.temporal_predictor_hidden_dim
        ),
        "--changepoint-stable-quantile", str(args.changepoint_stable_quantile),
        "--changepoint-change-quantile", str(args.changepoint_change_quantile),
        "--changepoint-absolute-floor", str(args.changepoint_absolute_floor),
        "--changepoint-null-mad-multiplier", str(
            args.changepoint_null_mad_multiplier
        ),
        "--changepoint-rank-margin", str(args.changepoint_rank_margin),
        "--changepoint-view-consistency-weight", str(
            args.changepoint_view_consistency_weight
        ),
        "--effective-minimum-duration-weight", str(
            args.effective_minimum_duration_weight
        ),
        "--utilization-weight", str(args.utilization_weight),
        "--assignment-confidence-weight", str(args.assignment_confidence_weight),
        "--codebook-diversity-weight", str(args.codebook_diversity_weight),
        "--transition-budget-weight", str(args.transition_budget_weight),
    ])
    return command


def _command_value(command: Sequence[str], option: str) -> str:
    positions = [index for index, token in enumerate(command) if token == option]
    if len(positions) != 1 or positions[0] + 1 >= len(command):
        raise RuntimeError(f"Command does not contain exactly one value for {option}: {command}.")
    return str(command[positions[0] + 1])


def _build_identity(
    args: argparse.Namespace,
    profiles: Sequence[str],
    folds: Sequence[int],
    seeds: Sequence[int],
    checkpoints: Mapping[tuple[int, int], Path],
) -> dict[str, Any]:
    parameters = {
        key: value
        for key, value in vars(args).items()
        if key not in {"skip_existing", "resume", "dry_run"}
    }
    parameters["npz_path"] = str(Path(args.npz_path).expanduser().resolve())
    parameters["window_pretrain_root"] = (
        str(Path(args.window_pretrain_root).expanduser().resolve())
        if args.window_pretrain_root is not None
        else None
    )
    parameters["output_root"] = str(Path(args.output_root).expanduser().resolve())
    checkpoint_rows = [
        {
            "fold": fold,
            "seed": seed,
            "path": str(path.resolve()),
            "sha256": _sha256_file(path),
        }
        for (fold, seed), path in sorted(checkpoints.items())
    ]
    payload: dict[str, Any] = {
        "schema": CV_SCHEMA,
        "profiles": list(profiles),
        "folds": list(folds),
        "seeds": list(seeds),
        "registered_splits": [registered_subject_split(fold).audit_dict() for fold in folds],
        "selection_checkpoints": {
            profile: [f"best_{head}" for head in selection_heads_for_profile(profile)]
            for profile in profiles
        },
        "parameters": parameters,
        "encoder_initialization": str(args.encoder_initialization),
        "window_pretrain_checkpoints": checkpoint_rows,
        "train_script": str(TRAIN_SCRIPT.resolve()),
        "train_script_sha256": _sha256_file(TRAIN_SCRIPT),
    }
    payload["identity_sha256"] = _canonical_hash(payload)
    return payload


def _validated_existing_cv_manifest(path: Path, identity: Mapping[str, Any]) -> None:
    try:
        recorded = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"Cannot read existing CV manifest {path}: {error}.") from error
    claimed_hash = recorded.get("identity_sha256")
    unsigned = {key: value for key, value in recorded.items() if key != "identity_sha256"}
    if claimed_hash != _canonical_hash(unsigned):
        raise RuntimeError(f"Existing CV manifest failed its integrity hash: {path}.")
    if claimed_hash != identity.get("identity_sha256"):
        raise RuntimeError(
            "Output root records a different profile/fold/seed/protocol/checkpoint identity; "
            "use a new --output-root."
        )


def _expected_member_arguments(
    command: Sequence[str], profile: str, split: SubjectSplit, seed: int
) -> dict[str, Any]:
    expected = {
        "profile": profile,
        "seed": int(seed),
        "uschad_cv_fold": int(split.fold),
        "uschad_train_subjects": _csv_subjects(split.train),
        "offline_val_subjects": _csv_subjects(split.validation),
        "uschad_test_subjects": _csv_subjects(split.outer_test),
        "uschad_npz_path": _command_value(command, "--npz-path"),
        "encoder_initialization": _command_value(
            command, "--encoder-initialization"
        ),
        "selection_head": _command_value(command, "--selection-head"),
        "selection_metric": _command_value(command, "--selection-metric"),
    }
    if expected["encoder_initialization"] == "warmstart":
        expected["trial_encoder_checkpoint"] = _command_value(
            command, "--trial-encoder-checkpoint"
        )
    else:
        expected["trial_encoder_checkpoint"] = ""
    return expected


def validate_completed_member(
    run_dir: Path,
    *,
    command: Sequence[str],
    profile: str,
    split: SubjectSplit,
    seed: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Load a complete member and reject stale or mixed protocol artifacts."""

    required = ["manifest.json", "summary.json", "checkpoint_best.pt", "checkpoint_last.pt"]
    required.extend(
        f"checkpoint_best_{head}.pt"
        for head in selection_heads_for_profile(profile, _command_value(command, "--selection-head"))
    )
    missing = [name for name in required if not (run_dir / name).is_file()]
    if missing:
        raise RuntimeError(f"Incomplete run {run_dir}; missing {missing}.")
    try:
        manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
        summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"Invalid completed run metadata below {run_dir}: {error}.") from error
    if manifest.get("schema") != "hhr_motion_primitive_trajectory_manifest_v3":
        raise RuntimeError(f"Unexpected member manifest schema in {run_dir}: {manifest.get('schema')!r}.")
    if summary.get("schema") != "hhr_motion_primitive_trajectory_summary_v3":
        raise RuntimeError(f"Unexpected member summary schema in {run_dir}: {summary.get('schema')!r}.")
    expected = _expected_member_arguments(command, profile, split, seed)
    arguments = manifest.get("arguments")
    if not isinstance(arguments, Mapping):
        raise RuntimeError(f"Member manifest lacks arguments: {run_dir}.")
    mismatches = {
        key: {"expected": value, "observed": arguments.get(key)}
        for key, value in expected.items()
        if arguments.get(key) != value
    }
    if manifest.get("profile") != profile or summary.get("profile") != profile:
        mismatches["profile_metadata"] = {
            "expected": profile,
            "observed": [manifest.get("profile"), summary.get("profile")],
        }
    expected_primary_head = _command_value(command, "--selection-head")
    if summary.get("selection_head") != expected_primary_head:
        mismatches["primary_selection_head"] = {
            "expected": expected_primary_head,
            "observed": summary.get("selection_head"),
        }
    if summary.get("test_metrics_used_for_selection") is not False:
        mismatches["test_metrics_used_for_selection"] = {
            "expected": False,
            "observed": summary.get("test_metrics_used_for_selection"),
        }
    evaluations = summary.get("evaluations_by_selection")
    expected_selection_heads = set(
        selection_heads_for_profile(profile, _command_value(command, "--selection-head"))
    )
    if not isinstance(evaluations, Mapping) or set(evaluations) != expected_selection_heads:
        mismatches["evaluations_by_selection"] = {
            "expected": sorted(expected_selection_heads),
            "observed": sorted(evaluations) if isinstance(evaluations, Mapping) else evaluations,
        }
    elif any(
        not isinstance(evaluation, Mapping)
        or evaluation.get("selection_head") != head
        or evaluation.get("selection_metric") != "macro_f1"
        or evaluation.get("test_metrics_used_for_selection") is not False
        for head, evaluation in evaluations.items()
    ):
        mismatches["evaluation_selection_contract"] = {
            "expected": "validation-only macro_f1 for every named head",
            "observed": evaluations,
        }
    if summary.get("test_evaluation_count") != len(expected_selection_heads):
        mismatches["test_evaluation_count"] = {
            "expected": len(expected_selection_heads),
            "observed": summary.get("test_evaluation_count"),
        }
    if mismatches:
        raise RuntimeError(f"Existing run {run_dir} is incompatible: {mismatches}.")
    return manifest, summary


def _member_state(run_dir: Path) -> str:
    if not run_dir.exists():
        return "missing"
    if not run_dir.is_dir():
        return "invalid"
    if not any(run_dir.iterdir()):
        return "empty"
    required = ("manifest.json", "summary.json", "checkpoint_best.pt", "checkpoint_last.pt")
    return "complete" if all((run_dir / name).is_file() for name in required) else "incomplete"


def _metric_row(
    *,
    profile: str,
    split: SubjectSplit,
    seed: int,
    evaluation_split: str,
    manifest: Mapping[str, Any],
    summary: Mapping[str, Any],
    summary_path: Path,
    report_head: str,
    selection_head: str,
) -> dict[str, Any]:
    evaluations = summary.get("evaluations_by_selection")
    if not isinstance(evaluations, Mapping) or selection_head not in evaluations:
        raise RuntimeError(
            f"Run {summary_path.parent} lacks validation selection {selection_head!r}."
        )
    selection = evaluations[selection_head]
    if selection.get("test_metrics_used_for_selection") is not False:
        raise RuntimeError(f"Run {summary_path.parent} used outer-test metrics for selection.")
    metrics_container = (
        selection["best_validation"] if evaluation_split == "validation" else selection["test"]
    )
    heads = metrics_container.get("heads")
    if not isinstance(heads, Mapping) or report_head not in heads:
        raise RuntimeError(
            f"Run {summary_path.parent} lacks report head {report_head!r} for {evaluation_split}."
        )
    metrics = heads[report_head]
    architecture = manifest.get("architecture", {})
    arguments = manifest.get("arguments", {})
    dataset_key = "validation" if evaluation_split == "validation" else "test"
    dataset = manifest.get("datasets", {}).get(dataset_key, {})
    old_classes = [int(value) for value in manifest.get("old_classes_physical", [])]
    per_class = metrics.get("per_class", {})
    if not old_classes or set(per_class) != {str(index) for index in range(len(old_classes))}:
        raise RuntimeError(f"Incomplete per-class metrics in {summary_path} for {evaluation_split}.")
    row: dict[str, Any] = {
        "profile": profile,
        "report_head": report_head,
        "checkpoint_selection": f"best_{selection_head}_validation_macro_f1",
        "fold": split.fold,
        "seed": int(seed),
        "evaluation_split": evaluation_split,
        "train_subjects": _csv_subjects(split.train),
        "validation_subjects": _csv_subjects(split.validation),
        "outer_test_subjects": _csv_subjects(split.outer_test),
        "sample_unit": "motion_primitive_trajectory",
        "representation": "variable_length_motion_primitive_trajectory",
        "view_mode": arguments.get("trial_view_mode"),
        "window_size": architecture.get("window_size"),
        "window_stride": architecture.get("window_stride"),
        "tail_policy": architecture.get("tail_policy"),
        "normalization_eps": arguments.get("uschad_norm_eps"),
        "normalization_mode": dataset.get("normalization_mode"),
        "normalization_stat_subjects": _csv_subjects(dataset.get("normalization_stat_subjects", [])),
        "normalization_stat_classes": _csv_subjects(dataset.get("normalization_stat_classes", [])),
        "selection_epoch": int(selection["best_epoch"]),
        "selection_metric": selection["selection_metric"],
        "selection_score": float(selection["best_validation_score"]),
        "num_samples": int(metrics["sample_count"]),
        "overall_accuracy": float(metrics["accuracy"]),
        "mean_class_accuracy": float(metrics["balanced_accuracy"]),
        "macro_f1": float(metrics["macro_f1"]),
        "confusion_matrix": json.dumps(metrics["confusion_matrix"], separators=(",", ":")),
        "trial_id_sha256": metrics_container.get("trial_id_sha256"),
        "dataset_identity_sha256": dataset.get("identity_sha256"),
        "npz_sha256": manifest.get("npz_sha256"),
        "encoder_initialization": manifest.get("encoder_initialization", {}).get("mode"),
        "encoder_freeze_epochs": manifest.get("encoder_initialization", {}).get("freeze_epochs"),
        "encoder_checkpoint": manifest.get("encoder_initialization", {}).get("checkpoint"),
        "encoder_checkpoint_sha256": manifest.get("encoder_initialization", {}).get("checkpoint_sha256"),
        "metrics_path": str(summary_path.resolve()),
    }
    for head_name, head_metrics in sorted(heads.items()):
        row[f"head_{head_name}_overall_accuracy"] = float(head_metrics["accuracy"])
        row[f"head_{head_name}_mean_class_accuracy"] = float(head_metrics["balanced_accuracy"])
        row[f"head_{head_name}_macro_f1"] = float(head_metrics["macro_f1"])
    for class_index, physical_label in enumerate(old_classes):
        values = per_class[str(class_index)]
        row[f"class_{class_index}_physical_label"] = physical_label
        row[f"class_{class_index}_accuracy"] = float(values["recall"])
        row[f"class_{class_index}_precision"] = float(values["precision"])
        row[f"class_{class_index}_f1"] = float(values["f1"])
        row[f"class_{class_index}_support"] = int(values["support"])
    return row


def validate_result_grid(
    rows: Sequence[Mapping[str, Any]], folds: Sequence[int], seeds: Sequence[int]
) -> None:
    expected = {
        (int(fold), int(seed), evaluation_split)
        for fold in folds
        for seed in seeds
        for evaluation_split in EVALUATION_SPLITS
    }
    observed_list = [
        (int(row["fold"]), int(row["seed"]), str(row["evaluation_split"])) for row in rows
    ]
    observed = set(observed_list)
    duplicates = sorted(key for key in observed if observed_list.count(key) != 1)
    missing = sorted(expected - observed)
    unexpected = sorted(observed - expected)
    if duplicates or missing or unexpected:
        raise RuntimeError(
            "Incomplete or ambiguous offline result grid: "
            f"duplicates={duplicates}, missing={missing}, unexpected={unexpected}."
        )


def aggregate_rows(rows: Sequence[Mapping[str, Any]], folds: Sequence[int], seeds: Sequence[int]) -> dict[str, Any]:
    """Aggregate seeds inside folds and folds with equal weight."""

    validate_result_grid(rows, folds, seeds)
    result: dict[str, Any] = {
        "schema": "hhr_motion_primitive_offline_subject_cv_summary_v3",
        "statistical_unit": "held-out subject fold after averaging seeds",
        "fold_count": len(folds),
        "seed_count": len(seeds),
        "splits": {},
    }
    for evaluation_split in EVALUATION_SPLITS:
        split_rows = [row for row in rows if row["evaluation_split"] == evaluation_split]
        metrics: dict[str, Any] = {}
        for metric in ("overall_accuracy", "mean_class_accuracy", "macro_f1"):
            fold_means = {
                int(fold): sum(
                    float(row[metric]) for row in split_rows if int(row["fold"]) == int(fold)
                ) / len(seeds)
                for fold in folds
            }
            metrics[metric] = {
                "mean": sum(fold_means.values()) / len(fold_means),
                "fold_means": {str(key): value for key, value in sorted(fold_means.items())},
            }
        result["splits"][evaluation_split] = metrics
    return result


def collect_profile_results(
    output_root: Path,
    profile: str,
    folds: Sequence[int],
    seeds: Sequence[int],
    commands: Mapping[tuple[str, int, int], Sequence[str]],
    *,
    selection_head: str,
    report_head: Optional[str] = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    report_head = str(report_head or selection_head)
    profile_root = Path(output_root) / f"profile_{profile}"
    member_root = profile_root
    expected_summary_paths = {
        (
            member_directory(
                output_root, profile, fold, seed, selection_head=selection_head
            ) / "summary.json"
        ).resolve()
        for fold in folds
        for seed in seeds
    }
    observed_summary_paths = {
        path.resolve() for path in member_root.rglob("summary.json") if path.is_file()
    }
    unexpected_paths = sorted(observed_summary_paths - expected_summary_paths)
    if unexpected_paths:
        raise RuntimeError(
            f"Unexpected summary files below {member_root}; refusing ambiguous aggregation: "
            f"{[str(path) for path in unexpected_paths]}."
        )
    rows: list[dict[str, Any]] = []
    for fold in folds:
        split = registered_subject_split(fold)
        for seed in seeds:
            run_dir = member_directory(
                output_root, profile, fold, seed, selection_head=selection_head
            )
            manifest, summary = validate_completed_member(
                run_dir,
                command=commands[(profile, fold, seed)],
                profile=profile,
                split=split,
                seed=seed,
            )
            summary_path = run_dir / "summary.json"
            for evaluation_split in EVALUATION_SPLITS:
                rows.append(_metric_row(
                    profile=profile,
                    split=split,
                    seed=seed,
                    evaluation_split=evaluation_split,
                    manifest=manifest,
                    summary=summary,
                    summary_path=summary_path,
                    report_head=report_head,
                    selection_head=selection_head,
                ))
    validate_result_grid(rows, folds, seeds)
    rows.sort(key=lambda row: (int(row["fold"]), int(row["seed"]), EVALUATION_SPLITS.index(row["evaluation_split"])))
    return rows, aggregate_rows(rows, folds, seeds)


def _fieldnames(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    preferred = [
        "profile", "report_head", "checkpoint_selection", "fold", "seed", "evaluation_split",
        "train_subjects", "validation_subjects", "outer_test_subjects",
        "sample_unit", "representation", "view_mode", "window_size", "window_stride",
        "tail_policy", "normalization_eps", "normalization_mode",
        "normalization_stat_subjects", "normalization_stat_classes",
        "selection_epoch", "selection_metric", "selection_score", "num_samples",
        "overall_accuracy", "mean_class_accuracy", "macro_f1", "confusion_matrix",
        "trial_id_sha256", "dataset_identity_sha256", "npz_sha256",
        "encoder_initialization", "encoder_freeze_epochs", "encoder_checkpoint",
        "encoder_checkpoint_sha256", "metrics_path",
    ]
    available = set().union(*(row.keys() for row in rows))
    class_fields = sorted(
        available - set(preferred),
        key=lambda name: (
            int(name.split("_")[1]) if name.startswith("class_") else 10**9,
            name,
        ),
    )
    return [name for name in preferred if name in available] + class_fields


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Registered USC-HAD 7-fold single-stage primitive-joint runner",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--npz-path", required=True)
    parser.add_argument(
        "--encoder-initialization",
        choices=("random", "warmstart"),
        required=True,
        help="required experimental decision: random or ResNet1D warm-start",
    )
    parser.add_argument(
        "--encoder-warmstart-root",
        "--window-pretrain-root",
        dest="window_pretrain_root",
        default=None,
        help="required only with --encoder-initialization warmstart",
    )
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--profiles", default=PROFILE_JOINT)
    parser.add_argument("--folds", default="1,2,3,4,5,6,7")
    parser.add_argument("--seeds", default="0,5,50,500")
    parser.add_argument("--python-executable", default=sys.executable)
    parser.add_argument("--device", default="cuda")

    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--eval-num-workers", type=int, default=0)
    parser.add_argument("--print-frequency", type=int, default=10)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--encoder-lr-scale", type=float, default=1.0)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=5.0e-4)
    parser.add_argument("--encoder-freeze-epochs", type=int, default=0)

    parser.add_argument("--window-size", type=int, choices=(256,), default=256)
    parser.add_argument("--window-stride", type=int, choices=(128,), default=128)
    parser.add_argument("--trial-view-mode", choices=("full_full",), default="full_full")
    parser.add_argument("--trial-crop-ratio", type=float, default=2.0 / 3.0)
    parser.add_argument("--trial-min-windows", type=int, default=2)
    parser.add_argument("--har-aug-mode", choices=("none", "weak_strong"), default="weak_strong")
    parser.add_argument(
        "--har-weak-jitter-std", type=float, choices=(0.0,), default=0.0
    )
    parser.add_argument("--har-weak-scale-std", type=float, choices=(0.0,), default=0.0)
    parser.add_argument("--har-strong-jitter-std", type=float, default=0.0)
    parser.add_argument("--har-strong-scale-std", type=float, default=0.20)
    parser.add_argument("--har-time-mask-ratio", type=float, default=0.0)

    parser.add_argument("--selection-metric", choices=("macro_f1",), default="macro_f1")
    parser.add_argument("--normalization-epsilon", type=float, default=1.0e-6)
    parser.add_argument("--old-classes", default="0,1,2,3,4,5")
    parser.add_argument("--total-classes", type=int, default=12)
    parser.add_argument("--novel-classes-per-session", type=int, default=2)

    parser.add_argument("--motion-weight", type=float, default=1.0)
    parser.add_argument("--motion-ramp-start-epoch", type=int, default=0)
    parser.add_argument("--motion-ramp-end-epoch", type=int, default=0)
    parser.add_argument("--codebook-size", type=int, default=32)
    parser.add_argument("--run-state-dim", type=int, default=12)
    parser.add_argument("--boundary-initial-bias", type=float, default=-1.5)
    parser.add_argument("--trajectory-supcon-weight", type=float, default=0.25)
    parser.add_argument("--changepoint-weight", type=float, default=1.0)
    parser.add_argument("--content-boundary-alignment-weight", type=float, default=0.10)
    parser.add_argument("--noncollapse-weight", type=float, default=0.05)
    parser.add_argument("--temporal-prediction-weight", type=float, default=0.50)
    parser.add_argument("--temporal-prediction-mask-ratio", type=float, default=0.20)
    parser.add_argument("--temporal-predictor-hidden-dim", type=int, default=0)
    parser.add_argument("--changepoint-stable-quantile", type=float, default=0.25)
    parser.add_argument("--changepoint-change-quantile", type=float, default=0.75)
    parser.add_argument("--changepoint-absolute-floor", type=float, default=0.01)
    parser.add_argument("--changepoint-null-mad-multiplier", type=float, default=3.0)
    parser.add_argument("--changepoint-rank-margin", type=float, default=0.20)
    parser.add_argument("--changepoint-view-consistency-weight", type=float, default=0.50)
    parser.add_argument("--effective-minimum-duration-weight", type=float, default=0.02)
    parser.add_argument("--utilization-weight", type=float, default=0.0)
    parser.add_argument("--assignment-confidence-weight", type=float, default=0.0)
    parser.add_argument("--codebook-diversity-weight", type=float, default=0.0)
    parser.add_argument("--transition-budget-weight", type=float, default=0.02)

    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def validate_args(args: argparse.Namespace) -> argparse.Namespace:
    if args.skip_existing and args.resume:
        raise ValueError("--skip-existing and --resume are aliases at grid level; choose one.")
    for name in (
        "epochs", "batch_size", "eval_batch_size", "window_size", "window_stride",
        "encoder_freeze_epochs", "total_classes", "novel_classes_per_session", "codebook_size",
    ):
        if int(getattr(args, name)) < (0 if name == "encoder_freeze_epochs" else 1):
            raise ValueError(f"--{name.replace('_', '-')} has an invalid value.")
    for name in ("num_workers", "eval_num_workers", "print_frequency"):
        if int(getattr(args, name)) < 0:
            raise ValueError(f"--{name.replace('_', '-')} cannot be negative.")
    for name in ("lr", "encoder_lr_scale", "momentum", "weight_decay", "normalization_epsilon"):
        value = float(getattr(args, name))
        if not math.isfinite(value) or value < 0.0 or (name in {"lr", "encoder_lr_scale", "normalization_epsilon"} and value == 0.0):
            raise ValueError(f"--{name.replace('_', '-')} must be finite and valid.")
    if parse_profiles(args.profiles) != (PROFILE_JOINT,):
        raise ValueError("HHR exposes exactly one profile: motion_primitive_joint.")
    if float(args.motion_weight) <= 0.0:
        raise ValueError("The joint profile requires --motion-weight > 0.")
    if (int(args.motion_ramp_start_epoch), int(args.motion_ramp_end_epoch)) != (0, 0):
        raise ValueError(
            "The registered trajectory-only A2-MP route requires motion ramp "
            "start=end=0."
        )
    if int(args.run_state_dim) < 0:
        raise ValueError("--run-state-dim cannot be negative.")
    if int(args.temporal_predictor_hidden_dim) < 0:
        raise ValueError("--temporal-predictor-hidden-dim cannot be negative.")
    if (int(args.window_size), int(args.window_stride)) != (256, 128):
        raise ValueError("The registered E0 route requires window/stride 256/128.")
    if args.trial_view_mode != "full_full":
        raise ValueError("A2-MP requires aligned full-trial views.")
    if float(args.har_weak_jitter_std) != 0.0 or float(
        args.har_weak_scale_std
    ) != 0.0:
        raise ValueError(
            "A2-MP view[0] is the clean changepoint anchor; weak jitter and "
            "weak scaling must both be exactly zero."
        )
    for name in (
        "temporal_prediction_mask_ratio",
        "changepoint_stable_quantile",
        "changepoint_change_quantile",
    ):
        value = float(getattr(args, name))
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"--{name.replace('_', '-')} must lie in [0,1].")
    if not float(args.changepoint_stable_quantile) < float(
        args.changepoint_change_quantile
    ):
        raise ValueError("Stable changepoint quantile must be below change quantile.")
    if not math.isfinite(float(args.boundary_initial_bias)):
        raise ValueError("--boundary-initial-bias must be finite.")
    if not 0.0 <= float(args.changepoint_absolute_floor) <= 2.0:
        raise ValueError("--changepoint-absolute-floor must lie in [0,2].")
    for name in (
        "trajectory_supcon_weight",
        "changepoint_weight",
        "content_boundary_alignment_weight",
        "noncollapse_weight",
        "temporal_prediction_weight",
        "changepoint_rank_margin",
        "changepoint_view_consistency_weight",
        "changepoint_null_mad_multiplier",
        "effective_minimum_duration_weight",
        "utilization_weight",
        "assignment_confidence_weight",
        "codebook_diversity_weight",
        "transition_budget_weight",
    ):
        value = float(getattr(args, name))
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(
                f"--{name.replace('_', '-')} must be finite and non-negative."
            )
    if not Path(args.npz_path).expanduser().is_file():
        raise FileNotFoundError(f"USC-HAD NPZ not found: {args.npz_path}.")
    if not Path(args.python_executable).expanduser().is_file():
        raise FileNotFoundError(f"Python executable not found: {args.python_executable}.")
    if args.encoder_initialization == "random":
        if not math.isclose(
            float(args.encoder_lr_scale), 1.0, rel_tol=0.0, abs_tol=1.0e-12
        ):
            raise ValueError(
                "Random ResNet1D initialization requires --encoder-lr-scale 1.0."
            )
        if int(args.encoder_freeze_epochs) != 0:
            raise ValueError(
                "The single-stage random route requires --encoder-freeze-epochs 0."
            )
        if args.window_pretrain_root is not None:
            raise ValueError(
                "--window-pretrain-root is only valid with "
                "--encoder-initialization warmstart."
            )
    else:
        if args.window_pretrain_root is None:
            raise ValueError(
                "Warm-start initialization requires --window-pretrain-root."
            )
        if not Path(args.window_pretrain_root).expanduser().is_dir():
            raise FileNotFoundError(
                f"Window-pretrain root not found: {args.window_pretrain_root}."
            )
    return args


def run(args: argparse.Namespace) -> dict[str, Any]:
    args = validate_args(args)
    profiles = parse_profiles(args.profiles)
    folds = parse_integer_grid(args.folds, minimum=1, maximum=7)
    seeds = parse_integer_grid(args.seeds, minimum=0)
    checkpoints = (
        {
            (fold, seed): find_window_pretrain_checkpoint(
                args.window_pretrain_root, fold, seed
            )
            for fold in folds
            for seed in seeds
        }
        if args.encoder_initialization == "warmstart"
        else {}
    )
    identity = _build_identity(args, profiles, folds, seeds, checkpoints)
    output_root = Path(args.output_root).expanduser().resolve()
    manifest_path = output_root / "cv_manifest.json"
    if output_root.exists() and any(output_root.iterdir()):
        if not manifest_path.is_file():
            raise RuntimeError(f"Non-empty output root has no CV manifest: {output_root}.")
        _validated_existing_cv_manifest(manifest_path, identity)

    commands: dict[tuple[str, int, int], list[str]] = {}
    statuses: dict[tuple[str, int, int], str] = {}
    for profile in profiles:
        for fold in folds:
            split = registered_subject_split(fold)
            for seed in seeds:
                selection_head = "trajectory"
                run_dir = member_directory(
                    output_root,
                    profile,
                    fold,
                    seed,
                    selection_head=selection_head,
                )
                command = build_member_command(
                    args,
                    profile=profile,
                    split=split,
                    seed=seed,
                    checkpoint=checkpoints.get((fold, seed)),
                    output_dir=run_dir,
                    selection_head=selection_head,
                )
                key = (profile, fold, seed)
                commands[key] = command
                statuses[key] = _member_state(run_dir)

    reuse_existing = bool(args.skip_existing or args.resume)
    for key, status in statuses.items():
        if status == "complete":
            profile, fold, seed = key
            selection_head = "trajectory"
            validate_completed_member(
                member_directory(
                    output_root,
                    profile,
                    fold,
                    seed,
                    selection_head=selection_head,
                ),
                command=commands[key],
                profile=profile,
                split=registered_subject_split(fold),
                seed=seed,
            )
            if not reuse_existing:
                raise FileExistsError(
                    f"Completed member already exists for {key}; use --skip-existing/--resume or a new output root."
                )
        elif status in {"incomplete", "invalid"}:
            raise RuntimeError(
                f"Cannot safely resume {status} member {key} at "
                f"{member_directory(output_root, key[0], key[1], key[2])}. "
                "Preserve it for diagnosis, then use a clean member/output root."
            )

    for key in sorted(commands):
        state = statuses[key]
        action = "skip" if state == "complete" else "run"
        print(f"[{action}] {subprocess.list2cmdline(commands[key])}", flush=True)
    if args.dry_run:
        return {
            "dry_run": True,
            "identity": identity,
            "commands": [commands[key] for key in sorted(commands)],
            "statuses": {"|".join(map(str, key)): value for key, value in statuses.items()},
        }

    if not manifest_path.exists():
        output_root.mkdir(parents=True, exist_ok=True)
        _write_json(manifest_path, identity)

    for key in sorted(commands):
        if statuses[key] == "complete":
            continue
        subprocess.run(commands[key], cwd=PROJECT_ROOT, check=True)

    output_tables: dict[str, str] = {}
    aggregate_paths: dict[str, str] = {}
    for profile in profiles:
        profile_root = output_root / f"profile_{profile}"
        for selection_head in selection_heads_for_profile(profile):
            report_head = "trajectory"
            rows, aggregate = collect_profile_results(
                output_root,
                profile,
                folds,
                seeds,
                commands,
                selection_head=selection_head,
                report_head=report_head,
            )
            is_primary = profile == PROFILE_JOINT and selection_head == "trajectory"
            if is_primary:
                csv_path = profile_root / "subject_cv_runs.csv"
                aggregate_path = profile_root / "subject_cv_summary.json"
                table_key = profile
            else:
                # This name deliberately does not end in subject_cv_runs.csv;
                # passing the file itself to the parity loader remains valid,
                # while recursive profile-root discovery stays unambiguous.
                csv_path = profile_root / f"joint_at_best_{selection_head}.csv"
                aggregate_path = profile_root / f"joint_at_best_{selection_head}_summary.json"
                table_key = f"{profile}@best_{selection_head}"
            _write_csv(csv_path, rows, _fieldnames(rows))
            _write_json(aggregate_path, aggregate)
            output_tables[table_key] = str(csv_path.resolve())
            aggregate_paths[table_key] = str(aggregate_path.resolve())
    return {
        "dry_run": False,
        "identity": identity,
        "subject_cv_runs": output_tables,
        "summaries": aggregate_paths,
    }


def main(argv: Optional[Sequence[str]] = None) -> dict[str, Any]:
    result = run(build_parser().parse_args(argv))
    if not result["dry_run"]:
        for profile, path in result["subject_cv_runs"].items():
            print(f"[completed] profile={profile} table={path}", flush=True)
    return result


if __name__ == "__main__":
    main()
