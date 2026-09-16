"""Run the four fixed-trajectory motion-primitive readout ablations.

This is deliberately a post-processing experiment.  It consumes completed
KMeans32 change-point runs and never refits the encoder, segmentation model,
change-point threshold, PCA, or codebook.  Legacy masked-adapter runs and the
new motion-encoder runs use separate, explicit protocol identifiers.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from experiments.motion_primitive.core import EPS
from experiments.motion_primitive.motion_checkpoint import COMMAND_ARGUMENTS_V1
from experiments.motion_primitive.trajectory_ablation import (
    GROUP_NAMES,
    STATE_FEATURE_NAMES,
    SourceSignalRepository,
    build_trajectories,
    codebook_dynamic_cost,
    evaluate_distance_matrix,
    fit_local_scales,
    fit_state_scaler,
    jsonable,
    resolve_source_npz,
    run_controls,
    sha256_file,
    stable_text_hash,
    total_duration_distance_matrix,
    trajectory_distance_matrix,
    transform_trial_states,
    write_json,
)


RUN_PATTERN = re.compile(r"fold_(\d+)_seed_(\d+)_k(\d+)")
CANONICAL_RUN_PATTERN = re.compile(
    r"^fold_(\d{2})_seed_(0|5|50|500)_k32_ssl_feature_changepoint$"
)
MOTION_ENCODER_RUN_PATTERN = re.compile(
    r"^fold_(\d{2})_seed_(\d+)_k32_motion_encoder_changepoint(?:_(?:a[0-4]|custom))?$"
)
FIXED_WINDOW_RUN_PATTERN = re.compile(r"^fold_(\d{2})_seed_(\d+)_k(\d+)$")
MOTION_FIXED_WINDOW_RUN_PATTERN = re.compile(
    r"^fold_(\d{2})_seed_(\d+)_k(\d+)_motion_encoder_fixed_window(?:_(?:a[0-4]|custom))?$"
)
CANONICAL_FOLDS = list(range(1, 8))
CANONICAL_SEEDS = [0, 5, 50, 500]
CANONICAL_SOURCE_NPZ_SHA256 = (
    "7947db1f4a18dee045a80785281f5e5148270d998983895be2a46c0e8b37eefe"
)
CRITICAL_ENCODER_ARGUMENTS = COMMAND_ARGUMENTS_V1


def parse_int_list(value: str) -> list[int]:
    return sorted(
        set(
            int(token)
            for token in re.split(r"[\s,]+", str(value).strip())
            if token
        )
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Four-group trajectory-distance ablation on fixed SSL change-point "
            "segments and KMeans tokens."
        )
    )
    parser.add_argument("--input-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument(
        "--input-protocol",
        choices=["legacy_ssl_v1", "motion_encoder_v1"],
        default="legacy_ssl_v1",
        help=(
            "Explicitly identifies the immutable feature/segmentation protocol. "
            "motion_encoder_v1 remains an exploratory protocol and cannot be "
            "silently promoted through the legacy canonical gate."
        ),
    )
    parser.add_argument(
        "--expected-encoder-profile",
        choices=["A0", "A1", "A2", "A3", "A4", "CUSTOM", "a0", "a1", "a2", "a3", "a4", "custom"],
        default="",
        help=(
            "Required for motion_encoder_v1; every checkpoint must record this "
            "same A0-A4/CUSTOM training profile."
        ),
    )
    parser.add_argument(
        "--fixed-window-root",
        default="",
        help=(
            "Root of the paired fixed-window KMeans32 runs. It is required "
            "for canonical promotion and replaces a hard-coded reference score."
        ),
    )
    parser.add_argument(
        "--npz-path",
        default="",
        help="Optional native path override for the source USC-HAD window NPZ.",
    )
    parser.add_argument("--folds", default="1,2,3,4,5,6,7")
    parser.add_argument("--seeds", default="0,5,50,500")
    parser.add_argument("--expected-runs", type=int, default=28)
    parser.add_argument("--state-weight", type=float, default=0.25)
    parser.add_argument("--context-weight", type=float, default=0.15)
    parser.add_argument("--duration-weight", type=float, default=0.15)
    parser.add_argument("--control-shuffles", type=int, default=50)
    parser.add_argument("--sample-rate-hz", type=float, default=100.0)
    parser.add_argument("--old-class-count", type=int, default=6)
    parser.add_argument("--seed", type=int, default=20260902)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if not 0.0 <= args.state_weight <= 1.0:
        raise ValueError("--state-weight must lie in [0,1].")
    if args.context_weight < 0 or args.duration_weight < 0:
        raise ValueError("Context and duration weights must be non-negative.")
    if args.context_weight + args.duration_weight > 1.0:
        raise ValueError("Context and duration weights must sum to at most 1.")
    if args.control_shuffles < 0:
        raise ValueError("--control-shuffles must be non-negative.")
    if args.sample_rate_hz <= 0 or args.old_class_count <= 0:
        raise ValueError("Sample rate and old-class count must be positive.")
    if (
        args.input_protocol == "motion_encoder_v1"
        and not str(args.expected_encoder_profile).strip()
    ):
        raise ValueError(
            "--expected-encoder-profile is required for motion_encoder_v1."
        )


def discover_runs(input_root: Path, folds: list[int], seeds: list[int]) -> list[dict]:
    fold_set = set(folds)
    seed_set = set(seeds)
    runs = []
    for directory in sorted(input_root.iterdir()):
        if not directory.is_dir():
            continue
        match = RUN_PATTERN.search(directory.name)
        if match is None:
            continue
        fold, seed, primitive_num = (int(value) for value in match.groups())
        if fold not in fold_set or seed not in seed_set:
            continue
        required = [
            "primitive_codebook.npz",
            "segment_embeddings_and_tokens.npz",
            "trial_primitive_sequences.jsonl",
            "sequence_association_metrics.json",
            "experiment_config.json",
        ]
        missing = [name for name in required if not (directory / name).exists()]
        if missing:
            raise RuntimeError(f"Run {directory} is missing files: {missing}")
        runs.append(
            {
                "fold": fold,
                "seed": seed,
                "primitive_num": primitive_num,
                "run_dir": directory,
            }
        )
    runs.sort(key=lambda item: (item["fold"], item["seed"]))
    pairs = [(item["fold"], item["seed"]) for item in runs]
    if len(pairs) != len(set(pairs)):
        raise RuntimeError("Input root contains duplicate fold/seed run directories.")
    return runs


def _close(value, expected: float, atol: float = 1e-12) -> bool:
    try:
        return bool(np.isclose(float(value), float(expected), rtol=0.0, atol=atol))
    except (TypeError, ValueError):
        return False


def build_preflight_protocol_audit(
    args: argparse.Namespace,
    folds: list[int],
    seeds: list[int],
    runs: list[dict],
    repositories: dict[Path, SourceSignalRepository],
) -> dict:
    """Audit the canonical protocol before expensive distance construction."""
    motion_protocol = args.input_protocol == "motion_encoder_v1"
    expected_segmentation = (
        "motion_encoder_changepoint"
        if motion_protocol
        else "ssl_feature_changepoint"
    )
    fixed_window_root = (
        Path(args.fixed_window_root).expanduser().resolve()
        if str(args.fixed_window_root).strip()
        else None
    )
    runner_checks = {
        "folds_are_1_through_7": folds == CANONICAL_FOLDS,
        "seeds_are_0_5_50_500": seeds == CANONICAL_SEEDS,
        "expected_runs_is_28": int(args.expected_runs) == 28,
        "state_weight_is_0.25": _close(args.state_weight, 0.25),
        "context_weight_is_0.15": _close(args.context_weight, 0.15),
        "duration_weight_is_0.15": _close(args.duration_weight, 0.15),
        "control_shuffles_at_least_50": int(args.control_shuffles) >= 50,
        "sample_rate_is_100_hz": _close(args.sample_rate_hz, 100.0),
        "old_class_count_is_6": int(args.old_class_count) == 6,
        "control_seed_is_20260902": int(args.seed) == 20260902,
        "fixed_window_root_exists": bool(
            fixed_window_root is not None and fixed_window_root.is_dir()
        ),
    }

    hash_cache: dict[Path, str] = {}

    def file_hash(path: Path) -> str:
        resolved = Path(path).resolve()
        if resolved not in hash_cache:
            hash_cache[resolved] = sha256_file(resolved)
        return hash_cache[resolved]

    per_run = []
    split_records: dict[int, list[tuple]] = defaultdict(list)
    source_hashes = set()
    encoder_training_config_hashes = set()
    for run in runs:
        run_dir = run["run_dir"]
        config = json.loads(
            (run_dir / "experiment_config.json").read_text(encoding="utf-8")
        )
        arguments = config.get("arguments", {})
        metadata = config.get("checkpoint_metadata", {})
        codebook = config.get("codebook", {})
        data_config = config.get("data", {})
        segmentation = config.get("segmentation", {})
        split_path = run_dir / "split_audit.json"
        split = (
            json.loads(split_path.read_text(encoding="utf-8"))
            if split_path.exists()
            else {}
        )
        train_subjects = tuple(sorted(int(value) for value in metadata.get("uschad_train_subjects", [])))
        val_subjects = tuple(sorted(int(value) for value in metadata.get("offline_val_subjects", [])))
        test_subjects = tuple(sorted(int(value) for value in metadata.get("uschad_test_subjects", [])))
        split_records[int(run["fold"])].append(
            (train_subjects, val_subjects, test_subjects)
        )
        repository = repositories[run["source_npz"]]
        source_hashes.add(repository.npz_sha256)
        run["source_npz_sha256"] = repository.npz_sha256
        try:
            recorded_source = resolve_source_npz(run_dir, PROJECT_ROOT, "")
            recorded_source_hash = file_hash(recorded_source)
        except FileNotFoundError:
            recorded_source = None
            recorded_source_hash = ""
        name_pattern = (
            MOTION_ENCODER_RUN_PATTERN
            if motion_protocol
            else CANONICAL_RUN_PATTERN
        )
        name_match = name_pattern.fullmatch(run_dir.name)
        checkpoint_type = str(config.get("checkpoint_type", ""))
        checkpoint_schema = config.get("checkpoint_schema_version")
        feature_roles = config.get("feature_roles", {})
        encoder_training = config.get("encoder_training") or {}
        encoder_training_schedule = config.get("encoder_training_schedule") or {}
        encoder_command_arguments = config.get("encoder_command_arguments") or {}
        encoder_selection = config.get("encoder_selection") or {}
        encoder_source = config.get("encoder_source_checkpoint") or {}
        if motion_protocol:
            # Paths, device and seed legitimately vary across folds/runs.  All
            # remaining command arguments can alter the objective, optimiser,
            # augmentation, pseudo-boundary construction or data handling and
            # therefore belong to the experiment identity.
            normalized_encoder_arguments = {
                key: value
                for key, value in encoder_command_arguments.items()
                if key
                not in {
                    "source_checkpoint",
                    "npz_path",
                    "output_dir",
                    "seed",
                    "device",
                    "self_test",
                }
            }
            training_hash = stable_text_hash(
                [
                    json.dumps(
                        {
                            "resolved_training_config": encoder_training,
                            "training_schedule": encoder_training_schedule,
                            "normalized_command_arguments": normalized_encoder_arguments,
                            "implementation_fingerprint": config.get(
                                "encoder_implementation_fingerprint"
                            ),
                        },
                        sort_keys=True,
                        separators=(",", ":"),
                    )
                ]
            )
            encoder_training_config_hashes.add(training_hash)
        else:
            training_hash = None
        source_legacy_metadata = encoder_source.get(
            "legacy_experiment_metadata", {}
        )
        source_checkpoint_text = str(encoder_source.get("path", "")).replace(
            "\\", "/"
        ).lower()
        checks = {
            "canonical_directory_name": name_match is not None,
            "directory_identity_matches_discovery": bool(
                name_match is not None
                and int(name_match.group(1)) == int(run["fold"])
                and int(name_match.group(2)) == int(run["seed"])
            ),
            "config_fold_matches_directory": int(
                metadata.get("uschad_cv_fold", -1)
            )
            == int(run["fold"]),
            "config_seed_matches_directory": int(arguments.get("seed", -1))
            == int(run["seed"]),
            "k_is_32_everywhere": (
                int(run["primitive_num"]) == 32
                and int(arguments.get("primitive_num", -1)) == 32
                and int(codebook.get("primitive_num", -1)) == 32
            ),
            "segmentation_matches_input_protocol": (
                arguments.get("primitive_segmentation")
                == expected_segmentation
                and segmentation.get("method") == expected_segmentation
            ),
            "codebook_protocol_matches": (
                int(arguments.get("pca_dim", -1)) == 64
                and arguments.get("embedding_normalization") == "l2"
                and arguments.get("codebook_weighting") == "per_trial"
                and int(arguments.get("kmeans_n_init", -1)) == 20
                and int(arguments.get("kmeans_max_iter", -1)) == 300
                and codebook.get("assignment_metric") == "cosine"
            ),
            "split_protocol_matches": (
                int(arguments.get("old_class_count", -1)) == 6
                and arguments.get("fit_subjects", None) == ""
                and arguments.get("eval_subjects", None) == ""
                and arguments.get("allow_split_override", None) is False
                and arguments.get("allow_unverified_npz_normalization", None)
                is False
                and arguments.get("anomaly_policy") == "report"
            ),
            "sampling_protocol_matches": (
                _close(arguments.get("sample_rate_hz"), 100.0)
                and _close(data_config.get("sample_rate_hz"), 100.0)
                and int(metadata.get("uschad_window_size", -1)) == 256
                and int(data_config.get("window_size_samples", -1)) == 256
                and int(repository.window_size) == 256
            ),
            "changepoint_protocol_matches": (
                (
                    motion_protocol
                    or (
                        int(arguments.get("ssl_feature_dim", -1)) == 64
                        and int(arguments.get("ssl_epochs", -1)) == 25
                        and _close(arguments.get("ssl_learning_rate"), 0.001)
                        and _close(arguments.get("ssl_mask_ratio"), 0.15)
                        and _close(arguments.get("ssl_noise_std"), 0.02)
                    )
                )
                and int(arguments.get("changepoint_context_windows", -1)) == 2
                and _close(arguments.get("changepoint_score_quantile"), 0.9)
                and int(arguments.get("changepoint_min_segment_windows", -1))
                == 2
            ),
            "checkpoint_and_feature_roles_match_protocol": (
                (
                    checkpoint_type == "motion_primitive_encoder"
                    and int(checkpoint_schema or -1) == 1
                    and feature_roles.get("codebook")
                    == "motion_encoder_content_head"
                    and feature_roles.get("boundary")
                    == "motion_encoder_segmentation_head"
                    and segmentation.get("boundary_feature_source")
                    == "motion_encoder_segmentation_head"
                )
                if motion_protocol
                else checkpoint_type in {"", "legacy_happy_encoder"}
            ),
            "encoder_training_identity_matches_protocol": (
                (
                    str(encoder_training.get("ablation_profile", "")).upper()
                    == str(args.expected_encoder_profile).upper()
                    and encoder_training.get("backbone_bn_policy") == "frozen"
                    and (encoder_training.get("cp_anchor") or {}).get("source")
                    == "raw_frozen_consensus"
                    and encoder_selection.get("policy") == "final_epoch"
                    and not bool(metadata.get("smoke_test", False))
                )
                if motion_protocol
                else True
            ),
            "encoder_schedule_matches_protocol": (
                (
                    int(encoder_training_schedule.get("epochs", 0)) > 0
                    and int(
                        encoder_training_schedule.get(
                            "early_stopping_patience", -1
                        )
                    )
                    == 0
                    and encoder_training_schedule.get("selection_policy")
                    == "final_epoch"
                    and int(
                        encoder_training_schedule.get(
                            "smoke_max_train_trials", -1
                        )
                    )
                    == 0
                    and int(
                        encoder_training_schedule.get(
                            "smoke_max_val_trials", -1
                        )
                    )
                    == 0
                    and bool(
                        encoder_training_schedule.get("deterministic", False)
                    )
                    and encoder_selection.get("policy") == "final_epoch"
                    and encoder_selection.get("file_role") == "canonical_final"
                    and int(encoder_selection.get("outer_test_queries", -1)) == 0
                    and int(
                        encoder_selection.get("completed_epochs", -1)
                    )
                    == int(encoder_training_schedule.get("epochs", 0))
                    and int(
                        encoder_selection.get("selected_epoch_1based", -1)
                    )
                    == int(encoder_training_schedule.get("epochs", 0))
                )
                if motion_protocol
                else True
            ),
            "encoder_command_arguments_are_complete": (
                CRITICAL_ENCODER_ARGUMENTS.issubset(
                    set(encoder_command_arguments)
                )
                if motion_protocol
                else True
            ),
            "encoder_implementation_is_fingerprinted": (
                bool(config.get("encoder_implementation_fingerprint", {}).get(
                    "combined_sha256"
                ))
                if motion_protocol
                else True
            ),
            "encoder_source_fold_seed_matches_run": (
                (
                    int(source_legacy_metadata.get("uschad_cv_fold", -1))
                    == int(run["fold"])
                    and f"/seed_{int(run['seed'])}_offline/"
                    in source_checkpoint_text
                )
                if motion_protocol
                else True
            ),
            "split_audit_matches_checkpoint_subjects": (
                tuple(sorted(int(value) for value in split.get("fit_subjects", [])))
                == train_subjects
                and tuple(
                    sorted(int(value) for value in split.get("eval_subjects", []))
                )
                == test_subjects
                and not split.get("subject_overlap", ["missing"])
            ),
            "resolved_npz_matches_recorded_npz": (
                recorded_source is not None
                and repository.npz_sha256 == recorded_source_hash
            ),
            "source_npz_matches_canonical_fingerprint": (
                repository.npz_sha256 == CANONICAL_SOURCE_NPZ_SHA256
            ),
        }
        run["config_train_subjects"] = list(train_subjects)
        run["config_val_subjects"] = list(val_subjects)
        run["config_eval_subjects"] = list(test_subjects)
        run["checkpoint"] = str(config.get("checkpoint", ""))
        run["checkpoint_sha256"] = str(config.get("checkpoint_sha256", ""))
        run["checkpoint_type"] = checkpoint_type or "legacy_happy_encoder"
        per_run.append(
            {
                "fold": int(run["fold"]),
                "seed": int(run["seed"]),
                "run_dir": str(run_dir),
                "source_npz": str(repository.npz_path),
                "source_npz_sha256": repository.npz_sha256,
                "recorded_source_npz": (
                    str(recorded_source) if recorded_source is not None else None
                ),
                "train_subjects": list(train_subjects),
                "offline_val_subjects": list(val_subjects),
                "eval_subjects": list(test_subjects),
                "encoder_training_config_sha256": training_hash,
                "checks": checks,
                "passed": bool(all(checks.values())),
            }
        )

    fold_split_stability = {
        str(fold): len(set(split_records.get(fold, []))) == 1
        and len(split_records.get(fold, [])) == len(seeds)
        for fold in folds
    }
    representative_splits = {
        fold: split_records[fold][0]
        for fold in folds
        if split_records.get(fold)
    }
    eval_pairs = [set(values[2]) for values in representative_splits.values()]
    eval_pairs_disjoint = all(
        not (left & right)
        for index, left in enumerate(eval_pairs)
        for right in eval_pairs[index + 1 :]
    )
    eval_union = set().union(*eval_pairs) if eval_pairs else set()
    per_fold_partitions = all(
        not (set(train) & set(val))
        and not (set(train) & set(test))
        and not (set(val) & set(test))
        and set(train) | set(val) | set(test) == set(range(1, 15))
        for train, val, test in representative_splits.values()
    )
    cross_run_checks = {
        "each_fold_has_one_stable_split_across_four_seeds": bool(
            fold_split_stability
            and all(fold_split_stability.values())
            and set(representative_splits) == set(CANONICAL_FOLDS)
        ),
        "seven_eval_subject_pairs_are_disjoint": bool(
            len(eval_pairs) == 7 and eval_pairs_disjoint
        ),
        "seven_eval_subject_pairs_cover_1_through_14": eval_union
        == set(range(1, 15)),
        "each_fold_train_val_test_partition_covers_1_through_14": bool(
            len(representative_splits) == 7 and per_fold_partitions
        ),
        "single_source_npz_fingerprint": len(source_hashes) == 1,
        "single_resolved_encoder_training_config": (
            len(encoder_training_config_hashes) == 1
            if motion_protocol
            else True
        ),
    }
    execution_passed = bool(
        all(runner_checks.values())
        and len(per_run) == 28
        and all(item["passed"] for item in per_run)
        and all(cross_run_checks.values())
    )
    # The legacy gate was preregistered for a different encoder.  A clean
    # motion-protocol audit permits analysis but cannot retroactively satisfy
    # that promotion gate.
    passed = bool(execution_passed and not motion_protocol)
    return {
        "stage": "preflight_before_distance_construction",
        "input_protocol": args.input_protocol,
        "expected_segmentation": expected_segmentation,
        "protocol_status": (
            "exploratory_motion_encoder_v1"
            if motion_protocol
            else "legacy_canonical_candidate"
        ),
        "canonical_definition": {
            "folds": CANONICAL_FOLDS,
            "seeds": CANONICAL_SEEDS,
            "primitive_num": 32,
            "old_class_count": 6,
            "sample_rate_hz": 100.0,
            "weights": {"state": 0.25, "context": 0.15, "duration": 0.15},
            "minimum_control_shuffles": 50,
            "source_npz_sha256": CANONICAL_SOURCE_NPZ_SHA256,
        },
        "runner_checks": runner_checks,
        "per_run": per_run,
        "cross_run_checks": cross_run_checks,
        "fold_split_stability": fold_split_stability,
        "encoder_training_config_sha256_values": sorted(
            encoder_training_config_hashes
        ),
        "execution_protocol_passed": execution_passed,
        "legacy_promotion_eligible": not motion_protocol,
        "passed": passed,
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"Cannot write an empty CSV: {path}")
    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _checkpoint_signature(value: str) -> str:
    normalized = str(value).replace("\\", "/").lower()
    match = re.search(
        r"/fold_\d+/window_pretrain/seed_\d+_offline/.+/checkpoints/model_best\.pt$",
        normalized,
    )
    return match.group(0) if match else normalized


def trial_grid_hash_from_jsonl(path: Path) -> str:
    records = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            required = {
                "trial_global_id_within_npz",
                "subject_id",
                "activity_label_0based",
                "trial_number",
                "window_start_indices",
            }
            missing = required - set(record)
            if missing:
                raise RuntimeError(
                    f"Trial JSONL {path} is missing fields: {sorted(missing)}"
                )
            records.append(record)
    records.sort(
        key=lambda record: (
            int(record["subject_id"]),
            int(record["activity_label_0based"]),
            int(record["trial_number"]),
            int(record["trial_global_id_within_npz"]),
        )
    )
    identities = [
        (
            int(record["trial_global_id_within_npz"]),
            int(record["subject_id"]),
            int(record["activity_label_0based"]),
            int(record["trial_number"]),
        )
        for record in records
    ]
    if len(identities) != len(set(identities)):
        raise RuntimeError(f"Trial JSONL contains duplicate trial identities: {path}")
    return stable_text_hash(
        (
            f"{trial_id}|{subject}|{label}|{trial_number}|"
            f"{','.join(map(str, record['window_start_indices']))}"
        )
        for record, (trial_id, subject, label, trial_number) in zip(
            records, identities
        )
    )


def load_fixed_window_baselines(
    root: Path,
    expected_hashes: dict[tuple[int, int], str],
    expected_runs: list[dict],
) -> list[dict]:
    """Load paired tie-aware fixed-window scores after strict identity checks."""
    root = Path(root).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)
    expected_by_key = {
        (int(run["fold"]), int(run["seed"])): run for run in expected_runs
    }
    expected_checkpoint_types = {
        str(run.get("checkpoint_type", "legacy_happy_encoder"))
        for run in expected_runs
    }
    if len(expected_checkpoint_types) != 1:
        raise RuntimeError(
            "Trajectory runs mix checkpoint types: "
            f"{sorted(expected_checkpoint_types)}."
        )
    expects_motion = expected_checkpoint_types == {"motion_primitive_encoder"}
    discovered = {}
    for directory in root.iterdir():
        if not directory.is_dir():
            continue
        match = (
            MOTION_FIXED_WINDOW_RUN_PATTERN.fullmatch(directory.name)
            if expects_motion
            else FIXED_WINDOW_RUN_PATTERN.fullmatch(directory.name)
        )
        if match is None:
            continue
        fold, seed, primitive_num = (int(value) for value in match.groups())
        key = (fold, seed)
        if key not in expected_hashes:
            continue
        if key in discovered:
            raise RuntimeError(f"Duplicate fixed-window run for {key} under {root}.")
        discovered[key] = (directory, primitive_num)
    expected_keys = set(expected_hashes)
    if set(discovered) != expected_keys:
        raise RuntimeError(
            "Fixed-window fold/seed grid differs from the trajectory grid: "
            f"missing={sorted(expected_keys - set(discovered))}, "
            f"unexpected={sorted(set(discovered) - expected_keys)}."
        )
    rows = []
    for key in sorted(expected_keys):
        directory, primitive_num = discovered[key]
        required = [
            "experiment_config.json",
            "split_audit.json",
            "sequence_association_metrics.json",
            "trial_primitive_sequences.jsonl",
            "sequence_metric_revision.json",
        ]
        missing = [name for name in required if not (directory / name).exists()]
        if missing:
            raise RuntimeError(
                f"Fixed-window run {directory} is missing files: {missing}."
            )
        config = json.loads(
            (directory / "experiment_config.json").read_text(encoding="utf-8")
        )
        arguments = config.get("arguments", {})
        metadata = config.get("checkpoint_metadata", {})
        segmentation = config.get("segmentation", {})
        codebook = config.get("codebook", {})
        feature_roles = config.get("feature_roles", {})
        data_config = config.get("data", {})
        normalization = config.get("normalization", {})
        split = json.loads(
            (directory / "split_audit.json").read_text(encoding="utf-8")
        )
        metric_revision = str(
            config.get("postprocessing", {}).get("metric_revision", "")
        )
        fold, seed = key
        expected_run = expected_by_key[key]
        recorded_checkpoint_hash = str(config.get("checkpoint_sha256", ""))
        expected_checkpoint_hash = str(
            expected_run.get("checkpoint_sha256", "")
        )
        identity_checks = {
            "k_is_32": primitive_num == 32
            and int(arguments.get("primitive_num", -1)) == 32,
            "fold_matches": int(metadata.get("uschad_cv_fold", -1)) == fold,
            "seed_matches": int(arguments.get("seed", -1)) == seed,
            "fixed_window_segmentation_is_explicit": (
                arguments.get("primitive_segmentation") == "fixed_window"
                and segmentation.get("method") == "fixed_window"
            ),
            "codebook_protocol_matches": (
                int(arguments.get("pca_dim", -1)) == 64
                and arguments.get("embedding_normalization") == "l2"
                and arguments.get("codebook_weighting") == "per_trial"
                and int(arguments.get("kmeans_n_init", -1)) == 20
                and int(arguments.get("kmeans_max_iter", -1)) == 300
                and int(codebook.get("primitive_num", -1)) == 32
                and codebook.get("assignment_metric") == "cosine"
                and int(codebook.get("pca_dim", -1)) == 64
                and codebook.get("embedding_normalization") == "l2"
                and codebook.get("weighting") == "per_trial"
            ),
            "split_protocol_matches": (
                tuple(
                    sorted(
                        int(value)
                        for value in metadata.get("uschad_train_subjects", [])
                    )
                )
                == tuple(expected_run.get("config_train_subjects", []))
                and tuple(
                    sorted(
                        int(value)
                        for value in metadata.get("offline_val_subjects", [])
                    )
                )
                == tuple(expected_run.get("config_val_subjects", []))
                and arguments.get("fit_subjects") == ""
                and arguments.get("eval_subjects") == ""
                and arguments.get("allow_split_override") is False
                and arguments.get("allow_unverified_npz_normalization") is False
                and int(arguments.get("old_class_count", -1)) == 6
                and arguments.get("anomaly_policy") == "report"
                and tuple(sorted(int(value) for value in split.get("fit_subjects", [])))
                == tuple(expected_run.get("config_train_subjects", []))
                and tuple(sorted(int(value) for value in split.get("eval_subjects", [])))
                == tuple(expected_run.get("config_eval_subjects", []))
                and not split.get("subject_overlap", ["missing"])
                and split.get("old_class_ids_0based") == list(range(6))
                and split.get("anomaly_policy") == "report"
                and split.get("checkpoint_split_override_used") is False
                and split.get("checkpoint_split_override_explicitly_allowed")
                is False
                and split.get(
                    "unverified_npz_normalization_explicitly_allowed"
                )
                is False
            ),
            "normalization_protocol_matches": (
                metadata.get("uschad_recompute_norm_from_train_subjects") is True
                and normalization.get("mode")
                == "fold_train_subjects_old_classes"
                and normalization.get("raw_source")
                == "reconstructed_from_npz_windows_mean_std"
                and int(normalization.get("stat_window_count", 0)) > 0
            ),
            "sampling_and_source_match": (
                _close(arguments.get("sample_rate_hz"), 100.0)
                and _close(data_config.get("sample_rate_hz"), 100.0)
                and int(metadata.get("uschad_window_size", -1)) == 256
                and int(data_config.get("window_size_samples", -1)) == 256
                and str(config.get("npz_sha256", ""))
                == str(expected_run.get("source_npz_sha256", ""))
            ),
            "eval_subjects_match": sorted(
                int(value) for value in metadata.get("uschad_test_subjects", [])
            )
            == sorted(expected_run.get("config_eval_subjects", [])),
            "checkpoint_matches": (
                bool(recorded_checkpoint_hash)
                and recorded_checkpoint_hash == expected_checkpoint_hash
                if expects_motion
                else _checkpoint_signature(config.get("checkpoint", ""))
                == _checkpoint_signature(expected_run.get("checkpoint", ""))
            ),
            "checkpoint_type_matches": (
                (
                    config.get("checkpoint_type") == "motion_primitive_encoder"
                    and int(config.get("checkpoint_schema_version", -1)) == 1
                    and feature_roles.get("codebook")
                    == "motion_encoder_content_head"
                    and feature_roles.get("boundary") == "unused_fixed_window"
                )
                if expects_motion
                else config.get("checkpoint_type", "legacy_happy_encoder")
                in {"legacy_happy_encoder", ""}
            ),
            "metric_is_tie_aware_v2": "tie_aware_1nn" in metric_revision,
        }
        if not all(identity_checks.values()):
            raise RuntimeError(
                f"Fixed-window identity audit failed for fold={fold}, seed={seed}: "
                f"{identity_checks}."
            )
        actual_hash = trial_grid_hash_from_jsonl(
            directory / "trial_primitive_sequences.jsonl"
        )
        if actual_hash != expected_hashes[key]:
            raise RuntimeError(
                f"Fixed-window trial-grid hash mismatch for fold={fold}, "
                f"seed={seed}: {actual_hash} != {expected_hashes[key]}."
            )
        metrics = json.loads(
            (directory / "sequence_association_metrics.json").read_text(
                encoding="utf-8"
            )
        )
        observed = metrics["rle_sequence_full"]["all_classes"]["observed"]
        rows.append(
            {
                "fold": fold,
                "seed": seed,
                "fixed_window_all_1nn_accuracy": float(
                    observed["cross_subject_1nn_activity_accuracy"]
                ),
                "trial_grid_hash": actual_hash,
                "run_dir": str(directory),
                "identity_checks": identity_checks,
            }
        )
    return rows


def build_fixed_window_comparison(
    rows: list[dict], fixed_rows: list[dict], noninferiority_margin: float = -0.01
) -> dict:
    fixed_by_key = {
        (int(row["fold"]), int(row["seed"])): row for row in fixed_rows
    }
    g4_by_key = {
        (int(row["fold"]), int(row["seed"])): row
        for row in rows
        if row["group"] == GROUP_NAMES[3]
    }
    if set(fixed_by_key) != set(g4_by_key):
        raise RuntimeError("Fixed-window and G4 paired key grids differ.")
    run_deltas = []
    fold_deltas = []
    for fold in sorted({key[0] for key in g4_by_key}):
        values = []
        for key in sorted(key for key in g4_by_key if key[0] == fold):
            delta = float(g4_by_key[key]["all_1nn_accuracy"]) - float(
                fixed_by_key[key]["fixed_window_all_1nn_accuracy"]
            )
            values.append(delta)
            run_deltas.append(
                {"fold": key[0], "seed": key[1], "g4_minus_fixed": delta}
            )
        fold_deltas.append(float(np.mean(values)))
    raw_test = exact_sign_flip(fold_deltas)
    shifted_values = [value - float(noninferiority_margin) for value in fold_deltas]
    shifted_test = exact_sign_flip(shifted_values)
    complete_grid = len(g4_by_key) == 28 and len(fold_deltas) == 7
    passed = bool(
        complete_grid
        and raw_test["mean_delta"] >= float(noninferiority_margin)
        and shifted_test["one_sided_p_greater"] <= 0.05
    )
    return {
        "available": True,
        "comparison": "g4_dynamic_state_duration_context_minus_fixed_window_kmeans32",
        "statistical_unit": "fold mean across four paired seeds",
        "noninferiority_margin": float(noninferiority_margin),
        "paired_run_count": len(run_deltas),
        "run_deltas": run_deltas,
        "fold_deltas": fold_deltas,
        "raw_delta_test": raw_test,
        "margin_shifted_test": shifted_test,
        "folds_at_or_above_margin": int(
            np.sum(np.asarray(fold_deltas) >= float(noninferiority_margin))
        ),
        "fixed_window_mean_accuracy": float(
            np.mean(
                [
                    row["fixed_window_all_1nn_accuracy"]
                    for row in fixed_rows
                ]
            )
        ),
        "g4_mean_accuracy": float(
            np.mean([row["all_1nn_accuracy"] for row in g4_by_key.values()])
        ),
        "complete_canonical_grid": complete_grid,
        "passed": passed,
        "inference_caveat": (
            "The seven folds share overlapping training subjects; this exact "
            "sign-flip result is exploratory, not confirmatory."
        ),
    }


def build_final_protocol_audit(
    preflight: dict, results: list[dict], runs: list[dict]
) -> dict:
    """Complete the audit with trajectory, split, and raw-source invariants."""
    run_by_key = {
        (int(run["fold"]), int(run["seed"])): run for run in runs
    }
    result_by_key = {
        (int(result["fold"]), int(result["seed"])): result for result in results
    }
    expected_keys = {
        (fold, seed) for fold in CANONICAL_FOLDS for seed in CANONICAL_SEEDS
    }
    exact_grid = set(run_by_key) == expected_keys and set(result_by_key) == expected_keys
    trial_grid_by_fold = {}
    raw_manifest_by_fold = {}
    split_match_by_run = {}
    for fold in CANONICAL_FOLDS:
        fold_results = [
            result_by_key[(fold, seed)]
            for seed in CANONICAL_SEEDS
            if (fold, seed) in result_by_key
        ]
        trial_grid_by_fold[str(fold)] = sorted(
            {result["hashes"]["trial_grid_hash"] for result in fold_results}
        )
        raw_manifest_by_fold[str(fold)] = sorted(
            {
                result["source_audit"]["raw_signal_manifest_hash"]
                for result in fold_results
            }
        )
        for result in fold_results:
            key = (int(result["fold"]), int(result["seed"]))
            run = run_by_key[key]
            split_match_by_run[f"fold{key[0]:02d}_seed{key[1]}"] = bool(
                sorted(result["fit_subjects"])
                == sorted(run.get("config_train_subjects", []))
                and sorted(result["eval_subjects"])
                == sorted(run.get("config_eval_subjects", []))
                and not (
                    set(result["fit_subjects"]) & set(result["eval_subjects"])
                )
            )
    eval_pairs = []
    for fold in CANONICAL_FOLDS:
        candidates = [
            set(result_by_key[(fold, seed)]["eval_subjects"])
            for seed in CANONICAL_SEEDS
            if (fold, seed) in result_by_key
        ]
        if candidates and all(value == candidates[0] for value in candidates):
            eval_pairs.append(candidates[0])
    source_audits = [result["source_audit"] for result in results]
    checks = {
        "preflight_passed": bool(preflight.get("passed", False)),
        "exact_7_by_4_result_grid": exact_grid,
        "trial_grid_identical_across_four_seeds_within_each_fold": bool(
            len(trial_grid_by_fold) == 7
            and all(len(values) == 1 for values in trial_grid_by_fold.values())
        ),
        "actual_subject_splits_match_config": bool(
            len(split_match_by_run) == 28 and all(split_match_by_run.values())
        ),
        "actual_eval_pairs_disjoint_and_cover_1_through_14": bool(
            len(eval_pairs) == 7
            and all(
                not (left & right)
                for index, left in enumerate(eval_pairs)
                for right in eval_pairs[index + 1 :]
            )
            and set().union(*eval_pairs) == set(range(1, 15))
        ),
        "single_npz_file_fingerprint": len(
            {audit["source_npz_sha256"] for audit in source_audits}
        )
        == 1,
        "single_npz_semantic_fingerprint": len(
            {audit["dataset_semantic_hash"] for audit in source_audits}
        )
        == 1,
        "uniform_raw_source_within_every_run": bool(
            source_audits
            and all(audit["uniform_source_mode"] for audit in source_audits)
        ),
        "mat_and_npz_visible_spans_compared_for_every_used_trial": bool(
            source_audits
            and all(
                int(audit["mat_npz_compared_trial_count"])
                == int(audit["trial_count"])
                for audit in source_audits
            )
        ),
        "mat_npz_max_error_within_tolerance": bool(
            source_audits
            and all(
                audit["mat_npz_max_abs_error"] is not None
                and float(audit["mat_npz_max_abs_error"])
                <= float(audit["mat_npz_consistency_atol"])
                for audit in source_audits
            )
        ),
        "raw_signal_manifest_identical_across_seeds_within_fold": bool(
            len(raw_manifest_by_fold) == 7
            and all(len(values) == 1 for values in raw_manifest_by_fold.values())
        ),
    }
    return {
        "stage": "final_after_trajectory_and_source_invariants",
        "input_protocol": preflight.get("input_protocol", "legacy_ssl_v1"),
        "protocol_status": preflight.get("protocol_status"),
        "preflight_file": "preflight_protocol_audit.json",
        "checks": checks,
        "trial_grid_hashes_by_fold": trial_grid_by_fold,
        "raw_signal_manifest_hashes_by_fold": raw_manifest_by_fold,
        "actual_split_matches_by_run": split_match_by_run,
        "source_npz_sha256_values": sorted(
            {audit["source_npz_sha256"] for audit in source_audits}
        ),
        "dataset_semantic_hash_values": sorted(
            {audit["dataset_semantic_hash"] for audit in source_audits}
        ),
        "passed": bool(all(checks.values())),
    }


def compact_evaluation(evaluation: dict) -> dict:
    result = {}
    for key, value in evaluation.items():
        if key.endswith("_confusion") and isinstance(value, dict):
            result[key] = {
                inner_key: inner_value
                for inner_key, inner_value in value.items()
                if inner_key != "prediction_probabilities"
            }
        else:
            result[key] = value
    return result


def class_lookup(trials) -> tuple[list[int], list[str]]:
    mapping = {}
    for trial in trials:
        mapping[int(trial.activity_label)] = str(trial.activity_name)
    class_ids = sorted(mapping)
    return class_ids, [mapping[class_id] for class_id in class_ids]


def save_heatmap_grid(
    path: Path,
    matrices: dict[str, np.ndarray],
    class_names: list[str],
    title: str,
    cmap: str,
    vmin: float,
    vmax: float,
    value_format: str,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 4, figsize=(26, 7), constrained_layout=True)
    image = None
    for axis, group in zip(axes, GROUP_NAMES):
        matrix = np.asarray(matrices[group], dtype=np.float64)
        image = axis.imshow(matrix, cmap=cmap, vmin=vmin, vmax=vmax, aspect="equal")
        axis.set_title(group.replace("_", "\n"), fontsize=10)
        axis.set_xticks(np.arange(len(class_names)))
        axis.set_xticklabels(class_names, rotation=55, ha="right", fontsize=7)
        axis.set_yticks(np.arange(len(class_names)))
        axis.set_yticklabels(class_names, fontsize=7)
        for row in range(matrix.shape[0]):
            for column in range(matrix.shape[1]):
                value = matrix[row, column]
                if np.isfinite(value):
                    axis.text(
                        column,
                        row,
                        format(float(value), value_format),
                        ha="center",
                        va="center",
                        fontsize=5,
                        color="white" if value > 0.55 * vmax else "black",
                    )
    fig.suptitle(title, fontsize=14)
    if image is not None:
        fig.colorbar(image, ax=axes, shrink=0.72, pad=0.01)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def save_trajectory_prediction_plot(
    path: Path,
    trials,
    evaluations: dict[str, dict],
    primitive_num: int,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import BoundaryNorm, ListedColormap

    order = sorted(
        range(len(trials)),
        key=lambda index: (
            trials[index].activity_label,
            trials[index].subject_id,
            trials[index].trial_number,
        ),
    )
    ordered_trials = [trials[index] for index in order]
    max_runs = max(len(trial.runs) for trial in ordered_trials)
    token_grid = np.full((len(trials), max_runs), np.nan, dtype=np.float64)
    for row, trial in enumerate(ordered_trials):
        token_grid[row, : len(trial.runs)] = [run.token for run in trial.runs]

    class_ids, class_names = class_lookup(trials)
    class_to_position = {value: position for position, value in enumerate(class_ids)}
    prediction_grid = np.full((len(trials), 1 + len(GROUP_NAMES)), np.nan)
    true_class_probability = np.full(
        (len(trials), len(GROUP_NAMES)), np.nan, dtype=np.float64
    )
    cross_class_top_tie = np.zeros(
        (len(trials), len(GROUP_NAMES)), dtype=bool
    )
    for row, original_index in enumerate(order):
        prediction_grid[row, 0] = class_to_position[
            int(trials[original_index].activity_label)
        ]
        for column, group in enumerate(GROUP_NAMES, start=1):
            confusion = evaluations[group]["all_confusion"]
            probabilities = np.asarray(
                confusion["prediction_probabilities"], dtype=np.float64
            )[original_index]
            if np.all(np.isnan(probabilities)):
                continue
            maximum = float(np.nanmax(probabilities))
            top_positions = np.flatnonzero(
                np.isclose(probabilities, maximum, rtol=0.0, atol=1e-12)
            )
            if len(top_positions) == 1:
                predicted_position = int(top_positions[0])
                predicted_class_id = int(
                    confusion["class_ids"][predicted_position]
                )
                prediction_grid[row, column] = class_to_position[predicted_class_id]
            else:
                cross_class_top_tie[row, column - 1] = True
            true_class_id = int(trials[original_index].activity_label)
            true_position = list(confusion["class_ids"]).index(true_class_id)
            true_class_probability[row, column - 1] = probabilities[true_position]

    token_colors = plt.cm.turbo(np.linspace(0.0, 1.0, max(primitive_num, 2)))
    token_cmap = ListedColormap(token_colors)
    token_cmap.set_bad("white")
    class_colors = plt.cm.tab20(np.linspace(0.0, 1.0, max(len(class_ids), 2)))
    class_cmap = ListedColormap(class_colors)
    class_cmap.set_bad("#bdbdbd")

    fig, (token_axis, prediction_axis) = plt.subplots(
        1,
        2,
        figsize=(max(18, 0.38 * max_runs + 9), 24),
        gridspec_kw={"width_ratios": [max(max_runs, 8), 6]},
        sharey=True,
        constrained_layout=True,
    )
    token_image = token_axis.imshow(
        token_grid,
        interpolation="nearest",
        aspect="auto",
        cmap=token_cmap,
        norm=BoundaryNorm(np.arange(primitive_num + 1) - 0.5, primitive_num),
    )
    prediction_axis.imshow(
        prediction_grid,
        interpolation="nearest",
        aspect="auto",
        cmap=class_cmap,
        norm=BoundaryNorm(np.arange(len(class_ids) + 1) - 0.5, len(class_ids)),
    )
    prediction_axis.set_xticks(np.arange(1 + len(GROUP_NAMES)))
    prediction_axis.set_xticklabels(
        ["True", "G1", "G2", "G3", "G4"], rotation=45, ha="right"
    )
    for row in range(len(trials)):
        for column in range(1, prediction_grid.shape[1]):
            probability = true_class_probability[row, column - 1]
            if cross_class_top_tie[row, column - 1]:
                prediction_axis.text(
                    column,
                    row,
                    "?",
                    color="black",
                    ha="center",
                    va="center",
                    fontsize=7,
                    fontweight="bold",
                )
                continue
            if not np.isfinite(probability) or np.isclose(probability, 1.0):
                continue
            marker = "×" if np.isclose(probability, 0.0) else "·"
            prediction_axis.text(
                column,
                row,
                marker,
                color="white",
                ha="center",
                va="center",
                fontsize=7,
                fontweight="bold",
            )
    boundaries = []
    centers = []
    labels = []
    start = 0
    while start < len(ordered_trials):
        label = ordered_trials[start].activity_label
        end = start + 1
        while end < len(ordered_trials) and ordered_trials[end].activity_label == label:
            end += 1
        boundaries.append(end - 0.5)
        centers.append(0.5 * (start + end - 1))
        labels.append(ordered_trials[start].activity_name)
        start = end
    for boundary in boundaries[:-1]:
        token_axis.axhline(boundary, color="black", linewidth=0.8)
        prediction_axis.axhline(boundary, color="black", linewidth=0.8)
    token_axis.set_yticks(centers)
    token_axis.set_yticklabels(labels, fontsize=8)
    token_axis.set_xlabel("Fixed RLE run position")
    token_axis.set_ylabel("Held-out activity trials grouped by true activity")
    token_axis.set_title("Shared token trajectories (identical in G1-G4)")
    prediction_axis.set_title(
        "Tie-aware cross-subject 1-NN\n"
        "color=unique top; gray ?=top tie\n"
        "×=no true credit; ·=partial credit",
        fontsize=8,
    )
    fig.colorbar(token_image, ax=token_axis, shrink=0.45, label="Primitive token id")
    fig.suptitle(
        "Fixed motion-primitive trajectories and four readout predictions",
        fontsize=14,
    )
    fig.savefig(path, dpi=180)
    plt.close(fig)


def save_duration_weighted_trajectory_plot(
    path: Path,
    trials,
    primitive_num: int,
    time_bins: int = 160,
) -> None:
    """Plot token order with relative run width and absolute trial duration."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import BoundaryNorm, ListedColormap

    order = sorted(
        range(len(trials)),
        key=lambda index: (
            trials[index].activity_label,
            trials[index].subject_id,
            trials[index].trial_number,
        ),
    )
    ordered_trials = [trials[index] for index in order]
    grid = np.full((len(ordered_trials), int(time_bins)), np.nan, dtype=np.float64)
    total_durations = np.zeros((len(ordered_trials), 1), dtype=np.float64)
    for row, trial in enumerate(ordered_trials):
        durations = np.asarray(
            [run.duration_seconds for run in trial.runs], dtype=np.float64
        )
        total = max(float(np.sum(durations)), EPS)
        total_durations[row, 0] = total
        boundaries = np.rint(
            np.r_[0.0, np.cumsum(durations) / total] * int(time_bins)
        ).astype(int)
        boundaries[0] = 0
        boundaries[-1] = int(time_bins)
        for index, run in enumerate(trial.runs):
            begin = int(np.clip(boundaries[index], 0, int(time_bins) - 1))
            end = int(np.clip(boundaries[index + 1], begin + 1, int(time_bins)))
            grid[row, begin:end] = run.token

    token_colors = plt.cm.turbo(np.linspace(0.0, 1.0, max(primitive_num, 2)))
    token_cmap = ListedColormap(token_colors)
    token_cmap.set_bad("white")
    fig, (trajectory_axis, duration_axis) = plt.subplots(
        1,
        2,
        figsize=(22, 24),
        gridspec_kw={"width_ratios": [18, 1]},
        sharey=True,
        constrained_layout=True,
    )
    token_image = trajectory_axis.imshow(
        grid,
        interpolation="nearest",
        aspect="auto",
        cmap=token_cmap,
        norm=BoundaryNorm(np.arange(primitive_num + 1) - 0.5, primitive_num),
    )
    log_durations = np.log1p(total_durations)
    duration_image = duration_axis.imshow(
        log_durations,
        interpolation="nearest",
        aspect="auto",
        cmap="cividis",
        vmin=float(np.min(log_durations)),
        vmax=float(np.max(log_durations)),
    )
    group_boundaries = []
    centers = []
    labels = []
    start = 0
    while start < len(ordered_trials):
        label = ordered_trials[start].activity_label
        end = start + 1
        while end < len(ordered_trials) and ordered_trials[end].activity_label == label:
            end += 1
        group_boundaries.append(end - 0.5)
        centers.append(0.5 * (start + end - 1))
        labels.append(ordered_trials[start].activity_name)
        start = end
    for boundary in group_boundaries[:-1]:
        trajectory_axis.axhline(boundary, color="black", linewidth=0.8)
        duration_axis.axhline(boundary, color="black", linewidth=0.8)
    trajectory_axis.set_yticks(centers)
    trajectory_axis.set_yticklabels(labels, fontsize=8)
    trajectory_axis.set_xlabel("Normalized visible-trial time (run width = relative duration)")
    trajectory_axis.set_ylabel("Held-out trials grouped by activity")
    trajectory_axis.set_title("Fixed tokens with duration-proportional run widths")
    duration_axis.set_xticks([0])
    duration_axis.set_xticklabels(["total\nduration"], fontsize=8)
    duration_axis.set_title("Absolute")
    fig.colorbar(token_image, ax=trajectory_axis, shrink=0.45, label="Primitive token id")
    fig.colorbar(
        duration_image,
        ax=duration_axis,
        shrink=0.45,
        label="log(1 + visible seconds)",
    )
    fig.suptitle(
        "Duration-aware view of the shared motion-primitive trajectories",
        fontsize=14,
    )
    fig.savefig(path, dpi=180)
    plt.close(fig)


def per_class_rows(
    fold: int,
    seed: int,
    group: str,
    evaluation: dict,
    class_names: list[str],
) -> list[dict]:
    confusion = evaluation["all_confusion"]
    rows = []
    for position, class_id in enumerate(confusion["class_ids"]):
        rows.append(
            {
                "fold": fold,
                "seed": seed,
                "group": group,
                "class_id_0based": int(class_id),
                "activity_name": class_names[position],
                "recall": float(confusion["per_class_recall"][position]),
                "precision": float(confusion["per_class_precision"][position]),
                "f1": float(confusion["per_class_f1"][position]),
            }
        )
    return rows


def trajectory_sequence_rows(
    fold: int,
    seed: int,
    primitive_num: int,
    run_dir: Path,
    trials,
) -> list[dict]:
    segment_counts = {}
    with (run_dir / "trial_primitive_sequences.jsonl").open(
        "r", encoding="utf-8"
    ) as handle:
        for line in handle:
            record = json.loads(line)
            trial_id = int(record["trial_global_id_within_npz"])
            segmentation = record.get("primitive_segmentation", {})
            segment_counts[trial_id] = int(
                segmentation.get(
                    "segment_count",
                    len(segmentation.get("segments", [])),
                )
            )
    rows = []
    for trial in trials:
        tokens = [int(run.token) for run in trial.runs]
        durations = [float(run.duration_seconds) for run in trial.runs]
        boundaries = [int(trial.runs[0].start_sample)] + [
            int(run.end_sample_exclusive) for run in trial.runs
        ]
        rows.append(
            {
                "fold": int(fold),
                "seed": int(seed),
                "trial_global_id": int(trial.trial_global_id),
                "trial_key": str(trial.trial_key),
                "subject_id": int(trial.subject_id),
                "activity_label_0based": int(trial.activity_label),
                "activity_name": str(trial.activity_name),
                "trial_number": int(trial.trial_number),
                "codebook_size": int(primitive_num),
                "adaptive_segment_count_before_rle": int(
                    segment_counts[trial.trial_global_id]
                ),
                "rle_motion_primitive_run_count": len(tokens),
                "unique_primitive_id_count": len(set(tokens)),
                "rle_primitive_sequence": " -> ".join(map(str, tokens)),
                "run_durations_seconds": " | ".join(
                    f"{value:.6g}" for value in durations
                ),
                "partition_boundaries_samples": " | ".join(
                    map(str, boundaries)
                ),
                "visible_trial_duration_seconds": float(sum(durations)),
            }
        )
    return rows


def _mean_class_recall(evaluation: dict, names: list[str], class_names: list[str]) -> float:
    normalized = [name.strip().lower() for name in class_names]
    positions = [normalized.index(name.lower()) for name in names if name.lower() in normalized]
    values = np.asarray(
        evaluation["all_confusion"]["per_class_recall"], dtype=np.float64
    )
    return float(np.mean(values[positions])) if positions else float("nan")


def summary_row(
    fold: int,
    seed: int,
    group: str,
    evaluation: dict,
    class_names: list[str],
    state_weight: float,
    context_weight: float,
    duration_weight: float,
    controls: dict,
) -> dict:
    association = evaluation["all_association"]
    confusion = evaluation["all_confusion"]
    old_all_candidates = evaluation["old_queries_all_candidates"]
    novel_all_candidates = evaluation["novel_queries_all_candidates"]
    old_restricted = evaluation["old_confusion"]
    novel_restricted = evaluation["novel_confusion"]
    low_dynamic_names = ["Sitting", "Standing", "Elevator Up", "Elevator Down"]
    sit_stand_names = ["Sitting", "Standing"]
    motion_names = [
        "Walking Forward",
        "Walking Left",
        "Walking Right",
        "Walking Upstairs",
        "Walking Downstairs",
        "Running Forward",
        "Jumping Up",
        "Elevator Up",
        "Elevator Down",
    ]
    row = {
        "fold": fold,
        "seed": seed,
        "group": group,
        "all_1nn_accuracy": float(confusion["accuracy"]),
        "all_macro_recall": float(confusion["macro_recall"]),
        "all_macro_f1": float(confusion["macro_f1"]),
        "same_distance_mean": float(association["same_mean"]),
        "different_distance_mean": float(association["different_mean"]),
        "distance_margin": float(association["mean_margin_different_minus_same"]),
        "pairwise_auc": float(association["probability_same_distance_is_smaller"]),
        "zero_min_distance_query_ratio": float(
            confusion["zero_min_distance_query_ratio"]
        ),
        "mean_tied_nearest_count": float(confusion["mean_tied_nearest_count"]),
        "mean_minimum_distance": float(confusion["mean_minimum_distance"]),
        "old_query_12class_1nn_accuracy": float(old_all_candidates["accuracy"]),
        "novel_query_12class_1nn_accuracy_diagnostic": float(
            novel_all_candidates["accuracy"]
        ),
        "old_restricted_candidate_1nn_accuracy": float(
            old_restricted["accuracy"]
        ),
        "novel_restricted_candidate_1nn_accuracy_diagnostic": float(
            novel_restricted["accuracy"]
        ),
        "sit_stand_binary_1nn_accuracy": float(
            evaluation["sit_stand_binary_confusion"]["accuracy"]
        ),
        "sit_stand_zero_min_distance_query_ratio": float(
            evaluation["sit_stand_binary_confusion"][
                "zero_min_distance_query_ratio"
            ]
        ),
        "low_dynamic_restricted_1nn_accuracy": float(
            evaluation["low_dynamic_confusion"]["accuracy"]
        ),
        "sit_stand_12class_macro_recall": _mean_class_recall(
            evaluation, sit_stand_names, class_names
        ),
        "low_dynamic_12class_macro_recall": _mean_class_recall(
            evaluation, low_dynamic_names, class_names
        ),
        "motion_sensitive_12class_macro_recall": _mean_class_recall(
            evaluation, motion_names, class_names
        ),
        "state_weight": state_weight if group in GROUP_NAMES[2:] else 0.0,
        "context_weight": context_weight if group == GROUP_NAMES[3] else 0.0,
        "duration_weight": duration_weight if group == GROUP_NAMES[3] else 0.0,
        "order_control_accuracy_gain": float("nan"),
        "duration_alignment_control_accuracy_gain": float("nan"),
        "total_duration_control_accuracy_gain": float("nan"),
    }
    normalized_names = [name.strip().lower() for name in class_names]
    per_class_recall_values = np.asarray(confusion["per_class_recall"], dtype=float)
    for display_name, output_name in [
        ("Sitting", "sitting_12class_recall"),
        ("Standing", "standing_12class_recall"),
        ("Elevator Up", "elevator_up_12class_recall"),
        ("Elevator Down", "elevator_down_12class_recall"),
    ]:
        if display_name.lower() in normalized_names:
            position = normalized_names.index(display_name.lower())
            row[output_name] = float(per_class_recall_values[position])
    if group == GROUP_NAMES[3] and controls.get("shuffles", 0) > 0:
        row["order_control_accuracy_gain"] = float(
            controls["order"]["observed_minus_shuffled_accuracy"]
        )
        row["duration_alignment_control_accuracy_gain"] = float(
            controls["duration_alignment"]["observed_minus_shuffled_accuracy"]
        )
        row["total_duration_control_accuracy_gain"] = float(
            controls["total_duration"]["observed_minus_shuffled_accuracy"]
        )
    return row


def validate_hard_baseline(run_dir: Path, evaluation: dict) -> dict:
    path = run_dir / "sequence_association_metrics.json"
    existing = json.loads(path.read_text(encoding="utf-8"))
    expected = existing["rle_sequence_full"]["all_classes"]["observed"]
    actual = evaluation["all_association"]
    checks = {}
    for key in [
        "same_mean",
        "different_mean",
        "mean_margin_different_minus_same",
        "cross_subject_1nn_activity_accuracy",
    ]:
        difference = abs(float(expected[key]) - float(actual[key]))
        checks[key] = {
            "expected": float(expected[key]),
            "actual": float(actual[key]),
            "absolute_difference": difference,
            "matches": bool(difference <= 1e-7),
        }
    if not all(item["matches"] for item in checks.values()):
        raise RuntimeError(
            f"Recomputed hard baseline does not match {path}; fixed-trajectory "
            "ablation is invalid."
        )
    return checks


def analyze_run(
    run: dict,
    output_dir: Path,
    source: SourceSignalRepository,
    args: argparse.Namespace,
) -> dict:
    run_dir = run["run_dir"]
    with np.load(run_dir / "primitive_codebook.npz", allow_pickle=False) as data:
        centers = np.asarray(data["centers"], dtype=np.float32)
        segmentation = str(np.asarray(data["primitive_segmentation"]).item())
        assignment_metric = str(np.asarray(data["assignment_metric"]).item())
    if run["primitive_num"] != 32 or centers.shape[0] != 32:
        raise RuntimeError(
            f"This preregistered ablation requires K=32, got directory K="
            f"{run['primitive_num']} and centers={centers.shape}."
        )
    expected_segmentation = (
        "motion_encoder_changepoint"
        if args.input_protocol == "motion_encoder_v1"
        else "ssl_feature_changepoint"
    )
    if segmentation != expected_segmentation:
        raise RuntimeError(
            f"Input run {run_dir} uses segmentation={segmentation!r}; expected "
            f"{expected_segmentation!r} for protocol {args.input_protocol!r}."
        )
    if assignment_metric != "cosine":
        raise RuntimeError(
            f"G2 expects the existing L2/cosine codebook, got {assignment_metric!r}."
        )
    fit_trials, eval_trials, hashes = build_trajectories(
        run_dir,
        source,
        args.sample_rate_hz,
        args.old_class_count,
    )
    fit_subjects = sorted({trial.subject_id for trial in fit_trials})
    eval_subjects = sorted({trial.subject_id for trial in eval_trials})
    source_audit = source.source_audit(
        [trial.trial_global_id for trial in fit_trials + eval_trials]
    )
    if not source_audit["uniform_source_mode"]:
        raise RuntimeError(
            f"Run {run_dir} mixes MAT and NPZ-fallback raw sources: "
            f"{source_audit['source_mode_counts']}."
        )
    if set(fit_subjects) & set(eval_subjects):
        raise RuntimeError("Fit and evaluation subject sets overlap.")
    scaler = fit_state_scaler(fit_trials)
    fit_trials = transform_trial_states(fit_trials, scaler)
    eval_trials = transform_trial_states(eval_trials, scaler)
    scales = fit_local_scales(fit_trials)
    dynamic_cost = codebook_dynamic_cost(centers)
    class_ids, class_names = class_lookup(eval_trials)

    matrices = {}
    evaluations = {}
    for group in GROUP_NAMES:
        matrix = trajectory_distance_matrix(
            eval_trials,
            group,
            dynamic_cost,
            scales,
            args.state_weight,
            args.context_weight,
            args.duration_weight,
        )
        matrices[group] = matrix
        evaluations[group] = evaluate_distance_matrix(
            matrix, eval_trials, args.old_class_count
        )
    duration_only_matrix = total_duration_distance_matrix(eval_trials)
    duration_only_evaluation = evaluate_distance_matrix(
        duration_only_matrix, eval_trials, args.old_class_count
    )
    hard_validation = validate_hard_baseline(
        run_dir, evaluations["g1_hard_rle"]
    )
    controls = run_controls(
        eval_trials,
        evaluations[GROUP_NAMES[3]]["all_confusion"]["accuracy"],
        dynamic_cost,
        scales,
        args.state_weight,
        args.context_weight,
        args.duration_weight,
        args.control_shuffles,
        args.seed + run["fold"] * 10000 + run["seed"],
    )

    output_dir.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(
        output_dir / "trajectory_distance_matrices.npz",
        **{group: matrices[group] for group in GROUP_NAMES},
        total_duration_only=duration_only_matrix,
        trial_global_ids=np.asarray(
            [trial.trial_global_id for trial in eval_trials], dtype=np.int64
        ),
        subject_ids=np.asarray(
            [trial.subject_id for trial in eval_trials], dtype=np.int64
        ),
        activity_labels_0based=np.asarray(
            [trial.activity_label for trial in eval_trials], dtype=np.int64
        ),
    )
    np.savez_compressed(
        output_dir / "readout_cost_parameters.npz",
        codebook_dynamic_cost=dynamic_cost,
        state_center=scaler.center,
        state_scale=scaler.scale,
        state_feature_names=np.asarray(STATE_FEATURE_NAMES),
        state_distance_p95=np.asarray(scales["state_distance_p95"]),
        log_duration_difference_p95=np.asarray(
            scales["log_duration_difference_p95"]
        ),
    )
    np.savez_compressed(
        output_dir / "trajectory_confusions.npz",
        **{
            group: evaluations[group]["all_confusion"][
                "confusion_row_normalized"
            ]
            for group in GROUP_NAMES
        },
    )
    activity_matrices = {
        group: evaluations[group]["activity_distance_matrix"]
        for group in GROUP_NAMES
    }
    confusion_matrices = {
        group: evaluations[group]["all_confusion"]["confusion_row_normalized"]
        for group in GROUP_NAMES
    }
    save_heatmap_grid(
        output_dir / "ablation_activity_distance_heatmaps.png",
        activity_matrices,
        class_names,
        "Cross-subject activity distance: fixed trajectories, four readouts",
        "viridis",
        0.0,
        1.0,
        ".2f",
    )
    save_heatmap_grid(
        output_dir / "ablation_confusion_heatmaps.png",
        confusion_matrices,
        class_names,
        "Tie-aware cross-subject 1-NN confusion: fixed trajectories",
        "magma",
        0.0,
        1.0,
        ".2f",
    )
    save_trajectory_prediction_plot(
        output_dir / "fixed_trajectories_and_predictions.png",
        eval_trials,
        evaluations,
        run["primitive_num"],
    )
    save_duration_weighted_trajectory_plot(
        output_dir / "duration_weighted_trajectories.png",
        eval_trials,
        run["primitive_num"],
    )

    group_rows = []
    all_per_class_rows = []
    for group in GROUP_NAMES:
        group_rows.append(
            summary_row(
                run["fold"],
                run["seed"],
                group,
                evaluations[group],
                class_names,
                args.state_weight,
                args.context_weight,
                args.duration_weight,
                controls,
            )
        )
        all_per_class_rows.extend(
            per_class_rows(
                run["fold"],
                run["seed"],
                group,
                evaluations[group],
                class_names,
            )
        )
    write_csv(output_dir / "ablation_group_metrics.csv", group_rows)
    write_csv(output_dir / "ablation_per_class.csv", all_per_class_rows)
    sequence_rows = trajectory_sequence_rows(
        run["fold"], run["seed"], run["primitive_num"], run_dir, eval_trials
    )
    write_csv(output_dir / "trajectory_sequences.csv", sequence_rows)
    duration_only_row = {
        "fold": run["fold"],
        "seed": run["seed"],
        "control": "total_visible_duration_only",
        "all_1nn_accuracy": float(
            duration_only_evaluation["all_confusion"]["accuracy"]
        ),
        "all_macro_f1": float(
            duration_only_evaluation["all_confusion"]["macro_f1"]
        ),
        "distance_margin": float(
            duration_only_evaluation["all_association"][
                "mean_margin_different_minus_same"
            ]
        ),
        "pairwise_auc": float(
            duration_only_evaluation["all_association"][
                "probability_same_distance_is_smaller"
            ]
        ),
        "old_query_12class_1nn_accuracy": float(
            duration_only_evaluation["old_queries_all_candidates"]["accuracy"]
        ),
        "novel_query_12class_1nn_accuracy_diagnostic": float(
            duration_only_evaluation["novel_queries_all_candidates"]["accuracy"]
        ),
        "old_restricted_candidate_1nn_accuracy": float(
            duration_only_evaluation["old_confusion"]["accuracy"]
        ),
        "novel_restricted_candidate_1nn_accuracy_diagnostic": float(
            duration_only_evaluation["novel_confusion"]["accuracy"]
        ),
    }
    write_csv(output_dir / "shortcut_control_metrics.csv", [duration_only_row])
    details = {
        "scope": "fixed segmentation and KMeans32 trajectory-readout ablation",
        "study_design_status": "exploratory_post_hoc",
        "study_design_caveat": (
            "The state feature family was selected after inspecting earlier "
            "outer-test Sit/Stand diagnostics. Evaluation labels do not enter "
            "distance construction, but confirmation requires new untouched "
            "subjects or an external dataset."
        ),
        "input_run": str(run_dir.resolve()),
        "fold": run["fold"],
        "seed": run["seed"],
        "primitive_num": run["primitive_num"],
        "groups": list(GROUP_NAMES),
        "weights": {
            "state": args.state_weight,
            "context": args.context_weight,
            "duration": args.duration_weight,
            "g4_base_dynamic_state_weight": 1.0
            - args.context_weight
            - args.duration_weight,
        },
        "split_audit": {
            "fit_subjects": fit_subjects,
            "eval_subjects": eval_subjects,
            "fit_eval_subject_overlap": sorted(set(fit_subjects) & set(eval_subjects)),
            "fit_trial_count": len(fit_trials),
            "eval_trial_count": len(eval_trials),
            "state_scaler_source": "fit subjects and old classes only",
            "evaluation_labels_used_for_distance": False,
            "raw_state_source": (
                "original MAT sensor_readings clipped to NPZ-visible span; "
                "overlap-deduplicated NPZ inverse transform is fallback"
            ),
            "source_npz": str(source.npz_path),
            "source_fingerprints_and_consistency": source_audit,
        },
        "invariance_hashes": hashes,
        "hard_baseline_reproduction": hard_validation,
        "state_features": list(STATE_FEATURE_NAMES),
        "local_scales": scales,
        "controls": controls,
        "shortcut_controls": {
            "total_visible_duration_only": compact_evaluation(
                duration_only_evaluation
            )
        },
        "evaluations": {
            group: compact_evaluation(evaluations[group]) for group in GROUP_NAMES
        },
    }
    write_json(output_dir / "trajectory_ablation_summary.json", details)
    return {
        "fold": int(run["fold"]),
        "seed": int(run["seed"]),
        "group_rows": group_rows,
        "per_class_rows": all_per_class_rows,
        "sequence_rows": sequence_rows,
        "activity_matrices": activity_matrices,
        "confusion_matrices": confusion_matrices,
        "class_names": class_names,
        "hashes": hashes,
        "fit_subjects": fit_subjects,
        "eval_subjects": eval_subjects,
        "source_audit": source_audit,
        "shortcut_row": duration_only_row,
        "output_dir": output_dir,
    }


def exact_sign_flip(values: list[float]) -> dict:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or len(array) == 0:
        raise ValueError("Sign-flip test needs a non-empty vector.")
    observed = float(np.mean(array))
    statistics = []
    for mask in range(1 << len(array)):
        signs = np.asarray(
            [1.0 if mask & (1 << index) else -1.0 for index in range(len(array))]
        )
        statistics.append(float(np.mean(array * signs)))
    statistics = np.asarray(statistics, dtype=np.float64)
    return {
        "fold_deltas": array,
        "mean_delta": observed,
        "median_delta": float(np.median(array)),
        "positive_fold_count": int(np.sum(array > 0)),
        "negative_fold_count": int(np.sum(array < 0)),
        "zero_fold_count": int(np.sum(array == 0)),
        "one_sided_p_greater": float(np.mean(statistics >= observed - 1e-15)),
        "one_sided_p_less": float(np.mean(statistics <= observed + 1e-15)),
        "two_sided_p": float(
            np.mean(np.abs(statistics) >= abs(observed) - 1e-15)
        ),
    }


def holm_adjust(p_values: dict[str, float]) -> dict[str, float]:
    ordered = sorted(p_values, key=lambda key: p_values[key])
    adjusted = {}
    running = 0.0
    count = len(ordered)
    for rank, key in enumerate(ordered):
        candidate = min(1.0, (count - rank) * float(p_values[key]))
        running = max(running, candidate)
        adjusted[key] = running
    return adjusted


def build_paired_summary(rows: list[dict]) -> dict:
    by_key = {(row["fold"], row["seed"], row["group"]): row for row in rows}
    folds = sorted({int(row["fold"]) for row in rows})
    seeds_by_fold = {
        fold: sorted(
            {
                int(row["seed"])
                for row in rows
                if int(row["fold"]) == fold
            }
        )
        for fold in folds
    }
    comparisons = {
        "g2_minus_g1": (GROUP_NAMES[0], GROUP_NAMES[1]),
        "g3_minus_g2": (GROUP_NAMES[1], GROUP_NAMES[2]),
        "g4_minus_g3": (GROUP_NAMES[2], GROUP_NAMES[3]),
        "g4_minus_g1": (GROUP_NAMES[0], GROUP_NAMES[3]),
    }
    metrics = [
        "all_1nn_accuracy",
        "all_macro_f1",
        "pairwise_auc",
        "zero_min_distance_query_ratio",
        "mean_tied_nearest_count",
        "old_query_12class_1nn_accuracy",
        "novel_query_12class_1nn_accuracy_diagnostic",
        "old_restricted_candidate_1nn_accuracy",
        "novel_restricted_candidate_1nn_accuracy_diagnostic",
        "sit_stand_12class_macro_recall",
        "low_dynamic_12class_macro_recall",
        "motion_sensitive_12class_macro_recall",
        "sit_stand_binary_1nn_accuracy",
        "sit_stand_zero_min_distance_query_ratio",
    ]
    result = {"statistical_unit": "fold mean across seeds", "comparisons": {}}
    primary_metrics = {
        "g2_minus_g1": "all_1nn_accuracy",
        "g3_minus_g2": "sit_stand_12class_macro_recall",
        "g4_minus_g3": "motion_sensitive_12class_macro_recall",
    }
    primary_p = {}
    for comparison_name, (before, after) in comparisons.items():
        comparison = {"before": before, "after": after, "metrics": {}}
        for metric in metrics:
            fold_deltas = []
            for fold in folds:
                seed_deltas = []
                for seed in seeds_by_fold[fold]:
                    left = by_key[(fold, seed, before)][metric]
                    right = by_key[(fold, seed, after)][metric]
                    seed_deltas.append(float(right) - float(left))
                fold_deltas.append(float(np.mean(seed_deltas)))
            comparison["metrics"][metric] = exact_sign_flip(fold_deltas)
        result["comparisons"][comparison_name] = comparison
        if comparison_name != "g4_minus_g1":
            primary_metric = primary_metrics[comparison_name]
            comparison["primary_metric"] = primary_metric
            primary_p[comparison_name] = comparison["metrics"][primary_metric][
                "one_sided_p_greater"
            ]
    adjusted = holm_adjust(primary_p)
    for comparison_name, value in adjusted.items():
        result["comparisons"][comparison_name][
            "primary_metric_holm_adjusted_one_sided_p"
        ] = value
    result["multiple_testing_family"] = {
        comparison: metric for comparison, metric in primary_metrics.items()
    }
    result["overlap_caveat"] = (
        "The seven subject folds have overlapping training sets; exact sign-flip "
        "p-values are exploratory and seeds are not independent units."
    )
    return result


def group_metric_mean(rows: list[dict], group: str, metric: str) -> float:
    return float(np.mean([float(row[metric]) for row in rows if row["group"] == group]))


def build_gate_report(
    rows: list[dict],
    paired: dict,
    protocol_audit: dict,
    fixed_window_comparison: dict,
) -> dict:
    fold_ids = sorted({int(row["fold"]) for row in rows})
    group_counts = {
        group: sum(row["group"] == group for row in rows) for group in GROUP_NAMES
    }
    seeds_per_fold = {
        fold: sorted(
            {
                int(row["seed"])
                for row in rows
                if int(row["fold"]) == fold
            }
        )
        for fold in fold_ids
    }
    expected_keys = {
        (fold, seed, group)
        for fold in CANONICAL_FOLDS
        for seed in CANONICAL_SEEDS
        for group in GROUP_NAMES
    }
    observed_keys = [
        (int(row["fold"]), int(row["seed"]), str(row["group"])) for row in rows
    ]
    complete_exact_grid = (
        len(observed_keys) == len(set(observed_keys))
        and set(observed_keys) == expected_keys
    )
    promotion_eligible = bool(protocol_audit.get("passed", False) and complete_exact_grid)

    def delta(comparison: str, metric: str) -> float:
        return float(
            paired["comparisons"][comparison]["metrics"][metric]["mean_delta"]
        )

    def positive(comparison: str, metric: str) -> int:
        return int(
            paired["comparisons"][comparison]["metrics"][metric][
                "positive_fold_count"
            ]
        )

    g1_ties = group_metric_mean(rows, GROUP_NAMES[0], "mean_tied_nearest_count")
    g2_ties = group_metric_mean(rows, GROUP_NAMES[1], "mean_tied_nearest_count")
    tie_relative_reduction = (g1_ties - g2_ties) / max(g1_ties, 1e-12)
    common_non_degradation_g2 = (
        delta("g2_minus_g1", "old_query_12class_1nn_accuracy") >= -0.02
        and delta(
            "g2_minus_g1", "novel_query_12class_1nn_accuracy_diagnostic"
        )
        >= -0.02
    )

    def control_fold_statistics(metric: str) -> dict:
        fold_values = []
        for fold in fold_ids:
            values = [
                float(row[metric])
                for row in rows
                if row["group"] == GROUP_NAMES[3] and int(row["fold"]) == fold
            ]
            if not values or not np.all(np.isfinite(values)):
                return {
                    "available": False,
                    "metric": metric,
                    "reason": "Control shuffles were disabled or incomplete.",
                }
            fold_values.append(float(np.mean(values)))
        result = exact_sign_flip(fold_values)
        result.update({"available": True, "metric": metric})
        return result

    control_stats = {
        "order": control_fold_statistics("order_control_accuracy_gain"),
        "duration_alignment": control_fold_statistics(
            "duration_alignment_control_accuracy_gain"
        ),
        "total_duration": control_fold_statistics(
            "total_duration_control_accuracy_gain"
        ),
    }
    structural_control_p = {
        name: stats["one_sided_p_greater"]
        for name, stats in control_stats.items()
        if name in {"order", "duration_alignment"} and stats["available"]
    }
    structural_control_adjusted = (
        holm_adjust(structural_control_p) if len(structural_control_p) == 2 else {}
    )
    for name, adjusted_p in structural_control_adjusted.items():
        control_stats[name]["holm_adjusted_one_sided_p"] = adjusted_p

    def structural_control_passed(name: str) -> bool:
        stats = control_stats[name]
        return bool(
            stats.get("available", False)
            and float(stats["mean_delta"]) >= 0.02
            and int(stats["positive_fold_count"]) >= 6
            and float(stats.get("holm_adjusted_one_sided_p", 1.0)) <= 0.05
        )
    g2_checks = {
        "all_1nn_delta_at_least_0.03": delta("g2_minus_g1", "all_1nn_accuracy")
        >= 0.03,
        "pairwise_auc_delta_at_least_0.01": delta("g2_minus_g1", "pairwise_auc")
        >= 0.01,
        "mean_tied_nearest_relative_reduction_at_least_25pct": tie_relative_reduction
        >= 0.25,
        "at_least_6_of_7_positive_folds": positive(
            "g2_minus_g1", "all_1nn_accuracy"
        )
        >= 6,
        "primary_metric_holm_adjusted_one_sided_p_at_most_0.05": paired["comparisons"][
            "g2_minus_g1"
        ]["primary_metric_holm_adjusted_one_sided_p"]
        <= 0.05,
        "old_and_novel_drop_no_more_than_0.02": common_non_degradation_g2,
    }
    g3_checks = {
        "sit_stand_12class_macro_recall_delta_at_least_0.08": delta(
            "g3_minus_g2", "sit_stand_12class_macro_recall"
        )
        >= 0.08,
        "low_dynamic_12class_macro_recall_delta_at_least_0.05": delta(
            "g3_minus_g2", "low_dynamic_12class_macro_recall"
        )
        >= 0.05,
        "all_1nn_noninferior_minus_0.01": delta(
            "g3_minus_g2", "all_1nn_accuracy"
        )
        >= -0.01,
        "sit_stand_zero_min_ratio_drop_at_least_0.10": delta(
            "g3_minus_g2", "sit_stand_zero_min_distance_query_ratio"
        )
        <= -0.10,
        "at_least_6_of_7_positive_sit_stand_folds": positive(
            "g3_minus_g2", "sit_stand_12class_macro_recall"
        )
        >= 6,
        "primary_metric_holm_adjusted_one_sided_p_at_most_0.05": paired[
            "comparisons"
        ]["g3_minus_g2"]["primary_metric_holm_adjusted_one_sided_p"]
        <= 0.05,
        "old_and_novel_drop_no_more_than_0.02": (
            delta("g3_minus_g2", "old_query_12class_1nn_accuracy") >= -0.02
            and delta(
                "g3_minus_g2", "novel_query_12class_1nn_accuracy_diagnostic"
            )
            >= -0.02
        ),
    }
    g4_checks = {
        "motion_sensitive_macro_recall_delta_at_least_0.03": delta(
            "g4_minus_g3", "motion_sensitive_12class_macro_recall"
        )
        >= 0.03,
        "all_1nn_noninferior_minus_0.01": delta(
            "g4_minus_g3", "all_1nn_accuracy"
        )
        >= -0.01,
        "at_least_6_of_7_positive_motion_sensitive_folds": positive(
            "g4_minus_g3", "motion_sensitive_12class_macro_recall"
        )
        >= 6,
        "primary_metric_holm_adjusted_one_sided_p_at_most_0.05": paired[
            "comparisons"
        ]["g4_minus_g3"]["primary_metric_holm_adjusted_one_sided_p"]
        <= 0.05,
        "old_and_novel_drop_no_more_than_0.02": (
            delta("g4_minus_g3", "old_query_12class_1nn_accuracy") >= -0.02
            and delta(
                "g4_minus_g3", "novel_query_12class_1nn_accuracy_diagnostic"
            )
            >= -0.02
        ),
        "order_or_duration_alignment_control_passes_and_other_nonnegative": (
            (
                structural_control_passed("order")
                or structural_control_passed("duration_alignment")
            )
            and control_stats["order"].get("mean_delta", -math.inf) >= 0.0
            and control_stats["duration_alignment"].get(
                "mean_delta", -math.inf
            )
            >= 0.0
        ),
    }
    final_checks = {
        "g4_minus_g1_all_1nn_delta_at_least_0.05": delta(
            "g4_minus_g1", "all_1nn_accuracy"
        )
        >= 0.05,
        "paired_fixed_window_noninferiority_mean_delta_at_least_minus_0.01": bool(
            fixed_window_comparison.get("available", False)
            and fixed_window_comparison.get("raw_delta_test", {}).get(
                "mean_delta", -math.inf
            )
            >= -0.01
        ),
        "paired_fixed_window_noninferiority_one_sided_p_at_most_0.05": bool(
            fixed_window_comparison.get("available", False)
            and fixed_window_comparison.get("margin_shifted_test", {}).get(
                "one_sided_p_greater", 1.0
            )
            <= 0.05
        ),
    }
    return {
        "status_semantics": (
            "Exploratory engineering promotion gates. Thresholds were fixed "
            "before running this readout ablation, but the state feature family "
            "was chosen after inspecting earlier outer-test diagnostics; these "
            "passes are not confirmatory statistical evidence."
        ),
        "promotion_eligible": promotion_eligible,
        "promotion_eligibility_requirement": (
            "Exactly seven folds and four seeds per fold for every group. "
            "Smoke runs always report passed=false."
        ),
        "observed_grid": {
            "folds": fold_ids,
            "seeds_per_fold": seeds_per_fold,
            "group_row_counts": group_counts,
            "complete_exact_fold_seed_group_grid": complete_exact_grid,
        },
        "canonical_protocol_audit": protocol_audit,
        "fixed_window_paired_noninferiority": fixed_window_comparison,
        "g4_control_fold_statistics": control_stats,
        "g4_control_interpretation": (
            "Order and within-trial duration-alignment controls test structural "
            "trajectory information. Total-duration shuffle separately measures "
            "dependence on the activity-duration shortcut and is not a promotion "
            "criterion."
        ),
        "g2_dynamic_soft": {
            "checks": g2_checks,
            "passed": bool(promotion_eligible and all(g2_checks.values())),
            "mean_tied_nearest_relative_reduction": tie_relative_reduction,
        },
        "g3_state_residual": {
            "checks": g3_checks,
            "passed": bool(promotion_eligible and all(g3_checks.values())),
        },
        "g4_duration_context": {
            "checks": g4_checks,
            "passed": bool(promotion_eligible and all(g4_checks.values())),
        },
        "final_g4": {
            "checks": final_checks,
            "passed": bool(promotion_eligible and all(final_checks.values())),
        },
    }


def main() -> None:
    args = parse_args()
    validate_args(args)
    input_root = Path(args.input_root).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    if not input_root.exists():
        raise FileNotFoundError(input_root)
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(
            f"Output directory is not empty: {output_root}. Use a new path."
        )
    output_root.mkdir(parents=True, exist_ok=True)
    folds = parse_int_list(args.folds)
    seeds = parse_int_list(args.seeds)
    runs = discover_runs(input_root, folds, seeds)
    expected_pairs = {(fold, seed) for fold in folds for seed in seeds}
    actual_pairs = {(run["fold"], run["seed"]) for run in runs}
    if actual_pairs != expected_pairs:
        missing = sorted(expected_pairs - actual_pairs)
        unexpected = sorted(actual_pairs - expected_pairs)
        raise RuntimeError(
            "Discovered fold/seed grid does not match the requested Cartesian "
            f"grid; missing={missing}, unexpected={unexpected}."
        )
    if args.expected_runs > 0 and len(runs) != int(args.expected_runs):
        raise RuntimeError(
            f"Expected {args.expected_runs} runs but discovered {len(runs)} under "
            f"{input_root}."
        )
    if not runs:
        raise RuntimeError("No compatible input runs were found.")

    repositories: dict[Path, SourceSignalRepository] = {}
    for run in runs:
        source_npz = resolve_source_npz(
            run["run_dir"], PROJECT_ROOT, args.npz_path
        )
        if source_npz not in repositories:
            repositories[source_npz] = SourceSignalRepository(source_npz)
        run["source_npz"] = source_npz
    preflight = build_preflight_protocol_audit(
        args, folds, seeds, runs, repositories
    )
    write_json(output_root / "preflight_protocol_audit.json", preflight)
    if (
        args.input_protocol == "motion_encoder_v1"
        and not preflight["execution_protocol_passed"]
    ):
        raise RuntimeError(
            "motion_encoder_v1 preflight failed; refusing to construct "
            "trajectory distances. Inspect preflight_protocol_audit.json."
        )
    if not preflight["passed"]:
        print(
            "Protocol audit: non-canonical exploratory run; all promotion "
            "flags will remain false.",
            flush=True,
        )

    results = []
    all_rows = []
    all_per_class_rows = []
    all_sequence_rows = []
    for position, run in enumerate(runs, start=1):
        print(
            f"[{position}/{len(runs)}] fold={run['fold']} seed={run['seed']} "
            f"input={run['run_dir'].name}",
            flush=True,
        )
        run_output = output_root / run["run_dir"].name
        result = analyze_run(
            run, run_output, repositories[run["source_npz"]], args
        )
        results.append(result)
        all_rows.extend(result["group_rows"])
        all_per_class_rows.extend(result["per_class_rows"])
        all_sequence_rows.extend(result["sequence_rows"])

    protocol_audit = build_final_protocol_audit(preflight, results, runs)
    write_json(output_root / "canonical_protocol_audit.json", protocol_audit)
    expected_trial_hashes = {
        (int(result["fold"]), int(result["seed"])): result["hashes"][
            "trial_grid_hash"
        ]
        for result in results
    }
    if str(args.fixed_window_root).strip():
        fixed_rows = load_fixed_window_baselines(
            Path(args.fixed_window_root), expected_trial_hashes, runs
        )
        fixed_window_comparison = build_fixed_window_comparison(
            all_rows, fixed_rows, noninferiority_margin=-0.01
        )
        write_csv(output_root / "fixed_window_paired_metrics.csv", fixed_rows)
    else:
        fixed_rows = []
        fixed_window_comparison = {
            "available": False,
            "passed": False,
            "reason": "--fixed-window-root was not supplied.",
            "noninferiority_margin": -0.01,
        }
    write_json(
        output_root / "fixed_window_noninferiority.json",
        fixed_window_comparison,
    )
    write_csv(output_root / "ablation_run_metrics.csv", all_rows)
    write_csv(output_root / "ablation_per_class.csv", all_per_class_rows)
    write_csv(output_root / "trajectory_sequences.csv", all_sequence_rows)
    paired = build_paired_summary(all_rows)
    gate_report = build_gate_report(
        all_rows, paired, protocol_audit, fixed_window_comparison
    )
    write_json(output_root / "ablation_paired_summary.json", paired)
    write_json(output_root / "ablation_gate_report.json", gate_report)

    reference_names = results[0]["class_names"]
    for result in results[1:]:
        if result["class_names"] != reference_names:
            raise RuntimeError("Activity-name order differs between runs.")
    mean_activity = {
        group: np.mean(
            [result["activity_matrices"][group] for result in results], axis=0
        )
        for group in GROUP_NAMES
    }
    mean_confusion = {
        group: np.mean(
            [result["confusion_matrices"][group] for result in results], axis=0
        )
        for group in GROUP_NAMES
    }
    np.savez_compressed(
        output_root / "ablation_mean_matrices.npz",
        **{f"activity_{group}": value for group, value in mean_activity.items()},
        **{f"confusion_{group}": value for group, value in mean_confusion.items()},
        activity_names=np.asarray(reference_names),
    )
    save_heatmap_grid(
        output_root / "mean_ablation_activity_distance_heatmaps.png",
        mean_activity,
        reference_names,
        f"Mean cross-subject activity distance across {len(runs)} runs",
        "viridis",
        0.0,
        1.0,
        ".2f",
    )
    save_heatmap_grid(
        output_root / "mean_ablation_confusion_heatmaps.png",
        mean_confusion,
        reference_names,
        f"Mean tie-aware 1-NN confusion across {len(runs)} runs",
        "magma",
        0.0,
        1.0,
        ".2f",
    )
    summary = {
        "scope": "fixed SSL change-point/KMeans32 trajectory-readout ablation",
        "study_design_status": "exploratory_post_hoc",
        "input_root": str(input_root),
        "output_root": str(output_root),
        "run_count": len(runs),
        "folds": folds,
        "seeds": seeds,
        "groups": list(GROUP_NAMES),
        "weights": {
            "state": args.state_weight,
            "context": args.context_weight,
            "duration": args.duration_weight,
        },
        "control_shuffles": args.control_shuffles,
        "canonical_protocol_passed": protocol_audit["passed"],
        "fixed_window_paired_noninferiority": fixed_window_comparison,
        "group_metric_means": {
            group: {
                key: group_metric_mean(all_rows, group, key)
                for key in [
                    "all_1nn_accuracy",
                    "all_macro_f1",
                    "pairwise_auc",
                    "zero_min_distance_query_ratio",
                    "old_query_12class_1nn_accuracy",
                    "novel_query_12class_1nn_accuracy_diagnostic",
                    "sit_stand_12class_macro_recall",
                    "low_dynamic_12class_macro_recall",
                ]
            }
            for group in GROUP_NAMES
        },
        "paired_summary_file": "ablation_paired_summary.json",
        "gate_report_file": "ablation_gate_report.json",
        "protocol_audit_file": "canonical_protocol_audit.json",
        "limitations": [
            "The state feature family was chosen after inspecting earlier outer-test diagnostics; gates are exploratory engineering evidence only.",
            "This validates trajectory readout only; it does not repair codebook OOV or online drift.",
            "Novel-class labels are used only for post-hoc diagnostics, not distance construction.",
            "G2 uses frozen codebook-center cosine cost; raw constrained-DTW is a separate later ablation.",
            "Duration uses the NPZ-visible partition span and omits the incomplete raw-trial tail.",
        ],
    }
    write_json(output_root / "ablation_summary.json", summary)
    print(f"Completed {len(runs)} runs. Results: {output_root}", flush=True)


if __name__ == "__main__":
    main()
