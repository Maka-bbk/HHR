"""Resume-safe subject-CV wrapper for the peak/valley hierarchy experiment.

This entry point deliberately lives outside the motion-encoder trainer.  It may
invoke the existing formal trainer for a missing A0/A2/A3 checkpoint, but it
never edits encoder-defining source files or relaxes their checkpoint identity
checks.  Every downstream run is delegated to ``run_peak_valley_hierarchy.py``.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import inspect
import json
import math
import subprocess
import sys
import uuid
from collections import defaultdict
from itertools import product
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any, Iterable, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from experiments.motion_primitive.run_subject_cv import (  # noqa: E402
    find_checkpoint,
    find_motion_encoder_checkpoint,
    sha256_file,
    validate_motion_encoder_grid_identity,
)


SCHEMA = "peak_valley_hierarchy_cv_v2"
RUN_RESULT = "experiment_result.json"
ALLOWED_PROFILES = ("A0", "A2", "A3")
ALLOWED_ARMS = ("E0", "E1", "E2", "E3", "E4")
REGISTERED_WINDOW_GRIDS = frozenset({(256, 128), (128, 64)})
READOUT_VARIANTS = ("state", "no_state")
PRIMARY_READOUT_VARIANT = "state"
CONTROL_SPECS = {
    "C1": {"base_arm": "E2", "control_type": "matched_random_boundaries"},
    "C2": {"base_arm": "E4", "control_type": "matched_random_parents"},
}
CANONICAL_FOLDS = tuple(range(1, 8))
CANONICAL_SEEDS = (0, 5, 50, 500)
METRICS = ("all_accuracy", "old_accuracy", "new_accuracy", "h_score", "macro_f1")
CROSS_SUBJECT_DIAGNOSTIC_METRICS = (
    "mean_margin_different_minus_same",
    "probability_same_distance_is_smaller",
    "rank_separation_effect",
    "cross_subject_1nn_accuracy",
    "cluster_subject_nmi",
)
CROSS_SUBJECT_DIAGNOSTIC_DIRECTIONS = {
    "mean_margin_different_minus_same": "higher_is_better",
    "probability_same_distance_is_smaller": "higher_is_better",
    "rank_separation_effect": "higher_is_better",
    "cross_subject_1nn_accuracy": "higher_is_better",
    "cluster_subject_nmi": "lower_means_less_subject_identity_leakage",
}

# Each comparison is evaluated after averaging seeds within a held-out-subject
# fold.  This keeps the seven held-out subject-pair folds, rather than 28
# correlated seed runs, as the units of inference.  The primary contrast is
# kept outside the eight-
# item exploratory Holm family: it is the one primary comparison specified
# before this follow-up grid is run.  Earlier single-fold results have already
# been inspected, so this is deliberately not described as preregistration.
PRESPECIFIED_FOLLOWUP_PRIMARY_COMPARISON = (
    "A2_E2_state_minus_A0_E0_state",
    ("A2", "E2", "state"),
    ("A0", "E0", "state"),
)

# These eight secondary contrasts form one fixed Holm family.  With seven
# folds, the smallest attainable two-sided exact sign-flip p-value is 2/2^7;
# consequently the smallest possible first-step Holm adjusted p-value is
# 8*(2/2^7)=0.125.  The family therefore cannot reject at alpha=.05, and its
# adjusted values are reported as an explicit resolution audit rather than as
# a promise of attainable confirmatory significance.
PAIRED_COMPARISONS = (
    ("A2_E2_minus_E0", ("A2", "E2", "state"), ("A2", "E0", "state")),
    ("A2_E4_minus_E2", ("A2", "E4", "state"), ("A2", "E2", "state")),
    ("A2_E2_minus_C1", ("A2", "E2", "state"), ("A2", "C1", "state")),
    ("A2_E4_minus_C2", ("A2", "E4", "state"), ("A2", "C2", "state")),
    ("A2_minus_A0_E2", ("A2", "E2", "state"), ("A0", "E2", "state")),
    # A2 and A3 share changepoint/alignment losses; A3 alone enables
    # InfoNCE. Use A2-A3 so a positive delta consistently means that
    # removing InfoNCE helped the proposed E2 route.
    ("A2_minus_A3_E2", ("A2", "E2", "state"), ("A3", "E2", "state")),
    (
        "A2_E2_state_minus_no_state",
        ("A2", "E2", "state"),
        ("A2", "E2", "no_state"),
    ),
    (
        "A2_E4_state_minus_no_state",
        ("A2", "E4", "state"),
        ("A2", "E4", "no_state"),
    ),
)

# This fixed-window replication asks whether the A2-A3 direction is preserved
# when the segmentation is held at historical E0. It is exploratory and is
# deliberately kept outside the fixed eight-item Holm family.
EXPLORATORY_SIMPLE_COMPARISONS = (
    (
        "A2_minus_A3_E0_fixed_window",
        ("A2", "E0", "state"),
        ("A3", "E0", "state"),
    ),
)

# These difference-in-differences are exploratory and deliberately not folded
# into the fixed eight-item Holm family above. Their raw exact p-values and
# fold effect distributions are descriptive diagnostics, not confirmatory
# tests. The second interaction checks whether the effect of removing
# InfoNCE (A2-A3) changes between E2 and historical fixed-window E0.
FACTORIAL_INTERACTIONS = (
    (
        "A2_by_E2_state_difference_in_differences",
        (
            (1.0, ("A2", "E2", "state")),
            (-1.0, ("A2", "E0", "state")),
            (-1.0, ("A0", "E2", "state")),
            (1.0, ("A0", "E0", "state")),
        ),
        "(A2/E2-A2/E0)-(A0/E2-A0/E0)",
    ),
    (
        "A2_minus_A3_by_E2_state_difference_in_differences",
        (
            (1.0, ("A2", "E2", "state")),
            (-1.0, ("A3", "E2", "state")),
            (-1.0, ("A2", "E0", "state")),
            (1.0, ("A3", "E0", "state")),
        ),
        "(A2/E2-A3/E2)-(A2/E0-A3/E0)",
    ),
)
HOLM_SECONDARY_FAMILY_NAME = "eight_exploratory_paired_hscore_contrasts"
CANONICAL_TWO_SIDED_SIGN_FLIP_MIN_P = 2.0 / (2.0 ** len(CANONICAL_FOLDS))
CANONICAL_HOLM_FAMILY_SIZE = len(PAIRED_COMPARISONS)
if CANONICAL_HOLM_FAMILY_SIZE != 8:
    raise RuntimeError("The predeclared secondary Holm family must contain exactly 8 items.")
CANONICAL_HOLM_MIN_ADJUSTED_P = min(
    1.0, CANONICAL_HOLM_FAMILY_SIZE * CANONICAL_TWO_SIDED_SIGN_FLIP_MIN_P
)

COMMON_REQUIRED_GENERATED_FILES = frozenset(
    {
        "raw_cluster_predictions.npz",
        "session_manifest.csv",
        "session2_predictions.csv",
        "trial_trajectories.jsonl",
        "parent_catalog.csv",
        "activity_distance_matrices.npz",
        "descriptor_transforms.npz",
        "trajectory_sequences.png",
        "activity_distance_heatmaps.png",
        "confusion_matrices.png",
        "segment_duration_and_count_distributions.png",
        "split_audit.json",
        "cross_subject_trajectory_diagnostics.json",
        "cross_subject_trajectory_diagnostics.csv",
    }
)
ARM_REQUIRED_GENERATED_FILES = {
    "E0": frozenset({"e0_legacy_window_codebook_state.json"}),
    "E1": frozenset(
        {"e1_peak_valley_segmenter_state.json", "e1_child_codebook_state.json"}
    ),
    "E2": frozenset(
        {
            "e2_peak_valley_feature_segmenter_state.json",
            "e2_shared_child_codebook_state.json",
            "primitive_representatives.jsonl",
            "primitive_representatives.png",
            "peak_valley_waveform_examples.png",
        }
    ),
    "E3": frozenset({"e3_frequency_parent_catalog_state.json"}),
    "E4": frozenset({"e4_strict_parent_catalog_state.json"}),
}
CONTROL_REQUIRED_GENERATED_FILES = {
    "C1": frozenset({"c1_matched_random_codebook_state.json"}),
    "C2": frozenset({"c2_matched_negative_parent_catalog_state.json"}),
}


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _canonical_json(value: Any) -> str:
    return json.dumps(
        _jsonable(value), ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )


def _identity_hash(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_jsonable(value), ensure_ascii=False, indent=2), encoding="utf-8"
    )


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Cannot write an empty CSV: {path}")
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(str(key))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: (
                        _canonical_json(value)
                        if isinstance(value, (dict, list, tuple))
                        else value
                    )
                    for key, value in row.items()
                }
            )


def _parse_ints(value: str) -> tuple[int, ...]:
    values = sorted(
        {int(token.strip()) for token in str(value).split(",") if token.strip()}
    )
    if not values:
        raise ValueError("At least one integer is required.")
    return tuple(values)


def _parse_names(value: str, allowed: Iterable[str], label: str) -> tuple[str, ...]:
    allowed_set = {str(item).upper() for item in allowed}
    values: list[str] = []
    for token in str(value).split(","):
        normalized = token.strip().upper()
        if not normalized:
            continue
        if normalized not in allowed_set:
            raise ValueError(f"Unsupported {label} {normalized!r}; allowed={sorted(allowed_set)}.")
        if normalized not in values:
            values.append(normalized)
    if not values:
        raise ValueError(f"At least one {label} is required.")
    return tuple(values)


def _load_torch(path: Path) -> dict:
    import torch

    try:
        value = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        value = torch.load(path, map_location="cpu")
    if not isinstance(value, dict):
        raise TypeError(f"Checkpoint {path} must contain a dictionary.")
    return value


def _nested_close(mapping: Mapping[str, Any], path: Sequence[str], expected: Any) -> bool:
    value: Any = mapping
    for key in path:
        if not isinstance(value, Mapping) or key not in value:
            return False
        value = value[key]
    if isinstance(expected, float):
        try:
            return math.isclose(float(value), expected, rel_tol=0.0, abs_tol=1e-12)
        except (TypeError, ValueError):
            return False
    return value == expected


def _validate_profile_semantics(
    checkpoint: Path,
    profile: str,
    fold: int,
    seed: int,
    expected_npz_sha256: str,
    expected_window_size_samples: int | None = None,
) -> dict:
    payload = _load_torch(checkpoint)
    training = payload.get("resolved_training_config")
    metadata = payload.get("experiment_metadata")
    selection = payload.get("selection")
    split_audit = payload.get("split_audit")
    data = payload.get("data")
    arguments = payload.get("command_arguments")
    if (
        not isinstance(training, Mapping)
        or not isinstance(metadata, Mapping)
        or not isinstance(split_audit, Mapping)
        or not isinstance(data, Mapping)
        or not isinstance(arguments, Mapping)
    ):
        raise RuntimeError(f"Motion checkpoint lacks formal metadata: {checkpoint}")
    expected = {
        "A0": {
            ("window_aug_consistency",): "none",
            ("loss_weights", "window_augmentation"): 0.0,
            ("loss_weights", "changepoint"): 0.0,
            ("loss_weights", "content_boundary_alignment"): 0.0,
        },
        "A2": {
            ("window_aug_consistency",): "none",
            ("loss_weights", "window_augmentation"): 0.0,
            ("loss_weights", "changepoint"): 1.0,
            ("loss_weights", "content_boundary_alignment"): 0.1,
        },
        "A3": {
            ("window_aug_consistency",): "infonce",
            ("loss_weights", "window_augmentation"): 1.0,
            ("loss_weights", "changepoint"): 1.0,
            ("loss_weights", "content_boundary_alignment"): 0.1,
        },
    }[profile]
    expected.update(
        {
            ("backbone_bn_policy",): "frozen",
            ("window_aug_profile",): "basic",
            ("window_aug_one_window_per_trial",): True,
            ("cp_anchor", "source"): "raw_frozen_consensus",
            ("architecture", "content_residual"): True,
            ("loss_weights", "noncollapse"): 0.05,
            ("loss_weights", "temporal_prediction"): 0.5,
            ("loss_weights", "trial_auxiliary"): 0.1,
            ("loss_weights", "cross_subject"): 0.0,
        }
    )
    checks = {
        "profile": str(training.get("ablation_profile", "")).upper() == profile,
        "fold": int(metadata.get("uschad_cv_fold", -1)) == int(fold),
        "seed": int(metadata.get("motion_encoder_seed", -1)) == int(seed),
        "final_epoch": isinstance(selection, Mapping)
        and selection.get("policy") == "final_epoch",
        "not_smoke": not bool(metadata.get("smoke_test", False)),
        **{
            "/".join(path): _nested_close(training, path, value)
            for path, value in expected.items()
        },
        "argument_ablation_profile": str(arguments.get("ablation_profile", "")).upper()
        == profile,
        "argument_window_aug_weight_raw_one": _nested_close(
            arguments, ("window_aug_weight",), 1.0
        ),
        "argument_noncollapse_weight": _nested_close(
            arguments, ("noncollapse_weight",), 0.05
        ),
        "argument_prediction_weight": _nested_close(
            arguments, ("prediction_weight",), 0.5
        ),
        "argument_trial_weight": _nested_close(
            arguments, ("trial_weight",), 0.1
        ),
        "argument_cross_subject_weight": _nested_close(
            arguments, ("cross_subject_weight",), 0.0
        ),
    }
    recorded_npz_hashes = {
        payload.get("npz_sha256"),
        data.get("npz_sha256"),
        split_audit.get("npz_sha256"),
    }
    checks["npz_hash_fields_complete_and_consistent"] = (
        None not in recorded_npz_hashes and len(recorded_npz_hashes) == 1
    )
    checks["npz_hash_matches_requested_dataset"] = (
        recorded_npz_hashes == {str(expected_npz_sha256)}
    )
    if expected_window_size_samples is not None:
        checks["window_size_matches_explicit_request"] = (
            int(metadata.get("uschad_window_size", -1))
            == int(expected_window_size_samples)
        )
    if not all(checks.values()):
        raise RuntimeError(
            f"Checkpoint {checkpoint} violates {profile} semantics: {checks}."
        )
    return checks


def _encoder_output_dir(root: Path, profile: str, fold: int, seed: int) -> Path:
    return root / f"fold_{fold:02d}_seed_{seed}_{profile}_formal_v1"


def _unique_staging_output_path(
    final_directory: Path, *, staging_parent: Path | None = None
) -> Path:
    """Return a unique, currently non-existent staging path.

    The child process owns creation of this directory.  We intentionally do
    not clean stale siblings: an interrupted run remains available for audit,
    while a UUID makes the next retry independent of it.
    """

    final_directory = final_directory.resolve()
    parent = (
        final_directory.parent
        if staging_parent is None
        else staging_parent.expanduser().resolve()
    )
    parent.mkdir(parents=True, exist_ok=True)
    prefix = f".{final_directory.name}.staging-"
    for _ in range(128):
        candidate = parent / f"{prefix}{uuid.uuid4().hex}"
        if not candidate.exists() and not candidate.is_symlink():
            return candidate
    raise RuntimeError(f"Could not allocate a unique staging path below {parent}.")


def _atomic_publish_directory(staging: Path, final_directory: Path) -> None:
    """Atomically publish one validated staging directory without cleanup."""

    staging = staging.resolve()
    if final_directory.exists() or final_directory.is_symlink():
        raise FileExistsError(
            f"Refusing to replace an existing published directory: {final_directory}."
        )
    final_directory = final_directory.resolve()
    if not staging.is_dir():
        raise RuntimeError(f"Validated staging directory disappeared: {staging}.")
    final_directory.parent.mkdir(parents=True, exist_ok=True)
    if staging.stat().st_dev != final_directory.parent.stat().st_dev:
        raise RuntimeError(
            "Staging and final directories are on different filesystems; "
            f"atomic rename is unavailable: {staging} -> {final_directory}."
        )
    try:
        staging.rename(final_directory)
    except OSError as error:
        raise RuntimeError(
            f"Atomic staging promotion failed; staging was preserved: "
            f"{staging} -> {final_directory}."
        ) from error
    if staging.exists() or not final_directory.is_dir():
        raise RuntimeError(
            f"Atomic promotion invariant failed: {staging} -> {final_directory}."
        )


def _trainer_command(
    python: str,
    source: Path,
    npz_path: Path,
    output_dir: Path,
    profile: str,
    seed: int,
    device: str,
) -> list[str]:
    alignment = 0.0 if profile == "A0" else 0.1
    return [
        python,
        str(PROJECT_ROOT / "experiments/motion_primitive/train_motion_encoder.py"),
        "--source-checkpoint",
        str(source),
        "--npz-path",
        str(npz_path),
        "--output-dir",
        str(output_dir),
        "--ablation-profile",
        profile,
        # The formal trainer records the common raw value one for every
        # profile.  A0/A2 resolve it to an effective weight of zero because
        # their consistency method is ``none``.
        "--window-aug-weight",
        "1",
        "--content-boundary-alignment-weight",
        str(alignment),
        "--trial-weight",
        "0.1",
        "--segmentation-dim",
        "0",
        "--epochs",
        "30",
        "--trial-batch-size",
        "8",
        "--source-encode-batch-size",
        "512",
        "--learning-rate",
        "0.0001",
        "--minimum-learning-rate",
        "0.000001",
        "--weight-decay",
        "0.0001",
        "--gradient-clip-norm",
        "5",
        "--ema-momentum",
        "0.99",
        "--early-stopping-patience",
        "0",
        "--selection-policy",
        "final_epoch",
        "--known-anomaly-policy",
        "report",
        "--deterministic",
        "--device",
        device,
        "--seed",
        str(seed),
    ]


def _resolve_checkpoint(
    *,
    encoder_root: Path,
    cv_root: Path,
    npz_path: Path,
    profile: str,
    fold: int,
    seed: int,
    python: str,
    device: str,
    train_missing: bool,
    dry_run: bool,
) -> Path:
    try:
        return find_motion_encoder_checkpoint(encoder_root, fold, seed, profile)
    except RuntimeError as error:
        if "matched 0" not in str(error) or not train_missing:
            raise
    source = find_checkpoint(cv_root, fold, seed)
    output_dir = _encoder_output_dir(encoder_root, profile, fold, seed)
    if output_dir.exists() or output_dir.is_symlink():
        raise RuntimeError(
            f"Missing canonical {profile} checkpoint, but its target directory already "
            f"exists: {output_dir}. It will not be deleted or overwritten; preserve or "
            "move it before retrying."
        )
    # Keep encoder staging outside encoder_root so the legacy recursive
    # checkpoint finder cannot mistake a fully written but not-yet-promoted
    # interrupted staging checkpoint for a published canonical checkpoint.
    staging_dir = _unique_staging_output_path(
        output_dir, staging_parent=encoder_root.parent
    )
    command = _trainer_command(
        python, source, npz_path, staging_dir, profile, seed, device
    )
    print(
        f"[encoder staging] {staging_dir} -> {output_dir}\n"
        "[encoder command] " + subprocess.list2cmdline(command),
        flush=True,
    )
    if dry_run:
        return output_dir / "motion_encoder_final.pt"
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)
    staged_checkpoint = staging_dir / "motion_encoder_final.pt"
    if not staged_checkpoint.is_file():
        raise RuntimeError(
            f"Encoder trainer completed without staged final checkpoint: {staged_checkpoint}."
        )
    expected_npz_sha256 = sha256_file(npz_path)
    _validate_profile_semantics(
        staged_checkpoint, profile, fold, seed, expected_npz_sha256
    )
    # This invokes the shared checkpoint-integrity and formal-schedule audit on
    # the staged payload before its directory receives the canonical name.
    validate_motion_encoder_grid_identity(
        {(int(fold), int(seed)): staged_checkpoint}, profile
    )
    _atomic_publish_directory(staging_dir, output_dir)
    published_checkpoint = output_dir / "motion_encoder_final.pt"
    _validate_profile_semantics(
        published_checkpoint, profile, fold, seed, expected_npz_sha256
    )
    return published_checkpoint


def _runner_command(
    args: argparse.Namespace,
    checkpoint: Path,
    output_dir: Path,
    profile: str,
    fold: int,
    seed: int,
    arms: Sequence[str],
) -> list[str]:
    command = [
        str(args.python),
        str(PROJECT_ROOT / "experiments/motion_primitive/run_peak_valley_hierarchy.py"),
        "--checkpoint",
        str(checkpoint),
        "--npz-path",
        str(Path(args.npz_path).resolve()),
        "--output-dir",
        str(output_dir),
        "--expected-profile",
        profile,
        "--orchestrator-wiring-sha256",
        _runner_cli_wiring_sha256(),
        "--fold",
        str(fold),
        "--window-size-samples",
        str(args.window_size_samples),
        "--window-stride-samples",
        str(args.window_stride_samples),
        "--arms",
        ",".join(arms),
        "--old-class-count",
        str(args.old_class_count),
        "--primitive-num",
        str(args.primitive_num),
        "--pca-dim",
        str(args.pca_dim),
        "--child-feature-source",
        str(args.child_feature_source),
        "--trial-cluster-count",
        str(args.trial_cluster_count),
        "--sample-rate-hz",
        str(args.sample_rate_hz),
        "--batch-size",
        str(args.batch_size),
        "--device",
        str(args.device),
        "--seed",
        str(seed),
        "--smooth-seconds",
        str(args.smooth_seconds),
        "--prominence-mad",
        str(args.prominence_mad),
        "--extrema-distance-seconds",
        str(args.extrema_distance_seconds),
        "--vote-tolerance-seconds",
        str(args.vote_tolerance_seconds),
        "--minimum-axes",
        str(args.minimum_axes),
        "--minimum-segment-seconds",
        str(args.minimum_segment_seconds),
        "--feature-confirm-quantile",
        str(args.feature_confirm_quantile),
        "--feature-context-seconds",
        str(args.feature_context_seconds),
        "--shape-points",
        str(args.shape_points),
        "--parent-min-occurrences",
        str(args.parent_min_occurrences),
        "--parent-min-trials",
        str(args.parent_min_trials),
        "--parent-min-subjects",
        str(args.parent_min_subjects),
        "--parent-min-npmi",
        str(args.parent_min_npmi),
        "--parent-min-mdl-gain",
        str(args.parent_min_mdl_gain),
        "--parent-min-loso-stability",
        str(args.parent_min_loso_stability),
    ]
    if args.matched_random_controls:
        command.append("--matched-random-controls")
    else:
        command.append("--no-matched-random-controls")
    return command


def _runner_cli_wiring_sha256() -> str:
    """Fingerprint only the CV-to-member CLI wiring, not aggregate code."""

    source = inspect.getsource(_runner_command).encode("utf-8")
    return hashlib.sha256(source).hexdigest()


def _source_fingerprints() -> dict[str, str]:
    paths = {
        "runner": PROJECT_ROOT
        / "experiments/motion_primitive/run_peak_valley_hierarchy.py",
        "algorithm": PROJECT_ROOT
        / "experiments/motion_primitive/peak_valley_hierarchy.py",
        "core": PROJECT_ROOT / "experiments/motion_primitive/core.py",
        "run_experiment": PROJECT_ROOT
        / "experiments/motion_primitive/run_experiment.py",
        "frozen_hierarchical_readout": PROJECT_ROOT
        / "experiments/motion_primitive/frozen_hierarchical_readout.py",
        "online_hierarchical_gate": PROJECT_ROOT
        / "experiments/motion_primitive/run_online_hierarchical_gate_v2.py",
        "online_secondary_codebook": PROJECT_ROOT
        / "experiments/motion_primitive/run_online_secondary_codebook.py",
        "trajectory_ablation": PROJECT_ROOT
        / "experiments/motion_primitive/trajectory_ablation.py",
        "motion_encoder": PROJECT_ROOT
        / "experiments/motion_primitive/motion_encoder.py",
        "uschad_loader": PROJECT_ROOT / "data/uschad.py",
        "cv_wrapper": Path(__file__).resolve(),
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Peak/valley experiment source is missing: {missing}")
    return {name: sha256_file(path) for name, path in paths.items()}


def _member_source_fingerprints(
    source_fingerprints: Mapping[str, str],
) -> dict[str, str]:
    """Return only sources that define an individual run.

    The CV wrapper affects orchestration and aggregate statistics, but it does
    not affect a completed member's segmentation, features, or predictions.
    Keeping its hash out of member identities allows aggregate-only fixes to be
    applied without invalidating every expensive fold/seed run.
    """

    return {
        key: str(value)
        for key, value in source_fingerprints.items()
        if key != "cv_wrapper"
    }


def _protocol_settings(args: argparse.Namespace) -> dict[str, Any]:
    return {
        key: _jsonable(getattr(args, key))
        for key in (
            "old_class_count",
            "primitive_num",
            "pca_dim",
            "child_feature_source",
            "trial_cluster_count",
            "window_size_samples",
            "window_stride_samples",
            "batch_size",
            "device",
            "sample_rate_hz",
            "smooth_seconds",
            "prominence_mad",
            "extrema_distance_seconds",
            "vote_tolerance_seconds",
            "minimum_axes",
            "minimum_segment_seconds",
            "feature_confirm_quantile",
            "feature_context_seconds",
            "shape_points",
            "parent_min_occurrences",
            "parent_min_trials",
            "parent_min_subjects",
            "parent_min_npmi",
            "parent_min_mdl_gain",
            "parent_min_loso_stability",
            "matched_random_controls",
        )
    } | {
        # E0 is the previous KMeans-32 baseline over the explicitly registered
        # overlapping NPZ grid (historical default w256/s128; w128/s64 is the
        # prespecified window-duration screen), not arbitrary fixed fragments.
        "e0_segmentation": "legacy_npz_windows",
        "readout_variants": list(READOUT_VARIANTS),
        "evaluation_protocol": "isolated_cgcd_clustering_proxy_not_formal_happy_learner",
        "label_access_boundary": {
            "benchmark_split_builder": "label_aware_before_raw_prediction_freeze",
            "builder_internal_label_scope": (
                "closed_benchmark_split_construction_and_validation_only"
            ),
            "predictor_facing_builder_output": "trial_ids_and_subject_ids_only",
            "predictor_receives_held_out_activity_labels": False,
            "scoring_truth_join": "after_raw_prediction_artifact_is_written_and_hashed",
        },
    }


def _grid_protocol_settings(args: argparse.Namespace) -> dict[str, Any]:
    primary_name, primary_lhs, primary_rhs = PRESPECIFIED_FOLLOWUP_PRIMARY_COMPARISON
    return {
        **_protocol_settings(args),
        "staging_publication": {
            "member_staging": "unique_uuid_path_in_final_parent",
            "member_validation_before_publish": (
                "identity_required_files_size_and_sha256"
            ),
            "encoder_staging": (
                "unique_uuid_path_outside_encoder_root_on_same_filesystem"
            ),
            "encoder_validation_before_publish": (
                "profile_semantics_checkpoint_integrity_and_formal_schedule"
            ),
            "publication": "same_filesystem_atomic_directory_rename",
            "stale_staging_policy": "preserve_and_ignore_on_retry_never_auto_delete",
            "existing_invalid_final_policy": "fail_closed_never_overwrite",
        },
        "aggregate": {
            "bootstrap_replicates": int(args.bootstrap_replicates),
            "analysis_seed": int(args.analysis_seed),
            "seed_aggregation": "mean_within_held_out_subject_fold",
            "inference_unit": "held_out_subject_fold",
            "prespecified_followup_primary_hscore_comparison": {
                "name": primary_name,
                "lhs": list(primary_lhs),
                "rhs": list(primary_rhs),
                "multiplicity": "single_prespecified_followup_primary",
            },
            "secondary_holm_family": {
                "name": HOLM_SECONDARY_FAMILY_NAME,
                "comparison_names": [item[0] for item in PAIRED_COMPARISONS],
                "planned_family_size": CANONICAL_HOLM_FAMILY_SIZE,
            },
            "exploratory_unadjusted_simple_comparisons": [
                {
                    "name": name,
                    "lhs": list(lhs),
                    "rhs": list(rhs),
                    "formula": f"{lhs[0]}/{lhs[1]}-{rhs[0]}/{rhs[1]}",
                }
                for name, lhs, rhs in EXPLORATORY_SIMPLE_COMPARISONS
            ],
            "exploratory_factorial_interactions": [
                {
                    "name": name,
                    "formula": formula,
                    "terms": [
                        {"coefficient": coefficient, "endpoint": list(endpoint)}
                        for coefficient, endpoint in terms
                    ],
                }
                for name, terms, formula in FACTORIAL_INTERACTIONS
            ],
            "two_sided_exact_sign_flip_minimum_p_at_seven_folds": (
                CANONICAL_TWO_SIDED_SIGN_FLIP_MIN_P
            ),
            "holm_eight_minimum_attainable_adjusted_p_at_seven_folds": (
                CANONICAL_HOLM_MIN_ADJUSTED_P
            ),
            "bootstrap_confidence_intervals": (
                "descriptive_only_not_confirmatory_with_seven_overlapping_cv_folds"
            ),
            "cross_subject_trajectory_diagnostics": {
                "analysis_role": "post_truth_join_descriptive_diagnostic_only",
                "metrics": list(CROSS_SUBJECT_DIAGNOSTIC_METRICS),
                "favourable_directions": dict(
                    CROSS_SUBJECT_DIAGNOSTIC_DIRECTIONS
                ),
                "aggregation_order": (
                    "mean_seeds_within_held_out_subject_fold_then_summarize_folds"
                ),
                "independent_sample_size_is_not_fold_times_seed": True,
                "bootstrap_intervals": "descriptive_only_not_confirmatory",
                "not_part_of_primary_classification_test": True,
            },
        },
    }


def _run_identity(
    *,
    args: argparse.Namespace,
    profile: str,
    fold: int,
    seed: int,
    checkpoint: Path,
    arms: Sequence[str],
    source_fingerprints: Mapping[str, str],
) -> dict:
    return {
        "schema": "peak_valley_hierarchy_run_request_v1",
        "profile": profile,
        "fold": int(fold),
        "seed": int(seed),
        "arms": list(arms),
        "checkpoint_sha256": sha256_file(checkpoint),
        "npz_sha256": sha256_file(Path(args.npz_path).resolve()),
        "source_fingerprints": dict(source_fingerprints),
        "orchestrator_wiring_sha256": _runner_cli_wiring_sha256(),
        "protocol": _protocol_settings(args),
    }


def _required_generated_files(expected_identity: Mapping[str, Any]) -> set[str]:
    arms_value = expected_identity.get("arms")
    protocol = expected_identity.get("protocol")
    if not isinstance(arms_value, list) or not arms_value or not isinstance(protocol, Mapping):
        raise RuntimeError("Expected run identity lacks arms/protocol for artifact audit.")
    arms = tuple(str(value).upper() for value in arms_value)
    if len(arms) != len(set(arms)) or any(arm not in ALLOWED_ARMS for arm in arms):
        raise RuntimeError(f"Expected run identity has invalid arms: {arms}.")
    required = set(COMMON_REQUIRED_GENERATED_FILES)
    for arm in arms:
        required.update(ARM_REQUIRED_GENERATED_FILES[arm])
    # E3/E4 are overlays on the E2 child segmentation and codebook even when
    # E2 itself is not a requested output arm.
    if set(arms) & {"E3", "E4"}:
        required.update(ARM_REQUIRED_GENERATED_FILES["E2"])
    if bool(protocol.get("matched_random_controls", False)):
        if "E2" in arms:
            required.update(CONTROL_REQUIRED_GENERATED_FILES["C1"])
        if "E4" in arms:
            required.update(CONTROL_REQUIRED_GENERATED_FILES["C2"])
    return required


def _safe_manifest_relative_path(directory: Path, value: Any) -> tuple[str, Path]:
    if not isinstance(value, str) or not value:
        raise RuntimeError(f"Generated artifact has invalid relative_path={value!r}.")
    posix = PurePosixPath(value)
    windows = PureWindowsPath(value)
    if (
        posix.is_absolute()
        or windows.is_absolute()
        or ".." in posix.parts
        or ".." in windows.parts
        or "\\" in value
        or posix.as_posix() != value
        or value in {".", RUN_RESULT}
    ):
        raise RuntimeError(f"Unsafe generated artifact relative_path={value!r}.")
    root = directory.resolve()
    candidate = (root / Path(*posix.parts)).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as error:
        raise RuntimeError(
            f"Generated artifact resolves outside run directory: {value!r}."
        ) from error
    return value, candidate


def _validate_completed_run(directory: Path, expected_identity: Mapping[str, Any]) -> dict:
    result_path = directory / RUN_RESULT
    if not result_path.is_file():
        raise RuntimeError(f"Existing run is incomplete; missing {result_path}.")
    result = json.loads(result_path.read_text(encoding="utf-8"))
    observed = result.get("request_identity")
    if _canonical_json(observed) != _canonical_json(expected_identity):
        raise RuntimeError(f"Existing run identity differs: {directory}")
    generated = result.get("generated_files")
    if not isinstance(generated, list) or not generated:
        raise RuntimeError(f"Existing run lacks a generated-file manifest: {directory}")
    observed_paths: set[str] = set()
    for record in generated:
        if not isinstance(record, Mapping):
            raise RuntimeError(
                f"Generated-file manifest must contain content-addressed records: {directory}"
            )
        relative_path, candidate = _safe_manifest_relative_path(
            directory, record.get("relative_path")
        )
        if relative_path in observed_paths:
            raise RuntimeError(
                f"Generated-file manifest contains duplicate path {relative_path!r}."
            )
        observed_paths.add(relative_path)
        if not candidate.is_file():
            raise RuntimeError(
                f"Existing run is incomplete; missing artifact {relative_path!r}: {directory}"
            )
        expected_size = record.get("size_bytes")
        if (
            isinstance(expected_size, bool)
            or not isinstance(expected_size, int)
            or expected_size < 1
        ):
            raise RuntimeError(
                f"Artifact {relative_path!r} has invalid manifest size {expected_size!r}."
            )
        observed_size = int(candidate.stat().st_size)
        if observed_size != expected_size:
            raise RuntimeError(
                f"Artifact {relative_path!r} size mismatch: "
                f"manifest={expected_size}, observed={observed_size}."
            )
        expected_sha256 = record.get("sha256")
        if (
            not isinstance(expected_sha256, str)
            or len(expected_sha256) != 64
            or expected_sha256.lower() != expected_sha256
            or any(character not in "0123456789abcdef" for character in expected_sha256)
        ):
            raise RuntimeError(
                f"Artifact {relative_path!r} has invalid manifest SHA256."
            )
        observed_sha256 = sha256_file(candidate)
        if observed_sha256 != expected_sha256:
            raise RuntimeError(
                f"Artifact {relative_path!r} SHA256 mismatch: "
                f"manifest={expected_sha256}, observed={observed_sha256}."
            )
    required = _required_generated_files(expected_identity)
    missing_required = sorted(required - observed_paths)
    if missing_required:
        raise RuntimeError(
            "Existing run manifest lacks arm-required artifacts: "
            f"missing={missing_required}, directory={directory}."
        )
    return result


def _run_or_resume_member(
    *,
    args: argparse.Namespace,
    checkpoint: Path,
    final_directory: Path,
    profile: str,
    fold: int,
    seed: int,
    arms: Sequence[str],
    request_identity: Mapping[str, Any],
) -> dict:
    """Reuse a published member or build one through validated staging."""

    if final_directory.exists() or final_directory.is_symlink():
        if not bool(args.skip_existing):
            raise FileExistsError(
                f"Published run path already exists: {final_directory}; "
                "use --skip-existing only for a complete, valid member."
            )
        result = _validate_completed_run(final_directory, request_identity)
        print(f"[skip valid run] {profile} fold={fold} seed={seed}", flush=True)
        return result

    staging_directory = _unique_staging_output_path(final_directory)
    command = _runner_command(
        args, checkpoint, staging_directory, profile, fold, seed, arms
    )
    print(
        f"[run staging] {staging_directory} -> {final_directory}\n"
        "[run command] " + subprocess.list2cmdline(command),
        flush=True,
    )
    # Any failure deliberately leaves staging untouched.  A later retry gets a
    # new UUID staging path and therefore is not blocked by this residue.
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)
    staged_result = _validate_completed_run(staging_directory, request_identity)
    _atomic_publish_directory(staging_directory, final_directory)
    published_result = _validate_completed_run(final_directory, request_identity)
    if _canonical_json(staged_result) != _canonical_json(published_result):
        raise RuntimeError(
            f"Published run changed across atomic rename: {final_directory}."
        )
    return published_result


def _expected_arm_specs(
    requested_arms: Sequence[str], matched_random_controls: bool
) -> dict[str, dict[str, Any]]:
    specs = {
        str(arm): {
            "base_arm": str(arm),
            "is_control": False,
            "control_type": None,
        }
        for arm in requested_arms
    }
    if matched_random_controls:
        for control_id, spec in CONTROL_SPECS.items():
            if str(spec["base_arm"]) in specs:
                specs[control_id] = {
                    "base_arm": str(spec["base_arm"]),
                    "is_control": True,
                    "control_type": str(spec["control_type"]),
                }
    return specs


def _metric_rows(
    result: Mapping[str, Any],
    *,
    requested_arms: Sequence[str],
    matched_random_controls: bool,
) -> list[dict[str, Any]]:
    identity = result["request_identity"]
    rows: list[dict[str, Any]] = []
    arm_results = result.get("arm_results")
    if not isinstance(arm_results, Mapping) or not arm_results:
        raise RuntimeError("Single-run result lacks arm_results.")
    expected = _expected_arm_specs(requested_arms, matched_random_controls)
    observed_arm_ids = {str(key) for key in arm_results}
    if observed_arm_ids != set(expected):
        raise RuntimeError(
            "Single-run arm/control set differs from the requested protocol: "
            f"expected={sorted(expected)}, observed={sorted(observed_arm_ids)}."
        )
    for arm_id, value in sorted(arm_results.items()):
        if not isinstance(value, Mapping):
            raise RuntimeError(f"Invalid arm result {arm_id!r}.")
        arm_id = str(arm_id)
        spec = expected[arm_id]
        observed_metadata = {
            "base_arm": str(value.get("base_arm", arm_id)),
            "is_control": bool(value.get("is_control", False)),
            "control_type": value.get("control_type"),
        }
        if observed_metadata != spec:
            raise RuntimeError(
                f"Arm {arm_id!r} metadata differs from protocol: "
                f"expected={spec}, observed={observed_metadata}."
            )
        variants = value.get("readout_variants")
        if not isinstance(variants, Mapping):
            raise RuntimeError(
                f"Arm {arm_id!r} lacks state/no_state readout_variants."
            )
        observed_variants = {str(key) for key in variants}
        if observed_variants != set(READOUT_VARIANTS):
            raise RuntimeError(
                f"Arm {arm_id!r} readout variants differ: "
                f"expected={list(READOUT_VARIANTS)}, observed={sorted(observed_variants)}."
            )
        segmentation = value.get("segmentation", {})
        parent_catalog = value.get("parent_catalog", {})
        if not isinstance(segmentation, Mapping) or not isinstance(
            parent_catalog, Mapping
        ):
            raise RuntimeError(f"Arm {arm_id!r} has invalid diagnostics metadata.")
        for variant in READOUT_VARIANTS:
            variant_result = variants[variant]
            if not isinstance(variant_result, Mapping):
                raise RuntimeError(
                    f"Arm {arm_id!r}/{variant} readout result is invalid."
                )
            metrics = variant_result.get("metrics")
            readout = variant_result.get("readout", {})
            if not isinstance(metrics, Mapping) or not isinstance(readout, Mapping):
                raise RuntimeError(
                    f"Arm {arm_id!r}/{variant} lacks metrics/readout metadata."
                )
            numeric_metrics: dict[str, float] = {}
            for metric in METRICS:
                try:
                    metric_value = float(metrics[metric])
                except (KeyError, TypeError, ValueError) as error:
                    raise RuntimeError(
                        f"Arm {arm_id!r}/{variant} lacks finite {metric}."
                    ) from error
                if not math.isfinite(metric_value) or not 0.0 <= metric_value <= 1.0:
                    raise RuntimeError(
                        f"Arm {arm_id!r}/{variant} has invalid {metric}={metric_value}."
                    )
                numeric_metrics[metric] = metric_value
            rows.append(
                {
                    "profile": identity["profile"],
                    "fold": int(identity["fold"]),
                    "seed": int(identity["seed"]),
                    "arm_id": arm_id,
                    "readout_variant": variant,
                    "is_primary_readout": variant == PRIMARY_READOUT_VARIANT,
                    **observed_metadata,
                    **numeric_metrics,
                    "fit_segment_count": segmentation.get("fit_segment_count"),
                    "eval_segment_count": segmentation.get("eval_segment_count"),
                    "parent_count": parent_catalog.get("parent_count", 0),
                    "test_trial_count": readout.get("test_trial_count"),
                }
            )
    return rows


def _strict_cross_subject_metric(
    values: Mapping[str, Any],
    key: str,
    *,
    arm_id: str,
    variant: str,
    lower: float | None = None,
    upper: float | None = None,
) -> float:
    try:
        raw_value = values[key]
    except KeyError as error:
        raise RuntimeError(
            f"Arm {arm_id!r}/{variant} cross-subject diagnostics lack {key}."
        ) from error
    if isinstance(raw_value, (bool, np.bool_)):
        raise RuntimeError(
            f"Arm {arm_id!r}/{variant} has non-numeric {key}={raw_value!r}."
        )
    try:
        value = float(raw_value)
    except (TypeError, ValueError) as error:
        raise RuntimeError(
            f"Arm {arm_id!r}/{variant} has non-numeric {key}={raw_value!r}."
        ) from error
    if not math.isfinite(value):
        raise RuntimeError(
            f"Arm {arm_id!r}/{variant} has non-finite {key}={value}."
        )
    tolerance = 1e-12
    if lower is not None and value < float(lower) - tolerance:
        raise RuntimeError(
            f"Arm {arm_id!r}/{variant} has out-of-range {key}={value}."
        )
    if upper is not None and value > float(upper) + tolerance:
        raise RuntimeError(
            f"Arm {arm_id!r}/{variant} has out-of-range {key}={value}."
        )
    return value


def _cross_subject_diagnostic_rows(
    result: Mapping[str, Any],
    *,
    requested_arms: Sequence[str],
    matched_random_controls: bool,
) -> list[dict[str, Any]]:
    """Strictly flatten every arm/readout post-truth subject diagnostic."""

    identity = result.get("request_identity")
    if not isinstance(identity, Mapping):
        raise RuntimeError("Single-run result lacks request_identity.")
    try:
        profile = str(identity["profile"])
        fold = int(identity["fold"])
        seed = int(identity["seed"])
    except (KeyError, TypeError, ValueError) as error:
        raise RuntimeError("Single-run request_identity is malformed.") from error

    expected = _expected_arm_specs(requested_arms, matched_random_controls)
    arm_results = result.get("arm_results")
    if not isinstance(arm_results, Mapping) or not arm_results:
        raise RuntimeError("Single-run result lacks arm_results.")
    observed_arm_ids = {str(key) for key in arm_results}
    if observed_arm_ids != set(expected):
        raise RuntimeError(
            "Cross-subject diagnostic arm/control set differs from the requested "
            f"protocol: expected={sorted(expected)}, observed={sorted(observed_arm_ids)}."
        )

    rows: list[dict[str, Any]] = []
    for arm_key, arm_result in sorted(arm_results.items()):
        arm_id = str(arm_key)
        if not isinstance(arm_result, Mapping):
            raise RuntimeError(f"Invalid arm result {arm_id!r}.")
        expected_metadata = expected[arm_id]
        observed_metadata = {
            "base_arm": str(arm_result.get("base_arm", arm_id)),
            "is_control": bool(arm_result.get("is_control", False)),
            "control_type": arm_result.get("control_type"),
        }
        if observed_metadata != expected_metadata:
            raise RuntimeError(
                f"Arm {arm_id!r} metadata differs from protocol: "
                f"expected={expected_metadata}, observed={observed_metadata}."
            )
        variants = arm_result.get("readout_variants")
        if not isinstance(variants, Mapping):
            raise RuntimeError(
                f"Arm {arm_id!r} lacks state/no_state readout_variants."
            )
        observed_variants = {str(key) for key in variants}
        if observed_variants != set(READOUT_VARIANTS):
            raise RuntimeError(
                f"Arm {arm_id!r} cross-subject diagnostic variants differ: "
                f"expected={list(READOUT_VARIANTS)}, "
                f"observed={sorted(observed_variants)}."
            )
        for variant in READOUT_VARIANTS:
            variant_result = variants[variant]
            if not isinstance(variant_result, Mapping):
                raise RuntimeError(
                    f"Arm {arm_id!r}/{variant} readout result is invalid."
                )
            diagnostic = variant_result.get("cross_subject_diagnostics")
            if not isinstance(diagnostic, Mapping):
                raise RuntimeError(
                    f"Arm {arm_id!r}/{variant} lacks cross_subject_diagnostics."
                )
            if diagnostic.get("post_truth_join_diagnostic_only") is not True:
                raise RuntimeError(
                    f"Arm {arm_id!r}/{variant} cross-subject diagnostic is not "
                    "marked post-truth descriptive-only."
                )
            if diagnostic.get("used_to_fit_or_modify_predictions") is not False:
                raise RuntimeError(
                    f"Arm {arm_id!r}/{variant} cross-subject diagnostic may have "
                    "modified predictions."
                )
            block_names = (
                "cross_subject_distance_effect",
                "tie_aware_cross_subject_1nn",
                "cluster_subject_nmi",
            )
            blocks: dict[str, Mapping[str, Any]] = {}
            for block_name in block_names:
                block = diagnostic.get(block_name)
                if not isinstance(block, Mapping):
                    raise RuntimeError(
                        f"Arm {arm_id!r}/{variant} lacks diagnostic block "
                        f"{block_name}."
                    )
                if block.get("available") is not True:
                    raise RuntimeError(
                        f"Arm {arm_id!r}/{variant} diagnostic block {block_name} "
                        f"is unavailable: reason={block.get('reason')!r}."
                    )
                blocks[block_name] = block

            effect = blocks["cross_subject_distance_effect"]
            nearest = blocks["tie_aware_cross_subject_1nn"]
            nmi = blocks["cluster_subject_nmi"]
            margin = _strict_cross_subject_metric(
                effect,
                "mean_margin_different_minus_same",
                arm_id=arm_id,
                variant=variant,
            )
            probability = _strict_cross_subject_metric(
                effect,
                "probability_same_distance_is_smaller",
                arm_id=arm_id,
                variant=variant,
                lower=0.0,
                upper=1.0,
            )
            rank_effect = _strict_cross_subject_metric(
                effect,
                "rank_separation_effect",
                arm_id=arm_id,
                variant=variant,
                lower=-1.0,
                upper=1.0,
            )
            if not math.isclose(
                rank_effect, 2.0 * probability - 1.0, rel_tol=0.0, abs_tol=1e-12
            ):
                raise RuntimeError(
                    f"Arm {arm_id!r}/{variant} has inconsistent probability/rank "
                    "cross-subject diagnostics."
                )
            one_nn = _strict_cross_subject_metric(
                nearest,
                "accuracy",
                arm_id=arm_id,
                variant=variant,
                lower=0.0,
                upper=1.0,
            )
            subject_nmi = _strict_cross_subject_metric(
                nmi,
                "normalized_mutual_information",
                arm_id=arm_id,
                variant=variant,
                lower=0.0,
                upper=1.0,
            )
            rows.append(
                {
                    "profile": profile,
                    "fold": fold,
                    "seed": seed,
                    "arm_id": arm_id,
                    "readout_variant": variant,
                    "is_primary_readout": variant == PRIMARY_READOUT_VARIANT,
                    **observed_metadata,
                    "mean_margin_different_minus_same": margin,
                    "probability_same_distance_is_smaller": probability,
                    "rank_separation_effect": rank_effect,
                    "cross_subject_1nn_accuracy": one_nn,
                    "cluster_subject_nmi": subject_nmi,
                    "post_truth_join_diagnostic_only": True,
                    "used_to_fit_or_modify_predictions": False,
                }
            )
    return rows


def _validate_cross_subject_diagnostic_grid(
    rows: Sequence[Mapping[str, Any]],
    *,
    profiles: Sequence[str],
    folds: Sequence[int],
    seeds: Sequence[int],
    requested_arms: Sequence[str],
    matched_random_controls: bool,
) -> None:
    expected_arms = _expected_arm_specs(
        requested_arms, matched_random_controls
    )
    expected = {
        (profile, int(fold), int(seed), arm_id, variant)
        for profile in profiles
        for fold in folds
        for seed in seeds
        for arm_id in expected_arms
        for variant in READOUT_VARIANTS
    }
    counts: dict[tuple[str, int, int, str, str], int] = defaultdict(int)
    for row in rows:
        key = (
            str(row["profile"]),
            int(row["fold"]),
            int(row["seed"]),
            str(row["arm_id"]),
            str(row["readout_variant"]),
        )
        counts[key] += 1
    observed = set(counts)
    duplicates = sorted(key for key, count in counts.items() if count != 1)
    missing = sorted(expected - observed)
    unexpected = sorted(observed - expected)
    if duplicates or missing or unexpected:
        raise RuntimeError(
            "Cross-subject diagnostic grid is incomplete, duplicated, or mixed "
            f"with another protocol: duplicates={duplicates[:8]}, "
            f"missing={missing[:8]}, unexpected={unexpected[:8]}."
        )


def _aggregate_cross_subject_diagnostic_rows(
    rows: Sequence[Mapping[str, Any]], replicates: int, seed: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    by_fold: dict[
        tuple[str, str, str, int], list[Mapping[str, Any]]
    ] = defaultdict(list)
    for row in rows:
        by_fold[
            (
                str(row["profile"]),
                str(row["arm_id"]),
                str(row["readout_variant"]),
                int(row["fold"]),
            )
        ].append(row)

    fold_rows: list[dict[str, Any]] = []
    for (profile, arm_id, variant, fold), members in sorted(by_fold.items()):
        metadata = {
            (
                str(item["base_arm"]),
                bool(item["is_control"]),
                item.get("control_type"),
            )
            for item in members
        }
        if len(metadata) != 1:
            raise RuntimeError(
                f"Cross-subject diagnostic metadata varies within "
                f"{profile}/{arm_id}/{variant}/fold={fold}."
            )
        base_arm, is_control, control_type = next(iter(metadata))
        fold_rows.append(
            {
                "profile": profile,
                "arm_id": arm_id,
                "readout_variant": variant,
                "is_primary_readout": variant == PRIMARY_READOUT_VARIANT,
                "fold": fold,
                "seed_count": len(members),
                "seeds": sorted(int(item["seed"]) for item in members),
                "base_arm": base_arm,
                "is_control": is_control,
                "control_type": control_type,
                **{
                    metric: float(
                        np.mean([float(item[metric]) for item in members])
                    )
                    for metric in CROSS_SUBJECT_DIAGNOSTIC_METRICS
                },
                "aggregation_order": "mean_seeds_within_fold",
                "analysis_role": "post_truth_join_descriptive_diagnostic_only",
            }
        )

    grouped: dict[
        tuple[str, str, str], list[Mapping[str, Any]]
    ] = defaultdict(list)
    for row in fold_rows:
        grouped[
            (
                str(row["profile"]),
                str(row["arm_id"]),
                str(row["readout_variant"]),
            )
        ].append(row)

    aggregate: list[dict[str, Any]] = []
    counter = 0
    for (profile, arm_id, variant), members in sorted(grouped.items()):
        common = {
            "profile": profile,
            "arm_id": arm_id,
            "readout_variant": variant,
            "is_primary_readout": variant == PRIMARY_READOUT_VARIANT,
            "base_arm": members[0]["base_arm"],
            "is_control": members[0]["is_control"],
            "control_type": members[0]["control_type"],
            "folds": sorted(int(item["fold"]) for item in members),
            "seed_count_per_fold": sorted(
                {int(item["seed_count"]) for item in members}
            ),
            "aggregation_order": (
                "mean_seeds_within_fold_then_describe_fold_means"
            ),
            "analysis_role": "post_truth_join_descriptive_diagnostic_only",
            "inference_unit": "held_out_subject_fold_not_fold_times_seed_run",
            "bootstrap_ci_interpretation": "descriptive_only_not_confirmatory",
            "not_part_of_primary_classification_test": True,
        }
        for metric in CROSS_SUBJECT_DIAGNOSTIC_METRICS:
            summary = _bootstrap_mean(
                [float(item[metric]) for item in members],
                seed=int(seed) + counter,
                replicates=int(replicates),
            )
            counter += 1
            aggregate.append(
                {
                    **common,
                    "metric": metric,
                    "favourable_direction": (
                        CROSS_SUBJECT_DIAGNOSTIC_DIRECTIONS[metric]
                    ),
                    **summary,
                }
            )
    return fold_rows, aggregate


def _bootstrap_mean(values: Sequence[float], seed: int, replicates: int) -> dict:
    data = np.asarray(values, dtype=np.float64)
    if data.ndim != 1 or len(data) == 0 or not np.all(np.isfinite(data)):
        raise ValueError("Bootstrap input must be a finite non-empty vector.")
    rng = np.random.default_rng(int(seed))
    draws = rng.choice(data, size=(int(replicates), len(data)), replace=True).mean(axis=1)
    lower, upper = np.quantile(draws, [0.025, 0.975])
    return {
        "fold_mean": float(np.mean(data)),
        "fold_std": float(np.std(data, ddof=1)) if len(data) > 1 else 0.0,
        "fold_bootstrap_ci95_lower": float(lower),
        "fold_bootstrap_ci95_upper": float(upper),
        "fold_count": int(len(data)),
    }


def _aggregate_rows(
    rows: Sequence[Mapping[str, Any]], replicates: int, seed: int
) -> tuple[list[dict], list[dict]]:
    by_fold: dict[tuple[str, str, str, int], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        by_fold[
            (
                str(row["profile"]),
                str(row["arm_id"]),
                str(row["readout_variant"]),
                int(row["fold"]),
            )
        ].append(row)
    fold_rows: list[dict] = []
    for (profile, arm_id, variant, fold), members in sorted(by_fold.items()):
        seed_values = sorted(int(item["seed"]) for item in members)
        fold_rows.append(
            {
                "profile": profile,
                "arm_id": arm_id,
                "readout_variant": variant,
                "is_primary_readout": variant == PRIMARY_READOUT_VARIANT,
                "fold": fold,
                "seed_count": len(members),
                "seeds": seed_values,
                "base_arm": members[0]["base_arm"],
                "is_control": members[0]["is_control"],
                "control_type": members[0]["control_type"],
                **{
                    metric: float(np.mean([float(item[metric]) for item in members]))
                    for metric in METRICS
                },
                **{
                    field: float(
                        np.mean(
                            [
                                float(item[field])
                                for item in members
                                if item.get(field) is not None
                            ]
                        )
                    )
                    for field in (
                        "fit_segment_count",
                        "eval_segment_count",
                        "parent_count",
                        "test_trial_count",
                    )
                    if any(item.get(field) is not None for item in members)
                },
            }
        )
    grouped: dict[tuple[str, str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in fold_rows:
        grouped[
            (
                str(row["profile"]),
                str(row["arm_id"]),
                str(row["readout_variant"]),
            )
        ].append(row)
    aggregate: list[dict] = []
    counter = 0
    for (profile, arm_id, variant), members in sorted(grouped.items()):
        common = {
            "profile": profile,
            "arm_id": arm_id,
            "readout_variant": variant,
            "is_primary_readout": variant == PRIMARY_READOUT_VARIANT,
            "base_arm": members[0]["base_arm"],
            "is_control": members[0]["is_control"],
            "control_type": members[0]["control_type"],
            "folds": sorted(int(item["fold"]) for item in members),
            "seed_count_per_fold": sorted(
                {int(item["seed_count"]) for item in members}
            ),
            "bootstrap_ci_interpretation": "descriptive_only_not_confirmatory",
        }
        for metric in METRICS:
            summary = _bootstrap_mean(
                [float(item[metric]) for item in members],
                seed=int(seed) + counter,
                replicates=int(replicates),
            )
            counter += 1
            aggregate.append({**common, "metric": metric, **summary})
    return fold_rows, aggregate


def _validate_metric_grid(
    rows: Sequence[Mapping[str, Any]],
    *,
    profiles: Sequence[str],
    folds: Sequence[int],
    seeds: Sequence[int],
    requested_arms: Sequence[str],
    matched_random_controls: bool,
) -> None:
    expected_arms = _expected_arm_specs(
        requested_arms, matched_random_controls
    )
    expected = {
        (profile, int(fold), int(seed), arm_id, variant)
        for profile in profiles
        for fold in folds
        for seed in seeds
        for arm_id in expected_arms
        for variant in READOUT_VARIANTS
    }
    counts: dict[tuple[str, int, int, str, str], int] = defaultdict(int)
    for row in rows:
        key = (
            str(row["profile"]),
            int(row["fold"]),
            int(row["seed"]),
            str(row["arm_id"]),
            str(row["readout_variant"]),
        )
        counts[key] += 1
    observed = set(counts)
    duplicates = sorted(key for key, count in counts.items() if count != 1)
    missing = sorted(expected - observed)
    unexpected = sorted(observed - expected)
    if duplicates or missing or unexpected:
        raise RuntimeError(
            "Metric grid is incomplete, duplicated, or mixed with another protocol: "
            f"duplicates={duplicates[:8]}, missing={missing[:8]}, "
            f"unexpected={unexpected[:8]}."
        )


def _exact_sign_flip_pvalue(deltas: Sequence[float]) -> float:
    values = np.asarray(deltas, dtype=np.float64)
    if values.ndim != 1 or len(values) == 0 or not np.all(np.isfinite(values)):
        raise ValueError("Sign-flip input must be a finite non-empty vector.")
    observed = abs(float(np.mean(values)))
    null_statistics = np.asarray(
        [
            abs(float(np.mean(values * np.asarray(signs, dtype=np.float64))))
            for signs in product((-1.0, 1.0), repeat=len(values))
        ],
        dtype=np.float64,
    )
    # This is an exact enumeration, so no Monte-Carlo +1 correction is used.
    tolerance = np.finfo(np.float64).eps * max(1.0, observed) * 16.0
    return float(np.mean(null_statistics >= observed - tolerance))


def _minimum_two_sided_exact_sign_flip_pvalue(block_count: int) -> float:
    if (
        isinstance(block_count, bool)
        or int(block_count) != block_count
        or int(block_count) < 1
    ):
        raise ValueError("block_count must be a positive integer.")
    # The absolute statistic makes a sign vector and its global negation
    # identical, so at least two of the 2^n assignments are as extreme as a
    # uniquely maximal observed assignment.
    return float(min(1.0, 2.0 / (2.0 ** int(block_count))))


def _holm_adjusted_pvalues(pvalues: Sequence[float]) -> list[float]:
    values = np.asarray(pvalues, dtype=np.float64)
    if (
        values.ndim != 1
        or len(values) < 1
        or not np.all(np.isfinite(values))
        or np.any(values < 0.0)
        or np.any(values > 1.0)
    ):
        raise ValueError("Holm adjustment requires finite p-values in [0,1].")
    order = np.argsort(values, kind="stable")
    adjusted = np.empty_like(values)
    running_maximum = 0.0
    family_size = len(values)
    for rank, original_index in enumerate(order):
        candidate = (family_size - rank) * float(values[original_index])
        running_maximum = max(running_maximum, candidate)
        adjusted[original_index] = min(1.0, running_maximum)
    return adjusted.astype(float).tolist()


def _paired_hscore_rows(
    fold_rows: Sequence[Mapping[str, Any]],
    *,
    expected_folds: Sequence[int],
    bootstrap_replicates: int,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    index: dict[tuple[str, str, str, int], Mapping[str, Any]] = {}
    for row in fold_rows:
        key = (
            str(row["profile"]),
            str(row["arm_id"]),
            str(row["readout_variant"]),
            int(row["fold"]),
        )
        if key in index:
            raise RuntimeError(f"Duplicate fold-level metric row: {key}.")
        index[key] = row
    primary_name, primary_lhs, primary_rhs = PRESPECIFIED_FOLLOWUP_PRIMARY_COMPARISON
    specifications: list[dict[str, Any]] = [
        {
            "name": primary_name,
            "analysis_role": "prespecified_followup_primary",
            "comparison_kind": "simple_difference",
            "formula": f"{primary_lhs}-{primary_rhs}",
            "terms": ((1.0, primary_lhs), (-1.0, primary_rhs)),
            "lhs": primary_lhs,
            "rhs": primary_rhs,
            "multiplicity_family": "single_prespecified_followup_primary",
        }
    ]
    specifications.extend(
        {
            "name": name,
            "analysis_role": "exploratory_secondary",
            "comparison_kind": "simple_difference",
            "formula": f"{lhs}-{rhs}",
            "terms": ((1.0, lhs), (-1.0, rhs)),
            "lhs": lhs,
            "rhs": rhs,
            "multiplicity_family": HOLM_SECONDARY_FAMILY_NAME,
        }
        for name, lhs, rhs in PAIRED_COMPARISONS
    )
    specifications.extend(
        {
            "name": name,
            "analysis_role": "exploratory_fixed_window_replication",
            "comparison_kind": "simple_difference",
            "formula": f"{lhs[0]}/{lhs[1]}-{rhs[0]}/{rhs[1]}",
            "terms": ((1.0, lhs), (-1.0, rhs)),
            "lhs": lhs,
            "rhs": rhs,
            "multiplicity_family": None,
        }
        for name, lhs, rhs in EXPLORATORY_SIMPLE_COMPARISONS
    )
    specifications.extend(
        {
            "name": name,
            "analysis_role": "exploratory_factorial_interaction",
            "comparison_kind": "difference_in_differences",
            "formula": formula,
            "terms": terms,
            "lhs": None,
            "rhs": None,
            "multiplicity_family": None,
        }
        for name, terms, formula in FACTORIAL_INTERACTIONS
    )

    rows: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    expected_fold_set = {int(fold) for fold in expected_folds}
    ordered_folds = sorted(expected_fold_set)
    if not ordered_folds:
        raise ValueError("At least one expected fold is required for paired analysis.")
    for comparison_index, specification in enumerate(specifications):
        name = str(specification["name"])
        terms = tuple(specification["terms"])
        endpoint_folds = {
            endpoint: {
                fold
                for profile, arm, variant, fold in index
                if (profile, arm, variant) == endpoint
            }
            for _, endpoint in terms
        }
        missing_endpoints = [
            endpoint for endpoint, folds in endpoint_folds.items() if not folds
        ]
        contrast_terms = [
            {
                "coefficient": float(coefficient),
                "profile": endpoint[0],
                "arm": endpoint[1],
                "readout_variant": endpoint[2],
            }
            for coefficient, endpoint in terms
        ]
        if missing_endpoints:
            skipped.append(
                {
                    "comparison": name,
                    "analysis_role": specification["analysis_role"],
                    "reason": "endpoint_not_requested",
                    "missing_endpoints": [list(item) for item in missing_endpoints],
                    "contrast_terms": contrast_terms,
                }
            )
            continue
        incomplete = {
            endpoint: folds
            for endpoint, folds in endpoint_folds.items()
            if folds != expected_fold_set
        }
        if incomplete:
            raise RuntimeError(
                f"Paired comparison {name} lacks complete folds: "
                f"observed={{{', '.join(f'{key}: {sorted(value)}' for key, value in incomplete.items())}}}, "
                f"expected={ordered_folds}."
            )
        deltas = [
            float(
                sum(
                    float(coefficient)
                    * float(index[(*endpoint, fold)]["h_score"])
                    for coefficient, endpoint in terms
                )
            )
            for fold in ordered_folds
        ]
        bootstrap = _bootstrap_mean(
            deltas,
            seed=int(seed) + comparison_index,
            replicates=int(bootstrap_replicates),
        )
        lhs = specification["lhs"]
        rhs = specification["rhs"]
        exact_p = _exact_sign_flip_pvalue(deltas)
        analysis_role = str(specification["analysis_role"])
        rows.append(
            {
                "comparison": name,
                "analysis_role": analysis_role,
                "comparison_kind": specification["comparison_kind"],
                "contrast_formula": specification["formula"],
                "contrast_terms": contrast_terms,
                "metric": "h_score",
                "lhs_profile": None if lhs is None else lhs[0],
                "lhs_arm": None if lhs is None else lhs[1],
                "lhs_readout_variant": None if lhs is None else lhs[2],
                "rhs_profile": None if rhs is None else rhs[0],
                "rhs_arm": None if rhs is None else rhs[1],
                "rhs_readout_variant": None if rhs is None else rhs[2],
                "fold_count": len(deltas),
                "folds": ordered_folds,
                "mean_paired_delta": float(np.mean(deltas)),
                "median_paired_delta": float(np.median(deltas)),
                "bootstrap_ci95_lower": bootstrap["fold_bootstrap_ci95_lower"],
                "bootstrap_ci95_upper": bootstrap["fold_bootstrap_ci95_upper"],
                "bootstrap_ci_interpretation": "descriptive_only_not_confirmatory",
                "positive_fold_count": int(np.sum(np.asarray(deltas) > 0.0)),
                "negative_fold_count": int(np.sum(np.asarray(deltas) < 0.0)),
                "tie_fold_count": int(np.sum(np.asarray(deltas) == 0.0)),
                "exact_sign_flip_p_two_sided": exact_p,
                "minimum_attainable_exact_sign_flip_p_two_sided": (
                    _minimum_two_sided_exact_sign_flip_pvalue(len(deltas))
                ),
                "multiplicity_family": specification["multiplicity_family"],
                "planned_multiplicity_family_size": (
                    CANONICAL_HOLM_FAMILY_SIZE
                    if specification["multiplicity_family"]
                    == HOLM_SECONDARY_FAMILY_NAME
                    else 1 if analysis_role == "prespecified_followup_primary" else None
                ),
                "holm_adjusted_p_two_sided": None,
                "holm_family_status": (
                    "not_applicable_single_prespecified_followup_primary"
                    if analysis_role == "prespecified_followup_primary"
                    else "pending_family_completeness_check"
                    if specification["multiplicity_family"]
                    == HOLM_SECONDARY_FAMILY_NAME
                    else "not_applicable_unadjusted_exploratory"
                ),
                "holm_reject_at_alpha_0_05": None,
                "unadjusted_reject_at_alpha_0_05": (
                    bool(exact_p <= 0.05)
                    if analysis_role == "prespecified_followup_primary"
                    else None
                ),
                "fold_deltas": deltas,
                "inference_unit": "held_out_subject_fold_after_seed_average",
            }
        )

    secondary_rows = [
        row
        for row in rows
        if row["multiplicity_family"] == HOLM_SECONDARY_FAMILY_NAME
    ]
    if len(secondary_rows) == CANONICAL_HOLM_FAMILY_SIZE:
        adjusted = _holm_adjusted_pvalues(
            [float(row["exact_sign_flip_p_two_sided"]) for row in secondary_rows]
        )
        for row, adjusted_p in zip(secondary_rows, adjusted):
            row["holm_adjusted_p_two_sided"] = float(adjusted_p)
            row["holm_family_status"] = "complete_eight_item_family"
            row["holm_reject_at_alpha_0_05"] = bool(adjusted_p <= 0.05)
            row["minimum_attainable_holm_adjusted_p_two_sided"] = (
                CANONICAL_HOLM_MIN_ADJUSTED_P
            )
    else:
        for row in secondary_rows:
            row["holm_family_status"] = (
                "incomplete_planned_eight_item_family_no_adjusted_p"
            )
            row["minimum_attainable_holm_adjusted_p_two_sided"] = (
                CANONICAL_HOLM_MIN_ADJUSTED_P
            )
    return rows, skipped


def _save_metric_plot(path: Path, aggregate: Sequence[Mapping[str, Any]]) -> bool:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as error:  # pragma: no cover - optional plotting dependency
        print(f"[warning] aggregate metric plot skipped: {error}", flush=True)
        return False
    metrics = ("all_accuracy", "old_accuracy", "new_accuracy", "h_score")
    profiles = sorted({str(row["profile"]) for row in aggregate})
    arm_variants = sorted(
        {
            (str(row["arm_id"]), str(row["readout_variant"]))
            for row in aggregate
        }
    )
    indexed = {
        (
            str(row["profile"]),
            str(row["arm_id"]),
            str(row["readout_variant"]),
            str(row["metric"]),
        ): row
        for row in aggregate
    }
    fig, axes = plt.subplots(
        len(profiles), len(metrics), figsize=(4.5 * len(metrics), 3.8 * len(profiles)),
        squeeze=False, constrained_layout=True
    )
    for r, profile in enumerate(profiles):
        for c, metric in enumerate(metrics):
            axis = axes[r, c]
            available = [
                (arm, variant)
                for arm, variant in arm_variants
                if (profile, arm, variant, metric) in indexed
            ]
            values = [
                float(indexed[(profile, arm, variant, metric)]["fold_mean"])
                for arm, variant in available
            ]
            lower = [
                float(
                    indexed[(profile, arm, variant, metric)][
                        "fold_bootstrap_ci95_lower"
                    ]
                )
                for arm, variant in available
            ]
            upper = [
                float(
                    indexed[(profile, arm, variant, metric)][
                        "fold_bootstrap_ci95_upper"
                    ]
                )
                for arm, variant in available
            ]
            positions = np.arange(len(available))
            axis.bar(
                positions,
                values,
                color=[
                    "#718096"
                    if arm in CONTROL_SPECS
                    else ("#3182ce" if variant == "state" else "#ed8936")
                    for arm, variant in available
                ],
            )
            axis.errorbar(
                positions,
                values,
                yerr=[np.asarray(values) - np.asarray(lower), np.asarray(upper) - np.asarray(values)],
                fmt="none", ecolor="black", capsize=2, linewidth=1,
            )
            labels = [
                arm if variant == "state" else f"{arm}/no-state"
                for arm, variant in available
            ]
            axis.set_xticks(positions, labels, rotation=60, ha="right", fontsize=7)
            axis.set_ylim(0.0, 1.0)
            axis.set_title(f"{profile} · {metric}")
            axis.grid(axis="y", alpha=0.2)
    fig.suptitle("Peak/valley hierarchy CGCD metrics (seed mean within subject fold)")
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return True


def _save_mean_activity_heatmaps(
    path: Path, run_directories: Sequence[Path]
) -> bool:
    matrices: dict[tuple[str, str], list[np.ndarray]] = defaultdict(list)
    names: list[str] | None = None
    for directory in run_directories:
        artifact = directory / "activity_distance_matrices.npz"
        if not artifact.is_file():
            continue
        with np.load(artifact, allow_pickle=True) as data:
            current_names = [str(value) for value in data["activity_names"].tolist()]
            if names is None:
                names = current_names
            elif names != current_names:
                raise RuntimeError("Activity names differ across peak/valley runs.")
            profile = str(data["profile"].item())
            for key in data.files:
                if key.startswith("matrix__"):
                    matrices[(profile, key[len("matrix__") :])].append(
                        np.asarray(data[key], dtype=np.float64)
                    )
    if not matrices or names is None:
        return False
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as error:  # pragma: no cover
        print(f"[warning] aggregate heatmap skipped: {error}", flush=True)
        return False
    keys = sorted(matrices)
    columns = min(4, len(keys))
    rows = int(math.ceil(len(keys) / columns))
    fig, axes = plt.subplots(
        rows, columns, figsize=(5.2 * columns, 4.8 * rows), squeeze=False,
        constrained_layout=True,
    )
    for axis, key in zip(axes.ravel(), keys):
        mean = np.mean(np.stack(matrices[key], axis=0), axis=0)
        image = axis.imshow(mean, cmap="viridis", vmin=0.0)
        axis.set_title(f"{key[0]} · {key[1]}")
        axis.set_xticks(range(len(names)), names, rotation=70, ha="right", fontsize=6)
        axis.set_yticks(range(len(names)), names, fontsize=6)
        fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    for axis in axes.ravel()[len(keys) :]:
        axis.axis("off")
    fig.suptitle("Mean test-activity trajectory-feature distances")
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return True


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Resume-safe multi-fold A0/A2/A3 peak/valley hierarchy experiment."
    )
    parser.add_argument("--cv-root", required=True)
    parser.add_argument("--npz-path", required=True)
    parser.add_argument("--encoder-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--profiles", default="A0,A2,A3")
    parser.add_argument("--arms", default="E0,E1,E2,E3,E4")
    parser.add_argument("--folds", default="1,2,3,4,5,6,7")
    parser.add_argument("--seeds", default="0,5,50,500")
    parser.add_argument(
        "--window-size-samples",
        type=int,
        choices=(128, 256),
        default=256,
        help="Explicit temporal length of the registered NPZ/encoder window grid.",
    )
    parser.add_argument(
        "--window-stride-samples",
        type=int,
        choices=(64, 128),
        default=128,
        help="Explicit stride of the registered NPZ window grid.",
    )
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument(
        "--train-missing-encoders", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--matched-random-controls", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--old-class-count", type=int, choices=(6,), default=6)
    parser.add_argument(
        "--primitive-num",
        type=int,
        choices=(32,),
        default=32,
        help="This hypothesis-validation grid is deliberately pinned to K=32.",
    )
    parser.add_argument("--pca-dim", type=int, choices=(64,), default=64)
    parser.add_argument(
        "--child-feature-source",
        choices=("content_embedding", "hybrid", "raw_shape"),
        default="content_embedding",
        help=(
            "Feature used to cluster variable-length child segments. The default "
            "resamples each segment to the encoder input length and uses its frozen "
            "content embedding; raw_shape is a diagnostic only."
        ),
    )
    parser.add_argument("--trial-cluster-count", type=int, choices=(10,), default=10)
    parser.add_argument("--sample-rate-hz", type=float, default=100.0)
    parser.add_argument("--smooth-seconds", type=float, default=0.15)
    parser.add_argument("--prominence-mad", type=float, default=1.5)
    parser.add_argument("--extrema-distance-seconds", type=float, default=0.20)
    parser.add_argument("--vote-tolerance-seconds", type=float, default=0.10)
    parser.add_argument("--minimum-axes", type=int, default=2)
    parser.add_argument("--minimum-segment-seconds", type=float, default=0.25)
    parser.add_argument("--feature-confirm-quantile", type=float, default=0.50)
    parser.add_argument("--feature-context-seconds", type=float, default=0.50)
    parser.add_argument("--shape-points", type=int, default=64)
    parser.add_argument("--parent-min-occurrences", type=int, default=10)
    parser.add_argument("--parent-min-trials", type=int, default=6)
    parser.add_argument("--parent-min-subjects", type=int, default=2)
    parser.add_argument("--parent-min-npmi", type=float, default=0.0)
    parser.add_argument("--parent-min-mdl-gain", type=float, default=0.0)
    parser.add_argument("--parent-min-loso-stability", type=float, default=0.8)
    parser.add_argument("--bootstrap-replicates", type=int, default=10000)
    parser.add_argument("--analysis-seed", type=int, default=20260906)
    return parser


def run(args: argparse.Namespace) -> dict:
    folds = _parse_ints(args.folds)
    seeds = _parse_ints(args.seeds)
    profiles = _parse_names(args.profiles, ALLOWED_PROFILES, "profile")
    arms = _parse_names(args.arms, ALLOWED_ARMS, "arm")
    requested_window_grid = (
        int(args.window_size_samples),
        int(args.window_stride_samples),
    )
    if requested_window_grid not in REGISTERED_WINDOW_GRIDS:
        raise ValueError(
            "The registered window grid must be one of "
            f"{sorted(REGISTERED_WINDOW_GRIDS)}; got {requested_window_grid}."
        )
    if any(fold not in CANONICAL_FOLDS for fold in folds):
        raise ValueError(f"Folds must be within {CANONICAL_FOLDS}.")
    if int(args.primitive_num) != 32:
        raise ValueError("This experiment is pinned to --primitive-num 32.")
    if (
        int(args.old_class_count) != 6
        or int(args.pca_dim) != 64
        or int(args.trial_cluster_count) != 10
    ):
        raise ValueError(
            "The historical comparison is pinned to old6, PCA64, and K=10 "
            "trajectory readout."
        )
    if int(args.bootstrap_replicates) < 1:
        raise ValueError("--bootstrap-replicates must be positive.")
    cv_root = Path(args.cv_root).expanduser().resolve()
    npz_path = Path(args.npz_path).expanduser().resolve()
    encoder_root = Path(args.encoder_root).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    if not cv_root.is_dir() or not npz_path.is_file():
        raise FileNotFoundError(f"Missing cv-root or NPZ: {cv_root}, {npz_path}")
    if not args.dry_run:
        from experiments.motion_primitive.run_peak_valley_hierarchy import (
            _load_numeric_grid,
        )

        # Validate shape and every within-trial stride before any missing
        # encoder is trained.  The explicit request must agree with the NPZ;
        # filename conventions are never used as a silent fallback.
        _load_numeric_grid(
            npz_path,
            expected_window_size_samples=requested_window_grid[0],
            expected_stride_samples=requested_window_grid[1],
        )
    npz_sha256 = sha256_file(npz_path)
    encoder_root.mkdir(parents=True, exist_ok=True)
    output_root.mkdir(parents=True, exist_ok=True)
    source_fingerprints = _source_fingerprints()

    checkpoints: dict[tuple[str, int, int], Path] = {}
    for profile in profiles:
        for fold in folds:
            for seed in seeds:
                checkpoint = _resolve_checkpoint(
                    encoder_root=encoder_root,
                    cv_root=cv_root,
                    npz_path=npz_path,
                    profile=profile,
                    fold=fold,
                    seed=seed,
                    python=str(args.python),
                    device=str(args.device),
                    train_missing=bool(args.train_missing_encoders),
                    dry_run=bool(args.dry_run),
                )
                checkpoints[(profile, fold, seed)] = checkpoint
                if not args.dry_run:
                    _validate_profile_semantics(
                        checkpoint,
                        profile,
                        fold,
                        seed,
                        npz_sha256,
                        requested_window_grid[0],
                    )

    encoder_grid_audits: dict[str, dict] = {}
    if not args.dry_run:
        for profile in profiles:
            encoder_grid_audits[profile] = validate_motion_encoder_grid_identity(
                {
                    (fold, seed): checkpoints[(profile, fold, seed)]
                    for fold in folds
                    for seed in seeds
                },
                profile,
            )

    protocol_identity = {
        "schema": SCHEMA,
        "cv_root": str(cv_root),
        "npz_path": str(npz_path),
        "npz_sha256": npz_sha256,
        "encoder_root": str(encoder_root),
        "profiles": list(profiles),
        "folds": list(folds),
        "seeds": list(seeds),
        "arms": list(arms),
        "source_fingerprints": source_fingerprints,
        "protocol": _grid_protocol_settings(args),
        "orchestrator_wiring_sha256": _runner_cli_wiring_sha256(),
        "encoder_training_identities": {
            profile: audit["training_identity_sha256"]
            for profile, audit in encoder_grid_audits.items()
        },
    }
    protocol_identity["identity_sha256"] = _identity_hash(protocol_identity)
    manifest_path = output_root / "experiment_manifest.json"
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if _canonical_json(existing) != _canonical_json(protocol_identity):
            raise RuntimeError(
                f"Output root records a different grid identity: {output_root}. "
                "Use a new --output-root."
            )
    elif not args.dry_run:
        _write_json(manifest_path, protocol_identity)

    results: list[dict] = []
    run_directories: list[Path] = []
    for profile in profiles:
        for fold in folds:
            for seed in seeds:
                checkpoint = checkpoints[(profile, fold, seed)]
                directory = (
                    output_root
                    / "runs"
                    / profile
                    / f"fold_{fold:02d}_seed_{seed}_peak_valley_hierarchy_v1"
                )
                if args.dry_run:
                    staging_directory = _unique_staging_output_path(directory)
                    command = _runner_command(
                        args,
                        checkpoint,
                        staging_directory,
                        profile,
                        fold,
                        seed,
                        arms,
                    )
                    print(
                        f"[run staging] {staging_directory} -> {directory}\n"
                        "[run command] " + subprocess.list2cmdline(command),
                        flush=True,
                    )
                    continue
                request_identity = _run_identity(
                    args=args,
                    profile=profile,
                    fold=fold,
                    seed=seed,
                    checkpoint=checkpoint,
                    arms=arms,
                    source_fingerprints=_member_source_fingerprints(
                        source_fingerprints
                    ),
                )
                result = _run_or_resume_member(
                    args=args,
                    checkpoint=checkpoint,
                    final_directory=directory,
                    profile=profile,
                    fold=fold,
                    seed=seed,
                    arms=arms,
                    request_identity=request_identity,
                )
                results.append(result)
                run_directories.append(directory)

    if args.dry_run:
        return {
            "schema": SCHEMA,
            "dry_run": True,
            "run_count": len(profiles) * len(folds) * len(seeds),
            "output_root": str(output_root),
        }
    expected_members = len(profiles) * len(folds) * len(seeds)
    observed_members = {
        (str(result["request_identity"]["profile"]), int(result["request_identity"]["fold"]), int(result["request_identity"]["seed"]))
        for result in results
    }
    if len(results) != expected_members or len(observed_members) != expected_members:
        raise RuntimeError("Completed peak/valley grid is incomplete or duplicated.")
    rows = [
        row
        for result in results
        for row in _metric_rows(
            result,
            requested_arms=arms,
            matched_random_controls=bool(args.matched_random_controls),
        )
    ]
    _validate_metric_grid(
        rows,
        profiles=profiles,
        folds=folds,
        seeds=seeds,
        requested_arms=arms,
        matched_random_controls=bool(args.matched_random_controls),
    )
    cross_subject_rows = [
        row
        for result in results
        for row in _cross_subject_diagnostic_rows(
            result,
            requested_arms=arms,
            matched_random_controls=bool(args.matched_random_controls),
        )
    ]
    _validate_cross_subject_diagnostic_grid(
        cross_subject_rows,
        profiles=profiles,
        folds=folds,
        seeds=seeds,
        requested_arms=arms,
        matched_random_controls=bool(args.matched_random_controls),
    )
    fold_rows, aggregate = _aggregate_rows(
        rows, int(args.bootstrap_replicates), int(args.analysis_seed)
    )
    cross_subject_fold_rows, cross_subject_aggregate = (
        _aggregate_cross_subject_diagnostic_rows(
            cross_subject_rows,
            int(args.bootstrap_replicates),
            int(args.analysis_seed) + 200_000,
        )
    )
    paired_rows, skipped_comparisons = _paired_hscore_rows(
        fold_rows,
        expected_folds=folds,
        bootstrap_replicates=int(args.bootstrap_replicates),
        seed=int(args.analysis_seed) + 100_000,
    )
    aggregate_dir = output_root / "aggregate"
    aggregate_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(aggregate_dir / "run_metrics.csv", rows)
    _write_csv(aggregate_dir / "fold_metrics_after_seed_average.csv", fold_rows)
    _write_csv(aggregate_dir / "aggregate_metrics_across_folds.csv", aggregate)
    _write_csv(
        aggregate_dir / "cross_subject_diagnostics_by_run.csv",
        cross_subject_rows,
    )
    _write_csv(
        aggregate_dir / "cross_subject_diagnostics_by_fold_after_seed_average.csv",
        cross_subject_fold_rows,
    )
    _write_csv(
        aggregate_dir / "cross_subject_diagnostics_across_folds.csv",
        cross_subject_aggregate,
    )
    if paired_rows:
        _write_csv(aggregate_dir / "paired_hscore_comparisons.csv", paired_rows)
    metric_plot = _save_metric_plot(aggregate_dir / "aggregate_cgcd_metrics.png", aggregate)
    heatmap_plot = _save_mean_activity_heatmaps(
        aggregate_dir / "mean_activity_distance_heatmaps.png", run_directories
    )
    summary = {
        "schema": SCHEMA,
        "protocol_identity": protocol_identity,
        "run_count": len(results),
        "expected_run_count": expected_members,
        "profiles": list(profiles),
        "folds": list(folds),
        "seeds": list(seeds),
        "arms": list(arms),
        "readout_variants": list(READOUT_VARIANTS),
        "primary_readout_variant": PRIMARY_READOUT_VARIANT,
        "staging_publication": protocol_identity["protocol"]["staging_publication"],
        "aggregation": (
            "Seeds are averaged within each held-out-subject fold. Bootstrap "
            "intervals resample the resulting seven fold means and are descriptive "
            "only, not confirmatory confidence statements."
        ),
        "paired_inference": (
            "Two-sided exact sign-flip tests use held-out-subject fold H-score "
            "differences after averaging seeds within each fold. The single "
            "prespecified follow-up primary contrast is tested without multiplicity "
            "adjustment; the fixed eight secondary contrasts receive Holm "
            "adjustment; the fixed-window replication and factorial interactions "
            "are exploratory and unadjusted."
        ),
        "cross_subject_diagnostic_aggregation": {
            "analysis_role": "post_truth_join_descriptive_diagnostic_only",
            "metrics": list(CROSS_SUBJECT_DIAGNOSTIC_METRICS),
            "favourable_directions": dict(CROSS_SUBJECT_DIAGNOSTIC_DIRECTIONS),
            "aggregation_order": (
                "mean_seeds_within_held_out_subject_fold_then_summarize_folds"
            ),
            "run_row_count": len(cross_subject_rows),
            "fold_row_count": len(cross_subject_fold_rows),
            "across_fold_metric_row_count": len(cross_subject_aggregate),
            "fold_count": len(folds),
            "seed_runs_are_not_independent_inference_units": True,
            "bootstrap_ci_interpretation": "descriptive_only_not_confirmatory",
            "not_part_of_primary_classification_test": True,
        },
        "prespecified_followup_primary_comparison": (
            PRESPECIFIED_FOLLOWUP_PRIMARY_COMPARISON[0]
        ),
        "exploratory_unadjusted_simple_comparisons": [
            item[0] for item in EXPLORATORY_SIMPLE_COMPARISONS
        ],
        "exploratory_factorial_interactions": [
            item[0] for item in FACTORIAL_INTERACTIONS
        ],
        "secondary_holm_family": {
            "name": HOLM_SECONDARY_FAMILY_NAME,
            "planned_size": CANONICAL_HOLM_FAMILY_SIZE,
            "observed_size": sum(
                row.get("multiplicity_family") == HOLM_SECONDARY_FAMILY_NAME
                for row in paired_rows
            ),
        },
        "statistical_resolution_audit": {
            "held_out_subject_fold_count": len(folds),
            "canonical_seven_fold_two_sided_exact_sign_flip_minimum_p": (
                CANONICAL_TWO_SIDED_SIGN_FLIP_MIN_P
            ),
            "canonical_eight_item_holm_minimum_attainable_adjusted_p": (
                CANONICAL_HOLM_MIN_ADJUSTED_P
            ),
            "holm_eight_can_reject_at_alpha_0_05_with_seven_folds": False,
            "bootstrap_ci_interpretation": "descriptive_only_not_confirmatory",
            "overlapping_training_folds_are_not_independent_experiments": True,
        },
        "paired_comparison_count": len(paired_rows),
        "skipped_paired_comparisons": skipped_comparisons,
        "metric_plot_saved": metric_plot,
        "mean_activity_heatmap_saved": heatmap_plot,
        "generated_files": [
            "run_metrics.csv",
            "fold_metrics_after_seed_average.csv",
            "aggregate_metrics_across_folds.csv",
            "cross_subject_diagnostics_by_run.csv",
            "cross_subject_diagnostics_by_fold_after_seed_average.csv",
            "cross_subject_diagnostics_across_folds.csv",
            *( ["paired_hscore_comparisons.csv"] if paired_rows else [] ),
            *( ["aggregate_cgcd_metrics.png"] if metric_plot else [] ),
            *( ["mean_activity_distance_heatmaps.png"] if heatmap_plot else [] ),
        ],
    }
    _write_json(aggregate_dir / "summary.json", summary)
    return summary


def main() -> None:
    result = run(build_parser().parse_args())
    print(json.dumps(_jsonable(result), ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
