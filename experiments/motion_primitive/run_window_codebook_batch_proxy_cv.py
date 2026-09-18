"""Run the paired window-size/codebook-size batch trajectory proxy.

This launcher is intentionally separate from the registered three-session
CGCD route.  For every requested fold/seed it:

1. prepares the matching USC-HAD window grid when it is absent;
2. trains the fold-matched ResNet1D window warm-up and A2 encoder;
3. fits the motion-primitive codebook on Offline old-class trajectories; and
4. delegates one label-free, transductive 12-cluster readout over all 120
   held-out-subject trajectories to :mod:`window_codebook_batch_proxy`.

The default three arms are a targeted set rather than a Cartesian grid.  They
form two pre-registered contrasts: K64 versus K128 on the shared W128 grid,
and W64 versus W128 at fixed K128.  The latter still co-varies stride,
separately trained encoders, and physical A2 context duration.  Global
Hungarian alignment is scorer-only and is an upper-bound clustering
diagnostic, not a deployable Online CGCD classifier.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from experiments.motion_primitive import pretrain_window_encoder, train_motion_encoder
from experiments.motion_primitive.frozen_e0 import (
    LEGACY_DESCRIPTOR_PROFILE,
    descriptor_profile_spec,
)
from experiments.motion_primitive.strict_artifacts import write_csv, write_json
from experiments.motion_primitive.strict_cv_common import (
    bootstrap_fold_mean,
    canonical_hash,
    member_directory,
    parse_integer_grid,
    validate_or_create_grid_manifest,
)
from experiments.motion_primitive.strict_protocol import sha256_file


SCHEMA = "hhr_window_codebook_batch_proxy_cv_v5"
WINDOW_SCHEMA = "hhr_har_window_pretrain_v2"
A2_SCHEMA = train_motion_encoder.COMPLETE_SCHEMA
MEMBER_SCHEMA = "hhr_window_codebook_batch_proxy_v3"
DEFAULT_ARMS = "64:32:128,128:64:64,128:64:128"
W64_K128_CONFIG_ID = "w64_s32_k128"
W128_K64_CONFIG_ID = "w128_s64_k64"
W128_K128_CONFIG_ID = "w128_s64_k128"
REQUIRED_CONFIG_IDS = frozenset(
    {
        W64_K128_CONFIG_ID,
        W128_K64_CONFIG_ID,
        W128_K128_CONFIG_ID,
    }
)
DEFAULT_FOLDS = (1, 2, 3)
DEFAULT_SEEDS = (0, 5)
EXPECTED_MEMBER_COUNT = 18
EXPECTED_OUTER_TRIAL_COUNT = 120
CHANNEL_NAMES = (
    "acc_x",
    "acc_y",
    "acc_z",
    "gyro_x",
    "gyro_y",
    "gyro_z",
)
PERFORMANCE_METRICS = (
    "all_accuracy",
    "old_accuracy",
    "new_accuracy",
    "h_score",
    "macro_f1",
    "ari",
    "nmi",
)
DIAGNOSTIC_METRICS = (
    "used_primitive_k",
    "effective_primitive_k",
    "dead_primitive_fraction",
)


@dataclass(frozen=True, order=True)
class ExperimentArm:
    """One paired temporal-resolution and primitive-vocabulary setting."""

    window_size: int
    window_stride: int
    primitive_num: int

    @property
    def config_id(self) -> str:
        return (
            f"w{int(self.window_size)}_s{int(self.window_stride)}_"
            f"k{int(self.primitive_num)}"
        )

    def as_dict(self) -> dict[str, int | str]:
        return {
            "config_id": self.config_id,
            "window_size": int(self.window_size),
            "window_stride": int(self.window_stride),
            "primitive_num": int(self.primitive_num),
        }


@dataclass(frozen=True)
class PairedContrast:
    """One pre-registered same-fold/same-seed targeted comparison."""

    contrast_id: str
    minuend_config_id: str
    subtrahend_config_id: str
    controlled_dimension: str
    interpretation: str
    limitation: str

    def as_dict(self) -> dict[str, str]:
        return {
            "contrast_id": self.contrast_id,
            "minuend_config_id": self.minuend_config_id,
            "subtrahend_config_id": self.subtrahend_config_id,
            "delta_definition": (
                f"{self.minuend_config_id}_minus_{self.subtrahend_config_id}"
            ),
            "controlled_dimension": self.controlled_dimension,
            "interpretation": self.interpretation,
            "limitation": self.limitation,
        }


ISOLATED_PAIRED_CONTRASTS = (
    PairedContrast(
        contrast_id="w128_k128_minus_w128_k64",
        minuend_config_id=W128_K128_CONFIG_ID,
        subtrahend_config_id=W128_K64_CONFIG_ID,
        controlled_dimension="fixed_window_grid_w128_s64",
        interpretation=(
            "Primitive codebook capacity contrast at a fixed window grid; the "
            "fold/seed-matched ResNet1D and A2 checkpoint is shared."
        ),
        limitation=(
            "Changing K necessarily changes the trajectory descriptor dimension, "
            "so the delta is the end-to-end capacity/readout effect rather than a "
            "centre-count-only geometric effect."
        ),
    ),
    PairedContrast(
        contrast_id="w128_k128_minus_w64_k128",
        minuend_config_id=W128_K128_CONFIG_ID,
        subtrahend_config_id=W64_K128_CONFIG_ID,
        controlled_dimension="fixed_primitive_capacity_k128",
        interpretation=(
            "Window-grid contrast at fixed primitive codebook capacity K=128."
        ),
        limitation=(
            "This is not a pure window-length main effect: stride, separately "
            "trained encoders, and the physical duration represented by the fixed "
            "A2 cp-context-windows all co-vary with window size."
        ),
    ),
)


def parse_arms(value: str) -> tuple[ExperimentArm, ...]:
    """Parse ``window:stride:primitive`` triples while preserving arm order."""

    tokens = [item.strip() for item in str(value).split(",") if item.strip()]
    if not tokens:
        raise ValueError("At least one experiment arm is required.")
    arms: list[ExperimentArm] = []
    for token in tokens:
        pieces = [piece.strip() for piece in token.split(":")]
        if len(pieces) != 3:
            raise ValueError(
                f"Invalid arm {token!r}; expected window:stride:primitive."
            )
        try:
            window_size, window_stride, primitive_num = map(int, pieces)
        except ValueError as error:
            raise ValueError(f"Experiment arm contains a non-integer: {token!r}.") from error
        if window_size < 1 or window_stride < 1 or primitive_num < 2:
            raise ValueError(f"Experiment arm contains a non-positive size: {token!r}.")
        if 2 * window_stride != window_size:
            raise ValueError(
                f"Experiment arm {token!r} must use a 50% overlap "
                "(stride=window/2)."
            )
        arms.append(ExperimentArm(window_size, window_stride, primitive_num))
    if len({arm.config_id for arm in arms}) != len(arms):
        raise ValueError("Experiment arms contain a duplicate configuration.")
    return tuple(arms)


def validate_experiment_grid(
    arms: Sequence[ExperimentArm],
    folds: Sequence[int],
    seeds: Sequence[int],
) -> tuple[tuple[ExperimentArm, ...], tuple[int, ...], tuple[int, ...]]:
    """Validate the requested 3-arm x 3-fold x 2-seed experiment."""

    arm_grid = tuple(arms)
    fold_grid = tuple(sorted(int(value) for value in folds))
    seed_grid = tuple(sorted(int(value) for value in seeds))
    if len(arm_grid) != 3:
        raise ValueError("This targeted experiment requires exactly three arms.")
    if len({arm.config_id for arm in arm_grid}) != len(arm_grid):
        raise ValueError("Experiment arms contain duplicate identities.")
    if any(
        arm.window_size < 1
        or arm.window_stride < 1
        or arm.primitive_num < 2
        or 2 * arm.window_stride != arm.window_size
        for arm in arm_grid
    ):
        raise ValueError("Every arm must be positive and use exactly 50% overlap.")
    observed_config_ids = {arm.config_id for arm in arm_grid}
    if observed_config_ids != REQUIRED_CONFIG_IDS:
        raise ValueError(
            "This targeted experiment requires exactly the three registered arms; "
            f"expected={sorted(REQUIRED_CONFIG_IDS)}, "
            f"observed={sorted(observed_config_ids)}."
        )
    if len(fold_grid) != 3 or len(set(fold_grid)) != 3:
        raise ValueError("This experiment requires exactly three distinct folds.")
    if any(fold < 1 or fold > 7 for fold in fold_grid):
        raise ValueError("USC-HAD folds must lie in 1..7.")
    if len(seed_grid) != 2 or len(set(seed_grid)) != 2:
        raise ValueError("This experiment requires exactly two distinct seeds.")
    if any(seed < 0 for seed in seed_grid):
        raise ValueError("Seeds must be non-negative.")
    if len(arm_grid) * len(fold_grid) * len(seed_grid) != EXPECTED_MEMBER_COUNT:
        raise AssertionError("The validated experiment grid is not 18 members.")
    return arm_grid, fold_grid, seed_grid


def expected_member_keys(
    arms: Sequence[ExperimentArm],
    folds: Sequence[int],
    seeds: Sequence[int],
) -> set[tuple[str, int, int]]:
    """Return exact ``(config_id, fold, seed)`` keys for completeness checks."""

    return {
        (arm.config_id, int(fold), int(seed))
        for arm in arms
        for fold in folds
        for seed in seeds
    }


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"Cannot read JSON artifact {path}.") from error
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON artifact is not an object: {path}.")
    return value


def _quoted(command: Sequence[str]) -> str:
    return " ".join(f'"{item}"' for item in command)


def _run_command(command: Sequence[str], *, stage: str) -> None:
    print("[command] " + _quoted(command), flush=True)
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
            f"Window/codebook batch proxy stage {stage!r} failed with "
            f"exit code {error.returncode}."
        ) from error


def _implementation_hashes() -> dict[str, str]:
    relative = (
        "preprocess.py",
        "models/resnet1d.py",
        "models/window_pretrain.py",
        "experiments/motion_primitive/run_window_codebook_batch_proxy_cv.py",
        "experiments/motion_primitive/window_codebook_batch_proxy.py",
        "experiments/motion_primitive/pretrain_window_encoder.py",
        "experiments/motion_primitive/train_motion_encoder.py",
        "experiments/motion_primitive/motion_encoder.py",
        "experiments/motion_primitive/motion_checkpoint.py",
        "experiments/motion_primitive/frozen_e0.py",
        "experiments/motion_primitive/strict_artifacts.py",
        "experiments/motion_primitive/strict_protocol.py",
        "experiments/motion_primitive/strict_metrics.py",
    )
    missing = [name for name in relative if not (PROJECT_ROOT / name).is_file()]
    if missing:
        raise RuntimeError(f"Experiment implementation is incomplete: {missing}.")
    return {name: sha256_file(PROJECT_ROOT / name) for name in relative}


def _preprocessed_directory(processed_root: Path, arm: ExperimentArm) -> Path:
    return processed_root / (
        f"uschad_w{arm.window_size}_s{arm.window_stride}_train17stats"
    )


def _window_grid_id(arm: ExperimentArm) -> str:
    """Identity shared by arms that differ only in primitive vocabulary size."""

    return f"w{int(arm.window_size)}_s{int(arm.window_stride)}"


def _preprocess_command(dataset_root: Path, target: Path, arm: ExperimentArm) -> list[str]:
    return [
        sys.executable,
        str(PROJECT_ROOT / "preprocess.py"),
        "--root",
        str(dataset_root),
        "--out_dir",
        str(target),
        "--window_size",
        str(int(arm.window_size)),
        "--stride",
        str(int(arm.window_stride)),
        "--channels",
        *CHANNEL_NAMES,
        "--old_classes",
        *[str(value) for value in range(1, 7)],
        "--all_classes",
        *[str(value) for value in range(1, 13)],
        "--source_subjects",
        *[str(value) for value in range(1, 8)],
        "--target_subjects",
        *[str(value) for value in range(10, 15)],
    ]


def _validate_preprocessed_grid(directory: Path, arm: ExperimentArm) -> dict[str, Any]:
    """Fail closed on an existing preprocessing cache with another identity."""

    npz_path = directory / "uschad_windows.npz"
    meta_path = directory / "meta.json"
    if not npz_path.is_file() or not meta_path.is_file():
        raise RuntimeError(f"Preprocessed grid is incomplete: {directory}.")
    meta = _read_json(meta_path)
    expected_meta = {
        "dataset": "USC-HAD",
        "num_valid_trials": 840,
        "window_size": int(arm.window_size),
        "stride": int(arm.window_stride),
        "channels": list(CHANNEL_NAMES),
        "num_channels": 6,
        "old_classes": list(range(1, 7)),
        "all_classes": list(range(1, 13)),
        "source_subjects": list(range(1, 8)),
        "target_subjects": list(range(10, 15)),
    }
    mismatches = {
        key: {"expected": expected, "observed": meta.get(key)}
        for key, expected in expected_meta.items()
        if meta.get(key) != expected
    }
    if mismatches:
        raise RuntimeError(
            f"Preprocessed grid metadata differs for {arm.config_id}: {mismatches}."
        )
    required = {
        "windows",
        "labels",
        "labels_1based",
        "subject_ids",
        "trial_global_ids",
        "window_indices",
        "window_start_indices",
        "mean",
        "std",
    }
    with np.load(npz_path, allow_pickle=False) as archive:
        missing = required - set(archive.files)
        if missing:
            raise RuntimeError(f"Preprocessed NPZ lacks fields {sorted(missing)}.")
        windows = np.asarray(archive["windows"], dtype=np.float32)
        labels = np.asarray(archive["labels"], dtype=np.int64)
        labels_1based = np.asarray(archive["labels_1based"], dtype=np.int64)
        subjects = np.asarray(archive["subject_ids"], dtype=np.int64)
        trial_ids = np.asarray(archive["trial_global_ids"], dtype=np.int64)
        window_indices = np.asarray(archive["window_indices"], dtype=np.int64)
        starts = np.asarray(archive["window_start_indices"], dtype=np.int64)
        mean = np.asarray(archive["mean"], dtype=np.float32)
        std = np.asarray(archive["std"], dtype=np.float32)
    count = len(windows)
    if windows.shape != (count, 6, int(arm.window_size)):
        raise RuntimeError(
            f"Preprocessed window shape differs for {arm.config_id}: {windows.shape}."
        )
    vectors = (labels, labels_1based, subjects, trial_ids, window_indices, starts)
    if any(value.shape != (count,) for value in vectors):
        raise RuntimeError("A preprocessed row field differs from the window count.")
    if not np.array_equal(labels_1based, labels + 1):
        raise RuntimeError("Preprocessed zero/one-based labels disagree.")
    if set(np.unique(labels).tolist()) != set(range(12)):
        raise RuntimeError("Preprocessed activity labels are not exactly 0..11.")
    if set(np.unique(subjects).tolist()) != set(range(1, 15)):
        raise RuntimeError("Preprocessed subjects are not exactly 1..14.")
    unique_trials = np.unique(trial_ids)
    if len(unique_trials) != 840 or not np.array_equal(unique_trials, np.arange(840)):
        raise RuntimeError("Preprocessed trial identities are not exactly 0..839.")
    if mean.shape != (1, 6, 1) or std.shape != (1, 6, 1):
        raise RuntimeError("Preprocessed normalization arrays have unexpected shapes.")
    if not np.all(np.isfinite(windows)) or not np.all(np.isfinite(mean)) or np.any(std <= 0):
        raise RuntimeError("Preprocessed sensor grid contains invalid numeric values.")
    for trial_id in unique_trials.tolist():
        rows = np.flatnonzero(trial_ids == int(trial_id))
        order = np.argsort(starts[rows], kind="stable")
        rows = rows[order]
        local_starts = starts[rows]
        if local_starts[0] != 0:
            raise RuntimeError(f"Trial {trial_id} does not start at sample zero.")
        if len(rows) > 1 and np.any(np.diff(local_starts) != int(arm.window_stride)):
            raise RuntimeError(f"Trial {trial_id} violates the requested stride.")
        if not np.array_equal(window_indices[rows], np.arange(len(rows))):
            raise RuntimeError(f"Trial {trial_id} has an incomplete ordered window grid.")
    if int(meta.get("num_windows", -1)) != count:
        raise RuntimeError("Preprocessing metadata/window row count differs.")
    return {
        **arm.as_dict(),
        "directory": str(directory.resolve()),
        "npz_path": str(npz_path.resolve()),
        "npz_sha256": sha256_file(npz_path),
        "meta_sha256": sha256_file(meta_path),
        "trial_count": 840,
        "window_count": int(count),
    }


def _prepare_preprocessed_grids(
    args: argparse.Namespace,
    arms: Sequence[ExperimentArm],
) -> dict[str, dict[str, Any]]:
    dataset_root = Path(args.dataset_root).expanduser().resolve()
    processed_root = Path(args.processed_root).expanduser().resolve()
    if not dataset_root.is_dir():
        raise FileNotFoundError(dataset_root)
    result: dict[str, dict[str, Any]] = {}
    validated_by_window: dict[tuple[int, int], dict[str, Any]] = {}
    for arm in arms:
        window_key = (int(arm.window_size), int(arm.window_stride))
        if window_key in validated_by_window:
            result[arm.config_id] = {
                **validated_by_window[window_key],
                **arm.as_dict(),
            }
            continue
        target = _preprocessed_directory(processed_root, arm)
        npz_path = target / "uschad_windows.npz"
        meta_path = target / "meta.json"
        if not (npz_path.is_file() and meta_path.is_file()):
            if target.exists() and any(target.iterdir()):
                raise RuntimeError(
                    f"Non-empty preprocessing directory is incomplete: {target}."
                )
            if not bool(args.generate_missing_preprocessed):
                raise FileNotFoundError(
                    f"Missing preprocessing grid for {arm.config_id}: {target}."
                )
            _run_command(
                _preprocess_command(dataset_root, target, arm),
                stage=f"preprocess {arm.config_id}",
            )
        validated = _validate_preprocessed_grid(target, arm)
        validated_by_window[window_key] = validated
        result[arm.config_id] = validated
    return result


def _window_command(
    args: argparse.Namespace,
    arm: ExperimentArm,
    npz_path: Path,
    fold: int,
    seed: int,
    output: Path,
) -> list[str]:
    command = [
        sys.executable,
        str(PROJECT_ROOT / "experiments" / "motion_primitive" / "pretrain_window_encoder.py"),
        "--npz-path",
        str(npz_path),
        "--output-dir",
        str(output),
        "--fold",
        str(int(fold)),
        "--seed",
        str(int(seed)),
        "--window-size",
        str(int(arm.window_size)),
        "--window-stride",
        str(int(arm.window_stride)),
        "--epochs",
        str(int(args.window_epochs)),
        "--batch-size",
        str(int(args.window_batch_size)),
        "--eval-batch-size",
        str(int(args.window_eval_batch_size)),
        "--learning-rate",
        str(float(args.window_learning_rate)),
        "--weight-decay",
        str(float(args.window_weight_decay)),
        "--weak-scale-std",
        str(float(args.window_weak_scale_std)),
        "--strong-scale-std",
        str(float(args.window_strong_scale_std)),
        "--num-workers",
        str(int(args.num_workers)),
        "--smoke-max-windows",
        str(int(args.smoke_max_windows)),
        "--device",
        str(args.device),
        "--deterministic",
        "--selection-policy",
        str(args.window_selection_policy),
    ]
    if bool(args.resume):
        command.append("--resume")
    return command


def _a2_command(
    args: argparse.Namespace,
    npz_path: Path,
    source_checkpoint: Path,
    seed: int,
    output: Path,
) -> list[str]:
    command = [
        sys.executable,
        str(PROJECT_ROOT / "experiments" / "motion_primitive" / "train_motion_encoder.py"),
        "--source-checkpoint",
        str(source_checkpoint),
        "--npz-path",
        str(npz_path),
        "--output-dir",
        str(output),
        "--ablation-profile",
        "A2",
        "--window-aug-consistency",
        "none",
        "--window-aug-profile",
        "basic",
        "--window-aug-weight",
        "1.0",
        "--rotation-max-degrees",
        "0.0",
        "--cp-weight",
        "1.0",
        "--content-boundary-alignment-weight",
        "0.1",
        "--cp-anchor-source",
        "raw_frozen_consensus",
        "--cp-context-windows",
        str(int(args.cp_context_windows)),
        "--cp-low-quantile",
        "0.50",
        "--cp-high-quantile",
        "0.90",
        "--cp-raw-scales",
        "1,2,4",
        "--cp-raw-frequency-bins",
        "16",
        "--cp-rank-margin",
        "0.20",
        "--cp-equivariance-weight",
        "0.50",
        "--cp-equivariance-delta",
        "1.0",
        "--noncollapse-weight",
        "0.05",
        "--noncollapse-target-std",
        "1.0",
        "--noncollapse-variance-weight",
        "1.0",
        "--noncollapse-covariance-weight",
        "1.0",
        "--noncollapse-windows-per-trial",
        "4",
        "--prediction-weight",
        "0.5",
        "--prediction-mask-ratio",
        "0.20",
        "--prediction-loss",
        "cosine",
        "--trial-weight",
        "0.1",
        "--cross-subject-weight",
        "0.0",
        "--segmentation-dim",
        "0",
        "--content-dim",
        "256",
        "--content-residual",
        "--augmentation-dim",
        "128",
        "--projection-hidden-dim",
        "256",
        "--trial-hidden-dim",
        "128",
        "--trial-peak-quantile",
        "0.90",
        "--trial-dropout",
        "0.0",
        "--predictor-hidden-dim",
        "0",
        "--backbone-layers",
        "2,2,2",
        "--old-class-count",
        "6",
        "--epochs",
        str(int(args.a2_epochs)),
        "--trial-batch-size",
        str(int(args.a2_trial_batch_size)),
        "--source-encode-batch-size",
        str(int(args.a2_source_encode_batch_size)),
        "--learning-rate",
        str(float(args.a2_learning_rate)),
        "--minimum-learning-rate",
        str(float(args.a2_minimum_learning_rate)),
        "--weight-decay",
        str(float(args.a2_weight_decay)),
        "--gradient-clip-norm",
        "5.0",
        "--ema-momentum",
        "0.99",
        "--freeze-backbone-epochs",
        "0",
        "--backbone-bn-policy",
        "frozen",
        "--early-stopping-patience",
        "0",
        "--minimum-improvement",
        "0.0",
        "--selection-policy",
        "final_epoch",
        "--normalization-eps",
        "1e-6",
        "--known-anomaly-policy",
        "report",
        "--smoke-max-train-trials",
        str(int(args.smoke_max_train_trials)),
        "--smoke-max-val-trials",
        str(int(args.smoke_max_val_trials)),
        "--seed",
        str(int(seed)),
        "--device",
        str(args.device),
        "--deterministic",
    ]
    if bool(args.resume):
        command.append("--resume")
    return command


def _member_command(
    args: argparse.Namespace,
    arm: ExperimentArm,
    npz_path: Path,
    a2_checkpoint: Path,
    fold: int,
    seed: int,
    output: Path,
) -> list[str]:
    command = [
        sys.executable,
        str(PROJECT_ROOT / "experiments" / "motion_primitive" / "window_codebook_batch_proxy.py"),
        "--a2-checkpoint",
        str(a2_checkpoint),
        "--npz-path",
        str(npz_path),
        "--output-dir",
        str(output),
        "--fold",
        str(int(fold)),
        "--seed",
        str(int(seed)),
        "--window-size",
        str(int(arm.window_size)),
        "--window-stride",
        str(int(arm.window_stride)),
        "--primitive-num",
        str(int(arm.primitive_num)),
        "--pca-dim",
        str(int(args.pca_dim)),
        "--activity-cluster-count",
        str(int(args.activity_cluster_count)),
        "--descriptor-pca-dim",
        str(int(args.descriptor_pca_dim)),
        "--descriptor-profile",
        str(args.descriptor_profile),
        "--signed-vertical-distance-weight",
        str(float(args.signed_vertical_distance_weight)),
        "--subject-nuisance-max-rank",
        str(int(args.subject_nuisance_max_rank)),
        "--subject-nuisance-explained-variance",
        str(float(args.subject_nuisance_explained_variance)),
        "--kmeans-n-init",
        str(int(args.kmeans_n_init)),
        "--kmeans-max-iter",
        str(int(args.kmeans_max_iter)),
        "--encode-batch-size",
        str(int(args.encode_batch_size)),
        "--device",
        str(args.device),
    ]
    if bool(args.allow_smoke_a2):
        command.append("--allow-smoke-a2")
    if bool(args.resume):
        command.append("--resume")
    return command


def _validated_window_checkpoint(
    output: Path,
    *,
    arm: ExperimentArm,
    fold: int,
    seed: int,
    npz_sha256: str,
    selection_policy: str,
) -> Path:
    complete_path = output / "complete.json"
    if not complete_path.is_file():
        raise RuntimeError(f"Window warm-up did not complete: {output}.")
    complete = _read_json(complete_path)
    if complete.get("schema") != WINDOW_SCHEMA or complete.get("complete") is not True:
        raise RuntimeError(f"Window warm-up completion marker is invalid: {output}.")
    if (int(complete.get("fold", -1)), int(complete.get("seed", -1))) != (fold, seed):
        raise RuntimeError("Window warm-up fold/seed identity differs.")
    identity = complete.get("identity")
    if not isinstance(identity, Mapping):
        raise RuntimeError("Window warm-up lacks its immutable identity.")
    arguments = identity.get("arguments")
    if not isinstance(arguments, Mapping):
        raise RuntimeError("Window warm-up identity lacks arguments.")
    expected = {
        "window_size": int(arm.window_size),
        "window_stride": int(arm.window_stride),
        "fold": int(fold),
        "seed": int(seed),
    }
    for key, value in expected.items():
        if int(arguments.get(key, -1)) != value:
            raise RuntimeError(f"Window warm-up argument {key!r} differs.")
    if arguments.get("deterministic") is not True:
        raise RuntimeError("Window warm-up was not trained in deterministic mode.")
    if str(arguments.get("selection_policy", "")) != str(selection_policy):
        raise RuntimeError("Window warm-up checkpoint-selection policy differs.")
    if identity.get("npz_sha256") != npz_sha256:
        raise RuntimeError("Window warm-up is bound to another NPZ.")
    checkpoint = output / str(complete.get("checkpoint", "model_best.pt"))
    if not checkpoint.is_file() or sha256_file(checkpoint) != complete.get("checkpoint_sha256"):
        raise RuntimeError("Window warm-up checkpoint is absent or changed.")
    checkpoint_payload = pretrain_window_encoder._load_checkpoint(checkpoint)
    if checkpoint_payload.get("run_identity") != identity:
        raise RuntimeError("Window warm-up checkpoint records another identity.")
    observed_full, observed_backbone = (
        pretrain_window_encoder._validate_checkpoint_state_hashes(
            checkpoint_payload
        )
    )
    if complete.get("model_state_dict_sha256") != observed_full:
        raise RuntimeError("Window warm-up selected tensor-state SHA256 mismatch.")
    if complete.get("backbone_state_dict_sha256") != observed_backbone:
        raise RuntimeError("Window warm-up selected backbone SHA256 mismatch.")
    runtime = complete.get("determinism")
    if not isinstance(runtime, Mapping):
        raise RuntimeError("Window warm-up lacks deterministic runtime metadata.")
    pretrain_window_encoder._validate_determinism_record(
        runtime, require_python_hash_seed=True
    )
    if checkpoint_payload.get("determinism") != runtime:
        raise RuntimeError(
            "Window warm-up checkpoint/completion deterministic runtime differs."
        )
    return checkpoint


def _validated_a2_checkpoint(
    output: Path,
    *,
    fold: int,
    seed: int,
    npz_sha256: str,
    source_checkpoint_sha256: str,
) -> Path:
    complete_path = output / "complete.json"
    if not complete_path.is_file():
        raise RuntimeError(f"A2 encoder did not complete: {output}.")
    complete = _read_json(complete_path)
    expected = {
        "schema": A2_SCHEMA,
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
        raise RuntimeError(f"A2 completion identity differs: {observed} != {expected}.")
    checkpoint = output / "motion_encoder_final.pt"
    if not checkpoint.is_file() or sha256_file(checkpoint) != complete.get("final_checkpoint_sha256"):
        raise RuntimeError("A2 final checkpoint is absent or changed.")
    payload = train_motion_encoder._load_checkpoint(checkpoint)
    train_motion_encoder._validate_output_checkpoint(payload)
    runtime = complete.get("determinism")
    if not isinstance(runtime, Mapping):
        raise RuntimeError("A2 completion lacks deterministic runtime metadata.")
    train_motion_encoder._validate_determinism_record(
        dict(runtime), require_python_hash_seed=True
    )
    if payload.get("determinism") != runtime:
        raise RuntimeError("A2 checkpoint/completion deterministic runtime differs.")
    if payload.get("model_state_dict_sha256") != complete.get(
        "final_model_state_dict_sha256"
    ):
        raise RuntimeError("A2 final model tensor-state SHA256 mismatch.")
    if payload.get("ema_teacher_state_dict_sha256") != complete.get(
        "final_ema_teacher_state_dict_sha256"
    ):
        raise RuntimeError("A2 final EMA-teacher tensor-state SHA256 mismatch.")
    return checkpoint


def _validated_member_row(
    output: Path,
    *,
    arm: ExperimentArm,
    fold: int,
    seed: int,
    descriptor_profile: str = LEGACY_DESCRIPTOR_PROFILE,
) -> dict[str, Any]:
    path = output / "complete.json"
    if not path.is_file():
        raise RuntimeError(f"Batch proxy member did not complete: {output}.")
    complete = _read_json(path)
    exact = {
        "schema": MEMBER_SCHEMA,
        "fold": int(fold),
        "seed": int(seed),
        "window_size": int(arm.window_size),
        "window_stride": int(arm.window_stride),
        "primitive_num": int(arm.primitive_num),
        "descriptor_profile": str(descriptor_profile),
        "activity_cluster_count": 12,
        "outer_trial_count": EXPECTED_OUTER_TRIAL_COUNT,
        "complete": True,
    }
    for key, expected in exact.items():
        if complete.get(key) != expected:
            raise RuntimeError(
                f"Batch proxy member field {key!r} differs in {output}: "
                f"{complete.get(key)!r} != {expected!r}."
            )
    numeric: dict[str, float] = {}
    for metric in (*PERFORMANCE_METRICS, *DIAGNOSTIC_METRICS):
        value = float(complete.get(metric, float("nan")))
        if not math.isfinite(value):
            raise RuntimeError(f"Batch proxy metric {metric!r} is not finite in {output}.")
        numeric[metric] = value
    if not 0 <= numeric["used_primitive_k"] <= arm.primitive_num:
        raise RuntimeError("Used primitive count lies outside codebook capacity.")
    if not 0 <= numeric["effective_primitive_k"] <= arm.primitive_num + 1e-6:
        raise RuntimeError("Effective primitive count lies outside codebook capacity.")
    if not 0 <= numeric["dead_primitive_fraction"] <= 1:
        raise RuntimeError("Dead primitive fraction lies outside [0,1].")
    return {
        "config_id": arm.config_id,
        "window_size": int(arm.window_size),
        "window_stride": int(arm.window_stride),
        "primitive_num": int(arm.primitive_num),
        "fold": int(fold),
        "seed": int(seed),
        "profile": str(complete.get("profile", "")),
        "descriptor_profile": str(complete.get("descriptor_profile", "")),
        "activity_cluster_count": int(complete["activity_cluster_count"]),
        "outer_trial_count": int(complete["outer_trial_count"]),
        **numeric,
        "member_dir": str(output.resolve()),
        "member_complete_sha256": sha256_file(path),
    }


def _aggregate_configuration(
    rows: Sequence[Mapping[str, Any]],
    *,
    arm: ExperimentArm,
    folds: Sequence[int],
    seeds: Sequence[int],
    bootstrap_seed: int,
    bootstrap_replicates: int,
    arm_index: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    selected = [row for row in rows if row["config_id"] == arm.config_id]
    expected = {(int(fold), int(seed)) for fold in folds for seed in seeds}
    observed = [(int(row["fold"]), int(row["seed"])) for row in selected]
    if len(observed) != len(set(observed)) or set(observed) != expected:
        raise RuntimeError(f"Incomplete member grid for {arm.config_id}.")
    metrics: dict[str, Any] = {}
    flat_rows: list[dict[str, Any]] = []
    for metric_index, metric in enumerate((*PERFORMANCE_METRICS, *DIAGNOSTIC_METRICS)):
        fold_means: list[float] = []
        for fold in folds:
            values = [
                float(row[metric])
                for row in selected
                if int(row["fold"]) == int(fold)
            ]
            if len(values) != len(seeds):
                raise RuntimeError(
                    f"Fold {fold} in {arm.config_id} lacks two seed values for {metric}."
                )
            fold_means.append(float(np.mean(values)))
        estimate = bootstrap_fold_mean(
            fold_means,
            seed=int(bootstrap_seed) + 1000 * arm_index + metric_index,
            replicates=int(bootstrap_replicates),
        )
        metrics[metric] = {**estimate, "fold_means_after_averaging_seeds": fold_means}
        flat_rows.append(
            {
                **arm.as_dict(),
                "metric": metric,
                **estimate,
                "fold_means_after_averaging_seeds": json.dumps(fold_means),
            }
        )
    return {
        **arm.as_dict(),
        "member_count": len(expected),
        "aggregation_unit": "fold_after_averaging_two_seeds_within_fold",
        "metrics": metrics,
    }, flat_rows


def _isolated_paired_comparisons(
    rows: Sequence[Mapping[str, Any]],
    *,
    folds: Sequence[int],
    seeds: Sequence[int],
    bootstrap_seed: int,
    bootstrap_replicates: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Compute the two pre-registered contrasts without breaking pairing.

    A difference is first calculated for each same-fold/same-seed member pair.
    The two seed differences are then averaged inside each held-out-subject fold,
    and only those fold means enter the bootstrap.  This keeps the statistical
    unit aligned with the outer subject split.
    """

    indexed = {
        (str(row["config_id"]), int(row["fold"]), int(row["seed"])): row
        for row in rows
    }
    if len(indexed) != len(rows):
        raise RuntimeError("Cannot pair duplicate member rows.")

    results: dict[str, Any] = {}
    flat: list[dict[str, Any]] = []
    for contrast_index, contrast in enumerate(ISOLATED_PAIRED_CONTRASTS):
        metric_results: dict[str, Any] = {}
        for metric_index, metric in enumerate(PERFORMANCE_METRICS):
            seed_deltas_by_fold: dict[str, list[float]] = {}
            fold_deltas: list[float] = []
            for fold in folds:
                seed_deltas: list[float] = []
                for seed in seeds:
                    minuend_key = (
                        contrast.minuend_config_id,
                        int(fold),
                        int(seed),
                    )
                    subtrahend_key = (
                        contrast.subtrahend_config_id,
                        int(fold),
                        int(seed),
                    )
                    if minuend_key not in indexed or subtrahend_key not in indexed:
                        raise RuntimeError(
                            "Isolated paired comparison lacks "
                            f"{minuend_key} or {subtrahend_key}."
                        )
                    delta = float(indexed[minuend_key][metric]) - float(
                        indexed[subtrahend_key][metric]
                    )
                    if not math.isfinite(delta):
                        raise RuntimeError(
                            f"Non-finite paired delta for {contrast.contrast_id} "
                            f"metric={metric} fold={fold} seed={seed}."
                        )
                    seed_deltas.append(delta)
                seed_deltas_by_fold[str(int(fold))] = seed_deltas
                fold_deltas.append(float(np.mean(seed_deltas)))

            estimate = bootstrap_fold_mean(
                fold_deltas,
                seed=(
                    int(bootstrap_seed)
                    + 20_000
                    + 1_000 * contrast_index
                    + metric_index
                ),
                replicates=int(bootstrap_replicates),
            )
            payload = {
                **estimate,
                "same_fold_same_seed_deltas": seed_deltas_by_fold,
                "fold_mean_paired_deltas_after_averaging_seeds": fold_deltas,
            }
            metric_results[metric] = payload
            flat.append(
                {
                    **contrast.as_dict(),
                    "metric": metric,
                    **estimate,
                    "same_fold_same_seed_deltas": json.dumps(
                        seed_deltas_by_fold, sort_keys=True
                    ),
                    "fold_mean_paired_deltas_after_averaging_seeds": json.dumps(
                        fold_deltas
                    ),
                }
            )
        results[contrast.contrast_id] = {
            **contrast.as_dict(),
            "pairing_unit": (
                "same_fold_same_seed_difference_then_average_seeds_within_fold"
            ),
            "bootstrap_unit": "held_out_subject_fold_mean",
            "metrics": metric_results,
        }

    return {
        "contrast_count": len(ISOLATED_PAIRED_CONTRASTS),
        "performance_metrics": list(PERFORMANCE_METRICS),
        "pairing_unit": (
            "same_fold_same_seed_difference_then_average_seeds_within_fold"
        ),
        "bootstrap_unit": "held_out_subject_fold_mean",
        "contrasts": results,
    }, flat


def _write_reports(
    root: Path,
    rows: Sequence[Mapping[str, Any]],
    *,
    arms: Sequence[ExperimentArm],
    folds: Sequence[int],
    seeds: Sequence[int],
    grid_identity_sha256: str,
    bootstrap_seed: int,
    bootstrap_replicates: int,
) -> dict[str, Any]:
    expected = expected_member_keys(arms, folds, seeds)
    observed = [
        (str(row["config_id"]), int(row["fold"]), int(row["seed"]))
        for row in rows
    ]
    if len(rows) != EXPECTED_MEMBER_COUNT or len(observed) != len(set(observed)):
        raise RuntimeError("Batch proxy result rows are duplicate or not exactly 18.")
    if set(observed) != expected:
        raise RuntimeError(
            f"Batch proxy result grid differs: missing={sorted(expected-set(observed))}, "
            f"extra={sorted(set(observed)-expected)}."
        )
    ordered = sorted(
        (dict(row) for row in rows),
        key=lambda row: (
            [arm.config_id for arm in arms].index(str(row["config_id"])),
            int(row["fold"]),
            int(row["seed"]),
        ),
    )
    write_csv(root / "batch_proxy_runs.csv", ordered)
    configurations: dict[str, Any] = {}
    summary_rows: list[dict[str, Any]] = []
    for arm_index, arm in enumerate(arms):
        aggregate, flat = _aggregate_configuration(
            ordered,
            arm=arm,
            folds=folds,
            seeds=seeds,
            bootstrap_seed=int(bootstrap_seed),
            bootstrap_replicates=int(bootstrap_replicates),
            arm_index=arm_index,
        )
        configurations[arm.config_id] = aggregate
        summary_rows.extend(flat)
    isolated_paired, isolated_paired_rows = _isolated_paired_comparisons(
        ordered,
        folds=folds,
        seeds=seeds,
        bootstrap_seed=int(bootstrap_seed),
        bootstrap_replicates=int(bootstrap_replicates),
    )
    write_csv(root / "batch_proxy_summary.csv", summary_rows)
    write_csv(root / "isolated_paired_contrasts.csv", isolated_paired_rows)
    write_json(root / "isolated_paired_contrasts.json", isolated_paired)
    summary = {
        "schema": SCHEMA,
        "grid_identity_sha256": str(grid_identity_sha256),
        "experiment_role": "transductive_batch_GCD_trajectory_clusterability_proxy",
        "not_deployable_online_cgcd": True,
        "gate_used": False,
        "multi_session_updates_used": False,
        "activity_truth_used_by_learner": False,
        "global_hungarian_role": "post_hoc_scoring_only",
        "arms_are_targeted_not_cartesian": True,
        "pre_registered_isolated_contrasts": [
            contrast.as_dict() for contrast in ISOLATED_PAIRED_CONTRASTS
        ],
        "independent_global_window_or_codebook_effect_identifiable": False,
        "folds": list(folds),
        "seeds": list(seeds),
        "member_count": EXPECTED_MEMBER_COUNT,
        "outer_trial_count_per_member": EXPECTED_OUTER_TRIAL_COUNT,
        "activity_cluster_count": 12,
        "primary_metric": "h_score",
        "statistical_unit": "held_out_subject_fold_after_averaging_two_seeds",
        "configurations": configurations,
        "isolated_paired_contrasts": isolated_paired,
    }
    write_json(root / "batch_proxy_summary.json", summary)
    artifacts = {
        name: sha256_file(root / name)
        for name in (
            "grid_manifest.json",
            "batch_proxy_runs.csv",
            "batch_proxy_summary.csv",
            "batch_proxy_summary.json",
            "isolated_paired_contrasts.csv",
            "isolated_paired_contrasts.json",
        )
    }
    complete = {
        **summary,
        "artifact_sha256": artifacts,
        "complete": True,
    }
    write_json(root / "complete.json", complete)
    return complete


def _grid_identity(
    args: argparse.Namespace,
    *,
    arms: Sequence[ExperimentArm],
    folds: Sequence[int],
    seeds: Sequence[int],
    preprocessed: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    return {
        "schema": SCHEMA,
        "project_root": str(Path(args.project_root).expanduser().resolve()),
        "experiment_role": "transductive_batch_GCD_trajectory_clusterability_proxy",
        "protocol": {
            "offline_encoder_fit": "train_subjects_old6_with_validation_subjects_old6",
            "primitive_codebook_fit": "offline_train_subjects_old6_only",
            "trajectory_fit_and_assignment": "outer_test_subjects_all12_all120_unlabeled",
            "activity_truth_access": "post_hoc_global_hungarian_scoring_only",
            "gate_used": False,
            "multi_session_updates_used": False,
        },
        "arms": [arm.as_dict() for arm in arms],
        "arms_are_targeted_not_cartesian": True,
        "pre_registered_isolated_contrasts": [
            contrast.as_dict() for contrast in ISOLATED_PAIRED_CONTRASTS
        ],
        "folds": list(folds),
        "seeds": list(seeds),
        "expected_member_count": EXPECTED_MEMBER_COUNT,
        "preprocessed": [dict(preprocessed[arm.config_id]) for arm in arms],
        "encoder": {
            "route": "ResNet1D_window_warmup_then_A2_final_frozen",
            "sharing": "one_fold_seed_encoder_per_unique_window_grid",
            "window_epochs": int(args.window_epochs),
            "window_batch_size": int(args.window_batch_size),
            "window_eval_batch_size": int(args.window_eval_batch_size),
            "window_learning_rate": float(args.window_learning_rate),
            "window_weight_decay": float(args.window_weight_decay),
            "window_weak_scale_std": float(args.window_weak_scale_std),
            "window_strong_scale_std": float(args.window_strong_scale_std),
            "window_deterministic": True,
            "window_selection_policy": str(args.window_selection_policy),
            "a2_epochs": int(args.a2_epochs),
            "a2_trial_batch_size": int(args.a2_trial_batch_size),
            "a2_source_encode_batch_size": int(args.a2_source_encode_batch_size),
            "a2_learning_rate": float(args.a2_learning_rate),
            "a2_minimum_learning_rate": float(args.a2_minimum_learning_rate),
            "a2_weight_decay": float(args.a2_weight_decay),
            "cp_context_windows": int(args.cp_context_windows),
            "selection_policy": "final_epoch",
        },
        "proxy": {
            "descriptor_profile": str(args.descriptor_profile),
            "descriptor_profile_spec": {
                "signed_vertical": bool(
                    descriptor_profile_spec(args.descriptor_profile).signed_vertical
                ),
                "duration_invariant": bool(
                    descriptor_profile_spec(args.descriptor_profile).duration_invariant
                ),
                "subject_debias": bool(
                    descriptor_profile_spec(args.descriptor_profile).subject_debias
                ),
            },
            "primitive_pca_dim": int(args.pca_dim),
            "activity_cluster_count": int(args.activity_cluster_count),
            "descriptor_pca_dim": int(args.descriptor_pca_dim),
            "kmeans_n_init": int(args.kmeans_n_init),
            "kmeans_max_iter": int(args.kmeans_max_iter),
            "encode_batch_size": int(args.encode_batch_size),
            "signed_vertical_distance_weight": float(
                args.signed_vertical_distance_weight
            ),
            "subject_nuisance_max_rank": int(args.subject_nuisance_max_rank),
            "subject_nuisance_explained_variance": float(
                args.subject_nuisance_explained_variance
            ),
            "outer_trial_count": EXPECTED_OUTER_TRIAL_COUNT,
        },
        "runtime": {
            "device": str(args.device),
            "num_workers": int(args.num_workers),
            "allow_smoke_a2": bool(args.allow_smoke_a2),
            "smoke_max_windows": int(args.smoke_max_windows),
            "smoke_max_train_trials": int(args.smoke_max_train_trials),
            "smoke_max_val_trials": int(args.smoke_max_val_trials),
            "bootstrap_seed": int(args.bootstrap_seed),
            "bootstrap_replicates": int(args.bootstrap_replicates),
        },
        "implementation_sha256": _implementation_hashes(),
    }


def validate_args(
    args: argparse.Namespace,
) -> tuple[argparse.Namespace, tuple[ExperimentArm, ...], tuple[int, ...], tuple[int, ...]]:
    requested_project_root = Path(args.project_root).expanduser().resolve()
    if requested_project_root != PROJECT_ROOT.resolve():
        raise ValueError(
            f"--project-root must resolve to this checkout {PROJECT_ROOT.resolve()}, "
            f"got {requested_project_root}."
        )
    arms = parse_arms(args.arms)
    folds = parse_integer_grid(args.folds, minimum=1, maximum=7)
    seeds = parse_integer_grid(args.seeds, minimum=0)
    arms, folds, seeds = validate_experiment_grid(arms, folds, seeds)
    positive_integer_names = (
        "window_epochs",
        "window_batch_size",
        "window_eval_batch_size",
        "a2_epochs",
        "a2_trial_batch_size",
        "a2_source_encode_batch_size",
        "pca_dim",
        "activity_cluster_count",
        "descriptor_pca_dim",
        "kmeans_n_init",
        "kmeans_max_iter",
        "encode_batch_size",
        "subject_nuisance_max_rank",
        "bootstrap_replicates",
    )
    if any(int(getattr(args, name)) < 1 for name in positive_integer_names):
        raise ValueError("Epoch, batch, PCA, KMeans and bootstrap values must be positive.")
    if int(args.activity_cluster_count) != 12:
        raise ValueError("USC-HAD all-class batch proxy requires activity-cluster-count=12.")
    if int(args.pca_dim) > 256:
        raise ValueError("Primitive PCA dimension cannot exceed A2 content dimension 256.")
    if int(args.cp_context_windows) < 1:
        raise ValueError("cp-context-windows must be positive.")
    descriptor_profile_spec(args.descriptor_profile)
    if str(args.descriptor_profile) != LEGACY_DESCRIPTOR_PROFILE:
        raise ValueError(
            "The three-arm scale/capacity runner is locked to legacy_state_v1. "
            "Use run_trajectory_descriptor_ablation_cv.py for non-legacy descriptor "
            "profiles so every fold/seed reuses one exact frozen codebook."
        )
    if not 0.0 < float(args.signed_vertical_distance_weight) < 1.0:
        raise ValueError("signed-vertical-distance-weight must lie in (0,1).")
    if not 0.0 < float(args.subject_nuisance_explained_variance) <= 1.0:
        raise ValueError("subject-nuisance-explained-variance must lie in (0,1].")
    for name in (
        "window_learning_rate",
        "a2_learning_rate",
        "a2_minimum_learning_rate",
    ):
        if float(getattr(args, name)) <= 0:
            raise ValueError(f"{name} must be positive.")
    for name in ("window_weight_decay", "a2_weight_decay"):
        if float(getattr(args, name)) < 0:
            raise ValueError(f"{name} must be non-negative.")
    for name in ("smoke_max_windows", "smoke_max_train_trials", "smoke_max_val_trials"):
        if int(getattr(args, name)) < 0:
            raise ValueError(f"{name} must be non-negative.")
    smoke = any(
        int(getattr(args, name)) > 0
        for name in ("smoke_max_windows", "smoke_max_train_trials", "smoke_max_val_trials")
    )
    if smoke and not bool(args.allow_smoke_a2):
        raise ValueError("Smoke limits require --allow-smoke-a2 for the downstream proxy.")
    return args, arms, folds, seeds


def run(args: argparse.Namespace) -> dict[str, Any]:
    args, arms, folds, seeds = validate_args(args)
    output_root = Path(args.output_root).expanduser().resolve()
    processed_root = Path(args.processed_root).expanduser().resolve()

    if bool(args.dry_run):
        commands: list[list[str]] = []
        dataset_root = Path(args.dataset_root).expanduser().resolve()
        planned_preprocessing: set[tuple[int, int]] = set()
        planned_encoders: set[tuple[int, int, int, int]] = set()
        for arm in arms:
            target = _preprocessed_directory(processed_root, arm)
            window_key = (int(arm.window_size), int(arm.window_stride))
            if (
                window_key not in planned_preprocessing
                and not (
                    (target / "uschad_windows.npz").is_file()
                    and (target / "meta.json").is_file()
                )
            ):
                commands.append(_preprocess_command(dataset_root, target, arm))
                planned_preprocessing.add(window_key)
            npz_path = target / "uschad_windows.npz"
            for fold in folds:
                for seed in seeds:
                    encoder_key = (
                        int(arm.window_size),
                        int(arm.window_stride),
                        int(fold),
                        int(seed),
                    )
                    window_dir = member_directory(
                        output_root
                        / "encoders"
                        / _window_grid_id(arm)
                        / "window_pretrain",
                        fold,
                        seed,
                    )
                    a2_dir = member_directory(
                        output_root / "encoders" / _window_grid_id(arm) / "a2",
                        fold,
                        seed,
                    )
                    proxy_dir = member_directory(
                        output_root / "members" / arm.config_id, fold, seed
                    )
                    if encoder_key not in planned_encoders:
                        commands.append(
                            _window_command(args, arm, npz_path, fold, seed, window_dir)
                        )
                        commands.append(
                            _a2_command(
                                args,
                                npz_path,
                                window_dir / (
                                    "model_best.pt"
                                    if str(args.window_selection_policy)
                                    == "best_val_macro_f1"
                                    else "model_last.pt"
                                ),
                                seed,
                                a2_dir,
                            )
                        )
                        planned_encoders.add(encoder_key)
                    commands.append(
                        _member_command(
                            args,
                            arm,
                            npz_path,
                            a2_dir / "motion_encoder_final.pt",
                            fold,
                            seed,
                            proxy_dir,
                        )
                    )
        return {
            "schema": SCHEMA,
            "dry_run": True,
            "arms": [arm.as_dict() for arm in arms],
            "member_count": EXPECTED_MEMBER_COUNT,
            "commands": commands,
        }

    preprocessed = _prepare_preprocessed_grids(args, arms)
    identity = _grid_identity(
        args,
        arms=arms,
        folds=folds,
        seeds=seeds,
        preprocessed=preprocessed,
    )
    validate_or_create_grid_manifest(output_root, identity)
    grid_sha = canonical_hash(identity)
    complete_path = output_root / "complete.json"
    if complete_path.is_file() and not bool(args.resume):
        raise FileExistsError(
            f"Completed experiment exists: {output_root}; pass --resume to verify/reuse."
        )

    rows: list[dict[str, Any]] = []
    encoder_checkpoints: dict[tuple[int, int, int, int], Path] = {}
    for arm in arms:
        npz_path = Path(preprocessed[arm.config_id]["npz_path"])
        npz_sha = str(preprocessed[arm.config_id]["npz_sha256"])
        for fold in folds:
            for seed in seeds:
                encoder_key = (
                    int(arm.window_size),
                    int(arm.window_stride),
                    int(fold),
                    int(seed),
                )
                if encoder_key not in encoder_checkpoints:
                    window_dir = member_directory(
                        output_root
                        / "encoders"
                        / _window_grid_id(arm)
                        / "window_pretrain",
                        fold,
                        seed,
                    )
                    window_command = _window_command(
                        args, arm, npz_path, fold, seed, window_dir
                    )
                    _run_command(
                        window_command,
                        stage=(
                            f"window warm-up {_window_grid_id(arm)} "
                            f"fold={fold} seed={seed}"
                        ),
                    )
                    source_checkpoint = _validated_window_checkpoint(
                        window_dir,
                        arm=arm,
                        fold=int(fold),
                        seed=int(seed),
                        npz_sha256=npz_sha,
                        selection_policy=str(args.window_selection_policy),
                    )

                    a2_dir = member_directory(
                        output_root / "encoders" / _window_grid_id(arm) / "a2",
                        fold,
                        seed,
                    )
                    a2_command = _a2_command(
                        args, npz_path, source_checkpoint, seed, a2_dir
                    )
                    _run_command(
                        a2_command,
                        stage=f"A2 {_window_grid_id(arm)} fold={fold} seed={seed}",
                    )
                    encoder_checkpoints[encoder_key] = _validated_a2_checkpoint(
                        a2_dir,
                        fold=int(fold),
                        seed=int(seed),
                        npz_sha256=npz_sha,
                        source_checkpoint_sha256=sha256_file(source_checkpoint),
                    )
                a2_checkpoint = encoder_checkpoints[encoder_key]

                member_dir = member_directory(
                    output_root / "members" / arm.config_id, fold, seed
                )
                command = _member_command(
                    args,
                    arm,
                    npz_path,
                    a2_checkpoint,
                    fold,
                    seed,
                    member_dir,
                )
                _run_command(
                    command,
                    stage=f"batch proxy {arm.config_id} fold={fold} seed={seed}",
                )
                rows.append(
                    _validated_member_row(
                        member_dir,
                        arm=arm,
                        fold=int(fold),
                        seed=int(seed),
                        descriptor_profile=str(args.descriptor_profile),
                    )
                )
                write_csv(output_root / "batch_proxy_runs.partial.csv", rows)

    return _write_reports(
        output_root,
        rows,
        arms=arms,
        folds=folds,
        seeds=seeds,
        grid_identity_sha256=grid_sha,
        bootstrap_seed=int(args.bootstrap_seed),
        bootstrap_replicates=int(args.bootstrap_replicates),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Three targeted window/codebook arms over 3 folds x 2 seeds, followed "
            "by one no-gate/no-session all-trajectory batch clustering proxy."
        )
    )
    parser.add_argument(
        "--project-root",
        default=str(PROJECT_ROOT),
        help="Compatibility/audit path; must resolve to this HHR checkout.",
    )
    parser.add_argument(
        "--dataset-root",
        default=str(PROJECT_ROOT.parent / "DataSet" / "USC-HAD"),
    )
    parser.add_argument("--processed-root", default=str(PROJECT_ROOT / "processed"))
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--arms", default=DEFAULT_ARMS)
    parser.add_argument("--folds", default=",".join(map(str, DEFAULT_FOLDS)))
    parser.add_argument("--seeds", default=",".join(map(str, DEFAULT_SEEDS)))
    parser.add_argument(
        "--generate-missing-preprocessed",
        action=argparse.BooleanOptionalAction,
        default=True,
    )

    # Keep training batches identical to the previous registered experiment;
    # larger no-gradient batches improve throughput without changing the
    # optimisation protocol across the three targeted arms.
    parser.add_argument("--window-epochs", type=int, default=60)
    parser.add_argument("--window-batch-size", type=int, default=256)
    parser.add_argument("--window-eval-batch-size", type=int, default=1024)
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
    parser.add_argument("--a2-source-encode-batch-size", type=int, default=1024)
    parser.add_argument("--a2-learning-rate", type=float, default=1e-4)
    parser.add_argument("--a2-minimum-learning-rate", type=float, default=1e-6)
    parser.add_argument("--a2-weight-decay", type=float, default=1e-4)
    parser.add_argument(
        "--cp-context-windows",
        type=int,
        default=2,
        help=(
            "Held at two windows to reproduce A2 configuration; physical context "
            "therefore co-varies with window scale and is recorded as a limitation."
        ),
    )
    parser.add_argument("--pca-dim", type=int, default=64)
    parser.add_argument("--activity-cluster-count", type=int, default=12)
    parser.add_argument("--descriptor-pca-dim", type=int, default=32)
    parser.add_argument(
        "--descriptor-profile",
        default=LEGACY_DESCRIPTOR_PROFILE,
        choices=(LEGACY_DESCRIPTOR_PROFILE,),
    )
    parser.add_argument("--signed-vertical-distance-weight", type=float, default=0.15)
    parser.add_argument("--subject-nuisance-max-rank", type=int, default=4)
    parser.add_argument(
        "--subject-nuisance-explained-variance", type=float, default=0.90
    )
    parser.add_argument("--kmeans-n-init", type=int, default=50)
    parser.add_argument("--kmeans-max-iter", type=int, default=300)
    parser.add_argument("--encode-batch-size", type=int, default=1024)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--bootstrap-seed", type=int, default=20260914)
    parser.add_argument("--bootstrap-replicates", type=int, default=10_000)
    parser.add_argument("--smoke-max-windows", type=int, default=0)
    parser.add_argument("--smoke-max-train-trials", type=int, default=0)
    parser.add_argument("--smoke-max-val-trials", type=int, default=0)
    parser.add_argument("--allow-smoke-a2", action="store_true")
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
    "DEFAULT_ARMS",
    "DEFAULT_FOLDS",
    "DEFAULT_SEEDS",
    "EXPECTED_MEMBER_COUNT",
    "ExperimentArm",
    "ISOLATED_PAIRED_CONTRASTS",
    "PairedContrast",
    "REQUIRED_CONFIG_IDS",
    "SCHEMA",
    "W64_K128_CONFIG_ID",
    "W128_K64_CONFIG_ID",
    "W128_K128_CONFIG_ID",
    "build_parser",
    "expected_member_keys",
    "main",
    "parse_arms",
    "run",
    "validate_args",
    "validate_experiment_grid",
]
