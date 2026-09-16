"""Resume-safe A0/A3 multi-fold adaptive-codebook experiment pipeline.

The wrapper deliberately orchestrates existing single-purpose entry points rather
than duplicating their scientific logic:

1. train one formal motion encoder for every profile/fold/seed;
2. build the matched fixed-window K=32 primitive run;
3. run the frozen/adaptive K=32-or-34 Session-2 experiment; and
4. execute the strict paired A0/A3 analysis.

Completed artifacts are skipped only after their recorded identities and content
hashes have been checked.  A directory without its stage completion marker is
moved to a recoverable ``_incomplete`` quarantine before retrying.  A directory
with a completion marker that fails validation is never modified automatically.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence
from uuid import uuid4


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.motion_primitive.run_subject_cv import (  # noqa: E402
    find_checkpoint,
    motion_encoder_grid_identity,
    sha256_file,
    validate_existing_run,
    validate_motion_encoder_grid_identity,
)


PROFILES = ("A0", "A3")
CANONICAL_FOLDS = tuple(range(1, 8))
CANONICAL_SEEDS = (0, 5, 50, 500)
SCHEMA = "adaptive_codebook_cv_pipeline_v1"

TRAINING_PROTOCOL = {
    "window_aug_weight": 1.0,
    "content_boundary_alignment_weight": {"A0": 0.0, "A3": 0.1},
    "trial_weight": 0.1,
    "segmentation_dim": 0,
    "epochs": 30,
    "trial_batch_size": 8,
    "source_encode_batch_size": 512,
    "learning_rate": 1e-4,
    "minimum_learning_rate": 1e-6,
    "weight_decay": 1e-4,
    "gradient_clip_norm": 5.0,
    "ema_momentum": 0.99,
    "early_stopping_patience": 0,
    "selection_policy": "final_epoch",
    "deterministic": True,
    "old_class_count": 6,
    "known_anomaly_policy": "report",
    "smoke_max_train_trials": 0,
    "smoke_max_val_trials": 0,
}

FIXED_PROTOCOL = {
    "primitive_num": 32,
    "primitive_segmentation": "fixed_window",
    "pca_dim": 64,
    "embedding_normalization": "l2",
    "codebook_weighting": "per_trial",
    "kmeans_n_init": 20,
    "kmeans_max_iter": 300,
    "old_class_count": 6,
    "anomaly_policy": "report",
    "sample_rate_hz": 100.0,
    "edge_trim_ratio": 0.10,
    "label_permutations": 1000,
    "order_shuffles": 50,
    "batch_size": 512,
    "changepoint_context_windows": 2,
    "changepoint_score_quantile": 0.90,
    "changepoint_min_segment_windows": 2,
}

ONLINE_PROTOCOL = {
    "minimum_token_fraction": 0.50,
    "registered_minimum_support": 10,
    "minimum_parent_trials": 6,
    "minimum_parent_subjects": 2,
    "minimum_child_trials": 3,
    "minimum_child_fraction": 0.15,
    "minimum_child_subjects": 2,
    "minimum_silhouette": 0.25,
    "minimum_loo_distortion_reduction": 0.15,
    "minimum_loo_stability": 0.80,
    "minimum_loso_stability": 0.80,
    "maximum_subject_nmi": 0.25,
    "confidence_radius_quantile": 0.95,
    "static_radius_quantile": 0.95,
    "minimum_motion_energy_ratio": 2.0,
    "minimum_motion_energy_gap": 0.25,
    "minimum_gate_loo_enabled_fraction": 1.0,
    "minimum_gate_loo_agreement": 0.90,
}

ENCODER_COMPLETION_FILES = (
    "motion_encoder_final.pt",
    "motion_encoder_final.pt.sha256",
    "motion_encoder_best.pt",
    "motion_encoder_best.pt.sha256",
    "resolved_config.json",
    "split_audit.json",
    "pseudo_boundary_calibration.json",
    "history.json",
)

FIXED_COMPLETION_FILES = (
    "summary.json",
    "experiment_config.json",
    "split_audit.json",
    "window_embeddings_and_tokens.npz",
    "segment_embeddings_and_tokens.npz",
    "primitive_codebook.npz",
    "trial_primitive_sequences.jsonl",
    "activity_sequence_distance_matrix.csv",
)

ONLINE_RESULT_NAME = "online_adaptive_codebook_v2_results.json"
ONLINE_COMPLETION_FILES = (
    ONLINE_RESULT_NAME,
    "session2_predictions.csv",
    "session2_adaptive_trajectories.csv",
    "session2_adaptive_trajectories.png",
    "session2_global_confusions.png",
    "session2_activity_distance_heatmaps.png",
    "trial_bootstrap_confidence_intervals.csv",
)

ANALYSIS_RESULT_NAME = "adaptive_codebook_v2_analysis.json"


@dataclass(frozen=True)
class GridMember:
    profile: str
    fold: int
    seed: int

    @property
    def key(self) -> tuple[str, int, int]:
        return self.profile, self.fold, self.seed


def parse_int_list(value: str) -> tuple[int, ...]:
    values = tuple(sorted({int(token.strip()) for token in value.split(",") if token.strip()}))
    if not values:
        raise argparse.ArgumentTypeError("at least one integer is required")
    return values


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"Cannot read JSON artifact {path}: {error}") from error
    if not isinstance(value, dict):
        raise RuntimeError(f"Expected a JSON object: {path}")
    return value


def _json_normalize(value: Any) -> Any:
    """Normalize tuple/list representation before comparing JSON sidecars."""

    return json.loads(json.dumps(value, ensure_ascii=False))


def _same_number(actual: Any, expected: Any) -> bool:
    if isinstance(expected, bool):
        return actual is expected
    if isinstance(expected, (int, float)) and not isinstance(expected, bool):
        try:
            return float(actual) == float(expected)
        except (TypeError, ValueError):
            return False
    return actual == expected


def _require_mapping(value: Any, context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise RuntimeError(f"{context} must be a mapping.")
    return value


def _assert_fields(
    actual: Mapping[str, Any], expected: Mapping[str, Any], context: str
) -> None:
    mismatches = {
        key: {"expected": value, "actual": actual.get(key, "<missing>")}
        for key, value in expected.items()
        if key not in actual or not _same_number(actual.get(key), value)
    }
    if mismatches:
        raise RuntimeError(f"{context} identity mismatch: {mismatches}")


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def quarantine_incomplete(path: Path, root: Path, dry_run: bool) -> Path | None:
    """Move one incomplete run aside without deleting it."""

    path = path.resolve()
    root = root.resolve()
    if not path.exists():
        return None
    if not _is_relative_to(path, root) or path == root:
        raise RuntimeError(f"Refusing to quarantine path outside its stage root: {path}")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    destination = root / "_incomplete" / f"{path.name}.{stamp}.{uuid4().hex[:8]}"
    print(f"[quarantine] {path} -> {destination}", flush=True)
    if not dry_run:
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(path), str(destination))
    return destination


def _prepare_directory(
    path: Path,
    root: Path,
    completion_marker: str,
    dry_run: bool,
) -> bool:
    """Return True when a completed-looking directory must be validated."""

    if not path.exists():
        return False
    if not path.is_dir():
        raise RuntimeError(f"Expected a run directory, found a non-directory: {path}")
    if (path / completion_marker).is_file():
        return True
    quarantine_incomplete(path, root, dry_run=dry_run)
    return False


def _validate_or_refresh_incompatible_online(
    path: Path,
    root: Path,
    validator: Callable[[], None],
    *,
    refresh_incompatible: bool,
    dry_run: bool,
) -> bool:
    """Validate a completed online run or recoverably retire an old version.

    Normal resume remains fail-closed.  The explicit refresh switch is intended
    for a scientifically material runner fix: an output produced by the old
    implementation is moved aside, never deleted or silently accepted into the
    new aggregate.
    """

    try:
        validator()
    except (FileNotFoundError, RuntimeError, ValueError) as error:
        if not refresh_incompatible:
            raise
        print(
            f"[refresh incompatible online] {path}: "
            f"{type(error).__name__}: {error}",
            flush=True,
        )
        quarantine_incomplete(path, root, dry_run=dry_run)
        return False
    return True


def _load_torch(path: Path) -> dict[str, Any]:
    import torch

    try:
        value = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        value = torch.load(path, map_location="cpu")
    if not isinstance(value, dict):
        raise RuntimeError(f"Checkpoint is not a dictionary: {path}")
    return value


def _source_metadata(path: Path, fold: int) -> dict[str, Any]:
    payload = _load_torch(path)
    metadata = payload.get("experiment_metadata")
    if not isinstance(metadata, dict):
        raise RuntimeError(f"Source checkpoint lacks experiment_metadata: {path}")
    expected = {
        "uschad_cv_fold": int(fold),
        "uschad_sample_unit": "window",
        "uschad_split_mode": "subject",
        "uschad_recompute_norm_from_train_subjects": True,
        "uschad_window_size": 256,
        "har_in_channels": 6,
    }
    _assert_fields(metadata, expected, f"source checkpoint {path}")
    train_subjects = {int(value) for value in metadata.get("uschad_train_subjects", [])}
    val_subjects = {int(value) for value in metadata.get("offline_val_subjects", [])}
    test_subjects = {int(value) for value in metadata.get("uschad_test_subjects", [])}
    if not train_subjects or len(val_subjects) != 2 or len(test_subjects) != 2:
        raise RuntimeError(f"Invalid source subject split in {path}.")
    if train_subjects & val_subjects or train_subjects & test_subjects or val_subjects & test_subjects:
        raise RuntimeError(f"Overlapping source subject split in {path}.")
    return metadata


def discover_sources(
    cv_root: Path, folds: Sequence[int], seeds: Sequence[int]
) -> dict[tuple[int, int], Path]:
    sources: dict[tuple[int, int], Path] = {}
    for fold in folds:
        for seed in seeds:
            path = find_checkpoint(cv_root, int(fold), int(seed))
            _source_metadata(path, int(fold))
            sources[(int(fold), int(seed))] = path
    return sources


def _validate_sha_sidecar(path: Path) -> None:
    sidecar = path.with_suffix(path.suffix + ".sha256")
    if not sidecar.is_file():
        raise RuntimeError(f"Checkpoint hash sidecar is missing: {sidecar}")
    recorded = sidecar.read_text(encoding="ascii").strip().split()[0].lower()
    observed = sha256_file(path)
    if recorded != observed:
        raise RuntimeError(f"Checkpoint hash mismatch for {path}: {recorded} != {observed}")


def validate_encoder_run(
    directory: Path,
    source_checkpoint: Path,
    npz_path: Path,
    member: GridMember,
) -> Path:
    from experiments.motion_primitive.motion_checkpoint import (
        validate_motion_encoder_checkpoint_integrity,
    )
    from experiments.motion_primitive.train_motion_encoder import (
        _implementation_fingerprint,
    )

    missing = [name for name in ENCODER_COMPLETION_FILES if not (directory / name).is_file()]
    if missing:
        raise RuntimeError(f"Encoder run {directory} is incomplete; missing {missing}.")
    final_path = directory / "motion_encoder_final.pt"
    best_path = directory / "motion_encoder_best.pt"
    _validate_sha_sidecar(final_path)
    _validate_sha_sidecar(best_path)
    payload = _load_torch(final_path)
    validate_motion_encoder_checkpoint_integrity(payload)

    metadata = _require_mapping(payload.get("experiment_metadata"), "encoder metadata")
    training = _require_mapping(payload.get("resolved_training_config"), "encoder training")
    arguments = _require_mapping(payload.get("command_arguments"), "encoder arguments")
    selection = _require_mapping(payload.get("selection"), "encoder selection")
    source = _require_mapping(payload.get("source_checkpoint"), "encoder source")
    split = _require_mapping(payload.get("split_audit"), "encoder split")
    expected_arguments = {
        "ablation_profile": member.profile,
        "window_aug_weight": TRAINING_PROTOCOL["window_aug_weight"],
        "content_boundary_alignment_weight": TRAINING_PROTOCOL[
            "content_boundary_alignment_weight"
        ][member.profile],
        "trial_weight": TRAINING_PROTOCOL["trial_weight"],
        "segmentation_dim": TRAINING_PROTOCOL["segmentation_dim"],
        "epochs": TRAINING_PROTOCOL["epochs"],
        "trial_batch_size": TRAINING_PROTOCOL["trial_batch_size"],
        "source_encode_batch_size": TRAINING_PROTOCOL["source_encode_batch_size"],
        "learning_rate": TRAINING_PROTOCOL["learning_rate"],
        "minimum_learning_rate": TRAINING_PROTOCOL["minimum_learning_rate"],
        "weight_decay": TRAINING_PROTOCOL["weight_decay"],
        "gradient_clip_norm": TRAINING_PROTOCOL["gradient_clip_norm"],
        "ema_momentum": TRAINING_PROTOCOL["ema_momentum"],
        "early_stopping_patience": 0,
        "selection_policy": "final_epoch",
        "deterministic": True,
        "old_class_count": 6,
        "known_anomaly_policy": "report",
        "smoke_max_train_trials": 0,
        "smoke_max_val_trials": 0,
        "seed": member.seed,
    }
    _assert_fields(arguments, expected_arguments, f"encoder arguments {directory}")
    _assert_fields(
        metadata,
        {
            "uschad_cv_fold": member.fold,
            "motion_encoder_seed": member.seed,
            "old_class_count": 6,
            "smoke_test": False,
            "outer_test_used_during_encoder_training": False,
        },
        f"encoder metadata {directory}",
    )
    _assert_fields(training, {"ablation_profile": member.profile}, f"encoder training {directory}")
    _assert_fields(
        selection,
        {
            "policy": "final_epoch",
            "file_role": "canonical_final",
            "completed_epochs": TRAINING_PROTOCOL["epochs"],
            "selected_epoch_1based": TRAINING_PROTOCOL["epochs"],
            "outer_test_queries": 0,
        },
        f"encoder selection {directory}",
    )
    _assert_fields(
        split,
        {
            "outer_test_sensor_windows_selected": 0,
            "outer_test_model_forward_calls": 0,
            "smoke_test": False,
        },
        f"encoder split {directory}",
    )
    if str(source.get("sha256", "")) != sha256_file(source_checkpoint):
        raise RuntimeError(f"Encoder {directory} was trained from a different source checkpoint.")
    npz_sha = sha256_file(npz_path)
    if str(payload.get("npz_sha256", "")) != npz_sha:
        raise RuntimeError(f"Encoder {directory} was trained from a different NPZ.")
    if payload.get("implementation_fingerprint") != _implementation_fingerprint():
        raise RuntimeError(
            f"Encoder {directory} implementation fingerprint differs from current source."
        )
    if _read_json(directory / "resolved_config.json") != _json_normalize(training):
        raise RuntimeError(f"Encoder resolved_config sidecar differs from checkpoint: {directory}")
    if _read_json(directory / "split_audit.json") != _json_normalize(split):
        raise RuntimeError(f"Encoder split_audit sidecar differs from checkpoint: {directory}")
    motion_encoder_grid_identity(payload)
    return final_path.resolve()


def build_encoder_command(
    python: str,
    source_checkpoint: Path,
    npz_path: Path,
    output_dir: Path,
    member: GridMember,
    device: str,
) -> list[str]:
    alignment = TRAINING_PROTOCOL["content_boundary_alignment_weight"][member.profile]
    return [
        python,
        str(PROJECT_ROOT / "experiments/motion_primitive/train_motion_encoder.py"),
        "--source-checkpoint", str(source_checkpoint),
        "--npz-path", str(npz_path),
        "--output-dir", str(output_dir),
        "--ablation-profile", member.profile,
        "--window-aug-weight", "1",
        "--content-boundary-alignment-weight", str(alignment),
        "--trial-weight", "0.1",
        "--segmentation-dim", "0",
        "--epochs", "30",
        "--trial-batch-size", "8",
        "--source-encode-batch-size", "512",
        "--learning-rate", "0.0001",
        "--minimum-learning-rate", "0.000001",
        "--weight-decay", "0.0001",
        "--gradient-clip-norm", "5",
        "--ema-momentum", "0.99",
        "--early-stopping-patience", "0",
        "--selection-policy", "final_epoch",
        "--known-anomaly-policy", "report",
        "--deterministic",
        "--device", device,
        "--seed", str(member.seed),
    ]


def _fixed_namespace(profile: str, device: str) -> argparse.Namespace:
    return argparse.Namespace(
        motion_encoder_root="enabled",
        expected_encoder_profile=profile,
        primitive_num=32,
        pca_dim=64,
        label_permutations=1000,
        order_shuffles=50,
        batch_size=512,
        device=device,
        anomaly_policy="report",
        primitive_segmentation="fixed_window",
        ssl_feature_dim=64,
        ssl_epochs=25,
        ssl_learning_rate=1e-3,
        ssl_mask_ratio=0.15,
        ssl_noise_std=0.02,
        changepoint_context_windows=2,
        changepoint_score_quantile=0.90,
        changepoint_min_segment_windows=2,
    )


def fixed_run_name(member: GridMember) -> str:
    return (
        f"fold_{member.fold:02d}_seed_{member.seed}_k32_"
        f"motion_encoder_fixed_window_{member.profile.lower()}"
    )


def validate_fixed_run(
    directory: Path,
    checkpoint: Path,
    npz_path: Path,
    member: GridMember,
    device: str,
) -> None:
    from experiments.motion_primitive.run_online_hierarchical_gate import (
        _validate_registered_upstream,
    )
    from experiments.motion_primitive.run_online_secondary_codebook import (
        load_segment_artifacts,
    )

    missing = [name for name in FIXED_COMPLETION_FILES if not (directory / name).is_file()]
    if missing:
        raise RuntimeError(f"Fixed-window run {directory} is incomplete; missing {missing}.")
    validate_existing_run(
        directory,
        checkpoint,
        member.fold,
        member.seed,
        _fixed_namespace(member.profile, device),
    )
    config = _read_json(directory / "experiment_config.json")
    split = _read_json(directory / "split_audit.json")
    _assert_fields(
        _require_mapping(config.get("checkpoint_metadata"), "fixed checkpoint metadata"),
        {
            "uschad_cv_fold": member.fold,
            "motion_encoder_seed": member.seed,
            "smoke_test": False,
            "outer_test_used_during_encoder_training": False,
        },
        f"fixed metadata {directory}",
    )
    if str(config.get("npz_sha256", "")) != sha256_file(npz_path):
        raise RuntimeError(f"Fixed run {directory} uses a different NPZ.")
    if str(config.get("checkpoint_sha256", "")) != sha256_file(checkpoint):
        raise RuntimeError(f"Fixed run {directory} uses different encoder bytes.")
    fit = sorted(int(value) for value in split.get("fit_subjects", []))
    evaluation = sorted(int(value) for value in split.get("eval_subjects", []))
    metadata = _require_mapping(config.get("checkpoint_metadata"), "fixed metadata")
    if fit != sorted(int(value) for value in metadata.get("uschad_train_subjects", [])):
        raise RuntimeError(f"Fixed run {directory} fit subjects differ from the encoder.")
    if evaluation != sorted(int(value) for value in metadata.get("uschad_test_subjects", [])):
        raise RuntimeError(f"Fixed run {directory} eval subjects differ from the encoder.")
    artifacts = load_segment_artifacts(directory, directory / "primitive_codebook.npz")
    _validate_registered_upstream(config, artifacts)


def build_fixed_command(
    python: str,
    checkpoint: Path,
    npz_path: Path,
    output_dir: Path,
    member: GridMember,
    device: str,
) -> list[str]:
    return [
        python,
        str(PROJECT_ROOT / "experiments/motion_primitive/run_experiment.py"),
        "--checkpoint", str(checkpoint),
        "--npz-path", str(npz_path),
        "--output-dir", str(output_dir),
        "--primitive-num", "32",
        "--primitive-segmentation", "fixed_window",
        "--pca-dim", "64",
        "--embedding-normalization", "l2",
        "--codebook-weighting", "per_trial",
        "--kmeans-n-init", "20",
        "--kmeans-max-iter", "300",
        "--old-class-count", "6",
        "--anomaly-policy", "report",
        "--sample-rate-hz", "100",
        "--edge-trim-ratio", "0.10",
        "--label-permutations", "1000",
        "--order-shuffles", "50",
        "--batch-size", "512",
        "--changepoint-context-windows", "2",
        "--changepoint-score-quantile", "0.90",
        "--changepoint-min-segment-windows", "2",
        "--device", device,
        "--seed", str(member.seed),
    ]


def build_fixed_aggregate_command(
    python: str,
    cv_root: Path,
    encoder_root: Path,
    output_root: Path,
    profile: str,
    folds: Sequence[int],
    seeds: Sequence[int],
    device: str,
) -> list[str]:
    return [
        python,
        str(PROJECT_ROOT / "experiments/motion_primitive/run_subject_cv.py"),
        "--cv-root", str(cv_root),
        "--motion-encoder-root", str(encoder_root),
        "--expected-encoder-profile", profile,
        "--output-root", str(output_root),
        "--folds", ",".join(str(value) for value in folds),
        "--seeds", ",".join(str(value) for value in seeds),
        "--primitive-num", "32",
        "--primitive-segmentation", "fixed_window",
        "--pca-dim", "64",
        "--label-permutations", "1000",
        "--order-shuffles", "50",
        "--batch-size", "512",
        "--device", device,
        "--anomaly-policy", "report",
        "--skip-existing",
    ]


def online_run_name(member: GridMember) -> str:
    return f"fold_{member.fold:02d}_seed_{member.seed}_{member.profile}_fixed_adaptive_v2"


def _current_online_fingerprint() -> dict[str, str]:
    paths = {
        "runner": PROJECT_ROOT / "experiments/motion_primitive/run_online_hierarchical_gate_v2.py",
        "adaptive_codebook": PROJECT_ROOT / "experiments/motion_primitive/adaptive_codebook.py",
        "frozen_readout": PROJECT_ROOT / "experiments/motion_primitive/frozen_hierarchical_readout.py",
        "hierarchical_gate_v2": PROJECT_ROOT / "experiments/motion_primitive/hierarchical_gate_v2.py",
        "historical_hierarchical_gate_v1": PROJECT_ROOT / "experiments/motion_primitive/hierarchical_gate.py",
    }
    return {name: sha256_file(path) for name, path in paths.items()}


def validate_online_run(
    directory: Path,
    fixed_dir: Path,
    npz_path: Path,
    member: GridMember,
    bootstrap_resamples: int,
) -> None:
    from experiments.motion_primitive.analyze_online_hierarchical_gate_v2 import (
        load_audited_run,
    )

    missing = [name for name in ONLINE_COMPLETION_FILES if not (directory / name).is_file()]
    if missing:
        raise RuntimeError(f"Online run {directory} is incomplete; missing {missing}.")
    result_path = directory / ONLINE_RESULT_NAME
    result = _read_json(result_path)
    arguments = _require_mapping(result.get("arguments"), "online arguments")
    expected_arguments = {
        "seed": member.seed,
        "bootstrap_resamples": int(bootstrap_resamples),
        **ONLINE_PROTOCOL,
    }
    _assert_fields(arguments, expected_arguments, f"online arguments {directory}")
    recorded_fold = arguments.get("fold")
    if recorded_fold is not None and int(recorded_fold) != member.fold:
        raise RuntimeError(f"Online run {directory} records fold={recorded_fold}, expected {member.fold}.")
    audit = _require_mapping(result.get("input_audit"), "online input audit")
    _assert_fields(
        audit,
        {
            "primitive_segmentation": "fixed_window",
            "encoder_ablation_profile": member.profile,
            "run_config_sha256": sha256_file(fixed_dir / "experiment_config.json"),
            "segment_artifact_sha256": sha256_file(fixed_dir / "segment_embeddings_and_tokens.npz"),
            "codebook_artifact_sha256": sha256_file(fixed_dir / "primitive_codebook.npz"),
            "npz_sha256": sha256_file(npz_path),
            "base_centers_unchanged": True,
        },
        f"online input audit {directory}",
    )
    manifest = _require_mapping(result.get("session_manifest_audit"), "session manifest")
    _assert_fields(manifest, {"seed": member.seed}, f"session manifest {directory}")
    evaluated_session = manifest.get("evaluated_session")
    if evaluated_session is not None and int(evaluated_session) != 2:
        raise RuntimeError(f"Online run {directory} is not a Session-2 result.")
    implementation = _require_mapping(audit.get("implementation_fingerprint"), "online fingerprint")
    current = _current_online_fingerprint()
    observed = {
        name: _require_mapping(implementation.get(name), f"online fingerprint {name}").get("sha256")
        for name in current
    }
    if observed != current:
        raise RuntimeError(
            f"Online run {directory} implementation differs from current source: "
            f"observed={observed}, current={current}."
        )
    generated = result.get("generated_files", [])
    if not isinstance(generated, list) or any(not (directory / str(name)).is_file() for name in generated):
        raise RuntimeError(f"Online run {directory} has missing generated files.")
    audited = load_audited_run(result_path, old_class_count=6)
    if (audited.fold, audited.seed, audited.profile, audited.segmentation, audited.session) != (
        member.fold,
        member.seed,
        member.profile,
        "fixed_window",
        2,
    ):
        raise RuntimeError(f"Online result identity mismatch in {directory}.")


def build_online_command(
    python: str,
    fixed_dir: Path,
    npz_path: Path,
    output_dir: Path,
    member: GridMember,
    bootstrap_resamples: int,
) -> list[str]:
    command = [
        python,
        str(PROJECT_ROOT / "experiments/motion_primitive/run_online_hierarchical_gate_v2.py"),
        "--run-dir", str(fixed_dir),
        "--npz-path", str(npz_path),
        "--output-dir", str(output_dir),
        "--fold", str(member.fold),
        "--seed", str(member.seed),
        "--bootstrap-resamples", str(int(bootstrap_resamples)),
    ]
    option_names = {
        "minimum_token_fraction": "--minimum-token-fraction",
        "registered_minimum_support": "--registered-minimum-support",
        "minimum_parent_trials": "--minimum-parent-trials",
        "minimum_parent_subjects": "--minimum-parent-subjects",
        "minimum_child_trials": "--minimum-child-trials",
        "minimum_child_fraction": "--minimum-child-fraction",
        "minimum_child_subjects": "--minimum-child-subjects",
        "minimum_silhouette": "--minimum-silhouette",
        "minimum_loo_distortion_reduction": "--minimum-loo-distortion-reduction",
        "minimum_loo_stability": "--minimum-loo-stability",
        "minimum_loso_stability": "--minimum-loso-stability",
        "maximum_subject_nmi": "--maximum-subject-nmi",
        "confidence_radius_quantile": "--confidence-radius-quantile",
        "static_radius_quantile": "--static-radius-quantile",
        "minimum_motion_energy_ratio": "--minimum-motion-energy-ratio",
        "minimum_motion_energy_gap": "--minimum-motion-energy-gap",
        "minimum_gate_loo_enabled_fraction": "--minimum-gate-loo-enabled-fraction",
        "minimum_gate_loo_agreement": "--minimum-gate-loo-agreement",
    }
    for key, flag in option_names.items():
        command.extend((flag, str(ONLINE_PROTOCOL[key])))
    return command


def build_analysis_command(
    python: str,
    online_root: Path,
    output_dir: Path,
    folds: Sequence[int],
    seeds: Sequence[int],
    bootstrap_replicates: int,
    analysis_seed: int,
) -> list[str]:
    return [
        python,
        str(PROJECT_ROOT / "experiments/motion_primitive/analyze_online_hierarchical_gate_v2.py"),
        "--input-root", str(online_root),
        "--output-dir", str(output_dir),
        "--old-class-count", "6",
        "--bootstrap-replicates", str(int(bootstrap_replicates)),
        "--seed", str(int(analysis_seed)),
        "--expected-folds", ",".join(str(value) for value in folds),
        "--expected-seeds", ",".join(str(value) for value in seeds),
    ]


def validate_analysis(
    directory: Path,
    online_root: Path,
    folds: Sequence[int],
    seeds: Sequence[int],
    bootstrap_replicates: int,
    analysis_seed: int,
) -> None:
    from experiments.motion_primitive.analyze_online_hierarchical_gate_v2 import (
        discover_result_paths,
    )

    result_path = directory / ANALYSIS_RESULT_NAME
    if not result_path.is_file():
        raise RuntimeError(f"Analysis result is missing: {result_path}")
    result = _read_json(result_path)
    generated = result.get("generated_files")
    if not isinstance(generated, list) or not generated:
        raise RuntimeError(f"Analysis generated-file manifest is absent: {directory}")
    missing = [str(name) for name in generated if not (directory / str(name)).is_file()]
    if missing:
        raise RuntimeError(f"Analysis {directory} is incomplete; missing {missing}.")
    arguments = _require_mapping(result.get("arguments"), "analysis arguments")
    _assert_fields(
        arguments,
        {
            "old_class_count": 6,
            "bootstrap_replicates": int(bootstrap_replicates),
            "seed": int(analysis_seed),
        },
        f"analysis arguments {directory}",
    )
    if [int(value) for value in arguments.get("expected_folds", [])] != list(folds):
        raise RuntimeError(f"Analysis {directory} fold grid differs from this request.")
    if [int(value) for value in arguments.get("expected_seeds", [])] != list(seeds):
        raise RuntimeError(f"Analysis {directory} seed grid differs from this request.")
    current_inputs = discover_result_paths([online_root])
    recorded_inputs = result.get("input_results")
    if not isinstance(recorded_inputs, list):
        raise RuntimeError(f"Analysis {directory} lacks input_results.")
    expected_hashes = sorted(sha256_file(path) for path in current_inputs)
    observed_hashes = sorted(str(item.get("sha256", "")) for item in recorded_inputs if isinstance(item, dict))
    if observed_hashes != expected_hashes:
        raise RuntimeError(f"Analysis {directory} was computed from different online results.")
    fingerprint = _require_mapping(result.get("implementation_fingerprint"), "analysis fingerprint")
    analyzer_path = PROJECT_ROOT / "experiments/motion_primitive/analyze_online_hierarchical_gate_v2.py"
    if str(fingerprint.get("analyzer_sha256", "")) != sha256_file(analyzer_path):
        raise RuntimeError(f"Analysis {directory} uses a different analyzer implementation.")
    grid = _require_mapping(result.get("grid_audit"), "analysis grid audit")
    expected_pairs = len(folds) * len(seeds)
    if int(grid.get("configuration_count", -1)) != 2:
        raise RuntimeError(f"Analysis {directory} does not contain exactly A0 and A3.")
    configurations = _require_mapping(grid.get("configurations"), "analysis configurations")
    for profile in PROFILES:
        key = f"profile={profile}|segmentation=fixed_window|session=2"
        entry = _require_mapping(configurations.get(key), f"analysis configuration {key}")
        if int(entry.get("run_count", -1)) != expected_pairs:
            raise RuntimeError(f"Analysis {directory}/{key} has an incomplete run grid.")


def _run_command(
    command: Sequence[str], dry_run: bool, *, context: str = "pipeline subprocess"
) -> None:
    printable = " ".join(json.dumps(str(item)) for item in command)
    print(f"[command] {printable}", flush=True)
    if not dry_run:
        try:
            subprocess.run(list(command), cwd=PROJECT_ROOT, check=True)
        except subprocess.CalledProcessError as error:
            raise RuntimeError(
                f"{context} failed with exit code {error.returncode}; "
                "the child traceback is printed immediately above this message."
            ) from error


def _members(folds: Sequence[int], seeds: Sequence[int]) -> list[GridMember]:
    return [
        GridMember(profile, int(fold), int(seed))
        for profile in PROFILES
        for fold in folds
        for seed in seeds
    ]


def _encoder_directory(root: Path, member: GridMember) -> Path:
    return root / f"fold_{member.fold:02d}_seed_{member.seed}_{member.profile}_formal_v1"


def _pipeline_identity(
    args: argparse.Namespace,
    sources: Mapping[tuple[int, int], Path],
    folds: Sequence[int],
    seeds: Sequence[int],
) -> dict[str, Any]:
    return {
        "schema": SCHEMA,
        "cv_root": str(Path(args.cv_root).resolve()),
        "npz_path": str(Path(args.npz_path).resolve()),
        "npz_sha256": sha256_file(Path(args.npz_path).resolve()),
        "encoder_root": str(Path(args.encoder_root).resolve()),
        "folds": list(folds),
        "seeds": list(seeds),
        "profiles": list(PROFILES),
        "source_checkpoints": [
            {
                "fold": fold,
                "seed": seed,
                "path": str(path),
                "sha256": sha256_file(path),
            }
            for (fold, seed), path in sorted(sources.items())
        ],
        "training_protocol": TRAINING_PROTOCOL,
        "fixed_protocol": FIXED_PROTOCOL,
        "online_protocol": {
            **ONLINE_PROTOCOL,
            "bootstrap_resamples": int(args.bootstrap_resamples),
        },
        "analysis_protocol": {
            "bootstrap_replicates": int(args.analysis_bootstrap_replicates),
            "seed": int(args.analysis_seed),
            "old_class_count": 6,
        },
    }


def _prepare_manifest(work_root: Path, identity: Mapping[str, Any], dry_run: bool) -> None:
    path = work_root / "pipeline_manifest.json"
    if path.is_file():
        recorded = _read_json(path)
        if recorded.get("protocol_identity") != identity:
            raise RuntimeError(
                f"Work root {work_root} records a different pipeline identity. "
                "Use a new --work-root rather than mixing experiments."
            )
        return
    if work_root.exists() and any(work_root.iterdir()):
        allowed = {"_incomplete"}
        unexpected = [item.name for item in work_root.iterdir() if item.name not in allowed]
        if unexpected:
            raise RuntimeError(
                f"Non-empty work root has no pipeline manifest: {work_root}; "
                f"unexpected={unexpected}."
            )
    print(f"[manifest] create {path}", flush=True)
    if not dry_run:
        work_root.mkdir(parents=True, exist_ok=True)
        payload = {
            "protocol_identity": identity,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "wrapper_path": str(Path(__file__).resolve()),
            "wrapper_sha256": sha256_file(Path(__file__).resolve()),
        }
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def run_pipeline(args: argparse.Namespace) -> dict[str, Any]:
    folds = parse_int_list(args.folds)
    seeds = parse_int_list(args.seeds)
    if set(folds) - set(CANONICAL_FOLDS):
        raise ValueError(f"Unsupported fold IDs: {folds}")
    if set(seeds) - set(CANONICAL_SEEDS):
        raise ValueError(f"Unsupported seed IDs: {seeds}")
    if int(args.bootstrap_resamples) < 1 or int(args.analysis_bootstrap_replicates) < 1:
        raise ValueError("Bootstrap counts must be positive.")

    cv_root = Path(args.cv_root).expanduser().resolve()
    npz_path = Path(args.npz_path).expanduser().resolve()
    encoder_root = Path(args.encoder_root).expanduser().resolve()
    work_root = Path(args.work_root).expanduser().resolve()
    fixed_parent = work_root / "fixed_window"
    online_root = work_root / "online"
    analysis_dir = work_root / "paired_analysis"
    if not cv_root.is_dir() or not npz_path.is_file():
        raise FileNotFoundError(f"Missing cv-root or NPZ: {cv_root}, {npz_path}")

    sources = discover_sources(cv_root, folds, seeds)
    identity = _pipeline_identity(args, sources, folds, seeds)
    _prepare_manifest(work_root, identity, dry_run=bool(args.dry_run))
    members = _members(folds, seeds)
    counts = {"encoder_run": 0, "encoder_skip": 0, "fixed_run": 0, "fixed_skip": 0, "online_run": 0, "online_skip": 0}

    checkpoints: dict[tuple[str, int, int], Path] = {}
    for member in members:
        directory = _encoder_directory(encoder_root, member)
        completed = _prepare_directory(
            directory, encoder_root, "motion_encoder_final.pt.sha256", bool(args.dry_run)
        )
        if completed:
            checkpoint = validate_encoder_run(
                directory, sources[(member.fold, member.seed)], npz_path, member
            )
            print(f"[skip encoder] {member.key}: {checkpoint}", flush=True)
            counts["encoder_skip"] += 1
        else:
            command = build_encoder_command(
                args.python,
                sources[(member.fold, member.seed)],
                npz_path,
                directory,
                member,
                args.device,
            )
            _run_command(
                command,
                bool(args.dry_run),
                context=(
                    f"encoder stage profile={member.profile}, fold={member.fold}, "
                    f"seed={member.seed}"
                ),
            )
            counts["encoder_run"] += 1
            checkpoint = (
                directory / "motion_encoder_final.pt"
                if args.dry_run
                else validate_encoder_run(
                    directory, sources[(member.fold, member.seed)], npz_path, member
                )
            )
        checkpoints[member.key] = checkpoint

    if not args.dry_run:
        for profile in PROFILES:
            grid = {
                (member.fold, member.seed): checkpoints[member.key]
                for member in members
                if member.profile == profile
            }
            validate_motion_encoder_grid_identity(grid, profile)
    if args.stop_after == "encoders":
        return {"counts": counts, "work_root": str(work_root), "dry_run": bool(args.dry_run)}

    for member in members:
        profile_root = fixed_parent / member.profile
        directory = profile_root / fixed_run_name(member)
        completed = _prepare_directory(directory, profile_root, "summary.json", bool(args.dry_run))
        if completed:
            validate_fixed_run(
                directory, checkpoints[member.key], npz_path, member, args.device
            )
            print(f"[skip fixed] {member.key}: {directory}", flush=True)
            counts["fixed_skip"] += 1
        else:
            _run_command(
                build_fixed_command(
                    args.python,
                    checkpoints[member.key],
                    npz_path,
                    directory,
                    member,
                    args.device,
                ),
                bool(args.dry_run),
                context=(
                    f"fixed-window stage profile={member.profile}, "
                    f"fold={member.fold}, seed={member.seed}"
                ),
            )
            counts["fixed_run"] += 1
            if not args.dry_run:
                validate_fixed_run(
                    directory, checkpoints[member.key], npz_path, member, args.device
                )

    for profile in PROFILES:
        command = build_fixed_aggregate_command(
            args.python,
            cv_root,
            encoder_root,
            fixed_parent / profile,
            profile,
            folds,
            seeds,
            args.device,
        )
        _run_command(
            command,
            bool(args.dry_run),
            context=f"fixed-window aggregate profile={profile}",
        )
    if args.stop_after == "fixed":
        return {"counts": counts, "work_root": str(work_root), "dry_run": bool(args.dry_run)}

    for member in members:
        fixed_dir = fixed_parent / member.profile / fixed_run_name(member)
        directory = online_root / online_run_name(member)
        completed = _prepare_directory(directory, online_root, ONLINE_RESULT_NAME, bool(args.dry_run))
        if completed:
            completed = _validate_or_refresh_incompatible_online(
                directory,
                online_root,
                lambda: validate_online_run(
                    directory,
                    fixed_dir,
                    npz_path,
                    member,
                    int(args.bootstrap_resamples),
                ),
                refresh_incompatible=bool(args.refresh_incompatible_online),
                dry_run=bool(args.dry_run),
            )
        if completed:
            print(f"[skip online] {member.key}: {directory}", flush=True)
            counts["online_skip"] += 1
        else:
            _run_command(
                build_online_command(
                    args.python,
                    fixed_dir,
                    npz_path,
                    directory,
                    member,
                    int(args.bootstrap_resamples),
                ),
                bool(args.dry_run),
                context=(
                    f"online stage profile={member.profile}, fold={member.fold}, "
                    f"seed={member.seed}"
                ),
            )
            counts["online_run"] += 1
            if not args.dry_run:
                validate_online_run(
                    directory,
                    fixed_dir,
                    npz_path,
                    member,
                    int(args.bootstrap_resamples),
                )
    if args.stop_after == "online":
        return {"counts": counts, "work_root": str(work_root), "dry_run": bool(args.dry_run)}

    analysis_complete = _prepare_directory(
        analysis_dir, work_root, ANALYSIS_RESULT_NAME, bool(args.dry_run)
    )
    if analysis_complete:
        validate_analysis(
            analysis_dir,
            online_root,
            folds,
            seeds,
            int(args.analysis_bootstrap_replicates),
            int(args.analysis_seed),
        )
        print(f"[skip analysis] {analysis_dir}", flush=True)
    else:
        _run_command(
            build_analysis_command(
                args.python,
                online_root,
                analysis_dir,
                folds,
                seeds,
                int(args.analysis_bootstrap_replicates),
                int(args.analysis_seed),
            ),
            bool(args.dry_run),
            context="paired A0/A3 analysis stage",
        )
        if not args.dry_run:
            validate_analysis(
                analysis_dir,
                online_root,
                folds,
                seeds,
                int(args.analysis_bootstrap_replicates),
                int(args.analysis_seed),
            )
    return {"counts": counts, "work_root": str(work_root), "dry_run": bool(args.dry_run)}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Resume-safe 7-fold x 4-seed A0/A3 adaptive-codebook pipeline."
    )
    parser.add_argument("--cv-root", required=True)
    parser.add_argument("--npz-path", required=True)
    parser.add_argument("--encoder-root", required=True)
    parser.add_argument("--work-root", required=True)
    parser.add_argument("--folds", default="1,2,3,4,5,6,7")
    parser.add_argument("--seeds", default="0,5,50,500")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--bootstrap-resamples", type=int, default=10000)
    parser.add_argument("--analysis-bootstrap-replicates", type=int, default=10000)
    parser.add_argument("--analysis-seed", type=int, default=20260904)
    parser.add_argument(
        "--stop-after",
        choices=("encoders", "fixed", "online", "analysis"),
        default="analysis",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Audit existing artifacts and print missing-stage commands without writing or running them.",
    )
    parser.add_argument(
        "--refresh-incompatible-online",
        action="store_true",
        help=(
            "Recoverably move completed online runs that fail current source/identity "
            "validation into online/_incomplete before rebuilding them."
        ),
    )
    return parser.parse_args(argv)


def main() -> None:
    result = run_pipeline(parse_args())
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
