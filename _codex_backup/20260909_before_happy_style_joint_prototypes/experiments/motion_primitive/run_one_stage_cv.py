"""Run and aggregate the seven-fold J0-U/J0-T screening grid."""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.motion_primitive.train_one_stage import (  # noqa: E402
    CHECKPOINT_SELECTION_POLICIES,
    CODEBOOK_UPDATE_MODES,
    CORE_SOURCE_RELATIVE_PATHS as TRAIN_CORE_SOURCE_RELATIVE_PATHS,
    REQUIRED_COMPLETION_ARTIFACTS,
    SCHEMA_VERSION as MEMBER_SCHEMA_VERSION,
)
from models.motion_trajectory import (  # noqa: E402
    LOCAL_ENCODER_TYPES,
    TRAJECTORY_INPUT_MODES,
)


TRAIN_SCRIPT = Path(__file__).resolve().with_name("train_one_stage.py")
CV_SCHEMA_VERSION = "one_stage_motion_trajectory_cv_v4"
PROFILES = ("J0-U", "J0-T")
METRICS = ("all_accuracy", "old_accuracy", "new_accuracy", "h_score", "macro_f1")
DIAGNOSTIC_METRICS = (
    "sitting_recall",
    "standing_recall",
    "sit_stand_balanced_accuracy",
    "order_h_score_drop",
    "occupancy_fraction",
    "effective_code_count",
    "validation_hard_occupancy_fraction",
    "validation_hard_effective_code_count",
    "validation_hard_effective_code_fraction",
    "validation_hard_max_code_share",
    "validation_assignment_margin",
    "validation_local_feature_effective_rank",
    "validation_local_feature_centroid_norm",
    "token_subject_nmi",
    "primitive_segment_start_repeatability_margin",
    "mean_primitive_segment_count",
)
RUN_CORE_SOURCE_RELATIVE_PATHS = tuple(TRAIN_CORE_SOURCE_RELATIVE_PATHS)
CV_SOURCE_RELATIVE_PATHS = (
    *RUN_CORE_SOURCE_RELATIVE_PATHS,
    "experiments/motion_primitive/run_one_stage_cv.py",
)


def _parse_ints(value: str, *, minimum: int | None = None, maximum: int | None = None) -> list[int]:
    pieces = [item.strip() for item in str(value).replace(" ", ",").split(",") if item.strip()]
    if not pieces:
        raise ValueError("Expected at least one integer.")
    result = [int(item) for item in pieces]
    if len(result) != len(set(result)):
        raise ValueError(f"Duplicate integer in {value!r}.")
    if minimum is not None and any(item < minimum for item in result):
        raise ValueError(f"Values must be >= {minimum}.")
    if maximum is not None and any(item > maximum for item in result):
        raise ValueError(f"Values must be <= {maximum}.")
    return result


def _parse_profiles(value: str) -> list[str]:
    result = [item.strip().upper() for item in str(value).split(",") if item.strip()]
    if not result or len(result) != len(set(result)):
        raise ValueError("--profiles must contain unique J0-U/J0-T values.")
    unknown = sorted(set(result) - set(PROFILES))
    if unknown:
        raise ValueError(f"Unknown profiles: {unknown}.")
    return result


def _canonical_hash(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _source_identity(relative_paths: Sequence[str]) -> dict[str, Any]:
    files = {}
    for relative in relative_paths:
        path = PROJECT_ROOT / relative
        if not path.is_file():
            raise FileNotFoundError(f"Required one-stage source file is missing: {path}.")
        files[relative] = _file_sha256(path)
    return {
        "schema": "one_stage_core_source_sha256_v1",
        "files": files,
        "identity_sha256": _canonical_hash(files),
    }


def _write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    temporary.replace(path)


def _bootstrap_interval(values: Sequence[float], replicates: int, seed: int) -> dict[str, float]:
    data = np.asarray(values, dtype=np.float64)
    if data.ndim != 1 or not len(data) or not np.all(np.isfinite(data)):
        raise ValueError("Bootstrap values must be a finite non-empty vector.")
    rng = np.random.default_rng(int(seed))
    indices = rng.integers(0, len(data), size=(int(replicates), len(data)))
    means = data[indices].mean(axis=1)
    return {
        "mean": float(data.mean()),
        "standard_deviation": float(data.std(ddof=1)) if len(data) > 1 else 0.0,
        "ci95_low": float(np.quantile(means, 0.025)),
        "ci95_high": float(np.quantile(means, 0.975)),
    }


def _exact_sign_flip_paired(differences: Sequence[float]) -> dict[str, Any]:
    values = np.asarray(differences, dtype=np.float64)
    if values.ndim != 1 or not len(values) or not np.all(np.isfinite(values)):
        raise ValueError("Paired differences must be finite and non-empty.")
    observed = float(values.mean())
    permuted = []
    for signs in itertools.product((-1.0, 1.0), repeat=len(values)):
        permuted.append(float(np.mean(values * np.asarray(signs))))
    permuted_array = np.asarray(permuted)
    return {
        "mean_J0_T_minus_J0_U": observed,
        "exact_one_sided_p_greater": float(np.mean(permuted_array >= observed - 1e-15)),
        "fold_difference_values": values.tolist(),
        "permutation_count": int(len(permuted_array)),
    }


def _aggregation_protocol(bootstrap_replicates: int, aggregate_seed: int) -> dict[str, Any]:
    return {
        "schema": "one_stage_fold_blocked_aggregation_v2",
        "bootstrap_replicates": int(bootstrap_replicates),
        "aggregate_seed": int(aggregate_seed),
        "seed_reduction_within_fold": "arithmetic_mean",
        "bootstrap_unit": "subject_fold_after_seed_reduction",
        "paired_test": "exact_fold_level_sign_flip_J0_T_minus_J0_U",
        "prespecified_primary_endpoint": "h_score",
        "prespecified_alternative": "J0-T greater than J0-U",
        "secondary_endpoint_policy": (
            "all_accuracy, old_accuracy, new_accuracy, and macro_f1 are "
            "exploratory; their unadjusted p-values are not confirmatory"
        ),
    }


def _command(args: argparse.Namespace, profile: str, fold: int, seed: int, output: Path) -> list[str]:
    command = [
        sys.executable,
        str(TRAIN_SCRIPT),
        "--data-root",
        str(args.data_root),
        "--output-dir",
        str(output),
        "--profile",
        profile,
        "--fold",
        str(fold),
        "--seed",
        str(seed),
        "--epochs",
        str(args.epochs),
        "--batch-size",
        str(args.batch_size),
        "--eval-batch-size",
        str(args.eval_batch_size),
        "--num-workers",
        str(args.num_workers),
        "--device",
        str(args.device),
        "--learning-rate",
        str(args.learning_rate),
        "--weight-decay",
        str(args.weight_decay),
        "--gradient-clip",
        str(args.gradient_clip),
        "--warmup-epochs",
        str(args.warmup_epochs),
        "--early-stop-patience",
        str(args.early_stop_patience),
        "--frame-size",
        str(args.frame_size),
        "--frame-stride",
        str(args.frame_stride),
        "--local-encoder",
        str(args.local_encoder),
        "--codebook-size",
        str(args.codebook_size),
        "--codebook-init",
        str(args.codebook_init),
        "--codebook-init-windows-per-trial",
        str(args.codebook_init_windows_per_trial),
        "--codebook-init-n-init",
        str(args.codebook_init_n_init),
        "--codebook-update",
        str(args.codebook_update),
        "--codebook-ema-decay",
        str(args.codebook_ema_decay),
        "--trajectory-input-mode",
        str(args.trajectory_input_mode),
        "--temperature-start",
        str(args.temperature_start),
        "--temperature-end",
        str(args.temperature_end),
        "--trajectory-mask-ratio",
        str(args.trajectory_mask_ratio),
        "--labelled-fraction",
        str(args.labelled_fraction),
        "--anomaly-policy",
        str(args.anomaly_policy),
        "--cluster-restarts",
        str(args.cluster_restarts),
        "--order-shuffles",
        str(args.order_shuffles),
        "--minimum-segment-windows",
        str(args.minimum_segment_windows),
        "--minimum-hard-code-fraction",
        str(args.minimum_hard_code_fraction),
        "--minimum-hard-effective-code-fraction",
        str(args.minimum_hard_effective_code_fraction),
        "--maximum-hard-code-share",
        str(args.maximum_hard_code_share),
        "--minimum-local-feature-effective-rank",
        str(args.minimum_local_feature_effective_rank),
        "--maximum-local-feature-centroid-norm",
        str(args.maximum_local_feature_centroid_norm),
        "--minimum-assignment-margin",
        str(args.minimum_assignment_margin),
        "--checkpoint-selection",
        str(args.checkpoint_selection),
        "--deterministic" if args.deterministic else "--no-deterministic",
        "--use-gumbel-training"
        if args.use_gumbel_training
        else "--no-use-gumbel-training",
    ]
    if args.resume:
        command.append("--resume")
    return command


def _load_rows(
    root: Path,
    profiles: Sequence[str],
    folds: Sequence[int],
    seeds: Sequence[int],
    *,
    expected_member_source_identity: str | None = None,
    expected_trajectory_input_mode: str | None = None,
    expected_checkpoint_selection: str | None = None,
    expected_codebook_update: str | None = None,
) -> list[dict[str, Any]]:
    rows = []
    shared_runtime_identity: str | None = None
    shared_raw_manifest_identity: str | None = None
    for profile in profiles:
        for fold in folds:
            for seed in seeds:
                run_dir = root / f"profile_{profile}" / f"fold_{fold:02d}_seed_{seed}"
                complete = run_dir / "complete.json"
                summary_path = run_dir / "summary.json"
                manifest_path = run_dir / "run_manifest.json"
                checkpoint_path = run_dir / "checkpoint_best.pt"
                if not all(
                    path.is_file()
                    for path in (complete, summary_path, manifest_path, checkpoint_path)
                ):
                    raise RuntimeError(f"Incomplete expected run: {run_dir}.")
                completion = json.loads(complete.read_text(encoding="utf-8"))
                summary = json.loads(summary_path.read_text(encoding="utf-8"))
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                if (
                    summary.get("schema") != MEMBER_SCHEMA_VERSION
                    or manifest.get("schema") != MEMBER_SCHEMA_VERSION
                ):
                    raise RuntimeError(
                        f"Member run schema is not {MEMBER_SCHEMA_VERSION} in {run_dir}."
                    )
                identities = {
                    str(completion.get("run_identity_sha256", "")),
                    str(summary.get("run_identity_sha256", "")),
                    str(manifest.get("run_identity_sha256", "")),
                }
                if len(identities) != 1 or "" in identities:
                    raise RuntimeError(f"Run artifact identities disagree in {run_dir}.")
                run_identity = next(iter(identities))
                identity_payload = {
                    key: value
                    for key, value in manifest.items()
                    if key
                    not in {
                        "run_identity_sha256",
                        "data_audit",
                        "boundary_calibration_sha256",
                        "command",
                    }
                }
                if _canonical_hash(identity_payload) != run_identity:
                    raise RuntimeError(
                        f"run_manifest.json identity payload failed verification in {run_dir}."
                    )
                runtime_environment = manifest.get("runtime_environment")
                if not isinstance(runtime_environment, dict):
                    raise RuntimeError(
                        f"Run manifest runtime environment is malformed in {run_dir}."
                    )
                runtime_identity = runtime_environment.get("identity_sha256")
                runtime_payload = {
                    key: value
                    for key, value in runtime_environment.items()
                    if key != "identity_sha256"
                }
                if (
                    not isinstance(runtime_identity, str)
                    or not runtime_identity
                    or _canonical_hash(runtime_payload) != runtime_identity
                ):
                    raise RuntimeError(
                        f"Runtime environment identity failed verification in {run_dir}."
                    )
                data_audit = manifest.get("data_audit")
                if not isinstance(data_audit, dict):
                    raise RuntimeError(f"Run manifest data audit is malformed in {run_dir}.")
                raw_manifest_identity = data_audit.get("manifest_sha256")
                if (
                    not isinstance(raw_manifest_identity, str)
                    or not raw_manifest_identity
                    or data_audit.get("identity_sha256")
                    != manifest.get("data_protocol_identity")
                ):
                    raise RuntimeError(
                        f"Run manifest data identity failed verification in {run_dir}."
                    )
                if shared_runtime_identity is None:
                    shared_runtime_identity = runtime_identity
                elif shared_runtime_identity != runtime_identity:
                    raise RuntimeError(
                        "CV members were produced by different runtime environments; "
                        f"first={shared_runtime_identity}, observed={runtime_identity}."
                    )
                if shared_raw_manifest_identity is None:
                    shared_raw_manifest_identity = raw_manifest_identity
                elif shared_raw_manifest_identity != raw_manifest_identity:
                    raise RuntimeError(
                        "CV members were produced from different raw USC-HAD manifests; "
                        f"first={shared_raw_manifest_identity}, observed={raw_manifest_identity}."
                    )
                if completion.get("summary_sha256") != _file_sha256(summary_path):
                    raise RuntimeError(f"summary.json integrity check failed in {run_dir}.")
                checkpoint_hash = _file_sha256(checkpoint_path)
                if (
                    completion.get("selected_checkpoint_sha256") != checkpoint_hash
                    or summary.get("selected_checkpoint_sha256") != checkpoint_hash
                ):
                    raise RuntimeError(
                        f"checkpoint_best.pt integrity check failed in {run_dir}."
                    )
                recorded_artifacts = completion.get("artifact_sha256")
                if not isinstance(recorded_artifacts, dict):
                    raise RuntimeError(
                        f"Completed run has no artifact_sha256 integrity manifest: {run_dir}."
                    )
                required_artifacts = set(REQUIRED_COMPLETION_ARTIFACTS)
                if set(recorded_artifacts) != required_artifacts:
                    raise RuntimeError(
                        f"Completed artifact manifest has a missing or unexpected file set in {run_dir}."
                    )
                for relative in REQUIRED_COMPLETION_ARTIFACTS:
                    artifact = run_dir / relative
                    if not artifact.is_file():
                        raise RuntimeError(
                            f"Completed run is missing required artifact: {artifact}."
                        )
                    if recorded_artifacts[relative] != _file_sha256(artifact):
                        raise RuntimeError(
                            f"Completed artifact failed its SHA256 check in {run_dir}: {relative}."
                        )
                observed_source_identity = (
                    manifest.get("source_identity", {}).get("identity_sha256")
                )
                if (
                    expected_member_source_identity is not None
                    and observed_source_identity != expected_member_source_identity
                ):
                    raise RuntimeError(
                        f"One-stage source identity mismatch in {manifest_path}."
                    )
                if (
                    summary.get("profile") != profile
                    or int(summary.get("fold", -1)) != int(fold)
                    or int(summary.get("seed", -1)) != int(seed)
                    or manifest.get("profile") != profile
                    or int(manifest.get("fold", -1)) != int(fold)
                    or int(manifest.get("seed", -1)) != int(seed)
                ):
                    raise RuntimeError(f"Run identity fields disagree in {summary_path}.")
                trajectory_input_mode = str(summary.get("trajectory_input_mode", ""))
                trajectory_assignment_forward = str(
                    summary.get("trajectory_assignment_forward", "")
                )
                checkpoint_selection = str(summary.get("checkpoint_selection", ""))
                codebook_update = str(summary.get("codebook_update", ""))
                local_encoder = str(summary.get("local_encoder", ""))
                manifest_model_config = manifest.get("model_config")
                manifest_training_parameters = manifest.get("training_parameters")
                if not isinstance(manifest_model_config, dict) or not isinstance(
                    manifest_training_parameters, dict
                ):
                    raise RuntimeError(f"Run manifest configuration is malformed in {run_dir}.")
                if (
                    manifest_model_config.get("trajectory_input_mode")
                    != trajectory_input_mode
                    or manifest_model_config.get("local_encoder_type")
                    != local_encoder
                    or manifest_training_parameters.get("checkpoint_selection")
                    != checkpoint_selection
                    or manifest_training_parameters.get("codebook_update")
                    != codebook_update
                ):
                    raise RuntimeError(
                        f"Summary and manifest trajectory protocol disagree in {run_dir}."
                    )
                expected_assignment_forward = (
                    "hard_one_hot_straight_through_gumbel"
                    if bool(manifest_model_config.get("use_gumbel_training"))
                    else "hard_one_hot_straight_through_deterministic"
                )
                if trajectory_assignment_forward != expected_assignment_forward:
                    raise RuntimeError(
                        f"Summary and manifest assignment protocol disagree in {run_dir}."
                    )
                if (
                    expected_trajectory_input_mode is not None
                    and trajectory_input_mode != expected_trajectory_input_mode
                ):
                    raise RuntimeError(
                        f"Unexpected trajectory input mode in {summary_path}: "
                        f"{trajectory_input_mode!r}."
                    )
                if (
                    expected_checkpoint_selection is not None
                    and checkpoint_selection != expected_checkpoint_selection
                ):
                    raise RuntimeError(
                        f"Unexpected checkpoint selection policy in {summary_path}: "
                        f"{checkpoint_selection!r}."
                    )
                if (
                    expected_codebook_update is not None
                    and codebook_update != expected_codebook_update
                ):
                    raise RuntimeError(
                        f"Unexpected codebook update mode in {summary_path}: "
                        f"{codebook_update!r}."
                    )
                primary_marker = summary.get("primary_motion_primitive_cgcd_arm")
                if not isinstance(primary_marker, bool):
                    raise RuntimeError(
                        f"Primary motion-primitive arm marker is not boolean in {summary_path}."
                    )
                primary_arm = primary_marker
                if primary_arm != (
                    trajectory_input_mode == "primitive_only"
                    and codebook_update == "ema"
                    and local_encoder == "resnet1d"
                    and trajectory_assignment_forward
                    == "hard_one_hot_straight_through_deterministic"
                ):
                    raise RuntimeError(
                        f"Primary motion-primitive arm marker is inconsistent in {summary_path}."
                    )
                metrics = summary["evaluation"]["cgcd_metrics"]
                recalls = metrics["per_class_recall"]
                sitting_recall = float(recalls["Sitting"])
                standing_recall = float(recalls["Standing"])
                diagnostics = summary["evaluation"]["codebook_diagnostics"]
                selection_validation = summary.get("selection_validation", {})
                if not bool(selection_validation.get("checkpoint_eligible", False)):
                    raise RuntimeError(
                        f"Selected checkpoint failed the hard non-collapse gate in {run_dir}."
                    )
                repeatability = summary["evaluation"][
                    "primitive_segment_start_repeatability"
                ]
                row = {
                    "profile": profile,
                    "trajectory_input_mode": trajectory_input_mode,
                    "trajectory_assignment_forward": trajectory_assignment_forward,
                    "checkpoint_selection": checkpoint_selection,
                    "codebook_update": codebook_update,
                    "local_encoder": local_encoder,
                    "primary_motion_primitive_cgcd_arm": primary_arm,
                    "runtime_identity_sha256": runtime_identity,
                    "raw_manifest_sha256": raw_manifest_identity,
                    "fold": int(fold),
                    "seed": int(seed),
                    "run_dir": str(run_dir),
                    "selected_epoch": int(summary["selected_epoch"]),
                    **{key: float(metrics[key]) for key in METRICS},
                    "sitting_recall": sitting_recall,
                    "standing_recall": standing_recall,
                    "sit_stand_balanced_accuracy": float(
                        0.5 * (sitting_recall + standing_recall)
                    ),
                    "order_h_score_drop": float(
                        summary["evaluation"]["order_shuffle_control"][
                            "identity_control_minus_shuffled"
                        ]["h_score"]
                    ),
                    "effective_code_count": float(
                        diagnostics["effective_code_count"]
                    ),
                    "validation_hard_occupancy_fraction": float(
                        selection_validation["hard_occupancy_fraction"]
                    ),
                    "validation_hard_effective_code_count": float(
                        selection_validation["hard_effective_code_count"]
                    ),
                    "validation_hard_effective_code_fraction": float(
                        selection_validation["hard_effective_code_fraction"]
                    ),
                    "validation_hard_max_code_share": float(
                        selection_validation["hard_max_code_share"]
                    ),
                    "validation_assignment_margin": float(
                        selection_validation["mean_assignment_margin"]
                    ),
                    "validation_local_feature_effective_rank": float(
                        selection_validation["local_feature_effective_rank"]
                    ),
                    "validation_local_feature_centroid_norm": float(
                        selection_validation["local_feature_centroid_norm"]
                    ),
                    "token_subject_nmi": float(
                        diagnostics["token_subject_nmi"]
                    ),
                    "occupancy_fraction": float(diagnostics["occupancy_fraction"]),
                    "primitive_segment_start_repeatability_margin": float(
                        repeatability["same_minus_different_margin"]
                    ),
                    "mean_primitive_segment_count": float(
                        summary["evaluation"]["primitive_segment_counts"]["mean"]
                    ),
                }
                rows.append(row)
    return rows


def _aggregate(rows: Sequence[dict[str, Any]], profiles: Sequence[str], folds: Sequence[int], seeds: Sequence[int], bootstrap_replicates: int, seed: int) -> dict[str, Any]:
    expected_count = len(profiles) * len(folds) * len(seeds)
    if len(rows) != expected_count:
        raise RuntimeError(f"Expected {expected_count} result rows, got {len(rows)}.")
    aggregate: dict[str, Any] = {
        "grid": {
            "profiles": list(profiles),
            "folds": list(folds),
            "seeds": list(seeds),
            "run_count": len(rows),
        },
        "aggregation_protocol": _aggregation_protocol(
            int(bootstrap_replicates), int(seed)
        ),
        "profiles": {},
    }
    fold_means_by_profile: dict[str, dict[int, dict[str, float]]] = {}
    for profile_index, profile in enumerate(profiles):
        selected = [item for item in rows if item["profile"] == profile]
        fold_means = {}
        for fold in folds:
            fold_rows = [item for item in selected if item["fold"] == fold]
            if len(fold_rows) != len(seeds):
                raise RuntimeError(f"Unbalanced seeds for profile={profile}, fold={fold}.")
            fold_means[int(fold)] = {
                key: float(np.mean([item[key] for item in fold_rows]))
                for key in (*METRICS, *DIAGNOSTIC_METRICS)
            }
        fold_means_by_profile[profile] = fold_means
        aggregate["profiles"][profile] = {
            key: _bootstrap_interval(
                [fold_means[fold][key] for fold in folds],
                int(bootstrap_replicates),
                int(seed) + 1009 * profile_index + 31 * key_index,
            )
            for key_index, key in enumerate(
                (*METRICS, *DIAGNOSTIC_METRICS)
            )
        }
        aggregate["profiles"][profile]["fold_seed_averages"] = fold_means

    if set(PROFILES).issubset(profiles):
        paired = {}
        for metric in METRICS:
            differences = [
                fold_means_by_profile["J0-T"][fold][metric]
                - fold_means_by_profile["J0-U"][fold][metric]
                for fold in folds
            ]
            paired[metric] = {
                **_exact_sign_flip_paired(differences),
                "inference_role": (
                    "prespecified_primary"
                    if metric == "h_score"
                    else "exploratory_unadjusted"
                ),
                "bootstrap_ci": _bootstrap_interval(
                    differences, int(bootstrap_replicates), int(seed) + 7001
                ),
            }
        aggregate["paired_J0_T_vs_J0_U"] = paired
    aggregate["statistical_unit"] = (
        "subject fold after averaging seeds; folds have overlapping train sets, so p-values are screening evidence"
    )
    return aggregate


def _write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    fields = [
        "profile",
        "trajectory_input_mode",
        "trajectory_assignment_forward",
        "checkpoint_selection",
        "codebook_update",
        "local_encoder",
        "primary_motion_primitive_cgcd_arm",
        "runtime_identity_sha256",
        "raw_manifest_sha256",
        "fold",
        "seed",
        *METRICS,
        *DIAGNOSTIC_METRICS,
        "selected_epoch",
        "run_dir",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _plot_aggregate(path: Path, aggregate: dict[str, Any], profiles: Sequence[str]) -> None:
    metrics = ("all_accuracy", "old_accuracy", "new_accuracy", "h_score")
    x = np.arange(len(metrics))
    width = 0.34 if len(profiles) == 2 else 0.7 / max(1, len(profiles))
    figure, axis = plt.subplots(figsize=(9, 5.5), constrained_layout=True)
    for index, profile in enumerate(profiles):
        values = [aggregate["profiles"][profile][metric]["mean"] for metric in metrics]
        low = [
            value - aggregate["profiles"][profile][metric]["ci95_low"]
            for value, metric in zip(values, metrics)
        ]
        high = [
            aggregate["profiles"][profile][metric]["ci95_high"] - value
            for value, metric in zip(values, metrics)
        ]
        offset = (index - (len(profiles) - 1) / 2.0) * width
        axis.bar(
            x + offset,
            values,
            width=width,
            label=profile,
            yerr=np.asarray([low, high]),
            capsize=4,
        )
    axis.set_xticks(x, ["All", "Old", "New", "H-score"])
    axis.set_ylim(0.0, 1.0)
    axis.set_ylabel("Accuracy / harmonic score")
    axis.set_title("One-stage motion-primitive trajectory: fold-blocked screening")
    axis.legend()
    axis.grid(axis="y", alpha=0.25)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Seven-fold one-stage J0 grid runner.")
    default_data = r"D:\WorkDir\DataSet\USC-HAD" if os.name == "nt" else "/mnt/d/WorkDir/DataSet/USC-HAD"
    parser.add_argument("--data-root", default=default_data)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--profiles", default="J0-U,J0-T")
    parser.add_argument("--folds", default="1,2,3,4,5,6,7")
    parser.add_argument("--seeds", default="50")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--gradient-clip", type=float, default=5.0)
    parser.add_argument("--warmup-epochs", type=int, default=10)
    parser.add_argument("--early-stop-patience", type=int, default=20)
    parser.add_argument("--frame-size", type=int, default=128)
    parser.add_argument("--frame-stride", type=int, default=64)
    parser.add_argument(
        "--local-encoder", choices=LOCAL_ENCODER_TYPES, default="resnet1d"
    )
    parser.add_argument("--codebook-size", type=int, default=32)
    parser.add_argument(
        "--codebook-init", choices=("kmeans++", "random"), default="kmeans++"
    )
    parser.add_argument("--codebook-init-windows-per-trial", type=int, default=4)
    parser.add_argument("--codebook-init-n-init", type=int, default=10)
    parser.add_argument(
        "--codebook-update", choices=CODEBOOK_UPDATE_MODES, default="ema"
    )
    parser.add_argument("--codebook-ema-decay", type=float, default=0.99)
    parser.add_argument(
        "--trajectory-input-mode",
        choices=TRAJECTORY_INPUT_MODES,
        default="primitive_only",
        help=(
            "Classifier ingress. primitive_only is the primary motion-primitive "
            "CGCD arm; other modes are attribution controls."
        ),
    )
    parser.add_argument("--temperature-start", type=float, default=2.0)
    parser.add_argument("--temperature-end", type=float, default=0.25)
    parser.add_argument(
        "--use-gumbel-training",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--trajectory-mask-ratio", type=float, default=0.15)
    parser.add_argument("--labelled-fraction", type=float, default=0.8)
    parser.add_argument("--anomaly-policy", choices=("report", "exclude"), default="report")
    parser.add_argument("--cluster-restarts", type=int, default=10)
    parser.add_argument("--order-shuffles", type=int, default=10)
    parser.add_argument("--minimum-segment-windows", type=int, default=2)
    parser.add_argument("--minimum-hard-code-fraction", type=float, default=0.25)
    parser.add_argument(
        "--minimum-hard-effective-code-fraction", type=float, default=0.20
    )
    parser.add_argument("--maximum-hard-code-share", type=float, default=0.50)
    parser.add_argument(
        "--minimum-local-feature-effective-rank", type=float, default=4.0
    )
    parser.add_argument(
        "--maximum-local-feature-centroid-norm", type=float, default=0.98
    )
    parser.add_argument("--minimum-assignment-margin", type=float, default=0.01)
    parser.add_argument(
        "--checkpoint-selection",
        choices=CHECKPOINT_SELECTION_POLICIES,
        default="common_unsupervised",
    )
    parser.add_argument("--bootstrap-replicates", type=int, default=10000)
    parser.add_argument("--aggregate-seed", type=int, default=20260908)
    parser.add_argument("--deterministic", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def run(args: argparse.Namespace) -> dict[str, Any]:
    profiles = _parse_profiles(args.profiles)
    folds = _parse_ints(args.folds, minimum=1, maximum=7)
    seeds = _parse_ints(args.seeds, minimum=0)
    for name in (
        "epochs",
        "batch_size",
        "eval_batch_size",
        "frame_size",
        "frame_stride",
        "codebook_size",
        "codebook_init_windows_per_trial",
        "codebook_init_n_init",
        "cluster_restarts",
        "order_shuffles",
        "bootstrap_replicates",
        "minimum_segment_windows",
    ):
        if int(getattr(args, name)) < 1:
            raise ValueError(f"--{name.replace('_', '-')} must be positive.")
    for name in ("num_workers", "warmup_epochs", "early_stop_patience", "aggregate_seed"):
        if int(getattr(args, name)) < 0:
            raise ValueError(f"--{name.replace('_', '-')} cannot be negative.")
    for name in ("learning_rate", "gradient_clip", "temperature_start", "temperature_end"):
        value = float(getattr(args, name))
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive and finite.")
    weight_decay = float(args.weight_decay)
    if not math.isfinite(weight_decay) or weight_decay < 0:
        raise ValueError("--weight-decay must be finite and non-negative.")
    if int(args.codebook_size) < 2:
        raise ValueError("--codebook-size must be at least two.")
    if int(args.frame_stride) > int(args.frame_size):
        raise ValueError("--frame-stride cannot exceed --frame-size in J0.")
    for name in (
        "minimum_hard_code_fraction",
        "minimum_hard_effective_code_fraction",
        "maximum_hard_code_share",
        "maximum_local_feature_centroid_norm",
    ):
        value = float(getattr(args, name))
        if not math.isfinite(value) or not 0.0 < value <= 1.0:
            raise ValueError(f"--{name.replace('_', '-')} must lie in (0,1].")
    ema_decay = float(args.codebook_ema_decay)
    if not math.isfinite(ema_decay) or not 0.0 <= ema_decay < 1.0:
        raise ValueError("--codebook-ema-decay must lie in [0,1).")
    for name in (
        "minimum_local_feature_effective_rank",
        "minimum_assignment_margin",
    ):
        value = float(getattr(args, name))
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(f"--{name.replace('_', '-')} must be finite and non-negative.")
    mask_ratio = float(args.trajectory_mask_ratio)
    if not math.isfinite(mask_ratio) or not 0.0 < mask_ratio < 1.0:
        raise ValueError("--trajectory-mask-ratio must lie strictly between zero and one.")
    labelled_fraction = float(args.labelled_fraction)
    if not math.isfinite(labelled_fraction) or not 0.0 < labelled_fraction < 1.0:
        raise ValueError("--labelled-fraction must lie strictly between zero and one.")
    output_root = Path(args.output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    identity = {
        "schema": CV_SCHEMA_VERSION,
        "profiles": profiles,
        "folds": folds,
        "seeds": seeds,
        "data_root": str(Path(args.data_root).expanduser().resolve()),
        "training_parameters": {
            name: getattr(args, name)
            for name in (
                "epochs",
                "batch_size",
                "eval_batch_size",
                "num_workers",
                "device",
                "learning_rate",
                "weight_decay",
                "gradient_clip",
                "warmup_epochs",
                "early_stop_patience",
                "frame_size",
                "frame_stride",
                "local_encoder",
                "codebook_size",
                "codebook_init",
                "codebook_init_windows_per_trial",
                "codebook_init_n_init",
                "codebook_update",
                "codebook_ema_decay",
                "trajectory_input_mode",
                "temperature_start",
                "temperature_end",
                "use_gumbel_training",
                "trajectory_mask_ratio",
                "labelled_fraction",
                "anomaly_policy",
                "cluster_restarts",
                "order_shuffles",
                "minimum_segment_windows",
                "minimum_hard_code_fraction",
                "minimum_hard_effective_code_fraction",
                "maximum_hard_code_share",
                "minimum_local_feature_effective_rank",
                "maximum_local_feature_centroid_norm",
                "minimum_assignment_margin",
                "checkpoint_selection",
                "deterministic",
            )
        },
        "primary_motion_primitive_cgcd_arm": bool(
            args.trajectory_input_mode == "primitive_only"
            and not args.use_gumbel_training
            and args.codebook_update == "ema"
            and args.local_encoder == "resnet1d"
        ),
        "trajectory_assignment_forward": (
            "hard_one_hot_straight_through_gumbel"
            if args.use_gumbel_training
            else "hard_one_hot_straight_through_deterministic"
        ),
        "aggregation_protocol": _aggregation_protocol(
            int(args.bootstrap_replicates), int(args.aggregate_seed)
        ),
        "one_stage_train_script": str(TRAIN_SCRIPT),
        "member_source_identity": _source_identity(
            RUN_CORE_SOURCE_RELATIVE_PATHS
        ),
        "cv_source_identity": _source_identity(CV_SOURCE_RELATIVE_PATHS),
    }
    identity["identity_sha256"] = _canonical_hash(identity)
    manifest_path = output_root / "cv_manifest.json"
    if manifest_path.exists():
        recorded = json.loads(manifest_path.read_text(encoding="utf-8"))
        recorded_payload = {
            key: value for key, value in recorded.items() if key != "identity_sha256"
        }
        if _canonical_hash(recorded_payload) != recorded.get("identity_sha256"):
            raise RuntimeError("Existing CV manifest failed its identity integrity check.")
        if recorded.get("identity_sha256") != identity["identity_sha256"]:
            raise RuntimeError(
                "Output root records a different CV identity/member set; use a new --output-root."
            )
    elif any(output_root.iterdir()):
        raise RuntimeError(
            f"Non-empty output root has no CV manifest: {output_root}."
        )
    else:
        _write_json(manifest_path, identity)

    commands = []
    for profile in profiles:
        for fold in folds:
            for seed in seeds:
                run_dir = output_root / f"profile_{profile}" / f"fold_{fold:02d}_seed_{seed}"
                command = _command(args, profile, fold, seed, run_dir)
                commands.append(command)
                print("[command] " + subprocess.list2cmdline(command), flush=True)
                if not args.dry_run:
                    subprocess.run(command, cwd=PROJECT_ROOT, check=True)
    if args.dry_run:
        return {"dry_run": True, "commands": commands, "identity": identity}

    rows = _load_rows(
        output_root,
        profiles,
        folds,
        seeds,
        expected_member_source_identity=identity["member_source_identity"][
            "identity_sha256"
        ],
        expected_trajectory_input_mode=str(args.trajectory_input_mode),
        expected_checkpoint_selection=str(args.checkpoint_selection),
        expected_codebook_update=str(args.codebook_update),
    )
    aggregate = _aggregate(
        rows,
        profiles,
        folds,
        seeds,
        int(args.bootstrap_replicates),
        int(args.aggregate_seed),
    )
    result = {"identity": identity, "rows": rows, "aggregate": aggregate}
    _write_json(output_root / "aggregate_summary.json", result)
    _write_csv(output_root / "per_run_metrics.csv", rows)
    _plot_aggregate(output_root / "aggregate_metrics.png", aggregate, profiles)
    return result


def main() -> None:
    result = run(build_parser().parse_args())
    if result.get("dry_run"):
        print(f"dry-run generated {len(result['commands'])} commands", flush=True)
    else:
        for profile, metrics in result["aggregate"]["profiles"].items():
            print(
                f"{profile}: all={metrics['all_accuracy']['mean']:.4f} "
                f"old={metrics['old_accuracy']['mean']:.4f} "
                f"new={metrics['new_accuracy']['mean']:.4f} "
                f"H={metrics['h_score']['mean']:.4f}",
                flush=True,
            )


if __name__ == "__main__":
    main()
