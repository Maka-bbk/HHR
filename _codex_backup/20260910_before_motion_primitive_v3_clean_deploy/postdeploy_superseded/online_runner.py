"""Run one trajectory-only USC-HAD continual-GCD stream.

Activity labels remain available to the protocol audit and final evaluator,
but the online optimisation API receives only the two unlabelled views.  Test
labels never enter a loss, hyper-parameter choice, or checkpoint selection.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import math
import os
import random
import sys
from argparse import Namespace
from dataclasses import fields
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Optional, Sequence

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.uschad_har import get_uschad_datasets  # noqa: E402
from experiments.motion_primitive.motion_online import (  # noqa: E402
    CodebookExpansionDecision,
    MotionPrimitiveOnlineConfig,
    extract_local_primitive_features,
    run_online_cgcd,
    select_kmeans_codebook_centres,
    select_residual_adaptive_codebook_centres,
)
from experiments.motion_primitive.profiles import PROFILE_JOINT, normalize_profile  # noqa: E402
from models.motion_primitive_cgcd import (  # noqa: E402
    MotionPrimitiveCGCDModel,
    MotionPrimitiveConfig,
)


OFFLINE_MANIFEST_SCHEMA = "hhr_motion_primitive_trajectory_manifest_v3"
OFFLINE_CHECKPOINT_SCHEMA = "hhr_motion_primitive_trajectory_offline_v3"
ONLINE_MANIFEST_SCHEMA = "hhr_motion_primitive_online_run_manifest_v3"
ONLINE_ROWS_SCHEMA = "hhr_motion_primitive_online_runs_v4"
CODEBOOK_POLICIES = ("none", "fixed_delta", "residual_adaptive")


def report_head_for_profile(profile: str) -> str:
    if normalize_profile(profile) == PROFILE_JOINT:
        return "trajectory"
    raise ValueError(f"Unknown online profile {profile!r}.")


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


def _load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"Cannot read JSON metadata {path}: {error}.") from error
    if not isinstance(payload, dict):
        raise RuntimeError(f"JSON metadata must contain an object: {path}.")
    return payload


def _torch_load(path: Path, *, map_location: torch.device | str) -> Mapping[str, Any]:
    try:
        value = torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:  # pragma: no cover - old torch compatibility
        value = torch.load(path, map_location=map_location)
    if not isinstance(value, Mapping):
        raise RuntimeError(f"Checkpoint is not a mapping: {path}.")
    return value


def _comma(values: Sequence[int]) -> str:
    return ",".join(str(int(value)) for value in values)


def _sha256_int_rows(*columns: Iterable[int]) -> str:
    arrays = [np.asarray(list(column), dtype=np.int64).reshape(-1) for column in columns]
    if not arrays:
        return hashlib.sha256(b"").hexdigest()
    if any(len(array) != len(arrays[0]) for array in arrays):
        raise ValueError("Identity-hash columns have different lengths.")
    matrix = np.stack(arrays, axis=1)
    order = np.lexsort(
        tuple(matrix[:, index] for index in reversed(range(matrix.shape[1])))
    )
    return hashlib.sha256(matrix[order].tobytes()).hexdigest()


def _dataset_summary(dataset: Any) -> dict[str, Any]:
    """Recompute the offline dataset identity without importing its trainer."""

    trial_ids = np.asarray(dataset.trial_global_ids, dtype=np.int64)
    subjects = np.asarray(dataset.subject_ids, dtype=np.int64)
    targets = np.asarray(dataset.targets, dtype=np.int64)
    return {
        "trial_count": int(len(dataset)),
        "trial_ids_sha256": _sha256_int_rows(trial_ids),
        "identity_sha256": _sha256_int_rows(trial_ids, subjects, targets),
        "subjects": sorted(set(int(value) for value in subjects.tolist())),
        "class_counts_physical_labels": {
            str(class_id): int(np.sum(targets == class_id))
            for class_id in sorted(set(int(value) for value in targets.tolist()))
        },
        "window_size": int(dataset.actual_window_size),
        "channels": int(dataset.actual_num_channels),
        "normalization_mode": str(dataset.normalization_mode),
        "normalization_stat_subjects": [
            int(value) for value in dataset.normalization_stat_subjects
        ],
        "normalization_stat_classes": [
            int(value) for value in dataset.normalization_stat_classes
        ],
    }


def _set_target_transform(dataset: Any, transform: Any) -> None:
    if dataset is None:
        return
    if isinstance(dataset, list):
        for item in dataset:
            _set_target_transform(item, transform)
        return
    dataset.target_transform = transform


def build_datasets(args: argparse.Namespace) -> tuple[Mapping[str, Any], np.ndarray]:
    """Rebuild the exact registered USC-HAD stream from offline arguments."""

    old_count = len(args.old_classes_parsed)
    novel_count = int(args.total_classes) - old_count
    if novel_count <= 0 or novel_count % int(args.novel_classes_per_session):
        raise ValueError(
            "total_classes - number of old classes must be positive and divisible "
            "by novel_classes_per_session."
        )
    args.continual_session_num = novel_count // int(args.novel_classes_per_session)
    args.num_novel_class_per_session = int(args.novel_classes_per_session)
    split_config = {
        "continual_session_num": args.continual_session_num,
        "online_old_seen_num": int(args.online_old_trials),
        "online_novel_unseen_num": int(args.online_novel_unseen_trials),
        "online_novel_seen_num": int(args.online_novel_seen_trials),
        "sample_unit": "trial",
    }
    datasets, novel_order = get_uschad_datasets(
        train_transform=None,
        test_transform=None,
        config_dict=split_config,
        train_classes=args.old_classes_parsed,
        prop_train_labels=1.0,
        split_train_val=False,
        is_shuffle=args.shuffle_novel_classes,
        seed=args.seed,
        args=args,
    )
    class_order = list(args.old_classes_parsed) + [
        int(value) for value in novel_order.tolist()
    ]
    target_mapping = {physical: index for index, physical in enumerate(class_order)}
    for dataset in datasets.values():
        _set_target_transform(
            dataset, lambda label, mapping=target_mapping: mapping[int(label)]
        )
    return datasets, novel_order


def resolve_device(specification: str) -> torch.device:
    requested = str(specification).strip().lower()
    if requested == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device(specification)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            f"CUDA device {specification!r} was requested but CUDA is unavailable."
        )
    return device


def seed_everything(seed: int, *, use_cuda: bool) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if use_cuda:
        torch.cuda.manual_seed_all(int(seed))
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def _resolve_npz_from_manifest(manifest: Mapping[str, Any]) -> Path:
    manifest_path = Path(str(manifest.get("npz_path", ""))).expanduser().resolve()
    arguments = manifest.get("arguments")
    if not isinstance(arguments, Mapping):
        raise RuntimeError("Offline manifest has no arguments mapping.")
    argument_path = Path(str(arguments.get("uschad_npz_path", ""))).expanduser().resolve()
    if manifest_path != argument_path:
        raise RuntimeError(
            "Offline manifest records two different NPZ paths: "
            f"{manifest_path} and {argument_path}."
        )
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Recorded USC-HAD NPZ does not exist: {manifest_path}.")
    expected_hash = str(manifest.get("npz_sha256", ""))
    observed_hash = _sha256_file(manifest_path)
    if not expected_hash or expected_hash != observed_hash:
        raise RuntimeError(
            "Recorded USC-HAD NPZ hash differs from the current file: "
            f"expected={expected_hash!r}, observed={observed_hash}."
        )
    return manifest_path


def _architecture_from_manifest(
    manifest: Mapping[str, Any], checkpoint: Mapping[str, Any]
) -> MotionPrimitiveConfig:
    architecture = manifest.get("architecture")
    checkpoint_architecture = checkpoint.get("architecture")
    if not isinstance(architecture, Mapping):
        raise RuntimeError("Offline manifest lacks architecture metadata.")
    if checkpoint_architecture != architecture:
        raise RuntimeError("Offline manifest and checkpoint architectures differ.")
    known = {field.name for field in fields(MotionPrimitiveConfig)}
    unknown = sorted(set(architecture) - known)
    missing = sorted(known - set(architecture))
    if unknown or missing:
        raise RuntimeError(
            f"Architecture metadata is not exact: missing={missing}, unknown={unknown}."
        )
    return MotionPrimitiveConfig(**dict(architecture)).validated()


def load_offline_model(
    offline_run_dir: Path,
    checkpoint_path: Path,
    *,
    device: torch.device,
) -> tuple[
    MotionPrimitiveCGCDModel,
    dict[str, Any],
    Mapping[str, Any],
    MotionPrimitiveConfig,
]:
    """Strictly reconstruct one validation-selected offline trajectory model."""

    run_dir = Path(offline_run_dir).expanduser().resolve()
    checkpoint_path = Path(checkpoint_path).expanduser().resolve()
    manifest_path = run_dir / "manifest.json"
    if not run_dir.is_dir() or not manifest_path.is_file():
        raise FileNotFoundError(f"Offline run manifest not found below {run_dir}.")
    if checkpoint_path.parent != run_dir:
        raise RuntimeError(
            "The selected checkpoint must be a direct member of --offline-run-dir; "
            "cross-run checkpoint mixing is forbidden."
        )
    if checkpoint_path.name != "checkpoint_best_trajectory.pt":
        raise RuntimeError(
            "Online CGCD must start from checkpoint_best_trajectory.pt; "
            "last-epoch or pooled-era aliases are not accepted."
        )
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Offline checkpoint not found: {checkpoint_path}.")

    manifest = _load_json(manifest_path)
    if manifest.get("schema") != OFFLINE_MANIFEST_SCHEMA:
        raise RuntimeError(
            f"Unexpected offline manifest schema: {manifest.get('schema')!r}."
        )
    try:
        profile = normalize_profile(str(manifest.get("profile", "")))
    except ValueError as error:
        raise RuntimeError(str(error)) from error
    manifest = dict(manifest)
    manifest["profile"] = profile
    checkpoint = _torch_load(checkpoint_path, map_location="cpu")
    if checkpoint.get("schema") != OFFLINE_CHECKPOINT_SCHEMA:
        raise RuntimeError(
            f"Unexpected offline checkpoint schema: {checkpoint.get('schema')!r}."
        )
    try:
        checkpoint_profile = normalize_profile(str(checkpoint.get("profile", "")))
    except ValueError as error:
        raise RuntimeError(str(error)) from error
    if checkpoint_profile != profile:
        raise RuntimeError("Offline manifest and checkpoint profiles differ.")
    if checkpoint.get("selection_head") != "trajectory":
        raise RuntimeError(
            "Online CGCD requires the validation-selected trajectory checkpoint."
        )
    if checkpoint.get("test_metrics_used_for_selection") is not False:
        raise RuntimeError("Offline checkpoint was not selected exclusively by validation.")
    if checkpoint.get("old_classes_physical") != manifest.get("old_classes_physical"):
        raise RuntimeError("Offline checkpoint and manifest old-class orders differ.")
    initialization = manifest.get("encoder_initialization")
    if not isinstance(initialization, Mapping) or initialization.get("mode") not in {
        "random", "warmstart"
    }:
        raise RuntimeError(
            "Offline manifest must explicitly record encoder initialization; "
            "the online stage is not allowed to infer or choose it."
        )
    checkpoint_initialization = checkpoint.get("encoder_initialization")
    if checkpoint_initialization == "checkpoint":
        checkpoint_initialization = "warmstart"
    if checkpoint_initialization != initialization.get("mode"):
        raise RuntimeError(
            "Offline checkpoint and manifest encoder-initialization modes differ."
        )

    config = _architecture_from_manifest(manifest, checkpoint)
    model = MotionPrimitiveCGCDModel(config)
    state = checkpoint.get("model")
    if not isinstance(state, Mapping):
        raise RuntimeError("Offline checkpoint has no model state dictionary.")
    incompatible = model.load_state_dict(state, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(f"Offline state is incompatible: {incompatible}.")
    if int(model.class_count) != int(config.old_class_count):
        raise RuntimeError("Reconstructed offline classifier has the wrong class count.")
    return model.to(device), manifest, checkpoint, config


def reconstruct_datasets(
    manifest: Mapping[str, Any]
) -> tuple[Mapping[str, Any], np.ndarray, Namespace]:
    """Rebuild and identity-check the exact offline/online USC-HAD split."""

    npz_path = _resolve_npz_from_manifest(manifest)
    raw_arguments = manifest.get("arguments")
    if not isinstance(raw_arguments, Mapping):
        raise RuntimeError("Offline manifest lacks argument metadata.")
    args = Namespace(**dict(raw_arguments))
    args.uschad_npz_path = str(npz_path)
    old_classes = manifest.get("old_classes_physical")
    if not isinstance(old_classes, list) or not old_classes:
        raise RuntimeError("Offline manifest lacks a non-empty old-class order.")
    args.old_classes_parsed = [int(value) for value in old_classes]
    if int(getattr(args, "seed")) < 0:
        raise RuntimeError("Offline seed must be non-negative.")

    datasets, novel_order = build_datasets(args)
    expected_novel = [int(value) for value in manifest.get("novel_class_order_physical", [])]
    observed_novel = [int(value) for value in np.asarray(novel_order).tolist()]
    if observed_novel != expected_novel:
        raise RuntimeError(
            "Rebuilt novel-class order differs from the offline manifest: "
            f"expected={expected_novel}, observed={observed_novel}."
        )

    expected_summaries = manifest.get("datasets")
    if not isinstance(expected_summaries, Mapping):
        raise RuntimeError("Offline manifest lacks dataset identity summaries.")
    key_map = {
        "train": "offline_train_dataset",
        "validation": "offline_val_dataset",
        "test": "offline_test_dataset",
    }
    for manifest_key, dataset_key in key_map.items():
        observed = _dataset_summary(datasets[dataset_key])
        expected = expected_summaries.get(manifest_key)
        if observed != expected:
            raise RuntimeError(
                f"Rebuilt {manifest_key} dataset identity differs from the offline manifest."
            )
    return datasets, novel_order, args


def configure_logger(output_dir: Path) -> logging.Logger:
    logger = logging.getLogger(f"hhr.motion_online_cli.{Path(output_dir).resolve()}")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(formatter)
    file_handler = logging.FileHandler(Path(output_dir) / "online.log", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(stream)
    logger.addHandler(file_handler)
    return logger


def build_online_config(
    args: argparse.Namespace,
    manifest: Mapping[str, Any],
    architecture: MotionPrimitiveConfig,
) -> MotionPrimitiveOnlineConfig:
    profile = normalize_profile(str(manifest["profile"]))
    if profile != PROFILE_JOINT:
        raise ValueError("HHR online accepts only motion_primitive_joint.")
    offline_arguments = manifest["arguments"]
    return MotionPrimitiveOnlineConfig(
        old_class_count=int(architecture.old_class_count),
        total_class_count=int(offline_arguments["total_classes"]),
        novel_classes_per_session=int(offline_arguments["novel_classes_per_session"]),
        online_old_trials_per_class=int(offline_arguments["online_old_trials"]),
        online_new_first_trials_per_class=int(offline_arguments["online_novel_unseen_trials"]),
        online_new_seen_trials_per_class=int(offline_arguments["online_novel_seen_trials"]),
        epochs_per_session=int(args.epochs_per_session),
        batch_size=int(args.batch_size),
        evaluation_batch_size=int(args.eval_batch_size),
        num_workers=int(args.num_workers),
        evaluation_num_workers=int(args.eval_num_workers),
        learning_rate=float(args.lr),
        encoder_lr_scale=float(args.encoder_lr_scale),
        momentum=float(args.momentum),
        weight_decay=float(args.weight_decay),
        cosine_minimum_ratio=float(args.cosine_minimum_ratio),
        student_temperature=float(args.student_temperature),
        warmup_teacher_temperature=float(args.warmup_teacher_temperature),
        teacher_temperature=float(args.teacher_temperature),
        warmup_teacher_epochs=int(args.warmup_teacher_epochs),
        grouped_memax_old_new_weight=float(args.memax_old_new_weight),
        grouped_memax_old_in_weight=float(args.memax_old_in_weight),
        grouped_memax_new_in_weight=float(args.memax_new_in_weight),
        initialize_new_trajectory_head_with_kmeans=bool(
            args.initialize_new_trajectory_head_with_kmeans
        ),
        kmeans_random_state=int(args.kmeans_random_state),
        trajectory_cluster_weight=float(args.trajectory_cluster_weight),
        trajectory_logit_distillation_weight=float(
            args.trajectory_logit_distillation_weight
        ),
        trajectory_feature_distillation_weight=float(
            args.trajectory_feature_distillation_weight
        ),
        primitive_feature_distillation_weight=float(
            args.primitive_feature_distillation_weight
        ),
        old_codebook_anchor_weight=float(args.old_codebook_anchor_weight),
        trajectory_view_consistency_weight=float(
            args.trajectory_view_consistency_weight
        ),
        vq_commitment_weight=float(args.vq_commitment_weight),
        vq_codebook_weight=float(args.vq_codebook_weight),
    ).validated()


def build_codebook_expansion_policy(
    args: argparse.Namespace,
    profile: str,
) -> Optional[Callable[..., CodebookExpansionDecision]]:
    if args.codebook_expansion == "none":
        return None
    if profile != PROFILE_JOINT:
        raise ValueError(
            f"{args.codebook_expansion} codebook expansion is joint-profile only."
        )
    random_state = int(args.codebook_kmeans_random_state)

    if args.codebook_expansion == "fixed_delta":
        delta = int(args.codebook_fixed_delta)
        if delta < 1:
            raise ValueError("--codebook-fixed-delta must be positive.")

        def fixed_delta_policy(session, model, loader, device):
            # The number is registered before seeing data. KMeans only estimates
            # the vectors, so this remains the non-adaptive mechanism ablation.
            centres = select_kmeans_codebook_centres(
                model,
                loader,
                delta,
                device=device,
                random_state=random_state + int(session) - 1,
            )
            before = int(model.codebook_size)
            return CodebookExpansionDecision(
                centres=centres,
                selected_delta=delta,
                audit={
                    "schema": "hhr_fixed_delta_codebook_decision_v1",
                    "policy": "fixed_delta",
                    "activity_labels_used": False,
                    "session": int(session),
                    "selected_delta": delta,
                    "codebook_size_before": before,
                    "codebook_size_after": before + delta,
                    "selection_reason": "registered_fixed_delta_ablation",
                    "candidate_evaluations": [
                        {
                            "delta": delta,
                            "candidate_codebook_size": before + delta,
                            "accepted": True,
                            "reason": "fixed_before_observing_session_data",
                        }
                    ],
                },
            )

        return fixed_delta_policy

    if args.codebook_expansion != "residual_adaptive":
        raise ValueError(f"Unknown codebook expansion policy {args.codebook_expansion!r}.")

    expected_offline_k = int(args.expected_offline_codebook_size)

    def residual_adaptive_policy(session, model, loader, device):
        session_number = int(session)
        observed_k = int(model.codebook_size)
        if session_number == 1 and observed_k != expected_offline_k:
            raise RuntimeError(
                "residual_adaptive requires the registered offline codebook size "
                f"K={expected_offline_k}, observed K={observed_k}."
            )
        if session_number < 1 or observed_k < expected_offline_k:
            raise RuntimeError("Invalid session number or regressed online codebook size.")
        extracted = extract_local_primitive_features(model, loader, device)
        raw = select_residual_adaptive_codebook_centres(
            extracted.features,
            model.codebook.vectors.detach(),
            trial_ids=extracted.trial_ids,
            subject_ids=extracted.subject_ids,
            max_delta=int(args.codebook_adaptive_max_delta),
            residual_quantile=float(args.codebook_residual_quantile),
            minimum_residual_support=int(args.codebook_minimum_residual_support),
            minimum_cluster_support=int(args.codebook_minimum_cluster_support),
            minimum_cluster_trials=int(args.codebook_minimum_cluster_trials),
            minimum_cluster_subjects=int(args.codebook_minimum_cluster_subjects),
            minimum_relative_improvement=float(
                args.codebook_minimum_relative_improvement
            ),
            complexity_penalty=float(args.codebook_complexity_penalty),
            random_state=random_state + session_number - 1,
        )
        audit = dict(raw.audit)
        audit.update(
            {
                "session": session_number,
                "expected_offline_codebook_size": expected_offline_k,
            }
        )
        return CodebookExpansionDecision(raw.centres, raw.selected_delta, audit)

    return residual_adaptive_policy


def _balanced_accuracy(confusion: np.ndarray) -> float:
    support = confusion.sum(axis=1)
    recalls = np.divide(
        np.diag(confusion),
        support,
        out=np.zeros_like(support, dtype=np.float64),
        where=support > 0,
    )
    return float(recalls.mean())


def session_metric_row(
    record: Mapping[str, Any],
    *,
    report_head: str,
    profile: str,
    fold: int,
    seed: int,
    manifest: Mapping[str, Any],
    offline_checkpoint_payload: Mapping[str, Any],
    offline_run_dir: Path,
    offline_checkpoint: Path,
    offline_checkpoint_sha256: str,
    metrics_path: Path,
    codebook_policy: str,
) -> dict[str, Any]:
    metrics_container = record.get("metrics")
    if not isinstance(metrics_container, Mapping):
        raise RuntimeError("Online session record lacks metrics.")
    if report_head != "trajectory":
        raise RuntimeError("The pure online runner reports only trajectory predictions.")
    metrics = metrics_container.get("trajectory")
    if not isinstance(metrics, Mapping):
        raise RuntimeError("Online session lacks trajectory metrics.")
    evaluation_layers = metrics.get("evaluation_layers")
    required_evaluation_layers = (
        "standard_global_hungarian",
        "constrained_old_fixed",
        "direct_head",
    )
    if not isinstance(evaluation_layers, Mapping) or any(
        name not in evaluation_layers for name in required_evaluation_layers
    ):
        raise RuntimeError(
            "Online session lacks the required stratified evaluation layers: "
            f"{required_evaluation_layers}."
        )
    if metrics.get("prediction_alignment") != "constrained_old_fixed":
        raise RuntimeError(
            "The headline online metrics must use constrained_old_fixed alignment."
        )
    confusion = np.asarray(metrics["confusion_matrix"], dtype=np.int64)
    class_count = int(record["class_count_after_expansion"])
    if confusion.shape != (class_count, class_count):
        raise RuntimeError("Online confusion matrix has the wrong shape.")
    support = confusion.sum(axis=1)
    predicted = confusion.sum(axis=0)
    diagonal = np.diag(confusion)
    recall = np.divide(diagonal, support, out=np.zeros(class_count), where=support > 0)
    precision = np.divide(
        diagonal, predicted, out=np.zeros(class_count), where=predicted > 0
    )
    f1 = np.divide(
        2.0 * precision * recall,
        precision + recall,
        out=np.zeros(class_count),
        where=(precision + recall) > 0,
    )

    architecture = manifest["architecture"]
    offline_arguments = manifest["arguments"]
    online = manifest.get("online_config")
    if not isinstance(online, Mapping):
        raise RuntimeError("Online manifest data lacks the registered online_config.")
    subjects = manifest["subjects"]
    class_order = [int(value) for value in manifest["old_classes_physical"]]
    class_order.extend(int(value) for value in manifest["novel_class_order_physical"])
    offline_selection = (
        f"best_{offline_checkpoint_payload.get('selection_head')}_validation_"
        f"{offline_checkpoint_payload.get('selection_metric')}"
    )
    raw_codebook_decision = record.get("codebook_expansion_decision")
    codebook_decision = (
        dict(raw_codebook_decision)
        if isinstance(raw_codebook_decision, Mapping)
        else {}
    )
    selected_delta = codebook_decision.get("selected_delta")
    if selected_delta is None:
        before = record.get("codebook_size_before")
        after = record.get("codebook_size_after")
        selected_delta = (
            int(after) - int(before) if before is not None and after is not None else 0
        )
    row: dict[str, Any] = {
        "schema": ONLINE_ROWS_SCHEMA,
        "profile": profile,
        "report_head": report_head,
        "primary_evaluation_layer": "constrained_old_fixed",
        "fold": int(fold),
        "seed": int(seed),
        "session": int(record["session"]),
        "train_subjects": _comma(subjects["train"]),
        "validation_subjects": _comma(subjects["validation"]),
        "outer_test_subjects": _comma(subjects["test"]),
        "npz_path": str(Path(manifest["npz_path"]).resolve()),
        "npz_sha256": manifest["npz_sha256"],
        "window_size": int(architecture["window_size"]),
        "window_stride": int(architecture["window_stride"]),
        "sample_unit": "complete_trial_as_ordered_primitive_trajectory",
        "representation": "variable_length_motion_primitive_trajectory",
        "view_mode": offline_arguments["trial_view_mode"],
        "normalization_eps": float(offline_arguments["uschad_norm_eps"]),
        "trial_crop_ratio": float(offline_arguments["trial_crop_ratio"]),
        "trial_min_windows": int(offline_arguments["trial_min_windows"]),
        "har_aug_mode": str(offline_arguments["har_aug_mode"]),
        "har_weak_jitter_std": float(offline_arguments["har_weak_jitter_std"]),
        "har_weak_scale_std": float(offline_arguments["har_weak_scale_std"]),
        "har_strong_jitter_std": float(offline_arguments["har_strong_jitter_std"]),
        "har_strong_scale_std": float(offline_arguments["har_strong_scale_std"]),
        "har_time_mask_ratio": float(offline_arguments["har_time_mask_ratio"]),
        "online_old_seen_trials": int(offline_arguments["online_old_trials"]),
        "online_novel_unseen_trials": int(
            offline_arguments["online_novel_unseen_trials"]
        ),
        "online_novel_seen_trials": int(offline_arguments["online_novel_seen_trials"]),
        "online_epochs_per_session": int(online["epochs_per_session"]),
        "online_batch_size": int(online["batch_size"]),
        "online_eval_batch_size": int(online["evaluation_batch_size"]),
        "online_num_workers": int(online["num_workers"]),
        "online_eval_num_workers": int(online["evaluation_num_workers"]),
        "online_learning_rate": float(online["learning_rate"]),
        "online_encoder_lr_scale": float(online["encoder_lr_scale"]),
        "online_momentum": float(online["momentum"]),
        "online_weight_decay": float(online["weight_decay"]),
        "online_cosine_minimum_ratio": float(online["cosine_minimum_ratio"]),
        "online_n_views": int(online["n_views"]),
        "online_student_temperature": float(online["student_temperature"]),
        "online_warmup_teacher_temperature": float(
            online["warmup_teacher_temperature"]
        ),
        "online_teacher_temperature": float(online["teacher_temperature"]),
        "online_warmup_teacher_epochs": int(online["warmup_teacher_epochs"]),
        "online_memax_old_new_weight": float(
            online["grouped_memax_old_new_weight"]
        ),
        "online_memax_old_in_weight": float(online["grouped_memax_old_in_weight"]),
        "online_memax_new_in_weight": float(online["grouped_memax_new_in_weight"]),
        "online_initialize_new_trajectory_head_with_kmeans": bool(
            online.get("initialize_new_trajectory_head_with_kmeans", False)
        ),
        "online_kmeans_random_state": int(online["kmeans_random_state"]),
        "online_trajectory_cluster_weight": float(
            online["trajectory_cluster_weight"]
        ),
        "online_trajectory_logit_distillation_weight": float(
            online["trajectory_logit_distillation_weight"]
        ),
        "online_trajectory_feature_distillation_weight": float(
            online["trajectory_feature_distillation_weight"]
        ),
        "online_primitive_feature_distillation_weight": float(
            online["primitive_feature_distillation_weight"]
        ),
        "online_old_codebook_anchor_weight": float(
            online["old_codebook_anchor_weight"]
        ),
        "online_trajectory_view_consistency_weight": float(
            online["trajectory_view_consistency_weight"]
        ),
        "online_vq_commitment_weight": float(online["vq_commitment_weight"]),
        "online_vq_codebook_weight": float(online["vq_codebook_weight"]),
        "checkpoint_selection": "final_epoch",
        "selection_epoch": int(manifest["online_config"]["epochs_per_session"])
        if "online_config" in manifest
        else "",
        "offline_checkpoint_selection": offline_selection,
        "offline_run_dir": str(Path(offline_run_dir).resolve()),
        "offline_checkpoint": str(Path(offline_checkpoint).resolve()),
        "offline_checkpoint_sha256": offline_checkpoint_sha256,
        "num_classes": class_count,
        "num_samples": int(metrics["sample_count"]),
        "labels": json.dumps(list(range(class_count)), separators=(",", ":")),
        "physical_labels": json.dumps(class_order[:class_count], separators=(",", ":")),
        "support": json.dumps(support.tolist(), separators=(",", ":")),
        "confusion_matrix": json.dumps(confusion.tolist(), separators=(",", ":")),
        "metrics_path": str(Path(metrics_path).resolve()),
        "overall_accuracy": float(metrics["all_accuracy"]),
        "mean_class_accuracy": _balanced_accuracy(confusion),
        "macro_f1": float(metrics["macro_f1"]),
        "gcd_all_accuracy": float(metrics["all_accuracy"]),
        "gcd_old_accuracy": float(metrics["old_accuracy"]),
        "gcd_new_accuracy": float(metrics["new_accuracy"]),
        "h_score": float(metrics["h_score"]),
        "gcd_soft_all_accuracy": float(metrics["all_accuracy"]),
        "gcd_seen_accuracy": float(metrics["seen_accuracy"]),
        "gcd_unseen_accuracy": float(metrics["unseen_accuracy"]),
        "codebook_expansion_policy": codebook_policy,
        "codebook_expansion_is_adaptive": codebook_policy == "residual_adaptive",
        "codebook_size_before": record.get("codebook_size_before"),
        "codebook_size_after": record.get("codebook_size_after"),
        "codebook_expanded": bool(record.get("codebook_expanded", False)),
        "codebook_selected_delta": int(selected_delta),
        "codebook_selection_reason": codebook_decision.get("selection_reason", ""),
        "codebook_residual_threshold": codebook_decision.get(
            "residual_threshold", ""
        ),
        "codebook_residual_support": codebook_decision.get("residual_support", ""),
        "codebook_residual_distinct_trial_support": codebook_decision.get(
            "residual_distinct_trial_support", ""
        ),
        "codebook_residual_distinct_subject_support": codebook_decision.get(
            "residual_distinct_subject_support", ""
        ),
        "codebook_minimum_cluster_trials": codebook_decision.get(
            "minimum_cluster_trials", ""
        ),
        "codebook_minimum_cluster_subjects": codebook_decision.get(
            "minimum_cluster_subjects", ""
        ),
        "codebook_baseline_mean_cosine_residual": codebook_decision.get(
            "baseline_mean_cosine_residual", ""
        ),
        "codebook_candidate_evaluations": json.dumps(
            codebook_decision.get("candidate_evaluations", []),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ),
        "codebook_expansion_decision": json.dumps(
            codebook_decision,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ),
        "new_class_initialization": json.dumps(
            record.get("new_class_initialization", {}),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ),
        "optimizer_steps": int(record["optimizer_steps"]),
        "backward_calls": int(record["backward_calls"]),
    }
    for layer_name in required_evaluation_layers:
        layer = evaluation_layers[layer_name]
        if not isinstance(layer, Mapping):
            raise RuntimeError(f"Evaluation layer {layer_name!r} is not a mapping.")
        layer_confusion = np.asarray(layer.get("confusion_matrix"), dtype=np.int64)
        if layer_confusion.shape != (class_count, class_count):
            raise RuntimeError(
                f"Evaluation layer {layer_name!r} has the wrong confusion shape."
            )
        prefix = f"eval_{layer_name}"
        row[f"{prefix}_all_accuracy"] = float(layer["all_accuracy"])
        row[f"{prefix}_old_accuracy"] = float(layer["old_accuracy"])
        row[f"{prefix}_new_accuracy"] = float(layer["new_accuracy"])
        row[f"{prefix}_h_score"] = float(layer["h_score"])
        row[f"{prefix}_seen_accuracy"] = float(layer["seen_accuracy"])
        row[f"{prefix}_unseen_accuracy"] = float(layer["unseen_accuracy"])
        row[f"{prefix}_macro_f1"] = float(layer["macro_f1"])
        row[f"{prefix}_confusion_matrix"] = json.dumps(
            layer_confusion.tolist(), separators=(",", ":")
        )
        row[f"{prefix}_assignment_pred_to_true"] = json.dumps(
            layer.get("assignment_pred_to_true", []), separators=(",", ":")
        )
        row[f"{prefix}_alignment_uses_test_labels"] = bool(
            layer.get("alignment_uses_test_labels", True)
        )
    for class_index in range(class_count):
        row[f"class_{class_index}_physical_label"] = class_order[class_index]
        row[f"class_{class_index}_accuracy"] = float(recall[class_index])
        row[f"class_{class_index}_precision"] = float(precision[class_index])
        row[f"class_{class_index}_f1"] = float(f1[class_index])
        row[f"class_{class_index}_support"] = int(support[class_index])
    return row


def online_fieldnames(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    preferred = [
        "schema", "profile", "report_head", "primary_evaluation_layer",
        "fold", "seed", "session",
        "train_subjects", "validation_subjects", "outer_test_subjects",
        "npz_path", "npz_sha256", "window_size", "window_stride", "sample_unit",
        "representation", "view_mode", "normalization_eps",
        "trial_crop_ratio", "trial_min_windows", "har_aug_mode",
        "har_weak_jitter_std", "har_weak_scale_std", "har_strong_jitter_std",
        "har_strong_scale_std", "har_time_mask_ratio",
        "online_old_seen_trials", "online_novel_unseen_trials",
        "online_novel_seen_trials", "online_epochs_per_session", "online_batch_size",
        "online_eval_batch_size", "online_num_workers", "online_eval_num_workers",
        "online_learning_rate", "online_encoder_lr_scale", "online_momentum",
        "online_weight_decay", "online_cosine_minimum_ratio", "online_n_views",
        "online_student_temperature", "online_warmup_teacher_temperature",
        "online_teacher_temperature", "online_warmup_teacher_epochs",
        "online_memax_old_new_weight", "online_memax_old_in_weight",
        "online_memax_new_in_weight",
        "online_initialize_new_trajectory_head_with_kmeans",
        "online_kmeans_random_state",
        "online_trajectory_cluster_weight",
        "online_trajectory_logit_distillation_weight",
        "online_trajectory_feature_distillation_weight",
        "online_trajectory_view_consistency_weight",
        "online_vq_commitment_weight", "online_vq_codebook_weight",
        "checkpoint_selection", "selection_epoch",
        "offline_checkpoint_selection", "offline_run_dir", "offline_checkpoint",
        "offline_checkpoint_sha256", "num_classes", "num_samples", "labels",
        "physical_labels", "support", "confusion_matrix", "metrics_path",
        "overall_accuracy", "mean_class_accuracy", "macro_f1",
        "gcd_all_accuracy", "gcd_old_accuracy", "gcd_new_accuracy", "h_score",
        "gcd_soft_all_accuracy", "gcd_seen_accuracy", "gcd_unseen_accuracy",
        "eval_standard_global_hungarian_all_accuracy",
        "eval_standard_global_hungarian_old_accuracy",
        "eval_standard_global_hungarian_new_accuracy",
        "eval_standard_global_hungarian_h_score",
        "eval_standard_global_hungarian_seen_accuracy",
        "eval_standard_global_hungarian_unseen_accuracy",
        "eval_standard_global_hungarian_macro_f1",
        "eval_standard_global_hungarian_confusion_matrix",
        "eval_standard_global_hungarian_assignment_pred_to_true",
        "eval_standard_global_hungarian_alignment_uses_test_labels",
        "eval_constrained_old_fixed_all_accuracy",
        "eval_constrained_old_fixed_old_accuracy",
        "eval_constrained_old_fixed_new_accuracy",
        "eval_constrained_old_fixed_h_score",
        "eval_constrained_old_fixed_seen_accuracy",
        "eval_constrained_old_fixed_unseen_accuracy",
        "eval_constrained_old_fixed_macro_f1",
        "eval_constrained_old_fixed_confusion_matrix",
        "eval_constrained_old_fixed_assignment_pred_to_true",
        "eval_constrained_old_fixed_alignment_uses_test_labels",
        "eval_direct_head_all_accuracy", "eval_direct_head_old_accuracy",
        "eval_direct_head_new_accuracy", "eval_direct_head_h_score",
        "eval_direct_head_seen_accuracy", "eval_direct_head_unseen_accuracy",
        "eval_direct_head_macro_f1", "eval_direct_head_confusion_matrix",
        "eval_direct_head_assignment_pred_to_true",
        "eval_direct_head_alignment_uses_test_labels",
        "codebook_expansion_policy", "codebook_expansion_is_adaptive",
        "codebook_size_before", "codebook_size_after", "codebook_expanded",
        "codebook_selected_delta", "codebook_selection_reason",
        "codebook_residual_threshold", "codebook_residual_support",
        "codebook_baseline_mean_cosine_residual",
        "codebook_candidate_evaluations", "codebook_expansion_decision",
        "new_class_initialization",
        "optimizer_steps", "backward_calls",
    ]
    available = set().union(*(row.keys() for row in rows))
    trailing = sorted(
        available - set(preferred),
        key=lambda name: (
            int(name.split("_")[1]) if name.startswith("class_") else 10**9,
            name,
        ),
    )
    return [name for name in preferred if name in available] + trailing


def validate_args(args: argparse.Namespace) -> argparse.Namespace:
    for name in (
        "epochs_per_session", "batch_size", "eval_batch_size",
    ):
        if int(getattr(args, name)) < 1:
            raise ValueError(f"--{name.replace('_', '-')} must be positive.")
    for name in ("num_workers", "eval_num_workers", "warmup_teacher_epochs"):
        if int(getattr(args, name)) < 0:
            raise ValueError(f"--{name.replace('_', '-')} cannot be negative.")
    if int(args.warmup_teacher_epochs) > int(args.epochs_per_session):
        raise ValueError("--warmup-teacher-epochs cannot exceed online epochs.")
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
    return args


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run one motion-primitive trajectory online CGCD stream",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--offline-run-dir", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="auto")

    # Registered trajectory-only online settings.
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
        "--codebook-expansion", choices=CODEBOOK_POLICIES,
        default="residual_adaptive",
        help="Main route: begin at offline K=32 and add only evidence-supported codes.",
    )
    parser.add_argument("--codebook-fixed-delta", type=int, default=2)
    parser.add_argument("--codebook-kmeans-random-state", type=int, default=0)
    parser.add_argument("--expected-offline-codebook-size", type=int, default=32)
    parser.add_argument("--codebook-adaptive-max-delta", type=int, default=4)
    parser.add_argument("--codebook-residual-quantile", type=float, default=0.90)
    parser.add_argument("--codebook-minimum-residual-support", type=int, default=32)
    parser.add_argument("--codebook-minimum-cluster-support", type=int, default=8)
    parser.add_argument(
        "--codebook-minimum-cluster-trials",
        type=int,
        default=3,
        help="Each proposed new code must be supported by this many distinct trials.",
    )
    parser.add_argument(
        "--codebook-minimum-cluster-subjects",
        type=int,
        default=2,
        help="Each proposed new code must be shared by this many distinct subjects.",
    )
    parser.add_argument(
        "--codebook-minimum-relative-improvement", type=float, default=0.10
    )
    parser.add_argument("--codebook-complexity-penalty", type=float, default=0.01)
    return parser


def run(args: argparse.Namespace) -> dict[str, Any]:
    args = validate_args(args)
    output_dir = Path(args.output_dir).expanduser().resolve()
    if output_dir.exists() and not output_dir.is_dir():
        raise RuntimeError(f"Online output path is not a directory: {output_dir}.")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            f"Refusing to mix or overwrite a non-empty online output directory: {output_dir}."
        )

    device = resolve_device(args.device)
    offline_run_dir = Path(args.offline_run_dir).expanduser().resolve()
    checkpoint_path = Path(args.checkpoint).expanduser().resolve()

    # Reuse the offline seed; online never chooses the encoder initialization.
    preliminary_manifest = _load_json(offline_run_dir / "manifest.json")
    offline_seed = int(preliminary_manifest.get("arguments", {}).get("seed", -1))
    if offline_seed < 0:
        raise RuntimeError("Offline manifest does not record a valid seed.")
    seed_everything(offline_seed, use_cuda=device.type == "cuda")
    model, manifest, checkpoint, architecture = load_offline_model(
        offline_run_dir, checkpoint_path, device=device
    )
    datasets, novel_order, offline_args = reconstruct_datasets(manifest)
    online_config = build_online_config(args, manifest, architecture)
    profile = normalize_profile(str(manifest["profile"]))
    policy = build_codebook_expansion_policy(args, profile)
    report_head = report_head_for_profile(profile)
    fold = int(getattr(offline_args, "uschad_cv_fold", -1))
    if not 1 <= fold <= 7:
        raise RuntimeError("Formal online aggregation requires an offline cv_fold in [1,7].")

    checkpoint_hash = _sha256_file(checkpoint_path)
    manifest_path = offline_run_dir / "manifest.json"
    identity: dict[str, Any] = {
        "schema": ONLINE_MANIFEST_SCHEMA,
        "profile": profile,
        "report_head": report_head,
        "fold": fold,
        "seed": offline_seed,
        "offline_run_dir": str(offline_run_dir),
        "offline_manifest": str(manifest_path.resolve()),
        "offline_manifest_sha256": _sha256_file(manifest_path),
        "offline_checkpoint": str(checkpoint_path),
        "offline_checkpoint_sha256": checkpoint_hash,
        "offline_checkpoint_selection_head": checkpoint.get("selection_head"),
        "offline_checkpoint_selection_metric": checkpoint.get("selection_metric"),
        "npz_path": str(Path(manifest["npz_path"]).resolve()),
        "npz_sha256": manifest["npz_sha256"],
        "subjects": manifest["subjects"],
        "old_classes_physical": manifest["old_classes_physical"],
        "novel_class_order_physical": manifest["novel_class_order_physical"],
        "architecture": architecture.audit_dict(),
        "offline_encoder_initialization": manifest["encoder_initialization"],
        "online_config": online_config.audit_dict(),
        "online_representation": {
            "primary_head": report_head,
            "primary_input": "variable_length_motion_primitive_trajectory",
            "primitive_identity": "fixed_dimensional_codebook_embedding",
            "trajectory_input_is_codebook_size_invariant": True,
            "complete_trial_pooling": False,
            "pooled_or_fused_prediction": False,
        },
        "codebook_expansion": {
            "policy": args.codebook_expansion,
            "fixed_delta_per_session_ablation": (
                int(args.codebook_fixed_delta)
                if args.codebook_expansion == "fixed_delta"
                else 0
            ),
            "kmeans_random_state": int(args.codebook_kmeans_random_state),
            "adaptive": args.codebook_expansion == "residual_adaptive",
            "activity_labels_used_for_decision": False,
            "expected_offline_codebook_size": int(
                args.expected_offline_codebook_size
            ),
            "adaptive_max_delta": int(args.codebook_adaptive_max_delta),
            "residual_quantile": float(args.codebook_residual_quantile),
            "minimum_residual_support": int(
                args.codebook_minimum_residual_support
            ),
            "minimum_cluster_support": int(
                args.codebook_minimum_cluster_support
            ),
            "minimum_cluster_trials": int(args.codebook_minimum_cluster_trials),
            "minimum_cluster_subjects": int(args.codebook_minimum_cluster_subjects),
            "minimum_relative_improvement": float(
                args.codebook_minimum_relative_improvement
            ),
            "complexity_penalty": float(args.codebook_complexity_penalty),
            "interpretation": {
                "none": "fixed_K_no_expansion_control",
                "fixed_delta": "registered_fixed_delta_mechanism_ablation",
                "residual_adaptive": "unlabeled_residual_adaptive_expansion",
            }[args.codebook_expansion],
        },
        "label_contract": {
            "activity_labels_available_for_protocol_audit": True,
            "activity_labels_enter_online_loss": False,
            "activity_labels_enter_codebook_expansion_decision": False,
            "test_labels_enter_hyperparameter_or_checkpoint_selection": False,
            "test_evaluations_per_session": 1,
        },
        "arguments": {
            key: value for key, value in vars(args).items()
            if key not in {"output_dir"}
        },
        "source_sha256": {
            str(Path(__file__).resolve().relative_to(PROJECT_ROOT)): _sha256_file(Path(__file__)),
            "experiments/motion_primitive/motion_online.py": _sha256_file(
                Path(__file__).with_name("motion_online.py")
            ),
            "experiments/motion_primitive/trajectory_distillation.py": _sha256_file(
                Path(__file__).with_name("trajectory_distillation.py")
            ),
            "models/motion_primitive_cgcd.py": _sha256_file(
                PROJECT_ROOT / "models" / "motion_primitive_cgcd.py"
            ),
        },
    }
    identity["identity_sha256"] = _canonical_hash(identity)
    run_manifest = dict(identity)
    run_manifest["created_utc"] = datetime.now(timezone.utc).isoformat()

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(output_dir / "manifest.json", run_manifest)
    logger = configure_logger(output_dir)
    logger.info(
        "online profile=%s fold=%d seed=%d checkpoint=%s expansion=%s",
        profile,
        fold,
        offline_seed,
        checkpoint_path,
        args.codebook_expansion,
    )
    _, records = run_online_cgcd(
        model,
        datasets,
        novel_order,
        manifest["old_classes_physical"],
        online_config,
        device=device,
        seed=offline_seed,
        output_dir=output_dir,
        logger=logger,
        codebook_expansion_policy=policy,
    )
    if len(records) != online_config.session_count:
        raise RuntimeError("Online API returned an incomplete session sequence.")
    rows = [
        session_metric_row(
            record,
            report_head=report_head,
            profile=profile,
            fold=fold,
            seed=offline_seed,
            manifest={**manifest, "online_config": online_config.audit_dict()},
            offline_checkpoint_payload=checkpoint,
            offline_run_dir=offline_run_dir,
            offline_checkpoint=checkpoint_path,
            offline_checkpoint_sha256=checkpoint_hash,
            metrics_path=output_dir / f"metrics_session_{int(record['session'])}.json",
            codebook_policy=args.codebook_expansion,
        )
        for record in records
    ]
    _write_csv(output_dir / "online_runs.csv", rows, online_fieldnames(rows))
    logger.info(
        "completed profile=%s fold=%d seed=%d session3 all=%.4f old=%.4f new=%.4f H=%.4f",
        profile,
        fold,
        offline_seed,
        rows[-1]["gcd_all_accuracy"],
        rows[-1]["gcd_old_accuracy"],
        rows[-1]["gcd_new_accuracy"],
        rows[-1]["h_score"],
    )
    return {
        "manifest": str((output_dir / "manifest.json").resolve()),
        "online_runs": str((output_dir / "online_runs.csv").resolve()),
        "profile": profile,
        "fold": fold,
        "seed": offline_seed,
        "session_count": len(rows),
    }


def main(argv: Optional[Sequence[str]] = None) -> dict[str, Any]:
    return run(build_parser().parse_args(argv))


if __name__ == "__main__":
    main()
