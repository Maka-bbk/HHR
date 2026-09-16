"""Offline builder for the frozen A2 + E0 + state + K32 route."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import sys
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
from sklearn.metrics import confusion_matrix, f1_score

from experiments.motion_primitive.frozen_e0_state import (
    PrimitiveTrial,
    codebook_usage,
    descriptor_matrix,
    fit_descriptor_transform,
    fit_e0_codebook,
    tokenize_e0_trial,
)
from experiments.motion_primitive.strict_artifacts import (
    encode_sensor_trials,
    load_frozen_a2_encoder,
    representation_sha256,
    save_descriptor_transform,
    save_e0_codebook,
    write_csv,
    write_json,
)
from experiments.motion_primitive.strict_protocol import (
    OLD_CLASSES,
    SensorTrial,
    build_registered_protocol,
    sha256_file,
)
from experiments.motion_primitive.strict_registry import (
    LabelFreeTrial,
    OldClassReference,
    StrictRegistryConfig,
    fit_old_registry,
    route_label_free_trials,
    serialize_registry_state,
)


LOGGER = logging.getLogger("hhr_strict_offline")
OFFLINE_SCHEMA = "hhr_frozen_a2_e0_state_offline_v1"
PROFILE = "frozen_a2_e0_state_k32"
RUN_IDENTITY_SCHEMA = "hhr_frozen_a2_e0_state_offline_identity_v1"


_OUTPUT_ARTIFACTS = (
    "codebook_usage.json",
    "e0_codebook.json",
    "e0_codebook.npz",
    "metrics_outer_test.json",
    "metrics_validation.json",
    "old_registry.json",
    "outer_test_predictions.csv",
    "raw_predictions_outer_test.npz",
    "raw_predictions_validation.npz",
    "state_descriptor_transform.json",
    "state_descriptor_transform.npz",
    "trajectories_label_free.jsonl",
    "trajectories_outer_test_scored.jsonl",
)


def _read_json_object(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"Cannot read JSON artifact {path}.") from error
    if not isinstance(payload, dict):
        raise RuntimeError(f"JSON artifact {path} is not an object.")
    return payload


def _canonical_hash(payload: Mapping[str, Any]) -> str:
    text = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _implementation_hashes() -> dict[str, str]:
    relative = (
        "models/resnet1d.py",
        "experiments/motion_primitive/motion_encoder.py",
        "experiments/motion_primitive/motion_checkpoint.py",
        "experiments/motion_primitive/frozen_e0_state.py",
        "experiments/motion_primitive/strict_artifacts.py",
        "experiments/motion_primitive/strict_protocol.py",
        "experiments/motion_primitive/strict_registry.py",
        "experiments/motion_primitive/strict_offline_runner.py",
    )
    return {name: sha256_file(PROJECT_ROOT / name) for name in relative}


def _run_identity(args: argparse.Namespace) -> dict[str, Any]:
    npz_path = Path(args.npz_path).expanduser().resolve()
    checkpoint_path = Path(args.a2_checkpoint).expanduser().resolve()
    if not npz_path.is_file():
        raise FileNotFoundError(npz_path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    return {
        "schema": RUN_IDENTITY_SCHEMA,
        "profile": PROFILE,
        "fold": int(args.fold),
        "seed": int(args.seed),
        "npz_path": str(npz_path),
        "npz_sha256": sha256_file(npz_path),
        "a2_checkpoint_path": str(checkpoint_path),
        "a2_checkpoint_sha256": sha256_file(checkpoint_path),
        "route": {
            "encoder": "A2_final_frozen_content",
            "segmentation": "E0_fixed_window_w256_s128",
            "codebook": "trial_equal_PCA64_KMeans32_cosine_hard",
            "descriptor": "state_3739_train_only_PCA32_L2",
        },
        "fixed_parameters": {
            "window_size": int(args.window_size),
            "window_stride": int(args.window_stride),
            "primitive_num": int(args.primitive_num),
            "pca_dim": int(args.pca_dim),
            "descriptor_pca_dim": int(args.descriptor_pca_dim),
            "include_state": bool(args.include_state),
            "shuffle_novel_classes": bool(args.shuffle_novel_classes),
        },
        "registry_config": {
            key: getattr(args, key)
            for key in (
                "old_distance_alpha",
                "old_ratio_alpha",
                "novel_distance_alpha",
                "minimum_cluster_trials",
                "minimum_cluster_subjects",
                "minimum_cluster_silhouette",
                "bootstrap_replicates",
                "minimum_bootstrap_stability",
                "minimum_registry_separation",
                "minimum_candidate_separation",
            )
        },
        "runtime": {
            "encode_batch_size": int(args.encode_batch_size),
            "requested_device": str(args.device),
            "resolved_device": str(_resolve_device(args.device)),
            "allow_smoke_a2": bool(args.allow_smoke_a2),
        },
        "implementation_sha256": _implementation_hashes(),
    }


def _prepare_run_identity(output: Path, args: argparse.Namespace) -> dict[str, Any]:
    identity = _run_identity(args)
    expected = {**identity, "identity_sha256": _canonical_hash(identity)}
    path = output / "run_identity.json"
    if path.is_file():
        observed = _read_json_object(path)
        if observed != expected:
            raise RuntimeError(
                f"Offline output {output} records another run identity; use a new directory."
            )
        return expected
    if output.exists():
        allowed = {"cv_member_manifest.json"}
        unidentified = [item.name for item in output.iterdir() if item.name not in allowed]
        if unidentified:
            raise RuntimeError(
                f"Offline output {output} has artifacts but no run_identity.json; "
                "refusing to adopt it."
            )
    output.mkdir(parents=True, exist_ok=True)
    write_json(path, expected)
    return expected


def validate_completed_output(
    output: Path,
    *,
    expected_identity: Mapping[str, Any],
) -> dict[str, Any]:
    identity_path = output / "run_identity.json"
    complete_path = output / "complete.json"
    manifest_path = output / "manifest.json"
    if not identity_path.is_file() or not complete_path.is_file() or not manifest_path.is_file():
        raise RuntimeError(f"Offline completed member is incomplete: {output}.")
    if _read_json_object(identity_path) != dict(expected_identity):
        raise RuntimeError("Offline run identity changed after fitting.")
    complete = _read_json_object(complete_path)
    manifest = _read_json_object(manifest_path)
    if complete.get("schema") != OFFLINE_SCHEMA or manifest.get("schema") != OFFLINE_SCHEMA:
        raise RuntimeError("Offline completed member has another schema.")
    if complete.get("profile") != PROFILE or manifest.get("profile") != PROFILE:
        raise RuntimeError("Offline completed member has another profile.")
    if complete.get("complete") is not True:
        raise RuntimeError("Offline completion marker is false.")
    for key in ("fold", "seed"):
        if int(complete.get(key, -1)) != int(expected_identity[key]):
            raise RuntimeError(f"Offline completion {key} differs from run identity.")
        if int(manifest.get(key, -1)) != int(expected_identity[key]):
            raise RuntimeError(f"Offline manifest {key} differs from run identity.")
    if complete.get("run_identity_sha256") != expected_identity["identity_sha256"]:
        raise RuntimeError("Offline completion is not bound to the current run identity.")
    if manifest.get("run_identity_sha256") != expected_identity["identity_sha256"]:
        raise RuntimeError("Offline manifest is not bound to the current run identity.")
    if complete.get("manifest_sha256") != sha256_file(manifest_path):
        raise RuntimeError("Offline manifest SHA256 mismatch.")
    artifact_hashes = complete.get("artifact_sha256")
    if not isinstance(artifact_hashes, Mapping) or set(artifact_hashes) != set(_OUTPUT_ARTIFACTS):
        raise RuntimeError("Offline completion has an incomplete artifact hash inventory.")
    if manifest.get("artifact_sha256") != artifact_hashes:
        raise RuntimeError("Offline manifest/completion artifact inventories differ.")
    for name, expected_hash in artifact_hashes.items():
        path = output / str(name)
        if not path.is_file() or sha256_file(path) != expected_hash:
            raise RuntimeError(f"Offline artifact SHA256 mismatch: {name}.")
    if manifest.get("npz_sha256") != expected_identity["npz_sha256"]:
        raise RuntimeError("Offline manifest NPZ SHA256 differs from run identity.")
    encoder = manifest.get("a2_encoder")
    if not isinstance(encoder, Mapping):
        raise RuntimeError("Offline manifest lacks A2 encoder audit.")
    if encoder.get("checkpoint_sha256") != expected_identity["a2_checkpoint_sha256"]:
        raise RuntimeError("Offline manifest A2 checkpoint SHA256 differs from run identity.")
    if complete.get("representation_sha256") != manifest.get("representation_sha256"):
        raise RuntimeError("Offline representation SHA256 differs between artifacts.")
    old_registry = manifest.get("old_registry")
    if not isinstance(old_registry, Mapping):
        raise RuntimeError("Offline manifest lacks old registry audit.")
    if complete.get("old_registry_state_sha256") != old_registry.get("state_sha256"):
        raise RuntimeError("Offline registry state SHA256 differs between artifacts.")
    validation = _read_json_object(output / "metrics_validation.json")
    outer = _read_json_object(output / "metrics_outer_test.json")
    if complete.get("validation") != validation.get("metrics"):
        raise RuntimeError("Offline validation metrics differ between artifacts.")
    if complete.get("outer_test") != outer.get("metrics"):
        raise RuntimeError("Offline outer-test metrics differ between artifacts.")
    return complete


def _resolve_device(value: str) -> torch.device:
    if str(value).lower() == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")
    return device


def _labels(protocol, trials: Sequence[SensorTrial]) -> np.ndarray:
    values, _, _ = protocol.truth.join([item.trial_id for item in trials])
    return values


def _primitive_descriptors(
    encoded_trials,
    codebook,
    transform=None,
) -> tuple[list[PrimitiveTrial], np.ndarray | None]:
    primitive = [tokenize_e0_trial(item, codebook) for item in encoded_trials]
    if transform is None:
        return primitive, None
    return primitive, descriptor_matrix(primitive, transform, include_state=True)


def _old_metrics(labels: np.ndarray, predictions: np.ndarray) -> dict[str, Any]:
    labels = np.asarray(labels, dtype=np.int64)
    predictions = np.asarray(predictions, dtype=np.int64)
    confusion = np.zeros((6, 7), dtype=np.int64)
    for truth, prediction in zip(labels.tolist(), predictions.tolist()):
        column = 6 if int(prediction) < 0 or int(prediction) >= 6 else int(prediction)
        confusion[int(truth), column] += 1
    per_class = np.diag(confusion[:, :6]) / np.maximum(confusion.sum(axis=1), 1)
    return {
        "all_accuracy": float(np.mean(labels == predictions)),
        "old_accuracy": float(np.mean(labels == predictions)),
        "new_accuracy": None,
        "h_score": None,
        "macro_f1": float(f1_score(labels, predictions, labels=list(OLD_CLASSES), average="macro", zero_division=0)),
        "balanced_accuracy": float(per_class.mean()),
        "unknown_fraction": float(np.mean(predictions < 0)),
        "per_class_accuracy": per_class.tolist(),
        "confusion_matrix_with_unknown_column": confusion.tolist(),
        "confusion_columns": list(OLD_CLASSES) + ["unknown"],
        "sample_count": int(len(labels)),
    }


def _raw_prediction_artifact(
    path: Path,
    trials: Sequence[SensorTrial],
    descriptors: np.ndarray,
    predictions: np.ndarray,
) -> str:
    np.savez_compressed(
        path,
        trial_ids=np.asarray([item.trial_id for item in trials], dtype=np.int64),
        subject_ids=np.asarray([item.subject_id for item in trials], dtype=np.int64),
        descriptors=np.asarray(descriptors, dtype=np.float32),
        raw_registry_predictions=np.asarray(predictions, dtype=np.int64),
    )
    return sha256_file(path)


def _trajectory_rows(
    role: str,
    sensor_trials: Sequence[SensorTrial],
    primitive_trials: Sequence[PrimitiveTrial],
    *,
    labels: np.ndarray | None = None,
    names: Sequence[str] | None = None,
) -> list[dict[str, Any]]:
    rows = []
    for index, (sensor, primitive) in enumerate(zip(sensor_trials, primitive_trials)):
        row: dict[str, Any] = {
            "role": role,
            "trial_id": int(sensor.trial_id),
            "subject_id": int(sensor.subject_id),
            "window_count": len(primitive.child_tokens),
            "primitive_occurrence_count": len(primitive.child_tokens),
            "unique_primitive_type_count": len(set(primitive.child_tokens.tolist())),
            "used_primitive_ids": sorted(set(primitive.child_tokens.astype(int).tolist())),
            "primitive_sequence": primitive.child_tokens.astype(int).tolist(),
            "ownership_start_samples": primitive.starts.astype(int).tolist(),
            "ownership_end_samples_exclusive": primitive.ends.astype(int).tolist(),
            "quantization_distances": primitive.child_distances.astype(float).tolist(),
            "original_window_start_samples": sensor.window_starts.astype(int).tolist(),
            "raw_window_channel_means": sensor.raw_windows.mean(axis=2).astype(float).tolist(),
        }
        if labels is not None:
            row["activity_label"] = int(labels[index])
            row["activity_name"] = str(names[index]) if names is not None else ""
        rows.append(row)
    return rows


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")


def validate_args(args: argparse.Namespace) -> argparse.Namespace:
    if str(args.profile) != PROFILE:
        raise ValueError(f"The public route accepts only --profile {PROFILE}.")
    if (int(args.window_size), int(args.window_stride)) != (256, 128):
        raise ValueError("The formal route is pinned to E0 w256/s128; w128/s64 is a separate ablation.")
    if int(args.primitive_num) != 32 or int(args.pca_dim) != 64 or int(args.descriptor_pca_dim) != 32:
        raise ValueError("The formal route is pinned to K32/PCA64/descriptor-PCA32.")
    if not bool(args.include_state):
        raise ValueError("The formal route requires the complete state descriptor.")
    if int(args.seed) < 0:
        raise ValueError("Seed must be non-negative.")
    return args


def run(args: argparse.Namespace) -> dict[str, Any]:
    args = validate_args(args)
    output = Path(args.output_dir).expanduser().resolve()
    complete_path = output / "complete.json"
    if output.exists() and not bool(args.resume):
        raise FileExistsError(f"Output already exists: {output}")
    run_identity = _prepare_run_identity(output, args)
    if complete_path.is_file():
        if not bool(args.resume):
            raise FileExistsError(f"Completed output already exists: {output}")
        return validate_completed_output(
            output,
            expected_identity=run_identity,
        )
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[logging.FileHandler(output / "offline.log", encoding="utf-8"), logging.StreamHandler()],
        force=True,
    )
    device = _resolve_device(args.device)
    protocol = build_registered_protocol(
        args.npz_path,
        fold=int(args.fold),
        seed=int(args.seed),
        window_size=int(args.window_size),
        stride=int(args.window_stride),
        shuffle_novel_classes=bool(args.shuffle_novel_classes),
    )
    encoder, encoder_audit = load_frozen_a2_encoder(
        args.a2_checkpoint,
        expected_npz_sha256=protocol.npz_sha256,
        expected_fold=int(args.fold),
        expected_seed=int(args.seed),
        expected_split=protocol.split,
        expected_window_size=int(args.window_size),
        allow_smoke=bool(args.allow_smoke_a2),
    )
    LOGGER.info("Encoding offline train/validation content rows; boundary head is unused by E0.")
    encoded_train = encode_sensor_trials(
        encoder, protocol.offline_train, device=device, batch_size=int(args.encode_batch_size)
    )
    encoded_validation = encode_sensor_trials(
        encoder, protocol.offline_validation, device=device, batch_size=int(args.encode_batch_size)
    )
    codebook = fit_e0_codebook(encoded_train, seed=int(args.seed), primitive_num=32, pca_dim=64)
    train_primitive, _ = _primitive_descriptors(encoded_train, codebook)
    descriptor, train_features = fit_descriptor_transform(
        train_primitive, maximum_components=32, include_state=True
    )
    validation_primitive, validation_features = _primitive_descriptors(
        encoded_validation, codebook, descriptor
    )
    if train_features is None or validation_features is None:
        raise AssertionError("Descriptor feature construction unexpectedly returned None.")
    representation_hash = representation_sha256(
        encoder_audit["model_state_dict_sha256"], codebook, descriptor
    )
    train_labels = _labels(protocol, protocol.offline_train)
    validation_labels = _labels(protocol, protocol.offline_validation)
    references = []
    for class_id in OLD_CLASSES:
        train_mask = train_labels == class_id
        validation_mask = validation_labels == class_id
        references.append(OldClassReference(
            registry_id=class_id,
            fit_descriptors=train_features[train_mask],
            calibration_descriptors=validation_features[validation_mask],
            fit_trial_ids=tuple(np.asarray([item.trial_id for item in protocol.offline_train])[train_mask].astype(int).tolist()),
            fit_subject_ids=tuple(np.asarray([item.subject_id for item in protocol.offline_train])[train_mask].astype(int).tolist()),
            calibration_trial_ids=tuple(np.asarray([item.trial_id for item in protocol.offline_validation])[validation_mask].astype(int).tolist()),
        ))
    registry_config = StrictRegistryConfig(
        old_distance_alpha=float(args.old_distance_alpha),
        old_ratio_alpha=float(args.old_ratio_alpha),
        novel_distance_alpha=float(args.novel_distance_alpha),
        max_new_classes_per_session=2,
        minimum_cluster_trials=int(args.minimum_cluster_trials),
        minimum_cluster_subjects=int(args.minimum_cluster_subjects),
        minimum_cluster_silhouette=float(args.minimum_cluster_silhouette),
        bootstrap_replicates=int(args.bootstrap_replicates),
        minimum_bootstrap_stability=float(args.minimum_bootstrap_stability),
        minimum_registry_separation=float(args.minimum_registry_separation),
        minimum_candidate_separation=float(args.minimum_candidate_separation),
        random_seed=int(args.seed),
    ).validated()
    registry = fit_old_registry(
        references, representation_sha256=representation_hash, config=registry_config
    )

    validation_records = tuple(LabelFreeTrial(
        trial_id=item.trial_id,
        subject_id=item.subject_id,
        session_id=1,
        descriptor=feature,
    ) for item, feature in zip(protocol.offline_validation, validation_features))
    validation_predictions = np.asarray([
        item.registry_id for item in route_label_free_trials(registry, validation_records)
    ], dtype=np.int64)
    validation_raw_hash = _raw_prediction_artifact(
        output / "raw_predictions_validation.npz",
        protocol.offline_validation,
        validation_features,
        validation_predictions,
    )
    validation_metrics = _old_metrics(validation_labels, validation_predictions)
    write_json(output / "metrics_validation.json", {
        "schema": OFFLINE_SCHEMA,
        "selection_role": "gate_calibration_diagnostic_only",
        "raw_predictions_sha256": validation_raw_hash,
        "metrics": validation_metrics,
    })

    # The outer test is encoded exactly once only after every train/validation
    # fitted object and old semantic threshold has been frozen.
    encoded_outer = encode_sensor_trials(
        encoder, protocol.offline_outer_test, device=device, batch_size=int(args.encode_batch_size)
    )
    outer_primitive, outer_features = _primitive_descriptors(encoded_outer, codebook, descriptor)
    if outer_features is None:
        raise AssertionError("Outer descriptor construction unexpectedly returned None.")
    outer_records = tuple(LabelFreeTrial(
        trial_id=item.trial_id,
        subject_id=item.subject_id,
        session_id=1,
        descriptor=feature,
    ) for item, feature in zip(protocol.offline_outer_test, outer_features))
    outer_predictions = np.asarray([
        item.registry_id for item in route_label_free_trials(registry, outer_records)
    ], dtype=np.int64)
    outer_raw_hash = _raw_prediction_artifact(
        output / "raw_predictions_outer_test.npz",
        protocol.offline_outer_test,
        outer_features,
        outer_predictions,
    )
    outer_labels, outer_names, outer_subjects = protocol.truth.join(
        [item.trial_id for item in protocol.offline_outer_test]
    )
    outer_metrics = _old_metrics(outer_labels, outer_predictions)
    write_json(output / "metrics_outer_test.json", {
        "schema": OFFLINE_SCHEMA,
        "raw_predictions_frozen_before_truth_join": True,
        "raw_predictions_sha256": outer_raw_hash,
        "outer_test_evaluations": 1,
        "metrics": outer_metrics,
    })

    codebook_metadata = save_e0_codebook(output / "e0_codebook.npz", codebook)
    descriptor_metadata = save_descriptor_transform(
        output / "state_descriptor_transform.npz", descriptor
    )
    registry_payload = serialize_registry_state(registry)
    write_json(output / "old_registry.json", registry_payload)
    old_registry_hash = sha256_file(output / "old_registry.json")
    usage = {
        "offline_train": codebook_usage(train_primitive),
        "offline_validation": codebook_usage(validation_primitive),
        "offline_outer_test": codebook_usage(outer_primitive),
    }
    write_json(output / "codebook_usage.json", usage)

    label_free_rows = (
        _trajectory_rows("offline_train", protocol.offline_train, train_primitive)
        + _trajectory_rows("offline_validation", protocol.offline_validation, validation_primitive)
        + _trajectory_rows("offline_outer_test", protocol.offline_outer_test, outer_primitive)
    )
    _write_jsonl(output / "trajectories_label_free.jsonl", label_free_rows)
    scored_rows = _trajectory_rows(
        "offline_outer_test",
        protocol.offline_outer_test,
        outer_primitive,
        labels=outer_labels,
        names=outer_names,
    )
    _write_jsonl(output / "trajectories_outer_test_scored.jsonl", scored_rows)
    write_csv(output / "outer_test_predictions.csv", [
        {
            "trial_id": int(trial.trial_id),
            "subject_id": int(subject),
            "activity_label": int(label),
            "activity_name": str(name),
            "raw_registry_prediction": int(prediction),
            "accepted_known": bool(prediction >= 0),
        }
        for trial, subject, label, name, prediction in zip(
            protocol.offline_outer_test, outer_subjects, outer_labels, outer_names, outer_predictions
        )
    ])

    artifact_hashes = {
        name: sha256_file(output / name)
        for name in _OUTPUT_ARTIFACTS
    }
    manifest = {
        "schema": OFFLINE_SCHEMA,
        "profile": PROFILE,
        "fold": int(args.fold),
        "seed": int(args.seed),
        "run_identity_sha256": run_identity["identity_sha256"],
        "npz_path": protocol.npz_path,
        "npz_sha256": protocol.npz_sha256,
        "split": protocol.split.as_dict(),
        "window_grid": {"size": 256, "stride": 128, "segmentation": "E0_fixed_window"},
        "a2_encoder": encoder_audit,
        "codebook": codebook_metadata,
        "descriptor": descriptor_metadata,
        "representation_sha256": representation_hash,
        "old_registry": {
            "path": "old_registry.json",
            "artifact_sha256": old_registry_hash,
            "state_sha256": registry.state_sha256,
            "old_anchor_sha256": registry.old_anchor_sha256,
        },
        "registry_config": registry_config.audit_dict(),
        "trial_counts": {
            "offline_train": len(protocol.offline_train),
            "offline_validation": len(protocol.offline_validation),
            "offline_outer_test": len(protocol.offline_outer_test),
        },
        "fit_scope_audit": {
            "encoder_fit": "train_subjects_old6; validation selection; outer test unused",
            "e0_codebook_fit": "offline_train_subjects_old6_only",
            "descriptor_transform_fit": "offline_train_subjects_old6_only",
            "old_prototype_fit": "offline_train_subjects_old6_only",
            "old_gate_calibration": "offline_validation_subjects_old6_only",
            "outer_test_used_for_selection": False,
            "outer_test_evaluations": 1,
        },
        "mutable_online_components": ["novel_registry_append", "unknown_buffer"],
        "frozen_online_components": ["A2_encoder", "E0_PCA64", "E0_KMeans32", "state_descriptor_transform", "old_registry_rows"],
        "artifact_sha256": artifact_hashes,
        "arguments": vars(args),
    }
    write_json(output / "manifest.json", manifest)
    manifest_hash = sha256_file(output / "manifest.json")
    complete = {
        "schema": OFFLINE_SCHEMA,
        "profile": PROFILE,
        "fold": int(args.fold),
        "seed": int(args.seed),
        "run_identity_sha256": run_identity["identity_sha256"],
        "manifest_sha256": manifest_hash,
        "artifact_sha256": artifact_hashes,
        "representation_sha256": representation_hash,
        "old_registry_state_sha256": registry.state_sha256,
        "validation": validation_metrics,
        "outer_test": outer_metrics,
        "complete": True,
    }
    write_json(complete_path, complete)
    LOGGER.info(
        "completed profile=%s fold=%d seed=%d offline_old=%.4f used_k=%d",
        PROFILE, int(args.fold), int(args.seed), outer_metrics["old_accuracy"], usage["offline_outer_test"]["used_k"],
    )
    return complete


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build frozen A2/E0/state/K32 offline state.")
    parser.add_argument("--a2-checkpoint", required=True)
    parser.add_argument("--npz-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--profile", default=PROFILE, choices=(PROFILE,))
    parser.add_argument("--fold", type=int, required=True, choices=range(1, 8))
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--window-size", type=int, default=256)
    parser.add_argument("--window-stride", type=int, default=128)
    parser.add_argument("--primitive-num", type=int, default=32)
    parser.add_argument("--pca-dim", type=int, default=64)
    parser.add_argument("--descriptor-pca-dim", type=int, default=32)
    parser.add_argument("--include-state", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--encode-batch-size", type=int, default=512)
    parser.add_argument("--old-distance-alpha", type=float, default=0.05)
    parser.add_argument("--old-ratio-alpha", type=float, default=0.05)
    parser.add_argument("--novel-distance-alpha", type=float, default=0.05)
    parser.add_argument("--minimum-cluster-trials", type=int, default=3)
    parser.add_argument("--minimum-cluster-subjects", type=int, default=2)
    parser.add_argument("--minimum-cluster-silhouette", type=float, default=0.20)
    parser.add_argument("--bootstrap-replicates", type=int, default=100)
    parser.add_argument("--minimum-bootstrap-stability", type=float, default=0.80)
    parser.add_argument("--minimum-registry-separation", type=float, default=0.10)
    parser.add_argument("--minimum-candidate-separation", type=float, default=0.10)
    parser.add_argument("--shuffle-novel-classes", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--allow-smoke-a2", action="store_true")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--resume", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> dict[str, Any]:
    result = run(build_parser().parse_args(argv))
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return result


if __name__ == "__main__":
    main()
