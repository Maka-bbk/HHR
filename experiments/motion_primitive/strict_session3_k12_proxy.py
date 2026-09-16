"""Strict Session-3 K=12 clustering proxy over frozen A2/E0 trajectories.

This is an isolated mechanism probe, not the deployable online registry.  It
recreates the validated legacy trial readout without activity-label access:

1. load one completed strict Offline member and freeze its A2 encoder/E0 K32;
2. build raw E0 ``state`` trajectory descriptors for all three incoming
   sessions (22 + 26 + 30 = 78 unique trials);
3. fit constant filtering, z-score, PCA<=32, L2 and KMeans12 on those 78 rows;
4. transform and predict the disjoint Session-3 evaluation set (42 trials);
5. durably freeze raw cluster IDs before opening scorer-only activity truth;
6. use one global Hungarian assignment for post-hoc clustering metrics only.

The old-class-fitted ``state_descriptor_transform.npz`` is verified as a
source artifact but deliberately never applied: this proxy must refit the
legacy readout transform from the 78 label-free incoming trajectories.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import sklearn
import torch
from sklearn.cluster import KMeans

from experiments.motion_primitive.frozen_e0_state import (
    HISTORICAL_STATE_SCHEMA_SHA256,
    PrimitiveTrial,
    l2_normalize,
    tokenize_e0_trial,
    trajectory_descriptor,
)
from experiments.motion_primitive.motion_checkpoint import motion_state_dict_sha256
from experiments.motion_primitive.strict_artifacts import (
    encode_sensor_trials,
    load_descriptor_transform,
    load_e0_codebook,
    load_frozen_a2_encoder,
    write_csv,
    write_json,
)
from experiments.motion_primitive.strict_cv_common import canonical_hash
from experiments.motion_primitive.strict_metrics import strict_three_layer_metrics
from experiments.motion_primitive.strict_offline_runner import PROFILE
from experiments.motion_primitive.strict_online_runner import _strict_offline_manifest
from experiments.motion_primitive.strict_protocol import (
    SensorTrial,
    build_registered_protocol,
    sha256_file,
)


SCHEMA = "hhr_strict_session3_k12_proxy_v1"
IDENTITY_SCHEMA = "hhr_strict_session3_k12_proxy_identity_v1"
CLUSTER_COUNT = 12
DESCRIPTOR_PCA_DIM = 32
INCOMING_SESSION_COUNTS = (22, 26, 30)
INCOMING_TOTAL = 78
EVALUATION_COUNT = 42
LOGGER = logging.getLogger("hhr_strict_session3_k12_proxy")


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"Cannot read JSON artifact {path}.") from error
    if not isinstance(payload, dict):
        raise RuntimeError(f"JSON artifact {path} is not an object.")
    return payload


def _array_sha256(*arrays: np.ndarray) -> str:
    digest = hashlib.sha256()
    for index, value in enumerate(arrays):
        array = np.ascontiguousarray(np.asarray(value))
        if array.dtype.kind == "f" and not np.all(np.isfinite(array)):
            raise ValueError("Cannot hash a non-finite array.")
        digest.update(str(index).encode("ascii"))
        digest.update(array.dtype.str.encode("ascii"))
        digest.update(str(array.shape).encode("ascii"))
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _ids_sha256(values: Sequence[int]) -> str:
    return _array_sha256(np.asarray(values, dtype="<i8"))


def _atomic_npz(path: Path, **arrays: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    temporary.replace(path)


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False, allow_nan=False) + "\n")
    temporary.replace(path)


def _verified_identity(path: Path) -> dict[str, Any]:
    payload = _read_json(path)
    recorded = payload.get("identity_sha256")
    body = {key: value for key, value in payload.items() if key != "identity_sha256"}
    if not isinstance(recorded, str) or canonical_hash(body) != recorded:
        raise RuntimeError(f"Identity SHA256 does not verify in {path}.")
    return payload


def _resolve_device(value: str) -> torch.device:
    if str(value).lower() == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")
    return device


@dataclass(frozen=True)
class IncomingDescriptorTransform:
    """Train-only constant-filter/z-score/PCA/L2 state."""

    raw_dim: int
    keep_columns: np.ndarray
    mean: np.ndarray
    scale: np.ndarray
    pca_mean: np.ndarray
    pca_components: np.ndarray
    fit_row_count: int
    requested_pca_dim: int

    def validate(self) -> "IncomingDescriptorTransform":
        keep = np.asarray(self.keep_columns, dtype=np.int64)
        mean = np.asarray(self.mean, dtype=np.float64)
        scale = np.asarray(self.scale, dtype=np.float64)
        pca_mean = np.asarray(self.pca_mean, dtype=np.float64)
        components = np.asarray(self.pca_components, dtype=np.float64)
        kept = len(keep)
        if int(self.raw_dim) < 1 or keep.ndim != 1 or not kept:
            raise ValueError("Descriptor transform has no retained columns.")
        if np.any(keep < 0) or np.any(keep >= int(self.raw_dim)) or len(np.unique(keep)) != kept:
            raise ValueError("Descriptor transform retained columns are invalid.")
        if mean.shape != (kept,) or scale.shape != (kept,) or pca_mean.shape != (kept,):
            raise ValueError("Descriptor transform z-score/PCA means differ from retained dimension.")
        maximum = min(int(self.requested_pca_dim), int(self.fit_row_count) - 1, kept)
        if components.shape != (maximum, kept) or maximum < 1:
            raise ValueError("Descriptor PCA shape violates sample/dimension constraints.")
        if np.any(scale <= 0) or not all(
            np.all(np.isfinite(value)) for value in (mean, scale, pca_mean, components)
        ):
            raise ValueError("Descriptor transform contains invalid numeric state.")
        return self

    @property
    def output_dim(self) -> int:
        return int(np.asarray(self.pca_components).shape[0])

    @property
    def state_sha256(self) -> str:
        self.validate()
        metadata = np.asarray(
            [self.raw_dim, self.fit_row_count, self.requested_pca_dim, self.output_dim],
            dtype="<i8",
        )
        return _array_sha256(
            metadata,
            np.asarray(self.keep_columns, dtype="<i8"),
            np.asarray(self.mean, dtype="<f8"),
            np.asarray(self.scale, dtype="<f8"),
            np.asarray(self.pca_mean, dtype="<f8"),
            np.asarray(self.pca_components, dtype="<f8"),
        )

    def transform(self, values: np.ndarray) -> np.ndarray:
        self.validate()
        matrix = np.asarray(values, dtype=np.float64)
        if matrix.ndim != 2 or matrix.shape[1] != int(self.raw_dim):
            raise ValueError("Raw descriptor matrix dimension mismatch.")
        if not np.all(np.isfinite(matrix)):
            raise ValueError("Raw descriptor matrix contains non-finite values.")
        selected = matrix[:, np.asarray(self.keep_columns, dtype=np.int64)]
        standardized = (selected - self.mean) / self.scale
        projected = (standardized - self.pca_mean) @ self.pca_components.T
        return l2_normalize(projected.astype(np.float32))


def fit_incoming_descriptor_transform(
    values: np.ndarray,
    *,
    maximum_components: int = DESCRIPTOR_PCA_DIM,
    constant_tolerance: float = 1e-10,
) -> IncomingDescriptorTransform:
    """Fit strictly on cumulative incoming rows; no evaluation argument exists."""

    matrix = np.asarray(values, dtype=np.float64)
    if matrix.ndim != 2 or len(matrix) < 2 or matrix.shape[1] < 1:
        raise ValueError("Descriptor fit matrix must be [N>=2,D>=1].")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("Descriptor fit matrix contains non-finite values.")
    requested = int(maximum_components)
    if requested < 1:
        raise ValueError("maximum_components must be positive.")
    keep = np.flatnonzero(matrix.std(axis=0) > float(constant_tolerance))
    if not len(keep):
        raise RuntimeError("Every incoming trajectory descriptor column is constant.")
    selected = matrix[:, keep]
    mean = selected.mean(axis=0)
    scale = np.maximum(selected.std(axis=0), 1e-8)
    standardized = (selected - mean) / scale
    pca_mean = standardized.mean(axis=0)
    output_dim = min(requested, len(standardized) - 1, standardized.shape[1])
    _, _, right = np.linalg.svd(standardized - pca_mean, full_matrices=False)
    components = right[:output_dim].copy()
    # Fix the algebraically arbitrary component signs for byte-stable audits.
    for row in components:
        pivot = int(np.argmax(np.abs(row)))
        if row[pivot] < 0:
            row *= -1.0
    return IncomingDescriptorTransform(
        raw_dim=int(matrix.shape[1]),
        keep_columns=keep.astype(np.int64),
        mean=mean.astype(np.float64),
        scale=scale.astype(np.float64),
        pca_mean=pca_mean.astype(np.float64),
        pca_components=components.astype(np.float64),
        fit_row_count=int(len(matrix)),
        requested_pca_dim=requested,
    ).validate()


def _raw_descriptor_matrix(
    primitive_trials: Sequence[PrimitiveTrial],
) -> tuple[np.ndarray, tuple[str, ...]]:
    rows: list[np.ndarray] = []
    schema: tuple[str, ...] | None = None
    for trial in primitive_trials:
        values, names = trajectory_descriptor(trial, 32, include_state=True)
        if schema is None:
            schema = tuple(names)
        elif tuple(names) != schema:
            raise RuntimeError("Raw state descriptor schema differs between trials.")
        rows.append(np.asarray(values, dtype=np.float64))
    if not rows or schema is None:
        raise ValueError("At least one primitive trajectory is required.")
    return np.stack(rows), schema


def _encode_primitives(
    encoder: torch.nn.Module,
    trials: Sequence[SensorTrial],
    codebook: Any,
    *,
    device: torch.device,
    batch_size: int,
) -> list[PrimitiveTrial]:
    encoded = encode_sensor_trials(encoder, trials, device=device, batch_size=int(batch_size))
    return [tokenize_e0_trial(item, codebook) for item in encoded]


def _trajectory_rows(
    trials: Sequence[SensorTrial],
    primitive: Sequence[PrimitiveTrial],
    *,
    role: str,
    session_by_trial: Mapping[int, int],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for sensor, item in zip(trials, primitive):
        rows.append(
            {
                "role": str(role),
                "session": int(session_by_trial[int(sensor.trial_id)]),
                "trial_id": int(sensor.trial_id),
                "subject_id": int(sensor.subject_id),
                "window_count": int(len(item.child_tokens)),
                "primitive_sequence": np.asarray(item.child_tokens, dtype=np.int64).tolist(),
                "ownership_start_samples": np.asarray(item.starts, dtype=np.int64).tolist(),
                "ownership_end_samples_exclusive": np.asarray(item.ends, dtype=np.int64).tolist(),
                "quantization_distances": np.asarray(item.child_distances, dtype=float).tolist(),
            }
        )
    return rows


def _implementation_hashes() -> dict[str, str]:
    names = (
        "experiments/motion_primitive/strict_session3_k12_proxy.py",
        "experiments/motion_primitive/strict_offline_runner.py",
        "experiments/motion_primitive/strict_artifacts.py",
        "experiments/motion_primitive/strict_protocol.py",
        "experiments/motion_primitive/strict_metrics.py",
        "experiments/motion_primitive/frozen_e0_state.py",
        "experiments/motion_primitive/motion_checkpoint.py",
        "experiments/motion_primitive/motion_encoder.py",
        "models/resnet1d.py",
    )
    return {name: sha256_file(PROJECT_ROOT / name) for name in names}


def _identity(
    args: argparse.Namespace,
    *,
    offline_dir: Path,
    offline_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema": IDENTITY_SCHEMA,
        "profile": PROFILE,
        "fold": int(args.fold),
        "seed": int(args.seed),
        "offline_run_dir": str(offline_dir),
        "offline_manifest_sha256": sha256_file(offline_dir / "manifest.json"),
        "offline_complete_sha256": sha256_file(offline_dir / "complete.json"),
        "source_representation_sha256": str(offline_manifest["representation_sha256"]),
        "proxy": {
            "incoming_sessions": [1, 2, 3],
            "evaluation_session": 3,
            "expected_incoming_counts": list(INCOMING_SESSION_COUNTS),
            "expected_cumulative_incoming_count": INCOMING_TOTAL,
            "expected_evaluation_count": EVALUATION_COUNT,
            "cluster_count": int(args.cluster_count),
            "descriptor_pca_dim": int(args.descriptor_pca_dim),
            "constant_tolerance": float(args.constant_tolerance),
            "kmeans_n_init": int(args.kmeans_n_init),
            "kmeans_max_iter": int(args.kmeans_max_iter),
            "raw_state_descriptor": True,
            "offline_descriptor_transform_applied": False,
            "global_hungarian_scoring_only": True,
        },
        "runtime": {
            "encode_batch_size": int(args.encode_batch_size),
            "requested_device": str(args.device),
            "resolved_device": str(_resolve_device(args.device)),
            "allow_smoke_a2": bool(args.allow_smoke_a2),
        },
        "implementation_sha256": _implementation_hashes(),
    }


def _prepare_identity(
    output: Path,
    args: argparse.Namespace,
    *,
    offline_dir: Path,
    offline_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    body = _identity(args, offline_dir=offline_dir, offline_manifest=offline_manifest)
    expected = {**body, "identity_sha256": canonical_hash(body)}
    path = output / "run_identity.json"
    if path.is_file():
        if _verified_identity(path) != expected:
            raise RuntimeError(f"Output {output} records another proxy identity.")
        return expected
    if output.exists():
        allowed = {"cv_member_manifest.json"}
        unexpected = [item.name for item in output.iterdir() if item.name not in allowed]
        if unexpected:
            raise RuntimeError(f"Unidentified proxy output cannot be adopted: {unexpected}.")
    output.mkdir(parents=True, exist_ok=True)
    write_json(path, expected)
    return expected


def _output_artifacts() -> tuple[str, ...]:
    return (
        "cluster_model.json",
        "cluster_model.npz",
        "fit_manifest.json",
        "incoming_transform.json",
        "incoming_transform.npz",
        "leakage_audit.json",
        "metrics.json",
        "predictions.csv",
        "proxy_summary.json",
        "raw_predictions.npz",
        "trajectories_label_free.jsonl",
    )


def validate_completed_output(
    output: Path,
    *,
    expected_identity: Mapping[str, Any],
) -> dict[str, Any]:
    complete_path = output / "complete.json"
    summary_path = output / "proxy_summary.json"
    identity_path = output / "run_identity.json"
    if not all(path.is_file() for path in (complete_path, summary_path, identity_path)):
        raise RuntimeError(f"Proxy completed member is incomplete: {output}.")
    if _verified_identity(identity_path) != dict(expected_identity):
        raise RuntimeError("Proxy run identity changed after execution.")
    complete = _read_json(complete_path)
    summary = _read_json(summary_path)
    if complete.get("schema") != SCHEMA or summary.get("schema") != SCHEMA:
        raise RuntimeError("Proxy completion belongs to another schema.")
    if complete.get("complete") is not True:
        raise RuntimeError("Proxy complete marker is false.")
    if complete.get("run_identity_sha256") != expected_identity["identity_sha256"]:
        raise RuntimeError("Proxy complete marker is not bound to its run identity.")
    if complete.get("proxy_summary_sha256") != sha256_file(summary_path):
        raise RuntimeError("Proxy summary SHA256 mismatch.")
    for key, value in summary.items():
        if complete.get(key) != value:
            raise RuntimeError(f"Proxy completion/summary field {key!r} differs.")
    hashes = complete.get("artifact_sha256")
    if not isinstance(hashes, Mapping) or set(hashes) != set(_output_artifacts()):
        raise RuntimeError("Proxy completion has an incomplete artifact inventory.")
    for name, expected in hashes.items():
        path = output / str(name)
        if not path.is_file() or sha256_file(path) != expected:
            raise RuntimeError(f"Proxy artifact SHA256 mismatch: {name}.")
    if int(summary.get("incoming_count", -1)) != INCOMING_TOTAL:
        raise RuntimeError("Proxy summary does not contain 78 incoming trials.")
    if int(summary.get("evaluation_count", -1)) != EVALUATION_COUNT:
        raise RuntimeError("Proxy summary does not contain 42 evaluation trials.")
    if summary.get("raw_predictions_frozen_before_truth_join") is not True:
        raise RuntimeError("Proxy does not prove prediction freezing before scoring.")
    if summary.get("evaluation_used_to_fit_transform_or_kmeans") is not False:
        raise RuntimeError("Proxy reports evaluation leakage.")
    return complete


def validate_args(args: argparse.Namespace) -> argparse.Namespace:
    if str(args.profile) != PROFILE:
        raise ValueError(f"K12 proxy accepts only {PROFILE}.")
    if int(args.cluster_count) != CLUSTER_COUNT:
        raise ValueError("Final Session-3 proxy is pinned to K=12.")
    if int(args.descriptor_pca_dim) != DESCRIPTOR_PCA_DIM:
        raise ValueError("Final Session-3 proxy is pinned to descriptor PCA32.")
    if int(args.encode_batch_size) < 1 or int(args.kmeans_n_init) < 1 or int(args.kmeans_max_iter) < 1:
        raise ValueError("Batch size and KMeans iteration parameters must be positive.")
    if float(args.constant_tolerance) <= 0:
        raise ValueError("constant-tolerance must be positive.")
    _resolve_device(args.device)
    return args


def run(args: argparse.Namespace) -> dict[str, Any]:
    args = validate_args(args)
    offline_dir = Path(args.offline_run_dir).expanduser().resolve()
    manifest = _strict_offline_manifest(offline_dir)
    if (int(manifest["fold"]), int(manifest["seed"])) != (int(args.fold), int(args.seed)):
        raise RuntimeError("Proxy fold/seed differs from the Offline member.")
    output = Path(args.output_dir).expanduser().resolve()
    if output.exists() and not bool(args.resume):
        raise FileExistsError(f"Output already exists: {output}")
    identity = _prepare_identity(
        output, args, offline_dir=offline_dir, offline_manifest=manifest
    )
    if (output / "complete.json").is_file():
        if not bool(args.resume):
            raise FileExistsError(f"Completed output already exists: {output}")
        return validate_completed_output(output, expected_identity=identity)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[logging.FileHandler(output / "proxy.log", encoding="utf-8"), logging.StreamHandler()],
        force=True,
    )
    protocol = build_registered_protocol(
        manifest["npz_path"],
        fold=int(args.fold),
        seed=int(args.seed),
        window_size=256,
        stride=128,
        shuffle_novel_classes=False,
    )
    if protocol.npz_sha256 != manifest["npz_sha256"] or protocol.split.as_dict() != manifest["split"]:
        raise RuntimeError("Proxy protocol differs from the completed Offline member.")

    device = _resolve_device(args.device)
    encoder, encoder_audit = load_frozen_a2_encoder(
        manifest["a2_encoder"]["checkpoint_path"],
        expected_npz_sha256=protocol.npz_sha256,
        expected_fold=int(args.fold),
        expected_seed=int(args.seed),
        expected_split=protocol.split,
        expected_window_size=256,
        allow_smoke=bool(args.allow_smoke_a2),
    )
    if encoder_audit["checkpoint_sha256"] != manifest["a2_encoder"]["checkpoint_sha256"]:
        raise RuntimeError("A2 checkpoint bytes changed after Offline fitting.")
    if encoder_audit["model_state_dict_sha256"] != manifest["a2_encoder"]["model_state_dict_sha256"]:
        raise RuntimeError("A2 encoder state changed after Offline fitting.")
    codebook = load_e0_codebook(offline_dir / "e0_codebook.npz")
    if codebook.state_sha256 != manifest["codebook"]["state_sha256"]:
        raise RuntimeError("E0 K32 state differs from the Offline manifest.")
    # Verify the historical descriptor artifact, then intentionally do not use
    # it.  The proxy refits its own transform from raw incoming descriptors.
    offline_descriptor = load_descriptor_transform(offline_dir / "state_descriptor_transform.npz")
    if offline_descriptor.state_sha256 != manifest["descriptor"]["state_sha256"]:
        raise RuntimeError("Offline descriptor state differs from its manifest.")

    sessions = tuple(protocol.sessions)
    counts = tuple(len(item.incoming) for item in sessions)
    if counts != INCOMING_SESSION_COUNTS:
        raise RuntimeError(f"Registered incoming counts changed: {counts}.")
    incoming_trials = tuple(trial for session in sessions for trial in session.incoming)
    evaluation_trials = tuple(sessions[-1].evaluation)
    incoming_ids = [int(item.trial_id) for item in incoming_trials]
    evaluation_ids = [int(item.trial_id) for item in evaluation_trials]
    if len(incoming_ids) != INCOMING_TOTAL or len(set(incoming_ids)) != INCOMING_TOTAL:
        raise RuntimeError("Cumulative Session-3 incoming set is not 78 unique trials.")
    if len(evaluation_ids) != EVALUATION_COUNT or len(set(evaluation_ids)) != EVALUATION_COUNT:
        raise RuntimeError("Session-3 evaluation set is not 42 unique trials.")
    overlap = sorted(set(incoming_ids) & set(evaluation_ids))
    if overlap:
        raise RuntimeError(f"Incoming/evaluation trial leakage: {overlap}.")
    incoming_session = {
        int(trial.trial_id): int(session.session)
        for session in sessions for trial in session.incoming
    }
    evaluation_session = {int(trial.trial_id): 3 for trial in evaluation_trials}

    LOGGER.info("encoding cumulative incoming=%d and final evaluation=%d", len(incoming_trials), len(evaluation_trials))
    initial_encoder_hash = encoder_audit["model_state_dict_sha256"]
    incoming_primitive = _encode_primitives(
        encoder, incoming_trials, codebook, device=device, batch_size=int(args.encode_batch_size)
    )
    evaluation_primitive = _encode_primitives(
        encoder, evaluation_trials, codebook, device=device, batch_size=int(args.encode_batch_size)
    )
    incoming_raw, descriptor_names = _raw_descriptor_matrix(incoming_primitive)
    evaluation_raw, evaluation_names = _raw_descriptor_matrix(evaluation_primitive)
    if evaluation_names != descriptor_names:
        raise RuntimeError("Incoming/evaluation raw descriptor schemas differ.")
    schema_sha = hashlib.sha256("\n".join(descriptor_names).encode("utf-8")).hexdigest()
    if schema_sha != HISTORICAL_STATE_SCHEMA_SHA256:
        raise RuntimeError("Raw state descriptor schema differs from the validated E0 route.")

    fit_manifest = {
        "schema": SCHEMA,
        "fold": int(args.fold),
        "seed": int(args.seed),
        "fit_role": "cumulative_sessions_1_2_3_incoming_only",
        "fit_session_counts": list(counts),
        "fit_trial_count": len(incoming_ids),
        "fit_trial_ids": incoming_ids,
        "fit_trial_ids_sha256": _ids_sha256(incoming_ids),
        "fit_subject_ids": [int(item.subject_id) for item in incoming_trials],
        "evaluation_role": "session_3_evaluation_predict_only",
        "evaluation_trial_count": len(evaluation_ids),
        "evaluation_trial_ids": evaluation_ids,
        "evaluation_trial_ids_sha256": _ids_sha256(evaluation_ids),
        "evaluation_subject_ids": [int(item.subject_id) for item in evaluation_trials],
        "train_evaluation_overlap": overlap,
        "activity_labels_present": False,
        "raw_descriptor_schema_sha256": schema_sha,
        "raw_descriptor_dim": int(incoming_raw.shape[1]),
    }
    write_json(output / "fit_manifest.json", fit_manifest)

    # Neither fitting function accepts evaluation rows.  Evaluation is passed
    # only to frozen ``transform`` and ``predict`` calls below.
    transform = fit_incoming_descriptor_transform(
        incoming_raw,
        maximum_components=int(args.descriptor_pca_dim),
        constant_tolerance=float(args.constant_tolerance),
    )
    incoming_features = transform.transform(incoming_raw)
    evaluation_features = transform.transform(evaluation_raw)
    _atomic_npz(
        output / "incoming_transform.npz",
        raw_dim=np.asarray(transform.raw_dim, dtype=np.int64),
        keep_columns=transform.keep_columns,
        mean=transform.mean,
        scale=transform.scale,
        pca_mean=transform.pca_mean,
        pca_components=transform.pca_components,
        fit_row_count=np.asarray(transform.fit_row_count, dtype=np.int64),
        requested_pca_dim=np.asarray(transform.requested_pca_dim, dtype=np.int64),
    )
    transform_metadata = {
        "schema": SCHEMA,
        "steps": ["constant_filter", "z_score", "pca", "l2"],
        "fit_scope": "78_cumulative_incoming_only",
        "evaluation_fit_row_count": 0,
        "raw_dim": transform.raw_dim,
        "kept_dim": len(transform.keep_columns),
        "output_dim": transform.output_dim,
        "fit_row_count": transform.fit_row_count,
        "requested_pca_dim": transform.requested_pca_dim,
        "constant_tolerance": float(args.constant_tolerance),
        "state_sha256": transform.state_sha256,
        "artifact_sha256": sha256_file(output / "incoming_transform.npz"),
        "offline_old_class_transform_state_sha256_verified": offline_descriptor.state_sha256,
        "offline_old_class_transform_applied": False,
    }
    write_json(output / "incoming_transform.json", transform_metadata)

    model = KMeans(
        n_clusters=CLUSTER_COUNT,
        n_init=int(args.kmeans_n_init),
        max_iter=int(args.kmeans_max_iter),
        random_state=int(args.seed),
        algorithm="lloyd",
    )
    incoming_predictions = model.fit_predict(incoming_features).astype(np.int64)
    if set(np.unique(incoming_predictions).tolist()) != set(range(CLUSTER_COUNT)):
        raise RuntimeError("Incoming KMeans fit did not use all 12 clusters.")
    evaluation_predictions = model.predict(evaluation_features).astype(np.int64)
    centers = np.asarray(model.cluster_centers_, dtype=np.float32)
    model_state_sha = _array_sha256(
        np.asarray([CLUSTER_COUNT, int(args.seed), int(model.n_iter_)], dtype="<i8"), centers
    )
    _atomic_npz(
        output / "cluster_model.npz",
        centers=centers,
        incoming_cluster_ids=incoming_predictions,
        fit_trial_ids=np.asarray(incoming_ids, dtype=np.int64),
    )
    model_metadata = {
        "schema": SCHEMA,
        "algorithm": "sklearn_KMeans_lloyd",
        "cluster_count": CLUSTER_COUNT,
        "fit_scope": "78_cumulative_incoming_only",
        "evaluation_fit_row_count": 0,
        "seed": int(args.seed),
        "n_init": int(args.kmeans_n_init),
        "max_iter": int(args.kmeans_max_iter),
        "iterations": int(model.n_iter_),
        "inertia": float(model.inertia_),
        "used_cluster_count": int(len(np.unique(incoming_predictions))),
        "feature_dim": int(incoming_features.shape[1]),
        "sklearn_version": sklearn.__version__,
        "state_sha256": model_state_sha,
        "artifact_sha256": sha256_file(output / "cluster_model.npz"),
    }
    write_json(output / "cluster_model.json", model_metadata)

    # This file is the irreversible boundary between learner and scorer.
    # Activity truth is not opened anywhere above this write and hash.
    _atomic_npz(
        output / "raw_predictions.npz",
        incoming_trial_ids=np.asarray(incoming_ids, dtype=np.int64),
        incoming_subject_ids=np.asarray([item.subject_id for item in incoming_trials], dtype=np.int64),
        incoming_session_ids=np.asarray([incoming_session[value] for value in incoming_ids], dtype=np.int64),
        incoming_features=incoming_features.astype(np.float32),
        incoming_cluster_ids=incoming_predictions,
        evaluation_trial_ids=np.asarray(evaluation_ids, dtype=np.int64),
        evaluation_subject_ids=np.asarray([item.subject_id for item in evaluation_trials], dtype=np.int64),
        evaluation_features=evaluation_features.astype(np.float32),
        raw_evaluation_cluster_ids=evaluation_predictions,
        transform_state_sha256=np.asarray(transform.state_sha256),
        cluster_model_state_sha256=np.asarray(model_state_sha),
        source_representation_sha256=np.asarray(str(manifest["representation_sha256"])),
    )
    raw_prediction_sha = sha256_file(output / "raw_predictions.npz")

    # Scorer-only truth access begins here, after raw predictions are durable.
    targets, names, subjects = protocol.truth.join(evaluation_ids)
    score_bundle = strict_three_layer_metrics(
        targets,
        evaluation_predictions,
        class_count=12,
        old_class_count=6,
        seen_class_count_before_session=10,
        registered_class_ids=tuple(range(12)),
    )
    global_metrics = score_bundle["layers"]["global_hungarian_upper_bound"]
    confusion = np.asarray(
        [row[:12] for row in global_metrics["confusion_matrix_with_unknown_column"]],
        dtype=np.int64,
    )
    row_totals = confusion.sum(axis=1)
    per_class_recall = np.divide(
        np.diag(confusion),
        row_totals,
        out=np.zeros(12, dtype=np.float64),
        where=row_totals > 0,
    )
    metrics = {
        "schema": SCHEMA,
        "primary_layer": "global_hungarian_scoring_only",
        "raw_predictions_path": "raw_predictions.npz",
        "raw_predictions_sha256": raw_prediction_sha,
        "raw_predictions_frozen_before_truth_join": True,
        "global_hungarian_uses_evaluation_labels_for_scoring_only": True,
        "scoring_does_not_write_back_to_transform_or_kmeans": True,
        "all_accuracy": global_metrics["all_accuracy"],
        "old_accuracy": global_metrics["old_accuracy"],
        "new_accuracy": global_metrics["new_accuracy"],
        "h_score": global_metrics["h_score"],
        "macro_f1": global_metrics["macro_f1"],
        "assignment_cluster_to_activity": global_metrics["assignment_pred_to_true"],
        "confusion_matrix": confusion.tolist(),
        "confusion_rows": list(range(12)),
        "confusion_columns": list(range(12)),
        "per_class_recall": per_class_recall.tolist(),
        "per_class_support": row_totals.tolist(),
        "sample_count": int(global_metrics["sample_count"]),
    }
    write_json(output / "metrics.json", metrics)
    aligned = np.asarray(global_metrics["aligned_predictions"], dtype=np.int64)
    write_csv(
        output / "predictions.csv",
        [
            {
                "trial_id": int(trial.trial_id),
                "subject_id": int(subject),
                "activity_label": int(target),
                "activity_name": str(name),
                "raw_cluster_id": int(raw),
                "global_hungarian_prediction": int(prediction),
            }
            for trial, subject, target, name, raw, prediction in zip(
                evaluation_trials, subjects, targets, names, evaluation_predictions, aligned
            )
        ],
    )
    _write_jsonl(
        output / "trajectories_label_free.jsonl",
        _trajectory_rows(
            incoming_trials, incoming_primitive, role="incoming_fit", session_by_trial=incoming_session
        )
        + _trajectory_rows(
            evaluation_trials,
            evaluation_primitive,
            role="session3_evaluation_predict_only",
            session_by_trial=evaluation_session,
        ),
    )

    final_encoder_hash = motion_state_dict_sha256(
        {key: value.detach().cpu() for key, value in encoder.state_dict().items()}
    )
    if final_encoder_hash != initial_encoder_hash:
        raise RuntimeError("Frozen A2 parameters or BatchNorm buffers changed during proxy execution.")
    if codebook.state_sha256 != manifest["codebook"]["state_sha256"]:
        raise RuntimeError("Frozen E0 K32 state changed during proxy execution.")
    leakage_audit = {
        "schema": SCHEMA,
        "learner_activity_labels_used": False,
        "activity_truth_first_access_stage": "after_raw_predictions_npz_was_written_and_hashed",
        "raw_predictions_sha256": raw_prediction_sha,
        "incoming_sessions_used_for_fit": [1, 2, 3],
        "incoming_session_counts": list(counts),
        "incoming_fit_trial_count": len(incoming_ids),
        "evaluation_session": 3,
        "evaluation_predict_only_trial_count": len(evaluation_ids),
        "incoming_evaluation_trial_overlap": overlap,
        "evaluation_rows_used_for_constant_filter": 0,
        "evaluation_rows_used_for_z_score": 0,
        "evaluation_rows_used_for_pca": 0,
        "evaluation_rows_used_for_kmeans": 0,
        "offline_old_class_descriptor_transform_verified": True,
        "offline_old_class_descriptor_transform_applied": False,
        "proxy_transform_fit_scope": "raw_state_descriptors_of_78_cumulative_incoming_trials_only",
        "kmeans_fit_scope": "l2_features_of_78_cumulative_incoming_trials_only",
        "global_hungarian_role": "post_hoc_scoring_only",
        "global_hungarian_writeback": False,
        "frozen_a2_initial_sha256": initial_encoder_hash,
        "frozen_a2_final_sha256": final_encoder_hash,
        "frozen_e0_codebook_sha256": codebook.state_sha256,
    }
    write_json(output / "leakage_audit.json", leakage_audit)

    summary = {
        "schema": SCHEMA,
        "profile": PROFILE,
        "fold": int(args.fold),
        "seed": int(args.seed),
        "proxy_name": "session3_cumulative_incoming_k12_global_hungarian",
        "incoming_count": len(incoming_ids),
        "incoming_session_counts": list(counts),
        "evaluation_count": len(evaluation_ids),
        "cluster_count": CLUSTER_COUNT,
        "descriptor_output_dim": transform.output_dim,
        "source_representation_sha256": str(manifest["representation_sha256"]),
        "proxy_transform_state_sha256": transform.state_sha256,
        "cluster_model_state_sha256": model_state_sha,
        "raw_predictions_sha256": raw_prediction_sha,
        "raw_predictions_frozen_before_truth_join": True,
        "evaluation_used_to_fit_transform_or_kmeans": False,
        "offline_old_class_descriptor_transform_applied": False,
        "global_hungarian_scoring_only": True,
        "seen_class_count_before_session": 10,
        "all_accuracy": metrics["all_accuracy"],
        "old_accuracy": metrics["old_accuracy"],
        "new_accuracy": metrics["new_accuracy"],
        "h_score": metrics["h_score"],
        "macro_f1": metrics["macro_f1"],
    }
    write_json(output / "proxy_summary.json", summary)
    artifact_hashes = {name: sha256_file(output / name) for name in _output_artifacts()}
    complete = {
        **summary,
        "run_identity_sha256": identity["identity_sha256"],
        "proxy_summary_sha256": sha256_file(output / "proxy_summary.json"),
        "artifact_sha256": artifact_hashes,
        "complete": True,
    }
    write_json(output / "complete.json", complete)
    LOGGER.info(
        "completed K12 proxy fold=%d seed=%d all=%.4f old=%.4f new=%.4f H=%.4f macro_f1=%.4f",
        int(args.fold), int(args.seed), metrics["all_accuracy"], metrics["old_accuracy"],
        metrics["new_accuracy"], metrics["h_score"], metrics["macro_f1"],
    )
    return complete


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Strict final Session-3 K12 proxy: fit 78 incoming trajectories, predict 42 evaluation trajectories."
    )
    parser.add_argument("--offline-run-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--profile", default=PROFILE, choices=(PROFILE,))
    parser.add_argument("--fold", type=int, required=True, choices=range(1, 8))
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--cluster-count", type=int, default=CLUSTER_COUNT)
    parser.add_argument("--descriptor-pca-dim", type=int, default=DESCRIPTOR_PCA_DIM)
    parser.add_argument("--constant-tolerance", type=float, default=1e-10)
    parser.add_argument("--kmeans-n-init", type=int, default=50)
    parser.add_argument("--kmeans-max-iter", type=int, default=300)
    parser.add_argument("--encode-batch-size", type=int, default=512)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--allow-smoke-a2", action="store_true")
    parser.add_argument("--resume", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> dict[str, Any]:
    result = run(build_parser().parse_args(argv))
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return result


if __name__ == "__main__":
    main()


__all__ = [
    "CLUSTER_COUNT",
    "DESCRIPTOR_PCA_DIM",
    "EVALUATION_COUNT",
    "IDENTITY_SCHEMA",
    "INCOMING_SESSION_COUNTS",
    "INCOMING_TOTAL",
    "IncomingDescriptorTransform",
    "SCHEMA",
    "build_parser",
    "fit_incoming_descriptor_transform",
    "main",
    "run",
    "validate_args",
    "validate_completed_output",
]
