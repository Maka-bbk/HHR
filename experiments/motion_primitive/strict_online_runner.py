"""Strict three-session Online CGCD over frozen motion-primitive trajectories."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch

from experiments.motion_primitive.frozen_e0_state import (
    PrimitiveTrial,
    codebook_usage,
    descriptor_matrix,
    tokenize_e0_trial,
)
from experiments.motion_primitive.strict_artifacts import (
    encode_sensor_trials,
    load_descriptor_transform,
    load_e0_codebook,
    load_frozen_a2_encoder,
    write_csv,
    write_json,
)
from experiments.motion_primitive.strict_metrics import strict_three_layer_metrics
from experiments.motion_primitive.strict_offline_runner import (
    OFFLINE_SCHEMA,
    PROFILE,
    validate_completed_output as validate_completed_offline_output,
)
from experiments.motion_primitive.strict_cv_common import canonical_hash
from experiments.motion_primitive.strict_protocol import (
    SensorTrial,
    build_registered_protocol,
    sha256_file,
)
from experiments.motion_primitive.strict_registry import (
    LabelFreeTrial,
    StrictRegistryConfig,
    advance_registry_session,
    deserialize_registry_state,
    route_label_free_trials,
    serialize_registry_state,
)


LOGGER = logging.getLogger("hhr_strict_online")
ONLINE_SCHEMA = "hhr_frozen_a2_e0_state_online_v1"
RUN_IDENTITY_SCHEMA = "hhr_frozen_a2_e0_state_online_identity_v1"


def _output_artifacts() -> tuple[str, ...]:
    names = ["online_runs.csv", "online_summary.json", "trajectories_label_free.jsonl"]
    for session in range(1, 4):
        names.extend(
            [
                f"discovery_session_{session}.json",
                f"incoming_label_free_manifest_session_{session}.json",
                f"metrics_session_{session}.json",
                f"predictions_session_{session}.csv",
                f"raw_predictions_session_{session}.npz",
                f"registry_session_{session}.json",
            ]
        )
    return tuple(sorted(names))


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"JSON artifact {path} is not an object.")
    return value


def _resolve_device(value: str) -> torch.device:
    if str(value).lower() == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")
    return device


def _implementation_hashes() -> dict[str, str]:
    relative = (
        "models/resnet1d.py",
        "experiments/motion_primitive/motion_encoder.py",
        "experiments/motion_primitive/motion_checkpoint.py",
        "experiments/motion_primitive/frozen_e0_state.py",
        "experiments/motion_primitive/strict_artifacts.py",
        "experiments/motion_primitive/strict_protocol.py",
        "experiments/motion_primitive/strict_registry.py",
        "experiments/motion_primitive/strict_metrics.py",
        "experiments/motion_primitive/strict_offline_runner.py",
        "experiments/motion_primitive/strict_online_runner.py",
    )
    return {name: sha256_file(PROJECT_ROOT / name) for name in relative}


def _verified_identity(path: Path) -> dict[str, Any]:
    payload = _read_json(path)
    recorded = payload.get("identity_sha256")
    identity = {key: value for key, value in payload.items() if key != "identity_sha256"}
    if not isinstance(recorded, str) or canonical_hash(identity) != recorded:
        raise RuntimeError(f"Identity SHA256 does not verify in {path}.")
    return payload


def _run_identity(
    args: argparse.Namespace,
    *,
    offline_dir: Path,
    offline_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    offline_complete = offline_dir / "complete.json"
    offline_manifest_path = offline_dir / "manifest.json"
    return {
        "schema": RUN_IDENTITY_SCHEMA,
        "profile": PROFILE,
        "fold": int(args.fold),
        "seed": int(args.seed),
        "offline_run_dir": str(offline_dir),
        "offline_manifest_sha256": sha256_file(offline_manifest_path),
        "offline_complete_sha256": sha256_file(offline_complete),
        "offline_representation_sha256": str(offline_manifest["representation_sha256"]),
        "offline_old_registry_state_sha256": str(
            offline_manifest["old_registry"]["state_sha256"]
        ),
        "runtime": {
            "encode_batch_size": int(args.encode_batch_size),
            "requested_device": str(args.device),
            "resolved_device": str(_resolve_device(args.device)),
            "allow_smoke_a2": bool(args.allow_smoke_a2),
        },
        "protocol": {
            "sessions": [1, 2, 3],
            "online_activity_labels_used_by_learner": False,
            "test_labels_used_only_for_scoring": True,
            "frozen_representation": True,
            "append_only_activity_registry": True,
        },
        "implementation_sha256": _implementation_hashes(),
    }


def _prepare_run_identity(
    output: Path,
    args: argparse.Namespace,
    *,
    offline_dir: Path,
    offline_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    identity = _run_identity(args, offline_dir=offline_dir, offline_manifest=offline_manifest)
    expected = {**identity, "identity_sha256": canonical_hash(identity)}
    path = output / "run_identity.json"
    if path.is_file():
        if _verified_identity(path) != expected:
            raise RuntimeError(
                f"Online output {output} records another run identity; use a new directory."
            )
        return expected
    if output.exists():
        allowed = {"cv_member_manifest.json"}
        unidentified = [item.name for item in output.iterdir() if item.name not in allowed]
        if unidentified:
            raise RuntimeError(
                f"Online output {output} has artifacts but no run_identity.json; "
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
    summary_path = output / "online_summary.json"
    if not identity_path.is_file() or not complete_path.is_file() or not summary_path.is_file():
        raise RuntimeError(f"Online completed member is incomplete: {output}.")
    if _verified_identity(identity_path) != dict(expected_identity):
        raise RuntimeError("Online run identity changed after execution.")
    complete = _read_json(complete_path)
    summary = _read_json(summary_path)
    if complete.get("schema") != ONLINE_SCHEMA or summary.get("schema") != ONLINE_SCHEMA:
        raise RuntimeError("Online completed member has another schema.")
    if complete.get("profile") != PROFILE or summary.get("profile") != PROFILE:
        raise RuntimeError("Online completed member has another profile.")
    if complete.get("complete") is not True:
        raise RuntimeError("Online completion marker is false.")
    if complete.get("run_identity_sha256") != expected_identity["identity_sha256"]:
        raise RuntimeError("Online completion is not bound to its run identity.")
    if complete.get("online_summary_sha256") != sha256_file(summary_path):
        raise RuntimeError("Online summary SHA256 mismatch.")
    for key, value in summary.items():
        if complete.get(key) != value:
            raise RuntimeError(f"Online completion/summary field {key!r} differs.")
    expected_names = set(_output_artifacts())
    artifact_hashes = complete.get("artifact_sha256")
    if not isinstance(artifact_hashes, Mapping) or set(artifact_hashes) != expected_names:
        raise RuntimeError("Online completion has an incomplete artifact hash inventory.")
    for name, expected_hash in artifact_hashes.items():
        path = output / str(name)
        if not path.is_file() or sha256_file(path) != expected_hash:
            raise RuntimeError(f"Online artifact SHA256 mismatch: {name}.")
    initial = complete.get("frozen_hashes_initial")
    final = complete.get("frozen_hashes_final")
    if not isinstance(initial, Mapping) or initial != final:
        raise RuntimeError("Online frozen component hashes changed.")
    if initial.get("representation_sha256") != expected_identity["offline_representation_sha256"]:
        raise RuntimeError("Online representation differs from its Offline identity.")
    if complete.get("online_activity_labels_used_by_learner") is not False:
        raise RuntimeError("Online learner reports access to activity labels.")
    if complete.get("test_labels_used_only_for_scoring") is not True:
        raise RuntimeError("Online scorer label isolation is not recorded.")
    return complete


def _strict_offline_manifest(run_dir: Path) -> dict[str, Any]:
    manifest_path = run_dir / "manifest.json"
    complete_path = run_dir / "complete.json"
    identity_path = run_dir / "run_identity.json"
    if not manifest_path.is_file() or not complete_path.is_file() or not identity_path.is_file():
        raise FileNotFoundError("Offline member is incomplete.")
    identity = _verified_identity(identity_path)
    validate_completed_offline_output(run_dir, expected_identity=identity)
    manifest = _read_json(manifest_path)
    complete = _read_json(complete_path)
    if manifest.get("schema") != OFFLINE_SCHEMA or complete.get("schema") != OFFLINE_SCHEMA:
        raise RuntimeError("Offline member belongs to another route/schema.")
    if manifest.get("profile") != PROFILE or complete.get("profile") != PROFILE:
        raise RuntimeError("Offline member profile is not frozen A2/E0/state/K32.")
    if complete.get("complete") is not True:
        raise RuntimeError("Offline complete marker is false.")
    if complete.get("manifest_sha256") != sha256_file(manifest_path):
        raise RuntimeError("Offline manifest SHA256 does not verify.")
    return manifest


def _build_primitive_features(
    encoder,
    sensor_trials: Sequence[SensorTrial],
    codebook,
    descriptor,
    *,
    session_id: int,
    device: torch.device,
    batch_size: int,
) -> tuple[list[PrimitiveTrial], np.ndarray, tuple[LabelFreeTrial, ...]]:
    encoded = encode_sensor_trials(
        encoder, sensor_trials, device=device, batch_size=int(batch_size)
    )
    primitive = [tokenize_e0_trial(item, codebook) for item in encoded]
    features = descriptor_matrix(primitive, descriptor, include_state=True)
    records = tuple(LabelFreeTrial(
        trial_id=int(sensor.trial_id),
        subject_id=int(sensor.subject_id),
        session_id=int(session_id),
        descriptor=feature,
    ) for sensor, feature in zip(sensor_trials, features))
    return primitive, features, records


def _save_raw_predictions(
    path: Path,
    trials: Sequence[SensorTrial],
    descriptors: np.ndarray,
    raw_predictions: np.ndarray,
    *,
    registry_state_sha256: str,
    representation_sha256: str,
) -> str:
    np.savez_compressed(
        path,
        trial_ids=np.asarray([item.trial_id for item in trials], dtype=np.int64),
        subject_ids=np.asarray([item.subject_id for item in trials], dtype=np.int64),
        descriptors=np.asarray(descriptors, dtype=np.float32),
        raw_registry_predictions=np.asarray(raw_predictions, dtype=np.int64),
        registry_state_sha256=np.asarray(str(registry_state_sha256)),
        representation_sha256=np.asarray(str(representation_sha256)),
    )
    return sha256_file(path)


def _trajectory_rows(
    session: int,
    role: str,
    sensor_trials: Sequence[SensorTrial],
    primitive_trials: Sequence[PrimitiveTrial],
    predictions: Sequence[int] | None = None,
) -> list[dict[str, Any]]:
    rows = []
    for index, (sensor, primitive) in enumerate(zip(sensor_trials, primitive_trials)):
        row: dict[str, Any] = {
            "session": int(session),
            "role": str(role),
            "trial_id": int(sensor.trial_id),
            "subject_id": int(sensor.subject_id),
            "window_count": int(len(primitive.child_tokens)),
            "primitive_occurrence_count": int(len(primitive.child_tokens)),
            "unique_primitive_type_count": int(len(np.unique(primitive.child_tokens))),
            "used_primitive_count": int(len(np.unique(primitive.child_tokens))),
            "used_primitive_ids": sorted(np.unique(primitive.child_tokens).astype(int).tolist()),
            "primitive_sequence": primitive.child_tokens.astype(int).tolist(),
            "ownership_start_samples": primitive.starts.astype(int).tolist(),
            "ownership_end_samples_exclusive": primitive.ends.astype(int).tolist(),
            "quantization_distances": primitive.child_distances.astype(float).tolist(),
            "original_window_start_samples": sensor.window_starts.astype(int).tolist(),
            "raw_window_channel_means": sensor.raw_windows.mean(axis=2).astype(float).tolist(),
        }
        if predictions is not None:
            row["raw_registry_prediction"] = int(predictions[index])
        rows.append(row)
    return rows


def _append_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")


def _registry_config(manifest: Mapping[str, Any]) -> StrictRegistryConfig:
    payload = manifest.get("registry_config")
    if not isinstance(payload, Mapping):
        raise RuntimeError("Offline manifest lacks registry_config.")
    known = StrictRegistryConfig.__dataclass_fields__
    if set(payload) != set(known):
        raise RuntimeError("Offline registry configuration schema differs.")
    return StrictRegistryConfig(**dict(payload)).validated()


def validate_args(args: argparse.Namespace) -> argparse.Namespace:
    if str(args.profile) != PROFILE:
        raise ValueError(f"Online main route accepts only {PROFILE}.")
    if int(args.encode_batch_size) < 1:
        raise ValueError("encode-batch-size must be positive.")
    return args


def run(args: argparse.Namespace) -> dict[str, Any]:
    args = validate_args(args)
    offline_dir = Path(args.offline_run_dir).expanduser().resolve()
    manifest = _strict_offline_manifest(offline_dir)
    if int(args.fold) != int(manifest["fold"]) or int(args.seed) != int(manifest["seed"]):
        raise RuntimeError("Online fold/seed differs from the offline member.")
    output = Path(args.output_dir).expanduser().resolve()
    complete_path = output / "complete.json"
    if output.exists() and not bool(args.resume):
        raise FileExistsError(f"Output already exists: {output}")
    run_identity = _prepare_run_identity(
        output,
        args,
        offline_dir=offline_dir,
        offline_manifest=manifest,
    )
    if complete_path.is_file():
        if not bool(args.resume):
            raise FileExistsError(f"Completed output already exists: {output}")
        return validate_completed_output(output, expected_identity=run_identity)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[logging.FileHandler(output / "online.log", encoding="utf-8"), logging.StreamHandler()],
        force=True,
    )
    npz_path = Path(str(manifest["npz_path"]))
    protocol = build_registered_protocol(
        npz_path,
        fold=int(args.fold),
        seed=int(args.seed),
        window_size=256,
        stride=128,
        shuffle_novel_classes=False,
    )
    if protocol.npz_sha256 != manifest.get("npz_sha256"):
        raise RuntimeError("Online NPZ SHA256 differs from the offline member.")
    if protocol.split.as_dict() != manifest.get("split"):
        raise RuntimeError("Online subject split differs from the offline member.")
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
        raise RuntimeError("A2 checkpoint bytes changed after offline fitting.")
    if encoder_audit["model_state_dict_sha256"] != manifest["a2_encoder"]["model_state_dict_sha256"]:
        raise RuntimeError("A2 state changed after offline fitting.")
    codebook = load_e0_codebook(offline_dir / "e0_codebook.npz")
    descriptor = load_descriptor_transform(offline_dir / "state_descriptor_transform.npz")
    if codebook.state_sha256 != manifest["codebook"]["state_sha256"]:
        raise RuntimeError("E0 K32 state differs from offline manifest.")
    if descriptor.state_sha256 != manifest["descriptor"]["state_sha256"]:
        raise RuntimeError("Descriptor transform differs from offline manifest.")
    registry_path = offline_dir / str(manifest["old_registry"]["path"])
    if sha256_file(registry_path) != manifest["old_registry"]["artifact_sha256"]:
        raise RuntimeError("Old registry artifact SHA256 differs from offline manifest.")
    state = deserialize_registry_state(_read_json(registry_path))
    if state.state_sha256 != manifest["old_registry"]["state_sha256"]:
        raise RuntimeError("Old registry numerical state differs from offline manifest.")
    if state.representation_sha256 != manifest["representation_sha256"]:
        raise RuntimeError("Old registry is bound to another frozen representation.")
    config = _registry_config(manifest)

    trajectory_path = output / "trajectories_label_free.jsonl"
    if trajectory_path.exists():
        trajectory_path.unlink()
    metric_rows: list[dict[str, Any]] = []
    initial_encoder_hash = encoder_audit["model_state_dict_sha256"]
    initial_frozen_hashes = {
        "representation_sha256": state.representation_sha256,
        "encoder_state_sha256": initial_encoder_hash,
        "codebook_state_sha256": codebook.state_sha256,
        "descriptor_state_sha256": descriptor.state_sha256,
        "old_anchor_sha256": state.old_anchor_sha256,
    }

    for session in protocol.sessions:
        LOGGER.info(
            "session=%d incoming=%d evaluation=%d registry_k=%d buffer=%d",
            session.session, len(session.incoming), len(session.evaluation), len(state.entries), len(state.unknown_buffer),
        )
        incoming_primitive, incoming_features, incoming_records = _build_primitive_features(
            encoder,
            session.incoming,
            codebook,
            descriptor,
            session_id=session.session,
            device=device,
            batch_size=int(args.encode_batch_size),
        )
        write_json(output / f"incoming_label_free_manifest_session_{session.session}.json", {
            "schema": ONLINE_SCHEMA,
            "session": session.session,
            "trial_ids": [item.trial_id for item in incoming_records],
            "subject_ids": [item.subject_id for item in incoming_records],
            "trial_ids_sha256": session.incoming_trial_ids_sha256,
            "descriptor_sha256": [item.audit_dict()["descriptor_sha256"] for item in incoming_records],
            "activity_labels_present": False,
        })
        update = advance_registry_session(state, incoming_records, config=config)
        state = update.state
        if state.representation_sha256 != initial_frozen_hashes["representation_sha256"]:
            raise RuntimeError("Frozen representation changed during Online CGCD.")
        if state.old_anchor_sha256 != initial_frozen_hashes["old_anchor_sha256"]:
            raise RuntimeError("Old semantic registry changed during Online CGCD.")
        write_json(output / f"discovery_session_{session.session}.json", update.audit_dict())
        registry_file = output / f"registry_session_{session.session}.json"
        write_json(registry_file, serialize_registry_state(state))

        evaluation_primitive, evaluation_features, evaluation_records = _build_primitive_features(
            encoder,
            session.evaluation,
            codebook,
            descriptor,
            session_id=session.session,
            device=device,
            batch_size=int(args.encode_batch_size),
        )
        raw_predictions = np.asarray([
            decision.registry_id
            for decision in route_label_free_trials(state, evaluation_records)
        ], dtype=np.int64)
        raw_path = output / f"raw_predictions_session_{session.session}.npz"
        raw_hash = _save_raw_predictions(
            raw_path,
            session.evaluation,
            evaluation_features,
            raw_predictions,
            registry_state_sha256=state.state_sha256,
            representation_sha256=state.representation_sha256,
        )
        # Truth is opened only after the raw prediction artifact is durable.
        targets, names, subjects = protocol.truth.join(
            [item.trial_id for item in session.evaluation]
        )
        class_count = 6 + 2 * int(session.session)
        metrics = strict_three_layer_metrics(
            targets,
            raw_predictions,
            class_count=class_count,
            old_class_count=6,
            seen_class_count_before_session=6 + 2 * (int(session.session) - 1),
            registered_class_ids=tuple(entry.registry_id for entry in state.entries),
        )
        metrics.update({
            "session": int(session.session),
            "raw_predictions_path": raw_path.name,
            "raw_predictions_sha256": raw_hash,
            "raw_predictions_frozen_before_truth_join": True,
            "registry_state_sha256": state.state_sha256,
            "previous_registry_state_sha256": state.previous_state_sha256,
            "representation_sha256": state.representation_sha256,
            "expected_class_count": class_count,
            "registered_class_count": len(state.entries),
            "registered_novel_count": len(state.entries) - 6,
            "registered_this_session": list(update.registered_ids),
            "unknown_buffer_size": len(state.unknown_buffer),
            "incoming_codebook_usage": codebook_usage(incoming_primitive),
            "evaluation_codebook_usage": codebook_usage(evaluation_primitive),
        })
        write_json(output / f"metrics_session_{session.session}.json", metrics)
        primary = metrics["layers"]["old_fixed_novel_hungarian"]
        metric_rows.append({
            "profile": PROFILE,
            "fold": int(args.fold),
            "seed": int(args.seed),
            "session": int(session.session),
            "all_accuracy": primary["all_accuracy"],
            "old_accuracy": primary["old_accuracy"],
            "new_accuracy": primary["new_accuracy"],
            "h_score": primary["h_score"],
            "macro_f1": primary["macro_f1"],
            "unknown_fraction": primary["unknown_prediction_fraction"],
            "registry_k": len(state.entries),
            "registered_this_session": len(update.registered_ids),
            "unknown_buffer_size": len(state.unknown_buffer),
            "used_primitive_k": metrics["evaluation_codebook_usage"]["used_k"],
            "representation_sha256": state.representation_sha256,
            "registry_state_sha256": state.state_sha256,
        })
        write_csv(output / f"predictions_session_{session.session}.csv", [
            {
                "trial_id": int(trial.trial_id),
                "subject_id": int(subject),
                "activity_label": int(target),
                "activity_name": str(name),
                "raw_registry_prediction": int(prediction),
                "primary_aligned_prediction": int(aligned),
            }
            for trial, subject, target, name, prediction, aligned in zip(
                session.evaluation,
                subjects,
                targets,
                names,
                raw_predictions,
                primary["aligned_predictions"],
            )
        ])
        _append_jsonl(
            trajectory_path,
            _trajectory_rows(session.session, "incoming", session.incoming, incoming_primitive)
            + _trajectory_rows(
                session.session, "evaluation", session.evaluation, evaluation_primitive, raw_predictions
            ),
        )

    final_encoder_state = {key: value.detach().cpu() for key, value in encoder.state_dict().items()}
    from experiments.motion_primitive.motion_checkpoint import motion_state_dict_sha256
    if motion_state_dict_sha256(final_encoder_state) != initial_encoder_hash:
        raise RuntimeError("A2 encoder parameters or BatchNorm buffers changed online.")
    final_frozen_hashes = {
        "representation_sha256": state.representation_sha256,
        "encoder_state_sha256": initial_encoder_hash,
        "codebook_state_sha256": codebook.state_sha256,
        "descriptor_state_sha256": descriptor.state_sha256,
        "old_anchor_sha256": state.old_anchor_sha256,
    }
    if final_frozen_hashes != initial_frozen_hashes:
        raise RuntimeError("A frozen Online component changed.")
    write_csv(output / "online_runs.csv", metric_rows)
    final_row = metric_rows[-1]
    summary = {
        "schema": ONLINE_SCHEMA,
        "profile": PROFILE,
        "fold": int(args.fold),
        "seed": int(args.seed),
        "session_count": len(metric_rows),
        "primary_layer": "old_fixed_novel_hungarian",
        "final_session": final_row,
        "all_sessions": metric_rows,
        "frozen_hashes_initial": initial_frozen_hashes,
        "frozen_hashes_final": final_frozen_hashes,
        "final_registry_state_sha256": state.state_sha256,
        "final_registry_k": len(state.entries),
        "unresolved_trial_count": len(state.unknown_buffer),
        "online_activity_labels_used_by_learner": False,
        "test_labels_used_only_for_scoring": True,
    }
    write_json(output / "online_summary.json", summary)
    artifact_hashes = {
        name: sha256_file(output / name)
        for name in _output_artifacts()
    }
    complete = {
        **summary,
        "run_identity_sha256": run_identity["identity_sha256"],
        "online_summary_sha256": sha256_file(output / "online_summary.json"),
        "artifact_sha256": artifact_hashes,
        "complete": True,
    }
    write_json(complete_path, complete)
    LOGGER.info(
        "completed profile=%s fold=%d seed=%d all=%.4f old=%.4f new=%.4f H=%.4f registry_k=%d",
        PROFILE,
        int(args.fold),
        int(args.seed),
        final_row["all_accuracy"],
        final_row["old_accuracy"],
        final_row["new_accuracy"],
        final_row["h_score"],
        final_row["registry_k"],
    )
    return complete


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Strict three-session frozen-trajectory Online CGCD.")
    parser.add_argument("--offline-run-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--profile", default=PROFILE, choices=(PROFILE,))
    parser.add_argument("--fold", type=int, required=True, choices=range(1, 8))
    parser.add_argument("--seed", type=int, required=True)
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
