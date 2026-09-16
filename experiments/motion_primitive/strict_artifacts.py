"""Strict loading, encoding, and persistence for the frozen HHR route."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from experiments.motion_primitive.frozen_e0_state import (
    DescriptorTransform,
    FrozenE0Codebook,
    WindowTrial,
)
from experiments.motion_primitive.motion_checkpoint import (
    motion_state_dict_sha256,
    validate_motion_encoder_checkpoint_integrity,
)
from experiments.motion_primitive.motion_encoder import MotionPrimitiveEncoder
from experiments.motion_primitive.strict_protocol import SensorTrial, SubjectSplit, sha256_file


A2_REQUIRED_WEIGHTS = {
    "window_augmentation": 0.0,
    "noncollapse": 0.05,
    "changepoint": 1.0,
    "content_boundary_alignment": 0.1,
    "temporal_prediction": 0.5,
    "trial_auxiliary": 0.1,
    "cross_subject": 0.0,
}


def jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [jsonable(item) for item in value]
    return value


def write_json(path: str | Path, payload: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".tmp")
    temporary.write_text(
        json.dumps(jsonable(payload), ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    temporary.replace(target)


def write_csv(path: str | Path, rows: Sequence[Mapping[str, Any]]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(str(key))
    temporary = target.with_name(target.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows([{key: jsonable(row.get(key, "")) for key in fields} for row in rows])
    temporary.replace(target)


def _load_torch(path: Path) -> Mapping[str, Any]:
    try:
        value = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        value = torch.load(path, map_location="cpu")
    if not isinstance(value, Mapping):
        raise TypeError(f"Checkpoint {path} is not a mapping.")
    return value


def _exact_float_mapping(observed: Mapping[str, Any], expected: Mapping[str, float]) -> None:
    if set(observed) != set(expected):
        raise RuntimeError(
            f"A2 loss keys differ: observed={sorted(observed)}, expected={sorted(expected)}."
        )
    mismatched = {
        key: (observed[key], expected[key])
        for key in expected
        if not np.isclose(float(observed[key]), float(expected[key]), rtol=0.0, atol=1e-12)
    }
    if mismatched:
        raise RuntimeError(f"A2 loss weights differ from the registered route: {mismatched}.")


def load_frozen_a2_encoder(
    checkpoint_path: str | Path,
    *,
    expected_npz_sha256: str,
    expected_fold: int,
    expected_seed: int,
    expected_split: SubjectSplit,
    expected_window_size: int = 256,
    allow_smoke: bool = False,
) -> tuple[MotionPrimitiveEncoder, dict[str, Any]]:
    """Fail closed unless a checkpoint is the exact registered A2 member."""

    path = Path(checkpoint_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    checkpoint = dict(_load_torch(path))
    state = validate_motion_encoder_checkpoint_integrity(checkpoint)
    config = checkpoint.get("resolved_training_config")
    metadata = checkpoint.get("experiment_metadata")
    split_audit = checkpoint.get("split_audit")
    architecture = checkpoint.get("architecture")
    selection = checkpoint.get("selection")
    for name, value in (("resolved_training_config", config), ("experiment_metadata", metadata),
                        ("split_audit", split_audit), ("architecture", architecture),
                        ("selection", selection)):
        if not isinstance(value, Mapping):
            raise RuntimeError(f"A2 checkpoint lacks {name} metadata.")
    if str(config.get("ablation_profile", "")).upper() != "A2":
        raise RuntimeError("Frozen route requires an A2 motion encoder checkpoint.")
    if config.get("window_aug_consistency") != "none":
        raise RuntimeError("Registered A2 must not contain the InfoNCE window loss.")
    if config.get("backbone_bn_policy") != "frozen":
        raise RuntimeError("Registered A2 requires frozen BatchNorm running statistics.")
    _exact_float_mapping(config.get("loss_weights", {}), A2_REQUIRED_WEIGHTS)
    if path.name != "motion_encoder_final.pt":
        raise RuntimeError("Registered A2 must be loaded from motion_encoder_final.pt.")
    required_selection = {
        "policy": "final_epoch",
        "file_role": "canonical_final",
        "outer_test_queries": 0,
    }
    for key, expected in required_selection.items():
        if selection.get(key) != expected:
            raise RuntimeError(
                f"Registered A2 selection field {key!r} differs: "
                f"{selection.get(key)!r} != {expected!r}."
            )
    selected_epoch = int(selection.get("selected_epoch_1based", -1))
    completed_epochs = int(selection.get("completed_epochs", -1))
    if selected_epoch < 1 or selected_epoch != completed_epochs:
        raise RuntimeError("Registered A2 is not the completed final training epoch.")
    exact_architecture = {
        "in_channels": 6,
        "backbone_dim": 256,
        "base_channels": 64,
        "backbone_layers": [2, 2, 2],
        "backbone_dropout": 0.0,
        "segmentation_dim": 256,
        "segmentation_residual": True,
        "content_dim": 256,
        "content_residual": True,
        "augmentation_dim": 128,
        "projection_hidden_dim": 256,
        "num_classes": 6,
        "trial_hidden_dim": 128,
        "trial_peak_quantile": 0.9,
        "trial_dropout": 0.0,
        "predictor_hidden_dim": None,
    }
    if dict(architecture) != exact_architecture:
        raise RuntimeError(
            "A2 architecture differs from the registered ResNet1D/content-residual route."
        )
    expected_subjects = {
        "uschad_train_subjects": list(expected_split.train),
        "offline_val_subjects": list(expected_split.validation),
        "uschad_test_subjects": list(expected_split.outer_test),
    }
    for key, expected in expected_subjects.items():
        observed = sorted(int(value) for value in metadata.get(key, []))
        if observed != sorted(expected):
            raise RuntimeError(f"A2 {key} differs: {observed} != {sorted(expected)}.")
    if int(metadata.get("uschad_cv_fold", -1)) != int(expected_fold):
        raise RuntimeError("A2 checkpoint fold differs from the requested member.")
    if int(metadata.get("motion_encoder_seed", metadata.get("seed", -1))) != int(expected_seed):
        raise RuntimeError("A2 checkpoint seed differs from the requested member.")
    if int(metadata.get("uschad_window_size", -1)) != int(expected_window_size):
        raise RuntimeError("A2 checkpoint window size differs from the registered E0 grid.")
    observed_npz_hash = str(checkpoint.get("npz_sha256", ""))
    if observed_npz_hash != str(expected_npz_sha256):
        raise RuntimeError("A2 checkpoint NPZ SHA256 differs from the current dataset.")
    if metadata.get("outer_test_used_during_encoder_training") is not False:
        raise RuntimeError("A2 checkpoint does not prove zero outer-test use.")
    if int(split_audit.get("outer_test_sensor_windows_selected", -1)) != 0:
        raise RuntimeError("A2 split audit selected outer-test sensor windows.")
    if int(split_audit.get("outer_test_model_forward_calls", -1)) != 0:
        raise RuntimeError("A2 split audit queried outer-test windows.")
    if bool(metadata.get("smoke_test", False)) and not bool(allow_smoke):
        raise RuntimeError("A smoke-test A2 checkpoint cannot enter a formal run.")
    model = MotionPrimitiveEncoder(**dict(architecture))
    incompatible = model.load_state_dict(state, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(f"A2 state is incompatible: {incompatible}.")
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    audit = {
        "checkpoint_path": str(path),
        "checkpoint_sha256": sha256_file(path),
        "model_state_dict_sha256": motion_state_dict_sha256(state),
        "profile": "A2",
        "fold": int(expected_fold),
        "seed": int(expected_seed),
        "window_size": int(expected_window_size),
        "content_role": "codebook_input",
        "segmentation_role_used_by_e0": False,
        "outer_test_queries": 0,
    }
    return model, audit


@torch.inference_mode()
def encode_sensor_trials(
    encoder: MotionPrimitiveEncoder,
    trials: Sequence[SensorTrial],
    *,
    device: torch.device,
    batch_size: int = 512,
) -> list[WindowTrial]:
    """Encode only A2 ``content`` rows; E0 never invokes the boundary head."""

    if int(batch_size) < 1:
        raise ValueError("Encoding batch size must be positive.")
    ordered = list(trials)
    if not ordered:
        return []
    rows = np.concatenate([np.asarray(item.windows, dtype=np.float32) for item in ordered])
    output = np.empty((len(rows), int(encoder.content_dim)), dtype=np.float32)
    encoder = encoder.to(device).eval()
    for begin in range(0, len(rows), int(batch_size)):
        end = min(begin + int(batch_size), len(rows))
        batch = torch.from_numpy(rows[begin:end]).to(device=device)
        # ``encode_windows`` returns independent roles.  Accessing only content
        # makes it impossible for the learned segmentation head to define E0.
        values = encoder.encode_windows(batch)["content"]
        output[begin:end] = values.detach().cpu().numpy().astype(np.float32)
    if not np.all(np.isfinite(output)):
        raise RuntimeError("A2 content encoder produced non-finite values.")
    result: list[WindowTrial] = []
    cursor = 0
    for trial in ordered:
        count = len(trial.windows)
        result.append(WindowTrial(
            trial_id=int(trial.trial_id),
            subject_id=int(trial.subject_id),
            window_starts=np.asarray(trial.window_starts, dtype=np.int64).copy(),
            raw_windows=np.asarray(trial.raw_windows, dtype=np.float32).copy(),
            content_embeddings=output[cursor : cursor + count].copy(),
        ).validate(window_size=int(trial.windows.shape[-1])))
        cursor += count
    if cursor != len(output):
        raise RuntimeError("A2 encoded-window regrouping lost rows.")
    return result


def save_e0_codebook(path: str | Path, state: FrozenE0Codebook) -> dict[str, Any]:
    state.validate()
    target = Path(path)
    np.savez_compressed(
        target,
        pca_mean=state.pca_mean,
        pca_components=state.pca_components,
        cluster_centers=state.cluster_centers,
    )
    metadata = {
        "schema": "hhr_frozen_e0_codebook_v1",
        "primitive_num": state.primitive_num,
        "pca_dim": state.pca_dim,
        "input_dim": state.input_dim,
        "seed": state.seed,
        "fit_trial_count": state.fit_trial_count,
        "fit_window_count": state.fit_window_count,
        "fit_subject_count": state.fit_subject_count,
        "fit_trial_ids_sha256": state.fit_trial_ids_sha256,
        "fit_used_k": state.fit_used_k,
        "inertia": state.inertia,
        "iterations": state.iterations,
        "sklearn_version": state.sklearn_version,
        "state_sha256": state.state_sha256,
        "artifact_sha256": sha256_file(target),
        "fit_scope": "offline_train_subjects_old6_only",
        "online_mutable": False,
    }
    write_json(target.with_suffix(".json"), metadata)
    return metadata


def load_e0_codebook(path: str | Path) -> FrozenE0Codebook:
    target = Path(path).expanduser().resolve()
    metadata = json.loads(target.with_suffix(".json").read_text(encoding="utf-8"))
    if metadata.get("schema") != "hhr_frozen_e0_codebook_v1":
        raise RuntimeError("Unexpected E0 codebook schema.")
    if metadata.get("artifact_sha256") != sha256_file(target):
        raise RuntimeError("E0 codebook artifact SHA256 mismatch.")
    with np.load(target, allow_pickle=False) as archive:
        state = FrozenE0Codebook(
            primitive_num=metadata["primitive_num"],
            pca_dim=metadata["pca_dim"],
            input_dim=metadata["input_dim"],
            pca_mean=np.asarray(archive["pca_mean"]),
            pca_components=np.asarray(archive["pca_components"]),
            cluster_centers=np.asarray(archive["cluster_centers"]),
            seed=metadata["seed"],
            fit_trial_count=metadata["fit_trial_count"],
            fit_window_count=metadata["fit_window_count"],
            fit_subject_count=metadata["fit_subject_count"],
            fit_trial_ids_sha256=metadata["fit_trial_ids_sha256"],
            fit_used_k=metadata["fit_used_k"],
            inertia=metadata["inertia"],
            iterations=metadata["iterations"],
            sklearn_version=metadata["sklearn_version"],
        ).validate()
    if state.state_sha256 != metadata.get("state_sha256"):
        raise RuntimeError("E0 codebook numerical state SHA256 mismatch.")
    return state


def save_descriptor_transform(path: str | Path, state: DescriptorTransform) -> dict[str, Any]:
    state.validate()
    target = Path(path)
    np.savez_compressed(
        target,
        keep_columns=state.keep_columns,
        mean=state.mean,
        scale=state.scale,
        pca_mean=np.asarray([], dtype=np.float64) if state.pca_mean is None else state.pca_mean,
        pca_components=np.empty((0, 0), dtype=np.float64) if state.pca_components is None else state.pca_components,
        schema_names=np.asarray(state.schema_names),
    )
    metadata = {
        "schema": "hhr_frozen_state_descriptor_transform_v1",
        "raw_dim": len(state.schema_names),
        "kept_dim": len(state.keep_columns),
        "output_dim": state.output_dim,
        "raw_schema_sha256": state.schema_sha256,
        "fit_trial_ids_sha256": state.fit_trial_ids_sha256,
        "state_sha256": state.state_sha256,
        "artifact_sha256": sha256_file(target),
        "fit_scope": "offline_train_subjects_old6_only",
        "l2_after_pca": True,
        "online_mutable": False,
    }
    write_json(target.with_suffix(".json"), metadata)
    return metadata


def load_descriptor_transform(path: str | Path) -> DescriptorTransform:
    target = Path(path).expanduser().resolve()
    metadata = json.loads(target.with_suffix(".json").read_text(encoding="utf-8"))
    if metadata.get("schema") != "hhr_frozen_state_descriptor_transform_v1":
        raise RuntimeError("Unexpected descriptor transform schema.")
    if metadata.get("artifact_sha256") != sha256_file(target):
        raise RuntimeError("Descriptor transform artifact SHA256 mismatch.")
    with np.load(target, allow_pickle=False) as archive:
        pca_mean = np.asarray(archive["pca_mean"])
        components = np.asarray(archive["pca_components"])
        state = DescriptorTransform(
            keep_columns=np.asarray(archive["keep_columns"]),
            mean=np.asarray(archive["mean"]),
            scale=np.asarray(archive["scale"]),
            pca_mean=None if not len(pca_mean) else pca_mean,
            pca_components=None if not len(components) else components,
            schema_names=tuple(str(value) for value in archive["schema_names"].tolist()),
            fit_trial_ids_sha256=metadata["fit_trial_ids_sha256"],
        ).validate()
    if state.state_sha256 != metadata.get("state_sha256"):
        raise RuntimeError("Descriptor transform numerical state SHA256 mismatch.")
    return state


def representation_sha256(
    encoder_state_sha256: str,
    codebook: FrozenE0Codebook,
    descriptor: DescriptorTransform,
) -> str:
    value = {
        "route": "frozen_a2_e0_state_k32_v1",
        "encoder_state_sha256": str(encoder_state_sha256),
        "codebook_state_sha256": codebook.state_sha256,
        "descriptor_state_sha256": descriptor.state_sha256,
    }
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


__all__ = [
    "A2_REQUIRED_WEIGHTS",
    "encode_sensor_trials",
    "jsonable",
    "load_descriptor_transform",
    "load_e0_codebook",
    "load_frozen_a2_encoder",
    "representation_sha256",
    "save_descriptor_transform",
    "save_e0_codebook",
    "write_csv",
    "write_json",
]
