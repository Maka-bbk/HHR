"""Train the independent motion-primitive encoder used by the feasibility study.

This entry point is intentionally separate from ``train_happy.py``.  It uses
only old-class trials from the source checkpoint's training subjects for
optimisation and only old-class trials from its validation subjects for model
selection.  Outer-test sensor windows are never selected or encoded here.

The training unit is a complete observed trial (the span covered by the NPZ
windows).  Stored normalised windows are first reconstructed into one raw
sensor stream, trial augmentation is then applied in raw physical units, and
the resulting stream is finally normalised with statistics fitted on the
current fold's train-subject/old-class windows before it is sliced again.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import logging
import math
import os
import random
import re
import sys
import tempfile
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Optional, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from experiments.motion_primitive.motion_augmentation import (
    MotionAugmentationConfig,
    make_augmented_trial_pair,
    normalize_and_slice_windows,
)
from experiments.motion_primitive.motion_encoder import (
    MotionPrimitiveEncoder,
    batched_feature_change_scores,
    boundary_loss_terms,
    masked_temporal_prediction_loss,
    symmetric_info_nce,
    variance_covariance_terms,
)
from experiments.motion_primitive.motion_checkpoint import (
    motion_state_dict_sha256,
    validate_motion_encoder_checkpoint_integrity,
)
from experiments.motion_primitive.raw_changepoint import (
    boundary_candidate_masks,
    consensus_boundary_masks,
    fit_trial_equal_robust_component_scaler,
    fit_trial_equal_score_thresholds,
    parse_raw_scales,
    raw_component_names,
    raw_formula_metadata,
    raw_multiscale_boundary_components,
    transform_raw_boundary_components,
)
from models.resnet1d import ResNet1D


LOGGER = logging.getLogger("motion_encoder_training")
CHECKPOINT_TYPE = "motion_primitive_encoder"
SCHEMA_VERSION = 1
RUN_IDENTITY_SCHEMA = "hhr_motion_encoder_training_identity_v1"
EXPECTED_CHANNELS = ("acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z")
KNOWN_LABEL_ANOMALIES = (
    {
        "subject_id": 14,
        "activity_label_1based": 3,
        "trial_number": 2,
        "source_key": "Subject14/a3t2.mat",
        "reason": "filename label conflicts with the MAT-internal activity metadata",
    },
)
ARCHITECTURE_FIELDS = (
    "in_channels",
    "backbone_dim",
    "base_channels",
    "backbone_layers",
    "backbone_dropout",
    "segmentation_dim",
    "segmentation_residual",
    "content_dim",
    "content_residual",
    "augmentation_dim",
    "projection_hidden_dim",
    "num_classes",
    "trial_hidden_dim",
    "trial_peak_quantile",
    "trial_dropout",
    "predictor_hidden_dim",
)


@dataclass
class TrialExample:
    """One ordered, raw-domain trial and its fixed clean/source supervision."""

    trial_id: int
    subject_id: int
    label: int
    label_1based: int
    trial_number: int
    starts: torch.Tensor
    raw_trial: torch.Tensor
    clean_windows: torch.Tensor
    source_scores: Optional[torch.Tensor] = None
    raw_scores: Optional[torch.Tensor] = None
    stable_mask: Optional[torch.Tensor] = None
    change_mask: Optional[torch.Tensor] = None

    @property
    def length(self) -> int:
        return int(self.clean_windows.shape[0])


def _jsonable(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    return value


def _write_json(path: Path, value) -> None:
    path.write_text(
        json.dumps(_jsonable(value), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while True:
            chunk = stream.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _state_dict_sha256(state_dict: dict[str, torch.Tensor]) -> str:
    """Hash keys, dtypes, shapes and exact contiguous tensor bytes."""

    digest = hashlib.sha256()
    for key in sorted(state_dict):
        tensor = state_dict[key]
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"State value {key!r} is not a tensor.")
        value = tensor.detach().cpu().contiguous()
        digest.update(key.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def _implementation_fingerprint() -> dict:
    """Fingerprint encoder-defining source because this workspace has no Git."""

    relative_paths = (
        "experiments/motion_primitive/train_motion_encoder.py",
        "experiments/motion_primitive/motion_encoder.py",
        "experiments/motion_primitive/motion_augmentation.py",
        "experiments/motion_primitive/raw_changepoint.py",
        "models/resnet1d.py",
    )
    files = {
        relative: _sha256_file(PROJECT_ROOT / relative)
        for relative in relative_paths
    }
    digest = hashlib.sha256()
    for relative, file_hash in sorted(files.items()):
        digest.update(relative.encode("utf-8"))
        digest.update(file_hash.encode("ascii"))
    return {
        "algorithm": "sha256_path_and_file_sha256_v1",
        "files": files,
        "combined_sha256": digest.hexdigest(),
    }


def _load_checkpoint(path: Path) -> dict:
    try:
        value = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        value = torch.load(path, map_location="cpu")
    if not isinstance(value, dict):
        raise TypeError(f"Checkpoint must be a dictionary, got {type(value).__name__}.")
    return value


def _normalise_path_text(value: str) -> str:
    """Accept native Windows paths and old WSL /mnt/<drive>/ paths."""

    text = str(value).strip()
    match = re.fullmatch(r"/mnt/([a-zA-Z])/(.*)", text)
    if match and os.name == "nt":
        return f"{match.group(1).upper()}:/{match.group(2)}"
    return text


def _resolve_input_path(value: str, *, relative_to: Path = PROJECT_ROOT) -> Path:
    text = _normalise_path_text(value)
    candidate = Path(text).expanduser()
    if not candidate.is_absolute():
        candidate = relative_to / candidate
    return candidate.resolve()


def _parse_layers(value: str | Sequence[int]) -> list[int]:
    if isinstance(value, str):
        tokens = [token for token in re.split(r"[\s,]+", value.strip()) if token]
        layers = [int(token) for token in tokens]
    else:
        layers = [int(item) for item in value]
    if len(layers) != 3 or any(item < 1 for item in layers):
        raise ValueError("backbone_layers must contain exactly three positive integers.")
    return layers


def _subject_list(metadata: dict, key: str) -> list[int]:
    value = metadata.get(key)
    if not isinstance(value, (tuple, list)) or not value:
        raise RuntimeError(f"Source checkpoint metadata requires a non-empty {key!r} list.")
    result = sorted(set(int(item) for item in value))
    if len(result) != len(value):
        raise RuntimeError(f"Source checkpoint metadata {key!r} contains duplicates.")
    return result


def _validate_source_metadata(metadata: dict, old_class_count: int) -> dict:
    if not isinstance(metadata, dict):
        raise RuntimeError(
            "The legacy source checkpoint lacks experiment_metadata; a verified "
            "subject-disjoint train/validation/test split is required."
        )
    train_subjects = _subject_list(metadata, "uschad_train_subjects")
    val_subjects = _subject_list(metadata, "offline_val_subjects")
    test_subjects = _subject_list(metadata, "uschad_test_subjects")
    groups = {"train": set(train_subjects), "validation": set(val_subjects), "test": set(test_subjects)}
    for left, right in (("train", "validation"), ("train", "test"), ("validation", "test")):
        overlap = sorted(groups[left] & groups[right])
        if overlap:
            raise RuntimeError(f"Source split leakage: {left}/{right} overlap={overlap}.")
    if int(old_class_count) < 2:
        raise ValueError("old_class_count must be at least two.")
    required_arch = ("har_in_channels", "har_feat_dim", "har_base_channels")
    missing = [key for key in required_arch if key not in metadata]
    if missing:
        raise RuntimeError(f"Source checkpoint metadata lacks architecture fields: {missing}.")
    return {
        "train_subjects": train_subjects,
        "validation_subjects": val_subjects,
        "test_subjects": test_subjects,
        "subject_overlap": {
            "train_validation": [],
            "train_test": [],
            "validation_test": [],
        },
        "old_class_ids_0based": list(range(int(old_class_count))),
    }


def _validate_source_npz_binding(metadata: dict, requested_npz_path: Path) -> dict:
    """Bind a legacy window checkpoint to the NPZ that trained it.

    Legacy Happy checkpoints do not contain a data-file digest. Their recorded
    NPZ path is therefore the only available guard against combining, for
    example, an original-label backbone with the old/new-swapped dataset (both
    of which have the same channel and window shapes).
    """

    recorded = str(metadata.get("uschad_npz_path", "")).strip()
    if not recorded:
        raise RuntimeError(
            "Source checkpoint metadata lacks uschad_npz_path; its training "
            "dataset cannot be bound to the requested motion-encoder run."
        )
    requested_path = Path(requested_npz_path).expanduser().resolve()
    recorded_sha = str(metadata.get("uschad_npz_sha256", "")).strip().lower()
    if recorded_sha:
        requested_sha = _sha256_file(requested_path)
        if len(recorded_sha) != 64 or any(char not in "0123456789abcdef" for char in recorded_sha):
            raise RuntimeError("Source checkpoint uschad_npz_sha256 is malformed.")
        if recorded_sha != requested_sha:
            raise RuntimeError(
                "Source checkpoint/NPZ SHA-256 mismatch: "
                f"recorded={recorded_sha}, requested={requested_sha}."
            )
        recorded_path = _resolve_input_path(recorded)
        return {
            "source_checkpoint_npz_path_recorded": str(recorded),
            "source_checkpoint_npz_path_resolved": str(recorded_path),
            "requested_npz_path_resolved": str(requested_path),
            "source_checkpoint_npz_sha256": recorded_sha,
            "requested_npz_sha256": requested_sha,
            "source_checkpoint_npz_identity_matches_requested": True,
            "identity_authority": "sha256",
            "path_matches": recorded_path == requested_path,
        }
    recorded_path = _resolve_input_path(recorded)
    if recorded_path != requested_path:
        raise RuntimeError(
            "Source checkpoint/NPZ identity mismatch: the window encoder records "
            f"{recorded_path}, but the motion encoder requested {requested_path}. "
            "Train or select a fold/seed-matched window encoder on this exact NPZ."
        )
    return {
        "source_checkpoint_npz_path_recorded": str(recorded),
        "source_checkpoint_npz_path_resolved": str(recorded_path),
        "requested_npz_path_resolved": str(requested_path),
        "source_checkpoint_npz_path_matches_requested": True,
        "identity_authority": "legacy_path_only",
        "identity_limit": (
            "legacy source checkpoint records a path but no NPZ digest; downstream "
            "motion checkpoint binds the requested NPZ by SHA256"
        ),
    }


def _extract_backbone_state(
    source_checkpoint: dict,
    target_backbone: ResNet1D,
) -> tuple[dict[str, torch.Tensor], dict[str, str]]:
    source_state = source_checkpoint.get("model", source_checkpoint.get("model_state_dict"))
    if not isinstance(source_state, dict):
        raise TypeError("Source checkpoint has no dictionary model/model_state_dict.")
    duplicate = source_checkpoint.get("model_state_dict")
    if duplicate is not None:
        if not isinstance(duplicate, dict) or set(duplicate) != set(source_state):
            raise RuntimeError("Window checkpoint model aliases differ in key set.")
        if any(not torch.equal(source_state[key].detach().cpu(), duplicate[key].detach().cpu()) for key in source_state):
            raise RuntimeError("Window checkpoint model aliases differ in tensor values.")
    recorded_hash = str(source_checkpoint.get("model_state_dict_sha256", "")).strip()
    if recorded_hash:
        observed_hash = motion_state_dict_sha256(source_state)
        if observed_hash != recorded_hash:
            raise RuntimeError(
                "Window checkpoint state hash mismatch: "
                f"recorded={recorded_hash}, observed={observed_hash}."
            )
    prefixes = ("0.window_encoder.", "window_encoder.", "0.", "backbone.", "")
    extracted: dict[str, torch.Tensor] = {}
    mapping: dict[str, str] = {}
    for target_key, target_value in target_backbone.state_dict().items():
        matches = [prefix + target_key for prefix in prefixes if prefix + target_key in source_state]
        if not matches:
            raise RuntimeError(f"Source checkpoint lacks backbone tensor {target_key!r}.")
        source_key = matches[0]
        source_value = source_state[source_key]
        if not isinstance(source_value, torch.Tensor):
            raise TypeError(f"Source checkpoint value {source_key!r} is not a tensor.")
        if tuple(source_value.shape) != tuple(target_value.shape):
            raise RuntimeError(
                f"Backbone tensor shape mismatch for {source_key}: "
                f"source={tuple(source_value.shape)}, target={tuple(target_value.shape)}."
            )
        extracted[target_key] = source_value.detach().cpu().clone()
        mapping[target_key] = source_key
    # Strict here refers to every tensor of the complete target backbone.  Extra
    # legacy projection/classifier tensors are deliberately outside this model.
    target_backbone.load_state_dict(extracted, strict=True)
    return extracted, mapping


def _anomaly_mask(
    subject_ids: np.ndarray,
    labels_1based: np.ndarray,
    trial_numbers: np.ndarray,
) -> np.ndarray:
    result = np.zeros(len(subject_ids), dtype=bool)
    for anomaly in KNOWN_LABEL_ANOMALIES:
        result |= (
            (subject_ids == int(anomaly["subject_id"]))
            & (labels_1based == int(anomaly["activity_label_1based"]))
            & (trial_numbers == int(anomaly["trial_number"]))
        )
    return result


def _reconstruct_raw_trial(
    raw_windows: np.ndarray,
    starts: np.ndarray,
    window_indices: np.ndarray,
) -> tuple[torch.Tensor, torch.Tensor, float]:
    order = np.argsort(starts, kind="stable")
    starts = np.asarray(starts[order], dtype=np.int64)
    window_indices = np.asarray(window_indices[order], dtype=np.int64)
    raw_windows = np.asarray(raw_windows[order], dtype=np.float32)
    if starts.ndim != 1 or len(starts) == 0:
        raise RuntimeError("A trial must contain at least one window.")
    if int(starts[0]) != 0 or (len(starts) > 1 and np.any(np.diff(starts) <= 0)):
        raise RuntimeError(f"Trial window starts must begin at zero and increase: {starts.tolist()}.")
    if not np.array_equal(window_indices, np.arange(len(window_indices), dtype=np.int64)):
        raise RuntimeError(
            "Trial window_indices must be complete and ordered after sorting by start; "
            f"got {window_indices.tolist()}."
        )
    window_size = int(raw_windows.shape[2])
    temporal_length = int(starts[-1]) + window_size
    accumulator = np.zeros((raw_windows.shape[1], temporal_length), dtype=np.float64)
    counts = np.zeros(temporal_length, dtype=np.int32)
    max_overlap_error = 0.0
    for window, begin in zip(raw_windows, starts):
        begin = int(begin)
        end = begin + window_size
        overlap = counts[begin:end] > 0
        if np.any(overlap):
            existing = accumulator[:, begin:end][:, overlap] / counts[begin:end][overlap][None, :]
            max_overlap_error = max(
                max_overlap_error,
                float(np.max(np.abs(existing - window[:, overlap]))),
            )
        accumulator[:, begin:end] += window.astype(np.float64)
        counts[begin:end] += 1
    if np.any(counts == 0):
        raise RuntimeError("NPZ windows leave a gap inside the observed trial span.")
    scale = max(1.0, float(np.max(np.abs(raw_windows))))
    if max_overlap_error > 5e-5 * scale:
        raise RuntimeError(
            "Overlapping NPZ windows disagree after raw reconstruction: "
            f"max_abs_error={max_overlap_error:.6g}, scale={scale:.6g}."
        )
    reconstructed = (accumulator / counts[None, :]).astype(np.float32)
    return (
        torch.from_numpy(reconstructed),
        torch.from_numpy(starts.copy()).long(),
        max_overlap_error,
    )


def _count_by(values: Iterable[int]) -> dict[str, int]:
    result: dict[str, int] = {}
    for value in values:
        key = str(int(value))
        result[key] = result.get(key, 0) + 1
    return dict(sorted(result.items(), key=lambda item: int(item[0])))


def _trial_identity_sha256(records: Sequence[TrialExample]) -> str:
    text = "\n".join(
        f"{record.trial_id}:{record.subject_id}:{record.label}:{record.trial_number}"
        for record in sorted(records, key=lambda item: item.trial_id)
    )
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _smoke_subsample_trials(
    records: Sequence[TrialExample], maximum: int
) -> list[TrialExample]:
    """Deterministic class/subject round-robin subset for integration smoke runs."""

    maximum = int(maximum)
    if maximum <= 0 or maximum >= len(records):
        return list(records)
    buckets: dict[tuple[int, int], list[TrialExample]] = {}
    for record in records:
        buckets.setdefault((record.label, record.subject_id), []).append(record)
    for values in buckets.values():
        values.sort(key=lambda item: (item.trial_number, item.trial_id))
    selected: list[TrialExample] = []
    keys = sorted(buckets)
    offset = 0
    while len(selected) < maximum:
        progressed = False
        for key in keys:
            if offset < len(buckets[key]):
                selected.append(buckets[key][offset])
                progressed = True
                if len(selected) == maximum:
                    break
        if not progressed:
            break
        offset += 1
    return sorted(selected, key=lambda item: (item.subject_id, item.label, item.trial_number))


def _apply_smoke_limits(
    train_records: Sequence[TrialExample],
    val_records: Sequence[TrialExample],
    split_audit: dict,
    max_train: int,
    max_val: int,
) -> tuple[list[TrialExample], list[TrialExample]]:
    if int(max_train) < 0 or int(max_val) < 0:
        raise ValueError("Smoke trial limits must be non-negative.")
    original = {
        "train_trial_count": len(train_records),
        "validation_trial_count": len(val_records),
        "train_window_count": int(sum(record.length for record in train_records)),
        "validation_window_count": int(sum(record.length for record in val_records)),
    }
    train = _smoke_subsample_trials(train_records, int(max_train))
    validation = _smoke_subsample_trials(val_records, int(max_val))
    smoke = int(max_train) > 0 or int(max_val) > 0
    split_audit["smoke_test"] = smoke
    split_audit["smoke_trial_limits"] = {
        "max_train_trials": int(max_train),
        "max_validation_trials": int(max_val),
    }
    split_audit["pre_smoke_limit_counts"] = original
    split_audit.update(
        {
            "train_window_count": int(sum(record.length for record in train)),
            "validation_window_count": int(sum(record.length for record in validation)),
            "train_trial_count": len(train),
            "validation_trial_count": len(validation),
            "train_trials_by_subject": _count_by(record.subject_id for record in train),
            "validation_trials_by_subject": _count_by(record.subject_id for record in validation),
            "train_trials_by_class_0based": _count_by(record.label for record in train),
            "validation_trials_by_class_0based": _count_by(record.label for record in validation),
            "train_trial_identity_sha256": _trial_identity_sha256(train),
            "validation_trial_identity_sha256": _trial_identity_sha256(validation),
        }
    )
    if not train or not validation:
        raise RuntimeError("Smoke limits produced an empty train or validation split.")
    return train, validation


def _build_records(
    *,
    raw_windows: np.ndarray,
    labels: np.ndarray,
    labels_1based: np.ndarray,
    subject_ids: np.ndarray,
    trial_numbers: np.ndarray,
    trial_ids: np.ndarray,
    window_indices: np.ndarray,
    window_starts: np.ndarray,
    fold_mean: torch.Tensor,
    fold_std: torch.Tensor,
    window_size: int,
) -> tuple[list[TrialExample], float]:
    records: list[TrialExample] = []
    maximum_overlap_error = 0.0
    for trial_id in np.unique(trial_ids):
        positions = np.flatnonzero(trial_ids == trial_id)
        scalar_fields = {
            "label": np.unique(labels[positions]),
            "label_1based": np.unique(labels_1based[positions]),
            "subject": np.unique(subject_ids[positions]),
            "trial_number": np.unique(trial_numbers[positions]),
        }
        inconsistent = {key: value.tolist() for key, value in scalar_fields.items() if len(value) != 1}
        if inconsistent:
            raise RuntimeError(f"Trial {int(trial_id)} has inconsistent metadata: {inconsistent}.")
        raw_trial, starts, overlap_error = _reconstruct_raw_trial(
            raw_windows[positions],
            window_starts[positions],
            window_indices[positions],
        )
        maximum_overlap_error = max(maximum_overlap_error, overlap_error)
        clean = normalize_and_slice_windows(
            raw_trial,
            fold_mean,
            fold_std,
            starts,
            int(window_size),
        )
        records.append(
            TrialExample(
                trial_id=int(trial_id),
                subject_id=int(scalar_fields["subject"][0]),
                label=int(scalar_fields["label"][0]),
                label_1based=int(scalar_fields["label_1based"][0]),
                trial_number=int(scalar_fields["trial_number"][0]),
                starts=starts,
                raw_trial=raw_trial,
                clean_windows=clean,
            )
        )
    records.sort(key=lambda item: (item.subject_id, item.label, item.trial_number))
    return records, maximum_overlap_error


def load_train_validation_trials(
    npz_path: Path,
    split: dict,
    old_class_count: int,
    norm_eps: float,
    anomaly_policy: str,
) -> tuple[list[TrialExample], list[TrialExample], torch.Tensor, torch.Tensor, dict]:
    """Select train/validation old-class windows; never construct a test set."""

    required = {
        "windows",
        "labels",
        "labels_1based",
        "subject_ids",
        "trial_numbers",
        "trial_global_ids",
        "window_indices",
        "window_start_indices",
        "mean",
        "std",
    }
    with np.load(npz_path, allow_pickle=True) as npz:
        missing = required - set(npz.files)
        if missing:
            raise RuntimeError(f"USC-HAD NPZ lacks required fields: {sorted(missing)}.")
        labels = np.asarray(npz["labels"], dtype=np.int64)
        labels_1based = np.asarray(npz["labels_1based"], dtype=np.int64)
        if "original_labels_1based" in npz.files:
            anomaly_labels_1based = np.asarray(
                npz["original_labels_1based"], dtype=np.int64
            )
            anomaly_label_source = "original_labels_1based"
        else:
            anomaly_labels_1based = labels_1based
            anomaly_label_source = "labels_1based"
        subject_ids = np.asarray(npz["subject_ids"], dtype=np.int64)
        trial_numbers = np.asarray(npz["trial_numbers"], dtype=np.int64)
        trial_ids = np.asarray(npz["trial_global_ids"], dtype=np.int64)
        window_indices = np.asarray(npz["window_indices"], dtype=np.int64)
        window_starts = np.asarray(npz["window_start_indices"], dtype=np.int64)
        sample_count = len(labels)
        for name, values in (
            ("labels_1based", labels_1based),
            (anomaly_label_source, anomaly_labels_1based),
            ("subject_ids", subject_ids),
            ("trial_numbers", trial_numbers),
            ("trial_global_ids", trial_ids),
            ("window_indices", window_indices),
            ("window_start_indices", window_starts),
        ):
            if len(values) != sample_count:
                raise RuntimeError(f"NPZ {name} length differs from labels length.")
        if "channel_names" in npz.files:
            channels = tuple(str(item) for item in np.asarray(npz["channel_names"], dtype=object).tolist())
            if channels != EXPECTED_CHANNELS:
                raise RuntimeError(f"Unexpected channel order {channels}; expected {EXPECTED_CHANNELS}.")

        train_mask = np.isin(subject_ids, np.asarray(split["train_subjects"], dtype=np.int64))
        val_mask = np.isin(subject_ids, np.asarray(split["validation_subjects"], dtype=np.int64))
        old_mask = (labels >= 0) & (labels < int(old_class_count))
        train_mask &= old_mask
        val_mask &= old_mask
        anomalies = _anomaly_mask(
            subject_ids, anomaly_labels_1based, trial_numbers
        )
        anomaly_counts = {
            "train_windows_before_policy": int(np.sum(train_mask & anomalies)),
            "validation_windows_before_policy": int(np.sum(val_mask & anomalies)),
        }
        if anomaly_policy == "exclude":
            train_mask &= ~anomalies
            val_mask &= ~anomalies
        elif anomaly_policy != "report":
            raise ValueError("anomaly_policy must be 'report' or 'exclude'.")
        train_indices = np.flatnonzero(train_mask)
        val_indices = np.flatnonzero(val_mask)
        if not len(train_indices) or not len(val_indices):
            raise RuntimeError("Train and validation selections must both be non-empty.")
        if np.intersect1d(train_indices, val_indices).size:
            raise RuntimeError("A sensor window was selected for train and validation.")

        stored_mean = np.asarray(npz["mean"], dtype=np.float32).reshape(-1)
        stored_std = np.asarray(npz["std"], dtype=np.float32).reshape(-1)
        if stored_mean.shape != (6,) or stored_std.shape != (6,) or np.any(stored_std <= 0):
            raise RuntimeError("NPZ mean/std must contain six finite positive-channel statistics.")
        # NPZ is a zip container, so NumPy decompresses its single windows.npy
        # member before row indexing.  Only train/validation rows survive this
        # expression and no outer-test row is exposed to a dataset or model.
        selected_indices = np.concatenate((train_indices, val_indices))
        selected_stored = np.asarray(npz["windows"], dtype=np.float32)[selected_indices]

    if selected_stored.ndim != 3 or selected_stored.shape[1] != 6:
        raise RuntimeError(f"Expected selected windows [N,6,T], got {selected_stored.shape}.")
    if not np.all(np.isfinite(selected_stored)):
        raise RuntimeError("Selected train/validation windows contain non-finite values.")
    window_size = int(selected_stored.shape[2])
    raw_selected = (
        selected_stored * stored_std.reshape(1, 6, 1)
        + stored_mean.reshape(1, 6, 1)
    ).astype(np.float32)
    train_count = len(train_indices)
    raw_train_windows = raw_selected[:train_count]
    fold_mean_np = raw_train_windows.mean(axis=(0, 2), dtype=np.float64).astype(np.float32)
    fold_std_np = raw_train_windows.std(axis=(0, 2), dtype=np.float64).astype(np.float32)
    fold_std_np = np.maximum(fold_std_np, np.float32(norm_eps))
    fold_mean = torch.from_numpy(fold_mean_np)
    fold_std = torch.from_numpy(fold_std_np)

    train_records, train_overlap = _build_records(
        raw_windows=raw_selected[:train_count],
        labels=labels[train_indices],
        labels_1based=labels_1based[train_indices],
        subject_ids=subject_ids[train_indices],
        trial_numbers=trial_numbers[train_indices],
        trial_ids=trial_ids[train_indices],
        window_indices=window_indices[train_indices],
        window_starts=window_starts[train_indices],
        fold_mean=fold_mean,
        fold_std=fold_std,
        window_size=window_size,
    )
    val_records, val_overlap = _build_records(
        raw_windows=raw_selected[train_count:],
        labels=labels[val_indices],
        labels_1based=labels_1based[val_indices],
        subject_ids=subject_ids[val_indices],
        trial_numbers=trial_numbers[val_indices],
        trial_ids=trial_ids[val_indices],
        window_indices=window_indices[val_indices],
        window_starts=window_starts[val_indices],
        fold_mean=fold_mean,
        fold_std=fold_std,
        window_size=window_size,
    )
    observed_subjects = sorted(set(record.subject_id for record in train_records + val_records))
    forbidden = sorted(set(observed_subjects) & set(split["test_subjects"]))
    if forbidden:
        raise RuntimeError(f"Outer-test subjects entered constructed records: {forbidden}.")
    audit = {
        "protocol": "source_checkpoint_split_train_old6_validation_old6_no_outer_test_evaluation",
        "train_subjects": split["train_subjects"],
        "validation_subjects": split["validation_subjects"],
        "outer_test_subjects_metadata_only": split["test_subjects"],
        "sensor_data_subjects_selected": observed_subjects,
        "outer_test_sensor_windows_selected": 0,
        "outer_test_model_forward_calls": 0,
        "train_window_count": int(sum(record.length for record in train_records)),
        "validation_window_count": int(sum(record.length for record in val_records)),
        "train_trial_count": len(train_records),
        "validation_trial_count": len(val_records),
        "train_trials_by_subject": _count_by(record.subject_id for record in train_records),
        "validation_trials_by_subject": _count_by(record.subject_id for record in val_records),
        "train_trials_by_class_0based": _count_by(record.label for record in train_records),
        "validation_trials_by_class_0based": _count_by(record.label for record in val_records),
        "train_trial_identity_sha256": _trial_identity_sha256(train_records),
        "validation_trial_identity_sha256": _trial_identity_sha256(val_records),
        "train_validation_trial_overlap": sorted(
            set(record.trial_id for record in train_records)
            & set(record.trial_id for record in val_records)
        ),
        "old_class_ids_0based": list(range(int(old_class_count))),
        "known_anomaly_policy": anomaly_policy,
        "anomaly_label_source": anomaly_label_source,
        "known_anomalies": list(KNOWN_LABEL_ANOMALIES),
        "known_anomaly_counts": anomaly_counts,
        "normalization": {
            "mode": "fold_train_subjects_old_classes",
            "raw_source": "reconstructed_from_npz_windows_mean_std",
            "stat_window_count": int(train_count),
            "mean": fold_mean_np.tolist(),
            "std": fold_std_np.tolist(),
            "eps": float(norm_eps),
        },
        "raw_trial_scope": "observed_span_through_last_complete_npz_window; final_incomplete_tail_unavailable",
        "window_size_samples": window_size,
        "maximum_raw_overlap_reconstruction_error": float(max(train_overlap, val_overlap)),
    }
    if audit["train_validation_trial_overlap"]:
        raise RuntimeError("A trial was reconstructed into both train and validation.")
    return train_records, val_records, fold_mean, fold_std, audit


@torch.no_grad()
def build_source_pseudo_boundaries(
    source_encoder: ResNet1D,
    train_records: Sequence[TrialExample],
    val_records: Sequence[TrialExample],
    device: torch.device,
    encode_batch_size: int,
    context_windows: int,
    low_quantile: float,
    high_quantile: float,
    anchor_source: str = "raw_frozen_consensus",
    raw_scales: Sequence[float] = (1.0, 2.0, 4.0),
    raw_frequency_bins: int = 16,
    raw_epsilon: float = 1e-8,
    raw_scale_floor: float = 1e-6,
    raw_z_clip: float = 10.0,
) -> dict:
    """Fit train-old-only frozen/raw thresholds and assign pseudo anchors.

    ``raw_frozen_consensus`` requires both independent signals to be high for
    a change anchor and both to be low for a stable anchor.  There is no
    per-trial top-k fallback, so an entire trial may contain no change anchor.
    The EMA teacher used later by temporal prediction is not consulted here.
    """

    if not 0.0 <= float(low_quantile) < float(high_quantile) <= 1.0:
        raise ValueError("CP quantiles must satisfy 0 <= low < high <= 1.")
    if anchor_source not in {"raw_frozen_consensus", "frozen_legacy"}:
        raise ValueError("anchor_source must be raw_frozen_consensus or frozen_legacy.")
    parsed_scales = parse_raw_scales(raw_scales)
    records = list(train_records) + list(val_records)
    lengths = [record.length for record in records]
    flat = torch.cat([record.clean_windows for record in records], dim=0)
    source_encoder = source_encoder.to(device).eval()
    encoded_parts = []
    for begin in range(0, len(flat), int(encode_batch_size)):
        batch = flat[begin : begin + int(encode_batch_size)].to(device=device)
        encoded_parts.append(F.normalize(source_encoder(batch), dim=-1).cpu())
    encoded = torch.cat(encoded_parts, dim=0)
    cursor = 0
    frozen_scores_by_trial: list[np.ndarray] = []
    for index, (record, length) in enumerate(zip(records, lengths)):
        features = encoded[cursor : cursor + length].unsqueeze(0)
        cursor += length
        scores, valid = batched_feature_change_scores(features, int(context_windows))
        score = scores[0][valid[0]].cpu()
        if score.numel() != max(0, record.length - 1):
            raise RuntimeError(f"Frozen-source boundary count mismatch for trial {record.trial_id}.")
        record.source_scores = score
        frozen_scores_by_trial.append(score.numpy().astype(np.float64, copy=False))
    frozen_thresholds = fit_trial_equal_score_thresholds(
        frozen_scores_by_trial[: len(train_records)],
        float(low_quantile),
        float(high_quantile),
    )

    raw_components_by_trial: list[np.ndarray] = []
    raw_scores_by_trial: list[np.ndarray] = []
    raw_calibration = None
    raw_thresholds = None
    if anchor_source == "raw_frozen_consensus":
        component_names = raw_component_names(parsed_scales)
        for record in records:
            components = raw_multiscale_boundary_components(
                record.raw_trial,
                record.starts,
                int(record.clean_windows.shape[-1]),
                scales=parsed_scales,
                frequency_bins=int(raw_frequency_bins),
                epsilon=float(raw_epsilon),
            )
            if len(components) != max(0, record.length - 1):
                raise RuntimeError(f"Raw boundary count mismatch for trial {record.trial_id}.")
            raw_components_by_trial.append(components)
        # This is the only scaler fit.  Validation components are deliberately
        # excluded and are transformed below with this frozen train-old state.
        raw_calibration = fit_trial_equal_robust_component_scaler(
            raw_components_by_trial[: len(train_records)],
            component_names,
            scale_floor=float(raw_scale_floor),
            z_clip=float(raw_z_clip),
        )
        raw_scores_by_trial = [
            transform_raw_boundary_components(components, raw_calibration)
            for components in raw_components_by_trial
        ]
        raw_thresholds = fit_trial_equal_score_thresholds(
            raw_scores_by_trial[: len(train_records)],
            float(low_quantile),
            float(high_quantile),
        )
        for record, score in zip(records, raw_scores_by_trial):
            record.raw_scores = torch.from_numpy(score.astype(np.float32, copy=False))

    trial_diagnostics: list[dict] = []
    for trial_index, record in enumerate(records):
        frozen_score = frozen_scores_by_trial[trial_index]
        if anchor_source == "raw_frozen_consensus":
            assert raw_thresholds is not None
            stable, change, diagnostic = consensus_boundary_masks(
                raw_scores_by_trial[trial_index],
                frozen_score,
                raw_thresholds,
                frozen_thresholds,
                anchor_source,
            )
        else:
            stable, change = boundary_candidate_masks(frozen_score, frozen_thresholds)
            diagnostic = {
                "boundary_count": int(len(frozen_score)),
                "frozen_low_count": int(stable.sum()),
                "frozen_high_count": int(change.sum()),
                "selected_stable_count": int(stable.sum()),
                "selected_change_count": int(change.sum()),
            }
        record.stable_mask = torch.from_numpy(stable.astype(np.bool_, copy=False))
        record.change_mask = torch.from_numpy(change.astype(np.bool_, copy=False))
        diagnostic.update(
            {
                "trial_id": int(record.trial_id),
                "split": "train" if trial_index < len(train_records) else "validation",
            }
        )
        trial_diagnostics.append(diagnostic)

    def coverage(split_name: str) -> dict:
        selected = [item for item in trial_diagnostics if item["split"] == split_name]
        boundary_count = int(sum(item["boundary_count"] for item in selected))
        count_keys = sorted(
            key for key in selected[0] if key.endswith("_count") and key != "boundary_count"
        ) if selected else []
        summary = {
            "trial_count": len(selected),
            "boundary_count": boundary_count,
            "boundary_bearing_trial_count": int(sum(item["boundary_count"] > 0 for item in selected)),
            "trials_without_selected_change": int(sum(item["selected_change_count"] == 0 for item in selected)),
            "trials_without_selected_stable": int(sum(item["selected_stable_count"] == 0 for item in selected)),
            "trials_with_selected_anchor_pair": int(
                sum(
                    item["selected_change_count"] > 0
                    and item["selected_stable_count"] > 0
                    for item in selected
                )
            ),
        }
        for key in count_keys:
            count = int(sum(item[key] for item in selected))
            summary[key] = count
            summary[key.removesuffix("_count") + "_rate"] = (
                count / boundary_count if boundary_count else None
            )
        return summary

    train_coverage = coverage("train")
    validation_coverage = coverage("validation")
    source_encoder.to("cpu")
    return {
        "source": anchor_source,
        "pseudo_anchor_provider": (
            "raw physical-unit kinematics intersected with frozen legacy encoder"
            if anchor_source == "raw_frozen_consensus"
            else "frozen legacy encoder only"
        ),
        "ema_teacher_generates_pseudo_anchors": False,
        "ema_teacher_role": "masked temporal prediction target only",
        "threshold_fit_split": "train_subjects_old_classes_only; validation transform only",
        "context_windows": int(context_windows),
        "low_quantile": float(low_quantile),
        "high_quantile": float(high_quantile),
        "anchor_rules": {
            "raw_frozen_consensus": {
                "change": "raw_score>=raw_high AND raw_score>raw_low AND frozen_score>=frozen_high AND frozen_score>frozen_low",
                "stable": "raw_score<=raw_low AND frozen_score<=frozen_low AND NOT change",
            },
            "frozen_legacy": {
                "change": "frozen_score>=frozen_high AND frozen_score>frozen_low",
                "stable": "frozen_score<=frozen_low AND NOT change",
            },
        },
        "frozen_legacy": {
            "source": "frozen_legacy_source_encoder_clean_fold_normalized_train_old_features",
            "feature_normalization": "per_window_l2_before_local_mean_cosine",
            "score_formula": "1-cosine(mean(L2(window_features)_left_context),mean(L2(window_features)_right_context))",
            "thresholds": frozen_thresholds,
        },
        "raw_kinematic": {
            "enabled": anchor_source == "raw_frozen_consensus",
            "formula": raw_formula_metadata(parsed_scales, int(raw_frequency_bins), float(raw_epsilon)),
            "robust_component_calibration": raw_calibration,
            "thresholds": raw_thresholds,
            "validation_policy": "transform_only_with_train_fitted_component_scaler_and_thresholds",
        },
        "coverage": {"train": train_coverage, "validation": validation_coverage},
        # Preserve the evidence needed to verify that the within-trial ranking
        # term was genuinely active; aggregate stable/change totals alone are
        # insufficient because they could come from disjoint trials.
        "trial_diagnostics": trial_diagnostics,
        # Backward-compatible top-level counts used by logs and older readers.
        "low_threshold": frozen_thresholds["low_threshold"],
        "high_threshold": frozen_thresholds["high_threshold"],
        "train_boundary_count": train_coverage["boundary_count"],
        "train_boundary_bearing_trial_count": train_coverage["boundary_bearing_trial_count"],
        "train_stable_anchor_count": train_coverage["selected_stable_count"],
        "train_change_anchor_count": train_coverage["selected_change_count"],
        "train_trials_without_change_anchor": train_coverage["trials_without_selected_change"],
        "train_trials_with_selected_anchor_pair": train_coverage[
            "trials_with_selected_anchor_pair"
        ],
        "validation_stable_anchor_count": validation_coverage["selected_stable_count"],
        "validation_change_anchor_count": validation_coverage["selected_change_count"],
        "validation_trials_without_change_anchor": validation_coverage["trials_without_selected_change"],
        "validation_trials_with_selected_anchor_pair": validation_coverage[
            "trials_with_selected_anchor_pair"
        ],
        "per_trial_change_anchor_is_forced": False,
    }


def _pad_windows(views: Sequence[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    if not views:
        raise ValueError("Cannot pad an empty trial batch.")
    maximum = max(int(view.shape[0]) for view in views)
    channels, width = map(int, views[0].shape[1:])
    output = views[0].new_zeros((len(views), maximum, channels, width))
    valid = torch.zeros((len(views), maximum), dtype=torch.bool, device=views[0].device)
    for row, view in enumerate(views):
        if view.ndim != 3 or tuple(view.shape[1:]) != (channels, width):
            raise ValueError("Every trial view must have the same [C,T] window shape.")
        length = int(view.shape[0])
        output[row, :length] = view
        valid[row, :length] = True
    return output, valid


def _pad_boundary_masks(
    records: Sequence[TrialExample],
    maximum_windows: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    width = max(0, int(maximum_windows) - 1)
    stable = torch.zeros((len(records), width), dtype=torch.bool)
    change = torch.zeros_like(stable)
    for row, record in enumerate(records):
        if record.stable_mask is None or record.change_mask is None:
            raise RuntimeError("Pseudo-boundary masks were not prepared.")
        boundary_count = max(0, record.length - 1)
        if tuple(record.stable_mask.shape) != (boundary_count,) or tuple(record.change_mask.shape) != (boundary_count,):
            raise RuntimeError(f"Pseudo-boundary shape mismatch for trial {record.trial_id}.")
        stable[row, :boundary_count] = record.stable_mask
        change[row, :boundary_count] = record.change_mask
    return stable, change


def _prediction_mask(
    valid_mask: torch.Tensor,
    ratio: float,
    generator: Optional[torch.Generator],
    deterministic: bool,
) -> torch.Tensor:
    result = torch.zeros_like(valid_mask)
    for row in range(valid_mask.shape[0]):
        length = int(valid_mask[row].sum())
        if length < 3 or ratio <= 0:
            continue
        candidates = torch.arange(1, length - 1)
        count = max(1, min(len(candidates), int(round(length * float(ratio)))))
        if deterministic:
            positions = torch.linspace(0, len(candidates) - 1, steps=count).round().long()
            chosen = candidates[positions]
        else:
            chosen = candidates[torch.randperm(len(candidates), generator=generator)[:count]]
        result[row, chosen] = True
    return result


def _trial_equal_varcov(
    output: dict[str, torch.Tensor],
    valid: torch.Tensor,
    target_std: float,
    variance_weight: float,
    covariance_weight: float,
    windows_per_trial: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Global anti-collapse loss on a trial-balanced cross-trial sample.

    This deliberately does *not* regularise each trial separately.  A valid
    activity such as running-forward may have a nearly constant one-segment
    trajectory; only collapse of the representation across the batch is
    discouraged.  Every trial contributes the same number (at most four by
    default) of uniformly spaced windows.
    """

    if int(windows_per_trial) < 1:
        raise ValueError("windows_per_trial must be positive.")
    common_count = min(
        int(windows_per_trial),
        min(int(valid[row].sum()) for row in range(valid.shape[0])),
    )
    if common_count < 1:
        raise RuntimeError("A trial-balanced noncollapse batch contains an empty trial.")
    # The pretrained content path is intentionally excluded: whitening it on
    # the first step would defeat the identity residual that preserves legacy
    # local geometry.  Segmentation needs an explicit anti-collapse term;
    # content is protected by the frozen initial skip plus prediction/trial and
    # low-weight boundary-alignment objectives.
    selected_by_head: dict[str, list[torch.Tensor]] = {
        "segmentation_raw": [],
    }
    for row in range(valid.shape[0]):
        length = int(valid[row].sum())
        indices = torch.linspace(
            0,
            length - 1,
            steps=common_count,
            device=valid.device,
        ).round().long()
        for head in selected_by_head:
            selected_by_head[head].append(output[head][row, indices])
    totals = []
    variance_values = []
    covariance_values = []
    for head, per_trial in selected_by_head.items():
        del head
        terms = variance_covariance_terms(
            torch.cat(per_trial, dim=0),
            target_std=float(target_std),
        )
        totals.append(
            float(variance_weight) * terms["variance"]
            + float(covariance_weight) * terms["covariance"]
        )
        variance_values.append(terms["variance"])
        covariance_values.append(terms["covariance"])
    return (
        torch.stack(totals).mean(),
        torch.stack(variance_values).mean(),
        torch.stack(covariance_values).mean(),
    )


def _trial_equal_prediction_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    prediction_mask: torch.Tensor,
    valid: torch.Tensor,
    loss_type: str,
    huber_delta: float,
) -> torch.Tensor:
    values = []
    for row in range(valid.shape[0]):
        length = int(valid[row].sum())
        values.append(
            masked_temporal_prediction_loss(
                prediction[row : row + 1, :length],
                target[row : row + 1, :length],
                prediction_mask[row : row + 1, :length],
                loss_type=loss_type,
                huber_delta=float(huber_delta),
            )
        )
    return torch.stack(values).mean()


def _trial_equal_boundary_loss(
    first: torch.Tensor,
    second: torch.Tensor,
    records: Sequence[TrialExample],
    context_windows: int,
    rank_margin: float,
    equivariance_delta: float,
    equivariance_weight: float,
) -> dict[str, torch.Tensor]:
    collected: dict[str, list[torch.Tensor]] = {
        "loss": [], "stable": [], "ranking": [], "equivariance": []
    }
    for row, record in enumerate(records):
        length = record.length
        assert record.stable_mask is not None and record.change_mask is not None
        terms = boundary_loss_terms(
            first[row : row + 1, :length],
            second[row : row + 1, :length],
            record.stable_mask.to(first.device).unsqueeze(0),
            record.change_mask.to(first.device).unsqueeze(0),
            context_windows=int(context_windows),
            rank_margin=float(rank_margin),
            equivariance_delta=float(equivariance_delta),
            equivariance_weight=float(equivariance_weight),
        )
        for name in collected:
            collected[name].append(terms[name])
    return {name: torch.stack(values).mean() for name, values in collected.items()}


def _window_consistency_loss(
    first_output: dict[str, torch.Tensor],
    second_output: dict[str, torch.Tensor],
    records: Sequence[TrialExample],
    valid: torch.Tensor,
    anchor_indices: Sequence[int],
    method: str,
    temperature: float,
    one_window_per_trial: bool,
    vicreg_invariance_weight: float,
    vicreg_variance_weight: float,
    vicreg_covariance_weight: float,
    vicreg_target_std: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    zero = first_output["augmentation_raw"].sum() * 0.0 + second_output["augmentation_raw"].sum() * 0.0
    if method == "none":
        return zero, {"invariance": zero, "variance": zero, "covariance": zero}
    if one_window_per_trial:
        rows = torch.arange(len(records), device=valid.device)
        indices = torch.tensor(anchor_indices, dtype=torch.long, device=valid.device)
        first_raw = first_output["augmentation_raw"][rows, indices]
        second_raw = second_output["augmentation_raw"][rows, indices]
        first_normalized = first_output["augmentation"][rows, indices]
        second_normalized = second_output["augmentation"][rows, indices]
    else:
        if method == "infonce":
            raise ValueError("InfoNCE requires exactly one anchor window per trial.")
        first_raw = first_output["augmentation_raw"][valid]
        second_raw = second_output["augmentation_raw"][valid]
        first_normalized = first_output["augmentation"][valid]
        second_normalized = second_output["augmentation"][valid]
    if method == "infonce":
        trial_ids = torch.tensor([record.trial_id for record in records], device=valid.device)
        loss = symmetric_info_nce(
            first_normalized,
            second_normalized,
            temperature=float(temperature),
            trial_ids=trial_ids,
        )
        return loss, {"invariance": loss, "variance": zero, "covariance": zero}
    if method != "vicreg":
        raise ValueError(f"Unknown window consistency method {method!r}.")
    invariance = F.mse_loss(first_raw, second_raw)
    first_terms = variance_covariance_terms(first_raw, target_std=float(vicreg_target_std))
    second_terms = variance_covariance_terms(second_raw, target_std=float(vicreg_target_std))
    variance = 0.5 * (first_terms["variance"] + second_terms["variance"])
    covariance = 0.5 * (first_terms["covariance"] + second_terms["covariance"])
    loss = (
        float(vicreg_invariance_weight) * invariance
        + float(vicreg_variance_weight) * variance
        + float(vicreg_covariance_weight) * covariance
    )
    return loss, {"invariance": invariance, "variance": variance, "covariance": covariance}


def _batch_slices(count: int, batch_size: int, permutation: torch.Tensor) -> list[torch.Tensor]:
    batches = [permutation[begin : begin + batch_size] for begin in range(0, count, batch_size)]
    # A singleton has no InfoNCE negatives.  Move one trial from the preceding
    # batch when possible while preserving exactly-once trial sampling.
    if len(batches) > 1 and len(batches[-1]) == 1 and len(batches[-2]) > 2:
        batches[-1] = torch.cat((batches[-2][-1:], batches[-1]))
        batches[-2] = batches[-2][:-1]
    elif len(batches) > 1 and len(batches[-1]) == 1:
        batches[-2] = torch.cat((batches[-2], batches[-1]))
        batches.pop()
    return batches


@torch.no_grad()
def _ema_update(teacher: nn.Module, student: nn.Module, momentum: float) -> None:
    student_parameters = dict(student.named_parameters())
    for name, target in teacher.named_parameters():
        target.mul_(float(momentum)).add_(student_parameters[name].detach(), alpha=1.0 - float(momentum))
    student_buffers = dict(student.named_buffers())
    for name, target in teacher.named_buffers():
        source = student_buffers[name].detach()
        if target.is_floating_point():
            target.mul_(float(momentum)).add_(source, alpha=1.0 - float(momentum))
        else:
            target.copy_(source)


def _gradient_parameter_groups(
    model: MotionPrimitiveEncoder,
) -> dict[str, list[nn.Parameter]]:
    groups = {
        "backbone": list(model.backbone.parameters()),
        "segmentation_head": list(model.segmentation_head.parameters()),
        "content_head": list(model.content_head.parameters()),
        "augmentation_head": list(model.augmentation_head.parameters()),
        "trial_auxiliary": list(model.trial_fusion.parameters()) + list(model.trial_classifier.parameters()),
        "temporal_predictor": list(model.temporal_predictor.parameters()),
    }
    for name, module in (
        ("segmentation_head", model.segmentation_skip),
        ("content_head", model.content_skip),
    ):
        if module is not None:
            groups[name].extend(module.parameters())
    parameter_ids = [id(parameter) for values in groups.values() for parameter in values]
    if len(parameter_ids) != len(set(parameter_ids)):
        raise RuntimeError("Gradient clipping groups contain duplicate parameters.")
    if set(parameter_ids) != {id(parameter) for parameter in model.parameters()}:
        raise RuntimeError("Gradient clipping groups do not cover the complete model.")
    return groups


def _gradient_norms(model: MotionPrimitiveEncoder) -> dict[str, float]:
    groups = _gradient_parameter_groups(model)
    groups = {"total": list(model.parameters()), **groups}
    result = {}
    for name, parameters in groups.items():
        squares = [parameter.grad.detach().float().square().sum() for parameter in parameters if parameter.grad is not None]
        result[name] = float(torch.sqrt(torch.stack(squares).sum()).cpu()) if squares else 0.0
    return result


def _profile_defaults(profile: str) -> dict:
    table = {
        "A0": {"window_aug_consistency": "none", "cp_weight": 0.0, "content_boundary_alignment_weight": 0.0, "window_aug_profile": "basic", "cp_anchor_source": "raw_frozen_consensus"},
        "A1": {"window_aug_consistency": "infonce", "cp_weight": 0.0, "content_boundary_alignment_weight": 0.0, "window_aug_profile": "basic", "cp_anchor_source": "raw_frozen_consensus"},
        "A2": {"window_aug_consistency": "none", "cp_weight": 1.0, "content_boundary_alignment_weight": 0.1, "window_aug_profile": "basic", "cp_anchor_source": "raw_frozen_consensus"},
        "A3": {"window_aug_consistency": "infonce", "cp_weight": 1.0, "content_boundary_alignment_weight": 0.1, "window_aug_profile": "basic", "cp_anchor_source": "raw_frozen_consensus"},
        "A4": {"window_aug_consistency": "infonce", "cp_weight": 1.0, "content_boundary_alignment_weight": 0.1, "window_aug_profile": "basic_rotation", "cp_anchor_source": "raw_frozen_consensus"},
        "CUSTOM": {"window_aug_consistency": "infonce", "cp_weight": 1.0, "content_boundary_alignment_weight": 0.1, "window_aug_profile": "basic", "cp_anchor_source": "raw_frozen_consensus"},
    }
    return table[str(profile).upper()].copy()


def resolve_configuration(args: argparse.Namespace, metadata: dict) -> dict:
    positive_integer_options = {
        "epochs": args.epochs,
        "trial_batch_size": args.trial_batch_size,
        "source_encode_batch_size": args.source_encode_batch_size,
        "cp_context_windows": args.cp_context_windows,
        "noncollapse_windows_per_trial": args.noncollapse_windows_per_trial,
    }
    invalid_positive = {
        name: value
        for name, value in positive_integer_options.items()
        if isinstance(value, bool) or int(value) != value or int(value) < 1
    }
    if invalid_positive:
        raise ValueError(
            f"Positive integer training options are invalid: {invalid_positive}."
        )
    if not 0.0 <= float(args.prediction_mask_ratio) < 1.0:
        raise ValueError("prediction_mask_ratio must lie in [0,1).")
    if not math.isfinite(float(args.learning_rate)) or float(args.learning_rate) <= 0:
        raise ValueError("learning_rate must be positive and finite.")
    if (
        not math.isfinite(float(args.minimum_learning_rate))
        or not 0.0 <= float(args.minimum_learning_rate) <= float(args.learning_rate)
    ):
        raise ValueError("minimum_learning_rate must lie in [0, learning_rate].")
    for name in (
        "weight_decay",
        "gradient_clip_norm",
        "minimum_improvement",
    ):
        value = float(getattr(args, name))
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be finite and non-negative.")
    if int(args.freeze_backbone_epochs) < 0 or int(args.early_stopping_patience) < 0:
        raise ValueError("freeze/early-stopping epoch counts must be non-negative.")
    if (
        not math.isfinite(float(args.normalization_eps))
        or float(args.normalization_eps) <= 0
    ):
        raise ValueError("normalization_eps must be positive and finite.")
    defaults = _profile_defaults(args.ablation_profile)
    profile_name = str(args.ablation_profile).upper()
    if profile_name == "CUSTOM":
        missing_identity = [
            flag
            for flag, value in (
                ("--window-aug-consistency", args.window_aug_consistency),
                ("--window-aug-profile", args.window_aug_profile),
                ("--cp-weight", args.cp_weight),
                (
                    "--content-boundary-alignment-weight",
                    args.content_boundary_alignment_weight,
                ),
                ("--rotation-max-degrees", args.rotation_max_degrees),
                ("--cp-anchor-source", args.cp_anchor_source),
            )
            if value is None
        ]
        if missing_identity:
            raise ValueError(
                "CUSTOM requires every identity-defining option explicitly: "
                + ", ".join(missing_identity)
            )
    else:
        expected_rotation = 3.0 if profile_name == "A4" else 0.0
        conflicts = []
        if args.window_aug_consistency is not None and args.window_aug_consistency != defaults["window_aug_consistency"]:
            conflicts.append(
                f"window_aug_consistency={args.window_aug_consistency!r} (expected {defaults['window_aug_consistency']!r})"
            )
        if args.window_aug_profile is not None and args.window_aug_profile != defaults["window_aug_profile"]:
            conflicts.append(
                f"window_aug_profile={args.window_aug_profile!r} (expected {defaults['window_aug_profile']!r})"
            )
        if args.cp_weight is not None and not math.isclose(float(args.cp_weight), float(defaults["cp_weight"]), abs_tol=1e-12):
            conflicts.append(f"cp_weight={args.cp_weight!r} (expected {defaults['cp_weight']!r})")
        if (
            args.content_boundary_alignment_weight is not None
            and not math.isclose(
                float(args.content_boundary_alignment_weight),
                float(defaults["content_boundary_alignment_weight"]),
                abs_tol=1e-12,
            )
        ):
            conflicts.append(
                "content_boundary_alignment_weight="
                f"{args.content_boundary_alignment_weight!r} (expected "
                f"{defaults['content_boundary_alignment_weight']!r})"
            )
        if args.rotation_max_degrees is not None and not math.isclose(float(args.rotation_max_degrees), expected_rotation, abs_tol=1e-12):
            conflicts.append(f"rotation_max_degrees={args.rotation_max_degrees!r} (expected {expected_rotation!r})")
        if args.cp_anchor_source is not None and args.cp_anchor_source != defaults["cp_anchor_source"]:
            conflicts.append(
                f"cp_anchor_source={args.cp_anchor_source!r} "
                f"(expected {defaults['cp_anchor_source']!r})"
            )
        if args.backbone_bn_policy != "frozen":
            conflicts.append(
                f"backbone_bn_policy={args.backbone_bn_policy!r} "
                "(expected 'frozen')"
            )
        if not bool(args.content_residual):
            conflicts.append("content_residual=False (expected True)")
        if not math.isclose(
            float(args.window_aug_weight), 1.0, rel_tol=0.0, abs_tol=1e-12
        ):
            conflicts.append(
                f"window_aug_weight={args.window_aug_weight!r} (expected 1.0)"
            )
        if conflicts:
            raise ValueError(
                f"{profile_name} identity conflict: " + "; ".join(conflicts)
                + ". Use --ablation-profile CUSTOM for a free combination."
            )
    consistency = defaults["window_aug_consistency"] if args.window_aug_consistency is None else args.window_aug_consistency
    cp_weight = defaults["cp_weight"] if args.cp_weight is None else float(args.cp_weight)
    content_boundary_alignment_weight = (
        defaults["content_boundary_alignment_weight"]
        if args.content_boundary_alignment_weight is None
        else float(args.content_boundary_alignment_weight)
    )
    aug_profile = defaults["window_aug_profile"] if args.window_aug_profile is None else args.window_aug_profile
    anchor_source = defaults["cp_anchor_source"] if args.cp_anchor_source is None else args.cp_anchor_source
    rotation = (3.0 if aug_profile == "basic_rotation" else 0.0) if args.rotation_max_degrees is None else float(args.rotation_max_degrees)
    backbone_dim = int(metadata["har_feat_dim"])
    segmentation_dim = (
        backbone_dim if int(args.segmentation_dim) == 0 else int(args.segmentation_dim)
    )
    architecture = {
        "in_channels": int(metadata["har_in_channels"]),
        "backbone_dim": backbone_dim,
        "base_channels": int(metadata["har_base_channels"]),
        "backbone_layers": _parse_layers(args.backbone_layers),
        "backbone_dropout": float(metadata.get("har_dropout", 0.0)),
        "segmentation_dim": segmentation_dim,
        "segmentation_residual": True,
        "content_dim": int(args.content_dim),
        "content_residual": bool(args.content_residual),
        "augmentation_dim": int(args.augmentation_dim),
        "projection_hidden_dim": int(args.projection_hidden_dim),
        "num_classes": int(args.old_class_count),
        "trial_hidden_dim": int(args.trial_hidden_dim),
        "trial_peak_quantile": float(args.trial_peak_quantile),
        "trial_dropout": float(args.trial_dropout),
        "predictor_hidden_dim": None if int(args.predictor_hidden_dim) == 0 else int(args.predictor_hidden_dim),
    }
    if architecture["segmentation_dim"] != architecture["backbone_dim"]:
        raise ValueError(
            "The formal A0-A4/CUSTOM protocol uses an exact identity-residual "
            "segmentation baseline; --segmentation-dim must be 0 or equal the "
            f"legacy backbone dimension ({architecture['backbone_dim']})."
        )
    if architecture["content_residual"] and architecture["content_dim"] != architecture["backbone_dim"]:
        raise ValueError(
            "--content-residual requires --content-dim to equal the legacy backbone dimension "
            f"({architecture['backbone_dim']})."
        )
    augmentation = MotionAugmentationConfig(
        noise_std_ratio=float(args.noise_std_ratio),
        acc_scale_range=(float(args.acc_scale_min), float(args.acc_scale_max)),
        gyro_scale_range=(float(args.gyro_scale_min), float(args.gyro_scale_max)),
        time_shift_max_samples=int(args.time_shift_max_samples),
        time_mask_min_samples=int(args.time_mask_min_samples),
        time_mask_max_samples=int(args.time_mask_max_samples),
        rotation_max_degrees=rotation,
    )
    aligned_augmentation = replace(augmentation, time_shift_max_samples=0)
    loss_weights = {
        "window_augmentation": float(args.window_aug_weight),
        "noncollapse": float(args.noncollapse_weight),
        "changepoint": float(cp_weight),
        "content_boundary_alignment": float(content_boundary_alignment_weight),
        "temporal_prediction": float(args.prediction_weight),
        "trial_auxiliary": float(args.trial_weight),
        "cross_subject": float(args.cross_subject_weight),
    }
    if any(not math.isfinite(value) or value < 0 for value in loss_weights.values()):
        raise ValueError("Every loss weight must be finite and non-negative.")
    if loss_weights["cross_subject"] != 0.0:
        raise ValueError("Cross-subject loss is intentionally deferred to E3; set --cross-subject-weight 0.")
    if consistency == "none":
        loss_weights["window_augmentation"] = 0.0
    if consistency == "infonce" and not args.window_aug_one_window_per_trial:
        raise ValueError("InfoNCE forbids --no-window-aug-one-window-per-trial (same-trial false negatives).")
    if int(args.trial_batch_size) < 2 and consistency == "infonce":
        raise ValueError("InfoNCE requires --trial-batch-size >= 2.")
    if not 0.0 < float(args.ema_momentum) < 1.0:
        raise ValueError("ema_momentum must lie strictly between zero and one.")
    raw_scales = parse_raw_scales(args.cp_raw_scales)
    if args.backbone_bn_policy not in {"frozen", "update"}:
        raise ValueError("backbone_bn_policy must be frozen or update.")
    if int(args.cp_raw_frequency_bins) < 2:
        raise ValueError("cp_raw_frequency_bins must be at least two.")
    for name, value in (
        ("cp_raw_epsilon", args.cp_raw_epsilon),
        ("cp_raw_scale_floor", args.cp_raw_scale_floor),
        ("cp_raw_z_clip", args.cp_raw_z_clip),
    ):
        if not math.isfinite(float(value)) or float(value) <= 0:
            raise ValueError(f"{name} must be positive and finite.")
    return {
        "ablation_profile": profile_name,
        "backbone_bn_policy": str(args.backbone_bn_policy),
        "gradient_clip_mode": "independent_parameter_groups_v1",
        "window_aug_consistency": consistency,
        "window_aug_profile": aug_profile,
        "window_aug_one_window_per_trial": bool(args.window_aug_one_window_per_trial),
        "augmentation": asdict(augmentation),
        "cp_aligned_augmentation": asdict(aligned_augmentation),
        "architecture": architecture,
        "loss_weights": loss_weights,
        "cp_anchor": {
            "source": str(anchor_source),
            "raw_scales": list(raw_scales),
            "raw_frequency_bins": int(args.cp_raw_frequency_bins),
            "raw_epsilon": float(args.cp_raw_epsilon),
            "raw_scale_floor": float(args.cp_raw_scale_floor),
            "raw_z_clip": float(args.cp_raw_z_clip),
        },
    }


def _make_augmented_views(
    records: Sequence[TrialExample],
    fold_mean: torch.Tensor,
    fold_std: torch.Tensor,
    config: MotionAugmentationConfig,
    generator: torch.Generator,
    anchor_indices: Optional[Sequence[int]],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    first_views = []
    second_views = []
    for index, record in enumerate(records):
        anchor = None
        if anchor_indices is not None:
            begin = int(record.starts[int(anchor_indices[index])])
            anchor = (begin, begin + int(record.clean_windows.shape[-1]))
        first, second = make_augmented_trial_pair(
            record.raw_trial,
            fold_mean,
            fold_std,
            record.starts,
            int(record.clean_windows.shape[-1]),
            config,
            generator=generator,
            time_mask_anchor=anchor,
        )
        first_views.append(first)
        second_views.append(second)
    first, valid = _pad_windows(first_views)
    second, second_valid = _pad_windows(second_views)
    if not torch.equal(valid, second_valid):
        raise RuntimeError("Augmented views produced different window-validity masks.")
    return first, second, valid


def _metric_accumulator() -> dict[str, float]:
    names = (
        "total_loss", "window_aug_loss", "window_aug_invariance", "window_aug_variance",
        "window_aug_covariance", "noncollapse_loss", "noncollapse_variance",
        "noncollapse_covariance", "cp_loss", "cp_stable", "cp_ranking",
        "cp_equivariance", "content_cp_loss", "content_cp_stable",
        "content_cp_ranking", "content_cp_equivariance", "prediction_loss",
        "trial_loss",
    )
    return {name: 0.0 for name in names}


def _add_metrics(accumulator: dict[str, float], values: dict[str, torch.Tensor], weight: int) -> None:
    for name in accumulator:
        accumulator[name] += float(values[name].detach().cpu()) * int(weight)


def _finish_metrics(accumulator: dict[str, float], count: int) -> dict[str, float]:
    return {name: value / max(1, int(count)) for name, value in accumulator.items()}


def _set_backbone_batchnorm_policy(
    model: MotionPrimitiveEncoder, policy: str
) -> int:
    """Apply the requested BN running-stat policy without touching other heads.

    This helper must be called after ``model.train()`` each epoch because that
    call recursively re-enables training mode.  ``eval()`` on BatchNorm1d
    freezes only its running mean/variance; affine parameters remain trainable.
    """

    if policy not in {"frozen", "update"}:
        raise ValueError("BatchNorm policy must be frozen or update.")
    modules = [
        module for module in model.backbone.modules()
        if isinstance(module, nn.BatchNorm1d)
    ]
    if policy == "frozen":
        for module in modules:
            module.eval()
    return len(modules)


def train_one_epoch(
    model: MotionPrimitiveEncoder,
    teacher: MotionPrimitiveEncoder,
    records: Sequence[TrialExample],
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    fold_mean: torch.Tensor,
    fold_std: torch.Tensor,
    config: dict,
    args: argparse.Namespace,
    generator: torch.Generator,
) -> dict:
    model.train()
    _set_backbone_batchnorm_policy(model, config["backbone_bn_policy"])
    teacher.eval()
    permutation = torch.randperm(len(records), generator=generator)
    batches = _batch_slices(len(records), int(args.trial_batch_size), permutation)
    metrics = _metric_accumulator()
    gradient_sums: dict[str, float] = {}
    gradient_maxima: dict[str, float] = {}
    post_clip_gradient_sums: dict[str, float] = {}
    post_clip_gradient_maxima: dict[str, float] = {}
    clip_scale_sums: dict[str, float] = {}
    group_clipped_steps: dict[str, int] = {}
    clipped_steps = 0
    optimizer_steps = 0
    seen = 0
    correct = 0
    augmentation = MotionAugmentationConfig(**config["augmentation"])
    aligned_augmentation = MotionAugmentationConfig(**config["cp_aligned_augmentation"])
    weights = config["loss_weights"]
    for positions in batches:
        batch_records = [records[int(index)] for index in positions]
        batch_size = len(batch_records)
        clean_cpu, valid_cpu = _pad_windows([record.clean_windows for record in batch_records])
        clean = clean_cpu.to(device=device)
        valid = valid_cpu.to(device=device)
        labels = torch.tensor([record.label for record in batch_records], dtype=torch.long, device=device)
        anchor_indices = [
            int(torch.randint(record.length, (1,), generator=generator).item())
            for record in batch_records
        ]

        clean_output = model(clean, valid)
        trial_loss = F.cross_entropy(clean_output["trial_logits"], labels)
        noncollapse, nc_variance, nc_covariance = _trial_equal_varcov(
            clean_output,
            valid,
            float(args.noncollapse_target_std),
            float(args.noncollapse_variance_weight),
            float(args.noncollapse_covariance_weight),
            int(args.noncollapse_windows_per_trial),
        )
        prediction_mask_cpu = _prediction_mask(
            valid_cpu,
            float(args.prediction_mask_ratio),
            generator,
            deterministic=False,
        )
        prediction_mask = prediction_mask_cpu.to(device=device)
        prediction = model.predict_masked_content(clean_output["content"], prediction_mask, valid)
        with torch.no_grad():
            teacher_target = teacher(clean, valid)["content"]
        prediction_loss = _trial_equal_prediction_loss(
            prediction,
            teacher_target,
            prediction_mask,
            valid,
            args.prediction_loss,
            float(args.prediction_huber_delta),
        )

        zero = clean_output["content"].sum() * 0.0
        aug_loss = zero
        aug_terms = {"invariance": zero, "variance": zero, "covariance": zero}
        if config["window_aug_consistency"] != "none":
            first_cpu, second_cpu, aug_valid_cpu = _make_augmented_views(
                batch_records,
                fold_mean,
                fold_std,
                augmentation,
                generator,
                anchor_indices,
            )
            first_output = model(first_cpu.to(device), aug_valid_cpu.to(device))
            second_output = model(second_cpu.to(device), aug_valid_cpu.to(device))
            aug_loss, aug_terms = _window_consistency_loss(
                first_output,
                second_output,
                batch_records,
                aug_valid_cpu.to(device),
                anchor_indices,
                config["window_aug_consistency"],
                float(args.window_aug_temperature),
                bool(config["window_aug_one_window_per_trial"]),
                float(args.vicreg_invariance_weight),
                float(args.vicreg_variance_weight),
                float(args.vicreg_covariance_weight),
                float(args.vicreg_target_std),
            )

        cp_terms = {"loss": zero, "stable": zero, "ranking": zero, "equivariance": zero}
        content_cp_terms = {"loss": zero, "stable": zero, "ranking": zero, "equivariance": zero}
        if (
            weights["changepoint"] > 0
            or weights["content_boundary_alignment"] > 0
        ):
            cp_first_cpu, cp_second_cpu, cp_valid_cpu = _make_augmented_views(
                batch_records,
                fold_mean,
                fold_std,
                aligned_augmentation,
                generator,
                anchor_indices=None,
            )
            cp_first_output = model(
                cp_first_cpu.to(device), cp_valid_cpu.to(device)
            )
            cp_second_output = model(
                cp_second_cpu.to(device), cp_valid_cpu.to(device)
            )
            cp_terms = _trial_equal_boundary_loss(
                cp_first_output["segmentation"],
                cp_second_output["segmentation"],
                batch_records,
                int(args.cp_context_windows),
                float(args.cp_rank_margin),
                float(args.cp_equivariance_delta),
                float(args.cp_equivariance_weight),
            )
            content_cp_terms = _trial_equal_boundary_loss(
                cp_first_output["content"],
                cp_second_output["content"],
                batch_records,
                int(args.cp_context_windows),
                float(args.cp_rank_margin),
                float(args.cp_equivariance_delta),
                float(args.cp_equivariance_weight),
            )

        total = (
            weights["window_augmentation"] * aug_loss
            + weights["noncollapse"] * noncollapse
            + weights["changepoint"] * cp_terms["loss"]
            + weights["content_boundary_alignment"] * content_cp_terms["loss"]
            + weights["temporal_prediction"] * prediction_loss
            + weights["trial_auxiliary"] * trial_loss
        )
        if not bool(torch.isfinite(total).item()):
            raise RuntimeError("A non-finite training loss was encountered.")
        optimizer.zero_grad(set_to_none=True)
        total.backward()
        gradient = _gradient_norms(model)
        clip_scales = {
            name: 1.0 for name in _gradient_parameter_groups(model)
        }
        if float(args.gradient_clip_norm) > 0:
            for name, parameters in _gradient_parameter_groups(model).items():
                pre_clip_norm = float(
                    torch.nn.utils.clip_grad_norm_(
                        parameters, float(args.gradient_clip_norm)
                    )
                )
                clip_scales[name] = min(
                    1.0,
                    float(args.gradient_clip_norm) / max(pre_clip_norm, 1e-12),
                )
        post_clip_gradient = _gradient_norms(model)
        optimizer.step()
        _ema_update(teacher, model, float(args.ema_momentum))

        values = {
            "total_loss": total,
            "window_aug_loss": aug_loss,
            "window_aug_invariance": aug_terms["invariance"],
            "window_aug_variance": aug_terms["variance"],
            "window_aug_covariance": aug_terms["covariance"],
            "noncollapse_loss": noncollapse,
            "noncollapse_variance": nc_variance,
            "noncollapse_covariance": nc_covariance,
            "cp_loss": cp_terms["loss"],
            "cp_stable": cp_terms["stable"],
            "cp_ranking": cp_terms["ranking"],
            "cp_equivariance": cp_terms["equivariance"],
            "content_cp_loss": content_cp_terms["loss"],
            "content_cp_stable": content_cp_terms["stable"],
            "content_cp_ranking": content_cp_terms["ranking"],
            "content_cp_equivariance": content_cp_terms["equivariance"],
            "prediction_loss": prediction_loss,
            "trial_loss": trial_loss,
        }
        _add_metrics(metrics, values, batch_size)
        for name, value in gradient.items():
            gradient_sums[name] = gradient_sums.get(name, 0.0) + value * batch_size
            gradient_maxima[name] = max(gradient_maxima.get(name, 0.0), value)
        for name, value in post_clip_gradient.items():
            post_clip_gradient_sums[name] = (
                post_clip_gradient_sums.get(name, 0.0) + value * batch_size
            )
            post_clip_gradient_maxima[name] = max(
                post_clip_gradient_maxima.get(name, 0.0), value
            )
        for name, scale in clip_scales.items():
            clip_scale_sums[name] = clip_scale_sums.get(name, 0.0) + scale
            group_clipped_steps[name] = group_clipped_steps.get(name, 0) + int(
                scale < 1.0
            )
        clipped_steps += int(any(scale < 1.0 for scale in clip_scales.values()))
        optimizer_steps += 1
        correct += int((clean_output["trial_logits"].argmax(dim=1) == labels).sum())
        seen += batch_size
    result = _finish_metrics(metrics, seen)
    weighted = {
        "window_augmentation": weights["window_augmentation"] * result["window_aug_loss"],
        "noncollapse": weights["noncollapse"] * result["noncollapse_loss"],
        "segmentation_changepoint": weights["changepoint"] * result["cp_loss"],
        "content_boundary_alignment": weights["content_boundary_alignment"] * result["content_cp_loss"],
        "temporal_prediction": weights["temporal_prediction"] * result["prediction_loss"],
        "trial_auxiliary": weights["trial_auxiliary"] * result["trial_loss"],
    }
    weighted_total = sum(weighted.values())
    result["weighted_loss_contributions"] = weighted
    result["weighted_loss_fractions"] = {
        name: value / max(1e-12, weighted_total) for name, value in weighted.items()
    }
    result["trial_accuracy"] = correct / max(1, seen)
    result["gradient_norm_mean"] = {name: value / max(1, seen) for name, value in gradient_sums.items()}
    result["gradient_norm_max"] = gradient_maxima
    result["post_clip_gradient_norm_mean"] = {
        name: value / max(1, seen)
        for name, value in post_clip_gradient_sums.items()
    }
    result["post_clip_gradient_norm_max"] = post_clip_gradient_maxima
    result["gradient_clip"] = {
        "mode": "independent_parameter_groups_v1",
        "max_norm": float(args.gradient_clip_norm),
        "optimizer_steps": int(optimizer_steps),
        "clipped_steps": int(clipped_steps),
        "clipped_step_fraction": clipped_steps / max(1, optimizer_steps),
        "mean_applied_scale_by_group": {
            name: value / max(1, optimizer_steps)
            for name, value in clip_scale_sums.items()
        },
        "clipped_steps_by_group": group_clipped_steps,
    }
    result["trial_count"] = seen
    return result


@torch.inference_mode()
def validate(
    model: MotionPrimitiveEncoder,
    teacher: MotionPrimitiveEncoder,
    records: Sequence[TrialExample],
    device: torch.device,
    config: dict,
    args: argparse.Namespace,
) -> dict:
    """Deterministic clean-view validation; no outer-test object exists."""

    model.eval()
    teacher.eval()
    metrics = _metric_accumulator()
    weights = config["loss_weights"]
    seen = 0
    correct = 0
    confusion = np.zeros((int(args.old_class_count), int(args.old_class_count)), dtype=np.int64)
    cp_change_values: list[float] = []
    cp_stable_values: list[float] = []
    ordered = torch.arange(len(records))
    for positions in _batch_slices(len(records), int(args.trial_batch_size), ordered):
        batch_records = [records[int(index)] for index in positions]
        batch_size = len(batch_records)
        clean_cpu, valid_cpu = _pad_windows([record.clean_windows for record in batch_records])
        clean = clean_cpu.to(device)
        valid = valid_cpu.to(device)
        labels = torch.tensor([record.label for record in batch_records], dtype=torch.long, device=device)
        output = model(clean, valid)
        trial_loss = F.cross_entropy(output["trial_logits"], labels)
        noncollapse, nc_variance, nc_covariance = _trial_equal_varcov(
            output,
            valid,
            float(args.noncollapse_target_std),
            float(args.noncollapse_variance_weight),
            float(args.noncollapse_covariance_weight),
            int(args.noncollapse_windows_per_trial),
        )
        pred_mask = _prediction_mask(
            valid_cpu,
            float(args.prediction_mask_ratio),
            generator=None,
            deterministic=True,
        ).to(device)
        prediction = model.predict_masked_content(output["content"], pred_mask, valid)
        target = teacher(clean, valid)["content"]
        prediction_loss = _trial_equal_prediction_loss(
            prediction,
            target,
            pred_mask,
            valid,
            args.prediction_loss,
            float(args.prediction_huber_delta),
        )
        zero = output["content"].sum() * 0.0
        cp_terms = {"loss": zero, "stable": zero, "ranking": zero, "equivariance": zero}
        content_cp_terms = {"loss": zero, "stable": zero, "ranking": zero, "equivariance": zero}
        if weights["changepoint"] > 0 or weights["content_boundary_alignment"] > 0:
            cp_terms = _trial_equal_boundary_loss(
                output["segmentation"],
                output["segmentation"],
                batch_records,
                int(args.cp_context_windows),
                float(args.cp_rank_margin),
                float(args.cp_equivariance_delta),
                float(args.cp_equivariance_weight),
            )
            content_cp_terms = _trial_equal_boundary_loss(
                output["content"],
                output["content"],
                batch_records,
                int(args.cp_context_windows),
                float(args.cp_rank_margin),
                float(args.cp_equivariance_delta),
                float(args.cp_equivariance_weight),
            )
        scores, score_valid = batched_feature_change_scores(
            output["segmentation"], int(args.cp_context_windows), valid
        )
        for row, record in enumerate(batch_records):
            count = max(0, record.length - 1)
            row_scores = scores[row, :count]
            assert record.change_mask is not None and record.stable_mask is not None
            if torch.any(record.change_mask):
                cp_change_values.extend(row_scores[record.change_mask.to(device)].cpu().tolist())
            if torch.any(record.stable_mask):
                cp_stable_values.extend(row_scores[record.stable_mask.to(device)].cpu().tolist())
            if not bool(score_valid[row, :count].all().item()):
                raise RuntimeError("A real validation boundary was marked invalid.")
        total = (
            weights["noncollapse"] * noncollapse
            + weights["changepoint"] * cp_terms["loss"]
            + weights["content_boundary_alignment"] * content_cp_terms["loss"]
            + weights["temporal_prediction"] * prediction_loss
            + weights["trial_auxiliary"] * trial_loss
        )
        values = {
            "total_loss": total,
            "window_aug_loss": zero,
            "window_aug_invariance": zero,
            "window_aug_variance": zero,
            "window_aug_covariance": zero,
            "noncollapse_loss": noncollapse,
            "noncollapse_variance": nc_variance,
            "noncollapse_covariance": nc_covariance,
            "cp_loss": cp_terms["loss"],
            "cp_stable": cp_terms["stable"],
            "cp_ranking": cp_terms["ranking"],
            "cp_equivariance": cp_terms["equivariance"],
            "content_cp_loss": content_cp_terms["loss"],
            "content_cp_stable": content_cp_terms["stable"],
            "content_cp_ranking": content_cp_terms["ranking"],
            "content_cp_equivariance": content_cp_terms["equivariance"],
            "prediction_loss": prediction_loss,
            "trial_loss": trial_loss,
        }
        _add_metrics(metrics, values, batch_size)
        predictions = output["trial_logits"].argmax(dim=1)
        correct += int((predictions == labels).sum())
        for truth, predicted in zip(labels.cpu().tolist(), predictions.cpu().tolist()):
            confusion[int(truth), int(predicted)] += 1
        seen += batch_size
    result = _finish_metrics(metrics, seen)
    weighted = {
        "window_augmentation": 0.0,
        "noncollapse": weights["noncollapse"] * result["noncollapse_loss"],
        "segmentation_changepoint": weights["changepoint"] * result["cp_loss"],
        "content_boundary_alignment": weights["content_boundary_alignment"] * result["content_cp_loss"],
        "temporal_prediction": weights["temporal_prediction"] * result["prediction_loss"],
        "trial_auxiliary": weights["trial_auxiliary"] * result["trial_loss"],
    }
    weighted_total = sum(weighted.values())
    result["weighted_loss_contributions"] = weighted
    result["weighted_loss_fractions"] = {
        name: value / max(1e-12, weighted_total) for name, value in weighted.items()
    }
    result["trial_accuracy"] = correct / max(1, seen)
    class_recall = np.diag(confusion) / np.maximum(confusion.sum(axis=1), 1)
    result["trial_balanced_accuracy"] = float(np.mean(class_recall))
    result["confusion_matrix"] = confusion.tolist()
    result["cp_change_score_mean"] = float(np.mean(cp_change_values)) if cp_change_values else None
    result["cp_stable_score_mean"] = float(np.mean(cp_stable_values)) if cp_stable_values else None
    result["cp_change_minus_stable_gap"] = (
        float(np.mean(cp_change_values) - np.mean(cp_stable_values))
        if cp_change_values and cp_stable_values
        else None
    )
    result["trial_count"] = seen
    result["outer_test_evaluations"] = 0
    return result


def _instantiate(architecture: dict) -> MotionPrimitiveEncoder:
    return MotionPrimitiveEncoder(**{key: architecture[key] for key in ARCHITECTURE_FIELDS})


def _validate_output_checkpoint(checkpoint: dict) -> None:
    validate_motion_encoder_checkpoint_integrity(checkpoint)
    if checkpoint.get("checkpoint_type") != CHECKPOINT_TYPE:
        raise RuntimeError("Output checkpoint_type is invalid.")
    if checkpoint.get("schema_version") != SCHEMA_VERSION:
        raise RuntimeError("Output schema_version is invalid.")
    run_identity = checkpoint.get("run_identity")
    if not isinstance(run_identity, dict) or run_identity.get("schema") != RUN_IDENTITY_SCHEMA:
        raise RuntimeError("Output checkpoint lacks a valid run identity.")
    architecture = checkpoint.get("architecture")
    if not isinstance(architecture, dict):
        raise RuntimeError("Output checkpoint lacks architecture dictionary.")
    missing = [key for key in ARCHITECTURE_FIELDS if key not in architecture]
    if missing:
        raise RuntimeError(f"Output architecture lacks fields: {missing}.")
    state = checkpoint.get("model_state_dict")
    if not isinstance(state, dict):
        raise RuntimeError("Output checkpoint lacks model_state_dict.")
    probe = _instantiate(architecture)
    probe.load_state_dict(state, strict=True)
    observed_hash = _state_dict_sha256(state)
    if observed_hash != checkpoint.get("model_state_dict_sha256"):
        raise RuntimeError("Output checkpoint model_state_dict SHA256 mismatch.")
    audit = checkpoint.get("split_audit", {})
    if audit.get("outer_test_sensor_windows_selected") != 0 or audit.get("outer_test_model_forward_calls") != 0:
        raise RuntimeError("Output split audit does not prove zero outer-test use.")
    groups = [
        set(audit.get("train_subjects", [])),
        set(audit.get("validation_subjects", [])),
        set(audit.get("outer_test_subjects_metadata_only", [])),
    ]
    if any(groups[left] & groups[right] for left, right in ((0, 1), (0, 2), (1, 2))):
        raise RuntimeError("Output checkpoint split audit contains subject overlap.")


def _checkpoint_payload(
    *,
    model_state: dict[str, torch.Tensor],
    teacher_state: dict[str, torch.Tensor],
    architecture: dict,
    config: dict,
    args: argparse.Namespace,
    source_checkpoint: Path,
    source_sha256: str,
    npz_path: Path,
    npz_sha256: str,
    source_metadata: dict,
    source_key_mapping: dict[str, str],
    split_audit: dict,
    pseudo_boundary: dict,
    history: list[dict],
    selected_epoch: int,
    selected_metric: float,
    selection_policy: str,
    best_epoch: int,
    best_metric: float,
    file_role: str,
    run_identity: dict,
) -> dict:
    state = {key: value.detach().cpu().clone() for key, value in model_state.items()}
    teacher = {key: value.detach().cpu().clone() for key, value in teacher_state.items()}
    return {
        "checkpoint_type": CHECKPOINT_TYPE,
        "schema_version": SCHEMA_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "architecture": copy.deepcopy(architecture),
        "model_state_dict": state,
        # Compatibility alias for generic project checkpoint readers.  Strict
        # consumers should prefer model_state_dict plus schema_version.
        "model": state,
        "ema_teacher_state_dict": teacher,
        "model_state_dict_sha256": _state_dict_sha256(state),
        "implementation_fingerprint": _implementation_fingerprint(),
        "run_identity": copy.deepcopy(run_identity),
        "npz_sha256": npz_sha256,
        "source_checkpoint": {
            "path": str(source_checkpoint),
            "sha256": source_sha256,
            "legacy_experiment_metadata": copy.deepcopy(source_metadata),
            "strict_backbone_key_mapping": source_key_mapping,
        },
        "data": {"npz_path": str(npz_path), "npz_sha256": npz_sha256},
        "experiment_metadata": {
            "uschad_npz_path": str(npz_path),
            "uschad_train_subjects": split_audit["train_subjects"],
            "offline_val_subjects": split_audit["validation_subjects"],
            "uschad_test_subjects": split_audit["outer_test_subjects_metadata_only"],
            "uschad_cv_fold": source_metadata.get("uschad_cv_fold"),
            "uschad_window_size": int(split_audit["window_size_samples"]),
            "uschad_recompute_norm_from_train_subjects": True,
            "uschad_norm_eps": float(args.normalization_eps),
            "har_in_channels": int(architecture["in_channels"]),
            "har_feat_dim": int(architecture["backbone_dim"]),
            "har_base_channels": int(architecture["base_channels"]),
            "har_dropout": float(architecture["backbone_dropout"]),
            "motion_encoder_seed": int(args.seed),
            "seed": int(args.seed),
            "old_class_count": int(args.old_class_count),
            "backbone_bn_policy": str(config["backbone_bn_policy"]),
            "primitive_segmentation": "motion_encoder_changepoint",
            "feature_roles": {"codebook": "content", "boundary": "segmentation"},
            "smoke_test": bool(split_audit.get("smoke_test", False)),
            "outer_test_used_during_encoder_training": False,
        },
        "resolved_training_config": copy.deepcopy(config),
        "command_arguments": vars(args).copy(),
        "split_audit": copy.deepcopy(split_audit),
        "pseudo_boundary_calibration": copy.deepcopy(pseudo_boundary),
        "selection": {
            "split": "validation_subjects_old_classes",
            "metric": "total_loss",
            "mode": "min",
            "policy": str(selection_policy),
            "file_role": str(file_role),
            "selected_epoch_1based": int(selected_epoch),
            "selected_metric": float(selected_metric),
            "best_epoch_1based": int(best_epoch),
            "best_metric": float(best_metric),
            "completed_epochs": int(len(history)),
            "outer_test_queries": 0,
        },
        "history": copy.deepcopy(history),
    }


def _configure_logging(output_dir: Path) -> None:
    LOGGER.handlers.clear()
    LOGGER.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(formatter)
    file_handler = logging.FileHandler(output_dir / "train.log", encoding="utf-8")
    file_handler.setFormatter(formatter)
    LOGGER.addHandler(stream)
    LOGGER.addHandler(file_handler)


def _close_logging() -> None:
    for handler in list(LOGGER.handlers):
        handler.flush()
        handler.close()
        LOGGER.removeHandler(handler)


def _seed_everything(seed: int, deterministic: bool) -> torch.Generator:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    if deterministic:
        torch.use_deterministic_algorithms(True)
        if torch.backends.cudnn.is_available():
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed) + 7919)
    return generator


def _choose_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")
    return device


def _normalised_run_arguments(
    args: argparse.Namespace,
    *,
    source_path: Path,
    npz_path: Path,
) -> dict:
    """Return every training-relevant parsed argument in a stable form."""

    ignored = {"output_dir", "resume", "self_test"}
    arguments = {
        key: value for key, value in vars(args).items() if key not in ignored
    }
    arguments["source_checkpoint"] = str(source_path.resolve())
    arguments["npz_path"] = str(npz_path.resolve())
    return _jsonable(arguments)


def _build_run_identity(
    args: argparse.Namespace,
    *,
    source_path: Path,
    source_sha256: str,
    npz_path: Path,
    npz_sha256: str,
    fold: int,
    resolved_config: dict,
    resolved_device: torch.device,
) -> dict:
    """Build the exact identity required for safe completed-run reuse."""

    return {
        "schema": RUN_IDENTITY_SCHEMA,
        "source_checkpoint_path": str(source_path.resolve()),
        "source_checkpoint_sha256": str(source_sha256),
        "npz_path": str(npz_path.resolve()),
        "npz_sha256": str(npz_sha256),
        "fold": int(fold),
        "seed": int(args.seed),
        "arguments": _normalised_run_arguments(
            args,
            source_path=source_path,
            npz_path=npz_path,
        ),
        "resolved_training_config": _jsonable(copy.deepcopy(resolved_config)),
        "resolved_device": str(resolved_device),
        "implementation_fingerprint": _implementation_fingerprint(),
    }


def resolve_run_identity(args: argparse.Namespace) -> dict:
    """Resolve an A2 command to the same identity used by the trainer.

    The strict grid runner uses this before accepting an existing member.  It
    intentionally performs only read-only source/data checks and never selects
    or encodes outer-test samples.
    """

    if not args.source_checkpoint:
        raise ValueError("--source-checkpoint is required outside --self-test.")
    source_path = _resolve_input_path(args.source_checkpoint)
    if not source_path.is_file():
        raise FileNotFoundError(source_path)
    source_sha = _sha256_file(source_path)
    source_checkpoint = _load_checkpoint(source_path)
    source_metadata = source_checkpoint.get("experiment_metadata")
    _validate_source_metadata(source_metadata, int(args.old_class_count))
    npz_recorded = args.npz_path or str(source_metadata.get("uschad_npz_path", ""))
    if not npz_recorded:
        raise ValueError("No USC-HAD NPZ path is available; pass --npz-path.")
    npz_path = _resolve_input_path(npz_recorded)
    if not npz_path.is_file():
        raise FileNotFoundError(npz_path)
    _validate_source_npz_binding(source_metadata, npz_path)
    config = resolve_configuration(args, source_metadata)
    fold = int(source_metadata.get("uschad_cv_fold", -1))
    return _build_run_identity(
        args,
        source_path=source_path,
        source_sha256=source_sha,
        npz_path=npz_path,
        npz_sha256=_sha256_file(npz_path),
        fold=fold,
        resolved_config=config,
        resolved_device=_choose_device(str(args.device)),
    )


def _run_training_impl(args: argparse.Namespace) -> Path:
    if not args.source_checkpoint:
        raise ValueError("--source-checkpoint is required outside --self-test.")
    source_path = _resolve_input_path(args.source_checkpoint)
    if not source_path.is_file():
        raise FileNotFoundError(source_path)
    source_sha = _sha256_file(source_path)
    source_checkpoint = _load_checkpoint(source_path)
    source_metadata = source_checkpoint.get("experiment_metadata")
    split = _validate_source_metadata(source_metadata, int(args.old_class_count))
    npz_recorded = args.npz_path or str(source_metadata.get("uschad_npz_path", ""))
    if not npz_recorded:
        raise ValueError("No USC-HAD NPZ path is available; pass --npz-path.")
    npz_path = _resolve_input_path(npz_recorded)
    if not npz_path.is_file():
        raise FileNotFoundError(npz_path)
    source_npz_binding = _validate_source_npz_binding(source_metadata, npz_path)
    npz_sha = _sha256_file(npz_path)

    config = resolve_configuration(args, source_metadata)
    fold = int(source_metadata.get("uschad_cv_fold", -1))
    device = _choose_device(str(args.device))
    run_identity = _build_run_identity(
        args,
        source_path=source_path,
        source_sha256=source_sha,
        npz_path=npz_path,
        npz_sha256=npz_sha,
        fold=fold,
        resolved_config=config,
        resolved_device=device,
    )
    if args.output_dir:
        output_dir = _resolve_input_path(args.output_dir)
    else:
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        output_dir = PROJECT_ROOT / "results" / "motion_primitive" / "encoder_training" / (
            f"fold_{fold:02d}_seed_{int(args.seed)}_{config['ablation_profile']}_{stamp}"
        )
    if output_dir.exists() and any(output_dir.iterdir()):
        complete_path = output_dir / "complete.json"
        final_path = output_dir / "motion_encoder_final.pt"
        if bool(getattr(args, "resume", False)) and complete_path.is_file() and final_path.is_file():
            completion = json.loads(complete_path.read_text(encoding="utf-8"))
            if completion.get("identity") != run_identity:
                raise RuntimeError(
                    "Completed motion-encoder directory records another run identity; "
                    "use a new output directory."
                )
            if completion.get("complete") is not True:
                raise RuntimeError("Motion-encoder completion marker is false.")
            observed_file_hash = _sha256_file(final_path)
            if observed_file_hash != completion.get("final_checkpoint_sha256"):
                raise RuntimeError("Completed motion-encoder checkpoint SHA256 mismatch.")
            checkpoint = _load_checkpoint(final_path)
            _validate_output_checkpoint(checkpoint)
            if checkpoint.get("run_identity") != run_identity:
                raise RuntimeError("Completed motion encoder records another run identity.")
            if checkpoint.get("npz_sha256") != npz_sha:
                raise RuntimeError("Completed motion encoder is bound to another NPZ.")
            LOGGER.info("Resume verified completed motion encoder: %s", final_path)
            return final_path
        raise FileExistsError(
            f"Output directory is not empty or is incomplete: {output_dir}. "
            "Completed members can be reused with --resume; preserve an interrupted "
            "directory under another name before restarting it."
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    _configure_logging(output_dir)
    LOGGER.info("Source checkpoint: %s", source_path)
    LOGGER.info("USC-HAD NPZ: %s", npz_path)
    LOGGER.info("Protocol: train-subject old classes -> validation-subject old classes; outer test unused")

    generator = _seed_everything(int(args.seed), bool(args.deterministic))
    train_records, val_records, fold_mean, fold_std, split_audit = load_train_validation_trials(
        npz_path,
        split,
        int(args.old_class_count),
        float(args.normalization_eps),
        args.known_anomaly_policy,
    )
    split_audit["source_checkpoint_npz_binding"] = source_npz_binding
    # Bind the selected windows/splits to the exact source bytes in both the
    # standalone audit and the final checkpoint.  Paths alone are not stable
    # enough to establish data identity across machines.
    split_audit["npz_sha256"] = npz_sha
    train_records, val_records = _apply_smoke_limits(
        train_records,
        val_records,
        split_audit,
        int(args.smoke_max_train_trials),
        int(args.smoke_max_val_trials),
    )
    recorded_window_size = source_metadata.get("uschad_window_size")
    actual_window_size = int(train_records[0].clean_windows.shape[-1])
    if recorded_window_size is not None and int(recorded_window_size) != actual_window_size:
        raise RuntimeError(
            f"Checkpoint/NPZ window size mismatch: {recorded_window_size} vs {actual_window_size}."
        )
    LOGGER.info(
        "Selected %d train trials/%d windows and %d validation trials/%d windows; test windows=0",
        len(train_records), split_audit["train_window_count"],
        len(val_records), split_audit["validation_window_count"],
    )

    model = _instantiate(config["architecture"])
    backbone_state, source_mapping = _extract_backbone_state(source_checkpoint, model.backbone)
    # Keep initialisation explicit at the training call site as well as inside
    # the strict extractor; the EMA teacher is created only after this load.
    model.backbone.load_state_dict(backbone_state, strict=True)
    source_encoder = ResNet1D(
        in_channels=config["architecture"]["in_channels"],
        feat_dim=config["architecture"]["backbone_dim"],
        base_channels=config["architecture"]["base_channels"],
        layers=config["architecture"]["backbone_layers"],
        dropout=config["architecture"]["backbone_dropout"],
    )
    source_encoder.load_state_dict(backbone_state, strict=True)
    source_encoder.eval()
    for parameter in source_encoder.parameters():
        parameter.requires_grad_(False)
    pseudo = build_source_pseudo_boundaries(
        source_encoder,
        train_records,
        val_records,
        device,
        int(args.source_encode_batch_size),
        int(args.cp_context_windows),
        float(args.cp_low_quantile),
        float(args.cp_high_quantile),
        anchor_source=config["cp_anchor"]["source"],
        raw_scales=config["cp_anchor"]["raw_scales"],
        raw_frequency_bins=config["cp_anchor"]["raw_frequency_bins"],
        raw_epsilon=config["cp_anchor"]["raw_epsilon"],
        raw_scale_floor=config["cp_anchor"]["raw_scale_floor"],
        raw_z_clip=config["cp_anchor"]["raw_z_clip"],
    )
    LOGGER.info(
        "CP anchor calibration (%s): frozen low=%.6f high=%.6f train change anchors=%d trials without change=%d",
        pseudo["source"],
        pseudo["low_threshold"], pseudo["high_threshold"],
        pseudo["train_change_anchor_count"], pseudo["train_trials_without_change_anchor"],
    )
    if (
        config["loss_weights"]["changepoint"] > 0
        or config["loss_weights"]["content_boundary_alignment"] > 0
    ) and (
        int(pseudo["train_change_anchor_count"]) == 0
        or int(pseudo["train_stable_anchor_count"]) == 0
        or int(pseudo["train_trials_with_selected_anchor_pair"]) == 0
    ):
        message = (
            "The CP loss has no usable within-trial train-old consensus "
            "anchor pair: "
            f"stable={pseudo['train_stable_anchor_count']}, "
            f"change={pseudo['train_change_anchor_count']}, "
            "trials_with_both="
            f"{pseudo['train_trials_with_selected_anchor_pair']}."
        )
        if bool(split_audit.get("smoke_test", False)):
            LOGGER.warning("%s Synthetic/smoke execution continues for plumbing only.", message)
        else:
            raise RuntimeError(message)

    model = model.to(device)
    teacher = copy.deepcopy(model).to(device).eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(args.learning_rate),
        weight_decay=float(args.weight_decay),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, int(args.epochs)),
        eta_min=float(args.minimum_learning_rate),
    )
    history: list[dict] = []
    best_metric = float("inf")
    best_epoch = 0
    best_model_state = None
    best_teacher_state = None
    stale_epochs = 0
    _write_json(output_dir / "resolved_config.json", config)
    _write_json(output_dir / "split_audit.json", split_audit)
    _write_json(output_dir / "pseudo_boundary_calibration.json", pseudo)
    for epoch in range(1, int(args.epochs) + 1):
        if int(args.freeze_backbone_epochs) > 0:
            trainable = epoch > int(args.freeze_backbone_epochs)
            for parameter in model.backbone.parameters():
                parameter.requires_grad_(trainable)
        learning_rate = float(optimizer.param_groups[0]["lr"])
        train_metrics = train_one_epoch(
            model,
            teacher,
            train_records,
            optimizer,
            device,
            fold_mean,
            fold_std,
            config,
            args,
            generator,
        )
        val_metrics = validate(model, teacher, val_records, device, config, args)
        entry = {
            "epoch_1based": epoch,
            "learning_rate": learning_rate,
            "train": train_metrics,
            "validation": val_metrics,
        }
        history.append(entry)
        selection_value = float(val_metrics["total_loss"])
        improved = selection_value < best_metric - float(args.minimum_improvement)
        if improved:
            best_metric = selection_value
            best_epoch = epoch
            best_model_state = copy.deepcopy(model.state_dict())
            best_teacher_state = copy.deepcopy(teacher.state_dict())
            stale_epochs = 0
        else:
            stale_epochs += 1
        _write_json(output_dir / "history.json", history)
        LOGGER.info(
            "Epoch %d/%d | train total=%.5f acc=%.3f | val total=%.5f acc=%.3f cp_gap=%s | best=%d",
            epoch, int(args.epochs), train_metrics["total_loss"], train_metrics["trial_accuracy"],
            val_metrics["total_loss"], val_metrics["trial_accuracy"],
            "NA" if val_metrics["cp_change_minus_stable_gap"] is None else f"{val_metrics['cp_change_minus_stable_gap']:.5f}",
            best_epoch,
        )
        scheduler.step()
        if int(args.early_stopping_patience) > 0 and stale_epochs >= int(args.early_stopping_patience):
            LOGGER.info("Early stopping after %d stale validation epochs.", stale_epochs)
            break
    if best_model_state is None or best_teacher_state is None or best_epoch < 1:
        raise RuntimeError("Training ended without a finite validation-selected checkpoint.")
    last_epoch = int(history[-1]["epoch_1based"])
    last_metric = float(history[-1]["validation"]["total_loss"])
    best_payload = _checkpoint_payload(
        model_state=best_model_state,
        teacher_state=best_teacher_state,
        architecture=config["architecture"],
        config=config,
        args=args,
        source_checkpoint=source_path,
        source_sha256=source_sha,
        npz_path=npz_path,
        npz_sha256=npz_sha,
        source_metadata=source_metadata,
        source_key_mapping=source_mapping,
        split_audit=split_audit,
        pseudo_boundary=pseudo,
        history=history,
        selected_epoch=best_epoch,
        selected_metric=best_metric,
        selection_policy="best_val_total",
        best_epoch=best_epoch,
        best_metric=best_metric,
        file_role="best_validation",
        run_identity=run_identity,
    )
    if args.selection_policy == "best_val_total":
        final_model_state = best_model_state
        final_teacher_state = best_teacher_state
        final_epoch = best_epoch
        final_metric = best_metric
    else:
        final_model_state = model.state_dict()
        final_teacher_state = teacher.state_dict()
        final_epoch = last_epoch
        final_metric = last_metric
    final_payload = _checkpoint_payload(
        model_state=final_model_state,
        teacher_state=final_teacher_state,
        architecture=config["architecture"],
        config=config,
        args=args,
        source_checkpoint=source_path,
        source_sha256=source_sha,
        npz_path=npz_path,
        npz_sha256=npz_sha,
        source_metadata=source_metadata,
        source_key_mapping=source_mapping,
        split_audit=split_audit,
        pseudo_boundary=pseudo,
        history=history,
        selected_epoch=final_epoch,
        selected_metric=final_metric,
        selection_policy=args.selection_policy,
        best_epoch=best_epoch,
        best_metric=best_metric,
        file_role="canonical_final",
        run_identity=run_identity,
    )
    _validate_output_checkpoint(best_payload)
    _validate_output_checkpoint(final_payload)
    best_path = output_dir / "motion_encoder_best.pt"
    final_path = output_dir / "motion_encoder_final.pt"
    torch.save(best_payload, best_path)
    torch.save(final_payload, final_path)
    for path in (best_path, final_path):
        file_hash = _sha256_file(path)
        path.with_suffix(path.suffix + ".sha256").write_text(
            f"{file_hash}  {path.name}\n", encoding="ascii"
        )
        reloaded = _load_checkpoint(path)
        _validate_output_checkpoint(reloaded)
    _write_json(
        output_dir / "complete.json",
        {
            "schema": "hhr_motion_encoder_training_complete_v1",
            "identity": run_identity,
            "source_checkpoint_sha256": source_sha,
            "npz_sha256": npz_sha,
            "fold": fold,
            "seed": int(args.seed),
            "ablation_profile": str(config["ablation_profile"]),
            "selection_policy": str(args.selection_policy),
            "selected_epoch": int(final_epoch),
            "selected_validation_total_loss": float(final_metric),
            "best_validation_epoch": int(best_epoch),
            "best_validation_total_loss": float(best_metric),
            "best_checkpoint_sha256": _sha256_file(best_path),
            "final_checkpoint_sha256": _sha256_file(final_path),
            "smoke_test": bool(split_audit.get("smoke_test", False)),
            "outer_test_model_forward_calls": int(
                split_audit.get("outer_test_model_forward_calls", 0)
            ),
            "complete": True,
        },
    )
    LOGGER.info(
        "Saved canonical %s epoch %d (val total %.6f): %s; best-val epoch=%d",
        args.selection_policy, final_epoch, final_metric, final_path, best_epoch,
    )
    return final_path


def run_training(args: argparse.Namespace) -> Path:
    """Run one encoder training job and always release log file handles.

    This wrapper matters on Windows: a fail-closed protocol check may raise
    after ``train.log`` is opened, and an unclosed handler otherwise prevents
    smoke-test temporary directories from being removed.
    """

    try:
        return _run_training_impl(args)
    finally:
        _close_logging()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train the independent USC-HAD motion-primitive encoder."
    )
    parser.add_argument("--source-checkpoint", default="", help="Legacy fold-specific old6 checkpoint.")
    parser.add_argument("--npz-path", default="", help="Optional USC-HAD NPZ override.")
    parser.add_argument("--output-dir", default="", help="Empty/new output directory.")
    parser.add_argument("--self-test", action="store_true", help="Run a tiny synthetic CPU end-to-end smoke test.")
    parser.add_argument("--ablation-profile", choices=["A0", "A1", "A2", "A3", "A4", "CUSTOM", "a0", "a1", "a2", "a3", "a4", "custom"], default="A3")
    parser.add_argument("--window-aug-consistency", choices=["none", "infonce", "vicreg"], default=None)
    parser.add_argument("--window-aug-profile", choices=["basic", "basic_rotation"], default=None)
    parser.add_argument("--window-aug-weight", "--lambda-aug", dest="window_aug_weight", type=float, default=1.0)
    parser.add_argument("--window-aug-temperature", type=float, default=0.2)
    parser.add_argument("--window-aug-one-window-per-trial", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--noise-std-ratio", type=float, default=0.01)
    parser.add_argument("--acc-scale-min", type=float, default=0.95)
    parser.add_argument("--acc-scale-max", type=float, default=1.05)
    parser.add_argument("--gyro-scale-min", type=float, default=0.95)
    parser.add_argument("--gyro-scale-max", type=float, default=1.05)
    parser.add_argument("--time-shift-max-samples", type=int, default=8)
    parser.add_argument("--time-mask-min-samples", type=int, default=4)
    parser.add_argument("--time-mask-max-samples", type=int, default=12)
    parser.add_argument("--rotation-max-degrees", type=float, default=None)
    parser.add_argument("--vicreg-invariance-weight", type=float, default=25.0)
    parser.add_argument("--vicreg-variance-weight", type=float, default=25.0)
    parser.add_argument("--vicreg-covariance-weight", type=float, default=1.0)
    parser.add_argument("--vicreg-target-std", type=float, default=1.0)

    parser.add_argument("--cp-weight", "--lambda-cp", dest="cp_weight", type=float, default=None)
    parser.add_argument(
        "--content-boundary-alignment-weight",
        type=float,
        default=None,
        help=(
            "Low-weight CP supervision on the content head so detected "
            "boundaries remain visible to segment means/KMeans; A2-A4 use 0.1."
        ),
    )
    parser.add_argument(
        "--cp-anchor-source",
        choices=["raw_frozen_consensus", "frozen_legacy"],
        default=None,
        help="Pseudo-boundary source; consensus intersects raw kinematics and the frozen legacy encoder.",
    )
    parser.add_argument("--cp-context-windows", type=int, default=2)
    parser.add_argument("--cp-low-quantile", type=float, default=0.50)
    parser.add_argument("--cp-high-quantile", type=float, default=0.90)
    parser.add_argument("--cp-raw-scales", default="1,2,4", help="Median-stride radius multipliers.")
    parser.add_argument("--cp-raw-frequency-bins", type=int, default=16)
    parser.add_argument("--cp-raw-epsilon", type=float, default=1e-8)
    parser.add_argument("--cp-raw-scale-floor", type=float, default=1e-6)
    parser.add_argument("--cp-raw-z-clip", type=float, default=10.0)
    parser.add_argument("--cp-rank-margin", type=float, default=0.20)
    parser.add_argument("--cp-equivariance-weight", type=float, default=0.50)
    parser.add_argument("--cp-equivariance-delta", type=float, default=1.0)
    parser.add_argument("--noncollapse-weight", "--lambda-nc", dest="noncollapse_weight", type=float, default=0.05)
    parser.add_argument("--noncollapse-target-std", type=float, default=1.0)
    parser.add_argument("--noncollapse-variance-weight", type=float, default=1.0)
    parser.add_argument("--noncollapse-covariance-weight", type=float, default=1.0)
    parser.add_argument("--noncollapse-windows-per-trial", type=int, default=4)
    parser.add_argument("--prediction-weight", "--lambda-pred", dest="prediction_weight", type=float, default=0.5)
    parser.add_argument("--prediction-mask-ratio", type=float, default=0.20)
    parser.add_argument("--prediction-loss", choices=["cosine", "huber"], default="cosine")
    parser.add_argument("--prediction-huber-delta", type=float, default=1.0)
    parser.add_argument(
        "--trial-weight",
        "--lambda-trial",
        dest="trial_weight",
        type=float,
        default=0.1,
        help=(
            "Train-only activity-semantic auxiliary weight. The conservative "
            "0.1 default prevents the randomly initialized trial classifier "
            "from dominating local boundary/content gradients."
        ),
    )
    parser.add_argument("--cross-subject-weight", "--lambda-xsub", dest="cross_subject_weight", type=float, default=0.0)

    parser.add_argument(
        "--segmentation-dim",
        type=int,
        default=0,
        help=(
            "Boundary feature dimension; 0 uses the legacy backbone dimension "
            "and is required for the exact identity-residual A0-A4 baseline."
        ),
    )
    parser.add_argument("--content-dim", type=int, default=256)
    parser.add_argument("--content-residual", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--augmentation-dim", type=int, default=128)
    parser.add_argument("--projection-hidden-dim", type=int, default=256)
    parser.add_argument("--trial-hidden-dim", type=int, default=128)
    parser.add_argument("--trial-peak-quantile", type=float, default=0.90)
    parser.add_argument("--trial-dropout", type=float, default=0.0)
    parser.add_argument("--predictor-hidden-dim", type=int, default=0, help="0 uses content_dim.")
    parser.add_argument("--backbone-layers", default="2,2,2")
    parser.add_argument("--old-class-count", type=int, default=6)

    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--trial-batch-size", type=int, default=8)
    parser.add_argument("--source-encode-batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--minimum-learning-rate", type=float, default=1e-6)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--gradient-clip-norm", type=float, default=5.0)
    parser.add_argument("--ema-momentum", type=float, default=0.99)
    parser.add_argument("--freeze-backbone-epochs", type=int, default=0)
    parser.add_argument(
        "--backbone-bn-policy",
        choices=["frozen", "update"],
        default="frozen",
        help="Keep legacy BatchNorm1d running statistics fixed or update them during training.",
    )
    parser.add_argument("--early-stopping-patience", type=int, default=0)
    parser.add_argument("--minimum-improvement", type=float, default=0.0)
    parser.add_argument("--selection-policy", choices=["final_epoch", "best_val_total"], default="final_epoch")
    parser.add_argument("--normalization-eps", type=float, default=1e-6)
    parser.add_argument("--known-anomaly-policy", choices=["report", "exclude"], default="report")
    parser.add_argument("--smoke-max-train-trials", type=int, default=0, help="0 disables; positive values mark outputs smoke_test=true.")
    parser.add_argument("--smoke-max-val-trials", type=int, default=0, help="0 disables; positive values mark outputs smoke_test=true.")
    parser.add_argument("--seed", type=int, default=500)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:N")
    parser.add_argument("--deterministic", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Verify and reuse a completed identical member; incomplete members fail closed.",
    )
    return parser


def _make_synthetic_fixture(root: Path) -> tuple[Path, Path]:
    """Create a small but structurally real legacy checkpoint/USC-HAD NPZ."""

    rng = np.random.default_rng(17)
    window_size = 64
    stride = 32
    raw_windows = []
    labels = []
    labels_1based = []
    subject_ids = []
    trial_numbers = []
    trial_ids = []
    window_indices = []
    window_starts = []
    global_trial = 0
    for subject in range(1, 5):
        for label in range(2):
            time = np.arange(128, dtype=np.float32)
            channels = []
            for channel in range(6):
                signal = np.sin(time * (0.035 + 0.01 * label) + channel * 0.3)
                signal += 0.08 * subject + 0.15 * label
                signal += rng.normal(0.0, 0.01, size=len(time))
                channels.append(signal.astype(np.float32))
            raw = np.stack(channels)
            for index, start in enumerate(range(0, 128 - window_size + 1, stride)):
                raw_windows.append(raw[:, start : start + window_size])
                labels.append(label)
                labels_1based.append(label + 1)
                subject_ids.append(subject)
                trial_numbers.append(1)
                trial_ids.append(global_trial)
                window_indices.append(index)
                window_starts.append(start)
            global_trial += 1
    raw_windows_np = np.stack(raw_windows).astype(np.float32)
    mean = raw_windows_np.mean(axis=(0, 2), keepdims=True).astype(np.float32)
    std = raw_windows_np.std(axis=(0, 2), keepdims=True).astype(np.float32)
    windows = ((raw_windows_np - mean) / std).astype(np.float32)
    npz_path = root / "synthetic_uschad.npz"
    np.savez_compressed(
        npz_path,
        windows=windows,
        labels=np.asarray(labels, dtype=np.int64),
        labels_1based=np.asarray(labels_1based, dtype=np.int64),
        subject_ids=np.asarray(subject_ids, dtype=np.int64),
        trial_numbers=np.asarray(trial_numbers, dtype=np.int64),
        trial_global_ids=np.asarray(trial_ids, dtype=np.int64),
        window_indices=np.asarray(window_indices, dtype=np.int64),
        window_start_indices=np.asarray(window_starts, dtype=np.int64),
        mean=mean,
        std=std,
        channel_names=np.asarray(EXPECTED_CHANNELS, dtype=object),
    )
    torch.manual_seed(23)
    source = ResNet1D(in_channels=6, feat_dim=16, base_channels=4, layers=[2, 2, 2], dropout=0.0)
    state = {f"0.{key}": value for key, value in source.state_dict().items()}
    checkpoint_path = root / "synthetic_source.pt"
    torch.save(
        {
            "model": state,
        "experiment_metadata": {
                "uschad_npz_path": str(npz_path),
                "uschad_window_size": window_size,
                "uschad_split_mode": "subject",
                "uschad_recompute_norm_from_train_subjects": True,
                "uschad_train_subjects": [1, 2],
                "offline_val_subjects": [3],
                "uschad_test_subjects": [4],
                "uschad_cv_fold": 99,
                "har_in_channels": 6,
                "har_feat_dim": 16,
                "har_base_channels": 4,
                "har_dropout": 0.0,
            },
        },
        checkpoint_path,
    )
    return checkpoint_path, npz_path


def run_self_test() -> dict:
    try:
        with tempfile.TemporaryDirectory(prefix="motion_encoder_smoke_") as temporary:
            root = Path(temporary)
            checkpoint, npz = _make_synthetic_fixture(root)
            parser = build_parser()
            args = parser.parse_args(
                [
                    "--source-checkpoint", str(checkpoint),
                    "--npz-path", str(npz),
                    "--output-dir", str(root / "output"),
                    "--ablation-profile", "A3",
                    "--old-class-count", "2",
                    "--content-dim", "16",
                    "--augmentation-dim", "8",
                    "--projection-hidden-dim", "16",
                    "--trial-hidden-dim", "8",
                    "--predictor-hidden-dim", "8",
                    "--epochs", "1",
                    "--trial-batch-size", "2",
                    "--source-encode-batch-size", "8",
                    "--smoke-max-train-trials", "4",
                    "--smoke-max-val-trials", "2",
                    "--device", "cpu",
                    "--seed", "31",
                ]
            )
            final_path = run_training(args)
            checkpoint_value = _load_checkpoint(final_path)
            _validate_output_checkpoint(checkpoint_value)
            audit = checkpoint_value["split_audit"]
            result = {
                "status": "passed",
                "checkpoint_type": checkpoint_value["checkpoint_type"],
                "schema_version": checkpoint_value["schema_version"],
                "train_trials": audit["train_trial_count"],
                "validation_trials": audit["validation_trial_count"],
                "outer_test_sensor_windows_selected": audit["outer_test_sensor_windows_selected"],
                "outer_test_model_forward_calls": audit["outer_test_model_forward_calls"],
                "strict_reload": True,
                "finite_train_loss": math.isfinite(checkpoint_value["history"][0]["train"]["total_loss"]),
                "finite_validation_loss": math.isfinite(checkpoint_value["history"][0]["validation"]["total_loss"]),
            }
            if not all((result["strict_reload"], result["finite_train_loss"], result["finite_validation_loss"])):
                raise RuntimeError(f"Synthetic smoke test failed: {result}")
            return result
    finally:
        _close_logging()


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.self_test:
        print(json.dumps(run_self_test(), ensure_ascii=False, indent=2))
        return
    final_path = run_training(args)
    print(f"Motion primitive encoder checkpoint: {final_path}")


if __name__ == "__main__":
    main()
