"""Run the frozen-encoder motion-primitive sequence feasibility experiment.

This script intentionally stops before trajectory memory, novelty detection, or
Happy online training.  It asks only whether train-only latent motion tokens are
non-collapsed and whether held-out-subject trials from the same activity have
more related token sequences than trials from different activities.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import os
import platform
import re
import sys
import warnings
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Iterable, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count() or 1))
warnings.filterwarnings(
    "ignore", message="Could not find the number of physical cores.*"
)
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    module=r"joblib\.externals\.loky\.backend\.context",
)
import sklearn
import torch
from sklearn.cluster import KMeans

from experiments.motion_primitive.core import (
    activity_distance_matrix,
    assign_to_codebook,
    association_summary,
    build_trial_sequence,
    cross_subject_nearest_neighbor_accuracy,
    cross_subject_pair_indices,
    distance_matrix_from_histograms,
    distance_matrix_from_sequences,
    fit_weighted_pca,
    inverse_trial_frequency_weights,
    l2_normalize,
    label_permutation_test,
    length_distance_matrix,
    primitive_histogram,
    shuffle_valid_rle_tokens,
)
from experiments.motion_primitive.segmentation import (
    build_segmented_features,
    calibrate_changepoint_threshold,
    codebook_segment_weights,
    segmentation_summary,
    train_masked_denoising_adapter,
    transform_with_adapter,
)
from experiments.motion_primitive.motion_encoder import MotionPrimitiveEncoder
from experiments.motion_primitive.motion_checkpoint import (
    validate_motion_encoder_checkpoint_integrity,
)
from models.resnet1d import ResNet1D


LOGGER = logging.getLogger("motion_primitive")
SEQUENCE_METRIC_REVISION = "tie_aware_1nn_and_valid_rle_shuffle_v2"


# This is reported by default, not silently removed.  The policy can be changed
# to ``exclude`` for an explicit sensitivity run.
KNOWN_LABEL_ANOMALIES = [
    {
        "subject_id": 14,
        "activity_label_1based": 3,
        "trial_number": 2,
        "source_key": "Subject14/a3t2.mat",
        "reason": (
            "filename says activity 3, while internal activity_number/text say "
            "activity 2; current preprocessing follows the filename"
        ),
    }
]


def parse_int_list(value: Optional[str]) -> Optional[list[int]]:
    if value is None or str(value).strip() == "":
        return None
    tokens = re.split(r"[\s,]+", str(value).strip())
    return sorted(set(int(token) for token in tokens if token))


def jsonable(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    return value


def write_json(path: Path, value) -> None:
    path.write_text(
        json.dumps(jsonable(value), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(int(chunk_size))
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def load_torch_checkpoint(path: Path) -> dict:
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location="cpu")
    if not isinstance(checkpoint, dict):
        raise TypeError(f"Checkpoint must contain a dictionary, got {type(checkpoint)}.")
    return checkpoint


def resolve_npz_path(checkpoint_path: Path, metadata: dict, explicit: str) -> Path:
    if explicit:
        return Path(explicit).expanduser().resolve()
    recorded = str(metadata.get("uschad_npz_path", "")).strip()
    if not recorded:
        raise ValueError("Checkpoint metadata has no uschad_npz_path; pass --npz-path.")
    candidate = Path(recorded)
    if not candidate.is_absolute():
        candidate = PROJECT_ROOT / candidate
    return candidate.resolve()


def checkpoint_seed(checkpoint_path: Path, fallback: int) -> int:
    match = re.search(r"seed_(\d+)_offline", str(checkpoint_path))
    return int(match.group(1)) if match else int(fallback)


def identify_checkpoint_type(checkpoint: dict, metadata: dict) -> str:
    """Return an explicit checkpoint family without guessing from its path."""

    top_level = str(checkpoint.get("checkpoint_type", "")).strip()
    metadata_type = str(metadata.get("checkpoint_type", "")).strip()
    if top_level and metadata_type and top_level != metadata_type:
        raise RuntimeError(
            "Conflicting checkpoint_type values at top level and in metadata: "
            f"{top_level!r} != {metadata_type!r}."
        )
    resolved = top_level or metadata_type
    if not resolved:
        return "legacy_happy_encoder"
    if resolved != "motion_primitive_encoder":
        raise RuntimeError(f"Unsupported checkpoint_type {resolved!r}.")
    if int(checkpoint.get("schema_version", -1)) != 1:
        raise RuntimeError(
            "Motion-primitive checkpoints require schema_version=1; got "
            f"{checkpoint.get('schema_version')!r}."
        )
    return resolved


def configure_output_directory(args, metadata: dict) -> Path:
    if args.output_dir:
        output_dir = Path(args.output_dir).expanduser().resolve()
    else:
        fold = int(metadata.get("uschad_cv_fold", -1))
        seed = int(
            metadata.get(
                "motion_encoder_seed",
                checkpoint_seed(Path(args.checkpoint), args.seed),
            )
        )
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        segmentation_suffix = (
            "" if args.primitive_segmentation == "fixed_window"
            else f"_{args.primitive_segmentation}"
        )
        output_dir = (
            PROJECT_ROOT
            / "results"
            / "motion_primitive"
            / (
                f"fold_{fold:02d}_seed_{seed}_k{args.primitive_num}"
                f"{segmentation_suffix}_{timestamp}"
            )
        )
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            f"Output directory is not empty: {output_dir}. Choose a new directory."
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def configure_logging(output_dir: Path) -> None:
    LOGGER.setLevel(logging.INFO)
    LOGGER.handlers.clear()
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(formatter)
    file_handler = logging.FileHandler(output_dir / "run.log", encoding="utf-8")
    file_handler.setFormatter(formatter)
    LOGGER.addHandler(stream)
    LOGGER.addHandler(file_handler)


def load_npz_arrays(path: Path) -> dict[str, np.ndarray]:
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
    with np.load(path, allow_pickle=True) as npz:
        missing = required - set(npz.files)
        if missing:
            raise RuntimeError(f"NPZ is missing fields: {sorted(missing)}")
        arrays = {name: np.asarray(npz[name]) for name in required}
        arrays["_activity_names_raw"] = (
            np.asarray(npz["activity_names"], dtype=object)
            if "activity_names" in npz.files
            else np.asarray([], dtype=object)
        )
    arrays["windows"] = np.asarray(arrays["windows"], dtype=np.float32)
    for name in [
        "labels",
        "labels_1based",
        "subject_ids",
        "trial_numbers",
        "trial_global_ids",
        "window_indices",
        "window_start_indices",
    ]:
        arrays[name] = np.asarray(arrays[name], dtype=np.int64)
    arrays["mean"] = np.asarray(arrays["mean"], dtype=np.float32)
    arrays["std"] = np.asarray(arrays["std"], dtype=np.float32)
    sample_count = len(arrays["windows"])
    for name in required - {"windows", "mean", "std"}:
        if len(arrays[name]) != sample_count:
            raise RuntimeError(f"{name} length does not match windows.shape[0].")
    if arrays["windows"].ndim != 3:
        raise RuntimeError(f"Expected windows [N,C,T], got {arrays['windows'].shape}.")
    if not np.all(np.isfinite(arrays["windows"])):
        raise RuntimeError("NPZ windows contain non-finite values.")
    raw_names = arrays.pop("_activity_names_raw")
    class_count = int(arrays["labels"].max()) + 1
    if raw_names.shape == (sample_count,):
        class_names = []
        for label in range(class_count):
            names = sorted(
                set(str(value) for value in raw_names[arrays["labels"] == label])
            )
            if len(names) != 1:
                raise RuntimeError(
                    f"Activity label {label} maps to inconsistent names: {names}."
                )
            class_names.append(names[0])
        arrays["activity_names"] = np.asarray(class_names, dtype=object)
    elif raw_names.shape == (class_count,):
        arrays["activity_names"] = raw_names
    else:
        arrays["activity_names"] = np.asarray(
            [f"activity_{index + 1}" for index in range(class_count)], dtype=object
        )
    return arrays


def anomaly_window_mask(arrays: dict[str, np.ndarray]) -> np.ndarray:
    mask = np.zeros(len(arrays["labels"]), dtype=bool)
    for anomaly in KNOWN_LABEL_ANOMALIES:
        mask |= (
            (arrays["subject_ids"] == anomaly["subject_id"])
            & (arrays["labels_1based"] == anomaly["activity_label_1based"])
            & (arrays["trial_numbers"] == anomaly["trial_number"])
        )
    return mask


def infer_stride(arrays: dict[str, np.ndarray]) -> tuple[int, dict[int, int]]:
    differences = []
    for trial_id in np.unique(arrays["trial_global_ids"]):
        indices = np.flatnonzero(arrays["trial_global_ids"] == trial_id)
        starts = np.sort(arrays["window_start_indices"][indices])
        differences.extend(np.diff(starts).tolist())
    if not differences:
        raise RuntimeError("Cannot infer stride from single-window trials only.")
    counts = Counter(int(value) for value in differences)
    stride, _ = counts.most_common(1)[0]
    return int(stride), dict(sorted(counts.items()))


def build_split_masks(
    arrays: dict[str, np.ndarray],
    fit_subjects: Iterable[int],
    eval_subjects: Iterable[int],
    old_class_count: int,
    anomaly_policy: str,
) -> tuple[np.ndarray, np.ndarray, dict]:
    fit_subjects = sorted(set(int(value) for value in fit_subjects))
    eval_subjects = sorted(set(int(value) for value in eval_subjects))
    overlap = sorted(set(fit_subjects) & set(eval_subjects))
    if overlap:
        raise RuntimeError(f"Fit/evaluation subject overlap: {overlap}")
    fit_mask = np.isin(arrays["subject_ids"], fit_subjects)
    fit_mask &= arrays["labels"] < int(old_class_count)
    eval_mask = np.isin(arrays["subject_ids"], eval_subjects)

    anomaly_mask = anomaly_window_mask(arrays)
    anomaly_trials = sorted(
        set(arrays["trial_global_ids"][anomaly_mask].astype(int).tolist())
    )
    if anomaly_policy == "exclude":
        fit_mask &= ~anomaly_mask
        eval_mask &= ~anomaly_mask
    elif anomaly_policy != "report":
        raise ValueError(f"Unknown anomaly policy {anomaly_policy!r}.")

    if not np.any(fit_mask):
        raise RuntimeError("No Stage-0 fit windows were selected.")
    if not np.any(eval_mask):
        raise RuntimeError("No held-out-subject evaluation windows were selected.")
    if np.any(fit_mask & eval_mask):
        raise RuntimeError("A window was selected for both codebook fit and evaluation.")
    return fit_mask, eval_mask, {
        "fit_subjects": fit_subjects,
        "eval_subjects": eval_subjects,
        "subject_overlap": overlap,
        "old_class_ids_0based": list(range(int(old_class_count))),
        "anomaly_policy": anomaly_policy,
        "known_anomalies": KNOWN_LABEL_ANOMALIES,
        "known_anomaly_trial_global_ids_in_npz": anomaly_trials,
        "known_anomaly_windows_in_fit_before_policy": int(
            np.sum(anomaly_mask & np.isin(arrays["subject_ids"], fit_subjects))
        ),
        "known_anomaly_windows_in_eval_before_policy": int(
            np.sum(anomaly_mask & np.isin(arrays["subject_ids"], eval_subjects))
        ),
    }


def normalize_for_checkpoint(
    arrays: dict[str, np.ndarray],
    fit_mask: np.ndarray,
    selected_indices: np.ndarray,
    metadata: dict,
    eps: float,
) -> tuple[np.ndarray, dict]:
    windows = arrays["windows"]
    if bool(metadata.get("uschad_recompute_norm_from_train_subjects", False)):
        stored_mean = arrays["mean"]
        stored_std = arrays["std"]
        raw_fit = windows[fit_mask] * stored_std + stored_mean
        fold_mean = raw_fit.mean(axis=(0, 2), keepdims=True, dtype=np.float64).astype(
            np.float32
        )
        fold_std = raw_fit.std(axis=(0, 2), keepdims=True, dtype=np.float64).astype(
            np.float32
        )
        fold_std = np.maximum(fold_std, np.float32(eps))
        raw_selected = windows[selected_indices] * stored_std + stored_mean
        normalized = ((raw_selected - fold_mean) / fold_std).astype(np.float32)
        details = {
            "mode": "fold_train_subjects_old_classes",
            "raw_source": "reconstructed_from_npz_windows_mean_std",
            "stat_window_count": int(np.sum(fit_mask)),
            "mean": fold_mean,
            "std": fold_std,
        }
    else:
        normalized = windows[selected_indices].astype(np.float32, copy=True)
        details = {
            "mode": "npz_stored",
            "raw_source": "not_reconstructed",
            "stat_window_count": None,
            "mean": arrays["mean"],
            "std": arrays["std"],
        }
    if not np.all(np.isfinite(normalized)):
        raise RuntimeError("Checkpoint-normalized windows contain non-finite values.")
    return normalized, details


def build_frozen_encoder(checkpoint: dict, metadata: dict) -> ResNet1D:
    encoder = ResNet1D(
        in_channels=int(metadata.get("har_in_channels", 6)),
        feat_dim=int(metadata.get("har_feat_dim", 256)),
        base_channels=int(metadata.get("har_base_channels", 64)),
        dropout=float(metadata.get("har_dropout", 0.0)),
    )
    source_state = checkpoint.get("model", checkpoint)
    if not isinstance(source_state, dict):
        raise TypeError("Checkpoint model state must be a dictionary.")
    extracted = {}
    prefixes = ["0.window_encoder.", "window_encoder.", "0.", ""]
    source_keys = {}
    for target_key, target_value in encoder.state_dict().items():
        matches = [prefix + target_key for prefix in prefixes if prefix + target_key in source_state]
        if not matches:
            raise RuntimeError(f"Checkpoint has no ResNet1D tensor for {target_key}.")
        source_key = matches[0]
        source_value = source_state[source_key]
        if tuple(source_value.shape) != tuple(target_value.shape):
            raise RuntimeError(
                f"Shape mismatch for {source_key}: {tuple(source_value.shape)} vs "
                f"{tuple(target_value.shape)}."
            )
        extracted[target_key] = source_value
        source_keys[target_key] = source_key
    encoder.load_state_dict(extracted, strict=True)
    encoder.eval()
    for parameter in encoder.parameters():
        parameter.requires_grad_(False)
    encoder.loaded_source_keys = source_keys
    return encoder


def build_frozen_motion_encoder(
    checkpoint: dict, metadata: dict
) -> MotionPrimitiveEncoder:
    """Strictly reconstruct the separately trained dual-role encoder."""

    source_state = validate_motion_encoder_checkpoint_integrity(checkpoint)
    if identify_checkpoint_type(checkpoint, metadata) != "motion_primitive_encoder":
        raise RuntimeError("Expected a motion_primitive_encoder checkpoint.")
    feature_roles = metadata.get("feature_roles")
    if feature_roles != {"codebook": "content", "boundary": "segmentation"}:
        raise RuntimeError(
            "Motion checkpoint must declare feature_roles exactly as "
            "{'codebook': 'content', 'boundary': 'segmentation'}; got "
            f"{feature_roles!r}."
        )
    architecture = checkpoint.get("architecture")
    if not isinstance(architecture, dict):
        raise RuntimeError("Motion checkpoint lacks an architecture dictionary.")
    required = {
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
    }
    missing = required - set(architecture)
    if missing:
        raise RuntimeError(
            "Motion checkpoint architecture is incomplete; missing "
            f"{sorted(missing)}."
        )
    model = MotionPrimitiveEncoder(
        in_channels=int(architecture["in_channels"]),
        backbone_dim=int(architecture["backbone_dim"]),
        base_channels=int(architecture["base_channels"]),
        backbone_layers=[int(value) for value in architecture["backbone_layers"]],
        backbone_dropout=float(architecture["backbone_dropout"]),
        segmentation_dim=int(architecture["segmentation_dim"]),
        segmentation_residual=bool(architecture["segmentation_residual"]),
        content_dim=int(architecture["content_dim"]),
        content_residual=bool(architecture["content_residual"]),
        augmentation_dim=int(architecture["augmentation_dim"]),
        projection_hidden_dim=int(architecture["projection_hidden_dim"]),
        num_classes=int(architecture["num_classes"]),
        trial_hidden_dim=int(architecture["trial_hidden_dim"]),
        trial_peak_quantile=float(architecture["trial_peak_quantile"]),
        trial_dropout=float(architecture["trial_dropout"]),
        predictor_hidden_dim=(
            None
            if architecture["predictor_hidden_dim"] is None
            else int(architecture["predictor_hidden_dim"])
        ),
    )
    # strict=True is intentional: silently falling back to a partial backbone
    # would invalidate the declared segmentation/content feature roles.
    model.load_state_dict(source_state, strict=True)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.checkpoint_architecture = dict(architecture)
    return model


def choose_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false.")
    return device


def encode_windows(
    encoder: ResNet1D,
    windows: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    encoder = encoder.to(device)
    output = np.empty((len(windows), encoder.feat_dim), dtype=np.float32)
    with torch.inference_mode():
        for begin in range(0, len(windows), int(batch_size)):
            end = min(begin + int(batch_size), len(windows))
            batch = torch.from_numpy(windows[begin:end]).to(device=device)
            features = encoder(batch)
            if features.shape != (end - begin, encoder.feat_dim):
                raise RuntimeError(f"Unexpected encoder output {tuple(features.shape)}.")
            output[begin:end] = features.detach().cpu().numpy().astype(np.float32)
    return output


def encode_motion_windows(
    encoder: MotionPrimitiveEncoder,
    windows: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> dict[str, np.ndarray]:
    """Encode content and boundary roles without trial pooling or adaptation."""

    encoder = encoder.to(device)
    dimensions = {
        "backbone": int(encoder.backbone_dim),
        "content": int(encoder.content_dim),
        "segmentation": int(encoder.segmentation_dim),
    }
    output = {
        name: np.empty((len(windows), dim), dtype=np.float32)
        for name, dim in dimensions.items()
    }
    with torch.inference_mode():
        for begin in range(0, len(windows), int(batch_size)):
            end = min(begin + int(batch_size), len(windows))
            batch = torch.from_numpy(windows[begin:end]).to(device=device)
            encoded = encoder.encode_windows(batch)
            for name, dimension in dimensions.items():
                values = encoded[name]
                if tuple(values.shape) != (end - begin, dimension):
                    raise RuntimeError(
                        f"Unexpected {name} output {tuple(values.shape)}."
                    )
                output[name][begin:end] = (
                    values.detach().cpu().numpy().astype(np.float32)
                )
    for name, values in output.items():
        if not np.all(np.isfinite(values)):
            raise RuntimeError(f"Motion encoder produced non-finite {name} features.")
    return output


def prepare_primitive_features(
    fit_embeddings: np.ndarray,
    eval_embeddings: np.ndarray,
    fit_weights: np.ndarray,
    pca_dim: int,
    normalization: str,
) -> tuple[np.ndarray, np.ndarray, Optional[object]]:
    if normalization == "l2":
        fit_base = l2_normalize(fit_embeddings)
        eval_base = l2_normalize(eval_embeddings)
    elif normalization == "none":
        fit_base = np.asarray(fit_embeddings, dtype=np.float32)
        eval_base = np.asarray(eval_embeddings, dtype=np.float32)
    else:
        raise ValueError(f"Unknown embedding normalization {normalization!r}.")

    pca = None
    if int(pca_dim) > 0:
        pca = fit_weighted_pca(fit_base, int(pca_dim), fit_weights)
        fit_features = pca.transform(fit_base)
        eval_features = pca.transform(eval_base)
    else:
        fit_features = fit_base.copy()
        eval_features = eval_base.copy()
    if normalization == "l2":
        fit_features = l2_normalize(fit_features)
        eval_features = l2_normalize(eval_features)
    return fit_features, eval_features, pca


def usage_summary(tokens: np.ndarray, primitive_num: int) -> dict:
    counts = np.bincount(tokens, minlength=int(primitive_num)).astype(np.int64)
    probabilities = counts / max(int(counts.sum()), 1)
    positive = probabilities > 0
    entropy = float(-np.sum(probabilities[positive] * np.log(probabilities[positive])))
    used = int(np.sum(counts > 0))
    return {
        "token_count": int(counts.sum()),
        "counts": counts.tolist(),
        "frequencies": probabilities.tolist(),
        "used_primitives": used,
        "total_primitives": int(primitive_num),
        "utilization": float(used / int(primitive_num)),
        "perplexity_effective_k": float(np.exp(entropy)),
        "normalized_entropy": float(entropy / max(np.log(int(primitive_num)), 1e-12)),
        "max_token_share": float(probabilities.max(initial=0.0)),
        "collapse_flags": {
            "utilization_below_half": bool(used < 0.5 * int(primitive_num)),
            "single_token_above_half": bool(probabilities.max(initial=0.0) > 0.5),
        },
    }


def boundary_to_codebook_alignment(
    segmented, segment_tokens: np.ndarray
) -> dict:
    """Diagnose whether detected adjacent segments remain distinct downstream."""

    tokens = np.asarray(segment_tokens, dtype=np.int64)
    trial_ids = np.asarray(segmented.segment_trial_ids, dtype=np.int64)
    features = np.asarray(segmented.segment_features, dtype=np.float32)
    if len(tokens) != len(trial_ids) or len(tokens) != len(features):
        raise ValueError("Segment tokens/features/trial ids must have equal length.")
    pair_count = 0
    changed_token_count = 0
    cosine_distances = []
    for trial_id in np.unique(trial_ids):
        positions = np.flatnonzero(trial_ids == trial_id)
        if len(positions) < 2:
            continue
        left = positions[:-1]
        right = positions[1:]
        pair_count += len(left)
        changed_token_count += int(np.sum(tokens[left] != tokens[right]))
        left_values = features[left]
        right_values = features[right]
        denominator = np.maximum(
            np.linalg.norm(left_values, axis=1)
            * np.linalg.norm(right_values, axis=1),
            1e-8,
        )
        cosine = np.clip(
            np.sum(left_values * right_values, axis=1) / denominator,
            -1.0,
            1.0,
        )
        cosine_distances.extend((1.0 - cosine).tolist())
    return {
        "adjacent_segment_pair_count": int(pair_count),
        "different_token_pair_count": int(changed_token_count),
        "different_token_ratio": (
            float(changed_token_count / pair_count) if pair_count else None
        ),
        "same_token_after_boundary_ratio": (
            float(1.0 - changed_token_count / pair_count) if pair_count else None
        ),
        "adjacent_segment_content_cosine_distance_mean": (
            float(np.mean(cosine_distances)) if cosine_distances else None
        ),
        "adjacent_segment_content_cosine_distance_median": (
            float(np.median(cosine_distances)) if cosine_distances else None
        ),
    }


def build_trial_records(
    arrays: dict[str, np.ndarray],
    eval_global_indices: np.ndarray,
    eval_tokens: np.ndarray,
    eval_distances: np.ndarray,
    eval_segmentation,
    primitive_segmentation: str,
    primitive_num: int,
    window_size: int,
    sample_rate_hz: float,
    edge_trim_ratio: float,
) -> list[dict]:
    activity_names = [str(value) for value in arrays["activity_names"].tolist()]
    eval_trial_ids = arrays["trial_global_ids"][eval_global_indices]
    records = []
    for trial_id in np.unique(eval_trial_ids):
        local = np.flatnonzero(eval_trial_ids == trial_id)
        global_indices = eval_global_indices[local]
        order = np.argsort(arrays["window_start_indices"][global_indices], kind="stable")
        local = local[order]
        global_indices = global_indices[order]
        starts = arrays["window_start_indices"][global_indices]
        window_indices = arrays["window_indices"][global_indices]
        if np.any(np.diff(starts) <= 0):
            raise RuntimeError(f"Trial {trial_id} starts are not strictly increasing.")
        if not np.array_equal(window_indices, np.arange(len(window_indices))):
            raise RuntimeError(f"Trial {trial_id} window indices are incomplete or unordered.")

        fields = {}
        for name in ["labels", "labels_1based", "subject_ids", "trial_numbers"]:
            values = np.unique(arrays[name][global_indices])
            if len(values) != 1:
                raise RuntimeError(f"Trial {trial_id} has inconsistent {name}: {values}.")
            fields[name] = int(values[0])
        tokens = eval_tokens[local]
        distances = eval_distances[local]
        ordered_segment_ids = eval_segmentation.window_segment_ids[local]
        if np.any(np.diff(ordered_segment_ids) < 0):
            raise RuntimeError(f"Trial {trial_id} segment ids are not ordered.")
        segment_starts = np.flatnonzero(
            np.r_[True, ordered_segment_ids[1:] != ordered_segment_ids[:-1]]
        )
        segment_ends = np.r_[segment_starts[1:], len(ordered_segment_ids)]
        primitive_segments = []
        for begin, end in zip(segment_starts, segment_ends):
            segment_id = int(ordered_segment_ids[int(begin)])
            member_tokens = np.unique(tokens[int(begin) : int(end)])
            if len(member_tokens) != 1:
                raise RuntimeError(
                    f"Trial {trial_id} segment {segment_id} has multiple tokens."
                )
            boundary_score = float(
                eval_segmentation.segment_boundary_scores[segment_id]
            )
            primitive_segments.append(
                {
                    "segment_id_within_eval_split": segment_id,
                    "primitive_id": int(member_tokens[0]),
                    "first_window_offset": int(begin),
                    "last_window_offset": int(end - 1),
                    "window_count": int(end - begin),
                    "support_start_sample": int(
                        eval_segmentation.segment_support_start_samples[segment_id]
                    ),
                    "support_end_sample_exclusive": int(
                        eval_segmentation.segment_support_end_samples_exclusive[
                            segment_id
                        ]
                    ),
                    "partition_start_sample": int(
                        eval_segmentation.segment_partition_start_samples[segment_id]
                    ),
                    "partition_end_sample_exclusive": int(
                        eval_segmentation.segment_partition_end_samples_exclusive[
                            segment_id
                        ]
                    ),
                    "observed_support_span_seconds": float(
                        (
                            eval_segmentation.segment_support_end_samples_exclusive[
                                segment_id
                            ]
                            - eval_segmentation.segment_support_start_samples[segment_id]
                        )
                        / float(sample_rate_hz)
                    ),
                    "partition_duration_seconds": float(
                        (
                            eval_segmentation.segment_partition_end_samples_exclusive[
                                segment_id
                            ]
                            - eval_segmentation.segment_partition_start_samples[
                                segment_id
                            ]
                        )
                        / float(sample_rate_hz)
                    ),
                    "boundary_score_at_start": (
                        None if np.isnan(boundary_score) else boundary_score
                    ),
                    "nearest_center_distance_mean": float(
                        np.mean(distances[int(begin) : int(end)])
                    ),
                }
            )
        variants = {}
        variant_indices = {"full": np.arange(len(tokens), dtype=np.int64)}
        variant_indices["nonoverlap_every_second_window"] = np.arange(
            0, len(tokens), 2, dtype=np.int64
        )
        trim = int(np.floor(len(tokens) * float(edge_trim_ratio)))
        if trim > 0 and len(tokens) - 2 * trim >= 1:
            variant_indices["edge_trimmed"] = np.arange(trim, len(tokens) - trim)
        else:
            variant_indices["edge_trimmed"] = np.arange(len(tokens), dtype=np.int64)
        for variant_name, selected in variant_indices.items():
            sequence = build_trial_sequence(
                tokens[selected], starts[selected], int(window_size), float(sample_rate_hz)
            )
            sequence["primitive_histogram"] = primitive_histogram(
                tokens[selected], primitive_num
            ).tolist()
            variants[variant_name] = sequence

        label = fields["labels"]
        subject_id = fields["subject_ids"]
        trial_number = fields["trial_numbers"]
        records.append(
            {
                "trial_key": f"S{subject_id:02d}-A{label + 1:02d}-T{trial_number}",
                "trial_global_id_within_npz": int(trial_id),
                "subject_id": subject_id,
                "activity_label_0based": label,
                "activity_label_1based": fields["labels_1based"],
                "activity_name": activity_names[label],
                "trial_number": trial_number,
                "window_global_indices": global_indices.astype(int).tolist(),
                "window_start_indices": starts.astype(int).tolist(),
                "nearest_center_distance_mean": float(np.mean(distances)),
                "nearest_center_distance_max": float(np.max(distances)),
                "primitive_segmentation": {
                    "method": str(primitive_segmentation),
                    "segment_count": int(len(primitive_segments)),
                    "segments": primitive_segments,
                    "note": (
                        "Support spans may overlap because source windows overlap; "
                        "partition spans are non-overlapping midpoint partitions."
                    ),
                },
                "sequence_variants": variants,
            }
        )
    records.sort(key=lambda record: (record["subject_id"], record["activity_label_0based"], record["trial_number"]))
    return records


def analyze_distance_matrix(
    matrix: np.ndarray,
    labels: np.ndarray,
    subjects: np.ndarray,
    old_class_count: int,
    permutations: int,
    seed: int,
) -> dict:
    class_groups = {
        "all_classes": list(range(int(labels.max()) + 1)),
        "old_classes": list(range(int(old_class_count))),
        "novel_classes_diagnostic_only": list(
            range(int(old_class_count), int(labels.max()) + 1)
        ),
    }
    results = {}
    for offset, (name, class_ids) in enumerate(class_groups.items()):
        if len(class_ids) < 2 or not np.all(np.isin(class_ids, np.unique(labels))):
            continue
        results[name] = label_permutation_test(
            matrix,
            labels,
            subjects,
            class_ids,
            permutations=int(permutations),
            seed=int(seed) + offset * 1009,
        )
    return results


def fast_margin_and_nn(
    matrix: np.ndarray,
    labels: np.ndarray,
    subjects: np.ndarray,
    class_ids: list[int],
) -> tuple[float, float]:
    left, right = cross_subject_pair_indices(labels, subjects, class_ids)
    pair_distances = matrix[left, right]
    same = labels[left] == labels[right]
    margin = float(np.mean(pair_distances[~same]) - np.mean(pair_distances[same]))
    accuracy, _ = cross_subject_nearest_neighbor_accuracy(
        matrix, labels, subjects, class_ids
    )
    return margin, accuracy


def order_shuffle_control(
    full_sequences: list[list[int]],
    observed_matrix: np.ndarray,
    labels: np.ndarray,
    subjects: np.ndarray,
    old_class_count: int,
    shuffles: int,
    seed: int,
) -> dict:
    class_groups = {
        "all_classes": list(range(int(labels.max()) + 1)),
        "old_classes": list(range(int(old_class_count))),
        "novel_classes_diagnostic_only": list(
            range(int(old_class_count), int(labels.max()) + 1)
        ),
    }
    observed = {
        name: fast_margin_and_nn(observed_matrix, labels, subjects, class_ids)
        for name, class_ids in class_groups.items()
    }
    if int(shuffles) <= 0:
        return {
            name: {
                "shuffles": 0,
                "observed_margin": values[0],
                "observed_1nn_accuracy": values[1],
            }
            for name, values in observed.items()
        }
    rng = np.random.default_rng(int(seed))
    margins = {name: [] for name in class_groups}
    accuracies = {name: [] for name in class_groups}
    for _ in range(int(shuffles)):
        shuffled_sequences = [
            shuffle_valid_rle_tokens(sequence, rng) for sequence in full_sequences
        ]
        shuffled_matrix = distance_matrix_from_sequences(shuffled_sequences)
        for name, class_ids in class_groups.items():
            margin, accuracy = fast_margin_and_nn(
                shuffled_matrix, labels, subjects, class_ids
            )
            margins[name].append(margin)
            accuracies[name].append(accuracy)
    results = {}
    for name in class_groups:
        observed_margin, observed_accuracy = observed[name]
        null_margin = np.asarray(margins[name], dtype=np.float64)
        null_accuracy = np.asarray(accuracies[name], dtype=np.float64)
        results[name] = {
            "shuffles": int(shuffles),
            "shuffle_algorithm": "valid_rle_permutation_v2",
            "control_preserves": (
                "RLE token multiset and run count without equal adjacent tokens; "
                "destroys run order"
            ),
            "observed_margin": observed_margin,
            "shuffled_margin_mean": float(np.mean(null_margin)),
            "observed_minus_shuffled_margin": float(
                observed_margin - np.mean(null_margin)
            ),
            "margin_p_value_observed_greater": float(
                (1 + np.sum(null_margin >= observed_margin)) / (int(shuffles) + 1)
            ),
            "observed_1nn_accuracy": observed_accuracy,
            "shuffled_1nn_accuracy_mean": float(np.mean(null_accuracy)),
            "one_nn_p_value_observed_greater": float(
                (1 + np.sum(null_accuracy >= observed_accuracy)) / (int(shuffles) + 1)
            ),
        }
    return results


def primitive_coverage_statistics(records: list[dict], primitive_num: int) -> list[dict]:
    windows = np.zeros(int(primitive_num), dtype=np.int64)
    trials = [set() for _ in range(int(primitive_num))]
    subjects = [set() for _ in range(int(primitive_num))]
    activities = [Counter() for _ in range(int(primitive_num))]
    run_lengths = [[] for _ in range(int(primitive_num))]
    for record in records:
        full = record["sequence_variants"]["full"]
        for token in full["raw_tokens"]:
            windows[int(token)] += 1
            trials[int(token)].add(record["trial_key"])
            subjects[int(token)].add(record["subject_id"])
            activities[int(token)][record["activity_label_1based"]] += 1
        for run in full["runs"]:
            run_lengths[run["primitive_id"]].append(run["run_length_windows"])
    result = []
    for primitive_id in range(int(primitive_num)):
        top_activities = [
            {"activity_label_1based": int(label), "window_count": int(count)}
            for label, count in activities[primitive_id].most_common(5)
        ]
        result.append(
            {
                "primitive_id": primitive_id,
                "window_count": int(windows[primitive_id]),
                "trial_count": int(len(trials[primitive_id])),
                "subject_count": int(len(subjects[primitive_id])),
                "activity_count": int(len(activities[primitive_id])),
                "mean_run_length_windows": (
                    float(np.mean(run_lengths[primitive_id]))
                    if run_lengths[primitive_id]
                    else 0.0
                ),
                "top_activities": top_activities,
            }
        )
    return result


def grouped_oov_statistics(
    distances: np.ndarray,
    labels: np.ndarray,
    subjects: np.ndarray,
    threshold: float,
    old_class_count: int,
) -> dict:
    distances = np.asarray(distances, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    subjects = np.asarray(subjects, dtype=np.int64)
    if distances.shape != labels.shape or labels.shape != subjects.shape:
        raise ValueError("OOV distances, labels, and subjects must have equal shapes.")

    def summarize(mask: np.ndarray) -> dict:
        selected = distances[mask]
        if len(selected) == 0:
            return {"window_count": 0, "fit_p95_exceedance_ratio": None}
        return {
            "window_count": int(len(selected)),
            "fit_p95_exceedance_ratio": float(np.mean(selected > float(threshold))),
            "distance_mean": float(np.mean(selected)),
            "distance_median": float(np.median(selected)),
        }

    result = {
        "definition": "window distance exceeds the 95th percentile of train-old fit distances",
        "fit_distance_p95_threshold": float(threshold),
        "all_evaluation": summarize(np.ones(len(labels), dtype=bool)),
        "old_classes": summarize(labels < int(old_class_count)),
        "novel_classes_diagnostic_only": summarize(labels >= int(old_class_count)),
        "by_activity_1based": {},
        "by_subject": {},
    }
    for label in np.unique(labels):
        result["by_activity_1based"][str(int(label) + 1)] = summarize(labels == label)
    for subject_id in np.unique(subjects):
        result["by_subject"][str(int(subject_id))] = summarize(subjects == subject_id)
    return result


def save_trial_sequences(path: Path, records: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(jsonable(record), ensure_ascii=False) + "\n")


def save_activity_matrix_csv(
    path: Path, matrix: np.ndarray, activity_names: list[str]
) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["activity"] + activity_names)
        for name, row in zip(activity_names, matrix):
            writer.writerow([name] + [f"{float(value):.8f}" for value in row])


def save_pairwise_csv(
    path: Path,
    records: list[dict],
    sequence_matrix: np.ndarray,
    histogram_matrix: np.ndarray,
    count_matrix: np.ndarray,
) -> None:
    labels = np.asarray([record["activity_label_0based"] for record in records])
    subjects = np.asarray([record["subject_id"] for record in records])
    left, right = cross_subject_pair_indices(labels, subjects, None)
    with path.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = [
            "left_trial_key",
            "right_trial_key",
            "left_subject",
            "right_subject",
            "left_activity",
            "right_activity",
            "same_activity",
            "rle_edit_distance",
            "histogram_js_distance",
            "window_count_distance",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for left_index, right_index in zip(left, right):
            writer.writerow(
                {
                    "left_trial_key": records[left_index]["trial_key"],
                    "right_trial_key": records[right_index]["trial_key"],
                    "left_subject": int(subjects[left_index]),
                    "right_subject": int(subjects[right_index]),
                    "left_activity": int(labels[left_index] + 1),
                    "right_activity": int(labels[right_index] + 1),
                    "same_activity": bool(labels[left_index] == labels[right_index]),
                    "rle_edit_distance": f"{float(sequence_matrix[left_index, right_index]):.8f}",
                    "histogram_js_distance": f"{float(histogram_matrix[left_index, right_index]):.8f}",
                    "window_count_distance": f"{float(count_matrix[left_index, right_index]):.8f}",
                }
            )


def save_heatmap(
    path: Path,
    matrix: np.ndarray,
    activity_names: list[str],
    primitive_segmentation: str = "fixed_window",
) -> bool:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as error:
        LOGGER.warning("Heatmap skipped because matplotlib is unavailable: %s", error)
        return False
    figure, axis = plt.subplots(figsize=(10, 8))
    image = axis.imshow(matrix, cmap="viridis", vmin=0.0, vmax=1.0)
    axis.set_xticks(range(len(activity_names)), activity_names, rotation=55, ha="right")
    axis.set_yticks(range(len(activity_names)), activity_names)
    axis.set_title(
        "Mean cross-subject RLE token-sequence distance\n"
        f"segmentation={primitive_segmentation}"
    )
    figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)
    return True


def build_trajectory_plot_arrays(
    records: list[dict],
) -> tuple[list[dict], np.ndarray, np.ndarray]:
    """Return ordered window-token rows and explicit segment-boundary mask."""

    if not records:
        raise ValueError("At least one trial record is required for a trajectory plot.")
    ordered = sorted(
        records,
        key=lambda record: (
            record["activity_label_1based"],
            record["subject_id"],
            record["trial_number"],
        ),
    )
    max_length = max(
        len(record["sequence_variants"]["full"]["raw_tokens"])
        for record in ordered
    )
    matrix = np.full((len(ordered), max_length), np.nan, dtype=np.float32)
    boundaries = np.zeros((len(ordered), max_length), dtype=bool)
    for row, record in enumerate(ordered):
        tokens = record["sequence_variants"]["full"]["raw_tokens"]
        matrix[row, : len(tokens)] = np.asarray(tokens, dtype=np.float32)
        for segment in record["primitive_segmentation"]["segments"][1:]:
            offset = int(segment["first_window_offset"])
            if not 0 < offset < len(tokens):
                raise RuntimeError(
                    f"Invalid segment boundary {offset} in {record['trial_key']}."
                )
            boundaries[row, offset] = True
    return ordered, matrix, boundaries


def save_trajectory_sequence_plot(
    path: Path,
    records: list[dict],
    primitive_num: int,
    stride_seconds: float,
    primitive_segmentation: str,
) -> bool:
    """Plot comparable per-window token trajectories and detected segment edges."""

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.colors import BoundaryNorm, ListedColormap
    except Exception as error:
        LOGGER.warning("Trajectory plot skipped because matplotlib is unavailable: %s", error)
        return False

    ordered, matrix, boundaries = build_trajectory_plot_arrays(records)
    masked = np.ma.masked_invalid(matrix)
    base = plt.get_cmap("gist_ncar", int(primitive_num))
    cmap = ListedColormap([base(index) for index in range(int(primitive_num))])
    cmap.set_bad("#e5e7eb")
    norm = BoundaryNorm(
        np.arange(-0.5, int(primitive_num) + 0.5), int(primitive_num)
    )
    figure, axis = plt.subplots(figsize=(18, 18))
    image = axis.imshow(
        masked, aspect="auto", interpolation="nearest", cmap=cmap, norm=norm
    )
    groups = []
    labels = sorted(set(record["activity_label_1based"] for record in ordered))
    for label in labels:
        indices = [
            index
            for index, record in enumerate(ordered)
            if record["activity_label_1based"] == label
        ]
        groups.append((ordered[indices[0]]["activity_name"], min(indices), max(indices)))
    axis.set_yticks(
        [(start + end) / 2 for _, start, end in groups],
        [name for name, _, _ in groups],
    )
    for _, _, end in groups[:-1]:
        axis.axhline(end + 0.5, color="black", linewidth=1.0)
    if primitive_segmentation in {
        "ssl_feature_changepoint",
        "motion_encoder_changepoint",
    }:
        rows, columns = np.nonzero(boundaries)
        if len(rows):
            axis.scatter(
                columns - 0.5,
                rows,
                marker="|",
                s=12,
                linewidths=0.45,
                color="black",
                alpha=0.8,
                label="detected segment boundary",
            )
            axis.legend(loc="upper right", fontsize=8)
    axis.set_xlabel(
        f"Original window index within trial (stride {float(stride_seconds):.2f} s)"
    )
    axis.set_ylabel("Held-out activity trials")
    axis.set_title(
        "Per-window primitive trajectories | "
        f"segmentation={primitive_segmentation} | K={int(primitive_num)}"
    )
    colorbar = figure.colorbar(
        image,
        ax=axis,
        ticks=np.arange(int(primitive_num)),
        fraction=0.025,
        pad=0.02,
    )
    colorbar.ax.set_yticklabels(
        [f"P{index:02d}" for index in range(int(primitive_num))]
    )
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)
    return True


def derive_feasibility_decision(
    primitive_stats: dict,
    association_metrics: dict,
    order_control: dict,
) -> dict:
    full = association_metrics["rle_sequence_full"]["all_classes"]
    nonoverlap = association_metrics["rle_sequence_nonoverlap"]["all_classes"]
    observed = full["observed"]
    nonoverlap_observed = nonoverlap["observed"]
    order = order_control["all_classes"]
    noncollapse = not any(primitive_stats["eval_usage"]["collapse_flags"].values())
    class_association = (
        observed["mean_margin_different_minus_same"] > 0
        and full.get("margin_p_value_greater", 1.0) <= 0.05
    )
    overlap_robust = nonoverlap_observed["mean_margin_different_minus_same"] > 0
    order_specific = (
        order.get("observed_minus_shuffled_margin", 0.0) > 0
        and order.get("margin_p_value_observed_greater", 1.0) <= 0.05
    )
    old_oov_ratio = primitive_stats["oov_statistics"]["old_classes"][
        "fit_p95_exceedance_ratio"
    ]
    # A matched fit distribution would exceed its own p95 about 5% of the time.
    # Twenty percent is a deliberately permissive diagnostic warning boundary,
    # not a tuned decision threshold.
    old_fit_distance_mismatch_warning = bool(
        old_oov_ratio is not None and old_oov_ratio > 0.20
    )
    association_gate = bool(noncollapse and class_association and overlap_robust)
    stable_candidate_gate = bool(
        association_gate and not old_fit_distance_mismatch_warning
    )
    if stable_candidate_gate and order_specific:
        status = "provisional_support_for_stable_ordered_primitive_candidates"
    elif association_gate:
        status = "partial_support_sequence_association_but_codebook_shift_or_order_limit"
    else:
        status = "not_supported_in_this_run"
    return {
        "codebook_noncollapse": bool(noncollapse),
        "held_out_subject_same_class_association": bool(class_association),
        "association_survives_nonoverlap_control": bool(overlap_robust),
        "evidence_for_order_beyond_token_composition": bool(order_specific),
        "old_class_fit_p95_exceedance_ratio": old_oov_ratio,
        "old_fit_distance_mismatch_warning_above_0_20": (
            old_fit_distance_mismatch_warning
        ),
        "old_subject_shift_warning_above_0_20": (
            old_fit_distance_mismatch_warning
        ),
        "sequence_association_gate_passed": association_gate,
        "stable_motion_primitive_candidate_gate_passed": stable_candidate_gate,
        "status": status,
        "interpretation_limits": [
            "KMeans always produces discrete clusters; non-collapse and held-out-subject association are the meaningful checks.",
            "The tokens are latent motion-state candidates, not validated semantic motion primitives.",
            "A positive closed-set sequence association result does not establish CGCD benefit.",
            "Observed spans omit the final incomplete raw-trial tail and are not exact activity durations.",
            "Window count is a known activity shortcut; compare the window_count_only control before attributing association to primitives.",
            "The fit-distance p95 exceedance mixes in-sample clustering optimism, weighting, and distribution shift; it is not a pure subject-shift estimate.",
            "The 0.20 old-class warning boundary is a permissive diagnostic heuristic, not a validation-tuned threshold.",
            "The valid-RLE order shuffle is a heuristic Monte Carlo control, not a uniform conditional permutation test.",
        ],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Frozen-encoder motion-primitive sequence feasibility experiment."
    )
    parser.add_argument("--checkpoint", required=True, help="Window/trial checkpoint containing ResNet1D weights.")
    parser.add_argument("--npz-path", default="", help="Override checkpoint metadata USC-HAD NPZ path.")
    parser.add_argument("--output-dir", default="", help="New, empty output directory; default is timestamped under results/.")
    parser.add_argument("--primitive-num", type=int, default=32, help="KMeans codebook size. Pre-register 32; report 16/64 as sensitivity runs.")
    parser.add_argument(
        "--primitive-segmentation",
        choices=[
            "fixed_window",
            "ssl_feature_changepoint",
            "motion_encoder_changepoint",
        ],
        default="fixed_window",
        help=(
            "fixed_window reproduces the original KMeans-window baseline; "
            "ssl_feature_changepoint learns a train-only masked-denoising feature "
            "adapter; motion_encoder_changepoint consumes the separately trained "
            "segmentation head while its content head supplies KMeans features."
        ),
    )
    parser.add_argument("--pca-dim", type=int, default=64, help="Train-only weighted PCA dimension; 0 disables PCA.")
    parser.add_argument("--embedding-normalization", choices=["l2", "none"], default="l2")
    parser.add_argument("--codebook-weighting", choices=["per_trial", "per_window"], default="per_trial")
    parser.add_argument("--kmeans-n-init", type=int, default=20)
    parser.add_argument("--kmeans-max-iter", type=int, default=300)
    parser.add_argument("--old-class-count", type=int, default=6)
    parser.add_argument("--fit-subjects", default="", help="CSV override; default checkpoint train subjects.")
    parser.add_argument("--eval-subjects", default="", help="CSV override; default checkpoint outer-test subjects.")
    parser.add_argument(
        "--allow-split-override",
        action="store_true",
        help="Explicitly authorize subjects that differ from checkpoint metadata.",
    )
    parser.add_argument(
        "--allow-unverified-npz-normalization",
        action="store_true",
        help="Explicitly authorize stored NPZ statistics whose fit subjects cannot be verified.",
    )
    parser.add_argument("--anomaly-policy", choices=["report", "exclude"], default="report")
    parser.add_argument("--sample-rate-hz", type=float, default=100.0)
    parser.add_argument("--edge-trim-ratio", type=float, default=0.10)
    parser.add_argument("--label-permutations", type=int, default=1000)
    parser.add_argument("--order-shuffles", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--ssl-feature-dim", type=int, default=64)
    parser.add_argument("--ssl-epochs", type=int, default=25)
    parser.add_argument("--ssl-learning-rate", type=float, default=1e-3)
    parser.add_argument("--ssl-mask-ratio", type=float, default=0.15)
    parser.add_argument("--ssl-noise-std", type=float, default=0.02)
    parser.add_argument("--changepoint-context-windows", type=int, default=2)
    parser.add_argument("--changepoint-score-quantile", type=float, default=0.90)
    parser.add_argument("--changepoint-min-segment-windows", type=int, default=2)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:N")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.primitive_num < 2:
        raise ValueError("--primitive-num must be at least 2.")
    if args.pca_dim < 0:
        raise ValueError("--pca-dim must be non-negative.")
    if args.old_class_count < 1 or args.old_class_count >= 12:
        raise ValueError("--old-class-count must be between 1 and 11.")
    if not 0.0 <= args.edge_trim_ratio < 0.5:
        raise ValueError("--edge-trim-ratio must be in [0,0.5).")
    if args.sample_rate_hz <= 0 or args.batch_size < 1:
        raise ValueError("Sample rate and batch size must be positive.")
    if args.label_permutations < 0 or args.order_shuffles < 0:
        raise ValueError("Permutation/shuffle counts must be non-negative.")
    if args.ssl_feature_dim < 1 or args.ssl_epochs < 1:
        raise ValueError("SSL feature dimension and epochs must be positive.")
    if args.ssl_learning_rate <= 0:
        raise ValueError("--ssl-learning-rate must be positive.")
    if not 0.0 <= args.ssl_mask_ratio < 1.0 or args.ssl_noise_std < 0.0:
        raise ValueError("SSL mask ratio must be in [0,1) and noise std non-negative.")
    if args.changepoint_context_windows < 1:
        raise ValueError("--changepoint-context-windows must be positive.")
    if not 0.0 <= args.changepoint_score_quantile <= 1.0:
        raise ValueError("--changepoint-score-quantile must be in [0,1].")
    if args.changepoint_min_segment_windows < 1:
        raise ValueError("--changepoint-min-segment-windows must be positive.")
    if args.primitive_segmentation == "ssl_feature_changepoint" and args.batch_size < 2:
        raise ValueError("SSL feature training requires --batch-size at least 2.")


def main() -> None:
    args = parse_args()
    validate_args(args)
    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    if not checkpoint_path.exists():
        raise FileNotFoundError(checkpoint_path)
    checkpoint = load_torch_checkpoint(checkpoint_path)
    metadata = checkpoint.get("experiment_metadata")
    if not isinstance(metadata, dict):
        raise RuntimeError("Checkpoint lacks experiment_metadata; split safety cannot be verified.")
    checkpoint_type = identify_checkpoint_type(checkpoint, metadata)
    if (
        args.primitive_segmentation == "motion_encoder_changepoint"
        and checkpoint_type != "motion_primitive_encoder"
    ):
        raise RuntimeError(
            "motion_encoder_changepoint requires a schema-v1 "
            "motion_primitive_encoder checkpoint."
        )
    if (
        args.primitive_segmentation == "ssl_feature_changepoint"
        and checkpoint_type == "motion_primitive_encoder"
    ):
        raise RuntimeError(
            "ssl_feature_changepoint is the legacy masked-adapter protocol and "
            "cannot be combined with a motion_primitive_encoder checkpoint. Use "
            "motion_encoder_changepoint, or fixed_window for its matched control."
        )
    output_dir = configure_output_directory(args, metadata)
    configure_logging(output_dir)
    npz_path = resolve_npz_path(checkpoint_path, metadata, args.npz_path)
    if not npz_path.exists():
        raise FileNotFoundError(npz_path)
    checkpoint_sha256 = sha256_file(checkpoint_path)
    npz_sha256 = sha256_file(npz_path)
    if checkpoint_type == "motion_primitive_encoder":
        recorded_npz_sha256 = str(checkpoint.get("npz_sha256", "")).strip()
        if not recorded_npz_sha256:
            raise RuntimeError("Motion checkpoint lacks its source NPZ SHA256.")
        if recorded_npz_sha256 != npz_sha256:
            raise RuntimeError(
                "Motion checkpoint/source dataset fingerprint mismatch: "
                f"{recorded_npz_sha256} != {npz_sha256}."
            )

    metadata_fit_subjects = [
        int(value) for value in metadata.get("uschad_train_subjects", [])
    ]
    metadata_eval_subjects = [
        int(value) for value in metadata.get("uschad_test_subjects", [])
    ]
    fit_subjects = parse_int_list(args.fit_subjects) or metadata_fit_subjects
    eval_subjects = parse_int_list(args.eval_subjects) or metadata_eval_subjects
    if not fit_subjects or not eval_subjects:
        raise RuntimeError("Fit/eval subjects must be provided or stored in checkpoint metadata.")
    if checkpoint_type == "motion_primitive_encoder" and int(
        metadata.get("old_class_count", -1)
    ) != int(args.old_class_count):
        raise RuntimeError(
            "Motion checkpoint old_class_count does not match this run: "
            f"{metadata.get('old_class_count')!r} != {args.old_class_count}."
        )
    split_differs = (
        sorted(fit_subjects) != sorted(metadata_fit_subjects)
        or sorted(eval_subjects) != sorted(metadata_eval_subjects)
    )
    if split_differs and not args.allow_split_override:
        raise RuntimeError(
            "Fit/eval subject overrides differ from checkpoint metadata. Use "
            "--allow-split-override only for an explicitly unsafe/custom analysis."
        )
    if (
        not bool(metadata.get("uschad_recompute_norm_from_train_subjects", False))
        and not args.allow_unverified_npz_normalization
    ):
        raise RuntimeError(
            "Checkpoint would use stored NPZ normalization whose fitting subjects "
            "cannot be verified for this outer split. Explicitly pass "
            "--allow-unverified-npz-normalization to run a marked custom analysis."
        )

    LOGGER.info("[Primitive Experiment] scope=decomposition_and_sequence_association_only")
    LOGGER.info("Checkpoint: %s", checkpoint_path)
    LOGGER.info("Checkpoint SHA256: %s", checkpoint_sha256)
    LOGGER.info("NPZ: %s", npz_path)
    LOGGER.info("NPZ SHA256: %s", npz_sha256)
    LOGGER.info(
        "Checkpoint type: %s | encoder frozen: true (eval mode and requires_grad=false)",
        checkpoint_type,
    )
    LOGGER.info("Fit subjects: %s | Eval subjects: %s", fit_subjects, eval_subjects)

    arrays = load_npz_arrays(npz_path)
    window_size = int(arrays["windows"].shape[2])
    expected_window_size = int(metadata.get("uschad_window_size", window_size))
    if window_size != expected_window_size:
        raise RuntimeError(
            f"Checkpoint window size {expected_window_size} != NPZ {window_size}."
        )
    stride, stride_counts = infer_stride(arrays)
    fit_mask, eval_mask, split_details = build_split_masks(
        arrays,
        fit_subjects,
        eval_subjects,
        args.old_class_count,
        args.anomaly_policy,
    )
    split_details["checkpoint_split_override_used"] = bool(split_differs)
    split_details["checkpoint_split_override_explicitly_allowed"] = bool(
        args.allow_split_override
    )
    split_details["unverified_npz_normalization_explicitly_allowed"] = bool(
        args.allow_unverified_npz_normalization
    )
    selected_mask = fit_mask | eval_mask
    selected_indices = np.flatnonzero(selected_mask)
    normalized_windows, normalization_details = normalize_for_checkpoint(
        arrays,
        fit_mask,
        selected_indices,
        metadata,
        eps=float(metadata.get("uschad_norm_eps", 1e-6)),
    )
    fit_in_selected = fit_mask[selected_indices]
    eval_in_selected = eval_mask[selected_indices]

    device = choose_device(args.device)
    if checkpoint_type == "motion_primitive_encoder":
        encoder = build_frozen_motion_encoder(checkpoint, metadata)
        encoder_feature_dim = int(encoder.content_dim)
    else:
        encoder = build_frozen_encoder(checkpoint, metadata)
        encoder_feature_dim = int(encoder.feat_dim)
    LOGGER.info(
        "Window input shape: [N,%d,%d] | feature dim: %d | device: %s",
        arrays["windows"].shape[1],
        window_size,
        encoder_feature_dim,
        device,
    )
    LOGGER.info(
        "Encoding %d selected windows (%d fit, %d held-out evaluation)",
        len(selected_indices),
        int(np.sum(fit_in_selected)),
        int(np.sum(eval_in_selected)),
    )
    if checkpoint_type == "motion_primitive_encoder":
        motion_embeddings = encode_motion_windows(
            encoder, normalized_windows, device, args.batch_size
        )
        encoder_embeddings = motion_embeddings["content"]
        backbone_embeddings = motion_embeddings["backbone"]
        trained_boundary_embeddings = motion_embeddings["segmentation"]
    else:
        encoder_embeddings = encode_windows(
            encoder, normalized_windows, device, args.batch_size
        )
        backbone_embeddings = encoder_embeddings
        trained_boundary_embeddings = None
    fit_embeddings = encoder_embeddings[fit_in_selected]
    eval_embeddings = encoder_embeddings[eval_in_selected]
    fit_global_indices = selected_indices[fit_in_selected]
    eval_global_indices = selected_indices[eval_in_selected]

    if args.codebook_weighting == "per_trial":
        fit_window_weights = inverse_trial_frequency_weights(
            arrays["trial_global_ids"][fit_global_indices]
        )
    else:
        fit_window_weights = np.ones(len(fit_embeddings), dtype=np.float64)
    primitive_fit_windows, primitive_eval_windows, pca = prepare_primitive_features(
        fit_embeddings,
        eval_embeddings,
        fit_window_weights,
        args.pca_dim,
        args.embedding_normalization,
    )

    fit_trial_ids = arrays["trial_global_ids"][fit_global_indices]
    eval_trial_ids = arrays["trial_global_ids"][eval_global_indices]
    fit_window_starts = arrays["window_start_indices"][fit_global_indices]
    eval_window_starts = arrays["window_start_indices"][eval_global_indices]
    ssl_bundle = None
    changepoint_calibration = None
    if args.primitive_segmentation == "ssl_feature_changepoint":
        LOGGER.info(
            "Training label-free masked-denoising feature adapter: dim=%d | epochs=%d",
            args.ssl_feature_dim,
            args.ssl_epochs,
        )
        ssl_bundle = train_masked_denoising_adapter(
            fit_embeddings=fit_embeddings,
            sample_weights=fit_window_weights,
            output_dim=args.ssl_feature_dim,
            epochs=args.ssl_epochs,
            batch_size=args.batch_size,
            learning_rate=args.ssl_learning_rate,
            mask_ratio=args.ssl_mask_ratio,
            noise_std=args.ssl_noise_std,
            device=device,
            seed=args.seed + 31013,
        )
        fit_boundary_features = transform_with_adapter(
            ssl_bundle, fit_embeddings, device, args.batch_size
        )
        eval_boundary_features = transform_with_adapter(
            ssl_bundle, eval_embeddings, device, args.batch_size
        )
        changepoint_threshold, changepoint_calibration = (
            calibrate_changepoint_threshold(
                fit_boundary_features,
                fit_trial_ids,
                fit_window_starts,
                args.changepoint_context_windows,
                args.changepoint_score_quantile,
            )
        )
        LOGGER.info(
            "Train-only change-point threshold: %.6f | quantile=%.3f | context=%d",
            changepoint_threshold,
            args.changepoint_score_quantile,
            args.changepoint_context_windows,
        )
        boundary_feature_source = "legacy_train_only_masked_denoising_adapter"
    elif args.primitive_segmentation == "motion_encoder_changepoint":
        if trained_boundary_embeddings is None:
            raise RuntimeError(
                "Internal error: the trained segmentation head was not encoded."
            )
        fit_boundary_features = trained_boundary_embeddings[fit_in_selected]
        eval_boundary_features = trained_boundary_embeddings[eval_in_selected]
        changepoint_threshold, changepoint_calibration = (
            calibrate_changepoint_threshold(
                fit_boundary_features,
                fit_trial_ids,
                fit_window_starts,
                args.changepoint_context_windows,
                args.changepoint_score_quantile,
            )
        )
        boundary_feature_source = "motion_encoder_segmentation_head"
        LOGGER.info(
            "Train-only motion-head change-point threshold: %.6f | "
            "quantile=%.3f | context=%d",
            changepoint_threshold,
            args.changepoint_score_quantile,
            args.changepoint_context_windows,
        )
    else:
        fit_boundary_features = primitive_fit_windows
        eval_boundary_features = primitive_eval_windows
        changepoint_threshold = None
        boundary_feature_source = "unused_fixed_window"

    fit_segmentation = build_segmented_features(
        primitive_fit_windows,
        fit_boundary_features,
        fit_trial_ids,
        fit_window_starts,
        window_size,
        args.primitive_segmentation,
        args.changepoint_context_windows,
        changepoint_threshold,
        args.changepoint_min_segment_windows,
        args.embedding_normalization == "l2",
    )
    eval_segmentation = build_segmented_features(
        primitive_eval_windows,
        eval_boundary_features,
        eval_trial_ids,
        eval_window_starts,
        window_size,
        args.primitive_segmentation,
        args.changepoint_context_windows,
        changepoint_threshold,
        args.changepoint_min_segment_windows,
        args.embedding_normalization == "l2",
    )
    fit_segment_weights = codebook_segment_weights(
        fit_segmentation, args.codebook_weighting
    )
    if args.primitive_num > len(fit_segmentation.segment_features):
        raise ValueError(
            "Codebook size exceeds the number of fit segments: "
            f"K={args.primitive_num}, segments={len(fit_segmentation.segment_features)}."
        )
    LOGGER.info(
        "Fitting KMeans: K=%d | fit_segments=%d | primitive_dim=%d | "
        "segmentation=%s | weighting=%s | n_init=%d",
        args.primitive_num,
        len(fit_segmentation.segment_features),
        fit_segmentation.segment_features.shape[1],
        args.primitive_segmentation,
        args.codebook_weighting,
        args.kmeans_n_init,
    )
    kmeans = KMeans(
        n_clusters=args.primitive_num,
        random_state=args.seed,
        n_init=args.kmeans_n_init,
        max_iter=args.kmeans_max_iter,
        algorithm="lloyd",
    )
    kmeans.fit(
        fit_segmentation.segment_features, sample_weight=fit_segment_weights
    )
    assignment_metric = "cosine" if args.embedding_normalization == "l2" else "euclidean"
    fit_segment_tokens, fit_segment_distances, stored_centers = assign_to_codebook(
        fit_segmentation.segment_features, kmeans.cluster_centers_, assignment_metric
    )
    eval_segment_tokens, eval_segment_distances, _ = assign_to_codebook(
        eval_segmentation.segment_features, stored_centers, assignment_metric
    )
    # Expand segment assignments back to the original window grid.  This keeps
    # RLE, occupancy, non-overlap, edge-trim, and window-count controls directly
    # comparable with the fixed-window KMeans32 baseline.
    fit_tokens = fit_segment_tokens[fit_segmentation.window_segment_ids]
    fit_distances = fit_segment_distances[fit_segmentation.window_segment_ids]
    eval_tokens = eval_segment_tokens[eval_segmentation.window_segment_ids]
    eval_distances = eval_segment_distances[eval_segmentation.window_segment_ids]
    primitive_fit = fit_segmentation.segment_features[
        fit_segmentation.window_segment_ids
    ]
    primitive_eval = eval_segmentation.segment_features[
        eval_segmentation.window_segment_ids
    ]

    records = build_trial_records(
        arrays,
        eval_global_indices,
        eval_tokens,
        eval_distances,
        eval_segmentation,
        args.primitive_segmentation,
        args.primitive_num,
        window_size,
        args.sample_rate_hz,
        args.edge_trim_ratio,
    )
    labels = np.asarray([record["activity_label_0based"] for record in records], dtype=np.int64)
    subjects = np.asarray([record["subject_id"] for record in records], dtype=np.int64)
    if set(subjects.tolist()) != set(eval_subjects):
        raise RuntimeError("Trial records do not cover exactly the held-out evaluation subjects.")

    matrices = {}
    for variant_name in ["full", "nonoverlap_every_second_window", "edge_trimmed"]:
        sequences = [
            record["sequence_variants"][variant_name]["rle_tokens"]
            for record in records
        ]
        matrices[f"rle_sequence_{'nonoverlap' if variant_name.startswith('nonoverlap') else variant_name}"] = (
            distance_matrix_from_sequences(sequences)
        )
    full_histograms = np.asarray(
        [
            record["sequence_variants"]["full"]["primitive_histogram"]
            for record in records
        ],
        dtype=np.float32,
    )
    matrices["primitive_histogram_full"] = distance_matrix_from_histograms(
        full_histograms
    )
    window_counts = np.asarray(
        [record["sequence_variants"]["full"]["window_count"] for record in records],
        dtype=np.int64,
    )
    matrices["window_count_only"] = length_distance_matrix(window_counts)

    association_metrics = {}
    for offset, (name, matrix) in enumerate(matrices.items()):
        LOGGER.info("Analyzing held-out cross-subject association: %s", name)
        association_metrics[name] = analyze_distance_matrix(
            matrix,
            labels,
            subjects,
            args.old_class_count,
            args.label_permutations,
            args.seed + offset * 10007,
        )
    full_sequences = [
        record["sequence_variants"]["full"]["rle_tokens"] for record in records
    ]
    order_control = order_shuffle_control(
        full_sequences,
        matrices["rle_sequence_full"],
        labels,
        subjects,
        args.old_class_count,
        args.order_shuffles,
        args.seed + 70001,
    )

    fit_usage = usage_summary(fit_tokens, args.primitive_num)
    eval_usage = usage_summary(eval_tokens, args.primitive_num)
    fit_segment_usage = usage_summary(fit_segment_tokens, args.primitive_num)
    eval_segment_usage = usage_summary(eval_segment_tokens, args.primitive_num)
    segmentation_stats = {
        "method": args.primitive_segmentation,
        "boundary_feature_source": boundary_feature_source,
        "checkpoint_type": checkpoint_type,
        "fit": segmentation_summary(fit_segmentation),
        "evaluation": segmentation_summary(eval_segmentation),
        "boundary_to_codebook_alignment": {
            "fit": boundary_to_codebook_alignment(
                fit_segmentation, fit_segment_tokens
            ),
            "evaluation": boundary_to_codebook_alignment(
                eval_segmentation, eval_segment_tokens
            ),
        },
        "changepoint_calibration": changepoint_calibration,
        "self_supervised_adapter": (
            ssl_bundle.training if ssl_bundle is not None else None
        ),
        "token_expansion": (
            "Each variable segment receives one KMeans token; that token and its "
            "center distance are expanded to all original windows in the segment "
            "for legacy sequence and occupancy controls."
        ),
        "boundary_resolution": (
            "Boundaries lie between existing 2.56-second source windows; with the "
            f"inferred stride their grid resolution is {stride / args.sample_rate_hz:.2f} seconds."
        ),
    }
    oov_threshold = float(np.percentile(fit_distances, 95.0))
    eval_window_labels = arrays["labels"][eval_global_indices]
    eval_window_subjects = arrays["subject_ids"][eval_global_indices]
    oov_statistics = grouped_oov_statistics(
        eval_distances,
        eval_window_labels,
        eval_window_subjects,
        oov_threshold,
        args.old_class_count,
    )
    unique_primitive_counts = np.asarray(
        [
            record["sequence_variants"]["full"]["unique_primitive_count"]
            for record in records
        ],
        dtype=np.int64,
    )
    run_counts = np.asarray(
        [record["sequence_variants"]["full"]["run_count"] for record in records],
        dtype=np.int64,
    )
    primitive_stats = {
        "fit_usage": fit_usage,
        "eval_usage": eval_usage,
        "fit_segment_usage": fit_segment_usage,
        "eval_segment_usage": eval_segment_usage,
        "fit_nearest_center_distance": {
            "mean": float(np.mean(fit_distances)),
            "median": float(np.median(fit_distances)),
            "p95": oov_threshold,
        },
        "eval_nearest_center_distance": {
            "mean": float(np.mean(eval_distances)),
            "median": float(np.median(eval_distances)),
            "p95": float(np.percentile(eval_distances, 95.0)),
            "fit_p95_exceedance_ratio": float(np.mean(eval_distances > oov_threshold)),
        },
        "oov_statistics": oov_statistics,
        "held_out_primitive_coverage": primitive_coverage_statistics(
            records, args.primitive_num
        ),
        "per_trial_decomposition": {
            "trial_count": int(len(records)),
            "unique_primitives_mean": float(np.mean(unique_primitive_counts)),
            "unique_primitives_median": float(np.median(unique_primitive_counts)),
            "unique_primitives_min": int(np.min(unique_primitive_counts)),
            "unique_primitives_max": int(np.max(unique_primitive_counts)),
            "multiple_primitive_trial_ratio": float(
                np.mean(unique_primitive_counts > 1)
            ),
            "run_count_mean": float(np.mean(run_counts)),
            "run_count_median": float(np.median(run_counts)),
            "run_count_min": int(np.min(run_counts)),
            "run_count_max": int(np.max(run_counts)),
            "multiple_run_trial_ratio": float(np.mean(run_counts > 1)),
            "window_count_min": int(window_counts.min()),
            "window_count_median": float(np.median(window_counts)),
            "window_count_max": int(window_counts.max()),
        },
    }
    feasibility = derive_feasibility_decision(
        primitive_stats, association_metrics, order_control
    )

    activity_names = [str(value) for value in arrays["activity_names"].tolist()]
    activity_matrix = activity_distance_matrix(
        matrices["rle_sequence_full"],
        labels,
        subjects,
        list(range(len(activity_names))),
    )
    split_details.update(
        {
            "fit_window_count": int(len(fit_global_indices)),
            "fit_segment_count": int(len(fit_segmentation.segment_features)),
            "fit_trial_count": int(len(np.unique(arrays["trial_global_ids"][fit_global_indices]))),
            "evaluation_window_count": int(len(eval_global_indices)),
            "evaluation_segment_count": int(
                len(eval_segmentation.segment_features)
            ),
            "evaluation_trial_count": int(len(records)),
            "fit_eval_subject_overlap": [],
            "fit_eval_window_overlap": int(
                len(set(fit_global_indices.tolist()) & set(eval_global_indices.tolist()))
            ),
            "fit_eval_trial_overlap": int(
                len(
                    set(arrays["trial_global_ids"][fit_global_indices].tolist())
                    & set(arrays["trial_global_ids"][eval_global_indices].tolist())
                )
            ),
        }
    )

    pca_payload = {}
    if pca is not None:
        pca_payload = {
            "pca_mean": pca.mean,
            "pca_components": pca.components,
            "pca_explained_variance": pca.explained_variance,
            "pca_explained_variance_ratio": pca.explained_variance_ratio,
        }
    np.savez_compressed(
        output_dir / "primitive_codebook.npz",
        centers=stored_centers,
        assignment_metric=np.asarray(assignment_metric),
        primitive_segmentation=np.asarray(args.primitive_segmentation),
        checkpoint_type=np.asarray(checkpoint_type),
        codebook_feature_source=np.asarray(
            "motion_encoder_content_head"
            if checkpoint_type == "motion_primitive_encoder"
            else "legacy_frozen_encoder"
        ),
        boundary_feature_source=np.asarray(boundary_feature_source),
        kmeans_inertia=np.asarray(float(kmeans.inertia_)),
        kmeans_n_iter=np.asarray(int(kmeans.n_iter_)),
        **pca_payload,
    )
    primitive_embeddings_ordered = np.empty(
        (len(selected_indices), primitive_fit.shape[1]), dtype=np.float32
    )
    primitive_embeddings_ordered[fit_in_selected] = primitive_fit
    primitive_embeddings_ordered[eval_in_selected] = primitive_eval
    primitive_tokens_ordered = np.empty(len(selected_indices), dtype=np.int64)
    primitive_tokens_ordered[fit_in_selected] = fit_tokens
    primitive_tokens_ordered[eval_in_selected] = eval_tokens
    primitive_distances_ordered = np.empty(len(selected_indices), dtype=np.float32)
    primitive_distances_ordered[fit_in_selected] = fit_distances
    primitive_distances_ordered[eval_in_selected] = eval_distances
    fit_segment_count = len(fit_segmentation.segment_features)
    primitive_segment_ids_ordered = np.empty(len(selected_indices), dtype=np.int64)
    primitive_segment_ids_ordered[fit_in_selected] = (
        fit_segmentation.window_segment_ids
    )
    primitive_segment_ids_ordered[eval_in_selected] = (
        eval_segmentation.window_segment_ids + fit_segment_count
    )
    boundary_embeddings_ordered = np.empty(
        (len(selected_indices), fit_boundary_features.shape[1]), dtype=np.float32
    )
    boundary_embeddings_ordered[fit_in_selected] = fit_boundary_features
    boundary_embeddings_ordered[eval_in_selected] = eval_boundary_features
    np.savez_compressed(
        output_dir / "window_embeddings_and_tokens.npz",
        checkpoint_type=np.asarray(checkpoint_type),
        global_window_indices=selected_indices,
        split_role=np.where(fit_in_selected, 0, 1).astype(np.int8),
        subject_ids=arrays["subject_ids"][selected_indices],
        activity_labels_0based=arrays["labels"][selected_indices],
        trial_numbers=arrays["trial_numbers"][selected_indices],
        trial_global_ids=arrays["trial_global_ids"][selected_indices],
        window_indices=arrays["window_indices"][selected_indices],
        window_start_indices=arrays["window_start_indices"][selected_indices],
        # Keep encoder_embeddings as a compatibility alias.  The explicit
        # arrays below prevent downstream code from confusing the codebook and
        # boundary roles of a motion checkpoint.
        encoder_embeddings=encoder_embeddings,
        backbone_embeddings=backbone_embeddings,
        content_embeddings=encoder_embeddings,
        boundary_embeddings=boundary_embeddings_ordered,
        primitive_embeddings=primitive_embeddings_ordered,
        primitive_tokens=primitive_tokens_ordered,
        primitive_segment_ids=primitive_segment_ids_ordered,
        nearest_center_distances=primitive_distances_ordered,
    )

    all_segment_features = np.concatenate(
        [fit_segmentation.segment_features, eval_segmentation.segment_features], axis=0
    )
    all_segment_tokens = np.concatenate(
        [fit_segment_tokens, eval_segment_tokens], axis=0
    )
    all_segment_distances = np.concatenate(
        [fit_segment_distances, eval_segment_distances], axis=0
    )
    all_segment_trial_ids = np.concatenate(
        [fit_segmentation.segment_trial_ids, eval_segmentation.segment_trial_ids]
    )
    all_segment_window_counts = np.concatenate(
        [
            fit_segmentation.segment_window_counts,
            eval_segmentation.segment_window_counts,
        ]
    )
    all_support_starts = np.concatenate(
        [
            fit_segmentation.segment_support_start_samples,
            eval_segmentation.segment_support_start_samples,
        ]
    )
    all_support_ends = np.concatenate(
        [
            fit_segmentation.segment_support_end_samples_exclusive,
            eval_segmentation.segment_support_end_samples_exclusive,
        ]
    )
    all_partition_starts = np.concatenate(
        [
            fit_segmentation.segment_partition_start_samples,
            eval_segmentation.segment_partition_start_samples,
        ]
    )
    all_partition_ends = np.concatenate(
        [
            fit_segmentation.segment_partition_end_samples_exclusive,
            eval_segmentation.segment_partition_end_samples_exclusive,
        ]
    )
    all_boundary_scores = np.concatenate(
        [
            fit_segmentation.segment_boundary_scores,
            eval_segmentation.segment_boundary_scores,
        ]
    )
    all_first_global_windows = np.concatenate(
        [
            fit_global_indices[fit_segmentation.segment_first_window_positions],
            eval_global_indices[eval_segmentation.segment_first_window_positions],
        ]
    )
    all_last_global_windows = np.concatenate(
        [
            fit_global_indices[fit_segmentation.segment_last_window_positions],
            eval_global_indices[eval_segmentation.segment_last_window_positions],
        ]
    )
    np.savez_compressed(
        output_dir / "segment_embeddings_and_tokens.npz",
        checkpoint_type=np.asarray(checkpoint_type),
        boundary_feature_source=np.asarray(boundary_feature_source),
        segment_ids=np.arange(len(all_segment_features), dtype=np.int64),
        split_role=np.r_[
            np.zeros(fit_segment_count, dtype=np.int8),
            np.ones(len(eval_segmentation.segment_features), dtype=np.int8),
        ],
        trial_global_ids=all_segment_trial_ids,
        window_counts=all_segment_window_counts,
        first_global_window_indices=all_first_global_windows,
        last_global_window_indices=all_last_global_windows,
        support_start_samples=all_support_starts,
        support_end_samples_exclusive=all_support_ends,
        partition_start_samples=all_partition_starts,
        partition_end_samples_exclusive=all_partition_ends,
        boundary_scores_at_start=all_boundary_scores,
        segment_embeddings=all_segment_features,
        primitive_tokens=all_segment_tokens,
        nearest_center_distances=all_segment_distances,
        primitive_segmentation=np.asarray(args.primitive_segmentation),
    )
    if ssl_bundle is not None:
        ssl_bundle.model.to("cpu")
        torch.save(
            {
                "state_dict": ssl_bundle.model.state_dict(),
                "input_mean": torch.from_numpy(ssl_bundle.input_mean),
                "input_std": torch.from_numpy(ssl_bundle.input_std),
                "training": ssl_bundle.training,
                "fit_subjects": fit_subjects,
                "old_class_count": int(args.old_class_count),
            },
            output_dir / "ssl_feature_adapter.pt",
        )
    save_trial_sequences(output_dir / "trial_primitive_sequences.jsonl", records)
    write_json(output_dir / "primitive_statistics.json", primitive_stats)
    write_json(output_dir / "segmentation_statistics.json", segmentation_stats)
    write_json(output_dir / "sequence_association_metrics.json", association_metrics)
    write_json(output_dir / "order_shuffle_control.json", order_control)
    write_json(output_dir / "split_audit.json", split_details)
    write_json(output_dir / "feasibility_decision.json", feasibility)
    save_activity_matrix_csv(
        output_dir / "activity_sequence_distance_matrix.csv",
        activity_matrix,
        activity_names,
    )
    save_pairwise_csv(
        output_dir / "cross_subject_pairwise_distances.csv",
        records,
        matrices["rle_sequence_full"],
        matrices["primitive_histogram_full"],
        matrices["window_count_only"],
    )
    heatmap_saved = save_heatmap(
        output_dir / "activity_sequence_distance_heatmap.png",
        activity_matrix,
        activity_names,
        args.primitive_segmentation,
    )
    trajectory_plot_saved = save_trajectory_sequence_plot(
        output_dir / "activity_trial_token_sequences.png",
        records,
        args.primitive_num,
        stride / args.sample_rate_hz,
        args.primitive_segmentation,
    )

    experiment_config = {
        "scope": "motion primitive decomposition and same-activity sequence association only",
        "arguments": vars(args),
        "checkpoint": str(checkpoint_path),
        "checkpoint_type": checkpoint_type,
        "checkpoint_schema_version": (
            int(checkpoint.get("schema_version", 0))
            if checkpoint_type == "motion_primitive_encoder"
            else None
        ),
        "checkpoint_sha256": checkpoint_sha256,
        "checkpoint_metadata": metadata,
        "encoder_training": (
            checkpoint.get("resolved_training_config")
            if checkpoint_type == "motion_primitive_encoder"
            else None
        ),
        "encoder_selection": (
            checkpoint.get("selection")
            if checkpoint_type == "motion_primitive_encoder"
            else None
        ),
        "encoder_training_schedule": (
            {
                key: checkpoint.get("command_arguments", {}).get(key)
                for key in [
                    "epochs",
                    "trial_batch_size",
                    "learning_rate",
                    "minimum_learning_rate",
                    "weight_decay",
                    "gradient_clip_norm",
                    "ema_momentum",
                    "freeze_backbone_epochs",
                    "early_stopping_patience",
                    "selection_policy",
                    "deterministic",
                    "smoke_max_train_trials",
                    "smoke_max_val_trials",
                ]
            }
            if checkpoint_type == "motion_primitive_encoder"
            else None
        ),
        "encoder_command_arguments": (
            checkpoint.get("command_arguments")
            if checkpoint_type == "motion_primitive_encoder"
            else None
        ),
        "encoder_implementation_fingerprint": (
            checkpoint.get("implementation_fingerprint")
            if checkpoint_type == "motion_primitive_encoder"
            else None
        ),
        "encoder_source_checkpoint": (
            checkpoint.get("source_checkpoint")
            if checkpoint_type == "motion_primitive_encoder"
            else None
        ),
        "npz_path": str(npz_path),
        "npz_sha256": npz_sha256,
        "output_dir": str(output_dir),
        "runtime": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "sklearn": sklearn.__version__,
            "numpy": np.__version__,
            "cuda_available": torch.cuda.is_available(),
            "device": str(device),
        },
        "data": {
            "window_shape": list(arrays["windows"].shape),
            "window_size_samples": window_size,
            "inferred_stride_samples": stride,
            "stride_value_counts": stride_counts,
            "sample_rate_hz": args.sample_rate_hz,
            "tail_limitation": "NPZ omits each raw trial's final incomplete window tail",
        },
        "normalization": normalization_details,
        "segmentation": segmentation_stats,
        "feature_roles": {
            "codebook": (
                "motion_encoder_content_head"
                if checkpoint_type == "motion_primitive_encoder"
                else "legacy_frozen_encoder"
            ),
            "boundary": boundary_feature_source,
            "legacy_encoder_embeddings_alias": "content_embeddings",
        },
        "codebook": {
            "primitive_num": args.primitive_num,
            "pca_dim": args.pca_dim,
            "embedding_normalization": args.embedding_normalization,
            "assignment_metric": assignment_metric,
            "weighting": args.codebook_weighting,
            "kmeans_inertia": float(kmeans.inertia_),
            "kmeans_n_iter": int(kmeans.n_iter_),
        },
        "order_shuffle_control": {
            "algorithm": "valid_rle_permutation_v2",
            "equal_adjacent_tokens_allowed": False,
        },
        "postprocessing": {
            "metric_revision": SEQUENCE_METRIC_REVISION,
            "generated_with_current_metrics": True,
            "encoder_pca_codebook_refit": False,
        },
        "heatmap_saved": heatmap_saved,
        "trajectory_plot_saved": trajectory_plot_saved,
    }
    write_json(output_dir / "experiment_config.json", experiment_config)
    write_json(
        output_dir / "sequence_metric_revision.json",
        {
            "revision": SEQUENCE_METRIC_REVISION,
            "reason": (
                "Use equal credit across exact 1-NN distance ties and prevent "
                "equal adjacent tokens in shuffled RLE controls."
            ),
            "encoder_pca_codebook_refit": False,
            "generated_with_current_metrics": True,
        },
    )
    write_json(
        output_dir / "summary.json",
        {
            "split_audit": split_details,
            "segmentation_statistics": segmentation_stats,
            "primitive_statistics": primitive_stats,
            "sequence_association_metrics": association_metrics,
            "order_shuffle_control": order_control,
            "feasibility_decision": feasibility,
        },
    )

    all_sequence = association_metrics["rle_sequence_full"]["all_classes"]
    LOGGER.info(
        "Codebook utilization: %d/%d (%.3f) | max token share: %.3f",
        eval_usage["used_primitives"],
        eval_usage["total_primitives"],
        eval_usage["utilization"],
        eval_usage["max_token_share"],
    )
    LOGGER.info(
        "Cross-subject RLE distance: same=%.4f | different=%.4f | ratio=%.4f | p=%.6f",
        all_sequence["observed"]["same_mean"],
        all_sequence["observed"]["different_mean"],
        all_sequence["observed"]["separation_ratio_different_over_same"],
        all_sequence.get("margin_p_value_greater", float("nan")),
    )
    LOGGER.info(
        "Cross-subject 1-NN activity retrieval: %.4f | order evidence: %s",
        all_sequence["observed"]["cross_subject_1nn_activity_accuracy"],
        feasibility["evidence_for_order_beyond_token_composition"],
    )
    LOGGER.info(
        "Feasibility status: %s | outputs: %s",
        feasibility["status"],
        output_dir,
    )


if __name__ == "__main__":
    main()
