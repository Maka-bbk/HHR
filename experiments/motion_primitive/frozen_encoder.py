"""Strict loading and batched inference for the frozen A2 motion encoder.

This module is deliberately small.  It accepts only schema-v1 checkpoints
produced by :mod:`train_motion_encoder`, verifies the declared feature roles,
and never exposes a training path.  E0 consumes ``content`` while future
adaptive-boundary ablations may consume ``segmentation``.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Mapping

import numpy as np
import torch

from experiments.motion_primitive.motion_checkpoint import (
    validate_motion_encoder_checkpoint_integrity,
)
from experiments.motion_primitive.motion_encoder import MotionPrimitiveEncoder


EXPECTED_FEATURE_ROLES = {"codebook": "content", "boundary": "segmentation"}


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(int(chunk_size)), b""):
            digest.update(block)
    return digest.hexdigest()


def load_torch_checkpoint(path: Path) -> dict:
    try:
        value = torch.load(Path(path), map_location="cpu", weights_only=False)
    except TypeError:  # PyTorch < 2.1
        value = torch.load(Path(path), map_location="cpu")
    if not isinstance(value, dict):
        raise TypeError("A motion-encoder checkpoint must contain a dictionary.")
    return value


def _architecture(checkpoint: Mapping) -> dict:
    architecture = checkpoint.get("architecture")
    if not isinstance(architecture, Mapping):
        raise RuntimeError("Motion checkpoint lacks an architecture mapping.")
    required = {
        "in_channels", "backbone_dim", "base_channels", "backbone_layers",
        "backbone_dropout", "segmentation_dim", "segmentation_residual",
        "content_dim", "content_residual", "augmentation_dim",
        "projection_hidden_dim", "num_classes", "trial_hidden_dim",
        "trial_peak_quantile", "trial_dropout", "predictor_hidden_dim",
    }
    missing = required - set(architecture)
    if missing:
        raise RuntimeError(
            f"Motion checkpoint architecture is incomplete: {sorted(missing)}."
        )
    return dict(architecture)


def build_frozen_a2_encoder(
    checkpoint: Mapping,
    *,
    expected_profile: str = "A2",
) -> MotionPrimitiveEncoder:
    """Reconstruct an integrity-checked A2 encoder with all gradients disabled."""

    state = validate_motion_encoder_checkpoint_integrity(checkpoint)
    metadata = checkpoint.get("experiment_metadata")
    if not isinstance(metadata, Mapping):
        raise RuntimeError("Motion checkpoint lacks experiment_metadata.")
    if dict(metadata.get("feature_roles", {})) != EXPECTED_FEATURE_ROLES:
        raise RuntimeError(
            "Motion checkpoint feature_roles must be exactly "
            f"{EXPECTED_FEATURE_ROLES!r}."
        )
    config = checkpoint.get("resolved_training_config")
    if not isinstance(config, Mapping):
        raise RuntimeError("Motion checkpoint lacks resolved_training_config.")
    observed_profile = str(config.get("ablation_profile", "")).upper()
    if observed_profile != str(expected_profile).upper():
        raise RuntimeError(
            f"Expected encoder profile {expected_profile!r}, got {observed_profile!r}."
        )
    if observed_profile == "A2":
        expected_weights = {
            "window_augmentation": 0.0,
            "changepoint": 1.0,
            "content_boundary_alignment": 0.1,
            "noncollapse": 0.05,
            "temporal_prediction": 0.5,
            "trial_auxiliary": 0.1,
            "cross_subject": 0.0,
        }
        weights = config.get("loss_weights")
        if not isinstance(weights, Mapping):
            raise RuntimeError("A2 checkpoint lacks loss_weights.")
        mismatches = {
            key: {"expected": expected, "observed": weights.get(key)}
            for key, expected in expected_weights.items()
            if key not in weights
            or not np.isclose(float(weights[key]), expected, rtol=0.0, atol=1e-12)
        }
        if mismatches:
            raise RuntimeError(f"A2 loss identity mismatch: {mismatches}.")

    architecture = _architecture(checkpoint)
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
    model.load_state_dict(state, strict=True)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.checkpoint_architecture = architecture
    model.checkpoint_sha256 = checkpoint.get("model_state_dict_sha256")
    return model


def choose_device(value: str) -> torch.device:
    if str(value) == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false.")
    return device


def encode_motion_windows(
    encoder: MotionPrimitiveEncoder,
    windows: np.ndarray,
    *,
    device: torch.device,
    batch_size: int = 512,
) -> dict[str, np.ndarray]:
    """Encode windows without trial pooling, adaptation, or gradient creation."""

    matrix = np.asarray(windows, dtype=np.float32)
    if matrix.ndim != 3 or len(matrix) == 0:
        raise ValueError("windows must be a non-empty [N,C,T] float array.")
    if int(batch_size) < 1:
        raise ValueError("batch_size must be positive.")
    encoder = encoder.to(device).eval()
    dimensions = {
        "backbone": int(encoder.backbone_dim),
        "content": int(encoder.content_dim),
        "segmentation": int(encoder.segmentation_dim),
    }
    result = {
        name: np.empty((len(matrix), dimension), dtype=np.float32)
        for name, dimension in dimensions.items()
    }
    with torch.inference_mode():
        for begin in range(0, len(matrix), int(batch_size)):
            end = min(begin + int(batch_size), len(matrix))
            encoded = encoder.encode_windows(
                torch.from_numpy(matrix[begin:end]).to(device=device)
            )
            for name, dimension in dimensions.items():
                values = encoded[name]
                if tuple(values.shape) != (end - begin, dimension):
                    raise RuntimeError(
                        f"Unexpected {name} shape {tuple(values.shape)}."
                    )
                result[name][begin:end] = values.detach().cpu().numpy()
    if any(not np.all(np.isfinite(values)) for values in result.values()):
        raise RuntimeError("Motion encoder produced non-finite features.")
    return result


__all__ = [
    "EXPECTED_FEATURE_ROLES",
    "build_frozen_a2_encoder",
    "choose_device",
    "encode_motion_windows",
    "load_torch_checkpoint",
    "sha256_file",
]
