"""Fail-closed integrity checks for schema-v1 motion encoder checkpoints."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping

import numpy as np
import torch


CHECKPOINT_TYPE = "motion_primitive_encoder"
SCHEMA_VERSION = 1

# Complete argparse payload produced by schema-v1 train_motion_encoder.py.
# Consumers use this to reject truncated metadata: comparing two equally
# incomplete dictionaries would otherwise make a heterogeneous experiment look
# homogeneous simply because the omitted parameter is invisible.
COMMAND_ARGUMENTS_V1 = frozenset(
    {
        "source_checkpoint", "npz_path", "output_dir", "self_test",
        "ablation_profile", "window_aug_consistency", "window_aug_profile",
        "window_aug_weight", "window_aug_temperature",
        "window_aug_one_window_per_trial", "noise_std_ratio", "acc_scale_min",
        "acc_scale_max", "gyro_scale_min", "gyro_scale_max",
        "time_shift_max_samples", "time_mask_min_samples",
        "time_mask_max_samples", "rotation_max_degrees",
        "vicreg_invariance_weight", "vicreg_variance_weight",
        "vicreg_covariance_weight", "vicreg_target_std", "cp_weight",
        "content_boundary_alignment_weight", "cp_anchor_source",
        "cp_context_windows", "cp_low_quantile", "cp_high_quantile",
        "cp_raw_scales", "cp_raw_frequency_bins", "cp_raw_epsilon",
        "cp_raw_scale_floor", "cp_raw_z_clip", "cp_rank_margin",
        "cp_equivariance_weight", "cp_equivariance_delta",
        "noncollapse_weight", "noncollapse_target_std",
        "noncollapse_variance_weight", "noncollapse_covariance_weight",
        "noncollapse_windows_per_trial", "prediction_weight",
        "prediction_mask_ratio", "prediction_loss", "prediction_huber_delta",
        "trial_weight", "cross_subject_weight", "segmentation_dim",
        "content_dim", "content_residual", "augmentation_dim",
        "projection_hidden_dim", "trial_hidden_dim", "trial_peak_quantile",
        "trial_dropout", "predictor_hidden_dim", "backbone_layers",
        "old_class_count", "epochs", "trial_batch_size",
        "source_encode_batch_size", "learning_rate", "minimum_learning_rate",
        "weight_decay", "gradient_clip_norm", "ema_momentum",
        "freeze_backbone_epochs", "backbone_bn_policy",
        "early_stopping_patience", "minimum_improvement", "selection_policy",
        "normalization_eps", "known_anomaly_policy", "smoke_max_train_trials",
        "smoke_max_val_trials", "seed", "device", "deterministic",
    }
)


def motion_state_dict_sha256(state_dict: Mapping[str, torch.Tensor]) -> str:
    """Hash state keys, dtypes, shapes and exact contiguous tensor bytes.

    This intentionally matches the producer's schema-v1 hashing algorithm.
    The key order in a Python dictionary therefore cannot affect the digest.
    """

    if not isinstance(state_dict, Mapping):
        raise TypeError("Motion encoder state_dict must be a mapping.")
    if not state_dict:
        raise RuntimeError("Motion encoder state_dict cannot be empty.")
    if any(not isinstance(key, str) for key in state_dict):
        raise TypeError("Every motion encoder state_dict key must be a string.")
    digest = hashlib.sha256()
    for key in sorted(state_dict):
        tensor = state_dict[key]
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"Motion encoder state value {key!r} is not a tensor.")
        if tensor.layout != torch.strided:
            raise TypeError(f"Motion encoder state value {key!r} must be strided/dense.")
        value = tensor.detach().cpu().contiguous()
        digest.update(key.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
        try:
            digest.update(value.numpy().tobytes())
        except (TypeError, RuntimeError) as error:
            raise TypeError(
                f"Motion encoder state value {key!r} cannot be hashed as raw NumPy bytes."
            ) from error
    return digest.hexdigest()


def _state_dicts_are_exactly_equal(
    first: Mapping[str, torch.Tensor], second: Mapping[str, torch.Tensor]
) -> bool:
    if set(first) != set(second):
        return False
    for key in first:
        left = first[key]
        right = second[key]
        if not isinstance(left, torch.Tensor) or not isinstance(right, torch.Tensor):
            return False
        if left.dtype != right.dtype or tuple(left.shape) != tuple(right.shape):
            return False
        if not torch.equal(left.detach().cpu(), right.detach().cpu()):
            return False
    return True


def validate_motion_encoder_checkpoint_integrity(
    checkpoint: Mapping,
) -> Mapping[str, torch.Tensor]:
    """Validate schema identity, duplicate state aliases, and content digest.

    Returns the canonical ``model_state_dict`` after validation.  Consumers
    must call this before constructing/loading a :class:`MotionPrimitiveEncoder`.
    Both aliases are mandatory in schema v1: accepting only one would allow a
    generic reader and a strict reader to silently consume different weights.
    """

    if not isinstance(checkpoint, Mapping):
        raise TypeError("Motion encoder checkpoint must be a mapping.")
    if checkpoint.get("checkpoint_type") != CHECKPOINT_TYPE:
        raise RuntimeError(
            f"Expected checkpoint_type={CHECKPOINT_TYPE!r}; "
            f"got {checkpoint.get('checkpoint_type')!r}."
        )
    schema = checkpoint.get("schema_version")
    if isinstance(schema, bool) or not isinstance(schema, int) or schema != SCHEMA_VERSION:
        raise RuntimeError(
            f"Expected motion encoder schema_version={SCHEMA_VERSION}; got {schema!r}."
        )
    model = checkpoint.get("model")
    canonical = checkpoint.get("model_state_dict")
    if not isinstance(model, Mapping) or not isinstance(canonical, Mapping):
        raise RuntimeError(
            "Schema-v1 motion checkpoint requires dictionary aliases "
            "'model' and 'model_state_dict'."
        )
    # Validate tensor types even when the two key sets differ, so malformed
    # state payloads fail with a useful error rather than inside torch.load_state_dict.
    motion_state_dict_sha256(model)
    observed_hash = motion_state_dict_sha256(canonical)
    if not _state_dicts_are_exactly_equal(model, canonical):
        raise RuntimeError(
            "Motion checkpoint aliases 'model' and 'model_state_dict' are not exactly equivalent."
        )
    recorded_hash = checkpoint.get("model_state_dict_sha256")
    if not isinstance(recorded_hash, str) or not recorded_hash:
        raise RuntimeError("Motion checkpoint lacks model_state_dict_sha256.")
    if recorded_hash != observed_hash:
        raise RuntimeError(
            "Motion checkpoint model_state_dict_sha256 mismatch: "
            f"recorded={recorded_hash!r}, observed={observed_hash!r}."
        )
    return canonical


__all__ = [
    "CHECKPOINT_TYPE",
    "COMMAND_ARGUMENTS_V1",
    "SCHEMA_VERSION",
    "motion_state_dict_sha256",
    "validate_motion_encoder_checkpoint_integrity",
]
