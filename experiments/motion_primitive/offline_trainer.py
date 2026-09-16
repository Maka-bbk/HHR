"""Legacy joint VQ/GRU USC-HAD motion-primitive offline trainer.

This implementation is retained for historical reproduction and is no longer
the public HHR entry point. It deliberately has no image-dataset dependency and no
complete-trial pooling branch.  Its only profile is a single-optimizer
motion-primitive trajectory model:

``motion_primitive_joint``
    ResNet1D -> discrete primitive runs -> trajectory readout. Every minibatch
    uses two forwards, one optimizer, and exactly one backward call.

Encoder initialization is intentionally not decided by the project: every run
must explicitly choose ``random`` or ``warmstart``.

Only validation subjects select the trajectory checkpoint.  Held-out test
subjects are evaluated once after training; their metrics never affect
checkpoint selection.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import os
import random
import sys
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, MutableMapping, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from torch.optim import SGD, lr_scheduler
from torch.utils.data import DataLoader


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.uschad_har import (  # noqa: E402
    get_uschad_datasets,
    parse_subject_ids,
    uschad_trial_collate,
)
from experiments.motion_primitive.joint_losses import (  # noqa: E402
    MaskedTemporalPredictor,
    MotionPrimitiveLossConfig,
    compose_joint_loss,
    sample_temporal_prediction_mask,
)
from experiments.motion_primitive.legacy_profiles import (  # noqa: E402
    PROFILE_JOINT,
    PROFILES,
    normalize_profile,
)
from models.motion_primitive_cgcd import (  # noqa: E402
    MotionPrimitiveCGCDModel,
    MotionPrimitiveConfig,
    forward_two_views,
)
from models.batch_utils import move_trial_batch_to_device  # noqa: E402


class ScalarAccumulator:
    def __init__(self) -> None:
        self.total = 0.0
        self.weight = 0

    def update(self, value: float, weight: int) -> None:
        self.total += float(value) * int(weight)
        self.weight += int(weight)

    @property
    def mean(self) -> float:
        return self.total / max(1, self.weight)


def _comma_separated_ints(value: str) -> list[int]:
    parsed = parse_subject_ids(value)
    if parsed is None:
        return []
    return [int(item) for item in parsed]


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.device):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(_json_safe(payload), handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, path)


def _append_jsonl(path: Path, payload: Mapping[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(_json_safe(payload), ensure_ascii=False, sort_keys=True) + "\n")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_int_rows(*columns: Iterable[int]) -> str:
    arrays = [np.asarray(list(column), dtype=np.int64).reshape(-1) for column in columns]
    if len(arrays) == 0:
        return hashlib.sha256(b"").hexdigest()
    if any(len(array) != len(arrays[0]) for array in arrays):
        raise ValueError("Identity-hash columns have different lengths.")
    matrix = np.stack(arrays, axis=1)
    order = np.lexsort(tuple(matrix[:, index] for index in reversed(range(matrix.shape[1]))))
    return hashlib.sha256(matrix[order].tobytes()).hexdigest()


def configure_logger(output_dir: Path) -> logging.Logger:
    logger = logging.getLogger(f"motion_primitive_offline.{output_dir.resolve()}")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    for existing in tuple(logger.handlers):
        logger.removeHandler(existing)
        existing.close()
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(formatter)
    file_handler = logging.FileHandler(output_dir / "train.log", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(stream)
    logger.addHandler(file_handler)
    return logger


def close_logger(logger: logging.Logger) -> None:
    """Flush and close per-run handlers so Windows can release ``train.log``."""

    for handler in tuple(logger.handlers):
        logger.removeHandler(handler)
        handler.flush()
        handler.close()


def seed_everything(seed: int, use_cuda: bool) -> None:
    """Apply the registered deterministic HAR seed policy."""

    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if use_cuda:
        torch.cuda.manual_seed_all(int(seed))
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def seed_dataloader_worker(worker_id: int) -> None:
    del worker_id
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    torch.manual_seed(worker_seed)


def dataloader_generator(seed: int) -> torch.Generator:
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    return generator


def resolve_device(specification: str) -> torch.device:
    requested = str(specification).strip().lower()
    if requested == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device(specification)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device {specification!r} was requested but CUDA is unavailable.")
    return device


def registered_single_stage_profile(args: argparse.Namespace) -> dict[str, Any]:
    """Audit the registered trajectory-only single-stage defaults."""

    expected = {
        "epochs": 100,
        "batch_size": 16,
        "eval_batch_size": 16,
        "num_workers": 0,
        "eval_num_workers": 0,
        "lr": 0.01,
        "encoder_lr_scale": 1.0,
        "momentum": 0.9,
        "weight_decay": 5.0e-4,
        "window_size": 256,
        "window_stride": 128,
        "in_channels": 6,
        "feature_dim": 256,
        "base_channels": 64,
        "backbone_dropout": 0.0,
        "trial_view_mode": "full_full",
        "trial_crop_ratio": 2.0 / 3.0,
        "trial_min_windows": 2,
        "har_aug_mode": "weak_strong",
        "har_weak_jitter_std": 0.0,
        "har_weak_scale_std": 0.0,
        "har_strong_jitter_std": 0.0,
        "har_strong_scale_std": 0.20,
        "har_time_mask_ratio": 0.0,
        "recompute_normalization": True,
        "selection_head": "trajectory",
        "selection_metric": "macro_f1",
        "effective_motion_weight": 1.0,
        "motion_ramp_start_epoch": 0,
        "motion_ramp_end_epoch": 0,
        "changepoint_absolute_floor": 0.01,
        "changepoint_null_mad_multiplier": 3.0,
        "utilization_weight": 0.0,
        "assignment_confidence_weight": 0.0,
        "codebook_diversity_weight": 0.0,
    }
    observed = {
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "eval_batch_size": args.eval_batch_size,
        "num_workers": args.num_workers,
        "eval_num_workers": args.eval_num_workers,
        "lr": args.lr,
        "encoder_lr_scale": args.encoder_lr_scale,
        "momentum": args.momentum,
        "weight_decay": args.weight_decay,
        "window_size": args.window_size,
        "window_stride": args.window_stride,
        "in_channels": args.in_channels,
        "feature_dim": args.feature_dim,
        "base_channels": args.base_channels,
        "backbone_dropout": args.backbone_dropout,
        "trial_view_mode": args.trial_view_mode,
        "trial_crop_ratio": args.trial_crop_ratio,
        "trial_min_windows": args.trial_min_windows,
        "har_aug_mode": args.har_aug_mode,
        "har_weak_jitter_std": args.har_weak_jitter_std,
        "har_weak_scale_std": args.har_weak_scale_std,
        "har_strong_jitter_std": args.har_strong_jitter_std,
        "har_strong_scale_std": args.har_strong_scale_std,
        "har_time_mask_ratio": args.har_time_mask_ratio,
        "recompute_normalization": args.uschad_recompute_norm_from_train_subjects,
        "selection_head": args.selection_head,
        "selection_metric": args.selection_metric,
        "effective_motion_weight": args.motion_weight,
        "motion_ramp_start_epoch": args.motion_ramp_start_epoch,
        "motion_ramp_end_epoch": args.motion_ramp_end_epoch,
        "changepoint_absolute_floor": args.changepoint_absolute_floor,
        "changepoint_null_mad_multiplier": args.changepoint_null_mad_multiplier,
        "utilization_weight": args.utilization_weight,
        "assignment_confidence_weight": args.assignment_confidence_weight,
        "codebook_diversity_weight": args.codebook_diversity_weight,
    }
    deviations = {
        key: {"expected": expected[key], "observed": observed[key]}
        for key in expected
        if observed[key] != expected[key]
    }
    return {
        "profile_id": "hhr_single_stage_resnet1d_trajectory_v1",
        "matches_registered_single_stage_profile": len(deviations) == 0,
        "deviations": deviations,
        "encoder_initialization_explicit": args.encoder_initialization in {
            "random",
            "warmstart",
        },
    }


def _model_config(args: argparse.Namespace) -> MotionPrimitiveConfig:
    return MotionPrimitiveConfig(
        in_channels=args.in_channels,
        window_size=args.window_size,
        window_stride=args.window_stride,
        tail_policy="drop",
        feature_dim=args.feature_dim,
        base_channels=args.base_channels,
        backbone_dropout=args.backbone_dropout,
        old_class_count=len(args.old_classes_parsed),
        codebook_size=args.codebook_size,
        codebook_temperature=args.codebook_temperature,
        trajectory_input_dim=args.trajectory_input_dim,
        trajectory_hidden_dim=args.trajectory_hidden_dim,
        trajectory_layers=args.trajectory_layers,
        trajectory_dropout=args.trajectory_dropout,
        run_state_dim=args.run_state_dim,
        boundary_threshold=args.boundary_threshold,
        boundary_initial_bias=args.boundary_initial_bias,
    ).validated()


def build_model(args: argparse.Namespace) -> MotionPrimitiveCGCDModel:
    if normalize_profile(args.profile) != PROFILE_JOINT:
        raise ValueError(f"Unknown profile {args.profile!r}.")
    return MotionPrimitiveCGCDModel(_model_config(args))


def _canonical_subject_ids(value: str) -> list[int]:
    return sorted(set(_comma_separated_ints(value)))


def load_trial_window_encoder(
    model: MotionPrimitiveCGCDModel,
    checkpoint_path: str,
    args: argparse.Namespace,
    logger: logging.Logger,
) -> None:
    """Strictly load only a ResNet1D window encoder for a warm-start run."""

    source_path = Path(checkpoint_path).resolve()
    checkpoint = torch.load(source_path, map_location="cpu")
    if not isinstance(checkpoint, Mapping):
        raise RuntimeError(
            f"Window encoder checkpoint is not a mapping: {source_path}."
        )
    split_audit = checkpoint.get("split_audit")
    checkpoint_npz_sha256 = checkpoint.get("npz_sha256")
    if not isinstance(split_audit, Mapping) or not isinstance(
        checkpoint_npz_sha256, str
    ) or not checkpoint_npz_sha256.strip():
        raise RuntimeError(
            "Window encoder checkpoint lacks mandatory split_audit/npz_sha256 "
            f"provenance: {source_path}."
        )
    required_provenance = (
        "old_class_ids_0based",
        "window_size_samples",
        "npz_sha256",
        "train_subjects",
        "validation_subjects",
        "outer_test_subjects_metadata_only",
    )
    missing_provenance = [
        key for key in required_provenance if key not in split_audit
    ]
    if missing_provenance:
        raise RuntimeError(
            "Window encoder checkpoint split_audit is incomplete for "
            f"{source_path}: missing {missing_provenance}."
        )
    current_npz_sha256 = _sha256_file(Path(args.uschad_npz_path).resolve())
    provenance_expected: dict[str, Any] = {
        "old_class_ids_0based": sorted(set(int(value) for value in args.old_classes_parsed)),
        "window_size_samples": int(args.uschad_window_size),
        "npz_sha256": current_npz_sha256,
        "train_subjects": _canonical_subject_ids(args.uschad_train_subjects),
        "validation_subjects": _canonical_subject_ids(args.offline_val_subjects),
        "outer_test_subjects_metadata_only": _canonical_subject_ids(
            args.uschad_test_subjects
        ),
    }
    provenance_observed: dict[str, Any] = {
        "old_class_ids_0based": sorted(
            set(int(value) for value in split_audit["old_class_ids_0based"])
        ),
        "window_size_samples": int(split_audit["window_size_samples"]),
        "npz_sha256": str(split_audit["npz_sha256"]),
        "train_subjects": sorted(
            set(int(value) for value in split_audit["train_subjects"])
        ),
        "validation_subjects": sorted(
            set(int(value) for value in split_audit["validation_subjects"])
        ),
        "outer_test_subjects_metadata_only": sorted(
            set(
                int(value)
                for value in split_audit["outer_test_subjects_metadata_only"]
            )
        ),
    }
    provenance_mismatches = [
        f"{key}={provenance_observed[key]!r} expected {expected!r}"
        for key, expected in provenance_expected.items()
        if provenance_observed[key] != expected
    ]
    if str(checkpoint_npz_sha256) != current_npz_sha256:
        provenance_mismatches.append(
            "checkpoint.npz_sha256="
            f"{checkpoint_npz_sha256!r} expected {current_npz_sha256!r}"
        )
    if provenance_mismatches:
        raise RuntimeError(
            f"Window encoder checkpoint provenance mismatch for {source_path}: "
            + "; ".join(provenance_mismatches)
        )
    metadata = checkpoint.get("experiment_metadata")
    if not isinstance(metadata, dict):
        raise RuntimeError(
            f"Window encoder checkpoint lacks experiment_metadata: {source_path}."
        )
    expected_metadata: dict[str, Any] = {
        "uschad_window_size": int(args.uschad_window_size),
        "har_in_channels": int(args.har_in_channels),
        "har_feat_dim": int(args.feature_dim),
        "har_base_channels": int(args.base_channels),
        "har_dropout": float(args.backbone_dropout),
    }
    if "seed" in metadata:
        expected_metadata["seed"] = int(args.seed)
    if "motion_encoder_seed" in metadata:
        expected_metadata["motion_encoder_seed"] = int(args.seed)
    if args.uschad_recompute_norm_from_train_subjects:
        expected_metadata.update(
            {
                "uschad_recompute_norm_from_train_subjects": True,
                "uschad_norm_eps": float(args.uschad_norm_eps),
                "uschad_train_subjects": _canonical_subject_ids(
                    args.uschad_train_subjects
                ),
                "offline_val_subjects": _canonical_subject_ids(
                    args.offline_val_subjects
                ),
                "uschad_test_subjects": _canonical_subject_ids(
                    args.uschad_test_subjects
                ),
                "uschad_cv_fold": int(args.uschad_cv_fold),
            }
        )
        # Historical A2 checkpoints predate the redundant split-mode field;
        # their explicit disjoint subject lists and fold id are authoritative.
        if "uschad_split_mode" in metadata:
            expected_metadata["uschad_split_mode"] = "subject"
    mismatches = []
    for key, expected in expected_metadata.items():
        if key not in metadata:
            mismatches.append(f"{key}=<missing> expected {expected!r}")
            continue
        observed = metadata[key]
        if isinstance(expected, float):
            equal = math.isclose(float(observed), expected, rel_tol=0.0, abs_tol=1.0e-12)
        else:
            equal = observed == expected
        if not equal:
            mismatches.append(f"{key}={observed!r} expected {expected!r}")
    if mismatches:
        raise RuntimeError(
            f"Window encoder checkpoint/data mismatch for {source_path}: "
            + "; ".join(mismatches)
        )

    source_state = checkpoint.get(
        "model_state_dict", checkpoint.get("model", checkpoint)
    )
    target_state = model.window_encoder.state_dict()
    prefixes = (
        "0.window_encoder.",
        "window_encoder.",
        "backbone.",
        "0.",
        "",
    )
    extracted: dict[str, torch.Tensor] = {}
    for target_key, target_value in target_state.items():
        match = next(
            (
                (prefix + target_key, source_state[prefix + target_key])
                for prefix in prefixes
                if prefix + target_key in source_state
            ),
            None,
        )
        if match is None:
            raise RuntimeError(
                f"Checkpoint {source_path} is missing window encoder key {target_key}."
            )
        source_key, source_value = match
        if tuple(source_value.shape) != tuple(target_value.shape):
            raise RuntimeError(
                f"Window encoder shape mismatch for {source_key}: "
                f"checkpoint={tuple(source_value.shape)}, "
                f"expected={tuple(target_value.shape)}."
            )
        extracted[target_key] = source_value
    model.window_encoder.load_state_dict(extracted, strict=True)
    logger.info(
        "Loaded strict ResNet1D window encoder from %s (%d tensors).",
        source_path,
        len(extracted),
    )


def configure_window_encoder_for_epoch(
    model: MotionPrimitiveCGCDModel, epoch_number: int, freeze_epochs: int
) -> bool:
    """Freeze parameters and BatchNorm state for the formal first five epochs."""

    frozen = int(epoch_number) <= int(freeze_epochs)
    for parameter in model.window_encoder.parameters():
        parameter.requires_grad = not frozen
    if frozen:
        model.window_encoder.eval()
    return frozen


def build_sgd_optimizer(
    model: nn.Module,
    args: argparse.Namespace,
    temporal_predictor: Optional[nn.Module] = None,
) -> SGD:
    """Build one optimizer for the deployable model and offline predictor."""

    grouped: MutableMapping[tuple[bool, bool], list[nn.Parameter]] = {}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        is_trial_encoder = name.startswith("window_encoder.")
        no_weight_decay = name.endswith(".bias") or parameter.ndim == 1
        grouped.setdefault((is_trial_encoder, no_weight_decay), []).append(parameter)
    if temporal_predictor is not None:
        for name, parameter in temporal_predictor.named_parameters():
            if not parameter.requires_grad:
                continue
            no_weight_decay = name.endswith(".bias") or parameter.ndim == 1
            grouped.setdefault((False, no_weight_decay), []).append(parameter)

    parameter_groups = []
    for (is_trial_encoder, no_weight_decay), parameters in grouped.items():
        group: dict[str, Any] = {
            "params": parameters,
            "lr": args.lr * args.encoder_lr_scale if is_trial_encoder else args.lr,
        }
        if no_weight_decay:
            group["weight_decay"] = 0.0
        parameter_groups.append(group)
    if not parameter_groups:
        raise RuntimeError("No trainable parameters were found for SGD.")
    return SGD(
        parameter_groups,
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )


def build_cosine_scheduler(
    optimizer: torch.optim.Optimizer,
    epochs: int,
    minimum_ratio: float = 1.0e-3,
) -> lr_scheduler.LambdaLR:
    """Cosine learning-rate multiplier retained from the audited HAR baseline."""

    if int(epochs) < 1:
        raise ValueError("Cosine scheduling requires at least one epoch.")
    if not 0.0 <= float(minimum_ratio) <= 1.0:
        raise ValueError("minimum_ratio must be in [0,1].")

    def multiplier(epoch: int) -> float:
        progress = min(max(float(epoch) / int(epochs), 0.0), 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return float(minimum_ratio) + (1.0 - float(minimum_ratio)) * cosine

    return lr_scheduler.LambdaLR(optimizer, lr_lambda=multiplier)


def build_joint_loss_config(args: argparse.Namespace) -> MotionPrimitiveLossConfig:
    """Build the sole complete-trajectory objective."""

    return MotionPrimitiveLossConfig(
        motion_total_weight=args.motion_weight,
        trajectory_ce_weight=args.trajectory_ce_weight,
        trajectory_supcon_weight=args.trajectory_supcon_weight,
        trajectory_view_consistency_weight=args.trajectory_view_consistency_weight,
        changepoint_weight=args.changepoint_weight,
        content_boundary_alignment_weight=args.content_boundary_alignment_weight,
        noncollapse_weight=args.noncollapse_weight,
        temporal_prediction_weight=args.temporal_prediction_weight,
        effective_minimum_duration_weight=args.effective_minimum_duration_weight,
        vq_commitment_weight=args.vq_commitment_weight,
        vq_codebook_weight=args.vq_codebook_weight,
        utilization_weight=args.utilization_weight,
        assignment_confidence_weight=args.assignment_confidence_weight,
        codebook_diversity_weight=args.codebook_diversity_weight,
        boundary_consistency_weight=args.boundary_consistency_weight,
        transition_budget_weight=args.transition_budget_weight,
        utilization_entropy_floor=args.utilization_entropy_floor,
        maximum_assignment_entropy=args.maximum_assignment_entropy,
        maximum_codebook_cosine=args.maximum_codebook_cosine,
        maximum_transition_rate=args.maximum_transition_rate,
        changepoint_stable_quantile=args.changepoint_stable_quantile,
        changepoint_change_quantile=args.changepoint_change_quantile,
        changepoint_absolute_floor=args.changepoint_absolute_floor,
        changepoint_null_mad_multiplier=args.changepoint_null_mad_multiplier,
        changepoint_rank_margin=args.changepoint_rank_margin,
        changepoint_view_consistency_weight=args.changepoint_view_consistency_weight,
        temporal_prediction_mask_ratio=args.temporal_prediction_mask_ratio,
        minimum_primitive_windows=args.minimum_primitive_windows,
        supcon_temperature=0.07,
        supcon_base_temperature=0.07,
    ).validated()


def scheduled_motion_weight(
    base_weight: float,
    *,
    epoch_number: int,
    start_epoch: int,
    end_epoch: int,
) -> float:
    """Linearly ramp the joint auxiliary without creating a training stage.

    Epoch numbers are one based.  With the registered ``start=0,end=10``
    schedule, epoch 1 uses 10% and epoch 10 onward uses the full weight.  Model
    parameters, optimizer and scheduler remain continuous throughout.
    """

    base = float(base_weight)
    epoch_number = int(epoch_number)
    start_epoch = int(start_epoch)
    end_epoch = int(end_epoch)
    if base < 0.0 or not math.isfinite(base):
        raise ValueError("base motion weight must be finite and non-negative.")
    if epoch_number < 1:
        raise ValueError("epoch_number must be one based and positive.")
    if start_epoch < 0 or end_epoch < start_epoch:
        raise ValueError("motion ramp must satisfy 0 <= start_epoch <= end_epoch.")
    if base == 0.0:
        return 0.0
    if end_epoch == start_epoch:
        return base if epoch_number > start_epoch else 0.0
    progress = (float(epoch_number) - float(start_epoch)) / float(
        end_epoch - start_epoch
    )
    return base * min(1.0, max(0.0, progress))


def training_step(
    model: MotionPrimitiveCGCDModel,
    views: Sequence[Mapping[str, torch.Tensor]],
    labels: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    args: argparse.Namespace,
    *,
    epoch_index: int,
    loss_config: MotionPrimitiveLossConfig,
    subject_ids: Optional[torch.Tensor] = None,
    temporal_predictor: Optional[MaskedTemporalPredictor] = None,
) -> dict[str, float]:
    """Perform two view forwards, one loss composition and one optimizer step."""

    outputs = forward_two_views(
        model,
        views,
        hard_codebook=False,
    )
    if loss_config.temporal_prediction_weight > 0.0:
        if temporal_predictor is None:
            raise RuntimeError(
                "A2-MP temporal prediction is enabled but no learnable predictor "
                "was supplied."
            )
        masks = [output["token_mask"].bool() for output in outputs]
        if masks[0].shape != masks[1].shape or not torch.equal(masks[0], masks[1]):
            raise RuntimeError(
                "A2-MP temporal prediction requires aligned full-trial views."
            )
        prediction_mask = sample_temporal_prediction_mask(
            masks[0], loss_config.temporal_prediction_mask_ratio
        )
        first_target = outputs[0]["primitive_features"].detach()
        outputs[0]["temporal_prediction"] = temporal_predictor(
            outputs[0]["primitive_features"], prediction_mask, masks[0]
        )
        outputs[0]["temporal_target"] = first_target
        outputs[0]["temporal_prediction_mask"] = prediction_mask
        outputs[1]["temporal_prediction"] = temporal_predictor(
            outputs[1]["primitive_features"], prediction_mask, masks[1]
        )
        outputs[1]["temporal_target"] = first_target
        outputs[1]["temporal_prediction_mask"] = prediction_mask
    visible = torch.ones_like(labels, dtype=torch.bool)
    effective_motion_weight = scheduled_motion_weight(
        loss_config.motion_total_weight,
        epoch_number=int(epoch_index) + 1,
        start_epoch=int(getattr(args, "motion_ramp_start_epoch", 0)),
        end_epoch=int(getattr(args, "motion_ramp_end_epoch", 0)),
    )
    effective_loss_config = replace(
        loss_config, motion_total_weight=effective_motion_weight
    )
    composed = compose_joint_loss(
        outputs,
        labels,
        visible,
        effective_loss_config,
        epoch=epoch_index,
        total_epochs=args.epochs,
        subject_ids=subject_ids,
    )
    with torch.no_grad():
        supervised_logits = torch.cat(
            [output["trajectory_logits"] for output in outputs],
            dim=0,
        )
        repeated_labels = torch.cat((labels, labels), dim=0)
        supervised_accuracy = float(
            (supervised_logits.argmax(dim=1) == repeated_labels)
            .float()
            .mean()
            .cpu()
            .item()
        )
    optimizer.zero_grad(set_to_none=True)
    composed.total.backward()
    if args.gradient_clip_norm > 0.0:
        parameters = list(model.parameters())
        if temporal_predictor is not None:
            parameters.extend(temporal_predictor.parameters())
        nn.utils.clip_grad_norm_(parameters, args.gradient_clip_norm)
    optimizer.step()
    # The codebook stores prototype directions explicitly.
    model.normalize_codebook_()

    metrics = {
        "total": float(composed.total.detach().cpu().item()),
        "motion_total": float(composed.motion_total.detach().cpu().item()),
        "effective_motion_weight": effective_motion_weight,
        "train_supervised_accuracy": supervised_accuracy,
    }
    metrics.update({
        name: float(value.detach().cpu().item())
        for name, value in composed.components.items()
    })
    if subject_ids is not None:
        coverage = cross_subject_positive_coverage(labels, subject_ids)
        metrics.update(
            {
                "cross_subject_positive_anchor_fraction": float(
                    coverage["eligible_anchor_fraction"]
                ),
                "cross_subject_positive_anchor_count": float(
                    coverage["eligible_anchor_count"]
                ),
                "cross_subject_positive_anchor_total": float(
                    coverage["anchor_count"]
                ),
                "cross_subject_positive_directed_pair_count": float(
                    coverage["directed_pair_count"]
                ),
                "cross_subject_positive_batch_covered": float(
                    coverage["batch_has_cross_subject_positive"]
                ),
            }
        )
    return metrics


def _classification_metrics(
    targets: np.ndarray,
    predictions: np.ndarray,
    *,
    class_count: int,
) -> dict[str, Any]:
    labels = np.arange(int(class_count), dtype=np.int64)
    precision, recall, per_class_f1, support = precision_recall_fscore_support(
        targets,
        predictions,
        labels=labels,
        zero_division=0,
    )
    return {
        "accuracy": float(accuracy_score(targets, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(targets, predictions)),
        "macro_f1": float(
            f1_score(targets, predictions, labels=labels, average="macro", zero_division=0)
        ),
        "sample_count": int(len(targets)),
        "confusion_matrix": confusion_matrix(targets, predictions, labels=labels).tolist(),
        "per_class": {
            str(class_id): {
                "precision": float(precision[class_id]),
                "recall": float(recall[class_id]),
                "f1": float(per_class_f1[class_id]),
                "support": int(support[class_id]),
            }
            for class_id in range(class_count)
        },
    }


def _usage_row(token_ids: np.ndarray, capacity_k: int) -> dict[str, Any]:
    token_ids = np.asarray(token_ids, dtype=np.int64).reshape(-1)
    capacity_k = int(capacity_k)
    if capacity_k < 1 or token_ids.size == 0:
        raise ValueError("Codebook usage requires positive capacity and valid tokens.")
    if int(token_ids.min()) < 0 or int(token_ids.max()) >= capacity_k:
        raise RuntimeError("A hard primitive token lies outside codebook capacity.")
    counts = np.bincount(token_ids, minlength=capacity_k).astype(np.int64)
    used_ids = np.flatnonzero(counts > 0).astype(np.int64)
    probabilities = counts[counts > 0].astype(np.float64) / float(counts.sum())
    entropy = float(-(probabilities * np.log(probabilities)).sum())
    normalized_entropy = (
        entropy / math.log(capacity_k) if capacity_k > 1 else 0.0
    )
    return {
        "capacity_k": capacity_k,
        "valid_window_count": int(token_ids.size),
        "used_k": int(used_ids.size),
        "used_ids": used_ids.tolist(),
        "dead_k": int(capacity_k - used_ids.size),
        "dead_fraction": float(1.0 - used_ids.size / capacity_k),
        "hard_counts": counts.tolist(),
        "usage_entropy_nats": entropy,
        "normalized_usage_entropy": float(normalized_entropy),
        "perplexity_effective_k": float(math.exp(entropy)),
    }


def _codebook_usage_diagnostics(
    token_ids: np.ndarray,
    token_class_ids: np.ndarray,
    token_subject_ids: np.ndarray,
    *,
    capacity_k: int,
    hard_boundary_count: int,
    valid_pair_count: int,
) -> dict[str, Any]:
    token_ids = np.asarray(token_ids, dtype=np.int64).reshape(-1)
    class_ids = np.asarray(token_class_ids, dtype=np.int64).reshape(-1)
    subject_ids = np.asarray(token_subject_ids, dtype=np.int64).reshape(-1)
    if not (len(token_ids) == len(class_ids) == len(subject_ids)):
        raise RuntimeError("Token, class and subject usage columns are misaligned.")
    overall = _usage_row(token_ids, capacity_k)
    overall.update(
        {
            "hard_boundary_count": int(hard_boundary_count),
            "valid_adjacent_pair_count": int(valid_pair_count),
            "hard_boundary_rate": (
                float(hard_boundary_count) / float(valid_pair_count)
                if valid_pair_count > 0
                else 0.0
            ),
            "by_class_model_index": {
                str(int(class_id)): _usage_row(
                    token_ids[class_ids == class_id], capacity_k
                )
                for class_id in np.unique(class_ids)
            },
            "by_subject": {
                str(int(subject_id)): _usage_row(
                    token_ids[subject_ids == subject_id], capacity_k
                )
                for subject_id in np.unique(subject_ids)
            },
            "capacity_is_not_learned_primitive_count": True,
        }
    )
    return overall


@torch.inference_mode()
def evaluate(
    model: MotionPrimitiveCGCDModel,
    loader: DataLoader,
    device: torch.device,
    args: argparse.Namespace,
) -> dict[str, Any]:
    model.eval()
    targets: list[np.ndarray] = []
    trial_ids: list[np.ndarray] = []
    trajectory_probabilities: list[np.ndarray] = []
    hard_token_rows: list[np.ndarray] = []
    token_class_rows: list[np.ndarray] = []
    token_subject_rows: list[np.ndarray] = []
    run_count_rows: list[np.ndarray] = []
    hard_boundary_count = 0
    valid_pair_count = 0
    subject_lookup = build_trial_subject_lookup(loader.dataset)

    for batch in loader:
        inputs, labels, batch_trial_ids = batch
        if isinstance(inputs, list):
            raise RuntimeError("Evaluation must use one unaugmented full-trial view.")
        prepared = move_trial_batch_to_device(inputs, device)
        output = model.forward_trajectory(prepared, hard_codebook=True)
        trajectory_probabilities.append(
            torch.softmax(output["trajectory_logits"], dim=-1).cpu().numpy()
        )
        token_mask = output["token_mask"].bool()
        hard_tokens = output["hard_tokens"].long()
        if tuple(hard_tokens.shape) != tuple(token_mask.shape):
            raise RuntimeError("Hard primitive tokens and token mask are misaligned.")
        hard_token_rows.append(hard_tokens[token_mask].cpu().numpy())
        expanded_classes = labels[:, None].expand_as(hard_tokens)
        token_class_rows.append(expanded_classes[token_mask.cpu()].numpy())
        batch_subjects = torch.as_tensor(
            [subject_lookup[int(value)] for value in batch_trial_ids.tolist()],
            dtype=torch.long,
        )
        expanded_subjects = batch_subjects[:, None].expand_as(hard_tokens.cpu())
        token_subject_rows.append(expanded_subjects[token_mask.cpu()].numpy())
        pair_mask = output["boundary_pair_mask"].bool()
        hard_pair_boundaries = output["hard_boundary_starts"].bool()[:, 1:]
        if tuple(pair_mask.shape) != tuple(hard_pair_boundaries.shape):
            raise RuntimeError("Hard boundaries and valid adjacent pairs are misaligned.")
        hard_boundary_count += int((hard_pair_boundaries & pair_mask).sum().item())
        valid_pair_count += int(pair_mask.sum().item())
        run_counts = output["run_lengths"].long().cpu().numpy().reshape(-1)
        if len(run_counts) != len(batch_trial_ids):
            raise RuntimeError("Run counts do not align with evaluation trials.")
        run_count_rows.append(run_counts)
        targets.append(labels.numpy())
        trial_ids.append(batch_trial_ids.numpy())

    truth = np.concatenate(targets)
    ids = np.concatenate(trial_ids)
    trajectory_probability = np.concatenate(trajectory_probabilities)
    codebook_usage = _codebook_usage_diagnostics(
        np.concatenate(hard_token_rows),
        np.concatenate(token_class_rows),
        np.concatenate(token_subject_rows),
        capacity_k=int(model.codebook.codebook_size),
        hard_boundary_count=hard_boundary_count,
        valid_pair_count=valid_pair_count,
    )
    run_counts = np.concatenate(run_count_rows).astype(np.int64)
    if np.any(run_counts < 1) or len(run_counts) != len(ids):
        raise RuntimeError("Every evaluated trial must contain at least one run.")
    run_count_diagnostics = {
        "trial_count": int(len(run_counts)),
        "mean": float(run_counts.mean()),
        "median": float(np.median(run_counts)),
        "minimum": int(run_counts.min()),
        "maximum": int(run_counts.max()),
        "one_run_fraction": float(np.mean(run_counts == 1)),
        "per_trial": {
            str(int(trial_id)): int(run_count)
            for trial_id, run_count in zip(ids.tolist(), run_counts.tolist())
        },
        "diagnostic_only_not_selection_penalty": True,
    }
    return {
        "sample_unit": "motion_primitive_trajectory",
        "trial_id_sha256": hashlib.sha256(np.sort(ids.astype(np.int64)).tobytes()).hexdigest(),
        "codebook_usage": codebook_usage,
        "run_count_diagnostics": run_count_diagnostics,
        "heads": {
            "trajectory": _classification_metrics(
                truth,
                trajectory_probability.argmax(axis=1),
                class_count=len(args.old_classes_parsed),
            )
        },
    }


def selection_score(metrics: Mapping[str, Any], args: argparse.Namespace) -> float:
    head = args.selection_head
    if head not in metrics["heads"]:
        raise KeyError(f"Selection head {head!r} is unavailable for {args.profile}.")
    return float(metrics["heads"][head][args.selection_metric])


def validation_selection_heads(args: argparse.Namespace) -> tuple[str, ...]:
    """Heads whose checkpoints are selected strictly from validation metrics."""
    if normalize_profile(args.profile) != PROFILE_JOINT:
        raise ValueError(f"Unknown profile {args.profile!r}.")
    return ("trajectory",)


def _dataset_summary(dataset: Any) -> dict[str, Any]:
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


def build_trial_subject_lookup(dataset: Any) -> dict[int, int]:
    """Build a fail-closed trial-id to subject-id map for SupCon masking."""

    trial_ids = np.asarray(dataset.trial_global_ids, dtype=np.int64).reshape(-1)
    subject_ids = np.asarray(dataset.subject_ids, dtype=np.int64).reshape(-1)
    if len(trial_ids) != len(subject_ids) or len(trial_ids) != len(dataset):
        raise RuntimeError("Trial and subject identity columns do not match dataset length.")
    lookup: dict[int, int] = {}
    for trial_id, subject_id in zip(trial_ids.tolist(), subject_ids.tolist()):
        trial_id = int(trial_id)
        subject_id = int(subject_id)
        previous = lookup.setdefault(trial_id, subject_id)
        if previous != subject_id:
            raise RuntimeError(
                f"Trial {trial_id} maps to multiple subjects: {previous}, {subject_id}."
            )
    if len(lookup) != len(trial_ids):
        raise RuntimeError("Trial-level dataset contains duplicate global trial IDs.")
    return lookup


def subject_ids_for_trial_batch(
    trial_ids: torch.Tensor,
    lookup: Mapping[int, int],
    *,
    device: torch.device,
) -> torch.Tensor:
    """Resolve batch subjects without exposing them through the model input."""

    flattened = trial_ids.detach().cpu().long().reshape(-1).tolist()
    missing = [int(trial_id) for trial_id in flattened if int(trial_id) not in lookup]
    if missing:
        raise RuntimeError(f"Training batch contains unknown trial IDs: {missing}.")
    return torch.as_tensor(
        [lookup[int(trial_id)] for trial_id in flattened],
        dtype=torch.long,
        device=device,
    )


def cross_subject_positive_coverage(
    labels: torch.Tensor,
    subject_ids: torch.Tensor,
) -> dict[str, float | int]:
    """Audit whether a random batch can actually apply cross-subject SupCon.

    An anchor is eligible only when another *trial* in the minibatch has the
    same activity label and a different subject ID.  The second augmented view
    of the same trial is deliberately not counted as cross-subject coverage.
    Pair counts are directed because the SupCon objective treats every trial
    as an anchor in turn.
    """

    if labels.ndim != 1 or subject_ids.ndim != 1:
        raise ValueError("labels and subject_ids must both have shape [B].")
    if labels.shape != subject_ids.shape:
        raise ValueError("labels and subject_ids must have the same shape.")
    if labels.numel() < 1:
        raise ValueError("Cross-subject coverage requires a non-empty batch.")
    labels = labels.detach().reshape(-1)
    subject_ids = subject_ids.detach().to(device=labels.device).reshape(-1)
    identity = torch.eye(len(labels), dtype=torch.bool, device=labels.device)
    directed_pairs = (
        labels[:, None].eq(labels[None, :])
        & subject_ids[:, None].ne(subject_ids[None, :])
        & ~identity
    )
    eligible = directed_pairs.any(dim=1)
    eligible_count = int(eligible.sum().cpu().item())
    anchor_count = int(labels.numel())
    pair_count = int(directed_pairs.sum().cpu().item())
    return {
        "eligible_anchor_count": eligible_count,
        "anchor_count": anchor_count,
        "eligible_anchor_fraction": eligible_count / anchor_count,
        "directed_pair_count": pair_count,
        "batch_has_cross_subject_positive": int(eligible_count > 0),
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
    old_count = len(args.old_classes_parsed)
    novel_count = int(args.total_classes) - old_count
    if novel_count <= 0 or novel_count % int(args.novel_classes_per_session) != 0:
        raise ValueError(
            "total_classes - number of old classes must be positive and divisible by "
            "--novel-classes-per-session."
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

    class_order = list(args.old_classes_parsed) + [int(value) for value in novel_order.tolist()]
    target_mapping = {physical: index for index, physical in enumerate(class_order)}
    for dataset in datasets.values():
        _set_target_transform(dataset, lambda label, mapping=target_mapping: mapping[int(label)])
    return datasets, novel_order


def build_loaders(
    datasets: Mapping[str, Any], args: argparse.Namespace
) -> tuple[DataLoader, DataLoader, DataLoader]:
    train_dataset = datasets["offline_train_dataset"]
    validation_dataset = datasets.get("offline_val_dataset")
    test_dataset = datasets["offline_test_dataset"]
    if validation_dataset is None or len(validation_dataset) == 0:
        raise RuntimeError("The subject-disjoint offline validation dataset is empty.")
    if len(train_dataset) < 2:
        raise RuntimeError("Offline training needs at least two complete trials.")
    drop_last = len(train_dataset) % int(args.batch_size) == 1
    common_train = {
        "num_workers": args.num_workers,
        "batch_size": args.batch_size,
        "shuffle": True,
        "drop_last": drop_last,
        "pin_memory": args.pin_memory,
        "generator": dataloader_generator(args.seed + 10),
        "worker_init_fn": seed_dataloader_worker,
        "collate_fn": uschad_trial_collate,
    }
    train_loader = DataLoader(train_dataset, **common_train)
    validation_loader = DataLoader(
        validation_dataset,
        num_workers=args.eval_num_workers,
        batch_size=args.eval_batch_size,
        shuffle=False,
        pin_memory=False,
        generator=dataloader_generator(args.seed + 11),
        worker_init_fn=seed_dataloader_worker,
        collate_fn=uschad_trial_collate,
    )
    test_loader = DataLoader(
        test_dataset,
        num_workers=args.eval_num_workers,
        batch_size=args.eval_batch_size,
        shuffle=False,
        pin_memory=False,
        generator=dataloader_generator(args.seed + 12),
        worker_init_fn=seed_dataloader_worker,
        collate_fn=uschad_trial_collate,
    )
    return train_loader, validation_loader, test_loader


def _checkpoint_payload(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: lr_scheduler.LambdaLR,
    *,
    epoch: int,
    validation: Mapping[str, Any],
    validation_score: float,
    args: argparse.Namespace,
    joint_config: MotionPrimitiveLossConfig,
    temporal_predictor: Optional[MaskedTemporalPredictor] = None,
    selection_head: Optional[str] = None,
) -> dict[str, Any]:
    selected_head = str(selection_head or args.selection_head)
    return {
        "schema": "hhr_motion_primitive_trajectory_offline_v3",
        "profile": args.profile,
        "epoch": int(epoch),
        "model": model.state_dict(),
        "offline_temporal_predictor": (
            temporal_predictor.state_dict() if temporal_predictor is not None else None
        ),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "validation_metrics": validation,
        "validation_score": float(validation_score),
        "selection_head": selected_head,
        "selection_metric": args.selection_metric,
        "trajectory_loss_config": joint_config.audit_dict(),
        "joint_loss_config": joint_config.audit_dict(),
        "a2_mp_contract": {
            "changepoint_anchor_source": (
                "clean_view_window_physical_descriptors_within_training_batch"
            ),
            "changepoint_anchor_scope": "fold_training_subjects_only",
            "changepoint_anchor_activity_labels_used": False,
            "changepoint_null_gate": "absolute_floor_plus_median_mad",
            "zero_change_trial_allowed": True,
            "view_alignment": "full_trial_same_window_coordinates",
            "content_boundary_representation": "primitive_features",
            "noncollapse": "variance_plus_off_diagonal_covariance",
            "temporal_prediction": (
                "learnable_masked_context_predictor_to_clean_stop_gradient_target"
            ),
            "temporal_predictor_deployment_role": "offline_auxiliary_only",
            "trial_auxiliary_classifier": False,
            "instance_infonce": False,
        },
        "architecture": model.config.audit_dict(),
        "encoder_initialization": args.encoder_initialization,
        "trial_encoder_checkpoint": (
            str(Path(args.trial_encoder_checkpoint).resolve())
            if args.encoder_initialization == "warmstart"
            else None
        ),
        "encoder_freeze_epochs": int(args.encoder_freeze_epochs),
        "old_classes_physical": list(args.old_classes_parsed),
        "test_metrics_used_for_selection": False,
    }


def _validate_subject_contract(args: argparse.Namespace) -> None:
    train = set(_comma_separated_ints(args.uschad_train_subjects))
    validation = set(_comma_separated_ints(args.offline_val_subjects))
    test = set(_comma_separated_ints(args.uschad_test_subjects))
    if not train or not validation or not test:
        raise ValueError("--train-subjects, --val-subjects and --test-subjects must all be non-empty.")
    overlaps = {
        "train/validation": sorted(train & validation),
        "train/test": sorted(train & test),
        "validation/test": sorted(validation & test),
    }
    nonempty = {name: values for name, values in overlaps.items() if values}
    if nonempty:
        raise ValueError(f"Subject split overlap is forbidden: {nonempty}.")


def validate_args(args: argparse.Namespace) -> argparse.Namespace:
    args.profile = normalize_profile(args.profile)
    args.old_classes_parsed = _comma_separated_ints(args.old_classes)
    if len(args.old_classes_parsed) < 2:
        raise ValueError("At least two old classes are required.")
    if len(set(args.old_classes_parsed)) != len(args.old_classes_parsed):
        raise ValueError("--old-classes contains duplicates.")
    if min(args.old_classes_parsed) < 0 or max(args.old_classes_parsed) >= args.total_classes:
        raise ValueError("--old-classes contains a label outside [0,total_classes).")
    _validate_subject_contract(args)
    if args.n_views != 2:
        raise ValueError("Single-stage trial training requires --n-views 2.")
    if args.trial_view_mode != "full_full":
        raise ValueError("A2-MP requires two aligned full-trial views.")
    if float(args.har_weak_jitter_std) != 0.0 or float(
        args.har_weak_scale_std
    ) != 0.0:
        raise ValueError(
            "A2-MP view[0] is the clean changepoint anchor; weak jitter and "
            "weak scaling must both be exactly zero."
        )
    if (args.uschad_window_size, args.window_stride) != (256, 128):
        raise ValueError("The registered E0 route requires window/stride 256/128.")
    if args.batch_size < 2:
        raise ValueError(
            "--batch-size must be at least 2 for supervised contrastive learning."
        )
    if args.epochs < 1:
        raise ValueError("--epochs must be positive.")
    if args.run_state_dim < 0:
        raise ValueError("--run-state-dim cannot be negative.")
    if args.temporal_predictor_hidden_dim < 0:
        raise ValueError("--temporal-predictor-hidden-dim cannot be negative.")
    if not math.isfinite(float(args.boundary_initial_bias)):
        raise ValueError("--boundary-initial-bias must be finite.")
    if not 0.0 <= float(args.changepoint_absolute_floor) <= 2.0:
        raise ValueError("--changepoint-absolute-floor must lie in [0,2].")
    if not math.isfinite(float(args.changepoint_null_mad_multiplier)) or float(
        args.changepoint_null_mad_multiplier
    ) < 0.0:
        raise ValueError(
            "--changepoint-null-mad-multiplier must be finite and non-negative."
        )
    if args.selection_head != "trajectory":
        raise ValueError(
            "motion_primitive_joint can only select the trajectory head."
        )
    if args.motion_weight < 0.0:
        raise ValueError("--motion-weight must be non-negative.")
    if (int(args.motion_ramp_start_epoch), int(args.motion_ramp_end_epoch)) != (0, 0):
        raise ValueError(
            "The registered trajectory-only A2-MP route requires motion ramp "
            "start=end=0; a ramp can otherwise create an empty-gradient epoch."
        )
    if args.motion_weight <= 0.0:
        raise ValueError(
            "The motion-primitive joint profile requires a positive "
            "--motion-weight."
        )
    if not math.isfinite(float(args.encoder_lr_scale)) or float(
        args.encoder_lr_scale
    ) <= 0.0:
        raise ValueError("--encoder-lr-scale must be finite and positive.")
    if args.encoder_initialization == "warmstart":
        if not str(args.trial_encoder_checkpoint).strip():
            raise ValueError(
                "Warm-start initialization requires --trial-encoder-checkpoint."
            )
        if not Path(args.trial_encoder_checkpoint).is_file():
            raise FileNotFoundError(
                f"Window encoder checkpoint not found: {args.trial_encoder_checkpoint}"
            )
    else:
        if not math.isclose(
            float(args.encoder_lr_scale), 1.0, rel_tol=0.0, abs_tol=1.0e-12
        ):
            raise ValueError(
                "Random ResNet1D initialization requires --encoder-lr-scale 1.0 "
                "so initialization and optimizer scale are not mixed."
            )
        if args.encoder_freeze_epochs != 0:
            raise ValueError(
                "Random encoder initialization requires --encoder-freeze-epochs 0."
            )
        if str(args.trial_encoder_checkpoint).strip():
            raise ValueError(
                "--trial-encoder-checkpoint is only valid with "
                "--encoder-initialization warmstart."
            )
    if not Path(args.uschad_npz_path).is_file():
        raise FileNotFoundError(f"USC-HAD NPZ not found: {args.uschad_npz_path}")
    return args


def train(args: argparse.Namespace) -> Mapping[str, Any]:
    args = validate_args(args)
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    selection_heads = validation_selection_heads(args)
    protected = [
        output_dir / "manifest.json",
        output_dir / "history.jsonl",
        output_dir / "checkpoint_best.pt",
        output_dir / "checkpoint_last.pt",
        output_dir / "summary.json",
    ] + [output_dir / f"checkpoint_best_{head}.pt" for head in selection_heads]
    existing = [str(path) for path in protected if path.exists()]
    if existing:
        raise FileExistsError(
            "Refusing to mix or overwrite an existing run. Use a new --output-dir: "
            f"{existing}"
        )
    logger = configure_logger(output_dir)
    args.logger = logger
    device = resolve_device(args.device)
    seed_everything(args.seed, use_cuda=device.type == "cuda")

    # Build before data access to keep module/data random-number ordering fixed.
    model = build_model(args).to(device)
    if args.encoder_initialization == "warmstart":
        load_trial_window_encoder(
            model,
            args.trial_encoder_checkpoint,
            args,
            logger,
        )
    joint_config = build_joint_loss_config(args)
    temporal_predictor = (
        MaskedTemporalPredictor(
            args.feature_dim,
            None if args.temporal_predictor_hidden_dim == 0
            else args.temporal_predictor_hidden_dim,
        ).to(device)
        if joint_config.temporal_prediction_weight > 0.0
        else None
    )
    optimizer = build_sgd_optimizer(model, args, temporal_predictor)
    scheduler = build_cosine_scheduler(optimizer, args.epochs, args.cosine_minimum_ratio)

    datasets, novel_order = build_datasets(args)
    train_loader, validation_loader, test_loader = build_loaders(datasets, args)
    train_trial_subject_lookup = build_trial_subject_lookup(
        datasets["offline_train_dataset"]
    )
    split_summary = {
        "train": _dataset_summary(datasets["offline_train_dataset"]),
        "validation": _dataset_summary(datasets["offline_val_dataset"]),
        "test": _dataset_summary(datasets["offline_test_dataset"]),
    }
    requested_subjects = {
        "train": sorted(_comma_separated_ints(args.uschad_train_subjects)),
        "validation": sorted(_comma_separated_ints(args.offline_val_subjects)),
        "test": sorted(_comma_separated_ints(args.uschad_test_subjects)),
    }
    observed_subjects = {
        name: summary["subjects"] for name, summary in split_summary.items()
    }
    if requested_subjects != observed_subjects:
        raise RuntimeError(
            "The constructed subject split differs from the explicit request: "
            f"requested={requested_subjects}, observed={observed_subjects}."
        )

    source_files = [
        Path(__file__).resolve(),
        PROJECT_ROOT / "data" / "uschad_har.py",
        PROJECT_ROOT / "data" / "uschad.py",
        PROJECT_ROOT / "models" / "motion_primitive_cgcd.py",
        PROJECT_ROOT / "models" / "motion_primitives.py",
        PROJECT_ROOT / "models" / "batch_utils.py",
        PROJECT_ROOT / "experiments" / "motion_primitive" / "joint_losses.py",
        PROJECT_ROOT / "experiments" / "motion_primitive" / "legacy_profiles.py",
        PROJECT_ROOT / "experiments" / "motion_primitive" / "trajectory_contrastive.py",
        PROJECT_ROOT / "models" / "resnet1d.py",
    ]
    manifest = {
        "schema": "hhr_motion_primitive_trajectory_manifest_v3",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "project_root": str(PROJECT_ROOT),
        "profile": args.profile,
        "npz_path": str(Path(args.uschad_npz_path).resolve()),
        "npz_sha256": _sha256_file(Path(args.uschad_npz_path).resolve()),
        "subjects": requested_subjects,
        "datasets": split_summary,
        "old_classes_physical": list(args.old_classes_parsed),
        "novel_class_order_physical": [int(value) for value in novel_order.tolist()],
        "architecture": model.config.audit_dict(),
        "single_stage_profile": registered_single_stage_profile(args),
        "trajectory_loss_config": joint_config.audit_dict(),
        "joint_loss_config": joint_config.audit_dict(),
        "motion_weight_schedule": {
            "type": "fixed_from_first_optimizer_step",
            "base_weight": float(joint_config.motion_total_weight),
            "start_epoch": int(args.motion_ramp_start_epoch),
            "end_epoch": int(args.motion_ramp_end_epoch),
            "optimizer_reset": False,
            "model_reset": False,
        },
        "optimizer": {
            "name": "SGD",
            "lr": args.lr,
            "encoder_lr_scale": args.encoder_lr_scale,
            "momentum": args.momentum,
            "weight_decay": args.weight_decay,
            "parameter_group_lrs": [float(group["lr"]) for group in optimizer.param_groups],
            "parameter_group_weight_decay": [
                float(group["weight_decay"]) for group in optimizer.param_groups
            ],
        },
        "encoder_initialization": {
            "mode": args.encoder_initialization,
            "checkpoint": (
                str(Path(args.trial_encoder_checkpoint).resolve())
                if args.encoder_initialization == "warmstart"
                else None
            ),
            "checkpoint_sha256": (
                _sha256_file(Path(args.trial_encoder_checkpoint).resolve())
                if args.encoder_initialization == "warmstart"
                else None
            ),
            "freeze_epochs": int(args.encoder_freeze_epochs),
            "provenance_validation": (
                "strict_split_old_classes_window_npz_sha256"
                if args.encoder_initialization == "warmstart"
                else "not_applicable"
            ),
        },
        "scheduler": {
            "name": "LambdaLR cosine",
            "epochs": args.epochs,
            "minimum_ratio": args.cosine_minimum_ratio,
        },
        "training_contract": {
            "sample_unit": "motion_primitive_trajectory",
            "representation": "variable_length_motion_primitive_trajectory",
            "activity_label_scope": "complete_old_class_trajectory_only",
            "window_activity_labels": False,
            "complete_trial_pooling": False,
            "pooled_or_fused_head": False,
            "subject_ids_used_only_for_cross_subject_supcon_mask": True,
            "random_batch_cross_subject_positive_coverage_audited": True,
            "class_subject_aware_sampler": False,
            "views_per_batch": 2,
            "forward_calls_per_batch": 2,
            "optimizers": 1,
            "backward_calls_per_batch": 1,
            "checkpoint_selection_split": "validation",
            "validation_selection_heads": list(selection_heads),
            "test_evaluations": len(selection_heads),
            "test_metrics_used_for_selection": False,
        },
        "arguments": {
            key: value
            for key, value in vars(args).items()
            if key not in {"logger", "old_classes_parsed"}
        },
        "source_sha256": {str(path.relative_to(PROJECT_ROOT)): _sha256_file(path) for path in source_files},
        "torch_version": torch.__version__,
        "numpy_version": np.__version__,
        "device": str(device),
    }
    _write_json(output_dir / "manifest.json", manifest)
    logger.info(
        "profile=%s device=%s train/val/test trials=%d/%d/%d",
        args.profile,
        device,
        len(datasets["offline_train_dataset"]),
        len(datasets["offline_val_dataset"]),
        len(datasets["offline_test_dataset"]),
    )
    logger.info(
        "Registered single-stage profile match: %s",
        manifest["single_stage_profile"]["matches_registered_single_stage_profile"],
    )
    logger.info("Motion trajectory weight: %.6g", args.motion_weight)

    best_scores = {head: -float("inf") for head in selection_heads}
    best_epochs = {head: -1 for head in selection_heads}
    best_validations: dict[str, Mapping[str, Any]] = {}
    optimizer_steps = 0
    backward_calls = 0
    cross_subject_coverage_by_epoch: list[dict[str, Any]] = []

    for epoch_index in range(args.epochs):
        model.train()
        if temporal_predictor is not None:
            temporal_predictor.train()
        encoder_frozen = configure_window_encoder_for_epoch(
            model,
            epoch_index + 1,
            args.encoder_freeze_epochs,
        )
        if epoch_index == 0 or epoch_index + 1 == args.encoder_freeze_epochs + 1:
            logger.info(
                "epoch=%d window_encoder_frozen=%s",
                epoch_index + 1,
                encoder_frozen,
            )
        accumulators: dict[str, ScalarAccumulator] = {}
        coverage_anchor_count = 0
        coverage_anchor_total = 0
        coverage_directed_pair_count = 0
        coverage_batch_count = 0
        coverage_covered_batch_count = 0
        for batch_index, batch in enumerate(train_loader):
            views, labels, batch_trial_ids = batch
            if not isinstance(views, list) or len(views) != 2:
                raise RuntimeError("Training data must return exactly two trial views.")
            prepared_views = [
                move_trial_batch_to_device(view, device) for view in views
            ]
            labels = labels.to(device=device, non_blocking=True)
            batch_subject_ids = subject_ids_for_trial_batch(
                batch_trial_ids,
                train_trial_subject_lookup,
                device=device,
            )
            batch_coverage = cross_subject_positive_coverage(
                labels, batch_subject_ids
            )
            batch_metrics = training_step(
                model,
                prepared_views,
                labels,
                optimizer,
                args,
                epoch_index=epoch_index,
                loss_config=joint_config,
                subject_ids=batch_subject_ids,
                temporal_predictor=temporal_predictor,
            )
            optimizer_steps += 1
            backward_calls += 1
            for name, value in batch_metrics.items():
                if name.startswith("cross_subject_positive_"):
                    continue
                accumulators.setdefault(name, ScalarAccumulator()).update(value, len(labels))
            coverage_anchor_count += int(batch_coverage["eligible_anchor_count"])
            coverage_anchor_total += int(batch_coverage["anchor_count"])
            coverage_directed_pair_count += int(
                batch_coverage["directed_pair_count"]
            )
            coverage_batch_count += 1
            coverage_covered_batch_count += int(
                batch_coverage["batch_has_cross_subject_positive"]
            )
            if batch_index % args.print_frequency == 0:
                logger.info(
                    "epoch=%d batch=%d/%d loss=%.6f motion=%.6f train_acc=%.4f",
                    epoch_index + 1,
                    batch_index,
                    len(train_loader),
                    batch_metrics["total"],
                    batch_metrics["motion_total"],
                    batch_metrics["train_supervised_accuracy"],
                )

        if coverage_anchor_total <= 0 or coverage_batch_count <= 0:
            raise RuntimeError("No training anchors were audited in this epoch.")
        epoch_coverage = {
            "eligible_anchor_count": coverage_anchor_count,
            "anchor_count": coverage_anchor_total,
            "eligible_anchor_fraction": coverage_anchor_count
            / coverage_anchor_total,
            "directed_pair_count": coverage_directed_pair_count,
            "covered_batch_count": coverage_covered_batch_count,
            "batch_count": coverage_batch_count,
            "covered_batch_fraction": coverage_covered_batch_count
            / coverage_batch_count,
            "same_trial_second_view_counted_as_cross_subject": False,
            "class_subject_aware_sampler": False,
        }
        cross_subject_coverage_by_epoch.append(epoch_coverage)

        validation = evaluate(model, validation_loader, device, args)
        selection_scores = {
            head: float(validation["heads"][head][args.selection_metric])
            for head in selection_heads
        }
        score = selection_scores[args.selection_head]
        train_metrics = {
            name: accumulator.mean for name, accumulator in accumulators.items()
        }
        train_metrics.update(
            {
                "cross_subject_positive_anchor_fraction": epoch_coverage[
                    "eligible_anchor_fraction"
                ],
                "cross_subject_positive_anchor_count": epoch_coverage[
                    "eligible_anchor_count"
                ],
                "cross_subject_positive_anchor_total": epoch_coverage["anchor_count"],
                "cross_subject_positive_directed_pair_count": epoch_coverage[
                    "directed_pair_count"
                ],
                "cross_subject_positive_batch_fraction": epoch_coverage[
                    "covered_batch_fraction"
                ],
            }
        )
        epoch_record = {
            "epoch": epoch_index + 1,
            "learning_rates": [float(group["lr"]) for group in optimizer.param_groups],
            "train": train_metrics,
            "cross_subject_supcon_coverage": epoch_coverage,
            "validation": validation,
            "selection_head": args.selection_head,
            "selection_metric": args.selection_metric,
            "selection_score": score,
            "selection_scores_by_head": selection_scores,
            "window_encoder_frozen": encoder_frozen,
            "cumulative_optimizer_steps": optimizer_steps,
            "cumulative_backward_calls": backward_calls,
        }
        _append_jsonl(output_dir / "history.jsonl", epoch_record)
        checkpoint = _checkpoint_payload(
            model,
            optimizer,
            scheduler,
            epoch=epoch_index + 1,
            validation=validation,
            validation_score=score,
            args=args,
            joint_config=joint_config,
            temporal_predictor=temporal_predictor,
            selection_head=args.selection_head,
        )
        torch.save(checkpoint, output_dir / "checkpoint_last.pt")
        for selection_head, head_score in selection_scores.items():
            if head_score <= best_scores[selection_head]:
                continue
            best_scores[selection_head] = head_score
            best_epochs[selection_head] = epoch_index + 1
            best_validations[selection_head] = validation
            head_checkpoint = _checkpoint_payload(
                model,
                optimizer,
                scheduler,
                epoch=epoch_index + 1,
                validation=validation,
                validation_score=head_score,
                args=args,
                joint_config=joint_config,
                temporal_predictor=temporal_predictor,
                selection_head=selection_head,
            )
            torch.save(
                head_checkpoint,
                output_dir / f"checkpoint_best_{selection_head}.pt",
            )
        logger.info(
            "epoch=%d validation %s/%s=%.4f best=%.4f@%d all_best=%s",
            epoch_index + 1,
            args.selection_head,
            args.selection_metric,
            score,
            best_scores[args.selection_head],
            best_epochs[args.selection_head],
            {
                head: {"score": best_scores[head], "epoch": best_epochs[head]}
                for head in selection_heads
            },
        )
        scheduler.step()

    incomplete_heads = [
        head for head in selection_heads
        if head not in best_validations or best_epochs[head] < 1
    ]
    if incomplete_heads:
        raise RuntimeError(
            "Training did not produce validation-selected checkpoints for "
            f"{incomplete_heads}."
        )
    evaluations_by_selection: dict[str, Any] = {}
    primary_checkpoint: Optional[MutableMapping[str, Any]] = None
    for selection_head in selection_heads:
        checkpoint_path = output_dir / f"checkpoint_best_{selection_head}.pt"
        selected_checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(selected_checkpoint["model"], strict=True)
        # The outer test is touched only after this head's validation-selected
        # epoch is fixed.  It never influences either checkpoint choice.
        selected_test_metrics = evaluate(model, test_loader, device, args)
        evaluation = {
            "selection_head": selection_head,
            "selection_metric": args.selection_metric,
            "best_epoch": best_epochs[selection_head],
            "best_validation_score": best_scores[selection_head],
            "best_validation": best_validations[selection_head],
            "test": selected_test_metrics,
            "test_metrics_used_for_selection": False,
        }
        evaluations_by_selection[selection_head] = evaluation
        selected_checkpoint["final_test_metrics"] = selected_test_metrics
        selected_checkpoint["test_metrics_used_for_selection"] = False
        torch.save(selected_checkpoint, checkpoint_path)
        if selection_head == args.selection_head:
            primary_checkpoint = selected_checkpoint

    primary = evaluations_by_selection[args.selection_head]
    total_eligible_anchors = sum(
        int(item["eligible_anchor_count"])
        for item in cross_subject_coverage_by_epoch
    )
    total_anchors = sum(
        int(item["anchor_count"]) for item in cross_subject_coverage_by_epoch
    )
    total_covered_batches = sum(
        int(item["covered_batch_count"])
        for item in cross_subject_coverage_by_epoch
    )
    total_batches = sum(
        int(item["batch_count"]) for item in cross_subject_coverage_by_epoch
    )
    cross_subject_coverage_summary = {
        "enabled_in_loss": bool(
            joint_config.trajectory_supcon_weight > 0.0
            and joint_config.cross_subject_supcon_only
        ),
        "sampler": "random_shuffle",
        "class_subject_aware_sampler": False,
        "epoch_count": len(cross_subject_coverage_by_epoch),
        "epoch_mean_eligible_anchor_fraction": float(
            np.mean(
                [
                    float(item["eligible_anchor_fraction"])
                    for item in cross_subject_coverage_by_epoch
                ]
            )
        ),
        "epoch_mean_covered_batch_fraction": float(
            np.mean(
                [
                    float(item["covered_batch_fraction"])
                    for item in cross_subject_coverage_by_epoch
                ]
            )
        ),
        "all_epochs_eligible_anchor_count": total_eligible_anchors,
        "all_epochs_anchor_count": total_anchors,
        "all_epochs_eligible_anchor_fraction": total_eligible_anchors
        / max(1, total_anchors),
        "all_epochs_covered_batch_count": total_covered_batches,
        "all_epochs_batch_count": total_batches,
        "all_epochs_covered_batch_fraction": total_covered_batches
        / max(1, total_batches),
        "same_trial_second_view_counted_as_cross_subject": False,
        "coverage_by_epoch": cross_subject_coverage_by_epoch,
    }
    summary = {
        "schema": "hhr_motion_primitive_trajectory_summary_v3",
        "profile": args.profile,
        "best_epoch": primary["best_epoch"],
        "selection_head": args.selection_head,
        "selection_metric": args.selection_metric,
        "best_validation_score": primary["best_validation_score"],
        "best_validation": primary["best_validation"],
        "test": primary["test"],
        "evaluations_by_selection": evaluations_by_selection,
        "test_evaluation_count": len(selection_heads),
        "test_metrics_used_for_selection": False,
        "optimizer_steps": optimizer_steps,
        "backward_calls": backward_calls,
        "single_optimizer_single_backward_contract_satisfied": optimizer_steps == backward_calls,
        "cross_subject_supcon_coverage": cross_subject_coverage_summary,
    }
    if primary_checkpoint is None:
        raise RuntimeError("Primary validation-selected checkpoint was not retained.")
    torch.save(primary_checkpoint, output_dir / "checkpoint_best.pt")
    _write_json(output_dir / "summary.json", summary)
    logger.info(
        "completed profile=%s best_epoch=%d validation=%.4f test_primary=%.4f",
        args.profile,
        primary["best_epoch"],
        primary["best_validation_score"],
        primary["test"]["heads"][args.selection_head]["accuracy"],
    )
    close_logger(logger)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="USC-HAD-only A2-MP motion-primitive trajectory trainer",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--profile",
        type=normalize_profile,
        choices=PROFILES,
        default=PROFILE_JOINT,
        help="The sole HHR motion-primitive trajectory experiment profile.",
    )
    parser.add_argument("--npz-path", dest="uschad_npz_path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--train-subjects", dest="uschad_train_subjects", required=True)
    parser.add_argument("--val-subjects", dest="offline_val_subjects", required=True)
    parser.add_argument("--test-subjects", dest="uschad_test_subjects", required=True)
    parser.add_argument("--old-classes", default="0,1,2,3,4,5")
    parser.add_argument("--total-classes", type=int, default=12)
    parser.add_argument("--novel-classes-per-session", type=int, default=2)
    parser.add_argument("--shuffle-novel-classes", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")

    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--eval-num-workers", type=int, default=0)
    parser.add_argument("--pin-memory", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--print-frequency", type=int, default=10)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument(
        "--encoder-lr-scale",
        type=float,
        default=1.0,
        help=(
            "Use 1.0 for random initialization; a warm-start experiment may "
            "explicitly choose a smaller ResNet1D learning-rate scale."
        ),
    )
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=5.0e-4)
    parser.add_argument("--cosine-minimum-ratio", type=float, default=1.0e-3)
    parser.add_argument("--gradient-clip-norm", type=float, default=0.0)

    parser.add_argument("--n-views", dest="n_views", type=int, default=2)
    parser.add_argument(
        "--trial-view-mode", choices=("full_full",), default="full_full"
    )
    parser.add_argument("--trial-crop-ratio", type=float, default=2.0 / 3.0)
    parser.add_argument("--trial-min-windows", type=int, default=2)
    parser.add_argument("--har-aug-mode", choices=("none", "weak_strong"), default="weak_strong")
    parser.add_argument(
        "--har-weak-jitter-std", type=float, choices=(0.0,), default=0.0
    )
    parser.add_argument("--har-weak-scale-std", type=float, choices=(0.0,), default=0.0)
    parser.add_argument("--har-strong-jitter-std", type=float, default=0.0)
    parser.add_argument("--har-strong-scale-std", type=float, default=0.20)
    parser.add_argument("--har-time-mask-ratio", type=float, default=0.0)

    parser.add_argument(
        "--window-size", dest="uschad_window_size", type=int, choices=(256,), default=256
    )
    parser.add_argument("--window-stride", type=int, choices=(128,), default=128)
    parser.add_argument("--in-channels", dest="har_in_channels", type=int, default=6)
    parser.add_argument("--feature-dim", type=int, default=256)
    parser.add_argument("--base-channels", type=int, default=64)
    parser.add_argument("--backbone-dropout", type=float, default=0.0)
    parser.add_argument(
        "--encoder-initialization",
        choices=("random", "warmstart"),
        required=True,
        help=(
            "Required experimental decision: random ResNet1D initialization or "
            "explicit ResNet1D checkpoint warm-start."
        ),
    )
    parser.add_argument("--trial-encoder-checkpoint", default="")
    parser.add_argument("--encoder-freeze-epochs", type=int, default=0)

    parser.add_argument("--motion-weight", type=float, default=1.0)
    parser.add_argument("--motion-ramp-start-epoch", type=int, default=0)
    parser.add_argument("--motion-ramp-end-epoch", type=int, default=0)
    parser.add_argument("--codebook-size", type=int, default=32)
    parser.add_argument("--codebook-temperature", type=float, default=0.25)
    parser.add_argument("--trajectory-input-dim", type=int, default=128)
    parser.add_argument("--trajectory-hidden-dim", type=int, default=128)
    parser.add_argument("--trajectory-layers", type=int, default=1)
    parser.add_argument("--trajectory-dropout", type=float, default=0.0)
    parser.add_argument(
        "--run-state-dim",
        type=int,
        default=12,
        help="Projected run-level physical descriptor size; zero disables it.",
    )
    parser.add_argument("--boundary-threshold", type=float, default=0.5)
    parser.add_argument(
        "--boundary-initial-bias",
        type=float,
        default=-1.5,
        help="Negative prior prevents random code jitter from over-segmenting epoch one.",
    )
    parser.add_argument("--trajectory-ce-weight", type=float, default=1.0)
    parser.add_argument("--trajectory-supcon-weight", type=float, default=0.25)
    parser.add_argument("--trajectory-view-consistency-weight", type=float, default=0.0)
    parser.add_argument("--changepoint-weight", type=float, default=1.0)
    parser.add_argument(
        "--content-boundary-alignment-weight", type=float, default=0.10
    )
    parser.add_argument("--noncollapse-weight", type=float, default=0.05)
    parser.add_argument("--temporal-prediction-weight", type=float, default=0.50)
    parser.add_argument("--temporal-prediction-mask-ratio", type=float, default=0.20)
    parser.add_argument("--temporal-predictor-hidden-dim", type=int, default=0)
    parser.add_argument("--changepoint-stable-quantile", type=float, default=0.25)
    parser.add_argument("--changepoint-change-quantile", type=float, default=0.75)
    parser.add_argument(
        "--changepoint-absolute-floor",
        type=float,
        default=0.01,
        help=(
            "Pre-registered cosine-change floor that permits a trial to have "
            "no change anchors; it remains an ablation parameter."
        ),
    )
    parser.add_argument(
        "--changepoint-null-mad-multiplier", type=float, default=3.0
    )
    parser.add_argument("--changepoint-rank-margin", type=float, default=0.20)
    parser.add_argument(
        "--changepoint-view-consistency-weight", type=float, default=0.50
    )
    parser.add_argument("--effective-minimum-duration-weight", type=float, default=0.02)
    parser.add_argument("--minimum-primitive-windows", type=int, default=2)
    parser.add_argument("--vq-commitment-weight", type=float, default=0.25)
    parser.add_argument("--vq-codebook-weight", type=float, default=0.25)
    parser.add_argument("--utilization-weight", type=float, default=0.0)
    parser.add_argument("--assignment-confidence-weight", type=float, default=0.0)
    parser.add_argument("--codebook-diversity-weight", type=float, default=0.0)
    parser.add_argument("--boundary-consistency-weight", type=float, default=0.0)
    parser.add_argument("--transition-budget-weight", type=float, default=0.02)
    parser.add_argument("--utilization-entropy-floor", type=float, default=0.50)
    parser.add_argument("--maximum-assignment-entropy", type=float, default=0.50)
    parser.add_argument("--maximum-codebook-cosine", type=float, default=0.25)
    parser.add_argument("--maximum-transition-rate", type=float, default=0.35)
    parser.add_argument(
        "--selection-head", choices=("trajectory",), default="trajectory"
    )
    parser.add_argument(
        "--selection-metric",
        choices=("macro_f1",),
        default="macro_f1",
    )
    parser.add_argument(
        "--recompute-normalization",
        dest="uschad_recompute_norm_from_train_subjects",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--normalization-epsilon", dest="uschad_norm_eps", type=float, default=1.0e-6)
    parser.add_argument("--cv-fold", dest="uschad_cv_fold", type=int, default=-1)
    parser.add_argument("--online-old-trials", type=int, default=2)
    parser.add_argument("--online-novel-unseen-trials", type=int, default=5)
    parser.add_argument("--online-novel-seen-trials", type=int, default=2)
    return parser


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    args = build_parser().parse_args(argv)
    # Names consumed by data.uschad.py and the model builder are deliberately
    # materialized rather than hidden behind compatibility magic.
    args.uschad_sample_unit = "trial"
    args.uschad_split_mode = "subject"
    args.in_channels = args.har_in_channels
    args.window_size = args.uschad_window_size
    return args


def main(argv: Optional[Sequence[str]] = None) -> Mapping[str, Any]:
    return train(parse_args(argv))


if __name__ == "__main__":
    main()
