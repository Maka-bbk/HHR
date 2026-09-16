"""Online CGCD for the variable-length motion-primitive trajectory model.

There is deliberately no complete-trial pooling, pooled classifier, ProtoAug,
or fused prediction in this module.  Both optimisation and evaluation operate
only on the trajectory embedding produced from ordered primitive runs.
"""

from __future__ import annotations

import copy
import json
import logging
import math
import os
import random
import warnings
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import confusion_matrix, f1_score
from torch.optim import SGD, lr_scheduler
from torch.utils.data import DataLoader, Dataset

from data.uschad_har import uschad_trial_collate
from experiments.motion_primitive.trajectory_distillation import (
    CrossViewDistillationLoss,
)
from models.batch_utils import move_trial_batch_to_device
from models.motion_primitive_cgcd import MotionPrimitiveCGCDModel, forward_two_views
from models.motion_primitives import SoftMotionCodebook


ONLINE_CONFIG_SCHEMA = "hhr_motion_primitive_online_config_v3"
ONLINE_CHECKPOINT_SCHEMA = "hhr_motion_primitive_online_checkpoint_v3"
ONLINE_SUMMARY_SCHEMA = "hhr_motion_primitive_online_summary_v3"


@dataclass(frozen=True)
class MotionPrimitiveOnlineConfig:
    old_class_count: int = 6
    total_class_count: int = 12
    novel_classes_per_session: int = 2
    online_old_trials_per_class: int = 2
    online_new_first_trials_per_class: int = 5
    online_new_seen_trials_per_class: int = 2
    epochs_per_session: int = 30
    batch_size: int = 16
    evaluation_batch_size: int = 16
    num_workers: int = 0
    evaluation_num_workers: int = 0
    learning_rate: float = 0.01
    encoder_lr_scale: float = 0.1
    momentum: float = 0.9
    weight_decay: float = 5.0e-4
    cosine_minimum_ratio: float = 1.0e-3
    n_views: int = 2
    student_temperature: float = 0.10
    warmup_teacher_temperature: float = 0.05
    teacher_temperature: float = 0.05
    warmup_teacher_epochs: int = 10
    grouped_memax_old_new_weight: float = 1.0
    grouped_memax_old_in_weight: float = 1.0
    grouped_memax_new_in_weight: float = 1.0
    initialize_new_trajectory_head_with_kmeans: bool = True
    kmeans_random_state: int = 0
    trajectory_cluster_weight: float = 1.0
    trajectory_logit_distillation_weight: float = 1.0
    trajectory_feature_distillation_weight: float = 1.0
    primitive_feature_distillation_weight: float = 1.0
    old_codebook_anchor_weight: float = 1.0
    trajectory_view_consistency_weight: float = 0.0
    vq_commitment_weight: float = 0.25
    vq_codebook_weight: float = 0.25

    def validated(self) -> "MotionPrimitiveOnlineConfig":
        positive_integers = (
            "old_class_count", "total_class_count", "novel_classes_per_session",
            "online_old_trials_per_class", "online_new_first_trials_per_class",
            "online_new_seen_trials_per_class", "epochs_per_session", "batch_size",
            "evaluation_batch_size",
        )
        for name in positive_integers:
            value = getattr(self, name)
            if isinstance(value, bool) or int(value) != value or int(value) < 1:
                raise ValueError(f"{name} must be a positive integer, got {value!r}.")
        for name in ("num_workers", "evaluation_num_workers", "warmup_teacher_epochs"):
            value = getattr(self, name)
            if isinstance(value, bool) or int(value) != value or int(value) < 0:
                raise ValueError(f"{name} must be a non-negative integer.")
        novel = self.total_class_count - self.old_class_count
        if novel <= 0 or novel % self.novel_classes_per_session:
            raise ValueError("The novel class count must be positive and session-divisible.")
        if self.n_views != 2:
            raise ValueError("Online trajectory training requires exactly two views.")
        if self.warmup_teacher_epochs > self.epochs_per_session:
            raise ValueError("warmup_teacher_epochs cannot exceed epochs_per_session.")
        for name in (
            "learning_rate", "encoder_lr_scale", "student_temperature",
            "warmup_teacher_temperature", "teacher_temperature",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be positive and finite.")
        for name in (
            "momentum", "weight_decay", "grouped_memax_old_new_weight",
            "grouped_memax_old_in_weight", "grouped_memax_new_in_weight",
            "trajectory_cluster_weight", "trajectory_logit_distillation_weight",
            "trajectory_feature_distillation_weight",
            "primitive_feature_distillation_weight", "old_codebook_anchor_weight",
            "trajectory_view_consistency_weight", "vq_commitment_weight",
            "vq_codebook_weight",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative.")
        if not 0 <= self.cosine_minimum_ratio <= 1:
            raise ValueError("cosine_minimum_ratio must lie in [0,1].")
        return self

    @property
    def session_count(self) -> int:
        return (self.total_class_count - self.old_class_count) // self.novel_classes_per_session

    @property
    def class_counts(self) -> tuple[int, ...]:
        return tuple(
            self.old_class_count + self.novel_classes_per_session * (index + 1)
            for index in range(self.session_count)
        )

    def audit_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload.update(
            {
                "schema": ONLINE_CONFIG_SCHEMA,
                "session_class_counts": list(self.class_counts),
                "sample_unit": "complete_trial_as_ordered_primitive_trajectory",
                "optimization_activity_labels_used": False,
                "window_level_activity_supervision_used": False,
                "primary_representation": "variable_length_motion_primitive_trajectory",
                "complete_trial_pooling": False,
                "pooled_or_fused_prediction": False,
                "primary_prediction_alignment": "constrained_old_fixed",
                "primitive_feature_distillation": (
                    "mean_cosine_distance_on_same_valid_unlabelled_windows"
                ),
                "old_codebook_anchor": (
                    "mean_cosine_distance_for_pre_session_codebook_rows_only"
                ),
            }
        )
        return payload


@dataclass
class OnlineLossResult:
    total: torch.Tensor
    trajectory_total: torch.Tensor
    components: Mapping[str, torch.Tensor]


@dataclass(frozen=True)
class CodebookExpansionDecision:
    centres: Optional[torch.Tensor]
    selected_delta: int
    audit: Mapping[str, Any]

    def __post_init__(self) -> None:
        delta = int(self.selected_delta)
        if isinstance(self.selected_delta, bool) or delta != self.selected_delta or delta < 0:
            raise ValueError("selected_delta must be a non-negative integer.")
        if delta == 0 and self.centres is not None:
            raise ValueError("A zero-delta decision cannot contain centres.")
        if delta > 0 and (
            not isinstance(self.centres, torch.Tensor)
            or self.centres.ndim != 2
            or self.centres.shape[0] != delta
        ):
            raise ValueError("Positive-delta centres must have shape [delta, dimension].")


@dataclass(frozen=True)
class LocalPrimitiveFeatureSet:
    """Local primitive features with label-free trial/subject provenance."""

    features: torch.Tensor
    trial_ids: torch.Tensor
    subject_ids: torch.Tensor

    def validated(self) -> "LocalPrimitiveFeatureSet":
        features = torch.as_tensor(self.features)
        trial_ids = torch.as_tensor(self.trial_ids)
        subject_ids = torch.as_tensor(self.subject_ids)
        if features.ndim != 2 or not len(features):
            raise ValueError("Local primitive features must be a non-empty rank-2 tensor.")
        if not features.is_floating_point() or not torch.isfinite(features).all():
            raise ValueError("Local primitive features must be finite floating point values.")
        if trial_ids.ndim != 1 or subject_ids.ndim != 1:
            raise ValueError("Primitive trial_ids and subject_ids must be rank-1 tensors.")
        if len(trial_ids) != len(features) or len(subject_ids) != len(features):
            raise ValueError("Every primitive feature needs one trial_id and subject_id.")
        if trial_ids.dtype == torch.bool or trial_ids.is_floating_point():
            raise ValueError("Primitive trial_ids must be integer-valued.")
        if subject_ids.dtype == torch.bool or subject_ids.is_floating_point():
            raise ValueError("Primitive subject_ids must be integer-valued.")
        trial_ids = trial_ids.detach().long().cpu()
        subject_ids = subject_ids.detach().long().cpu()
        features = features.detach().cpu()
        for trial_id in torch.unique(trial_ids):
            owners = torch.unique(subject_ids[trial_ids == trial_id])
            if len(owners) != 1:
                raise ValueError(
                    f"Trial {int(trial_id)} maps to multiple subjects: {owners.tolist()}."
                )
        return LocalPrimitiveFeatureSet(features, trial_ids, subject_ids)


class HARUnlabelledPairDataset(Dataset):
    def __init__(self, old_dataset: Dataset, novel_dataset: Dataset) -> None:
        if old_dataset is None or novel_dataset is None or not len(old_dataset) or not len(novel_dataset):
            raise ValueError("Both old and novel online datasets must be non-empty.")
        self.old_unlabelled_dataset = old_dataset
        self.novel_unlabelled_dataset = novel_dataset
        trial_parts: list[np.ndarray] = []
        subject_parts: list[np.ndarray] = []
        for source in (old_dataset, novel_dataset):
            if not hasattr(source, "trial_global_ids") or not hasattr(
                source, "subject_ids"
            ):
                raise TypeError(
                    "Online trial datasets must expose trial_global_ids and subject_ids."
                )
            trials = np.asarray(source.trial_global_ids, dtype=np.int64).reshape(-1)
            subjects = np.asarray(source.subject_ids, dtype=np.int64).reshape(-1)
            if len(trials) != len(source) or len(subjects) != len(source):
                raise ValueError("Online trial/subject provenance length does not match dataset.")
            trial_parts.append(trials)
            subject_parts.append(subjects)
        self.trial_global_ids = np.concatenate(trial_parts)
        self.subject_ids = np.concatenate(subject_parts)
        if len(np.unique(self.trial_global_ids)) != len(self.trial_global_ids):
            raise ValueError("Old and novel online datasets contain duplicate trial ids.")

    def __len__(self) -> int:
        return len(self.old_unlabelled_dataset) + len(self.novel_unlabelled_dataset)

    def __getitem__(self, index: int):
        source = self.old_unlabelled_dataset
        local = int(index)
        if local >= len(source):
            local -= len(source)
            source = self.novel_unlabelled_dataset
        inputs, label, trial_id = source[local]
        return inputs, label, trial_id, np.asarray([0], dtype=np.int64)


def _raw_targets(dataset: Any) -> np.ndarray:
    if not hasattr(dataset, "targets"):
        raise TypeError(f"Dataset {type(dataset).__name__} has no auditable targets field.")
    return np.asarray(dataset.targets, dtype=np.int64).reshape(-1)


def _trial_ids(dataset: Any) -> np.ndarray:
    if not hasattr(dataset, "trial_global_ids"):
        raise TypeError(f"Dataset {type(dataset).__name__} has no trial_global_ids field.")
    return np.asarray(dataset.trial_global_ids, dtype=np.int64).reshape(-1)


def _counts(values: np.ndarray) -> dict[int, int]:
    unique, count = np.unique(values, return_counts=True)
    return {int(key): int(value) for key, value in zip(unique, count)}


def audit_uschad_online_protocol(
    datasets: Mapping[str, Any], novel_class_order: Sequence[int],
    old_classes: Sequence[int], config: MotionPrimitiveOnlineConfig,
) -> list[dict[str, Any]]:
    cfg = config.validated()
    old = [int(value) for value in old_classes]
    novel = [int(value) for value in novel_class_order]
    if len(old) != cfg.old_class_count or len(set(old)) != len(old):
        raise ValueError("old_classes does not match old_class_count.")
    if len(novel) != cfg.total_class_count - cfg.old_class_count or set(old) & set(novel):
        raise ValueError("novel_class_order is inconsistent with the class protocol.")
    streams = (
        datasets.get("online_old_dataset_unlabelled_list"),
        datasets.get("online_novel_dataset_unlabelled_list"),
        datasets.get("online_test_dataset_list"),
    )
    if not all(isinstance(value, list) and len(value) == cfg.session_count for value in streams):
        raise RuntimeError("USC-HAD online streams do not match session_count.")
    old_streams, novel_streams, test_streams = streams
    cumulative_train: set[int] = set()
    audits: list[dict[str, Any]] = []
    for index in range(cfg.session_count):
        old_dataset, novel_dataset, test_dataset = (
            old_streams[index], novel_streams[index], test_streams[index]
        )
        active_novel = novel[: (index + 1) * cfg.novel_classes_per_session]
        first_novel = set(active_novel[-cfg.novel_classes_per_session :])
        expected_old = {label: cfg.online_old_trials_per_class for label in old}
        expected_novel = {
            label: (cfg.online_new_first_trials_per_class if label in first_novel
                    else cfg.online_new_seen_trials_per_class)
            for label in active_novel
        }
        if _counts(_raw_targets(old_dataset)) != expected_old:
            raise RuntimeError(f"Session {index + 1} violates the old 2/class contract.")
        if _counts(_raw_targets(novel_dataset)) != expected_novel:
            raise RuntimeError(f"Session {index + 1} violates the novel 5-first/2-seen contract.")
        expected_test = set(old) | set(active_novel)
        if set(_counts(_raw_targets(test_dataset))) != expected_test:
            raise RuntimeError(f"Session {index + 1} test classes are incomplete.")
        old_ids, novel_ids = set(_trial_ids(old_dataset)), set(_trial_ids(novel_dataset))
        if old_ids & novel_ids or cumulative_train & (old_ids | novel_ids):
            raise RuntimeError(f"Session {index + 1} reuses online training trials.")
        cumulative_train |= old_ids | novel_ids
        test_ids = set(_trial_ids(test_dataset))
        if cumulative_train & test_ids:
            raise RuntimeError(f"Session {index + 1} has train/test trial leakage.")
        audits.append(
            {
                "session": index + 1,
                "class_count": cfg.class_counts[index],
                "seen_before": cfg.old_class_count + index * cfg.novel_classes_per_session,
                "old_trial_counts_physical": expected_old,
                "novel_trial_counts_physical": expected_novel,
                "test_classes_physical": sorted(expected_test),
                "train_trial_count": len(old_ids | novel_ids),
                "test_trial_count": len(test_ids),
                "cumulative_train_test_trial_overlap": 0,
            }
        )
    return audits


def build_online_session_datasets(
    datasets: Mapping[str, Any], config: MotionPrimitiveOnlineConfig
) -> list[HARUnlabelledPairDataset]:
    cfg = config.validated()
    old = datasets["online_old_dataset_unlabelled_list"]
    novel = datasets["online_novel_dataset_unlabelled_list"]
    return [HARUnlabelledPairDataset(old[i], novel[i]) for i in range(cfg.session_count)]


def _disable_augmentation(dataset: Any) -> Any:
    if isinstance(dataset, HARUnlabelledPairDataset):
        return HARUnlabelledPairDataset(
            _disable_augmentation(dataset.old_unlabelled_dataset),
            _disable_augmentation(dataset.novel_unlabelled_dataset),
        )
    result = copy.deepcopy(dataset)
    if not hasattr(result, "transform"):
        raise TypeError(f"Dataset {type(result).__name__} has no transform field.")
    result.transform = None
    return result


def make_online_loader(
    dataset: Dataset, config: MotionPrimitiveOnlineConfig, *, seed: int, training: bool
) -> DataLoader:
    cfg = config.validated()
    if training and len(dataset) < 2:
        raise RuntimeError("Online two-view training requires at least two trials.")
    generator = torch.Generator().manual_seed(int(seed))
    return DataLoader(
        dataset,
        batch_size=cfg.batch_size if training else cfg.evaluation_batch_size,
        shuffle=training,
        drop_last=bool(training and len(dataset) % cfg.batch_size == 1),
        num_workers=cfg.num_workers if training else cfg.evaluation_num_workers,
        pin_memory=training,
        generator=generator,
        worker_init_fn=_seed_online_worker,
        collate_fn=uschad_trial_collate,
    )


def _seed_online_worker(worker_id: int) -> None:
    del worker_id
    seed = torch.initial_seed() % (2**32)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def unlabelled_views_from_batch(batch: Sequence[Any]) -> Sequence[Mapping[str, torch.Tensor]]:
    if len(batch) != 4:
        raise ValueError("Online training expects (views, labels, trial_ids, flags).")
    views, _private_labels, _trial_ids_value, flags = batch
    flags = flags if isinstance(flags, torch.Tensor) else torch.as_tensor(flags)
    if torch.any(flags != 0):
        raise RuntimeError("Online optimisation received an activity-label flag.")
    if not isinstance(views, list) or len(views) != 2:
        raise RuntimeError("Online trajectory training requires exactly two views.")
    return views


def build_online_sgd(model: nn.Module, config: MotionPrimitiveOnlineConfig) -> SGD:
    cfg = config.validated()
    grouped: dict[tuple[bool, bool], list[nn.Parameter]] = {}
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            grouped.setdefault(
                (name.startswith("window_encoder."), name.endswith(".bias") or parameter.ndim == 1),
                [],
            ).append(parameter)
    groups = []
    for (encoder, no_decay), parameters in grouped.items():
        group: dict[str, Any] = {
            "params": parameters,
            "lr": cfg.learning_rate * (cfg.encoder_lr_scale if encoder else 1.0),
        }
        if no_decay:
            group["weight_decay"] = 0.0
        groups.append(group)
    if not groups:
        raise RuntimeError("No trainable online parameters were found.")
    return SGD(groups, lr=cfg.learning_rate, momentum=cfg.momentum, weight_decay=cfg.weight_decay)


def build_online_cosine_scheduler(
    optimizer: torch.optim.Optimizer, config: MotionPrimitiveOnlineConfig
) -> lr_scheduler.LambdaLR:
    cfg = config.validated()
    def multiplier(epoch: int) -> float:
        progress = min(max(float(epoch) / cfg.epochs_per_session, 0.0), 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return cfg.cosine_minimum_ratio + (1.0 - cfg.cosine_minimum_ratio) * cosine
    return lr_scheduler.LambdaLR(optimizer, lr_lambda=multiplier)


def clone_motion_primitive_model(model: MotionPrimitiveCGCDModel) -> MotionPrimitiveCGCDModel:
    if not isinstance(model, MotionPrimitiveCGCDModel):
        raise TypeError("Online CGCD requires MotionPrimitiveCGCDModel.")
    live = replace(
        model.config, old_class_count=model.class_count, codebook_size=model.codebook_size
    ).validated()
    with torch.random.fork_rng(devices=[], enabled=True):
        clone = MotionPrimitiveCGCDModel(live)
    reference = next(model.parameters())
    clone = clone.to(device=reference.device, dtype=reference.dtype)
    clone.load_state_dict(model.state_dict(), strict=True)
    clone.train(model.training)
    return clone


def expand_online_model(
    previous_model: MotionPrimitiveCGCDModel,
    target_class_count: int,
    config: MotionPrimitiveOnlineConfig,
) -> MotionPrimitiveCGCDModel:
    cfg = config.validated()
    seen, target = previous_model.class_count, int(target_class_count)
    if target != seen + cfg.novel_classes_per_session:
        raise ValueError("Each online session must add novel_classes_per_session rows.")
    current = clone_motion_primitive_model(previous_model)
    current.expand_class_heads(target)
    current.config = replace(current.config, old_class_count=target).validated()
    return current


@torch.inference_mode()
def extract_trajectory_embeddings(
    model: MotionPrimitiveCGCDModel, loader: DataLoader, device: torch.device
) -> torch.Tensor:
    model.eval()
    values = []
    for batch in loader:
        inputs = batch[0]
        if isinstance(inputs, list):
            raise RuntimeError("Trajectory discovery requires unaugmented trials.")
        output = model(move_trial_batch_to_device(inputs, device), hard_codebook=True)
        values.append(F.normalize(output["trajectory_embedding"], dim=-1).cpu())
    if not values:
        raise RuntimeError("Cannot discover trajectory centres from an empty stream.")
    return torch.cat(values)


def select_kmeans_new_trajectory_centres(
    representation_model: MotionPrimitiveCGCDModel,
    previous_model: MotionPrimitiveCGCDModel,
    evaluation_loader: DataLoader,
    target_class_count: int,
    config: MotionPrimitiveOnlineConfig,
    device: torch.device,
) -> torch.Tensor:
    cfg = config.validated()
    target = int(target_class_count)
    features = extract_trajectory_embeddings(representation_model, evaluation_loader, device)
    if len(features) < target:
        raise RuntimeError(f"Trajectory KMeans needs at least {target} trials, got {len(features)}.")
    estimator = KMeans(n_clusters=target, random_state=cfg.kmeans_random_state, n_init=10).fit(features.numpy())
    centres = F.normalize(
        torch.as_tensor(estimator.cluster_centers_, device=device,
                        dtype=previous_model.trajectory_classifier.weight.dtype), dim=-1
    )
    with torch.no_grad():
        confidence = previous_model.trajectory_classifier(centres).max(dim=-1).values
        indices = torch.topk(confidence, k=cfg.novel_classes_per_session, largest=False).indices
    return centres[indices]


@torch.no_grad()
def initialise_new_trajectory_rows_(
    model: MotionPrimitiveCGCDModel, seen_class_count: int, centres: torch.Tensor
) -> None:
    seen = int(seen_class_count)
    classifier = model.trajectory_classifier
    expected = (model.class_count - seen, classifier.in_features)
    if tuple(centres.shape) != expected:
        raise ValueError(f"Trajectory centres have {tuple(centres.shape)}, expected {expected}.")
    centres = F.normalize(centres.to(classifier.weight), dim=-1)
    reference_norm = classifier.weight[:seen].norm(dim=-1).mean()
    if not torch.isfinite(reference_norm) or reference_norm <= 0:
        raise RuntimeError("Existing trajectory classifier rows have invalid norms.")
    classifier.weight[seen:].copy_(centres * reference_norm)
    classifier.bias[seen:].zero_()


@torch.no_grad()
def expand_motion_codebook_(
    model: MotionPrimitiveCGCDModel, new_centres: torch.Tensor
) -> tuple[int, int]:
    old_codebook = model.codebook
    old_k = old_codebook.codebook_size
    centres = torch.as_tensor(new_centres, device=old_codebook.vectors.device,
                              dtype=old_codebook.vectors.dtype)
    if centres.ndim != 2 or centres.shape[1] != old_codebook.embedding_dim or not len(centres):
        raise ValueError("new_centres must have non-empty shape [delta_K, embedding_dim].")
    if not torch.isfinite(centres).all() or torch.any(centres.norm(dim=-1) <= old_codebook.epsilon):
        raise ValueError("Every new codebook centre must be finite and non-zero.")
    centres = F.normalize(centres, dim=-1, eps=old_codebook.epsilon)
    new_k = old_k + len(centres)
    with torch.random.fork_rng(devices=[], enabled=True):
        expanded_codebook = SoftMotionCodebook(
            embedding_dim=old_codebook.embedding_dim, codebook_size=new_k,
            temperature=old_codebook.temperature,
            use_gumbel_training=old_codebook.use_gumbel_training,
            commitment_beta=old_codebook.commitment_beta,
            dead_code_fraction=old_codebook.dead_code_fraction,
            epsilon=old_codebook.epsilon,
        ).to(old_codebook.vectors)
    expanded_codebook.train(old_codebook.training)
    expanded_codebook.vectors[:old_k].copy_(old_codebook.vectors)
    expanded_codebook.vectors[old_k:].copy_(centres)
    expanded_codebook.ema_cluster_size[:old_k].copy_(old_codebook.ema_cluster_size)
    expanded_codebook.ema_vector_sum[:old_k].copy_(old_codebook.ema_vector_sum)
    if bool(old_codebook.ema_initialized.item()):
        expanded_codebook.ema_cluster_size[old_k:].fill_(1.0)
        expanded_codebook.ema_vector_sum[old_k:].copy_(centres)
        expanded_codebook.ema_initialized.fill_(True)

    model.codebook = expanded_codebook
    model.config = replace(model.config, codebook_size=new_k).validated()
    return old_k, new_k


@torch.inference_mode()
def extract_local_primitive_features(
    model: MotionPrimitiveCGCDModel, loader: DataLoader, device: torch.device
) -> LocalPrimitiveFeatureSet:
    """Encode local windows and preserve their trial/subject provenance.

    Activity labels in the benchmark batch are never inspected. Subject ids are
    joined from immutable trial-dataset metadata using the global trial id.
    """

    source_dataset = getattr(loader, "dataset", None)
    if source_dataset is None or not hasattr(source_dataset, "trial_global_ids") or not hasattr(
        source_dataset, "subject_ids"
    ):
        raise TypeError(
            "Codebook discovery loader.dataset must expose trial_global_ids and subject_ids."
        )
    source_trials = np.asarray(source_dataset.trial_global_ids, dtype=np.int64).reshape(-1)
    source_subjects = np.asarray(source_dataset.subject_ids, dtype=np.int64).reshape(-1)
    if len(source_trials) != len(source_dataset) or len(source_subjects) != len(source_dataset):
        raise ValueError("Codebook discovery dataset provenance has the wrong length.")
    trial_to_subject: dict[int, int] = {}
    for trial_id, subject_id in zip(source_trials, source_subjects):
        old_subject = trial_to_subject.setdefault(int(trial_id), int(subject_id))
        if old_subject != int(subject_id):
            raise ValueError(f"Trial {int(trial_id)} has inconsistent subject provenance.")

    model.eval()
    values: list[torch.Tensor] = []
    primitive_trial_ids: list[torch.Tensor] = []
    primitive_subject_ids: list[torch.Tensor] = []
    for batch in loader:
        if len(batch) < 3:
            raise ValueError("Codebook discovery batches must include global trial ids.")
        inputs = batch[0]
        if isinstance(inputs, list):
            raise RuntimeError("Codebook discovery requires unaugmented trials.")
        prepared = move_trial_batch_to_device(inputs, device)
        mask = prepared["mask"].bool()
        batch_trial_ids = torch.as_tensor(batch[2]).detach().long().reshape(-1)
        if len(batch_trial_ids) != mask.shape[0]:
            raise ValueError("Batch trial ids do not align with the trial mask.")
        unknown = [
            int(trial_id)
            for trial_id in batch_trial_ids.tolist()
            if int(trial_id) not in trial_to_subject
        ]
        if unknown:
            raise RuntimeError(
                f"Batch contains trial ids absent from dataset provenance: {unknown}."
            )
        batch_subject_ids = torch.as_tensor(
            [trial_to_subject[int(trial_id)] for trial_id in batch_trial_ids.tolist()],
            dtype=torch.long,
        )
        local = model.primitive_input_norm(model.window_encoder(prepared["windows"][mask]))
        values.append(F.normalize(local, dim=-1).cpu())
        primitive_trial_ids.append(
            batch_trial_ids[:, None].expand_as(mask.cpu())[mask.cpu()].clone()
        )
        primitive_subject_ids.append(
            batch_subject_ids[:, None].expand_as(mask.cpu())[mask.cpu()].clone()
        )
    if not values:
        raise RuntimeError("Cannot discover codes from an empty online stream.")
    return LocalPrimitiveFeatureSet(
        torch.cat(values),
        torch.cat(primitive_trial_ids),
        torch.cat(primitive_subject_ids),
    ).validated()


def select_kmeans_codebook_centres(
    model: MotionPrimitiveCGCDModel, evaluation_loader: DataLoader,
    new_code_count: int, *, device: torch.device, random_state: int = 0,
) -> torch.Tensor:
    count = int(new_code_count)
    if count < 1:
        raise ValueError("new_code_count must be positive.")
    extracted = extract_local_primitive_features(model, evaluation_loader, device)
    features = extracted.features
    if len(features) < count:
        raise RuntimeError(f"KMeans code discovery needs {count} windows, got {len(features)}.")
    estimator = KMeans(n_clusters=count, random_state=int(random_state), n_init=10).fit(features.numpy())
    return F.normalize(torch.as_tensor(estimator.cluster_centers_, device=device,
                                       dtype=model.codebook.vectors.dtype), dim=-1)


def select_residual_adaptive_codebook_centres(
    local_features: torch.Tensor, existing_centres: torch.Tensor, *, max_delta: int,
    trial_ids: torch.Tensor, subject_ids: torch.Tensor,
    residual_quantile: float, minimum_residual_support: int,
    minimum_cluster_support: int, minimum_relative_improvement: float,
    minimum_cluster_trials: int, minimum_cluster_subjects: int,
    complexity_penalty: float, random_state: int = 0,
) -> CodebookExpansionDecision:
    features_input, centres_input = torch.as_tensor(local_features), torch.as_tensor(existing_centres)
    if features_input.ndim != 2 or centres_input.ndim != 2 or features_input.shape[1] != centres_input.shape[1]:
        raise ValueError("Feature and codebook matrices must be non-empty aligned rank-2 tensors.")
    if not len(features_input) or not len(centres_input) or not torch.isfinite(features_input).all() or not torch.isfinite(centres_input).all():
        raise ValueError("Features and codebook centres must be non-empty and finite.")
    provenance = LocalPrimitiveFeatureSet(
        features_input, torch.as_tensor(trial_ids), torch.as_tensor(subject_ids)
    ).validated()
    features_input = provenance.features
    feature_trial_ids = provenance.trial_ids
    feature_subject_ids = provenance.subject_ids
    limit, quantile = int(max_delta), float(residual_quantile)
    support_gate, cluster_gate = int(minimum_residual_support), int(minimum_cluster_support)
    trial_gate, subject_gate = int(minimum_cluster_trials), int(minimum_cluster_subjects)
    improvement_gate, penalty = float(minimum_relative_improvement), float(complexity_penalty)
    if limit < 1 or support_gate < 1 or cluster_gate < 1 or trial_gate < 1 or subject_gate < 1:
        raise ValueError("Adaptive integer gates must be positive.")
    if not 0 <= quantile < 1 or not 0 <= improvement_gate <= 1 or penalty < 0:
        raise ValueError("Adaptive floating-point gates are outside their valid range.")
    result_device, result_dtype = centres_input.device, centres_input.dtype
    features = F.normalize(features_input.detach().float().cpu(), dim=-1)
    old = F.normalize(centres_input.detach().float().cpu(), dim=-1)
    residuals = (1 - (features @ old.T).amax(dim=1).clamp(-1, 1)).clamp_min(0)
    threshold = torch.quantile(residuals, quantile)
    residual_mask = residuals >= threshold
    selected = features[residual_mask]
    selected_residuals = residuals[residual_mask]
    selected_trial_ids = feature_trial_ids[residual_mask]
    selected_subject_ids = feature_subject_ids[residual_mask]
    residual_trial_support = int(torch.unique(selected_trial_ids).numel())
    residual_subject_support = int(torch.unique(selected_subject_ids).numel())
    baseline = float(selected_residuals.mean()) if len(selected) else float("nan")
    epsilon = float(torch.finfo(features.dtype).eps * 16)
    candidates: list[dict[str, Any]] = []
    accepted: dict[int, torch.Tensor] = {}
    for delta in range(1, limit + 1):
        row: dict[str, Any] = {
            "delta": delta, "candidate_codebook_size": len(old) + delta,
            "evaluated": False, "accepted": False, "cluster_supports": [],
            "cluster_distinct_trial_supports": [],
            "cluster_distinct_subject_supports": [],
            "mean_cosine_residual_before": baseline,
            "mean_cosine_residual_after": None, "absolute_improvement": None,
            "relative_improvement": None, "penalized_score": None,
            "rejection_reasons": [],
        }
        reasons = row["rejection_reasons"]
        if len(selected) < support_gate: reasons.append("insufficient_residual_support")
        if residual_trial_support < trial_gate:
            reasons.append("insufficient_residual_trial_support")
        if residual_subject_support < subject_gate:
            reasons.append("insufficient_residual_subject_support")
        if len(selected) < delta: reasons.append("fewer_residual_windows_than_candidate_codes")
        if len(selected) < delta * cluster_gate: reasons.append("insufficient_support_for_all_candidate_clusters")
        if not math.isfinite(baseline) or baseline <= epsilon: reasons.append("no_positive_quantization_residual")
        if not reasons:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", ConvergenceWarning)
                estimator = KMeans(n_clusters=delta, random_state=int(random_state) + delta - 1,
                                   n_init=10).fit(selected.numpy())
            supports = np.bincount(estimator.labels_, minlength=delta)
            cluster_trial_supports = [
                int(torch.unique(selected_trial_ids[torch.as_tensor(estimator.labels_ == cluster)]).numel())
                for cluster in range(delta)
            ]
            cluster_subject_supports = [
                int(torch.unique(selected_subject_ids[torch.as_tensor(estimator.labels_ == cluster)]).numel())
                for cluster in range(delta)
            ]
            proposed = F.normalize(torch.as_tensor(estimator.cluster_centers_).float(), dim=-1)
            after = float((1 - (selected @ torch.cat((old, proposed)).T).amax(1).clamp(-1, 1)).clamp_min(0).mean())
            absolute = baseline - after
            relative = absolute / max(baseline, epsilon)
            score = absolute - penalty * delta
            row.update({"evaluated": True, "cluster_supports": supports.tolist(),
                        "cluster_distinct_trial_supports": cluster_trial_supports,
                        "cluster_distinct_subject_supports": cluster_subject_supports,
                        "mean_cosine_residual_after": after, "absolute_improvement": absolute,
                        "relative_improvement": relative, "penalized_score": score})
            if int(supports.min()) < cluster_gate: reasons.append("minimum_cluster_support_not_met")
            if min(cluster_trial_supports) < trial_gate:
                reasons.append("minimum_cluster_trial_support_not_met")
            if min(cluster_subject_supports) < subject_gate:
                reasons.append("minimum_cluster_subject_support_not_met")
            if relative < improvement_gate: reasons.append("minimum_relative_improvement_not_met")
            if score <= 0: reasons.append("non_positive_penalized_improvement")
            if not reasons:
                row["accepted"] = True
                accepted[delta] = proposed
        candidates.append(row)
    accepted_rows = [row for row in candidates if row["accepted"]]
    if accepted_rows:
        best = max(accepted_rows, key=lambda row: (row["penalized_score"], -row["delta"]))
        delta = int(best["delta"])
        centres = accepted[delta].to(device=result_device, dtype=result_dtype)
        reason = "best_positive_penalized_quantization_improvement"
    else:
        delta, centres = 0, None
        if len(selected) < support_gate: reason = "insufficient_residual_support"
        elif residual_trial_support < trial_gate: reason = "insufficient_residual_trial_support"
        elif residual_subject_support < subject_gate: reason = "insufficient_residual_subject_support"
        elif not math.isfinite(baseline) or baseline <= epsilon: reason = "no_positive_quantization_residual"
        else: reason = "no_candidate_passed_unlabeled_quality_gates"
    audit = {
        "schema": "hhr_residual_adaptive_codebook_decision_v2",
        "policy": "residual_adaptive", "activity_labels_used": False,
        "provenance_used": ["trial_global_id", "subject_id"],
        "feature_count": len(features), "embedding_dim": features.shape[1],
        "distinct_trial_count": int(torch.unique(feature_trial_ids).numel()),
        "distinct_subject_count": int(torch.unique(feature_subject_ids).numel()),
        "codebook_size_before": len(old), "codebook_size_after": len(old) + delta,
        "max_delta": limit, "residual_quantile": quantile,
        "residual_threshold": float(threshold), "residual_support": len(selected),
        "residual_distinct_trial_support": residual_trial_support,
        "residual_distinct_subject_support": residual_subject_support,
        "minimum_residual_support": support_gate, "minimum_cluster_support": cluster_gate,
        "minimum_cluster_trials": trial_gate,
        "minimum_cluster_subjects": subject_gate,
        "minimum_relative_improvement": improvement_gate,
        "complexity_penalty_per_code": penalty,
        "baseline_mean_cosine_residual": baseline,
        "candidate_evaluations": candidates, "selected_delta": delta,
        "selection_reason": reason,
    }
    return CodebookExpansionDecision(centres, delta, audit)


def grouped_memax_losses(
    logits: torch.Tensor, seen_class_count: int, *, temperature: float = 0.10
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if logits.ndim != 2 or not 0 < seen_class_count < logits.shape[1]:
        raise ValueError("seen_class_count must split rank-2 logits into old/new groups.")
    if not logits.is_floating_point() or not torch.isfinite(logits).all():
        raise ValueError("grouped MeMax logits must be finite floating-point values.")
    temperature = float(temperature)
    if not math.isfinite(temperature) or temperature <= 0.0:
        raise ValueError("grouped MeMax temperature must be positive and finite.")
    # Work entirely in log space until the final x*log(x) products.  At the
    # registered low temperature, an old or new group can otherwise underflow
    # to exact zero, causing 0*log(0), NaN gradients, and a false within-group
    # penalty.  logsumexp retains the relative distribution inside even a
    # group whose total probability is vanishingly small.
    working = logits.float() if logits.dtype in {torch.float16, torch.bfloat16} else logits
    log_probabilities = F.log_softmax(working / temperature, dim=1)
    log_average = torch.logsumexp(log_probabilities, dim=0) - math.log(
        int(logits.shape[0])
    )
    log_old = log_average[:seen_class_count]
    log_new = log_average[seen_class_count:]
    log_old_mass = torch.logsumexp(log_old, dim=0)
    log_new_mass = torch.logsumexp(log_new, dim=0)
    group_log_mass = torch.stack((log_old_mass, log_new_mass))
    group_mass = torch.softmax(group_log_mass, dim=0)
    inter = (torch.sum(group_mass * group_log_mass) + math.log(2)).clamp_min(0.0)

    old_log_distribution = log_old - log_old_mass
    new_log_distribution = log_new - log_new_mass
    old_in = (
        torch.sum(old_log_distribution.exp() * old_log_distribution)
        + math.log(len(log_old))
    ).clamp_min(0.0)
    new_in = (
        torch.sum(new_log_distribution.exp() * new_log_distribution)
        + math.log(len(log_new))
    ).clamp_min(0.0)
    return inter, old_in, new_in


def _cat(outputs: Sequence[Mapping[str, torch.Tensor]], key: str) -> torch.Tensor:
    if len(outputs) != 2 or any(key not in item for item in outputs):
        raise ValueError(f"Two trajectory outputs containing {key!r} are required.")
    return torch.cat([item[key] for item in outputs])


def compute_online_trajectory_loss(
    outputs: Sequence[Mapping[str, torch.Tensor]],
    previous_outputs: Sequence[Mapping[str, torch.Tensor]],
    criterion: CrossViewDistillationLoss,
    config: MotionPrimitiveOnlineConfig,
    *, seen_class_count: int, epoch_index: int,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    cfg = config.validated()
    logits = _cat(outputs, "trajectory_logits")
    cluster = criterion(logits, logits.detach(), int(epoch_index))
    inter, old_in, new_in = grouped_memax_losses(
        logits, seen_class_count, temperature=cfg.student_temperature
    )
    cluster = cluster + cfg.grouped_memax_old_new_weight * inter \
        + cfg.grouped_memax_old_in_weight * old_in \
        + cfg.grouped_memax_new_in_weight * new_in
    teacher_logits = _cat(previous_outputs, "trajectory_logits").detach()
    seen, temperature = int(seen_class_count), cfg.student_temperature
    logit_distillation = F.kl_div(
        F.log_softmax(logits[:, :seen] / temperature, dim=-1),
        F.softmax(teacher_logits[:, :seen] / temperature, dim=-1),
        reduction="batchmean",
    ) * temperature**2
    embeddings = F.normalize(_cat(outputs, "trajectory_embedding"), dim=-1)
    teacher_embeddings = F.normalize(
        _cat(previous_outputs, "trajectory_embedding"), dim=-1
    ).detach()
    if embeddings.shape != teacher_embeddings.shape:
        raise ValueError("Current and previous trajectory embeddings must align.")
    feature_distillation = (embeddings - teacher_embeddings).square().sum() / len(embeddings)
    primitive_feature_terms: list[torch.Tensor] = []
    for view_index, (output, previous_output) in enumerate(
        zip(outputs, previous_outputs)
    ):
        primitive = output["primitive_features"]
        previous_primitive = previous_output["primitive_features"].detach()
        token_mask = output["token_mask"].bool()
        previous_mask = previous_output["token_mask"].bool()
        if (
            primitive.shape != previous_primitive.shape
            or token_mask.shape != primitive.shape[:2]
            or not torch.equal(token_mask, previous_mask)
        ):
            raise ValueError(
                f"Current/previous primitive features do not align in view {view_index}."
            )
        primitive_feature_terms.append(
            (
                1.0
                - F.cosine_similarity(
                    primitive[token_mask],
                    previous_primitive[token_mask],
                    dim=-1,
                    eps=1.0e-8,
                )
            ).mean()
        )
    primitive_feature_distillation = torch.stack(primitive_feature_terms).mean()

    current_codebook = outputs[0]["normalized_codebook"]
    previous_codebook = previous_outputs[0]["normalized_codebook"].detach()
    if (
        current_codebook.ndim != 2
        or previous_codebook.ndim != 2
        or current_codebook.shape[1] != previous_codebook.shape[1]
        or current_codebook.shape[0] < previous_codebook.shape[0]
    ):
        raise ValueError("Current/previous codebooks cannot be anchored by old rows.")
    old_codebook_anchor = (
        1.0
        - F.cosine_similarity(
            current_codebook[: previous_codebook.shape[0]],
            previous_codebook,
            dim=-1,
            eps=1.0e-8,
        )
    ).mean()
    view_consistency = (1 - F.cosine_similarity(
        outputs[0]["trajectory_embedding"], outputs[1]["trajectory_embedding"], dim=-1
    )).mean()
    commitment = 0.5 * (outputs[0]["commitment_loss"] + outputs[1]["commitment_loss"])
    codebook = 0.5 * (outputs[0]["codebook_embedding_loss"] + outputs[1]["codebook_embedding_loss"])
    total = cfg.trajectory_cluster_weight * cluster \
        + cfg.trajectory_logit_distillation_weight * logit_distillation \
        + cfg.trajectory_feature_distillation_weight * feature_distillation \
        + cfg.primitive_feature_distillation_weight * primitive_feature_distillation \
        + cfg.old_codebook_anchor_weight * old_codebook_anchor \
        + cfg.trajectory_view_consistency_weight * view_consistency \
        + cfg.vq_commitment_weight * commitment \
        + cfg.vq_codebook_weight * codebook
    return total, {
        "trajectory_cluster": cluster, "memax_old_new": inter,
        "memax_old_in": old_in, "memax_new_in": new_in,
        "old_trajectory_logit_distillation": logit_distillation,
        "old_trajectory_feature_distillation": feature_distillation,
        "old_primitive_feature_distillation": primitive_feature_distillation,
        "old_codebook_row_anchor": old_codebook_anchor,
        "trajectory_view_consistency": view_consistency,
        "vq_commitment": commitment, "vq_codebook": codebook,
    }


def online_training_step(
    model: MotionPrimitiveCGCDModel, previous_model: MotionPrimitiveCGCDModel,
    views: Sequence[Mapping[str, torch.Tensor]], optimizer: torch.optim.Optimizer,
    criterion: CrossViewDistillationLoss, config: MotionPrimitiveOnlineConfig, *,
    seen_class_count: int, epoch_index: int,
) -> OnlineLossResult:
    outputs = forward_two_views(model, views, hard_codebook=False)
    with torch.no_grad():
        previous_outputs = [previous_model(view, hard_codebook=False) for view in views]
    total, components = compute_online_trajectory_loss(
        outputs, previous_outputs, criterion, config,
        seen_class_count=seen_class_count, epoch_index=epoch_index,
    )
    optimizer.zero_grad(set_to_none=True)
    total.backward()
    optimizer.step()
    model.normalize_codebook_()
    detached = total.detach()
    return OnlineLossResult(detached, detached, {key: value.detach() for key, value in components.items()})


@torch.inference_mode()
def audit_primitive_retention(
    current_model: MotionPrimitiveCGCDModel,
    previous_model: MotionPrimitiveCGCDModel,
    loader: DataLoader,
    device: torch.device,
) -> dict[str, Any]:
    """Measure action-primitive retention on deterministic unlabelled trials.

    The loader must have augmentation disabled. Activity labels present in the
    benchmark tuple are deliberately ignored. Token ids are append-only, so an
    exact id comparison is meaningful for every code that existed before the
    current session; assignments to newly appended ids count as disagreements.
    """

    if not isinstance(current_model, MotionPrimitiveCGCDModel) or not isinstance(
        previous_model, MotionPrimitiveCGCDModel
    ):
        raise TypeError("Primitive retention audit requires two motion-primitive models.")
    old_k = int(previous_model.codebook_size)
    current_k = int(current_model.codebook_size)
    if current_k < old_k:
        raise RuntimeError("The current codebook cannot be smaller than its teacher.")
    current_centres = F.normalize(
        current_model.codebook.vectors[:old_k].detach(), dim=-1
    )
    previous_centres = F.normalize(
        previous_model.codebook.vectors.detach().to(current_centres), dim=-1
    )
    if current_centres.shape != previous_centres.shape:
        raise RuntimeError("Old codebook rows are not shape-compatible for retention audit.")
    row_cosine = F.cosine_similarity(current_centres, previous_centres, dim=-1)

    current_training = current_model.training
    previous_training = previous_model.training
    current_model.eval()
    previous_model.eval()
    local_cosines: list[torch.Tensor] = []
    exact_count = 0
    new_code_count = 0
    valid_count = 0
    trial_count = 0
    try:
        for batch in loader:
            inputs = batch[0]
            if isinstance(inputs, list):
                raise RuntimeError(
                    "Primitive retention audit requires augmentation-disabled trials."
                )
            prepared = move_trial_batch_to_device(inputs, device)
            current_output = current_model(prepared, hard_codebook=True)
            previous_output = previous_model(prepared, hard_codebook=True)
            current_mask = current_output["token_mask"].bool()
            previous_mask = previous_output["token_mask"].bool()
            if not torch.equal(current_mask, previous_mask):
                raise RuntimeError("Current/previous token masks differ in retention audit.")
            current_features = current_output["primitive_features"]
            previous_features = previous_output["primitive_features"]
            if current_features.shape != previous_features.shape:
                raise RuntimeError("Current/previous local feature shapes differ.")
            local_cosines.append(
                F.cosine_similarity(
                    current_features[current_mask],
                    previous_features[current_mask],
                    dim=-1,
                    eps=1.0e-8,
                ).detach().cpu()
            )
            current_tokens = current_output["hard_tokens"][current_mask]
            previous_tokens = previous_output["hard_tokens"][current_mask]
            if torch.any(previous_tokens < 0) or torch.any(previous_tokens >= old_k):
                raise RuntimeError("Teacher emitted a token outside its old codebook.")
            exact_count += int((current_tokens == previous_tokens).sum().item())
            new_code_count += int((current_tokens >= old_k).sum().item())
            valid_count += int(current_tokens.numel())
            trial_count += int(current_mask.shape[0])
    finally:
        current_model.train(current_training)
        previous_model.train(previous_training)
    if valid_count < 1 or not local_cosines:
        raise RuntimeError("Primitive retention audit received no valid local windows.")
    local = torch.cat(local_cosines).float()
    if len(local) != valid_count or not torch.isfinite(local).all():
        raise RuntimeError("Primitive retention audit produced invalid local similarities.")
    return {
        "schema": "hhr_motion_primitive_retention_audit_v1",
        "activity_labels_used": False,
        "augmentation_disabled": True,
        "comparison_scope": "current_session_unlabelled_train_trials",
        "trial_count": trial_count,
        "valid_window_count": valid_count,
        "old_codebook_size": old_k,
        "current_codebook_size": current_k,
        "new_code_count": current_k - old_k,
        "old_codebook_row_cosine_mean": float(row_cosine.mean().cpu()),
        "old_codebook_row_cosine_minimum": float(row_cosine.min().cpu()),
        "old_codebook_row_cosine_drift_mean": float((1.0 - row_cosine).mean().cpu()),
        "old_codebook_row_cosine_drift_maximum": float((1.0 - row_cosine).max().cpu()),
        "old_codebook_row_cosines": [float(value) for value in row_cosine.cpu().tolist()],
        "local_feature_cosine_mean": float(local.mean()),
        "local_feature_cosine_p05": float(torch.quantile(local, 0.05)),
        "local_feature_cosine_minimum": float(local.min()),
        "old_token_exact_agreement": exact_count / valid_count,
        "old_token_exact_agreement_count": exact_count,
        "new_code_usage_fraction": new_code_count / valid_count,
        "new_code_usage_count": new_code_count,
    }


def global_hungarian_alignment(
    targets: np.ndarray, predictions: np.ndarray, class_count: int
) -> tuple[np.ndarray, list[list[int]]]:
    truth, raw, count = np.asarray(targets).reshape(-1), np.asarray(predictions).reshape(-1), int(class_count)
    if truth.shape != raw.shape or not len(truth):
        raise ValueError("targets/predictions must be equal non-empty vectors.")
    contingency = np.zeros((count, count), dtype=np.int64)
    np.add.at(contingency, (raw, truth), 1)
    rows, columns = linear_sum_assignment(contingency.max() - contingency)
    mapping = {int(row): int(column) for row, column in zip(rows, columns)}
    return np.asarray([mapping[int(value)] for value in raw]), [[key, mapping[key]] for key in sorted(mapping)]


def constrained_old_fixed_alignment(
    targets: np.ndarray, predictions: np.ndarray, class_count: int, old_class_count: int
) -> tuple[np.ndarray, list[list[int]]]:
    truth, raw = np.asarray(targets).reshape(-1), np.asarray(predictions).reshape(-1)
    count, old_count = int(class_count), int(old_class_count)
    if truth.shape != raw.shape or not len(truth) or not 1 <= old_count < count:
        raise ValueError("Invalid old-fixed alignment inputs.")
    novel_count = count - old_count
    contingency = np.zeros((novel_count, novel_count), dtype=np.int64)
    mask = (raw >= old_count) & (truth >= old_count)
    if np.any(mask):
        np.add.at(contingency, (raw[mask] - old_count, truth[mask] - old_count), 1)
    rows, columns = linear_sum_assignment(contingency.max(initial=0) - contingency)
    mapping = {index: index for index in range(old_count)}
    mapping.update({int(row + old_count): int(column + old_count) for row, column in zip(rows, columns)})
    return np.asarray([mapping[int(value)] for value in raw]), [[key, mapping[key]] for key in sorted(mapping)]


def _harmonic(left: float, right: float) -> float:
    return 0.0 if left + right <= 0 else 2 * left * right / (left + right)


def _metrics(
    targets: np.ndarray, predictions: np.ndarray, *, class_count: int,
    old_class_count: int, seen_class_count_before_session: int,
    assignment: Sequence[Sequence[int]], prediction_alignment: str,
    alignment_uses_test_labels: bool,
) -> dict[str, Any]:
    truth, pred = np.asarray(targets).reshape(-1), np.asarray(predictions).reshape(-1)
    old = truth < old_class_count
    seen = truth < seen_class_count_before_session
    accuracy = lambda mask: float(np.mean(pred[mask] == truth[mask])) if np.any(mask) else 0.0
    old_accuracy, new_accuracy = accuracy(old), accuracy(~old)
    return {
        "all_accuracy": float(np.mean(pred == truth)), "old_accuracy": old_accuracy,
        "new_accuracy": new_accuracy, "h_score": _harmonic(old_accuracy, new_accuracy),
        "seen_accuracy": accuracy(seen), "unseen_accuracy": accuracy(~seen),
        "macro_f1": float(f1_score(truth, pred, labels=np.arange(class_count),
                                    average="macro", zero_division=0)),
        "confusion_matrix": confusion_matrix(truth, pred, labels=np.arange(class_count)).tolist(),
        "hungarian_assignment_pred_to_true": [list(map(int, pair)) for pair in assignment],
        "assignment_pred_to_true": [list(map(int, pair)) for pair in assignment],
        "prediction_alignment": prediction_alignment,
        "alignment_uses_test_labels": alignment_uses_test_labels,
        "sample_count": len(truth),
    }


def aligned_online_metrics(targets, raw_predictions, *, class_count, old_class_count,
                           seen_class_count_before_session):
    pred, assignment = global_hungarian_alignment(targets, raw_predictions, class_count)
    return _metrics(targets, pred, class_count=class_count, old_class_count=old_class_count,
                    seen_class_count_before_session=seen_class_count_before_session,
                    assignment=assignment, prediction_alignment="standard_global_hungarian",
                    alignment_uses_test_labels=True)


def constrained_old_fixed_online_metrics(targets, raw_predictions, *, class_count,
                                         old_class_count, seen_class_count_before_session):
    pred, assignment = constrained_old_fixed_alignment(targets, raw_predictions, class_count, old_class_count)
    return _metrics(targets, pred, class_count=class_count, old_class_count=old_class_count,
                    seen_class_count_before_session=seen_class_count_before_session,
                    assignment=assignment, prediction_alignment="constrained_old_fixed",
                    alignment_uses_test_labels=True)


def direct_head_online_metrics(targets, raw_predictions, *, class_count,
                               old_class_count, seen_class_count_before_session):
    assignment = [[index, index] for index in range(class_count)]
    return _metrics(targets, raw_predictions, class_count=class_count,
                    old_class_count=old_class_count,
                    seen_class_count_before_session=seen_class_count_before_session,
                    assignment=assignment, prediction_alignment="direct_head",
                    alignment_uses_test_labels=False)


def stratified_online_metrics(targets, raw_predictions, *, class_count,
                              old_class_count, seen_class_count_before_session):
    common = dict(class_count=class_count, old_class_count=old_class_count,
                  seen_class_count_before_session=seen_class_count_before_session)
    return {
        "standard_global_hungarian": aligned_online_metrics(targets, raw_predictions, **common),
        "constrained_old_fixed": constrained_old_fixed_online_metrics(targets, raw_predictions, **common),
        "direct_head": direct_head_online_metrics(targets, raw_predictions, **common),
    }


@torch.inference_mode()
def evaluate_online(
    model: MotionPrimitiveCGCDModel, loader: DataLoader, device: torch.device,
    config: MotionPrimitiveOnlineConfig, *, seen_class_count_before_session: int,
) -> dict[str, Any]:
    cfg = config.validated()
    model.eval()
    targets, predictions = [], []
    for batch in loader:
        inputs, labels = batch[0], batch[1]
        if isinstance(inputs, list):
            raise RuntimeError("Online evaluation requires one deterministic trial view.")
        output = model(move_trial_batch_to_device(inputs, device), hard_codebook=True)
        predictions.append(output["trajectory_logits"].argmax(1).cpu().numpy())
        targets.append(labels.numpy())
    truth, raw = np.concatenate(targets), np.concatenate(predictions)
    layers = stratified_online_metrics(
        truth, raw, class_count=model.class_count, old_class_count=cfg.old_class_count,
        seen_class_count_before_session=seen_class_count_before_session,
    )
    primary = dict(layers["constrained_old_fixed"])
    primary["evaluation_layers"] = layers
    return {
        "sample_unit": "complete_trial_as_ordered_primitive_trajectory",
        "primary_head": "trajectory", "primary_evaluation_layer": "constrained_old_fixed",
        "evaluation_layers": ["standard_global_hungarian", "constrained_old_fixed", "direct_head"],
        "trajectory": primary,
    }


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, path)


def run_online_cgcd(
    offline_model: MotionPrimitiveCGCDModel, datasets: Mapping[str, Any],
    novel_class_order: Sequence[int], old_classes: Sequence[int],
    config: MotionPrimitiveOnlineConfig, *, device: torch.device, seed: int,
    output_dir: Optional[Path] = None, logger: Optional[logging.Logger] = None,
    codebook_expansion_policy: Optional[Callable[[int, MotionPrimitiveCGCDModel, DataLoader, torch.device], CodebookExpansionDecision]] = None,
) -> tuple[MotionPrimitiveCGCDModel, list[dict[str, Any]]]:
    cfg = config.validated()
    if not isinstance(offline_model, MotionPrimitiveCGCDModel) or offline_model.class_count != cfg.old_class_count:
        raise TypeError("Online input must be the offline old-class motion-primitive model.")
    audit = audit_uschad_online_protocol(datasets, novel_class_order, old_classes, cfg)
    session_datasets = build_online_session_datasets(datasets, cfg)
    destination = Path(output_dir).resolve() if output_dir is not None else None
    log = logger or logging.getLogger("hhr.motion_primitive_online.null")
    if destination:
        destination.mkdir(parents=True, exist_ok=True)
        _write_json(destination / "online_protocol.json", {"config": cfg.audit_dict(), "sessions": audit})
    current = clone_motion_primitive_model(offline_model).to(device)
    results = []
    for index, target_count in enumerate(cfg.class_counts):
        seen = cfg.old_class_count + index * cfg.novel_classes_per_session
        previous = current.to(device).eval()
        current = expand_online_model(previous, target_count, cfg).to(device)
        initialization_dataset = _disable_augmentation(session_datasets[index])
        initialization_loader = make_online_loader(
            initialization_dataset, cfg, seed=seed + 10002 + index * 10, training=False
        )
        before, after = current.codebook_size, current.codebook_size
        if codebook_expansion_policy is None:
            decision = CodebookExpansionDecision(None, 0, {
                "schema": "hhr_codebook_expansion_decision_v1", "policy": "none",
                "activity_labels_used": False, "selected_delta": 0,
                "selection_reason": "codebook_expansion_disabled",
                "codebook_size_before": before, "codebook_size_after": before,
            })
        else:
            decision = codebook_expansion_policy(index + 1, current, initialization_loader, device)
            if decision.audit.get("activity_labels_used") is not False:
                raise RuntimeError("Codebook expansion must attest that labels were not used.")
            if decision.centres is not None:
                observed_before, after = expand_motion_codebook_(current, decision.centres)
                if observed_before != before:
                    raise RuntimeError("Codebook expansion observed a stale K.")
        if after != before + decision.selected_delta:
            raise RuntimeError("Codebook decision delta does not match the live K transition.")
        decision_audit = dict(decision.audit)
        decision_audit.update({"session": index + 1, "codebook_size_before": before,
                               "codebook_size_after": after, "selected_delta": after - before,
                               "expanded": after > before})
        if cfg.initialize_new_trajectory_head_with_kmeans:
            centres = select_kmeans_new_trajectory_centres(
                current, previous, initialization_loader, target_count, cfg, device
            )
            initialise_new_trajectory_rows_(current, seen, centres)
            initialization = "low_old_confidence_kmeans_in_trajectory_embedding_space"
        else:
            initialization = "random_new_trajectory_rows"
        train_loader = make_online_loader(session_datasets[index], cfg,
                                          seed=seed + 10000 + index * 10, training=True)
        test_loader = make_online_loader(_disable_augmentation(datasets["online_test_dataset_list"][index]),
                                         cfg, seed=seed + 10001 + index * 10, training=False)
        optimizer = build_online_sgd(current, cfg)
        scheduler = build_online_cosine_scheduler(optimizer, cfg)
        criterion = CrossViewDistillationLoss(
            cfg.warmup_teacher_epochs,
            cfg.epochs_per_session,
            cfg.n_views,
            cfg.warmup_teacher_temperature,
            cfg.teacher_temperature,
            cfg.student_temperature,
        ).to(device)
        steps = 0
        for epoch in range(cfg.epochs_per_session):
            current.train()
            for batch in train_loader:
                views = [move_trial_batch_to_device(view, device)
                         for view in unlabelled_views_from_batch(batch)]
                loss = online_training_step(current, previous, views, optimizer, criterion, cfg,
                                            seen_class_count=seen, epoch_index=epoch)
                steps += 1
            scheduler.step()
            log.info("online session=%d epoch=%d/%d trajectory_loss=%.6f K=%d",
                     index + 1, epoch + 1, cfg.epochs_per_session,
                     float(loss.trajectory_total.cpu()), current.codebook_size)
        retention_audit = audit_primitive_retention(
            current, previous, initialization_loader, device
        )
        log.info(
            "online session=%d retention old_code_cos=%.6f local_cos=%.6f "
            "token_agreement=%.6f new_code_usage=%.6f",
            index + 1,
            retention_audit["old_codebook_row_cosine_mean"],
            retention_audit["local_feature_cosine_mean"],
            retention_audit["old_token_exact_agreement"],
            retention_audit["new_code_usage_fraction"],
        )
        metrics = evaluate_online(current, test_loader, device, cfg,
                                  seen_class_count_before_session=seen)
        record = {
            "session": index + 1, "seen_class_count_before": seen,
            "class_count_after_expansion": target_count, "optimizer_steps": steps,
            "backward_calls": steps, "codebook_size_before": before,
            "codebook_size_after": after, "codebook_expanded": after > before,
            "codebook_expansion_decision": decision_audit,
            "new_class_initialization": {"activity_labels_used": False,
                                         "trajectory_head": initialization},
            "retention_audit": retention_audit,
            "metrics": metrics, "protocol": audit[index],
        }
        results.append(record)
        if destination:
            payload = {
                "schema": ONLINE_CHECKPOINT_SCHEMA,
                "architecture": current.config.audit_dict(),
                "model": current.state_dict(), "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(), "epoch": cfg.epochs_per_session,
                "session": index + 1, "class_count": target_count,
                "codebook_size": after, "codebook_expansion_decision": decision_audit,
                "new_class_initialization": record["new_class_initialization"],
                "retention_audit": retention_audit,
                "config": cfg.audit_dict(), "metrics": metrics,
            }
            torch.save(payload, destination / f"checkpoint_session_{index + 1}.pt")
            _write_json(destination / f"metrics_session_{index + 1}.json", record)
    if destination:
        _write_json(
            destination / "online_summary.json",
            {
                "schema": ONLINE_SUMMARY_SCHEMA,
                "config": cfg.audit_dict(),
                "sessions": results,
            },
        )
    return current, results


__all__ = [
    "ONLINE_CHECKPOINT_SCHEMA", "ONLINE_CONFIG_SCHEMA", "ONLINE_SUMMARY_SCHEMA",
    "CodebookExpansionDecision", "HARUnlabelledPairDataset",
    "LocalPrimitiveFeatureSet",
    "MotionPrimitiveOnlineConfig", "OnlineLossResult", "aligned_online_metrics",
    "audit_primitive_retention", "audit_uschad_online_protocol",
    "build_online_session_datasets", "build_online_sgd",
    "clone_motion_primitive_model", "compute_online_trajectory_loss",
    "constrained_old_fixed_alignment", "constrained_old_fixed_online_metrics",
    "direct_head_online_metrics", "evaluate_online", "expand_motion_codebook_",
    "expand_online_model", "extract_local_primitive_features",
    "extract_trajectory_embeddings", "global_hungarian_alignment",
    "grouped_memax_losses", "initialise_new_trajectory_rows_", "make_online_loader",
    "online_training_step", "run_online_cgcd", "select_kmeans_codebook_centres",
    "select_kmeans_new_trajectory_centres", "select_residual_adaptive_codebook_centres",
    "stratified_online_metrics", "unlabelled_views_from_batch", "uschad_trial_collate",
]
