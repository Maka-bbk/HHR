"""Train one checkpoint-free motion-primitive trajectory model on one fold.

This is the executable J0 screening experiment.  It uses one randomly
initialised model, one optimizer, and one continuous epoch lineage.  The only
pre-optimisation operation is deterministic raw-kinematic boundary
calibration on train-subject old-class trials.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import sys
from collections import defaultdict
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable, Mapping, Optional, Sequence

import numpy as np
import torch
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.uschad_trials import (  # noqa: E402
    ACTIVITY_NAMES,
    IGNORE_INDEX,
    USCHADFoldDatasets,
    build_uschad_fold_datasets,
    pad_trial_batch,
)
from experiments.motion_primitive.one_stage_boundaries import (  # noqa: E402
    build_one_stage_raw_boundaries,
)
from experiments.motion_primitive.one_stage_evaluation import (  # noqa: E402
    codebook_diagnostics,
    score_cgcd_clusters,
    semi_supervised_kmeans,
)
from experiments.motion_primitive.one_stage_losses import (  # noqa: E402
    J0_TRAJECTORY,
    J0_UNSUPERVISED,
    OneStageLossConfig,
    OneStageLossInputs,
    compose_one_stage_loss,
)
from experiments.motion_primitive.one_stage_visualization import (  # noqa: E402
    save_class_codebook_heatmap,
    save_confusion_heatmap,
    save_representative_trajectory_panel,
)
from models.motion_trajectory import (  # noqa: E402
    MotionTrajectoryConfig,
    MotionTrajectoryModel,
    hard_token_runs,
)


SCHEMA_VERSION = "one_stage_motion_trajectory_run_v1"


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, set | frozenset | tuple):
        return list(value)
    raise TypeError(f"Cannot serialize {type(value).__name__}.")


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        default=_json_default,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _sha256_document(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(
            value,
            default=_json_default,
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        ),
        encoding="utf-8",
    )
    temporary.replace(path)


def _append_jsonl(path: Path, value: Any) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(_canonical_json(value) + "\n")


def _checkpoint_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_torch_save(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def set_reproducible_seed(seed: int, deterministic: bool) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
        if torch.backends.cudnn.is_available():
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True


def resolve_device(requested: str) -> torch.device:
    value = str(requested).lower()
    if value == "auto":
        value = "cuda" if torch.cuda.is_available() else "cpu"
    if value.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false.")
    return torch.device(value)


def _full_frame_starts(length: int, frame_size: int, frame_stride: int) -> np.ndarray:
    if int(length) < int(frame_size):
        raise ValueError(
            f"Raw-boundary calibration requires length >= frame_size; got {length} < {frame_size}."
        )
    return np.arange(
        0,
        int(length) - int(frame_size) + 1,
        int(frame_stride),
        dtype=np.int64,
    )


def _boundary_records(dataset: Any, frame_size: int, frame_stride: int) -> list[Any]:
    records = []
    for index in range(len(dataset)):
        item = dataset[index]
        records.append(
            SimpleNamespace(
                trial_id=int(item["trial_id"]),
                subject_id=int(item["subject_id"]),
                label=int(item["class_index"]),
                raw_trial=item["physical_trial"],
                starts=_full_frame_starts(
                    int(item["length"]), int(frame_size), int(frame_stride)
                ),
            )
        )
    return records


def prepare_raw_boundary_targets(
    datasets: USCHADFoldDatasets,
    *,
    frame_size: int,
    frame_stride: int,
) -> tuple[dict[int, tuple[np.ndarray, np.ndarray]], dict[str, Any]]:
    train_records = _boundary_records(datasets.train, frame_size, frame_stride)
    validation_records = _boundary_records(
        datasets.validation, frame_size, frame_stride
    )
    audit = build_one_stage_raw_boundaries(
        train_records,
        validation_records,
        window_size_samples=int(frame_size),
        old_class_count=len(datasets.old_activity_ids),
        require_subject_disjoint=True,
    )
    targets: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for record in train_records + validation_records:
        stable = record.stable_mask.detach().cpu().numpy().astype(bool, copy=True)
        change = record.change_mask.detach().cpu().numpy().astype(bool, copy=True)
        if stable.shape != change.shape or np.any(stable & change):
            raise RuntimeError("Raw boundary targets are malformed.")
        targets[int(record.trial_id)] = (stable, change)
    audit["model_alignment"] = {
        "model_frame_policy": "include one right-padded partial tail frame when needed",
        "anchor_frame_policy": "full raw frames only",
        "partial_tail_boundary_policy": "uncertain_and_excluded_from_boundary_supervision",
        "frame_size_samples": int(frame_size),
        "frame_stride_samples": int(frame_stride),
    }
    return targets, audit


def _batch_raw_boundary_masks(
    trial_ids: torch.Tensor,
    valid_boundary_mask: torch.Tensor,
    targets: Mapping[int, tuple[np.ndarray, np.ndarray]],
) -> tuple[torch.Tensor, torch.Tensor]:
    stable = torch.zeros_like(valid_boundary_mask, dtype=torch.bool)
    change = torch.zeros_like(valid_boundary_mask, dtype=torch.bool)
    for row, trial_id in enumerate(trial_ids.detach().cpu().tolist()):
        if int(trial_id) not in targets:
            raise KeyError(f"No raw-boundary target for trial id {trial_id}.")
        stable_np, change_np = targets[int(trial_id)]
        valid_count = int(valid_boundary_mask[row].sum().item())
        if len(stable_np) > valid_count:
            raise RuntimeError(
                f"Raw anchor count exceeds model boundary count for trial {trial_id}."
            )
        if len(stable_np):
            stable[row, : len(stable_np)] = torch.as_tensor(
                stable_np, device=stable.device, dtype=torch.bool
            )
            change[row, : len(change_np)] = torch.as_tensor(
                change_np, device=change.device, dtype=torch.bool
            )
    if torch.any((stable | change) & ~valid_boundary_mask) or torch.any(stable & change):
        raise RuntimeError("Raw boundary masks selected an invalid/overlapping position.")
    return stable, change


def _deterministic_trajectory_mask(
    lengths: torch.Tensor,
    trial_ids: torch.Tensor,
    config: MotionTrajectoryConfig,
) -> torch.Tensor:
    frame_lengths = 1 + torch.div(
        torch.clamp(lengths - int(config.frame_size), min=0)
        + int(config.frame_stride)
        - 1,
        int(config.frame_stride),
        rounding_mode="floor",
    )
    maximum = int(frame_lengths.max().item())
    positions = torch.arange(maximum, device=lengths.device).unsqueeze(0)
    valid = positions < frame_lengths.unsqueeze(1)
    period = max(2, int(round(1.0 / max(float(config.trajectory_mask_ratio), 1e-6))))
    offsets = torch.remainder(trial_ids.to(lengths.device), period).unsqueeze(1)
    selected = (torch.remainder(positions + offsets, period) == 0) & valid
    for row in range(len(lengths)):
        if not bool(selected[row].any().item()):
            selected[row, 0] = True
    return selected


def _scheduled_loss_config(
    base: OneStageLossConfig,
    epoch: int,
    *,
    warmup_epochs: int,
) -> OneStageLossConfig:
    # This is a continuous curriculum: no model/optimizer reset and no second
    # checkpoint lineage.  A small nonzero floor keeps J0-U trajectory gradients
    # alive while early pseudo-token identities are still moving.
    ramp = max(0.05, min(1.0, float(epoch) / max(1, int(warmup_epochs))))
    values = {
        "masked_token_prediction_weight": base.masked_token_prediction_weight * ramp,
        "masked_state_reconstruction_weight": base.masked_state_reconstruction_weight * ramp,
        "stable_next_content_weight": base.stable_next_content_weight * ramp,
    }
    if base.profile == J0_TRAJECTORY:
        values["trajectory_ce_weight"] = base.trajectory_ce_weight * ramp
    return replace(base, **values).validated()


def _temperature(epoch: int, epochs: int, start: float, end: float) -> float:
    if int(epochs) <= 1:
        return float(end)
    progress = (int(epoch) - 1) / float(int(epochs) - 1)
    return float(start * ((end / start) ** progress))


def _to_device_inputs(batch: Mapping[str, Any], device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # Dataset/collate intentionally expose channel-first [B,C,T]; the model's
    # public contract is [B,T,C].
    model_trials = batch["model_trials"].to(device, non_blocking=True).transpose(1, 2)
    physical_trials = batch["physical_trials"].to(device, non_blocking=True).transpose(1, 2)
    lengths = batch["lengths"].to(device, non_blocking=True)
    return model_trials.contiguous(), physical_trials.contiguous(), lengths


def _loss_inputs(
    outputs: Mapping[str, torch.Tensor],
    batch: Mapping[str, Any],
    raw_targets: Mapping[int, tuple[np.ndarray, np.ndarray]],
    profile: str,
    device: torch.device,
) -> OneStageLossInputs:
    valid_boundary = outputs["boundary_pair_mask"]
    stable, change = _batch_raw_boundary_masks(
        batch["trial_ids"], valid_boundary, raw_targets
    )
    trajectory_mask = outputs["context_token_loss_mask"]
    pseudo_targets = torch.full(
        trajectory_mask.shape,
        int(IGNORE_INDEX),
        dtype=torch.long,
        device=device,
    )
    pseudo_targets[trajectory_mask] = outputs["context_token_targets"][
        trajectory_mask
    ].long()
    trajectory_kwargs: dict[str, Any] = {}
    if profile == J0_TRAJECTORY:
        trajectory_kwargs = {
            "trajectory_logits": outputs["trajectory_logits"],
            "trajectory_labels": batch["supervision_targets"].to(device),
            "labelled_old_trial_mask": batch["trajectory_label_mask"].to(device),
        }
    return OneStageLossInputs(
        encoded_states=outputs["encoded_states"],
        quantized_states=outputs["quantized_states"],
        reconstructed_content=outputs["reconstructed_content"],
        content_targets=outputs["content_targets"],
        next_content_predictions=outputs["next_content_predictions"],
        next_content_targets=outputs["next_content_targets"],
        assignment_logits=outputs["assignment_logits"],
        valid_window_mask=outputs["token_mask"],
        boundary_logits=outputs["boundary_logits"],
        valid_boundary_mask=valid_boundary,
        raw_stable_mask=stable,
        raw_change_mask=change,
        reconstructed_states=outputs["codebook_state_reconstruction"],
        state_targets=outputs["codebook_state_target"],
        masked_token_logits=outputs["context_token_logits"],
        pseudo_token_targets=pseudo_targets,
        masked_state_predictions=outputs["context_state_reconstruction"],
        masked_trajectory_mask=outputs["context_state_loss_mask"],
        state_target_source="raw_kinematic_state_descriptor",
        **trajectory_kwargs,
    )


def _mean_dict(sums: Mapping[str, float], denominator: int) -> dict[str, float]:
    return {key: float(value / max(1, denominator)) for key, value in sums.items()}


def train_epoch(
    model: MotionTrajectoryModel,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_config: OneStageLossConfig,
    raw_targets: Mapping[int, tuple[np.ndarray, np.ndarray]],
    device: torch.device,
    *,
    temperature: float,
    gradient_clip: float,
) -> dict[str, Any]:
    model.train()
    sums: dict[str, float] = defaultdict(float)
    samples = 0
    labelled = 0
    for batch in loader:
        model_trials, physical_trials, lengths = _to_device_inputs(batch, device)
        outputs = model(
            model_trials,
            lengths,
            state_trials=physical_trials,
            hard_codebook=False,
            codebook_temperature=float(temperature),
            apply_random_trajectory_mask=True,
        )
        result = compose_one_stage_loss(
            _loss_inputs(outputs, batch, raw_targets, loss_config.profile, device),
            loss_config,
        )
        optimizer.zero_grad(set_to_none=True)
        result.total.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), float(gradient_clip)
        )
        if not bool(torch.isfinite(torch.as_tensor(gradient_norm)).item()):
            raise RuntimeError("Non-finite gradient norm encountered.")
        optimizer.step()
        batch_size = int(len(lengths))
        samples += batch_size
        labelled += int(result.labelled_old_trial_count)
        sums["loss"] += float(result.total.detach()) * batch_size
        sums["gradient_norm"] += float(gradient_norm) * batch_size
        for name, value in result.components.items():
            sums[f"loss_{name}"] += float(value.detach()) * batch_size
        sums["effective_code_count"] += float(
            result.metrics["effective_code_count"]
        ) * batch_size
    return {
        **_mean_dict(sums, samples),
        "trial_count": int(samples),
        "labelled_trial_count": int(labelled),
        "temperature": float(temperature),
    }


@torch.no_grad()
def validate_epoch(
    model: MotionTrajectoryModel,
    loader: DataLoader,
    loss_config: OneStageLossConfig,
    raw_targets: Mapping[int, tuple[np.ndarray, np.ndarray]],
    device: torch.device,
) -> dict[str, Any]:
    model.eval()
    sums: dict[str, float] = defaultdict(float)
    samples = 0
    true_labels: list[int] = []
    predicted_labels: list[int] = []
    for batch in loader:
        model_trials, physical_trials, lengths = _to_device_inputs(batch, device)
        mask = _deterministic_trajectory_mask(
            lengths,
            batch["trial_ids"].to(device),
            model.config,
        )
        outputs = model(
            model_trials,
            lengths,
            state_trials=physical_trials,
            hard_codebook=True,
            trajectory_mask=mask,
            apply_random_trajectory_mask=False,
        )
        result = compose_one_stage_loss(
            _loss_inputs(outputs, batch, raw_targets, loss_config.profile, device),
            loss_config,
        )
        batch_size = int(len(lengths))
        samples += batch_size
        sums["loss"] += float(result.total.detach()) * batch_size
        for name, value in result.components.items():
            sums[f"loss_{name}"] += float(value.detach()) * batch_size
        sums["effective_code_count"] += float(
            result.metrics["effective_code_count"]
        ) * batch_size
        if loss_config.profile == J0_TRAJECTORY:
            visible = batch["trajectory_label_mask"].bool()
            true_labels.extend(batch["supervision_targets"][visible].tolist())
            predicted_labels.extend(
                outputs["trajectory_logits"].argmax(dim=1).cpu()[visible].tolist()
            )
    metrics: dict[str, Any] = {
        **_mean_dict(sums, samples),
        "trial_count": int(samples),
    }
    if loss_config.profile == J0_TRAJECTORY:
        metrics["trajectory_macro_f1"] = float(
            f1_score(
                true_labels,
                predicted_labels,
                labels=np.arange(model.config.num_classes),
                average="macro",
                zero_division=0,
            )
        )
        metrics["trajectory_accuracy"] = float(
            np.mean(np.asarray(true_labels) == np.asarray(predicted_labels))
        )
    return metrics


def _selection_key(profile: str, validation: Mapping[str, Any]) -> tuple[float, float]:
    if profile == J0_TRAJECTORY:
        return float(validation["trajectory_macro_f1"]), -float(validation["loss"])
    return -float(validation["loss"]), float(validation["effective_code_count"])


def _frame_partition_boundaries(
    length: int, frame_count: int, frame_size: int, frame_stride: int
) -> np.ndarray:
    starts = np.arange(frame_count, dtype=np.float64) * int(frame_stride)
    centres = np.minimum(starts + 0.5 * (int(frame_size) - 1), int(length) - 1)
    inner = np.floor(0.5 * (centres[:-1] + centres[1:]) + 0.5).astype(np.int64)
    return np.concatenate(
        [np.asarray([0], dtype=np.int64), inner, np.asarray([length], dtype=np.int64)]
    )


def _runs_with_sample_ranges(
    export: Mapping[str, Any], length: int, frame_size: int, frame_stride: int
) -> list[dict[str, Any]]:
    boundaries = _frame_partition_boundaries(
        int(length), int(export["token_count"]), int(frame_size), int(frame_stride)
    )
    result = []
    for run in export["runs"]:
        start = int(run["start_token_index"])
        end = int(run["end_token_index_exclusive"])
        enriched = dict(run)
        enriched.update(
            {
                "start_sample": int(boundaries[start]),
                "end_sample_exclusive": int(boundaries[end]),
                "duration_samples": int(boundaries[end] - boundaries[start]),
                "duration_seconds": float(
                    (boundaries[end] - boundaries[start]) / 100.0
                ),
            }
        )
        result.append(enriched)
    return result


def _run_block_permutation(
    outputs: Mapping[str, torch.Tensor],
    trial_ids: torch.Tensor,
    seed: int,
) -> torch.Tensor:
    mask = outputs["token_mask"]
    tokens = outputs["hard_tokens"].detach()
    boundaries = outputs["boundary_probabilities"].detach() >= 0.5
    batch_size, maximum = mask.shape
    permutation = torch.arange(maximum, device=mask.device).expand(batch_size, -1).clone()
    for row, trial_id in enumerate(trial_ids.tolist()):
        length = int(mask[row].sum().item())
        starts = [0]
        for position in range(1, length):
            if (
                int(tokens[row, position]) != int(tokens[row, position - 1])
                or bool(boundaries[row, position - 1])
            ):
                starts.append(position)
        ends = starts[1:] + [length]
        if len(starts) <= 1:
            continue
        rng = np.random.default_rng(
            np.random.SeedSequence([int(seed) & 0xFFFFFFFF, int(trial_id)])
        )
        order = rng.permutation(len(starts)).tolist()
        indices = [
            index
            for block in order
            for index in range(starts[block], ends[block])
        ]
        permutation[row, :length] = torch.as_tensor(indices, device=mask.device)
    return permutation


@torch.no_grad()
def encode_dataset(
    model: MotionTrajectoryModel,
    dataset: Any,
    device: torch.device,
    *,
    batch_size: int,
    num_workers: int,
    shuffle_seed: Optional[int] = None,
    strict_relation_recompute: bool = False,
    export_trajectories: bool = True,
) -> dict[str, Any]:
    loader = DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=False,
        num_workers=int(num_workers),
        collate_fn=pad_trial_batch,
        pin_memory=device.type == "cuda",
    )
    model.eval()
    embeddings: list[np.ndarray] = []
    logits: list[np.ndarray] = []
    targets: list[int] = []
    subjects: list[int] = []
    trials: list[int] = []
    trial_ids: list[int] = []
    token_sequences: list[list[int]] = []
    records: list[dict[str, Any]] = []
    expanded_tokens: list[int] = []
    expanded_subjects: list[int] = []
    for batch in loader:
        model_trials, physical_trials, lengths = _to_device_inputs(batch, device)
        outputs = model(
            model_trials,
            lengths,
            state_trials=physical_trials,
            hard_codebook=True,
            apply_random_trajectory_mask=False,
        )
        selected_embedding = outputs["trajectory_embedding"]
        selected_logits = outputs["trajectory_logits"]
        if shuffle_seed is not None or strict_relation_recompute:
            if shuffle_seed is None:
                maximum = int(outputs["token_mask"].shape[1])
                permutation = torch.arange(
                    maximum, device=device
                ).expand(len(lengths), -1).clone()
            else:
                permutation = _run_block_permutation(
                    outputs, batch["trial_ids"], int(shuffle_seed)
                )
            shuffled = model.encode_trajectory(
                outputs["trajectory_assignments"],
                outputs["token_mask"],
                outputs["state_features"],
                outputs["boundary_probabilities"],
                outputs["transition_probabilities"],
                outputs["duration_proxy"],
                apply_random_mask=False,
                token_permutation=permutation,
                permutation_relations="recompute",
            )
            selected_embedding = shuffled["trajectory_embedding"]
            selected_logits = shuffled["trajectory_logits"]
        embeddings.append(selected_embedding.detach().cpu().numpy())
        logits.append(selected_logits.detach().cpu().numpy())
        targets.extend(batch["class_indices"].tolist())
        subjects.extend(batch["subject_ids"].tolist())
        trials.extend(batch["trial_numbers"].tolist())
        trial_ids.extend(batch["trial_ids"].tolist())
        if export_trajectories:
            boundary_starts = torch.zeros_like(outputs["token_mask"])
            boundary_starts[:, 0] = True
            if boundary_starts.shape[1] > 1:
                boundary_starts[:, 1:] = outputs["boundary_probabilities"] >= 0.5
            exported = hard_token_runs(
                outputs["hard_tokens"],
                outputs["frame_lengths"],
                boundary_starts=boundary_starts,
            )
            for row, export in enumerate(exported):
                sequence = [int(item) for item in export["tokens"]]
                token_sequences.append(sequence)
                expanded_tokens.extend(sequence)
                expanded_subjects.extend([int(batch["subject_ids"][row])] * len(sequence))
                runs = _runs_with_sample_ranges(
                    export,
                    int(batch["lengths"][row]),
                    model.config.frame_size,
                    model.config.frame_stride,
                )
                records.append(
                    {
                        "trial_id": int(batch["trial_ids"][row]),
                        "trial_key": batch["trial_keys"][row],
                        "subject_id": int(batch["subject_ids"][row]),
                        "trial_number": int(batch["trial_numbers"][row]),
                        "original_activity_class_index": int(batch["class_indices"][row]),
                        "original_activity_name": batch["activity_names"][row],
                        "sample_count": int(batch["lengths"][row]),
                        "frame_count": int(export["token_count"]),
                        "primitive_segment_count": int(export["run_count"]),
                        "unique_primitive_type_count": len(set(sequence)),
                        "frame_token_sequence": sequence,
                        "primitive_runs": runs,
                    }
                )
    return {
        "embeddings": np.concatenate(embeddings, axis=0),
        "logits": np.concatenate(logits, axis=0),
        "targets": np.asarray(targets, dtype=np.int64),
        "subject_ids": np.asarray(subjects, dtype=np.int64),
        "trial_numbers": np.asarray(trials, dtype=np.int64),
        "trial_ids": np.asarray(trial_ids, dtype=np.int64),
        "token_sequences": token_sequences,
        "trajectory_records": records,
        "expanded_tokens": np.asarray(expanded_tokens, dtype=np.int64),
        "expanded_subject_ids": np.asarray(expanded_subjects, dtype=np.int64),
    }


def _boundary_repeatability(records: Sequence[Mapping[str, Any]], bins: int = 20) -> dict[str, float]:
    histograms = []
    labels = []
    subjects = []
    for record in records:
        frame_count = int(record["frame_count"])
        starts = [int(run["start_token_index"]) for run in record["primitive_runs"]][1:]
        histogram = np.zeros(int(bins), dtype=np.float64)
        if starts and frame_count > 1:
            positions = np.asarray(starts, dtype=np.float64) / float(frame_count - 1)
            indices = np.minimum((positions * bins).astype(int), bins - 1)
            np.add.at(histogram, indices, 1.0)
            histogram /= histogram.sum()
        histograms.append(histogram)
        labels.append(int(record["original_activity_class_index"]))
        subjects.append(int(record["subject_id"]))
    values = np.asarray(histograms)
    labels_array = np.asarray(labels)
    subjects_array = np.asarray(subjects)
    similarities_same = []
    similarities_different = []
    for left in range(len(values)):
        for right in range(left + 1, len(values)):
            if subjects_array[left] == subjects_array[right]:
                continue
            denominator = max(
                np.linalg.norm(values[left]) * np.linalg.norm(values[right]), 1e-12
            )
            similarity = float(np.dot(values[left], values[right]) / denominator)
            if labels_array[left] == labels_array[right]:
                similarities_same.append(similarity)
            else:
                similarities_different.append(similarity)
    same = float(np.mean(similarities_same)) if similarities_same else 0.0
    different = float(np.mean(similarities_different)) if similarities_different else 0.0
    return {
        "same_activity_boundary_histogram_cosine": same,
        "different_activity_boundary_histogram_cosine": different,
        "same_minus_different_margin": same - different,
    }


def _primitive_count_summary(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    counts = np.asarray(
        [int(item["primitive_segment_count"]) for item in records], dtype=np.float64
    )
    by_activity: dict[str, Any] = {}
    for class_index in range(12):
        selected = [
            int(item["primitive_segment_count"])
            for item in records
            if int(item["original_activity_class_index"]) == class_index
        ]
        if selected:
            by_activity[ACTIVITY_NAMES[class_index + 1]] = {
                "mean": float(np.mean(selected)),
                "minimum": int(np.min(selected)),
                "maximum": int(np.max(selected)),
            }
    return {
        "mean": float(np.mean(counts)),
        "median": float(np.median(counts)),
        "minimum": int(np.min(counts)),
        "maximum": int(np.max(counts)),
        "by_original_activity": by_activity,
    }


def evaluate_selected_model(
    model: MotionTrajectoryModel,
    datasets: USCHADFoldDatasets,
    device: torch.device,
    output_dir: Path,
    *,
    batch_size: int,
    num_workers: int,
    cluster_restarts: int,
    order_shuffles: int,
    seed: int,
) -> dict[str, Any]:
    train = encode_dataset(
        model,
        datasets.train,
        device,
        batch_size=batch_size,
        num_workers=num_workers,
        export_trajectories=False,
    )
    test = encode_dataset(
        model,
        datasets.outer_test,
        device,
        batch_size=batch_size,
        num_workers=num_workers,
        export_trajectories=True,
    )
    clustering = semi_supervised_kmeans(
        train["embeddings"],
        train["targets"],
        test["embeddings"],
        num_classes=12,
        num_old_classes=6,
        seed=int(seed),
        n_init=int(cluster_restarts),
    )
    names = {index: ACTIVITY_NAMES[index + 1] for index in range(12)}
    scores = score_cgcd_clusters(
        clustering.assignments,
        test["targets"],
        num_classes=12,
        num_old_classes=6,
        activity_names=names,
    )
    predictions = scores.pop("predictions")
    for record, prediction in zip(test["trajectory_records"], predictions):
        record["predicted_activity_class_index"] = int(prediction)
        record["predicted_activity_name"] = names[int(prediction)]
        record["prediction_correct"] = bool(
            int(prediction) == int(record["original_activity_class_index"])
        )
    trajectory_path = output_dir / "outer_test_primitive_trajectories.jsonl"
    with trajectory_path.open("w", encoding="utf-8") as handle:
        for record in test["trajectory_records"]:
            handle.write(_canonical_json(record) + "\n")

    old_mask = test["targets"] < 6
    direct_predictions = np.argmax(test["logits"], axis=1)
    direct_old_accuracy = float(
        np.mean(direct_predictions[old_mask] == test["targets"][old_mask])
    )
    direct_old_f1 = float(
        f1_score(
            test["targets"][old_mask],
            direct_predictions[old_mask],
            labels=np.arange(6),
            average="macro",
            zero_division=0,
        )
    )

    shuffle_scores = []
    shuffle_direct_old = []
    train_order_control = encode_dataset(
        model,
        datasets.train,
        device,
        batch_size=batch_size,
        num_workers=num_workers,
        strict_relation_recompute=True,
        export_trajectories=False,
    )
    test_order_control = encode_dataset(
        model,
        datasets.outer_test,
        device,
        batch_size=batch_size,
        num_workers=num_workers,
        strict_relation_recompute=True,
        export_trajectories=False,
    )
    identity_control_clustering = semi_supervised_kmeans(
        train_order_control["embeddings"],
        train_order_control["targets"],
        test_order_control["embeddings"],
        num_classes=12,
        num_old_classes=6,
        seed=int(seed),
        n_init=int(cluster_restarts),
    )
    identity_control_scores = score_cgcd_clusters(
        identity_control_clustering.assignments,
        test["targets"],
        num_classes=12,
        num_old_classes=6,
    )
    identity_direct_predictions = np.argmax(test_order_control["logits"], axis=1)
    identity_direct_old_accuracy = float(
        np.mean(identity_direct_predictions[old_mask] == test["targets"][old_mask])
    )
    for repeat in range(int(order_shuffles)):
        shuffled = encode_dataset(
            model,
            datasets.outer_test,
            device,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle_seed=int(seed) + 100003 * (repeat + 1),
            strict_relation_recompute=True,
            export_trajectories=False,
        )
        shuffled_clustering = semi_supervised_kmeans(
            train_order_control["embeddings"],
            train_order_control["targets"],
            shuffled["embeddings"],
            num_classes=12,
            num_old_classes=6,
            seed=int(seed),
            n_init=int(cluster_restarts),
        )
        shuffled_score = score_cgcd_clusters(
            shuffled_clustering.assignments,
            test["targets"],
            num_classes=12,
            num_old_classes=6,
        )
        shuffle_scores.append(
            {key: float(shuffled_score[key]) for key in ("all_accuracy", "old_accuracy", "new_accuracy", "h_score", "macro_f1")}
        )
        shuffle_direct_old.append(
            float(
                np.mean(
                    np.argmax(shuffled["logits"], axis=1)[old_mask]
                    == test["targets"][old_mask]
                )
            )
        )
    mean_shuffled = {
        key: float(np.mean([item[key] for item in shuffle_scores]))
        for key in ("all_accuracy", "old_accuracy", "new_accuracy", "h_score", "macro_f1")
    }
    order_control = {
        "repeat_count": int(order_shuffles),
        "shuffle_unit": "predicted primitive run blocks; within-run frames retained",
        "relation_policy": (
            "identity and shuffled arms both recompute transition/duration from the presented "
            "assignment order and zero learned-boundary ingress"
        ),
        "identity_relation_control_metrics": {
            key: float(identity_control_scores[key])
            for key in ("all_accuracy", "old_accuracy", "new_accuracy", "h_score", "macro_f1")
        },
        "main_minus_identity_relation_control": {
            key: float(scores[key] - identity_control_scores[key])
            for key in ("all_accuracy", "old_accuracy", "new_accuracy", "h_score", "macro_f1")
        },
        "mean_shuffled_metrics": mean_shuffled,
        "identity_control_minus_shuffled": {
            key: float(identity_control_scores[key] - mean_shuffled[key])
            for key in mean_shuffled
        },
        "direct_old_accuracy_standard": direct_old_accuracy,
        "direct_old_accuracy_identity_relation_control": identity_direct_old_accuracy,
        "direct_old_accuracy_shuffled_mean": float(np.mean(shuffle_direct_old)),
        "direct_old_accuracy_drop": float(
            identity_direct_old_accuracy - np.mean(shuffle_direct_old)
        ),
    }

    diagnostics = codebook_diagnostics(
        test["expanded_tokens"],
        test["expanded_subject_ids"],
        num_codes=model.config.codebook_size,
    )
    class_names = [ACTIVITY_NAMES[index] for index in range(1, 13)]
    save_confusion_heatmap(
        np.asarray(scores["confusion_matrix"]),
        class_names,
        output_dir / "cgcd_confusion_heatmap.png",
    )
    save_class_codebook_heatmap(
        test["token_sequences"],
        test["targets"],
        class_names,
        model.config.codebook_size,
        output_dir / "activity_codebook_heatmap.png",
    )
    save_representative_trajectory_panel(
        test["token_sequences"],
        test["targets"],
        predictions,
        test["subject_ids"],
        test["trial_numbers"],
        class_names,
        model.config.codebook_size,
        output_dir / "fixed_trajectories_and_predictions.png",
    )
    return {
        "cgcd_metrics": scores,
        "semi_supervised_kmeans": {
            "transductive": True,
            "evaluation_labels_used_during_fit": False,
            "old_train_labels_used_as_fixed_anchors": True,
            "oracle_total_class_count": 12,
            "inertia": float(clustering.inertia),
            "iterations": int(clustering.iterations),
            "restart": int(clustering.restart),
        },
        "direct_old_trajectory_classifier": {
            "outer_test_old_accuracy": direct_old_accuracy,
            "outer_test_old_macro_f1": direct_old_f1,
            "new_class_prediction_supported": False,
        },
        "order_shuffle_control": order_control,
        "codebook_diagnostics": diagnostics,
        "boundary_repeatability": _boundary_repeatability(
            test["trajectory_records"]
        ),
        "primitive_segment_counts": _primitive_count_summary(
            test["trajectory_records"]
        ),
        "trajectory_export": str(trajectory_path),
        "plots": {
            "confusion": str(output_dir / "cgcd_confusion_heatmap.png"),
            "activity_codebook": str(output_dir / "activity_codebook_heatmap.png"),
            "representative_trajectories": str(
                output_dir / "fixed_trajectories_and_predictions.png"
            ),
        },
    }


def _create_loader(
    dataset: Any,
    *,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    seed: int,
    device: torch.device,
) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    return DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=bool(shuffle),
        num_workers=int(num_workers),
        collate_fn=pad_trial_batch,
        pin_memory=device.type == "cuda",
        generator=generator if shuffle else None,
        persistent_workers=int(num_workers) > 0,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="One-stage full-trial motion-primitive trajectory training."
    )
    default_data = (
        r"D:\WorkDir\DataSet\USC-HAD"
        if os.name == "nt"
        else "/mnt/d/WorkDir/DataSet/USC-HAD"
    )
    parser.add_argument("--data-root", default=default_data)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--profile", choices=(J0_UNSUPERVISED, J0_TRAJECTORY), required=True)
    parser.add_argument("--fold", type=int, required=True, choices=range(1, 8))
    parser.add_argument("--seed", type=int, default=50)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--gradient-clip", type=float, default=5.0)
    parser.add_argument("--warmup-epochs", type=int, default=10)
    parser.add_argument("--early-stop-patience", type=int, default=20)
    parser.add_argument("--frame-size", type=int, default=128)
    parser.add_argument("--frame-stride", type=int, default=64)
    parser.add_argument("--codebook-size", type=int, default=32)
    parser.add_argument("--temperature-start", type=float, default=2.0)
    parser.add_argument("--temperature-end", type=float, default=0.25)
    parser.add_argument("--trajectory-mask-ratio", type=float, default=0.15)
    parser.add_argument("--labelled-fraction", type=float, default=0.8)
    parser.add_argument("--anomaly-policy", choices=("report", "exclude"), default="report")
    parser.add_argument("--cluster-restarts", type=int, default=10)
    parser.add_argument("--order-shuffles", type=int, default=10)
    parser.add_argument("--minimum-segment-windows", type=int, default=2)
    parser.add_argument("--deterministic", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--resume", action="store_true")
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    for name in ("epochs", "batch_size", "eval_batch_size", "frame_size", "frame_stride", "codebook_size", "cluster_restarts", "order_shuffles"):
        if int(getattr(args, name)) < 1:
            raise ValueError(f"--{name.replace('_', '-')} must be positive.")
    for name in ("learning_rate", "gradient_clip", "temperature_start", "temperature_end"):
        value = float(getattr(args, name))
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive and finite.")
    if int(args.frame_stride) > int(args.frame_size):
        raise ValueError("--frame-stride cannot exceed --frame-size in J0.")
    if int(args.codebook_size) < 2:
        raise ValueError("--codebook-size must be at least two.")


def run(args: argparse.Namespace) -> dict[str, Any]:
    _validate_args(args)
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    complete_path = output_dir / "complete.json"
    if complete_path.exists() and args.resume:
        saved_manifest = json.loads(
            (output_dir / "run_manifest.json").read_text(encoding="utf-8")
        )
        requested = (str(args.profile), int(args.fold), int(args.seed), int(args.epochs))
        recorded = (
            saved_manifest.get("profile"),
            int(saved_manifest.get("fold", -1)),
            int(saved_manifest.get("seed", -1)),
            int(saved_manifest.get("target_epochs", -1)),
        )
        if recorded != requested:
            raise RuntimeError(
                "Completed output does not match requested profile/fold/seed/epochs; "
                "use a new --output-dir."
            )
        return json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
    existing = [item for item in output_dir.iterdir() if item.name not in {"checkpoint_last.pt"}]
    if existing and not args.resume:
        raise RuntimeError(
            f"Output directory is not empty: {output_dir}. Use a new directory or --resume."
        )

    set_reproducible_seed(int(args.seed), bool(args.deterministic))
    device = resolve_device(args.device)
    datasets = build_uschad_fold_datasets(
        args.data_root,
        fold=int(args.fold),
        label_regime=args.profile,
        anomaly_policy=args.anomaly_policy,
        labelled_fraction=float(args.labelled_fraction),
        label_seed=int(args.seed),
    )
    model_config = MotionTrajectoryConfig(
        frame_size=int(args.frame_size),
        frame_stride=int(args.frame_stride),
        codebook_size=int(args.codebook_size),
        codebook_temperature=float(args.temperature_end),
        trajectory_mask_ratio=float(args.trajectory_mask_ratio),
        num_classes=6,
        unlabelled_index=IGNORE_INDEX,
    ).validated()
    base_loss = (
        OneStageLossConfig.j0_u(
            minimum_segment_windows=int(args.minimum_segment_windows),
            unlabelled_index=IGNORE_INDEX,
        )
        if args.profile == J0_UNSUPERVISED
        else OneStageLossConfig.j0_t(
            minimum_segment_windows=int(args.minimum_segment_windows),
            unlabelled_index=IGNORE_INDEX,
        )
    )
    raw_targets, boundary_audit = prepare_raw_boundary_targets(
        datasets,
        frame_size=model_config.frame_size,
        frame_stride=model_config.frame_stride,
    )
    identity_payload = {
        "schema": SCHEMA_VERSION,
        "profile": args.profile,
        "fold": int(args.fold),
        "seed": int(args.seed),
        "target_epochs": int(args.epochs),
        "data_root": str(Path(args.data_root).expanduser().resolve()),
        "data_protocol_identity": datasets.protocol_audit["identity_sha256"],
        "model_config": asdict(model_config),
        "loss_config": base_loss.to_audit(),
        "optimizer": {
            "name": "AdamW",
            "learning_rate": float(args.learning_rate),
            "weight_decay": float(args.weight_decay),
        },
        "one_model_one_optimizer_continuous_run": True,
        "happy_checkpoint_loaded": False,
    }
    run_identity = _sha256_document(identity_payload)
    manifest = {
        **identity_payload,
        "run_identity_sha256": run_identity,
        "data_audit": datasets.audit_dict(),
        "boundary_calibration_sha256": boundary_audit["calibration"]["calibration_sha256"],
        "command": sys.argv,
    }
    manifest_path = output_dir / "run_manifest.json"
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        if previous.get("run_identity_sha256") != run_identity:
            raise RuntimeError(
                "Existing output records a different run identity; use a new --output-dir."
            )
    else:
        _write_json(manifest_path, manifest)
    _write_json(output_dir / "boundary_calibration.json", boundary_audit)

    model = MotionTrajectoryModel(model_config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(args.learning_rate),
        weight_decay=float(args.weight_decay),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, int(args.epochs))
    )
    train_loader = _create_loader(
        datasets.train,
        batch_size=int(args.batch_size),
        shuffle=True,
        num_workers=int(args.num_workers),
        seed=int(args.seed),
        device=device,
    )
    validation_loader = _create_loader(
        datasets.validation,
        batch_size=int(args.eval_batch_size),
        shuffle=False,
        num_workers=int(args.num_workers),
        seed=int(args.seed),
        device=device,
    )

    history_path = output_dir / "history.jsonl"
    start_epoch = 1
    best_key: Optional[tuple[float, float]] = None
    best_epoch = 0
    patience = 0
    last_path = output_dir / "checkpoint_last.pt"
    best_path = output_dir / "checkpoint_best.pt"
    if args.resume and last_path.exists():
        checkpoint = torch.load(last_path, map_location=device, weights_only=False)
        if checkpoint.get("run_identity_sha256") != run_identity:
            raise RuntimeError("checkpoint_last.pt belongs to another run identity.")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best_key_value = checkpoint.get("best_key")
        best_key = tuple(best_key_value) if best_key_value is not None else None
        best_epoch = int(checkpoint.get("best_epoch", 0))
        patience = int(checkpoint.get("patience", 0))

    for epoch in range(start_epoch, int(args.epochs) + 1):
        scheduled = _scheduled_loss_config(
            base_loss, epoch, warmup_epochs=int(args.warmup_epochs)
        )
        temperature = _temperature(
            epoch,
            int(args.epochs),
            float(args.temperature_start),
            float(args.temperature_end),
        )
        train_metrics = train_epoch(
            model,
            train_loader,
            optimizer,
            scheduled,
            raw_targets,
            device,
            temperature=temperature,
            gradient_clip=float(args.gradient_clip),
        )
        # Checkpoint selection always evaluates the full, fixed base objective;
        # otherwise early low curriculum weights would unfairly favour epoch 1.
        validation_metrics = validate_epoch(
            model, validation_loader, base_loss, raw_targets, device
        )
        key = _selection_key(args.profile, validation_metrics)
        improved = best_key is None or key > best_key
        if improved:
            best_key = key
            best_epoch = epoch
            patience = 0
            _atomic_torch_save(
                best_path,
                {
                    "schema": SCHEMA_VERSION,
                    "run_identity_sha256": run_identity,
                    "epoch": int(epoch),
                    "model_config": asdict(model_config),
                    "model": model.state_dict(),
                    "validation": validation_metrics,
                },
            )
        else:
            patience += 1
        scheduler.step()
        state = {
            "schema": SCHEMA_VERSION,
            "run_identity_sha256": run_identity,
            "epoch": int(epoch),
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "best_key": list(best_key) if best_key is not None else None,
            "best_epoch": int(best_epoch),
            "patience": int(patience),
        }
        _atomic_torch_save(last_path, state)
        record = {
            "epoch": int(epoch),
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "scheduled_loss_weights": asdict(scheduled),
            "train": train_metrics,
            "validation": validation_metrics,
            "best_epoch": int(best_epoch),
            "improved": bool(improved),
        }
        _append_jsonl(history_path, record)
        print(
            f"epoch={epoch:03d} profile={args.profile} "
            f"train_loss={train_metrics['loss']:.5f} "
            f"val_loss={validation_metrics['loss']:.5f} "
            f"effective_K={validation_metrics['effective_code_count']:.2f} "
            f"best={best_epoch}",
            flush=True,
        )
        if int(args.early_stop_patience) > 0 and patience >= int(
            args.early_stop_patience
        ):
            break

    if not best_path.is_file():
        raise RuntimeError("Training completed without a selected checkpoint.")
    selected = torch.load(best_path, map_location=device, weights_only=False)
    if selected.get("run_identity_sha256") != run_identity:
        raise RuntimeError("Selected checkpoint identity mismatch.")
    model.load_state_dict(selected["model"])
    selected_sha256 = _checkpoint_sha256(best_path)
    datasets.outer_test.unlock_outer_test_for_evaluation(selected_sha256)
    evaluation = evaluate_selected_model(
        model,
        datasets,
        device,
        output_dir,
        batch_size=int(args.eval_batch_size),
        num_workers=int(args.num_workers),
        cluster_restarts=int(args.cluster_restarts),
        order_shuffles=int(args.order_shuffles),
        seed=int(args.seed),
    )
    summary = {
        "schema": SCHEMA_VERSION,
        "run_identity_sha256": run_identity,
        "profile": args.profile,
        "fold": int(args.fold),
        "seed": int(args.seed),
        "selected_epoch": int(selected["epoch"]),
        "selected_checkpoint": str(best_path),
        "selected_checkpoint_sha256": selected_sha256,
        "selection_validation": selected["validation"],
        "evaluation": evaluation,
        "runtime_sensor_access": datasets.audit_dict()["runtime_sensor_access"],
        "interpretation_limits": [
            "Outer-test CGCD clustering is transductive and uses an oracle total class count of 12.",
            "Novel-cluster Hungarian naming uses ground truth only after clustering for scoring.",
            "J0-U is label-free representation training, not a deployable label-free classifier.",
            "Online adaptive codebook expansion is not part of J0 and remains a later experiment.",
        ],
    }
    _write_json(output_dir / "summary.json", summary)
    _write_json(
        complete_path,
        {
            "run_identity_sha256": run_identity,
            "summary_sha256": _checkpoint_sha256(output_dir / "summary.json"),
            "selected_checkpoint_sha256": selected_sha256,
        },
    )
    return summary


def main() -> None:
    args = build_parser().parse_args()
    summary = run(args)
    metrics = summary["evaluation"]["cgcd_metrics"]
    print(
        "completed "
        f"profile={summary['profile']} fold={summary['fold']} seed={summary['seed']} "
        f"all={metrics['all_accuracy']:.4f} old={metrics['old_accuracy']:.4f} "
        f"new={metrics['new_accuracy']:.4f} H={metrics['h_score']:.4f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
