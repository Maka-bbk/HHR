"""Train-only feature segmentation utilities for motion primitive experiments.

The ``ssl_feature_changepoint`` path deliberately keeps the existing frozen
window encoder and learns only a small masked-denoising adapter without labels.
The ``motion_encoder_changepoint`` path consumes the dedicated segmentation
head from a separately trained motion-primitive encoder.  In both cases,
change-point calibration uses fit-subject/old-class trials only.  Detected
variable-length segments are represented by the mean of their codebook
features; their assigned token is later expanded back to the original window
grid so the legacy sequence controls remain comparable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


EPS = 1e-8


class MaskedDenoisingAdapter(nn.Module):
    """Small self-supervised bottleneck over frozen window embeddings."""

    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        input_dim = int(input_dim)
        output_dim = int(output_dim)
        hidden_dim = max(input_dim, 2 * output_dim)
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(output_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, input_dim),
        )

    def forward(self, values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        latent = self.encoder(values)
        return latent, self.decoder(latent)

    def encode(self, values: torch.Tensor) -> torch.Tensor:
        return self.encoder(values)


@dataclass
class SSLAdapterBundle:
    model: MaskedDenoisingAdapter
    input_mean: np.ndarray
    input_std: np.ndarray
    training: dict


@dataclass
class SegmentedFeatures:
    """Segment-level features plus a mapping back to input windows."""

    method: str
    segment_features: np.ndarray
    segment_trial_ids: np.ndarray
    segment_window_counts: np.ndarray
    segment_first_window_positions: np.ndarray
    segment_last_window_positions: np.ndarray
    segment_support_start_samples: np.ndarray
    segment_support_end_samples_exclusive: np.ndarray
    segment_partition_start_samples: np.ndarray
    segment_partition_end_samples_exclusive: np.ndarray
    segment_boundary_scores: np.ndarray
    window_segment_ids: np.ndarray
    trial_statistics: list[dict]


def _standardization(values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    values = np.asarray(values, dtype=np.float32)
    if values.ndim != 2 or len(values) < 2:
        raise ValueError("Self-supervised adapter needs at least two 2D feature rows.")
    if not np.all(np.isfinite(values)):
        raise ValueError("Adapter input contains non-finite values.")
    mean = values.mean(axis=0, dtype=np.float64).astype(np.float32)
    std = values.std(axis=0, dtype=np.float64).astype(np.float32)
    std = np.maximum(std, np.float32(1e-6))
    standardized = ((values - mean) / std).astype(np.float32)
    return standardized, mean, std


def train_masked_denoising_adapter(
    fit_embeddings: np.ndarray,
    sample_weights: np.ndarray,
    output_dim: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    mask_ratio: float,
    noise_std: float,
    device: torch.device,
    seed: int,
) -> SSLAdapterBundle:
    """Fit a label-free masked-denoising adapter on fit embeddings only."""

    standardized, input_mean, input_std = _standardization(fit_embeddings)
    sample_weights = np.asarray(sample_weights, dtype=np.float32)
    if sample_weights.shape != (len(standardized),):
        raise ValueError("SSL sample weights must match fit embeddings.")
    if np.any(sample_weights <= 0) or not np.all(np.isfinite(sample_weights)):
        raise ValueError("SSL sample weights must be positive and finite.")
    sample_weights = sample_weights / float(np.mean(sample_weights))
    input_dim = int(standardized.shape[1])
    if not 1 <= int(output_dim) <= input_dim:
        raise ValueError(
            f"SSL adapter dimension must be in [1,{input_dim}], got {output_dim}."
        )
    if int(epochs) < 1 or int(batch_size) < 2 or float(learning_rate) <= 0:
        raise ValueError("SSL epochs/lr must be positive and batch size at least 2.")
    if not 0.0 <= float(mask_ratio) < 1.0 or float(noise_std) < 0.0:
        raise ValueError("SSL mask ratio must be in [0,1) and noise std non-negative.")

    # The seed controls initialization, shuffling, masks, and feature noise.
    torch.manual_seed(int(seed))
    if device.type == "cuda":
        torch.cuda.manual_seed_all(int(seed))
    model = MaskedDenoisingAdapter(input_dim, int(output_dim)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(learning_rate))
    source = torch.from_numpy(standardized)
    source_weights = torch.from_numpy(sample_weights)
    losses: list[float] = []
    model.train()
    for _ in range(int(epochs)):
        permutation = torch.randperm(len(source))
        total_loss = 0.0
        total_count = 0
        for begin in range(0, len(source), int(batch_size)):
            positions = permutation[begin : begin + int(batch_size)]
            clean = source[positions].to(device=device)
            weights = source_weights[positions].to(device=device)
            keep = (torch.rand_like(clean) >= float(mask_ratio)).to(clean.dtype)
            corrupted = clean * keep
            if float(noise_std) > 0.0:
                corrupted = corrupted + torch.randn_like(corrupted) * float(noise_std)
            _, reconstruction = model(corrupted)
            per_sample_loss = F.mse_loss(
                reconstruction, clean, reduction="none"
            ).mean(dim=1)
            loss = torch.sum(per_sample_loss * weights) / torch.sum(weights)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            count = int(len(clean))
            total_loss += float(loss.detach().cpu()) * count
            total_count += count
        losses.append(total_loss / max(total_count, 1))

    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return SSLAdapterBundle(
        model=model,
        input_mean=input_mean,
        input_std=input_std,
        training={
            "objective": "masked_denoising_reconstruction_of_frozen_window_embeddings",
            "uses_activity_labels": False,
            "sample_weighting": "caller-supplied train-only codebook weighting",
            "fit_sample_count": int(len(standardized)),
            "input_dim": input_dim,
            "output_dim": int(output_dim),
            "epochs": int(epochs),
            "batch_size": int(batch_size),
            "learning_rate": float(learning_rate),
            "mask_ratio": float(mask_ratio),
            "noise_std": float(noise_std),
            "loss_first": float(losses[0]),
            "loss_last": float(losses[-1]),
            "loss_min": float(min(losses)),
            "loss_by_epoch": losses,
        },
    )


def transform_with_adapter(
    bundle: SSLAdapterBundle,
    embeddings: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    """Transform embeddings with train-only statistics and L2-normalize output."""

    values = np.asarray(embeddings, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] != bundle.model.input_dim:
        raise ValueError("Embedding shape does not match the SSL adapter input.")
    standardized = ((values - bundle.input_mean) / bundle.input_std).astype(np.float32)
    output = np.empty((len(values), bundle.model.output_dim), dtype=np.float32)
    bundle.model.to(device)
    with torch.inference_mode():
        for begin in range(0, len(values), int(batch_size)):
            end = min(begin + int(batch_size), len(values))
            batch = torch.from_numpy(standardized[begin:end]).to(device=device)
            latent = F.normalize(bundle.model.encode(batch), dim=1)
            output[begin:end] = latent.cpu().numpy().astype(np.float32)
    if not np.all(np.isfinite(output)):
        raise RuntimeError("SSL adapter produced non-finite features.")
    return output


def ordered_trial_positions(
    trial_ids: np.ndarray, window_starts: np.ndarray
) -> list[tuple[int, np.ndarray]]:
    trial_ids = np.asarray(trial_ids, dtype=np.int64)
    window_starts = np.asarray(window_starts, dtype=np.int64)
    if trial_ids.ndim != 1 or trial_ids.shape != window_starts.shape:
        raise ValueError("trial_ids and window_starts must be equally sized 1D arrays.")
    groups = []
    for trial_id in np.unique(trial_ids):
        positions = np.flatnonzero(trial_ids == trial_id)
        order = np.argsort(window_starts[positions], kind="stable")
        positions = positions[order]
        if len(positions) > 1 and np.any(np.diff(window_starts[positions]) <= 0):
            raise ValueError(f"Trial {trial_id} window starts are not strictly increasing.")
        groups.append((int(trial_id), positions))
    return groups


def feature_change_scores(features: np.ndarray, context_windows: int) -> np.ndarray:
    """Cosine distance between local left/right feature means at every boundary."""

    values = np.asarray(features, dtype=np.float32)
    if values.ndim != 2 or len(values) == 0:
        raise ValueError("features must be a non-empty [N,D] array.")
    if int(context_windows) < 1:
        raise ValueError("context_windows must be positive.")
    if len(values) == 1:
        return np.empty(0, dtype=np.float32)
    scores = np.empty(len(values) - 1, dtype=np.float32)
    radius = int(context_windows)
    for boundary in range(1, len(values)):
        left = values[max(0, boundary - radius) : boundary].mean(axis=0)
        right = values[boundary : min(len(values), boundary + radius)].mean(axis=0)
        denominator = max(float(np.linalg.norm(left) * np.linalg.norm(right)), EPS)
        cosine = float(np.dot(left, right) / denominator)
        scores[boundary - 1] = np.float32(1.0 - np.clip(cosine, -1.0, 1.0))
    return scores


def weighted_quantile(
    values: np.ndarray, weights: np.ndarray, quantile: float
) -> float:
    values = np.asarray(values, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    if values.ndim != 1 or values.shape != weights.shape or len(values) == 0:
        raise ValueError("Weighted quantile inputs must be non-empty equal 1D arrays.")
    if not 0.0 <= float(quantile) <= 1.0:
        raise ValueError("quantile must be in [0,1].")
    if np.any(weights < 0) or float(weights.sum()) <= 0:
        raise ValueError("Weighted quantile needs non-negative positive-total weights.")
    order = np.argsort(values, kind="stable")
    ordered_values = values[order]
    ordered_weights = weights[order]
    cumulative = np.cumsum(ordered_weights)
    target = float(quantile) * float(cumulative[-1])
    index = int(np.searchsorted(cumulative, target, side="left"))
    return float(ordered_values[min(index, len(ordered_values) - 1)])


def calibrate_changepoint_threshold(
    fit_boundary_features: np.ndarray,
    fit_trial_ids: np.ndarray,
    fit_window_starts: np.ndarray,
    context_windows: int,
    score_quantile: float,
) -> tuple[float, dict]:
    """Calibrate one threshold using train-old trial-equal score mass only."""

    values = []
    weights = []
    trial_score_counts = {}
    for trial_id, positions in ordered_trial_positions(fit_trial_ids, fit_window_starts):
        scores = feature_change_scores(
            fit_boundary_features[positions], int(context_windows)
        )
        trial_score_counts[str(trial_id)] = int(len(scores))
        if len(scores):
            values.append(scores.astype(np.float64))
            weights.append(np.full(len(scores), 1.0 / len(scores), dtype=np.float64))
    if not values:
        raise RuntimeError("No fit trial contains a candidate change-point boundary.")
    pooled = np.concatenate(values)
    pooled_weights = np.concatenate(weights)
    threshold = weighted_quantile(pooled, pooled_weights, float(score_quantile))
    return threshold, {
        "source": "fit_subjects_old_classes_only",
        "trial_weighting": "each fit trial contributes total score mass 1",
        "context_windows": int(context_windows),
        "score_quantile": float(score_quantile),
        "threshold": float(threshold),
        "candidate_score_count": int(len(pooled)),
        "fit_trial_count": int(len(trial_score_counts)),
        "score_mean": float(np.mean(pooled)),
        "score_median": float(np.median(pooled)),
        "score_p95": float(np.percentile(pooled, 95.0)),
        "trial_score_counts": trial_score_counts,
    }


def select_changepoints(
    scores: np.ndarray,
    threshold: float,
    min_segment_windows: int,
) -> np.ndarray:
    """Greedily retain strongest eligible boundaries under a minimum gap."""

    scores = np.asarray(scores, dtype=np.float64)
    if scores.ndim != 1:
        raise ValueError("scores must be 1D.")
    minimum = int(min_segment_windows)
    if minimum < 1:
        raise ValueError("min_segment_windows must be positive.")
    window_count = int(len(scores) + 1)
    candidates = [
        boundary
        for boundary in range(1, window_count)
        if scores[boundary - 1] > float(threshold)
        and boundary >= minimum
        and window_count - boundary >= minimum
    ]
    candidates.sort(key=lambda boundary: (-float(scores[boundary - 1]), boundary))
    selected: list[int] = []
    for boundary in candidates:
        if all(abs(boundary - existing) >= minimum for existing in selected):
            selected.append(int(boundary))
    return np.asarray(sorted(selected), dtype=np.int64)


def _partition_sample_boundaries(
    ordered_starts: np.ndarray,
    segment_boundaries: np.ndarray,
    window_size: int,
) -> np.ndarray:
    values = [int(ordered_starts[0])]
    centers = ordered_starts.astype(np.float64) + float(window_size) / 2.0
    for boundary in segment_boundaries[1:-1]:
        midpoint = 0.5 * (centers[int(boundary) - 1] + centers[int(boundary)])
        values.append(int(round(midpoint)))
    values.append(int(ordered_starts[-1] + int(window_size)))
    return np.asarray(values, dtype=np.int64)


def build_segmented_features(
    codebook_window_features: np.ndarray,
    boundary_window_features: np.ndarray,
    trial_ids: np.ndarray,
    window_starts: np.ndarray,
    window_size: int,
    method: str,
    context_windows: int,
    threshold: float | None,
    min_segment_windows: int,
    normalize_segment_features: bool,
) -> SegmentedFeatures:
    """Create fixed-window or change-point segments without crossing trials."""

    codebook_values = np.asarray(codebook_window_features, dtype=np.float32)
    boundary_values = np.asarray(boundary_window_features, dtype=np.float32)
    trial_ids = np.asarray(trial_ids, dtype=np.int64)
    window_starts = np.asarray(window_starts, dtype=np.int64)
    if codebook_values.ndim != 2 or boundary_values.ndim != 2:
        raise ValueError("Codebook and boundary features must be 2D.")
    if not (
        len(codebook_values)
        == len(boundary_values)
        == len(trial_ids)
        == len(window_starts)
    ):
        raise ValueError("All segmentation inputs must have equal first dimensions.")
    changepoint_methods = {
        "ssl_feature_changepoint",
        "motion_encoder_changepoint",
    }
    if method not in {"fixed_window", *changepoint_methods}:
        raise ValueError(f"Unknown primitive segmentation method {method!r}.")
    if method in changepoint_methods and threshold is None:
        raise ValueError("Change-point segmentation requires a calibrated threshold.")
    if int(window_size) <= 0:
        raise ValueError("window_size must be positive.")

    segment_features = []
    segment_trial_ids = []
    segment_window_counts = []
    segment_first_positions = []
    segment_last_positions = []
    support_starts = []
    support_ends = []
    partition_starts = []
    partition_ends = []
    boundary_scores = []
    window_segment_ids = np.full(len(codebook_values), -1, dtype=np.int64)
    trial_statistics = []

    for trial_id, positions in ordered_trial_positions(trial_ids, window_starts):
        starts = window_starts[positions]
        scores = feature_change_scores(boundary_values[positions], int(context_windows))
        if method == "fixed_window":
            internal = np.arange(1, len(positions), dtype=np.int64)
        else:
            internal = select_changepoints(
                scores, float(threshold), int(min_segment_windows)
            )
        boundaries = np.r_[0, internal, len(positions)].astype(np.int64)
        partition = _partition_sample_boundaries(starts, boundaries, int(window_size))
        trial_segment_ids = []
        for segment_offset, (begin, end) in enumerate(
            zip(boundaries[:-1], boundaries[1:])
        ):
            member_positions = positions[int(begin) : int(end)]
            feature = codebook_values[member_positions].mean(axis=0)
            # A one-window segment must remain bit-for-bit compatible with the
            # legacy fixed-window KMeans input.  Multi-window means need to be
            # projected back to the unit sphere for cosine assignment.
            if normalize_segment_features and len(member_positions) > 1:
                norm = max(float(np.linalg.norm(feature)), EPS)
                feature = feature / norm
            segment_id = len(segment_features)
            segment_features.append(np.asarray(feature, dtype=np.float32))
            segment_trial_ids.append(int(trial_id))
            segment_window_counts.append(int(len(member_positions)))
            segment_first_positions.append(int(member_positions[0]))
            segment_last_positions.append(int(member_positions[-1]))
            support_starts.append(int(starts[int(begin)]))
            support_ends.append(int(starts[int(end) - 1] + int(window_size)))
            partition_starts.append(int(partition[segment_offset]))
            partition_ends.append(int(partition[segment_offset + 1]))
            boundary_scores.append(
                np.nan if int(begin) == 0 else float(scores[int(begin) - 1])
            )
            window_segment_ids[member_positions] = int(segment_id)
            trial_segment_ids.append(int(segment_id))
        trial_statistics.append(
            {
                "trial_id": int(trial_id),
                "window_count": int(len(positions)),
                "segment_count": int(len(boundaries) - 1),
                "segment_window_counts": np.diff(boundaries).astype(int).tolist(),
                "change_point_window_offsets": internal.astype(int).tolist(),
                "candidate_score_count": int(len(scores)),
                "change_score_mean": float(np.mean(scores)) if len(scores) else None,
                "change_score_max": float(np.max(scores)) if len(scores) else None,
                "segment_ids": trial_segment_ids,
            }
        )
    if np.any(window_segment_ids < 0):
        raise RuntimeError("At least one window was not assigned to a segment.")
    return SegmentedFeatures(
        method=method,
        segment_features=np.asarray(segment_features, dtype=np.float32),
        segment_trial_ids=np.asarray(segment_trial_ids, dtype=np.int64),
        segment_window_counts=np.asarray(segment_window_counts, dtype=np.int64),
        segment_first_window_positions=np.asarray(segment_first_positions, dtype=np.int64),
        segment_last_window_positions=np.asarray(segment_last_positions, dtype=np.int64),
        segment_support_start_samples=np.asarray(support_starts, dtype=np.int64),
        segment_support_end_samples_exclusive=np.asarray(support_ends, dtype=np.int64),
        segment_partition_start_samples=np.asarray(partition_starts, dtype=np.int64),
        segment_partition_end_samples_exclusive=np.asarray(partition_ends, dtype=np.int64),
        segment_boundary_scores=np.asarray(boundary_scores, dtype=np.float32),
        window_segment_ids=window_segment_ids,
        trial_statistics=trial_statistics,
    )


def segmentation_summary(segmented: SegmentedFeatures) -> dict:
    counts = np.asarray(segmented.segment_window_counts, dtype=np.int64)
    per_trial = np.asarray(
        [row["segment_count"] for row in segmented.trial_statistics], dtype=np.int64
    )
    return {
        "method": segmented.method,
        "window_count": int(np.sum(counts)),
        "segment_count": int(len(counts)),
        "trial_count": int(len(per_trial)),
        "segments_per_trial_mean": float(np.mean(per_trial)),
        "segments_per_trial_median": float(np.median(per_trial)),
        "segments_per_trial_min": int(np.min(per_trial)),
        "segments_per_trial_max": int(np.max(per_trial)),
        "segment_windows_mean": float(np.mean(counts)),
        "segment_windows_median": float(np.median(counts)),
        "segment_windows_min": int(np.min(counts)),
        "segment_windows_max": int(np.max(counts)),
        "single_segment_trial_ratio": float(np.mean(per_trial == 1)),
    }


def codebook_segment_weights(
    segmented: SegmentedFeatures, weighting: str
) -> np.ndarray:
    if weighting == "per_window":
        weights = segmented.segment_window_counts.astype(np.float64)
    elif weighting == "per_trial":
        # Preserve the legacy time-occupancy measure: every trial has equal
        # total mass, while a segment receives mass proportional to the number
        # of original windows it represents.  Giving every variable segment
        # equal mass would confound segmentation with a new weighting scheme.
        unique_trials, inverse = np.unique(
            segmented.segment_trial_ids, return_inverse=True
        )
        trial_window_totals = np.zeros(len(unique_trials), dtype=np.float64)
        np.add.at(
            trial_window_totals,
            inverse,
            segmented.segment_window_counts.astype(np.float64),
        )
        weights = (
            segmented.segment_window_counts.astype(np.float64)
            / trial_window_totals[inverse]
        )
        weights *= len(weights) / weights.sum()
    else:
        raise ValueError(f"Unknown codebook weighting {weighting!r}.")
    return weights.astype(np.float64)
