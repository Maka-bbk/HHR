"""Trajectory-only single-stage objective for motion-primitive HAR-CGCD.

This module intentionally has no complete-trial pooling loss, pooled
classification head, fused prediction, or image-CGCD objective. Offline class
labels supervise the complete variable-length primitive trajectory; no window
is assigned an activity target.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from experiments.motion_primitive.trajectory_contrastive import (
    SupervisedContrastiveLoss,
)


def _finite_nonnegative(name: str, value: float) -> float:
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be finite and non-negative, got {value!r}.")
    return result


@dataclass(frozen=True)
class MotionPrimitiveLossConfig:
    """Weights for one trajectory-only forward/backward objective."""

    motion_total_weight: float = 1.0
    trajectory_ce_weight: float = 1.0
    trajectory_supcon_weight: float = 0.25
    trajectory_view_consistency_weight: float = 0.0
    # A2-MP keeps the useful local representation objectives from A2, but
    # replaces A2's pooled trial auxiliary with trajectory supervision.
    changepoint_weight: float = 1.0
    content_boundary_alignment_weight: float = 0.10
    noncollapse_weight: float = 0.05
    temporal_prediction_weight: float = 0.50
    effective_minimum_duration_weight: float = 0.02
    vq_commitment_weight: float = 0.25
    vq_codebook_weight: float = 0.25
    utilization_weight: float = 0.0
    assignment_confidence_weight: float = 0.0
    codebook_diversity_weight: float = 0.0
    boundary_consistency_weight: float = 0.0
    transition_budget_weight: float = 0.02
    minimum_primitive_windows: int = 2
    utilization_entropy_floor: float = 0.50
    maximum_assignment_entropy: float = 0.50
    maximum_codebook_cosine: float = 0.25
    maximum_transition_rate: float = 0.35
    changepoint_stable_quantile: float = 0.25
    changepoint_change_quantile: float = 0.75
    # A within-trial quantile alone always labels a fraction of any non-flat
    # signal as changing.  This absolute + robust-null gate permits an entire
    # static or low-amplitude trial to contain no change anchor.
    changepoint_absolute_floor: float = 0.01
    changepoint_null_mad_multiplier: float = 3.0
    changepoint_rank_margin: float = 0.20
    changepoint_view_consistency_weight: float = 0.50
    temporal_prediction_mask_ratio: float = 0.20
    supcon_temperature: float = 0.07
    supcon_base_temperature: float = 0.07
    cross_subject_supcon_only: bool = True
    ignore_index: int = -100

    def validated(self) -> "MotionPrimitiveLossConfig":
        for name, value in asdict(self).items():
            if name.endswith("_weight"):
                _finite_nonnegative(name, value)
        if float(self.motion_total_weight) <= 0.0:
            raise ValueError("motion_total_weight must be positive.")
        if float(self.trajectory_ce_weight) <= 0.0:
            raise ValueError(
                "The primitive route requires complete-trajectory cross entropy."
            )
        for name in ("supcon_temperature", "supcon_base_temperature"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive.")
        if int(self.minimum_primitive_windows) < 1:
            raise ValueError("minimum_primitive_windows must be at least one.")
        for name in (
            "utilization_entropy_floor",
            "maximum_assignment_entropy",
            "maximum_transition_rate",
            "changepoint_stable_quantile",
            "changepoint_change_quantile",
            "temporal_prediction_mask_ratio",
        ):
            value = float(getattr(self, name))
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must lie in [0, 1].")
        if not (
            float(self.changepoint_stable_quantile)
            < float(self.changepoint_change_quantile)
        ):
            raise ValueError(
                "changepoint stable quantile must be below change quantile."
            )
        if not 0.0 <= float(self.changepoint_absolute_floor) <= 2.0:
            raise ValueError("changepoint_absolute_floor must lie in [0, 2].")
        if not math.isfinite(float(self.changepoint_null_mad_multiplier)) or float(
            self.changepoint_null_mad_multiplier
        ) < 0.0:
            raise ValueError(
                "changepoint_null_mad_multiplier must be finite and non-negative."
            )
        if not math.isfinite(float(self.changepoint_rank_margin)) or float(
            self.changepoint_rank_margin
        ) < 0.0:
            raise ValueError("changepoint_rank_margin must be finite and non-negative.")
        if not math.isfinite(float(self.changepoint_view_consistency_weight)) or float(
            self.changepoint_view_consistency_weight
        ) < 0.0:
            raise ValueError(
                "changepoint_view_consistency_weight must be finite and non-negative."
            )
        if not -1.0 <= float(self.maximum_codebook_cosine) <= 1.0:
            raise ValueError("maximum_codebook_cosine must lie in [-1, 1].")
        if int(self.ignore_index) >= 0:
            raise ValueError("ignore_index must be negative.")
        return self

    def audit_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload.update(
            {
                "schema": "hhr_motion_primitive_trajectory_loss_v1",
                "single_optimizer": True,
                "single_backward_per_batch": True,
                "activity_supervision_level": "complete_primitive_trajectory",
                "window_activity_labels_used": False,
                "complete_trial_pooling_used": False,
                "pooled_auxiliary_used": False,
                "fused_prediction_used": False,
                "a2_mother_route": True,
                "a2_trial_auxiliary_replaced_by": "trajectory_ce",
                "instance_infonce_used": False,
                "image_losses_used": False,
            }
        )
        return payload


@dataclass
class MotionPrimitiveLossResult:
    total: torch.Tensor
    motion_total: torch.Tensor
    components: Mapping[str, torch.Tensor]
    visible_trial_count: int


class MaskedTemporalPredictor(nn.Module):
    """Predict a masked local feature from neighbouring primitive features.

    This is deliberately an offline-only auxiliary module.  It is optimized in
    the same optimizer/backward call as the trajectory model, but is stored
    separately from the deployable model checkpoint so online inference cannot
    accidentally depend on it.
    """

    def __init__(self, feature_dim: int, hidden_dim: int | None = None) -> None:
        super().__init__()
        feature_dim = int(feature_dim)
        hidden_dim = feature_dim if hidden_dim is None else int(hidden_dim)
        if feature_dim < 1 or hidden_dim < 1:
            raise ValueError("temporal predictor dimensions must be positive.")
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.network = nn.Sequential(
            nn.Conv1d(feature_dim + 1, hidden_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(hidden_dim, feature_dim, kernel_size=3, padding=1),
        )

    def forward(
        self,
        features: torch.Tensor,
        prediction_mask: torch.Tensor,
        token_mask: torch.Tensor,
    ) -> torch.Tensor:
        if features.ndim != 3:
            raise ValueError("features must have shape [B,L,D].")
        if features.shape[-1] != self.feature_dim:
            raise ValueError("temporal predictor feature dimension mismatch.")
        if tuple(prediction_mask.shape) != tuple(features.shape[:2]) or tuple(
            token_mask.shape
        ) != tuple(features.shape[:2]):
            raise ValueError("prediction_mask/token_mask must match features [B,L].")
        valid = token_mask.bool()
        selected = prediction_mask.bool()
        if torch.any(selected & ~valid):
            raise ValueError("temporal prediction selected a padded token.")
        hidden = features.masked_fill((selected | ~valid).unsqueeze(-1), 0.0)
        mask_channel = selected.to(features.dtype).unsqueeze(-1)
        prediction = self.network(
            torch.cat((hidden, mask_channel), dim=-1).transpose(1, 2)
        ).transpose(1, 2)
        return prediction.masked_fill(~valid.unsqueeze(-1), 0.0)


def sample_temporal_prediction_mask(
    token_mask: torch.Tensor, ratio: float
) -> torch.Tensor:
    """Sample at least one valid target per non-empty trial when ratio > 0."""

    if token_mask.ndim != 2:
        raise ValueError("token_mask must have shape [B,L].")
    ratio = float(ratio)
    if not 0.0 <= ratio <= 1.0:
        raise ValueError("temporal prediction mask ratio must lie in [0,1].")
    valid = token_mask.bool()
    selected = torch.zeros_like(valid)
    if ratio == 0.0:
        return selected
    for row in range(valid.shape[0]):
        indices = torch.nonzero(valid[row], as_tuple=False).flatten()
        if len(indices) == 0:
            raise ValueError("Every trial must contain at least one valid token.")
        count = max(1, min(len(indices), int(round(len(indices) * ratio))))
        chosen = indices[torch.randperm(len(indices), device=indices.device)[:count]]
        selected[row, chosen] = True
    return selected


def _graph_zero(*tensors: torch.Tensor) -> torch.Tensor:
    for tensor in tensors:
        if isinstance(tensor, torch.Tensor):
            return tensor.sum() * 0.0
    raise ValueError("At least one tensor is required to create a graph zero.")


def _visible_mask(
    labels: torch.Tensor, visible_mask: torch.Tensor, ignore_index: int
) -> torch.Tensor:
    if labels.ndim != 1 or visible_mask.shape != labels.shape:
        raise ValueError("labels and visible_mask must have shape [B].")
    visible = visible_mask.bool()
    if torch.any((~visible) & (labels != int(ignore_index))):
        raise ValueError("A hidden trial carries a non-ignore activity target.")
    if torch.any(visible & (labels < 0)):
        raise ValueError("A visible trial carries an invalid activity target.")
    if not torch.any(visible):
        raise ValueError("Offline trajectory training needs visible old trials.")
    return visible


def _supervised_contrastive_loss(
    first: torch.Tensor,
    second: torch.Tensor,
    labels: torch.Tensor,
    config: MotionPrimitiveLossConfig,
    *,
    subject_ids: torch.Tensor | None,
) -> torch.Tensor:
    features = F.normalize(torch.stack((first, second), dim=1), dim=-1)
    criterion = SupervisedContrastiveLoss(
        temperature=float(config.supcon_temperature),
        base_temperature=float(config.supcon_base_temperature),
    )
    if subject_ids is None or not config.cross_subject_supcon_only:
        return criterion(features, labels=labels)
    if subject_ids.shape != labels.shape:
        raise ValueError("labels and subject_ids must both have shape [B].")
    same_class = labels[:, None].eq(labels[None, :])
    different_subject = subject_ids[:, None].ne(subject_ids[None, :])
    identity = torch.eye(len(labels), dtype=torch.bool, device=labels.device)
    positive_mask = (identity | (same_class & different_subject)).to(torch.float32)
    return criterion(features, mask=positive_mask)


def effective_minimum_duration_loss(
    final_boundary_probabilities: torch.Tensor,
    boundary_pair_mask: torch.Tensor,
    minimum_windows: int,
) -> torch.Tensor:
    """Penalise final boundaries which can create too-short primitive runs."""

    if final_boundary_probabilities.ndim != 2:
        raise ValueError("final_boundary_probabilities must have shape [B,L-1].")
    if boundary_pair_mask.shape != final_boundary_probabilities.shape:
        raise ValueError("boundary_pair_mask must match final boundary shape.")
    minimum_windows = int(minimum_windows)
    if minimum_windows < 1:
        raise ValueError("minimum_windows must be at least one.")
    if minimum_windows == 1:
        return _graph_zero(final_boundary_probabilities)
    probabilities = final_boundary_probabilities
    mask = boundary_pair_mask.bool()
    losses: list[torch.Tensor] = []
    for row in range(probabilities.shape[0]):
        valid_count = int(mask[row].sum().item())
        if valid_count == 0:
            continue
        if not torch.all(mask[row, :valid_count]) or torch.any(mask[row, valid_count:]):
            raise ValueError("boundary_pair_mask must be a contiguous valid prefix.")
        boundary = probabilities[row, :valid_count]
        token_count = valid_count + 1
        indices = torch.arange(valid_count, device=boundary.device)
        edge_violation = ((indices + 1) < minimum_windows) | (
            (token_count - indices - 1) < minimum_windows
        )
        if torch.any(edge_violation):
            losses.append(boundary[edge_violation].mean())
        distance = indices[None, :] - indices[:, None]
        close_pairs = (distance > 0) & (distance < minimum_windows)
        if torch.any(close_pairs):
            losses.append((boundary[:, None] * boundary[None, :])[close_pairs].mean())
    return torch.stack(losses).mean() if losses else _graph_zero(
        final_boundary_probabilities
    )


def _required(
    output: Mapping[str, torch.Tensor], key: str, view_index: int
) -> torch.Tensor:
    if key not in output:
        raise KeyError(f"view {view_index} is missing trajectory output {key!r}.")
    return output[key]


def _mean_or_zero(values: Sequence[torch.Tensor], zero: torch.Tensor) -> torch.Tensor:
    return torch.stack(tuple(values)).mean() if values else zero


def _adjacent_change_scores(
    features: torch.Tensor, pair_mask: torch.Tensor
) -> torch.Tensor:
    """Cosine change of adjacent local features, masked at padded boundaries."""

    if features.ndim != 3:
        raise ValueError("local features must have shape [B,L,D].")
    expected = (features.shape[0], max(0, features.shape[1] - 1))
    if tuple(pair_mask.shape) != expected:
        raise ValueError(f"pair_mask must have shape {expected}.")
    if features.shape[1] < 2:
        return features.new_zeros(expected)
    scores = 1.0 - F.cosine_similarity(
        features[:, :-1], features[:, 1:], dim=-1, eps=1.0e-8
    )
    return scores.clamp(0.0, 2.0).masked_fill(~pair_mask.bool(), 0.0)


def _physical_changepoint_anchors(
    outputs: Sequence[Mapping[str, torch.Tensor]],
    config: MotionPrimitiveLossConfig,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build label-free stable/change anchors from two physical sensor views.

    Quantiles are computed independently inside each trial, but a high-quantile
    candidate is accepted only when it also exceeds an absolute floor and a
    median-plus-MAD robust null threshold.  Consequently a complete trial may
    legitimately contain no change anchor.
    """

    descriptors = [
        _required(output, "window_physical_descriptors", index).detach()
        for index, output in enumerate(outputs)
    ]
    pair_masks = [
        _required(output, "boundary_pair_mask", index).bool()
        for index, output in enumerate(outputs)
    ]
    if descriptors[0].shape != descriptors[1].shape:
        raise ValueError("physical descriptor views must have equal shapes.")
    if pair_masks[0].shape != pair_masks[1].shape or not torch.equal(
        pair_masks[0], pair_masks[1]
    ):
        raise ValueError(
            "A2-MP changepoint supervision requires aligned full-trial views."
        )
    pair_mask = pair_masks[0]
    # View zero is the registered clean full-trial view.  The augmented second
    # view must learn the same boundary anchors, never redefine them.
    physical = descriptors[0]
    score = _adjacent_change_scores(physical, pair_mask)
    stable = torch.zeros_like(pair_mask)
    change = torch.zeros_like(pair_mask)
    for row in range(score.shape[0]):
        valid_values = score[row, pair_mask[row]]
        if valid_values.numel() == 0:
            continue
        low = torch.quantile(valid_values, float(config.changepoint_stable_quantile))
        high = torch.quantile(valid_values, float(config.changepoint_change_quantile))
        stable[row] = pair_mask[row] & (score[row] <= low)
        median = torch.median(valid_values)
        mad = torch.median(torch.abs(valid_values - median))
        robust_null = median + float(
            config.changepoint_null_mad_multiplier
        ) * 1.4826 * mad
        null_threshold = torch.maximum(
            robust_null,
            high.new_tensor(float(config.changepoint_absolute_floor)),
        )
        # The absolute/robust threshold, rather than a required q75-q25 gap,
        # lets a single salient transition survive in a long otherwise-stable
        # trial while still returning no changes for low-amplitude noise.  The
        # robust-null comparison is strict: a uniform non-zero sequence has
        # score == median and is not evidence of a changepoint.
        change[row] = (
            pair_mask[row]
            & (score[row] >= high)
            & (score[row] > null_threshold)
        )
        # Quantile ties must never make one pair both a stable and a change
        # anchor; that would give the ranking objective contradictory labels.
        stable[row] &= ~change[row]
    return stable, change, score


def _ranking_loss(
    scores: torch.Tensor,
    stable_mask: torch.Tensor,
    change_mask: torch.Tensor,
    margin: float,
) -> torch.Tensor:
    terms: list[torch.Tensor] = []
    for row_scores, row_stable, row_change in zip(
        scores, stable_mask, change_mask
    ):
        negatives = row_scores[row_stable]
        positives = row_scores[row_change]
        if positives.numel() and negatives.numel():
            terms.append(
                F.relu(float(margin) - positives[:, None] + negatives[None, :]).mean()
            )
    return _mean_or_zero(terms, _graph_zero(scores))


def _a2_local_losses(
    outputs: Sequence[Mapping[str, torch.Tensor]],
    config: MotionPrimitiveLossConfig,
    zero: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """A2-derived local objectives without A2's pooled trial classifier."""

    needs_boundary = (
        config.changepoint_weight > 0.0
        or config.content_boundary_alignment_weight > 0.0
    )
    if needs_boundary:
        stable, change, physical_score = _physical_changepoint_anchors(outputs, config)
        learned_logits = [
            _required(output, "learned_boundary_logits", index)
            for index, output in enumerate(outputs)
        ]
        anchor = stable | change
        target = change.to(learned_logits[0].dtype)
        cp_anchor_terms = [
            F.binary_cross_entropy_with_logits(logits[anchor], target[anchor])
            if torch.any(anchor)
            else _graph_zero(logits)
            for logits in learned_logits
        ]
        cp_equivariance = F.smooth_l1_loss(
            torch.sigmoid(learned_logits[0])[stable | change],
            torch.sigmoid(learned_logits[1])[stable | change],
        ) if torch.any(stable | change) else _graph_zero(*learned_logits)
        changepoint = 0.5 * (cp_anchor_terms[0] + cp_anchor_terms[1]) + float(
            config.changepoint_view_consistency_weight
        ) * cp_equivariance

        content_scores = [
            _adjacent_change_scores(
                _required(output, "primitive_features", index),
                _required(output, "boundary_pair_mask", index).bool(),
            )
            for index, output in enumerate(outputs)
        ]
        stable_term = 0.5 * (
            content_scores[0][stable].mean() + content_scores[1][stable].mean()
        ) if torch.any(stable) else _graph_zero(*content_scores)
        rank_term = 0.5 * (
            _ranking_loss(
                content_scores[0], stable, change, config.changepoint_rank_margin
            )
            + _ranking_loss(
                content_scores[1], stable, change, config.changepoint_rank_margin
            )
        )
        valid_pairs = _required(outputs[0], "boundary_pair_mask", 0).bool()
        content_equivariance = F.smooth_l1_loss(
            content_scores[0][valid_pairs], content_scores[1][valid_pairs]
        ) if torch.any(valid_pairs) else _graph_zero(*content_scores)
        content_boundary_alignment = (
            stable_term
            + rank_term
            + float(config.changepoint_view_consistency_weight)
            * content_equivariance
        )
        pair_mask = _required(
            outputs[0], "boundary_pair_mask", 0
        ).bool()
        valid_pair_denominator = pair_mask.sum().clamp_min(1)
        anchor_stable_rate = stable.sum().to(zero.dtype) / valid_pair_denominator
        anchor_change_rate = change.sum().to(zero.dtype) / valid_pair_denominator
        hard_boundary_starts = _required(
            outputs[0], "hard_boundary_starts", 0
        ).bool()
        hard_pair_boundaries = hard_boundary_starts[:, 1:] & pair_mask
        hard_boundary_rate = (
            hard_pair_boundaries.sum().to(zero.dtype) / valid_pair_denominator
        )
    else:
        physical_score = zero
        cp_equivariance = zero
        changepoint = zero
        stable_term = zero
        rank_term = zero
        content_equivariance = zero
        content_boundary_alignment = zero
        anchor_stable_rate = zero.detach()
        anchor_change_rate = zero.detach()
        hard_boundary_rate = zero.detach()

    if config.noncollapse_weight > 0.0:
        values = []
        for index, output in enumerate(outputs):
            features = _required(output, "primitive_features", index)
            mask = _required(output, "token_mask", index).bool()
            values.append(features[mask])
        flattened = torch.cat(values, dim=0)
        if flattened.shape[0] < 2:
            noncollapse_variance = zero
            noncollapse_covariance = zero
        else:
            centred = flattened - flattened.mean(dim=0, keepdim=True)
            std = torch.sqrt(centred.square().mean(dim=0) + 1.0e-4)
            noncollapse_variance = F.relu(1.0 - std).mean()
            covariance = centred.T @ centred / max(1, flattened.shape[0] - 1)
            off_diagonal = covariance - torch.diag_embed(torch.diagonal(covariance))
            dimension = int(flattened.shape[1])
            noncollapse_covariance = off_diagonal.square().sum() / max(
                1, dimension * (dimension - 1)
            )
        noncollapse = noncollapse_variance + noncollapse_covariance
    else:
        noncollapse_variance = zero
        noncollapse_covariance = zero
        noncollapse = zero

    if config.temporal_prediction_weight > 0.0:
        temporal_terms: list[torch.Tensor] = []
        for index, output in enumerate(outputs):
            prediction = _required(output, "temporal_prediction", index)
            target = _required(output, "temporal_target", index).detach()
            prediction_mask = _required(output, "temporal_prediction_mask", index).bool()
            token_mask = _required(output, "token_mask", index).bool()
            selected = prediction_mask & token_mask
            temporal_terms.append(
                (
                    1.0
                    - F.cosine_similarity(
                        prediction[selected], target[selected], dim=-1, eps=1.0e-8
                    )
                ).mean()
                if torch.any(selected)
                else _graph_zero(prediction)
            )
        temporal_prediction = torch.stack(temporal_terms).mean()
    else:
        temporal_prediction = zero

    return {
        "changepoint": changepoint,
        "changepoint_equivariance": cp_equivariance,
        "physical_changepoint_score_mean": (
            physical_score.mean()
            if isinstance(physical_score, torch.Tensor) and physical_score.numel()
            else zero
        ),
        "content_boundary_alignment": content_boundary_alignment,
        "content_boundary_stable": stable_term,
        "content_boundary_ranking": rank_term,
        "content_boundary_equivariance": content_equivariance,
        "changepoint_anchor_stable_rate": anchor_stable_rate.detach(),
        "changepoint_anchor_change_rate": anchor_change_rate.detach(),
        "hard_boundary_rate": hard_boundary_rate.detach(),
        "noncollapse": noncollapse,
        "noncollapse_variance": noncollapse_variance,
        "noncollapse_covariance": noncollapse_covariance,
        "temporal_prediction": temporal_prediction,
    }


def compose_joint_loss(
    outputs: Sequence[Mapping[str, torch.Tensor]],
    labels: torch.Tensor,
    visible_mask: torch.Tensor,
    config: MotionPrimitiveLossConfig,
    *,
    epoch: int,
    total_epochs: int,
    subject_ids: torch.Tensor | None = None,
) -> MotionPrimitiveLossResult:
    """Compose the sole HHR offline objective; the caller owns ``backward``."""

    del epoch
    if int(total_epochs) <= 0:
        raise ValueError("total_epochs must be positive.")
    cfg = config.validated()
    if len(outputs) != 2:
        raise ValueError("Motion-primitive training requires exactly two views.")
    visible = _visible_mask(labels, visible_mask, cfg.ignore_index)
    if subject_ids is not None:
        if subject_ids.shape != labels.shape:
            raise ValueError("subject_ids must have the same shape as labels.")
        subject_ids = subject_ids.to(device=labels.device, dtype=torch.long)

    first_logits = _required(outputs[0], "trajectory_logits", 0)
    second_logits = _required(outputs[1], "trajectory_logits", 1)
    if first_logits.shape != second_logits.shape:
        raise ValueError("trajectory logit views must have equal shapes.")
    logits = torch.cat((first_logits[visible], second_logits[visible]), dim=0)
    repeated_labels = torch.cat((labels[visible], labels[visible]), dim=0)
    trajectory_ce = F.cross_entropy(logits, repeated_labels)
    zero = _graph_zero(logits)

    if cfg.trajectory_supcon_weight > 0.0:
        visible_subjects = subject_ids[visible] if subject_ids is not None else None
        trajectory_supcon = _supervised_contrastive_loss(
            _required(outputs[0], "trajectory_embedding", 0)[visible],
            _required(outputs[1], "trajectory_embedding", 1)[visible],
            labels[visible],
            cfg,
            subject_ids=visible_subjects,
        )
    else:
        trajectory_supcon = zero
    if cfg.trajectory_view_consistency_weight > 0.0:
        trajectory_view_consistency = (
            1.0
            - F.cosine_similarity(
                _required(outputs[0], "trajectory_embedding", 0),
                _required(outputs[1], "trajectory_embedding", 1),
                dim=-1,
            )
        ).mean()
    else:
        trajectory_view_consistency = zero
    vq_commitment = (
        0.5
        * (
            _required(outputs[0], "commitment_loss", 0)
            + _required(outputs[1], "commitment_loss", 1)
        )
        if cfg.vq_commitment_weight > 0.0
        else zero
    )
    vq_codebook = (
        0.5
        * (
            _required(outputs[0], "codebook_embedding_loss", 0)
            + _required(outputs[1], "codebook_embedding_loss", 1)
        )
        if cfg.vq_codebook_weight > 0.0
        else zero
    )
    a2_components = _a2_local_losses(outputs, cfg, zero)

    duration_terms: list[torch.Tensor] = []
    utilization_terms: list[torch.Tensor] = []
    confidence_terms: list[torch.Tensor] = []
    boundary_terms: list[torch.Tensor] = []
    transition_terms: list[torch.Tensor] = []
    for view_index, output in enumerate(outputs):
        pair_mask = _required(output, "boundary_pair_mask", view_index).bool()
        final_boundary = _required(
            output, "final_boundary_probabilities", view_index
        )
        if cfg.effective_minimum_duration_weight > 0.0:
            duration_terms.append(
                effective_minimum_duration_loss(
                    final_boundary, pair_mask, cfg.minimum_primitive_windows
                )
            )
        needs_assignments = any(
            weight > 0.0
            for weight in (
                cfg.utilization_weight,
                cfg.assignment_confidence_weight,
                cfg.boundary_consistency_weight,
            )
        )
        if needs_assignments:
            assignments = _required(output, "trajectory_assignments", view_index)
            token_mask = _required(output, "token_mask", view_index).bool()
        if cfg.utilization_weight > 0.0:
            weights = token_mask.to(assignments.dtype)
            weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1.0)
            marginal = (assignments * weights.unsqueeze(-1)).sum(dim=1).mean(dim=0)
            marginal = marginal / marginal.sum().clamp_min(1.0e-12)
            entropy = -torch.sum(
                marginal * torch.log(marginal.clamp_min(1.0e-12))
            ) / math.log(float(marginal.numel()))
            utilization_terms.append(
                F.relu(entropy.new_tensor(cfg.utilization_entropy_floor) - entropy)
            )
        if cfg.assignment_confidence_weight > 0.0:
            posterior = _required(output, "assignment_probabilities", view_index)[
                token_mask
            ]
            entropy = -torch.sum(
                posterior * torch.log(posterior.clamp_min(1.0e-12)), dim=-1
            ) / math.log(float(posterior.shape[-1]))
            confidence_terms.append(
                F.relu(entropy.mean() - entropy.new_tensor(cfg.maximum_assignment_entropy))
            )
        if cfg.boundary_consistency_weight > 0.0:
            token_change = 1.0 - torch.sum(
                assignments[:, :-1] * assignments[:, 1:], dim=-1
            )
            boundary_terms.append(
                F.binary_cross_entropy(
                    final_boundary[pair_mask], token_change.detach()[pair_mask]
                )
                if torch.any(pair_mask)
                else _graph_zero(final_boundary)
            )
        if cfg.transition_budget_weight > 0.0:
            transition_terms.append(
                F.relu(
                    final_boundary[pair_mask].mean()
                    - final_boundary.new_tensor(cfg.maximum_transition_rate)
                )
                if torch.any(pair_mask)
                else _graph_zero(final_boundary)
            )

    effective_duration = _mean_or_zero(duration_terms, zero)
    utilization = _mean_or_zero(utilization_terms, zero)
    assignment_confidence = _mean_or_zero(confidence_terms, zero)
    boundary_consistency = _mean_or_zero(boundary_terms, zero)
    transition_budget = _mean_or_zero(transition_terms, zero)
    if cfg.codebook_diversity_weight > 0.0:
        codebook = _required(outputs[0], "normalized_codebook", 0)
        similarity = codebook @ codebook.T
        off_diagonal = ~torch.eye(
            len(codebook), dtype=torch.bool, device=similarity.device
        )
        codebook_diversity = F.relu(
            similarity[off_diagonal] - cfg.maximum_codebook_cosine
        ).square().mean()
    else:
        codebook_diversity = zero

    components = {
        "trajectory_ce": trajectory_ce,
        "trajectory_supcon": trajectory_supcon,
        "trajectory_view_consistency": trajectory_view_consistency,
        "vq_commitment": vq_commitment,
        "vq_codebook": vq_codebook,
        "effective_minimum_duration": effective_duration,
        "utilization": utilization,
        "assignment_confidence": assignment_confidence,
        "codebook_diversity": codebook_diversity,
        "boundary_consistency": boundary_consistency,
        "transition_budget": transition_budget,
        **a2_components,
    }
    motion_total = (
        cfg.trajectory_ce_weight * trajectory_ce
        + cfg.trajectory_supcon_weight * trajectory_supcon
        + cfg.trajectory_view_consistency_weight * trajectory_view_consistency
        + cfg.changepoint_weight * a2_components["changepoint"]
        + cfg.content_boundary_alignment_weight
        * a2_components["content_boundary_alignment"]
        + cfg.noncollapse_weight * a2_components["noncollapse"]
        + cfg.temporal_prediction_weight * a2_components["temporal_prediction"]
        + cfg.vq_commitment_weight * vq_commitment
        + cfg.vq_codebook_weight * vq_codebook
        + cfg.effective_minimum_duration_weight * effective_duration
        + cfg.utilization_weight * utilization
        + cfg.assignment_confidence_weight * assignment_confidence
        + cfg.codebook_diversity_weight * codebook_diversity
        + cfg.boundary_consistency_weight * boundary_consistency
        + cfg.transition_budget_weight * transition_budget
    )
    total = cfg.motion_total_weight * motion_total
    return MotionPrimitiveLossResult(
        total=total,
        motion_total=motion_total,
        components=components,
        visible_trial_count=int(visible.sum().item()),
    )


# Short neutral aliases retain the public trajectory-only interface.
JointLossConfig = MotionPrimitiveLossConfig
JointLossResult = MotionPrimitiveLossResult


__all__ = [
    "JointLossConfig",
    "JointLossResult",
    "MotionPrimitiveLossConfig",
    "MotionPrimitiveLossResult",
    "MaskedTemporalPredictor",
    "compose_joint_loss",
    "effective_minimum_duration_loss",
    "sample_temporal_prediction_mask",
]
