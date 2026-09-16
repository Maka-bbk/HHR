"""Encoder and losses for motion-primitive boundary representation learning.

This module deliberately has no dependency on the trajectory readout code.  It
contains the trainable representation, the exact PyTorch counterpart of the
feature-space change score used by :mod:`segmentation`, and small, independently
ablatable loss functions.

Conventions
-----------
``N`` is a flat window batch, ``B`` is a trial batch, ``L`` is the padded number
of windows in a trial, and ``D`` is a feature dimension.  Padded trial masks
must be left aligned (valid windows followed by padding).  A boundary at index
``t`` lies between windows ``t`` and ``t + 1`` and therefore has shape
``[B, L - 1]``.
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

try:  # Running with HHR as the working directory (the project's usual mode).
    from models.resnet1d import ResNet1D
except ModuleNotFoundError:  # Importing as HHR.experiments.motion_primitive.
    from ...models.resnet1d import ResNet1D


TensorDict = Dict[str, torch.Tensor]


def _positive_int(name: str, value: int) -> int:
    if isinstance(value, bool) or int(value) != value or int(value) < 1:
        raise ValueError(f"{name} must be a positive integer, got {value!r}.")
    return int(value)


def _floating_tensor(name: str, value: torch.Tensor, ndim: int) -> None:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor, got {type(value).__name__}.")
    if value.ndim != ndim:
        raise ValueError(f"{name} must have {ndim} dimensions, got {tuple(value.shape)}.")
    if not torch.is_floating_point(value):
        raise TypeError(f"{name} must have floating dtype, got {value.dtype}.")


def _prefix_mask(
    mask: Optional[torch.Tensor],
    shape: Tuple[int, int],
    device: torch.device,
    *,
    allow_empty_trials: bool,
    name: str = "valid_mask",
) -> torch.Tensor:
    if mask is None:
        return torch.ones(shape, dtype=torch.bool, device=device)
    if not isinstance(mask, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor, got {type(mask).__name__}.")
    if tuple(mask.shape) != shape:
        raise ValueError(f"{name} must have shape {shape}, got {tuple(mask.shape)}.")
    if mask.device != device:
        raise ValueError(f"{name} must be on {device}, got {mask.device}.")
    mask = mask.bool()
    # A false-to-true transition means valid values occur after padding.  Local
    # boundary contexts are only unambiguous for the standard left-padded form.
    if shape[1] > 1 and torch.any((~mask[:, :-1]) & mask[:, 1:]):
        raise ValueError(f"{name} must be left aligned (valid values, then padding).")
    if not allow_empty_trials and torch.any(mask.sum(dim=1) == 0):
        raise ValueError(f"{name} contains an empty trial.")
    return mask


def _mask_like(name: str, mask: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    if not isinstance(mask, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor, got {type(mask).__name__}.")
    if tuple(mask.shape) != tuple(reference.shape):
        raise ValueError(
            f"{name} must have shape {tuple(reference.shape)}, got {tuple(mask.shape)}."
        )
    if mask.device != reference.device:
        raise ValueError(f"{name} must be on {reference.device}, got {mask.device}.")
    return mask.bool()


def _differentiable_zero(*values: torch.Tensor) -> torch.Tensor:
    """Return scalar zero connected to all supplied computation graphs."""

    if not values:
        raise ValueError("At least one tensor is required to construct a zero loss.")
    zero = values[0].sum() * 0.0
    for value in values[1:]:
        zero = zero + value.sum() * 0.0
    return zero


class ProjectionHead(nn.Module):
    """Small LayerNorm-MLP projection used by the three representation heads."""

    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int) -> None:
        super().__init__()
        input_dim = _positive_int("input_dim", input_dim)
        output_dim = _positive_int("output_dim", output_dim)
        hidden_dim = _positive_int("hidden_dim", hidden_dim)
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features)


class MaskedTemporalPredictor(nn.Module):
    """Predict masked content embeddings from their short temporal context.

    Masked positions are replaced by zero and an explicit binary mask channel
    is appended before two temporal convolutions.  Consequently the predictor
    cannot copy its target embedding directly.  The module predicts every
    position; :func:`masked_temporal_prediction_loss` selects only requested
    valid targets.
    """

    def __init__(
        self,
        feature_dim: int,
        hidden_dim: Optional[int] = None,
        kernel_size: int = 3,
    ) -> None:
        super().__init__()
        self.feature_dim = _positive_int("feature_dim", feature_dim)
        self.hidden_dim = _positive_int(
            "hidden_dim", self.feature_dim if hidden_dim is None else hidden_dim
        )
        self.kernel_size = _positive_int("kernel_size", kernel_size)
        if self.kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd so temporal length is preserved.")
        padding = self.kernel_size // 2
        self.network = nn.Sequential(
            nn.Conv1d(self.feature_dim + 1, self.hidden_dim, self.kernel_size, padding=padding),
            nn.GELU(),
            nn.Conv1d(self.hidden_dim, self.feature_dim, self.kernel_size, padding=padding),
        )

    def forward(
        self,
        content: torch.Tensor,
        prediction_mask: torch.Tensor,
        valid_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        _floating_tensor("content", content, 3)
        batch_size, length, feature_dim = content.shape
        if feature_dim != self.feature_dim:
            raise ValueError(
                f"content feature dimension must be {self.feature_dim}, got {feature_dim}."
            )
        valid = _prefix_mask(
            valid_mask,
            (batch_size, length),
            content.device,
            allow_empty_trials=True,
        )
        prediction = _mask_like("prediction_mask", prediction_mask, valid)
        if torch.any(prediction & ~valid):
            raise ValueError("prediction_mask selects padded positions.")

        hidden = content.masked_fill((prediction | ~valid).unsqueeze(-1), 0.0)
        mask_channel = prediction.to(content.dtype).unsqueeze(-1)
        predictor_input = torch.cat((hidden, mask_channel), dim=-1).transpose(1, 2)
        output = self.network(predictor_input).transpose(1, 2)
        return output.masked_fill((~valid).unsqueeze(-1), 0.0)


class MotionPrimitiveEncoder(nn.Module):
    """Shared ResNet1D with segmentation, content, augmentation and trial heads.

    Flat input ``[N,C,T]`` returns per-window tensors.  Padded trial input
    ``[B,L,C,T]`` additionally returns a masked mean/robust-peak trial embedding
    and old-class auxiliary logits.  Segmentation and augmentation projections
    are returned both before (``*_raw``) and after L2 normalisation.  Change
    scores/InfoNCE use the normalised tensors, whereas VICReg-style variance and
    covariance regularisation must use ``*_raw``: unit-sphere coordinates cannot
    all attain the default target standard deviation of one.  The content
    projection remains metric-unconstrained for state-residual and temporal-
    prediction objectives.  With ``content_residual=True`` (the experiment
    default), its zero-initialised delta is added to the pretrained backbone,
    so the codebook path initially preserves the legacy local representation.
    The segmentation path uses the same residual initialisation by default.
    When its dimension equals ``backbone_dim`` (required by the formal trainer),
    initial boundary features are exactly the L2-normalised legacy backbone
    features rather than an untrained random projection.
    """

    def __init__(
        self,
        in_channels: int = 6,
        backbone_dim: int = 256,
        base_channels: int = 64,
        backbone_layers: Optional[list[int]] = None,
        backbone_dropout: float = 0.0,
        segmentation_dim: int = 128,
        segmentation_residual: bool = True,
        content_dim: int = 256,
        content_residual: bool = True,
        augmentation_dim: int = 128,
        projection_hidden_dim: int = 256,
        num_classes: int = 6,
        trial_hidden_dim: int = 128,
        trial_peak_quantile: float = 0.90,
        trial_dropout: float = 0.0,
        predictor_hidden_dim: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.in_channels = _positive_int("in_channels", in_channels)
        self.backbone_dim = _positive_int("backbone_dim", backbone_dim)
        self.segmentation_dim = _positive_int("segmentation_dim", segmentation_dim)
        self.segmentation_residual = bool(segmentation_residual)
        self.content_dim = _positive_int("content_dim", content_dim)
        self.content_residual = bool(content_residual)
        self.augmentation_dim = _positive_int("augmentation_dim", augmentation_dim)
        projection_hidden_dim = _positive_int(
            "projection_hidden_dim", projection_hidden_dim
        )
        self.num_classes = _positive_int("num_classes", num_classes)
        trial_hidden_dim = _positive_int("trial_hidden_dim", trial_hidden_dim)
        self.trial_peak_quantile = float(trial_peak_quantile)
        if not 0.5 <= self.trial_peak_quantile <= 1.0:
            raise ValueError("trial_peak_quantile must be in [0.5, 1.0].")
        if not 0.0 <= float(trial_dropout) < 1.0:
            raise ValueError("trial_dropout must be in [0, 1).")
        if not 0.0 <= float(backbone_dropout) < 1.0:
            raise ValueError("backbone_dropout must be in [0, 1).")

        layers = [2, 2, 2] if backbone_layers is None else list(backbone_layers)
        if len(layers) != 3 or any(_positive_int("backbone layer", x) < 1 for x in layers):
            raise ValueError("backbone_layers must contain three positive integers.")
        self.backbone = ResNet1D(
            in_channels=self.in_channels,
            feat_dim=self.backbone_dim,
            base_channels=_positive_int("base_channels", base_channels),
            layers=layers,
            dropout=float(backbone_dropout),
        )
        self.segmentation_head = ProjectionHead(
            self.backbone_dim, self.segmentation_dim, projection_hidden_dim
        )
        if self.segmentation_residual:
            if self.backbone_dim == self.segmentation_dim:
                self.segmentation_skip = nn.Identity()
            else:
                # Direct construction supports a deterministic coordinate
                # projection for small unit tests/custom research.  The formal
                # A0-A4 trainer requires equal dimensions so its skip is exact.
                projection = nn.Linear(
                    self.backbone_dim, self.segmentation_dim, bias=False
                )
                with torch.no_grad():
                    projection.weight.zero_()
                    diagonal = min(self.backbone_dim, self.segmentation_dim)
                    projection.weight[:diagonal, :diagonal].copy_(
                        torch.eye(diagonal, dtype=projection.weight.dtype)
                    )
                projection.weight.requires_grad_(False)
                self.segmentation_skip = projection
            segmentation_delta = self.segmentation_head.net[-1]
            nn.init.zeros_(segmentation_delta.weight)
            if segmentation_delta.bias is not None:
                nn.init.zeros_(segmentation_delta.bias)
        else:
            self.segmentation_skip = None
        self.content_head = ProjectionHead(
            self.backbone_dim, self.content_dim, projection_hidden_dim
        )
        if self.content_residual:
            self.content_skip = (
                nn.Identity()
                if self.backbone_dim == self.content_dim
                else nn.Linear(self.backbone_dim, self.content_dim, bias=False)
            )
            # Start exactly from the pretrained backbone representation when
            # dimensions match.  The trainable branch then learns only the
            # task-specific residual instead of erasing useful local phase on
            # the first optimisation step.
            final_delta = self.content_head.net[-1]
            nn.init.zeros_(final_delta.weight)
            if final_delta.bias is not None:
                nn.init.zeros_(final_delta.bias)
        else:
            self.content_skip = None
        self.augmentation_head = ProjectionHead(
            self.backbone_dim, self.augmentation_dim, projection_hidden_dim
        )

        # A train-only trial auxiliary head.  Pooling is deliberately fixed and
        # low capacity so it cannot replace the trajectory readout at evaluation.
        self.trial_fusion = nn.Sequential(
            nn.LayerNorm(2 * self.content_dim),
            nn.Linear(2 * self.content_dim, trial_hidden_dim),
            nn.GELU(),
            nn.Dropout(float(trial_dropout)),
            nn.Linear(trial_hidden_dim, self.content_dim),
            nn.LayerNorm(self.content_dim),
        )
        self.trial_classifier = nn.Linear(self.content_dim, self.num_classes)
        self.temporal_predictor = MaskedTemporalPredictor(
            self.content_dim, predictor_hidden_dim
        )

    def _encode_flat(self, windows: torch.Tensor) -> TensorDict:
        _floating_tensor("windows", windows, 3)
        if windows.shape[0] < 1 or windows.shape[2] < 1:
            raise ValueError("windows must contain at least one item and time sample.")
        if windows.shape[1] != self.in_channels:
            raise ValueError(
                f"windows channel dimension must be {self.in_channels}, got {windows.shape[1]}."
            )
        backbone = self.backbone(windows)
        segmentation_delta = self.segmentation_head(backbone)
        segmentation_raw = (
            self.segmentation_skip(backbone) + segmentation_delta
            if self.segmentation_skip is not None
            else segmentation_delta
        )
        segmentation = F.normalize(segmentation_raw, dim=-1)
        content_delta = self.content_head(backbone)
        content = (
            self.content_skip(backbone) + content_delta
            if self.content_skip is not None
            else content_delta
        )
        augmentation_raw = self.augmentation_head(backbone)
        augmentation = F.normalize(augmentation_raw, dim=-1)
        return {
            "backbone": backbone,
            "segmentation_raw": segmentation_raw,
            "segmentation": segmentation,
            "content": content,
            "content_delta": content_delta,
            "augmentation_raw": augmentation_raw,
            "augmentation": augmentation,
        }

    def encode_windows(self, windows: torch.Tensor) -> TensorDict:
        """Encode a non-padded flat batch shaped ``[N,C,T]``."""

        return self._encode_flat(windows)

    def encode_trials(
        self,
        windows: torch.Tensor,
        valid_mask: Optional[torch.Tensor] = None,
    ) -> TensorDict:
        """Encode padded trials shaped ``[B,L,C,T]`` without processing padding."""

        _floating_tensor("windows", windows, 4)
        batch_size, length, channels, time = windows.shape
        if batch_size < 1 or length < 1 or time < 1:
            raise ValueError("windows must have non-empty batch, trial and time dimensions.")
        if channels != self.in_channels:
            raise ValueError(
                f"windows channel dimension must be {self.in_channels}, got {channels}."
            )
        valid = _prefix_mask(
            valid_mask,
            (batch_size, length),
            windows.device,
            allow_empty_trials=False,
        )
        flat_windows = windows.reshape(batch_size * length, channels, time)
        flat_valid = valid.reshape(-1)
        encoded_valid = self._encode_flat(flat_windows[flat_valid])

        encoded: TensorDict = {}
        for name, values in encoded_valid.items():
            output = values.new_zeros((batch_size * length, values.shape[-1]))
            output[flat_valid] = values
            encoded[name] = output.reshape(batch_size, length, values.shape[-1])

        content = encoded["content"]
        counts = valid.sum(dim=1, keepdim=True).to(content.dtype)
        mean = (content * valid.unsqueeze(-1).to(content.dtype)).sum(dim=1) / counts
        # Feature-wise q-th order statistic; padded values never enter quantile.
        peak = torch.stack(
            [
                torch.quantile(row[row_mask], self.trial_peak_quantile, dim=0)
                for row, row_mask in zip(content, valid)
            ],
            dim=0,
        )
        trial_embedding = self.trial_fusion(torch.cat((mean, peak), dim=-1))
        encoded["trial_embedding"] = trial_embedding
        encoded["trial_logits"] = self.trial_classifier(trial_embedding)
        encoded["valid_mask"] = valid
        return encoded

    def predict_masked_content(
        self,
        content: torch.Tensor,
        prediction_mask: torch.Tensor,
        valid_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return self.temporal_predictor(content, prediction_mask, valid_mask)

    def forward(
        self,
        windows: torch.Tensor,
        valid_mask: Optional[torch.Tensor] = None,
    ) -> TensorDict:
        if not isinstance(windows, torch.Tensor):
            raise TypeError(
                f"windows must be a torch.Tensor, got {type(windows).__name__}."
            )
        if windows.ndim == 3:
            if valid_mask is not None:
                raise ValueError("valid_mask is only accepted with padded [B,L,C,T] input.")
            return self.encode_windows(windows)
        if windows.ndim == 4:
            return self.encode_trials(windows, valid_mask)
        raise ValueError(
            "windows must have shape [N,C,T] or [B,L,C,T], got "
            f"{tuple(windows.shape)}."
        )


def batched_feature_change_scores(
    features: torch.Tensor,
    context_windows: int,
    valid_mask: Optional[torch.Tensor] = None,
    eps: float = 1e-8,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute local-mean cosine change scores for padded trial batches.

    This is numerically equivalent to ``segmentation.feature_change_scores``
    for every unpadded trial, including its truncated contexts at both ends.
    Invalid padded boundaries receive score zero and are identified by the
    returned boolean mask.
    """

    _floating_tensor("features", features, 3)
    batch_size, length, feature_dim = features.shape
    if batch_size < 1 or feature_dim < 1 or length < 1:
        raise ValueError("features must have non-empty [B,L,D] dimensions.")
    radius = _positive_int("context_windows", context_windows)
    eps = float(eps)
    if not math.isfinite(eps) or eps <= 0.0:
        raise ValueError("eps must be finite and positive.")
    valid = _prefix_mask(
        valid_mask,
        (batch_size, length),
        features.device,
        allow_empty_trials=True,
    )
    if length == 1:
        return features.new_zeros((batch_size, 0)), valid.new_zeros((batch_size, 0))

    scores = []
    boundary_validity = []
    for boundary in range(1, length):
        left_slice = slice(max(0, boundary - radius), boundary)
        right_slice = slice(boundary, min(length, boundary + radius))
        left_mask = valid[:, left_slice]
        right_mask = valid[:, right_slice]
        left_count = left_mask.sum(dim=1, keepdim=True)
        right_count = right_mask.sum(dim=1, keepdim=True)
        boundary_valid = (
            valid[:, boundary - 1]
            & valid[:, boundary]
            & (left_count[:, 0] > 0)
            & (right_count[:, 0] > 0)
        )
        left = (
            features[:, left_slice]
            * left_mask.unsqueeze(-1).to(features.dtype)
        ).sum(dim=1) / left_count.clamp_min(1).to(features.dtype)
        right = (
            features[:, right_slice]
            * right_mask.unsqueeze(-1).to(features.dtype)
        ).sum(dim=1) / right_count.clamp_min(1).to(features.dtype)
        denominator = (left.norm(dim=-1) * right.norm(dim=-1)).clamp_min(eps)
        cosine = ((left * right).sum(dim=-1) / denominator).clamp(-1.0, 1.0)
        score = 1.0 - cosine
        scores.append(torch.where(boundary_valid, score, torch.zeros_like(score)))
        boundary_validity.append(boundary_valid)
    return torch.stack(scores, dim=1), torch.stack(boundary_validity, dim=1)


def feature_change_scores_torch(
    features: torch.Tensor,
    context_windows: int,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Unbatched ``[L,D]`` convenience wrapper matching the NumPy function."""

    _floating_tensor("features", features, 2)
    if features.shape[0] < 1 or features.shape[1] < 1:
        raise ValueError("features must be a non-empty [L,D] tensor.")
    scores, _ = batched_feature_change_scores(
        features.unsqueeze(0), context_windows=context_windows, eps=eps
    )
    return scores[0]


def symmetric_info_nce(
    view_one: torch.Tensor,
    view_two: torch.Tensor,
    temperature: float = 0.2,
    trial_ids: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Symmetric cross-view InfoNCE with exactly one anchor per trial.

    ``trial_ids`` is optional but, when supplied, must be unique.  This explicit
    check prevents overlapping windows from the same trial entering the batch as
    false negatives.  A one-trial batch has no negatives and returns graph-linked
    zero rather than the vacuous zero-valued cross entropy.
    """

    _floating_tensor("view_one", view_one, 2)
    _floating_tensor("view_two", view_two, 2)
    if tuple(view_one.shape) != tuple(view_two.shape):
        raise ValueError(
            f"InfoNCE views must have identical shape, got {tuple(view_one.shape)} "
            f"and {tuple(view_two.shape)}."
        )
    if view_one.device != view_two.device or view_one.dtype != view_two.dtype:
        raise ValueError("InfoNCE views must have the same device and dtype.")
    if view_one.shape[1] < 1:
        raise ValueError("InfoNCE projections must have a non-empty feature dimension.")
    temperature = float(temperature)
    if not math.isfinite(temperature) or temperature <= 0.0:
        raise ValueError("temperature must be finite and positive.")
    batch_size = view_one.shape[0]
    if trial_ids is not None:
        if not isinstance(trial_ids, torch.Tensor):
            raise TypeError("trial_ids must be a torch.Tensor.")
        if tuple(trial_ids.shape) != (batch_size,):
            raise ValueError(f"trial_ids must have shape ({batch_size},).")
        if trial_ids.device != view_one.device:
            raise ValueError("trial_ids must be on the same device as the views.")
        if torch.unique(trial_ids).numel() != batch_size:
            raise ValueError("InfoNCE requires at most one anchor window per trial.")
    if batch_size < 2:
        return _differentiable_zero(view_one, view_two)

    first = F.normalize(view_one, dim=-1)
    second = F.normalize(view_two, dim=-1)
    logits = first @ second.transpose(0, 1) / temperature
    targets = torch.arange(batch_size, device=view_one.device)
    return 0.5 * (
        F.cross_entropy(logits, targets)
        + F.cross_entropy(logits.transpose(0, 1), targets)
    )


def variance_covariance_terms(
    features: torch.Tensor,
    valid_mask: Optional[torch.Tensor] = None,
    target_std: float = 1.0,
    eps: float = 1e-4,
) -> TensorDict:
    """VICReg-style variance and off-diagonal covariance penalties.

    With the default ``target_std=1``, pass an *unnormalised* projection (for
    example ``encoder_output["segmentation_raw"]``), not per-sample unit vectors.
    If unit-normalised features are intentionally supplied, their feasible
    coordinate scale is approximately ``1 / sqrt(D)`` and ``target_std`` must be
    chosen explicitly and recorded in the experiment configuration.
    """

    if not isinstance(features, torch.Tensor) or features.ndim not in (2, 3):
        shape = tuple(features.shape) if isinstance(features, torch.Tensor) else None
        raise ValueError(f"features must have shape [N,D] or [B,L,D], got {shape}.")
    if not torch.is_floating_point(features):
        raise TypeError("features must have floating dtype.")
    target_std = float(target_std)
    eps = float(eps)
    if not math.isfinite(target_std) or target_std <= 0.0:
        raise ValueError("target_std must be finite and positive.")
    if not math.isfinite(eps) or eps <= 0.0:
        raise ValueError("eps must be finite and positive.")

    if features.ndim == 2:
        if valid_mask is not None:
            if not isinstance(valid_mask, torch.Tensor):
                raise TypeError("valid_mask must be a torch.Tensor.")
            if tuple(valid_mask.shape) != (features.shape[0],):
                raise ValueError("A flat valid_mask must have shape [N].")
            if valid_mask.device != features.device:
                raise ValueError("valid_mask must be on the same device as features.")
            values = features[valid_mask.bool()]
        else:
            values = features
    else:
        valid = _prefix_mask(
            valid_mask,
            tuple(features.shape[:2]),
            features.device,
            allow_empty_trials=True,
        )
        values = features[valid]
    if values.shape[-1] < 1:
        raise ValueError("features must have a non-empty feature dimension.")
    if values.shape[0] < 2:
        zero = _differentiable_zero(features)
        return {"variance": zero, "covariance": zero, "loss": zero}

    centred = values - values.mean(dim=0, keepdim=True)
    variance = centred.square().sum(dim=0) / (values.shape[0] - 1)
    std = torch.sqrt(variance + eps)
    variance_loss = F.relu(target_std - std).mean()

    covariance = centred.transpose(0, 1) @ centred / (values.shape[0] - 1)
    diagonal = torch.diagonal(covariance)
    off_diagonal = covariance - torch.diag_embed(diagonal)
    off_diagonal_square_sum = off_diagonal.square().sum()
    feature_dim = int(values.shape[1])
    # Use the mean over off-diagonal entries, not VICReg's sum/D form.  The
    # latter grows approximately linearly with representation dimension and
    # overwhelmed every other objective at D=256 in the real USC-HAD smoke
    # audit.  The mean is dimension-stable while retaining the same minimum.
    covariance_loss = off_diagonal_square_sum / max(
        1, feature_dim * (feature_dim - 1)
    )
    return {
        "variance": variance_loss,
        "covariance": covariance_loss,
        "loss": variance_loss + covariance_loss,
    }


def variance_covariance_loss(
    features: torch.Tensor,
    valid_mask: Optional[torch.Tensor] = None,
    target_std: float = 1.0,
    eps: float = 1e-4,
) -> torch.Tensor:
    return variance_covariance_terms(features, valid_mask, target_std, eps)["loss"]


def stable_boundary_loss(scores: torch.Tensor, stable_mask: torch.Tensor) -> torch.Tensor:
    """Drive high-confidence stable-boundary scores toward zero."""

    _floating_tensor("scores", scores, 2)
    stable = _mask_like("stable_mask", stable_mask, scores)
    if not torch.any(stable):
        return _differentiable_zero(scores)
    return scores[stable].mean()


def boundary_ranking_loss(
    scores: torch.Tensor,
    change_mask: torch.Tensor,
    stable_mask: torch.Tensor,
    margin: float = 0.2,
) -> torch.Tensor:
    """Rank change candidates above stable candidates within each trial."""

    _floating_tensor("scores", scores, 2)
    change = _mask_like("change_mask", change_mask, scores)
    stable = _mask_like("stable_mask", stable_mask, scores)
    if torch.any(change & stable):
        raise ValueError("change_mask and stable_mask must be disjoint.")
    margin = float(margin)
    if not math.isfinite(margin) or margin < 0.0:
        raise ValueError("margin must be finite and non-negative.")
    per_trial = []
    for row, row_change, row_stable in zip(scores, change, stable):
        positives = row[row_change]
        negatives = row[row_stable]
        if positives.numel() and negatives.numel():
            per_trial.append(
                F.relu(margin - positives[:, None] + negatives[None, :]).mean()
            )
    if not per_trial:
        return _differentiable_zero(scores)
    return torch.stack(per_trial).mean()


def cross_view_boundary_loss(
    scores_one: torch.Tensor,
    scores_two: torch.Tensor,
    consistency_mask: Optional[torch.Tensor] = None,
    delta: float = 1.0,
) -> torch.Tensor:
    """Huber consistency of aligned boundary scores from two augmented views.

    Inputs must already share the same time coordinates.  For independently
    shifted/cropped views, undo the known augmentation mapping before calling
    this function (or omit temporal shifts from the boundary-consistency views).
    """

    _floating_tensor("scores_one", scores_one, 2)
    _floating_tensor("scores_two", scores_two, 2)
    if tuple(scores_one.shape) != tuple(scores_two.shape):
        raise ValueError("Cross-view boundary score tensors must have identical shape.")
    if scores_one.device != scores_two.device or scores_one.dtype != scores_two.dtype:
        raise ValueError("Cross-view boundary scores must have the same device and dtype.")
    delta = float(delta)
    if not math.isfinite(delta) or delta <= 0.0:
        raise ValueError("delta must be finite and positive.")
    if consistency_mask is None:
        selected = torch.ones_like(scores_one, dtype=torch.bool)
    else:
        selected = _mask_like("consistency_mask", consistency_mask, scores_one)
    if not torch.any(selected):
        return _differentiable_zero(scores_one, scores_two)
    return F.huber_loss(
        scores_one[selected], scores_two[selected], reduction="mean", delta=delta
    )


def boundary_loss_terms(
    segmentation_view_one: torch.Tensor,
    segmentation_view_two: torch.Tensor,
    stable_mask: torch.Tensor,
    change_mask: torch.Tensor,
    valid_mask: Optional[torch.Tensor] = None,
    context_windows: int = 2,
    rank_margin: float = 0.2,
    equivariance_delta: float = 1.0,
    equivariance_weight: float = 0.5,
) -> TensorDict:
    """Joint stable, ranking and aligned cross-view boundary objective."""

    _floating_tensor("segmentation_view_one", segmentation_view_one, 3)
    _floating_tensor("segmentation_view_two", segmentation_view_two, 3)
    if tuple(segmentation_view_one.shape) != tuple(segmentation_view_two.shape):
        raise ValueError("Segmentation views must have identical [B,L,D] shape.")
    if (
        segmentation_view_one.device != segmentation_view_two.device
        or segmentation_view_one.dtype != segmentation_view_two.dtype
    ):
        raise ValueError("Segmentation views must have the same device and dtype.")
    scores_one, valid_one = batched_feature_change_scores(
        segmentation_view_one, context_windows, valid_mask
    )
    scores_two, valid_two = batched_feature_change_scores(
        segmentation_view_two, context_windows, valid_mask
    )
    stable = _mask_like("stable_mask", stable_mask, scores_one)
    change = _mask_like("change_mask", change_mask, scores_one)
    valid_boundary = valid_one & valid_two
    if torch.any((stable | change) & ~valid_boundary):
        raise ValueError("Boundary supervision masks select a padded/invalid boundary.")
    if torch.any(stable & change):
        raise ValueError("stable_mask and change_mask must be disjoint.")
    equivariance_weight = float(equivariance_weight)
    if not math.isfinite(equivariance_weight) or equivariance_weight < 0.0:
        raise ValueError("equivariance_weight must be finite and non-negative.")

    stable_term = 0.5 * (
        stable_boundary_loss(scores_one, stable)
        + stable_boundary_loss(scores_two, stable)
    )
    ranking_term = 0.5 * (
        boundary_ranking_loss(scores_one, change, stable, rank_margin)
        + boundary_ranking_loss(scores_two, change, stable, rank_margin)
    )
    equivariance_term = cross_view_boundary_loss(
        scores_one, scores_two, valid_boundary, equivariance_delta
    )
    total = stable_term + ranking_term + equivariance_weight * equivariance_term
    return {
        "loss": total,
        "stable": stable_term,
        "ranking": ranking_term,
        "equivariance": equivariance_term,
        "scores_one": scores_one,
        "scores_two": scores_two,
        "valid_boundary_mask": valid_boundary,
    }


def masked_temporal_prediction_loss(
    prediction: torch.Tensor,
    teacher_target: torch.Tensor,
    prediction_mask: torch.Tensor,
    valid_mask: Optional[torch.Tensor] = None,
    loss_type: str = "cosine",
    stop_gradient_target: bool = True,
    huber_delta: float = 1.0,
) -> torch.Tensor:
    """Loss on valid masked positions against an optional stop-gradient teacher.

    ``teacher_target`` should be a clean-view embedding produced by an EMA
    teacher.  Detaching the online encoder's own target prevents direct target
    gradients but does not provide the stabilisation of an EMA teacher.
    """

    _floating_tensor("prediction", prediction, 3)
    _floating_tensor("teacher_target", teacher_target, 3)
    if tuple(prediction.shape) != tuple(teacher_target.shape):
        raise ValueError("prediction and teacher_target must have identical [B,L,D] shape.")
    if prediction.device != teacher_target.device or prediction.dtype != teacher_target.dtype:
        raise ValueError("prediction and teacher_target must have the same device and dtype.")
    valid = _prefix_mask(
        valid_mask,
        tuple(prediction.shape[:2]),
        prediction.device,
        allow_empty_trials=True,
    )
    selected = _mask_like("prediction_mask", prediction_mask, valid)
    if torch.any(selected & ~valid):
        raise ValueError("prediction_mask selects padded positions.")
    selected = selected & valid
    if not torch.any(selected):
        return _differentiable_zero(prediction, teacher_target)
    target = teacher_target.detach() if stop_gradient_target else teacher_target
    prediction_values = prediction[selected]
    target_values = target[selected]
    loss_type = str(loss_type).lower()
    if loss_type == "cosine":
        return (1.0 - F.cosine_similarity(prediction_values, target_values, dim=-1)).mean()
    if loss_type == "huber":
        huber_delta = float(huber_delta)
        if not math.isfinite(huber_delta) or huber_delta <= 0.0:
            raise ValueError("huber_delta must be finite and positive.")
        return F.huber_loss(
            prediction_values, target_values, reduction="mean", delta=huber_delta
        )
    raise ValueError("loss_type must be 'cosine' or 'huber'.")


__all__ = [
    "MaskedTemporalPredictor",
    "MotionPrimitiveEncoder",
    "ProjectionHead",
    "batched_feature_change_scores",
    "boundary_loss_terms",
    "boundary_ranking_loss",
    "cross_view_boundary_loss",
    "feature_change_scores_torch",
    "masked_temporal_prediction_loss",
    "stable_boundary_loss",
    "symmetric_info_nce",
    "variance_covariance_loss",
    "variance_covariance_terms",
]
