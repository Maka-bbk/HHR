"""HAR-only window warm-up components used before the A2 encoder.

The module contains only the small ResNet1D projection/classification stack and
the two losses actually active in the validated USC-HAD checkpoint.  No image
backbone, torchvision registry, or generic Happy training dependency is
imported.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class WindowProjectionHead(nn.Module):
    """State-dict-compatible form of the historical three-layer DINO head."""

    def __init__(
        self,
        in_dim: int = 256,
        out_dim: int = 6,
        *,
        hidden_dim: int = 2048,
        bottleneck_dim: int = 256,
        norm_last_layer: bool = True,
    ) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(int(in_dim), int(hidden_dim)),
            nn.GELU(),
            nn.Linear(int(hidden_dim), int(hidden_dim)),
            nn.GELU(),
            nn.Linear(int(hidden_dim), int(bottleneck_dim)),
        )
        self.apply(self._init_weights)
        # Keep ``in_dim`` here for exact compatibility with the source model.
        self.last_layer = nn.utils.weight_norm(
            nn.Linear(int(in_dim), int(out_dim), bias=False)
        )
        self.last_layer.weight_g.data.fill_(1.0)
        if bool(norm_last_layer):
            self.last_layer.weight_g.requires_grad = False

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0.0)

    def forward(self, values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        projection = self.mlp(values)
        logits = self.last_layer(F.normalize(values, dim=-1, p=2))
        return projection, logits


class SupervisedContrastiveLoss(nn.Module):
    """Two-view supervised contrastive loss used by the window warm-up."""

    def __init__(self, temperature: float = 0.07, base_temperature: float = 0.07) -> None:
        super().__init__()
        self.temperature = float(temperature)
        self.base_temperature = float(base_temperature)

    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        if features.ndim != 3 or features.shape[1] != 2:
            raise ValueError("SupervisedContrastiveLoss expects [B,2,D].")
        batch_size = int(features.shape[0])
        labels = labels.contiguous().view(-1, 1)
        if len(labels) != batch_size:
            raise ValueError("Labels and features differ in batch size.")
        positive = torch.eq(labels, labels.T).to(dtype=features.dtype, device=features.device)
        contrast = torch.cat(torch.unbind(features, dim=1), dim=0)
        logits = contrast @ contrast.T / self.temperature
        logits = logits - logits.max(dim=1, keepdim=True).values.detach()
        positive = positive.repeat(2, 2)
        self_mask = torch.eye(2 * batch_size, dtype=torch.bool, device=features.device)
        positive = positive.masked_fill(self_mask, 0.0)
        exp_logits = torch.exp(logits).masked_fill(self_mask, 0.0)
        log_probability = logits - torch.log(exp_logits.sum(dim=1, keepdim=True).clamp_min(1e-12))
        counts = positive.sum(dim=1)
        if torch.any(counts <= 0):
            raise RuntimeError("A supervised-contrastive anchor has no positive view.")
        mean_positive = (positive * log_probability).sum(dim=1) / counts
        return -(self.temperature / self.base_temperature) * mean_positive.mean()


def info_nce_loss(features: torch.Tensor, temperature: float = 0.1) -> torch.Tensor:
    """Instance InfoNCE for concatenated ``[view0; view1]`` projections."""

    if features.ndim != 2 or len(features) < 4 or len(features) % 2:
        raise ValueError("InfoNCE expects an even [2B,D] matrix with B>=2.")
    batch = len(features) // 2
    identity = torch.arange(batch, device=features.device).repeat(2)
    positive_mask = identity[:, None].eq(identity[None, :])
    diagonal = torch.eye(2 * batch, dtype=torch.bool, device=features.device)
    normalized = F.normalize(features, dim=1)
    similarities = normalized @ normalized.T
    positives = similarities[(positive_mask & ~diagonal)].view(2 * batch, 1)
    negatives = similarities[(~positive_mask) & ~diagonal].view(2 * batch, -1)
    logits = torch.cat([positives, negatives], dim=1) / float(temperature)
    targets = torch.zeros(2 * batch, dtype=torch.long, device=features.device)
    return F.cross_entropy(logits, targets)


__all__ = [
    "SupervisedContrastiveLoss",
    "WindowProjectionHead",
    "info_nce_loss",
]
