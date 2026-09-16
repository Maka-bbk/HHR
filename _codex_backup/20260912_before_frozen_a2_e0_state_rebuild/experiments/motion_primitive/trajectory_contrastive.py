"""Small, HAR-only supervised contrastive objective for trajectories.

The canonical HHR path deliberately keeps this implementation separate from
the historical SimGCD utility module, which also contains image-oriented
heads and helpers.  ``SupervisedContrastiveLoss`` accepts either class labels
or an explicit positive-pair mask and preserves the standard two-view SupCon
formulation.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


class SupervisedContrastiveLoss(nn.Module):
    """Supervised Contrastive Learning (Khosla et al., 2020)."""

    def __init__(
        self,
        temperature: float = 0.07,
        contrast_mode: str = "all",
        base_temperature: float = 0.07,
    ) -> None:
        super().__init__()
        self.temperature = float(temperature)
        self.contrast_mode = str(contrast_mode)
        self.base_temperature = float(base_temperature)
        if self.temperature <= 0.0 or self.base_temperature <= 0.0:
            raise ValueError("SupCon temperatures must be positive.")
        if self.contrast_mode not in {"one", "all"}:
            raise ValueError(f"Unknown contrast mode: {self.contrast_mode!r}.")

    def forward(
        self,
        features: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if features.ndim < 3:
            raise ValueError("features must have shape [batch, views, ...].")
        if features.ndim > 3:
            features = features.reshape(features.shape[0], features.shape[1], -1)
        batch_size = int(features.shape[0])
        if batch_size < 1:
            raise ValueError("SupCon requires a non-empty batch.")
        if labels is not None and mask is not None:
            raise ValueError("Cannot define both labels and mask.")

        device = features.device
        dtype = features.dtype
        if labels is None and mask is None:
            positive_mask = torch.eye(batch_size, dtype=dtype, device=device)
        elif labels is not None:
            labels = labels.contiguous().reshape(-1, 1)
            if int(labels.shape[0]) != batch_size:
                raise ValueError("Number of labels does not match features.")
            positive_mask = labels.eq(labels.T).to(dtype=dtype, device=device)
        else:
            if tuple(mask.shape) != (batch_size, batch_size):
                raise ValueError(
                    "Explicit SupCon mask must have shape [batch,batch]."
                )
            positive_mask = mask.to(dtype=dtype, device=device)

        contrast_count = int(features.shape[1])
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == "one":
            anchor_feature = features[:, 0]
            anchor_count = 1
        else:
            anchor_feature = contrast_feature
            anchor_count = contrast_count

        logits = anchor_feature @ contrast_feature.T / self.temperature
        logits = logits - logits.max(dim=1, keepdim=True).values.detach()
        positive_mask = positive_mask.repeat(anchor_count, contrast_count)
        self_mask = torch.ones_like(positive_mask)
        self_indices = torch.arange(
            batch_size * anchor_count, device=device
        ).reshape(-1, 1)
        self_mask.scatter_(1, self_indices, 0.0)
        positive_mask = positive_mask * self_mask

        positive_count = positive_mask.sum(dim=1)
        if torch.any(positive_count <= 0):
            raise ValueError(
                "Every SupCon anchor needs at least one non-self positive pair."
            )
        exp_logits = torch.exp(logits) * self_mask
        log_probability = logits - torch.log(
            exp_logits.sum(dim=1, keepdim=True).clamp_min(torch.finfo(dtype).tiny)
        )
        mean_positive_log_probability = (
            positive_mask * log_probability
        ).sum(dim=1) / positive_count
        loss = -(
            self.temperature / self.base_temperature
        ) * mean_positive_log_probability
        return loss.reshape(anchor_count, batch_size).mean()


__all__ = ["SupervisedContrastiveLoss"]
