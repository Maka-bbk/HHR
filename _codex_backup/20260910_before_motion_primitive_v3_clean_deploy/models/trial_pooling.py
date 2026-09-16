import torch
import torch.nn as nn


def masked_feature_quantile(features, mask, quantile):
    """Compute a feature-wise quantile from only valid bag elements."""
    if features.ndim != 3:
        raise ValueError(
            f"Expected features [B,L,D], got {tuple(features.shape)}."
        )
    if mask.shape != features.shape[:2]:
        raise ValueError(
            f"Mask shape {tuple(mask.shape)} does not match "
            f"features {tuple(features.shape[:2])}."
        )
    quantile = float(quantile)
    if not 0.0 <= quantile <= 1.0:
        raise ValueError(f"Quantile must be in [0, 1], got {quantile}.")
    mask = mask.bool()
    if torch.any(mask.sum(dim=1) <= 0):
        raise ValueError("Masked quantile received an empty trial bag.")
    return torch.stack(
        [
            torch.quantile(
                sample_features[sample_mask],
                q=quantile,
                dim=0,
                interpolation="linear",
            )
            for sample_features, sample_mask in zip(features, mask)
        ],
        dim=0,
    )


def move_trial_batch_to_device(batch, device, non_blocking=True):
    """Move a padded trial-batch dictionary without changing its structure."""
    if not isinstance(batch, dict):
        raise TypeError(f"Expected a trial batch dict, got {type(batch).__name__}.")
    required = {"windows", "positions", "mask", "lengths"}
    missing = required - set(batch)
    if missing:
        raise KeyError(f"Trial batch is missing keys: {sorted(missing)}.")
    return {
        key: value.to(device=device, non_blocking=non_blocking)
        for key, value in batch.items()
    }


class TrialMeanPool(nn.Module):
    def __init__(self, feature_dim):
        super().__init__()
        self.feature_dim = int(feature_dim)
        self.norm = nn.LayerNorm(self.feature_dim)

    def forward(self, features, mask, positions=None):
        del positions
        valid_counts = mask.sum(dim=1, keepdim=True)
        if torch.any(valid_counts <= 0):
            raise ValueError("Mean pooling received an empty trial bag.")
        weights = mask.to(features.dtype) / valid_counts.to(features.dtype)
        pooled = torch.sum(features * weights.unsqueeze(-1), dim=1)
        return self.norm(pooled), weights


class TrialMeanRobustMaxPool(nn.Module):
    """Fuse masked mean features with a fixed-quantile robust peak."""

    def __init__(
        self,
        feature_dim,
        quantile=0.90,
        fusion_dim=64,
        dropout=0.0,
    ):
        super().__init__()
        self.feature_dim = int(feature_dim)
        self.quantile = float(quantile)
        self.fusion_dim = int(fusion_dim)

        if not 0.5 <= self.quantile < 1.0:
            raise ValueError(
                "Robust-max quantile must be in [0.5, 1.0); "
                f"got {self.quantile}."
            )
        if self.fusion_dim < 1:
            raise ValueError("Robust-max fusion_dim must be positive.")
        if not 0.0 <= float(dropout) < 1.0:
            raise ValueError("Robust-max fusion dropout must be in [0, 1).")

        self.mean_norm = nn.LayerNorm(self.feature_dim)
        self.peak_norm = nn.LayerNorm(self.feature_dim)
        self.fusion = nn.Sequential(
            nn.Linear(2 * self.feature_dim, self.fusion_dim),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(self.fusion_dim, self.feature_dim),
        )
        self.output_norm = nn.LayerNorm(self.feature_dim)

        # Begin close to mean pooling while retaining a trainable peak branch.
        nn.init.normal_(self.fusion[-1].weight, mean=0.0, std=1e-3)
        nn.init.zeros_(self.fusion[-1].bias)

    def forward(self, features, mask, positions=None):
        del positions
        if features.ndim != 3:
            raise ValueError(
                "Robust mean/max pooling expects features [B,L,D], got "
                f"{tuple(features.shape)}."
            )
        if mask.shape != features.shape[:2]:
            raise ValueError(
                f"Robust mean/max mask shape {tuple(mask.shape)} does not match "
                f"features {tuple(features.shape[:2])}."
            )

        mask = mask.bool()
        valid_counts = mask.sum(dim=1, keepdim=True)
        if torch.any(valid_counts <= 0):
            raise ValueError("Robust mean/max pooling received an empty trial bag.")

        mean_weights = mask.to(features.dtype) / valid_counts.to(features.dtype)
        mean_pooled = torch.sum(features * mean_weights.unsqueeze(-1), dim=1)

        # Each feature dimension gets its own q-th order statistic. Iterating over
        # bags keeps padded values entirely outside the quantile calculation.
        robust_peak = masked_feature_quantile(
            features,
            mask,
            self.quantile,
        )

        normalized_mean = self.mean_norm(mean_pooled)
        normalized_peak = self.peak_norm(robust_peak)
        correction = self.fusion(
            torch.cat([normalized_mean, normalized_peak], dim=-1)
        )
        pooled = self.output_norm(normalized_mean + correction)

        # Unlike scalar attention, feature-wise quantiles have no single window
        # weight vector that can be interpreted as attention diagnostics.
        return pooled, None


class GatedAttentionPool(nn.Module):
    """Low-capacity gated MIL attention with a fixed mean-pooling residual."""

    def __init__(
        self,
        feature_dim,
        position_dim=3,
        attention_dim=64,
        dropout=0.1,
        temperature=1.0,
        mean_mix=0.5,
    ):
        super().__init__()
        self.feature_dim = int(feature_dim)
        self.position_dim = int(position_dim)
        self.attention_dim = int(attention_dim)
        self.temperature = float(temperature)
        self.mean_mix = float(mean_mix)

        if self.attention_dim < 1:
            raise ValueError("attention_dim must be positive.")
        if self.temperature <= 0:
            raise ValueError("attention temperature must be positive.")
        if not 0.0 <= self.mean_mix <= 1.0:
            raise ValueError("mean_mix must be in [0, 1].")

        input_dim = self.feature_dim + self.position_dim
        self.dropout = nn.Dropout(float(dropout))
        self.tanh_projection = nn.Linear(input_dim, self.attention_dim)
        self.sigmoid_projection = nn.Linear(input_dim, self.attention_dim)
        self.score = nn.Linear(self.attention_dim, 1)
        self.norm = nn.LayerNorm(self.feature_dim)

    def forward(self, features, mask, positions):
        if positions is None:
            raise ValueError("Gated attention pooling requires relative positions.")
        if positions.shape[:2] != features.shape[:2]:
            raise ValueError(
                "Position/features bag dimensions differ: "
                f"{tuple(positions.shape)} vs {tuple(features.shape)}."
            )
        valid_counts = mask.sum(dim=1, keepdim=True)
        if torch.any(valid_counts <= 0):
            raise ValueError("Attention pooling received an empty trial bag.")

        attention_input = self.dropout(torch.cat([features, positions], dim=-1))
        gated = torch.tanh(self.tanh_projection(attention_input))
        gated = gated * torch.sigmoid(self.sigmoid_projection(attention_input))
        scores = self.score(gated).squeeze(-1) / self.temperature
        scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
        weights = torch.softmax(scores, dim=1)
        weights = weights * mask.to(weights.dtype)
        weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-12)

        attention_pooled = torch.sum(features * weights.unsqueeze(-1), dim=1)
        mean_weights = mask.to(features.dtype) / valid_counts.to(features.dtype)
        mean_pooled = torch.sum(features * mean_weights.unsqueeze(-1), dim=1)
        pooled = (
            (1.0 - self.mean_mix) * mean_pooled
            + self.mean_mix * attention_pooled
        )
        return self.norm(pooled), weights


class TrialMILBackbone(nn.Module):
    """Encode valid windows and aggregate one feature per padded trial bag."""

    def __init__(self, window_encoder, pooling):
        super().__init__()
        self.window_encoder = window_encoder
        self.pooling = pooling
        self.feat_dim = int(window_encoder.feat_dim)
        self.last_attention_weights = None
        self.last_attention_mask = None

    def forward(self, batch):
        if not isinstance(batch, dict):
            raise TypeError(
                f"TrialMILBackbone expects a dict batch, got {type(batch).__name__}."
            )
        windows = batch["windows"]
        positions = batch["positions"]
        mask = batch["mask"].bool()
        lengths = batch["lengths"].long()

        if windows.ndim != 4:
            raise ValueError(
                f"Trial windows must be [B,L,C,T], got {tuple(windows.shape)}."
            )
        if mask.shape != windows.shape[:2]:
            raise ValueError(
                f"Trial mask shape {tuple(mask.shape)} does not match "
                f"windows {tuple(windows.shape[:2])}."
            )
        if positions.shape[:2] != windows.shape[:2]:
            raise ValueError("Trial positions and windows have different [B,L].")
        observed_lengths = mask.sum(dim=1)
        if not torch.equal(observed_lengths, lengths):
            raise ValueError(
                f"Trial lengths/mask mismatch: {lengths.tolist()} vs "
                f"{observed_lengths.tolist()}."
            )
        if torch.any(observed_lengths <= 0):
            raise ValueError("Trial batch contains an empty bag.")

        valid_windows = windows[mask]
        valid_features = self.window_encoder(valid_windows)
        if valid_features.ndim != 2 or valid_features.size(1) != self.feat_dim:
            raise RuntimeError(
                "Window encoder returned an unexpected feature shape: "
                f"{tuple(valid_features.shape)}."
            )

        padded_features = valid_features.new_zeros(
            windows.size(0), windows.size(1), self.feat_dim
        )
        padded_features[mask] = valid_features
        pooled, weights = self.pooling(padded_features, mask, positions)
        if weights is None:
            self.last_attention_weights = None
            self.last_attention_mask = None
        else:
            self.last_attention_weights = weights.detach()
            self.last_attention_mask = mask.detach()
        return pooled


def build_trial_backbone(window_encoder, args):
    pooling_name = str(getattr(args, "trial_pooling", "mean_robust_max"))
    if pooling_name == "mean":
        pooling = TrialMeanPool(window_encoder.feat_dim)
    elif pooling_name == "mean_robust_max":
        pooling = TrialMeanRobustMaxPool(
            feature_dim=window_encoder.feat_dim,
            quantile=getattr(args, "trial_robust_max_quantile", 0.90),
            fusion_dim=getattr(args, "trial_pool_fusion_dim", 64),
            dropout=getattr(args, "trial_pool_fusion_dropout", 0.0),
        )
    elif pooling_name == "gated_attention":
        pooling = GatedAttentionPool(
            feature_dim=window_encoder.feat_dim,
            position_dim=3,
            attention_dim=getattr(args, "trial_attention_dim", 64),
            dropout=getattr(args, "trial_attention_dropout", 0.1),
            temperature=getattr(args, "trial_attention_temperature", 1.0),
            mean_mix=getattr(args, "trial_attention_mean_mix", 0.5),
        )
    else:
        raise ValueError(
            f"Unknown trial pooling {pooling_name!r}; expected mean, "
            "mean_robust_max, or gated_attention."
        )
    return TrialMILBackbone(window_encoder, pooling)
