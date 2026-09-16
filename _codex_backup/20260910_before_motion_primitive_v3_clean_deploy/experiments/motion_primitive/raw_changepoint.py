"""Label-free multi-scale kinematic change scores in raw sensor units.

The functions in this module deliberately know nothing about activity labels
or subject identities.  A caller supplies one reconstructed physical-unit
trial and its ordered window starts.  Component calibration is fitted by the
training entry point on train-subject/old-class trials only and can then be
applied unchanged to validation trials.
"""

from __future__ import annotations

import copy
import math
import re
from typing import Sequence

import numpy as np
import torch


RAW_COMPONENTS_PER_SCALE = (
    "acc_mean_l2",
    "gyro_mean_l2",
    "acc_std_l2",
    "gyro_std_l2",
    "acc_log_energy_l2",
    "gyro_log_energy_l2",
    "gravity_direction_angle_rad",
    "acc_fft_amplitude_l2",
    "gyro_fft_amplitude_l2",
)


def parse_raw_scales(value: str | Sequence[float]) -> tuple[float, ...]:
    """Parse positive, unique stride multipliers without silently sorting."""

    if isinstance(value, str):
        tokens = [item for item in re.split(r"[\s,]+", value.strip()) if item]
        scales = tuple(float(item) for item in tokens)
    else:
        scales = tuple(float(item) for item in value)
    if not scales or any(not math.isfinite(item) or item <= 0 for item in scales):
        raise ValueError("Raw CP scales must be a non-empty list of positive finite values.")
    if len(set(scales)) != len(scales):
        raise ValueError("Raw CP scales must not contain duplicates.")
    return scales


def _scale_token(value: float) -> str:
    return format(float(value), ".8g").replace(".", "p")


def raw_component_names(scales: Sequence[float]) -> tuple[str, ...]:
    parsed = parse_raw_scales(scales)
    return tuple(
        f"stride_x{_scale_token(scale)}__{component}"
        for scale in parsed
        for component in RAW_COMPONENTS_PER_SCALE
    )


def _as_raw_numpy(raw_trial: torch.Tensor | np.ndarray) -> np.ndarray:
    if isinstance(raw_trial, torch.Tensor):
        raw = raw_trial.detach().cpu().numpy()
    else:
        raw = np.asarray(raw_trial)
    raw = np.asarray(raw, dtype=np.float64)
    if raw.ndim != 2 or raw.shape[0] != 6 or raw.shape[1] < 1:
        raise ValueError(f"raw_trial must have shape [6,T], got {raw.shape}.")
    if not np.all(np.isfinite(raw)):
        raise ValueError("raw_trial contains non-finite values.")
    return raw


def _as_starts_numpy(starts: torch.Tensor | np.ndarray | Sequence[int]) -> np.ndarray:
    if isinstance(starts, torch.Tensor):
        values = starts.detach().cpu().numpy()
    else:
        values = np.asarray(starts)
    values = np.asarray(values, dtype=np.int64)
    if values.ndim != 1:
        raise ValueError("Window starts must be one-dimensional.")
    if len(values) and (values[0] < 0 or np.any(np.diff(values) <= 0)):
        raise ValueError("Window starts must be non-negative and strictly increasing.")
    return values


def raw_boundary_anchor_samples(
    starts: torch.Tensor | np.ndarray | Sequence[int],
    window_size: int,
) -> np.ndarray:
    """Map adjacent window features to midpoints between their sample centres."""

    positions = _as_starts_numpy(starts)
    if int(window_size) < 1:
        raise ValueError("window_size must be positive.")
    if len(positions) < 2:
        return np.empty(0, dtype=np.int64)
    centres = positions.astype(np.float64) + 0.5 * (int(window_size) - 1)
    # Round half up rather than Python's banker rounding.  The convention is
    # deterministic and is recorded in calibration metadata.
    return np.floor(0.5 * (centres[:-1] + centres[1:]) + 0.5).astype(np.int64)


def _fixed_frequency_amplitude(signal: np.ndarray, bins: int) -> np.ndarray:
    """One-sided demeaned FFT amplitude interpolated to fixed normalised bins."""

    if signal.ndim != 2 or signal.shape[1] < 2:
        return np.zeros((signal.shape[0], int(bins)), dtype=np.float64)
    centred = signal - signal.mean(axis=1, keepdims=True)
    amplitude = np.abs(np.fft.rfft(centred, axis=1)) / float(signal.shape[1])
    amplitude = amplitude[:, 1:]  # DC is already represented by mean features.
    if amplitude.shape[1] == 0:
        return np.zeros((signal.shape[0], int(bins)), dtype=np.float64)
    old_grid = np.linspace(0.0, 1.0, amplitude.shape[1], dtype=np.float64)
    new_grid = np.linspace(0.0, 1.0, int(bins), dtype=np.float64)
    return np.stack([np.interp(new_grid, old_grid, row) for row in amplitude], axis=0)


def _vector_rms_difference(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(left - right), dtype=np.float64)))


def _components_for_pair(
    left: np.ndarray,
    right: np.ndarray,
    *,
    frequency_bins: int,
    epsilon: float,
) -> np.ndarray:
    left_mean = left.mean(axis=1)
    right_mean = right.mean(axis=1)
    left_std = left.std(axis=1)
    right_std = right.std(axis=1)
    left_energy = np.mean(np.square(left), axis=1, dtype=np.float64)
    right_energy = np.mean(np.square(right), axis=1, dtype=np.float64)

    gravity_left = left_mean[:3]
    gravity_right = right_mean[:3]
    left_norm = float(np.linalg.norm(gravity_left))
    right_norm = float(np.linalg.norm(gravity_right))
    if left_norm <= float(epsilon) or right_norm <= float(epsilon):
        # Direction is undefined when either mean acceleration is effectively
        # zero.  Magnitude changes remain represented by acc_mean_l2; assigning
        # a 90-degree angle here would manufacture a direction change.
        gravity_angle = 0.0
    else:
        cosine = float(
            np.clip(np.dot(gravity_left, gravity_right) / (left_norm * right_norm), -1.0, 1.0)
        )
        gravity_angle = float(np.arccos(cosine))

    left_fft = _fixed_frequency_amplitude(left, int(frequency_bins))
    right_fft = _fixed_frequency_amplitude(right, int(frequency_bins))
    return np.asarray(
        [
            _vector_rms_difference(left_mean[:3], right_mean[:3]),
            _vector_rms_difference(left_mean[3:], right_mean[3:]),
            _vector_rms_difference(left_std[:3], right_std[:3]),
            _vector_rms_difference(left_std[3:], right_std[3:]),
            _vector_rms_difference(
                np.log(left_energy[:3] + epsilon), np.log(right_energy[:3] + epsilon)
            ),
            _vector_rms_difference(
                np.log(left_energy[3:] + epsilon), np.log(right_energy[3:] + epsilon)
            ),
            gravity_angle,
            _vector_rms_difference(left_fft[:3], right_fft[:3]),
            _vector_rms_difference(left_fft[3:], right_fft[3:]),
        ],
        dtype=np.float64,
    )


def raw_multiscale_boundary_components(
    raw_trial: torch.Tensor | np.ndarray,
    starts: torch.Tensor | np.ndarray | Sequence[int],
    window_size: int,
    scales: Sequence[float] = (1.0, 2.0, 4.0),
    frequency_bins: int = 16,
    epsilon: float = 1e-8,
) -> np.ndarray:
    """Return one physical-unit kinematic component vector per window boundary.

    For scale ``s``, each side uses ``s * median_window_stride`` requested
    samples.  Near an observed-trial edge, both sides are shortened to the
    same available radius, preventing unequal-window summary bias.
    """

    raw = _as_raw_numpy(raw_trial)
    positions = _as_starts_numpy(starts)
    parsed_scales = parse_raw_scales(scales)
    if int(window_size) < 1 or int(window_size) > raw.shape[1]:
        raise ValueError("window_size must fit inside raw_trial.")
    if int(frequency_bins) < 2:
        raise ValueError("frequency_bins must be at least two.")
    if not math.isfinite(float(epsilon)) or float(epsilon) <= 0:
        raise ValueError("epsilon must be positive and finite.")
    if len(positions) and int(positions[-1]) + int(window_size) > raw.shape[1]:
        raise ValueError("A window extends beyond the reconstructed raw trial.")
    names = raw_component_names(parsed_scales)
    if len(positions) < 2:
        return np.empty((0, len(names)), dtype=np.float64)

    strides = np.diff(positions)
    base_stride = float(np.median(strides))
    if not math.isfinite(base_stride) or base_stride <= 0:
        raise ValueError("Could not derive a positive median window stride.")
    anchors = raw_boundary_anchor_samples(positions, int(window_size))
    output = np.empty((len(anchors), len(names)), dtype=np.float64)
    for boundary_index, anchor in enumerate(anchors):
        column = 0
        for scale in parsed_scales:
            requested = max(2, int(round(float(scale) * base_stride)))
            radius = min(requested, int(anchor), int(raw.shape[1] - anchor))
            if radius < 2:
                raise ValueError(
                    f"Boundary {boundary_index} has fewer than two raw samples per side."
                )
            left = raw[:, int(anchor) - radius : int(anchor)]
            right = raw[:, int(anchor) : int(anchor) + radius]
            values = _components_for_pair(
                left,
                right,
                frequency_bins=int(frequency_bins),
                epsilon=float(epsilon),
            )
            output[boundary_index, column : column + len(values)] = values
            column += len(values)
    if not np.all(np.isfinite(output)):
        raise RuntimeError("Raw boundary component extraction produced non-finite values.")
    return output


def _weighted_quantile(values: np.ndarray, weights: np.ndarray, quantile: float) -> float:
    values = np.asarray(values, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    if values.ndim != 1 or weights.shape != values.shape or not len(values):
        raise ValueError("Weighted quantile requires equally sized non-empty vectors.")
    if not np.all(np.isfinite(values)) or not np.all(np.isfinite(weights)) or np.any(weights <= 0):
        raise ValueError("Weighted quantile inputs must be finite and weights positive.")
    if not 0.0 <= float(quantile) <= 1.0:
        raise ValueError("quantile must lie in [0,1].")
    order = np.argsort(values, kind="stable")
    sorted_values = values[order]
    cumulative = np.cumsum(weights[order])
    target = float(quantile) * float(cumulative[-1])
    # Equal-trial weights such as 1/9 and 1/90 have slightly different binary
    # summation error.  A tiny scale-aware tolerance makes exact CDF ties use
    # the same lower-quantile convention regardless of boundary replication.
    tolerance = 32.0 * np.finfo(np.float64).eps * max(1.0, abs(float(cumulative[-1])))
    index = int(np.searchsorted(cumulative, target - tolerance, side="left"))
    return float(sorted_values[min(index, len(sorted_values) - 1)])


def _trial_equal_flatten(values_by_trial: Sequence[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    nonempty: list[np.ndarray] = []
    weights: list[np.ndarray] = []
    width = None
    for values in values_by_trial:
        array = np.asarray(values, dtype=np.float64)
        if array.ndim != 2:
            raise ValueError("Each raw component trial must have shape [boundaries,components].")
        if width is None:
            width = int(array.shape[1])
        elif array.shape[1] != width:
            raise ValueError("Raw component trial widths differ.")
        if not np.all(np.isfinite(array)):
            raise ValueError("Raw component values must be finite.")
        if len(array):
            nonempty.append(array)
            weights.append(np.full(len(array), 1.0 / len(array), dtype=np.float64))
    if not nonempty:
        raise ValueError("At least one training trial must contain a boundary.")
    return np.concatenate(nonempty, axis=0), np.concatenate(weights, axis=0)


def fit_trial_equal_robust_component_scaler(
    train_components: Sequence[np.ndarray],
    component_names: Sequence[str],
    *,
    scale_floor: float = 1e-6,
    z_clip: float = 10.0,
) -> dict:
    """Fit component-wise weighted median/IQR using equal total trial mass."""

    if not math.isfinite(float(scale_floor)) or float(scale_floor) <= 0:
        raise ValueError("scale_floor must be positive and finite.")
    if not math.isfinite(float(z_clip)) or float(z_clip) <= 0:
        raise ValueError("z_clip must be positive and finite.")
    flat, weights = _trial_equal_flatten(train_components)
    names = tuple(str(item) for item in component_names)
    if flat.shape[1] != len(names) or len(set(names)) != len(names):
        raise ValueError("component_names must uniquely match the component width.")
    centre = []
    q25 = []
    q75 = []
    scales = []
    floor_used = []
    for column in range(flat.shape[1]):
        values = flat[:, column]
        lower = _weighted_quantile(values, weights, 0.25)
        median = _weighted_quantile(values, weights, 0.50)
        upper = _weighted_quantile(values, weights, 0.75)
        adaptive_floor = float(scale_floor) * max(1.0, abs(lower), abs(median), abs(upper))
        spread = upper - lower
        scale = max(spread, adaptive_floor)
        q25.append(lower)
        centre.append(median)
        q75.append(upper)
        scales.append(scale)
        floor_used.append(spread < adaptive_floor)
    return {
        "fit_split": "train_subjects_old_classes_only",
        "fit_weighting": "each_boundary_weight=1/n_boundaries_in_its_trial",
        "centre_statistic": "weighted_median",
        "scale_statistic": "max(weighted_q75-weighted_q25,relative_scale_floor)",
        "component_names": list(names),
        "q25": q25,
        "centre": centre,
        "q75": q75,
        "scale": scales,
        "relative_scale_floor": float(scale_floor),
        "scale_floor_used": floor_used,
        "z_clip": float(z_clip),
        "aggregate_formula": "raw_score=mean_j(clip((component_j-centre_j)/scale_j,-z_clip,+z_clip))",
        "train_boundary_count": int(len(flat)),
        "train_boundary_bearing_trial_count": int(sum(len(item) > 0 for item in train_components)),
    }


def transform_raw_boundary_components(components: np.ndarray, calibration: dict) -> np.ndarray:
    """Apply a frozen train-fitted robust transform and average dimensions."""

    values = np.asarray(components, dtype=np.float64)
    names = calibration.get("component_names", [])
    if values.ndim != 2 or values.shape[1] != len(names):
        raise ValueError("Raw component shape does not match scaler calibration.")
    centre = np.asarray(calibration["centre"], dtype=np.float64)
    scale = np.asarray(calibration["scale"], dtype=np.float64)
    z_clip = float(calibration["z_clip"])
    if np.any(scale <= 0) or not np.all(np.isfinite(scale)):
        raise ValueError("Scaler contains an invalid scale.")
    if not len(values):
        return np.empty(0, dtype=np.float64)
    z = np.clip((values - centre[None, :]) / scale[None, :], -z_clip, z_clip)
    return z.mean(axis=1, dtype=np.float64)


def fit_trial_equal_score_thresholds(
    train_scores: Sequence[np.ndarray], low_quantile: float, high_quantile: float
) -> dict:
    """Fit low/high thresholds with equal total weight for every nonempty trial."""

    if not 0.0 <= float(low_quantile) < float(high_quantile) <= 1.0:
        raise ValueError("Threshold quantiles must satisfy 0 <= low < high <= 1.")
    matrices = [np.asarray(item, dtype=np.float64).reshape(-1, 1) for item in train_scores]
    flat, weights = _trial_equal_flatten(matrices)
    values = flat[:, 0]
    return {
        "low_quantile": float(low_quantile),
        "high_quantile": float(high_quantile),
        "low_threshold": _weighted_quantile(values, weights, float(low_quantile)),
        "high_threshold": _weighted_quantile(values, weights, float(high_quantile)),
        "threshold_fit_split": "train_subjects_old_classes_only",
        "threshold_weighting": "each_boundary_weight=1/n_boundaries_in_its_trial",
        "train_boundary_count": int(len(values)),
        "train_boundary_bearing_trial_count": int(sum(len(item) > 0 for item in train_scores)),
    }


def boundary_candidate_masks(scores: np.ndarray, thresholds: dict) -> tuple[np.ndarray, np.ndarray]:
    """Return disjoint low/stable and high/change candidates; high is never forced."""

    values = np.asarray(scores, dtype=np.float64).reshape(-1)
    low = float(thresholds["low_threshold"])
    high = float(thresholds["high_threshold"])
    if high < low:
        raise ValueError("High boundary threshold is below low threshold.")
    high_mask = (values >= high) & (values > low)
    low_mask = (values <= low) & ~high_mask
    return low_mask, high_mask


def consensus_boundary_masks(
    raw_scores: np.ndarray,
    frozen_scores: np.ndarray,
    raw_thresholds: dict,
    frozen_thresholds: dict,
    source: str,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Combine raw/frozen candidates without a per-trial top-k fallback."""

    raw = np.asarray(raw_scores, dtype=np.float64).reshape(-1)
    frozen = np.asarray(frozen_scores, dtype=np.float64).reshape(-1)
    if raw.shape != frozen.shape:
        raise ValueError("Raw and frozen boundary score lengths differ.")
    raw_low, raw_high = boundary_candidate_masks(raw, raw_thresholds)
    frozen_low, frozen_high = boundary_candidate_masks(frozen, frozen_thresholds)
    if source == "raw_frozen_consensus":
        stable = raw_low & frozen_low
        change = raw_high & frozen_high
    elif source == "frozen_legacy":
        stable = frozen_low
        change = frozen_high
    else:
        raise ValueError("source must be raw_frozen_consensus or frozen_legacy.")
    if np.any(stable & change):
        raise RuntimeError("Stable and change masks overlap.")
    diagnostics = {
        "boundary_count": int(len(raw)),
        "raw_low_count": int(raw_low.sum()),
        "raw_high_count": int(raw_high.sum()),
        "frozen_low_count": int(frozen_low.sum()),
        "frozen_high_count": int(frozen_high.sum()),
        "joint_low_count": int((raw_low & frozen_low).sum()),
        "joint_high_count": int((raw_high & frozen_high).sum()),
        "selected_stable_count": int(stable.sum()),
        "selected_change_count": int(change.sum()),
    }
    return stable, change, diagnostics


def raw_formula_metadata(
    scales: Sequence[float], frequency_bins: int, epsilon: float
) -> dict:
    """Machine-readable audit description stored with pseudo calibration."""

    parsed = parse_raw_scales(scales)
    return {
        "input": "reconstructed raw physical-unit trial before fold standardization",
        "component_extractor_uses_activity_label_or_subject_id": False,
        "calibration_population": "caller-selected train-subject old-class trials only",
        "boundary_sample_formula": (
            "round_half_up(mean(start_i+(window_size-1)/2,"
            "start_(i+1)+(window_size-1)/2))"
        ),
        "base_radius_samples": "median(diff(window_start_indices))",
        "scale_stride_multipliers": list(parsed),
        "edge_policy": "symmetric_radius=min(requested_radius,left_available,right_available)",
        "components_per_scale": list(RAW_COMPONENTS_PER_SCALE),
        "mean_std_formula": "RMS across the three sensor axes of left/right summary difference",
        "energy_formula": "RMS across axes of log(mean(x^2)+epsilon) left/right difference",
        "gravity_formula": (
            "acos(cosine(mean_acc_left,mean_acc_right)); kinematic gravity-direction proxy; "
            "zero when either mean norm<=epsilon because direction is undefined"
        ),
        "frequency_formula": (
            "RMS difference of per-axis |rFFT(x-mean(x))|/n, DC excluded, "
            "linearly interpolated to fixed normalized-frequency bins"
        ),
        "frequency_bins": int(frequency_bins),
        "epsilon": float(epsilon),
        "component_names": list(raw_component_names(parsed)),
    }


def calibration_copy(calibration: dict) -> dict:
    """Small explicit helper used by tests to assert transform-only behaviour."""

    return copy.deepcopy(calibration)


__all__ = [
    "RAW_COMPONENTS_PER_SCALE",
    "boundary_candidate_masks",
    "calibration_copy",
    "consensus_boundary_masks",
    "fit_trial_equal_robust_component_scaler",
    "fit_trial_equal_score_thresholds",
    "parse_raw_scales",
    "raw_boundary_anchor_samples",
    "raw_component_names",
    "raw_formula_metadata",
    "raw_multiscale_boundary_components",
    "transform_raw_boundary_components",
]
