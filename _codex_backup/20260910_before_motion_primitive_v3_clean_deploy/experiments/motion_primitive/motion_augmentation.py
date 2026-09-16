"""Trial-level, semantics-preserving augmentation for motion primitives.

The functions in this module deliberately operate on a complete raw USC-HAD
trial before normalisation and window extraction.  This keeps overlapping
windows physically consistent: a sample shared by two windows receives the
same noise, mask, rotation, scale, and time shift in both windows.

The expected channel order is strictly::

    acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z

Normalisation statistics must come from the current fold's training subjects.
No statistics are estimated by this module, which helps keep the train/test
boundary explicit and auditable.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Integral, Real
from typing import Optional, Sequence, Tuple

import torch
import torch.nn.functional as F


_EXPECTED_CHANNELS = 6
_ALLOWED_STAT_SHAPES_DESCRIPTION = "[C], [C,1], or [1,C,1]"


@dataclass(frozen=True)
class MotionAugmentationConfig:
    """Configuration for one independently sampled raw-trial view.

    Setting an augmentation's magnitude to zero (or a scale range to
    ``(1, 1)``) disables that augmentation.  Defaults implement the proposed
    conservative ``basic`` profile.  Set ``rotation_max_degrees=3.0`` for the
    separate ``basic_rotation`` ablation.

    ``noise_std_ratio`` is relative to each fold-training channel standard
    deviation.  One scale is shared by the three accelerometer axes and a
    second independently sampled scale is shared by the three gyroscope axes.
    """

    noise_std_ratio: float = 0.01
    acc_scale_range: Tuple[float, float] = (0.95, 1.05)
    gyro_scale_range: Tuple[float, float] = (0.95, 1.05)
    time_shift_max_samples: int = 8
    time_mask_min_samples: int = 4
    time_mask_max_samples: int = 12
    rotation_max_degrees: float = 0.0

    def __post_init__(self) -> None:
        noise_ratio = _finite_real(self.noise_std_ratio, "noise_std_ratio")
        if not 0.0 <= noise_ratio <= 0.25:
            raise ValueError("noise_std_ratio must be in [0, 0.25].")

        _validate_scale_range(self.acc_scale_range, "acc_scale_range")
        _validate_scale_range(self.gyro_scale_range, "gyro_scale_range")

        _nonnegative_int(self.time_shift_max_samples, "time_shift_max_samples")
        mask_min = _nonnegative_int(
            self.time_mask_min_samples, "time_mask_min_samples"
        )
        mask_max = _nonnegative_int(
            self.time_mask_max_samples, "time_mask_max_samples"
        )
        if (mask_min == 0) != (mask_max == 0):
            raise ValueError(
                "time-mask bounds must either both be zero (disabled) or both "
                "be positive."
            )
        if mask_min > mask_max:
            raise ValueError(
                "time_mask_min_samples cannot exceed time_mask_max_samples."
            )

        max_degrees = _finite_real(
            self.rotation_max_degrees, "rotation_max_degrees"
        )
        if not 0.0 <= max_degrees <= 15.0:
            raise ValueError(
                "rotation_max_degrees must be in [0, 15]; larger rotations are "
                "not a conservative sensor-mount simulation."
            )


@dataclass(frozen=True)
class MotionAugmentationMetadata:
    """Sampled temporal transform needed for exact cross-view alignment.

    With ``time_shift_samples=s``, augmented coordinate ``t`` reads source
    coordinate ``t-s`` away from reflected trial edges.  A source boundary at
    coordinate ``b`` therefore appears near augmented coordinate ``b+s``.
    Mask coordinates refer to the augmented output after time shifting.
    """

    time_shift_samples: int
    time_mask_start_sample: Optional[int]
    time_mask_length_samples: int


def _finite_real(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number, got {type(value).__name__}.")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite, got {result!r}.")
    return result


def _nonnegative_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer, got {type(value).__name__}.")
    result = int(value)
    if result < 0:
        raise ValueError(f"{name} must be non-negative, got {result}.")
    return result


def _positive_int(value: object, name: str) -> int:
    result = _nonnegative_int(value, name)
    if result == 0:
        raise ValueError(f"{name} must be positive.")
    return result


def _validate_scale_range(value: object, name: str) -> Tuple[float, float]:
    if not isinstance(value, (tuple, list)) or len(value) != 2:
        raise TypeError(f"{name} must be a two-element tuple/list.")
    lower = _finite_real(value[0], f"{name}[0]")
    upper = _finite_real(value[1], f"{name}[1]")
    if not 0.5 <= lower <= 1.0 <= upper <= 1.5:
        raise ValueError(
            f"{name} must satisfy 0.5 <= lower <= 1 <= upper <= 1.5, "
            f"got ({lower}, {upper})."
        )
    return lower, upper


def _validate_raw_trial(raw_trial: torch.Tensor) -> None:
    if not isinstance(raw_trial, torch.Tensor):
        raise TypeError(
            f"raw_trial must be a torch.Tensor, got {type(raw_trial).__name__}."
        )
    if raw_trial.ndim != 2:
        raise ValueError(
            "raw_trial must have shape [C,T], got "
            f"{tuple(raw_trial.shape)}."
        )
    if raw_trial.shape[0] != _EXPECTED_CHANNELS:
        raise ValueError(
            "raw_trial must contain exactly six ordered channels "
            "[acc_x,acc_y,acc_z,gyro_x,gyro_y,gyro_z], got "
            f"C={raw_trial.shape[0]}."
        )
    if raw_trial.shape[1] < 2:
        raise ValueError(
            "raw_trial must contain at least two temporal samples; reflection "
            f"padding is undefined for T={raw_trial.shape[1]}."
        )
    if not raw_trial.is_floating_point():
        raise TypeError(
            f"raw_trial must use a floating dtype, got {raw_trial.dtype}."
        )
    if not bool(torch.isfinite(raw_trial).all().item()):
        raise ValueError("raw_trial contains NaN or infinite values.")


def _channel_vector(
    values: torch.Tensor,
    channels: int,
    name: str,
    *,
    reference: torch.Tensor,
) -> torch.Tensor:
    if not isinstance(values, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor, got {type(values).__name__}.")
    allowed_shapes = {(channels,), (channels, 1), (1, channels, 1)}
    if tuple(values.shape) not in allowed_shapes:
        raise ValueError(
            f"{name} must have shape {_ALLOWED_STAT_SHAPES_DESCRIPTION} with "
            f"C={channels}, got {tuple(values.shape)}."
        )
    if not values.is_floating_point():
        raise TypeError(f"{name} must use a floating dtype, got {values.dtype}.")
    result = values.reshape(channels).to(
        device=reference.device, dtype=reference.dtype
    )
    if not bool(torch.isfinite(result).all().item()):
        raise ValueError(f"{name} contains NaN or infinite values.")
    return result


def _validate_statistics(
    raw_trial: torch.Tensor,
    train_channel_mean: torch.Tensor,
    train_channel_std: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    mean = _channel_vector(
        train_channel_mean,
        int(raw_trial.shape[0]),
        "train_channel_mean",
        reference=raw_trial,
    )
    std = _channel_vector(
        train_channel_std,
        int(raw_trial.shape[0]),
        "train_channel_std",
        reference=raw_trial,
    )
    if not bool((std > 0).all().item()):
        raise ValueError("train_channel_std must be strictly positive in every channel.")
    return mean, std


def _random_device(
    target_device: torch.device, generator: Optional[torch.Generator]
) -> torch.device:
    if generator is None:
        return target_device
    if not isinstance(generator, torch.Generator):
        raise TypeError(
            "generator must be torch.Generator or None, got "
            f"{type(generator).__name__}."
        )
    return torch.device(generator.device)


def _rand(
    shape: Sequence[int],
    *,
    reference: torch.Tensor,
    generator: Optional[torch.Generator],
) -> torch.Tensor:
    random_device = _random_device(reference.device, generator)
    sample = torch.rand(
        tuple(int(size) for size in shape),
        dtype=torch.float64,
        device=random_device,
        generator=generator,
    )
    return sample.to(device=reference.device, dtype=reference.dtype)


def _randn(
    shape: Sequence[int],
    *,
    reference: torch.Tensor,
    generator: Optional[torch.Generator],
) -> torch.Tensor:
    random_device = _random_device(reference.device, generator)
    sample = torch.randn(
        tuple(int(size) for size in shape),
        dtype=torch.float64,
        device=random_device,
        generator=generator,
    )
    return sample.to(device=reference.device, dtype=reference.dtype)


def _randint_inclusive(
    lower: int,
    upper: int,
    *,
    reference: torch.Tensor,
    generator: Optional[torch.Generator],
) -> int:
    if lower > upper:
        raise ValueError(f"Invalid integer sampling interval [{lower}, {upper}].")
    random_device = _random_device(reference.device, generator)
    return int(
        torch.randint(
            lower,
            upper + 1,
            (1,),
            dtype=torch.int64,
            device=random_device,
            generator=generator,
        ).item()
    )


def _uniform_scalar(
    lower: float,
    upper: float,
    *,
    reference: torch.Tensor,
    generator: Optional[torch.Generator],
) -> torch.Tensor:
    if lower == upper:
        return reference.new_tensor(lower)
    unit = _rand((), reference=reference, generator=generator)
    return unit * (upper - lower) + lower


def _axis_angle_rotation(
    max_degrees: float,
    *,
    reference: torch.Tensor,
    generator: Optional[torch.Generator],
) -> torch.Tensor:
    """Sample a proper 3-D rotation using Rodrigues' formula."""

    axis = _randn((3,), reference=reference, generator=generator)
    axis_norm = torch.linalg.vector_norm(axis)
    if float(axis_norm.item()) <= torch.finfo(reference.dtype).eps:
        axis = reference.new_tensor((1.0, 0.0, 0.0))
    else:
        axis = axis / axis_norm

    max_radians = math.radians(max_degrees)
    angle = _uniform_scalar(
        -max_radians,
        max_radians,
        reference=reference,
        generator=generator,
    )
    x, y, z = axis.unbind()
    zero = reference.new_zeros(())
    skew = torch.stack(
        (
            torch.stack((zero, -z, y)),
            torch.stack((z, zero, -x)),
            torch.stack((-y, x, zero)),
        )
    )
    identity = torch.eye(3, dtype=reference.dtype, device=reference.device)
    sine = torch.sin(angle)
    cosine = torch.cos(angle)
    return identity + sine * skew + (1.0 - cosine) * (skew @ skew)


def _reflect_time_shift(raw_trial: torch.Tensor, shift: int) -> torch.Tensor:
    """Shift without wraparound, filling the exposed edge by reflection."""

    if shift == 0:
        return raw_trial
    amount = abs(int(shift))
    temporal_length = int(raw_trial.shape[1])
    if amount >= temporal_length:
        raise ValueError(
            "Absolute time shift must be smaller than trial length for reflection "
            f"padding, got |shift|={amount}, T={temporal_length}."
        )
    padded = F.pad(raw_trial.unsqueeze(0), (amount, amount), mode="reflect").squeeze(0)
    start = amount - int(shift)
    return padded[:, start : start + temporal_length]


def _validated_time_mask_anchor(
    time_mask_anchor: Optional[Tuple[int, int]], temporal_length: int
) -> Optional[Tuple[int, int]]:
    if time_mask_anchor is None:
        return None
    if not isinstance(time_mask_anchor, (tuple, list)) or len(time_mask_anchor) != 2:
        raise TypeError(
            "time_mask_anchor must be None or a two-element (start, end) tuple/list."
        )
    begin = time_mask_anchor[0]
    end = time_mask_anchor[1]
    if (
        isinstance(begin, bool)
        or not isinstance(begin, Integral)
        or isinstance(end, bool)
        or not isinstance(end, Integral)
    ):
        raise TypeError("time_mask_anchor start and end must be integers.")
    begin = int(begin)
    end = int(end)
    if not 0 <= begin < end <= temporal_length:
        raise ValueError(
            "time_mask_anchor must satisfy 0 <= start < end <= T, got "
            f"({begin}, {end}) for T={temporal_length}."
        )
    return begin, end


def _augment_raw_trial_with_metadata_impl(
    raw_trial: torch.Tensor,
    train_channel_mean: torch.Tensor,
    train_channel_std: torch.Tensor,
    config: MotionAugmentationConfig,
    *,
    generator: Optional[torch.Generator] = None,
    time_mask_anchor: Optional[Tuple[int, int]] = None,
) -> Tuple[torch.Tensor, MotionAugmentationMetadata]:
    """Implement one raw-trial view and retain sampled alignment metadata.

    Args:
        raw_trial: Raw sensor values with shape ``[6,T]`` and canonical USC-HAD
            channel order.
        train_channel_mean/std: Current fold-training statistics in shape
            ``[C]``, ``[C,1]``, or ``[1,C,1]``.  The standard deviation sets
            the physical scale of Gaussian noise; the mean is the raw-domain
            fill value for time masking.
        config: Conservative augmentation magnitudes.
        generator: Optional deterministic PyTorch generator. Every stochastic
            operation in this function consumes this generator.
        time_mask_anchor: Optional half-open output-grid interval ``[start,end)``.
            When provided, the full mask is sampled inside this interval. Use
            the selected InfoNCE window support here so the anchor cannot miss
            the otherwise sparse trial-level mask.

    Returns:
        The augmented raw sensor tensor and its sampled temporal metadata.
    """

    _validate_raw_trial(raw_trial)
    if not isinstance(config, MotionAugmentationConfig):
        raise TypeError(
            "config must be MotionAugmentationConfig, got "
            f"{type(config).__name__}."
        )
    mean, std = _validate_statistics(
        raw_trial, train_channel_mean, train_channel_std
    )
    _random_device(raw_trial.device, generator)  # Validate before any mutation.

    temporal_length = int(raw_trial.shape[1])
    mask_anchor = _validated_time_mask_anchor(time_mask_anchor, temporal_length)
    shift_max = int(config.time_shift_max_samples)
    if shift_max >= temporal_length:
        raise ValueError(
            "time_shift_max_samples must be smaller than raw trial length, got "
            f"{shift_max} >= {temporal_length}."
        )
    mask_max = int(config.time_mask_max_samples)
    if mask_max > temporal_length:
        raise ValueError(
            "time_mask_max_samples cannot exceed raw trial length, got "
            f"{mask_max} > {temporal_length}."
        )
    if mask_anchor is not None and mask_max > mask_anchor[1] - mask_anchor[0]:
        raise ValueError(
            "time_mask_max_samples must fit completely inside time_mask_anchor, "
            f"got mask_max={mask_max}, anchor={mask_anchor}."
        )

    result = raw_trial.clone()
    sampled_shift = 0
    sampled_mask_start: Optional[int] = None
    sampled_mask_length = 0

    acc_lower, acc_upper = _validate_scale_range(
        config.acc_scale_range, "acc_scale_range"
    )
    gyro_lower, gyro_upper = _validate_scale_range(
        config.gyro_scale_range, "gyro_scale_range"
    )
    acc_scale = _uniform_scalar(
        acc_lower, acc_upper, reference=result, generator=generator
    )
    gyro_scale = _uniform_scalar(
        gyro_lower, gyro_upper, reference=result, generator=generator
    )
    result[:3] = result[:3] * acc_scale
    result[3:6] = result[3:6] * gyro_scale

    if config.rotation_max_degrees > 0:
        rotation = _axis_angle_rotation(
            float(config.rotation_max_degrees),
            reference=result,
            generator=generator,
        )
        # One SO(3) matrix is deliberately shared by both sensor triads.
        result[:3] = rotation @ result[:3]
        result[3:6] = rotation @ result[3:6]

    if config.noise_std_ratio > 0:
        noise = _randn(result.shape, reference=result, generator=generator)
        result = result + noise * std[:, None] * float(config.noise_std_ratio)

    if shift_max > 0:
        sampled_shift = _randint_inclusive(
            -shift_max,
            shift_max,
            reference=result,
            generator=generator,
        )
        result = _reflect_time_shift(result, sampled_shift)

    if mask_max > 0:
        sampled_mask_length = _randint_inclusive(
            int(config.time_mask_min_samples),
            mask_max,
            reference=result,
            generator=generator,
        )
        mask_region_begin = 0 if mask_anchor is None else mask_anchor[0]
        mask_region_end = temporal_length if mask_anchor is None else mask_anchor[1]
        if sampled_mask_length > mask_region_end - mask_region_begin:
            raise ValueError(
                "The sampled time mask cannot fit inside time_mask_anchor: "
                f"mask_length={sampled_mask_length}, anchor={mask_anchor}."
            )
        sampled_mask_start = _randint_inclusive(
            mask_region_begin,
            mask_region_end - sampled_mask_length,
            reference=result,
            generator=generator,
        )
        result[
            :,
            sampled_mask_start : sampled_mask_start + sampled_mask_length,
        ] = mean[:, None]

    if not bool(torch.isfinite(result).all().item()):
        raise RuntimeError("Augmentation produced NaN or infinite values.")
    metadata = MotionAugmentationMetadata(
        time_shift_samples=sampled_shift,
        time_mask_start_sample=sampled_mask_start,
        time_mask_length_samples=sampled_mask_length,
    )
    return result, metadata


def augment_raw_trial(
    raw_trial: torch.Tensor,
    train_channel_mean: torch.Tensor,
    train_channel_std: torch.Tensor,
    config: MotionAugmentationConfig,
    *,
    generator: Optional[torch.Generator] = None,
    time_mask_anchor: Optional[Tuple[int, int]] = None,
) -> torch.Tensor:
    """Create one independently augmented complete raw trial.

    ``time_mask_anchor`` is an optional half-open output sample interval. Pass the
    selected InfoNCE window support to guarantee that this anchor observes a
    short mask while augmentation remains globally consistent across every
    overlapping window. Use :func:`augment_raw_trial_with_metadata` when a
    nonzero time shift must later be mapped back exactly.
    """

    result, _ = _augment_raw_trial_with_metadata_impl(
        raw_trial,
        train_channel_mean,
        train_channel_std,
        config,
        generator=generator,
        time_mask_anchor=time_mask_anchor,
    )
    return result


def augment_raw_trial_with_metadata(
    raw_trial: torch.Tensor,
    train_channel_mean: torch.Tensor,
    train_channel_std: torch.Tensor,
    config: MotionAugmentationConfig,
    *,
    generator: Optional[torch.Generator] = None,
    time_mask_anchor: Optional[Tuple[int, int]] = None,
) -> Tuple[torch.Tensor, MotionAugmentationMetadata]:
    """Create one raw-trial view and return its temporal alignment metadata."""

    return _augment_raw_trial_with_metadata_impl(
        raw_trial,
        train_channel_mean,
        train_channel_std,
        config,
        generator=generator,
        time_mask_anchor=time_mask_anchor,
    )


def _validated_starts(
    starts: Sequence[int] | torch.Tensor,
    *,
    temporal_length: int,
    window_size: int,
    device: torch.device,
) -> torch.Tensor:
    if isinstance(starts, torch.Tensor):
        if starts.dtype == torch.bool or starts.is_floating_point() or starts.is_complex():
            raise TypeError(f"starts must use an integer dtype, got {starts.dtype}.")
        indices = starts.to(device=device, dtype=torch.long)
    else:
        if isinstance(starts, (str, bytes)):
            raise TypeError("starts must be a one-dimensional integer sequence.")
        try:
            raw_values = list(starts)
        except TypeError as error:
            raise TypeError(
                "starts must be a one-dimensional integer sequence or tensor."
            ) from error
        if any(isinstance(value, bool) or not isinstance(value, Integral) for value in raw_values):
            raise TypeError("Every element of starts must be an integer.")
        indices = torch.tensor(raw_values, dtype=torch.long, device=device)

    if indices.ndim != 1:
        raise ValueError(f"starts must be one-dimensional, got {tuple(indices.shape)}.")
    if indices.numel() == 0:
        raise ValueError("starts cannot be empty.")
    if bool((indices < 0).any().item()):
        raise ValueError("starts cannot contain negative indices.")
    if indices.numel() > 1 and not bool((indices[1:] > indices[:-1]).all().item()):
        raise ValueError("starts must be strictly increasing and contain no duplicates.")
    last_end = int(indices[-1].item()) + int(window_size)
    if last_end > temporal_length:
        raise ValueError(
            "A requested window exceeds the raw trial: "
            f"last_start+window_size={last_end} > T={temporal_length}."
        )
    return indices


def normalize_and_slice_windows(
    raw_trial: torch.Tensor,
    train_channel_mean: torch.Tensor,
    train_channel_std: torch.Tensor,
    starts: Sequence[int] | torch.Tensor,
    window_size: int,
) -> torch.Tensor:
    """Fold-normalise a complete raw trial, then extract ordered windows."""

    _validate_raw_trial(raw_trial)
    size = _positive_int(window_size, "window_size")
    if size > int(raw_trial.shape[1]):
        raise ValueError(
            f"window_size={size} exceeds trial length T={raw_trial.shape[1]}."
        )
    mean, std = _validate_statistics(
        raw_trial, train_channel_mean, train_channel_std
    )
    indices = _validated_starts(
        starts,
        temporal_length=int(raw_trial.shape[1]),
        window_size=size,
        device=raw_trial.device,
    )
    normalised = (raw_trial - mean[:, None]) / std[:, None]
    windows = torch.stack(
        [normalised[:, int(start) : int(start) + size] for start in indices],
        dim=0,
    )
    expected_shape = (int(indices.numel()), _EXPECTED_CHANNELS, size)
    if tuple(windows.shape) != expected_shape:
        raise RuntimeError(
            f"Internal window extraction error: {tuple(windows.shape)} != "
            f"{expected_shape}."
        )
    return windows


def make_augmented_trial_view(
    raw_trial: torch.Tensor,
    train_channel_mean: torch.Tensor,
    train_channel_std: torch.Tensor,
    starts: Sequence[int] | torch.Tensor,
    window_size: int,
    config: MotionAugmentationConfig,
    *,
    generator: Optional[torch.Generator] = None,
    time_mask_anchor: Optional[Tuple[int, int]] = None,
) -> torch.Tensor:
    """Augment in raw trial space, normalise, and return ``[N,6,W]``."""

    augmented = augment_raw_trial(
        raw_trial,
        train_channel_mean,
        train_channel_std,
        config,
        generator=generator,
        time_mask_anchor=time_mask_anchor,
    )
    return normalize_and_slice_windows(
        augmented,
        train_channel_mean,
        train_channel_std,
        starts,
        window_size,
    )


def make_augmented_trial_view_with_metadata(
    raw_trial: torch.Tensor,
    train_channel_mean: torch.Tensor,
    train_channel_std: torch.Tensor,
    starts: Sequence[int] | torch.Tensor,
    window_size: int,
    config: MotionAugmentationConfig,
    *,
    generator: Optional[torch.Generator] = None,
    time_mask_anchor: Optional[Tuple[int, int]] = None,
) -> Tuple[torch.Tensor, MotionAugmentationMetadata]:
    """Return an augmented ``[N,6,W]`` view plus alignment metadata."""

    augmented, metadata = augment_raw_trial_with_metadata(
        raw_trial,
        train_channel_mean,
        train_channel_std,
        config,
        generator=generator,
        time_mask_anchor=time_mask_anchor,
    )
    windows = normalize_and_slice_windows(
        augmented,
        train_channel_mean,
        train_channel_std,
        starts,
        window_size,
    )
    return windows, metadata


def make_augmented_trial_pair(
    raw_trial: torch.Tensor,
    train_channel_mean: torch.Tensor,
    train_channel_std: torch.Tensor,
    starts: Sequence[int] | torch.Tensor,
    window_size: int,
    config: MotionAugmentationConfig,
    *,
    generator: Optional[torch.Generator] = None,
    time_mask_anchor: Optional[Tuple[int, int]] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Create two independently sampled views on the same window grid.

    A single optional generator is consumed sequentially, so reseeding a new
    generator to the same seed reproduces the *pair*, while the two members of
    a pair remain independently sampled. Equal window indices are exact
    physical-time matches only when ``config.time_shift_max_samples == 0``;
    otherwise use :func:`make_augmented_trial_pair_with_metadata` to align them.
    """

    first = make_augmented_trial_view(
        raw_trial,
        train_channel_mean,
        train_channel_std,
        starts,
        window_size,
        config,
        generator=generator,
        time_mask_anchor=time_mask_anchor,
    )
    second = make_augmented_trial_view(
        raw_trial,
        train_channel_mean,
        train_channel_std,
        starts,
        window_size,
        config,
        generator=generator,
        time_mask_anchor=time_mask_anchor,
    )
    return first, second


def make_augmented_trial_pair_with_metadata(
    raw_trial: torch.Tensor,
    train_channel_mean: torch.Tensor,
    train_channel_std: torch.Tensor,
    starts: Sequence[int] | torch.Tensor,
    window_size: int,
    config: MotionAugmentationConfig,
    *,
    generator: Optional[torch.Generator] = None,
    time_mask_anchor: Optional[Tuple[int, int]] = None,
) -> Tuple[
    Tuple[torch.Tensor, MotionAugmentationMetadata],
    Tuple[torch.Tensor, MotionAugmentationMetadata],
]:
    """Create two independent views and retain each temporal mapping.

    For change-point equivariance or masked temporal prediction, the simplest
    first-round protocol is still ``time_shift_max_samples=0``. If shifts are
    enabled for the InfoNCE-only ablation, the returned metadata prevents code
    from accidentally treating equal output indices as exact physical times.
    """

    first = make_augmented_trial_view_with_metadata(
        raw_trial,
        train_channel_mean,
        train_channel_std,
        starts,
        window_size,
        config,
        generator=generator,
        time_mask_anchor=time_mask_anchor,
    )
    second = make_augmented_trial_view_with_metadata(
        raw_trial,
        train_channel_mean,
        train_channel_std,
        starts,
        window_size,
        config,
        generator=generator,
        time_mask_anchor=time_mask_anchor,
    )
    return first, second


__all__ = [
    "MotionAugmentationConfig",
    "MotionAugmentationMetadata",
    "augment_raw_trial",
    "augment_raw_trial_with_metadata",
    "make_augmented_trial_pair",
    "make_augmented_trial_pair_with_metadata",
    "make_augmented_trial_view",
    "make_augmented_trial_view_with_metadata",
    "normalize_and_slice_windows",
]
