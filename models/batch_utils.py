"""Device-transfer helpers for padded motion-primitive batches.

This module is intentionally representation-neutral.  A padded batch describes
an ordered sequence of local windows; it does not imply that those windows are
pooled into a whole-trial embedding.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch


_REQUIRED_BATCH_KEYS = frozenset({"windows", "positions", "mask", "lengths"})


def move_trial_batch_to_device(
    batch: Mapping[str, Any],
    device: torch.device | str,
    non_blocking: bool = True,
) -> dict[str, Any]:
    """Move tensor values in one padded sequence batch to ``device``.

    Non-tensor metadata is preserved unchanged.  Requiring the four structural
    keys catches accidental use with the old flat-window loader while keeping
    the helper independent of any pooling implementation.
    """

    if not isinstance(batch, Mapping):
        raise TypeError(
            f"Expected a padded batch mapping, got {type(batch).__name__}."
        )
    missing = _REQUIRED_BATCH_KEYS - set(batch)
    if missing:
        raise KeyError(f"Padded batch is missing keys: {sorted(missing)}.")
    return {
        key: (
            value.to(device=device, non_blocking=non_blocking)
            if isinstance(value, torch.Tensor)
            else value
        )
        for key, value in batch.items()
    }


__all__ = ["move_trial_batch_to_device"]
