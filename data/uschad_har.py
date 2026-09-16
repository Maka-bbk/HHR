"""Public USC-HAD-only data API for the HHR training path.

This module is an import boundary, not a second dataset implementation.  The
USC-HAD implementation already lives in :mod:`data.uschad`; importing the old
``data.get_datasets`` registry would also import CIFAR, ImageNet, CUB,
Aircraft, Cars and ``torchvision`` at module-import time.  New HHR entry
points must therefore import HAR symbols from this module and never from that
legacy image/HAR registry.

Keeping aliases (rather than wrapper functions) deliberately preserves the
audited HHR call signatures and object identities.
"""

from __future__ import annotations

from .uschad import (
    HARContrastiveTransform,
    HARPlainTransform,
    HARStrongTransform,
    HARTrialContrastiveTransform,
    HARWeakTransform,
    USCHADTrialDataset,
    USCHADWindowDataset,
    build_har_train_transform,
    build_har_trial_train_transform,
    convert_window_dataset_to_trials,
    get_uschad_datasets,
    parse_subject_ids,
    recompute_subject_train_normalization,
    uschad_trial_collate,
)


__all__ = (
    "HARContrastiveTransform",
    "HARPlainTransform",
    "HARStrongTransform",
    "HARTrialContrastiveTransform",
    "HARWeakTransform",
    "USCHADTrialDataset",
    "USCHADWindowDataset",
    "build_har_train_transform",
    "build_har_trial_train_transform",
    "convert_window_dataset_to_trials",
    "get_uschad_datasets",
    "parse_subject_ids",
    "recompute_subject_train_normalization",
    "uschad_trial_collate",
)
