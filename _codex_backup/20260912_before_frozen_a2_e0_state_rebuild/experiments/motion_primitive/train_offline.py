"""Canonical offline entry for single-stage motion-primitive HAR-CGCD."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.motion_primitive.offline_trainer import (  # noqa: F401
    build_datasets,
    build_model,
    build_parser,
    evaluate,
    get_uschad_datasets,
    load_trial_window_encoder,
    main,
    parse_args,
    parse_subject_ids,
    resolve_device,
    train,
    training_step,
    uschad_trial_collate,
    validate_args,
)


if __name__ == "__main__":
    main()
