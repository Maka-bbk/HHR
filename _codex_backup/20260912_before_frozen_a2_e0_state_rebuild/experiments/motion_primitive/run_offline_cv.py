"""Canonical subject-CV launcher for offline HHR training."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.motion_primitive.offline_cv_runner import (  # noqa: F401
    build_parser,
    main,
    run,
)


if __name__ == "__main__":
    main()
