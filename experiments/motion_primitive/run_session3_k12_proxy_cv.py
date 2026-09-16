"""Public canonical 7-fold x 4-seed launcher for the final Session-3 K12 proxy."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.motion_primitive.strict_session3_k12_proxy_cv import build_parser, main, run


if __name__ == "__main__":
    main()
