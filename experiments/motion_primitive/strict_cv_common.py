"""Shared grid, identity, and aggregation helpers for strict HHR CV."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from experiments.motion_primitive.strict_artifacts import write_json


CANONICAL_FOLDS = tuple(range(1, 8))
CANONICAL_SEEDS = (0, 5, 50, 500)
METRICS = ("all_accuracy", "old_accuracy", "new_accuracy", "h_score", "macro_f1")


def parse_integer_grid(
    value: str,
    *,
    minimum: int,
    maximum: int | None = None,
) -> tuple[int, ...]:
    tokens = [item.strip() for item in str(value).split(",") if item.strip()]
    if not tokens:
        raise ValueError("Experiment grid cannot be empty.")
    values = tuple(int(item) for item in tokens)
    if len(values) != len(set(values)):
        raise ValueError(f"Experiment grid contains duplicates: {values}.")
    if any(item < minimum or (maximum is not None and item > maximum) for item in values):
        raise ValueError("Experiment grid contains an out-of-range value.")
    return tuple(sorted(values))


def canonical_hash(payload: Mapping[str, Any]) -> str:
    text = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def validate_or_create_grid_manifest(root: Path, identity: Mapping[str, Any]) -> None:
    path = root / "grid_manifest.json"
    expected = {**dict(identity), "identity_sha256": canonical_hash(identity)}
    if path.is_file():
        observed = json.loads(path.read_text(encoding="utf-8"))
        if observed != expected:
            raise RuntimeError(
                f"Output root {root} records another experiment identity; use a new directory."
            )
    else:
        if root.exists() and any(root.iterdir()):
            raise RuntimeError(
                f"Existing output root {root} has artifacts but no grid_manifest.json; "
                "refusing to mix or adopt an unidentified experiment."
            )
        root.mkdir(parents=True, exist_ok=True)
        write_json(path, expected)


def find_a2_checkpoint(root: str | Path, fold: int, seed: int) -> Path:
    """Resolve the sole canonical final-epoch A2 checkpoint for a member.

    A validation-best checkpoint is a different selection policy and must not
    silently enter the registered frozen route when the final checkpoint is
    absent.
    """

    base = Path(root).expanduser().resolve()
    directories = (
        base / f"fold_{int(fold):02d}_seed_{int(seed)}",
        base / f"fold_{int(fold):02d}_seed_{int(seed)}_A2_formal_v1",
    )
    filename = "motion_encoder_final.pt"
    existing = [directory / filename for directory in directories if (directory / filename).is_file()]
    if len(existing) > 1:
        raise RuntimeError(
            f"Duplicate A2 {filename} candidates for fold={fold}, seed={seed}: "
            f"{[str(path) for path in existing]}."
        )
    if existing:
        return existing[0]
    best = [
        directory / "motion_encoder_best.pt"
        for directory in directories
        if (directory / "motion_encoder_best.pt").is_file()
    ]
    if best:
        raise RuntimeError(
            f"Only validation-best A2 checkpoint(s) exist for fold={fold}, seed={seed}; "
            "the registered route requires motion_encoder_final.pt."
        )
    raise RuntimeError(
        f"No motion_encoder_final.pt found for fold={fold}, seed={seed} below {base}."
    )


def member_directory(root: str | Path, fold: int, seed: int) -> Path:
    return Path(root).expanduser().resolve() / f"fold_{int(fold):02d}_seed_{int(seed)}"


def bootstrap_fold_mean(
    values: Sequence[float],
    *,
    seed: int,
    replicates: int = 10_000,
) -> dict[str, float]:
    data = np.asarray(values, dtype=np.float64)
    if data.ndim != 1 or not len(data) or not np.all(np.isfinite(data)):
        raise ValueError("Bootstrap needs a finite non-empty fold vector.")
    rng = np.random.default_rng(int(seed))
    estimates = np.empty(int(replicates), dtype=np.float64)
    for index in range(int(replicates)):
        estimates[index] = rng.choice(data, size=len(data), replace=True).mean()
    return {
        "mean": float(data.mean()),
        "std_across_folds": float(data.std(ddof=1)) if len(data) > 1 else 0.0,
        "ci95_low": float(np.quantile(estimates, 0.025)),
        "ci95_high": float(np.quantile(estimates, 0.975)),
        "fold_count": int(len(data)),
    }


def aggregate_online_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    folds: Sequence[int],
    seeds: Sequence[int],
    bootstrap_seed: int,
) -> dict[str, Any]:
    expected = {
        (int(fold), int(seed), int(session))
        for fold in folds for seed in seeds for session in (1, 2, 3)
    }
    observed = [(int(row["fold"]), int(row["seed"]), int(row["session"])) for row in rows]
    if len(observed) != len(set(observed)):
        raise RuntimeError("Online CV rows contain duplicate fold/seed/session keys.")
    missing = sorted(expected - set(observed))
    extra = sorted(set(observed) - expected)
    if missing or extra:
        raise RuntimeError(f"Online CV grid incomplete: missing={missing}, extra={extra}.")
    result: dict[str, Any] = {
        "aggregation_unit": "fold_after_averaging_seeds_within_fold",
        "folds": list(folds),
        "seeds": list(seeds),
        "row_count": len(rows),
        "sessions": {},
    }
    for session in (1, 2, 3):
        session_rows = [row for row in rows if int(row["session"]) == session]
        metric_result: dict[str, Any] = {}
        for metric in METRICS:
            fold_values = []
            for fold in folds:
                values = [float(row[metric]) for row in session_rows if int(row["fold"]) == int(fold)]
                if len(values) != len(seeds):
                    raise RuntimeError("A fold does not contain the expected seed count.")
                fold_values.append(float(np.mean(values)))
            metric_result[metric] = {
                **bootstrap_fold_mean(
                    fold_values,
                    seed=int(bootstrap_seed) + 100 * session + METRICS.index(metric),
                ),
                "fold_means": fold_values,
            }
        result["sessions"][str(session)] = metric_result
    return result


__all__ = [
    "CANONICAL_FOLDS",
    "CANONICAL_SEEDS",
    "METRICS",
    "aggregate_online_rows",
    "canonical_hash",
    "find_a2_checkpoint",
    "member_directory",
    "parse_integer_grid",
    "validate_or_create_grid_manifest",
]
