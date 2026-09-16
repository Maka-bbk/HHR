"""Canonical 7-fold x 4-seed wrapper for the strict Session-3 K12 proxy."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from experiments.motion_primitive.strict_artifacts import write_csv, write_json
from experiments.motion_primitive.strict_cv_common import (
    CANONICAL_FOLDS,
    CANONICAL_SEEDS,
    METRICS,
    canonical_hash,
    member_directory,
    parse_integer_grid,
)
from experiments.motion_primitive.strict_online_cv_runner import validate_offline_grid
from experiments.motion_primitive.strict_protocol import sha256_file
from experiments.motion_primitive.strict_session3_k12_proxy import (
    CLUSTER_COUNT,
    DESCRIPTOR_PCA_DIM,
    SCHEMA as MEMBER_RUN_SCHEMA,
    validate_completed_output,
)


SCHEMA = "hhr_strict_session3_k12_proxy_cv_v1"
MEMBER_SCHEMA = "hhr_strict_session3_k12_proxy_cv_member_v1"
EXPECTED_MEMBER_COUNT = 28


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"JSON artifact {path} is not an object.")
    return payload


def _verified_identity(path: Path) -> dict[str, Any]:
    payload = _read_json(path)
    body = {key: value for key, value in payload.items() if key != "identity_sha256"}
    if canonical_hash(body) != payload.get("identity_sha256"):
        raise RuntimeError(f"Identity SHA256 does not verify in {path}.")
    return payload


def _implementation_hashes() -> dict[str, str]:
    names = (
        "experiments/motion_primitive/strict_session3_k12_proxy.py",
        "experiments/motion_primitive/strict_session3_k12_proxy_cv.py",
        "experiments/motion_primitive/strict_protocol.py",
        "experiments/motion_primitive/frozen_e0_state.py",
        "experiments/motion_primitive/strict_metrics.py",
    )
    return {name: sha256_file(PROJECT_ROOT / name) for name in names}


def validate_canonical_grid(
    folds: Sequence[int], seeds: Sequence[int]
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    fold_grid = tuple(sorted(int(value) for value in folds))
    seed_grid = tuple(sorted(int(value) for value in seeds))
    if fold_grid != CANONICAL_FOLDS or seed_grid != CANONICAL_SEEDS:
        raise ValueError("Formal K12 proxy CV requires folds=1..7 and seeds=0,5,50,500.")
    return fold_grid, seed_grid


def _grid_identity(
    args: argparse.Namespace,
    *,
    offline_root: Path,
    offline_grid: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema": SCHEMA,
        "offline_cv_root": str(offline_root),
        "offline_grid_manifest_sha256": sha256_file(offline_root / "grid_manifest.json"),
        "offline_grid_identity_sha256": offline_grid["identity_sha256"],
        "folds": list(CANONICAL_FOLDS),
        "seeds": list(CANONICAL_SEEDS),
        "expected_member_count": EXPECTED_MEMBER_COUNT,
        "incoming_sessions": [1, 2, 3],
        "evaluation_session": 3,
        "expected_incoming_count": 78,
        "expected_evaluation_count": 42,
        "cluster_count": CLUSTER_COUNT,
        "descriptor_pca_dim": DESCRIPTOR_PCA_DIM,
        "kmeans_n_init": int(args.kmeans_n_init),
        "kmeans_max_iter": int(args.kmeans_max_iter),
        "constant_tolerance": float(args.constant_tolerance),
        "encode_batch_size": int(args.encode_batch_size),
        "device": str(args.device),
        "allow_smoke_a2": bool(args.allow_smoke_a2),
        "implementation_sha256": _implementation_hashes(),
    }


def _ensure_identity(path: Path, body: Mapping[str, Any]) -> dict[str, Any]:
    expected = {**dict(body), "identity_sha256": canonical_hash(body)}
    if path.is_file():
        if _verified_identity(path) != expected:
            raise RuntimeError(f"Output identity differs: {path}.")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        write_json(path, expected)
    return expected


def _member_body(
    grid_sha: str,
    offline: Any,
    args: argparse.Namespace,
) -> dict[str, Any]:
    return {
        "schema": MEMBER_SCHEMA,
        "grid_identity_sha256": str(grid_sha),
        "fold": int(offline.fold),
        "seed": int(offline.seed),
        "offline_run_dir": str(offline.path),
        "offline_manifest_sha256": str(offline.manifest_sha256),
        "offline_complete_sha256": str(offline.complete_sha256),
        "offline_representation_sha256": str(offline.representation_sha256),
        "cluster_count": CLUSTER_COUNT,
        "descriptor_pca_dim": DESCRIPTOR_PCA_DIM,
        "kmeans_n_init": int(args.kmeans_n_init),
        "kmeans_max_iter": int(args.kmeans_max_iter),
        "constant_tolerance": float(args.constant_tolerance),
        "encode_batch_size": int(args.encode_batch_size),
        "device": str(args.device),
        "allow_smoke_a2": bool(args.allow_smoke_a2),
    }


def _member_command(args: argparse.Namespace, offline: Any, output: Path) -> list[str]:
    command = [
        sys.executable,
        str(PROJECT_ROOT / "experiments" / "motion_primitive" / "strict_session3_k12_proxy.py"),
        "--offline-run-dir", str(offline.path),
        "--output-dir", str(output),
        "--fold", str(int(offline.fold)),
        "--seed", str(int(offline.seed)),
        "--cluster-count", str(CLUSTER_COUNT),
        "--descriptor-pca-dim", str(DESCRIPTOR_PCA_DIM),
        "--constant-tolerance", str(float(args.constant_tolerance)),
        "--kmeans-n-init", str(int(args.kmeans_n_init)),
        "--kmeans-max-iter", str(int(args.kmeans_max_iter)),
        "--encode-batch-size", str(int(args.encode_batch_size)),
        "--device", str(args.device),
        "--resume",
    ]
    if bool(args.allow_smoke_a2):
        command.append("--allow-smoke-a2")
    return command


def _read_member(target: Path, fold: int, seed: int) -> dict[str, Any]:
    cv_identity = _verified_identity(target / "cv_member_manifest.json")
    run_identity = _verified_identity(target / "run_identity.json")
    if cv_identity.get("schema") != MEMBER_SCHEMA:
        raise RuntimeError(f"Member CV schema differs: {target}.")
    if (int(cv_identity["fold"]), int(cv_identity["seed"])) != (int(fold), int(seed)):
        raise RuntimeError(f"Member CV fold/seed differs: {target}.")
    complete = validate_completed_output(target, expected_identity=run_identity)
    if complete.get("schema") != MEMBER_RUN_SCHEMA:
        raise RuntimeError(f"Member run schema differs: {target}.")
    return {
        "fold": int(fold),
        "seed": int(seed),
        **{metric: float(complete[metric]) for metric in METRICS},
        "incoming_count": int(complete["incoming_count"]),
        "evaluation_count": int(complete["evaluation_count"]),
        "cluster_count": int(complete["cluster_count"]),
        "member_complete_sha256": sha256_file(target / "complete.json"),
    }


def _write_reports(root: Path, rows: Sequence[Mapping[str, Any]], grid_sha: str) -> dict[str, Any]:
    if len(rows) != EXPECTED_MEMBER_COUNT:
        raise RuntimeError(f"Expected 28 proxy members, observed {len(rows)}.")
    keys = [(int(row["fold"]), int(row["seed"])) for row in rows]
    expected = {(fold, seed) for fold in CANONICAL_FOLDS for seed in CANONICAL_SEEDS}
    if len(keys) != len(set(keys)) or set(keys) != expected:
        raise RuntimeError("Proxy fold/seed grid is duplicate or incomplete.")
    ordered = sorted((dict(row) for row in rows), key=lambda row: (row["fold"], row["seed"]))
    write_csv(root / "k12_proxy_runs.csv", ordered)
    aggregate: dict[str, Any] = {}
    for metric in METRICS:
        fold_means = [
            float(np.mean([row[metric] for row in ordered if int(row["fold"]) == fold]))
            for fold in CANONICAL_FOLDS
        ]
        aggregate[metric] = {
            "mean": float(np.mean(fold_means)),
            "std_across_folds": float(np.std(fold_means, ddof=1)),
            "fold_means_after_averaging_four_seeds": fold_means,
        }
    summary = {
        "schema": SCHEMA,
        "grid_identity_sha256": str(grid_sha),
        "member_count": EXPECTED_MEMBER_COUNT,
        "folds": list(CANONICAL_FOLDS),
        "seeds": list(CANONICAL_SEEDS),
        "primary_layer": "global_hungarian_scoring_only",
        "statistical_unit": "held_out_subject_fold_after_averaging_four_seeds",
        "incoming_count_per_member": 78,
        "evaluation_count_per_member": 42,
        "cluster_count": CLUSTER_COUNT,
        "aggregate": aggregate,
    }
    write_json(root / "k12_proxy_summary.json", summary)
    complete = {
        **summary,
        "k12_proxy_runs_sha256": sha256_file(root / "k12_proxy_runs.csv"),
        "k12_proxy_summary_sha256": sha256_file(root / "k12_proxy_summary.json"),
        "complete": True,
    }
    write_json(root / "complete.json", complete)
    return complete


def validate_args(args: argparse.Namespace) -> argparse.Namespace:
    folds = parse_integer_grid(args.folds, minimum=1, maximum=7)
    seeds = parse_integer_grid(args.seeds, minimum=0)
    validate_canonical_grid(folds, seeds)
    if int(args.encode_batch_size) < 1 or int(args.kmeans_n_init) < 1 or int(args.kmeans_max_iter) < 1:
        raise ValueError("Batch size and KMeans parameters must be positive.")
    if float(args.constant_tolerance) <= 0:
        raise ValueError("constant-tolerance must be positive.")
    return args


def run(args: argparse.Namespace) -> dict[str, Any]:
    args = validate_args(args)
    offline_root = Path(args.offline_cv_root).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    offline_grid, members = validate_offline_grid(
        offline_root, CANONICAL_FOLDS, CANONICAL_SEEDS
    )
    body = _grid_identity(args, offline_root=offline_root, offline_grid=offline_grid)
    if output_root.exists() and any(output_root.iterdir()) and not (output_root / "grid_manifest.json").is_file():
        raise RuntimeError("Non-empty output root has no grid identity; refusing adoption.")
    grid_identity = _ensure_identity(output_root / "grid_manifest.json", body)
    grid_sha = str(grid_identity["identity_sha256"])
    commands = [
        _member_command(args, members[(fold, seed)], member_directory(output_root, fold, seed))
        for fold in CANONICAL_FOLDS for seed in CANONICAL_SEEDS
    ]
    if bool(args.dry_run):
        return {"schema": SCHEMA, "dry_run": True, "member_count": len(commands), "commands": commands}

    rows: list[dict[str, Any]] = []
    for fold in CANONICAL_FOLDS:
        for seed in CANONICAL_SEEDS:
            offline = members[(fold, seed)]
            target = member_directory(output_root, fold, seed)
            cv_path = target / "cv_member_manifest.json"
            if target.exists() and any(target.iterdir()) and not cv_path.is_file():
                raise RuntimeError(f"Existing member has no CV identity: {target}.")
            _ensure_identity(cv_path, _member_body(grid_sha, offline, args))
            if (target / "complete.json").is_file():
                if not bool(args.resume):
                    raise FileExistsError(f"Completed member exists: {target}; use --resume.")
            else:
                command = _member_command(args, offline, target)
                print("[command] " + " ".join(f'\"{item}\"' for item in command), flush=True)
                try:
                    subprocess.run(command, cwd=PROJECT_ROOT, check=True)
                except subprocess.CalledProcessError as error:
                    raise RuntimeError(
                        f"K12 proxy fold={fold} seed={seed} failed with exit code {error.returncode}."
                    ) from error
            rows.append(_read_member(target, fold, seed))
            write_csv(output_root / "k12_proxy_runs.partial.csv", rows)
    return _write_reports(output_root, rows, grid_sha)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Strict final Session-3 K12 trajectory proxy over canonical 7-fold x 4-seed CV."
    )
    parser.add_argument("--offline-cv-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--folds", default="1,2,3,4,5,6,7")
    parser.add_argument("--seeds", default="0,5,50,500")
    parser.add_argument("--cluster-count", type=int, default=CLUSTER_COUNT, choices=(CLUSTER_COUNT,))
    parser.add_argument("--descriptor-pca-dim", type=int, default=DESCRIPTOR_PCA_DIM, choices=(DESCRIPTOR_PCA_DIM,))
    parser.add_argument("--constant-tolerance", type=float, default=1e-10)
    parser.add_argument("--kmeans-n-init", type=int, default=50)
    parser.add_argument("--kmeans-max-iter", type=int, default=300)
    parser.add_argument("--encode-batch-size", type=int, default=512)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--allow-smoke-a2", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> dict[str, Any]:
    result = run(build_parser().parse_args(argv))
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return result


if __name__ == "__main__":
    main()


__all__ = ["EXPECTED_MEMBER_COUNT", "MEMBER_SCHEMA", "SCHEMA", "build_parser", "main", "run", "validate_args", "validate_canonical_grid"]
