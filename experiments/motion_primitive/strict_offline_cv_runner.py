"""Multi-fold/multi-seed orchestration for frozen A2/E0/state offline state."""

from __future__ import annotations

import argparse
import json
import platform
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import sklearn
import torch

from experiments.motion_primitive.strict_artifacts import write_csv, write_json
from experiments.motion_primitive.strict_cv_common import (
    CANONICAL_FOLDS,
    CANONICAL_SEEDS,
    canonical_hash,
    find_a2_checkpoint,
    member_directory,
    parse_integer_grid,
    validate_or_create_grid_manifest,
)
from experiments.motion_primitive.strict_offline_runner import (
    OFFLINE_SCHEMA,
    PROFILE,
    RUN_IDENTITY_SCHEMA,
    validate_completed_output,
)
from experiments.motion_primitive.strict_protocol import sha256_file


CV_SCHEMA = "hhr_frozen_a2_e0_state_offline_cv_v1"
MEMBER_SCHEMA = "hhr_frozen_a2_e0_state_offline_cv_member_v1"
_MEMBER_PATTERN = re.compile(r"^fold_(\d{2})_seed_(\d+)$")


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"Cannot read JSON artifact {path}.") from error
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON artifact {path} is not an object.")
    return value


def _verified_identity(path: Path) -> dict[str, Any]:
    payload = _read_json(path)
    recorded = payload.get("identity_sha256")
    identity = {key: value for key, value in payload.items() if key != "identity_sha256"}
    if not isinstance(recorded, str) or canonical_hash(identity) != recorded:
        raise RuntimeError(f"Identity SHA256 does not verify in {path}.")
    return payload


def _resolved_device(value: str) -> str:
    requested = str(value).lower()
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested for Offline CV but is unavailable.")
    return str(device)


def _implementation_hashes() -> dict[str, str]:
    relative = (
        "models/resnet1d.py",
        "experiments/motion_primitive/motion_encoder.py",
        "experiments/motion_primitive/motion_checkpoint.py",
        "experiments/motion_primitive/frozen_e0_state.py",
        "experiments/motion_primitive/strict_artifacts.py",
        "experiments/motion_primitive/strict_protocol.py",
        "experiments/motion_primitive/strict_registry.py",
        "experiments/motion_primitive/strict_offline_runner.py",
        "experiments/motion_primitive/strict_offline_cv_runner.py",
    )
    return {name: sha256_file(PROJECT_ROOT / name) for name in relative}


def _validate_member_directory_set(
    root: Path,
    folds: Sequence[int],
    seeds: Sequence[int],
    *,
    allow_missing: bool,
) -> None:
    expected = {(int(fold), int(seed)) for fold in folds for seed in seeds}
    observed: set[tuple[int, int]] = set()
    malformed: list[str] = []
    if root.is_dir():
        for child in root.iterdir():
            if not child.is_dir() or not child.name.startswith("fold_"):
                continue
            matched = _MEMBER_PATTERN.fullmatch(child.name)
            if matched is None:
                malformed.append(child.name)
                continue
            observed.add((int(matched.group(1)), int(matched.group(2))))
    missing = sorted(expected - observed)
    extra = sorted(observed - expected)
    if malformed or extra or (missing and not allow_missing):
        raise RuntimeError(
            "Offline member directory set is ambiguous: "
            f"malformed={sorted(malformed)}, missing={missing}, extra={extra}."
        )


def _checkpoint_members(
    a2_root: str | Path,
    folds: Sequence[int],
    seeds: Sequence[int],
) -> dict[tuple[int, int], Path]:
    members: dict[tuple[int, int], Path] = {}
    for fold in folds:
        for seed in seeds:
            checkpoint = find_a2_checkpoint(a2_root, int(fold), int(seed))
            if checkpoint.name != "motion_encoder_final.pt":
                raise RuntimeError("Strict Offline CV requires the final-epoch A2 checkpoint.")
            members[(int(fold), int(seed))] = checkpoint.resolve()
    return members


def _grid_identity(
    args: argparse.Namespace,
    *,
    folds: Sequence[int],
    seeds: Sequence[int],
    npz_path: Path,
    checkpoints: Mapping[tuple[int, int], Path],
) -> dict[str, Any]:
    return {
        "schema": CV_SCHEMA,
        "profile": PROFILE,
        "folds": list(folds),
        "seeds": list(seeds),
        "npz_path": str(npz_path),
        "npz_sha256": sha256_file(npz_path),
        "a2_root": str(Path(args.a2_root).expanduser().resolve()),
        "a2_members": [
            {
                "fold": int(fold),
                "seed": int(seed),
                "path": str(checkpoints[(int(fold), int(seed))]),
                "sha256": sha256_file(checkpoints[(int(fold), int(seed))]),
            }
            for fold in folds for seed in seeds
        ],
        "route": {
            "encoder": "A2_final_frozen_content",
            "segmentation": "E0_fixed_window_w256_s128",
            "codebook": "trial_equal_PCA64_KMeans32_cosine_hard",
            "trajectory": "state_3739_train_only_PCA32_L2",
        },
        "registry": {
            key: getattr(args, key)
            for key in (
                "old_distance_alpha", "old_ratio_alpha", "novel_distance_alpha",
                "minimum_cluster_trials", "minimum_cluster_subjects",
                "minimum_cluster_silhouette", "bootstrap_replicates",
                "minimum_bootstrap_stability", "minimum_registry_separation",
                "minimum_candidate_separation",
            )
        },
        "runtime": {
            "encode_batch_size": int(args.encode_batch_size),
            "requested_device": str(args.device),
            "resolved_device": _resolved_device(args.device),
            "allow_smoke_a2": bool(args.allow_smoke_a2),
            "python": platform.python_version(),
            "numpy": np.__version__,
            "sklearn": sklearn.__version__,
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
        },
        "implementation_sha256": _implementation_hashes(),
    }


def _member_identity(
    *,
    grid_identity_sha256: str,
    grid_identity: Mapping[str, Any],
    fold: int,
    seed: int,
    checkpoint: Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
    return {
        "schema": MEMBER_SCHEMA,
        "grid_identity_sha256": str(grid_identity_sha256),
        "profile": PROFILE,
        "fold": int(fold),
        "seed": int(seed),
        "npz_path": str(grid_identity["npz_path"]),
        "npz_sha256": str(grid_identity["npz_sha256"]),
        "a2_checkpoint_path": str(checkpoint.resolve()),
        "a2_checkpoint_sha256": sha256_file(checkpoint),
        "registry": dict(grid_identity["registry"]),
        "encode_batch_size": int(args.encode_batch_size),
        "requested_device": str(args.device),
        "resolved_device": _resolved_device(args.device),
        "allow_smoke_a2": bool(args.allow_smoke_a2),
    }


def _ensure_member_identity(
    target: Path,
    identity: Mapping[str, Any],
    *,
    resume: bool,
) -> None:
    path = target / "cv_member_manifest.json"
    expected = {**dict(identity), "identity_sha256": canonical_hash(identity)}
    if target.exists() and any(target.iterdir()) and not path.is_file():
        raise RuntimeError(f"Existing Offline member has no CV identity: {target}.")
    if path.is_file():
        if _verified_identity(path) != expected:
            raise RuntimeError(f"Offline member contains another identity: {target}.")
    else:
        target.mkdir(parents=True, exist_ok=True)
        write_json(path, expected)
    other = [item for item in target.iterdir() if item.name != path.name]
    if other and not bool(resume):
        raise FileExistsError(
            f"Offline member already contains run artifacts: {target}; use --resume."
        )


def _member_command(args, fold: int, seed: int, a2_checkpoint: Path, output: Path) -> list[str]:
    command = [
        sys.executable,
        str(PROJECT_ROOT / "experiments" / "motion_primitive" / "strict_offline_runner.py"),
        "--a2-checkpoint", str(a2_checkpoint),
        "--npz-path", str(Path(args.npz_path).expanduser().resolve()),
        "--output-dir", str(output),
        "--profile", PROFILE,
        "--fold", str(int(fold)),
        "--seed", str(int(seed)),
        "--window-size", "256",
        "--window-stride", "128",
        "--primitive-num", "32",
        "--pca-dim", "64",
        "--descriptor-pca-dim", "32",
        "--include-state",
        "--encode-batch-size", str(int(args.encode_batch_size)),
        "--old-distance-alpha", str(float(args.old_distance_alpha)),
        "--old-ratio-alpha", str(float(args.old_ratio_alpha)),
        "--novel-distance-alpha", str(float(args.novel_distance_alpha)),
        "--minimum-cluster-trials", str(int(args.minimum_cluster_trials)),
        "--minimum-cluster-subjects", str(int(args.minimum_cluster_subjects)),
        "--minimum-cluster-silhouette", str(float(args.minimum_cluster_silhouette)),
        "--bootstrap-replicates", str(int(args.bootstrap_replicates)),
        "--minimum-bootstrap-stability", str(float(args.minimum_bootstrap_stability)),
        "--minimum-registry-separation", str(float(args.minimum_registry_separation)),
        "--minimum-candidate-separation", str(float(args.minimum_candidate_separation)),
        "--no-shuffle-novel-classes",
        "--device", str(args.device),
        "--resume",
    ]
    if bool(args.allow_smoke_a2):
        command.append("--allow-smoke-a2")
    return command


def _validate_member(
    path: Path,
    fold: int,
    seed: int,
    *,
    expected_cv_identity: Mapping[str, Any],
) -> dict[str, Any]:
    cv_identity_path = path / "cv_member_manifest.json"
    run_identity_path = path / "run_identity.json"
    complete_path = path / "complete.json"
    manifest_path = path / "manifest.json"
    if not all(
        item.is_file()
        for item in (cv_identity_path, run_identity_path, complete_path, manifest_path)
    ):
        raise RuntimeError(f"Offline member is incomplete: {path}.")
    expected_cv = {
        **dict(expected_cv_identity),
        "identity_sha256": canonical_hash(expected_cv_identity),
    }
    if _verified_identity(cv_identity_path) != expected_cv:
        raise RuntimeError(f"Offline member CV identity changed: {path}.")
    run_identity = _verified_identity(run_identity_path)
    expected_fields = {
        "schema": RUN_IDENTITY_SCHEMA,
        "profile": PROFILE,
        "fold": int(fold),
        "seed": int(seed),
        "npz_path": str(expected_cv_identity["npz_path"]),
        "npz_sha256": str(expected_cv_identity["npz_sha256"]),
        "a2_checkpoint_path": str(expected_cv_identity["a2_checkpoint_path"]),
        "a2_checkpoint_sha256": str(expected_cv_identity["a2_checkpoint_sha256"]),
    }
    for key, expected in expected_fields.items():
        if run_identity.get(key) != expected:
            raise RuntimeError(f"Offline run identity field {key!r} differs: {path}.")
    runtime = run_identity.get("runtime")
    if not isinstance(runtime, Mapping):
        raise RuntimeError(f"Offline run identity lacks runtime settings: {path}.")
    for key in ("encode_batch_size", "requested_device", "resolved_device", "allow_smoke_a2"):
        if runtime.get(key) != expected_cv_identity[key]:
            raise RuntimeError(f"Offline run runtime field {key!r} differs: {path}.")
    if run_identity.get("registry_config") != expected_cv_identity["registry"]:
        raise RuntimeError(f"Offline registry parameters differ: {path}.")
    complete = validate_completed_output(path, expected_identity=run_identity)
    return complete


def _aggregate(rows: Sequence[dict[str, Any]], folds: Sequence[int], seeds: Sequence[int]) -> dict[str, Any]:
    keys = [(int(row["fold"]), int(row["seed"])) for row in rows]
    expected = {(int(fold), int(seed)) for fold in folds for seed in seeds}
    if len(keys) != len(set(keys)) or set(keys) != expected:
        raise RuntimeError("Offline result grid has duplicate, missing, or extra members.")
    metrics = ("old_accuracy", "macro_f1", "balanced_accuracy", "unknown_fraction")
    result = {"aggregation_unit": "fold_after_averaging_seeds_within_fold", "metrics": {}}
    for metric in metrics:
        fold_values = []
        for fold in folds:
            values = [float(row[metric]) for row in rows if int(row["fold"]) == int(fold)]
            fold_values.append(float(np.mean(values)))
        result["metrics"][metric] = {
            "mean": float(np.mean(fold_values)),
            "std_across_folds": float(np.std(fold_values, ddof=1)) if len(fold_values) > 1 else 0.0,
            "fold_means": fold_values,
        }
    return result


def run(args: argparse.Namespace) -> dict[str, Any]:
    folds = parse_integer_grid(args.folds, minimum=1, maximum=7)
    seeds = parse_integer_grid(args.seeds, minimum=0)
    root = Path(args.output_root).expanduser().resolve()
    npz_path = Path(args.npz_path).expanduser().resolve()
    if not npz_path.is_file():
        raise FileNotFoundError(npz_path)
    if int(args.encode_batch_size) < 1:
        raise ValueError("encode-batch-size must be positive.")
    checkpoints = _checkpoint_members(args.a2_root, folds, seeds)
    identity = _grid_identity(
        args,
        folds=folds,
        seeds=seeds,
        npz_path=npz_path,
        checkpoints=checkpoints,
    )
    validate_or_create_grid_manifest(root, identity)
    grid_identity_sha256 = canonical_hash(identity)
    _validate_member_directory_set(root, folds, seeds, allow_missing=True)
    if bool(args.dry_run):
        commands = []
        for fold in folds:
            for seed in seeds:
                a2 = checkpoints[(int(fold), int(seed))]
                commands.append(_member_command(args, fold, seed, a2, member_directory(root, fold, seed)))
        return {"schema": CV_SCHEMA, "dry_run": True, "commands": commands}
    rows: list[dict[str, Any]] = []
    for fold in folds:
        for seed in seeds:
            target = member_directory(root, fold, seed)
            a2 = checkpoints[(int(fold), int(seed))]
            member_identity = _member_identity(
                grid_identity_sha256=grid_identity_sha256,
                grid_identity=identity,
                fold=int(fold),
                seed=int(seed),
                checkpoint=a2,
                args=args,
            )
            _ensure_member_identity(target, member_identity, resume=bool(args.resume))
            complete_path = target / "complete.json"
            if complete_path.is_file():
                # A corrupt completed member is evidence, not scratch space.
                # It is never overwritten under --resume.
                complete = _validate_member(
                    target,
                    fold,
                    seed,
                    expected_cv_identity=member_identity,
                )
                if not bool(args.resume):
                    raise FileExistsError(
                        f"Completed Offline member already exists: {target}; use --resume."
                    )
            else:
                command = _member_command(args, fold, seed, a2, target)
                print("[command] " + " ".join(f'"{item}"' for item in command), flush=True)
                try:
                    subprocess.run(command, cwd=PROJECT_ROOT, check=True)
                except subprocess.CalledProcessError as error:
                    raise RuntimeError(
                        f"Offline member fold={fold} seed={seed} failed with exit code {error.returncode}."
                    ) from error
                complete = _validate_member(
                    target,
                    fold,
                    seed,
                    expected_cv_identity=member_identity,
                )
            metrics = complete["outer_test"]
            rows.append({
                "profile": PROFILE,
                "fold": int(fold),
                "seed": int(seed),
                "old_accuracy": float(metrics["old_accuracy"]),
                "macro_f1": float(metrics["macro_f1"]),
                "balanced_accuracy": float(metrics["balanced_accuracy"]),
                "unknown_fraction": float(metrics["unknown_fraction"]),
                "representation_sha256": complete["representation_sha256"],
                "old_registry_state_sha256": complete["old_registry_state_sha256"],
            })
            write_csv(root / "offline_runs.csv", rows)
    _validate_member_directory_set(root, folds, seeds, allow_missing=False)
    summary = {
        "schema": CV_SCHEMA,
        "profile": PROFILE,
        "folds": list(folds),
        "seeds": list(seeds),
        "member_count": len(rows),
        "grid_identity_sha256": grid_identity_sha256,
        "aggregate": _aggregate(rows, folds, seeds),
    }
    write_json(root / "offline_summary.json", summary)
    complete = {
        **summary,
        "grid_manifest_sha256": sha256_file(root / "grid_manifest.json"),
        "offline_summary_sha256": sha256_file(root / "offline_summary.json"),
        "offline_runs_sha256": sha256_file(root / "offline_runs.csv"),
        "complete": True,
    }
    write_json(root / "complete.json", complete)
    return complete


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Frozen A2/E0/state offline subject CV.")
    parser.add_argument("--a2-root", required=True)
    parser.add_argument("--npz-path", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--folds", default=",".join(map(str, CANONICAL_FOLDS)))
    parser.add_argument("--seeds", default=",".join(map(str, CANONICAL_SEEDS)))
    parser.add_argument("--encode-batch-size", type=int, default=512)
    parser.add_argument("--old-distance-alpha", type=float, default=0.05)
    parser.add_argument("--old-ratio-alpha", type=float, default=0.05)
    parser.add_argument("--novel-distance-alpha", type=float, default=0.05)
    parser.add_argument("--minimum-cluster-trials", type=int, default=3)
    parser.add_argument("--minimum-cluster-subjects", type=int, default=2)
    parser.add_argument("--minimum-cluster-silhouette", type=float, default=0.20)
    parser.add_argument("--bootstrap-replicates", type=int, default=100)
    parser.add_argument("--minimum-bootstrap-stability", type=float, default=0.80)
    parser.add_argument("--minimum-registry-separation", type=float, default=0.10)
    parser.add_argument("--minimum-candidate-separation", type=float, default=0.10)
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
