"""One-command formal frozen A2/E0/state/K32 HAR-CGCD experiment.

The driver executes four identity-locked stages in order:

1. HAR window ResNet1D warm-up and fold/seed-matched A2 training;
2. frozen E0/PCA64/KMeans32 and offline state/old registry fitting;
3. strict three-session label-isolated Online CGCD;
4. post-hoc visual audit for one representative fold/seed member.

Every child stage has its own durable manifest.  ``--resume`` reuses only
verified complete members and never adopts a directory with another identity.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.motion_primitive.strict_artifacts import write_json
from experiments.motion_primitive.strict_cv_common import (
    CANONICAL_FOLDS,
    CANONICAL_SEEDS,
    parse_integer_grid,
    validate_or_create_grid_manifest,
)
from experiments.motion_primitive.strict_protocol import sha256_file


FULL_SCHEMA = "hhr_frozen_a2_e0_state_full_cv_v1"
PROFILE = "frozen_a2_e0_state_k32"


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"JSON artifact is not an object: {path}.")
    return value


def _quoted(command: Sequence[str]) -> str:
    return " ".join(f'"{item}"' for item in command)


def _run_command(command: Sequence[str], *, stage: str) -> None:
    print("[command] " + _quoted(command), flush=True)
    try:
        subprocess.run(list(command), cwd=PROJECT_ROOT, check=True)
    except subprocess.CalledProcessError as error:
        raise RuntimeError(
            f"Full experiment stage {stage!r} failed with exit code {error.returncode}."
        ) from error


def _canonical_grid(args: argparse.Namespace) -> tuple[tuple[int, ...], tuple[int, ...]]:
    folds = parse_integer_grid(args.folds, minimum=1, maximum=7)
    seeds = parse_integer_grid(args.seeds, minimum=0)
    if folds != CANONICAL_FOLDS or seeds != CANONICAL_SEEDS:
        raise ValueError(
            "The formal full command is fixed to folds 1..7 and seeds 0,5,50,500. "
            "Use the single-stage runners for smoke tests."
        )
    return folds, seeds


def _encoder_command(args: argparse.Namespace, root: Path) -> list[str]:
    command = [
        sys.executable,
        str(PROJECT_ROOT / "experiments" / "motion_primitive" / "strict_encoder_cv_runner.py"),
        "--npz-path", str(Path(args.npz_path).expanduser().resolve()),
        "--output-root", str(root / "encoders"),
        "--folds", str(args.folds),
        "--seeds", str(args.seeds),
        "--window-epochs", str(int(args.window_epochs)),
        "--window-batch-size", str(int(args.window_batch_size)),
        "--window-eval-batch-size", str(int(args.window_eval_batch_size)),
        "--window-learning-rate", str(float(args.window_learning_rate)),
        "--window-weight-decay", str(float(args.window_weight_decay)),
        "--window-weak-scale-std", str(float(args.window_weak_scale_std)),
        "--window-strong-scale-std", str(float(args.window_strong_scale_std)),
        "--a2-epochs", str(int(args.a2_epochs)),
        "--a2-trial-batch-size", str(int(args.a2_trial_batch_size)),
        "--a2-source-encode-batch-size", str(int(args.encode_batch_size)),
        "--a2-learning-rate", str(float(args.a2_learning_rate)),
        "--a2-minimum-learning-rate", str(float(args.a2_minimum_learning_rate)),
        "--a2-weight-decay", str(float(args.a2_weight_decay)),
        "--num-workers", str(int(args.num_workers)),
        "--device", str(args.device),
    ]
    if bool(args.resume):
        command.append("--resume")
    return command


def _offline_command(args: argparse.Namespace, root: Path) -> list[str]:
    command = [
        sys.executable,
        str(PROJECT_ROOT / "experiments" / "motion_primitive" / "strict_offline_cv_runner.py"),
        "--a2-root", str(root / "encoders" / "a2"),
        "--npz-path", str(Path(args.npz_path).expanduser().resolve()),
        "--output-root", str(root / "offline"),
        "--folds", str(args.folds),
        "--seeds", str(args.seeds),
        "--encode-batch-size", str(int(args.encode_batch_size)),
        "--old-distance-alpha", str(float(args.old_distance_alpha)),
        "--old-ratio-alpha", str(float(args.old_ratio_alpha)),
        "--novel-distance-alpha", str(float(args.novel_distance_alpha)),
        "--minimum-cluster-trials", str(int(args.minimum_cluster_trials)),
        "--minimum-cluster-subjects", str(int(args.minimum_cluster_subjects)),
        "--minimum-cluster-silhouette", str(float(args.minimum_cluster_silhouette)),
        "--bootstrap-replicates", str(int(args.discovery_bootstrap_replicates)),
        "--minimum-bootstrap-stability", str(float(args.minimum_bootstrap_stability)),
        "--minimum-registry-separation", str(float(args.minimum_registry_separation)),
        "--minimum-candidate-separation", str(float(args.minimum_candidate_separation)),
        "--device", str(args.device),
    ]
    if bool(args.resume):
        command.append("--resume")
    return command


def _online_command(args: argparse.Namespace, root: Path) -> list[str]:
    command = [
        sys.executable,
        str(PROJECT_ROOT / "experiments" / "motion_primitive" / "strict_online_cv_runner.py"),
        "--offline-cv-root", str(root / "offline"),
        "--output-root", str(root / "online"),
        "--folds", str(args.folds),
        "--seeds", str(args.seeds),
        "--encode-batch-size", str(int(args.encode_batch_size)),
        "--bootstrap-seed", str(int(args.summary_bootstrap_seed)),
        "--device", str(args.device),
    ]
    if bool(args.resume):
        command.append("--resume")
    return command


def _visual_command(args: argparse.Namespace, root: Path) -> list[str]:
    member = f"fold_{int(args.visual_fold):02d}_seed_{int(args.visual_seed)}"
    command = [
        sys.executable,
        str(PROJECT_ROOT / "experiments" / "motion_primitive" / "strict_visualization.py"),
        "--offline-run-dir", str(root / "offline" / member),
        "--online-run-dir", str(root / "online" / member),
        "--output-dir", str(root / "visualizations" / member),
        "--maximum-trial-plots", str(int(args.maximum_trial_plots)),
        "--sample-rate-hz", "100.0",
        "--dpi", str(int(args.visual_dpi)),
    ]
    if bool(args.resume):
        command.append("--resume")
    return command


def _implementation_hashes() -> dict[str, str]:
    names = (
        "experiments/motion_primitive/run_full_cv.py",
        "experiments/motion_primitive/strict_encoder_cv_runner.py",
        "experiments/motion_primitive/pretrain_window_encoder.py",
        "experiments/motion_primitive/train_motion_encoder.py",
        "experiments/motion_primitive/motion_encoder.py",
        "experiments/motion_primitive/motion_augmentation.py",
        "experiments/motion_primitive/raw_changepoint.py",
        "experiments/motion_primitive/motion_checkpoint.py",
        "experiments/motion_primitive/strict_offline_cv_runner.py",
        "experiments/motion_primitive/strict_offline_runner.py",
        "experiments/motion_primitive/strict_online_cv_runner.py",
        "experiments/motion_primitive/strict_online_runner.py",
        "experiments/motion_primitive/strict_visualization.py",
        "experiments/motion_primitive/strict_artifacts.py",
        "experiments/motion_primitive/strict_cv_common.py",
        "experiments/motion_primitive/strict_protocol.py",
        "experiments/motion_primitive/frozen_e0_state.py",
        "experiments/motion_primitive/strict_registry.py",
        "experiments/motion_primitive/strict_metrics.py",
        "models/resnet1d.py",
        "models/window_pretrain.py",
    )
    return {name: sha256_file(PROJECT_ROOT / name) for name in names}


def _identity(args: argparse.Namespace, folds: Sequence[int], seeds: Sequence[int]) -> dict[str, Any]:
    npz = Path(args.npz_path).expanduser().resolve()
    return {
        "schema": FULL_SCHEMA,
        "profile": PROFILE,
        "folds": list(folds),
        "seeds": list(seeds),
        "npz_path": str(npz),
        "npz_sha256": sha256_file(npz),
        "route": {
            "window_encoder": "ResNet1D_HAR_warmup",
            "motion_encoder": "A2_final_frozen",
            "segmentation": "E0_fixed_window_w256_s128",
            "codebook": "window_L2_trial_equal_PCA64_KMeans32_cosine_hard",
            "trajectory": "state_descriptor_3739_constant_filter_zscore_PCA32_L2",
            "online": "strict_label_free_append_only_activity_registry_three_sessions",
        },
        "training": {
            "window_epochs": int(args.window_epochs),
            "window_batch_size": int(args.window_batch_size),
            "window_eval_batch_size": int(args.window_eval_batch_size),
            "window_learning_rate": float(args.window_learning_rate),
            "window_weight_decay": float(args.window_weight_decay),
            "window_weak_scale_std": float(args.window_weak_scale_std),
            "window_strong_scale_std": float(args.window_strong_scale_std),
            "a2_epochs": int(args.a2_epochs),
            "a2_trial_batch_size": int(args.a2_trial_batch_size),
            "a2_learning_rate": float(args.a2_learning_rate),
            "a2_minimum_learning_rate": float(args.a2_minimum_learning_rate),
            "a2_weight_decay": float(args.a2_weight_decay),
        },
        "registry": {
            key: getattr(args, key)
            for key in (
                "old_distance_alpha",
                "old_ratio_alpha",
                "novel_distance_alpha",
                "minimum_cluster_trials",
                "minimum_cluster_subjects",
                "minimum_cluster_silhouette",
                "discovery_bootstrap_replicates",
                "minimum_bootstrap_stability",
                "minimum_registry_separation",
                "minimum_candidate_separation",
            )
        },
        "runtime": {
            "encode_batch_size": int(args.encode_batch_size),
            "num_workers": int(args.num_workers),
            "device": str(args.device),
            "summary_bootstrap_seed": int(args.summary_bootstrap_seed),
        },
        "visualization": {
            "enabled": bool(args.visualize),
            "fold": int(args.visual_fold),
            "seed": int(args.visual_seed),
            "maximum_trial_plots": int(args.maximum_trial_plots),
            "dpi": int(args.visual_dpi),
            "cross_fold_token_id_averaging": False,
        },
        "implementation_sha256": _implementation_hashes(),
    }


def _verify_stage(path: Path, *, schema: str, expected: Mapping[str, Any] = {}) -> dict[str, Any]:
    if not path.is_file():
        raise RuntimeError(f"Stage did not produce its completion marker: {path}.")
    complete = _read_json(path)
    if complete.get("schema") != schema or complete.get("complete") is not True:
        raise RuntimeError(f"Stage completion marker has another schema/status: {path}.")
    for key, value in expected.items():
        if complete.get(key) != value:
            raise RuntimeError(f"Stage completion field {key!r} differs in {path}.")
    return complete


def run(args: argparse.Namespace) -> dict[str, Any]:
    folds, seeds = _canonical_grid(args)
    if int(args.visual_fold) not in folds or int(args.visual_seed) not in seeds:
        raise ValueError("Representative visual fold/seed must belong to the formal grid.")
    if int(args.maximum_trial_plots) < 0 or int(args.visual_dpi) < 50:
        raise ValueError("Visualization limits are invalid.")
    root = Path(args.output_root).expanduser().resolve()
    identity = _identity(args, folds, seeds)
    validate_or_create_grid_manifest(root, identity)
    commands = (
        _encoder_command(args, root),
        _offline_command(args, root),
        _online_command(args, root),
    )
    visual_command = _visual_command(args, root)
    if bool(args.dry_run):
        return {
            "schema": FULL_SCHEMA,
            "dry_run": True,
            "commands": [*commands, *([visual_command] if bool(args.visualize) else [])],
        }
    for stage, command in zip(("encoders", "offline", "online"), commands):
        _run_command(command, stage=stage)
    encoder_complete = _verify_stage(
        root / "encoders" / "complete.json",
        schema="hhr_frozen_a2_encoder_cv_v1",
        expected={"folds": list(folds), "seeds": list(seeds), "member_count": 28},
    )
    offline_complete = _verify_stage(
        root / "offline" / "complete.json",
        schema="hhr_frozen_a2_e0_state_offline_cv_v1",
        expected={
            "profile": PROFILE,
            "folds": list(folds),
            "seeds": list(seeds),
            "member_count": 28,
        },
    )
    online_complete = _verify_stage(
        root / "online" / "complete.json",
        schema="hhr_frozen_a2_e0_state_online_cv_v1",
        expected={
            "profile": PROFILE,
            "folds": list(folds),
            "seeds": list(seeds),
            "member_count": 28,
            "session_row_count": 84,
        },
    )
    visual_manifest: Path | None = None
    visual_complete: dict[str, Any] | None = None
    if bool(args.visualize):
        _run_command(visual_command, stage="visualization")
        member = f"fold_{int(args.visual_fold):02d}_seed_{int(args.visual_seed)}"
        visual_manifest = root / "visualizations" / member / "visualization_manifest.json"
        visual_complete = _verify_stage(
            visual_manifest,
            schema="hhr_strict_cgcd_visualization_v1",
            expected={"profile": PROFILE, "fold": int(args.visual_fold), "seed": int(args.visual_seed)},
        )
    completion = {
        "schema": FULL_SCHEMA,
        "profile": PROFILE,
        "folds": list(folds),
        "seeds": list(seeds),
        "encoder_member_count": int(encoder_complete["member_count"]),
        "offline_member_count": int(offline_complete["member_count"]),
        "online_member_count": int(online_complete["member_count"]),
        "online_session_row_count": int(online_complete["session_row_count"]),
        "aggregation_unit": "fold_after_averaging_four_seeds_within_fold",
        "primary_metric_layer": "old_fixed_novel_hungarian",
        "stage_complete_sha256": {
            "encoders": sha256_file(root / "encoders" / "complete.json"),
            "offline": sha256_file(root / "offline" / "complete.json"),
            "online": sha256_file(root / "online" / "complete.json"),
            **(
                {"visualization": sha256_file(visual_manifest)}
                if visual_manifest is not None else {}
            ),
        },
        "stage_grid_manifest_sha256": {
            stage: sha256_file(root / stage / "grid_manifest.json")
            for stage in ("encoders", "offline", "online")
        },
        "visualization": visual_complete,
        "complete": True,
    }
    write_json(root / "complete.json", completion)
    return completion


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the complete formal frozen A2/E0/state/K32 HAR-CGCD experiment."
    )
    parser.add_argument("--npz-path", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--folds", default=",".join(map(str, CANONICAL_FOLDS)))
    parser.add_argument("--seeds", default=",".join(map(str, CANONICAL_SEEDS)))
    parser.add_argument("--window-epochs", type=int, default=60)
    parser.add_argument("--window-batch-size", type=int, default=64)
    parser.add_argument("--window-eval-batch-size", type=int, default=256)
    parser.add_argument("--window-learning-rate", type=float, default=0.1)
    parser.add_argument("--window-weight-decay", type=float, default=5e-4)
    parser.add_argument("--window-weak-scale-std", type=float, default=0.1)
    parser.add_argument("--window-strong-scale-std", type=float, default=0.2)
    parser.add_argument("--a2-epochs", type=int, default=30)
    parser.add_argument("--a2-trial-batch-size", type=int, default=8)
    parser.add_argument("--a2-learning-rate", type=float, default=1e-4)
    parser.add_argument("--a2-minimum-learning-rate", type=float, default=1e-6)
    parser.add_argument("--a2-weight-decay", type=float, default=1e-4)
    parser.add_argument("--encode-batch-size", type=int, default=512)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--old-distance-alpha", type=float, default=0.05)
    parser.add_argument("--old-ratio-alpha", type=float, default=0.05)
    parser.add_argument("--novel-distance-alpha", type=float, default=0.05)
    parser.add_argument("--minimum-cluster-trials", type=int, default=3)
    parser.add_argument("--minimum-cluster-subjects", type=int, default=2)
    parser.add_argument("--minimum-cluster-silhouette", type=float, default=0.20)
    parser.add_argument("--discovery-bootstrap-replicates", type=int, default=100)
    parser.add_argument("--minimum-bootstrap-stability", type=float, default=0.80)
    parser.add_argument("--minimum-registry-separation", type=float, default=0.10)
    parser.add_argument("--minimum-candidate-separation", type=float, default=0.10)
    parser.add_argument("--summary-bootstrap-seed", type=int, default=20260912)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--visualize", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--visual-fold", type=int, default=1)
    parser.add_argument("--visual-seed", type=int, default=0)
    parser.add_argument("--maximum-trial-plots", type=int, default=0)
    parser.add_argument("--visual-dpi", type=int, default=180)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> dict[str, Any]:
    result = run(build_parser().parse_args(argv))
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return result


if __name__ == "__main__":
    main()
