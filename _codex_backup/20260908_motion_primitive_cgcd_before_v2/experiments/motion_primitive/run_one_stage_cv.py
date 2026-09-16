"""Run and aggregate the seven-fold J0-U/J0-T screening grid."""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
TRAIN_SCRIPT = Path(__file__).resolve().with_name("train_one_stage.py")
PROFILES = ("J0-U", "J0-T")
METRICS = ("all_accuracy", "old_accuracy", "new_accuracy", "h_score", "macro_f1")


def _parse_ints(value: str, *, minimum: int | None = None, maximum: int | None = None) -> list[int]:
    pieces = [item.strip() for item in str(value).replace(" ", ",").split(",") if item.strip()]
    if not pieces:
        raise ValueError("Expected at least one integer.")
    result = [int(item) for item in pieces]
    if len(result) != len(set(result)):
        raise ValueError(f"Duplicate integer in {value!r}.")
    if minimum is not None and any(item < minimum for item in result):
        raise ValueError(f"Values must be >= {minimum}.")
    if maximum is not None and any(item > maximum for item in result):
        raise ValueError(f"Values must be <= {maximum}.")
    return result


def _parse_profiles(value: str) -> list[str]:
    result = [item.strip().upper() for item in str(value).split(",") if item.strip()]
    if not result or len(result) != len(set(result)):
        raise ValueError("--profiles must contain unique J0-U/J0-T values.")
    unknown = sorted(set(result) - set(PROFILES))
    if unknown:
        raise ValueError(f"Unknown profiles: {unknown}.")
    return result


def _canonical_hash(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    temporary.replace(path)


def _bootstrap_interval(values: Sequence[float], replicates: int, seed: int) -> dict[str, float]:
    data = np.asarray(values, dtype=np.float64)
    if data.ndim != 1 or not len(data) or not np.all(np.isfinite(data)):
        raise ValueError("Bootstrap values must be a finite non-empty vector.")
    rng = np.random.default_rng(int(seed))
    indices = rng.integers(0, len(data), size=(int(replicates), len(data)))
    means = data[indices].mean(axis=1)
    return {
        "mean": float(data.mean()),
        "standard_deviation": float(data.std(ddof=1)) if len(data) > 1 else 0.0,
        "ci95_low": float(np.quantile(means, 0.025)),
        "ci95_high": float(np.quantile(means, 0.975)),
    }


def _exact_sign_flip_paired(differences: Sequence[float]) -> dict[str, Any]:
    values = np.asarray(differences, dtype=np.float64)
    if values.ndim != 1 or not len(values) or not np.all(np.isfinite(values)):
        raise ValueError("Paired differences must be finite and non-empty.")
    observed = float(values.mean())
    permuted = []
    for signs in itertools.product((-1.0, 1.0), repeat=len(values)):
        permuted.append(float(np.mean(values * np.asarray(signs))))
    permuted_array = np.asarray(permuted)
    return {
        "mean_J0_T_minus_J0_U": observed,
        "exact_one_sided_p_greater": float(np.mean(permuted_array >= observed - 1e-15)),
        "fold_difference_values": values.tolist(),
        "permutation_count": int(len(permuted_array)),
    }


def _command(args: argparse.Namespace, profile: str, fold: int, seed: int, output: Path) -> list[str]:
    command = [
        sys.executable,
        str(TRAIN_SCRIPT),
        "--data-root",
        str(args.data_root),
        "--output-dir",
        str(output),
        "--profile",
        profile,
        "--fold",
        str(fold),
        "--seed",
        str(seed),
        "--epochs",
        str(args.epochs),
        "--batch-size",
        str(args.batch_size),
        "--eval-batch-size",
        str(args.eval_batch_size),
        "--num-workers",
        str(args.num_workers),
        "--device",
        str(args.device),
        "--learning-rate",
        str(args.learning_rate),
        "--weight-decay",
        str(args.weight_decay),
        "--gradient-clip",
        str(args.gradient_clip),
        "--warmup-epochs",
        str(args.warmup_epochs),
        "--early-stop-patience",
        str(args.early_stop_patience),
        "--frame-size",
        str(args.frame_size),
        "--frame-stride",
        str(args.frame_stride),
        "--codebook-size",
        str(args.codebook_size),
        "--temperature-start",
        str(args.temperature_start),
        "--temperature-end",
        str(args.temperature_end),
        "--trajectory-mask-ratio",
        str(args.trajectory_mask_ratio),
        "--labelled-fraction",
        str(args.labelled_fraction),
        "--anomaly-policy",
        str(args.anomaly_policy),
        "--cluster-restarts",
        str(args.cluster_restarts),
        "--order-shuffles",
        str(args.order_shuffles),
        "--minimum-segment-windows",
        str(args.minimum_segment_windows),
        "--deterministic" if args.deterministic else "--no-deterministic",
    ]
    if args.resume:
        command.append("--resume")
    return command


def _load_rows(root: Path, profiles: Sequence[str], folds: Sequence[int], seeds: Sequence[int]) -> list[dict[str, Any]]:
    rows = []
    for profile in profiles:
        for fold in folds:
            for seed in seeds:
                run_dir = root / f"profile_{profile}" / f"fold_{fold:02d}_seed_{seed}"
                complete = run_dir / "complete.json"
                summary_path = run_dir / "summary.json"
                if not complete.is_file() or not summary_path.is_file():
                    raise RuntimeError(f"Incomplete expected run: {run_dir}.")
                summary = json.loads(summary_path.read_text(encoding="utf-8"))
                if (
                    summary.get("profile") != profile
                    or int(summary.get("fold", -1)) != int(fold)
                    or int(summary.get("seed", -1)) != int(seed)
                ):
                    raise RuntimeError(f"Run identity fields disagree in {summary_path}.")
                metrics = summary["evaluation"]["cgcd_metrics"]
                row = {
                    "profile": profile,
                    "fold": int(fold),
                    "seed": int(seed),
                    "run_dir": str(run_dir),
                    "selected_epoch": int(summary["selected_epoch"]),
                    **{key: float(metrics[key]) for key in METRICS},
                    "order_h_score_drop": float(
                        summary["evaluation"]["order_shuffle_control"][
                            "identity_control_minus_shuffled"
                        ]["h_score"]
                    ),
                    "effective_code_count": float(
                        summary["evaluation"]["codebook_diagnostics"][
                            "effective_code_count"
                        ]
                    ),
                    "token_subject_nmi": float(
                        summary["evaluation"]["codebook_diagnostics"][
                            "token_subject_nmi"
                        ]
                    ),
                }
                rows.append(row)
    return rows


def _aggregate(rows: Sequence[dict[str, Any]], profiles: Sequence[str], folds: Sequence[int], seeds: Sequence[int], bootstrap_replicates: int, seed: int) -> dict[str, Any]:
    expected_count = len(profiles) * len(folds) * len(seeds)
    if len(rows) != expected_count:
        raise RuntimeError(f"Expected {expected_count} result rows, got {len(rows)}.")
    aggregate: dict[str, Any] = {
        "grid": {
            "profiles": list(profiles),
            "folds": list(folds),
            "seeds": list(seeds),
            "run_count": len(rows),
        },
        "profiles": {},
    }
    fold_means_by_profile: dict[str, dict[int, dict[str, float]]] = {}
    for profile_index, profile in enumerate(profiles):
        selected = [item for item in rows if item["profile"] == profile]
        fold_means = {}
        for fold in folds:
            fold_rows = [item for item in selected if item["fold"] == fold]
            if len(fold_rows) != len(seeds):
                raise RuntimeError(f"Unbalanced seeds for profile={profile}, fold={fold}.")
            fold_means[int(fold)] = {
                key: float(np.mean([item[key] for item in fold_rows]))
                for key in (*METRICS, "order_h_score_drop", "effective_code_count", "token_subject_nmi")
            }
        fold_means_by_profile[profile] = fold_means
        aggregate["profiles"][profile] = {
            key: _bootstrap_interval(
                [fold_means[fold][key] for fold in folds],
                int(bootstrap_replicates),
                int(seed) + 1009 * profile_index + 31 * key_index,
            )
            for key_index, key in enumerate(
                (*METRICS, "order_h_score_drop", "effective_code_count", "token_subject_nmi")
            )
        }
        aggregate["profiles"][profile]["fold_seed_averages"] = fold_means

    if set(PROFILES).issubset(profiles):
        paired = {}
        for metric in METRICS:
            differences = [
                fold_means_by_profile["J0-T"][fold][metric]
                - fold_means_by_profile["J0-U"][fold][metric]
                for fold in folds
            ]
            paired[metric] = {
                **_exact_sign_flip_paired(differences),
                "bootstrap_ci": _bootstrap_interval(
                    differences, int(bootstrap_replicates), int(seed) + 7001
                ),
            }
        aggregate["paired_J0_T_vs_J0_U"] = paired
    aggregate["statistical_unit"] = (
        "subject fold after averaging seeds; folds have overlapping train sets, so p-values are screening evidence"
    )
    return aggregate


def _write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    fields = [
        "profile",
        "fold",
        "seed",
        *METRICS,
        "order_h_score_drop",
        "effective_code_count",
        "token_subject_nmi",
        "selected_epoch",
        "run_dir",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _plot_aggregate(path: Path, aggregate: dict[str, Any], profiles: Sequence[str]) -> None:
    metrics = ("all_accuracy", "old_accuracy", "new_accuracy", "h_score")
    x = np.arange(len(metrics))
    width = 0.34 if len(profiles) == 2 else 0.7 / max(1, len(profiles))
    figure, axis = plt.subplots(figsize=(9, 5.5), constrained_layout=True)
    for index, profile in enumerate(profiles):
        values = [aggregate["profiles"][profile][metric]["mean"] for metric in metrics]
        low = [
            value - aggregate["profiles"][profile][metric]["ci95_low"]
            for value, metric in zip(values, metrics)
        ]
        high = [
            aggregate["profiles"][profile][metric]["ci95_high"] - value
            for value, metric in zip(values, metrics)
        ]
        offset = (index - (len(profiles) - 1) / 2.0) * width
        axis.bar(
            x + offset,
            values,
            width=width,
            label=profile,
            yerr=np.asarray([low, high]),
            capsize=4,
        )
    axis.set_xticks(x, ["All", "Old", "New", "H-score"])
    axis.set_ylim(0.0, 1.0)
    axis.set_ylabel("Accuracy / harmonic score")
    axis.set_title("One-stage motion-primitive trajectory: fold-blocked screening")
    axis.legend()
    axis.grid(axis="y", alpha=0.25)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Seven-fold one-stage J0 grid runner.")
    default_data = r"D:\WorkDir\DataSet\USC-HAD" if os.name == "nt" else "/mnt/d/WorkDir/DataSet/USC-HAD"
    parser.add_argument("--data-root", default=default_data)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--profiles", default="J0-U,J0-T")
    parser.add_argument("--folds", default="1,2,3,4,5,6,7")
    parser.add_argument("--seeds", default="50")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--gradient-clip", type=float, default=5.0)
    parser.add_argument("--warmup-epochs", type=int, default=10)
    parser.add_argument("--early-stop-patience", type=int, default=20)
    parser.add_argument("--frame-size", type=int, default=128)
    parser.add_argument("--frame-stride", type=int, default=64)
    parser.add_argument("--codebook-size", type=int, default=32)
    parser.add_argument("--temperature-start", type=float, default=2.0)
    parser.add_argument("--temperature-end", type=float, default=0.25)
    parser.add_argument("--trajectory-mask-ratio", type=float, default=0.15)
    parser.add_argument("--labelled-fraction", type=float, default=0.8)
    parser.add_argument("--anomaly-policy", choices=("report", "exclude"), default="report")
    parser.add_argument("--cluster-restarts", type=int, default=10)
    parser.add_argument("--order-shuffles", type=int, default=10)
    parser.add_argument("--minimum-segment-windows", type=int, default=2)
    parser.add_argument("--bootstrap-replicates", type=int, default=10000)
    parser.add_argument("--aggregate-seed", type=int, default=20260908)
    parser.add_argument("--deterministic", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def run(args: argparse.Namespace) -> dict[str, Any]:
    profiles = _parse_profiles(args.profiles)
    folds = _parse_ints(args.folds, minimum=1, maximum=7)
    seeds = _parse_ints(args.seeds)
    for name in (
        "epochs",
        "batch_size",
        "eval_batch_size",
        "frame_size",
        "frame_stride",
        "codebook_size",
        "cluster_restarts",
        "order_shuffles",
        "bootstrap_replicates",
    ):
        if int(getattr(args, name)) < 1:
            raise ValueError(f"--{name.replace('_', '-')} must be positive.")
    output_root = Path(args.output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    identity = {
        "schema": "one_stage_motion_trajectory_cv_v1",
        "profiles": profiles,
        "folds": folds,
        "seeds": seeds,
        "data_root": str(Path(args.data_root).expanduser().resolve()),
        "training_parameters": {
            name: getattr(args, name)
            for name in (
                "epochs",
                "batch_size",
                "eval_batch_size",
                "num_workers",
                "device",
                "learning_rate",
                "weight_decay",
                "gradient_clip",
                "warmup_epochs",
                "early_stop_patience",
                "frame_size",
                "frame_stride",
                "codebook_size",
                "temperature_start",
                "temperature_end",
                "trajectory_mask_ratio",
                "labelled_fraction",
                "anomaly_policy",
                "cluster_restarts",
                "order_shuffles",
                "minimum_segment_windows",
                "deterministic",
            )
        },
        "one_stage_train_script": str(TRAIN_SCRIPT),
    }
    identity["identity_sha256"] = _canonical_hash(identity)
    manifest_path = output_root / "cv_manifest.json"
    if manifest_path.exists():
        recorded = json.loads(manifest_path.read_text(encoding="utf-8"))
        if recorded.get("identity_sha256") != identity["identity_sha256"]:
            raise RuntimeError(
                "Output root records a different CV identity/member set; use a new --output-root."
            )
    elif any(output_root.iterdir()):
        raise RuntimeError(
            f"Non-empty output root has no CV manifest: {output_root}."
        )
    else:
        _write_json(manifest_path, identity)

    commands = []
    for profile in profiles:
        for fold in folds:
            for seed in seeds:
                run_dir = output_root / f"profile_{profile}" / f"fold_{fold:02d}_seed_{seed}"
                command = _command(args, profile, fold, seed, run_dir)
                commands.append(command)
                print("[command] " + subprocess.list2cmdline(command), flush=True)
                if not args.dry_run:
                    subprocess.run(command, cwd=PROJECT_ROOT, check=True)
    if args.dry_run:
        return {"dry_run": True, "commands": commands, "identity": identity}

    rows = _load_rows(output_root, profiles, folds, seeds)
    aggregate = _aggregate(
        rows,
        profiles,
        folds,
        seeds,
        int(args.bootstrap_replicates),
        int(args.aggregate_seed),
    )
    result = {"identity": identity, "rows": rows, "aggregate": aggregate}
    _write_json(output_root / "aggregate_summary.json", result)
    _write_csv(output_root / "per_run_metrics.csv", rows)
    _plot_aggregate(output_root / "aggregate_metrics.png", aggregate, profiles)
    return result


def main() -> None:
    result = run(build_parser().parse_args())
    if result.get("dry_run"):
        print(f"dry-run generated {len(result['commands'])} commands", flush=True)
    else:
        for profile, metrics in result["aggregate"]["profiles"].items():
            print(
                f"{profile}: all={metrics['all_accuracy']['mean']:.4f} "
                f"old={metrics['old_accuracy']['mean']:.4f} "
                f"new={metrics['new_accuracy']['mean']:.4f} "
                f"H={metrics['h_score']['mean']:.4f}",
                flush=True,
            )


if __name__ == "__main__":
    main()
