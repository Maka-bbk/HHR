"""Upgrade completed runs to tie-aware 1-NN and valid-RLE order controls.

This is a deterministic post-processing migration. It does not refit the
encoder, PCA, or KMeans codebook, and it preserves one backup of the legacy
derived metrics before replacing them.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.motion_primitive.core import (
    distance_matrix_from_histograms,
    distance_matrix_from_sequences,
    length_distance_matrix,
)
from experiments.motion_primitive.run_experiment import (
    analyze_distance_matrix,
    derive_feasibility_decision,
    order_shuffle_control,
    write_json,
)


REVISION = "tie_aware_1nn_and_valid_rle_shuffle_v2"


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def read_records(path: Path) -> list[dict]:
    records = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not records:
        raise RuntimeError(f"No trial records found in {path}.")
    return records


def backup_once(source: Path, backup_name: str) -> None:
    backup = source.with_name(backup_name)
    if source.is_file() and not backup.exists():
        shutil.copy2(source, backup)


def recompute_run(run_dir: Path, force: bool) -> bool:
    config_path = run_dir / "experiment_config.json"
    summary_path = run_dir / "summary.json"
    records_path = run_dir / "trial_primitive_sequences.jsonl"
    for path in [config_path, summary_path, records_path]:
        if not path.is_file():
            raise RuntimeError(f"Incomplete run {run_dir}; missing {path.name}.")

    config = read_json(config_path)
    postprocessing = config.get("postprocessing", {})
    if postprocessing.get("metric_revision") == REVISION and not force:
        summary = read_json(summary_path)
        feasibility = derive_feasibility_decision(
            summary["primitive_statistics"],
            summary["sequence_association_metrics"],
            summary["order_shuffle_control"],
        )
        summary["feasibility_decision"] = feasibility
        write_json(summary_path, summary)
        write_json(run_dir / "feasibility_decision.json", feasibility)
        print(f"[refresh-decision] {run_dir}", flush=True)
        return False

    arguments = config["arguments"]
    label_permutations = int(arguments["label_permutations"])
    order_shuffles = int(arguments["order_shuffles"])
    old_class_count = int(arguments["old_class_count"])
    seed = int(arguments["seed"])
    if label_permutations <= 0 or order_shuffles <= 0:
        raise RuntimeError(
            f"Run {run_dir} has non-positive permutation/shuffle counts."
        )

    records = read_records(records_path)
    labels = np.asarray(
        [record["activity_label_0based"] for record in records], dtype=np.int64
    )
    subjects = np.asarray(
        [record["subject_id"] for record in records], dtype=np.int64
    )
    matrices = {}
    for variant_name in ["full", "nonoverlap_every_second_window", "edge_trimmed"]:
        sequences = [
            record["sequence_variants"][variant_name]["rle_tokens"]
            for record in records
        ]
        output_name = (
            "nonoverlap" if variant_name.startswith("nonoverlap") else variant_name
        )
        matrices[f"rle_sequence_{output_name}"] = distance_matrix_from_sequences(
            sequences
        )
    histograms = np.asarray(
        [
            record["sequence_variants"]["full"]["primitive_histogram"]
            for record in records
        ],
        dtype=np.float32,
    )
    matrices["primitive_histogram_full"] = distance_matrix_from_histograms(
        histograms
    )
    window_counts = np.asarray(
        [record["sequence_variants"]["full"]["window_count"] for record in records],
        dtype=np.int64,
    )
    matrices["window_count_only"] = length_distance_matrix(window_counts)

    association_metrics = {}
    for offset, (name, matrix) in enumerate(matrices.items()):
        association_metrics[name] = analyze_distance_matrix(
            matrix,
            labels,
            subjects,
            old_class_count,
            label_permutations,
            seed + offset * 10007,
        )
    full_sequences = [
        record["sequence_variants"]["full"]["rle_tokens"] for record in records
    ]
    order_control = order_shuffle_control(
        full_sequences,
        matrices["rle_sequence_full"],
        labels,
        subjects,
        old_class_count,
        order_shuffles,
        seed + 70001,
    )

    summary = read_json(summary_path)
    feasibility = derive_feasibility_decision(
        summary["primitive_statistics"], association_metrics, order_control
    )

    backup_once(
        run_dir / "sequence_association_metrics.json",
        "sequence_association_metrics_legacy_first_tie_v1.json",
    )
    backup_once(
        run_dir / "order_shuffle_control.json",
        "order_shuffle_control_legacy_invalid_rle_v1.json",
    )
    backup_once(summary_path, "summary_legacy_sequence_metrics_v1.json")

    write_json(run_dir / "sequence_association_metrics.json", association_metrics)
    write_json(run_dir / "order_shuffle_control.json", order_control)
    write_json(run_dir / "feasibility_decision.json", feasibility)
    summary["sequence_association_metrics"] = association_metrics
    summary["order_shuffle_control"] = order_control
    summary["feasibility_decision"] = feasibility
    write_json(summary_path, summary)

    config["order_shuffle_control"] = {
        "algorithm": "valid_rle_permutation_v2",
        "equal_adjacent_tokens_allowed": False,
        "interpretation": (
            "Heuristic Monte Carlo control over valid RLE arrangements; the "
            "fallback scheduler is not a uniform sampler over all arrangements."
        ),
    }
    config["postprocessing"] = {
        "metric_revision": REVISION,
        "updated_utc": datetime.now(timezone.utc).isoformat(),
        "legacy_metric_backups_preserved": True,
    }
    write_json(config_path, config)
    write_json(
        run_dir / "sequence_metric_revision.json",
        {
            "revision": REVISION,
            "reason": (
                "Use equal credit across exact 1-NN distance ties and prevent "
                "equal adjacent tokens in shuffled RLE controls."
            ),
            "encoder_pca_codebook_refit": False,
            "legacy_metric_backups_preserved": True,
        },
    )
    print(f"[upgraded] {run_dir}", flush=True)
    return True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", required=True)
    parser.add_argument(
        "--folds",
        default="",
        help="Optional comma-separated fold numbers for parallel/restartable migration.",
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root).expanduser().resolve()
    selected_folds = {
        int(token.strip()) for token in args.folds.split(",") if token.strip()
    }
    run_dirs = sorted(
        path
        for path in output_root.glob("fold_*_seed_*_k*")
        if path.is_dir()
        and (path / "summary.json").is_file()
        and (
            not selected_folds
            or int(path.name.split("_", maxsplit=2)[1]) in selected_folds
        )
    )
    if not run_dirs:
        raise RuntimeError(f"No completed run directories found in {output_root}.")
    upgraded = sum(recompute_run(run_dir, bool(args.force)) for run_dir in run_dirs)
    print(
        json.dumps(
            {
                "revision": REVISION,
                "run_count": len(run_dirs),
                "upgraded_count": upgraded,
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
