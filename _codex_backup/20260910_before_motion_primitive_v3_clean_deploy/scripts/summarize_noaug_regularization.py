#!/usr/bin/env python3
"""Summarize validation-only offline screening runs."""

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev


METRIC_NAMES = (
    "macro_f1",
    "overall_accuracy",
    "mean_class_accuracy",
)
GRADIENT_LOSS_NAMES = ("cls", "cluster", "contrast", "supcon")
GRADIENT_PAIRS = tuple(
    f"{name_a}__{name_b}"
    for index, name_a in enumerate(GRADIENT_LOSS_NAMES)
    for name_b in GRADIENT_LOSS_NAMES[index + 1:]
)


def find_config_name(run_dir: Path, root: Path) -> str:
    for parent in run_dir.parents:
        if parent == root.parent:
            break
        if parent.name.endswith("_offline"):
            return parent.name[:-len("_offline")]
    return "unknown"


def load_json(path: Path):
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def load_epoch_rows(path: Path):
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def read_seed(run_dir: Path):
    args_path = run_dir / "args.txt"
    if not args_path.exists():
        return None
    match = re.search(r"'seed':\s*(\d+)", args_path.read_text(encoding="utf-8"))
    return int(match.group(1)) if match else None


def sample_std(values):
    return stdev(values) if len(values) > 1 else 0.0


def write_gradient_summaries(root, output_prefix):
    gradient_files = sorted(root.rglob("offline_gradient_diagnostics.jsonl"))
    if not gradient_files:
        return

    gradient_rows = []
    for gradient_file in gradient_files:
        run_dir = gradient_file.parent
        for record in load_epoch_rows(gradient_file):
            row = {
                "config": find_config_name(run_dir, root),
                "run_dir": str(run_dir.relative_to(root)),
                "seed": read_seed(run_dir),
                "epoch": int(record["epoch"]),
                "batch_index": int(record["batch_index"]),
                "auxiliary_loss_scale": float(record["auxiliary_loss_scale"]),
            }
            for name in GRADIENT_LOSS_NAMES:
                row[f"loss_{name}"] = float(record["loss_values"][name])
                row[f"weight_{name}"] = float(
                    record["effective_loss_weights"][name]
                )
                row[f"raw_gradient_norm_{name}"] = float(
                    record["raw_gradient_norms"][name]
                )
                row[f"weighted_gradient_norm_{name}"] = float(
                    record["weighted_gradient_norms"][name]
                )
            for pair in GRADIENT_PAIRS:
                value = record["pairwise_gradient_cosines"][pair]
                row[f"gradient_cosine_{pair}"] = (
                    None if value is None else float(value)
                )
            gradient_rows.append(row)

    gradient_csv = root / f"{output_prefix}_gradient_diagnostics.csv"
    with gradient_csv.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(gradient_rows[0]))
        writer.writeheader()
        writer.writerows(gradient_rows)

    grouped = defaultdict(list)
    for row in gradient_rows:
        grouped[(row["config"], row["epoch"])].append(row)

    summary_rows = []
    identity_fields = {
        "config", "run_dir", "seed", "epoch", "batch_index"
    }
    numeric_fields = [
        key for key in gradient_rows[0]
        if key not in identity_fields
    ]
    for (config, epoch), rows in sorted(grouped.items()):
        summary = {
            "config": config,
            "epoch": epoch,
            "num_runs": len(rows),
        }
        for key in numeric_fields:
            values = [float(row[key]) for row in rows if row[key] is not None]
            summary[f"mean_{key}"] = mean(values) if values else None
            summary[f"std_{key}"] = sample_std(values) if values else None
        summary_rows.append(summary)

    gradient_summary_csv = root / f"{output_prefix}_gradient_summary.csv"
    with gradient_summary_csv.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)

    print(f"Wrote {len(gradient_rows)} gradient rows to {gradient_csv}")
    print(
        f"Wrote {len(summary_rows)} aggregated gradient rows to "
        f"{gradient_summary_csv}"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Summarize no-augmentation regularization screening outputs."
    )
    parser.add_argument("--root", required=True, help="Screening result root directory.")
    parser.add_argument(
        "--output-prefix",
        default="noaug_regularization",
        help="Prefix for generated CSV/JSON files.",
    )
    args = parser.parse_args()

    root = Path(args.root).resolve()
    epoch_files = sorted(root.rglob("offline_epoch_metrics.jsonl"))
    if not epoch_files:
        raise FileNotFoundError(f"No offline_epoch_metrics.jsonl files found under {root}")

    run_rows = []
    for epoch_file in epoch_files:
        run_dir = epoch_file.parent
        validation_file = run_dir / "offline_best_validation_classification_metrics.json"
        final_test_file = run_dir / "offline_best_classification_metrics.json"
        metric_file = final_test_file if final_test_file.exists() else validation_file
        if not metric_file.exists():
            raise FileNotFoundError(
                f"Missing selected-checkpoint metrics beside {epoch_file}"
            )

        rows = load_epoch_rows(epoch_file)
        if not rows:
            raise RuntimeError(f"No epoch metrics in {epoch_file}")
        selected = load_json(metric_file)
        best = max(rows, key=lambda row: row["validation_metrics"]["macro_f1"])
        last = rows[-1]
        validation_macro_f1 = [
            float(epoch_row["validation_metrics"]["macro_f1"])
            for epoch_row in rows
        ]
        tail_size = min(10, len(validation_macro_f1))
        one_epoch_drops = [
            previous - current
            for previous, current in zip(
                validation_macro_f1,
                validation_macro_f1[1:],
            )
        ]
        best_index = max(
            range(len(validation_macro_f1)),
            key=validation_macro_f1.__getitem__,
        )
        post_peak_one_epoch_drops = one_epoch_drops[best_index:]

        row = {
            "config": find_config_name(run_dir, root),
            "run_dir": str(run_dir.relative_to(root)),
            "seed": read_seed(run_dir),
            "evaluation_source": "final_test" if final_test_file.exists() else "validation_only",
            "epochs_ran": len(rows),
            "selection_epoch": int(selected["selection_epoch"]),
            "selection_metric": selected["selection_metric"],
            "selection_score": float(selected["selection_score"]),
            "validation_peak_epoch": int(best["epoch"]),
            "validation_peak_macro_f1": float(best["validation_metrics"]["macro_f1"]),
            "validation_last_macro_f1": float(last["validation_metrics"]["macro_f1"]),
            "validation_macro_f1_drop": float(
                best["validation_metrics"]["macro_f1"]
                - last["validation_metrics"]["macro_f1"]
            ),
            "validation_tail10_macro_f1": mean(validation_macro_f1[-tail_size:]),
            "validation_max_one_epoch_drop": max(one_epoch_drops, default=0.0),
            "validation_max_post_peak_one_epoch_drop": max(
                post_peak_one_epoch_drops,
                default=0.0,
            ),
            "validation_peak_accuracy": float(best["validation_metrics"]["overall_accuracy"]),
            "validation_last_accuracy": float(last["validation_metrics"]["overall_accuracy"]),
            "train_accuracy_at_peak": float(best["train_supervised_accuracy"]),
            "train_accuracy_last": float(last["train_supervised_accuracy"]),
            "train_loss_at_peak": float(best["train_loss"]),
            "train_loss_last": float(last["train_loss"]),
        }
        for metric_name in METRIC_NAMES:
            row[f"selected_{metric_name}"] = float(selected[metric_name])
        run_rows.append(row)

    run_csv = root / f"{args.output_prefix}_runs.csv"
    with run_csv.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(run_rows[0]))
        writer.writeheader()
        writer.writerows(run_rows)

    grouped = defaultdict(list)
    for row in run_rows:
        grouped[row["config"]].append(row)

    summary_rows = []
    for config, config_rows in sorted(grouped.items()):
        summary = {"config": config, "num_runs": len(config_rows)}
        for key in (
            "selection_epoch",
            "selection_score",
            "validation_peak_macro_f1",
            "validation_last_macro_f1",
            "validation_macro_f1_drop",
            "validation_tail10_macro_f1",
            "validation_max_one_epoch_drop",
            "validation_max_post_peak_one_epoch_drop",
            "validation_peak_accuracy",
            "validation_last_accuracy",
            "train_accuracy_at_peak",
            "train_accuracy_last",
            "selected_macro_f1",
            "selected_overall_accuracy",
            "selected_mean_class_accuracy",
        ):
            values = [float(row[key]) for row in config_rows]
            summary[f"mean_{key}"] = mean(values)
            summary[f"std_{key}"] = sample_std(values)
        summary_rows.append(summary)

    summary_csv = root / f"{args.output_prefix}_summary.csv"
    with summary_csv.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)

    summary_json = root / f"{args.output_prefix}_summary.json"
    with summary_json.open("w", encoding="utf-8") as handle:
        json.dump(summary_rows, handle, ensure_ascii=False, indent=2)

    write_gradient_summaries(root, args.output_prefix)

    print(f"Wrote {len(run_rows)} run rows to {run_csv}")
    print(f"Wrote {len(summary_rows)} configuration rows to {summary_csv}")
    for summary in summary_rows:
        print(
            "{config}: selected_macro_f1={mean_selected_macro_f1:.4f}+/-{std_selected_macro_f1:.4f}, "
            "peak-to-last-drop={mean_validation_macro_f1_drop:.4f}+/-{std_validation_macro_f1_drop:.4f}".format(
                **summary
            )
        )


if __name__ == "__main__":
    main()
