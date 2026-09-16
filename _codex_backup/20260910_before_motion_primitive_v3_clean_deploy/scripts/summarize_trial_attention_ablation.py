#!/usr/bin/env python3
"""Summarize validation or final-test metrics from the trial-attention runs."""

import argparse
import csv
import json
import math
import re
import statistics
from pathlib import Path


METRIC_FILES = {
    "offline_best_validation_classification_metrics.json": "validation",
    "offline_best_classification_metrics.json": "final_test",
}

TRIAL_CONFIGS = {
    "mean_full_full",
    "gated_attention_full_full",
    "mean_full_random_crop",
    "gated_attention_full_random_crop",
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output-prefix", default="trial_attention")
    return parser.parse_args()


def read_saved_arg(text, name, default=None):
    pattern = re.compile(
        rf"'{re.escape(name)}':\s*(" 
        r"'(?:\\.|[^'])*'|"
        r"True|False|None|"
        r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
        r")"
    )
    match = pattern.search(text)
    if match is None:
        return default
    token = match.group(1)
    if token.startswith("'"):
        return token[1:-1]
    if token == "True":
        return True
    if token == "False":
        return False
    if token == "None":
        return None
    try:
        return int(token)
    except ValueError:
        return float(token)


def finite_number(value):
    if value is None:
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def find_seed(relative_parts):
    for part in relative_parts:
        match = re.fullmatch(r"seed_(-?\d+)_offline", part)
        if match:
            return int(match.group(1))
    raise RuntimeError(f"Could not infer seed from path parts: {relative_parts}")


def collect_runs(root):
    rows = []
    seen = set()
    for metric_path in sorted(root.rglob("*_classification_metrics.json")):
        if metric_path.name not in METRIC_FILES:
            continue
        relative = metric_path.relative_to(root)
        if len(relative.parts) < 2:
            raise RuntimeError(f"Unexpected result path: {metric_path}")
        config = relative.parts[0]
        # The shared window encoder is an initialization stage, not a fifth
        # trial-attention ablation configuration.
        if config == "window_pretrain":
            continue
        if config not in TRIAL_CONFIGS:
            raise RuntimeError(
                f"Unexpected trial-attention configuration {config!r}: {metric_path}"
            )
        seed = find_seed(relative.parts)
        evaluation_split = METRIC_FILES[metric_path.name]
        identity = (config, seed, evaluation_split)
        if identity in seen:
            raise RuntimeError(
                f"Duplicate metrics for config/seed/split {identity}: {metric_path}"
            )
        seen.add(identity)

        with metric_path.open("r", encoding="utf-8") as handle:
            metrics = json.load(handle)
        args_path = metric_path.parent / "args.txt"
        args_text = args_path.read_text(encoding="utf-8") if args_path.exists() else ""
        diagnostics = metrics.get("prediction_diagnostics") or {}
        row = {
            "config": config,
            "seed": seed,
            "evaluation_split": evaluation_split,
            "sample_unit": metrics.get(
                "sample_unit", read_saved_arg(args_text, "uschad_sample_unit")
            ),
            "pooling": read_saved_arg(args_text, "trial_pooling"),
            "view_mode": read_saved_arg(args_text, "trial_view_mode"),
            "crop_ratio": read_saved_arg(args_text, "trial_crop_ratio"),
            "selection_epoch": metrics.get("selection_epoch"),
            "selection_score": finite_number(metrics.get("selection_score")),
            "overall_accuracy": finite_number(metrics.get("overall_accuracy")),
            "mean_class_accuracy": finite_number(metrics.get("mean_class_accuracy")),
            "macro_f1": finite_number(metrics.get("macro_f1")),
            "num_samples": metrics.get("num_samples"),
            "mean_confidence": finite_number(diagnostics.get("mean_confidence")),
            "mean_predictive_entropy": finite_number(
                diagnostics.get("mean_predictive_entropy")
            ),
            "mean_attention_entropy": finite_number(
                diagnostics.get("mean_attention_entropy")
            ),
            "mean_effective_windows": finite_number(
                diagnostics.get("mean_effective_windows")
            ),
            "metrics_path": str(metric_path),
        }
        rows.append(row)
    if not rows:
        expected = ", ".join(METRIC_FILES)
        raise FileNotFoundError(
            f"No supported metric files found under {root}. Expected: {expected}"
        )
    return rows


def mean_std(rows, key):
    values = [float(row[key]) for row in rows if row.get(key) is not None]
    if not values:
        return None, None
    mean = statistics.fmean(values)
    std = statistics.stdev(values) if len(values) > 1 else 0.0
    return mean, std


def build_summary(rows):
    groups = {}
    for row in rows:
        groups.setdefault((row["config"], row["evaluation_split"]), []).append(row)

    summary = []
    metric_keys = {
        "selection_epoch": "selection_epoch",
        "selection_score": "selection_score",
        "overall_accuracy": "overall_accuracy",
        "mean_class_accuracy": "class_accuracy",
        "macro_f1": "macro_f1",
        "mean_confidence": "confidence",
        "mean_predictive_entropy": "predictive_entropy",
        "mean_attention_entropy": "attention_entropy",
        "mean_effective_windows": "effective_windows",
    }
    for (config, evaluation_split), group_rows in sorted(groups.items()):
        seeds = sorted(int(row["seed"]) for row in group_rows)
        if len(seeds) != len(set(seeds)):
            raise RuntimeError(f"Duplicate seeds in {config}/{evaluation_split}: {seeds}")
        record = {
            "config": config,
            "evaluation_split": evaluation_split,
            "num_runs": len(group_rows),
            "seeds": " ".join(str(seed) for seed in seeds),
            "sample_unit": group_rows[0].get("sample_unit"),
            "pooling": group_rows[0].get("pooling"),
            "view_mode": group_rows[0].get("view_mode"),
        }
        for key, output_name in metric_keys.items():
            mean, std = mean_std(group_rows, key)
            record[f"mean_{output_name}"] = mean
            record[f"std_{output_name}"] = std
        summary.append(record)
    return summary


def write_csv(path, rows):
    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    root = args.root.resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Result root not found: {root}")
    runs = collect_runs(root)
    summary = build_summary(runs)

    runs_path = root / f"{args.output_prefix}_runs.csv"
    summary_path = root / f"{args.output_prefix}_summary.csv"
    json_path = root / f"{args.output_prefix}_summary.json"
    write_csv(runs_path, runs)
    write_csv(summary_path, summary)
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)

    print("config,evaluation,num_runs,accuracy,macro_f1,attention_entropy,effective_windows")
    for row in summary:
        def show(key):
            value = row.get(key)
            return "NA" if value is None else f"{float(value):.6f}"

        print(
            f"{row['config']},{row['evaluation_split']},{row['num_runs']},"
            f"{show('mean_overall_accuracy')},{show('mean_macro_f1')},"
            f"{show('mean_attention_entropy')},"
            f"{show('mean_effective_windows')}"
        )
    print(f"Run table: {runs_path}")
    print(f"Summary CSV: {summary_path}")
    print(f"Summary JSON: {json_path}")


if __name__ == "__main__":
    main()
