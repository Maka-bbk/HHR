#!/usr/bin/env python3
"""Paired descriptive comparison of mean and mean+robust-max online CV runs."""

import argparse
import csv
import statistics
from pathlib import Path


METRICS = [
    "overall_accuracy",
    "mean_class_accuracy",
    "macro_f1",
    "gcd_all_accuracy",
    "gcd_old_accuracy",
    "gcd_new_accuracy",
    "gcd_soft_all_accuracy",
    "gcd_seen_accuracy",
    "gcd_unseen_accuracy",
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mean-root", type=Path, required=True)
    parser.add_argument("--mean-robust-max-root", type=Path, required=True)
    parser.add_argument(
        "--output-prefix", default="mean_robust_max_vs_mean_online_paired"
    )
    return parser.parse_args()


def read_csv(path):
    if not path.is_file():
        raise FileNotFoundError(f"Summary run table not found: {path}")
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def index_rows(rows, expected_pooling, source):
    indexed = {}
    for row in rows:
        if row.get("pooling") != expected_pooling:
            raise RuntimeError(
                f"Expected pooling={expected_pooling!r}, found {row.get('pooling')!r}: "
                f"{source}"
            )
        key = (int(row["fold"]), int(row["seed"]), int(row["session"]))
        if key in indexed:
            raise RuntimeError(f"Duplicate fold/seed/session {key}: {source}")
        indexed[key] = row
    return indexed


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


def mean_std(values):
    return (
        statistics.fmean(values),
        statistics.stdev(values) if len(values) > 1 else 0.0,
    )


def summarize_deltas(rows, scope):
    summary = {"scope": scope, "num_pairs": len(rows)}
    for metric in METRICS:
        key = f"delta_{metric}"
        values = [float(row[key]) for row in rows]
        mean, std = mean_std(values)
        summary[f"mean_{key}"] = mean
        summary[f"std_{key}"] = std
        summary[f"wins_{metric}"] = sum(value > 0 for value in values)
        summary[f"ties_{metric}"] = sum(value == 0 for value in values)
        summary[f"losses_{metric}"] = sum(value < 0 for value in values)
    return summary


def main():
    args = parse_args()
    mean_root = args.mean_root.resolve()
    robust_root = args.mean_robust_max_root.resolve()
    mean_path = mean_root / "mean_subject_cv_online_runs.csv"
    robust_path = robust_root / "mean_robust_max_subject_cv_online_runs.csv"
    mean_rows = index_rows(read_csv(mean_path), "mean", mean_path)
    robust_rows = index_rows(
        read_csv(robust_path), "mean_robust_max", robust_path
    )
    if set(mean_rows) != set(robust_rows):
        raise RuntimeError(
            "Mean and mean_robust_max do not contain the same fold/seed/session grid; "
            f"mean_only={sorted(set(mean_rows) - set(robust_rows))}, "
            f"robust_only={sorted(set(robust_rows) - set(mean_rows))}."
        )

    matched_fields = [
        "train_subjects",
        "validation_subjects",
        "outer_test_subjects",
        "npz_path",
        "window_size",
        "sample_unit",
        "view_mode",
        "normalization_eps",
        "online_old_seen_trials",
        "online_novel_unseen_trials",
        "online_novel_seen_trials",
        "checkpoint_selection",
        "selection_epoch",
        "num_classes",
        "num_samples",
        "labels",
        "support",
    ]
    paired = []
    for key in sorted(mean_rows):
        baseline = mean_rows[key]
        candidate = robust_rows[key]
        mismatches = [
            field
            for field in matched_fields
            if baseline.get(field) != candidate.get(field)
        ]
        if mismatches:
            raise RuntimeError(
                f"Unmatched data/protocol fields for fold/seed/session {key}: {mismatches}"
            )
        row = {
            "fold": key[0],
            "seed": key[1],
            "session": key[2],
            "outer_test_subjects": baseline["outer_test_subjects"],
            "num_samples": int(baseline["num_samples"]),
        }
        for metric in METRICS:
            mean_value = float(baseline[metric])
            robust_value = float(candidate[metric])
            row[f"mean_{metric}"] = mean_value
            row[f"mean_robust_max_{metric}"] = robust_value
            row[f"delta_{metric}"] = robust_value - mean_value
        paired.append(row)

    continual = []
    fold_seed_pairs = sorted(set((row["fold"], row["seed"]) for row in paired))
    for fold, seed in fold_seed_pairs:
        run_rows = [
            row for row in paired if row["fold"] == fold and row["seed"] == seed
        ]
        if sorted(row["session"] for row in run_rows) != [1, 2, 3]:
            raise RuntimeError(f"Fold {fold}, seed {seed} lacks all three sessions.")
        record = {"fold": fold, "seed": seed, "scope": "mean_1_3"}
        for metric in METRICS:
            for prefix in ["mean_", "mean_robust_max_", "delta_"]:
                field = f"{prefix}{metric}"
                record[field] = statistics.fmean(float(row[field]) for row in run_rows)
        continual.append(record)

    summaries = []
    for session in (1, 2, 3):
        summaries.append(
            summarize_deltas(
                [row for row in paired if row["session"] == session],
                f"session_{session}",
            )
        )
    summaries.append(summarize_deltas(continual, "mean_sessions_1_3"))

    output_root = robust_root
    pair_path = output_root / f"{args.output_prefix}_runs.csv"
    continual_path = output_root / f"{args.output_prefix}_continual_by_run.csv"
    summary_path = output_root / f"{args.output_prefix}_summary.csv"
    write_csv(pair_path, paired)
    write_csv(continual_path, continual)
    write_csv(summary_path, summaries)

    print("scope,pairs,delta_all,delta_new,delta_unseen,delta_macro_f1")
    for row in summaries:
        print(
            f"{row['scope']},{row['num_pairs']},"
            f"{row['mean_delta_gcd_all_accuracy']:.6f},"
            f"{row['mean_delta_gcd_new_accuracy']:.6f},"
            f"{row['mean_delta_gcd_unseen_accuracy']:.6f},"
            f"{row['mean_delta_macro_f1']:.6f}"
        )
    print(f"Paired runs: {pair_path}")
    print(f"Continual paired runs: {continual_path}")
    print(f"Paired summary: {summary_path}")
    print("Deltas are mean_robust_max minus mean; this script reports descriptive effects only.")


if __name__ == "__main__":
    main()
