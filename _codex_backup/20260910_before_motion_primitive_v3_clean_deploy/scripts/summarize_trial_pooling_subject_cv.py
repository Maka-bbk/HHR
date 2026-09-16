#!/usr/bin/env python3
"""Summarize subject-level trial-pooling cross-validation runs."""

import argparse
import csv
import json
import math
import re
import statistics
from pathlib import Path

import numpy as np


METRIC_FILES = {
    "offline_best_validation_classification_metrics.json": "validation",
    "offline_best_classification_metrics.json": "outer_test",
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output-prefix", default="mean_robust_max_subject_cv")
    parser.add_argument(
        "--expected-pooling",
        choices=["mean", "mean_robust_max", "gated_attention"],
        default="mean_robust_max",
    )
    return parser.parse_args()


def read_saved_arg(text, name, default=None):
    pattern = re.compile(
        rf"'{re.escape(name)}':\s*("
        r"'(?:\\.|[^'])*'|"
        r"True|False|None|"
        r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)"
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


def parse_subjects(value):
    if value is None:
        return []
    tokens = str(value).replace(",", " ").split()
    return sorted(set(int(token) for token in tokens))


def finite_number(value, name):
    if value is None:
        return None
    number = float(value)
    if not math.isfinite(number):
        raise RuntimeError(f"Non-finite {name}: {value!r}")
    return number


def find_seed(parts):
    for part in parts:
        match = re.fullmatch(r"seed_(-?\d+)_offline", part)
        if match:
            return int(match.group(1))
    raise RuntimeError(f"Could not infer seed from path: {parts}")


def validate_fold_split(fold, train_subjects, val_subjects, test_subjects):
    split_sets = {
        "train": set(train_subjects),
        "validation": set(val_subjects),
        "outer_test": set(test_subjects),
    }
    for left, right in [
        ("train", "validation"),
        ("train", "outer_test"),
        ("validation", "outer_test"),
    ]:
        overlap = split_sets[left] & split_sets[right]
        if overlap:
            raise RuntimeError(
                f"Fold {fold} has {left}/{right} subject overlap: {sorted(overlap)}"
            )
    if set.union(*split_sets.values()) != set(range(1, 15)):
        raise RuntimeError(
            f"Fold {fold} does not partition USC-HAD subjects 1..14 exactly."
        )
    observed_sizes = tuple(len(split_sets[name]) for name in split_sets)
    if observed_sizes != (10, 2, 2):
        raise RuntimeError(
            f"Fold {fold} expected train/validation/test sizes 10/2/2, "
            f"got {observed_sizes}."
        )


def validate_metrics(metrics, source):
    required = {
        "num_samples",
        "labels",
        "overall_accuracy",
        "mean_class_accuracy",
        "macro_f1",
        "per_class_accuracy",
        "per_class_f1",
        "support",
        "confusion_matrix",
    }
    missing = required - set(metrics)
    if missing:
        raise RuntimeError(f"Missing metrics {sorted(missing)} in {source}")
    labels = [int(value) for value in metrics["labels"]]
    support = np.asarray(metrics["support"], dtype=np.int64)
    confusion = np.asarray(metrics["confusion_matrix"], dtype=np.int64)
    if confusion.shape != (len(labels), len(labels)):
        raise RuntimeError(
            f"Invalid confusion matrix shape {confusion.shape} in {source}."
        )
    if len(support) != len(labels) or not np.array_equal(
        support, confusion.sum(axis=1)
    ):
        raise RuntimeError(f"Support/confusion mismatch in {source}.")
    if int(confusion.sum()) != int(metrics["num_samples"]):
        raise RuntimeError(f"Confusion total/num_samples mismatch in {source}.")


def build_row(
    metric_path,
    metrics,
    split_name,
    expected_pooling,
    parent_metrics=None,
):
    args_path = metric_path.parent / "args.txt"
    if not args_path.is_file():
        raise FileNotFoundError(f"Missing args.txt beside {metric_path}")
    args_text = args_path.read_text(encoding="utf-8")
    relative_parts = metric_path.parts
    seed = int(read_saved_arg(args_text, "seed", find_seed(relative_parts)))
    fold = int(read_saved_arg(args_text, "uschad_cv_fold", -1))
    if fold < 1:
        raise RuntimeError(f"Missing valid uschad_cv_fold in {args_path}")

    train_subjects = parse_subjects(
        read_saved_arg(args_text, "uschad_train_subjects")
    )
    val_subjects = parse_subjects(
        read_saved_arg(args_text, "offline_val_subjects")
    )
    test_subjects = parse_subjects(
        read_saved_arg(args_text, "uschad_test_subjects")
    )
    validate_fold_split(fold, train_subjects, val_subjects, test_subjects)

    pooling = read_saved_arg(args_text, "trial_pooling")
    if pooling != expected_pooling:
        raise RuntimeError(
            f"Expected trial_pooling={expected_pooling!r}, found {pooling!r}: "
            f"{metric_path}"
        )
    if read_saved_arg(
        args_text, "uschad_recompute_norm_from_train_subjects", False
    ) is not True:
        raise RuntimeError(f"Fold normalization was not enabled: {metric_path}")

    validate_metrics(metrics, metric_path)
    diagnostics = metrics.get("prediction_diagnostics") or {}
    selection_source = parent_metrics if parent_metrics is not None else metrics
    row = {
        "fold": fold,
        "seed": seed,
        "evaluation_split": split_name,
        "train_subjects": ",".join(map(str, train_subjects)),
        "validation_subjects": ",".join(map(str, val_subjects)),
        "outer_test_subjects": ",".join(map(str, test_subjects)),
        "sample_unit": metrics.get(
            "sample_unit", read_saved_arg(args_text, "uschad_sample_unit")
        ),
        "pooling": pooling,
        "view_mode": read_saved_arg(args_text, "trial_view_mode"),
        "robust_quantile": None,
        "filtered_upper_fraction": None,
        "fusion_dim": None,
        "fusion_dropout": None,
        "attention_dim": None,
        "attention_dropout": None,
        "attention_temperature": None,
        "attention_mean_mix": None,
        "normalization_eps": finite_number(
            read_saved_arg(args_text, "uschad_norm_eps"), "normalization_eps"
        ),
        "selection_epoch": selection_source.get("selection_epoch"),
        "selection_metric": selection_source.get("selection_metric"),
        "selection_score": finite_number(
            selection_source.get("selection_score"), "selection_score"
        ),
        "num_samples": int(metrics["num_samples"]),
        "overall_accuracy": finite_number(
            metrics["overall_accuracy"], "overall_accuracy"
        ),
        "mean_class_accuracy": finite_number(
            metrics["mean_class_accuracy"], "mean_class_accuracy"
        ),
        "macro_f1": finite_number(metrics["macro_f1"], "macro_f1"),
        "mean_confidence": finite_number(
            diagnostics.get("mean_confidence"), "mean_confidence"
        ),
        "mean_predictive_entropy": finite_number(
            diagnostics.get("mean_predictive_entropy"),
            "mean_predictive_entropy",
        ),
        "mean_attention_entropy": finite_number(
            diagnostics.get("mean_attention_entropy"),
            "mean_attention_entropy",
        ),
        "mean_effective_windows": finite_number(
            diagnostics.get("mean_effective_windows"),
            "mean_effective_windows",
        ),
        "confusion_matrix": json.dumps(
            metrics["confusion_matrix"], separators=(",", ":")
        ),
        "metrics_path": str(metric_path.resolve()),
    }
    if pooling == "mean_robust_max":
        row["robust_quantile"] = finite_number(
            read_saved_arg(args_text, "trial_robust_max_quantile"),
            "robust_quantile",
        )
        row["filtered_upper_fraction"] = round(
            1.0 - row["robust_quantile"], 12
        )
        row["fusion_dim"] = read_saved_arg(args_text, "trial_pool_fusion_dim")
        row["fusion_dropout"] = finite_number(
            read_saved_arg(args_text, "trial_pool_fusion_dropout"),
            "fusion_dropout",
        )
    elif pooling == "gated_attention":
        row["attention_dim"] = read_saved_arg(args_text, "trial_attention_dim")
        row["attention_dropout"] = finite_number(
            read_saved_arg(args_text, "trial_attention_dropout"),
            "attention_dropout",
        )
        row["attention_temperature"] = finite_number(
            read_saved_arg(args_text, "trial_attention_temperature"),
            "attention_temperature",
        )
        row["attention_mean_mix"] = finite_number(
            read_saved_arg(args_text, "trial_attention_mean_mix"),
            "attention_mean_mix",
        )

    labels = [int(value) for value in metrics["labels"]]
    for index, label in enumerate(labels):
        row[f"class_{label}_accuracy"] = finite_number(
            metrics["per_class_accuracy"][index], f"class_{label}_accuracy"
        )
        row[f"class_{label}_f1"] = finite_number(
            metrics["per_class_f1"][index], f"class_{label}_f1"
        )
        row[f"class_{label}_support"] = int(metrics["support"][index])
    return row


def collect_rows(root, expected_pooling):
    rows = []
    seen = set()
    for metric_path in sorted(root.rglob("*_classification_metrics.json")):
        if "window_pretrain" in metric_path.parts:
            continue
        if metric_path.name not in METRIC_FILES:
            continue
        with metric_path.open("r", encoding="utf-8") as handle:
            metrics = json.load(handle)

        metric_sets = []
        file_split = METRIC_FILES[metric_path.name]
        if file_split == "validation":
            metric_sets.append(("validation", metrics, None))
        else:
            validation_metrics = metrics.get("validation_metrics")
            if not isinstance(validation_metrics, dict):
                raise RuntimeError(
                    f"Outer-test metrics lack nested validation_metrics: {metric_path}"
                )
            metric_sets.append(("validation", validation_metrics, metrics))
            metric_sets.append(("outer_test", metrics, None))

        for split_name, split_metrics, parent_metrics in metric_sets:
            row = build_row(
                metric_path,
                split_metrics,
                split_name,
                expected_pooling,
                parent_metrics=parent_metrics,
            )
            identity = (row["fold"], row["seed"], split_name)
            if identity in seen:
                raise RuntimeError(f"Duplicate fold/seed/split result: {identity}")
            seen.add(identity)
            rows.append(row)
    if not rows:
        raise FileNotFoundError(f"No subject-CV trial metrics found under {root}")
    return rows


def mean_std(rows, key):
    values = [float(row[key]) for row in rows if row.get(key) is not None]
    if not values:
        return None, None
    return (
        statistics.fmean(values),
        statistics.stdev(values) if len(values) > 1 else 0.0,
    )


def metrics_from_confusion(confusion):
    confusion = np.asarray(confusion, dtype=np.float64)
    support = confusion.sum(axis=1)
    predicted = confusion.sum(axis=0)
    true_positive = np.diag(confusion)
    total = confusion.sum()
    accuracy = float(true_positive.sum() / total) if total > 0 else None
    denominator = support + predicted
    f1 = np.divide(
        2.0 * true_positive,
        denominator,
        out=np.zeros_like(true_positive),
        where=denominator > 0,
    )
    valid_classes = support > 0
    macro_f1 = float(f1[valid_classes].mean()) if np.any(valid_classes) else None
    return accuracy, macro_f1


def aggregate_group(rows, extra=None):
    record = dict(extra or {})
    record.update({
        "num_runs": len(rows),
        "num_folds": len(set(int(row["fold"]) for row in rows)),
        "folds": " ".join(
            str(value) for value in sorted(set(int(row["fold"]) for row in rows))
        ),
        "num_seeds": len(set(int(row["seed"]) for row in rows)),
        "seeds": " ".join(
            str(value) for value in sorted(set(int(row["seed"]) for row in rows))
        ),
    })
    for key in [
        "sample_unit",
        "pooling",
        "view_mode",
        "robust_quantile",
        "filtered_upper_fraction",
        "fusion_dim",
        "fusion_dropout",
        "attention_dim",
        "attention_dropout",
        "attention_temperature",
        "attention_mean_mix",
        "normalization_eps",
    ]:
        values = {row.get(key) for row in rows}
        if len(values) != 1:
            raise RuntimeError(
                f"Mixed {key} values within one CV aggregate: {sorted(values)}"
            )
        record[key] = values.pop()
    for key in [
        "selection_epoch",
        "selection_score",
        "overall_accuracy",
        "mean_class_accuracy",
        "macro_f1",
        "mean_confidence",
        "mean_predictive_entropy",
        "mean_attention_entropy",
        "mean_effective_windows",
    ]:
        mean, std = mean_std(rows, key)
        record[f"mean_{key}"] = mean
        record[f"std_{key}"] = std
    for label in range(6):
        for metric_name in ["accuracy", "f1"]:
            key = f"class_{label}_{metric_name}"
            mean, std = mean_std(rows, key)
            record[f"mean_{key}"] = mean
            record[f"std_{key}"] = std

    confusion = sum(
        (
            np.asarray(json.loads(row["confusion_matrix"]), dtype=np.int64)
            for row in rows
        ),
        start=np.zeros((6, 6), dtype=np.int64),
    )
    pooled_accuracy, pooled_macro_f1 = metrics_from_confusion(confusion)
    record["pooled_accuracy"] = pooled_accuracy
    record["pooled_macro_f1"] = pooled_macro_f1
    record["pooled_num_samples"] = int(confusion.sum())
    record["pooled_confusion_matrix"] = json.dumps(
        confusion.tolist(), separators=(",", ":")
    )
    return record


def build_aggregates(rows):
    by_seed = []
    summary = []
    for split_name in sorted(set(row["evaluation_split"] for row in rows)):
        split_rows = [row for row in rows if row["evaluation_split"] == split_name]
        for seed in sorted(set(int(row["seed"]) for row in split_rows)):
            seed_rows = [row for row in split_rows if int(row["seed"]) == seed]
            by_seed.append(
                aggregate_group(
                    seed_rows,
                    {"evaluation_split": split_name, "seed": seed},
                )
            )
        summary.append(
            aggregate_group(split_rows, {"evaluation_split": split_name})
        )
    return by_seed, summary


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

    rows = collect_rows(root, args.expected_pooling)
    by_seed, summary = build_aggregates(rows)
    runs_path = root / f"{args.output_prefix}_runs.csv"
    seed_path = root / f"{args.output_prefix}_by_seed.csv"
    summary_path = root / f"{args.output_prefix}_summary.csv"
    json_path = root / f"{args.output_prefix}_summary.json"
    write_csv(runs_path, rows)
    write_csv(seed_path, by_seed)
    write_csv(summary_path, summary)
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(
            {"runs": rows, "by_seed": by_seed, "summary": summary},
            handle,
            ensure_ascii=False,
            indent=2,
        )

    print("split,runs,folds,seeds,mean_accuracy,mean_macro_f1,pooled_accuracy")
    for row in summary:
        print(
            f"{row['evaluation_split']},{row['num_runs']},"
            f"{row['num_folds']},{row['num_seeds']},"
            f"{row['mean_overall_accuracy']:.6f},"
            f"{row['mean_macro_f1']:.6f},{row['pooled_accuracy']:.6f}"
        )
    print(f"Run table: {runs_path}")
    print(f"By-seed table: {seed_path}")
    print(f"Summary CSV: {summary_path}")
    print(f"Summary JSON: {json_path}")


if __name__ == "__main__":
    main()
