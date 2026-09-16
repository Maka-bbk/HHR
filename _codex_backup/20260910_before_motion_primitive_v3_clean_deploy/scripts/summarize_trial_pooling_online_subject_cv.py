#!/usr/bin/env python3
"""Summarize final-epoch online metrics from trial-level subject CV runs."""

import argparse
import csv
import json
import math
import re
import statistics
from pathlib import Path

import numpy as np


METRIC_KEYS = [
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
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output-prefix", default="mean_robust_max_subject_cv_online")
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
    return sorted(set(int(token) for token in str(value).replace(",", " ").split()))


def finite_number(value, name):
    if value is None:
        return None
    number = float(value)
    if not math.isfinite(number):
        raise RuntimeError(f"Non-finite {name}: {value!r}")
    return number


def find_seed(parts):
    for part in parts:
        match = re.fullmatch(r"seed_(-?\d+)_online", part)
        if match:
            return int(match.group(1))
    raise RuntimeError(f"Could not infer online seed from path: {parts}")


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
        raise RuntimeError(f"Fold {fold} does not partition subjects 1..14 exactly.")
    sizes = tuple(len(split_sets[name]) for name in split_sets)
    if sizes != (10, 2, 2):
        raise RuntimeError(
            f"Fold {fold} expected train/validation/test sizes 10/2/2, got {sizes}."
        )


def validate_metrics(metrics, session, source):
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
        "gcd_all_accuracy",
        "gcd_old_accuracy",
        "gcd_new_accuracy",
        "gcd_soft_all_accuracy",
        "gcd_seen_accuracy",
        "gcd_unseen_accuracy",
        "checkpoint_selection",
        "selection_epoch",
    }
    missing = required - set(metrics)
    if missing:
        raise RuntimeError(f"Missing metrics {sorted(missing)} in {source}")
    labels = [int(value) for value in metrics["labels"]]
    expected_labels = list(range(6 + 2 * session))
    if labels != expected_labels:
        raise RuntimeError(
            f"Session {session} expected labels {expected_labels}, got {labels}: {source}"
        )
    support = np.asarray(metrics["support"], dtype=np.int64)
    confusion = np.asarray(metrics["confusion_matrix"], dtype=np.int64)
    expected_shape = (len(labels), len(labels))
    if confusion.shape != expected_shape:
        raise RuntimeError(
            f"Invalid confusion shape {confusion.shape}, expected {expected_shape}: {source}"
        )
    if len(support) != len(labels) or not np.array_equal(
        support, confusion.sum(axis=1)
    ):
        raise RuntimeError(f"Support/confusion mismatch: {source}")
    if int(confusion.sum()) != int(metrics["num_samples"]):
        raise RuntimeError(f"Confusion total/num_samples mismatch: {source}")
    if metrics["checkpoint_selection"] != "final_epoch":
        raise RuntimeError(f"Online metrics were not selected by final_epoch: {source}")
    for key in METRIC_KEYS:
        finite_number(metrics[key], f"{key} in {source}")


def build_row(metric_path, metrics, session, expected_pooling):
    args_path = metric_path.parent / "args.txt"
    if not args_path.is_file():
        raise FileNotFoundError(f"Missing args.txt beside {metric_path}")
    args_text = args_path.read_text(encoding="utf-8")
    seed = int(read_saved_arg(args_text, "seed", find_seed(metric_path.parts)))
    fold = int(read_saved_arg(args_text, "uschad_cv_fold", -1))
    if fold < 1:
        raise RuntimeError(f"Missing valid uschad_cv_fold in {args_path}")

    train_subjects = parse_subjects(read_saved_arg(args_text, "uschad_train_subjects"))
    val_subjects = parse_subjects(read_saved_arg(args_text, "offline_val_subjects"))
    test_subjects = parse_subjects(read_saved_arg(args_text, "uschad_test_subjects"))
    validate_fold_split(fold, train_subjects, val_subjects, test_subjects)

    required_args = {
        "train_session": "online",
        "uschad_sample_unit": "trial",
        "trial_pooling": expected_pooling,
        "online_checkpoint_selection": "final_epoch",
        "uschad_split_mode": "subject",
        "uschad_recompute_norm_from_train_subjects": True,
        "num_novel_classes_per_session": 2,
        "continual_session_num": 3,
        "online_old_seen_trials": 2,
        "online_novel_unseen_trials": 5,
        "online_novel_seen_trials": 2,
        "shuffle_classes": False,
    }
    for name, expected in required_args.items():
        observed = read_saved_arg(args_text, name)
        if observed != expected:
            raise RuntimeError(
                f"Expected {name}={expected!r}, found {observed!r}: {metric_path}"
            )
    load_offline_id = read_saved_arg(args_text, "load_offline_id")
    if not load_offline_id:
        raise RuntimeError(f"Online run does not record load_offline_id: {metric_path}")

    validate_metrics(metrics, session, metric_path)
    online_epochs = int(read_saved_arg(args_text, "epochs_online_per_session", -1))
    if int(metrics["selection_epoch"]) != online_epochs:
        raise RuntimeError(
            f"Final selection epoch {metrics['selection_epoch']} does not match "
            f"epochs_online_per_session={online_epochs}: {metric_path}"
        )

    diagnostics = metrics.get("prediction_diagnostics") or {}
    row = {
        "fold": fold,
        "seed": seed,
        "session": session,
        "train_subjects": ",".join(map(str, train_subjects)),
        "validation_subjects": ",".join(map(str, val_subjects)),
        "outer_test_subjects": ",".join(map(str, test_subjects)),
        "npz_path": read_saved_arg(args_text, "uschad_npz_path"),
        "window_size": int(read_saved_arg(args_text, "uschad_window_size")),
        "sample_unit": "trial",
        "pooling": expected_pooling,
        "view_mode": read_saved_arg(args_text, "trial_view_mode"),
        "robust_quantile": None,
        "filtered_upper_fraction": None,
        "fusion_dim": None,
        "fusion_dropout": None,
        "normalization_eps": finite_number(
            read_saved_arg(args_text, "uschad_norm_eps"), "normalization_eps"
        ),
        "online_old_seen_trials": 2,
        "online_novel_unseen_trials": 5,
        "online_novel_seen_trials": 2,
        "checkpoint_selection": "final_epoch",
        "selection_epoch": int(metrics["selection_epoch"]),
        "load_offline_id": load_offline_id,
        "num_classes": len(metrics["labels"]),
        "num_samples": int(metrics["num_samples"]),
        "mean_confidence": finite_number(
            diagnostics.get("mean_confidence"), "mean_confidence"
        ),
        "mean_predictive_entropy": finite_number(
            diagnostics.get("mean_predictive_entropy"), "mean_predictive_entropy"
        ),
        "labels": json.dumps(metrics["labels"], separators=(",", ":")),
        "support": json.dumps(metrics["support"], separators=(",", ":")),
        "confusion_matrix": json.dumps(
            metrics["confusion_matrix"], separators=(",", ":")
        ),
        "metrics_path": str(metric_path.resolve()),
    }
    for key in METRIC_KEYS:
        row[key] = finite_number(metrics[key], key)
    if expected_pooling == "mean_robust_max":
        row["robust_quantile"] = finite_number(
            read_saved_arg(args_text, "trial_robust_max_quantile"),
            "robust_quantile",
        )
        row["filtered_upper_fraction"] = round(1.0 - row["robust_quantile"], 12)
        row["fusion_dim"] = int(read_saved_arg(args_text, "trial_pool_fusion_dim"))
        row["fusion_dropout"] = finite_number(
            read_saved_arg(args_text, "trial_pool_fusion_dropout"),
            "fusion_dropout",
        )
    for index, label in enumerate(metrics["labels"]):
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
    pattern = re.compile(
        r"online_session_(\d+)_(final|best)_classification_metrics\.json"
    )
    for metric_path in sorted(root.rglob("online_session_*_classification_metrics.json")):
        if "window_pretrain" in metric_path.parts:
            continue
        match = pattern.fullmatch(metric_path.name)
        if match is None:
            raise RuntimeError(f"Unexpected online metric filename: {metric_path}")
        session = int(match.group(1))
        if match.group(2) != "final":
            raise RuntimeError(
                f"Found test-selected online metrics; expected final_epoch only: {metric_path}"
            )
        with metric_path.open("r", encoding="utf-8") as handle:
            metrics = json.load(handle)
        row = build_row(metric_path, metrics, session, expected_pooling)
        identity = (row["fold"], row["seed"], row["session"])
        if identity in seen:
            raise RuntimeError(f"Duplicate fold/seed/session online result: {identity}")
        seen.add(identity)
        rows.append(row)
    if not rows:
        raise FileNotFoundError(f"No online subject-CV metrics found under {root}")

    folds = sorted(set(int(row["fold"]) for row in rows))
    seeds = sorted(set(int(row["seed"]) for row in rows))
    expected = {
        (fold, seed, session)
        for fold in folds
        for seed in seeds
        for session in (1, 2, 3)
    }
    observed = {(row["fold"], row["seed"], row["session"]) for row in rows}
    if observed != expected:
        missing = sorted(expected - observed)
        unexpected = sorted(observed - expected)
        raise RuntimeError(
            f"Incomplete online fold/seed/session grid; missing={missing}, "
            f"unexpected={unexpected}."
        )
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


def aggregate_group(rows, extra=None, include_confusion=True):
    record = dict(extra or {})
    record.update(
        {
            "num_runs": len(rows),
            "num_folds": len(set(int(row["fold"]) for row in rows)),
            "folds": " ".join(
                str(value) for value in sorted(set(int(row["fold"]) for row in rows))
            ),
            "num_seeds": len(set(int(row["seed"]) for row in rows)),
            "seeds": " ".join(
                str(value) for value in sorted(set(int(row["seed"]) for row in rows))
            ),
        }
    )
    for key in [
        "sample_unit",
        "pooling",
        "view_mode",
        "robust_quantile",
        "filtered_upper_fraction",
        "fusion_dim",
        "fusion_dropout",
        "normalization_eps",
        "online_old_seen_trials",
        "online_novel_unseen_trials",
        "online_novel_seen_trials",
        "checkpoint_selection",
        "selection_epoch",
    ]:
        values = {row.get(key) for row in rows}
        if len(values) != 1:
            raise RuntimeError(f"Mixed {key} values in one online aggregate: {values}")
        record[key] = values.pop()
    for key in METRIC_KEYS + ["num_samples", "mean_confidence", "mean_predictive_entropy"]:
        mean, std = mean_std(rows, key)
        record[f"mean_{key}"] = mean
        record[f"std_{key}"] = std

    if include_confusion:
        labels = {row["labels"] for row in rows}
        if len(labels) != 1:
            raise RuntimeError("Cannot pool confusion matrices with different labels.")
        first = np.asarray(json.loads(rows[0]["confusion_matrix"]), dtype=np.int64)
        confusion = sum(
            (
                np.asarray(json.loads(row["confusion_matrix"]), dtype=np.int64)
                for row in rows
            ),
            start=np.zeros_like(first),
        )
        pooled_accuracy, pooled_macro_f1 = metrics_from_confusion(confusion)
        record["labels"] = labels.pop()
        record["pooled_accuracy"] = pooled_accuracy
        record["pooled_macro_f1"] = pooled_macro_f1
        record["pooled_num_samples"] = int(confusion.sum())
        record["pooled_confusion_matrix"] = json.dumps(
            confusion.tolist(), separators=(",", ":")
        )
    return record


def build_aggregates(rows):
    by_seed_session = []
    by_session = []
    for session in (1, 2, 3):
        session_rows = [row for row in rows if int(row["session"]) == session]
        for seed in sorted(set(int(row["seed"]) for row in session_rows)):
            seed_rows = [row for row in session_rows if int(row["seed"]) == seed]
            by_seed_session.append(
                aggregate_group(seed_rows, {"session": session, "seed": seed})
            )
        by_session.append(aggregate_group(session_rows, {"session": session}))

    continual_rows = []
    identities = sorted(set((int(row["fold"]), int(row["seed"])) for row in rows))
    for fold, seed in identities:
        run_rows = [
            row for row in rows if int(row["fold"]) == fold and int(row["seed"]) == seed
        ]
        run_rows.sort(key=lambda row: int(row["session"]))
        if [int(row["session"]) for row in run_rows] != [1, 2, 3]:
            raise RuntimeError(f"Fold {fold}, seed {seed} lacks all three sessions.")
        record = {
            key: run_rows[0][key]
            for key in [
                "sample_unit",
                "pooling",
                "view_mode",
                "robust_quantile",
                "filtered_upper_fraction",
                "fusion_dim",
                "fusion_dropout",
                "normalization_eps",
                "online_old_seen_trials",
                "online_novel_unseen_trials",
                "online_novel_seen_trials",
                "checkpoint_selection",
                "selection_epoch",
            ]
        }
        record.update({"fold": fold, "seed": seed, "session_scope": "mean_1_3"})
        for key in METRIC_KEYS + ["num_samples", "mean_confidence", "mean_predictive_entropy"]:
            values = [row[key] for row in run_rows if row.get(key) is not None]
            record[key] = statistics.fmean(values) if values else None
        continual_rows.append(record)

    continual_summary = aggregate_group(
        continual_rows,
        {"session": "mean_1_3"},
        include_confusion=False,
    )
    return by_seed_session, by_session + [continual_summary], continual_rows


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
    by_seed_session, summary, continual_rows = build_aggregates(rows)
    runs_path = root / f"{args.output_prefix}_runs.csv"
    seed_path = root / f"{args.output_prefix}_by_seed_session.csv"
    continual_path = root / f"{args.output_prefix}_continual_by_run.csv"
    summary_path = root / f"{args.output_prefix}_summary.csv"
    json_path = root / f"{args.output_prefix}_summary.json"
    write_csv(runs_path, rows)
    write_csv(seed_path, by_seed_session)
    write_csv(continual_path, continual_rows)
    write_csv(summary_path, summary)
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "runs": rows,
                "by_seed_session": by_seed_session,
                "continual_by_run": continual_rows,
                "summary": summary,
            },
            handle,
            ensure_ascii=False,
            indent=2,
        )

    print("session,runs,folds,seeds,all_accuracy,new_accuracy,unseen_accuracy,macro_f1")
    for row in summary:
        print(
            f"{row['session']},{row['num_runs']},{row['num_folds']},"
            f"{row['num_seeds']},{row['mean_gcd_all_accuracy']:.6f},"
            f"{row['mean_gcd_new_accuracy']:.6f},"
            f"{row['mean_gcd_unseen_accuracy']:.6f},"
            f"{row['mean_macro_f1']:.6f}"
        )
    print(f"Run table: {runs_path}")
    print(f"By-seed/session table: {seed_path}")
    print(f"Continual by-run table: {continual_path}")
    print(f"Summary CSV: {summary_path}")
    print(f"Summary JSON: {json_path}")


if __name__ == "__main__":
    main()
