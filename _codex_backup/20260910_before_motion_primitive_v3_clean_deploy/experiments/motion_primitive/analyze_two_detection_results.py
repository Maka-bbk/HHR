"""Compare the old/new-swap and w128 probes against the A2/E0 baseline.

The three inputs are peak/valley result roots.  Only the seven members with
profile=A2, arm=E0 and seed=0 are admitted.  The swap comparison deliberately
does not subtract aggregate CGCD metrics: its Session-2 physical activity set
is different.  It reports paired fold deltas only for physical activities
present in both roots (plus the Sitting/Standing mean recall).  The w128 root
must use exactly the baseline physical label space, so all registered metrics
may be paired by held-out-subject fold.

This module is analysis-only.  It never imports a trainer or rewrites a run.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


EXPECTED_FOLDS = tuple(range(1, 8))
EXPECTED_PROFILE = "A2"
EXPECTED_ARM = "E0"
EXPECTED_SEED = 0
EXPECTED_TEST_TRIAL_COUNT = 52
VARIANTS = ("state", "no_state")
METRICS = (
    "all_accuracy",
    "old_accuracy",
    "new_accuracy",
    "h_score",
    "macro_f1",
)
SIT_STAND = ("Sitting", "Standing")


@dataclass(frozen=True)
class ClassRecall:
    activity_name: str
    protocol_label: int
    support: int
    correct: int
    recall: float


@dataclass(frozen=True)
class Member:
    condition: str
    fold: int
    result_path: Path
    prediction_path: Path
    metrics: Mapping[str, Mapping[str, float]]
    class_recalls: Mapping[str, Mapping[str, ClassRecall]]
    label_by_activity: Mapping[str, int]
    test_trial_ids: tuple[int, ...]


def _read_json_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"Could not read JSON object {path}: {error}") from error
    if not isinstance(value, dict):
        raise RuntimeError(f"Expected a JSON object: {path}")
    return value


def _exact_int(value: Any, context: str) -> int:
    if isinstance(value, bool):
        raise RuntimeError(f"{context} must be an integer, not a boolean.")
    try:
        numeric = float(value)
    except (TypeError, ValueError) as error:
        raise RuntimeError(f"{context} must be an integer, got {value!r}.") from error
    if not math.isfinite(numeric) or not numeric.is_integer():
        raise RuntimeError(f"{context} must be an integer, got {value!r}.")
    return int(numeric)


def _unit_float(value: Any, context: str) -> float:
    if isinstance(value, bool):
        raise RuntimeError(f"{context} must be numeric, not a boolean.")
    try:
        numeric = float(value)
    except (TypeError, ValueError) as error:
        raise RuntimeError(f"{context} must be numeric, got {value!r}.") from error
    if not math.isfinite(numeric) or not 0.0 <= numeric <= 1.0:
        raise RuntimeError(f"{context} must lie in [0,1], got {value!r}.")
    return numeric


def _close(observed: float, expected: float, context: str) -> None:
    if not math.isclose(
        float(observed), float(expected), rel_tol=0.0, abs_tol=1e-10
    ):
        raise RuntimeError(
            f"{context} disagrees with session2_predictions.csv: "
            f"JSON={observed}, recomputed={expected}."
        )


def _macro_f1(labels: Sequence[int], predictions: Sequence[int]) -> float:
    class_ids = sorted(set(int(value) for value in labels))
    values = []
    for class_id in class_ids:
        tp = sum(
            int(truth == class_id and predicted == class_id)
            for truth, predicted in zip(labels, predictions)
        )
        fp = sum(
            int(truth != class_id and predicted == class_id)
            for truth, predicted in zip(labels, predictions)
        )
        fn = sum(
            int(truth == class_id and predicted != class_id)
            for truth, predicted in zip(labels, predictions)
        )
        denominator = 2 * tp + fp + fn
        values.append(2.0 * tp / denominator if denominator else 0.0)
    return float(sum(values) / len(values))


def _h_score(old_accuracy: float, new_accuracy: float) -> float:
    denominator = float(old_accuracy) + float(new_accuracy)
    return (
        2.0 * float(old_accuracy) * float(new_accuracy) / denominator
        if denominator > 0.0
        else 0.0
    )


def _candidate_identity(payload: Mapping[str, Any]) -> tuple[str, int, int] | None:
    request = payload.get("request_identity")
    if not isinstance(request, Mapping):
        return None
    try:
        profile = str(request.get("profile", "")).upper()
        seed = _exact_int(request.get("seed"), "request_identity.seed")
        fold = _exact_int(request.get("fold"), "request_identity.fold")
    except RuntimeError:
        return None
    return profile, seed, fold


def _load_prediction_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing prediction artifact: {path}")
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            required = {
                "trial_global_id",
                "subject_id",
                "activity_label_0based",
                "activity_name",
                "E0_state_aligned_prediction",
                "E0_no_state_aligned_prediction",
            }
            missing = required - set(reader.fieldnames or ())
            if missing:
                raise RuntimeError(
                    f"{path} lacks required columns: {sorted(missing)}."
                )
            rows = list(reader)
    except OSError as error:
        raise RuntimeError(f"Could not read {path}: {error}") from error
    if not rows:
        raise RuntimeError(f"Prediction artifact is empty: {path}")
    return rows


def _extract_member(
    condition: str, fold: int, result_path: Path, payload: Mapping[str, Any]
) -> Member:
    request = payload.get("request_identity")
    if not isinstance(request, Mapping):
        raise RuntimeError(f"Run lacks request_identity: {result_path}")
    requested_arms = request.get("arms")
    if not isinstance(requested_arms, list) or EXPECTED_ARM not in {
        str(value) for value in requested_arms
    }:
        raise RuntimeError(f"Run did not request arm E0: {result_path}")

    arm_results = payload.get("arm_results")
    if not isinstance(arm_results, Mapping) or not isinstance(
        arm_results.get(EXPECTED_ARM), Mapping
    ):
        raise RuntimeError(f"Run lacks arm_results.E0: {result_path}")
    variant_results = arm_results[EXPECTED_ARM].get("readout_variants")
    if not isinstance(variant_results, Mapping) or set(variant_results) != set(
        VARIANTS
    ):
        raise RuntimeError(
            f"Run must contain exactly E0 state/no_state variants: {result_path}"
        )

    prediction_path = result_path.with_name("session2_predictions.csv")
    rows = _load_prediction_rows(prediction_path)
    if len(rows) != EXPECTED_TEST_TRIAL_COUNT:
        raise RuntimeError(
            f"{condition} fold {fold} has {len(rows)} test trials; "
            f"expected {EXPECTED_TEST_TRIAL_COUNT}."
        )

    trial_ids: list[int] = []
    labels: list[int] = []
    activity_names: list[str] = []
    predictions_by_variant: dict[str, list[int]] = {
        variant: [] for variant in VARIANTS
    }
    for row_number, row in enumerate(rows, start=2):
        prefix = f"{prediction_path}:row {row_number}"
        trial_ids.append(_exact_int(row["trial_global_id"], f"{prefix}:trial"))
        _exact_int(row["subject_id"], f"{prefix}:subject")
        labels.append(
            _exact_int(row["activity_label_0based"], f"{prefix}:label")
        )
        name = str(row["activity_name"]).strip()
        if not name:
            raise RuntimeError(f"{prefix}: activity_name is empty.")
        activity_names.append(name)
        for variant in VARIANTS:
            prediction = _exact_int(
                row[f"E0_{variant}_aligned_prediction"],
                f"{prefix}:E0_{variant}_aligned_prediction",
            )
            if prediction < 0:
                raise RuntimeError(f"{prefix}: aligned prediction is negative.")
            predictions_by_variant[variant].append(prediction)
    if len(trial_ids) != len(set(trial_ids)):
        raise RuntimeError(
            f"{condition} fold {fold} contains duplicate test trial IDs."
        )
    if sorted(set(labels)) != list(range(10)):
        raise RuntimeError(
            f"{condition} fold {fold} must expose protocol labels 0..9, got "
            f"{sorted(set(labels))}."
        )

    label_by_activity: dict[str, int] = {}
    activity_by_label: dict[int, str] = {}
    for name, label in zip(activity_names, labels):
        previous_label = label_by_activity.setdefault(name, label)
        previous_name = activity_by_label.setdefault(label, name)
        if previous_label != label or previous_name != name:
            raise RuntimeError(
                f"{condition} fold {fold} has a non-bijective activity/label map."
            )
    if len(label_by_activity) != 10 or len(activity_by_label) != 10:
        raise RuntimeError(
            f"{condition} fold {fold} must contain ten physical activities."
        )
    missing_focus = set(SIT_STAND) - set(label_by_activity)
    if missing_focus:
        raise RuntimeError(
            f"{condition} fold {fold} lacks focus activities: {sorted(missing_focus)}."
        )

    extracted_metrics: dict[str, dict[str, float]] = {}
    class_recalls: dict[str, dict[str, ClassRecall]] = {}
    old_mask = [label < 6 for label in labels]
    new_mask = [not value for value in old_mask]
    for variant in VARIANTS:
        block = variant_results[variant]
        if not isinstance(block, Mapping) or not isinstance(
            block.get("metrics"), Mapping
        ):
            raise RuntimeError(
                f"Run lacks E0/{variant}/metrics: {result_path}"
            )
        metrics = block["metrics"]
        if metrics.get("alignment_scope") != "single_global_hungarian" or _exact_int(
            metrics.get("hungarian_call_count"),
            f"{condition} fold {fold} {variant} hungarian_call_count",
        ) != 1:
            raise RuntimeError(
                f"{condition} fold {fold} {variant} did not use one global Hungarian map."
            )
        extracted_metrics[variant] = {
            name: _unit_float(
                metrics.get(name), f"{condition} fold {fold} {variant} {name}"
            )
            for name in METRICS
        }

        predictions = predictions_by_variant[variant]
        recorded_aligned = metrics.get("aligned_predictions")
        if not isinstance(recorded_aligned, list) or [
            _exact_int(value, f"{condition} fold {fold} aligned_predictions")
            for value in recorded_aligned
        ] != predictions:
            raise RuntimeError(
                f"{condition} fold {fold} {variant} JSON/CSV aligned predictions differ."
            )
        correct = [truth == predicted for truth, predicted in zip(labels, predictions)]
        all_accuracy = sum(correct) / len(correct)
        old_accuracy = sum(
            int(value) for value, selected in zip(correct, old_mask) if selected
        ) / sum(old_mask)
        new_accuracy = sum(
            int(value) for value, selected in zip(correct, new_mask) if selected
        ) / sum(new_mask)
        recomputed = {
            "all_accuracy": all_accuracy,
            "old_accuracy": old_accuracy,
            "new_accuracy": new_accuracy,
            "h_score": _h_score(old_accuracy, new_accuracy),
            "macro_f1": _macro_f1(labels, predictions),
        }
        for name, value in recomputed.items():
            _close(
                extracted_metrics[variant][name],
                value,
                f"{condition} fold {fold} {variant} {name}",
            )

        per_name: dict[str, ClassRecall] = {}
        recorded_recalls = metrics.get("per_class_recall")
        if not isinstance(recorded_recalls, Mapping):
            raise RuntimeError(
                f"{condition} fold {fold} {variant} lacks per_class_recall."
            )
        for name, label in sorted(
            label_by_activity.items(), key=lambda value: (value[1], value[0])
        ):
            positions = [
                index
                for index, observed_label in enumerate(labels)
                if observed_label == label
            ]
            count = len(positions)
            correct_count = sum(int(correct[index]) for index in positions)
            recall = correct_count / count
            _close(
                _unit_float(
                    recorded_recalls.get(str(label)),
                    f"{condition} fold {fold} {variant} recall label {label}",
                ),
                recall,
                f"{condition} fold {fold} {variant} recall label {label}",
            )
            per_name[name] = ClassRecall(
                activity_name=name,
                protocol_label=label,
                support=count,
                correct=correct_count,
                recall=recall,
            )
        class_recalls[variant] = per_name

        readout = block.get("readout")
        if not isinstance(readout, Mapping) or _exact_int(
            readout.get("test_trial_count"),
            f"{condition} fold {fold} {variant} test_trial_count",
        ) != len(rows):
            raise RuntimeError(
                f"{condition} fold {fold} {variant} readout test count differs."
            )

    return Member(
        condition=condition,
        fold=fold,
        result_path=result_path.resolve(),
        prediction_path=prediction_path.resolve(),
        metrics=extracted_metrics,
        class_recalls=class_recalls,
        label_by_activity=dict(sorted(label_by_activity.items())),
        test_trial_ids=tuple(trial_ids),
    )


def discover_members(root: Path, condition: str) -> dict[int, Member]:
    root = Path(root).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"{condition} root is not a directory: {root}")
    members: dict[int, Member] = {}
    for result_path in sorted(root.rglob("experiment_result.json")):
        payload = _read_json_object(result_path)
        identity = _candidate_identity(payload)
        if identity is None:
            continue
        profile, seed, fold = identity
        if profile != EXPECTED_PROFILE or seed != EXPECTED_SEED:
            continue
        arm_results = payload.get("arm_results")
        if not isinstance(arm_results, Mapping) or EXPECTED_ARM not in arm_results:
            continue
        if fold not in EXPECTED_FOLDS:
            raise RuntimeError(
                f"{condition} contains an A2/E0/seed0 member outside folds 1..7: "
                f"fold={fold}, path={result_path}."
            )
        if fold in members:
            raise RuntimeError(
                f"{condition} has duplicate A2/E0/seed0 fold {fold} members: "
                f"{members[fold].result_path} and {result_path}."
            )
        members[fold] = _extract_member(condition, fold, result_path, payload)
    missing = sorted(set(EXPECTED_FOLDS) - set(members))
    extra = sorted(set(members) - set(EXPECTED_FOLDS))
    if missing or extra:
        raise RuntimeError(
            f"{condition} must contain exactly A2/E0/seed0 folds 1..7; "
            f"missing={missing}, extra={extra}, root={root}."
        )

    reference = members[EXPECTED_FOLDS[0]].label_by_activity
    for fold in EXPECTED_FOLDS[1:]:
        if members[fold].label_by_activity != reference:
            raise RuntimeError(
                f"{condition} physical label mapping changes between folds 1 and {fold}."
            )
    return members


def _condition_audit(members: Mapping[int, Member]) -> dict[str, Any]:
    first = members[EXPECTED_FOLDS[0]]
    label_by_activity = dict(first.label_by_activity)
    old_names = sorted(
        name for name, label in label_by_activity.items() if int(label) < 6
    )
    new_names = sorted(
        name for name, label in label_by_activity.items() if int(label) >= 6
    )
    return {
        "profile": EXPECTED_PROFILE,
        "arm": EXPECTED_ARM,
        "seed": EXPECTED_SEED,
        "folds": list(EXPECTED_FOLDS),
        "member_count": len(members),
        "physical_activity_names": sorted(label_by_activity),
        "protocol_label_by_activity_name": label_by_activity,
        "old_activity_names_protocol_labels_0_to_5": old_names,
        "new_activity_names_protocol_labels_6_to_9_at_session2": new_names,
        "test_trial_count_by_fold": {
            str(fold): len(members[fold].test_trial_ids) for fold in EXPECTED_FOLDS
        },
        "result_paths_by_fold": {
            str(fold): str(members[fold].result_path) for fold in EXPECTED_FOLDS
        },
    }


def _sit_stand_balance(member: Member, variant: str) -> float:
    return float(
        sum(member.class_recalls[variant][name].recall for name in SIT_STAND)
        / len(SIT_STAND)
    )


def _fold_metric_rows(all_members: Mapping[str, Mapping[int, Member]]) -> list[dict]:
    rows = []
    for condition, members in all_members.items():
        for fold in EXPECTED_FOLDS:
            member = members[fold]
            for variant in VARIANTS:
                sitting = member.class_recalls[variant]["Sitting"]
                standing = member.class_recalls[variant]["Standing"]
                rows.append(
                    {
                        "condition": condition,
                        "profile": EXPECTED_PROFILE,
                        "arm": EXPECTED_ARM,
                        "seed": EXPECTED_SEED,
                        "fold": fold,
                        "readout_variant": variant,
                        **member.metrics[variant],
                        "sitting_recall": sitting.recall,
                        "sitting_support": sitting.support,
                        "standing_recall": standing.recall,
                        "standing_support": standing.support,
                        "sit_stand_balanced_accuracy": _sit_stand_balance(
                            member, variant
                        ),
                        "test_trial_count": len(member.test_trial_ids),
                        "result_path": str(member.result_path),
                    }
                )
    return rows


def _class_recall_rows(
    all_members: Mapping[str, Mapping[int, Member]]
) -> list[dict]:
    rows = []
    for condition, members in all_members.items():
        for fold in EXPECTED_FOLDS:
            member = members[fold]
            for variant in VARIANTS:
                values = sorted(
                    member.class_recalls[variant].values(),
                    key=lambda value: (value.protocol_label, value.activity_name),
                )
                for item in values:
                    rows.append(
                        {
                            "condition": condition,
                            "profile": EXPECTED_PROFILE,
                            "arm": EXPECTED_ARM,
                            "seed": EXPECTED_SEED,
                            "fold": fold,
                            "readout_variant": variant,
                            "activity_name": item.activity_name,
                            "protocol_label_0based": item.protocol_label,
                            "protocol_old_or_new": (
                                "old" if item.protocol_label < 6 else "new"
                            ),
                            "support": item.support,
                            "correct": item.correct,
                            "recall": item.recall,
                        }
                    )
    return rows


def _paired_row(
    *,
    comparison: str,
    validity: str,
    endpoint_type: str,
    endpoint: str,
    variant: str,
    fold: int,
    baseline_value: float,
    comparator_value: float,
    activity_name: str = "",
    baseline_support: int | str = "",
    comparator_support: int | str = "",
    physical_label_space_same: bool,
    test_trial_ids_same: bool,
    note: str,
) -> dict[str, Any]:
    return {
        "comparison": comparison,
        "comparison_validity": validity,
        "endpoint_type": endpoint_type,
        "endpoint": endpoint,
        "activity_name": activity_name,
        "readout_variant": variant,
        "fold": int(fold),
        "baseline_value": float(baseline_value),
        "comparator_value": float(comparator_value),
        "delta_comparator_minus_baseline": float(
            comparator_value - baseline_value
        ),
        "baseline_support": baseline_support,
        "comparator_support": comparator_support,
        "physical_label_space_same": bool(physical_label_space_same),
        "test_trial_ids_same": bool(test_trial_ids_same),
        "note": note,
    }


def _paired_rows(
    baseline: Mapping[int, Member],
    swap: Mapping[int, Member],
    w128: Mapping[int, Member],
) -> tuple[list[dict], list[str]]:
    baseline_map = dict(baseline[1].label_by_activity)
    swap_map = dict(swap[1].label_by_activity)
    w128_map = dict(w128[1].label_by_activity)
    if w128_map != baseline_map:
        raise RuntimeError(
            "w128 must use exactly the baseline physical activity-to-protocol-label "
            f"mapping; baseline={baseline_map}, w128={w128_map}."
        )
    if set(swap_map) == set(baseline_map):
        raise RuntimeError(
            "swap Session-2 unexpectedly has the same physical activity set as "
            "baseline; the registered swap analysis assumes different sets."
        )
    common = sorted(set(baseline_map) & set(swap_map))
    if not common or not set(SIT_STAND).issubset(common):
        raise RuntimeError(
            "baseline/swap common physical activities must include Sitting and "
            f"Standing; observed common={common}."
        )

    rows: list[dict] = []
    for fold in EXPECTED_FOLDS:
        base = baseline[fold]
        changed_window = w128[fold]
        exchanged = swap[fold]
        w128_trials_same = set(base.test_trial_ids) == set(
            changed_window.test_trial_ids
        )
        swap_trials_same = set(base.test_trial_ids) == set(exchanged.test_trial_ids)
        for variant in VARIANTS:
            for metric in METRICS:
                rows.append(
                    _paired_row(
                        comparison="w128_minus_baseline",
                        validity="paired_same_physical_label_space",
                        endpoint_type="cgcd_metric",
                        endpoint=metric,
                        variant=variant,
                        fold=fold,
                        baseline_value=base.metrics[variant][metric],
                        comparator_value=changed_window.metrics[variant][metric],
                        physical_label_space_same=True,
                        test_trial_ids_same=w128_trials_same,
                        note=(
                            "Fold-paired w128 comparison; physical activity and "
                            "protocol-label mapping are identical."
                        ),
                    )
                )
            rows.append(
                _paired_row(
                    comparison="w128_minus_baseline",
                    validity="paired_same_physical_label_space",
                    endpoint_type="sit_stand_balanced_accuracy",
                    endpoint="sit_stand_balanced_accuracy",
                    variant=variant,
                    fold=fold,
                    baseline_value=_sit_stand_balance(base, variant),
                    comparator_value=_sit_stand_balance(changed_window, variant),
                    baseline_support=sum(
                        base.class_recalls[variant][name].support
                        for name in SIT_STAND
                    ),
                    comparator_support=sum(
                        changed_window.class_recalls[variant][name].support
                        for name in SIT_STAND
                    ),
                    physical_label_space_same=True,
                    test_trial_ids_same=w128_trials_same,
                    note="Mean of Sitting and Standing class recall within the fold.",
                )
            )
            for activity_name in sorted(baseline_map):
                left = base.class_recalls[variant][activity_name]
                right = changed_window.class_recalls[variant][activity_name]
                rows.append(
                    _paired_row(
                        comparison="w128_minus_baseline",
                        validity="paired_same_physical_label_space",
                        endpoint_type="physical_class_recall",
                        endpoint="class_recall",
                        activity_name=activity_name,
                        variant=variant,
                        fold=fold,
                        baseline_value=left.recall,
                        comparator_value=right.recall,
                        baseline_support=left.support,
                        comparator_support=right.support,
                        physical_label_space_same=True,
                        test_trial_ids_same=w128_trials_same,
                        note=(
                            "Recall inherits each member's one global Hungarian "
                            "mapping; it is not a separately aligned class score."
                        ),
                    )
                )

            rows.append(
                _paired_row(
                    comparison="swap_minus_baseline",
                    validity="descriptive_common_physical_classes_only",
                    endpoint_type="sit_stand_balanced_accuracy",
                    endpoint="sit_stand_balanced_accuracy",
                    variant=variant,
                    fold=fold,
                    baseline_value=_sit_stand_balance(base, variant),
                    comparator_value=_sit_stand_balance(exchanged, variant),
                    baseline_support=sum(
                        base.class_recalls[variant][name].support
                        for name in SIT_STAND
                    ),
                    comparator_support=sum(
                        exchanged.class_recalls[variant][name].support
                        for name in SIT_STAND
                    ),
                    physical_label_space_same=False,
                    test_trial_ids_same=swap_trials_same,
                    note=(
                        "Descriptive fold delta only: class status, test support, "
                        "and the ten-class Hungarian context can differ."
                    ),
                )
            )
            for activity_name in common:
                left = base.class_recalls[variant][activity_name]
                right = exchanged.class_recalls[variant][activity_name]
                rows.append(
                    _paired_row(
                        comparison="swap_minus_baseline",
                        validity="descriptive_common_physical_classes_only",
                        endpoint_type="physical_class_recall",
                        endpoint="class_recall",
                        activity_name=activity_name,
                        variant=variant,
                        fold=fold,
                        baseline_value=left.recall,
                        comparator_value=right.recall,
                        baseline_support=left.support,
                        comparator_support=right.support,
                        physical_label_space_same=False,
                        test_trial_ids_same=swap_trials_same,
                        note=(
                            "Only the same named physical activity is compared; "
                            "recall still inherits a different global Hungarian map."
                        ),
                    )
                )
    return rows, common


def _delta_summaries(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[
        tuple[str, str, str, str, str, str], list[float]
    ] = defaultdict(list)
    for row in rows:
        key = (
            str(row["comparison"]),
            str(row["comparison_validity"]),
            str(row["endpoint_type"]),
            str(row["endpoint"]),
            str(row["activity_name"]),
            str(row["readout_variant"]),
        )
        grouped[key].append(float(row["delta_comparator_minus_baseline"]))
    summaries = []
    for key, values in sorted(grouped.items()):
        if len(values) != len(EXPECTED_FOLDS):
            raise RuntimeError(
                f"Paired delta group {key} has {len(values)} folds, expected 7."
            )
        comparison, validity, endpoint_type, endpoint, activity, variant = key
        summaries.append(
            {
                "comparison": comparison,
                "comparison_validity": validity,
                "endpoint_type": endpoint_type,
                "endpoint": endpoint,
                "activity_name": activity,
                "readout_variant": variant,
                "fold_count": len(values),
                "mean_delta": statistics.fmean(values),
                "sample_std_delta": statistics.stdev(values),
                "median_delta": statistics.median(values),
                "minimum_delta": min(values),
                "maximum_delta": max(values),
                "positive_fold_count": sum(value > 0.0 for value in values),
                "zero_fold_count": sum(value == 0.0 for value in values),
                "negative_fold_count": sum(value < 0.0 for value in values),
            }
        )
    return summaries


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise RuntimeError(f"Refusing to write an empty CSV: {path}")
    fieldnames = list(rows[0].keys())
    if any(set(row) != set(fieldnames) for row in rows):
        raise RuntimeError(f"CSV rows have inconsistent fields: {path}")
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def analyze(
    baseline_root: Path,
    swap_root: Path,
    w128_root: Path,
    output_dir: Path,
) -> dict[str, Any]:
    output_dir = Path(output_dir).expanduser().resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    all_members = {
        "baseline": discover_members(Path(baseline_root), "baseline"),
        "swap": discover_members(Path(swap_root), "swap"),
        "w128": discover_members(Path(w128_root), "w128"),
    }
    audits = {
        condition: _condition_audit(members)
        for condition, members in all_members.items()
    }
    fold_metrics = _fold_metric_rows(all_members)
    class_recalls = _class_recall_rows(all_members)
    paired_deltas, common_swap_activities = _paired_rows(
        all_members["baseline"], all_members["swap"], all_members["w128"]
    )
    delta_summaries = _delta_summaries(paired_deltas)

    csv_files = {
        "fold_metrics": "fold_metrics.csv",
        "class_recalls": "class_recalls.csv",
        "paired_fold_deltas": "paired_fold_deltas.csv",
        "paired_delta_summaries": "paired_delta_summaries.csv",
    }
    _write_csv(output_dir / csv_files["fold_metrics"], fold_metrics)
    _write_csv(output_dir / csv_files["class_recalls"], class_recalls)
    _write_csv(output_dir / csv_files["paired_fold_deltas"], paired_deltas)
    _write_csv(
        output_dir / csv_files["paired_delta_summaries"], delta_summaries
    )

    result = {
        "schema": "two_detection_result_analysis_v1",
        "selection": {
            "profile": EXPECTED_PROFILE,
            "arm": EXPECTED_ARM,
            "seed": EXPECTED_SEED,
            "folds": list(EXPECTED_FOLDS),
            "readout_variants": list(VARIANTS),
            "metrics_from_experiment_result": list(METRICS),
            "class_recall_source": (
                "session2_predictions.csv aligned predictions, grouped by "
                "activity_name"
            ),
        },
        "condition_audits": audits,
        "comparison_policy": {
            "swap_minus_baseline": {
                "overall_cgcd_metric_delta_is_valid": False,
                "overall_h_score_delta_is_valid": False,
                "reason": (
                    "The swap Session-2 physical activity set differs from the "
                    "baseline set, and old/new membership changes. All/Old/New/H/"
                    "macro-F1 are retained per condition but are not subtracted."
                ),
                "reported_paired_endpoints": (
                    "same-named common physical-class recall and mean Sitting/"
                    "Standing recall only"
                ),
                "common_physical_activity_names": common_swap_activities,
                "excluded_baseline_only_activity_names": sorted(
                    set(audits["baseline"]["physical_activity_names"])
                    - set(common_swap_activities)
                ),
                "excluded_swap_only_activity_names": sorted(
                    set(audits["swap"]["physical_activity_names"])
                    - set(common_swap_activities)
                ),
            },
            "w128_minus_baseline": {
                "overall_cgcd_metric_delta_is_valid": True,
                "overall_h_score_delta_is_valid": True,
                "reason": (
                    "The physical activity set and activity-to-protocol-label "
                    "mapping were required to match exactly before pairing folds."
                ),
            },
            "hungarian_caveat": (
                "Every class recall uses the member's single global ten-class "
                "Hungarian mapping. A swap class-recall delta is therefore still "
                "affected by the different competing physical-class roster."
            ),
            "causal_caveat": (
                "The old/new swap jointly changes supervised encoder exposure, "
                "normalization, codebook fit data, online sampling/support, and the "
                "Hungarian context. Improvement does not by itself identify online "
                "codebook synchronization as the cause."
            ),
            "inference_unit": (
                "held-out-subject fold; one seed only; seven fold deltas are "
                "descriptive and seeds are not additional independent samples"
            ),
        },
        "fold_metrics": fold_metrics,
        "class_recalls": class_recalls,
        "paired_fold_deltas": paired_deltas,
        "paired_delta_summaries": delta_summaries,
        "csv_files": csv_files,
    }
    json_path = output_dir / "two_detection_analysis.json"
    json_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze seven-fold seed-0 A2/E0 old/new-swap and w128 probes."
        )
    )
    parser.add_argument("--baseline-root", required=True)
    parser.add_argument("--swap-root", required=True)
    parser.add_argument("--w128-root", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = analyze(
        Path(args.baseline_root),
        Path(args.swap_root),
        Path(args.w128_root),
        Path(args.output_dir),
    )
    print(
        json.dumps(
            {
                "schema": result["schema"],
                "output_dir": str(Path(args.output_dir).expanduser().resolve()),
                "common_swap_activity_count": len(
                    result["comparison_policy"]["swap_minus_baseline"][
                        "common_physical_activity_names"
                    ]
                ),
                "paired_fold_delta_row_count": len(
                    result["paired_fold_deltas"]
                ),
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
