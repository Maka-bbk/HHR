from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path
from typing import Sequence

from experiments.motion_primitive.analyze_two_detection_results import analyze


BASELINE_NAMES = (
    "Walking Forward",
    "Walking Left",
    "Walking Right",
    "Walking Upstairs",
    "Walking Downstairs",
    "Running Forward",
    "Jumping Up",
    "Sitting",
    "Standing",
    "Sleeping",
)
SWAP_NAMES = (
    "Jumping Up",
    "Sitting",
    "Standing",
    "Sleeping",
    "Elevator Up",
    "Elevator Down",
    "Walking Forward",
    "Walking Left",
    "Walking Right",
    "Walking Upstairs",
)
VARIANTS = ("state", "no_state")


def _macro_f1(labels: Sequence[int], predictions: Sequence[int]) -> float:
    values = []
    for class_id in sorted(set(labels)):
        tp = sum(
            truth == class_id and predicted == class_id
            for truth, predicted in zip(labels, predictions)
        )
        fp = sum(
            truth != class_id and predicted == class_id
            for truth, predicted in zip(labels, predictions)
        )
        fn = sum(
            truth == class_id and predicted != class_id
            for truth, predicted in zip(labels, predictions)
        )
        denominator = 2 * tp + fp + fn
        values.append(2.0 * tp / denominator if denominator else 0.0)
    return sum(values) / len(values)


def _metrics(labels: list[int], predictions: list[int]) -> dict:
    correct = [truth == predicted for truth, predicted in zip(labels, predictions)]
    old_positions = [index for index, label in enumerate(labels) if label < 6]
    new_positions = [index for index, label in enumerate(labels) if label >= 6]
    old_accuracy = sum(correct[index] for index in old_positions) / len(old_positions)
    new_accuracy = sum(correct[index] for index in new_positions) / len(new_positions)
    denominator = old_accuracy + new_accuracy
    recalls = {}
    for label in range(10):
        positions = [index for index, value in enumerate(labels) if value == label]
        recalls[str(label)] = sum(correct[index] for index in positions) / len(
            positions
        )
    return {
        "all_accuracy": sum(correct) / len(correct),
        "old_accuracy": old_accuracy,
        "new_accuracy": new_accuracy,
        "h_score": (
            2.0 * old_accuracy * new_accuracy / denominator
            if denominator
            else 0.0
        ),
        "macro_f1": _macro_f1(labels, predictions),
        "alignment_scope": "single_global_hungarian",
        "hungarian_call_count": 1,
        "aligned_predictions": predictions,
        "per_class_recall": recalls,
    }


def _predictions(
    labels: list[int], condition: str, variant: str, fold: int
) -> list[int]:
    predictions = list(labels)
    if condition == "baseline":
        for index, label in enumerate(labels):
            if label == 7 or (label == 8 and (index + fold) % 2 == 0):
                predictions[index] = (label + 1) % 10
            elif variant == "no_state" and label == 6 and index % 2 == 0:
                predictions[index] = 5
    elif condition == "w128":
        for index, label in enumerate(labels):
            if label == 9 and (index + fold) % 5 == 0:
                predictions[index] = 8
    return predictions


def _write_member(
    root: Path,
    *,
    condition: str,
    fold: int,
    names_by_label: Sequence[str],
    trial_id_offset: int,
) -> Path:
    member_dir = root / f"fold_{fold:02d}"
    member_dir.mkdir(parents=True)
    labels = [label for label in range(10) for _ in range(5)] + [0, 1]
    predictions = {
        variant: _predictions(labels, condition, variant, fold)
        for variant in VARIANTS
    }
    prediction_path = member_dir / "session2_predictions.csv"
    with prediction_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "trial_global_id",
                "subject_id",
                "activity_label_0based",
                "activity_name",
                "E0_state_aligned_prediction",
                "E0_no_state_aligned_prediction",
            ),
        )
        writer.writeheader()
        for index, label in enumerate(labels):
            writer.writerow(
                {
                    "trial_global_id": trial_id_offset + index,
                    "subject_id": fold,
                    "activity_label_0based": label,
                    "activity_name": names_by_label[label],
                    "E0_state_aligned_prediction": predictions["state"][index],
                    "E0_no_state_aligned_prediction": predictions["no_state"][
                        index
                    ],
                }
            )

    payload = {
        "request_identity": {
            "profile": "A2",
            "fold": fold,
            "seed": 0,
            "arms": ["E0"],
        },
        "arm_results": {
            "E0": {
                "readout_variants": {
                    variant: {
                        "metrics": _metrics(labels, predictions[variant]),
                        "readout": {"test_trial_count": len(labels)},
                    }
                    for variant in VARIANTS
                }
            }
        },
    }
    result_path = member_dir / "experiment_result.json"
    result_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return result_path


def _build_condition(
    root: Path,
    condition: str,
    names_by_label: Sequence[str],
    folds: Sequence[int] = tuple(range(1, 8)),
) -> list[Path]:
    paths = []
    for fold in folds:
        # Baseline and w128 use identical test trial identities; swap represents
        # a different Session-2 roster after changing old/new membership.
        base_offset = fold * 1_000
        if condition == "swap":
            base_offset += 100_000
        paths.append(
            _write_member(
                root,
                condition=condition,
                fold=fold,
                names_by_label=names_by_label,
                trial_id_offset=base_offset,
            )
        )
    return paths


class AnalyzeTwoDetectionResultsTests(unittest.TestCase):
    def test_complete_analysis_reports_only_valid_paired_endpoints(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            baseline = root / "baseline"
            swap = root / "swap"
            w128 = root / "w128"
            _build_condition(baseline, "baseline", BASELINE_NAMES)
            _build_condition(swap, "swap", SWAP_NAMES)
            _build_condition(w128, "w128", BASELINE_NAMES)

            result = analyze(baseline, swap, w128, root / "analysis")

            self.assertEqual(result["schema"], "two_detection_result_analysis_v1")
            self.assertEqual(len(result["fold_metrics"]), 42)
            self.assertEqual(len(result["class_recalls"]), 420)
            self.assertEqual(len(result["paired_fold_deltas"]), 350)
            self.assertEqual(len(result["paired_delta_summaries"]), 50)
            policy = result["comparison_policy"]
            self.assertFalse(
                policy["swap_minus_baseline"]["overall_h_score_delta_is_valid"]
            )
            self.assertEqual(
                policy["swap_minus_baseline"]["common_physical_activity_names"],
                sorted(set(BASELINE_NAMES) & set(SWAP_NAMES)),
            )

            swap_rows = [
                row
                for row in result["paired_fold_deltas"]
                if row["comparison"] == "swap_minus_baseline"
            ]
            self.assertTrue(swap_rows)
            self.assertFalse(
                any(row["endpoint_type"] == "cgcd_metric" for row in swap_rows)
            )
            self.assertFalse(any(row["endpoint"] == "h_score" for row in swap_rows))
            swap_class_names = {
                row["activity_name"]
                for row in swap_rows
                if row["endpoint_type"] == "physical_class_recall"
            }
            self.assertEqual(
                swap_class_names, set(BASELINE_NAMES) & set(SWAP_NAMES)
            )

            w128_state_h = [
                row
                for row in result["paired_fold_deltas"]
                if row["comparison"] == "w128_minus_baseline"
                and row["readout_variant"] == "state"
                and row["endpoint"] == "h_score"
            ]
            self.assertEqual(len(w128_state_h), 7)
            self.assertTrue(all(row["test_trial_ids_same"] for row in w128_state_h))
            for filename in (
                "two_detection_analysis.json",
                "fold_metrics.csv",
                "class_recalls.csv",
                "paired_fold_deltas.csv",
                "paired_delta_summaries.csv",
            ):
                self.assertTrue((root / "analysis" / filename).is_file())

    def test_missing_fold_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _build_condition(
                root / "baseline",
                "baseline",
                BASELINE_NAMES,
                folds=tuple(range(1, 7)),
            )
            _build_condition(root / "swap", "swap", SWAP_NAMES)
            _build_condition(root / "w128", "w128", BASELINE_NAMES)

            with self.assertRaisesRegex(RuntimeError, r"missing=\[7\]"):
                analyze(
                    root / "baseline",
                    root / "swap",
                    root / "w128",
                    root / "analysis",
                )

    def test_w128_mapping_mismatch_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _build_condition(root / "baseline", "baseline", BASELINE_NAMES)
            _build_condition(root / "swap", "swap", SWAP_NAMES)
            mismatched = list(BASELINE_NAMES)
            mismatched[0], mismatched[1] = mismatched[1], mismatched[0]
            _build_condition(root / "w128", "w128", mismatched)

            with self.assertRaisesRegex(
                RuntimeError, "w128 must use exactly the baseline"
            ):
                analyze(
                    root / "baseline",
                    root / "swap",
                    root / "w128",
                    root / "analysis",
                )

    def test_json_csv_prediction_mismatch_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            baseline_paths = _build_condition(
                root / "baseline", "baseline", BASELINE_NAMES
            )
            _build_condition(root / "swap", "swap", SWAP_NAMES)
            _build_condition(root / "w128", "w128", BASELINE_NAMES)
            result_path = baseline_paths[0]
            payload = json.loads(result_path.read_text(encoding="utf-8"))
            recorded = payload["arm_results"]["E0"]["readout_variants"][
                "state"
            ]["metrics"]["aligned_predictions"]
            recorded[0] = (int(recorded[0]) + 1) % 10
            result_path.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(
                RuntimeError, "JSON/CSV aligned predictions differ"
            ):
                analyze(
                    root / "baseline",
                    root / "swap",
                    root / "w128",
                    root / "analysis",
                )


if __name__ == "__main__":
    unittest.main()
