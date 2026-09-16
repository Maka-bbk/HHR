"""Paired confirmation analysis for the four fold06 Session-2 proxy runs.

This script deliberately does not refit a codebook or use labels as model
inputs.  It audits four already-frozen prediction files and quantifies paired
accuracy differences by resampling trials within subject-by-activity strata.
The resulting intervals describe stability conditional on held-out subjects
S4/S5; they are not confidence intervals for unseen-subject generalisation.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ARMS = ("U0_coarse", "U1_residual", "U2_gravity", "U3_joint")
METRICS = ("all_accuracy", "old_accuracy", "new_accuracy")
EXPECTED_RUNS = {
    "A0_fixed": ("A0", "fixed_window"),
    "A3_fixed": ("A3", "fixed_window"),
    "A0_changepoint": ("A0", "motion_encoder_changepoint"),
    "A3_changepoint": ("A3", "motion_encoder_changepoint"),
}


@dataclass(frozen=True)
class FrozenRun:
    key: str
    directory: Path
    result: dict
    trial_ids: np.ndarray
    subjects: np.ndarray
    truth: np.ndarray
    aligned_predictions: Mapping[str, np.ndarray]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require_new_output_dir(path: Path) -> Path:
    resolved = path.resolve()
    if resolved.exists():
        raise FileExistsError(f"Output directory must not already exist: {resolved}")
    return resolved


def _read_json(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def _read_prediction_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise RuntimeError(f"Prediction CSV is empty: {path}")
    return rows


def load_frozen_run(key: str, directory: Path) -> FrozenRun:
    if key not in EXPECTED_RUNS:
        raise ValueError(f"Unknown registered run key {key!r}.")
    directory = Path(directory).resolve()
    result = _read_json(directory / "online_secondary_codebook_results.json")
    rows = _read_prediction_rows(directory / "session2_predictions.csv")
    expected_profile, expected_segmentation = EXPECTED_RUNS[key]
    audit = result.get("input_audit", {})
    observed_identity = (
        str(audit.get("encoder_ablation_profile")),
        str(audit.get("primitive_segmentation")),
    )
    if observed_identity != (expected_profile, expected_segmentation):
        raise RuntimeError(
            f"{key} identity mismatch: {observed_identity} != "
            f"{(expected_profile, expected_segmentation)}."
        )
    manifest = result.get("session_manifest_audit", {})
    if not bool(manifest.get("registered_protocol_verified")):
        raise RuntimeError(f"{key} did not verify the registered Session-2 protocol.")
    if int(manifest.get("future_feature_trial_count", -1)) != 0:
        raise RuntimeError(f"{key} includes future-session feature trials.")
    firewall = result.get("label_firewall", {})
    if not bool(firewall.get("metadata_aware_repository_created_after_predictions")):
        raise RuntimeError(f"{key} lacks the post-freeze metadata firewall audit.")
    implementation = audit.get("experiment_implementation_fingerprint", {})
    if not implementation.get("runner_sha256") or not implementation.get(
        "secondary_codebook_helper_sha256"
    ):
        raise RuntimeError(f"{key} lacks runner/helper implementation hashes.")

    trial_ids = np.asarray([int(row["trial_global_id"]) for row in rows], dtype=np.int64)
    subjects = np.asarray([int(row["subject_id"]) for row in rows], dtype=np.int64)
    truth = np.asarray(
        [int(row["activity_label_0based"]) for row in rows], dtype=np.int64
    )
    if len(np.unique(trial_ids)) != len(trial_ids):
        raise RuntimeError(f"{key} prediction CSV repeats trial ids.")
    expected_test_ids = np.asarray(
        manifest.get("session_2_test_trial_ids", []), dtype=np.int64
    )
    if not np.array_equal(trial_ids, expected_test_ids):
        raise RuntimeError(f"{key} prediction order differs from its test manifest.")
    predictions = {}
    for arm in ARMS:
        column = f"{arm}_aligned_prediction"
        if any(column not in row for row in rows):
            raise RuntimeError(f"{key} prediction CSV lacks {column}.")
        values = np.asarray([int(row[column]) for row in rows], dtype=np.int64)
        reported = result.get("arms", {}).get(arm, {}).get("global_metrics", {})
        json_values = np.asarray(reported.get("aligned_predictions", []), dtype=np.int64)
        if not np.array_equal(values, json_values):
            raise RuntimeError(f"{key}/{arm} CSV and JSON predictions disagree.")
        correct = values == truth
        observed_metrics = {
            "all_accuracy": float(np.mean(correct)),
            "old_accuracy": float(np.mean(correct[truth < 6])),
            "new_accuracy": float(np.mean(correct[truth >= 6])),
        }
        for metric, observed in observed_metrics.items():
            if not np.isclose(observed, float(reported[metric]), atol=1e-12):
                raise RuntimeError(
                    f"{key}/{arm}/{metric} does not reproduce the reported value."
                )
        predictions[arm] = values
    return FrozenRun(
        key=key,
        directory=directory,
        result=result,
        trial_ids=trial_ids,
        subjects=subjects,
        truth=truth,
        aligned_predictions=predictions,
    )


def validate_paired_runs(runs: Mapping[str, FrozenRun]) -> dict:
    if set(runs) != set(EXPECTED_RUNS):
        raise RuntimeError(
            f"Four registered runs are required: {sorted(EXPECTED_RUNS)}."
        )
    reference = runs["A0_fixed"]
    reference_manifest = reference.directory / "session_manifest.csv"
    manifest_bytes = reference_manifest.read_bytes()
    reference_implementation = reference.result["input_audit"][
        "experiment_implementation_fingerprint"
    ]
    result_hashes = {}
    for key, run in runs.items():
        if not np.array_equal(run.trial_ids, reference.trial_ids):
            raise RuntimeError(f"{key} test trial ids/order are not paired.")
        if not np.array_equal(run.subjects, reference.subjects):
            raise RuntimeError(f"{key} subject ids are not paired.")
        if not np.array_equal(run.truth, reference.truth):
            raise RuntimeError(f"{key} ground truth is not paired.")
        if (run.directory / "session_manifest.csv").read_bytes() != manifest_bytes:
            raise RuntimeError(f"{key} session_manifest.csv differs byte-for-byte.")
        npz_sha = run.result.get("input_audit", {}).get("npz_sha256")
        reference_sha = reference.result.get("input_audit", {}).get("npz_sha256")
        if npz_sha != reference_sha:
            raise RuntimeError(f"{key} uses a different source NPZ.")
        implementation = run.result["input_audit"][
            "experiment_implementation_fingerprint"
        ]
        for field in ("runner_sha256", "secondary_codebook_helper_sha256"):
            if implementation.get(field) != reference_implementation.get(field):
                raise RuntimeError(f"{key} uses a different {field}.")
        result_hashes[key] = _sha256_file(
            run.directory / "online_secondary_codebook_results.json"
        )
    observed_subjects = sorted(set(reference.subjects.tolist()))
    observed_classes = sorted(set(reference.truth.tolist()))
    if observed_subjects != [4, 5] or observed_classes != list(range(10)):
        raise RuntimeError(
            "Registered fold06 confirmation requires subjects [4,5] and classes 0..9; "
            f"got subjects={observed_subjects}, classes={observed_classes}."
        )
    return {
        "paired_test_trial_count": int(len(reference.trial_ids)),
        "paired_subject_ids": observed_subjects,
        "paired_activity_ids": observed_classes,
        "session_manifest_byte_identical": True,
        "npz_sha256_identical": True,
        "runner_and_helper_sha256_identical": True,
        "input_result_sha256": result_hashes,
    }


def stratified_bootstrap_indices(
    subjects: Sequence[int],
    labels: Sequence[int],
    replicates: int,
    seed: int,
) -> np.ndarray:
    subjects = np.asarray(subjects, dtype=np.int64)
    labels = np.asarray(labels, dtype=np.int64)
    if subjects.ndim != 1 or labels.shape != subjects.shape or len(subjects) == 0:
        raise ValueError("subjects/labels must be equal non-empty 1D arrays.")
    if int(replicates) < 1:
        raise ValueError("bootstrap replicates must be positive.")
    strata = [
        np.flatnonzero((subjects == subject) & (labels == label))
        for subject in sorted(set(subjects.tolist()))
        for label in sorted(set(labels.tolist()))
        if np.any((subjects == subject) & (labels == label))
    ]
    if any(len(indices) == 0 for indices in strata):
        raise RuntimeError("An empty bootstrap stratum was constructed.")
    rng = np.random.default_rng(int(seed))
    samples = np.empty((int(replicates), len(subjects)), dtype=np.int64)
    for replicate in range(int(replicates)):
        cursor = 0
        for indices in strata:
            selected = rng.choice(indices, size=len(indices), replace=True)
            samples[replicate, cursor : cursor + len(indices)] = selected
            cursor += len(indices)
        if cursor != len(subjects):
            raise RuntimeError("Bootstrap strata did not preserve the trial count.")
    return samples


def _metric_mask(truth: np.ndarray, metric: str) -> np.ndarray:
    if metric == "all_accuracy":
        return np.ones(len(truth), dtype=bool)
    if metric == "old_accuracy":
        return truth < 6
    if metric == "new_accuracy":
        return truth >= 6
    raise ValueError(f"Unknown metric {metric!r}.")


def paired_bootstrap_effect(
    truth: Sequence[int],
    left_predictions: Sequence[int],
    right_predictions: Sequence[int],
    bootstrap_indices: np.ndarray,
    metric: str,
) -> dict:
    """Return left-minus-right accuracy with full-test mappings held fixed."""

    truth = np.asarray(truth, dtype=np.int64)
    left = np.asarray(left_predictions, dtype=np.int64)
    right = np.asarray(right_predictions, dtype=np.int64)
    samples = np.asarray(bootstrap_indices, dtype=np.int64)
    if left.shape != truth.shape or right.shape != truth.shape:
        raise ValueError("Paired predictions and truth must have the same shape.")
    if samples.ndim != 2 or samples.shape[1] != len(truth):
        raise ValueError("bootstrap_indices must have shape [B,N].")
    mask = _metric_mask(truth, metric)
    trial_delta = (left == truth).astype(np.float64) - (
        right == truth
    ).astype(np.float64)
    selected_mask = mask[samples]
    numerator = np.sum(trial_delta[samples] * selected_mask, axis=1)
    denominator = np.sum(selected_mask, axis=1)
    if np.any(denominator <= 0):
        raise RuntimeError(f"A bootstrap sample contains no trials for {metric}.")
    bootstrap = numerator / denominator
    point = float(np.mean(trial_delta[mask]))
    lower, upper = np.percentile(bootstrap, [2.5, 97.5]).tolist()
    return {
        "metric": metric,
        "left_minus_right": point,
        "ci95_percentile": [float(lower), float(upper)],
        "bootstrap_mean": float(np.mean(bootstrap)),
        "bootstrap_fraction_positive": float(np.mean(bootstrap > 0.0)),
        "bootstrap_fraction_nonpositive": float(np.mean(bootstrap <= 0.0)),
        "evaluated_trial_count": int(np.sum(mask)),
        "mapping_policy": "each run's full-test global Hungarian mapping held fixed",
    }


def _comparisons(runs: Mapping[str, FrozenRun]) -> list[tuple[str, str, str, str]]:
    comparisons: list[tuple[str, str, str, str]] = []
    for segmentation in ("fixed", "changepoint"):
        for arm in ARMS:
            comparisons.append(
                (
                    "encoder_effect",
                    f"A3_minus_A0/{segmentation}/{arm}",
                    f"A3_{segmentation}",
                    f"A0_{segmentation}",
                )
            )
    for profile in ("A0", "A3"):
        for arm in ARMS:
            comparisons.append(
                (
                    "segmentation_effect",
                    f"changepoint_minus_fixed/{profile}/{arm}",
                    f"{profile}_changepoint",
                    f"{profile}_fixed",
                )
            )
    for key in EXPECTED_RUNS:
        for arm in ARMS[1:]:
            comparisons.append(
                (
                    "secondary_split_effect",
                    f"{arm}_minus_U0/{key}",
                    key,
                    key,
                )
            )
    return comparisons


def analyze(
    runs: Mapping[str, FrozenRun],
    replicates: int,
    seed: int,
) -> dict:
    pairing_audit = validate_paired_runs(runs)
    reference = runs["A0_fixed"]
    samples = stratified_bootstrap_indices(
        reference.subjects, reference.truth, replicates=replicates, seed=seed
    )
    effects = []
    for family, name, left_key, right_key in _comparisons(runs):
        if family == "secondary_split_effect":
            left_arm = name.split("_minus_U0/", 1)[0]
            right_arm = "U0_coarse"
        else:
            left_arm = right_arm = name.rsplit("/", 1)[-1]
        left_run, right_run = runs[left_key], runs[right_key]
        for metric in METRICS:
            effect = paired_bootstrap_effect(
                reference.truth,
                left_run.aligned_predictions[left_arm],
                right_run.aligned_predictions[right_arm],
                samples,
                metric,
            )
            effect.update(
                {
                    "family": family,
                    "comparison": name,
                    "left_run": left_key,
                    "left_arm": left_arm,
                    "right_run": right_key,
                    "right_arm": right_arm,
                }
            )
            effects.append(effect)
    return {
        "protocol": {
            "name": "fold06_session2_paired_trial_bootstrap_v1",
            "replicates": int(replicates),
            "seed": int(seed),
            "resampling_unit": "trial",
            "strata": "subject_id x activity_label",
            "pairing": "same resampled trial indices for both frozen predictions",
            "hungarian_policy": (
                "do not re-estimate mappings inside bootstrap samples; reuse each "
                "run's single mapping estimated on the complete Session-2 test set"
            ),
            "scope_caveat": (
                "conditional trial-level stability for held-out S4/S5 only; not an "
                "unseen-subject generalisation confidence interval"
            ),
        },
        "pairing_audit": pairing_audit,
        "effects": effects,
    }


def _write_effects_csv(path: Path, effects: Sequence[dict]) -> None:
    rows = []
    for effect in effects:
        row = dict(effect)
        row["ci95_low"] = float(effect["ci95_percentile"][0])
        row["ci95_high"] = float(effect["ci95_percentile"][1])
        del row["ci95_percentile"]
        rows.append(row)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _save_effect_plot(path: Path, effects: Sequence[dict]) -> None:
    selected = [
        effect
        for effect in effects
        if effect["family"] == "encoder_effect"
        and effect["metric"] in ("all_accuracy", "new_accuracy")
    ]
    labels = [
        f"{effect['comparison']} [{effect['metric'].replace('_accuracy', '')}]"
        for effect in selected
    ]
    point = np.asarray([effect["left_minus_right"] for effect in selected]) * 100.0
    lower = np.asarray([effect["ci95_percentile"][0] for effect in selected]) * 100.0
    upper = np.asarray([effect["ci95_percentile"][1] for effect in selected]) * 100.0
    y = np.arange(len(selected))
    fig, axis = plt.subplots(figsize=(12, max(7.0, 0.48 * len(selected))))
    axis.errorbar(
        point,
        y,
        xerr=np.vstack([point - lower, upper - point]),
        fmt="o",
        color="navy",
        ecolor="steelblue",
        capsize=3,
    )
    axis.axvline(0.0, color="black", linewidth=1, linestyle="--")
    axis.set_yticks(y, labels, fontsize=8)
    axis.invert_yaxis()
    axis.set_xlabel("A3 - A0 accuracy (percentage points)")
    axis.set_title(
        "Fold06 Session-2 paired trial bootstrap\n"
        "95% intervals conditional on held-out subjects S4/S5"
    )
    axis.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pair four frozen Session-2 runs and bootstrap accuracy differences."
    )
    parser.add_argument("--a0-fixed-dir", required=True)
    parser.add_argument("--a3-fixed-dir", required=True)
    parser.add_argument("--a0-changepoint-dir", required=True)
    parser.add_argument("--a3-changepoint-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--bootstrap-replicates", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260903)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = _require_new_output_dir(Path(args.output_dir))
    directories = {
        "A0_fixed": Path(args.a0_fixed_dir),
        "A3_fixed": Path(args.a3_fixed_dir),
        "A0_changepoint": Path(args.a0_changepoint_dir),
        "A3_changepoint": Path(args.a3_changepoint_dir),
    }
    runs = {key: load_frozen_run(key, value) for key, value in directories.items()}
    result = analyze(
        runs,
        replicates=int(args.bootstrap_replicates),
        seed=int(args.seed),
    )
    result["arguments"] = {
        "directories": {key: str(value.resolve()) for key, value in directories.items()},
        "output_dir": str(output_dir),
    }
    result["analysis_implementation"] = {
        "path": str(Path(__file__).resolve()),
        "sha256": _sha256_file(Path(__file__).resolve()),
    }
    result["generated_files"] = [
        "paired_trial_bootstrap.json",
        "paired_trial_bootstrap.csv",
        "paired_encoder_effects.png",
    ]
    output_dir.mkdir(parents=True)
    (output_dir / "paired_trial_bootstrap.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    _write_effects_csv(output_dir / "paired_trial_bootstrap.csv", result["effects"])
    _save_effect_plot(output_dir / "paired_encoder_effects.png", result["effects"])
    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "pairing_audit": result["pairing_audit"],
                "encoder_effects": [
                    effect
                    for effect in result["effects"]
                    if effect["family"] == "encoder_effect"
                ],
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
