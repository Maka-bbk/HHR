"""Paired four-run analysis for the Session-2 hierarchical-gate ablation."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.motion_primitive.analyze_online_secondary_confirmations import (
    paired_bootstrap_effect,
    stratified_bootstrap_indices,
)
from experiments.motion_primitive.run_online_hierarchical_gate import (
    ARMS,
    REGISTERED_SEGMENTATION_PARAMETERS,
    REGISTERED_UPSTREAM_CODEBOOK,
)
from experiments.motion_primitive.trajectory_ablation import jsonable, sha256_file


METRICS = ("all_accuracy", "old_accuracy", "new_accuracy")
EXPECTED_RUNS = {
    "A0_fixed": ("A0", "fixed_window"),
    "A3_fixed": ("A3", "fixed_window"),
    "A0_changepoint": ("A0", "motion_encoder_changepoint"),
    "A3_changepoint": ("A3", "motion_encoder_changepoint"),
}
REGISTERED_SEED = 500
REGISTERED_GATE_PARAMETERS = {
    "static_radius_quantile": 0.95,
    "minimum_token_fraction": 0.50,
    "minimum_motion_energy_ratio": 2.0,
    "minimum_motion_energy_gap": 0.25,
}
CURRENT_IMPLEMENTATION_PATHS = {
    "runner_sha256": PROJECT_ROOT
    / "experiments"
    / "motion_primitive"
    / "run_online_hierarchical_gate.py",
    "hierarchical_helper_sha256": PROJECT_ROOT
    / "experiments"
    / "motion_primitive"
    / "hierarchical_gate.py",
    "frozen_secondary_helper_sha256": PROJECT_ROOT
    / "experiments"
    / "motion_primitive"
    / "online_secondary_codebook.py",
    "frozen_secondary_runner_sha256": PROJECT_ROOT
    / "experiments"
    / "motion_primitive"
    / "run_online_secondary_codebook.py",
}


def _is_sha256(value) -> bool:
    digest = str(value or "")
    return len(digest) == 64 and all(
        character in "0123456789abcdefABCDEF" for character in digest
    )


def _finite_float(mapping: Mapping, field: str, context: str) -> float:
    try:
        value = float(mapping[field])
    except (KeyError, TypeError, ValueError) as error:
        raise RuntimeError(f"{context} lacks a valid {field}.") from error
    if not np.isfinite(value):
        raise RuntimeError(f"{context} {field} must be finite.")
    return value


def _validate_gate_status(
    key: str,
    gate_fit: Mapping,
    minimum_ratio: float,
    minimum_gap: float,
    expected_fit_trial_count: int,
) -> None:
    enabled = gate_fit.get("gate_enabled")
    if type(enabled) is not bool:
        raise RuntimeError(f"{key} gate_enabled must be a JSON boolean.")
    reason = gate_fit.get("gate_disable_reason")
    if enabled and reason is not None:
        raise RuntimeError(f"{key} enabled gate must have null gate_disable_reason.")
    if not enabled and (not isinstance(reason, str) or not reason.strip()):
        raise RuntimeError(f"{key} disabled gate must report gate_disable_reason.")
    ratio = _finite_float(gate_fit, "motion_energy_ratio", f"{key} gate fit")
    gap = _finite_float(gate_fit, "motion_energy_gap", f"{key} gate fit")
    if ratio < 1.0 or gap < 0.0:
        raise RuntimeError(f"{key} gate motion-energy diagnostics are invalid.")
    medians = np.asarray(
        gate_fit.get("residual_child_normalized_motion_component_medians", []),
        dtype=np.float64,
    )
    if medians.shape != (2, 2) or not np.all(np.isfinite(medians)):
        raise RuntimeError(
            f"{key} must report finite 2x2 residual-child motion-component medians."
        )
    try:
        static_child = int(gate_fit["static_residual_child"])
        fit_count = int(gate_fit["fit_trial_count"])
        confident_count = int(gate_fit["confident_static_fit_count"])
    except (KeyError, TypeError, ValueError) as error:
        raise RuntimeError(f"{key} gate count/static-child diagnostics are invalid.") from error
    if static_child not in (0, 1):
        raise RuntimeError(f"{key} static_residual_child must be 0 or 1.")
    if fit_count != int(expected_fit_trial_count) or confident_count < 0:
        raise RuntimeError(f"{key} gate fit counts disagree with the frozen fit set.")
    dynamic_child = 1 - static_child
    components_agree = bool(
        np.all(medians[static_child] <= medians[dynamic_child] + 1e-12)
    )
    ratio_passes = ratio + 1e-12 >= float(minimum_ratio)
    gap_passes = gap + 1e-12 >= float(minimum_gap)
    if enabled:
        if not ratio_passes or not gap_passes or not components_agree:
            raise RuntimeError(f"{key} enabled a hierarchy that fails its gate criteria.")
        if confident_count < 2:
            raise RuntimeError(f"{key} enabled a hierarchy with fewer than two static fits.")
    else:
        if not ratio_passes and "motion_energy_ratio_below_threshold" not in reason:
            raise RuntimeError(f"{key} gate_disable_reason omits the failed energy ratio.")
        if not gap_passes and "motion_energy_gap_below_threshold" not in reason:
            raise RuntimeError(f"{key} gate_disable_reason omits the failed energy gap.")
        if not components_agree and "motion_component_directions_disagree" not in reason:
            raise RuntimeError(f"{key} gate_disable_reason omits component disagreement.")


@dataclass(frozen=True)
class FrozenGateRun:
    key: str
    directory: Path
    result: dict
    trial_ids: np.ndarray
    subjects: np.ndarray
    truth: np.ndarray
    aligned_predictions: Mapping[str, np.ndarray]


def _read_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise RuntimeError(f"Prediction CSV is empty: {path}")
    return rows


def _read_result(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def load_frozen_gate_run(key: str, directory: Path) -> FrozenGateRun:
    if key not in EXPECTED_RUNS:
        raise ValueError(f"Unknown registered run key {key!r}.")
    directory = Path(directory).resolve()
    result = _read_result(directory / "online_hierarchical_gate_results.json")
    rows = _read_rows(directory / "session2_predictions.csv")
    expected_profile, expected_segmentation = EXPECTED_RUNS[key]
    audit = result.get("input_audit", {})
    identity = (
        str(audit.get("encoder_ablation_profile")),
        str(audit.get("primitive_segmentation")),
    )
    if identity != (expected_profile, expected_segmentation):
        raise RuntimeError(
            f"{key} identity mismatch: {identity} != "
            f"{(expected_profile, expected_segmentation)}."
        )
    manifest = result.get("session_manifest_audit", {})
    if not bool(manifest.get("registered_protocol_verified")):
        raise RuntimeError(f"{key} did not verify the registered Session-2 protocol.")
    if int(manifest.get("future_feature_trial_count", -1)) != 0:
        raise RuntimeError(f"{key} includes future-session feature trials.")
    session_one_ids = [
        int(value) for value in manifest.get("session_1_train_trial_ids", [])
    ]
    session_two_ids = [
        int(value) for value in manifest.get("session_2_train_trial_ids", [])
    ]
    cumulative_ids = [
        int(value) for value in manifest.get("cumulative_train_trial_ids", [])
    ]
    test_manifest_ids = [
        int(value) for value in manifest.get("session_2_test_trial_ids", [])
    ]
    if (
        not session_one_ids
        or not session_two_ids
        or len(cumulative_ids) != len(set(cumulative_ids))
        or set(cumulative_ids) != set(session_one_ids) | set(session_two_ids)
    ):
        raise RuntimeError(f"{key} cumulative training manifest is inconsistent.")
    manifest_rows = _read_rows(directory / "session_manifest.csv")
    observed_manifest_rows = [
        (
            str(row.get("split", "")),
            int(row.get("session", -1)),
            int(row.get("trial_global_id", -1)),
        )
        for row in manifest_rows
    ]
    expected_manifest_rows = [
        (split_name, session, trial_id)
        for split_name, session, ids in (
            ("online_train", 1, session_one_ids),
            ("online_train", 2, session_two_ids),
            ("online_test", 2, test_manifest_ids),
        )
        for trial_id in ids
    ]
    if observed_manifest_rows != expected_manifest_rows:
        raise RuntimeError(f"{key} session_manifest.csv disagrees with its JSON audit.")
    firewall = result.get("label_firewall", {})
    if not bool(firewall.get("metadata_aware_repository_created_after_predictions")):
        raise RuntimeError(f"{key} lacks the post-freeze metadata firewall audit.")
    if bool(firewall.get("gate_fit_uses_activity_labels_or_names", True)):
        raise RuntimeError(f"{key} reports label use during gate fitting.")
    if bool(firewall.get("test_motion_energy_used_for_gate_assignment", True)):
        raise RuntimeError(f"{key} used test motion energy in gate assignment.")
    upstream = audit.get("registered_upstream")
    if not isinstance(upstream, Mapping):
        raise RuntimeError(f"{key} lacks the registered upstream audit.")
    if upstream.get("registered_upstream_codebook") != REGISTERED_UPSTREAM_CODEBOOK:
        raise RuntimeError(f"{key} upstream K32/PCA64 protocol is not registered.")
    if (
        upstream.get("registered_segmentation_parameters")
        != REGISTERED_SEGMENTATION_PARAMETERS
        or upstream.get("observed_segmentation_parameters")
        != REGISTERED_SEGMENTATION_PARAMETERS
    ):
        raise RuntimeError(f"{key} upstream segmentation parameters drifted.")
    if upstream.get("artifact_center_shape") != [32, 64] or int(
        upstream.get("artifact_embedding_dim", -1)
    ) != 64:
        raise RuntimeError(f"{key} upstream artifact is not K32/PCA64.")
    if upstream.get("checkpoint_smoke_test") is not False or upstream.get(
        "outer_test_used_during_encoder_training"
    ) is not False:
        raise RuntimeError(f"{key} upstream encoder provenance is unsafe.")
    for field in (
        "encoder_implementation_combined_sha256",
        "encoder_source_checkpoint_sha256",
        "motion_encoder_checkpoint_sha256",
    ):
        if not _is_sha256(upstream.get(field)):
            raise RuntimeError(f"{key} lacks a valid upstream {field}.")
    if upstream["motion_encoder_checkpoint_sha256"] != audit.get(
        "checkpoint_sha256"
    ):
        raise RuntimeError(f"{key} upstream/final checkpoint hashes disagree.")
    if not _is_sha256(audit.get("npz_sha256")):
        raise RuntimeError(f"{key} lacks a valid source NPZ SHA-256 audit.")
    implementation = audit.get("implementation_fingerprint", {})
    for field in (
        "runner_sha256",
        "hierarchical_helper_sha256",
        "frozen_secondary_helper_sha256",
        "frozen_secondary_runner_sha256",
    ):
        if not implementation.get(field):
            raise RuntimeError(f"{key} lacks {field}.")
        current = sha256_file(CURRENT_IMPLEMENTATION_PATHS[field])
        if implementation[field] != current:
            raise RuntimeError(
                f"{key} {field} does not match the current frozen implementation."
            )
    arguments = result.get("arguments", {})
    if int(arguments.get("seed", -1)) != REGISTERED_SEED:
        raise RuntimeError(f"{key} does not use registered seed {REGISTERED_SEED}.")
    if int(manifest.get("seed", -1)) != REGISTERED_SEED:
        raise RuntimeError(f"{key} manifest does not use registered seed {REGISTERED_SEED}.")
    for field, expected in REGISTERED_GATE_PARAMETERS.items():
        observed = _finite_float(arguments, field, key)
        if observed != float(expected):
            raise RuntimeError(
                f"{key} {field}={observed!r}, expected registered {expected}."
            )
    candidate_ids = [int(value) for value in result.get("candidate", {}).get("dominant_trial_ids", [])]
    fit_ids = [int(value) for value in result.get("gate_fit", {}).get("fit_trial_ids", [])]
    if not candidate_ids or fit_ids != candidate_ids:
        raise RuntimeError(f"{key} gate fit ids do not match dominant train candidates.")
    cumulative_train = set(
        int(value) for value in manifest.get("cumulative_train_trial_ids", [])
    )
    if not set(fit_ids).issubset(cumulative_train):
        raise RuntimeError(f"{key} gate fit includes a non-train trial.")
    gate_fit = result.get("gate_fit", {})
    if not isinstance(gate_fit, Mapping):
        raise RuntimeError(f"{key} lacks gate-fit diagnostics.")
    _validate_gate_status(
        key,
        gate_fit,
        minimum_ratio=float(arguments["minimum_motion_energy_ratio"]),
        minimum_gap=float(arguments["minimum_motion_energy_gap"]),
        expected_fit_trial_count=len(fit_ids),
    )
    for field in REGISTERED_GATE_PARAMETERS:
        fit_value = _finite_float(gate_fit, field, f"{key} gate fit")
        if fit_value != float(arguments[field]):
            raise RuntimeError(f"{key} gate-fit {field} disagrees with arguments.")

    trial_ids = np.asarray([int(row["trial_global_id"]) for row in rows], dtype=np.int64)
    subjects = np.asarray([int(row["subject_id"]) for row in rows], dtype=np.int64)
    truth = np.asarray(
        [int(row["activity_label_0based"]) for row in rows], dtype=np.int64
    )
    expected_ids = np.asarray(manifest.get("session_2_test_trial_ids", []), dtype=np.int64)
    if not np.array_equal(trial_ids, expected_ids):
        raise RuntimeError(f"{key} prediction order differs from its test manifest.")
    if len(np.unique(trial_ids)) != len(trial_ids):
        raise RuntimeError(f"{key} repeats a test trial id.")
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
        reproduced = {
            "all_accuracy": float(np.mean(correct)),
            "old_accuracy": float(np.mean(correct[truth < 6])),
            "new_accuracy": float(np.mean(correct[truth >= 6])),
        }
        for metric, value in reproduced.items():
            if not np.isclose(value, float(reported[metric]), atol=1e-12):
                raise RuntimeError(f"{key}/{arm}/{metric} cannot be reproduced.")
        predictions[arm] = values
    if gate_fit["gate_enabled"] is False:
        for suffix in ("raw_cluster", "aligned_prediction"):
            coarse = np.asarray(
                [int(row[f"G0_coarse_{suffix}"]) for row in rows], dtype=np.int64
            )
            gated = np.asarray(
                [int(row[f"G3_hierarchical_{suffix}"]) for row in rows], dtype=np.int64
            )
            if not np.array_equal(coarse, gated):
                raise RuntimeError(
                    f"{key} disabled hierarchy does not reproduce G0 {suffix}."
                )
    return FrozenGateRun(
        key=key,
        directory=directory,
        result=result,
        trial_ids=trial_ids,
        subjects=subjects,
        truth=truth,
        aligned_predictions=predictions,
    )


def validate_paired_gate_runs(runs: Mapping[str, FrozenGateRun]) -> dict:
    if set(runs) != set(EXPECTED_RUNS):
        raise RuntimeError(f"Four registered runs are required: {sorted(EXPECTED_RUNS)}.")
    reference = runs["A0_fixed"]
    reference_manifest = (reference.directory / "session_manifest.csv").read_bytes()
    reference_npz = reference.result["input_audit"]["npz_sha256"]
    reference_implementation = reference.result["input_audit"]["implementation_fingerprint"]
    reference_upstream = reference.result["input_audit"]["registered_upstream"]
    reference_encoder_implementation = reference_upstream[
        "encoder_implementation_combined_sha256"
    ]
    reference_source_checkpoint = reference_upstream[
        "encoder_source_checkpoint_sha256"
    ]
    reference_manifest_audit = reference.result.get("session_manifest_audit", {})
    reference_manifest_ids = {
        field: reference_manifest_audit.get(field)
        for field in (
            "session_1_train_trial_ids",
            "session_2_train_trial_ids",
            "cumulative_train_trial_ids",
            "session_2_test_trial_ids",
        )
    }
    if any(value is None for value in reference_manifest_ids.values()):
        raise RuntimeError("Reference run lacks the JSON session manifest ids.")
    reference_parameters = {
        field: float(reference.result["arguments"][field])
        for field in REGISTERED_GATE_PARAMETERS
    }
    hashes = {}
    for key, run in runs.items():
        if not np.array_equal(run.trial_ids, reference.trial_ids):
            raise RuntimeError(f"{key} test trial ids/order are not paired.")
        if not np.array_equal(run.subjects, reference.subjects):
            raise RuntimeError(f"{key} subject ids are not paired.")
        if not np.array_equal(run.truth, reference.truth):
            raise RuntimeError(f"{key} ground truth is not paired.")
        if (run.directory / "session_manifest.csv").read_bytes() != reference_manifest:
            raise RuntimeError(f"{key} session_manifest.csv differs byte-for-byte.")
        manifest_ids = {
            field: run.result.get("session_manifest_audit", {}).get(field)
            for field in reference_manifest_ids
        }
        if manifest_ids != reference_manifest_ids:
            raise RuntimeError(f"{key} JSON session manifest ids are not paired.")
        audit = run.result["input_audit"]
        if audit.get("npz_sha256") != reference_npz:
            raise RuntimeError(f"{key} uses a different source NPZ.")
        implementation = audit["implementation_fingerprint"]
        for field in (
            "runner_sha256",
            "hierarchical_helper_sha256",
            "frozen_secondary_helper_sha256",
            "frozen_secondary_runner_sha256",
        ):
            if implementation.get(field) != reference_implementation.get(field):
                raise RuntimeError(f"{key} uses a different {field}.")
        upstream = audit["registered_upstream"]
        if (
            upstream["encoder_implementation_combined_sha256"]
            != reference_encoder_implementation
        ):
            raise RuntimeError(f"{key} uses a different encoder implementation.")
        if upstream["encoder_source_checkpoint_sha256"] != reference_source_checkpoint:
            raise RuntimeError(f"{key} uses a different legacy source checkpoint.")
        if upstream["registered_upstream_codebook"] != reference_upstream[
            "registered_upstream_codebook"
        ]:
            raise RuntimeError(f"{key} uses a different upstream codebook protocol.")
        if upstream["observed_segmentation_parameters"] != reference_upstream[
            "observed_segmentation_parameters"
        ]:
            raise RuntimeError(f"{key} uses different segmentation parameters.")
        for field, reference_value in reference_parameters.items():
            observed = float(run.result["arguments"][field])
            if observed != reference_value:
                raise RuntimeError(f"{key} uses a different {field}.")
        hashes[key] = sha256_file(
            run.directory / "online_hierarchical_gate_results.json"
        )
    if sorted(set(reference.subjects.tolist())) != [4, 5]:
        raise RuntimeError("The registered fold06 analysis requires Subjects 4 and 5.")
    if sorted(set(reference.truth.tolist())) != list(range(10)):
        raise RuntimeError("The registered Session-2 analysis requires classes 0..9.")
    audit = {
        "paired_test_trial_count": int(len(reference.trial_ids)),
        "paired_subject_ids": [4, 5],
        "paired_activity_ids": list(range(10)),
        "session_manifest_byte_identical": True,
        "session_manifest_json_ids_identical": True,
        "source_npz_sha256_identical": True,
        "implementation_sha256_identical": True,
        "encoder_implementation_sha256_identical": True,
        "encoder_source_checkpoint_sha256_identical": True,
        "upstream_codebook_protocol_identical": True,
        "segmentation_parameters_identical": True,
        "gate_parameters_identical": True,
        "registered_gate_parameters": reference_parameters,
        "input_result_sha256": hashes,
    }
    profile_checkpoints = {}
    for profile in ("A0", "A3"):
        fixed = runs[f"{profile}_fixed"]
        changepoint = runs[f"{profile}_changepoint"]
        fixed_checkpoint = fixed.result["input_audit"].get("checkpoint_sha256")
        changepoint_checkpoint = changepoint.result["input_audit"].get(
            "checkpoint_sha256"
        )
        if not fixed_checkpoint or fixed_checkpoint != changepoint_checkpoint:
            raise RuntimeError(
                f"{profile} fixed/changepoint do not share one frozen checkpoint."
            )
        profile_checkpoints[profile] = fixed_checkpoint
    if profile_checkpoints["A0"] == profile_checkpoints["A3"]:
        raise RuntimeError("A0 and A3 unexpectedly use the same motion-encoder checkpoint.")
    audit["fixed_changepoint_checkpoint_sha256_identical_within_profile"] = True
    audit["a0_a3_motion_encoder_checkpoints_distinct"] = True
    audit["motion_encoder_checkpoint_sha256_by_profile"] = profile_checkpoints
    audit["registered_seed"] = REGISTERED_SEED
    audit["gate_status_by_run"] = {
        key: {
            field: runs[key].result["gate_fit"].get(field)
            for field in (
                "gate_enabled",
                "gate_disable_reason",
                "motion_energy_ratio",
                "motion_energy_gap",
                "static_residual_child",
                "fit_trial_count",
                "confident_static_fit_count",
                "residual_child_normalized_motion_component_medians",
            )
        }
        for key in EXPECTED_RUNS
    }
    return audit


def _comparisons() -> list[tuple[str, str, str, str, str, str]]:
    result = []
    for segmentation in ("fixed", "changepoint"):
        for arm in ARMS:
            result.append(
                (
                    "encoder_effect",
                    f"A3_minus_A0/{segmentation}/{arm}",
                    f"A3_{segmentation}",
                    arm,
                    f"A0_{segmentation}",
                    arm,
                )
            )
    for profile in ("A0", "A3"):
        for arm in ARMS:
            result.append(
                (
                    "segmentation_effect",
                    f"changepoint_minus_fixed/{profile}/{arm}",
                    f"{profile}_changepoint",
                    arm,
                    f"{profile}_fixed",
                    arm,
                )
            )
    for run_key in EXPECTED_RUNS:
        for arm in ARMS[1:]:
            result.append(
                (
                    "refinement_effect",
                    f"{arm}_minus_G0/{run_key}",
                    run_key,
                    arm,
                    run_key,
                    "G0_coarse",
                )
            )
        for baseline in ("G1_residual_flat", "G2_gravity_flat"):
            result.append(
                (
                    "hierarchy_effect",
                    f"G3_minus_{baseline}/{run_key}",
                    run_key,
                    "G3_hierarchical",
                    run_key,
                    baseline,
                )
            )
    return result


def analyze_gate_runs(
    runs: Mapping[str, FrozenGateRun], replicates: int, seed: int
) -> dict:
    audit = validate_paired_gate_runs(runs)
    reference = runs["A0_fixed"]
    samples = stratified_bootstrap_indices(
        reference.subjects, reference.truth, replicates=int(replicates), seed=int(seed)
    )
    effects = []
    for family, name, left_run, left_arm, right_run, right_arm in _comparisons():
        for metric in METRICS:
            summary = paired_bootstrap_effect(
                reference.truth,
                runs[left_run].aligned_predictions[left_arm],
                runs[right_run].aligned_predictions[right_arm],
                samples,
                metric,
            )
            summary.update(
                {
                    "family": family,
                    "comparison": name,
                    "left_run": left_run,
                    "left_arm": left_arm,
                    "right_run": right_run,
                    "right_arm": right_arm,
                }
            )
            effects.append(summary)
    return {
        "protocol": {
            "name": "fold06_session2_hierarchical_gate_paired_analysis_v1",
            "replicates": int(replicates),
            "seed": int(seed),
            "resampling_unit": "trial",
            "stratification": "subject_id x activity_label",
            "hungarian_policy": (
                "reuse each run/arm complete-test-set mapping inside bootstrap"
            ),
            "multiplicity_adjustment": "none; single-fold exploratory screening",
            "generalisation_limit": (
                "conditional trial stability for Subjects 4/5, not unseen-subject CI"
            ),
        },
        "pairing_audit": audit,
        "effects": effects,
    }


def _write_effects(path: Path, effects: list[dict]) -> None:
    rows = []
    for effect in effects:
        rows.append(
            {
                key: effect[key]
                for key in (
                    "family",
                    "comparison",
                    "metric",
                    "left_run",
                    "left_arm",
                    "right_run",
                    "right_arm",
                    "left_minus_right",
                    "ci95_percentile",
                    "bootstrap_mean",
                    "bootstrap_fraction_positive",
                    "evaluated_trial_count",
                )
            }
        )
        rows[-1]["ci_lower"] = effect["ci95_percentile"][0]
        rows[-1]["ci_upper"] = effect["ci95_percentile"][1]
        del rows[-1]["ci95_percentile"]
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _save_primary_plot(path: Path, effects: list[dict]) -> None:
    wanted = [
        ("A3_minus_A0/fixed/G3_hierarchical", "A3-A0 fixed"),
        ("A3_minus_A0/changepoint/G3_hierarchical", "A3-A0 changepoint"),
        ("G3_hierarchical_minus_G0/A0_fixed", "G3-G0 A0 fixed"),
        ("G3_hierarchical_minus_G0/A3_fixed", "G3-G0 A3 fixed"),
        ("G3_hierarchical_minus_G0/A0_changepoint", "G3-G0 A0 changepoint"),
        ("G3_hierarchical_minus_G0/A3_changepoint", "G3-G0 A3 changepoint"),
        ("G3_minus_G2_gravity_flat/A3_changepoint", "G3-G2 A3 changepoint"),
    ]
    selected = {
        (effect["comparison"], effect["metric"]): effect for effect in effects
    }
    fig, axes = plt.subplots(1, 3, figsize=(18, 7), sharey=True)
    colors = {"all_accuracy": "#3366cc", "old_accuracy": "#dc3912", "new_accuracy": "#109618"}
    for axis, metric in zip(axes, METRICS):
        points = [selected[(name, metric)] for name, _ in wanted]
        values = np.asarray([point["left_minus_right"] for point in points]) * 100.0
        lower = np.asarray([point["ci95_percentile"][0] for point in points]) * 100.0
        upper = np.asarray([point["ci95_percentile"][1] for point in points]) * 100.0
        positions = np.arange(len(points))
        axis.hlines(positions, lower, upper, color=colors[metric], linewidth=1.5)
        axis.scatter(values, positions, color=colors[metric], s=28, zorder=3)
        axis.axvline(0.0, color="black", linewidth=1, linestyle="--")
        axis.set_title(metric.replace("_", " "))
        axis.set_xlabel("left - right (percentage points)")
        axis.grid(axis="x", alpha=0.25)
        if axis is axes[0]:
            axis.set_yticks(positions)
            axis.set_yticklabels([label for _, label in wanted])
            axis.invert_yaxis()
        else:
            axis.tick_params(axis="y", labelleft=False)
    fig.suptitle("Hierarchical-gate paired trial bootstrap (95% percentile intervals)")
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _require_new_output_dir(path: Path) -> Path:
    resolved = path.resolve()
    if resolved.exists():
        raise FileExistsError(f"Output directory must not already exist: {resolved}")
    return resolved


def run(args: argparse.Namespace) -> dict:
    output_dir = _require_new_output_dir(Path(args.output_dir))
    runs = {
        "A0_fixed": load_frozen_gate_run("A0_fixed", Path(args.a0_fixed_dir)),
        "A3_fixed": load_frozen_gate_run("A3_fixed", Path(args.a3_fixed_dir)),
        "A0_changepoint": load_frozen_gate_run(
            "A0_changepoint", Path(args.a0_changepoint_dir)
        ),
        "A3_changepoint": load_frozen_gate_run(
            "A3_changepoint", Path(args.a3_changepoint_dir)
        ),
    }
    result = analyze_gate_runs(runs, replicates=int(args.bootstrap_replicates), seed=int(args.seed))
    result["arguments"] = {
        "output_dir": str(output_dir),
        "bootstrap_replicates": int(args.bootstrap_replicates),
        "seed": int(args.seed),
        "input_directories": {key: str(run.directory) for key, run in runs.items()},
    }
    output_dir.mkdir(parents=True)
    _write_effects(output_dir / "paired_hierarchical_gate_effects.csv", result["effects"])
    _save_primary_plot(
        output_dir / "paired_hierarchical_gate_effects.png", result["effects"]
    )
    result["implementation_fingerprint"] = {
        "analyzer_path": str(Path(__file__).resolve()),
        "analyzer_sha256": sha256_file(Path(__file__).resolve()),
        "bootstrap_helper_path": str(
            PROJECT_ROOT
            / "experiments"
            / "motion_primitive"
            / "analyze_online_secondary_confirmations.py"
        ),
        "bootstrap_helper_sha256": sha256_file(
            PROJECT_ROOT
            / "experiments"
            / "motion_primitive"
            / "analyze_online_secondary_confirmations.py"
        ),
    }
    result["generated_files"] = [
        "hierarchical_gate_confirmation.json",
        "paired_hierarchical_gate_effects.csv",
        "paired_hierarchical_gate_effects.png",
    ]
    (output_dir / "hierarchical_gate_confirmation.json").write_text(
        json.dumps(jsonable(result), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Paired analysis for four fold06 hierarchical-gate runs."
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
    if int(args.bootstrap_replicates) < 1:
        raise ValueError("bootstrap_replicates must be positive.")
    result = run(args)
    print(
        json.dumps(
            {
                "output_dir": result["arguments"]["output_dir"],
                "pairing_audit": result["pairing_audit"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
