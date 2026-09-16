"""Run the fold06 Session-2 residual-to-gravity hierarchical-gate ablation.

This remains an isolated trajectory-clustering proxy.  It does not modify or
train the formal Happy-CGCD online learner.  The purpose is to test whether the
A3 encoder residual can safely route only static occurrences to the gravity
split that separates Sitting from Standing.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import linear_sum_assignment

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.motion_primitive.core import run_length_encode  # noqa: E402
from experiments.motion_primitive.hierarchical_gate import (  # noqa: E402
    GATE_STATES,
    assign_hierarchical_gate,
    fit_hierarchical_gate,
    hierarchical_duration_histograms,
    hierarchical_gate_diagnostics,
    motion_components_from_token_partitions,
)
from experiments.motion_primitive.online_secondary_codebook import (  # noqa: E402
    assign_secondary_codebook,
    class_mean_histogram_distances,
    duration_histograms,
    fit_secondary_codebook,
    global_hungarian_metrics,
    posthoc_secondary_metrics,
    secondary_fit_diagnostics,
    select_dominant_token,
)
from experiments.motion_primitive.run_online_secondary_codebook import (  # noqa: E402
    LabelFreeSourceSignalRepository,
    _fit_trial_clusterer,
    _paired_bootstrap_comparisons,
    _positive_int,
    _require_new_output_dir,
    _truth_maps,
    _validate_fold06_run,
    _write_csv,
    build_fold06_session_manifest,
    build_token_occurrences,
    build_trial_token_durations,
    load_segment_artifacts,
)
from experiments.motion_primitive.trajectory_ablation import (  # noqa: E402
    SourceSignalRepository,
    jsonable,
    sha256_file,
)


ARMS = (
    "G0_coarse",
    "G1_residual_flat",
    "G2_gravity_flat",
    "G3_hierarchical",
)


PROTOCOL = {
    "name": "fold06_session2_residual_gravity_hierarchical_gate_v1",
    "scope": "isolated_session2_unlabelled_trajectory_clustering_proxy",
    "is_formal_happy_cgcd": False,
    "primary_question": (
        "Does an encoder-residual gate preserve the dynamic coarse token while "
        "allowing gravity to refine only confident static occurrences?"
    ),
    "arms": {
        "G0_coarse": "K=32 coarse duration histogram; no selected-token split",
        "G1_residual_flat": "selected token split globally by residual K-medoids",
        "G2_gravity_flat": "selected token split globally by gravity K-medoids",
        "G3_hierarchical": (
            "residual K-medoids -> train-only motion-energy static-child naming -> "
            "train static-radius gate -> gravity K-medoids; dynamic/low-confidence "
            "occurrences retain the original coarse token"
        ),
    },
    "registered_gate": {
        "residual_children": 2,
        "gravity_children_within_confident_static": 2,
        "static_radius_quantile_default": 0.95,
        "minimum_token_fraction_default": 0.50,
        "minimum_motion_energy_ratio_default": 2.0,
        "minimum_motion_energy_gap_default": 0.25,
        "threshold_status": (
            "fixed conservative exploratory heuristics; not validated optima and "
            "must not be tuned on Session-2 test labels. Passing them identifies a "
            "relative lower-motion proxy, not an absolutely proven static state"
        ),
        "lower_motion_child_naming": (
            "lower median of duration-weighted acceleration-deviation and gyroscope-"
            "magnitude score, scaled by cumulative-train p95 per component; both "
            "component medians must agree on which child is lower-motion"
        ),
        "test_motion_energy_used_for_assignment": False,
        "fallback": (
            "residual-dynamic, residual-static outside the train radius, and all "
            "non-selected tokens remain at their original coarse token id"
        ),
        "fail_closed": (
            "if train residual children do not reach both the motion-energy ratio "
            "and absolute-gap thresholds, disable G3 and reproduce G0 exactly"
        ),
    },
    "trial_readout": (
        "duration-normalized histogram -> cumulative-train-only KMeans K=10 -> "
        "frozen Session-2 test predictions"
    ),
    "evaluation": (
        "one complete-test-set global Hungarian alignment per arm; paired trial "
        "bootstrap holds each alignment fixed"
    ),
    "status": (
        "single-fold/single-seed exploratory gate ablation; classification benefit "
        "is required before any 7-fold x 4-seed expansion"
    ),
    "attribution_limits": [
        (
            "G3 still uses raw acceleration/gyroscope information to name the static "
            "training child and raw gravity inside that child; its benefit is not an "
            "encoder-only effect. Encoder attribution requires paired A3-vs-A0 runs."
        ),
        (
            "The readout uses token composition/duration only and does not test token "
            "order or transition information."
        ),
        (
            "Fold06 contains only held-out Subjects 4 and 5; trial bootstrap does not "
            "create unseen-subject evidence."
        ),
    ],
}


REGISTERED_UPSTREAM_CODEBOOK = {
    "primitive_num": 32,
    "pca_dim": 64,
    "embedding_normalization": "l2",
    "assignment_metric": "cosine",
    "weighting": "per_trial",
    "kmeans_n_init": 20,
    "kmeans_max_iter": 300,
    "old_class_count": 6,
}

REGISTERED_SEGMENTATION_PARAMETERS = {
    "changepoint_context_windows": 2,
    "changepoint_score_quantile": 0.90,
    "changepoint_min_segment_windows": 2,
}


def _validated_sha256(value, field: str) -> str:
    digest = str(value or "")
    if len(digest) != 64 or any(
        character not in "0123456789abcdefABCDEF" for character in digest
    ):
        raise RuntimeError(f"Upstream {field} is not a complete SHA-256 digest.")
    return digest.lower()


def _registered_integer(mapping: Mapping, field: str) -> int:
    raw = mapping.get(field)
    if isinstance(raw, bool):
        raise RuntimeError(f"Registered upstream {field} must be an integer.")
    try:
        numeric = float(raw)
    except (TypeError, ValueError) as error:
        raise RuntimeError(f"Registered upstream {field} must be an integer.") from error
    if not np.isfinite(numeric) or not numeric.is_integer():
        raise RuntimeError(f"Registered upstream {field} must be an integer.")
    return int(numeric)


def _validate_registered_upstream(config: Mapping, artifacts) -> dict:
    """Fail closed if the frozen K32/PCA64 upstream experiment drifts."""

    arguments = config.get("arguments", {})
    codebook = config.get("codebook", {})
    observed = {
        "primitive_num": _registered_integer(arguments, "primitive_num"),
        "pca_dim": _registered_integer(arguments, "pca_dim"),
        "embedding_normalization": str(
            arguments.get("embedding_normalization", "")
        ),
        "assignment_metric": str(codebook.get("assignment_metric", "")),
        "weighting": str(arguments.get("codebook_weighting", "")),
        "kmeans_n_init": _registered_integer(arguments, "kmeans_n_init"),
        "kmeans_max_iter": _registered_integer(arguments, "kmeans_max_iter"),
        "old_class_count": _registered_integer(arguments, "old_class_count"),
    }
    if observed != REGISTERED_UPSTREAM_CODEBOOK:
        raise RuntimeError(
            "Registered upstream K32/PCA64 protocol drifted: "
            f"observed={observed}, expected={REGISTERED_UPSTREAM_CODEBOOK}."
        )
    repeated = {
        "primitive_num": _registered_integer(codebook, "primitive_num"),
        "pca_dim": _registered_integer(codebook, "pca_dim"),
        "embedding_normalization": str(codebook.get("embedding_normalization", "")),
        "weighting": str(codebook.get("weighting", "")),
    }
    expected_repeated = {
        key: REGISTERED_UPSTREAM_CODEBOOK[key]
        for key in ("primitive_num", "pca_dim", "embedding_normalization", "weighting")
    }
    if repeated != expected_repeated:
        raise RuntimeError(
            "Config arguments/codebook audit disagree: "
            f"observed={repeated}, expected={expected_repeated}."
        )
    center_shape = tuple(int(value) for value in artifacts.centers.shape)
    if np.asarray(artifacts.embeddings).ndim != 2:
        raise RuntimeError("Segment embeddings must be a two-dimensional matrix.")
    embedding_dim = int(artifacts.embeddings.shape[1])
    if center_shape != (32, 64) or embedding_dim != 64:
        raise RuntimeError(
            "Segment/codebook artifact is not the registered K32/PCA64 representation: "
            f"centers={center_shape}, embedding_dim={embedding_dim}."
        )
    segmentation_observed = {}
    for field, expected in REGISTERED_SEGMENTATION_PARAMETERS.items():
        try:
            value = float(arguments[field])
        except (KeyError, TypeError, ValueError) as error:
            raise RuntimeError(f"Upstream segmentation parameter {field} is invalid.") from error
        if not np.isfinite(value) or value != float(expected):
            raise RuntimeError(
                f"Upstream segmentation parameter {field}={value!r}, expected {expected}."
            )
        segmentation_observed[field] = int(value) if isinstance(expected, int) else value

    metadata = config.get("checkpoint_metadata", {})
    if metadata.get("smoke_test") is not False or (
        metadata.get("outer_test_used_during_encoder_training") is not False
    ):
        raise RuntimeError("A smoke or outer-test-trained encoder entered the gate run.")
    encoder_fingerprint = config.get("encoder_implementation_fingerprint", {})
    source_checkpoint = config.get("encoder_source_checkpoint", {})
    combined = _validated_sha256(
        encoder_fingerprint.get("combined_sha256"),
        "encoder implementation combined_sha256",
    )
    source_sha = _validated_sha256(
        source_checkpoint.get("sha256"), "encoder source checkpoint sha256"
    )
    final_checkpoint_sha = _validated_sha256(
        config.get("checkpoint_sha256"), "motion encoder checkpoint_sha256"
    )
    return {
        "registered_upstream_codebook": dict(REGISTERED_UPSTREAM_CODEBOOK),
        "registered_segmentation_parameters": dict(REGISTERED_SEGMENTATION_PARAMETERS),
        "observed_segmentation_parameters": segmentation_observed,
        "artifact_center_shape": list(center_shape),
        "artifact_embedding_dim": embedding_dim,
        "encoder_implementation_combined_sha256": combined,
        "encoder_source_checkpoint_sha256": source_sha,
        "motion_encoder_checkpoint_sha256": final_checkpoint_sha,
        "checkpoint_smoke_test": False,
        "outer_test_used_during_encoder_training": False,
    }


def _build_motion_components(
    artifacts,
    source: LabelFreeSourceSignalRepository,
    trial_ids: Sequence[int],
    selected_token: int,
) -> dict[int, np.ndarray]:
    """Build train-only motion descriptors without loading metadata."""

    result = {}
    for trial_id_value in trial_ids:
        trial_id = int(trial_id_value)
        positions = np.flatnonzero(artifacts.trial_ids == trial_id)
        if len(positions) == 0:
            raise RuntimeError(f"Trial {trial_id} is absent from segment artifacts.")
        positions = positions[np.argsort(artifacts.starts[positions], kind="stable")]
        result[trial_id] = motion_components_from_token_partitions(
            source.sensor(trial_id),
            artifacts.starts[positions],
            artifacts.ends[positions],
            artifacts.tokens[positions],
            selected_token=int(selected_token),
        )
    return result


def _flat_child_map(model, occurrences) -> dict[int, int]:
    children = assign_secondary_codebook(model, occurrences)
    return {
        int(item.trial_global_id): int(child)
        for item, child in zip(occurrences, children.tolist())
    }


def _gate_state_map(assignments) -> dict[int, int]:
    return {
        int(trial_id): int(state)
        for trial_id, state in zip(
            assignments.trial_ids.tolist(), assignments.gate_states.tolist()
        )
    }


def _posthoc_gate_sit_stand(
    evaluation_trial_ids: Sequence[int],
    gate_state_by_trial: Mapping[int, int],
    labels_by_trial: Mapping[int, int],
    subjects_by_trial: Mapping[int, int],
    sitting_label: int = 7,
    standing_label: int = 8,
) -> dict:
    """Evaluate all Sit/Stand states, counting coarse fallback as a third cluster."""

    trial_ids = sorted(
        int(trial_id)
        for trial_id in evaluation_trial_ids
        if int(trial_id) in labels_by_trial
        and int(trial_id) in subjects_by_trial
        and int(labels_by_trial[int(trial_id)])
        in (int(sitting_label), int(standing_label))
    )
    if not trial_ids:
        raise RuntimeError("No Sitting/Standing trial reached the gate diagnostic.")
    truth = np.asarray([labels_by_trial[trial_id] for trial_id in trial_ids], dtype=np.int64)
    states = np.asarray(
        [gate_state_by_trial.get(trial_id, 0) for trial_id in trial_ids], dtype=np.int64
    )
    state_ids = sorted(set(states.tolist()))
    routed_state_ids = sorted(set(states.tolist()) & {1, 2})
    truth_ids = [int(sitting_label), int(standing_label)]
    contingency = np.zeros((len(routed_state_ids), len(truth_ids)), dtype=np.int64)
    state_position = {state: index for index, state in enumerate(routed_state_ids)}
    truth_position = {label: index for index, label in enumerate(truth_ids)}
    for state, label in zip(states.tolist(), truth.tolist()):
        if int(state) in state_position:
            contingency[state_position[int(state)], truth_position[int(label)]] += 1
    rows, columns = (
        linear_sum_assignment(contingency.max() - contingency)
        if len(routed_state_ids)
        else (np.asarray([], dtype=np.int64), np.asarray([], dtype=np.int64))
    )
    mapping = {
        int(routed_state_ids[int(row)]): int(truth_ids[int(column)])
        for row, column in zip(rows.tolist(), columns.tolist())
    }
    aligned = np.asarray([mapping.get(int(state), -1) for state in states], dtype=np.int64)
    recalls = []
    for label in truth_ids:
        mask = truth == int(label)
        recalls.append(float(np.mean(aligned[mask] == truth[mask])))
    by_subject = {}
    subject_values = np.asarray([subjects_by_trial[trial_id] for trial_id in trial_ids])
    for subject_id in sorted(set(subject_values.tolist())):
        mask = subject_values == int(subject_id)
        subject_truth = truth[mask]
        subject_aligned = aligned[mask]
        observed = sorted(set(subject_truth.tolist()))
        subject_recalls = [
            float(np.mean(subject_aligned[subject_truth == label] == label))
            if np.any(subject_truth == label)
            else None
            for label in truth_ids
        ]
        by_subject[str(int(subject_id))] = {
            "trial_count": int(np.sum(mask)),
            "observed_activity_ids": observed,
            "complete_binary_support": observed == truth_ids,
            "accuracy_using_global_mapping": float(
                np.mean(subject_aligned == subject_truth)
            ),
            "balanced_accuracy_using_global_mapping": (
                float(np.mean(subject_recalls)) if all(value is not None for value in subject_recalls) else None
            ),
        }
    confusion = np.zeros((2, 3), dtype=np.int64)
    for row, label in enumerate(truth_ids):
        mask = truth == label
        for column, predicted in enumerate(truth_ids):
            confusion[row, column] = int(np.sum(aligned[mask] == predicted))
        confusion[row, 2] = int(np.sum(aligned[mask] < 0))
    routed = states > 0
    return {
        "evaluation_is_posthoc": True,
        "fit_used_labels_or_names": False,
        "fallback_is_counted_not_dropped": True,
        "evaluated_trial_ids": trial_ids,
        "binary_trial_count": len(trial_ids),
        "observed_gate_states": state_ids,
        "routed_gate_states_used_for_binary_mapping": routed_state_ids,
        "missing_selected_token_counted_as_fallback": int(
            sum(trial_id not in gate_state_by_trial for trial_id in trial_ids)
        ),
        "coarse_fallback_trial_count": int(np.sum(states == 0)),
        "gate_state_meanings": {str(key): value for key, value in GATE_STATES.items()},
        "binary_hungarian_accuracy": float(np.mean(aligned == truth)),
        "binary_balanced_accuracy": float(np.mean(recalls)),
        "routed_trial_count": int(np.sum(routed)),
        "routed_coverage": float(np.mean(routed)),
        "conditional_accuracy_on_routed_trials": (
            float(np.mean(aligned[routed] == truth[routed])) if np.any(routed) else None
        ),
        "binary_mapping_gate_state_to_activity": {
            str(key): value for key, value in sorted(mapping.items())
        },
        "unmapped_fallback_or_child_count": int(np.sum(aligned < 0)),
        "binary_per_class_recall": {
            str(label): value for label, value in zip(truth_ids, recalls)
        },
        "binary_confusion_counts_columns": [
            str(sitting_label),
            str(standing_label),
            "unmapped_or_fallback",
        ],
        "binary_confusion_counts": confusion,
        "binary_by_subject_using_same_global_mapping": by_subject,
    }


def _actual_segment_tokens(
    artifacts,
    trial_id: int,
    arm: str,
    selected_token: int,
    flat_children: Mapping[str, Mapping[int, int]],
    gate_states: Mapping[int, int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    positions = np.flatnonzero(artifacts.trial_ids == int(trial_id))
    positions = positions[np.argsort(artifacts.starts[positions], kind="stable")]
    tokens = artifacts.tokens[positions].copy()
    selected = tokens == int(selected_token)
    if np.any(selected) and arm in ("G1_residual_flat", "G2_gravity_flat"):
        child = int(flat_children[arm][int(trial_id)])
        if child == 1:
            tokens[selected] = len(artifacts.centers)
    elif np.any(selected) and arm == "G3_hierarchical":
        state = int(gate_states[int(trial_id)])
        if state > 0:
            tokens[selected] = len(artifacts.centers) + state - 1
    return tokens, artifacts.starts[positions], artifacts.ends[positions]


def _save_trajectory_plot(
    path: Path,
    artifacts,
    test_ids: Sequence[int],
    selected_token: int,
    flat_children: Mapping[str, Mapping[int, int]],
    gate_states: Mapping[int, int],
    labels: Mapping[int, int],
    subjects: Mapping[int, int],
) -> None:
    ordered = sorted(
        (int(value) for value in test_ids),
        key=lambda trial_id: (labels[trial_id], subjects[trial_id], trial_id),
    )
    height = max(14.0, 0.17 * len(ordered) * len(ARMS))
    fig, axes = plt.subplots(len(ARMS), 1, figsize=(19, height), sharex=True)
    palette = plt.get_cmap("tab20")
    special = {len(artifacts.centers): "black", len(artifacts.centers) + 1: "magenta"}
    for axis, arm in zip(axes, ARMS):
        for row, trial_id in enumerate(ordered):
            tokens, starts, ends = _actual_segment_tokens(
                artifacts,
                trial_id,
                arm,
                selected_token,
                flat_children,
                gate_states,
            )
            for token, start, end in zip(tokens, starts, ends):
                axis.barh(
                    row,
                    (int(end) - int(start)) / 100.0,
                    left=int(start) / 100.0,
                    height=0.82,
                    color=special.get(int(token), palette((int(token) % 20) / 19.0)),
                    linewidth=0,
                )
        axis.set_title(arm)
        axis.set_yticks(np.arange(len(ordered)))
        axis.set_yticklabels(
            [f"y{labels[t]}-S{subjects[t]}-T{t}" for t in ordered], fontsize=5
        )
        axis.set_ylabel("test trial")
        axis.invert_yaxis()
    axes[-1].set_xlabel("visible time (seconds)")
    fig.suptitle(
        f"Session-2 actual readout trajectories; selected coarse token q={selected_token}",
        y=0.995,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _save_candidate_focus_plot(
    path: Path,
    artifacts,
    test_ids: Sequence[int],
    selected_token: int,
    flat_children: Mapping[str, Mapping[int, int]],
    gate_states: Mapping[int, int],
    labels: Mapping[int, int],
    subjects: Mapping[int, int],
    names: Mapping[int, str],
) -> None:
    focused = sorted(
        (int(trial_id) for trial_id in test_ids if int(trial_id) in gate_states),
        key=lambda trial_id: (labels[trial_id], subjects[trial_id], trial_id),
    )
    if not focused:
        raise RuntimeError("No selected-token test trial is available for the focus plot.")
    fig, axes = plt.subplots(len(ARMS), 1, figsize=(18, max(9.0, 0.55 * len(focused) * len(ARMS))), sharex=True)
    palette = plt.get_cmap("tab20")
    special = {len(artifacts.centers): "black", len(artifacts.centers) + 1: "magenta"}
    for axis, arm in zip(axes, ARMS):
        for row, trial_id in enumerate(focused):
            tokens, starts, ends = _actual_segment_tokens(
                artifacts,
                trial_id,
                arm,
                selected_token,
                flat_children,
                gate_states,
            )
            for token, start, end in zip(tokens, starts, ends):
                axis.barh(
                    row,
                    (int(end) - int(start)) / 100.0,
                    left=int(start) / 100.0,
                    height=0.78,
                    color=special.get(int(token), palette((int(token) % 20) / 19.0)),
                    linewidth=0,
                )
        axis.set_title(arm)
        axis.set_yticks(np.arange(len(focused)))
        axis.set_yticklabels(
            [f"{names[t]} | S{subjects[t]} | T{t}" for t in focused], fontsize=7
        )
        axis.invert_yaxis()
    axes[-1].set_xlabel("visible time (seconds)")
    fig.suptitle(
        f"Selected-token test trajectories only; q={selected_token}, black/magenta=new static children",
        y=0.995,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _save_confusions(
    path: Path, arm_results: Mapping[str, dict], class_names: Sequence[str]
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    for axis, arm in zip(axes.flat, ARMS):
        matrix = np.asarray(
            arm_results[arm]["global_metrics"]["confusion_counts"], dtype=np.float64
        )
        normalized = np.divide(
            matrix,
            matrix.sum(axis=1, keepdims=True),
            out=np.zeros_like(matrix),
            where=matrix.sum(axis=1, keepdims=True) > 0,
        )
        image = axis.imshow(normalized, vmin=0.0, vmax=1.0, cmap="Blues")
        metrics = arm_results[arm]["global_metrics"]
        axis.set_title(
            f"{arm}: All={metrics['all_accuracy']:.3f}, New={metrics['new_accuracy']:.3f}"
        )
        axis.set_xticks(range(len(class_names)), class_names, rotation=55, ha="right", fontsize=7)
        axis.set_yticks(range(len(class_names)), class_names, fontsize=7)
        axis.set_xlabel("globally aligned prediction")
        axis.set_ylabel("true activity")
        fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    fig.suptitle("Session-2 test confusion matrices (one global Hungarian per arm)")
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _save_activity_heatmaps(
    path: Path, matrices: Mapping[str, np.ndarray], class_names: Sequence[str]
) -> None:
    upper = max(float(np.max(value)) for value in matrices.values())
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    for axis, arm in zip(axes.flat, ARMS):
        matrix = np.asarray(matrices[arm], dtype=np.float64)
        image = axis.imshow(matrix, vmin=0.0, vmax=max(upper, 1e-6), cmap="magma")
        axis.set_title(arm)
        axis.set_xticks(range(len(class_names)), class_names, rotation=55, ha="right", fontsize=7)
        axis.set_yticks(range(len(class_names)), class_names, fontsize=7)
        for row in range(len(matrix)):
            for column in range(len(matrix)):
                axis.text(
                    column,
                    row,
                    f"{matrix[row, column]:.2f}",
                    ha="center",
                    va="center",
                    fontsize=5,
                    color="white" if matrix[row, column] > 0.55 * max(upper, 1e-6) else "black",
                )
        fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    fig.suptitle("Session-2 activity mean histogram Jensen-Shannon distances")
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def run(args: argparse.Namespace) -> dict:
    bootstrap_resamples = int(args.bootstrap_resamples)
    radius_quantile = float(args.static_radius_quantile)
    minimum_energy_ratio = float(args.minimum_motion_energy_ratio)
    minimum_energy_gap = float(args.minimum_motion_energy_gap)
    minimum_token_fraction = float(args.minimum_token_fraction)
    if bootstrap_resamples < 1:
        raise ValueError("bootstrap_resamples must be positive.")
    if not 0.0 < radius_quantile <= 1.0:
        raise ValueError("static_radius_quantile must lie in (0,1].")
    if not np.isfinite(minimum_energy_ratio) or minimum_energy_ratio < 1.0:
        raise ValueError("minimum_motion_energy_ratio must be finite and >=1.")
    if not np.isfinite(minimum_energy_gap) or minimum_energy_gap < 0.0:
        raise ValueError("minimum_motion_energy_gap must be finite and non-negative.")
    if not 0.0 < minimum_token_fraction <= 1.0:
        raise ValueError("minimum_token_fraction must lie in (0,1].")
    run_dir = Path(args.run_dir).resolve()
    npz_path = Path(args.npz_path).resolve()
    output_dir = _require_new_output_dir(Path(args.output_dir))
    config, split, _, codebook_path = _validate_fold06_run(
        run_dir, npz_path, seed=int(args.seed)
    )
    validation_subjects = config.get("checkpoint_metadata", {}).get(
        "offline_val_subjects", []
    )
    manifest, _ = build_fold06_session_manifest(
        npz_path=npz_path,
        fit_subjects=split["fit_subjects"],
        eval_subjects=split["eval_subjects"],
        validation_subjects=validation_subjects,
        seed=int(args.seed),
    )
    train_ids = [int(value) for value in manifest["cumulative_train_trial_ids"]]
    test_ids = [int(value) for value in manifest["session_2_test_trial_ids"]]
    all_ids = train_ids + test_ids
    artifacts = load_segment_artifacts(run_dir, codebook_path)
    upstream_audit = _validate_registered_upstream(config, artifacts)
    source = LabelFreeSourceSignalRepository(npz_path)
    visible_end_by_trial = {
        trial_id: int(max(source.trial_window_starts(trial_id)) + source.window_size)
        for trial_id in all_ids
    }
    trial_token_durations = build_trial_token_durations(
        artifacts, all_ids, visible_end_by_trial=visible_end_by_trial
    )
    candidate = select_dominant_token(
        {trial_id: trial_token_durations[trial_id] for trial_id in train_ids},
        minimum_fraction=minimum_token_fraction,
        minimum_support=10,
    )
    selected_token = int(candidate["selected_token"])
    occurrences = build_token_occurrences(
        artifacts, source, all_ids, selected_token=selected_token
    )
    dominant_train_ids = [int(value) for value in candidate["dominant_trial_ids"]]
    fit_occurrences = [occurrences[trial_id] for trial_id in dominant_train_ids]
    all_occurrences = [occurrences[trial_id] for trial_id in sorted(occurrences)]

    residual_model = fit_secondary_codebook(
        fit_occurrences, arm="U1_residual", restarts=50, seed=int(args.seed)
    )
    gravity_model = fit_secondary_codebook(
        fit_occurrences, arm="U2_gravity", restarts=50, seed=int(args.seed)
    )
    flat_children = {
        "G1_residual_flat": _flat_child_map(residual_model, all_occurrences),
        "G2_gravity_flat": _flat_child_map(gravity_model, all_occurrences),
    }
    motion_components = _build_motion_components(
        artifacts, source, dominant_train_ids, selected_token
    )
    gate_model = fit_hierarchical_gate(
        fit_occurrences,
        motion_components,
        static_radius_quantile=radius_quantile,
        minimum_token_fraction=minimum_token_fraction,
        minimum_motion_energy_ratio=minimum_energy_ratio,
        minimum_motion_energy_gap=minimum_energy_gap,
        restarts=50,
        seed=int(args.seed),
    )
    gate_assignments = assign_hierarchical_gate(gate_model, all_occurrences)
    gate_states = _gate_state_map(gate_assignments)

    histograms_by_arm: dict[str, dict[str, np.ndarray]] = {}
    predictions_by_arm: dict[str, np.ndarray] = {}
    readout_audit_by_arm: dict[str, dict] = {}
    for arm in ARMS:
        if arm == "G0_coarse":
            train_histograms = duration_histograms(
                train_ids, trial_token_durations, primitive_num=len(artifacts.centers)
            )
            test_histograms = duration_histograms(
                test_ids, trial_token_durations, primitive_num=len(artifacts.centers)
            )
        elif arm in flat_children:
            train_histograms = duration_histograms(
                train_ids,
                trial_token_durations,
                primitive_num=len(artifacts.centers),
                selected_token=selected_token,
                child_by_trial=flat_children[arm],
            )
            test_histograms = duration_histograms(
                test_ids,
                trial_token_durations,
                primitive_num=len(artifacts.centers),
                selected_token=selected_token,
                child_by_trial=flat_children[arm],
            )
        else:
            train_histograms = hierarchical_duration_histograms(
                train_ids,
                trial_token_durations,
                primitive_num=len(artifacts.centers),
                selected_token=selected_token,
                gate_state_by_trial=gate_states,
            )
            test_histograms = hierarchical_duration_histograms(
                test_ids,
                trial_token_durations,
                primitive_num=len(artifacts.centers),
                selected_token=selected_token,
                gate_state_by_trial=gate_states,
            )
        if arm == "G3_hierarchical" and not gate_model.gate_enabled:
            predictions = predictions_by_arm["G0_coarse"].copy()
            readout_audit = dict(readout_audit_by_arm["G0_coarse"])
            readout_audit["disabled_hierarchy_reused_g0_readout"] = True
        else:
            predictions, readout_audit = _fit_trial_clusterer(
                train_histograms, test_histograms, seed=int(args.seed)
            )
        histograms_by_arm[arm] = {
            "train": train_histograms,
            "test": test_histograms,
        }
        predictions_by_arm[arm] = predictions
        readout_audit_by_arm[arm] = readout_audit

    if not gate_model.gate_enabled:
        coarse_train = histograms_by_arm["G0_coarse"]["train"]
        coarse_test = histograms_by_arm["G0_coarse"]["test"]
        gated_train = histograms_by_arm["G3_hierarchical"]["train"]
        gated_test = histograms_by_arm["G3_hierarchical"]["test"]
        primitive_num = len(artifacts.centers)
        if not (
            np.array_equal(gated_train[:, :primitive_num], coarse_train)
            and np.array_equal(gated_test[:, :primitive_num], coarse_test)
            and np.count_nonzero(gated_train[:, primitive_num:]) == 0
            and np.count_nonzero(gated_test[:, primitive_num:]) == 0
            and np.array_equal(
                predictions_by_arm["G3_hierarchical"],
                predictions_by_arm["G0_coarse"],
            )
        ):
            raise RuntimeError("Disabled hierarchy failed to reproduce G0 exactly.")

    # Persist the label-free decisions before creating a metadata-aware reader.
    output_dir.mkdir(parents=True)
    frozen_assignment_path = output_dir / "frozen_label_free_gate_assignments.npz"
    np.savez_compressed(
        frozen_assignment_path,
        trial_global_ids=gate_assignments.trial_ids,
        residual_children=gate_assignments.residual_children,
        assigned_residual_distances=gate_assignments.assigned_residual_distances,
        confident_static=gate_assignments.confident_static,
        gravity_children=gate_assignments.gravity_children,
        gate_states=gate_assignments.gate_states,
    )

    evaluation_source = SourceSignalRepository(npz_path)
    labels, subjects, names = _truth_maps(evaluation_source, test_ids)
    y_true = np.asarray([labels[trial_id] for trial_id in test_ids], dtype=np.int64)
    y_subject = np.asarray([subjects[trial_id] for trial_id in test_ids], dtype=np.int64)
    class_ids = sorted(set(y_true.tolist()))
    class_names = [
        names[next(trial_id for trial_id in test_ids if labels[trial_id] == class_id)]
        for class_id in class_ids
    ]
    fit_audits = {
        "G0_coarse": None,
        "G1_residual_flat": secondary_fit_diagnostics(residual_model, fit_occurrences),
        "G2_gravity_flat": secondary_fit_diagnostics(gravity_model, fit_occurrences),
        "G3_hierarchical": hierarchical_gate_diagnostics(gate_model, fit_occurrences),
    }
    arm_results = {}
    distance_matrices = {}
    for arm in ARMS:
        metrics = global_hungarian_metrics(
            y_true, predictions_by_arm[arm], old_class_count=6
        )
        if arm in flat_children:
            test_child_map = {
                trial_id: flat_children[arm][trial_id]
                for trial_id in test_ids
                if trial_id in flat_children[arm]
            }
            sit_stand = posthoc_secondary_metrics(
                test_child_map, labels, subjects, sitting_label=7, standing_label=8
            )
        elif arm == "G3_hierarchical":
            sit_stand = _posthoc_gate_sit_stand(
                test_ids,
                gate_states,
                labels,
                subjects,
                sitting_label=7,
                standing_label=8,
            )
        else:
            sit_stand = None
        distance = class_mean_histogram_distances(
            histograms_by_arm[arm]["test"], y_true, class_ids
        )
        distance_matrices[arm] = distance
        arm_results[arm] = {
            "codebook_or_gate_fit": fit_audits[arm],
            "trial_clusterer": readout_audit_by_arm[arm],
            "global_metrics": metrics,
            "sit_stand_posthoc": sit_stand,
            "activity_mean_histogram_distance_matrix": distance,
        }

    aligned = {
        arm: np.asarray(
            arm_results[arm]["global_metrics"]["aligned_predictions"], dtype=np.int64
        )
        for arm in ARMS
    }
    bootstrap, bootstrap_rows = _paired_bootstrap_comparisons(
        trial_ids=test_ids,
        y_true=y_true,
        subjects=y_subject,
        comparisons={
            f"{arm}_minus_G0_coarse": (aligned["G0_coarse"], aligned[arm])
            for arm in ARMS[1:]
        },
        resamples=bootstrap_resamples,
        seed=int(args.seed),
        scope="fold06_session2_subjects_4_5_hierarchical_gate",
    )

    _write_csv(
        output_dir / "session_manifest.csv",
        [
            {"split": split_name, "session": session, "trial_global_id": int(trial_id)}
            for split_name, session, ids in (
                ("online_train", 1, manifest["session_1_train_trial_ids"]),
                ("online_train", 2, manifest["session_2_train_trial_ids"]),
                ("online_test", 2, manifest["session_2_test_trial_ids"]),
            )
            for trial_id in ids
        ],
    )
    prediction_rows = []
    for row, trial_id in enumerate(test_ids):
        item = {
            "trial_global_id": trial_id,
            "subject_id": subjects[trial_id],
            "activity_label_0based": labels[trial_id],
            "activity_name": names[trial_id],
            "selected_token_present": trial_id in gate_states,
            "G1_residual_child": flat_children["G1_residual_flat"].get(trial_id, ""),
            "G2_gravity_child": flat_children["G2_gravity_flat"].get(trial_id, ""),
            "G3_gate_state": gate_states.get(trial_id, ""),
        }
        for arm in ARMS:
            item[f"{arm}_raw_cluster"] = int(predictions_by_arm[arm][row])
            item[f"{arm}_aligned_prediction"] = int(aligned[arm][row])
        prediction_rows.append(item)
    _write_csv(output_dir / "session2_predictions.csv", prediction_rows)

    assignment_index = {
        int(trial_id): position
        for position, trial_id in enumerate(gate_assignments.trial_ids.tolist())
    }
    gate_rows = []
    for trial_id in test_ids:
        if trial_id not in assignment_index:
            continue
        position = assignment_index[trial_id]
        state = int(gate_assignments.gate_states[position])
        gate_rows.append(
            {
                "trial_global_id": trial_id,
                "subject_id": subjects[trial_id],
                "activity_label_0based": labels[trial_id],
                "activity_name": names[trial_id],
                "residual_child": int(gate_assignments.residual_children[position]),
                "assigned_residual_distance": float(
                    gate_assignments.assigned_residual_distances[position]
                ),
                "confident_static": bool(gate_assignments.confident_static[position]),
                "gravity_child": int(gate_assignments.gravity_children[position]),
                "gate_state": state,
                "gate_state_meaning": GATE_STATES[state],
            }
        )
    _write_csv(output_dir / "session2_gate_assignments.csv", gate_rows)

    trajectory_rows = []
    for trial_id in test_ids:
        row = {
            "trial_global_id": trial_id,
            "subject_id": subjects[trial_id],
            "activity_label_0based": labels[trial_id],
            "activity_name": names[trial_id],
        }
        for arm in ARMS:
            tokens, starts, ends = _actual_segment_tokens(
                artifacts,
                trial_id,
                arm,
                selected_token,
                flat_children,
                gate_states,
            )
            rle, _ = run_length_encode(tokens.astype(int).tolist())
            row[f"{arm}_rle_tokens"] = json.dumps(rle.astype(int).tolist())
            row[f"{arm}_segment_tokens"] = json.dumps(tokens.astype(int).tolist())
            row[f"{arm}_durations_samples"] = json.dumps(
                (ends - starts).astype(int).tolist()
            )
        trajectory_rows.append(row)
    _write_csv(output_dir / "session2_gated_trajectories.csv", trajectory_rows)
    _write_csv(output_dir / "trial_bootstrap_confidence_intervals.csv", bootstrap_rows)

    _save_trajectory_plot(
        output_dir / "session2_gated_trajectories.png",
        artifacts,
        test_ids,
        selected_token,
        flat_children,
        gate_states,
        labels,
        subjects,
    )
    _save_confusions(
        output_dir / "session2_global_confusions.png", arm_results, class_names
    )
    _save_activity_heatmaps(
        output_dir / "session2_activity_distance_heatmaps.png",
        distance_matrices,
        class_names,
    )
    _save_candidate_focus_plot(
        output_dir / "session2_selected_token_focus_trajectories.png",
        artifacts,
        test_ids,
        selected_token,
        flat_children,
        gate_states,
        labels,
        subjects,
        names,
    )

    result = {
        "protocol": PROTOCOL,
        "arguments": {
            "run_dir": str(run_dir),
            "npz_path": str(npz_path),
            "output_dir": str(output_dir),
            "seed": int(args.seed),
            "static_radius_quantile": radius_quantile,
            "minimum_token_fraction": minimum_token_fraction,
            "minimum_motion_energy_ratio": minimum_energy_ratio,
            "minimum_motion_energy_gap": minimum_energy_gap,
            "bootstrap_resamples": bootstrap_resamples,
        },
        "input_audit": {
            "registered_upstream": upstream_audit,
            "primitive_segmentation": config.get("segmentation", {}).get("method"),
            "encoder_ablation_profile": config.get("encoder_training", {}).get(
                "ablation_profile"
            ),
            "checkpoint": config.get("checkpoint"),
            "checkpoint_sha256": config.get("checkpoint_sha256"),
            "run_dir": str(run_dir),
            "run_config_sha256": sha256_file(run_dir / "experiment_config.json"),
            "segment_artifact_sha256": sha256_file(
                run_dir / "segment_embeddings_and_tokens.npz"
            ),
            "codebook_artifact_sha256": sha256_file(codebook_path),
            "npz_sha256": sha256_file(npz_path),
            "label_free_raw_source": source.source_audit(all_ids),
            "postfreeze_metadata_and_mat_source_audit": evaluation_source.source_audit(
                all_ids
            ),
            "implementation_fingerprint": {
                "runner_path": str(Path(__file__).resolve()),
                "runner_sha256": sha256_file(Path(__file__).resolve()),
                "hierarchical_helper_path": str(
                    PROJECT_ROOT
                    / "experiments"
                    / "motion_primitive"
                    / "hierarchical_gate.py"
                ),
                "hierarchical_helper_sha256": sha256_file(
                    PROJECT_ROOT
                    / "experiments"
                    / "motion_primitive"
                    / "hierarchical_gate.py"
                ),
                "frozen_secondary_helper_sha256": sha256_file(
                    PROJECT_ROOT
                    / "experiments"
                    / "motion_primitive"
                    / "online_secondary_codebook.py"
                ),
                "frozen_secondary_runner_path": str(
                    PROJECT_ROOT
                    / "experiments"
                    / "motion_primitive"
                    / "run_online_secondary_codebook.py"
                ),
                "frozen_secondary_runner_sha256": sha256_file(
                    PROJECT_ROOT
                    / "experiments"
                    / "motion_primitive"
                    / "run_online_secondary_codebook.py"
                ),
            },
        },
        "session_manifest_audit": manifest,
        "label_firewall": {
            "gate_fit_uses_activity_labels_or_names": False,
            "static_child_naming_uses_train_only_raw_motion": True,
            "test_motion_energy_used_for_gate_assignment": False,
            "test_occurrences_used_to_fit_residual_or_gravity_codebook": False,
            "trial_kmeans_fit_uses_activity_labels": False,
            "metadata_aware_repository_created_after_predictions": True,
            "evaluation_ground_truth_join_time": (
                "after all gate states and test KMeans predictions were frozen"
            ),
            "future_feature_trial_count": int(manifest["future_feature_trial_count"]),
        },
        "candidate": candidate,
        "gate_fit": hierarchical_gate_diagnostics(gate_model, fit_occurrences),
        "arms": arm_results,
        "trial_bootstrap_confidence_intervals": bootstrap,
        "frozen_label_free_assignments": {
            "file": frozen_assignment_path.name,
            "sha256": sha256_file(frozen_assignment_path),
            "contains_activity_labels_or_names": False,
        },
        "generated_files": [
            "online_hierarchical_gate_results.json",
            "session_manifest.csv",
            "session2_predictions.csv",
            "session2_gate_assignments.csv",
            "session2_gated_trajectories.csv",
            "session2_gated_trajectories.png",
            "session2_selected_token_focus_trajectories.png",
            "session2_global_confusions.png",
            "session2_activity_distance_heatmaps.png",
            "trial_bootstrap_confidence_intervals.csv",
            "frozen_label_free_gate_assignments.npz",
        ],
    }
    (output_dir / "online_hierarchical_gate_results.json").write_text(
        json.dumps(jsonable(result), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return result


def _unit_interval(value: str) -> float:
    parsed = float(value)
    if not 0.0 < parsed <= 1.0:
        raise argparse.ArgumentTypeError("value must lie in (0,1]")
    return parsed


def _minimum_ratio(value: str) -> float:
    parsed = float(value)
    if not np.isfinite(parsed) or parsed < 1.0:
        raise argparse.ArgumentTypeError("value must be finite and >=1")
    return parsed


def _nonnegative_float(value: str) -> float:
    parsed = float(value)
    if not np.isfinite(parsed) or parsed < 0.0:
        raise argparse.ArgumentTypeError("value must be finite and non-negative")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fold06 Session-2 residual-to-gravity hierarchical gate ablation."
    )
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--npz-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=500)
    parser.add_argument("--static-radius-quantile", type=_unit_interval, default=0.95)
    parser.add_argument("--minimum-token-fraction", type=_unit_interval, default=0.50)
    parser.add_argument(
        "--minimum-motion-energy-ratio", type=_minimum_ratio, default=2.0
    )
    parser.add_argument(
        "--minimum-motion-energy-gap", type=_nonnegative_float, default=0.25
    )
    parser.add_argument("--bootstrap-resamples", type=_positive_int, default=10000)
    return parser.parse_args()


def main() -> None:
    result = run(parse_args())
    print(
        json.dumps(
            jsonable(
                {
                    "output_dir": result["arguments"]["output_dir"],
                    "candidate": result["candidate"],
                    "gate_fit": result["gate_fit"],
                    "metrics": {
                        arm: result["arms"][arm]["global_metrics"] for arm in ARMS
                    },
                }
            ),
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
