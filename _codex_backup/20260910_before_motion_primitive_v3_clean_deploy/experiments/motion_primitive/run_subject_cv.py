"""Run and aggregate subject-disjoint motion-primitive feasibility folds."""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from experiments.motion_primitive.motion_checkpoint import (
    COMMAND_ARGUMENTS_V1,
    validate_motion_encoder_checkpoint_integrity,
)

SINGLE_RUN_SCRIPT = Path(__file__).with_name("run_experiment.py")


def parse_int_list(value: str) -> list[int]:
    return sorted(set(int(token.strip()) for token in value.split(",") if token.strip()))


def find_checkpoint(cv_root: Path, fold: int, seed: int) -> Path:
    pattern = (
        f"fold_{fold:02d}/window_pretrain/seed_{seed}_offline/uschad/"
        "*/checkpoints/model_best.pt"
    )
    candidates = sorted(cv_root.glob(pattern))
    if len(candidates) != 1:
        raise RuntimeError(
            f"Expected exactly one checkpoint for fold={fold}, seed={seed}; "
            f"found {len(candidates)} using {pattern}."
        )
    return candidates[0].resolve()


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(int(chunk_size))
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _json_documents_equal(first, second) -> bool:
    """Compare JSON documents after normalising Python container types.

    Formal encoder checkpoints may retain tuples in fields such as layer
    layouts and augmentation ranges.  JSON persists those tuples as arrays,
    which are loaded back as lists.  Direct Python equality therefore rejects
    an unchanged identity after the first aggregate has been written.  A
    canonical JSON comparison preserves every serialized value while treating
    tuple/list representations of the same JSON array as equivalent.
    """

    def canonical(value) -> str:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )

    return canonical(first) == canonical(second)


def find_motion_encoder_checkpoint(
    root: Path, fold: int, seed: int, expected_profile: str
) -> Path:
    """Find a schema-v1 motion checkpoint by recorded fold/seed identity."""

    import torch

    # The trainer's canonical directory is
    # ``fold_XX_seed_<seed>_<profile>_<timestamp>`` directly below its root;
    # callers may also organise checkpoints under an extra fold directory.
    # Search recursively and rely on the recorded checkpoint identity below,
    # never on directory naming alone.
    candidates = sorted(root.rglob("motion_encoder_final.pt"))
    matched = []
    rejected = []
    for candidate in candidates:
        try:
            try:
                payload = torch.load(candidate, map_location="cpu", weights_only=False)
            except TypeError:
                payload = torch.load(candidate, map_location="cpu")
            validate_motion_encoder_checkpoint_integrity(payload)
            metadata = payload.get("experiment_metadata", {})
            training = payload.get("resolved_training_config", {})
            selection = payload.get("selection", {})
            source = payload.get("source_checkpoint", {})
            source_metadata = source.get("legacy_experiment_metadata", {})
            source_path = str(source.get("path", "")).replace("\\", "/").lower()
            identity = (
                payload.get("checkpoint_type") == "motion_primitive_encoder"
                and int(payload.get("schema_version", -1)) == 1
                and int(metadata.get("uschad_cv_fold", -1)) == int(fold)
                and int(metadata.get("motion_encoder_seed", -1)) == int(seed)
                and str(training.get("ablation_profile", "")).upper()
                == str(expected_profile).upper()
                and training.get("backbone_bn_policy") == "frozen"
                and (training.get("cp_anchor") or {}).get("source")
                == "raw_frozen_consensus"
                and selection.get("policy") == "final_epoch"
                and int(source_metadata.get("uschad_cv_fold", -1)) == int(fold)
                and f"/seed_{int(seed)}_offline/" in source_path
                and not bool(metadata.get("smoke_test", False))
            )
        except Exception as error:  # Keep every rejected candidate auditable.
            rejected.append(f"{candidate}: {type(error).__name__}: {error}")
            continue
        if identity:
            matched.append(candidate.resolve())
        else:
            rejected.append(f"{candidate}: recorded identity does not match")
    if len(matched) != 1:
        raise RuntimeError(
            "Expected exactly one schema-v1 motion checkpoint for "
            f"fold={fold}, seed={seed}, profile={expected_profile}; matched "
            f"{len(matched)} from "
            f"{len(candidates)} candidates. Rejected={rejected}."
        )
    return matched[0]


_GRID_VARIABLE_ENCODER_ARGUMENTS = frozenset(
    {
        # These identify a grid member or its execution location, not the
        # training method.  Everything else is deliberately retained so a
        # loss temperature, augmentation, optimiser or pseudo-boundary change
        # cannot silently enter one CV aggregate.
        "source_checkpoint",
        "npz_path",
        "output_dir",
        "seed",
        "device",
        "self_test",
    }
)


def _load_torch_dictionary(path: Path) -> dict:
    import torch

    try:
        value = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        value = torch.load(path, map_location="cpu")
    if not isinstance(value, dict):
        raise TypeError(
            f"Motion encoder checkpoint {path} must contain a dictionary; "
            f"got {type(value).__name__}."
        )
    return value


def motion_encoder_grid_identity(checkpoint: dict) -> dict:
    """Return the fold/seed-independent identity of one formal encoder run.

    The full command argument mapping is used instead of a hand-picked subset.
    This is intentional: parameters such as InfoNCE temperature, change-point
    quantiles, VICReg coefficients and prediction masking all change what was
    learned even when the short A0--A4 profile label is unchanged.
    """

    validate_motion_encoder_checkpoint_integrity(checkpoint)
    arguments = checkpoint.get("command_arguments")
    training = checkpoint.get("resolved_training_config")
    architecture = checkpoint.get("architecture")
    implementation = checkpoint.get("implementation_fingerprint")
    selection = checkpoint.get("selection")
    split = checkpoint.get("split_audit")
    if not isinstance(arguments, dict) or not arguments:
        raise RuntimeError("Motion checkpoint lacks complete command_arguments.")
    missing_arguments = sorted(COMMAND_ARGUMENTS_V1 - set(arguments))
    if missing_arguments:
        raise RuntimeError(
            "Motion checkpoint has truncated schema-v1 command_arguments; "
            f"missing {missing_arguments}."
        )
    if not isinstance(training, dict) or not training:
        raise RuntimeError("Motion checkpoint lacks resolved_training_config.")
    if not isinstance(architecture, dict) or not architecture:
        raise RuntimeError("Motion checkpoint lacks architecture identity.")
    if not isinstance(implementation, dict) or not implementation.get("files"):
        raise RuntimeError("Motion checkpoint lacks implementation_fingerprint files.")
    if not isinstance(selection, dict):
        raise RuntimeError("Motion checkpoint lacks selection audit.")
    if not isinstance(split, dict):
        raise RuntimeError("Motion checkpoint lacks split_audit.")

    epochs = arguments.get("epochs")
    schedule_checks = {
        "epochs_positive": (
            isinstance(epochs, int) and not isinstance(epochs, bool) and epochs > 0
        ),
        "early_stopping_disabled": arguments.get("early_stopping_patience") == 0,
        "argument_selection_is_final_epoch": (
            arguments.get("selection_policy") == "final_epoch"
        ),
        "train_smoke_limit_disabled": arguments.get("smoke_max_train_trials") == 0,
        "validation_smoke_limit_disabled": arguments.get("smoke_max_val_trials") == 0,
        "deterministic_training": arguments.get("deterministic") is True,
        "checkpoint_selection_is_final_epoch": selection.get("policy") == "final_epoch",
        "checkpoint_is_canonical_final": selection.get("file_role") == "canonical_final",
        "all_epochs_completed": (
            isinstance(epochs, int)
            and selection.get("completed_epochs") == epochs
            and selection.get("selected_epoch_1based") == epochs
        ),
        "zero_outer_test_queries": selection.get("outer_test_queries") == 0,
        "zero_outer_test_selection": (
            split.get("outer_test_sensor_windows_selected") == 0
        ),
        "zero_outer_test_forward": split.get("outer_test_model_forward_calls") == 0,
        "not_smoke_output": split.get("smoke_test") is False,
    }
    if not all(schedule_checks.values()):
        raise RuntimeError(
            "Motion encoder is not eligible for the formal CV grid: "
            f"{schedule_checks}."
        )

    recorded_npz_hashes = {
        checkpoint.get("npz_sha256"),
        (checkpoint.get("data") or {}).get("npz_sha256"),
        split.get("npz_sha256"),
    }
    if None in recorded_npz_hashes or len(recorded_npz_hashes) != 1:
        raise RuntimeError(
            "Motion checkpoint has missing or inconsistent USC-HAD NPZ SHA256 "
            f"values: {sorted(str(value) for value in recorded_npz_hashes)}."
        )
    normalized_arguments = {
        key: value
        for key, value in arguments.items()
        if key not in _GRID_VARIABLE_ENCODER_ARGUMENTS
    }
    if "ablation_profile" in normalized_arguments:
        normalized_arguments["ablation_profile"] = str(
            normalized_arguments["ablation_profile"]
        ).upper()
    return {
        "checkpoint_type": checkpoint.get("checkpoint_type"),
        "schema_version": checkpoint.get("schema_version"),
        "architecture": architecture,
        "resolved_training_config": training,
        "normalized_command_arguments": normalized_arguments,
        "implementation_fingerprint": implementation,
        "npz_sha256": next(iter(recorded_npz_hashes)),
    }


def validate_motion_encoder_grid_identity(
    checkpoints: dict[tuple[int, int], Path], expected_profile: str
) -> dict:
    """Fail before any downstream run if encoder methods differ in one grid."""

    if not checkpoints:
        raise ValueError("At least one motion encoder checkpoint is required.")
    groups: dict[str, list[tuple[int, int]]] = {}
    identities: dict[str, dict] = {}
    member_records = []
    for (fold, seed), path in sorted(checkpoints.items()):
        payload = _load_torch_dictionary(path)
        identity = motion_encoder_grid_identity(payload)
        recorded_profile = str(
            identity["resolved_training_config"].get("ablation_profile", "")
        ).upper()
        if recorded_profile != str(expected_profile).upper():
            raise RuntimeError(
                f"Motion encoder {path} records profile={recorded_profile!r}; "
                f"expected {str(expected_profile).upper()!r}."
            )
        serialized = json.dumps(identity, sort_keys=True, separators=(",", ":"))
        identity_hash = hashlib.sha256(serialized.encode("utf-8")).hexdigest()
        groups.setdefault(identity_hash, []).append((int(fold), int(seed)))
        identities[identity_hash] = identity
        member_records.append(
            {
                "fold": int(fold),
                "seed": int(seed),
                "checkpoint": str(path.resolve()),
                "checkpoint_sha256": sha256_file(path),
                "training_identity_sha256": identity_hash,
            }
        )
    if len(groups) != 1:
        raise RuntimeError(
            "Heterogeneous motion-encoder training identities would be mixed "
            f"inside one CV aggregate: {groups}."
        )
    identity_hash = next(iter(groups))
    return {
        "schema": "motion_encoder_cv_grid_identity_v1",
        "expected_profile": str(expected_profile).upper(),
        "training_identity_sha256": identity_hash,
        "training_identity": identities[identity_hash],
        "members": member_records,
    }


def nested(document: dict, *keys):
    value = document
    for key in keys:
        value = value[key]
    return value


def read_trial_records(run_dir: Path) -> list[dict]:
    records_path = run_dir / "trial_primitive_sequences.jsonl"
    records = [
        json.loads(line)
        for line in records_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not records:
        raise RuntimeError(f"No trial records found in {records_path}.")
    return records


def read_decomposition_metrics(run_dir: Path) -> dict:
    """Recover trial-level decomposition diagnostics from a completed run.

    Reading the JSONL keeps aggregation backward-compatible with runs produced
    before these diagnostics were added to the per-run summary.
    """
    records = read_trial_records(run_dir)
    unique_counts = np.asarray(
        [
            record["sequence_variants"]["full"]["unique_primitive_count"]
            for record in records
        ],
        dtype=np.int64,
    )
    run_counts = np.asarray(
        [record["sequence_variants"]["full"]["run_count"] for record in records],
        dtype=np.int64,
    )
    return {
        "decomposition_trial_count": int(len(records)),
        "unique_primitives_mean": float(np.mean(unique_counts)),
        "unique_primitives_median": float(np.median(unique_counts)),
        "multiple_primitive_trial_ratio": float(np.mean(unique_counts > 1)),
        "run_count_mean": float(np.mean(run_counts)),
        "run_count_median": float(np.median(run_counts)),
        "multiple_run_trial_ratio": float(np.mean(run_counts > 1)),
    }


def summarize_activity_decomposition(rows: list[dict]) -> list[dict]:
    summaries = []
    for activity_label in sorted(set(int(row["activity_label_1based"]) for row in rows)):
        selected = [
            row for row in rows if int(row["activity_label_1based"]) == activity_label
        ]
        unique_counts = np.asarray(
            [int(row["unique_primitive_count"]) for row in selected], dtype=np.int64
        )
        run_counts = np.asarray(
            [int(row["run_count"]) for row in selected], dtype=np.int64
        )
        summaries.append(
            {
                "activity_label_1based": activity_label,
                "activity_name": str(selected[0]["activity_name"]),
                "representation_count": int(len(selected)),
                "physical_trial_count": int(
                    len(set(str(row["trial_key"]) for row in selected))
                ),
                "multiple_primitive_trial_ratio": float(
                    np.mean(unique_counts > 1)
                ),
                "unique_primitives_mean": float(np.mean(unique_counts)),
                "unique_primitives_median": float(np.median(unique_counts)),
                "run_count_mean": float(np.mean(run_counts)),
                "run_count_median": float(np.median(run_counts)),
            }
        )
    return summaries


def extract_run_row(summary: dict, fold: int, seed: int, run_dir: Path) -> dict:
    split = summary["split_audit"]
    primitive = summary["primitive_statistics"]
    metrics = summary["sequence_association_metrics"]
    order = summary["order_shuffle_control"]["all_classes"]
    decision = summary["feasibility_decision"]
    segmentation = summary.get("segmentation_statistics", {})
    segmentation_method = segmentation.get("method", "fixed_window")
    fit_segmentation = segmentation.get("fit", {})
    eval_segmentation = segmentation.get("evaluation", {})

    def observed(representation: str) -> dict:
        return metrics[representation]["all_classes"]["observed"]

    def permutation_p(representation: str) -> float:
        return metrics[representation]["all_classes"]["margin_p_value_greater"]

    full = observed("rle_sequence_full")
    nonoverlap = observed("rle_sequence_nonoverlap")
    trimmed = observed("rle_sequence_edge_trimmed")
    histogram = observed("primitive_histogram_full")
    count_only = observed("window_count_only")
    decomposition = read_decomposition_metrics(run_dir)
    if decomposition["decomposition_trial_count"] != int(
        split["evaluation_trial_count"]
    ):
        raise RuntimeError(
            f"Trial-record count differs from summary in {run_dir}: "
            f"{decomposition['decomposition_trial_count']} != "
            f"{split['evaluation_trial_count']}."
        )
    return {
        "fold": int(fold),
        "seed": int(seed),
        "run_dir": str(run_dir),
        "primitive_segmentation": segmentation_method,
        "eval_subjects": ",".join(str(value) for value in split["eval_subjects"]),
        "fit_window_count": split["fit_window_count"],
        "fit_segment_count": fit_segmentation.get(
            "segment_count", split.get("fit_window_count")
        ),
        "evaluation_segment_count": eval_segmentation.get(
            "segment_count", split.get("evaluation_window_count")
        ),
        "evaluation_segments_per_trial_mean": eval_segmentation.get(
            "segments_per_trial_mean",
            split.get("evaluation_window_count", 0)
            / max(split.get("evaluation_trial_count", 1), 1),
        ),
        "evaluation_segment_windows_mean": eval_segmentation.get(
            "segment_windows_mean", 1.0
        ),
        "evaluation_single_segment_trial_ratio": eval_segmentation.get(
            "single_segment_trial_ratio", 0.0
        ),
        "evaluation_trial_count": split["evaluation_trial_count"],
        "codebook_utilization": primitive["eval_usage"]["utilization"],
        "max_token_share": primitive["eval_usage"]["max_token_share"],
        "effective_k": primitive["eval_usage"]["perplexity_effective_k"],
        "old_oov_ratio": nested(
            primitive,
            "oov_statistics",
            "old_classes",
            "fit_p95_exceedance_ratio",
        ),
        "novel_oov_ratio": nested(
            primitive,
            "oov_statistics",
            "novel_classes_diagnostic_only",
            "fit_p95_exceedance_ratio",
        ),
        "sequence_same_mean": full["same_mean"],
        "sequence_different_mean": full["different_mean"],
        "sequence_separation_ratio": full["separation_ratio_different_over_same"],
        "sequence_margin": full["mean_margin_different_minus_same"],
        "sequence_margin_p": permutation_p("rle_sequence_full"),
        "sequence_1nn_accuracy": full["cross_subject_1nn_activity_accuracy"],
        "nonoverlap_sequence_margin": nonoverlap["mean_margin_different_minus_same"],
        "nonoverlap_sequence_1nn_accuracy": nonoverlap[
            "cross_subject_1nn_activity_accuracy"
        ],
        "trimmed_sequence_margin": trimmed["mean_margin_different_minus_same"],
        "histogram_margin": histogram["mean_margin_different_minus_same"],
        "histogram_1nn_accuracy": histogram["cross_subject_1nn_activity_accuracy"],
        "window_count_margin": count_only["mean_margin_different_minus_same"],
        "window_count_1nn_accuracy": count_only[
            "cross_subject_1nn_activity_accuracy"
        ],
        "order_margin_gain": order.get("observed_minus_shuffled_margin"),
        "order_margin_p": order.get("margin_p_value_observed_greater"),
        "order_1nn_gain": (
            order.get("observed_1nn_accuracy", 0.0)
            - order.get("shuffled_1nn_accuracy_mean", 0.0)
        ),
        **decomposition,
        "sequence_association_gate": decision["sequence_association_gate_passed"],
        "stable_candidate_gate": decision[
            "stable_motion_primitive_candidate_gate_passed"
        ],
        "order_evidence": decision["evidence_for_order_beyond_token_composition"],
        "status": decision["status"],
    }


def numeric_summary(values: list[float]) -> dict:
    array = np.asarray(values, dtype=np.float64)
    return {
        "n": int(len(array)),
        "mean": float(np.mean(array)),
        "sample_std": float(np.std(array, ddof=1)) if len(array) > 1 else 0.0,
        "median": float(np.median(array)),
        "min": float(np.min(array)),
        "max": float(np.max(array)),
    }


def exact_sign_flip_p_greater(values: list[float]) -> float:
    """Exact one-sided sign-flip p-value for a positive paired mean."""
    values = np.asarray(values, dtype=np.float64)
    if len(values) == 0:
        raise ValueError("Sign-flip test requires values.")
    observed = float(np.mean(values))
    if len(values) > 20:
        raise ValueError("Exact sign-flip enumeration is limited to 20 blocks.")
    null = []
    for signs in itertools.product([-1.0, 1.0], repeat=len(values)):
        null.append(float(np.mean(values * np.asarray(signs))))
    null = np.asarray(null, dtype=np.float64)
    return float(np.mean(null >= observed - 1e-15))


def aggregate_rows(rows: list[dict]) -> dict:
    if not rows:
        raise ValueError("At least one completed run is required for aggregation.")
    pairs = [(int(row["fold"]), int(row["seed"])) for row in rows]
    if len(pairs) != len(set(pairs)):
        raise RuntimeError("Duplicate (fold, seed) rows were supplied.")
    segmentation_methods = {
        str(row.get("primitive_segmentation", "fixed_window")) for row in rows
    }
    if len(segmentation_methods) != 1:
        raise RuntimeError(
            f"Cannot aggregate mixed primitive segmentation methods: {segmentation_methods}."
        )
    seed_sets = {
        fold: frozenset(seed for row_fold, seed in pairs if row_fold == fold)
        for fold in sorted(set(fold for fold, _ in pairs))
    }
    if len(set(seed_sets.values())) != 1:
        raise RuntimeError(f"Unbalanced seed grid across folds: {seed_sets}.")

    numeric_fields = [
        "fit_segment_count",
        "evaluation_segment_count",
        "evaluation_segments_per_trial_mean",
        "evaluation_segment_windows_mean",
        "evaluation_single_segment_trial_ratio",
        "codebook_utilization",
        "max_token_share",
        "effective_k",
        "old_oov_ratio",
        "novel_oov_ratio",
        "sequence_same_mean",
        "sequence_different_mean",
        "sequence_separation_ratio",
        "sequence_margin",
        "sequence_1nn_accuracy",
        "nonoverlap_sequence_margin",
        "nonoverlap_sequence_1nn_accuracy",
        "trimmed_sequence_margin",
        "histogram_margin",
        "histogram_1nn_accuracy",
        "window_count_margin",
        "window_count_1nn_accuracy",
        "order_margin_gain",
        "order_1nn_gain",
        "unique_primitives_mean",
        "unique_primitives_median",
        "multiple_primitive_trial_ratio",
        "run_count_mean",
        "run_count_median",
        "multiple_run_trial_ratio",
    ]
    run_summary = {
        field: numeric_summary([float(row[field]) for row in rows])
        for field in numeric_fields
    }

    fold_rows = []
    for fold in sorted(set(int(row["fold"]) for row in rows)):
        selected = [row for row in rows if int(row["fold"]) == fold]
        fold_row = {"fold": fold, "seeds": [int(row["seed"]) for row in selected]}
        for field in numeric_fields:
            fold_row[field] = float(np.mean([float(row[field]) for row in selected]))
        fold_rows.append(fold_row)

    primary_tests = {}
    for field in [
        "sequence_margin",
        "nonoverlap_sequence_margin",
        "trimmed_sequence_margin",
        "order_margin_gain",
    ]:
        values = [row[field] for row in fold_rows]
        primary_tests[field] = {
            "fold_block_values": values,
            "mean": float(np.mean(values)),
            "exact_sign_flip_p_greater": exact_sign_flip_p_greater(values),
            "positive_fold_count": int(np.sum(np.asarray(values) > 0)),
            "fold_count": int(len(values)),
        }
    gate_fields = {
        "sequence_association": "sequence_association_gate",
        "stable_candidate": "stable_candidate_gate",
        "order_evidence": "order_evidence",
    }
    run_level_gates = {"run_count": int(len(rows))}
    fold_level_gates = {"fold_count": int(len(fold_rows))}
    for output_name, row_field in gate_fields.items():
        run_level_gates[f"{output_name}_passed"] = int(
            sum(bool(row[row_field]) for row in rows)
        )
        per_fold = [
            [bool(row[row_field]) for row in rows if int(row["fold"]) == fold]
            for fold in sorted(seed_sets)
        ]
        fold_level_gates[f"{output_name}_all_seeds_passed"] = int(
            sum(all(values) for values in per_fold)
        )
        fold_level_gates[f"{output_name}_any_seed_passed"] = int(
            sum(any(values) for values in per_fold)
        )

    return {
        "primitive_segmentation": next(iter(segmentation_methods)),
        "run_count": int(len(rows)),
        "fold_count": int(len(fold_rows)),
        "run_level_summary": run_summary,
        "fold_seed_averages": fold_rows,
        "fold_block_sign_flip_tests": primary_tests,
        "run_level_gate_counts": run_level_gates,
        "fold_level_gate_consistency": fold_level_gates,
        "interpretation": (
            "Run-level summaries and gate counts are descriptive. Folds, not seeds, "
            "are the blocks in the exact sign-flip tests; the inference still assumes "
            "sign symmetry and is limited by overlapping fold training sets. This remains "
            "a representation diagnostic, not a CGCD evaluation."
        ),
    }


def write_rows(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def read_activity_matrix(path: Path) -> tuple[list[str], np.ndarray]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.reader(handle))
    names = rows[0][1:]
    row_names = [row[0] for row in rows[1:]]
    if row_names != names:
        raise RuntimeError(f"Activity matrix rows/columns differ in {path}.")
    matrix = np.asarray([[float(value) for value in row[1:]] for row in rows[1:]])
    return names, matrix


def write_activity_matrix(path: Path, names: list[str], matrix: np.ndarray) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["activity"] + names)
        for name, row in zip(names, matrix):
            writer.writerow([name] + [f"{float(value):.8f}" for value in row])


def save_activity_heatmap(
    path: Path,
    names: list[str],
    matrix: np.ndarray,
    primitive_segmentation: str = "fixed_window",
) -> bool:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as error:
        print(f"[warning] aggregate heatmap skipped: {error}", flush=True)
        return False
    figure, axis = plt.subplots(figsize=(10, 8))
    image = axis.imshow(matrix, cmap="viridis", vmin=0.0, vmax=1.0)
    axis.set_xticks(range(len(names)), names, rotation=55, ha="right")
    axis.set_yticks(range(len(names)), names)
    axis.set_title(
        "Mean cross-subject RLE sequence distance across runs\n"
        f"segmentation={primitive_segmentation}"
    )
    figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)
    return True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Subject-CV wrapper for motion primitive feasibility runs.")
    parser.add_argument("--cv-root", required=True)
    parser.add_argument(
        "--motion-encoder-root",
        default="",
        help=(
            "Root recursively containing motion_encoder_final.pt files. Required for "
            "motion_encoder_changepoint and also selects its matched "
            "fixed-window control when used with fixed_window."
        ),
    )
    parser.add_argument(
        "--expected-encoder-profile",
        choices=["A0", "A1", "A2", "A3", "A4", "CUSTOM", "a0", "a1", "a2", "a3", "a4", "custom"],
        default="",
        help="Required with --motion-encoder-root; prevents mixed encoder ablations.",
    )
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--folds", default="1,2,3,4,5,6,7")
    parser.add_argument("--seeds", default="0")
    parser.add_argument("--primitive-num", type=int, default=32)
    parser.add_argument(
        "--primitive-segmentation",
        choices=[
            "fixed_window",
            "ssl_feature_changepoint",
            "motion_encoder_changepoint",
        ],
        default="fixed_window",
    )
    parser.add_argument("--pca-dim", type=int, default=64)
    parser.add_argument("--label-permutations", type=int, default=1000)
    parser.add_argument("--order-shuffles", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--ssl-feature-dim", type=int, default=64)
    parser.add_argument("--ssl-epochs", type=int, default=25)
    parser.add_argument("--ssl-learning-rate", type=float, default=1e-3)
    parser.add_argument("--ssl-mask-ratio", type=float, default=0.15)
    parser.add_argument("--ssl-noise-std", type=float, default=0.02)
    parser.add_argument("--changepoint-context-windows", type=int, default=2)
    parser.add_argument("--changepoint-score-quantile", type=float, default=0.90)
    parser.add_argument("--changepoint-min-segment-windows", type=int, default=2)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--anomaly-policy", choices=["report", "exclude"], default="report")
    parser.add_argument("--skip-existing", action="store_true")
    return parser.parse_args()


def validate_existing_run(
    run_dir: Path,
    checkpoint: Path,
    fold: int,
    seed: int,
    args: argparse.Namespace,
) -> None:
    """Refuse to mix stale runs whose configuration differs from this CV call."""
    required_files = [
        "summary.json",
        "experiment_config.json",
        "trial_primitive_sequences.jsonl",
        "activity_sequence_distance_matrix.csv",
    ]
    missing = [name for name in required_files if not (run_dir / name).is_file()]
    if missing:
        raise RuntimeError(f"Existing run {run_dir} is incomplete; missing {missing}.")
    config = json.loads(
        (run_dir / "experiment_config.json").read_text(encoding="utf-8")
    )
    actual_checkpoint = Path(config["checkpoint"]).expanduser().resolve()
    if actual_checkpoint != checkpoint.resolve():
        raise RuntimeError(
            f"Existing run checkpoint mismatch for fold={fold}, seed={seed}: "
            f"{actual_checkpoint} != {checkpoint.resolve()}."
        )
    actual_checkpoint_hash = config.get("checkpoint_sha256")
    expected_checkpoint_hash = sha256_file(checkpoint)
    if actual_checkpoint_hash is not None and actual_checkpoint_hash != expected_checkpoint_hash:
        raise RuntimeError(
            f"Existing run checkpoint content mismatch for fold={fold}, seed={seed}: "
            f"{actual_checkpoint_hash} != {expected_checkpoint_hash}."
        )
    expects_motion_checkpoint = bool(
        str(getattr(args, "motion_encoder_root", "")).strip()
    )
    if expects_motion_checkpoint:
        if actual_checkpoint_hash is None:
            raise RuntimeError(
                "A motion-encoder run cannot be reused without checkpoint_sha256."
            )
        if config.get("checkpoint_type") != "motion_primitive_encoder":
            raise RuntimeError(
                "A motion-encoder run cannot be reused with checkpoint_type="
                f"{config.get('checkpoint_type')!r}."
            )
        actual_profile = str(
            config.get("encoder_training", {}).get("ablation_profile", "")
        ).upper()
        expected_profile = str(args.expected_encoder_profile).upper()
        if actual_profile != expected_profile:
            raise RuntimeError(
                "A motion-encoder run cannot be reused with encoder profile "
                f"{actual_profile!r}; expected {expected_profile!r}."
            )
    expected_arguments = {
        "seed": int(seed),
        "primitive_num": int(args.primitive_num),
        "pca_dim": int(args.pca_dim),
        "label_permutations": int(args.label_permutations),
        "order_shuffles": int(args.order_shuffles),
        "batch_size": int(args.batch_size),
        "device": str(args.device),
        "anomaly_policy": str(args.anomaly_policy),
        "old_class_count": 6,
        "embedding_normalization": "l2",
        "codebook_weighting": "per_trial",
        "kmeans_n_init": 20,
        "kmeans_max_iter": 300,
        "edge_trim_ratio": 0.10,
        "sample_rate_hz": 100.0,
    }
    actual_arguments = config.get("arguments", {})
    actual_segmentation = actual_arguments.get(
        "primitive_segmentation", "fixed_window"
    )
    expected_segmentation = str(
        getattr(args, "primitive_segmentation", "fixed_window")
    )
    expected_arguments["primitive_segmentation"] = expected_segmentation
    if expected_segmentation == "ssl_feature_changepoint":
        expected_arguments.update(
            {
                "ssl_feature_dim": int(args.ssl_feature_dim),
                "ssl_epochs": int(args.ssl_epochs),
                "ssl_learning_rate": float(args.ssl_learning_rate),
                "ssl_mask_ratio": float(args.ssl_mask_ratio),
                "ssl_noise_std": float(args.ssl_noise_std),
            }
        )
    if expected_segmentation in {
        "ssl_feature_changepoint",
        "motion_encoder_changepoint",
    }:
        expected_arguments.update(
            {
                "changepoint_context_windows": int(
                    args.changepoint_context_windows
                ),
                "changepoint_score_quantile": float(
                    args.changepoint_score_quantile
                ),
                "changepoint_min_segment_windows": int(
                    args.changepoint_min_segment_windows
                ),
            }
        )
    def actual_argument(key: str):
        if key == "primitive_segmentation":
            return actual_segmentation
        return actual_arguments.get(key)

    mismatches = {
        key: {"expected": expected, "actual": actual_argument(key)}
        for key, expected in expected_arguments.items()
        if actual_argument(key) != expected
    }
    if actual_segmentation != expected_segmentation:
        mismatches["primitive_segmentation"] = {
            "expected": expected_segmentation,
            "actual": actual_segmentation,
        }
    if expected_segmentation in {
        "ssl_feature_changepoint",
        "motion_encoder_changepoint",
    }:
        method_files = [
            "segmentation_statistics.json",
            "segment_embeddings_and_tokens.npz",
            "activity_trial_token_sequences.png",
        ]
        if expected_segmentation == "ssl_feature_changepoint":
            method_files.append("ssl_feature_adapter.pt")
        method_missing = [
            name for name in method_files if not (run_dir / name).is_file()
        ]
        if method_missing:
            mismatches["segmentation_output_files"] = {
                "expected": method_files,
                "actual_missing": method_missing,
            }
    algorithm = config.get("order_shuffle_control", {}).get("algorithm")
    if algorithm != "valid_rle_permutation_v2":
        mismatches["order_shuffle_control.algorithm"] = {
            "expected": "valid_rle_permutation_v2",
            "actual": algorithm,
        }
    if mismatches:
        raise RuntimeError(
            f"Refusing --skip-existing for incompatible run {run_dir}: {mismatches}."
        )


def main() -> None:
    args = parse_args()
    if args.label_permutations <= 0 or args.order_shuffles <= 0:
        raise ValueError(
            "The CV wrapper requires positive label permutations and order shuffles."
        )
    if args.primitive_num < 2 or args.pca_dim < 0 or args.batch_size < 2:
        raise ValueError("Primitive number/batch size must be >=2 and PCA dim non-negative.")
    if args.ssl_feature_dim < 1 or args.ssl_epochs < 1 or args.ssl_learning_rate <= 0:
        raise ValueError("SSL feature dimension, epochs, and learning rate must be positive.")
    if not 0.0 <= args.ssl_mask_ratio < 1.0 or args.ssl_noise_std < 0.0:
        raise ValueError("SSL mask ratio must be in [0,1) and noise std non-negative.")
    if (
        args.changepoint_context_windows < 1
        or args.changepoint_min_segment_windows < 1
        or not 0.0 <= args.changepoint_score_quantile <= 1.0
    ):
        raise ValueError("Invalid change-point context/minimum/quantile configuration.")
    cv_root = Path(args.cv_root).expanduser().resolve()
    motion_encoder_root = (
        Path(args.motion_encoder_root).expanduser().resolve()
        if str(args.motion_encoder_root).strip()
        else None
    )
    if args.primitive_segmentation == "motion_encoder_changepoint" and motion_encoder_root is None:
        raise ValueError(
            "--motion-encoder-root is required for motion_encoder_changepoint."
        )
    if motion_encoder_root is not None and not str(args.expected_encoder_profile).strip():
        raise ValueError(
            "--expected-encoder-profile is required with --motion-encoder-root."
        )
    if args.primitive_segmentation == "ssl_feature_changepoint" and motion_encoder_root is not None:
        raise ValueError(
            "The legacy ssl_feature_changepoint protocol cannot use "
            "--motion-encoder-root."
        )
    folds = parse_int_list(args.folds)
    seeds = parse_int_list(args.seeds)
    if not folds or not seeds:
        raise ValueError("At least one fold and one seed are required.")
    checkpoint_grid = {
        (fold, seed): (
            find_motion_encoder_checkpoint(
                motion_encoder_root,
                fold,
                seed,
                str(args.expected_encoder_profile),
            )
            if motion_encoder_root is not None
            else find_checkpoint(cv_root, fold, seed)
        )
        for fold in folds
        for seed in seeds
    }
    encoder_grid_audit = (
        validate_motion_encoder_grid_identity(
            checkpoint_grid, str(args.expected_encoder_profile)
        )
        if motion_encoder_root is not None
        else None
    )
    output_root = Path(args.output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    if encoder_grid_audit is not None:
        audit_path = output_root / "motion_encoder_grid_identity.json"
        if audit_path.exists():
            existing_audit = json.loads(audit_path.read_text(encoding="utf-8"))
            if not _json_documents_equal(existing_audit, encoder_grid_audit):
                raise RuntimeError(
                    f"Output root {output_root} already records a different "
                    "motion-encoder grid identity/member set. Use a new "
                    "--output-root rather than mixing results."
                )
        else:
            audit_path.write_text(
                json.dumps(encoder_grid_audit, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
    rows = []
    decomposition_rows = []
    activity_names = None
    activity_matrices_by_fold = {fold: [] for fold in folds}
    if args.primitive_segmentation == "fixed_window" and motion_encoder_root is not None:
        segmentation_suffix = (
            "_motion_encoder_fixed_window_"
            f"{str(args.expected_encoder_profile).lower()}"
        )
    elif args.primitive_segmentation == "motion_encoder_changepoint":
        segmentation_suffix = (
            "_motion_encoder_changepoint_"
            f"{str(args.expected_encoder_profile).lower()}"
        )
    else:
        segmentation_suffix = (
            "" if args.primitive_segmentation == "fixed_window"
            else f"_{args.primitive_segmentation}"
        )
    for fold in folds:
        for seed in seeds:
            checkpoint = checkpoint_grid[(fold, seed)]
            run_dir = output_root / (
                f"fold_{fold:02d}_seed_{seed}_k{args.primitive_num}"
                f"{segmentation_suffix}"
            )
            summary_path = run_dir / "summary.json"
            if summary_path.exists() and args.skip_existing:
                validate_existing_run(run_dir, checkpoint, fold, seed, args)
                print(f"[skip] {run_dir}", flush=True)
            else:
                command = [
                    sys.executable,
                    str(SINGLE_RUN_SCRIPT),
                    "--checkpoint",
                    str(checkpoint),
                    "--output-dir",
                    str(run_dir),
                    "--primitive-num",
                    str(args.primitive_num),
                    "--primitive-segmentation",
                    str(args.primitive_segmentation),
                    "--pca-dim",
                    str(args.pca_dim),
                    "--label-permutations",
                    str(args.label_permutations),
                    "--order-shuffles",
                    str(args.order_shuffles),
                    "--batch-size",
                    str(args.batch_size),
                    "--ssl-feature-dim",
                    str(args.ssl_feature_dim),
                    "--ssl-epochs",
                    str(args.ssl_epochs),
                    "--ssl-learning-rate",
                    str(args.ssl_learning_rate),
                    "--ssl-mask-ratio",
                    str(args.ssl_mask_ratio),
                    "--ssl-noise-std",
                    str(args.ssl_noise_std),
                    "--changepoint-context-windows",
                    str(args.changepoint_context_windows),
                    "--changepoint-score-quantile",
                    str(args.changepoint_score_quantile),
                    "--changepoint-min-segment-windows",
                    str(args.changepoint_min_segment_windows),
                    "--device",
                    str(args.device),
                    "--anomaly-policy",
                    str(args.anomaly_policy),
                    "--seed",
                    str(seed),
                ]
                print(f"[run] fold={fold} seed={seed} checkpoint={checkpoint}", flush=True)
                subprocess.run(command, cwd=PROJECT_ROOT, check=True)
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            rows.append(extract_run_row(summary, fold, seed, run_dir))
            for record in read_trial_records(run_dir):
                full = record["sequence_variants"]["full"]
                decomposition_rows.append(
                    {
                        "fold": int(fold),
                        "seed": int(seed),
                        "trial_key": record["trial_key"],
                        "activity_label_1based": int(record["activity_label_1based"]),
                        "activity_name": record["activity_name"],
                        "unique_primitive_count": int(
                            full["unique_primitive_count"]
                        ),
                        "run_count": int(full["run_count"]),
                    }
                )
            names, matrix = read_activity_matrix(
                run_dir / "activity_sequence_distance_matrix.csv"
            )
            if activity_names is None:
                activity_names = names
            elif names != activity_names:
                raise RuntimeError("Activity name order differs across runs.")
            activity_matrices_by_fold[fold].append(matrix)

    write_rows(output_root / "cv_runs.csv", rows)
    aggregation = aggregate_rows(rows)
    activity_decomposition = summarize_activity_decomposition(decomposition_rows)
    write_rows(
        output_root / "activity_decomposition_summary.csv",
        activity_decomposition,
    )
    aggregation["activity_decomposition_summary"] = activity_decomposition
    aggregation["activity_decomposition_interpretation"] = (
        "Each physical held-out trial appears once per encoder seed; these counts are "
        "descriptive tokenization replicates, not independent trials."
    )
    fold_mean_activity_matrices = [
        np.mean(np.stack(activity_matrices_by_fold[fold], axis=0), axis=0)
        for fold in folds
    ]
    mean_activity_matrix = np.mean(
        np.stack(fold_mean_activity_matrices, axis=0), axis=0
    )
    write_activity_matrix(
        output_root / "mean_activity_sequence_distance_matrix.csv",
        activity_names,
        mean_activity_matrix,
    )
    aggregation["mean_activity_heatmap_saved"] = save_activity_heatmap(
        output_root / "mean_activity_sequence_distance_heatmap.png",
        activity_names,
        mean_activity_matrix,
        args.primitive_segmentation,
    )
    aggregation["primitive_segmentation"] = args.primitive_segmentation
    if encoder_grid_audit is not None:
        aggregation["motion_encoder_training_identity_sha256"] = (
            encoder_grid_audit["training_identity_sha256"]
        )
        aggregation["motion_encoder_grid_audit_file"] = str(
            (output_root / "motion_encoder_grid_identity.json").resolve()
        )
    aggregation["trajectory_plot_scope"] = (
        "Per-run activity_trial_token_sequences.png files are generated. Primitive "
        "IDs are not aligned across fold/seed runs, so token trajectories are not "
        "averaged; compare the same fold/seed against the fixed-window baseline."
    )
    aggregation["activity_matrix_aggregation"] = (
        "mean over seeds within each fold, followed by an equal-weight mean over folds"
    )
    (output_root / "cv_summary.json").write_text(
        json.dumps(aggregation, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(aggregation, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
