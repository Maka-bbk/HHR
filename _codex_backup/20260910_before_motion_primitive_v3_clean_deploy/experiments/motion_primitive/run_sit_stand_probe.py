"""Run the registered fold06 Sitting/Standing oracle representation probe."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.motion_primitive.sit_stand_probe import (  # noqa: E402
    PASS_CORRECT_PER_DIRECTION,
    REPRESENTATIONS,
    TrialProbeFeature,
    evaluate_bidirectional_probe,
    features_to_rows,
    robust_gravity_direction,
)
from experiments.motion_primitive.trajectory_ablation import (  # noqa: E402
    SourceSignalRepository,
    jsonable,
    sha256_file,
)


PROTOCOL = {
    "name": "sit_stand_bidirectional_subject_centroid_v1",
    "scope": "oracle_novel_label_representation_diagnostic_only",
    "is_cgcd_metric": False,
    "directions": "lower subject id -> higher subject id, then reverse",
    "classifier": (
        "nearest class centroid with source-subject-only LOTO block scaling; "
        "zero LOTO spread falls back to source inter-class prototype distance"
    ),
    "tie_policy": "equal class distances split prediction credit equally",
    "representations": {
        "token_only": "expanded-window-occupancy-normalized token histogram",
        "token_residual": (
            "token histogram + expanded-window-occupancy-weighted mean "
            "(embedding - assigned center)"
        ),
        "token_gravity": "token histogram + unit median raw acceleration over central 80%",
    },
    "block_combination": "equal mean of active source-calibrated block distances",
    "zero_scale_policy": (
        "use source inter-class prototype distance when source LOTO spread is zero; "
        "a block is inactive only when both are zero; all-inactive means a tie"
    ),
    "registered_gate": "both directions correct_credit >= 8 of 10",
    "gate_threshold_correct_per_direction": PASS_CORRECT_PER_DIRECTION,
    "forbidden_shortcuts": [
        "absolute token id",
        "unnormalized trial duration/window count as a predictor",
        "target-subject scaling or hyperparameter tuning",
        "encoder/PCA/KMeans refitting",
    ],
}

REGISTERED_CV_FOLD = 6
REGISTERED_OLD_CLASS_COUNT = 6
REGISTERED_EVAL_SUBJECTS = [4, 5]


def _read_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"Expected a JSON object: {path}")
    return value


def _require_new_output_dir(path: Path) -> Path:
    path = path.resolve()
    if path.exists():
        raise FileExistsError(f"Output directory must not already exist: {path}")
    return path


def _unique_int_list(payload: dict, key: str, context: str) -> list[int]:
    raw = payload.get(key)
    if not isinstance(raw, list) or not raw:
        raise RuntimeError(f"{context}.{key} must be a non-empty list.")
    try:
        values = [int(value) for value in raw]
    except (TypeError, ValueError) as error:
        raise RuntimeError(f"{context}.{key} must contain integer IDs.") from error
    if len(values) != len(set(values)):
        raise RuntimeError(f"{context}.{key} contains duplicate IDs: {values}.")
    return sorted(values)


def _validate_registered_run_metadata(
    config: dict, split: dict
) -> tuple[list[int], list[int]]:
    """Fail closed unless artifacts implement the registered fold06 protocol."""

    metadata = config.get("checkpoint_metadata")
    arguments = config.get("arguments")
    if not isinstance(metadata, dict) or not isinstance(arguments, dict):
        raise RuntimeError("Run lacks checkpoint_metadata or arguments audit data.")
    if metadata.get("outer_test_used_during_encoder_training") is not False:
        raise RuntimeError(
            "Probe requires outer_test_used_during_encoder_training=false."
        )
    if metadata.get("smoke_test") is not False:
        raise RuntimeError("Probe rejects smoke-test motion encoders.")
    if metadata.get("uschad_cv_fold") != REGISTERED_CV_FOLD:
        raise RuntimeError(
            f"Probe is registered for USC-HAD fold {REGISTERED_CV_FOLD}."
        )
    if metadata.get("old_class_count") != REGISTERED_OLD_CLASS_COUNT:
        raise RuntimeError(
            f"Checkpoint must use old_class_count={REGISTERED_OLD_CLASS_COUNT}."
        )
    if arguments.get("old_class_count") != REGISTERED_OLD_CLASS_COUNT:
        raise RuntimeError(
            f"Downstream run must use old_class_count={REGISTERED_OLD_CLASS_COUNT}."
        )

    fit_subjects = _unique_int_list(split, "fit_subjects", "split_audit")
    eval_subjects = _unique_int_list(split, "eval_subjects", "split_audit")
    checkpoint_train = _unique_int_list(
        metadata, "uschad_train_subjects", "checkpoint_metadata"
    )
    checkpoint_test = _unique_int_list(
        metadata, "uschad_test_subjects", "checkpoint_metadata"
    )
    if len(eval_subjects) != 2 or set(fit_subjects) & set(eval_subjects):
        raise RuntimeError(
            f"Expected two disjoint eval subjects, got fit={fit_subjects}, "
            f"eval={eval_subjects}."
        )
    if eval_subjects != REGISTERED_EVAL_SUBJECTS:
        raise RuntimeError(
            f"Fold06 probe requires eval subjects {REGISTERED_EVAL_SUBJECTS}, "
            f"got {eval_subjects}."
        )
    if fit_subjects != checkpoint_train:
        raise RuntimeError(
            "Split fit subjects differ from checkpoint training subjects: "
            f"{fit_subjects} != {checkpoint_train}."
        )
    if eval_subjects != checkpoint_test:
        raise RuntimeError(
            "Split eval subjects differ from checkpoint test subjects: "
            f"{eval_subjects} != {checkpoint_test}."
        )
    expected_old_ids = list(range(REGISTERED_OLD_CLASS_COUNT))
    old_ids = _unique_int_list(split, "old_class_ids_0based", "split_audit")
    if old_ids != expected_old_ids:
        raise RuntimeError(
            f"Registered old-class IDs must be {expected_old_ids}, got {old_ids}."
        )
    return fit_subjects, eval_subjects


def _validate_global_window_indices(global_indices: np.ndarray) -> np.ndarray:
    values = np.asarray(global_indices, dtype=np.int64)
    if values.ndim != 1:
        raise RuntimeError(
            f"global_window_indices must be 1D, got shape {values.shape}."
        )
    if len(values) != len(np.unique(values)):
        raise RuntimeError("global_window_indices contains duplicate window rows.")
    return values


def _validate_trial_window_coverage(
    trial_id: int, actual: np.ndarray, expected: np.ndarray
) -> None:
    actual_values = np.asarray(actual, dtype=np.int64)
    expected_values = np.asarray(expected, dtype=np.int64)
    if actual_values.ndim != 1 or expected_values.ndim != 1:
        raise RuntimeError(f"Trial {trial_id} window indices must be 1D.")
    if not np.array_equal(np.sort(actual_values), np.sort(expected_values)):
        raise RuntimeError(
            f"Trial {trial_id} window rows differ from source NPZ coverage: "
            f"actual_count={len(actual_values)}, expected_count={len(expected_values)}."
        )


def _validate_run(run_dir: Path, npz_path: Path) -> tuple[dict, dict, Path, Path]:
    required = [
        run_dir / "experiment_config.json",
        run_dir / "split_audit.json",
        run_dir / "window_embeddings_and_tokens.npz",
        run_dir / "primitive_codebook.npz",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Run directory lacks required files: {missing}")
    config = _read_json(required[0])
    split = _read_json(required[1])
    if config.get("checkpoint_type") != "motion_primitive_encoder":
        raise RuntimeError("Probe requires a formal motion_primitive_encoder run.")
    if int(config.get("checkpoint_schema_version", -1)) != 1:
        raise RuntimeError("Probe requires motion checkpoint schema version 1.")
    _validate_registered_run_metadata(config, split)
    actual_npz_sha = sha256_file(npz_path)
    if str(config.get("npz_sha256", "")) != actual_npz_sha:
        raise RuntimeError(
            "Source NPZ SHA256 differs from the downstream run: "
            f"{actual_npz_sha} != {config.get('npz_sha256')}"
        )
    return config, split, required[2], required[3]


def load_probe_features(
    run_dir: Path, npz_path: Path
) -> tuple[list[TrialProbeFeature], dict]:
    run_dir = run_dir.resolve()
    npz_path = npz_path.resolve()
    config, split, window_path, codebook_path = _validate_run(run_dir, npz_path)
    with np.load(codebook_path, allow_pickle=False) as codebook:
        if "centers" not in codebook.files:
            raise RuntimeError("Codebook has no centers array.")
        centers = np.asarray(codebook["centers"], dtype=np.float32)
    required_fields = {
        "global_window_indices",
        "split_role",
        "subject_ids",
        "activity_labels_0based",
        "trial_numbers",
        "trial_global_ids",
        "primitive_embeddings",
        "primitive_tokens",
    }
    with np.load(window_path, allow_pickle=False) as window_data:
        missing = required_fields - set(window_data.files)
        if missing:
            raise RuntimeError(f"Window output lacks fields: {sorted(missing)}")
        arrays = {
            key: np.asarray(window_data[key]).copy() for key in required_fields
        }
    row_count = len(arrays["split_role"])
    invalid = {
        key: tuple(value.shape)
        for key, value in arrays.items()
        if value.ndim == 0 or len(value) != row_count
    }
    if invalid:
        raise RuntimeError(f"Window-aligned arrays have invalid shapes: {invalid}")
    global_window_indices = _validate_global_window_indices(
        arrays["global_window_indices"]
    )
    split_roles = np.asarray(arrays["split_role"], dtype=np.int8)
    if split_roles.ndim != 1 or not np.all(np.isin(split_roles, [0, 1])):
        raise RuntimeError("split_role must be a 1D array containing only 0 or 1.")
    embeddings = np.asarray(arrays["primitive_embeddings"], dtype=np.float32)
    tokens = np.asarray(arrays["primitive_tokens"], dtype=np.int64)
    if embeddings.ndim != 2 or centers.shape != (len(centers), embeddings.shape[1]):
        raise RuntimeError(
            f"Embedding/codebook shapes disagree: {embeddings.shape}, {centers.shape}."
        )
    if np.any(tokens < 0) or np.any(tokens >= len(centers)):
        raise RuntimeError("Primitive token lies outside the codebook.")
    source = SourceSignalRepository(npz_path)
    eval_rows = split_roles == 1
    eval_trial_ids = sorted(
        int(value) for value in np.unique(arrays["trial_global_ids"][eval_rows])
    )
    selected_trial_ids = []
    metadata_by_trial = {}
    for trial_id in eval_trial_ids:
        metadata = source.metadata(trial_id)
        activity = str(metadata["activity_name"]).strip().lower()
        if activity in {"sitting", "standing"}:
            selected_trial_ids.append(trial_id)
            metadata_by_trial[trial_id] = metadata
    if not selected_trial_ids:
        raise RuntimeError("No held-out Sitting/Standing trials were found.")
    features = []
    for trial_id in selected_trial_ids:
        indices = np.flatnonzero(
            eval_rows
            & (np.asarray(arrays["trial_global_ids"], dtype=np.int64) == trial_id)
        )
        if len(indices) == 0:
            raise RuntimeError(f"Trial {trial_id} has no expanded window rows.")
        for key in ("subject_ids", "activity_labels_0based", "trial_numbers"):
            if len(np.unique(arrays[key][indices])) != 1:
                raise RuntimeError(f"Trial {trial_id} has inconsistent {key}.")
        global_indices = global_window_indices[indices]
        expected_global = np.asarray(
            source.trial_window_global_indices(trial_id), dtype=np.int64
        )
        _validate_trial_window_coverage(trial_id, global_indices, expected_global)
        trial_tokens = tokens[indices]
        counts = np.bincount(trial_tokens, minlength=len(centers)).astype(np.float64)
        histogram = (counts / counts.sum()).astype(np.float32)
        residuals = embeddings[indices] - centers[trial_tokens]
        mean_residual = np.mean(residuals.astype(np.float64), axis=0).astype(np.float32)
        gravity = robust_gravity_direction(source.sensor(trial_id), trim_fraction=0.10)
        metadata = metadata_by_trial[trial_id]
        label = int(np.unique(arrays["activity_labels_0based"][indices])[0])
        subject = int(np.unique(arrays["subject_ids"][indices])[0])
        trial_number = int(np.unique(arrays["trial_numbers"][indices])[0])
        if (
            label != int(metadata["label"])
            or subject != int(metadata["subject_id"])
            or trial_number != int(metadata["trial_number"])
        ):
            raise RuntimeError(f"Trial {trial_id} metadata disagrees across artifacts.")
        features.append(
            TrialProbeFeature(
                trial_global_id=trial_id,
                subject_id=subject,
                activity_label=label,
                activity_name=str(metadata["activity_name"]),
                trial_number=trial_number,
                token_histogram=histogram,
                mean_quantization_residual=mean_residual,
                gravity_direction=gravity,
            )
        )
    audit = {
        "run_dir": str(run_dir),
        "primitive_segmentation": config.get("segmentation", {}).get("method"),
        "checkpoint": config.get("checkpoint"),
        "checkpoint_sha256": config.get("checkpoint_sha256"),
        "checkpoint_type": config.get("checkpoint_type"),
        "checkpoint_schema_version": config.get("checkpoint_schema_version"),
        "npz_path": str(npz_path),
        "npz_sha256": source.npz_sha256,
        "window_artifact": str(window_path),
        "window_artifact_sha256": sha256_file(window_path),
        "codebook_artifact": str(codebook_path),
        "codebook_artifact_sha256": sha256_file(codebook_path),
        "fit_subjects": split.get("fit_subjects"),
        "eval_subjects": split.get("eval_subjects"),
        "selected_trial_ids": selected_trial_ids,
        "selected_trial_count": len(selected_trial_ids),
        "raw_source": source.source_audit(selected_trial_ids),
        "trial_selection_uses_activity_names": True,
        "numeric_feature_construction_uses_class_labels": False,
        "label_use": (
            "activity names select the oracle Sitting/Standing subset; source-subject "
            "labels build prototypes; target labels are used only after prediction"
        ),
        "encoder_pca_kmeans_refit": False,
        "implementation_fingerprint": {
            "experiments/motion_primitive/sit_stand_probe.py": sha256_file(
                Path(__file__).with_name("sit_stand_probe.py")
            ),
            "experiments/motion_primitive/run_sit_stand_probe.py": sha256_file(
                Path(__file__)
            ),
        },
    }
    return features, audit


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError("Cannot write an empty CSV.")
    fieldnames = list(rows[0])
    if any(list(row) != fieldnames for row in rows):
        raise ValueError("CSV rows must have identical fields in identical order.")

    def serialize(value):
        if isinstance(value, (dict, list, tuple, np.ndarray)):
            return json.dumps(
                jsonable(value),
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
        if isinstance(value, np.generic):
            return value.item()
        return value

    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(
            {key: serialize(value) for key, value in row.items()} for row in rows
        )


def _save_accuracy_plot(path: Path, results: dict) -> None:
    labels = list(REPRESENTATIONS)
    first = []
    second = []
    macro = []
    direction_names = None
    for name in labels:
        directions = results["representations"][name]["directions"]
        if direction_names is None:
            direction_names = [
                f"S{item['source_subject']}→S{item['target_subject']}"
                for item in directions
            ]
        first.append(float(directions[0]["accuracy"]))
        second.append(float(directions[1]["accuracy"]))
        macro.append(float(results["representations"][name]["macro_direction_accuracy"]))
    x = np.arange(len(labels), dtype=np.float64)
    width = 0.24
    fig, axis = plt.subplots(figsize=(10, 5.5))
    axis.bar(x - width, first, width, label=direction_names[0])
    axis.bar(x, second, width, label=direction_names[1])
    axis.bar(x + width, macro, width, label="direction macro")
    axis.axhline(0.5, color="black", linestyle="--", linewidth=1, label="chance=0.5")
    axis.axhline(0.8, color="tab:red", linestyle=":", linewidth=1.5, label="gate=0.8")
    axis.set_ylim(0.0, 1.05)
    axis.set_ylabel("Tie-aware accuracy")
    axis.set_xticks(x, [name.replace("_", "\n") for name in labels])
    axis.set_title("Sitting vs Standing cross-subject oracle representation probe")
    axis.legend(loc="lower right")
    axis.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _save_confusion_plot(path: Path, results: dict, trials: list[TrialProbeFeature]) -> None:
    activity_by_label = {
        int(trial.activity_label): trial.activity_name for trial in trials
    }
    fig, axes = plt.subplots(len(REPRESENTATIONS), 2, figsize=(9.5, 10.5))
    for row, representation in enumerate(REPRESENTATIONS):
        directions = results["representations"][representation]["directions"]
        for column, direction in enumerate(directions):
            axis = axes[row, column]
            matrix = np.asarray(direction["confusion_counts_tie_aware"], dtype=float)
            axis.imshow(matrix, vmin=0.0, vmax=5.0, cmap="Blues")
            for y in range(2):
                for x in range(2):
                    axis.text(x, y, f"{matrix[y, x]:.1f}", ha="center", va="center")
            names = [activity_by_label[int(label)] for label in direction["class_ids"]]
            axis.set_xticks([0, 1], names, rotation=25, ha="right")
            axis.set_yticks([0, 1], names)
            axis.set_xlabel("Predicted")
            axis.set_ylabel("True")
            axis.set_title(
                f"{representation} | S{direction['source_subject']}→S{direction['target_subject']}"
            )
    fig.suptitle("Sitting/Standing confusion matrices", y=0.995)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.975), h_pad=2.0, w_pad=2.0)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _save_gravity_plot(path: Path, trials: list[TrialProbeFeature]) -> None:
    labels = sorted({trial.activity_label for trial in trials})
    subjects = sorted({trial.subject_id for trial in trials})
    activity_by_label = {
        int(trial.activity_label): trial.activity_name for trial in trials
    }
    colors = {labels[0]: "tab:blue", labels[1]: "tab:orange"}
    markers = {subjects[0]: "o", subjects[1]: "^"}
    figure = plt.figure(figsize=(8, 6.5))
    axis = figure.add_subplot(111, projection="3d")
    for subject in subjects:
        for label in labels:
            selected = [
                trial
                for trial in trials
                if trial.subject_id == subject and trial.activity_label == label
            ]
            values = np.asarray([trial.gravity_direction for trial in selected])
            axis.scatter(
                values[:, 0],
                values[:, 1],
                values[:, 2],
                color=colors[label],
                marker=markers[subject],
                s=55,
                label=f"S{subject} {activity_by_label[label]}",
            )
    axis.set_xlabel("gravity x")
    axis.set_ylabel("gravity y")
    axis.set_zlabel("gravity z")
    axis.set_title("Robust raw-acceleration gravity directions (central 80%)")
    axis.legend(loc="best")
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the strict bidirectional USC-HAD Sitting/Standing representation probe."
    )
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--npz-path", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    run_dir = Path(args.run_dir).expanduser().resolve()
    npz_path = Path(args.npz_path).expanduser().resolve()
    output_dir = _require_new_output_dir(Path(args.output_dir).expanduser())
    features, audit = load_probe_features(run_dir, npz_path)
    results = evaluate_bidirectional_probe(features)
    output_dir.mkdir(parents=True, exist_ok=False)
    payload = {
        "protocol": PROTOCOL,
        "input_audit": audit,
        "results": results,
        "interpretation_limits": [
            "Sitting/Standing labels from one outer-test subject build the class prototypes.",
            "This tests information availability, not unsupervised category discovery.",
            "Only two subjects and one fold/seed are included.",
            "A positive gravity result does not prove the motion encoder caused the gain.",
        ],
    }
    (output_dir / "sit_stand_probe_results.json").write_text(
        json.dumps(jsonable(payload), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    _write_csv(output_dir / "sit_stand_trial_features.csv", features_to_rows(features))
    prediction_rows = []
    for representation in REPRESENTATIONS:
        for direction in results["representations"][representation]["directions"]:
            for row in direction["predictions"]:
                prediction_rows.append(
                    {
                        "representation": representation,
                        "source_subject": direction["source_subject"],
                        "target_subject": direction["target_subject"],
                        **row,
                    }
                )
    _write_csv(output_dir / "sit_stand_predictions.csv", prediction_rows)
    _save_accuracy_plot(output_dir / "sit_stand_probe_accuracy.png", results)
    _save_confusion_plot(
        output_dir / "sit_stand_probe_confusions.png", results, features
    )
    _save_gravity_plot(output_dir / "sit_stand_gravity_directions.png", features)
    print(json.dumps({
        name: {
            "direction_accuracies": [
                item["accuracy"]
                for item in results["representations"][name]["directions"]
            ],
            "gate_passed": results["representations"][name]["registered_gate"]["passed"],
        }
        for name in REPRESENTATIONS
    }, indent=2))
    print(f"Probe outputs: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
