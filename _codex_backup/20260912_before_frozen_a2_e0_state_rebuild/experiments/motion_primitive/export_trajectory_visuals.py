"""Export auditable motion-primitive trajectories and figures.

This is a post-training reader.  It never participates in checkpoint
selection or optimisation.  Given an offline run manifest and either an
offline or online checkpoint, it rebuilds the exact USC-HAD split, emits one
JSON/CSV row per complete trial, and renders:

* an activity-by-codebook duration heatmap;
* a cross-subject primitive trajectory panel;
* a confusion heatmap for the trajectory readout;
* one six-channel sensor trace per represented activity, with primitive
  boundaries and ids overlaid.

The sensor traces are reconstructed by overlap-averaging the stored windows.
When the fold-specific dataset is normalised, its recorded fold mean/std are
used to return the plot to sensor units before reconstruction.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import sys
from argparse import Namespace
from dataclasses import fields, replace
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
import torch
from torch.utils.data import DataLoader
from scipy.optimize import linear_sum_assignment


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.uschad_har import uschad_trial_collate  # noqa: E402
from experiments.motion_primitive.profiles import (  # noqa: E402
    PROFILE_JOINT,
    normalize_profile,
)
from experiments.motion_primitive.train_offline import (  # noqa: E402
    build_datasets,
    resolve_device,
)
from models.motion_primitive_cgcd import (  # noqa: E402
    MotionPrimitiveCGCDModel,
    MotionPrimitiveConfig,
)
from models.batch_utils import move_trial_batch_to_device  # noqa: E402


ACTIVITY_NAMES = {
    0: "Walking Forward",
    1: "Walking Left",
    2: "Walking Right",
    3: "Walking Upstairs",
    4: "Walking Downstairs",
    5: "Running Forward",
    6: "Jumping Up",
    7: "Sitting",
    8: "Standing",
    9: "Sleeping",
    10: "Elevator Up",
    11: "Elevator Down",
}
DEFAULT_CHANNEL_NAMES = (
    "acc_x",
    "acc_y",
    "acc_z",
    "gyro_x",
    "gyro_y",
    "gyro_z",
)


def _load_json(path: Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise TypeError(f"Expected a JSON object in {path}.")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    with Path(path).open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def _checkpoint_state(checkpoint: Mapping[str, Any]) -> Mapping[str, torch.Tensor]:
    state = checkpoint.get("model", checkpoint)
    if not isinstance(state, Mapping) or not state:
        raise TypeError("Checkpoint has no non-empty model state.")
    return state


def _trajectory_only_state(
    state: Mapping[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Read pure checkpoints and migrate the old shared encoder prefix only.

    Early joint checkpoints stored the same ResNet1D under ``0.window_encoder``
    and also carried an unused complete-trial branch.  For post-hoc trajectory
    auditing we copy only that encoder into the current neutral key and reject
    the historical pooling/classifier tensors by construction.
    """

    result: dict[str, torch.Tensor] = {}
    for key, value in state.items():
        if key.startswith("0.window_encoder."):
            result["window_encoder." + key[len("0.window_encoder.") :]] = value
        elif key.startswith("0.pooling.") or key.startswith("1."):
            continue
        else:
            result[key] = value
    return result


def rebuild_model(
    manifest: Mapping[str, Any], checkpoint: Mapping[str, Any]
) -> tuple[MotionPrimitiveCGCDModel, str]:
    """Reconstruct the live class/K dimensions directly from checkpoint state."""

    state = _trajectory_only_state(_checkpoint_state(checkpoint))
    profile = normalize_profile(
        str(checkpoint.get("profile", manifest.get("profile", "")))
    )
    if profile != PROFILE_JOINT:
        raise ValueError(f"Unsupported trajectory profile: {profile!r}.")
    architecture = checkpoint.get("architecture", manifest.get("architecture"))
    if not isinstance(architecture, Mapping):
        raise TypeError("Run manifest/checkpoint has no architecture object.")
    values = dict(architecture)
    head_key = "trajectory_classifier.weight"
    if head_key not in state:
        raise KeyError(f"Checkpoint is missing {head_key!r}.")
    values["old_class_count"] = int(state[head_key].shape[0])
    if "codebook.vectors" not in state:
        raise KeyError("Trajectory checkpoint is missing codebook.vectors.")
    values["codebook_size"] = int(state["codebook.vectors"].shape[0])
    known = {field.name for field in fields(MotionPrimitiveConfig)}
    legacy_unused = {
        "pool_quantile",
        "pool_fusion_dim",
        "pool_dropout",
        "projection_hidden_dim",
        "projection_bottleneck_dim",
    }
    unknown = set(values) - known
    unsupported = sorted(unknown - legacy_unused)
    if unsupported:
        raise ValueError(
            "Checkpoint architecture contains unsupported fields: "
            f"{unsupported}."
        )
    values = {name: value for name, value in values.items() if name in known}
    config = MotionPrimitiveConfig(**values).validated()
    model = MotionPrimitiveCGCDModel(config)
    model.load_state_dict(dict(state), strict=True)
    return model, profile


def rebuild_datasets_from_manifest(
    manifest: Mapping[str, Any], npz_override: Optional[str] = None
) -> tuple[Mapping[str, Any], np.ndarray]:
    saved = manifest.get("arguments")
    if not isinstance(saved, Mapping):
        raise TypeError("Offline manifest has no arguments object.")
    values = dict(saved)
    if npz_override:
        values["uschad_npz_path"] = str(Path(npz_override).expanduser())
    values["old_classes_parsed"] = [
        int(value) for value in manifest.get("old_classes_physical", [])
    ]
    if not values["old_classes_parsed"]:
        raise ValueError("Offline manifest does not record old_classes_physical.")
    return build_datasets(Namespace(**values))


def _dataset_for_split(
    datasets: Mapping[str, Any], split: str, online_session: int
) -> Any:
    if split == "offline_test":
        dataset = datasets["offline_test_dataset"]
    elif split == "offline_validation":
        dataset = datasets["offline_val_dataset"]
    elif split == "offline_train":
        dataset = datasets["offline_train_dataset"]
    elif split == "online_test":
        streams = datasets["online_test_dataset_list"]
        if not 1 <= int(online_session) <= len(streams):
            raise ValueError(
                f"--online-session must lie in [1,{len(streams)}] for online_test."
            )
        dataset = streams[int(online_session) - 1]
    else:
        raise ValueError(f"Unknown split {split!r}.")
    # This is a newly reconstructed dataset, so disabling augmentation in place
    # cannot affect training state.
    dataset.transform = None
    return dataset


def _trial_metadata(dataset: Any) -> dict[int, dict[str, Any]]:
    required = (
        "trial_global_ids",
        "subject_ids",
        "trial_numbers",
        "targets",
        "_trial_window_indices",
        "window_start_indices",
        "window_data",
    )
    missing = [name for name in required if not hasattr(dataset, name)]
    if missing:
        raise TypeError(f"Dataset lacks trial provenance fields: {missing}.")
    result: dict[int, dict[str, Any]] = {}
    mean = np.asarray(dataset.normalization_mean, dtype=np.float32).reshape(
        1, int(dataset.actual_num_channels), 1
    )
    std = np.asarray(dataset.normalization_std, dtype=np.float32).reshape(
        1, int(dataset.actual_num_channels), 1
    )
    for row, trial_id_value in enumerate(dataset.trial_global_ids):
        trial_id = int(trial_id_value)
        indices = np.asarray(dataset._trial_window_indices[row], dtype=np.int64)
        normalised = np.asarray(dataset.window_data[indices], dtype=np.float32)
        sensor_windows = normalised * std + mean
        result[trial_id] = {
            "subject_id": int(dataset.subject_ids[row]),
            "trial_number": int(dataset.trial_numbers[row]),
            "physical_label": int(dataset.targets[row]),
            "starts": np.asarray(dataset.window_start_indices[indices], dtype=np.int64),
            "sensor_windows": sensor_windows,
        }
    return result


def _enrich_runs(
    run_payload: Mapping[str, Any],
    starts: np.ndarray,
    window_size: int,
    *,
    final_boundary_probabilities: Optional[Sequence[float]] = None,
    learned_boundary_probabilities: Optional[Sequence[float]] = None,
    token_change_probabilities: Optional[Sequence[float]] = None,
) -> list[dict[str, Any]]:
    """Attach sensor-time provenance without discarding run composition."""

    runs: list[dict[str, Any]] = []
    window_count = len(starts)
    for item in run_payload["runs"]:
        start_index = int(item["start_token_index"])
        end_index = int(item["end_token_index_exclusive"])
        if not 0 <= start_index < end_index <= len(starts):
            raise RuntimeError(f"Invalid primitive run {item} for {len(starts)} windows.")
        # Adjacent sensor windows overlap.  A display boundary is placed at
        # the next run's first window start (and only the last run extends to
        # the final window end), so coloured spans partition time instead of
        # visually double-counting the overlap.
        end_sample = (
            int(starts[end_index])
            if end_index < len(starts)
            else int(starts[end_index - 1] + window_size)
        )
        enter_pair = start_index - 1
        run = {
                "primitive_id": int(item.get("dominant_token", item["token"])),
                "dominant_primitive_id": int(
                    item.get("dominant_token", item["token"])
                ),
                "active_primitive_ids": [
                    int(value)
                    for value in item.get("active_tokens", [item["token"]])
                ],
                "code_distribution": [
                    float(value) for value in item.get("code_distribution", [])
                ],
                "start_window_index": start_index,
                "end_window_index_exclusive": end_index,
                "duration_windows": int(end_index - start_index),
                "duration_fraction": float((end_index - start_index) / window_count),
                "start_sample": int(starts[start_index]),
                "end_sample_exclusive": end_sample,
            }
        if enter_pair >= 0:
            if final_boundary_probabilities is not None:
                run["enter_final_boundary_probability"] = float(
                    final_boundary_probabilities[enter_pair]
                )
            if learned_boundary_probabilities is not None:
                run["enter_learned_boundary_probability"] = float(
                    learned_boundary_probabilities[enter_pair]
                )
            if token_change_probabilities is not None:
                run["enter_token_change_probability"] = float(
                    token_change_probabilities[enter_pair]
                )
        runs.append(run)
    return runs


@torch.inference_mode()
def collect_trial_records(
    model: MotionPrimitiveCGCDModel,
    dataset: Any,
    *,
    device: torch.device,
    batch_size: int,
    split: str,
) -> list[dict[str, Any]]:
    loader = DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=False,
        num_workers=0,
        collate_fn=uschad_trial_collate,
    )
    metadata = _trial_metadata(dataset)
    model = model.to(device).eval()
    records: list[dict[str, Any]] = []
    for batch in loader:
        inputs, labels, trial_ids = batch[:3]
        prepared = move_trial_batch_to_device(inputs, device)
        output = model.forward_trajectory(prepared, hard_codebook=True)
        trajectory = torch.softmax(output["trajectory_logits"], dim=-1)
        for row, trial_id_tensor in enumerate(trial_ids):
            trial_id = int(trial_id_tensor)
            source = metadata[trial_id]
            record: dict[str, Any] = {
                "trial_id": trial_id,
                "subject_id": source["subject_id"],
                "trial_number": source["trial_number"],
                "physical_label_zero_based": source["physical_label"],
                "activity": ACTIVITY_NAMES.get(
                    source["physical_label"], f"Class {source['physical_label']}"
                ),
                "model_label": int(labels[row]),
                "window_count": int(prepared["lengths"][row]),
                "window_start_samples": [int(value) for value in source["starts"]],
                "trajectory_prediction_raw": int(trajectory[row].argmax().cpu()),
                "trajectory_confidence": float(trajectory[row].max().cpu()),
                # Internal plotting payloads are removed before JSON/CSV export.
                "_sensor_windows": source["sensor_windows"],
            }
            primitive = output["primitive_runs"][row]
            window_count = int(prepared["lengths"][row])
            final_boundary = output["final_boundary_probabilities"][
                row, : max(0, window_count - 1)
            ].detach().cpu().tolist()
            learned_boundary = output["learned_boundary_probabilities"][
                row, : max(0, window_count - 1)
            ].detach().cpu().tolist()
            token_change = output["token_change_probabilities"][
                row, : max(0, window_count - 1)
            ].detach().cpu().tolist()
            tokens = [int(value) for value in primitive["tokens"]]
            run_sequence = [int(item["token"]) for item in primitive["runs"]]
            run_count = int(primitive["run_count"])
            record.update(
                {
                    "primitive_count": run_count,
                    "run_count": run_count,
                    "boundary_count": max(0, run_count - 1),
                    "boundary_rate": float(
                        max(0, run_count - 1) / max(1, window_count - 1)
                    ),
                    "used_code_ids": sorted(set(tokens)),
                    "used_code_count": len(set(tokens)),
                    "primitive_ids_per_window": tokens,
                    "window_token_sequence": tokens,
                    "primitive_sequence": run_sequence,
                    "compressed_run_sequence": run_sequence,
                    "primitive_transitions": [
                        {
                            "transition_index": index,
                            "source_primitive_id": int(source_id),
                            "target_primitive_id": int(target_id),
                        }
                        for index, (source_id, target_id) in enumerate(
                            zip(run_sequence[:-1], run_sequence[1:])
                        )
                    ],
                    "final_boundary_probabilities_between_windows": final_boundary,
                    "learned_boundary_probabilities_between_windows": learned_boundary,
                    "token_change_probabilities_between_windows": token_change,
                    "primitive_runs": _enrich_runs(
                        primitive,
                        source["starts"],
                        int(dataset.actual_window_size),
                        final_boundary_probabilities=final_boundary,
                        learned_boundary_probabilities=learned_boundary,
                        token_change_probabilities=token_change,
                    ),
                }
            )
            records.append(record)
    return records


def align_online_predictions(
    records: list[dict[str, Any]], class_count: int, old_class_count: int
) -> list[list[int]]:
    """Keep supervised old rows fixed and align only discovered novel rows."""

    truth = np.asarray([item["model_label"] for item in records], dtype=np.int64)
    raw = np.asarray(
        [item["trajectory_prediction_raw"] for item in records], dtype=np.int64
    )
    count = int(class_count)
    old = int(old_class_count)
    if not 1 <= old <= count:
        raise ValueError("old_class_count must lie in [1,class_count].")
    if np.any(truth < 0) or np.any(truth >= count):
        raise ValueError("A target lies outside the model class range.")
    if np.any(raw < 0) or np.any(raw >= count):
        raise ValueError("A trajectory prediction lies outside the model class range.")
    novel_ids = np.arange(old, count, dtype=np.int64)
    contingency = np.zeros((len(novel_ids), len(novel_ids)), dtype=np.int64)
    for target, prediction in zip(truth, raw):
        if target >= old and prediction >= old:
            contingency[int(target - old), int(prediction - old)] += 1
    pairs: list[list[int]] = []
    mapping = {int(value): int(value) for value in range(old)}
    if len(novel_ids):
        target_rows, predicted_columns = linear_sum_assignment(-contingency)
        for target_row, predicted_column in zip(target_rows, predicted_columns):
            predicted_id = int(novel_ids[predicted_column])
            target_id = int(novel_ids[target_row])
            mapping[predicted_id] = target_id
            pairs.append([predicted_id, target_id])
    for item, value in zip(records, raw.tolist()):
        item["trajectory_prediction"] = int(mapping[int(value)])
    return pairs


def _primitive_cmap(codebook_size: int) -> ListedColormap:
    base = plt.get_cmap("turbo")
    return ListedColormap(base(np.linspace(0.02, 0.98, int(codebook_size))))


def _selected_representatives(
    records: Sequence[Mapping[str, Any]], per_activity: int
) -> list[Mapping[str, Any]]:
    selected: list[Mapping[str, Any]] = []
    labels = sorted({int(item["model_label"]) for item in records})
    for label in labels:
        candidates = sorted(
            (item for item in records if int(item["model_label"]) == label),
            key=lambda item: (
                int(item["subject_id"]),
                int(item["trial_number"]),
                int(item["trial_id"]),
            ),
        )
        used_subjects: set[int] = set()
        chosen: list[Mapping[str, Any]] = []
        for item in candidates:
            subject = int(item["subject_id"])
            if subject not in used_subjects:
                chosen.append(item)
                used_subjects.add(subject)
            if len(chosen) == int(per_activity):
                break
        if len(chosen) < int(per_activity):
            for item in candidates:
                if item not in chosen:
                    chosen.append(item)
                if len(chosen) == int(per_activity):
                    break
        selected.extend(chosen)
    return selected


def plot_trajectory_panel(
    records: Sequence[Mapping[str, Any]],
    output_path: Path,
    *,
    codebook_size: int,
    per_activity: int,
) -> None:
    selected = _selected_representatives(records, per_activity)
    if not selected:
        raise RuntimeError("No trajectory records are available for plotting.")
    columns = 160
    matrix = np.zeros((len(selected), columns), dtype=np.int64)
    for row, item in enumerate(selected):
        tokens = np.asarray(item["primitive_ids_per_window"], dtype=np.int64)
        positions = np.minimum(
            (np.arange(columns) * len(tokens) / columns).astype(np.int64),
            len(tokens) - 1,
        )
        matrix[row] = tokens[positions]
    height = max(5.0, 0.34 * len(selected) + 1.8)
    figure, axis = plt.subplots(figsize=(16, height), constrained_layout=True)
    image = axis.imshow(
        matrix,
        aspect="auto",
        interpolation="nearest",
        cmap=_primitive_cmap(codebook_size),
        vmin=-0.5,
        vmax=codebook_size - 0.5,
    )
    labels = [
        f"{item['activity']} | S{int(item['subject_id']):02d} "
        f"T{int(item['trial_number'])} | {int(item['primitive_count'])} primitives"
        for item in selected
    ]
    axis.set_yticks(np.arange(len(selected)), labels=labels, fontsize=8)
    axis.set_xticks(np.linspace(0, columns - 1, 6), labels=["0", ".2", ".4", ".6", ".8", "1"])
    axis.set_xlabel("Normalised trial progress")
    axis.set_title("Motion-primitive trajectories across activities and subjects")
    for row, item in enumerate(selected):
        token_count = max(1, int(item["window_count"]))
        for run in item["primitive_runs"][1:]:
            x = float(run["start_window_index"]) / token_count * columns - 0.5
            axis.vlines(x, row - 0.45, row + 0.45, color="white", linewidth=0.55)
    ticks = np.unique(np.linspace(0, codebook_size - 1, min(codebook_size, 12)).astype(int))
    colorbar = figure.colorbar(image, ax=axis, ticks=ticks, fraction=0.025, pad=0.01)
    colorbar.set_label("Primitive id")
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def plot_activity_codebook_heatmap(
    records: Sequence[Mapping[str, Any]],
    output_path: Path,
    csv_path: Path,
    *,
    codebook_size: int,
) -> None:
    labels = sorted({int(item["model_label"]) for item in records})
    matrix = np.zeros((len(labels), int(codebook_size)), dtype=np.float64)
    names: list[str] = []
    for row, label in enumerate(labels):
        members = [item for item in records if int(item["model_label"]) == label]
        names.append(str(members[0]["activity"]))
        for item in members:
            for primitive_id in item["window_token_sequence"]:
                matrix[row, int(primitive_id)] += 1
    totals = matrix.sum(axis=1, keepdims=True)
    fractions = np.divide(matrix, totals, out=np.zeros_like(matrix), where=totals > 0)
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["model_label", "activity"] + [f"P{index}" for index in range(codebook_size)])
        for label, name, values in zip(labels, names, fractions):
            writer.writerow([label, name] + [f"{value:.10g}" for value in values])
    figure, axis = plt.subplots(
        figsize=(max(12.0, codebook_size * 0.34), max(5.0, len(labels) * 0.5 + 1.8)),
        constrained_layout=True,
    )
    image = axis.imshow(fractions, aspect="auto", cmap="magma", vmin=0.0)
    axis.set_yticks(np.arange(len(names)), labels=names)
    step = max(1, math.ceil(codebook_size / 24))
    xticks = np.arange(0, codebook_size, step)
    axis.set_xticks(xticks, labels=[f"P{value}" for value in xticks], rotation=90)
    axis.set_xlabel("Motion primitive")
    axis.set_title("Activity × codebook window-token fraction")
    colorbar = figure.colorbar(image, ax=axis, fraction=0.025, pad=0.01)
    colorbar.set_label("Fraction of activity window tokens")
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def plot_confusion(
    records: Sequence[Mapping[str, Any]], output_path: Path
) -> None:
    truth = np.asarray([item["model_label"] for item in records], dtype=np.int64)
    predictions = np.asarray(
        [
            item.get("trajectory_prediction", item["trajectory_prediction_raw"])
            for item in records
        ],
        dtype=np.int64,
    )
    labels = sorted(set(truth.tolist()) | set(predictions.tolist()))
    matrix = np.zeros((len(labels), len(labels)), dtype=np.int64)
    index = {label: offset for offset, label in enumerate(labels)}
    for target, predicted in zip(truth, predictions):
        matrix[index[int(target)], index[int(predicted)]] += 1
    row_sum = matrix.sum(axis=1, keepdims=True)
    normalised = np.divide(matrix, row_sum, out=np.zeros_like(matrix, dtype=float), where=row_sum > 0)
    model_to_activity = {
        int(item["model_label"]): str(item["activity"]) for item in records
    }
    names = [model_to_activity.get(label, f"Class {label}") for label in labels]
    figure, axis = plt.subplots(figsize=(10, 9), constrained_layout=True)
    image = axis.imshow(normalised, cmap="Blues", vmin=0.0, vmax=1.0)
    axis.set_xticks(np.arange(len(labels)), labels=names, rotation=45, ha="right")
    axis.set_yticks(np.arange(len(labels)), labels=names)
    axis.set_xlabel("Predicted activity")
    axis.set_ylabel("True activity")
    axis.set_title("Trajectory confusion matrix (row-normalised)")
    for row in range(len(labels)):
        for column in range(len(labels)):
            if matrix[row, column]:
                axis.text(
                    column,
                    row,
                    f"{normalised[row, column]:.2f}\n({matrix[row, column]})",
                    ha="center",
                    va="center",
                    fontsize=7,
                    color="white" if normalised[row, column] > 0.55 else "black",
                )
    figure.colorbar(image, ax=axis, fraction=0.04, pad=0.02)
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def _overlap_average(windows: np.ndarray, starts: Sequence[int]) -> tuple[np.ndarray, int]:
    windows = np.asarray(windows, dtype=np.float64)
    starts = np.asarray(starts, dtype=np.int64)
    if windows.ndim != 3 or len(windows) != len(starts) or len(starts) == 0:
        raise ValueError("windows/starts must be non-empty [L,C,W]/[L].")
    origin = int(starts[0])
    relative = starts - origin
    length = int(relative[-1] + windows.shape[-1])
    signal = np.zeros((windows.shape[1], length), dtype=np.float64)
    count = np.zeros(length, dtype=np.float64)
    for window, begin in zip(windows, relative):
        end = int(begin + windows.shape[-1])
        signal[:, int(begin) : end] += window
        count[int(begin) : end] += 1.0
    signal /= np.maximum(count, 1.0)[None, :]
    return signal, origin


def _safe_filename(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def plot_sensor_traces(
    records: Sequence[Mapping[str, Any]],
    output_dir: Path,
    *,
    codebook_size: int,
    channel_names: Sequence[str],
    sample_rate_hz: float,
) -> list[str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    selected = _selected_representatives(records, 1)
    colors = _primitive_cmap(codebook_size)
    paths: list[str] = []
    for item in selected:
        signal, origin = _overlap_average(
            np.asarray(item["_sensor_windows"]), item["window_start_samples"]
        )
        time = np.arange(signal.shape[1]) / float(sample_rate_hz)
        figure, axes = plt.subplots(
            signal.shape[0],
            1,
            figsize=(15, 10),
            sharex=True,
            constrained_layout=True,
        )
        axes = np.atleast_1d(axes)
        for channel, axis in enumerate(axes):
            axis.plot(time, signal[channel], color="#172554", linewidth=0.8)
            axis.set_ylabel(channel_names[channel], fontsize=8)
            axis.grid(alpha=0.18)
            for run in item["primitive_runs"]:
                begin = (int(run["start_sample"]) - origin) / float(sample_rate_hz)
                end = (int(run["end_sample_exclusive"]) - origin) / float(sample_rate_hz)
                token = int(run["primitive_id"])
                axis.axvspan(begin, end, color=colors(token), alpha=0.10)
                if channel == 0:
                    axis.text(
                        0.5 * (begin + end),
                        0.98,
                        f"P{token}",
                        transform=axis.get_xaxis_transform(),
                        ha="center",
                        va="top",
                        fontsize=7,
                    )
        axes[-1].set_xlabel("Time (s)")
        figure.suptitle(
            f"{item['activity']} | subject {item['subject_id']} trial {item['trial_number']} "
            f"| {item['primitive_count']} motion primitives",
            fontsize=12,
        )
        path = output_dir / (
            f"class_{int(item['model_label']):02d}_{_safe_filename(str(item['activity']))}.png"
        )
        figure.savefig(path, dpi=170)
        plt.close(figure)
        paths.append(str(path.resolve()))
    return paths


def _public_record(record: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in record.items() if not key.startswith("_")}


def _write_records(records: Sequence[Mapping[str, Any]], output_dir: Path) -> None:
    jsonl_path = output_dir / "primitive_trajectories.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(_public_record(record), ensure_ascii=False, sort_keys=True) + "\n")
    csv_path = output_dir / "primitive_trajectories.csv"
    fields = [
        "trial_id",
        "subject_id",
        "trial_number",
        "physical_label_zero_based",
        "activity",
        "model_label",
        "window_count",
        "primitive_count",
        "boundary_count",
        "boundary_rate",
        "used_code_count",
        "used_code_ids",
        "window_token_sequence",
        "primitive_sequence",
        "run_durations_windows",
        "trajectory_prediction_raw",
        "trajectory_prediction",
        "trajectory_confidence",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for record in records:
            row = {key: record.get(key, "") for key in fields}
            row["used_code_ids"] = " ".join(
                f"P{value}" for value in record.get("used_code_ids", [])
            )
            row["window_token_sequence"] = " ".join(
                f"P{value}" for value in record.get("window_token_sequence", [])
            )
            row["primitive_sequence"] = " ".join(
                f"P{value}" for value in record.get("primitive_sequence", [])
            )
            row["run_durations_windows"] = " ".join(
                str(int(item["duration_windows"]))
                for item in record.get("primitive_runs", [])
            )
            row["trajectory_prediction"] = record.get(
                "trajectory_prediction", record["trajectory_prediction_raw"]
            )
            writer.writerow(row)


def segmentation_summary(
    records: Sequence[Mapping[str, Any]], codebook_size: int
) -> dict[str, Any]:
    """Return fail-readable segmentation diagnostics overall and by activity."""

    def summarize(items: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        run_counts = np.asarray([item["run_count"] for item in items], dtype=np.int64)
        window_counts = np.asarray([item["window_count"] for item in items], dtype=np.int64)
        boundary_counts = np.maximum(run_counts - 1, 0)
        possible_boundaries = np.maximum(window_counts - 1, 0)
        tokens = [
            int(token)
            for item in items
            for token in item["window_token_sequence"]
        ]
        durations = [
            int(run["duration_windows"])
            for item in items
            for run in item["primitive_runs"]
        ]
        used = sorted(set(tokens))
        return {
            "trial_count": int(len(items)),
            "window_token_count": int(window_counts.sum()),
            "motion_primitive_run_count": int(run_counts.sum()),
            "boundary_count": int(boundary_counts.sum()),
            "boundary_rate": float(
                boundary_counts.sum() / max(1, possible_boundaries.sum())
            ),
            "one_run_trial_count": int(np.sum(run_counts == 1)),
            "one_run_fraction": float(np.mean(run_counts == 1)),
            "run_count_per_trial": {
                "mean": float(run_counts.mean()),
                "median": float(np.median(run_counts)),
                "minimum": int(run_counts.min()),
                "maximum": int(run_counts.max()),
            },
            "run_duration_windows": {
                "mean": float(np.mean(durations)),
                "median": float(np.median(durations)),
                "minimum": int(np.min(durations)),
                "maximum": int(np.max(durations)),
            },
            "used_code_count": int(len(used)),
            "used_code_ids": used,
            "codebook_utilization_fraction": float(len(used) / int(codebook_size)),
        }

    labels = sorted({int(item["model_label"]) for item in records})
    by_activity = {}
    for label in labels:
        members = [item for item in records if int(item["model_label"]) == label]
        by_activity[str(label)] = {
            "activity": str(members[0]["activity"]),
            **summarize(members),
        }
    overall = summarize(records)
    overall["codebook_size"] = int(codebook_size)
    overall["unused_code_ids"] = sorted(
        set(range(int(codebook_size))) - set(overall["used_code_ids"])
    )
    flags = {
        "single_code_used": overall["used_code_count"] <= 1,
        "all_trials_one_run": overall["one_run_fraction"] >= 1.0,
        "zero_boundary_rate": overall["boundary_rate"] <= 0.0,
    }
    overall["diagnostic_flags"] = flags
    overall["requires_segmentation_review"] = any(flags.values())
    return {"overall": overall, "by_activity": by_activity}


def write_codebook_usage(
    records: Sequence[Mapping[str, Any]], output_path: Path, codebook_size: int
) -> None:
    total_windows = sum(int(item["window_count"]) for item in records)
    total_runs = sum(int(item["run_count"]) for item in records)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        fields = [
            "primitive_id",
            "window_assignments",
            "window_fraction",
            "dominant_runs",
            "dominant_run_fraction",
            "trial_support",
            "subject_support",
            "activity_support",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for primitive_id in range(int(codebook_size)):
            window_count = sum(
                item["window_token_sequence"].count(primitive_id) for item in records
            )
            run_count = sum(
                sum(
                    int(run["dominant_primitive_id"] == primitive_id)
                    for run in item["primitive_runs"]
                )
                for item in records
            )
            supported = [
                item for item in records if primitive_id in item["used_code_ids"]
            ]
            writer.writerow(
                {
                    "primitive_id": primitive_id,
                    "window_assignments": window_count,
                    "window_fraction": window_count / max(1, total_windows),
                    "dominant_runs": run_count,
                    "dominant_run_fraction": run_count / max(1, total_runs),
                    "trial_support": len(supported),
                    "subject_support": len(
                        {int(item["subject_id"]) for item in supported}
                    ),
                    "activity_support": len(
                        {int(item["model_label"]) for item in supported}
                    ),
                }
            )


def write_transition_counts(
    records: Sequence[Mapping[str, Any]], output_path: Path
) -> None:
    counts: dict[tuple[int, int], int] = {}
    trial_support: dict[tuple[int, int], set[int]] = {}
    for item in records:
        observed: set[tuple[int, int]] = set()
        for transition in item["primitive_transitions"]:
            key = (
                int(transition["source_primitive_id"]),
                int(transition["target_primitive_id"]),
            )
            counts[key] = counts.get(key, 0) + 1
            observed.add(key)
        for key in observed:
            trial_support.setdefault(key, set()).add(int(item["trial_id"]))
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["source_primitive_id", "target_primitive_id", "transition_count", "trial_support"]
        )
        for key in sorted(counts):
            writer.writerow([key[0], key[1], counts[key], len(trial_support[key])])


def export(args: argparse.Namespace) -> dict[str, Any]:
    offline_run = Path(args.offline_run_dir).expanduser().resolve()
    manifest_path = offline_run / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Offline manifest not found: {manifest_path}.")
    manifest = _load_json(manifest_path)
    checkpoint_path = (
        Path(args.checkpoint).expanduser().resolve()
        if args.checkpoint
        else offline_run / "checkpoint_best_trajectory.pt"
    )
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}.")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model, profile = rebuild_model(manifest, checkpoint)
    datasets, _novel_order = rebuild_datasets_from_manifest(manifest, args.npz_path)
    dataset = _dataset_for_split(datasets, args.split, args.online_session)
    device = resolve_device(args.device)
    records = collect_trial_records(
        model,
        dataset,
        device=device,
        batch_size=args.batch_size,
        split=args.split,
    )
    assignments: list[list[int]] = []
    if args.split == "online_test":
        assignments = align_online_predictions(
            records,
            model.class_count,
            old_class_count=len(manifest["old_classes_physical"]),
        )
    output_dir = Path(args.output_dir).expanduser().resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            f"Refusing to mix visual artifacts in non-empty {output_dir}."
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_records(records, output_dir)
    codebook_size = int(model.codebook_size)
    diagnostics = segmentation_summary(records, codebook_size)
    _write_json(output_dir / "segmentation_diagnostics.json", diagnostics)
    write_codebook_usage(records, output_dir / "codebook_usage.csv", codebook_size)
    write_transition_counts(records, output_dir / "primitive_transition_counts.csv")
    plot_trajectory_panel(
        records,
        output_dir / "trajectory_sequences.png",
        codebook_size=codebook_size,
        per_activity=args.trials_per_activity,
    )
    plot_activity_codebook_heatmap(
        records,
        output_dir / "activity_codebook_heatmap.png",
        output_dir / "activity_codebook_fractions.csv",
        codebook_size=codebook_size,
    )
    plot_confusion(records, output_dir / "confusion_trajectory.png")
    names = tuple(dataset.channel_names or DEFAULT_CHANNEL_NAMES)
    if len(names) != int(dataset.actual_num_channels):
        names = tuple(f"channel_{index}" for index in range(dataset.actual_num_channels))
    sensor_paths = plot_sensor_traces(
        records,
        output_dir / "raw_sensor_sequences",
        codebook_size=codebook_size,
        channel_names=names,
        sample_rate_hz=args.sample_rate_hz,
    )
    summary = {
        "schema": "hhr_motion_primitive_visual_export_v1",
        "offline_run_dir": str(offline_run),
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": _sha256(checkpoint_path),
        "profile": profile,
        "split": args.split,
        "online_session": int(args.online_session),
        "classification_head": "trajectory",
        "trial_count": len(records),
        "activity_count": len({item["model_label"] for item in records}),
        "codebook_size": codebook_size,
        "total_motion_primitive_runs": int(
            sum(item["primitive_count"] for item in records)
        ),
        "mean_motion_primitive_runs_per_trial": float(
            np.mean([item["primitive_count"] for item in records])
        ),
        "codebook_capacity": codebook_size,
        "motion_primitive_type_count_used": diagnostics["overall"]["used_code_count"],
        "motion_primitive_type_ids_used": diagnostics["overall"]["used_code_ids"],
        "one_run_trial_count": diagnostics["overall"]["one_run_trial_count"],
        "one_run_fraction": diagnostics["overall"]["one_run_fraction"],
        "boundary_rate": diagnostics["overall"]["boundary_rate"],
        "diagnostic_flags": diagnostics["overall"]["diagnostic_flags"],
        "requires_segmentation_review": diagnostics["overall"][
            "requires_segmentation_review"
        ],
        "novel_only_hungarian_assignments_predicted_to_target": assignments,
        "sensor_trace_files": sensor_paths,
        "post_training_only": True,
        "used_for_checkpoint_selection": False,
    }
    _write_json(output_dir / "visualization_summary.json", summary)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export trajectory-only motion-primitive sequences and figures",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--offline-run-dir", required=True)
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--npz-path", default="")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--split",
        choices=("offline_train", "offline_validation", "offline_test", "online_test"),
        default="offline_test",
    )
    parser.add_argument("--online-session", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--trials-per-activity", type=int, default=2)
    parser.add_argument("--sample-rate-hz", type=float, default=100.0)
    parser.add_argument("--device", default="auto")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> dict[str, Any]:
    args = build_parser().parse_args(argv)
    if args.batch_size < 1 or args.trials_per_activity < 1:
        raise ValueError("--batch-size and --trials-per-activity must be positive.")
    if not math.isfinite(args.sample_rate_hz) or args.sample_rate_hz <= 0:
        raise ValueError("--sample-rate-hz must be positive and finite.")
    result = export(args)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return result


if __name__ == "__main__":
    main()
