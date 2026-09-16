"""Auditable figures for the frozen E0 trajectory route."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap


def _colormap(codebook_size: int) -> ListedColormap:
    base = plt.get_cmap("turbo")
    return ListedColormap(base(np.linspace(0.0, 1.0, int(codebook_size))))


def write_trajectory_records(
    records: Sequence[Mapping[str, Any]], output_dir: Path
) -> None:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    public = [
        {key: value for key, value in record.items() if not key.startswith("_")}
        for record in records
    ]
    with (output / "primitive_trajectories.jsonl").open("w", encoding="utf-8") as handle:
        for record in public:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    fields = (
        "trial_id", "subject_id", "activity_id", "activity", "prediction_raw",
        "prediction_aligned", "window_count", "primitive_count", "used_primitive_count",
        "used_primitive_ids", "primitive_sequence", "primitive_durations_samples",
    )
    with (output / "primitive_trajectories.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for record in public:
            row = {key: record.get(key, "") for key in fields}
            for key in (
                "used_primitive_ids", "primitive_sequence",
                "primitive_durations_samples",
            ):
                row[key] = " ".join(str(value) for value in record.get(key, []))
            writer.writerow(row)


def plot_trajectory_sequences(
    records: Sequence[Mapping[str, Any]],
    output_path: Path,
    *,
    codebook_size: int = 32,
    per_activity: int = 2,
) -> None:
    selected: list[Mapping[str, Any]] = []
    for activity in sorted({int(item["activity_id"]) for item in records}):
        candidates = sorted(
            (item for item in records if int(item["activity_id"]) == activity),
            key=lambda item: (int(item["subject_id"]), int(item["trial_id"])),
        )
        chosen: list[Mapping[str, Any]] = []
        subjects: set[int] = set()
        for item in candidates:
            subject = int(item["subject_id"])
            if subject not in subjects:
                chosen.append(item)
                subjects.add(subject)
            if len(chosen) >= int(per_activity):
                break
        selected.extend(chosen)
    if not selected:
        raise ValueError("No trajectory record is available for plotting.")
    columns = 180
    matrix = np.zeros((len(selected), columns), dtype=np.int64)
    for row, item in enumerate(selected):
        tokens = np.asarray(item["primitive_sequence"], dtype=np.int64)
        positions = np.minimum(
            (np.arange(columns) * len(tokens) / columns).astype(np.int64),
            len(tokens) - 1,
        )
        matrix[row] = tokens[positions]
    figure, axis = plt.subplots(
        figsize=(16, max(5.0, 0.42 * len(selected) + 1.8)),
        constrained_layout=True,
    )
    image = axis.imshow(
        matrix,
        aspect="auto",
        interpolation="nearest",
        cmap=_colormap(codebook_size),
        vmin=-0.5,
        vmax=float(codebook_size) - 0.5,
    )
    axis.set_yticks(
        np.arange(len(selected)),
        labels=[
            f"{item['activity']} | S{int(item['subject_id']):02d} | "
            f"{int(item['primitive_count'])} E0 primitives / "
            f"{int(item['used_primitive_count'])} types"
            for item in selected
        ],
        fontsize=8,
    )
    axis.set_xticks(
        np.linspace(0, columns - 1, 6),
        labels=("0", ".2", ".4", ".6", ".8", "1"),
    )
    axis.set_xlabel("Normalised trial progress")
    axis.set_title("Frozen A2 → E0/K32 motion-primitive trajectories")
    colorbar = figure.colorbar(image, ax=axis, fraction=0.025, pad=0.01)
    colorbar.set_label("Motion primitive ID")
    figure.savefig(Path(output_path), dpi=180)
    plt.close(figure)


def plot_activity_codebook_heatmap(
    records: Sequence[Mapping[str, Any]],
    output_path: Path,
    csv_path: Path,
    *,
    codebook_size: int = 32,
) -> None:
    activity_ids = sorted({int(item["activity_id"]) for item in records})
    matrix = np.zeros((len(activity_ids), int(codebook_size)), dtype=np.float64)
    names: list[str] = []
    for row, activity in enumerate(activity_ids):
        members = [item for item in records if int(item["activity_id"]) == activity]
        names.append(str(members[0]["activity"]))
        for item in members:
            matrix[row] += np.bincount(
                np.asarray(item["primitive_sequence"], dtype=np.int64),
                minlength=int(codebook_size),
            )
    matrix = np.divide(
        matrix,
        matrix.sum(axis=1, keepdims=True),
        out=np.zeros_like(matrix),
        where=matrix.sum(axis=1, keepdims=True) > 0,
    )
    with Path(csv_path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["activity_id", "activity"] + [f"P{i}" for i in range(codebook_size)])
        for activity, name, row in zip(activity_ids, names, matrix):
            writer.writerow([activity, name] + row.tolist())
    figure, axis = plt.subplots(
        figsize=(max(12.0, codebook_size * 0.35), max(5.0, len(names) * 0.55 + 1.5)),
        constrained_layout=True,
    )
    image = axis.imshow(matrix, aspect="auto", cmap="magma", vmin=0.0)
    axis.set_yticks(np.arange(len(names)), labels=names)
    ticks = np.arange(0, codebook_size, max(1, math.ceil(codebook_size / 24)))
    axis.set_xticks(ticks, labels=[f"P{i}" for i in ticks], rotation=90)
    axis.set_xlabel("Motion primitive")
    axis.set_title("Activity × E0 token fraction")
    figure.colorbar(image, ax=axis, fraction=0.025, pad=0.01)
    figure.savefig(Path(output_path), dpi=180)
    plt.close(figure)


def plot_confusion_with_unknown(
    records: Sequence[Mapping[str, Any]], output_path: Path
) -> None:
    labels = sorted({int(item["activity_id"]) for item in records})
    row_index = {label: index for index, label in enumerate(labels)}
    matrix = np.zeros((len(labels), len(labels) + 1), dtype=np.int64)
    for item in records:
        row = row_index[int(item["activity_id"])]
        prediction = int(item["prediction_aligned"])
        column = row_index.get(prediction, len(labels)) if prediction >= 0 else len(labels)
        matrix[row, column] += 1
    fractions = np.divide(
        matrix,
        matrix.sum(axis=1, keepdims=True),
        out=np.zeros_like(matrix, dtype=np.float64),
        where=matrix.sum(axis=1, keepdims=True) > 0,
    )
    names = [
        str(next(item["activity"] for item in records if int(item["activity_id"]) == label))
        for label in labels
    ]
    figure, axis = plt.subplots(figsize=(11, 9), constrained_layout=True)
    image = axis.imshow(fractions, cmap="Blues", vmin=0.0, vmax=1.0, aspect="auto")
    axis.set_xticks(np.arange(len(names) + 1), labels=names + ["Unknown"], rotation=45, ha="right")
    axis.set_yticks(np.arange(len(names)), labels=names)
    axis.set_xlabel("Predicted activity")
    axis.set_ylabel("True activity")
    axis.set_title("Strict CGCD trajectory confusion (row normalised)")
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            if matrix[row, column]:
                axis.text(
                    column, row, f"{fractions[row, column]:.2f}\n({matrix[row, column]})",
                    ha="center", va="center", fontsize=7,
                    color="white" if fractions[row, column] > 0.55 else "black",
                )
    figure.colorbar(image, ax=axis, fraction=0.035, pad=0.02)
    figure.savefig(Path(output_path), dpi=180)
    plt.close(figure)


def _overlap_average(windows: np.ndarray, starts: np.ndarray) -> np.ndarray:
    windows = np.asarray(windows, dtype=np.float64)
    starts = np.asarray(starts, dtype=np.int64)
    length = int(starts[-1]) + int(windows.shape[-1])
    signal = np.zeros((windows.shape[1], length), dtype=np.float64)
    counts = np.zeros(length, dtype=np.float64)
    for window, begin in zip(windows, starts):
        end = int(begin) + int(windows.shape[-1])
        signal[:, int(begin):end] += window
        counts[int(begin):end] += 1.0
    return signal / np.maximum(counts, 1.0)[None, :]


def plot_original_sensor_sequences(
    records: Sequence[Mapping[str, Any]],
    output_dir: Path,
    *,
    sample_rate_hz: float = 100.0,
    codebook_size: int = 32,
) -> list[str]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    selected: list[Mapping[str, Any]] = []
    for activity in sorted({int(item["activity_id"]) for item in records}):
        selected.append(
            min(
                (item for item in records if int(item["activity_id"]) == activity),
                key=lambda item: (int(item["subject_id"]), int(item["trial_id"])),
            )
        )
    colour = _colormap(codebook_size)
    paths: list[str] = []
    for item in selected:
        signal = _overlap_average(
            np.asarray(item["_raw_windows"]), np.asarray(item["_window_starts"])
        )
        seconds = np.arange(signal.shape[1]) / float(sample_rate_hz)
        figure, axes = plt.subplots(6, 1, figsize=(15, 10), sharex=True, constrained_layout=True)
        for channel, axis in enumerate(axes):
            axis.plot(seconds, signal[channel], color="#172554", linewidth=0.8)
            axis.set_ylabel(("ax", "ay", "az", "gx", "gy", "gz")[channel])
            for token, begin, end in zip(
                item["primitive_sequence"], item["primitive_starts"], item["primitive_ends"]
            ):
                axis.axvspan(
                    float(begin) / sample_rate_hz,
                    float(end) / sample_rate_hz,
                    color=colour(int(token)), alpha=0.10,
                )
        axes[-1].set_xlabel("Time (s)")
        figure.suptitle(
            f"{item['activity']} | S{int(item['subject_id']):02d} | "
            f"trial {int(item['trial_id'])} | {int(item['primitive_count'])} primitives"
        )
        path = output / f"activity_{int(item['activity_id']):02d}_trial_{int(item['trial_id'])}.png"
        figure.savefig(path, dpi=170)
        plt.close(figure)
        paths.append(str(path))
    return paths


__all__ = [
    "plot_activity_codebook_heatmap", "plot_confusion_with_unknown",
    "plot_original_sensor_sequences", "plot_trajectory_sequences",
    "write_trajectory_records",
]
