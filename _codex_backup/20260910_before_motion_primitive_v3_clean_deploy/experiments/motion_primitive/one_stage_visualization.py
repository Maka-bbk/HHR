"""Headless plots for one-stage motion-primitive trajectory experiments."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _prepare_path(path: str | Path) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    return output


def save_confusion_heatmap(
    confusion: np.ndarray,
    class_names: Sequence[str],
    path: str | Path,
) -> Path:
    matrix = np.asarray(confusion, dtype=np.float64)
    names = [str(item) for item in class_names]
    if matrix.shape != (len(names), len(names)):
        raise ValueError("Confusion matrix and class names disagree.")
    row_sums = matrix.sum(axis=1, keepdims=True)
    normalized = np.divide(
        matrix,
        row_sums,
        out=np.zeros_like(matrix),
        where=row_sums > 0,
    )
    output = _prepare_path(path)
    figure, axis = plt.subplots(figsize=(11, 9), constrained_layout=True)
    image = axis.imshow(normalized, vmin=0.0, vmax=1.0, cmap="Blues", aspect="auto")
    axis.set_xticks(np.arange(len(names)), labels=names, rotation=45, ha="right")
    axis.set_yticks(np.arange(len(names)), labels=names)
    axis.set_xlabel("Predicted activity")
    axis.set_ylabel("True activity")
    axis.set_title("Outer-test CGCD confusion matrix (row normalized)")
    for row in range(len(names)):
        for column in range(len(names)):
            value = normalized[row, column]
            axis.text(
                column,
                row,
                f"{value:.2f}",
                ha="center",
                va="center",
                fontsize=7,
                color="white" if value >= 0.55 else "black",
            )
    figure.colorbar(image, ax=axis, label="Recall fraction")
    figure.savefig(output, dpi=180)
    plt.close(figure)
    return output


def save_class_codebook_heatmap(
    token_sequences: Sequence[Sequence[int]],
    targets: Sequence[int],
    class_names: Sequence[str],
    num_codes: int,
    path: str | Path,
) -> Path:
    if len(token_sequences) != len(targets):
        raise ValueError("token_sequences and targets differ in length.")
    names = [str(item) for item in class_names]
    counts = np.zeros((len(names), int(num_codes)), dtype=np.float64)
    for sequence, target in zip(token_sequences, targets):
        target = int(target)
        values = np.asarray(sequence, dtype=np.int64)
        if not 0 <= target < len(names):
            raise ValueError("Target is outside class_names.")
        if np.any(values < 0) or np.any(values >= int(num_codes)):
            raise ValueError("A token is outside the configured codebook.")
        if len(values):
            # Each trial contributes total mass one so long trials/activities
            # cannot dominate the visual solely through duration.
            np.add.at(counts[target], values, 1.0 / len(values))
    activity_counts = np.bincount(
        np.asarray(targets, dtype=np.int64), minlength=len(names)
    ).astype(np.float64)
    values = np.divide(
        counts,
        activity_counts[:, None],
        out=np.zeros_like(counts),
        where=activity_counts[:, None] > 0,
    )
    output = _prepare_path(path)
    figure, axis = plt.subplots(figsize=(15, 6.5), constrained_layout=True)
    image = axis.imshow(values, cmap="magma", aspect="auto", vmin=0.0)
    axis.set_xticks(np.arange(int(num_codes)))
    axis.set_xticklabels([str(index) for index in range(int(num_codes))], fontsize=7)
    axis.set_yticks(np.arange(len(names)), labels=names)
    axis.set_xlabel("Motion-primitive code id")
    axis.set_ylabel("Original activity")
    axis.set_title("Trial-equal activity × codebook occupancy")
    figure.colorbar(image, ax=axis, label="Mean within-trial token fraction")
    figure.savefig(output, dpi=180)
    plt.close(figure)
    return output


def _representative_indices(
    targets: np.ndarray,
    subjects: np.ndarray,
    per_class: int,
) -> list[int]:
    chosen: list[int] = []
    for target in sorted(np.unique(targets).tolist()):
        indices = np.flatnonzero(targets == target)
        used_subjects: set[int] = set()
        for index in indices:
            subject = int(subjects[index])
            if subject in used_subjects:
                continue
            chosen.append(int(index))
            used_subjects.add(subject)
            if len(used_subjects) >= int(per_class):
                break
    return chosen


def save_representative_trajectory_panel(
    token_sequences: Sequence[Sequence[int]],
    targets: Sequence[int],
    predictions: Sequence[int],
    subject_ids: Sequence[int],
    trial_numbers: Sequence[int],
    class_names: Sequence[str],
    num_codes: int,
    path: str | Path,
    *,
    trials_per_class: int = 2,
) -> Path:
    size = len(token_sequences)
    arrays = [
        np.asarray(targets, dtype=np.int64),
        np.asarray(predictions, dtype=np.int64),
        np.asarray(subject_ids, dtype=np.int64),
        np.asarray(trial_numbers, dtype=np.int64),
    ]
    if any(len(item) != size for item in arrays):
        raise ValueError("Trajectory-panel metadata lengths disagree.")
    truth, predicted, subjects, trials = arrays
    names = [str(item) for item in class_names]
    selected = _representative_indices(truth, subjects, int(trials_per_class))
    if not selected:
        raise ValueError("No representative trajectories were available.")
    normalized_width = 240
    canvas = np.full((len(selected), normalized_width), np.nan, dtype=np.float64)
    labels: list[str] = []
    for row, index in enumerate(selected):
        sequence = np.asarray(token_sequences[index], dtype=np.int64)
        if not len(sequence):
            raise ValueError("A representative token trajectory is empty.")
        positions = np.minimum(
            (np.arange(normalized_width) * len(sequence) / normalized_width).astype(int),
            len(sequence) - 1,
        )
        canvas[row] = sequence[positions]
        true_id = int(truth[index])
        predicted_id = int(predicted[index])
        labels.append(
            f"{names[true_id]} | S{int(subjects[index]):02d} T{int(trials[index])} | "
            f"pred={names[predicted_id]} | frames={len(sequence)}"
        )
    output = _prepare_path(path)
    height = max(7.0, 0.38 * len(selected) + 2.0)
    figure, axis = plt.subplots(figsize=(16, height), constrained_layout=True)
    color_map = plt.get_cmap("turbo", int(num_codes))
    image = axis.imshow(
        canvas,
        interpolation="nearest",
        aspect="auto",
        cmap=color_map,
        vmin=-0.5,
        vmax=int(num_codes) - 0.5,
    )
    axis.set_yticks(np.arange(len(labels)), labels=labels, fontsize=8)
    axis.set_xticks([0, 60, 120, 180, 239], labels=["0%", "25%", "50%", "75%", "100%"])
    axis.set_xlabel("Normalized position within complete trial")
    axis.set_ylabel("Original activity / subject / trial / prediction")
    axis.set_title("Representative hard motion-primitive trajectories")
    ticks = np.arange(int(num_codes))
    colorbar = figure.colorbar(image, ax=axis, ticks=ticks)
    colorbar.set_label("Motion-primitive code id")
    colorbar.ax.tick_params(labelsize=6)
    figure.savefig(output, dpi=180)
    plt.close(figure)
    return output


__all__ = [
    "save_class_codebook_heatmap",
    "save_confusion_heatmap",
    "save_representative_trajectory_panel",
]
