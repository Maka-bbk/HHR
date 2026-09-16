"""Create paired visual diagnostics for fixed-window and changepoint runs.

The two methods learn independent KMeans codebooks.  Primitive IDs are therefore
kept local to each panel and must never be interpreted as aligned semantics.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def read_records(path: Path) -> list[dict]:
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                records.append(json.loads(line))
    if not records:
        raise RuntimeError(f"No trial records found in {path}.")
    return sorted(
        records,
        key=lambda record: (
            int(record["activity_label_1based"]),
            int(record["subject_id"]),
            int(record["trial_number"]),
        ),
    )


def validate_paired_records(fixed: list[dict], changepoint: list[dict]) -> None:
    if len(fixed) != len(changepoint):
        raise RuntimeError(
            f"Trial count mismatch: fixed={len(fixed)}, changepoint={len(changepoint)}."
        )
    for left, right in zip(fixed, changepoint):
        if left["trial_key"] != right["trial_key"]:
            raise RuntimeError(
                f"Trial ordering mismatch: {left['trial_key']} != {right['trial_key']}."
            )
        if left["window_start_indices"] != right["window_start_indices"]:
            raise RuntimeError(f"Window grid mismatch for {left['trial_key']}.")
        left_tokens = left["sequence_variants"]["full"]["raw_tokens"]
        right_tokens = right["sequence_variants"]["full"]["raw_tokens"]
        if len(left_tokens) != len(right_tokens):
            raise RuntimeError(f"Window count mismatch for {left['trial_key']}.")


def token_matrix(records: list[dict]) -> np.ndarray:
    maximum = max(
        len(record["sequence_variants"]["full"]["raw_tokens"])
        for record in records
    )
    matrix = np.full((len(records), maximum), np.nan, dtype=np.float32)
    for row, record in enumerate(records):
        tokens = record["sequence_variants"]["full"]["raw_tokens"]
        matrix[row, : len(tokens)] = np.asarray(tokens, dtype=np.float32)
    return matrix


def boundary_mask(records: list[dict], width: int) -> np.ndarray:
    mask = np.zeros((len(records), width), dtype=bool)
    for row, record in enumerate(records):
        segmentation = record.get("primitive_segmentation")
        if not segmentation:
            continue
        tokens = record["sequence_variants"]["full"]["raw_tokens"]
        for segment in segmentation.get("segments", [])[1:]:
            offset = int(segment["first_window_offset"])
            if not 0 < offset < len(tokens):
                raise RuntimeError(
                    f"Invalid segment boundary {offset} in {record['trial_key']}."
                )
            mask[row, offset] = True
    return mask


def activity_groups(records: list[dict]) -> list[tuple[str, int, int]]:
    groups = []
    for label in sorted({int(record["activity_label_1based"]) for record in records}):
        indices = [
            index
            for index, record in enumerate(records)
            if int(record["activity_label_1based"]) == label
        ]
        groups.append((records[indices[0]]["activity_name"], min(indices), max(indices)))
    return groups


def rle_run_count(tokens: list[int]) -> int:
    if not tokens:
        return 0
    return 1 + sum(left != right for left, right in zip(tokens, tokens[1:]))


def method_statistics(records: list[dict], include_segments: bool) -> dict:
    token_sequences = [
        record["sequence_variants"]["full"]["raw_tokens"] for record in records
    ]
    result = {
        "trial_count": len(records),
        "subjects": sorted({int(record["subject_id"]) for record in records}),
        "mean_unique_tokens_per_trial": float(
            np.mean([len(set(tokens)) for tokens in token_sequences])
        ),
        "mean_token_rle_runs_per_trial": float(
            np.mean([rle_run_count(tokens) for tokens in token_sequences])
        ),
    }
    if include_segments:
        counts = [
            int(record["primitive_segmentation"]["segment_count"])
            for record in records
        ]
        result.update(
            {
                "mean_detected_segments_per_trial": float(np.mean(counts)),
                "median_detected_segments_per_trial": float(np.median(counts)),
                "single_detected_segment_trial_ratio": float(
                    np.mean(np.asarray(counts) == 1)
                ),
            }
        )
    return result


def save_trajectory_comparison(
    path: Path,
    fixed: list[dict],
    changepoint: list[dict],
    primitive_num: int,
    stride_seconds: float,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import BoundaryNorm, ListedColormap

    fixed_matrix = token_matrix(fixed)
    changepoint_matrix = token_matrix(changepoint)
    if fixed_matrix.shape != changepoint_matrix.shape:
        raise RuntimeError(
            f"Trajectory matrix mismatch: {fixed_matrix.shape} != "
            f"{changepoint_matrix.shape}."
        )
    boundaries = boundary_mask(changepoint, changepoint_matrix.shape[1])
    groups = activity_groups(fixed)
    norm = BoundaryNorm(
        np.arange(-0.5, int(primitive_num) + 0.5), int(primitive_num)
    )

    figure, axes = plt.subplots(1, 2, figsize=(25, 18), sharex=True, sharey=True)
    panels = [
        (axes[0], fixed_matrix, "gist_ncar", "Fixed windows + KMeans32", None),
        (
            axes[1],
            changepoint_matrix,
            "turbo",
            "SSL-feature changepoints + KMeans32",
            boundaries,
        ),
    ]
    for axis, matrix, cmap_name, title, panel_boundaries in panels:
        base = plt.get_cmap(cmap_name, int(primitive_num))
        cmap = ListedColormap([base(index) for index in range(int(primitive_num))])
        cmap.set_bad("#e5e7eb")
        image = axis.imshow(
            np.ma.masked_invalid(matrix),
            aspect="auto",
            interpolation="nearest",
            cmap=cmap,
            norm=norm,
        )
        for _, _, end in groups[:-1]:
            axis.axhline(end + 0.5, color="black", linewidth=0.9)
        if panel_boundaries is not None:
            rows, columns = np.nonzero(panel_boundaries)
            axis.scatter(
                columns - 0.5,
                rows,
                marker="|",
                s=12,
                linewidths=0.45,
                color="black",
                alpha=0.85,
                label="detected segment boundary",
            )
            axis.legend(loc="upper right", fontsize=8)
        axis.set_title(title, fontsize=14)
        axis.set_xlabel(
            f"Original window index within trial (stride {stride_seconds:.2f} s)"
        )
        colorbar = figure.colorbar(
            image,
            ax=axis,
            ticks=np.arange(int(primitive_num)),
            fraction=0.025,
            pad=0.018,
        )
        colorbar.ax.set_yticklabels(
            [f"P{index:02d}" for index in range(int(primitive_num))]
        )
        colorbar.ax.set_title("local\nID", fontsize=8)

    axes[0].set_yticks(
        [(start + end) / 2 for _, start, end in groups],
        [name for name, _, _ in groups],
    )
    axes[0].set_ylabel("Same 120 held-out activity trials")
    figure.suptitle(
        "Paired motion-primitive trajectories | fold 01, seed 50\n"
        "Primitive IDs and colors are codebook-local; they are not aligned across panels.",
        fontsize=17,
        y=0.995,
    )
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.972))
    figure.savefig(path, dpi=180)
    plt.close(figure)


def read_activity_matrix(path: Path) -> tuple[list[str], np.ndarray]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.reader(handle))
    names = rows[0][1:]
    row_names = [row[0] for row in rows[1:]]
    if names != row_names:
        raise RuntimeError(f"Activity matrix rows/columns differ in {path}.")
    matrix = np.asarray([[float(value) for value in row[1:]] for row in rows[1:]])
    return names, matrix


def save_heatmap_comparison(
    path: Path,
    names: list[str],
    fixed: np.ndarray,
    changepoint: np.ndarray,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    difference = changepoint - fixed
    difference_limit = max(0.05, float(np.max(np.abs(difference))))
    figure, axes = plt.subplots(1, 3, figsize=(24, 7.8))
    first = axes[0].imshow(fixed, cmap="viridis", vmin=0.0, vmax=1.0)
    axes[0].set_title("Fixed windows + KMeans32")
    axes[1].imshow(changepoint, cmap="viridis", vmin=0.0, vmax=1.0)
    axes[1].set_title("SSL-feature changepoints + KMeans32")
    third = axes[2].imshow(
        difference,
        cmap="coolwarm",
        vmin=-difference_limit,
        vmax=difference_limit,
    )
    axes[2].set_title("Difference: changepoint - fixed")
    for axis in axes:
        axis.set_xticks(range(len(names)), names, rotation=58, ha="right")
        axis.set_yticks(range(len(names)), names)
    figure.colorbar(first, ax=axes[:2], fraction=0.025, pad=0.015, label="RLE distance")
    figure.colorbar(third, ax=axes[2], fraction=0.046, pad=0.04, label="distance delta")
    figure.suptitle(
        "Mean cross-subject activity-sequence distance across 28 paired runs\n"
        "For the delta panel: lower diagonal and higher off-diagonal are favorable.",
        fontsize=16,
    )
    figure.subplots_adjust(left=0.06, right=0.96, bottom=0.24, top=0.84, wspace=0.30)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare fixed-window and SSL-feature changepoint results."
    )
    parser.add_argument("--fixed-run-dir", type=Path, required=True)
    parser.add_argument("--changepoint-run-dir", type=Path, required=True)
    parser.add_argument("--fixed-cv-root", type=Path, required=True)
    parser.add_argument("--changepoint-cv-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--primitive-num", type=int, default=32)
    parser.add_argument("--stride-seconds", type=float, default=1.28)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    fixed = read_records(
        args.fixed_run_dir.expanduser().resolve() / "trial_primitive_sequences.jsonl"
    )
    changepoint = read_records(
        args.changepoint_run_dir.expanduser().resolve()
        / "trial_primitive_sequences.jsonl"
    )
    validate_paired_records(fixed, changepoint)

    trajectory_path = output_dir / "fold01_seed50_trajectory_comparison.png"
    save_trajectory_comparison(
        trajectory_path,
        fixed,
        changepoint,
        int(args.primitive_num),
        float(args.stride_seconds),
    )

    fixed_names, fixed_matrix = read_activity_matrix(
        args.fixed_cv_root.expanduser().resolve()
        / "mean_activity_sequence_distance_matrix.csv"
    )
    changepoint_names, changepoint_matrix = read_activity_matrix(
        args.changepoint_cv_root.expanduser().resolve()
        / "mean_activity_sequence_distance_matrix.csv"
    )
    if fixed_names != changepoint_names:
        raise RuntimeError("Activity names differ between aggregate matrices.")
    heatmap_path = output_dir / "mean_activity_distance_comparison.png"
    save_heatmap_comparison(
        heatmap_path, fixed_names, fixed_matrix, changepoint_matrix
    )

    manifest = {
        "comparison_scope": "paired fold 01 / seed 50 trajectories and 28-run aggregate matrices",
        "primitive_id_warning": (
            "P00-P31 are independently learned in each method and are not aligned."
        ),
        "paired_trial_grid_verified": True,
        "fixed": method_statistics(fixed, include_segments=False),
        "ssl_feature_changepoint": method_statistics(
            changepoint, include_segments=True
        ),
        "trajectory_plot": str(trajectory_path),
        "heatmap_plot": str(heatmap_path),
    }
    (output_dir / "comparison_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
