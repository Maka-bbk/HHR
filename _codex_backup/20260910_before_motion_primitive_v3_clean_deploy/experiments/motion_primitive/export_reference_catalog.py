"""Export a readable primitive catalog and original trial token sequences."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.motion_primitive.core import normalized_levenshtein


CHANNEL_NAMES = ["Acc-X", "Acc-Y", "Acc-Z", "Gyro-X", "Gyro-Y", "Gyro-Z"]


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def read_records(path: Path) -> list[dict]:
    records = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not records:
        raise RuntimeError(f"No trial records found in {path}.")
    return records


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"Cannot write an empty table to {path}.")
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def token_text(tokens: list[int]) -> str:
    return " ".join(f"P{int(token):02d}" for token in tokens)


def compact_rle(full: dict) -> str:
    return " → ".join(
        f"P{int(run['primitive_id']):02d}"
        f"[{int(run['run_length_windows'])}w/{float(run['observed_span_seconds']):.2f}s]"
        for run in full["runs"]
    )


def medoid_record(records: list[dict]) -> dict:
    sequences = [record["sequence_variants"]["full"]["rle_tokens"] for record in records]
    totals = []
    for left, sequence in enumerate(sequences):
        totals.append(
            sum(
                normalized_levenshtein(sequence, other)
                for right, other in enumerate(sequences)
                if right != left
            )
        )
    return records[int(np.argmin(np.asarray(totals, dtype=np.float64)))]


def top_activity_text(
    labels: np.ndarray,
    mask: np.ndarray,
    activity_names: list[str],
    limit: int = 3,
) -> str:
    counts = Counter(int(value) for value in labels[mask].tolist())
    total = sum(counts.values())
    if total == 0:
        return "—"
    return "; ".join(
        f"{activity_names[label]} {count / total:.1%}"
        for label, count in counts.most_common(limit)
    )


def plot_activity_sequences(
    path: Path,
    records: list[dict],
    primitive_num: int,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import BoundaryNorm, ListedColormap

    ordered = sorted(
        records,
        key=lambda record: (
            record["activity_label_1based"],
            record["subject_id"],
            record["trial_number"],
        ),
    )
    max_length = max(
        len(record["sequence_variants"]["full"]["raw_tokens"])
        for record in ordered
    )
    matrix = np.ma.masked_all((len(ordered), max_length), dtype=np.float32)
    for row, record in enumerate(ordered):
        values = record["sequence_variants"]["full"]["raw_tokens"]
        matrix[row, : len(values)] = values

    base = plt.get_cmap("gist_ncar", primitive_num)
    colors = [base(index) for index in range(primitive_num)]
    cmap = ListedColormap(colors)
    cmap.set_bad("#e5e7eb")
    norm = BoundaryNorm(np.arange(-0.5, primitive_num + 0.5), primitive_num)
    figure, axis = plt.subplots(figsize=(18, 18))
    image = axis.imshow(matrix, aspect="auto", interpolation="nearest", cmap=cmap, norm=norm)
    groups = []
    for label in sorted(set(record["activity_label_1based"] for record in ordered)):
        indices = [
            index
            for index, record in enumerate(ordered)
            if record["activity_label_1based"] == label
        ]
        name = ordered[indices[0]]["activity_name"]
        groups.append((name, min(indices), max(indices)))
    axis.set_yticks(
        [(start + end) / 2 for _, start, end in groups],
        [name for name, _, _ in groups],
    )
    for _, _, end in groups[:-1]:
        axis.axhline(end + 0.5, color="black", linewidth=1.2)
    axis.set_xlabel("Window index within trial (stride 1.28 s)")
    axis.set_ylabel("Held-out activity trials (10 rows per activity)")
    axis.set_title("Reference run: original per-window primitive sequences")
    colorbar = figure.colorbar(
        image,
        ax=axis,
        ticks=np.arange(primitive_num),
        fraction=0.025,
        pad=0.02,
    )
    colorbar.ax.set_yticklabels([f"P{index:02d}" for index in range(primitive_num)])
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def plot_primitive_waveforms(
    output_dir: Path,
    representatives: list[dict],
    raw_windows: dict[int, np.ndarray],
    sample_rate_hz: float,
) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_paths = []
    colors = ["#2563eb", "#dc2626", "#16a34a"]
    for page_start in range(0, len(representatives), 8):
        page = representatives[page_start : page_start + 8]
        figure, axes = plt.subplots(len(page), 2, figsize=(18, 3.0 * len(page)))
        if len(page) == 1:
            axes = np.asarray([axes])
        for row, item in enumerate(page):
            primitive_id = int(item["primitive_id"])
            window = raw_windows[primitive_id]
            times = np.arange(window.shape[1], dtype=np.float32) / float(sample_rate_hz)
            for channel, color in zip(range(3), colors):
                axes[row, 0].plot(
                    times,
                    window[channel],
                    color=color,
                    linewidth=0.9,
                    label=CHANNEL_NAMES[channel],
                )
            for channel, color in zip(range(3, 6), colors):
                axes[row, 1].plot(
                    times,
                    window[channel],
                    color=color,
                    linewidth=0.9,
                    label=CHANNEL_NAMES[channel],
                )
            source = (
                f"S{int(item['representative_subject']):02d}-"
                f"A{int(item['representative_activity_label_1based']):02d}-"
                f"T{int(item['representative_trial_number'])}-"
                f"W{int(item['representative_window_index'])}"
            )
            axes[row, 0].set_title(
                f"P{primitive_id:02d} accelerometer | nearest fit window {source}"
            )
            axes[row, 1].set_title(
                f"P{primitive_id:02d} gyroscope | {item['representative_activity_name']}"
            )
            axes[row, 0].set_ylabel("Original-scale value")
            for axis in axes[row]:
                axis.set_xlabel("Time (s)")
                axis.grid(alpha=0.2)
                axis.legend(loc="upper right", ncol=3, fontsize=8)
        end = page_start + len(page) - 1
        figure.suptitle(
            "Representative raw sensor windows for run-specific primitive candidates",
            fontsize=16,
            y=1.002,
        )
        figure.tight_layout()
        path = output_dir / f"primitive_waveforms_P{page_start:02d}-P{end:02d}.png"
        figure.savefig(path, dpi=170, bbox_inches="tight")
        plt.close(figure)
        output_paths.append(path)
    return output_paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--npz-path", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    config = read_json(run_dir / "experiment_config.json")
    summary = read_json(run_dir / "summary.json")
    records = read_records(run_dir / "trial_primitive_sequences.jsonl")
    npz_path = (
        Path(args.npz_path).expanduser().resolve()
        if args.npz_path
        else Path(config["npz_path"]).expanduser().resolve()
    )
    token_data = np.load(run_dir / "window_embeddings_and_tokens.npz")
    source_data = np.load(npz_path, allow_pickle=True)
    activity_names = [str(value) for value in source_data["activity_names"].tolist()]
    if len(activity_names) != 12:
        labels = source_data["labels"].astype(np.int64)
        names_array = source_data["activity_names"]
        activity_names = [
            str(names_array[np.flatnonzero(labels == label)[0]])
            for label in sorted(np.unique(labels).tolist())
        ]

    tokens = token_data["primitive_tokens"].astype(np.int64)
    split_role = token_data["split_role"].astype(np.int8)
    labels = token_data["activity_labels_0based"].astype(np.int64)
    distances = token_data["nearest_center_distances"].astype(np.float64)
    primitive_num = int(config["codebook"]["primitive_num"])
    representatives = []
    raw_windows = {}
    source_mean = source_data["mean"].astype(np.float32)
    source_std = source_data["std"].astype(np.float32)
    for primitive_id in range(primitive_num):
        primitive_mask = tokens == primitive_id
        fit_mask = primitive_mask & (split_role == 0)
        eval_mask = primitive_mask & (split_role == 1)
        fit_indices = np.flatnonzero(fit_mask)
        if len(fit_indices) == 0:
            raise RuntimeError(f"Primitive P{primitive_id:02d} has no fit window.")
        representative_row = int(
            fit_indices[np.argmin(distances[fit_indices])]
        )
        global_index = int(token_data["global_window_indices"][representative_row])
        normalized_window = source_data["windows"][global_index].astype(np.float32)
        raw_window = normalized_window * source_std[0] + source_mean[0]
        raw_windows[primitive_id] = raw_window
        item = {
            "primitive_id": primitive_id,
            "fit_window_count": int(np.sum(fit_mask)),
            "heldout_window_count": int(np.sum(eval_mask)),
            "heldout_subject_count": int(
                len(np.unique(token_data["subject_ids"][eval_mask]))
            ),
            "heldout_activity_count": int(len(np.unique(labels[eval_mask]))),
            "fit_top_activities": top_activity_text(
                labels, fit_mask, activity_names
            ),
            "heldout_top_activities": top_activity_text(
                labels, eval_mask, activity_names
            ),
            "representative_global_window_index": global_index,
            "representative_subject": int(token_data["subject_ids"][representative_row]),
            "representative_activity_label_1based": int(labels[representative_row] + 1),
            "representative_activity_name": activity_names[int(labels[representative_row])],
            "representative_trial_number": int(token_data["trial_numbers"][representative_row]),
            "representative_window_index": int(token_data["window_indices"][representative_row]),
            "representative_start_seconds": float(
                token_data["window_start_indices"][representative_row] / 100.0
            ),
            "representative_center_distance": float(distances[representative_row]),
        }
        representatives.append(item)
    write_csv(output_dir / "primitive_catalog.csv", representatives)
    representative_stack = np.stack(
        [raw_windows[primitive_id] for primitive_id in range(primitive_num)], axis=0
    ).astype(np.float32)
    np.savez_compressed(
        output_dir / "primitive_representative_raw_windows.npz",
        primitive_ids=np.arange(primitive_num, dtype=np.int64),
        raw_windows=representative_stack,
        channel_names=np.asarray(CHANNEL_NAMES),
        sample_rate_hz=np.asarray(float(config["data"]["sample_rate_hz"])),
    )
    raw_window_rows = []
    for item in representatives:
        primitive_id = int(item["primitive_id"])
        window = raw_windows[primitive_id]
        for sample_index in range(window.shape[1]):
            raw_window_rows.append(
                {
                    "primitive_id": f"P{primitive_id:02d}",
                    "sample_index": sample_index,
                    "time_seconds": sample_index
                    / float(config["data"]["sample_rate_hz"]),
                    "acc_x": float(window[0, sample_index]),
                    "acc_y": float(window[1, sample_index]),
                    "acc_z": float(window[2, sample_index]),
                    "gyro_x": float(window[3, sample_index]),
                    "gyro_y": float(window[4, sample_index]),
                    "gyro_z": float(window[5, sample_index]),
                    "source_subject": int(item["representative_subject"]),
                    "source_activity": item["representative_activity_name"],
                    "source_trial": int(item["representative_trial_number"]),
                    "source_window_index": int(item["representative_window_index"]),
                }
            )
    write_csv(
        output_dir / "primitive_representative_raw_windows_long.csv",
        raw_window_rows,
    )

    trial_rows = []
    for record in sorted(
        records,
        key=lambda value: (
            value["activity_label_1based"],
            value["subject_id"],
            value["trial_number"],
        ),
    ):
        full = record["sequence_variants"]["full"]
        trial_rows.append(
            {
                "trial_key": record["trial_key"],
                "subject_id": record["subject_id"],
                "activity_label_1based": record["activity_label_1based"],
                "activity_name": record["activity_name"],
                "trial_number": record["trial_number"],
                "window_count": full["window_count"],
                "unique_primitive_count": full["unique_primitive_count"],
                "run_count": full["run_count"],
                "observed_trial_span_seconds": full["observed_trial_span_seconds"],
                "raw_window_token_sequence": token_text(full["raw_tokens"]),
                "rle_primitive_sequence": compact_rle(full),
            }
        )
    write_csv(output_dir / "activity_trial_sequences.csv", trial_rows)

    grouped = {}
    for record in records:
        grouped.setdefault(record["activity_label_1based"], []).append(record)
    activity_lines = [
        "# 12 类 activity 的参考动作元序列",
        "",
        "> 参考码本：fold 1 / seed 50；P00–P31 只在此 run 内有意义。",
        "",
        "记号 `Pxx[nw/ts]` 表示 primitive `Pxx` 连续覆盖 `n` 个窗口，观测跨度为 `t` 秒；窗口长 2.56 秒、stride 1.28 秒，因此相邻窗口存在 50% overlap。",
        "",
    ]
    activity_medoid_rows = []
    for label in sorted(grouped):
        selected = sorted(
            grouped[label], key=lambda value: (value["subject_id"], value["trial_number"])
        )
        medoid = medoid_record(selected)
        all_raw = [
            token
            for record in selected
            for token in record["sequence_variants"]["full"]["raw_tokens"]
        ]
        top_tokens = Counter(all_raw).most_common(6)
        total = len(all_raw)
        top_token_text = "; ".join(
            f"P{token:02d}={count / total:.1%}" for token, count in top_tokens
        )
        medoid_full = medoid["sequence_variants"]["full"]
        activity_medoid_rows.append(
            {
                "activity_label_1based": label,
                "activity_name": selected[0]["activity_name"],
                "medoid_trial_key": medoid["trial_key"],
                "window_count": medoid_full["window_count"],
                "unique_primitive_count": medoid_full["unique_primitive_count"],
                "run_count": medoid_full["run_count"],
                "top_primitives_across_10_trials": top_token_text,
                "medoid_raw_window_token_sequence": token_text(
                    medoid_full["raw_tokens"]
                ),
                "medoid_rle_primitive_sequence": compact_rle(medoid_full),
            }
        )
        activity_lines.extend(
            [
                f"## A{label:02d} {selected[0]['activity_name']}",
                "",
                "- 10 条 held-out trials 使用的主要 primitive："
                + top_token_text,
                f"- 组内 medoid trial：`{medoid['trial_key']}`；其 RLE 序列：",
                "",
                "```text",
                compact_rle(medoid["sequence_variants"]["full"]),
                "```",
                "",
                "全部 10 条原 trial 的 RLE 序列：",
                "",
                "```text",
            ]
        )
        for record in selected:
            activity_lines.append(
                f"{record['trial_key']}: {compact_rle(record['sequence_variants']['full'])}"
            )
        activity_lines.extend(["```", ""])
    (output_dir / "activity_sequence_examples.md").write_text(
        "\n".join(activity_lines), encoding="utf-8"
    )
    write_csv(output_dir / "activity_medoid_sequences.csv", activity_medoid_rows)

    plot_activity_sequences(
        output_dir / "activity_trial_token_sequences.png",
        records,
        primitive_num,
    )
    waveform_paths = plot_primitive_waveforms(
        output_dir,
        representatives,
        raw_windows,
        sample_rate_hz=float(config["data"]["sample_rate_hz"]),
    )

    decomposition = summary["primitive_statistics"]["per_trial_decomposition"]
    usage = summary["primitive_statistics"]["eval_usage"]
    catalog_lines = [
        "# 参考 run 的 Motion Primitive 目录",
        "",
        "## 如何理解“拆出了多少动作元”",
        "",
        f"- 参考 run 固定建立了 **{primitive_num} 个码字候选（P00–P{primitive_num - 1:02d}）**；",
        f"- held-out subjects 实际使用 **{usage['used_primitives']}/{usage['total_primitives']}** 个；",
        f"- 使用频率对应的 effective K 为 **{usage['perplexity_effective_k']:.2f}**，因此不能把 32 理解为 32 个同等稳定、同等常见的语义动作；",
        f"- 120 条 held-out trials 的 unique primitive 中位数为 **{decomposition['unique_primitives_median']}**，范围 **{decomposition['unique_primitives_min']}–{decomposition['unique_primitives_max']}**；",
        "- 码字编号只对 fold 1 / seed 50 有效，不能与其他 fold/seed 的同号 primitive 直接等同。",
        "",
        "## 32 个候选元动作",
        "",
        "`代表来源`选择该 primitive 在拟合集上距离 KMeans center 最近的原始 2.56 秒传感器窗口。Top activity 是统计关联，不是人工确认的动作语义。",
        "",
        "| ID | Fit windows | Held-out windows | Fit top activities | Held-out top activities | 代表来源 |",
        "|---:|---:|---:|---|---|---|",
    ]
    for item in representatives:
        source = (
            f"S{int(item['representative_subject']):02d}/"
            f"A{int(item['representative_activity_label_1based']):02d}/"
            f"T{int(item['representative_trial_number'])}/"
            f"W{int(item['representative_window_index'])} "
            f"({item['representative_start_seconds']:.2f}s)"
        )
        catalog_lines.append(
            f"| P{int(item['primitive_id']):02d} | {item['fit_window_count']} | "
            f"{item['heldout_window_count']} | {item['fit_top_activities']} | "
            f"{item['heldout_top_activities']} | {source} |"
        )
    catalog_lines.extend(
        [
            "",
            "## 文件索引",
            "",
            "- `primitive_catalog.csv`：32 个 primitive 的完整统计与代表窗口元数据；",
            "- `primitive_representative_raw_windows.npz`：32×6×256 的代表原始六轴窗口；",
            "- `primitive_representative_raw_windows_long.csv`：上述原始窗口的 8,192 行长表；",
            "- `activity_trial_sequences.csv`：120 条 trial 的 raw token 与 RLE 序列；",
            "- `activity_medoid_sequences.csv`：12 类 activity 各自的 medoid 原始/RLE 序列；",
            "- `activity_sequence_examples.md`：按 12 类 activity 分组的全部 RLE 序列和组内 medoid；",
            "- `activity_trial_token_sequences.png`：120 条原始窗口 token 序列总览；",
        ]
    )
    catalog_lines.extend(
        f"- `{path.name}`：P{index * 8:02d}–P{index * 8 + 7:02d} 的代表原始传感器窗口；"
        for index, path in enumerate(waveform_paths)
    )
    (output_dir / "REFERENCE_CATALOG.md").write_text(
        "\n".join(catalog_lines) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "run_dir": str(run_dir),
                "output_dir": str(output_dir),
                "primitive_num": primitive_num,
                "heldout_used_primitives": usage["used_primitives"],
                "effective_k": usage["perplexity_effective_k"],
                "trial_count": len(records),
                "waveform_pages": [str(path) for path in waveform_paths],
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
