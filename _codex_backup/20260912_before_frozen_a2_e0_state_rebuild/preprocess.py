# HHR USC-HAD preprocessing
# -*- coding: utf-8 -*-

"""
USC-HAD preprocessing for motion-primitive trajectory HAR-CGCD.

功能：
1. 递归读取 USC-HAD 的 .mat 文件；
2. 从 sensor_readings 字段读取 [T, 6] 原始传感器序列；
3. 按 window_size / stride 切分为滑动窗口；
4. 支持选择输入通道；
5. 支持额外加入 acc_mag / gyro_mag；
6. 将窗口保存为 [N, C, T]；
7. 使用指定训练类统计 z-score 的 mean/std；
8. 保存为 .npz 文件，供后续 Dataset 直接读取。

PyCharm Linux/WSL 示例（一行）：
python preprocess.py --root /mnt/d/WorkDir/DataSet/USC-HAD --out_dir ./processed/uschad_w256_s128_train17stats --window_size 256 --stride 128 --channels acc_x acc_y acc_z gyro_x gyro_y gyro_z --old_classes 1 2 3 4 5 6
"""

import argparse
import json
import os
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
from scipy.io import loadmat


ACTIVITY_NAMES = {
    1: "Walking Forward",
    2: "Walking Left",
    3: "Walking Right",
    4: "Walking Upstairs",
    5: "Walking Downstairs",
    6: "Running Forward",
    7: "Jumping Up",
    8: "Sitting",
    9: "Standing",
    10: "Sleeping",
    11: "Elevator Up",
    12: "Elevator Down",
}

RAW_CHANNEL_NAMES = [
    "acc_x",
    "acc_y",
    "acc_z",
    "gyro_x",
    "gyro_y",
    "gyro_z",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Preprocess USC-HAD .mat files into window-level npz files."
    )

    parser.add_argument(
        "--root",
        type=str,
        default="/mnt/d/WorkDir/DataSet/USC-HAD",
       # required=True,
        help="USC-HAD 数据集根目录。脚本会递归搜索其中所有 .mat 文件。",
    )

    parser.add_argument(
        "--out_dir",
        type=str,
        required=True,
        help="预处理结果保存目录。",
    )

    parser.add_argument(
        "--window_size",
        type=int,
        default=256,
        help="滑动窗口长度。100Hz 下 256 约等于 2.56 秒。",
    )

    parser.add_argument(
        "--stride",
        type=int,
        default=128,
        help="滑动窗口步长。128 表示 50%% overlap。",
    )

    parser.add_argument(
        "--channels",
        nargs="+",
        default=["acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"],
        help=(
            "选择输入通道。可选："
            "acc_x acc_y acc_z gyro_x gyro_y gyro_z acc_mag gyro_mag"
        ),
    )

    parser.add_argument(
        "--add_acc_mag",
        action="store_true",
        help="是否额外加入 acc_mag。等价于在 channels 中加入 acc_mag。",
    )

    parser.add_argument(
        "--add_gyro_mag",
        action="store_true",
        help="是否额外加入 gyro_mag。等价于在 channels 中加入 gyro_mag。",
    )

    parser.add_argument(
        "--old_classes",
        nargs="+",
        type=int,
        default=[1, 2, 3, 4, 5, 6],
        help="Stage-0 old classes，使用 1-based activity id。",
    )

    parser.add_argument(
        "--all_classes",
        nargs="+",
        type=int,
        default=list(range(1, 13)),
        help="参与预处理的全部类别，使用 1-based activity id。",
    )

    parser.add_argument(
        "--source_subjects",
        nargs="*",
        type=int,
        default=None,
        help=(
            "可选。指定 source subjects，用于统计 mean/std。"
            "如果不指定，则使用 old_classes 的所有窗口统计。"
        ),
    )

    parser.add_argument(
        "--target_subjects",
        nargs="*",
        type=int,
        default=None,
        help=(
            "可选。指定 target subjects。当前脚本只记录该字段，"
            "不强制过滤，后续 Dataset 可按 subject_id 使用。"
        ),
    )

    parser.add_argument(
        "--drop_last",
        action="store_true",
        help=(
            "Deprecated compatibility flag. Incomplete trial tails are always "
            "excluded because padded windows are not represented in the npz schema."
        ),
    )

    parser.add_argument(
        "--min_trial_len",
        type=int,
        default=1,
        help="最短 trial 长度，小于该长度的 .mat 文件会被跳过。",
    )

    parser.add_argument(
        "--save_raw",
        action="store_true",
        help="是否额外保存未标准化的 windows_raw。默认不保存，节省空间。",
    )

    parser.add_argument(
        "--eps",
        type=float,
        default=1e-6,
        help="z-score 标准化中的最小 std，防止除 0。",
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="是否打印每个 .mat 文件的处理信息。",
    )

    args = parser.parse_args()
    return args


def find_mat_files(root: str) -> List[Path]:
    root_path = Path(root)
    if not root_path.exists():
        raise FileNotFoundError(f"Dataset root does not exist: {root}")

    mat_files = sorted(root_path.rglob("*.mat"))
    if len(mat_files) == 0:
        raise FileNotFoundError(f"No .mat files found under: {root}")

    return mat_files


def parse_label_from_filename(path: Path) -> int:
    """
    USC-HAD 文件名类似：
    a1t1.mat
    a12t5.mat

    返回：
    activity_id，1-based。
    """
    name = path.name
    match = re.search(r"a(\d+)t(\d+)\.mat$", name, flags=re.IGNORECASE)
    if match is None:
        raise ValueError(
            f"Cannot parse activity id from filename: {path}. "
            f"Expected format like a1t1.mat."
        )

    activity_id = int(match.group(1))
    return activity_id


def parse_trial_from_filename(path: Path) -> int:
    """
    从 a1t1.mat 中解析 trial_id。
    """
    name = path.name
    match = re.search(r"a(\d+)t(\d+)\.mat$", name, flags=re.IGNORECASE)
    if match is None:
        raise ValueError(
            f"Cannot parse trial id from filename: {path}. "
            f"Expected format like a1t1.mat."
        )

    trial_id = int(match.group(2))
    return trial_id


def parse_subject_from_path(path: Path) -> int:
    """
    尝试从路径中解析 subject_id。

    兼容示例：
    Subject1/a1t1.mat
    subject_1/a1t1.mat
    s1/a1t1.mat
    subj1/a1t1.mat

    如果解析失败，返回 -1。
    """
    parts = list(path.parts)

    for part in reversed(parts):
        lower = part.lower()

        patterns = [
            r"subject[_\-]?(\d+)",
            r"subj[_\-]?(\d+)",
            r"^s(\d+)$",
            r"user[_\-]?(\d+)",
        ]

        for pattern in patterns:
            match = re.search(pattern, lower)
            if match is not None:
                return int(match.group(1))

    return -1


def load_sensor_readings(mat_path: Path) -> np.ndarray:
    """
    读取 .mat 文件中的 sensor_readings。

    期望 shape: [T, 6]
    """
    mat = loadmat(str(mat_path))

    if "sensor_readings" not in mat:
        keys = [k for k in mat.keys() if not k.startswith("__")]
        raise KeyError(
            f"'sensor_readings' not found in {mat_path}. "
            f"Available keys: {keys}"
        )

    x = mat["sensor_readings"]

    if not isinstance(x, np.ndarray):
        raise TypeError(f"sensor_readings is not ndarray in {mat_path}")

    x = np.asarray(x, dtype=np.float32)

    if x.ndim != 2:
        raise ValueError(
            f"Expected sensor_readings shape [T, 6], "
            f"but got {x.shape} in {mat_path}"
        )

    if x.shape[1] != 6 and x.shape[0] == 6:
        x = x.T

    if x.shape[1] != 6:
        raise ValueError(
            f"Expected 6 channels, but got shape {x.shape} in {mat_path}"
        )

    return x


def select_and_extend_channels(
    raw_x: np.ndarray,
    channels: List[str],
) -> np.ndarray:
    """
    输入：
    raw_x: [T, 6]

    输出：
    selected_x: [T, C]
    """
    channel_dict: Dict[str, np.ndarray] = {
        "acc_x": raw_x[:, 0],
        "acc_y": raw_x[:, 1],
        "acc_z": raw_x[:, 2],
        "gyro_x": raw_x[:, 3],
        "gyro_y": raw_x[:, 4],
        "gyro_z": raw_x[:, 5],
    }

    acc_mag = np.sqrt(
        raw_x[:, 0] ** 2 + raw_x[:, 1] ** 2 + raw_x[:, 2] ** 2
    )
    gyro_mag = np.sqrt(
        raw_x[:, 3] ** 2 + raw_x[:, 4] ** 2 + raw_x[:, 5] ** 2
    )

    channel_dict["acc_mag"] = acc_mag
    channel_dict["gyro_mag"] = gyro_mag

    invalid_channels = [c for c in channels if c not in channel_dict]
    if len(invalid_channels) > 0:
        raise ValueError(
            f"Invalid channels: {invalid_channels}. "
            f"Valid channels are: {list(channel_dict.keys())}"
        )

    selected = [channel_dict[c] for c in channels]
    selected_x = np.stack(selected, axis=1).astype(np.float32)

    return selected_x


def sliding_window_trial(
    x: np.ndarray,
    window_size: int,
    stride: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    输入：
    x: [T, C]

    输出：
    windows: [N, C, window_size][批次，通道数，窗口大小]
    start_indices: [N]
    """
    if x.ndim != 2:
        raise ValueError(f"Expected x shape [T, C], but got {x.shape}")

    total_len, num_channels = x.shape

    if total_len < window_size:
        return (
            np.empty((0, num_channels, window_size), dtype=np.float32),
            np.empty((0,), dtype=np.int64),
        )

    windows = []
    start_indices = []

    for start in range(0, total_len - window_size + 1, stride):
        end = start + window_size
        window = x[start:end, :]          # [T, C]
        window = window.T                 # [C, T]
        windows.append(window)
        start_indices.append(start)

    windows = np.stack(windows, axis=0).astype(np.float32)
    start_indices = np.asarray(start_indices, dtype=np.int64)

    return windows, start_indices


def compute_train_stats(
    windows: np.ndarray,
    labels_1based: np.ndarray,
    subject_ids: np.ndarray,
    old_classes: List[int],
    source_subjects: Optional[List[int]],
    eps: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    计算 z-score mean/std。

    windows: [N, C, T]
    labels_1based: [N]

    默认只使用 old_classes 的窗口统计。
    如果指定 source_subjects，则进一步限制到 source_subjects。
    """
    old_classes_arr = np.asarray(old_classes, dtype=np.int64)
    stat_mask = np.isin(labels_1based, old_classes_arr)

    if source_subjects is not None and len(source_subjects) > 0:
        source_subjects_arr = np.asarray(source_subjects, dtype=np.int64)
        stat_mask = stat_mask & np.isin(subject_ids, source_subjects_arr)

    if stat_mask.sum() == 0:
        raise ValueError(
            "No samples found for computing normalization statistics. "
            "Please check --old_classes and --source_subjects."
        )

    stat_windows = windows[stat_mask]     # [N_stat, C, T]

    mean = stat_windows.mean(axis=(0, 2), keepdims=True)   # [1, C, 1]
    std = stat_windows.std(axis=(0, 2), keepdims=True)     # [1, C, 1]
    std = np.maximum(std, eps)

    return mean.astype(np.float32), std.astype(np.float32), stat_mask


def preprocess_uschad(args: argparse.Namespace) -> None:
    if args.window_size < 1:
        raise ValueError(f"--window_size must be positive, got {args.window_size}.")
    if args.stride < 1:
        raise ValueError(f"--stride must be positive, got {args.stride}.")
    if args.min_trial_len < 1:
        raise ValueError(f"--min_trial_len must be positive, got {args.min_trial_len}.")
    if args.eps <= 0:
        raise ValueError(f"--eps must be positive, got {args.eps}.")
    if not set(args.old_classes).issubset(set(args.all_classes)):
        raise ValueError(
            "--old_classes must be a subset of --all_classes, got "
            f"old={args.old_classes}, all={args.all_classes}."
        )
    if args.drop_last:
        print(
            "[USC-HAD Preprocess] --drop_last is a deprecated compatibility flag; "
            "incomplete trial tails are always excluded."
        )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    channels = list(args.channels)

    if args.add_acc_mag and "acc_mag" not in channels:
        channels.append("acc_mag")

    if args.add_gyro_mag and "gyro_mag" not in channels:
        channels.append("gyro_mag")

    mat_files = find_mat_files(args.root)

    all_windows = []
    all_labels_1based = []
    all_labels_0based = []
    all_activity_names = []
    all_subject_ids = []
    all_trial_numbers = []
    all_trial_global_ids = []
    all_window_indices = []
    all_window_start_indices = []
    all_file_paths = []

    skipped_files = []
    trial_global_id = 0

    all_classes_set = set(args.all_classes)

    print("=" * 80)
    print("[USC-HAD Preprocess] Start")
    print(f"Root: {args.root}")
    print(f"Output dir: {args.out_dir}")
    print(f"Found .mat files: {len(mat_files)}")
    print(f"window_size: {args.window_size}")
    print(f"stride: {args.stride}")
    print(f"channels: {channels}")
    print(f"old_classes: {args.old_classes}")
    print(f"all_classes: {args.all_classes}")
    print(f"source_subjects: {args.source_subjects}")
    print("=" * 80)

    for mat_path in mat_files:
        try:
            activity_id = parse_label_from_filename(mat_path)
            trial_number = parse_trial_from_filename(mat_path)
            subject_id = parse_subject_from_path(mat_path)

            if activity_id not in all_classes_set:
                continue

            raw_x = load_sensor_readings(mat_path)

            if raw_x.shape[0] < args.min_trial_len:
                skipped_files.append((str(mat_path), f"too short: {raw_x.shape[0]}"))
                continue

            selected_x = select_and_extend_channels(raw_x, channels)

            windows, start_indices = sliding_window_trial(
                selected_x,
                window_size=args.window_size,
                stride=args.stride,
            )

            if windows.shape[0] == 0:
                skipped_files.append(
                    (str(mat_path), f"shorter than window_size: {raw_x.shape[0]}")
                )
                continue

            num_windows = windows.shape[0]

            all_windows.append(windows)
            all_labels_1based.extend([activity_id] * num_windows)
            all_labels_0based.extend([activity_id - 1] * num_windows)
            all_activity_names.extend([ACTIVITY_NAMES.get(activity_id, "Unknown")] * num_windows)
            all_subject_ids.extend([subject_id] * num_windows)
            all_trial_numbers.extend([trial_number] * num_windows)
            all_trial_global_ids.extend([trial_global_id] * num_windows)
            all_window_indices.extend(list(range(num_windows)))
            all_window_start_indices.extend(start_indices.tolist())
            all_file_paths.extend([str(mat_path)] * num_windows)

            if args.verbose:
                print(
                    f"[OK] {mat_path} | "
                    f"activity={activity_id} | "
                    f"subject={subject_id} | "
                    f"trial={trial_number} | "
                    f"raw_shape={raw_x.shape} | "
                    f"windows={num_windows}"
                )

            trial_global_id += 1

        except Exception as e:
            skipped_files.append((str(mat_path), repr(e)))
            print(f"[SKIP] {mat_path} | reason: {repr(e)}")

    if len(all_windows) == 0:
        raise RuntimeError("No valid windows generated. Please check dataset path and parameters.")

    windows_raw = np.concatenate(all_windows, axis=0).astype(np.float32)  # [N, C, T]

    labels_1based = np.asarray(all_labels_1based, dtype=np.int64)
    labels_0based = np.asarray(all_labels_0based, dtype=np.int64)
    subject_ids = np.asarray(all_subject_ids, dtype=np.int64)
    trial_numbers = np.asarray(all_trial_numbers, dtype=np.int64)
    trial_global_ids = np.asarray(all_trial_global_ids, dtype=np.int64)
    window_indices = np.asarray(all_window_indices, dtype=np.int64)
    window_start_indices = np.asarray(all_window_start_indices, dtype=np.int64)
    file_paths = np.asarray(all_file_paths, dtype=object)
    activity_names = np.asarray(all_activity_names, dtype=object)

    mean, std, stat_mask = compute_train_stats(
        windows=windows_raw,
        labels_1based=labels_1based,
        subject_ids=subject_ids,
        old_classes=args.old_classes,
        source_subjects=args.source_subjects,
        eps=args.eps,
    )

    windows = ((windows_raw - mean) / std).astype(np.float32)

    old_mask = np.isin(labels_1based, np.asarray(args.old_classes, dtype=np.int64))
    new_mask = ~old_mask

    source_mask = np.ones_like(labels_1based, dtype=bool)
    if args.source_subjects is not None and len(args.source_subjects) > 0:
        source_mask = np.isin(
            subject_ids,
            np.asarray(args.source_subjects, dtype=np.int64),
        )

    target_mask = np.ones_like(labels_1based, dtype=bool)
    if args.target_subjects is not None and len(args.target_subjects) > 0:
        target_mask = np.isin(
            subject_ids,
            np.asarray(args.target_subjects, dtype=np.int64),
        )

    stage0_labeled_mask = old_mask & source_mask

    # Stage-1 默认包含 old + new 的 target/source 全部无标签窗口。
    # 具体是否只用 target subjects，可在后续 Dataset 中按 target_mask 控制。
    stage1_unlabeled_mask = np.isin(
        labels_1based,
        np.asarray(args.all_classes, dtype=np.int64),
    )

    save_dict = {
        "windows": windows,                                      # [N, C, T], z-score 后
        "labels": labels_0based,                                # [N], 0-based
        "labels_1based": labels_1based,                         # [N], 1-based
        "subject_ids": subject_ids,                             # [N]
        "trial_numbers": trial_numbers,                         # [N]
        "trial_global_ids": trial_global_ids,                   # [N]
        "window_indices": window_indices,                       # [N]
        "window_start_indices": window_start_indices,           # [N]
        "file_paths": file_paths,                               # [N]
        "activity_names": activity_names,                       # [N]
        "channel_names": np.asarray(channels, dtype=object),     # [C]
        "mean": mean.astype(np.float32),                         # [1, C, 1]
        "std": std.astype(np.float32),                           # [1, C, 1]
        "stat_mask": stat_mask.astype(bool),                     # [N]
        "old_mask": old_mask.astype(bool),                       # [N]
        "new_mask": new_mask.astype(bool),                       # [N]
        "source_mask": source_mask.astype(bool),                 # [N]
        "target_mask": target_mask.astype(bool),                 # [N]
        "stage0_labeled_mask": stage0_labeled_mask.astype(bool), # [N]
        "stage1_unlabeled_mask": stage1_unlabeled_mask.astype(bool),
    }

    if args.save_raw:
        save_dict["windows_raw"] = windows_raw

    npz_path = out_dir / "uschad_windows.npz"
    np.savez_compressed(npz_path, **save_dict)

    meta = {
        "dataset": "USC-HAD",
        "root": str(Path(args.root).resolve()),
        "num_mat_files_found": len(mat_files),
        "num_valid_trials": int(trial_global_id),
        "num_windows": int(windows.shape[0]),
        "window_shape": list(windows.shape),
        "window_size": int(args.window_size),
        "stride": int(args.stride),
        "channels": channels,
        "num_channels": len(channels),
        "old_classes": args.old_classes,
        "all_classes": args.all_classes,
        "source_subjects": args.source_subjects,
        "target_subjects": args.target_subjects,
        "activity_names": ACTIVITY_NAMES,
        "label_format": {
            "labels": "0-based, used for PyTorch training",
            "labels_1based": "original USC-HAD activity id",
        },
        "normalization": {
            "type": "per-channel z-score",
            "mean_shape": list(mean.shape),
            "std_shape": list(std.shape),
            "stats_from": "old_classes only"
            if args.source_subjects is None
            else "old_classes and source_subjects",
        },
        "masks": {
            "old_mask": "labels_1based in old_classes",
            "new_mask": "labels_1based not in old_classes",
            "source_mask": "subject_ids in source_subjects, or all True if not specified",
            "target_mask": "subject_ids in target_subjects, or all True if not specified",
            "stage0_labeled_mask": "old_mask & source_mask",
            "stage1_unlabeled_mask": "all classes, default old + new",
        },
        "skipped_files": skipped_files,
    }

    meta_path = out_dir / "meta.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print("=" * 80)
    print("[USC-HAD Preprocess] Finished")
    print(f"Saved npz: {npz_path}")
    print(f"Saved meta: {meta_path}")
    print("-" * 80)
    print(f"windows shape: {windows.shape}  # [N, C, T]")
    print(f"labels shape: {labels_0based.shape}")
    print(f"num channels: {len(channels)}")
    print(f"num valid trials: {trial_global_id}")
    print(f"num skipped files: {len(skipped_files)}")
    print("-" * 80)
    print(f"old windows: {int(old_mask.sum())}")
    print(f"new windows: {int(new_mask.sum())}")
    print(f"stage0 labeled windows: {int(stage0_labeled_mask.sum())}")
    print(f"stage1 unlabeled windows: {int(stage1_unlabeled_mask.sum())}")
    print("-" * 80)

    unique_labels, label_counts = np.unique(labels_1based, return_counts=True)
    print("[Class window counts]")
    for label, count in zip(unique_labels, label_counts):
        name = ACTIVITY_NAMES.get(int(label), "Unknown")
        print(f"  class {int(label):02d} | {name:<20s} | windows={int(count)}")

    unique_subjects, subject_counts = np.unique(subject_ids, return_counts=True)
    print("-" * 80)
    print("[Subject window counts]")
    for sid, count in zip(unique_subjects, subject_counts):
        print(f"  subject {int(sid):02d} | windows={int(count)}")

    print("=" * 80)


def main() -> None:
    args = parse_args()
    preprocess_uschad(args)


if __name__ == "__main__":
    main()
# 6. 运行命令
# 6.1 V0：原始 6 通道版本
# python preprocess_uschad.py \
#   --root /path/to/USC-HAD \
#   --out_dir ./processed/uschad_v0 \
#   --window_size 256 \
#   --stride 128 \
#   --channels acc_x acc_y acc_z gyro_x gyro_y gyro_z \
#   --old_classes 1 2 3 4 5 6 \
#   --all_classes 1 2 3 4 5 6 7 8 9 10 11 12 \
#   --verbose
# 6.2 V1：加入 acc_mag 和 gyro_mag
# python preprocess_uschad.py \
#   --root /path/to/USC-HAD \
#   --out_dir ./processed/uschad_v1_mag \
#   --window_size 256 \
#   --stride 128 \
#   --channels acc_x acc_y acc_z gyro_x gyro_y gyro_z \
#   --add_acc_mag \
#   --add_gyro_mag \
#   --old_classes 1 2 3 4 5 6 \
#   --all_classes 1 2 3 4 5 6 7 8 9 10 11 12 \
#   --verbose
#
# 这个版本输出窗口形状会从：
#
# [N, 6, 256]
#
# 变成：
#
# [N, 8, 256]
# 6.3 cross-subject 预留版本
#
# 假设 1 2 3 4 5 6 7 是 source subjects，8 9 10 11 12 13 14 是 target subjects：
#
# python preprocess_uschad.py \
#   --root /path/to/USC-HAD \
#   --out_dir ./processed/uschad_cross_subject_v0 \
#   --window_size 256 \
#   --stride 128 \
#   --channels acc_x acc_y acc_z gyro_x gyro_y gyro_z \
#   --old_classes 1 2 3 4 5 6 \
#   --all_classes 1 2 3 4 5 6 7 8 9 10 11 12 \
#   --source_subjects 1 2 3 4 5 6 7 \
#   --target_subjects 8 9 10 11 12 13 14 \
#   --verbose
