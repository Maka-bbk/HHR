# -*- coding: utf-8 -*-  # 指定源码文件使用 UTF-8 编码，避免中文注释乱码

import math
from copy import deepcopy  # 导入 deepcopy，用于复制 Dataset，避免原数据集被原地修改
from pathlib import Path  # 导入 Path，用于更稳妥地处理文件路径
from typing import Dict, List, Optional, Sequence, Tuple  # 导入类型注解，方便后续阅读和调试

import numpy as np  # 导入 numpy，用于读取 npz、索引切分、数组拼接
import torch  # 导入 torch，用于把 numpy 窗口转成 tensor
from torch.utils.data import Dataset  # 导入 PyTorch Dataset 基类，用于自定义数据集


class HARPlainTransform(object):  # 定义 HAR 普通变换类，不做增强、不生成多视图
    def __init__(self, clone: bool = True):  # 初始化普通变换，clone 用于控制是否复制输入
        self.clone = bool(clone)  # 保存 clone 参数，确保它是布尔值

    def __call__(self, x: torch.Tensor) -> torch.Tensor:  # 让普通变换对象可以像函数一样被调用
        x = x.float()  # 确保输入数据类型是 float32，适合神经网络输入

        if self.clone:  # 如果要求复制输入
            x = x.clone()  # 复制一份 tensor，避免后续操作影响原始数据

        return x  # 直接返回原始窗口，形状仍然是 [C, T]
class HARWeakTransform(object):  # 定义 HAR 弱增强类，主要用于第一个 view
    def __init__(self, jitter_std: float = 0.01, scale_std: float = 0.05):  # 初始化弱增强参数
        self.jitter_std = float(jitter_std)  # 保存 jitter 噪声标准差，并转成 float
        self.scale_std = float(scale_std)  # 保存 scaling 标准差，并转成 float
        if self.jitter_std < 0 or self.scale_std < 0:
            raise ValueError(
                "HAR weak jitter/scale standard deviations must be non-negative."
            )

    def __call__(self, x: torch.Tensor) -> torch.Tensor:  # 让对象可以像函数一样被调用
        x = x.clone().float()  # 复制输入窗口，并确保类型为 float32，避免原始数据被原地修改

        if self.jitter_std > 0:  # 如果 jitter 标准差大于 0，就添加高斯噪声
            x = x + torch.randn_like(x) * self.jitter_std  # 对每个时间点和通道加入轻微随机噪声

        if self.scale_std > 0:  # 如果 scaling 标准差大于 0，就进行通道级缩放
            scale = torch.randn(x.size(0), 1, dtype=x.dtype, device=x.device) * self.scale_std + 1.0  # 为每个通道生成一个缩放系数
            x = x * scale  # 将每个通道乘以对应的缩放系数

        return x  # 返回弱增强后的窗口，形状仍为 [C, T]
class HARStrongTransform(object):  # 定义 HAR 强增强类，主要用于第二个 view
    def __init__(self, jitter_std: float = 0.02, scale_std: float = 0.10, time_mask_ratio: float = 0.10):  # 初始化强增强参数
        self.jitter_std = float(jitter_std)  # 保存更强的 jitter 噪声标准差
        self.scale_std = float(scale_std)  # 保存更强的 scaling 标准差
        self.time_mask_ratio = float(time_mask_ratio)  # 保存时间遮挡比例
        if self.jitter_std < 0 or self.scale_std < 0:
            raise ValueError(
                "HAR strong jitter/scale standard deviations must be non-negative."
            )
        if not 0.0 <= self.time_mask_ratio <= 1.0:
            raise ValueError("HAR strong time-mask ratio must be in [0, 1].")

    def __call__(self, x: torch.Tensor) -> torch.Tensor:  # 让强增强对象可以像函数一样调用
        x = x.clone().float()  # 复制输入窗口，并确保数据类型为 float32

        if self.jitter_std > 0:  # 如果 jitter 标准差大于 0，就添加高斯噪声
            x = x + torch.randn_like(x) * self.jitter_std  # 对窗口加入随机噪声

        if self.scale_std > 0:  # 如果 scaling 标准差大于 0，就执行通道级缩放
            scale = torch.randn(x.size(0), 1, dtype=x.dtype, device=x.device) * self.scale_std + 1.0  # 为每个通道生成缩放系数
            x = x * scale  # 应用缩放增强

        if self.time_mask_ratio > 0:  # 如果时间遮挡比例大于 0，就执行 time masking
            total_len = x.size(1)  # 获取时间长度 T
            mask_len = int(round(total_len * self.time_mask_ratio))  # 根据比例计算需要遮挡的长度
            mask_len = max(1, min(mask_len, total_len))  # 限制遮挡长度至少为 1，最多不超过整个窗口
            start_max = max(0, total_len - mask_len)  # 计算遮挡起点的最大值
            start = int(torch.randint(low=0, high=start_max + 1, size=(1,)).item())  # 随机采样遮挡起点
            x[:, start:start + mask_len] = 0.0  # 将所有通道在指定时间段内置零

        return x  # 返回强增强后的窗口，形状仍为 [C, T]
class HARContrastiveTransform(object):
    """
    Generate multiple views for HHR training.

    aug_mode:
        none:
            return n_views cloned raw tensors.
            This means no HAR data augmentation.

        weak_strong:
            view 0 uses weak HAR augmentation.
            view 1 and later views use strong HAR augmentation.
    """

    def __init__(
        self,
        n_views: int = 2,
        aug_mode: str = "none",
        weak_jitter_std: float = 0.01,
        weak_scale_std: float = 0.05,
        strong_jitter_std: float = 0.02,
        strong_scale_std: float = 0.10,
        strong_time_mask_ratio: float = 0.10,
    ):
        self.n_views = int(n_views)
        self.aug_mode = str(aug_mode)
        self.plain = HARPlainTransform(clone=True)
        self.weak = HARWeakTransform(
            jitter_std=weak_jitter_std,
            scale_std=weak_scale_std,
        )
        self.strong = HARStrongTransform(
            jitter_std=strong_jitter_std,
            scale_std=strong_scale_std,
            time_mask_ratio=strong_time_mask_ratio,
        )

        if self.n_views <= 0:
            raise ValueError(f"n_views must be positive, got {self.n_views}")

        if self.aug_mode not in ["none", "weak_strong"]:
            raise ValueError(f"Unknown HAR aug_mode: {self.aug_mode}")

    def __call__(self, x: torch.Tensor) -> List[torch.Tensor]:
        views = []

        if self.aug_mode == "none":
            for _ in range(self.n_views):
                views.append(self.plain(x))

            return views

        if self.aug_mode == "weak_strong":
            for view_id in range(self.n_views):
                if view_id == 0:
                    views.append(self.weak(x))
                else:
                    views.append(self.strong(x))

            return views

        raise ValueError(f"Unknown HAR aug_mode: {self.aug_mode}")


class HARTrialContrastiveTransform(object):
    """Create two trial-level views while preserving temporal window order."""

    def __init__(
        self,
        n_views: int = 2,
        view_mode: str = "full_random_crop",
        crop_ratio: float = 2.0 / 3.0,
        min_windows: int = 2,
        aug_mode: str = "none",
        weak_jitter_std: float = 0.01,
        weak_scale_std: float = 0.05,
        strong_jitter_std: float = 0.02,
        strong_scale_std: float = 0.10,
        strong_time_mask_ratio: float = 0.10,
    ):
        self.n_views = int(n_views)
        self.view_mode = str(view_mode)
        self.crop_ratio = float(crop_ratio)
        self.min_windows = int(min_windows)
        self.aug_mode = str(aug_mode)
        self.weak_jitter_std = float(weak_jitter_std)
        self.weak_scale_std = float(weak_scale_std)
        self.strong_jitter_std = float(strong_jitter_std)
        self.strong_scale_std = float(strong_scale_std)
        self.strong_time_mask_ratio = float(strong_time_mask_ratio)

        if self.n_views != 2:
            raise ValueError(
                f"Trial-level HHR requires exactly two views, got {self.n_views}."
            )
        if self.view_mode not in ["full_full", "full_random_crop"]:
            raise ValueError(
                f"Unknown trial view mode {self.view_mode!r}; expected "
                "'full_full' or 'full_random_crop'."
            )
        if not 0.0 < self.crop_ratio <= 1.0:
            raise ValueError(
                f"trial crop ratio must be in (0, 1], got {self.crop_ratio}."
            )
        if self.min_windows < 1:
            raise ValueError(
                f"trial min_windows must be at least 1, got {self.min_windows}."
            )
        if self.aug_mode not in ["none", "weak_strong"]:
            raise ValueError(f"Unknown HAR aug_mode: {self.aug_mode}")
        augmentation_stds = [
            self.weak_jitter_std,
            self.weak_scale_std,
            self.strong_jitter_std,
            self.strong_scale_std,
        ]
        if any(value < 0 for value in augmentation_stds):
            raise ValueError(
                "Trial jitter/scale standard deviations must be non-negative."
            )
        if not 0.0 <= self.strong_time_mask_ratio <= 1.0:
            raise ValueError(
                "strong_time_mask_ratio must be in [0, 1], got "
                f"{self.strong_time_mask_ratio}."
            )

    @staticmethod
    def _augment_bag(
        bag: torch.Tensor,
        starts: torch.Tensor,
        jitter_std: float,
        scale_std: float,
        time_mask_ratio: float,
    ) -> torch.Tensor:
        bag = bag.clone().float()
        if bag.ndim != 3:
            raise ValueError(
                f"A trial bag must have shape [L, C, T], got {tuple(bag.shape)}."
            )
        if len(bag) == 0:
            raise ValueError("Cannot augment an empty trial bag.")

        if jitter_std > 0:
            bag = bag + torch.randn_like(bag) * float(jitter_std)

        if scale_std > 0:
            # One scale per channel and view preserves continuity across windows.
            scale = (
                torch.randn(
                    bag.size(1), 1, dtype=bag.dtype, device=bag.device
                ) * float(scale_std) + 1.0
            )
            bag = bag * scale.unsqueeze(0)

        if time_mask_ratio > 0:
            starts = starts.to(dtype=torch.long, device=bag.device)
            window_size = int(bag.size(2))
            global_begin = int(starts.min().item())
            global_end = int(starts.max().item()) + window_size
            span = max(1, global_end - global_begin)
            mask_len = max(1, min(int(round(span * time_mask_ratio)), span))
            mask_begin = global_begin + int(
                torch.randint(0, span - mask_len + 1, size=(1,)).item()
            )
            mask_end = mask_begin + mask_len

            for window_id, window_begin_tensor in enumerate(starts):
                window_begin = int(window_begin_tensor.item())
                window_end = window_begin + window_size
                overlap_begin = max(window_begin, mask_begin)
                overlap_end = min(window_end, mask_end)
                if overlap_begin < overlap_end:
                    local_begin = overlap_begin - window_begin
                    local_end = overlap_end - window_begin
                    bag[window_id, :, local_begin:local_end] = 0.0

        return bag

    def __call__(
        self,
        bag: torch.Tensor,
        starts: torch.Tensor,
    ) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        if bag.ndim != 3 or starts.ndim != 1 or len(bag) != len(starts):
            raise ValueError(
                "Trial transform expects bag [L,C,T] and starts [L], got "
                f"{tuple(bag.shape)} and {tuple(starts.shape)}."
            )
        if len(bag) == 0:
            raise ValueError("Cannot create views from an empty trial bag.")

        full_indices = torch.arange(len(bag), dtype=torch.long)
        second_indices = full_indices
        if self.view_mode == "full_random_crop":
            crop_len = max(
                self.min_windows,
                int(math.ceil(len(bag) * self.crop_ratio - 1e-12)),
            )
            crop_len = min(crop_len, len(bag))
            crop_start = int(
                torch.randint(0, len(bag) - crop_len + 1, size=(1,)).item()
            )
            second_indices = full_indices[crop_start:crop_start + crop_len]

        view_specs = [
            (full_indices, self.weak_jitter_std, self.weak_scale_std, 0.0),
            (
                second_indices,
                self.strong_jitter_std,
                self.strong_scale_std,
                self.strong_time_mask_ratio,
            ),
        ]

        views = []
        for indices, jitter_std, scale_std, time_mask_ratio in view_specs:
            selected_bag = bag[indices]
            selected_starts = starts[indices]
            if self.aug_mode == "none":
                transformed = selected_bag.clone().float()
            else:
                transformed = self._augment_bag(
                    selected_bag,
                    selected_starts,
                    jitter_std,
                    scale_std,
                    time_mask_ratio,
                )
            views.append((transformed, selected_starts.clone()))

        return views


class USCHADWindowDataset(Dataset):  # 定义 USC-HAD 窗口级 Dataset
    def __init__(self, npz_path: str, transform=None, target_transform=None, indices: Optional[Sequence[int]] = None):  # 初始化 Dataset
        self.npz_path = str(npz_path)  # 保存 npz 路径为字符串
        self.transform = transform  # 保存输入数据增强或变换函数
        self.target_transform = target_transform  # 保存标签变换函数，后续 get_datasets.py 会设置

        if not Path(self.npz_path).exists():  # 检查 npz 文件是否存在
            raise FileNotFoundError(f"USC-HAD npz file not found: {self.npz_path}")  # 如果文件不存在，直接报错

        npz = np.load(self.npz_path, allow_pickle=True)  # 读取预处理生成的 uschad_windows.npz
        required_fields = {
            "windows",
            "mean",
            "std",
            "labels",
            "labels_1based",
            "subject_ids",
            "trial_numbers",
            "trial_global_ids",
            "window_indices",
            "window_start_indices",
            "stat_mask",
        }
        missing_fields = required_fields - set(npz.files)
        if missing_fields:
            raise RuntimeError(
                f"USC-HAD npz is missing required fields {sorted(missing_fields)}: "
                f"{self.npz_path}"
            )

        self.data = np.asarray(npz["windows"], dtype=np.float32)  # 读取窗口数据，形状为 [N, C, T]

        if self.data.ndim != 3:
            raise RuntimeError(
                f"Expected npz['windows'] shape [N, C, T], but got {self.data.shape}. "
                f"Please check the preprocessing output: {self.npz_path}"
            )

        self.actual_num_samples = int(self.data.shape[0])
        self.actual_num_channels = int(self.data.shape[1])
        self.actual_window_size = int(self.data.shape[2])

        self.targets = np.asarray(npz["labels"], dtype=np.int64).tolist()  # 读取 0-based 标签，并转成 list 以兼容原项目写法
        self.uq_idxs = np.arange(len(self.targets), dtype=np.int64)  # 为每个窗口生成唯一索引 uq_idx

        self.labels_1based = np.asarray(npz["labels_1based"], dtype=np.int64)  # 保存原始 1-based USC-HAD 标签
        self.subject_ids = np.asarray(npz["subject_ids"], dtype=np.int64)  # 保存每个窗口所属 subject id
        self.trial_numbers = np.asarray(npz["trial_numbers"], dtype=np.int64)  # 保存 subject 内原始 trial 编号
        self.trial_global_ids = np.asarray(npz["trial_global_ids"], dtype=np.int64)  # 保存每个窗口所属全局 trial id
        self.window_indices = np.asarray(npz["window_indices"], dtype=np.int64)  # 保存每个窗口在 trial 内的窗口序号
        self.window_start_indices = np.asarray(npz["window_start_indices"], dtype=np.int64)  # 保存每个窗口在 trial 内的起始帧位置
        self.stat_mask = np.asarray(npz["stat_mask"], dtype=bool)
        self.npz_mean = np.asarray(npz["mean"], dtype=np.float32)
        self.npz_std = np.asarray(npz["std"], dtype=np.float32)
        self.has_windows_raw = "windows_raw" in npz.files

        expected_stat_shape = (1, self.actual_num_channels, 1)
        if self.npz_mean.shape != expected_stat_shape:
            raise RuntimeError(
                f"USC-HAD npz mean shape must be {expected_stat_shape}, "
                f"got {self.npz_mean.shape}."
            )
        if self.npz_std.shape != expected_stat_shape:
            raise RuntimeError(
                f"USC-HAD npz std shape must be {expected_stat_shape}, "
                f"got {self.npz_std.shape}."
            )
        if not np.all(np.isfinite(self.npz_mean)):
            raise RuntimeError("USC-HAD npz mean contains non-finite values.")
        if not np.all(np.isfinite(self.npz_std)) or np.any(self.npz_std <= 0):
            raise RuntimeError(
                "USC-HAD npz std must contain only finite positive values."
            )
        if not np.all(np.isfinite(self.data)):
            raise RuntimeError("USC-HAD npz windows contain non-finite values.")

        sample_fields = {
            "labels": np.asarray(self.targets, dtype=np.int64),
            "labels_1based": self.labels_1based,
            "subject_ids": self.subject_ids,
            "trial_numbers": self.trial_numbers,
            "trial_global_ids": self.trial_global_ids,
            "window_indices": self.window_indices,
            "window_start_indices": self.window_start_indices,
            "stat_mask": self.stat_mask,
        }
        invalid_lengths = {
            name: int(len(values))
            for name, values in sample_fields.items()
            if len(values) != self.actual_num_samples
        }
        if invalid_lengths:
            raise RuntimeError(
                "USC-HAD npz per-window field lengths do not match windows.shape[0]="
                f"{self.actual_num_samples}: {invalid_lengths}."
            )
        if not np.array_equal(
            self.labels_1based,
            np.asarray(self.targets, dtype=np.int64) + 1,
        ):
            raise RuntimeError(
                "USC-HAD labels and labels_1based are inconsistent; expected "
                "labels_1based == labels + 1."
            )
        if not np.any(self.stat_mask):
            raise RuntimeError("USC-HAD npz stat_mask is empty.")

        self.normalization_mean = self.npz_mean.copy()
        self.normalization_std = self.npz_std.copy()
        self.normalization_mode = "npz_stored"
        self.normalization_raw_source = "not_reconstructed"
        self.normalization_stat_subjects = sorted(
            set(self.subject_ids[self.stat_mask].astype(np.int64).tolist())
        )
        self.normalization_stat_classes = sorted(
            set(
                np.asarray(self.targets, dtype=np.int64)[self.stat_mask].tolist()
            )
        )
        unique_trials, trial_counts = np.unique(
            self.trial_global_ids, return_counts=True
        )
        self.full_trial_window_counts = {
            int(trial_id): int(count)
            for trial_id, count in zip(unique_trials, trial_counts)
        }

        if "channel_names" in npz:  # 如果 npz 里保存了通道名
            self.channel_names = np.asarray(npz["channel_names"], dtype=object).tolist()  # 读取通道名列表
        else:  # 如果 npz 中没有通道名
            self.channel_names = None  # 将通道名设为 None，避免访问时报错
        npz.close()

        if indices is not None:  # 如果初始化时传入了子集索引
            self.select_indices(indices)  # 只保留指定索引对应的样本

    def select_indices(self, indices: Sequence[int]):  # 定义根据索引筛选子集的方法
        indices = np.asarray(indices, dtype=np.int64)  # 将输入索引转成 int64 numpy 数组

        self.data = self.data[indices]  # 按索引筛选窗口数据
        self.targets = np.asarray(self.targets, dtype=np.int64)[indices].tolist()  # 按索引筛选标签，并重新转成 list
        self.uq_idxs = self.uq_idxs[indices]  # 按索引筛选唯一窗口 id
        self.labels_1based = self.labels_1based[indices]  # 按索引筛选 1-based 标签
        self.subject_ids = self.subject_ids[indices]  # 按索引筛选 subject id
        self.trial_numbers = self.trial_numbers[indices]  # 按索引筛选 subject 内 trial 编号
        self.trial_global_ids = self.trial_global_ids[indices]  # 按索引筛选 trial id
        self.window_indices = self.window_indices[indices]  # 按索引筛选 trial 内窗口序号
        self.window_start_indices = self.window_start_indices[indices]  # 按索引筛选窗口起始帧
        self.stat_mask = self.stat_mask[indices]

        return self  # 返回自身，方便链式调用或外部接收

    def __getitem__(self, item: int):  # 定义 Dataset 取样逻辑
        x = torch.from_numpy(self.data[item]).float()  # 取出第 item 个窗口，并转成 torch tensor，形状为 [C, T]
        label = int(self.targets[item])  # 取出第 item 个样本的 0-based 标签
        uq_idx = int(self.uq_idxs[item])  # 取出第 item 个样本的唯一索引

        if self.target_transform is not None:  # 如果外部设置了标签映射函数
            label = self.target_transform(label)  # 对标签进行重新映射，适配 HHR 的 old/new 类顺序

        if self.transform is not None:  # 如果外部设置了输入变换函数
            x = self.transform(x)  # 对输入窗口进行增强或变换

        return x, label, uq_idx  # 返回 HHR Dataset 兼容的三元组

    def __len__(self):  # 定义 Dataset 长度
        return len(self.targets)  # 返回样本数量


def recompute_subject_train_normalization(
    training_set: USCHADWindowDataset,
    test_set: USCHADWindowDataset,
    train_subjects: Sequence[int],
    train_classes: Sequence[int],
    eps: float = 1e-6,
):
    """Apply one fold-specific z-score transform to both dataset views."""
    eps = float(eps)
    if not np.isfinite(eps) or eps <= 0:
        raise ValueError(f"Normalization eps must be finite and positive, got {eps}.")
    if training_set.normalization_mode != "npz_stored":
        raise RuntimeError("Fold normalization was requested more than once.")
    if test_set.normalization_mode != "npz_stored":
        raise RuntimeError("Fold normalization was requested more than once.")
    if Path(training_set.npz_path).resolve() != Path(test_set.npz_path).resolve():
        raise RuntimeError("Training/test dataset views use different USC-HAD npz files.")
    if training_set.data.shape != test_set.data.shape:
        raise RuntimeError(
            "Training/test dataset views have different window shapes before "
            "fold normalization."
        )

    metadata_fields = [
        "targets",
        "subject_ids",
        "trial_global_ids",
        "window_indices",
        "window_start_indices",
    ]
    for field in metadata_fields:
        left = np.asarray(getattr(training_set, field))
        right = np.asarray(getattr(test_set, field))
        if not np.array_equal(left, right):
            raise RuntimeError(
                f"Training/test dataset metadata differs for {field!r}."
            )

    train_subjects = sorted(set(int(value) for value in train_subjects))
    train_classes = sorted(set(int(value) for value in train_classes))
    if not train_subjects:
        raise ValueError("Fold normalization requires at least one training subject.")
    if not train_classes:
        raise ValueError("Fold normalization requires at least one training class.")

    subject_ids = np.asarray(training_set.subject_ids, dtype=np.int64)
    targets = np.asarray(training_set.targets, dtype=np.int64)
    stat_mask = np.isin(subject_ids, np.asarray(train_subjects, dtype=np.int64))
    stat_mask &= np.isin(targets, np.asarray(train_classes, dtype=np.int64))
    if not np.any(stat_mask):
        raise RuntimeError(
            "No old-class windows from the fold training subjects are available "
            "for normalization."
        )

    with np.load(training_set.npz_path, allow_pickle=True) as npz:
        if "windows_raw" in npz.files:
            raw_windows = np.asarray(npz["windows_raw"], dtype=np.float32)
            raw_source = "npz_windows_raw"
        else:
            raw_windows = (
                training_set.data * training_set.npz_std
                + training_set.npz_mean
            ).astype(np.float32)
            raw_source = "reconstructed_from_npz_windows_mean_std"

    if raw_windows.shape != training_set.data.shape:
        raise RuntimeError(
            f"Raw USC-HAD window shape {raw_windows.shape} does not match "
            f"normalized windows {training_set.data.shape}."
        )
    if not np.all(np.isfinite(raw_windows)):
        raise RuntimeError("Reconstructed USC-HAD raw windows contain non-finite values.")

    stat_windows = raw_windows[stat_mask]
    fold_mean = stat_windows.mean(
        axis=(0, 2), keepdims=True, dtype=np.float64
    ).astype(np.float32)
    fold_std = stat_windows.std(
        axis=(0, 2), keepdims=True, dtype=np.float64
    ).astype(np.float32)
    fold_std = np.maximum(fold_std, np.float32(eps))
    if not np.all(np.isfinite(fold_mean)):
        raise RuntimeError("Fold normalization mean contains non-finite values.")
    if not np.all(np.isfinite(fold_std)) or np.any(fold_std <= 0):
        raise RuntimeError("Fold normalization std must be finite and positive.")

    normalized_windows = ((raw_windows - fold_mean) / fold_std).astype(np.float32)
    if not np.all(np.isfinite(normalized_windows)):
        raise RuntimeError("Fold-normalized USC-HAD windows contain non-finite values.")

    for dataset, data in [
        (training_set, normalized_windows),
        (test_set, normalized_windows.copy()),
    ]:
        dataset.data = data
        dataset.stat_mask = stat_mask.copy()
        dataset.normalization_mean = fold_mean.copy()
        dataset.normalization_std = fold_std.copy()
        dataset.normalization_mode = "fold_train_subjects_old_classes"
        dataset.normalization_raw_source = raw_source
        dataset.normalization_stat_subjects = train_subjects.copy()
        dataset.normalization_stat_classes = train_classes.copy()

    if not np.array_equal(training_set.stat_mask, test_set.stat_mask):
        raise RuntimeError("Fold normalization produced inconsistent stat masks.")
    if not np.array_equal(
        training_set.normalization_mean, test_set.normalization_mean
    ) or not np.array_equal(
        training_set.normalization_std, test_set.normalization_std
    ):
        raise RuntimeError("Training/test dataset views received different fold stats.")

    return {
        "mode": training_set.normalization_mode,
        "raw_source": raw_source,
        "stat_subjects": train_subjects,
        "stat_classes": train_classes,
        "stat_window_count": int(stat_mask.sum()),
        "mean": fold_mean.copy(),
        "std": fold_std.copy(),
    }


class USCHADTrialDataset(Dataset):
    """Group all windows from one USC-HAD trial into one variable-length sample."""

    sample_unit = "trial"

    def __init__(self, window_dataset: USCHADWindowDataset, transform=None):
        if window_dataset is None or len(window_dataset) == 0:
            raise ValueError("USCHADTrialDataset requires a non-empty window dataset.")

        self.npz_path = window_dataset.npz_path
        self.transform = transform
        self.target_transform = window_dataset.target_transform
        self.channel_names = window_dataset.channel_names
        self.actual_num_channels = int(window_dataset.actual_num_channels)
        self.actual_window_size = int(window_dataset.actual_window_size)
        self.normalization_mean = window_dataset.normalization_mean.copy()
        self.normalization_std = window_dataset.normalization_std.copy()
        self.normalization_mode = str(window_dataset.normalization_mode)
        self.normalization_raw_source = str(
            window_dataset.normalization_raw_source
        )
        self.normalization_stat_subjects = list(
            window_dataset.normalization_stat_subjects
        )
        self.normalization_stat_classes = list(
            window_dataset.normalization_stat_classes
        )
        self.has_windows_raw = bool(window_dataset.has_windows_raw)
        self.window_data = window_dataset.data
        self.window_targets = np.asarray(window_dataset.targets, dtype=np.int64)
        self.window_labels_1based = np.asarray(
            window_dataset.labels_1based, dtype=np.int64
        )
        self.window_subject_ids = np.asarray(
            window_dataset.subject_ids, dtype=np.int64
        )
        self.window_trial_numbers = np.asarray(
            window_dataset.trial_numbers, dtype=np.int64
        )
        self.window_trial_global_ids = np.asarray(
            window_dataset.trial_global_ids, dtype=np.int64
        )
        self.window_indices = np.asarray(
            window_dataset.window_indices, dtype=np.int64
        )
        self.window_start_indices = np.asarray(
            window_dataset.window_start_indices, dtype=np.int64
        )
        self.window_uq_idxs = np.asarray(window_dataset.uq_idxs, dtype=np.int64)

        self._trial_window_indices = []
        trial_targets = []
        trial_labels_1based = []
        trial_subject_ids = []
        trial_numbers = []
        trial_global_ids = []

        for trial_id in np.unique(self.window_trial_global_ids):
            local_indices = np.flatnonzero(
                self.window_trial_global_ids == int(trial_id)
            )
            order = np.argsort(
                self.window_start_indices[local_indices], kind="stable"
            )
            local_indices = local_indices[order]
            targets = np.unique(self.window_targets[local_indices])
            labels_1based = np.unique(self.window_labels_1based[local_indices])
            subjects = np.unique(self.window_subject_ids[local_indices])
            source_trial_numbers = np.unique(
                self.window_trial_numbers[local_indices]
            )

            if len(local_indices) == 0:
                raise RuntimeError(f"Trial {int(trial_id)} contains no windows.")
            if (
                len(targets) != 1
                or len(labels_1based) != 1
                or len(subjects) != 1
                or len(source_trial_numbers) != 1
            ):
                raise RuntimeError(
                    f"Trial {int(trial_id)} is inconsistent: targets={targets.tolist()}, "
                    f"labels_1based={labels_1based.tolist()}, subjects={subjects.tolist()}, "
                    f"trial_numbers={source_trial_numbers.tolist()}."
                )

            expected_window_count = window_dataset.full_trial_window_counts.get(
                int(trial_id)
            )
            if expected_window_count is None or len(local_indices) != expected_window_count:
                raise RuntimeError(
                    f"Trial {int(trial_id)} is incomplete: selected "
                    f"{len(local_indices)} of {expected_window_count} windows. "
                    "Trial-level samples must contain every preprocessed window."
                )

            starts = self.window_start_indices[local_indices]
            if np.any(np.diff(starts) <= 0):
                raise RuntimeError(
                    f"Trial {int(trial_id)} window_start_indices are not strictly increasing: "
                    f"{starts.tolist()}."
                )
            ordered_window_indices = self.window_indices[local_indices]
            expected_window_indices = np.arange(len(local_indices), dtype=np.int64)
            if not np.array_equal(ordered_window_indices, expected_window_indices):
                raise RuntimeError(
                    f"Trial {int(trial_id)} window_indices are not contiguous from zero: "
                    f"{ordered_window_indices.tolist()}."
                )

            self._trial_window_indices.append(local_indices.astype(np.int64))
            trial_targets.append(int(targets[0]))
            trial_labels_1based.append(int(labels_1based[0]))
            trial_subject_ids.append(int(subjects[0]))
            trial_numbers.append(int(source_trial_numbers[0]))
            trial_global_ids.append(int(trial_id))

        self.targets = trial_targets
        self.labels_1based = np.asarray(trial_labels_1based, dtype=np.int64)
        self.subject_ids = np.asarray(trial_subject_ids, dtype=np.int64)
        self.trial_numbers = np.asarray(trial_numbers, dtype=np.int64)
        self.trial_global_ids = np.asarray(trial_global_ids, dtype=np.int64)
        self.uq_idxs = self.trial_global_ids.copy()
        self.actual_num_samples = len(self.targets)

    @staticmethod
    def _build_view(
        bag: torch.Tensor,
        selected_starts: torch.Tensor,
        full_starts: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        first_start = full_starts[0].float()
        denominator = torch.clamp(
            full_starts[-1].float() - first_start,
            min=1.0,
        )
        relative = (selected_starts.float() - first_start) / denominator
        positions = torch.stack(
            [
                relative,
                torch.sin(2.0 * math.pi * relative),
                torch.cos(2.0 * math.pi * relative),
            ],
            dim=1,
        )
        return {
            "windows": bag.float(),
            "positions": positions.float(),
        }

    def __getitem__(self, item: int):
        local_indices = self._trial_window_indices[item]
        bag = torch.from_numpy(self.window_data[local_indices]).float()
        full_starts = torch.from_numpy(
            self.window_start_indices[local_indices]
        ).long()
        label = int(self.targets[item])
        trial_id = int(self.trial_global_ids[item])

        if self.target_transform is not None:
            label = self.target_transform(label)

        if self.transform is None:
            x = self._build_view(bag, full_starts, full_starts)
        else:
            transformed_views = self.transform(bag, full_starts)
            x = [
                self._build_view(view_bag, view_starts, full_starts)
                for view_bag, view_starts in transformed_views
            ]

        return x, label, trial_id

    def __len__(self):
        return len(self.targets)


def _collate_trial_view(views: Sequence[Dict[str, torch.Tensor]]):
    if len(views) == 0:
        raise ValueError("Cannot collate an empty trial batch.")

    lengths = torch.as_tensor(
        [int(view["windows"].shape[0]) for view in views],
        dtype=torch.long,
    )
    if torch.any(lengths <= 0):
        raise ValueError(f"Trial batch contains an empty bag: {lengths.tolist()}.")

    max_len = int(lengths.max().item())
    channels = int(views[0]["windows"].shape[1])
    window_size = int(views[0]["windows"].shape[2])
    position_dim = int(views[0]["positions"].shape[1])
    windows = torch.zeros(
        len(views), max_len, channels, window_size, dtype=torch.float32
    )
    positions = torch.zeros(
        len(views), max_len, position_dim, dtype=torch.float32
    )
    mask = torch.zeros(len(views), max_len, dtype=torch.bool)

    for batch_id, view in enumerate(views):
        view_windows = view["windows"]
        view_positions = view["positions"]
        length = int(view_windows.shape[0])
        if tuple(view_windows.shape[1:]) != (channels, window_size):
            raise ValueError("All trial windows in a batch must share [C,T].")
        if tuple(view_positions.shape) != (length, position_dim):
            raise ValueError(
                "Trial positions must have shape [L,P], got "
                f"{tuple(view_positions.shape)} for L={length}."
            )
        windows[batch_id, :length] = view_windows
        positions[batch_id, :length] = view_positions
        mask[batch_id, :length] = True

    return {
        "windows": windows,
        "positions": positions,
        "mask": mask,
        "lengths": lengths,
    }


def uschad_trial_collate(batch):
    """Collate offline/test/online tuples containing variable-length trial bags."""
    if len(batch) == 0:
        raise ValueError("Cannot collate an empty batch.")
    tuple_size = len(batch[0])
    if tuple_size not in [3, 4]:
        raise ValueError(f"Unexpected USC-HAD trial item size: {tuple_size}.")

    inputs = [item[0] for item in batch]
    labels = torch.as_tensor([int(item[1]) for item in batch], dtype=torch.long)
    uq_idxs = torch.as_tensor([int(item[2]) for item in batch], dtype=torch.long)

    if isinstance(inputs[0], list):
        num_views = len(inputs[0])
        if any(not isinstance(item, list) or len(item) != num_views for item in inputs):
            raise ValueError("Every trial in a training batch must have the same view count.")
        collated_inputs = [
            _collate_trial_view([item[view_id] for item in inputs])
            for view_id in range(num_views)
        ]
    else:
        collated_inputs = _collate_trial_view(inputs)

    if tuple_size == 3:
        return collated_inputs, labels, uq_idxs

    flags = torch.as_tensor(
        [int(np.asarray(item[3]).reshape(-1)[0]) for item in batch],
        dtype=torch.long,
    )
    return collated_inputs, labels, uq_idxs, flags


def subsample_dataset(dataset: USCHADWindowDataset, idxs: Sequence[int]):  # 定义按样本索引抽取子数据集的函数
    if idxs is None or len(idxs) == 0:  # 如果索引为空
        return None  # 返回 None，表示该子集不存在

    dataset = deepcopy(dataset)  # 深拷贝数据集，避免修改原始 Dataset
    dataset = dataset.select_indices(idxs)  # 根据索引筛选数据
    return dataset  # 返回筛选后的子数据集

def parse_subject_ids(subject_ids_str):
    """
    Parse subject ids from command line string.

    Examples:
        ""          -> None
        "1,2,3"     -> [1, 2, 3]
        "1 2 3"     -> [1, 2, 3]
        "1, 2, 3"   -> [1, 2, 3]
    """

    if subject_ids_str is None:
        return None

    subject_ids_str = str(subject_ids_str).strip()

    if subject_ids_str == "":
        return None

    subject_ids_str = subject_ids_str.replace(",", " ")

    subject_ids = [
        int(x)
        for x in subject_ids_str.split()
        if x.strip() != ""
    ]

    if len(subject_ids) == 0:
        return None

    return subject_ids

def subsample_by_subjects(dataset: USCHADWindowDataset, include_subjects):
    """
    Select samples whose subject_id belongs to include_subjects.

    This is used for subject-level split.

    Input:
        dataset:
            USCHADWindowDataset or its subset.

        include_subjects:
            A list / set of subject ids.

    Output:
        A deepcopy of dataset containing only selected subjects.
    """

    if dataset is None:
        return None

    if include_subjects is None or len(include_subjects) == 0:
        return None

    include_subjects = set([int(s) for s in include_subjects])

    selected_indices = [
        i
        for i, subject_id in enumerate(dataset.subject_ids)
        if int(subject_id) in include_subjects
    ]

    if len(selected_indices) == 0:
        return None

    return subsample_dataset(dataset, selected_indices)

def build_subject_split(dataset: USCHADWindowDataset,prop_train_labels: float = 0.8,seed: int = 0,train_subjects=None,test_subjects=None,):
    """
    Build a global subject-level train/test split.

    Important:
        This split is global, not class-wise.

    Why:
        If we split subjects separately inside each class,
        the same subject may appear in train for class 0
        and test for class 1.

        That still causes subject leakage.

    Return:
        train_subjects_set, test_subjects_set
    """

    all_subjects = sorted(
        set([int(s) for s in np.asarray(dataset.subject_ids).tolist()])
    )

    if len(all_subjects) < 2:
        raise RuntimeError(
            f"Subject-level split requires at least 2 subjects, "
            f"but got {len(all_subjects)}."
        )

    train_subjects = None if train_subjects is None else set([int(s) for s in train_subjects])
    test_subjects = None if test_subjects is None else set([int(s) for s in test_subjects])

    all_subjects_set = set(all_subjects)

    # Case 1: user explicitly provides both train and test subjects.
    if train_subjects is not None and test_subjects is not None:
        overlap = train_subjects & test_subjects

        if len(overlap) > 0:
            raise RuntimeError(
                f"Invalid subject split: train/test subject overlap = {sorted(overlap)}"
            )

        unknown_subjects = (train_subjects | test_subjects) - all_subjects_set

        if len(unknown_subjects) > 0:
            raise RuntimeError(
                f"Invalid subject split: unknown subject ids = {sorted(unknown_subjects)}. "
                f"Available subjects = {all_subjects}"
            )

        return train_subjects, test_subjects

    # Case 2: user only provides test subjects.
    if train_subjects is None and test_subjects is not None:
        unknown_subjects = test_subjects - all_subjects_set

        if len(unknown_subjects) > 0:
            raise RuntimeError(
                f"Invalid test subjects = {sorted(unknown_subjects)}. "
                f"Available subjects = {all_subjects}"
            )

        train_subjects = all_subjects_set - test_subjects

        if len(train_subjects) == 0:
            raise RuntimeError("train_subjects is empty after using explicit test_subjects.")

        return train_subjects, test_subjects

    # Case 3: user only provides train subjects.
    if train_subjects is not None and test_subjects is None:
        unknown_subjects = train_subjects - all_subjects_set

        if len(unknown_subjects) > 0:
            raise RuntimeError(
                f"Invalid train subjects = {sorted(unknown_subjects)}. "
                f"Available subjects = {all_subjects}"
            )

        test_subjects = all_subjects_set - train_subjects

        if len(test_subjects) == 0:
            raise RuntimeError("test_subjects is empty after using explicit train_subjects.")

        return train_subjects, test_subjects

    # Case 4: no explicit subjects. Randomly split subjects globally.
    rng = np.random.default_rng(seed)

    shuffled_subjects = np.asarray(all_subjects, dtype=np.int64)
    rng.shuffle(shuffled_subjects)

    n_subjects = len(shuffled_subjects)

    n_train = int(round(n_subjects * float(prop_train_labels)))
    n_train = max(1, min(n_subjects - 1, n_train))

    train_subjects = set([int(s) for s in shuffled_subjects[:n_train]])
    test_subjects = set([int(s) for s in shuffled_subjects[n_train:]])

    if len(train_subjects & test_subjects) > 0:
        raise RuntimeError("Internal error: subject split overlap detected.")

    return train_subjects, test_subjects

def split_labeled_unlabeled_by_subject(dataset: USCHADWindowDataset,train_subjects,test_subjects) -> Tuple[np.ndarray, np.ndarray]:
    """
    Split one class dataset into labeled / unlabeled indices by subject.

    This function must match split_labeled_unlabeled_by_trial(...).

    Existing trial-level function returns:
        labeled_indices, unlabeled_indices

    Therefore this subject-level function also returns:
        labeled_indices, unlabeled_indices

    For old classes:
        labeled_indices   = samples from train subjects
        unlabeled_indices = samples from test subjects
    """

    if dataset is None or len(dataset) == 0:
        return np.asarray([], dtype=np.int64), np.asarray([], dtype=np.int64)

    subjects = np.asarray(dataset.subject_ids, dtype=np.int64)

    train_subjects = set([int(s) for s in train_subjects])
    test_subjects = set([int(s) for s in test_subjects])

    overlap = train_subjects & test_subjects
    if len(overlap) > 0:
        raise RuntimeError(
            f"Invalid subject split: train/test subject overlap = {sorted(overlap)}"
        )

    labeled_indices = np.asarray(
        [
            i
            for i, subject_id in enumerate(subjects)
            if int(subject_id) in train_subjects
        ],
        dtype=np.int64,
    )

    unlabeled_indices = np.asarray(
        [
            i
            for i, subject_id in enumerate(subjects)
            if int(subject_id) in test_subjects
        ],
        dtype=np.int64,
    )

    return labeled_indices, unlabeled_indices

def subsample_by_uq_idxs(dataset: USCHADWindowDataset, uq_idxs: Sequence[int]):
    """
    Select samples from dataset according to original uq_idx values.

    Why this function is needed:
        offline_train_dataset is built from whole_training_set.
        offline_test_dataset should be built from whole_test_set.
        Their local positions may differ after subsampling, but uq_idx is stable.

    Input:
        dataset:
            Usually whole_test_set.

        uq_idxs:
            Original unique window ids that should be selected.

    Output:
        A deepcopy of dataset containing only samples whose uq_idx is in uq_idxs.
    """

    if uq_idxs is None or len(uq_idxs) == 0:
        return None

    uq_set = set([int(x) for x in uq_idxs])

    selected_local_indices = [
        local_idx
        for local_idx, uq_idx in enumerate(dataset.uq_idxs)
        if int(uq_idx) in uq_set
    ]

    if len(selected_local_indices) == 0:
        return None

    return subsample_dataset(dataset, selected_local_indices)


def subsample_by_trial_ids(
    dataset: USCHADWindowDataset,
    trial_ids: Sequence[int],
):
    """Select every window belonging to the requested complete trials."""
    if dataset is None or trial_ids is None or len(trial_ids) == 0:
        return None
    trial_set = set(np.asarray(trial_ids, dtype=np.int64).tolist())
    selected_local_indices = np.flatnonzero(
        np.isin(
            np.asarray(dataset.trial_global_ids, dtype=np.int64),
            np.asarray(sorted(trial_set), dtype=np.int64),
        )
    )
    if len(selected_local_indices) == 0:
        return None
    selected = subsample_dataset(dataset, selected_local_indices)
    selected_trials = set(
        np.asarray(selected.trial_global_ids, dtype=np.int64).tolist()
    )
    if selected_trials != trial_set:
        raise RuntimeError(
            f"Trial selection mismatch: requested={sorted(trial_set)}, "
            f"selected={sorted(selected_trials)}."
        )
    return selected


def subsample_exclude_uq_or_trial(dataset: USCHADWindowDataset,exclude_uq_idxs=None,exclude_trial_ids=None,):
    """
    Exclude samples whose uq_idx or trial_global_id has been used by online train.

    This is used to prevent online train/test leakage.

    Why:
        online old/novel train samples are drawn from held-out/test subjects.
        online test is also drawn from held-out/test subjects.
        Therefore, without explicit exclusion, the exact same window or trial
        may appear in both online train and online test.
    """

    if dataset is None:
        return None

    exclude_uq_set = set()
    exclude_trial_set = set()

    if exclude_uq_idxs is not None and len(exclude_uq_idxs) > 0:
        exclude_uq_set = set([int(x) for x in np.asarray(exclude_uq_idxs, dtype=np.int64).tolist()])

    if exclude_trial_ids is not None and len(exclude_trial_ids) > 0:
        exclude_trial_set = set([int(x) for x in np.asarray(exclude_trial_ids, dtype=np.int64).tolist()])

    selected_local_indices = []

    for local_idx in range(len(dataset)):
        uq_idx = int(dataset.uq_idxs[local_idx])
        trial_id = int(dataset.trial_global_ids[local_idx])

        if uq_idx in exclude_uq_set:
            continue

        if trial_id in exclude_trial_set:
            continue

        selected_local_indices.append(local_idx)

    if len(selected_local_indices) == 0:
        return None

    return subsample_dataset(dataset, selected_local_indices)
def subsample_classes(dataset: USCHADWindowDataset, include_classes=(0, 1)):  # 定义按类别抽取子数据集的函数
    include_classes = set([int(c) for c in include_classes])  # 将目标类别列表转成 int 集合，方便判断
    cls_idxs = [i for i, target in enumerate(dataset.targets) if int(target) in include_classes]  # 找到属于目标类别的样本索引
    return subsample_dataset(dataset, cls_idxs)  # 根据样本索引返回子数据集

def subDataset_wholeDataset(datalist: Sequence[USCHADWindowDataset]):  # 定义将多个子数据集合并成一个数据集的函数
    valid_datasets = [dataset for dataset in datalist if dataset is not None and len(dataset) > 0]  # 过滤掉 None 或空数据集

    if len(valid_datasets) == 0:  # 如果没有任何有效数据集
        return None  # 返回 None，表示合并失败

    whole_dataset = deepcopy(valid_datasets[0])  # 以第一个有效数据集为模板进行深拷贝
    whole_dataset.data = np.concatenate([dataset.data for dataset in valid_datasets], axis=0)  # 拼接所有窗口数据
    whole_dataset.targets = np.concatenate([np.asarray(dataset.targets, dtype=np.int64) for dataset in valid_datasets], axis=0).tolist()  # 拼接所有标签
    whole_dataset.uq_idxs = np.concatenate([dataset.uq_idxs for dataset in valid_datasets], axis=0)  # 拼接所有唯一索引
    whole_dataset.labels_1based = np.concatenate([dataset.labels_1based for dataset in valid_datasets], axis=0)  # 拼接所有 1-based 标签
    whole_dataset.subject_ids = np.concatenate([dataset.subject_ids for dataset in valid_datasets], axis=0)  # 拼接所有 subject id
    whole_dataset.trial_numbers = np.concatenate([dataset.trial_numbers for dataset in valid_datasets], axis=0)  # 拼接 subject 内 trial 编号
    whole_dataset.trial_global_ids = np.concatenate([dataset.trial_global_ids for dataset in valid_datasets], axis=0)  # 拼接所有 trial id
    whole_dataset.window_indices = np.concatenate([dataset.window_indices for dataset in valid_datasets], axis=0)  # 拼接所有窗口序号
    whole_dataset.window_start_indices = np.concatenate([dataset.window_start_indices for dataset in valid_datasets], axis=0)  # 拼接所有窗口起始帧
    whole_dataset.stat_mask = np.concatenate([dataset.stat_mask for dataset in valid_datasets], axis=0)

    return whole_dataset  # 返回合并后的完整数据集

def safe_random_choice(indices: Sequence[int], sample_num: int, seed: Optional[int] = None):  # 定义安全随机采样函数
    indices = np.asarray(indices, dtype=np.int64)  # 将候选索引转成 numpy 数组
    sample_num = int(sample_num)  # 将采样数量转成 int

    if len(indices) == 0:  # 如果候选索引为空
        return np.asarray([], dtype=np.int64)  # 返回空数组

    sample_num = min(sample_num, len(indices))  # 防止采样数量超过候选样本数量

    rng = np.random.default_rng(seed)  # 根据 seed 创建随机数生成器
    selected = rng.choice(indices, size=sample_num, replace=False)  # 无放回随机采样
    return selected.astype(np.int64)  # 返回 int64 类型采样结果

def split_labeled_unlabeled_by_trial(dataset: USCHADWindowDataset, prop_train_labels: float, seed: int = 0) -> Tuple[np.ndarray, np.ndarray]:  # 定义按 trial 划分 labeled/unlabeled 的函数
    targets = np.asarray(dataset.targets, dtype=np.int64)  # 获取该数据集的标签数组
    unique_classes = np.unique(targets)  # 获取该数据集中包含的唯一类别

    if len(unique_classes) != 1:  # 该函数要求输入的是单类别数据集
        raise ValueError(f"split_labeled_unlabeled_by_trial expects one class, but got {unique_classes.tolist()}")  # 如果不止一个类别，直接报错

    trial_ids = np.asarray(dataset.trial_global_ids, dtype=np.int64)  # 获取每个窗口所属 trial id
    unique_trials = np.unique(trial_ids)  # 获取该类别下所有 trial id

    rng = np.random.default_rng(seed)  # 根据 seed 创建随机数生成器
    shuffled_trials = unique_trials.copy()  # 复制 trial id，避免原数组被修改
    rng.shuffle(shuffled_trials)  # 随机打乱 trial 顺序

    num_labeled_trials = int(round(len(shuffled_trials) * float(prop_train_labels)))  # 根据比例计算 labeled trial 数量
    num_labeled_trials = max(1, min(num_labeled_trials, len(shuffled_trials)))  # 限制 labeled trial 数量至少为 1，最多为全部 trial

    labeled_trials = set(shuffled_trials[:num_labeled_trials].tolist())  # 取前一部分 trial 作为 labeled trial
    labeled_indices = np.asarray([i for i, trial_id in enumerate(trial_ids) if int(trial_id) in labeled_trials], dtype=np.int64)  # 找到 labeled trial 对应的窗口索引
    unlabeled_indices = np.asarray([i for i, trial_id in enumerate(trial_ids) if int(trial_id) not in labeled_trials], dtype=np.int64)  # 找到 unlabeled trial 对应的窗口索引

    if len(unlabeled_indices) == 0 and len(labeled_indices) > 1:  # 如果 labeled 比例太大导致 unlabeled 为空
        all_indices = np.arange(len(dataset), dtype=np.int64)  # 创建该类别下所有窗口索引
        rng.shuffle(all_indices)  # 打乱窗口索引
        num_labeled = int(round(len(all_indices) * float(prop_train_labels)))  # 按窗口数量重新计算 labeled 数量
        num_labeled = max(1, min(num_labeled, len(all_indices) - 1))  # 确保至少留出一个 unlabeled 样本
        labeled_indices = all_indices[:num_labeled]  # 前一部分作为 labeled
        unlabeled_indices = all_indices[num_labeled:]  # 后一部分作为 unlabeled

    return labeled_indices, unlabeled_indices  # 返回 labeled 和 unlabeled 的样本索引

def build_har_train_transform(args):
    """
    Build USC-HAD train transform.

    Default:
        --har_aug_mode none

    Meaning:
        no jittering,
        no scaling,
        no time masking,
        but still return n_views cloned tensors for HHR training.
    """

    har_aug_mode = getattr(args, "har_aug_mode", "none")

    weak_jitter_std = getattr(args, "har_weak_jitter_std", 0.01)
    weak_scale_std = getattr(args, "har_weak_scale_std", 0.05)
    strong_jitter_std = getattr(args, "har_strong_jitter_std", 0.02)
    strong_scale_std = getattr(args, "har_strong_scale_std", 0.10)
    strong_time_mask_ratio = getattr(args, "har_time_mask_ratio", 0.10)

    print("=" * 80)
    print("[USCHAD HAR AUG CHECK]")
    print(f"har_aug_mode = {har_aug_mode}")
    print(f"weak_jitter_std = {weak_jitter_std}")
    print(f"weak_scale_std = {weak_scale_std}")
    print(f"strong_jitter_std = {strong_jitter_std}")
    print(f"strong_scale_std = {strong_scale_std}")
    print(f"strong_time_mask_ratio = {strong_time_mask_ratio}")
    print("=" * 80)

    return HARContrastiveTransform(
        n_views=args.n_views,
        aug_mode=har_aug_mode,
        weak_jitter_std=weak_jitter_std,
        weak_scale_std=weak_scale_std,
        strong_jitter_std=strong_jitter_std,
        strong_scale_std=strong_scale_std,
        strong_time_mask_ratio=strong_time_mask_ratio,
    )


def build_har_trial_train_transform(args):
    transform = HARTrialContrastiveTransform(
        n_views=args.n_views,
        view_mode=getattr(args, "trial_view_mode", "full_random_crop"),
        crop_ratio=getattr(args, "trial_crop_ratio", 2.0 / 3.0),
        min_windows=getattr(args, "trial_min_windows", 2),
        aug_mode=getattr(args, "har_aug_mode", "none"),
        weak_jitter_std=getattr(args, "har_weak_jitter_std", 0.01),
        weak_scale_std=getattr(args, "har_weak_scale_std", 0.05),
        strong_jitter_std=getattr(args, "har_strong_jitter_std", 0.02),
        strong_scale_std=getattr(args, "har_strong_scale_std", 0.10),
        strong_time_mask_ratio=getattr(args, "har_time_mask_ratio", 0.10),
    )
    print("=" * 80)
    print("[USCHAD TRIAL VIEW CHECK]")
    print(f"view_mode = {transform.view_mode}")
    print(f"crop_ratio = {transform.crop_ratio}")
    print(f"min_windows = {transform.min_windows}")
    print(f"har_aug_mode = {transform.aug_mode}")
    print(f"weak_jitter_std = {transform.weak_jitter_std}")
    print(f"weak_scale_std = {transform.weak_scale_std}")
    print(f"strong_jitter_std = {transform.strong_jitter_std}")
    print(f"strong_scale_std = {transform.strong_scale_std}")
    print(f"strong_time_mask_ratio = {transform.strong_time_mask_ratio}")
    print("view[0] = full trial")
    print(
        "view[1] = "
        + ("full trial" if transform.view_mode == "full_full" else "random contiguous crop")
    )
    print("=" * 80)
    return transform


def convert_window_dataset_to_trials(
    dataset: USCHADWindowDataset,
    transform,
):
    if dataset is None:
        return None
    return USCHADTrialDataset(dataset, transform=transform)


def get_uschad_datasets(train_transform, test_transform, config_dict, train_classes=range(6), prop_train_labels=0.8, split_train_val=False, is_shuffle=False, seed=0, args=None):  # 定义 USC-HAD 数据集构造入口
    if args is None:  # 检查是否传入 args
        raise ValueError("get_uschad_datasets requires args, because args.uschad_npz_path is needed.")  # 没有 args 就无法获取 npz 路径

    npz_path = getattr(args, "uschad_npz_path", "")  # 从 args 中读取预处理 npz 文件路径

    if npz_path is None or len(str(npz_path)) == 0:  # 检查 npz 路径是否为空
        raise ValueError("Please set --uschad_npz_path to your processed uschad_windows.npz")  # 如果为空，提示用户设置路径

    continual_session_num = int(config_dict["continual_session_num"])  # 自动推导后的 continual session 数量
    targets_per_session = int(
        getattr(
            args,
            "num_novel_class_per_session",
            getattr(args, "num_novel_classes_per_session", -1),
        )
    )  # 每个 online session 新增多少 novel class

    if targets_per_session < 1:
        raise RuntimeError(
            f"num_novel_class_per_session must be at least 1, "
            f"but got {targets_per_session}."
        )

    online_novel_unseen_num = int(config_dict["online_novel_unseen_num"])  # 读取每阶段新类未见样本采样数
    online_old_seen_num = int(config_dict["online_old_seen_num"])  # 读取每阶段旧类无标签样本采样数
    online_novel_seen_num = int(config_dict["online_novel_seen_num"])  # 读取已出现过的新类在后续阶段的采样数

    sample_unit = str(getattr(args, "uschad_sample_unit", "window"))
    if sample_unit not in ["window", "trial"]:
        raise ValueError(
            f"Unsupported --uschad_sample_unit={sample_unit!r}; expected window or trial."
        )

    if sample_unit == "trial":
        har_train_transform = None
        har_test_transform = None
        har_trial_train_transform = build_har_trial_train_transform(args)
    else:
        har_train_transform = build_har_train_transform(args)
        har_test_transform = HARPlainTransform(clone=True)
        har_trial_train_transform = None

    whole_training_set = USCHADWindowDataset(npz_path=npz_path, transform=har_train_transform)  # 构造完整训练集视角的数据集
    whole_test_set = USCHADWindowDataset(npz_path=npz_path, transform=har_test_transform)  # 构造完整测试集视角的数据集

    if whole_training_set.data.shape != whole_test_set.data.shape:
        raise RuntimeError(
            f"USC-HAD train/test npz shape mismatch: "
            f"train={whole_training_set.data.shape}, test={whole_test_set.data.shape}."
        )

    expected_channels = int(getattr(args, "har_in_channels", whole_training_set.actual_num_channels))
    if whole_training_set.actual_num_channels != expected_channels:
        raise RuntimeError(
            f"USC-HAD channel mismatch: npz has {whole_training_set.actual_num_channels}, "
            f"but --har_in_channels={expected_channels}."
        )

    expected_window_size = getattr(args, "uschad_window_size", None)
    if expected_window_size is not None:
        expected_window_size = int(expected_window_size)
        if whole_training_set.actual_window_size != expected_window_size:
            raise RuntimeError(
                f"USC-HAD window mismatch: npz windows.shape[2]={whole_training_set.actual_window_size}, "
                f"but --uschad_window_size={expected_window_size}."
            )

    print("=" * 80)
    print("[USCHAD NPZ INPUT CHECK]")
    print(f"npz_path = {npz_path}")
    print(f"npz windows shape = {whole_training_set.data.shape}  # [N, C, T]")
    print(f"actual_num_samples = {whole_training_set.actual_num_samples}")
    print(f"actual_num_channels = {whole_training_set.actual_num_channels}")
    print(f"expected_num_channels = {expected_channels}")
    print(f"actual_window_size = {whole_training_set.actual_window_size}")
    print(f"expected_window_size = {expected_window_size}")
    print(f"sample_unit = {sample_unit}")
    print(
        "online sampling counts = "
        f"old:{online_old_seen_num}, novel_unseen:{online_novel_unseen_num}, "
        f"novel_seen:{online_novel_seen_num} ({sample_unit} units per class)"
    )
    print("=" * 80)
    logger = getattr(args, "logger", None)
    if logger is not None:
        logger.info(
            "USC-HAD NPZ input: "
            f"path={npz_path}, shape={tuple(whole_training_set.data.shape)}, "
            f"channels={whole_training_set.actual_num_channels}, "
            f"window={whole_training_set.actual_window_size}, "
            f"sample_unit={sample_unit}"
        )

    uschad_split_mode = getattr(args, "uschad_split_mode", "trial")

    if uschad_split_mode not in ["trial", "subject"]:
        raise ValueError(
            f"Unsupported uschad_split_mode: {uschad_split_mode}. "
            f"Expected one of ['trial', 'subject']."
        )

    train_subjects_for_split = None
    test_subjects_for_split = None

    if uschad_split_mode == "subject":
        explicit_train_subjects = parse_subject_ids(
            getattr(args, "uschad_train_subjects", "")
        )
        explicit_test_subjects = parse_subject_ids(
            getattr(args, "uschad_test_subjects", "")
        )

        train_subjects_for_split, test_subjects_for_split = build_subject_split(
            dataset=whole_training_set,
            prop_train_labels=prop_train_labels,
            seed=seed,
            train_subjects=explicit_train_subjects,
            test_subjects=explicit_test_subjects,
        )

        print("=" * 80)
        print("[USCHAD Subject-Level Split]")
        print(f"split mode: {uschad_split_mode}")
        print(f"train subjects: {sorted(train_subjects_for_split)}")
        print(f"test subjects: {sorted(test_subjects_for_split)}")
        print(f"subject overlap: {sorted(train_subjects_for_split & test_subjects_for_split)}")
        print("=" * 80)
        if logger is not None:
            logger.info(
                "USC-HAD subject split: "
                f"train={sorted(train_subjects_for_split)}, "
                f"validation={parse_subject_ids(getattr(args, 'offline_val_subjects', '')) or []}, "
                f"test={sorted(test_subjects_for_split)}, overlap=0"
            )

        if bool(
            getattr(args, "uschad_recompute_norm_from_train_subjects", False)
        ):
            normalization_report = recompute_subject_train_normalization(
                whole_training_set,
                whole_test_set,
                train_subjects=train_subjects_for_split,
                train_classes=train_classes,
                eps=float(getattr(args, "uschad_norm_eps", 1e-6)),
            )
        else:
            normalization_report = {
                "mode": whole_training_set.normalization_mode,
                "raw_source": whole_training_set.normalization_raw_source,
                "stat_subjects": list(
                    whole_training_set.normalization_stat_subjects
                ),
                "stat_classes": list(
                    whole_training_set.normalization_stat_classes
                ),
                "stat_window_count": int(whole_training_set.stat_mask.sum()),
                "mean": whole_training_set.normalization_mean,
                "std": whole_training_set.normalization_std,
            }

        print("=" * 80)
        print("[USCHAD Normalization Config]")
        print(f"mode: {normalization_report['mode']}")
        print(f"raw source: {normalization_report['raw_source']}")
        print(f"npz contains windows_raw: {whole_training_set.has_windows_raw}")
        print(f"stat subjects: {normalization_report['stat_subjects']}")
        print(f"stat classes (0-based): {normalization_report['stat_classes']}")
        print(f"stat windows: {normalization_report['stat_window_count']}")
        print(
            "normalization mean: "
            f"{np.asarray(normalization_report['mean']).reshape(-1).tolist()}"
        )
        print(
            "normalization std: "
            f"{np.asarray(normalization_report['std']).reshape(-1).tolist()}"
        )
        print("=" * 80)
        if logger is not None:
            logger.info(
                "USC-HAD effective normalization: "
                f"mode={normalization_report['mode']}, "
                f"raw_source={normalization_report['raw_source']}, "
                f"stat_subjects={normalization_report['stat_subjects']}, "
                f"stat_classes_0based={normalization_report['stat_classes']}, "
                f"stat_windows={normalization_report['stat_window_count']}, "
                f"mean={np.asarray(normalization_report['mean']).reshape(-1).tolist()}, "
                f"std={np.asarray(normalization_report['std']).reshape(-1).tolist()}"
            )

        allowed_stat_mask = np.isin(
            np.asarray(whole_training_set.subject_ids, dtype=np.int64),
            np.asarray(sorted(train_subjects_for_split), dtype=np.int64),
        ) & np.isin(
            np.asarray(whole_training_set.targets, dtype=np.int64),
            np.asarray(list(train_classes), dtype=np.int64),
        )
        leaked_stat_mask = np.asarray(whole_training_set.stat_mask, dtype=bool) & ~allowed_stat_mask
        if np.any(leaked_stat_mask):
            leaked_subjects = sorted(
                set(
                    np.asarray(whole_training_set.subject_ids, dtype=np.int64)[
                        leaked_stat_mask
                    ].tolist()
                )
            )
            leaked_classes = sorted(
                set(
                    np.asarray(whole_training_set.targets, dtype=np.int64)[
                        leaked_stat_mask
                    ].tolist()
                )
            )
            raise RuntimeError(
                "USC-HAD normalization-stat leakage detected: npz stat_mask uses "
                f"subjects={leaked_subjects}, classes={leaked_classes} outside the "
                "offline training subjects/old classes. Regenerate the npz with "
                "--source_subjects equal to --uschad_train_subjects."
            )
        if bool(
            getattr(args, "uschad_recompute_norm_from_train_subjects", False)
        ) and not np.array_equal(
            np.asarray(whole_training_set.stat_mask, dtype=bool),
            allowed_stat_mask,
        ):
            raise RuntimeError(
                "Fold normalization stat_mask does not exactly match all old-class "
                "windows from the fold training subjects."
            )
        stat_subjects = sorted(
            set(
                np.asarray(whole_training_set.subject_ids, dtype=np.int64)[
                    whole_training_set.stat_mask
                ].tolist()
            )
        )
        stat_classes = sorted(
            set(
                np.asarray(whole_training_set.targets, dtype=np.int64)[
                    whole_training_set.stat_mask
                ].tolist()
            )
        )
        print(
            "[USCHAD Normalization Leakage Check] "
            f"stat_subjects={stat_subjects}, stat_classes={stat_classes}, "
            "outside_train_or_old=0"
        )
    old_dataset_all = subsample_classes(deepcopy(whole_training_set), include_classes=train_classes)  # 从完整训练集中抽取 old classes

    if old_dataset_all is None:  # 检查 old classes 是否为空
        raise RuntimeError(f"No old-class samples found for train_classes={list(train_classes)}")  # 如果为空，说明类别划分或标签有问题

    each_old_all_samples = [subsample_classes(deepcopy(old_dataset_all), include_classes=[target]) for target in list(train_classes)]  # 将 old 数据按类别拆成多个单类数据集

    each_old_labeled_samples = []  # 创建列表，用于保存每个 old class 的 labeled 子集
    each_old_unlabeled_samples = []  # 创建列表，用于保存每个 old class 的 unlabeled 子集

    for class_id, class_dataset in zip(list(train_classes), each_old_all_samples):  # 遍历每一个 old class 数据集
        if class_dataset is None or len(class_dataset) == 0:  # 检查该 old class 是否有样本
            raise RuntimeError(f"No samples found for old class {class_id}")  # 如果该类没有样本，直接报错

        if uschad_split_mode == "subject":
            labeled_indices, unlabeled_indices = split_labeled_unlabeled_by_subject(
                class_dataset,
                train_subjects=train_subjects_for_split,
                test_subjects=test_subjects_for_split,
            )

            if len(labeled_indices) == 0:
                raise RuntimeError(
                    f"Old class {class_id} has no samples in train subjects. "
                    f"train_subjects={sorted(train_subjects_for_split)}"
                )

            if len(unlabeled_indices) == 0:
                raise RuntimeError(
                    f"Old class {class_id} has no samples in test subjects. "
                    f"test_subjects={sorted(test_subjects_for_split)}"
                )

        else:
            labeled_indices, unlabeled_indices = split_labeled_unlabeled_by_trial(
                class_dataset,
                prop_train_labels=prop_train_labels,
                seed=seed,
            )

        each_old_labeled_samples.append(subsample_dataset(class_dataset, labeled_indices))
        each_old_unlabeled_samples.append(subsample_dataset(class_dataset, unlabeled_indices))

    offline_train_dataset = subDataset_wholeDataset(
        each_old_labeled_samples)  # 合并所有 old class labeled 样本作为 offline train dataset
    if offline_train_dataset is None or len(offline_train_dataset) == 0:
        raise RuntimeError(
            "offline_train_dataset is empty. "
            "Please check train_classes, uschad_split_mode, and train subjects."
        )
    # -------------------------------------------------------------------------
    # Build offline test dataset from held-out old samples only.
    #
    # Important:
    #   Do NOT use all old-class samples as offline_test_dataset.
    #   Otherwise offline_test_dataset will contain offline_train_dataset samples,
    #   causing data leakage and unrealistically high accuracy.
    #
    # Correct logic:
    #   offline train = old labeled trials
    #   offline test  = old unlabeled / held-out trials
    #
    # Since each_old_unlabeled_samples comes from whole_training_set and uses
    # train transform, we only use its uq_idxs here. Then we re-select the same
    # uq_idxs from whole_test_set, whose transform is HARPlainTransform.
    # -------------------------------------------------------------------------
    offline_test_uq_idx_list = [
        np.asarray(dataset.uq_idxs, dtype=np.int64)
        for dataset in each_old_unlabeled_samples
        if dataset is not None and len(dataset) > 0
    ]

    if len(offline_test_uq_idx_list) == 0:
        raise RuntimeError(
            "No held-out old samples found for offline_test_dataset. "
            "Please check prop_train_labels, uschad_split_mode, and test subjects."
        )

    offline_test_uq_idxs = np.concatenate(
        offline_test_uq_idx_list,
        axis=0,
    )

    offline_test_dataset = subsample_by_uq_idxs(
        deepcopy(whole_test_set),
        offline_test_uq_idxs,
    )

    if offline_test_dataset is None or len(offline_test_dataset) == 0:
        raise RuntimeError(
            "offline_test_dataset is empty. "
            "Please check prop_train_labels, uschad_split_mode, and subject/trial split."
        )

    # Check train/test uq_idx overlap.
    offline_train_uq_set = set(np.asarray(offline_train_dataset.uq_idxs, dtype=np.int64).tolist())
    offline_test_uq_set = set(np.asarray(offline_test_dataset.uq_idxs, dtype=np.int64).tolist())
    offline_overlap_uq = offline_train_uq_set & offline_test_uq_set

    if len(offline_overlap_uq) > 0:
        raise RuntimeError(
            f"Data leakage detected: offline train/test uq_idx overlap size = {len(offline_overlap_uq)}"
        )

    # Check train/test trial overlap.
    offline_train_trial_set = set(np.asarray(offline_train_dataset.trial_global_ids, dtype=np.int64).tolist())
    offline_test_trial_set = set(np.asarray(offline_test_dataset.trial_global_ids, dtype=np.int64).tolist())
    offline_overlap_trials = offline_train_trial_set & offline_test_trial_set

    if len(offline_overlap_trials) > 0:
        raise RuntimeError(
            f"Trial leakage detected: offline train/test trial overlap size = {len(offline_overlap_trials)}"
        )
    # Check train/test subject overlap.
    offline_train_subject_set = set(
        np.asarray(offline_train_dataset.subject_ids, dtype=np.int64).tolist()
    )
    offline_test_subject_set = set(
        np.asarray(offline_test_dataset.subject_ids, dtype=np.int64).tolist()
    )
    offline_overlap_subjects = offline_train_subject_set & offline_test_subject_set

    if uschad_split_mode == "subject" and len(offline_overlap_subjects) > 0:
        raise RuntimeError(
            f"Subject leakage detected: offline train/test subject overlap = "
            f"{sorted(offline_overlap_subjects)}"
        )

    # This optional split is used only for offline model selection. Keeping it
    # outside the online stream preserves the existing continual-data protocol.
    offline_val_dataset = None
    explicit_val_subjects = parse_subject_ids(
        getattr(args, "offline_val_subjects", "")
    )
    if explicit_val_subjects is not None:
        if uschad_split_mode != "subject":
            raise ValueError(
                "--offline_val_subjects requires --uschad_split_mode subject, "
                "because validation must be subject-disjoint from train and test."
            )

        all_subject_set = set(
            np.asarray(whole_test_set.subject_ids, dtype=np.int64).tolist()
        )
        val_subject_set = set(int(subject_id) for subject_id in explicit_val_subjects)
        unknown_val_subjects = val_subject_set - all_subject_set
        if len(unknown_val_subjects) > 0:
            raise ValueError(
                f"Unknown --offline_val_subjects: {sorted(unknown_val_subjects)}. "
                f"Available subjects: {sorted(all_subject_set)}"
            )

        overlap_with_train = val_subject_set & offline_train_subject_set
        overlap_with_test = val_subject_set & offline_test_subject_set
        if len(overlap_with_train) > 0 or len(overlap_with_test) > 0:
            raise RuntimeError(
                "Offline validation subject leakage detected: "
                f"val/train overlap={sorted(overlap_with_train)}, "
                f"val/test overlap={sorted(overlap_with_test)}."
            )

        offline_val_dataset = subsample_by_subjects(
            subsample_classes(deepcopy(whole_test_set), include_classes=train_classes),
            val_subject_set,
        )
        if offline_val_dataset is None or len(offline_val_dataset) == 0:
            raise RuntimeError(
                "offline_val_dataset is empty. Please choose validation subjects "
                "that contain all old classes."
            )

        val_classes = set(np.asarray(offline_val_dataset.targets, dtype=np.int64).tolist())
        missing_val_classes = set(int(class_id) for class_id in train_classes) - val_classes
        if len(missing_val_classes) > 0:
            raise RuntimeError(
                "offline_val_dataset is missing old classes "
                f"{sorted(missing_val_classes)} for validation subjects "
                f"{sorted(val_subject_set)}."
            )

        offline_val_uq_set = set(
            np.asarray(offline_val_dataset.uq_idxs, dtype=np.int64).tolist()
        )
        offline_val_trial_set = set(
            np.asarray(offline_val_dataset.trial_global_ids, dtype=np.int64).tolist()
        )
        for split_name, uq_set, trial_set in [
            ("train", offline_train_uq_set, offline_train_trial_set),
            ("test", offline_test_uq_set, offline_test_trial_set),
        ]:
            if len(offline_val_uq_set & uq_set) > 0:
                raise RuntimeError(
                    f"Offline validation {split_name} uq_idx leakage detected."
                )
            if len(offline_val_trial_set & trial_set) > 0:
                raise RuntimeError(
                    f"Offline validation {split_name} trial leakage detected."
                )

        print(
            "[USCHAD Offline Validation Check] "
            f"subjects={sorted(val_subject_set)}, samples={len(offline_val_dataset)}, "
            "subject/uq_idx/trial overlap with train/test=0"
        )

    online_old_dataset_unlabelled_list = []  # 创建列表，用于保存每个 online session 的 old unlabeled 数据集
    old_trial_pools = {}
    if sample_unit == "trial":
        required_old_trials = continual_session_num * online_old_seen_num
        for class_id, class_dataset in zip(
            list(train_classes), each_old_unlabeled_samples
        ):
            if class_dataset is None or len(class_dataset) == 0:
                raise RuntimeError(
                    f"Old class {int(class_id)} has no held-out trial candidates."
                )
            trial_pool = np.unique(
                np.asarray(class_dataset.trial_global_ids, dtype=np.int64)
            )
            if required_old_trials > len(trial_pool):
                raise RuntimeError(
                    f"Old class {int(class_id)} needs {required_old_trials} online "
                    f"train trials ({continual_session_num} sessions x "
                    f"{online_old_seen_num}), but only {len(trial_pool)} are available."
                )
            rng = np.random.default_rng(seed + 1000 + int(class_id))
            rng.shuffle(trial_pool)
            old_trial_pools[int(class_id)] = trial_pool

    for session_id in range(continual_session_num):  # 遍历每一个 continual session
        online_session_old_samples = []  # 创建列表，用于保存当前 session 中每个 old class 的无标签样本

        for class_id, class_dataset in zip(list(train_classes), each_old_unlabeled_samples):  # 遍历每一个 old class 的 unlabeled 数据集
            if class_dataset is None or len(class_dataset) == 0:  # 如果该 old class 没有 unlabeled 样本
                continue  # 跳过该类，避免报错

            if sample_unit == "trial":
                pool = old_trial_pools[int(class_id)]
                begin = session_id * online_old_seen_num
                end = begin + online_old_seen_num
                chosen_trial_ids = pool[begin:end]
                selected_dataset = subsample_by_trial_ids(
                    class_dataset, chosen_trial_ids
                )
            else:
                candidate_indices = np.arange(len(class_dataset), dtype=np.int64)
                chosen_indices = safe_random_choice(
                    candidate_indices,
                    sample_num=online_old_seen_num,
                    seed=seed + 1000 * (session_id + 1) + int(class_id),
                )
                selected_dataset = subsample_dataset(class_dataset, chosen_indices)
            online_session_old_samples.append(selected_dataset)

        online_session_old_dataset = subDataset_wholeDataset(online_session_old_samples)

        if online_session_old_dataset is None or len(online_session_old_dataset) == 0:
            raise RuntimeError(
                f"online_session_old_dataset is empty at session {session_id + 1}. "
                "Please check online_old_seen_num and old held-out samples."
            )

        online_old_dataset_unlabelled_list.append(online_session_old_dataset)

    old_class_set = set([int(class_id) for class_id in train_classes])  # 将 old class 列表转成集合
    all_class_set = set(np.asarray(whole_training_set.targets, dtype=np.int64).tolist())  # 获取整个 USC-HAD 数据集中出现过的所有类别
    novel_classes = sorted(list(all_class_set - old_class_set))  # 用全集减去 old classes 得到 novel classes
    novel_targets_shuffle = np.asarray(novel_classes, dtype=np.int64)  # 将 novel class 列表转成 numpy 数组

    if is_shuffle:  # 如果参数要求打乱新类顺序
        rng = np.random.default_rng(seed)  # 根据 seed 创建随机数生成器
        rng.shuffle(novel_targets_shuffle)  # 打乱 novel class 顺序

    novel_dataset_unlabelled = subsample_classes(deepcopy(whole_training_set), include_classes=novel_targets_shuffle.tolist())  # 从完整训练集中抽取 novel classes

    if novel_dataset_unlabelled is None:  # 检查 novel 数据集是否为空
        raise RuntimeError("No novel-class samples found.")  # 如果没有 novel 样本，说明类别划分有问题
    if uschad_split_mode == "subject":
        novel_dataset_unlabelled = subsample_by_subjects(
            novel_dataset_unlabelled,
            test_subjects_for_split,
        )

        if novel_dataset_unlabelled is None or len(novel_dataset_unlabelled) == 0:
            raise RuntimeError(
                "No novel-class samples found in test subjects. "
                "Please check --uschad_test_subjects or use another subject split."
            )
    total_novel_classes = int(len(novel_targets_shuffle))

    if total_novel_classes <= 0:
        raise RuntimeError(
            "No novel classes found after old/novel split. "
            "Please check --num_old_classes and dataset labels."
        )

    if total_novel_classes % targets_per_session != 0:
        raise RuntimeError(
            f"Invalid CGCD session config in USC-HAD: total novel classes "
            f"({total_novel_classes}) cannot be divided by "
            f"num_novel_class_per_session ({targets_per_session}). "
            f"Please choose a divisor of {total_novel_classes}."
        )

    expected_continual_session_num = total_novel_classes // targets_per_session

    if continual_session_num != expected_continual_session_num:
        raise RuntimeError(
            f"continual_session_num mismatch: config_dict gives {continual_session_num}, "
            f"but total_novel_classes ({total_novel_classes}) // "
            f"num_novel_class_per_session ({targets_per_session}) = "
            f"{expected_continual_session_num}. "
            f"Please make sure train_happy.py derives args.continual_session_num before get_datasets(...)."
        )

    print("=" * 80)
    print("[USCHAD CGCD SESSION CHECK]")
    print(f"total_novel_classes = {total_novel_classes}")
    print(f"num_novel_class_per_session = {targets_per_session}")
    print(f"derived continual_session_num = {continual_session_num}")
    print("=" * 80)

    novel_trial_pools = {}
    novel_trial_offsets = {}
    if sample_unit == "trial":
        for target_class in novel_targets_shuffle.tolist():
            class_dataset = subsample_classes(
                deepcopy(novel_dataset_unlabelled),
                include_classes=[int(target_class)],
            )
            if class_dataset is None or len(class_dataset) == 0:
                raise RuntimeError(
                    f"Novel class {int(target_class)} has no online trial candidates."
                )
            trial_pool = np.unique(
                np.asarray(class_dataset.trial_global_ids, dtype=np.int64)
            )
            rng = np.random.default_rng(seed + 2000 + int(target_class))
            rng.shuffle(trial_pool)
            novel_trial_pools[int(target_class)] = trial_pool
            novel_trial_offsets[int(target_class)] = 0

    online_novel_dataset_unlabelled_list = []  # 创建列表，用于保存每个 online session 的 novel unlabeled 数据集
    online_test_dataset_list = []  # 创建列表，用于保存每个 online session 的测试集

    for session_id in range(continual_session_num):  # 遍历每个 online session
        end_index = (session_id + 1) * targets_per_session  # 当前 session 后累计已出现 novel class 数量
        online_session_targets = novel_targets_shuffle[:end_index]  # 当前 session 测试时包含到目前为止所有已出现 novel classes
        online_session_each_novel_samples = [subsample_classes(deepcopy(novel_dataset_unlabelled), include_classes=[target]) for target in online_session_targets]  # 按 novel class 拆成多个单类数据集
        online_session_novel_samples = []  # 创建列表，用于保存当前 session 采样后的 novel 数据

        for class_position, class_dataset in enumerate(online_session_each_novel_samples):  # 遍历当前 session 涉及到的每个 novel class
            target_class = int(online_session_targets[class_position])  # 读取当前 novel class 的类别 id

            if class_dataset is None or len(class_dataset) == 0:  # 如果该 novel class 没有样本
                continue  # 跳过该类，避免报错

            if session_id >= 1 and class_position < session_id * targets_per_session:  # 判断该 novel class 是否属于以前 session 已经出现过的类
                sample_num = online_novel_seen_num  # 对已经出现过的新类，采样 seen 数量
            else:  # 如果该 novel class 是当前 session 第一次出现
                sample_num = online_novel_unseen_num  # 对新出现的新类，采样 unseen 数量

            if sample_unit == "trial":
                pool = novel_trial_pools[target_class]
                begin = novel_trial_offsets[target_class]
                end = begin + sample_num
                if end > len(pool):
                    raise RuntimeError(
                        f"Novel class {target_class} needs {sample_num} more trials at "
                        f"session {session_id + 1}, but only {len(pool) - begin} unused "
                        "trials remain. Reduce the trial-level online counts."
                    )
                chosen_trial_ids = pool[begin:end]
                novel_trial_offsets[target_class] = end
                selected_dataset = subsample_by_trial_ids(
                    class_dataset, chosen_trial_ids
                )
            else:
                candidate_indices = np.arange(len(class_dataset), dtype=np.int64)
                chosen_indices = safe_random_choice(
                    candidate_indices,
                    sample_num=sample_num,
                    seed=seed + 2000 * (session_id + 1) + target_class,
                )
                selected_dataset = subsample_dataset(class_dataset, chosen_indices)
            online_session_novel_samples.append(selected_dataset)

        online_session_novel_dataset = subDataset_wholeDataset(online_session_novel_samples)

        if online_session_novel_dataset is None or len(online_session_novel_dataset) == 0:
            raise RuntimeError(
                f"online_session_novel_dataset is empty at session {session_id + 1}. "
                "Please check novel classes, test subjects, and online_novel_unseen_num."
            )

        online_novel_dataset_unlabelled_list.append(online_session_novel_dataset)

        online_session_test_dataset = subsample_classes(
            deepcopy(whole_test_set),
            include_classes=list(train_classes) + online_session_targets.tolist(),
        )  # 构造当前 session 的 old + seen novel 测试集

        if uschad_split_mode == "subject":
            online_session_test_dataset = subsample_by_subjects(
                online_session_test_dataset,
                test_subjects_for_split,
            )

            if online_session_test_dataset is None or len(online_session_test_dataset) == 0:
                raise RuntimeError(
                    f"online_session_test_dataset is empty after subject filtering "
                    f"at session {session_id + 1}."
                )

        # ---------------------------------------------------------------------
        # Prevent online train/test leakage.
        #
        # Online train samples are drawn from held-out/test subjects.
        # Online test is also drawn from held-out/test subjects.
        #
        # Therefore, for session k, test set must exclude all samples/trials that
        # have been used by online train in sessions <= k.
        # ---------------------------------------------------------------------
        cumulative_online_train_uq_idxs = []
        cumulative_online_train_trial_ids = []

        for previous_session_id in range(session_id + 1):
            previous_old_dataset = online_old_dataset_unlabelled_list[previous_session_id]
            previous_novel_dataset = online_novel_dataset_unlabelled_list[previous_session_id]

            for previous_train_dataset in [previous_old_dataset, previous_novel_dataset]:
                if previous_train_dataset is None or len(previous_train_dataset) == 0:
                    continue

                cumulative_online_train_uq_idxs.append(
                    np.asarray(previous_train_dataset.uq_idxs, dtype=np.int64)
                )
                cumulative_online_train_trial_ids.append(
                    np.asarray(previous_train_dataset.trial_global_ids, dtype=np.int64)
                )

        if len(cumulative_online_train_uq_idxs) > 0:
            cumulative_online_train_uq_idxs = np.concatenate(
                cumulative_online_train_uq_idxs,
                axis=0,
            )
        else:
            cumulative_online_train_uq_idxs = np.asarray([], dtype=np.int64)

        if len(cumulative_online_train_trial_ids) > 0:
            cumulative_online_train_trial_ids = np.concatenate(
                cumulative_online_train_trial_ids,
                axis=0,
            )
        else:
            cumulative_online_train_trial_ids = np.asarray([], dtype=np.int64)

        online_session_test_dataset = subsample_exclude_uq_or_trial(
            online_session_test_dataset,
            exclude_uq_idxs=cumulative_online_train_uq_idxs,
            exclude_trial_ids=cumulative_online_train_trial_ids,
        )

        if online_session_test_dataset is None or len(online_session_test_dataset) == 0:
            sampling_options = (
                "--online_old_seen_trials / --online_novel_unseen_trials / "
                "--online_novel_seen_trials"
                if sample_unit == "trial"
                else "--online_old_seen_num / --online_novel_unseen_num / "
                "--online_novel_seen_num"
            )
            raise RuntimeError(
                f"online_session_test_dataset is empty after excluding online train "
                f"uq_idx/trial leakage at session {session_id + 1}. "
                f"This means online train has consumed all candidate test trials/windows. "
                f"Please reduce {sampling_options}, "
                f"or use a larger test subject split."
            )

        expected_test_classes = set(int(class_id) for class_id in train_classes)
        expected_test_classes.update(
            int(class_id) for class_id in online_session_targets.tolist()
        )
        observed_test_classes = set(
            np.asarray(online_session_test_dataset.targets, dtype=np.int64).tolist()
        )
        missing_test_classes = sorted(expected_test_classes - observed_test_classes)
        if missing_test_classes:
            sampling_options = (
                "--online_old_seen_trials / --online_novel_unseen_trials / "
                "--online_novel_seen_trials"
                if sample_unit == "trial"
                else "--online_old_seen_num / --online_novel_unseen_num / "
                "--online_novel_seen_num"
            )
            raise RuntimeError(
                f"Online session-{session_id + 1} test set lost expected classes "
                f"{missing_test_classes} after excluding cumulative online train data. "
                f"Reduce {sampling_options} or reserve a larger disjoint test pool."
            )

        remaining_test_units_by_class = {}
        test_targets = np.asarray(online_session_test_dataset.targets, dtype=np.int64)
        for class_id in sorted(expected_test_classes):
            class_mask = test_targets == class_id
            if sample_unit == "trial":
                class_units = len(
                    np.unique(
                        np.asarray(
                            online_session_test_dataset.trial_global_ids,
                            dtype=np.int64,
                        )[class_mask]
                    )
                )
            else:
                class_units = int(class_mask.sum())
            remaining_test_units_by_class[int(class_id)] = int(class_units)

        online_train_uq_set = set(
            [int(x) for x in np.asarray(cumulative_online_train_uq_idxs, dtype=np.int64).tolist()]
        )
        online_test_uq_set = set(
            [int(x) for x in np.asarray(online_session_test_dataset.uq_idxs, dtype=np.int64).tolist()]
        )
        online_overlap_uq = online_train_uq_set & online_test_uq_set

        if len(online_overlap_uq) > 0:
            raise RuntimeError(
                f"Online exact-window leakage detected at session {session_id + 1}: "
                f"train/test uq_idx overlap size = {len(online_overlap_uq)}"
            )

        online_train_trial_set = set(
            [int(x) for x in np.asarray(cumulative_online_train_trial_ids, dtype=np.int64).tolist()]
        )
        online_test_trial_set = set(
            [int(x) for x in np.asarray(online_session_test_dataset.trial_global_ids, dtype=np.int64).tolist()]
        )
        online_overlap_trials = online_train_trial_set & online_test_trial_set

        if len(online_overlap_trials) > 0:
            raise RuntimeError(
                f"Online trial leakage detected at session {session_id + 1}: "
                f"train/test trial overlap size = {len(online_overlap_trials)}"
            )

        print(
            f"[USCHAD Online Leakage Check] session {session_id + 1}: "
            f"excluded_train_uq={len(online_train_uq_set)}, "
            f"excluded_train_trials={len(online_train_trial_set)}, "
            f"test_after_exclusion={len(online_session_test_dataset)}, "
            f"uq_overlap=0, trial_overlap=0, "
            f"test_{sample_unit}s_per_class={remaining_test_units_by_class}"
        )

        online_test_dataset_list.append(online_session_test_dataset)  # 保存当前 session 的测试集

    if sample_unit == "trial":
        offline_train_window_count = len(offline_train_dataset)
        offline_test_window_count = len(offline_test_dataset)
        offline_val_window_count = (
            len(offline_val_dataset) if offline_val_dataset is not None else 0
        )
        offline_train_dataset = convert_window_dataset_to_trials(
            offline_train_dataset, har_trial_train_transform
        )
        offline_test_dataset = convert_window_dataset_to_trials(
            offline_test_dataset, None
        )
        offline_val_dataset = convert_window_dataset_to_trials(
            offline_val_dataset, None
        )
        online_old_dataset_unlabelled_list = [
            convert_window_dataset_to_trials(dataset, har_trial_train_transform)
            for dataset in online_old_dataset_unlabelled_list
        ]
        online_novel_dataset_unlabelled_list = [
            convert_window_dataset_to_trials(dataset, har_trial_train_transform)
            for dataset in online_novel_dataset_unlabelled_list
        ]
        online_test_dataset_list = [
            convert_window_dataset_to_trials(dataset, None)
            for dataset in online_test_dataset_list
        ]
        print(
            "[USCHAD Trial Conversion] "
            f"offline train {offline_train_window_count} windows -> "
            f"{len(offline_train_dataset)} trials; validation "
            f"{offline_val_window_count} windows -> "
            f"{len(offline_val_dataset) if offline_val_dataset is not None else 0} trials; "
            f"test {offline_test_window_count} windows -> "
            f"{len(offline_test_dataset)} trials."
        )

    all_datasets = {  # 构造 HHR offline/online 共用的数据集字典
        "offline_train_dataset": offline_train_dataset,  # 保存 offline 阶段 old labeled 训练集
        "offline_test_dataset": offline_test_dataset,  # 保存 offline 阶段 old test 测试集
        "online_old_dataset_unlabelled_list": online_old_dataset_unlabelled_list,  # 保存每个 online session 的 old unlabeled 数据
        "online_novel_dataset_unlabelled_list": online_novel_dataset_unlabelled_list,  # 保存每个 online session 的 novel unlabeled 数据
        "online_test_dataset_list": online_test_dataset_list,  # 保存每个 online session 的测试集
    }  # 数据集字典结束

    print("=" * 80)  # 打印分隔线，方便在终端中定位 USC-HAD 数据加载日志
    print("[USCHAD Dataset Loaded]")  # 打印 USC-HAD 数据加载完成标记
    print(f"npz_path: {npz_path}")  # 打印使用的 npz 文件路径
    print(f"window shape: {whole_training_set.data.shape}  # [N, C, T]")  # 打印窗口数据形状
    print(f"expected channels/window: C={expected_channels}, T={expected_window_size}")
    print(f"sample unit: {sample_unit}")
    print(f"train_classes(old): {list(train_classes)}")  # 打印 old class 列表
    print(f"novel_targets_shuffle: {novel_targets_shuffle.tolist()}")  # 打印 novel class 顺序
    print(f"offline_train_dataset: {len(offline_train_dataset) if offline_train_dataset is not None else 0}")  # 打印 offline 训练集大小
    print(f"offline_test_dataset: {len(offline_test_dataset) if offline_test_dataset is not None else 0}")  # 打印 offline 测试集大小
    if uschad_split_mode == "subject":
        all_datasets["offline_val_dataset"] = offline_val_dataset
        offline_train_subjects = sorted(
            set(np.asarray(offline_train_dataset.subject_ids, dtype=np.int64).tolist())
        )
        offline_val_subjects = sorted(
            set(np.asarray(offline_val_dataset.subject_ids, dtype=np.int64).tolist())
        ) if offline_val_dataset is not None else []
        offline_test_subjects = sorted(
            set(np.asarray(offline_test_dataset.subject_ids, dtype=np.int64).tolist())
        )
        print(f"offline train subjects: {offline_train_subjects}")
        print(f"offline validation dataset: {len(offline_val_dataset) if offline_val_dataset is not None else 0}")
        print(f"offline validation subjects: {offline_val_subjects}")
        print(f"offline test subjects: {offline_test_subjects}")
        print(f"offline subject overlap: {sorted(set(offline_train_subjects) & set(offline_test_subjects))}")
    for session_id in range(continual_session_num):  # 遍历每个 session，打印数据量
        old_len = len(online_old_dataset_unlabelled_list[session_id]) if online_old_dataset_unlabelled_list[session_id] is not None else 0  # 当前 session old unlabeled 数量
        novel_len = len(online_novel_dataset_unlabelled_list[session_id]) if online_novel_dataset_unlabelled_list[session_id] is not None else 0  # 当前 session novel unlabeled 数量
        test_len = len(online_test_dataset_list[session_id]) if online_test_dataset_list[session_id] is not None else 0  # 当前 session 测试样本数量
        print(f"session {session_id + 1}: old_unlabeled={old_len}, novel_unlabeled={novel_len}, test={test_len}")  # 打印当前 session 数据统计

    print("=" * 80)  # 打印结束分隔线

    return all_datasets, novel_targets_shuffle  # 返回数据集字典和 novel class 顺序
