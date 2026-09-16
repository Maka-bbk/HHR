"""Registered USC-HAD split and label-isolated three-session CGCD stream."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np


REGISTERED_TEST_SUBJECTS = (
    (10, 11),
    (2, 13),
    (3, 9),
    (1, 7),
    (8, 12),
    (4, 5),
    (6, 14),
)
REGISTERED_VALIDATION_SUBJECTS = (
    (2, 13),
    (3, 9),
    (1, 7),
    (8, 12),
    (4, 5),
    (6, 14),
    (10, 11),
)
OLD_CLASSES = tuple(range(6))
TOTAL_CLASSES = 12
EXPECTED_CHANNEL_NAMES = (
    "acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z",
)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _ids_sha256(values: Sequence[int]) -> str:
    return hashlib.sha256(np.asarray(values, dtype="<i8").tobytes()).hexdigest()


@dataclass(frozen=True)
class SubjectSplit:
    fold: int
    train: tuple[int, ...]
    validation: tuple[int, ...]
    outer_test: tuple[int, ...]

    def as_dict(self) -> dict:
        return {
            "fold": int(self.fold),
            "train_subjects": list(self.train),
            "validation_subjects": list(self.validation),
            "outer_test_subjects": list(self.outer_test),
        }


def registered_subject_split(fold: int) -> SubjectSplit:
    fold = int(fold)
    if not 1 <= fold <= 7:
        raise ValueError("USC-HAD fold must lie in 1..7.")
    test = tuple(sorted(REGISTERED_TEST_SUBJECTS[fold - 1]))
    validation = tuple(sorted(REGISTERED_VALIDATION_SUBJECTS[fold - 1]))
    excluded = set(test) | set(validation)
    train = tuple(subject for subject in range(1, 15) if subject not in excluded)
    groups = (set(train), set(validation), set(test))
    if any(groups[left] & groups[right] for left, right in ((0, 1), (0, 2), (1, 2))):
        raise RuntimeError("Registered train/validation/test subjects overlap.")
    if set().union(*groups) != set(range(1, 15)) or tuple(map(len, groups)) != (10, 2, 2):
        raise RuntimeError("Registered fold is not a complete 10/2/2 partition.")
    return SubjectSplit(fold, train, validation, test)


@dataclass(frozen=True)
class SensorTrial:
    """Learner-facing trial; activity identity is deliberately absent."""

    trial_id: int
    subject_id: int
    windows: np.ndarray
    raw_windows: np.ndarray
    window_starts: np.ndarray

    def validate(self, window_size: int) -> "SensorTrial":
        windows = np.asarray(self.windows, dtype=np.float32)
        raw = np.asarray(self.raw_windows, dtype=np.float32)
        starts = np.asarray(self.window_starts, dtype=np.int64)
        if windows.ndim != 3 or windows.shape[1:] != (6, int(window_size)):
            raise ValueError(f"SensorTrial windows must be [L,6,{window_size}].")
        if raw.shape != windows.shape or starts.shape != (len(windows),):
            raise ValueError("SensorTrial raw windows/starts differ from normalized windows.")
        if not len(starts) or starts[0] != 0 or np.any(np.diff(starts) <= 0):
            raise ValueError("SensorTrial windows are incomplete or out of order.")
        if not np.all(np.isfinite(windows)) or not np.all(np.isfinite(raw)):
            raise ValueError("SensorTrial contains non-finite sensor values.")
        return self


@dataclass(frozen=True)
class SessionStream:
    session: int
    incoming: tuple[SensorTrial, ...]
    evaluation: tuple[SensorTrial, ...]
    maximum_new_classes: int
    incoming_trial_ids_sha256: str
    evaluation_trial_ids_sha256: str


class TruthStore:
    """Scorer-only identities kept outside every learner-facing method."""

    def __init__(
        self,
        labels: Mapping[int, int],
        names: Mapping[int, str],
        subjects: Mapping[int, int],
    ) -> None:
        self.__labels = {int(key): int(value) for key, value in labels.items()}
        self.__names = {int(key): str(value) for key, value in names.items()}
        self.__subjects = {int(key): int(value) for key, value in subjects.items()}

    def join(self, trial_ids: Sequence[int]) -> tuple[np.ndarray, list[str], np.ndarray]:
        ids = [int(value) for value in trial_ids]
        missing = [value for value in ids if value not in self.__labels]
        if missing:
            raise KeyError(f"TruthStore lacks trial ids {missing[:5]}.")
        return (
            np.asarray([self.__labels[value] for value in ids], dtype=np.int64),
            [self.__names[value] for value in ids],
            np.asarray([self.__subjects[value] for value in ids], dtype=np.int64),
        )


@dataclass(frozen=True)
class RegisteredProtocol:
    npz_path: str
    npz_sha256: str
    fold: int
    seed: int
    window_size: int
    stride: int
    split: SubjectSplit
    fold_mean: np.ndarray
    fold_std: np.ndarray
    offline_train: tuple[SensorTrial, ...]
    offline_validation: tuple[SensorTrial, ...]
    offline_outer_test: tuple[SensorTrial, ...]
    sessions: tuple[SessionStream, ...]
    novel_class_order_for_scoring: tuple[int, ...]
    truth: TruthStore


def _load_grid(npz_path: Path, window_size: int, stride: int) -> dict[str, np.ndarray]:
    with np.load(npz_path, allow_pickle=True) as archive:
        required = (
            "windows", "labels", "subject_ids", "trial_global_ids",
            "window_indices", "window_start_indices", "mean", "std", "activity_names",
            "channel_names",
        )
        missing = [name for name in required if name not in archive]
        if missing:
            raise RuntimeError(f"USC-HAD NPZ lacks {missing}.")
        grid = {name: np.asarray(archive[name]) for name in required}
    windows = np.asarray(grid["windows"], dtype=np.float32)
    if windows.ndim != 3 or windows.shape[1:] != (6, int(window_size)):
        raise RuntimeError(
            f"Expected USC-HAD windows [N,6,{window_size}], got {windows.shape}."
        )
    observed_channels = tuple(
        str(value) for value in np.asarray(grid["channel_names"]).reshape(-1).tolist()
    )
    if observed_channels != EXPECTED_CHANNEL_NAMES:
        raise RuntimeError(
            "USC-HAD channel order is unsafe for gravity alignment: "
            f"observed={observed_channels}, expected={EXPECTED_CHANNEL_NAMES}."
        )
    starts = np.asarray(grid["window_start_indices"], dtype=np.int64)
    trial_ids = np.asarray(grid["trial_global_ids"], dtype=np.int64)
    observed_strides = []
    for trial_id in np.unique(trial_ids):
        local = np.sort(starts[trial_ids == trial_id])
        if len(local) > 1:
            observed_strides.extend(np.diff(local).tolist())
    if not observed_strides or set(observed_strides) != {int(stride)}:
        raise RuntimeError(
            f"NPZ stride differs from registered stride {stride}: {sorted(set(observed_strides))}."
        )
    labels = np.asarray(grid["labels"], dtype=np.int64)
    subjects = np.asarray(grid["subject_ids"], dtype=np.int64)
    if set(np.unique(labels).tolist()) != set(range(TOTAL_CLASSES)):
        raise RuntimeError("USC-HAD physical labels must be 0..11.")
    if set(np.unique(subjects).tolist()) != set(range(1, 15)):
        raise RuntimeError("USC-HAD subjects must be 1..14.")
    return grid


def _trial_rows(grid: Mapping[str, np.ndarray], trial_id: int) -> np.ndarray:
    rows = np.flatnonzero(np.asarray(grid["trial_global_ids"], dtype=np.int64) == int(trial_id))
    order = np.argsort(np.asarray(grid["window_start_indices"])[rows], kind="stable")
    rows = rows[order]
    if not len(rows):
        raise KeyError(f"Unknown trial {trial_id}.")
    starts = np.asarray(grid["window_start_indices"])[rows]
    indices = np.asarray(grid["window_indices"])[rows]
    if starts[0] != 0 or np.any(np.diff(starts) <= 0) or not np.array_equal(indices, np.arange(len(rows))):
        raise RuntimeError(f"Trial {trial_id} does not contain its complete ordered window grid.")
    return rows


def _trial_identity(grid: Mapping[str, np.ndarray], trial_id: int) -> tuple[int, int, str]:
    rows = _trial_rows(grid, trial_id)
    labels = np.unique(np.asarray(grid["labels"])[rows])
    subjects = np.unique(np.asarray(grid["subject_ids"])[rows])
    names = np.unique(np.asarray(grid["activity_names"], dtype=object)[rows])
    if len(labels) != 1 or len(subjects) != 1 or len(names) != 1:
        raise RuntimeError(f"Trial {trial_id} has inconsistent identity metadata.")
    return int(labels[0]), int(subjects[0]), str(names[0])


def build_registered_protocol(
    npz_path: str | Path,
    *,
    fold: int,
    seed: int,
    window_size: int = 256,
    stride: int = 128,
    shuffle_novel_classes: bool = False,
    online_old_trials: int = 2,
    online_novel_unseen_trials: int = 5,
    online_novel_seen_trials: int = 2,
) -> RegisteredProtocol:
    """Build the exact 300/60 offline split and three label-free online streams.

    Activity labels are used here only to construct the registered benchmark.
    The returned :class:`SensorTrial` and :class:`SessionStream` schemas contain
    no activity field; identities are available solely through ``truth.join``
    after predictions have been frozen.
    """

    path = Path(npz_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    if (int(window_size), int(stride)) not in ((256, 128), (128, 64), (64, 32)):
        raise ValueError("Registered grids are w256/s128, w128/s64, and w64/s32.")
    if (int(online_old_trials), int(online_novel_unseen_trials), int(online_novel_seen_trials)) != (2, 5, 2):
        raise ValueError("The formal USC-HAD stream is pinned to old=2, first-new=5, seen-new=2 trials.")
    split = registered_subject_split(int(fold))
    grid = _load_grid(path, int(window_size), int(stride))
    stored_windows = np.asarray(grid["windows"], dtype=np.float32)
    stored_mean = np.asarray(grid["mean"], dtype=np.float32).reshape(1, 6, 1)
    stored_std = np.asarray(grid["std"], dtype=np.float32).reshape(1, 6, 1)
    if np.any(stored_std <= 0):
        raise RuntimeError("NPZ normalization standard deviation is invalid.")
    raw_windows = stored_windows * stored_std + stored_mean
    labels = np.asarray(grid["labels"], dtype=np.int64)
    subjects = np.asarray(grid["subject_ids"], dtype=np.int64)
    fit_mask = np.isin(subjects, split.train) & np.isin(labels, OLD_CLASSES)
    fold_mean = raw_windows[fit_mask].mean(axis=(0, 2), keepdims=True).astype(np.float32)
    fold_std = np.maximum(
        raw_windows[fit_mask].std(axis=(0, 2), keepdims=True), np.float32(1e-6)
    ).astype(np.float32)
    normalized = ((raw_windows - fold_mean) / fold_std).astype(np.float32)

    all_trial_ids = np.unique(np.asarray(grid["trial_global_ids"], dtype=np.int64))
    truth_labels: dict[int, int] = {}
    truth_subjects: dict[int, int] = {}
    truth_names: dict[int, str] = {}
    records: dict[int, SensorTrial] = {}
    trials_by_subject_class: dict[tuple[int, int], list[int]] = {}
    for trial_id_value in all_trial_ids:
        trial_id = int(trial_id_value)
        rows = _trial_rows(grid, trial_id)
        label, subject, name = _trial_identity(grid, trial_id)
        truth_labels[trial_id] = label
        truth_subjects[trial_id] = subject
        truth_names[trial_id] = name
        record = SensorTrial(
            trial_id=trial_id,
            subject_id=subject,
            windows=normalized[rows].copy(),
            raw_windows=raw_windows[rows].copy(),
            window_starts=np.asarray(grid["window_start_indices"])[rows].astype(np.int64, copy=True),
        ).validate(int(window_size))
        records[trial_id] = record
        trials_by_subject_class.setdefault((subject, label), []).append(trial_id)

    def select(subject_set: Sequence[int], class_set: Sequence[int]) -> tuple[SensorTrial, ...]:
        ids = [
            trial_id for trial_id in all_trial_ids.astype(int).tolist()
            if truth_subjects[trial_id] in set(subject_set) and truth_labels[trial_id] in set(class_set)
        ]
        return tuple(records[value] for value in sorted(ids))

    offline_train = select(split.train, OLD_CLASSES)
    offline_validation = select(split.validation, OLD_CLASSES)
    offline_outer_test = select(split.outer_test, OLD_CLASSES)
    if (len(offline_train), len(offline_validation), len(offline_outer_test)) != (300, 60, 60):
        raise RuntimeError(
            "Registered USC-HAD offline trial counts changed: "
            f"{len(offline_train)}/{len(offline_validation)}/{len(offline_outer_test)}."
        )

    outer = set(split.outer_test)
    old_pools: dict[int, np.ndarray] = {}
    for class_id in OLD_CLASSES:
        pool = np.asarray(sorted(
            trial_id for trial_id in records
            if truth_subjects[trial_id] in outer and truth_labels[trial_id] == class_id
        ), dtype=np.int64)
        rng = np.random.default_rng(int(seed) + 1000 + int(class_id))
        rng.shuffle(pool)
        old_pools[class_id] = pool

    novel_order = np.arange(6, TOTAL_CLASSES, dtype=np.int64)
    if bool(shuffle_novel_classes):
        rng = np.random.default_rng(int(seed))
        rng.shuffle(novel_order)
    novel_pools: dict[int, np.ndarray] = {}
    novel_offsets: dict[int, int] = {}
    for class_id in novel_order.tolist():
        pool = np.asarray(sorted(
            trial_id for trial_id in records
            if truth_subjects[trial_id] in outer and truth_labels[trial_id] == int(class_id)
        ), dtype=np.int64)
        rng = np.random.default_rng(int(seed) + 2000 + int(class_id))
        rng.shuffle(pool)
        novel_pools[int(class_id)] = pool
        novel_offsets[int(class_id)] = 0

    cumulative_train: set[int] = set()
    sessions: list[SessionStream] = []
    for session_index in range(3):
        incoming_ids: list[int] = []
        for class_id in OLD_CLASSES:
            begin = session_index * int(online_old_trials)
            end = begin + int(online_old_trials)
            incoming_ids.extend(old_pools[class_id][begin:end].astype(int).tolist())
        seen_novel = novel_order[: (session_index + 1) * 2].astype(int).tolist()
        for position, class_id in enumerate(seen_novel):
            count = (
                int(online_novel_seen_trials)
                if session_index >= 1 and position < session_index * 2
                else int(online_novel_unseen_trials)
            )
            begin = novel_offsets[class_id]
            end = begin + count
            if end > len(novel_pools[class_id]):
                raise RuntimeError("Formal online sampling exhausted a novel-class trial pool.")
            incoming_ids.extend(novel_pools[class_id][begin:end].astype(int).tolist())
            novel_offsets[class_id] = end
        if len(incoming_ids) != len(set(incoming_ids)) or set(incoming_ids) & cumulative_train:
            raise RuntimeError("Online sessions reused a training trial.")
        cumulative_train.update(incoming_ids)
        visible_classes = set(OLD_CLASSES) | set(seen_novel)
        evaluation_ids = sorted(
            trial_id for trial_id in records
            if truth_subjects[trial_id] in outer
            and truth_labels[trial_id] in visible_classes
            and trial_id not in cumulative_train
        )
        if set(evaluation_ids) & cumulative_train:
            raise RuntimeError("Online train/evaluation trial leakage survived exclusion.")
        observed_classes = {truth_labels[value] for value in evaluation_ids}
        if observed_classes != visible_classes:
            raise RuntimeError(
                f"Session {session_index + 1} evaluation lost classes "
                f"{sorted(visible_classes - observed_classes)}."
            )
        sessions.append(SessionStream(
            session=session_index + 1,
            incoming=tuple(records[value] for value in sorted(incoming_ids)),
            evaluation=tuple(records[value] for value in evaluation_ids),
            maximum_new_classes=2,
            incoming_trial_ids_sha256=_ids_sha256(sorted(incoming_ids)),
            evaluation_trial_ids_sha256=_ids_sha256(evaluation_ids),
        ))

    return RegisteredProtocol(
        npz_path=str(path),
        npz_sha256=sha256_file(path),
        fold=int(fold),
        seed=int(seed),
        window_size=int(window_size),
        stride=int(stride),
        split=split,
        fold_mean=fold_mean.copy(),
        fold_std=fold_std.copy(),
        offline_train=offline_train,
        offline_validation=offline_validation,
        offline_outer_test=offline_outer_test,
        sessions=tuple(sessions),
        novel_class_order_for_scoring=tuple(int(value) for value in novel_order.tolist()),
        truth=TruthStore(truth_labels, truth_names, truth_subjects),
    )


__all__ = [
    "EXPECTED_CHANNEL_NAMES",
    "OLD_CLASSES",
    "REGISTERED_TEST_SUBJECTS",
    "REGISTERED_VALIDATION_SUBJECTS",
    "RegisteredProtocol",
    "SensorTrial",
    "SessionStream",
    "SubjectSplit",
    "TOTAL_CLASSES",
    "TruthStore",
    "build_registered_protocol",
    "registered_subject_split",
    "sha256_file",
]
