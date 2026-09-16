"""Run the fold06 Session-2 unlabelled secondary-codebook experiment."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.uschad import get_uschad_datasets  # noqa: E402
from experiments.motion_primitive.core import run_length_encode  # noqa: E402
from experiments.motion_primitive.online_secondary_codebook import (  # noqa: E402
    ARMS,
    SPLIT_ARMS,
    UnlabelledTokenOccurrence,
    assign_secondary_codebook,
    class_mean_histogram_distances,
    duration_histograms,
    duration_weighted_residual,
    fit_secondary_codebook,
    global_hungarian_metrics,
    gravity_from_token_partitions,
    monte_carlo_upper_tail_summary,
    posthoc_secondary_metrics,
    secondary_fit_diagnostics,
    select_dominant_token,
    shuffle_gravity_within_subject_split,
    stratified_paired_accuracy_bootstrap,
    validate_session_manifest,
)
from experiments.motion_primitive.run_sit_stand_probe import (  # noqa: E402
    _validate_run,
)
from experiments.motion_primitive.trajectory_ablation import (  # noqa: E402
    SourceSignalRepository,
    jsonable,
    sha256_file,
    stable_array_hash,
    stable_text_hash,
)


REGISTERED_NOVEL_ORDER = tuple(range(6, 12))
REGISTERED_SESSION2_ACTIVITY_IDS = tuple(range(10))
REGISTERED_SESSION_COUNTS = {
    "session_1_incremental_train_count": 22,
    "session_2_incremental_train_count": 26,
    "session_2_cumulative_train_count": 48,
    "session_2_test_count": 52,
}


PROTOCOL = {
    "name": "fold06_session2_unlabelled_secondary_codebook_v2",
    "scope": "isolated_session2_unlabelled_trajectory_clustering_proxy",
    "is_formal_happy_cgcd": False,
    "session_sampling": {
        "source": "data.uschad.get_uschad_datasets",
        "old_classes": list(range(6)),
        "novel_classes_per_session": 2,
        "continual_sessions": 3,
        "online_old_seen_trials": 2,
        "online_novel_unseen_trials": 5,
        "online_novel_seen_trials": 2,
        "target_session": 2,
    },
    "candidate_rule": (
        "Among cumulative Session1+2 train trials, select the token that covers "
        ">=50% non-overlapping partition duration in the greatest number of trials; "
        "require support>=10 and resolve ties by smaller token id."
    ),
    "secondary_clusterer": (
        "K=2 deterministic alternating k-medoids, 50 unlabelled restarts"
    ),
    "arms": {
        "U0_coarse": "no secondary split",
        "U1_residual": "duration-weighted codebook quantisation residual",
        "U2_gravity": "robust raw-acceleration gravity direction",
        "U3_joint": "equal residual/angular-distance fusion",
    },
    "trial_readout": (
        "duration-normalized refined histogram -> train-only KMeans K=10 -> "
        "frozen Session-2 test predictions"
    ),
    "evaluation": (
        "one global Hungarian alignment for All/Old/New/F1/recall; separate "
        "post-hoc binary alignment only for the secondary Sit/Stand diagnostic"
    ),
    "confirmatory_statistics": {
        "gravity_shuffle_negative_control": (
            "Shuffle the selected-token gravity descriptor only within each "
            "subject x split-role group; refit the U2/U3 secondary codebook and "
            "train-only KMeans for every permutation; freeze both test cluster "
            "predictions and selected-token child assignments before joining labels; "
            "evaluate global metrics with one complete-test-set Hungarian alignment "
            "per replicate and Sit/Stand balanced accuracy with one binary alignment "
            "per replicate. Also report U2-minus-fixed-U0, whose effect and upper-tail "
            "p value are algebraically identical to the absolute-U2 shuffle test."
        ),
        "paired_trial_bootstrap": (
            "Freeze each arm's complete-test-set Hungarian mapping first, then "
            "paired-bootstrap trial correctness within subject x activity strata "
            "for U1/U2/U3 minus U0 All/Old/New accuracy."
        ),
    },
    "statistical_status": {
        "multiplicity_adjustment": "none",
        "gravity_global_metrics": (
            "unadjusted diagnostic endpoints; report effect, null quantiles, and "
            "Monte Carlo p without family-wise confirmatory claims"
        ),
        "sit_stand_endpoint": (
            "exploratory post-hoc endpoint because the Sit/Stand pair was selected "
            "after inspecting this fold; independent held-out confirmation or a "
            "pre-specified pair-selection/maxT procedure is still required"
        ),
        "u2_minus_u0_shuffle": (
            "algebraic re-expression of the absolute-U2 shuffle test under fixed U0; "
            "not a test that U2 outperforms U0"
        ),
    },
    "attribution_caveats": [
        (
            "U2 uses raw acceleration gravity rather than an encoder feature; "
            "a U2 gain is not direct evidence that A3 encoder training helped."
        ),
        (
            "This isolated Session-2 clustering proxy is not the formal "
            "Happy-CGCD training/evaluation pipeline."
        ),
        (
            "Rejecting the U2 gravity-shuffle null shows that the trial-gravity "
            "association carries classification information; it still does not "
            "attribute that information to the learned encoder."
        ),
        (
            "Bootstrap intervals are conditional trial-level stability intervals "
            "for held-out Subjects 4 and 5, not new-subject generalisation intervals."
        ),
        (
            "Gravity-shuffle p values are unadjusted; the Sit/Stand endpoint is "
            "post-hoc and exploratory on this fold."
        ),
    ],
    "deferred": [
        "adaptive split/no-split model selection",
    ],
}


@dataclass(frozen=True)
class SegmentArtifacts:
    split_role: np.ndarray
    trial_ids: np.ndarray
    starts: np.ndarray
    ends: np.ndarray
    tokens: np.ndarray
    embeddings: np.ndarray
    centers: np.ndarray


class LabelFreeSourceSignalRepository:
    """Reconstruct raw trials without consulting activity metadata.

    This reader opens the NPZ with ``allow_pickle=False`` and loads only the
    numeric signal/grid fields needed for inverse-normalisation overlap-add.
    It deliberately has no label, activity-name, subject, file-path, or
    ``metadata`` attribute.  The ordinary metadata-aware repository is created
    only after secondary-codebook and KMeans predictions are frozen.
    """

    def __init__(self, npz_path: Path):
        self.npz_path = Path(npz_path).resolve()
        required = {
            "windows",
            "trial_global_ids",
            "window_start_indices",
            "mean",
            "std",
        }
        with np.load(self.npz_path, allow_pickle=False) as data:
            missing = required - set(data.files)
            if missing:
                raise RuntimeError(
                    f"Label-free source NPZ is missing fields: {sorted(missing)}"
                )
            self.windows = np.asarray(data["windows"], dtype=np.float32)
            self.trial_ids = np.asarray(
                data["trial_global_ids"], dtype=np.int64
            )
            self.window_starts = np.asarray(
                data["window_start_indices"], dtype=np.int64
            )
            self.stored_mean = np.asarray(data["mean"], dtype=np.float32)
            self.stored_std = np.asarray(data["std"], dtype=np.float32)
        if self.windows.ndim != 3 or self.windows.shape[1] < 6:
            raise RuntimeError(
                f"Expected source windows [N,>=6,T], got {self.windows.shape}."
            )
        if self.trial_ids.shape != (len(self.windows),) or self.window_starts.shape != (
            len(self.windows),
        ):
            raise RuntimeError("Label-free source grid fields have invalid shapes.")
        expected_stats = (1, self.windows.shape[1], 1)
        if self.stored_mean.shape != expected_stats or self.stored_std.shape != expected_stats:
            raise RuntimeError(
                "Label-free source normalization shapes disagree with windows: "
                f"{self.stored_mean.shape}, {self.stored_std.shape}, "
                f"expected {expected_stats}."
            )
        if (
            np.any(self.window_starts < 0)
            or not np.all(np.isfinite(self.windows))
            or not np.all(np.isfinite(self.stored_mean))
            or not np.all(np.isfinite(self.stored_std))
            or np.any(self.stored_std <= 0)
        ):
            raise RuntimeError("Label-free source contains invalid numeric values.")
        self.window_size = int(self.windows.shape[-1])
        self.npz_sha256 = sha256_file(self.npz_path)
        self.signal_grid_hash = stable_array_hash(
            self.trial_ids,
            self.window_starts,
            self.stored_mean,
            self.stored_std,
        )
        self._trial_indices = {
            int(trial_id): np.flatnonzero(self.trial_ids == trial_id)
            for trial_id in np.unique(self.trial_ids)
        }
        self._sensor_cache: dict[int, np.ndarray] = {}
        self._sensor_hash: dict[int, str] = {}

    def trial_window_starts(self, trial_id: int) -> tuple[int, ...]:
        indices = self._trial_indices.get(int(trial_id))
        if indices is None or len(indices) == 0:
            raise KeyError(f"Unknown trial_global_id {trial_id}.")
        starts = np.sort(self.window_starts[indices], kind="stable")
        return tuple(int(value) for value in starts)

    def _reconstruct_from_windows(self, trial_id: int) -> np.ndarray:
        indices = self._trial_indices[int(trial_id)]
        order = np.argsort(self.window_starts[indices], kind="stable")
        indices = indices[order]
        starts = self.window_starts[indices]
        raw = (
            self.windows[indices] * self.stored_std + self.stored_mean
        ).astype(np.float32)
        visible_end = int(starts[-1] + self.window_size)
        accumulator = np.zeros((raw.shape[1], visible_end), dtype=np.float64)
        counts = np.zeros(visible_end, dtype=np.int32)
        for window, start in zip(raw, starts):
            begin = int(start)
            end = begin + self.window_size
            accumulator[:, begin:end] += window
            counts[begin:end] += 1
        if np.any(counts == 0):
            raise RuntimeError(
                f"Trial {trial_id} has gaps in reconstructed NPZ coverage."
            )
        return (accumulator / counts[None, :]).astype(np.float32)

    def sensor(self, trial_id: int) -> np.ndarray:
        trial_id = int(trial_id)
        cached = self._sensor_cache.get(trial_id)
        if cached is not None:
            return cached
        if trial_id not in self._trial_indices:
            raise KeyError(f"Unknown trial_global_id {trial_id}.")
        sensor = self._reconstruct_from_windows(trial_id)
        if not np.all(np.isfinite(sensor)):
            raise RuntimeError(f"Trial {trial_id} raw sensor signal is non-finite.")
        self._sensor_cache[trial_id] = sensor
        self._sensor_hash[trial_id] = stable_array_hash(sensor)
        return sensor

    def source_audit(self, trial_ids: Sequence[int]) -> dict:
        ids = sorted(set(int(value) for value in trial_ids))
        for trial_id in ids:
            self.sensor(trial_id)
        return {
            "source_npz": str(self.npz_path),
            "source_npz_size_bytes": int(self.npz_path.stat().st_size),
            "source_npz_sha256": self.npz_sha256,
            "signal_grid_hash_without_activity_metadata": self.signal_grid_hash,
            "trial_count": len(ids),
            "source_mode_counts": {
                "npz_inverse_overlap_deduplicated_label_free": len(ids)
            },
            "uniform_source_mode": True,
            "allow_pickle": False,
            "loaded_npz_fields": [
                "mean",
                "std",
                "trial_global_ids",
                "window_start_indices",
                "windows",
            ],
            "label_name_subject_or_path_fields_loaded": False,
            "raw_signal_manifest_hash": stable_text_hash(
                f"{trial_id}|{self._sensor_hash[trial_id]}" for trial_id in ids
            ),
        }


def _require_new_output_dir(path: Path) -> Path:
    resolved = path.resolve()
    if resolved.exists():
        raise FileExistsError(f"Output directory must not already exist: {resolved}")
    return resolved


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return parsed


def _trial_ids(dataset) -> list[int]:
    return sorted(
        set(np.asarray(dataset.trial_global_ids, dtype=np.int64).tolist())
    )


def _union_trial_ids(*datasets) -> list[int]:
    result: set[int] = set()
    for dataset in datasets:
        if dataset is not None:
            result.update(_trial_ids(dataset))
    return sorted(result)


def _union_trial_labels(*datasets) -> dict[int, int]:
    """Read labels only for a fail-closed session-boundary audit."""

    result: dict[int, int] = {}
    for dataset in datasets:
        if dataset is None:
            continue
        trial_ids = np.asarray(dataset.trial_global_ids, dtype=np.int64)
        labels = np.asarray(dataset.targets, dtype=np.int64)
        if trial_ids.ndim != 1 or labels.shape != trial_ids.shape:
            raise RuntimeError("Trial ids and targets disagree in a session dataset.")
        for trial_id, label in zip(trial_ids.tolist(), labels.tolist()):
            trial_id, label = int(trial_id), int(label)
            if trial_id in result and result[trial_id] != label:
                raise RuntimeError(
                    f"Trial {trial_id} has conflicting session labels."
                )
            result[trial_id] = label
    return result


def _union_trial_subjects(*datasets) -> dict[int, int]:
    result: dict[int, int] = {}
    for dataset in datasets:
        if dataset is None:
            continue
        trial_ids = np.asarray(dataset.trial_global_ids, dtype=np.int64)
        subjects = np.asarray(dataset.subject_ids, dtype=np.int64)
        if trial_ids.ndim != 1 or subjects.shape != trial_ids.shape:
            raise RuntimeError("Trial ids and subjects disagree in a session dataset.")
        for trial_id, subject_id in zip(trial_ids.tolist(), subjects.tolist()):
            trial_id, subject_id = int(trial_id), int(subject_id)
            if trial_id in result and result[trial_id] != subject_id:
                raise RuntimeError(
                    f"Trial {trial_id} has conflicting session subjects."
                )
            result[trial_id] = subject_id
    return result


def _validate_registered_session_protocol(
    audit: dict,
    novel_order: Sequence[int],
    labels_by_feature_trial: Mapping[int, int],
    subjects_by_feature_trial: Mapping[int, int],
) -> dict:
    observed_order = tuple(int(value) for value in novel_order)
    if observed_order != REGISTERED_NOVEL_ORDER:
        raise RuntimeError(
            "Registered novel-class order drifted: "
            f"{list(observed_order)} != {list(REGISTERED_NOVEL_ORDER)}."
        )
    observed_counts = {
        key: int(audit.get(key, -1)) for key in REGISTERED_SESSION_COUNTS
    }
    if observed_counts != REGISTERED_SESSION_COUNTS:
        raise RuntimeError(
            "Registered Session-2 trial counts drifted: "
            f"{observed_counts} != {REGISTERED_SESSION_COUNTS}."
        )
    expected_feature_ids = set(
        int(value) for value in audit["cumulative_train_trial_ids"]
    ) | set(int(value) for value in audit["session_2_test_trial_ids"])
    observed_feature_ids = set(int(value) for value in labels_by_feature_trial)
    if observed_feature_ids != expected_feature_ids:
        missing = sorted(expected_feature_ids - observed_feature_ids)
        extra = sorted(observed_feature_ids - expected_feature_ids)
        raise RuntimeError(
            "Session label audit and feature-trial manifest disagree: "
            f"missing={missing}, extra={extra}."
        )
    observed_subject_ids = set(int(value) for value in subjects_by_feature_trial)
    if observed_subject_ids != expected_feature_ids:
        missing = sorted(expected_feature_ids - observed_subject_ids)
        extra = sorted(observed_subject_ids - expected_feature_ids)
        raise RuntimeError(
            "Session subject audit and feature-trial manifest disagree: "
            f"missing={missing}, extra={extra}."
        )
    feature_subjects = sorted(
        set(int(value) for value in subjects_by_feature_trial.values())
    )
    if feature_subjects != [4, 5]:
        raise RuntimeError(
            f"Registered Session-2 feature subjects drifted: {feature_subjects}."
        )
    allowed = set(REGISTERED_SESSION2_ACTIVITY_IDS)
    future_ids = sorted(
        trial_id
        for trial_id, label in labels_by_feature_trial.items()
        if int(label) not in allowed
    )
    observed_activity_ids = sorted(
        set(int(value) for value in labels_by_feature_trial.values())
    )
    if future_ids or observed_activity_ids != list(REGISTERED_SESSION2_ACTIVITY_IDS):
        raise RuntimeError(
            "Future or missing activity entered the Session-2 feature manifest: "
            f"observed_classes={observed_activity_ids}, future_trials={future_ids}."
        )
    audit.update(
        {
            "registered_protocol_verified": True,
            "registered_novel_class_order": list(REGISTERED_NOVEL_ORDER),
            "allowed_session_2_activity_ids": list(
                REGISTERED_SESSION2_ACTIVITY_IDS
            ),
            "feature_trial_count": len(expected_feature_ids),
            "future_feature_trial_ids": [],
            "future_feature_trial_count": 0,
            "feature_subject_ids": feature_subjects,
            "subject_by_feature_trial_protocol_only": {
                str(trial_id): int(subjects_by_feature_trial[trial_id])
                for trial_id in sorted(expected_feature_ids)
            },
        }
    )
    return audit


def build_fold06_session_manifest(
    npz_path: Path,
    fit_subjects: Sequence[int],
    eval_subjects: Sequence[int],
    validation_subjects: Sequence[int],
    seed: int = 500,
) -> tuple[dict, object]:
    """Use the production loader to reproduce the exact trial stream."""

    args = SimpleNamespace(
        uschad_npz_path=str(Path(npz_path).resolve()),
        num_novel_class_per_session=2,
        num_novel_classes_per_session=2,
        uschad_sample_unit="trial",
        n_views=2,
        trial_view_mode="full_random_crop",
        trial_crop_ratio=2.0 / 3.0,
        trial_min_windows=2,
        har_aug_mode="weak_strong",
        har_weak_jitter_std=0.0,
        har_weak_scale_std=0.1,
        har_strong_jitter_std=0.0,
        har_strong_scale_std=0.2,
        har_time_mask_ratio=0.0,
        uschad_split_mode="subject",
        uschad_train_subjects=",".join(str(int(value)) for value in fit_subjects),
        uschad_test_subjects=",".join(str(int(value)) for value in eval_subjects),
        offline_val_subjects=",".join(
            str(int(value)) for value in validation_subjects
        ),
        uschad_recompute_norm_from_train_subjects=True,
        uschad_norm_eps=1e-6,
        har_in_channels=6,
        uschad_window_size=256,
        logger=None,
    )
    config = {
        "continual_session_num": 3,
        "online_novel_unseen_num": 5,
        "online_old_seen_num": 2,
        "online_novel_seen_num": 2,
        "sample_unit": "trial",
    }
    datasets, novel_order = get_uschad_datasets(
        train_transform=None,
        test_transform=None,
        config_dict=config,
        train_classes=range(6),
        prop_train_labels=0.8,
        split_train_val=False,
        is_shuffle=False,
        seed=int(seed),
        args=args,
    )
    old_sessions = datasets["online_old_dataset_unlabelled_list"]
    novel_sessions = datasets["online_novel_dataset_unlabelled_list"]
    test_sessions = datasets["online_test_dataset_list"]
    session_one = _union_trial_ids(old_sessions[0], novel_sessions[0])
    session_two = _union_trial_ids(old_sessions[1], novel_sessions[1])
    session_two_test = _trial_ids(test_sessions[1])
    feature_trial_labels = _union_trial_labels(
        old_sessions[0],
        novel_sessions[0],
        old_sessions[1],
        novel_sessions[1],
        test_sessions[1],
    )
    feature_trial_subjects = _union_trial_subjects(
        old_sessions[0],
        novel_sessions[0],
        old_sessions[1],
        novel_sessions[1],
        test_sessions[1],
    )
    audit = validate_session_manifest(session_one, session_two, session_two_test)
    audit.update(
        {
            "builder": "data.uschad.get_uschad_datasets",
            "seed": int(seed),
            "novel_class_order_protocol_only": np.asarray(
                novel_order, dtype=np.int64
            ).tolist(),
            "fit_subjects": sorted(int(value) for value in fit_subjects),
            "eval_subjects": sorted(int(value) for value in eval_subjects),
            "validation_subjects": sorted(
                int(value) for value in validation_subjects
            ),
            "session_1_incremental_train_count": len(session_one),
            "session_2_incremental_train_count": len(session_two),
            "session_2_cumulative_train_count": len(
                audit["cumulative_train_trial_ids"]
            ),
            "session_2_test_count": len(session_two_test),
        }
    )
    _validate_registered_session_protocol(
        audit,
        novel_order=np.asarray(novel_order, dtype=np.int64).tolist(),
        labels_by_feature_trial=feature_trial_labels,
        subjects_by_feature_trial=feature_trial_subjects,
    )
    return audit, test_sessions[1]


def _validate_fold06_run(
    run_dir: Path, npz_path: Path, seed: int
) -> tuple[dict, dict, Path, Path]:
    config, split, window_path, codebook_path = _validate_run(run_dir, npz_path)
    metadata = config.get("checkpoint_metadata", {})
    fold = int(metadata.get("uschad_cv_fold", -1))
    run_seed = int(config.get("arguments", {}).get("seed", -1))
    if fold != 6:
        raise RuntimeError(f"This registered minimum experiment requires fold06, got {fold}.")
    if run_seed != int(seed):
        raise RuntimeError(
            f"Session sampling seed must match the downstream run: {seed} != {run_seed}."
        )
    if sorted(int(value) for value in split["eval_subjects"]) != [4, 5]:
        raise RuntimeError(
            "The registered fold06 experiment requires held-out subjects [4,5]."
        )
    return config, split, window_path, codebook_path


def load_segment_artifacts(run_dir: Path, codebook_path: Path) -> SegmentArtifacts:
    segment_path = Path(run_dir) / "segment_embeddings_and_tokens.npz"
    if not segment_path.is_file():
        raise FileNotFoundError(segment_path)
    required = {
        "split_role",
        "trial_global_ids",
        "partition_start_samples",
        "partition_end_samples_exclusive",
        "primitive_tokens",
        "segment_embeddings",
    }
    with np.load(segment_path, allow_pickle=False) as data:
        missing = required - set(data.files)
        if missing:
            raise RuntimeError(f"Segment artifact lacks fields: {sorted(missing)}")
        arrays = {name: np.asarray(data[name]).copy() for name in required}
    with np.load(codebook_path, allow_pickle=False) as data:
        if "centers" not in data.files:
            raise RuntimeError("Codebook artifact has no centers.")
        centers = np.asarray(data["centers"], dtype=np.float32)
    row_count = len(arrays["trial_global_ids"])
    for name, value in arrays.items():
        if value.ndim == 0 or len(value) != row_count:
            raise RuntimeError(f"Invalid segment row array {name}: {value.shape}.")
    embeddings = np.asarray(arrays["segment_embeddings"], dtype=np.float32)
    tokens = np.asarray(arrays["primitive_tokens"], dtype=np.int64)
    if embeddings.ndim != 2 or centers.shape != (len(centers), embeddings.shape[1]):
        raise RuntimeError(
            f"Segment/codebook shapes disagree: {embeddings.shape}, {centers.shape}."
        )
    if np.any(tokens < 0) or np.any(tokens >= len(centers)):
        raise RuntimeError("Segment token lies outside the coarse codebook.")
    return SegmentArtifacts(
        split_role=np.asarray(arrays["split_role"], dtype=np.int8),
        trial_ids=np.asarray(arrays["trial_global_ids"], dtype=np.int64),
        starts=np.asarray(arrays["partition_start_samples"], dtype=np.int64),
        ends=np.asarray(arrays["partition_end_samples_exclusive"], dtype=np.int64),
        tokens=tokens,
        embeddings=embeddings,
        centers=centers,
    )


def build_trial_token_durations(
    artifacts: SegmentArtifacts,
    trial_ids: Sequence[int],
    visible_end_by_trial: Mapping[int, int] | None = None,
) -> dict[int, dict[int, float]]:
    result = {}
    for trial_id_value in trial_ids:
        trial_id = int(trial_id_value)
        positions = np.flatnonzero(artifacts.trial_ids == trial_id)
        if len(positions) == 0:
            raise RuntimeError(f"Trial {trial_id} is absent from segment artifacts.")
        if np.any(artifacts.split_role[positions] != 1):
            raise RuntimeError(f"Online trial {trial_id} is not an eval-role artifact.")
        order = np.argsort(artifacts.starts[positions], kind="stable")
        positions = positions[order]
        starts = artifacts.starts[positions]
        ends = artifacts.ends[positions]
        if np.any(ends <= starts) or int(starts[0]) != 0:
            raise RuntimeError(
                f"Trial {trial_id} has invalid or incomplete partition coverage."
            )
        if np.any(starts[1:] != ends[:-1]):
            raise RuntimeError(
                f"Trial {trial_id} partition coverage contains an overlap or gap."
            )
        if visible_end_by_trial is not None:
            if trial_id not in visible_end_by_trial:
                raise RuntimeError(f"Trial {trial_id} lacks a visible-end audit value.")
            if int(ends[-1]) != int(visible_end_by_trial[trial_id]):
                raise RuntimeError(
                    f"Trial {trial_id} partitions end at {int(ends[-1])}, not its "
                    f"visible end {int(visible_end_by_trial[trial_id])}."
                )
        durations: dict[int, float] = {}
        for token, duration in zip(
            artifacts.tokens[positions], (ends - starts).astype(np.float64)
        ):
            durations[int(token)] = durations.get(int(token), 0.0) + float(duration)
        result[trial_id] = durations
    return result


def build_token_occurrences(
    artifacts: SegmentArtifacts,
    source: LabelFreeSourceSignalRepository,
    trial_ids: Sequence[int],
    selected_token: int,
) -> dict[int, UnlabelledTokenOccurrence]:
    occurrences = {}
    for trial_id_value in trial_ids:
        trial_id = int(trial_id_value)
        positions = np.flatnonzero(artifacts.trial_ids == trial_id)
        order = np.argsort(artifacts.starts[positions], kind="stable")
        positions = positions[order]
        selected = positions[artifacts.tokens[positions] == int(selected_token)]
        if len(selected) == 0:
            continue
        all_durations = artifacts.ends[positions] - artifacts.starts[positions]
        selected_durations = artifacts.ends[selected] - artifacts.starts[selected]
        residual = duration_weighted_residual(
            artifacts.embeddings[selected],
            artifacts.centers[int(selected_token)],
            selected_durations,
        )
        gravity = gravity_from_token_partitions(
            source.sensor(trial_id),
            artifacts.starts[positions],
            artifacts.ends[positions],
            artifacts.tokens[positions],
            selected_token=int(selected_token),
        )
        occurrences[trial_id] = UnlabelledTokenOccurrence(
            trial_global_id=trial_id,
            coarse_token=int(selected_token),
            token_fraction=float(np.sum(selected_durations) / np.sum(all_durations)),
            duration_samples=int(np.sum(selected_durations)),
            mean_quantization_residual=residual,
            gravity_direction=gravity,
        )
    return occurrences


def _fit_trial_clusterer(
    train_features: np.ndarray, test_features: np.ndarray, seed: int
) -> tuple[np.ndarray, dict]:
    if len(train_features) < 10:
        raise RuntimeError("Session-2 cumulative train has fewer than K=10 trials.")
    model = KMeans(
        n_clusters=10,
        n_init=50,
        max_iter=300,
        random_state=int(seed),
        algorithm="lloyd",
    )
    model.fit(np.asarray(train_features, dtype=np.float32))
    predictions = model.predict(np.asarray(test_features, dtype=np.float32)).astype(
        np.int64
    )
    return predictions, {
        "cluster_count": 10,
        "n_init": 50,
        "max_iter": 300,
        "inertia": float(model.inertia_),
        "n_iter": int(model.n_iter_),
        "fit_uses_activity_labels": False,
    }


def _gravity_shuffle_null_predictions(
    artifacts: SegmentArtifacts,
    occurrences: Mapping[int, UnlabelledTokenOccurrence],
    fit_trial_ids: Sequence[int],
    trial_token_durations: Mapping[int, Mapping[int, float]],
    train_ids: Sequence[int],
    test_ids: Sequence[int],
    subjects_by_trial: Mapping[int, int],
    selected_token: int,
    shuffles: int,
    seed: int,
) -> tuple[dict[str, np.ndarray], dict[str, dict[str, np.ndarray]], dict]:
    if int(shuffles) < 1:
        raise ValueError("gravity_shuffles must be positive.")
    train_set = set(int(value) for value in train_ids)
    test_set = set(int(value) for value in test_ids)
    if train_set & test_set:
        raise RuntimeError("Gravity-control train/test trial roles overlap.")
    fit_ids = [int(value) for value in fit_trial_ids]
    if not fit_ids or len(fit_ids) != len(set(fit_ids)):
        raise RuntimeError(
            "Gravity-control fit trial ids must be non-empty and unique."
        )
    leaked_fit_ids = sorted(set(fit_ids) - train_set)
    if leaked_fit_ids:
        raise RuntimeError(
            "Gravity-control secondary fitting received a non-train trial: "
            f"{leaked_fit_ids}."
        )
    ordered_occurrences = [occurrences[key] for key in sorted(occurrences)]
    test_occurrence_ids = np.asarray(
        [
            int(item.trial_global_id)
            for item in ordered_occurrences
            if int(item.trial_global_id) in test_set
        ],
        dtype=np.int64,
    )
    if len(test_occurrence_ids) == 0:
        raise RuntimeError("No selected-token test occurrence is available.")
    split_roles = {
        int(item.trial_global_id): (
            "cumulative_online_train"
            if int(item.trial_global_id) in train_set
            else "session_2_test"
        )
        for item in ordered_occurrences
    }
    if any(
        int(item.trial_global_id) not in train_set | test_set
        for item in ordered_occurrences
    ):
        raise RuntimeError("An occurrence lies outside the visible train/test pools.")

    rng = np.random.default_rng(int(seed) + 7301)
    null_predictions: dict[str, list[np.ndarray]] = {
        "U2_gravity": [],
        "U3_joint": [],
    }
    null_test_children: dict[str, list[np.ndarray]] = {
        "U2_gravity": [],
        "U3_joint": [],
    }
    moved_fractions = []
    group_sizes = None
    for _ in range(int(shuffles)):
        shuffled, shuffle_audit = shuffle_gravity_within_subject_split(
            ordered_occurrences,
            subjects_by_trial=subjects_by_trial,
            split_roles_by_trial=split_roles,
            rng=rng,
        )
        moved_fractions.append(float(shuffle_audit["moved_occurrence_fraction"]))
        if group_sizes is None:
            group_sizes = shuffle_audit["group_sizes"]
        elif group_sizes != shuffle_audit["group_sizes"]:
            raise RuntimeError("Gravity-shuffle groups changed between permutations.")
        shuffled_by_trial = {
            int(item.trial_global_id): item for item in shuffled
        }
        shuffled_fit = [shuffled_by_trial[trial_id] for trial_id in fit_ids]
        for arm in ("U2_gravity", "U3_joint"):
            model = fit_secondary_codebook(
                shuffled_fit,
                arm=arm,
                restarts=50,
                seed=int(seed),
            )
            children = assign_secondary_codebook(model, shuffled)
            child_map = {
                int(item.trial_global_id): int(child)
                for item, child in zip(shuffled, children.tolist())
            }
            null_test_children[arm].append(
                np.asarray(
                    [child_map[int(trial_id)] for trial_id in test_occurrence_ids],
                    dtype=np.int64,
                )
            )
            train_histograms = duration_histograms(
                train_ids,
                trial_token_durations,
                primitive_num=len(artifacts.centers),
                selected_token=selected_token,
                child_by_trial=child_map,
            )
            test_histograms = duration_histograms(
                test_ids,
                trial_token_durations,
                primitive_num=len(artifacts.centers),
                selected_token=selected_token,
                child_by_trial=child_map,
            )
            predictions, _ = _fit_trial_clusterer(
                train_histograms,
                test_histograms,
                seed=int(seed),
            )
            null_predictions[arm].append(predictions)
    if not any(value > 0.0 for value in moved_fractions):
        raise RuntimeError("All gravity shuffles were identity permutations.")
    prediction_matrices = {
        arm: np.stack(values, axis=0).astype(np.int64)
        for arm, values in null_predictions.items()
    }
    child_assignment_matrices = {
        arm: {
            "trial_ids": test_occurrence_ids.copy(),
            "child_ids": np.stack(values, axis=0).astype(np.int64),
        }
        for arm, values in null_test_children.items()
    }
    return prediction_matrices, child_assignment_matrices, {
        "permutation_count": int(shuffles),
        "seed": int(seed) + 7301,
        "permutation_unit": "selected-token trial occurrence gravity descriptor",
        "grouping": "subject_x_split_role",
        "split_roles": ["cumulative_online_train", "session_2_test"],
        "group_sizes": group_sizes,
        "test_child_assignment_trial_ids": test_occurrence_ids.tolist(),
        "secondary_fit_trial_count": len(fit_ids),
        "secondary_fit_trial_ids_are_unique": True,
        "secondary_fit_trial_ids_are_train_only": True,
        "mean_moved_occurrence_fraction": float(np.mean(moved_fractions)),
        "minimum_moved_occurrence_fraction": float(np.min(moved_fractions)),
        "maximum_moved_occurrence_fraction": float(np.max(moved_fractions)),
        "preserves_each_subject_split_gravity_marginal": True,
        "test_occurrences_used_to_fit_codebook_or_kmeans": False,
        "uses_activity_labels_for_permutation_or_fitting": False,
    }


def _paired_bootstrap_comparisons(
    trial_ids: Sequence[int],
    y_true: np.ndarray,
    subjects: np.ndarray,
    comparisons: Mapping[str, tuple[np.ndarray, np.ndarray]],
    resamples: int,
    seed: int,
    scope: str,
) -> tuple[dict, list[dict]]:
    masks = {
        "all_accuracy": np.ones(len(y_true), dtype=bool),
        "old_accuracy": np.asarray(y_true, dtype=np.int64) < 6,
        "new_accuracy": np.asarray(y_true, dtype=np.int64) >= 6,
    }
    result = {}
    rows = []
    for comparison_index, (name, pair) in enumerate(comparisons.items()):
        reference, challenger = pair
        result[name] = {}
        for metric_index, (metric, mask) in enumerate(masks.items()):
            summary = stratified_paired_accuracy_bootstrap(
                trial_ids=trial_ids,
                y_true=y_true,
                subjects=subjects,
                reference_aligned=reference,
                challenger_aligned=challenger,
                subset_mask=mask,
                resamples=int(resamples),
                seed=int(seed) + 9101 + 10 * comparison_index + metric_index,
                confidence_level=0.95,
            )
            result[name][metric] = summary
            rows.append(
                {
                    "scope": scope,
                    "comparison": name,
                    "metric": metric,
                    "observed_difference": summary["observed_difference"],
                    "bootstrap_mean_difference": summary[
                        "bootstrap_mean_difference"
                    ],
                    "ci_lower": summary["ci_lower"],
                    "ci_median": summary["ci_median"],
                    "ci_upper": summary["ci_upper"],
                    "bootstrap_probability_gt_zero": summary[
                        "bootstrap_probability_gt_zero"
                    ],
                    "trial_count": summary["trial_count"],
                    "resamples": summary["resamples"],
                    "confidence_level": summary["confidence_level"],
                    "stratification": summary["stratification"],
                    "hungarian_mapping_policy": summary[
                        "hungarian_mapping_policy"
                    ],
                }
            )
    return {
        "scope": scope,
        "comparisons": result,
        "resamples": int(resamples),
        "confidence_level": 0.95,
        "resampling_unit": "trial",
        "stratification": "subject_x_activity",
        "hungarian_mapping_policy": "frozen_on_complete_test_set_before_bootstrap",
        "interpretation_limit": (
            "Conditional trial-level uncertainty for held-out Subjects 4 and 5; "
            "not a confidence interval for generalisation to new subjects."
        ),
    }, rows


def _truth_maps(
    source: SourceSignalRepository, trial_ids: Sequence[int]
) -> tuple[dict[int, int], dict[int, int], dict[int, str]]:
    labels, subjects, names = {}, {}, {}
    for trial_id_value in trial_ids:
        trial_id = int(trial_id_value)
        metadata = source.metadata(trial_id)
        labels[trial_id] = int(metadata["label"])
        subjects[trial_id] = int(metadata["subject_id"])
        names[trial_id] = str(metadata["activity_name"])
    return labels, subjects, names


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"Cannot write empty CSV {path}.")
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _rle_tokens(values: Sequence[int]) -> list[int]:
    encoded, _ = run_length_encode([int(value) for value in values])
    return encoded.astype(int).tolist()


def _arm_segment_tokens(
    artifacts: SegmentArtifacts,
    trial_id: int,
    arm: str,
    selected_token: int,
    child_by_trial: Mapping[int, int] | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    positions = np.flatnonzero(artifacts.trial_ids == int(trial_id))
    positions = positions[np.argsort(artifacts.starts[positions], kind="stable")]
    tokens = artifacts.tokens[positions].copy()
    if arm != "U0_coarse":
        child = int(child_by_trial[int(trial_id)]) if int(trial_id) in child_by_trial else 0
        selected = tokens == int(selected_token)
        # Give both children explicit new display ids; the histogram uses q/K.
        tokens[selected] = len(artifacts.centers) + child
    return tokens, artifacts.starts[positions], artifacts.ends[positions]


def _save_trajectory_plot(
    path: Path,
    artifacts: SegmentArtifacts,
    test_ids: Sequence[int],
    selected_token: int,
    children_by_arm: Mapping[str, Mapping[int, int]],
    labels: Mapping[int, int],
    subjects: Mapping[int, int],
) -> None:
    ordered = sorted(
        (int(value) for value in test_ids),
        key=lambda trial_id: (labels[trial_id], subjects[trial_id], trial_id),
    )
    height = max(12.0, 0.18 * len(ordered) * len(ARMS))
    fig, axes = plt.subplots(len(ARMS), 1, figsize=(18, height), sharex=True)
    palette = plt.get_cmap("tab20")
    child_colors = {len(artifacts.centers): "black", len(artifacts.centers) + 1: "magenta"}
    for axis, arm in zip(axes, ARMS):
        mapping = children_by_arm.get(arm, {})
        for row, trial_id in enumerate(ordered):
            tokens, starts, ends = _arm_segment_tokens(
                artifacts, trial_id, arm, selected_token, mapping
            )
            for token, start, end in zip(tokens, starts, ends):
                color = child_colors.get(
                    int(token), palette((int(token) % 20) / 19.0)
                )
                axis.barh(
                    row,
                    (int(end) - int(start)) / 100.0,
                    left=int(start) / 100.0,
                    height=0.82,
                    color=color,
                    linewidth=0,
                )
        axis.set_title(arm)
        axis.set_ylabel("test trial")
        axis.set_yticks(np.arange(len(ordered)))
        axis.set_yticklabels(
            [f"y{labels[t]}-S{subjects[t]}-T{t}" for t in ordered], fontsize=5
        )
        axis.invert_yaxis()
    axes[-1].set_xlabel("visible time (seconds)")
    fig.suptitle(
        f"Session-2 original/refined trajectories; automatically selected q={selected_token}",
        y=0.995,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _save_confusions(path: Path, arm_results: Mapping[str, dict], class_names: Sequence[str]) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    for axis, arm in zip(axes.flat, ARMS):
        confusion = np.asarray(
            arm_results[arm]["global_metrics"]["confusion_counts"], dtype=np.float64
        )
        normalized = np.divide(
            confusion,
            confusion.sum(axis=1, keepdims=True),
            out=np.zeros_like(confusion),
            where=confusion.sum(axis=1, keepdims=True) > 0,
        )
        image = axis.imshow(normalized, vmin=0.0, vmax=1.0, cmap="Blues")
        axis.set_title(
            f"{arm}: All={arm_results[arm]['global_metrics']['all_accuracy']:.3f}, "
            f"New={arm_results[arm]['global_metrics']['new_accuracy']:.3f}"
        )
        axis.set_xticks(range(len(class_names)), class_names, rotation=55, ha="right", fontsize=7)
        axis.set_yticks(range(len(class_names)), class_names, fontsize=7)
        axis.set_xlabel("globally aligned prediction")
        axis.set_ylabel("true activity")
        fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    fig.suptitle("Session-2 test confusion matrices (one global Hungarian per arm)")
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _save_activity_heatmaps(
    path: Path,
    matrices: Mapping[str, np.ndarray],
    class_names: Sequence[str],
) -> None:
    upper = max(float(np.max(value)) for value in matrices.values())
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    for axis, arm in zip(axes.flat, ARMS):
        matrix = np.asarray(matrices[arm], dtype=np.float64)
        image = axis.imshow(matrix, vmin=0.0, vmax=max(upper, 1e-6), cmap="magma")
        axis.set_title(arm)
        axis.set_xticks(range(len(class_names)), class_names, rotation=55, ha="right", fontsize=7)
        axis.set_yticks(range(len(class_names)), class_names, fontsize=7)
        for row in range(len(matrix)):
            for column in range(len(matrix)):
                axis.text(
                    column,
                    row,
                    f"{matrix[row, column]:.2f}",
                    ha="center",
                    va="center",
                    fontsize=5,
                    color="white" if matrix[row, column] > 0.55 * max(upper, 1e-6) else "black",
                )
        fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    fig.suptitle("Session-2 activity mean refined-histogram Jensen-Shannon distances")
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def run(args: argparse.Namespace) -> dict:
    gravity_shuffles = int(args.gravity_shuffles)
    bootstrap_resamples = int(args.bootstrap_resamples)
    if gravity_shuffles < 1:
        raise ValueError("gravity_shuffles must be positive.")
    if bootstrap_resamples < 1:
        raise ValueError("bootstrap_resamples must be positive.")
    run_dir = Path(args.run_dir).resolve()
    npz_path = Path(args.npz_path).resolve()
    output_dir = _require_new_output_dir(Path(args.output_dir))
    config, split, _, codebook_path = _validate_fold06_run(
        run_dir, npz_path, seed=int(args.seed)
    )
    validation_subjects = config.get("checkpoint_metadata", {}).get(
        "offline_val_subjects", []
    )
    manifest, _ = build_fold06_session_manifest(
        npz_path=npz_path,
        fit_subjects=split["fit_subjects"],
        eval_subjects=split["eval_subjects"],
        validation_subjects=validation_subjects,
        seed=int(args.seed),
    )
    train_ids = [int(value) for value in manifest["cumulative_train_trial_ids"]]
    test_ids = [int(value) for value in manifest["session_2_test_trial_ids"]]
    all_ids = train_ids + test_ids
    subjects_by_feature_trial = {
        int(trial_id): int(subject_id)
        for trial_id, subject_id in manifest[
            "subject_by_feature_trial_protocol_only"
        ].items()
    }
    if set(subjects_by_feature_trial) != set(all_ids):
        raise RuntimeError(
            "Protocol-only subject map does not exactly cover the feature trials."
        )
    artifacts = load_segment_artifacts(run_dir, codebook_path)
    source = LabelFreeSourceSignalRepository(npz_path)
    visible_end_by_trial = {
        trial_id: int(max(source.trial_window_starts(trial_id)) + source.window_size)
        for trial_id in all_ids
    }
    trial_token_durations = build_trial_token_durations(
        artifacts,
        all_ids,
        visible_end_by_trial=visible_end_by_trial,
    )
    train_token_durations = {trial_id: trial_token_durations[trial_id] for trial_id in train_ids}
    candidate = select_dominant_token(
        train_token_durations, minimum_fraction=0.50, minimum_support=10
    )
    selected_token = int(candidate["selected_token"])
    occurrences = build_token_occurrences(
        artifacts, source, all_ids, selected_token=selected_token
    )
    dominant_train_ids = [
        int(value) for value in candidate["dominant_trial_ids"]
    ]
    fit_occurrences = [occurrences[trial_id] for trial_id in dominant_train_ids]

    predictions_by_arm: dict[str, np.ndarray] = {}
    histograms_by_arm: dict[str, dict[str, np.ndarray]] = {}
    children_by_arm: dict[str, dict[int, int]] = {"U0_coarse": {}}
    fit_audit_by_arm: dict[str, dict | None] = {"U0_coarse": None}
    readout_audit_by_arm: dict[str, dict] = {}
    for arm in ARMS:
        if arm == "U0_coarse":
            train_histograms = duration_histograms(
                train_ids,
                trial_token_durations,
                primitive_num=len(artifacts.centers),
            )
            test_histograms = duration_histograms(
                test_ids,
                trial_token_durations,
                primitive_num=len(artifacts.centers),
            )
        else:
            model = fit_secondary_codebook(
                fit_occurrences,
                arm=arm,
                restarts=50,
                seed=int(args.seed),
            )
            all_occurrences = [occurrences[trial_id] for trial_id in sorted(occurrences)]
            all_children = assign_secondary_codebook(model, all_occurrences)
            child_map = {
                int(item.trial_global_id): int(child)
                for item, child in zip(all_occurrences, all_children.tolist())
            }
            children_by_arm[arm] = child_map
            fit_audit_by_arm[arm] = secondary_fit_diagnostics(model, fit_occurrences)
            train_histograms = duration_histograms(
                train_ids,
                trial_token_durations,
                primitive_num=len(artifacts.centers),
                selected_token=selected_token,
                child_by_trial=child_map,
            )
            test_histograms = duration_histograms(
                test_ids,
                trial_token_durations,
                primitive_num=len(artifacts.centers),
                selected_token=selected_token,
                child_by_trial=child_map,
            )
        predictions, readout_audit = _fit_trial_clusterer(
            train_histograms, test_histograms, seed=int(args.seed)
        )
        predictions_by_arm[arm] = predictions
        histograms_by_arm[arm] = {
            "train": train_histograms,
            "test": test_histograms,
        }
        readout_audit_by_arm[arm] = readout_audit

    # This negative control is completed without an activity label/name and
    # before the metadata-aware repository exists.  Subject ids are used only
    # to constrain permutation groups and never enter either feature vector.
    (
        null_predictions_by_arm,
        null_test_children_by_arm,
        gravity_shuffle_fit_audit,
    ) = (
        _gravity_shuffle_null_predictions(
            artifacts=artifacts,
            occurrences=occurrences,
            fit_trial_ids=dominant_train_ids,
            trial_token_durations=trial_token_durations,
            train_ids=train_ids,
            test_ids=test_ids,
            subjects_by_trial=subjects_by_feature_trial,
            selected_token=selected_token,
            shuffles=gravity_shuffles,
            seed=int(args.seed),
        )
    )

    # Ground truth is joined only after every child assignment and Session-2
    # observed/null downstream prediction above has been frozen.
    evaluation_source = SourceSignalRepository(npz_path)
    labels, subjects, names = _truth_maps(evaluation_source, test_ids)
    y_true = np.asarray([labels[trial_id] for trial_id in test_ids], dtype=np.int64)
    class_ids = sorted(set(y_true.tolist()))
    class_names = [
        names[next(trial_id for trial_id in test_ids if labels[trial_id] == class_id)]
        for class_id in class_ids
    ]
    arm_results = {}
    distance_matrices = {}
    for arm in ARMS:
        global_metrics = global_hungarian_metrics(
            y_true, predictions_by_arm[arm], old_class_count=6
        )
        secondary = None
        if arm in SPLIT_ARMS:
            test_child_map = {
                trial_id: children_by_arm[arm][trial_id]
                for trial_id in test_ids
                if trial_id in children_by_arm[arm]
            }
            secondary = posthoc_secondary_metrics(
                test_child_map, labels, subjects, sitting_label=7, standing_label=8
            )
        distance = class_mean_histogram_distances(
            histograms_by_arm[arm]["test"], y_true, class_ids
        )
        distance_matrices[arm] = distance
        arm_results[arm] = {
            "secondary_codebook_fit": fit_audit_by_arm[arm],
            "trial_clusterer": readout_audit_by_arm[arm],
            "global_metrics": global_metrics,
            "secondary_sit_stand_posthoc": secondary,
            "activity_mean_histogram_distance_matrix": distance,
        }

    gravity_metric_names = (
        "all_accuracy",
        "old_accuracy",
        "new_accuracy",
        "macro_f1",
    )
    gravity_shuffle_rows = []
    gravity_shuffle_arms = {}
    global_null_values_by_arm = {}

    def append_gravity_summary_row(
        arm_or_contrast: str,
        statistic: str,
        metric: str,
        endpoint_status: str,
        summary: Mapping[str, object],
        mapping_policy: str,
        null_artifact: str,
        null_artifact_sha256: str,
    ) -> None:
        quantiles = summary["null_quantiles"]
        gravity_shuffle_rows.append(
            {
                "arm_or_contrast": arm_or_contrast,
                "statistic": statistic,
                "metric": metric,
                "endpoint_status": endpoint_status,
                "observed": summary["observed"],
                "null_mean": summary["null_mean"],
                "observed_minus_null_mean": summary[
                    "observed_minus_null_mean"
                ],
                "monte_carlo_upper_tail_p": summary[
                    "monte_carlo_upper_tail_p"
                ],
                "null_q025": quantiles["q025"],
                "null_q500": quantiles["q500"],
                "null_q975": quantiles["q975"],
                "permutation_count": summary["permutation_count"],
                "grouping": gravity_shuffle_fit_audit["grouping"],
                "null_mapping_policy": mapping_policy,
                "null_artifact": null_artifact,
                "null_artifact_sha256": null_artifact_sha256,
            }
        )

    for arm in ("U2_gravity", "U3_joint"):
        null_matrix = np.asarray(null_predictions_by_arm[arm], dtype=np.int64)
        if null_matrix.shape != (gravity_shuffles, len(test_ids)):
            raise RuntimeError(
                f"Unexpected {arm} gravity-null prediction shape {null_matrix.shape}."
            )
        null_global_metrics = [
            global_hungarian_metrics(y_true, predictions, old_class_count=6)
            for predictions in null_matrix
        ]
        metric_summaries = {}
        global_null_values_by_arm[arm] = {}
        prediction_hash = stable_array_hash(null_matrix)
        for metric in gravity_metric_names:
            null_values = np.asarray(
                [item[metric] for item in null_global_metrics], dtype=np.float64
            )
            global_null_values_by_arm[arm][metric] = null_values
            summary = monte_carlo_upper_tail_summary(
                observed=float(arm_results[arm]["global_metrics"][metric]),
                null_values=null_values,
            )
            summary["null_values"] = null_values.tolist()
            metric_summaries[metric] = summary
            append_gravity_summary_row(
                arm_or_contrast=arm,
                statistic="absolute_arm_metric",
                metric=metric,
                endpoint_status="diagnostic_unadjusted",
                summary=summary,
                mapping_policy=(
                    "one_global_complete_test_hungarian_per_permutation"
                ),
                null_artifact="session2_raw_cluster_prediction_matrix",
                null_artifact_sha256=prediction_hash,
            )

        child_payload = null_test_children_by_arm[arm]
        child_trial_ids = np.asarray(child_payload["trial_ids"], dtype=np.int64)
        child_matrix = np.asarray(child_payload["child_ids"], dtype=np.int64)
        expected_child_ids = np.asarray(
            gravity_shuffle_fit_audit["test_child_assignment_trial_ids"],
            dtype=np.int64,
        )
        if not np.array_equal(child_trial_ids, expected_child_ids):
            raise RuntimeError(f"{arm} null child trial order drifted.")
        if child_matrix.shape != (gravity_shuffles, len(child_trial_ids)):
            raise RuntimeError(
                f"Unexpected {arm} gravity-null child shape {child_matrix.shape}."
            )
        if np.any((child_matrix < 0) | (child_matrix > 1)):
            raise RuntimeError(f"{arm} null child matrix contains a non-binary id.")
        null_sit_stand_results = []
        for child_ids in child_matrix:
            child_map = {
                int(trial_id): int(child)
                for trial_id, child in zip(
                    child_trial_ids.tolist(), child_ids.tolist()
                )
            }
            null_sit_stand_results.append(
                posthoc_secondary_metrics(
                    child_map,
                    labels,
                    subjects,
                    sitting_label=7,
                    standing_label=8,
                )
            )
        sit_stand_null_values = np.asarray(
            [
                item["binary_balanced_accuracy"]
                for item in null_sit_stand_results
            ],
            dtype=np.float64,
        )
        sit_stand_summary = monte_carlo_upper_tail_summary(
            observed=float(
                arm_results[arm]["secondary_sit_stand_posthoc"][
                    "binary_balanced_accuracy"
                ]
            ),
            null_values=sit_stand_null_values,
        )
        sit_stand_summary["null_values"] = sit_stand_null_values.tolist()
        metric_summaries[
            "sit_stand_binary_balanced_accuracy"
        ] = sit_stand_summary
        child_hash = stable_array_hash(child_trial_ids, child_matrix)
        append_gravity_summary_row(
            arm_or_contrast=arm,
            statistic="absolute_arm_metric",
            metric="sit_stand_binary_balanced_accuracy",
            endpoint_status="exploratory_posthoc_unadjusted",
            summary=sit_stand_summary,
            mapping_policy=(
                "one_binary_hungarian_per_permutation_on_the_complete_frozen_"
                "selected_token_sit_stand_subset"
            ),
            null_artifact="selected_token_test_child_assignment_matrix",
            null_artifact_sha256=child_hash,
        )
        for subject_id in (4, 5):
            subject_key = str(subject_id)
            observed_subject = arm_results[arm][
                "secondary_sit_stand_posthoc"
            ]["binary_by_subject_using_same_global_mapping"][subject_key]
            if not observed_subject["complete_binary_support"]:
                raise RuntimeError(
                    f"Observed Sit/Stand support is incomplete for Subject {subject_id}."
                )
            subject_null_values = np.asarray(
                [
                    item["binary_by_subject_using_same_global_mapping"][
                        subject_key
                    ]["balanced_accuracy_using_global_binary_mapping"]
                    for item in null_sit_stand_results
                ],
                dtype=np.float64,
            )
            if not np.all(np.isfinite(subject_null_values)):
                raise RuntimeError(
                    f"Null Sit/Stand support is incomplete for Subject {subject_id}."
                )
            subject_metric = (
                f"sit_stand_subject_{subject_id}_balanced_accuracy_using_"
                "global_binary_mapping"
            )
            subject_summary = monte_carlo_upper_tail_summary(
                observed=float(
                    observed_subject[
                        "balanced_accuracy_using_global_binary_mapping"
                    ]
                ),
                null_values=subject_null_values,
            )
            subject_summary["null_values"] = subject_null_values.tolist()
            metric_summaries[subject_metric] = subject_summary
            append_gravity_summary_row(
                arm_or_contrast=arm,
                statistic="absolute_arm_metric",
                metric=subject_metric,
                endpoint_status=(
                    "exploratory_posthoc_subject_specific_unadjusted"
                ),
                summary=subject_summary,
                mapping_policy=(
                    "reuse_each_permutation_global_binary_hungarian_mapping_"
                    f"within_subject_{subject_id}"
                ),
                null_artifact="selected_token_test_child_assignment_matrix",
                null_artifact_sha256=child_hash,
            )
        gravity_shuffle_arms[arm] = {
            "null_prediction_matrix_shape": list(null_matrix.shape),
            "null_prediction_matrix_sha256": prediction_hash,
            "null_global_hungarian_evaluations": gravity_shuffles,
            "null_test_child_trial_ids": child_trial_ids.tolist(),
            "null_test_child_assignment_matrix_shape": list(child_matrix.shape),
            "null_test_child_assignment_matrix_sha256": child_hash,
            "null_sit_stand_binary_hungarian_evaluations": gravity_shuffles,
            "metrics": metric_summaries,
        }

    u2_minus_u0_equivalence = {}
    for metric in gravity_metric_names:
        fixed_u0 = float(arm_results["U0_coarse"]["global_metrics"][metric])
        observed_u2 = float(arm_results["U2_gravity"]["global_metrics"][metric])
        absolute_summary = gravity_shuffle_arms["U2_gravity"]["metrics"][metric]
        delta_null = global_null_values_by_arm["U2_gravity"][metric] - fixed_u0
        delta_summary = monte_carlo_upper_tail_summary(
            observed=observed_u2 - fixed_u0,
            null_values=delta_null,
        )
        delta_summary["null_values"] = delta_null.tolist()
        same_effect = bool(
            np.isclose(
                delta_summary["observed_minus_null_mean"],
                absolute_summary["observed_minus_null_mean"],
                rtol=0.0,
                atol=1e-12,
            )
        )
        same_p = bool(
            np.isclose(
                delta_summary["monte_carlo_upper_tail_p"],
                absolute_summary["monte_carlo_upper_tail_p"],
                rtol=0.0,
                atol=0.0,
            )
        )
        if not (same_effect and same_p):
            raise RuntimeError(
                f"Fixed-U0 gravity-null equivalence failed for {metric}."
            )
        u2_minus_u0_equivalence[metric] = {
            "fixed_u0_observed": fixed_u0,
            "absolute_u2_observed": observed_u2,
            "u2_minus_u0": delta_summary,
            "same_observed_minus_null_mean_as_absolute_u2": same_effect,
            "same_upper_tail_p_as_absolute_u2": same_p,
        }
        append_gravity_summary_row(
            arm_or_contrast="U2_gravity_minus_U0_coarse",
            statistic="challenger_minus_fixed_baseline",
            metric=metric,
            endpoint_status=(
                "algebraic_reexpression_not_U2_vs_U0_superiority_test"
            ),
            summary=delta_summary,
            mapping_policy=(
                "subtract_fixed_observed_U0_after_one_global_complete_test_"
                "hungarian_per_U2_permutation"
            ),
            null_artifact="session2_raw_cluster_prediction_matrix",
            null_artifact_sha256=gravity_shuffle_arms["U2_gravity"][
                "null_prediction_matrix_sha256"
            ],
        )
    gravity_shuffle_negative_control = {
        "name": "within_subject_split_gravity_shuffle_negative_control_v1",
        "fit_and_permutation_audit": gravity_shuffle_fit_audit,
        "labels_joined_only_after_all_null_predictions_frozen": True,
        "null_alignment_policy": (
            "one global Hungarian alignment on the complete Session-2 test set "
            "for each frozen null prediction vector"
        ),
        "tail": "upper_one_sided",
        "p_value_formula": "(1 + count(null >= observed)) / (B + 1)",
        "arms": gravity_shuffle_arms,
        "u2_minus_u0_fixed_baseline_equivalence": {
            "u0_is_unchanged_by_gravity_shuffle": True,
            "is_a_test_of_u2_superiority_to_u0": False,
            "reason": (
                "Subtracting the same frozen U0 value from the observed U2 and "
                "every shuffled U2 replicate preserves ordering and observed-minus-"
                "null-mean; therefore the Monte Carlo upper-tail p value and effect "
                "size are identical to the absolute-U2 test. U2 versus U0 performance "
                "must instead be read from the paired trial-bootstrap interval."
            ),
            "metrics": u2_minus_u0_equivalence,
        },
        "multiplicity_adjustment": "none",
        "sit_stand_endpoint_status": "exploratory_posthoc_unadjusted",
        "attribution_limit": (
            "A significant U2 result supports informative trial-gravity pairing, "
            "not a causal contribution from encoder training."
        ),
    }

    aligned_by_arm = {
        arm: np.asarray(
            arm_results[arm]["global_metrics"]["aligned_predictions"],
            dtype=np.int64,
        )
        for arm in ARMS
    }
    trial_bootstrap_confidence_intervals, bootstrap_rows = (
        _paired_bootstrap_comparisons(
            trial_ids=test_ids,
            y_true=y_true,
            subjects=np.asarray(
                [subjects[trial_id] for trial_id in test_ids], dtype=np.int64
            ),
            comparisons={
                f"{arm}_minus_U0_coarse": (
                    aligned_by_arm["U0_coarse"],
                    aligned_by_arm[arm],
                )
                for arm in SPLIT_ARMS
            },
            resamples=bootstrap_resamples,
            seed=int(args.seed),
            scope="fold06_session2_subjects_4_5",
        )
    )

    output_dir.mkdir(parents=True)
    raw_null_path = output_dir / "gravity_shuffle_null_assignments.npz"
    np.savez_compressed(
        raw_null_path,
        permutation_index=np.arange(gravity_shuffles, dtype=np.int64),
        session2_test_trial_ids=np.asarray(test_ids, dtype=np.int64),
        selected_token_test_trial_ids=np.asarray(
            null_test_children_by_arm["U2_gravity"]["trial_ids"],
            dtype=np.int64,
        ),
        U2_gravity_raw_cluster_predictions=np.asarray(
            null_predictions_by_arm["U2_gravity"], dtype=np.int64
        ),
        U3_joint_raw_cluster_predictions=np.asarray(
            null_predictions_by_arm["U3_joint"], dtype=np.int64
        ),
        U2_gravity_selected_token_child_ids=np.asarray(
            null_test_children_by_arm["U2_gravity"]["child_ids"],
            dtype=np.int64,
        ),
        U3_joint_selected_token_child_ids=np.asarray(
            null_test_children_by_arm["U3_joint"]["child_ids"],
            dtype=np.int64,
        ),
    )
    gravity_shuffle_negative_control["raw_null_artifact"] = {
        "file": raw_null_path.name,
        "sha256": sha256_file(raw_null_path),
        "contains_activity_labels_or_names": False,
        "arrays": [
            "permutation_index",
            "session2_test_trial_ids",
            "selected_token_test_trial_ids",
            "U2_gravity_raw_cluster_predictions",
            "U3_joint_raw_cluster_predictions",
            "U2_gravity_selected_token_child_ids",
            "U3_joint_selected_token_child_ids",
        ],
    }
    manifest_rows = []
    for split_name, session, values in [
        ("online_train", 1, manifest["session_1_train_trial_ids"]),
        ("online_train", 2, manifest["session_2_train_trial_ids"]),
        ("online_test", 2, manifest["session_2_test_trial_ids"]),
    ]:
        for trial_id in values:
            manifest_rows.append(
                {"split": split_name, "session": session, "trial_global_id": int(trial_id)}
            )
    _write_csv(output_dir / "session_manifest.csv", manifest_rows)

    prediction_rows = []
    for row, trial_id in enumerate(test_ids):
        item = {
            "trial_global_id": trial_id,
            "subject_id": subjects[trial_id],
            "activity_label_0based": labels[trial_id],
            "activity_name": names[trial_id],
        }
        for arm in ARMS:
            item[f"{arm}_raw_cluster"] = int(predictions_by_arm[arm][row])
            item[f"{arm}_aligned_prediction"] = int(
                arm_results[arm]["global_metrics"]["aligned_predictions"][row]
            )
            item[f"{arm}_selected_token_child"] = (
                children_by_arm.get(arm, {}).get(trial_id, "")
            )
        prediction_rows.append(item)
    _write_csv(output_dir / "session2_predictions.csv", prediction_rows)

    trajectory_rows = []
    for trial_id in test_ids:
        row = {
            "trial_global_id": trial_id,
            "subject_id": subjects[trial_id],
            "activity_label_0based": labels[trial_id],
            "activity_name": names[trial_id],
        }
        for arm in ARMS:
            tokens, starts, ends = _arm_segment_tokens(
                artifacts,
                trial_id,
                arm,
                selected_token,
                children_by_arm.get(arm, {}),
            )
            row[f"{arm}_rle_tokens"] = json.dumps(_rle_tokens(tokens.tolist()))
            row[f"{arm}_segment_tokens"] = json.dumps(tokens.astype(int).tolist())
            row[f"{arm}_durations_samples"] = json.dumps(
                (ends - starts).astype(int).tolist()
            )
        trajectory_rows.append(row)
    _write_csv(output_dir / "session2_refined_trajectories.csv", trajectory_rows)
    _write_csv(
        output_dir / "gravity_shuffle_negative_control.csv",
        gravity_shuffle_rows,
    )
    _write_csv(
        output_dir / "trial_bootstrap_confidence_intervals.csv",
        bootstrap_rows,
    )

    _save_trajectory_plot(
        output_dir / "session2_original_and_refined_trajectories.png",
        artifacts,
        test_ids,
        selected_token,
        children_by_arm,
        labels,
        subjects,
    )
    _save_confusions(
        output_dir / "session2_global_confusions.png", arm_results, class_names
    )
    _save_activity_heatmaps(
        output_dir / "session2_activity_distance_heatmaps.png",
        distance_matrices,
        class_names,
    )

    result = {
        "protocol": PROTOCOL,
        "arguments": {
            "run_dir": str(run_dir),
            "npz_path": str(npz_path),
            "output_dir": str(output_dir),
            "seed": int(args.seed),
            "gravity_shuffles": gravity_shuffles,
            "bootstrap_resamples": bootstrap_resamples,
        },
        "input_audit": {
            "primitive_segmentation": config.get("segmentation", {}).get("method"),
            "encoder_ablation_profile": config.get("encoder_training", {}).get(
                "ablation_profile"
            ),
            "encoder_implementation_fingerprint": config.get(
                "encoder_implementation_fingerprint"
            ),
            "experiment_implementation_fingerprint": {
                "runner_path": str(Path(__file__).resolve()),
                "runner_sha256": sha256_file(Path(__file__).resolve()),
                "secondary_codebook_helper_path": str(
                    PROJECT_ROOT
                    / "experiments"
                    / "motion_primitive"
                    / "online_secondary_codebook.py"
                ),
                "secondary_codebook_helper_sha256": sha256_file(
                    PROJECT_ROOT
                    / "experiments"
                    / "motion_primitive"
                    / "online_secondary_codebook.py"
                ),
            },
            "checkpoint": config.get("checkpoint"),
            "checkpoint_sha256": config.get("checkpoint_sha256"),
            "run_dir": str(run_dir),
            "run_config_sha256": sha256_file(run_dir / "experiment_config.json"),
            "segment_artifact_sha256": sha256_file(
                run_dir / "segment_embeddings_and_tokens.npz"
            ),
            "codebook_artifact_sha256": sha256_file(codebook_path),
            "npz_sha256": sha256_file(npz_path),
            "label_free_raw_source": source.source_audit(all_ids),
            "postfreeze_metadata_and_mat_source_audit": (
                evaluation_source.source_audit(all_ids)
            ),
        },
        "session_manifest_audit": manifest,
        "label_firewall": {
            "occurrence_type_has_label_or_name": False,
            "candidate_selection_uses_labels": False,
            "secondary_fit_uses_labels": False,
            "trial_kmeans_fit_uses_labels": False,
            "session_builder_label_use": (
                "protocol isolation audit only; labels are not passed to features, "
                "distances, secondary fitting, or KMeans"
            ),
            "raw_sensor_accessor": (
                "NPZ inverse-normalisation overlap-add without metadata()"
            ),
            "raw_sensor_accessor_calls_metadata_before_fit": False,
            "metadata_aware_repository_created_after_predictions": True,
            "evaluation_ground_truth_join_time": (
                "after all observed and gravity-null child assignments and test "
                "predictions were frozen"
            ),
            "gravity_shuffle_uses_activity_labels": False,
            "gravity_shuffle_uses_subject_only_for_grouping": True,
            "future_feature_trial_count": int(
                manifest["future_feature_trial_count"]
            ),
        },
        "candidate": candidate,
        "arms": arm_results,
        "gravity_shuffle_negative_control": gravity_shuffle_negative_control,
        "trial_bootstrap_confidence_intervals": (
            trial_bootstrap_confidence_intervals
        ),
        "generated_files": [
            "online_secondary_codebook_results.json",
            "session_manifest.csv",
            "session2_predictions.csv",
            "session2_refined_trajectories.csv",
            "session2_original_and_refined_trajectories.png",
            "session2_global_confusions.png",
            "session2_activity_distance_heatmaps.png",
            "gravity_shuffle_negative_control.csv",
            "gravity_shuffle_null_assignments.npz",
            "trial_bootstrap_confidence_intervals.csv",
        ],
    }
    (output_dir / "online_secondary_codebook_results.json").write_text(
        json.dumps(jsonable(result), indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fold06 Session-2 unlabelled within-token secondary codebook experiment."
    )
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--npz-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=500)
    parser.add_argument("--gravity-shuffles", type=_positive_int, default=200)
    parser.add_argument("--bootstrap-resamples", type=_positive_int, default=5000)
    return parser.parse_args()


def main() -> None:
    result = run(parse_args())
    print(json.dumps(jsonable({
        "output_dir": result["arguments"]["output_dir"],
        "candidate": result["candidate"],
        "metrics": {
            arm: result["arms"][arm]["global_metrics"]
            for arm in ARMS
        },
    }), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
