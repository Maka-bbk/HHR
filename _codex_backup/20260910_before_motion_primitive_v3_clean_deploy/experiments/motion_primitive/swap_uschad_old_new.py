#!/usr/bin/env python3
"""Build a diagnostic USC-HAD NPZ with the old/new class halves swapped.

The output uses a contiguous label space whose first six classes are the
original USC-HAD zero-based classes 6--11.  The transformation is diagnostic
only: it deliberately changes which semantic classes supply the saved
normalisation statistics and does not implement online codebook adaptation.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


CLASS_COUNT = 12
OLD_CLASS_COUNT = 6
CLASS_PERMUTATION = np.asarray(
    [*range(OLD_CLASS_COUNT, CLASS_COUNT), *range(OLD_CLASS_COUNT)],
    dtype=np.int64,
)
ORIGINAL_TO_NEW = np.argsort(CLASS_PERMUTATION).astype(np.int64)
REQUIRED_ARRAYS = (
    "windows",
    "labels",
    "labels_1based",
    "subject_ids",
    "activity_names",
    "mean",
    "std",
    "stat_mask",
    "old_mask",
    "new_mask",
    "source_mask",
    "stage0_labeled_mask",
)
PROVENANCE_ARRAYS = (
    "original_labels",
    "original_labels_1based",
    "original_activity_names",
    "class_permutation",
)


def sha256_file(path: Path, chunk_size: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _integer_list(values: Any, context: str) -> list[int]:
    if not isinstance(values, (list, tuple)) or not values:
        raise ValueError(f"{context} must be a non-empty integer list.")
    try:
        result = [int(value) for value in values]
    except (TypeError, ValueError) as error:
        raise ValueError(f"{context} must be a non-empty integer list.") from error
    if len(result) != len(set(result)):
        raise ValueError(f"{context} contains duplicate values: {result}.")
    return result


def _activity_lookup(meta: Mapping[str, Any]) -> dict[int, str]:
    raw = meta.get("activity_names")
    if not isinstance(raw, Mapping):
        raise ValueError("meta.json activity_names must be a label-to-name mapping.")
    result: dict[int, str] = {}
    for label_1based in range(1, CLASS_COUNT + 1):
        value = raw.get(str(label_1based), raw.get(label_1based))
        if value is None or not str(value).strip():
            raise ValueError(
                "meta.json activity_names lacks original 1-based label "
                f"{label_1based}."
            )
        result[label_1based] = str(value)
    return result


def _validate_vector(name: str, value: np.ndarray, sample_count: int) -> None:
    if value.ndim != 1 or len(value) != sample_count:
        raise ValueError(
            f"NPZ {name} must have shape ({sample_count},), got {value.shape}."
        )


def _validate_input(
    arrays: Mapping[str, np.ndarray], meta: Mapping[str, Any]
) -> tuple[np.ndarray, np.ndarray, list[int], dict[int, str]]:
    missing = [name for name in REQUIRED_ARRAYS if name not in arrays]
    if missing:
        raise ValueError(f"Input NPZ lacks required arrays: {missing}.")
    already_swapped = [name for name in PROVENANCE_ARRAYS if name in arrays]
    if already_swapped:
        raise ValueError(
            "Input already contains swap provenance and will not be transformed "
            f"again: {already_swapped}."
        )

    windows = np.asarray(arrays["windows"])
    if windows.ndim != 3 or windows.shape[0] == 0:
        raise ValueError(f"NPZ windows must be non-empty [N,C,T], got {windows.shape}.")
    if not np.issubdtype(windows.dtype, np.floating):
        raise ValueError(f"NPZ windows must be floating point, got {windows.dtype}.")
    if not np.all(np.isfinite(windows)):
        raise ValueError("NPZ windows contains non-finite values.")
    sample_count, channel_count, _ = windows.shape

    labels = np.asarray(arrays["labels"], dtype=np.int64)
    labels_1based = np.asarray(arrays["labels_1based"], dtype=np.int64)
    subject_ids = np.asarray(arrays["subject_ids"], dtype=np.int64)
    activity_names = np.asarray(arrays["activity_names"], dtype=object)
    for name, value in (
        ("labels", labels),
        ("labels_1based", labels_1based),
        ("subject_ids", subject_ids),
        ("activity_names", activity_names),
    ):
        _validate_vector(name, value, sample_count)

    expected_labels = np.arange(CLASS_COUNT, dtype=np.int64)
    if not np.array_equal(np.unique(labels), expected_labels):
        raise ValueError(
            "Input labels must contain every original USC-HAD class 0--11 exactly "
            "as the label vocabulary."
        )
    if not np.array_equal(labels_1based, labels + 1):
        raise ValueError("Input labels_1based must equal input labels + 1.")

    input_old_classes = _integer_list(meta.get("old_classes"), "meta.old_classes")
    input_all_classes = _integer_list(meta.get("all_classes"), "meta.all_classes")
    if input_old_classes != list(range(1, OLD_CLASS_COUNT + 1)):
        raise ValueError(
            "The source NPZ must use the canonical original old classes 1--6; "
            f"got {input_old_classes}."
        )
    if input_all_classes != list(range(1, CLASS_COUNT + 1)):
        raise ValueError(
            "The source NPZ must use all original classes 1--12; "
            f"got {input_all_classes}."
        )
    source_subjects = _integer_list(
        meta.get("source_subjects"), "meta.source_subjects"
    )
    if not set(source_subjects).issubset(set(subject_ids.tolist())):
        raise ValueError(
            "meta.source_subjects contains a subject absent from the input NPZ."
        )

    input_mean = np.asarray(arrays["mean"])
    input_std = np.asarray(arrays["std"])
    expected_stat_shape = (1, channel_count, 1)
    if input_mean.shape != expected_stat_shape or input_std.shape != expected_stat_shape:
        raise ValueError(
            "NPZ mean/std must both have shape "
            f"{expected_stat_shape}, got {input_mean.shape}/{input_std.shape}."
        )
    if not np.all(np.isfinite(input_mean)):
        raise ValueError("NPZ mean contains non-finite values.")
    if not np.all(np.isfinite(input_std)) or np.any(input_std <= 0):
        raise ValueError("NPZ std must be finite and strictly positive.")

    activity_lookup = _activity_lookup(meta)
    expected_names = np.asarray(
        [activity_lookup[int(label)] for label in labels_1based], dtype=object
    )
    if not np.array_equal(activity_names.astype(str), expected_names.astype(str)):
        raise ValueError("NPZ activity_names disagrees with meta.json activity_names.")

    source_mask = np.isin(subject_ids, np.asarray(source_subjects, dtype=np.int64))
    original_old_mask = labels < OLD_CLASS_COUNT
    expected_masks = {
        "old_mask": original_old_mask,
        "new_mask": ~original_old_mask,
        "source_mask": source_mask,
        "stat_mask": original_old_mask & source_mask,
        "stage0_labeled_mask": original_old_mask & source_mask,
    }
    for name, expected in expected_masks.items():
        observed = np.asarray(arrays[name])
        _validate_vector(name, observed, sample_count)
        if observed.dtype != np.bool_:
            raise ValueError(f"NPZ {name} must have boolean dtype, got {observed.dtype}.")
        if not np.array_equal(observed, expected):
            raise ValueError(f"Input NPZ {name} is inconsistent with its metadata.")

    return labels, subject_ids, source_subjects, activity_lookup


def transform_arrays(
    arrays: Mapping[str, np.ndarray],
    meta: Mapping[str, Any],
    *,
    eps: float = 1e-6,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Return transformed NPZ arrays plus audit values used by metadata."""
    eps = float(eps)
    if not np.isfinite(eps) or eps <= 0:
        raise ValueError(f"eps must be finite and positive, got {eps}.")

    original_labels, subject_ids, source_subjects, activity_lookup = _validate_input(
        arrays, meta
    )
    original_labels_1based = np.asarray(arrays["labels_1based"], dtype=np.int64)
    original_activity_names = np.asarray(arrays["activity_names"], dtype=object)

    labels = ORIGINAL_TO_NEW[original_labels]
    labels_1based = labels + 1
    old_mask = labels < OLD_CLASS_COUNT
    new_mask = ~old_mask
    source_mask = np.isin(subject_ids, np.asarray(source_subjects, dtype=np.int64))
    stat_mask = old_mask & source_mask
    if not np.any(stat_mask):
        raise ValueError("The swapped old-class/source-subject statistics set is empty.")

    input_windows = np.asarray(arrays["windows"], dtype=np.float32)
    input_mean = np.asarray(arrays["mean"], dtype=np.float32)
    input_std = np.asarray(arrays["std"], dtype=np.float32)
    raw_windows = (
        input_windows.astype(np.float64) * input_std.astype(np.float64)
        + input_mean.astype(np.float64)
    ).astype(np.float32)
    stat_windows = raw_windows[stat_mask]
    mean = stat_windows.mean(
        axis=(0, 2), keepdims=True, dtype=np.float64
    ).astype(np.float32)
    std = stat_windows.std(
        axis=(0, 2), keepdims=True, dtype=np.float64
    ).astype(np.float32)
    std = np.maximum(std, np.float32(eps))
    if not np.all(np.isfinite(mean)):
        raise ValueError("Swapped normalization mean contains non-finite values.")
    if not np.all(np.isfinite(std)) or np.any(std <= 0):
        raise ValueError("Swapped normalization std must be finite and positive.")
    windows = ((raw_windows - mean) / std).astype(np.float32)
    if not np.all(np.isfinite(windows)):
        raise ValueError("Swapped normalized windows contains non-finite values.")

    output = {name: np.array(value, copy=True) for name, value in arrays.items()}
    output.update(
        {
            "windows": windows,
            "labels": labels.astype(np.int64),
            "labels_1based": labels_1based.astype(np.int64),
            "activity_names": original_activity_names.copy(),
            "mean": mean,
            "std": std,
            "stat_mask": stat_mask.astype(bool),
            "old_mask": old_mask.astype(bool),
            "new_mask": new_mask.astype(bool),
            "source_mask": source_mask.astype(bool),
            "stage0_labeled_mask": stat_mask.astype(bool),
            "stage1_unlabeled_mask": np.ones(len(labels), dtype=bool),
            "original_labels": original_labels.astype(np.int64),
            "original_labels_1based": original_labels_1based.astype(np.int64),
            "original_activity_names": original_activity_names.copy(),
            "class_permutation": CLASS_PERMUTATION.copy(),
        }
    )

    target_subjects_raw = meta.get("target_subjects")
    if target_subjects_raw is not None:
        target_subjects = _integer_list(
            target_subjects_raw, "meta.target_subjects"
        )
        output["target_mask"] = np.isin(
            subject_ids, np.asarray(target_subjects, dtype=np.int64)
        )

    class_mapping = []
    for new_label, original_label in enumerate(CLASS_PERMUTATION.tolist()):
        class_mapping.append(
            {
                "new_label_0based": int(new_label),
                "new_label_1based": int(new_label + 1),
                "original_label_0based": int(original_label),
                "original_label_1based": int(original_label + 1),
                "activity_name": activity_lookup[int(original_label + 1)],
                "new_role": "old" if new_label < OLD_CLASS_COUNT else "novel",
            }
        )

    audit = {
        "source_subjects": source_subjects,
        "stat_window_count": int(np.sum(stat_mask)),
        "mean": mean.reshape(-1).astype(float).tolist(),
        "std": std.reshape(-1).astype(float).tolist(),
        "class_mapping": class_mapping,
        "activity_lookup": activity_lookup,
    }
    return output, audit


def build_output_meta(
    input_meta: Mapping[str, Any],
    audit: Mapping[str, Any],
    *,
    input_npz_path: Path,
    input_meta_path: Path,
    input_npz_sha256: str,
    input_meta_sha256: str,
    eps: float,
) -> dict[str, Any]:
    meta = copy.deepcopy(dict(input_meta))
    original_activity_names = {
        str(label): str(name)
        for label, name in sorted(audit["activity_lookup"].items())
    }
    remapped_activity_names = {
        str(item["new_label_1based"]): item["activity_name"]
        for item in audit["class_mapping"]
    }
    meta.update(
        {
            "diagnostic_only": True,
            "analysis_role": "diagnostic_only_old_new_semantic_swap",
            "diagnostic_scope": (
                "Sensitivity analysis for reversing the six old and six novel "
                "USC-HAD semantic classes; not evidence by itself for online "
                "codebook adaptation and not a deployment dataset."
            ),
            "old_classes": list(range(1, OLD_CLASS_COUNT + 1)),
            "all_classes": list(range(1, CLASS_COUNT + 1)),
            "activity_names": remapped_activity_names,
            "original_activity_names": original_activity_names,
            "class_permutation": CLASS_PERMUTATION.tolist(),
            "class_permutation_semantics": (
                "index is the new 0-based label; value is the original 0-based label"
            ),
            "inverse_class_permutation": ORIGINAL_TO_NEW.tolist(),
            "class_mapping": copy.deepcopy(audit["class_mapping"]),
            "semantic_split": {
                "new_old_labels_0based": list(range(OLD_CLASS_COUNT)),
                "new_novel_labels_0based": list(
                    range(OLD_CLASS_COUNT, CLASS_COUNT)
                ),
                "new_old_original_labels_0based": list(
                    range(OLD_CLASS_COUNT, CLASS_COUNT)
                ),
                "new_novel_original_labels_0based": list(range(OLD_CLASS_COUNT)),
            },
            "label_format": {
                "labels": "permuted contiguous 0-based labels used for training",
                "labels_1based": "permuted contiguous 1-based labels",
                "original_labels": "original USC-HAD contiguous 0-based labels",
                "original_labels_1based": "original USC-HAD activity ids",
            },
            "normalization": {
                "type": "per-channel z-score",
                "reconstruction_source": (
                    "input windows * input std + input mean"
                ),
                "stats_from": "swapped old classes and source subjects only",
                "stat_subjects": list(audit["source_subjects"]),
                "stat_new_labels_0based": list(range(OLD_CLASS_COUNT)),
                "stat_original_labels_0based": list(
                    range(OLD_CLASS_COUNT, CLASS_COUNT)
                ),
                "stat_window_count": int(audit["stat_window_count"]),
                "mean_shape": [1, len(audit["mean"]), 1],
                "std_shape": [1, len(audit["std"]), 1],
                "mean": list(audit["mean"]),
                "std": list(audit["std"]),
                "eps": float(eps),
            },
            "masks": {
                "old_mask": "permuted labels in 0..5",
                "new_mask": "permuted labels in 6..11",
                "source_mask": "subject_ids in meta.source_subjects",
                "target_mask": (
                    "subject_ids in meta.target_subjects, if target_subjects is set"
                ),
                "stat_mask": "old_mask & source_mask",
                "stage0_labeled_mask": "old_mask & source_mask",
                "stage1_unlabeled_mask": "all twelve permuted classes",
            },
            "input_artifacts": {
                "npz_path": str(input_npz_path.resolve()),
                "npz_sha256": input_npz_sha256,
                "meta_path": str(input_meta_path.resolve()),
                "meta_sha256": input_meta_sha256,
            },
        }
    )
    return meta


def create_swapped_dataset(
    input_npz_path: Path,
    input_meta_path: Path,
    output_dir: Path,
    *,
    eps: float = 1e-6,
) -> tuple[Path, Path]:
    input_npz_path = Path(input_npz_path)
    input_meta_path = Path(input_meta_path)
    output_dir = Path(output_dir)
    if not input_npz_path.is_file():
        raise FileNotFoundError(f"Input NPZ does not exist: {input_npz_path}")
    if not input_meta_path.is_file():
        raise FileNotFoundError(f"Input meta.json does not exist: {input_meta_path}")
    if output_dir.exists():
        raise FileExistsError(f"Output directory already exists: {output_dir}")

    input_npz_sha256 = sha256_file(input_npz_path)
    input_meta_sha256 = sha256_file(input_meta_path)
    with input_meta_path.open(encoding="utf-8") as handle:
        input_meta = json.load(handle)
    if not isinstance(input_meta, Mapping):
        raise ValueError("Input meta.json must contain a JSON object.")
    with np.load(input_npz_path, allow_pickle=True) as archive:
        arrays = {name: np.array(archive[name], copy=True) for name in archive.files}

    output_arrays, audit = transform_arrays(arrays, input_meta, eps=eps)
    output_meta = build_output_meta(
        input_meta,
        audit,
        input_npz_path=input_npz_path,
        input_meta_path=input_meta_path,
        input_npz_sha256=input_npz_sha256,
        input_meta_sha256=input_meta_sha256,
        eps=eps,
    )

    output_dir.mkdir(parents=True, exist_ok=False)
    output_npz_path = output_dir / "uschad_windows.npz"
    output_meta_path = output_dir / "meta.json"
    np.savez_compressed(output_npz_path, **output_arrays)
    output_meta["output_npz_sha256"] = sha256_file(output_npz_path)
    output_meta_path.write_text(
        json.dumps(output_meta, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return output_npz_path, output_meta_path


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create a diagnostic USC-HAD NPZ whose original classes 6--11 are "
            "old and original classes 0--5 are novel."
        )
    )
    parser.add_argument("--input-npz", type=Path, required=True)
    parser.add_argument("--input-meta", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--eps", type=float, default=1e-6)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    output_npz, output_meta = create_swapped_dataset(
        args.input_npz,
        args.input_meta,
        args.output_dir,
        eps=args.eps,
    )
    print(f"Wrote swapped USC-HAD NPZ: {output_npz}")
    print(f"Wrote audit metadata: {output_meta}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
