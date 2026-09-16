import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from experiments.motion_primitive.swap_uschad_old_new import (
    CLASS_PERMUTATION,
    ORIGINAL_TO_NEW,
    create_swapped_dataset,
    sha256_file,
    transform_arrays,
)


CLASS_NAMES = {
    index: name
    for index, name in enumerate(
        (
            "Walking Forward",
            "Walking Left",
            "Walking Right",
            "Walking Upstairs",
            "Walking Downstairs",
            "Running Forward",
            "Jumping Up",
            "Sitting",
            "Standing",
            "Sleeping",
            "Elevator Up",
            "Elevator Down",
        ),
        start=1,
    )
}


def build_fixture(root: Path) -> tuple[Path, Path, np.ndarray]:
    labels = np.tile(np.arange(12, dtype=np.int64), 3)
    subjects = np.repeat(np.asarray([1, 2, 3], dtype=np.int64), 12)
    rows = []
    for row, (label, subject) in enumerate(zip(labels, subjects)):
        time = np.arange(5, dtype=np.float32)
        rows.append(
            np.stack(
                (
                    label * 2.0 + subject * 0.2 + time * 0.1,
                    label * -0.5 + subject * 0.3 + time * 0.25 + row * 0.01,
                )
            )
        )
    raw = np.asarray(rows, dtype=np.float32)

    source_mask = np.isin(subjects, [1, 2])
    old_mask = labels < 6
    stat_mask = source_mask & old_mask
    mean = raw[stat_mask].mean(
        axis=(0, 2), keepdims=True, dtype=np.float64
    ).astype(np.float32)
    std = raw[stat_mask].std(
        axis=(0, 2), keepdims=True, dtype=np.float64
    ).astype(np.float32)
    windows = ((raw - mean) / std).astype(np.float32)
    activity_names = np.asarray(
        [CLASS_NAMES[int(label + 1)] for label in labels], dtype=object
    )

    input_npz = root / "input.npz"
    np.savez_compressed(
        input_npz,
        windows=windows,
        labels=labels,
        labels_1based=labels + 1,
        subject_ids=subjects,
        trial_numbers=np.arange(len(labels), dtype=np.int64) % 5 + 1,
        trial_global_ids=np.arange(len(labels), dtype=np.int64),
        window_indices=np.zeros(len(labels), dtype=np.int64),
        window_start_indices=np.zeros(len(labels), dtype=np.int64),
        file_paths=np.asarray(
            [f"subject-{subject}/class-{label}.mat" for subject, label in zip(subjects, labels)],
            dtype=object,
        ),
        activity_names=activity_names,
        channel_names=np.asarray(["acc_x", "acc_y"], dtype=object),
        mean=mean,
        std=std,
        stat_mask=stat_mask,
        old_mask=old_mask,
        new_mask=~old_mask,
        source_mask=source_mask,
        target_mask=subjects == 3,
        stage0_labeled_mask=stat_mask,
        stage1_unlabeled_mask=np.ones(len(labels), dtype=bool),
    )
    input_meta = root / "meta.json"
    input_meta.write_text(
        json.dumps(
            {
                "dataset": "USC-HAD",
                "root": "/dataset/USC-HAD",
                "num_windows": len(labels),
                "window_shape": list(windows.shape),
                "window_size": windows.shape[-1],
                "stride": 2,
                "channels": ["acc_x", "acc_y"],
                "num_channels": 2,
                "old_classes": list(range(1, 7)),
                "all_classes": list(range(1, 13)),
                "source_subjects": [1, 2],
                "target_subjects": [3],
                "activity_names": {
                    str(label): name for label, name in CLASS_NAMES.items()
                },
                "label_format": {
                    "labels": "0-based",
                    "labels_1based": "original activity id",
                },
                "normalization": {
                    "type": "per-channel z-score",
                    "stats_from": "old classes and source subjects",
                },
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return input_npz, input_meta, raw


class SwappedUSCHADTests(unittest.TestCase):
    def test_semantic_swap_relabels_and_recomputes_normalization(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            input_npz, input_meta, _ = build_fixture(root)
            output_npz, _ = create_swapped_dataset(
                input_npz, input_meta, root / "output"
            )

            with np.load(input_npz, allow_pickle=True) as source:
                original_labels = np.asarray(source["labels"], dtype=np.int64)
                original_windows = np.asarray(source["windows"], dtype=np.float32)
                original_mean = np.asarray(source["mean"], dtype=np.float32)
                original_std = np.asarray(source["std"], dtype=np.float32)
                recovered_raw = (
                    original_windows.astype(np.float64)
                    * original_std.astype(np.float64)
                    + original_mean.astype(np.float64)
                ).astype(np.float32)

            with np.load(output_npz, allow_pickle=True) as swapped:
                expected_labels = ORIGINAL_TO_NEW[original_labels]
                self.assertTrue(np.array_equal(swapped["labels"], expected_labels))
                self.assertTrue(
                    np.array_equal(swapped["labels_1based"], expected_labels + 1)
                )
                self.assertTrue(
                    np.array_equal(swapped["original_labels"], original_labels)
                )
                self.assertTrue(
                    np.array_equal(
                        swapped["original_labels_1based"], original_labels + 1
                    )
                )
                self.assertTrue(
                    np.array_equal(swapped["class_permutation"], CLASS_PERMUTATION)
                )
                self.assertTrue(
                    np.array_equal(
                        swapped["original_activity_names"], swapped["activity_names"]
                    )
                )

                expected_old = original_labels >= 6
                expected_source = np.isin(swapped["subject_ids"], [1, 2])
                expected_stat = expected_old & expected_source
                self.assertTrue(np.array_equal(swapped["old_mask"], expected_old))
                self.assertTrue(np.array_equal(swapped["new_mask"], ~expected_old))
                self.assertTrue(
                    np.array_equal(swapped["source_mask"], expected_source)
                )
                self.assertTrue(
                    np.array_equal(swapped["stat_mask"], expected_stat)
                )
                self.assertTrue(
                    np.array_equal(swapped["stage0_labeled_mask"], expected_stat)
                )
                self.assertTrue(np.all(swapped["stage1_unlabeled_mask"]))

                expected_mean = recovered_raw[expected_stat].mean(
                    axis=(0, 2), keepdims=True, dtype=np.float64
                ).astype(np.float32)
                expected_std = recovered_raw[expected_stat].std(
                    axis=(0, 2), keepdims=True, dtype=np.float64
                ).astype(np.float32)
                self.assertTrue(
                    np.allclose(swapped["mean"], expected_mean, atol=1e-7)
                )
                self.assertTrue(
                    np.allclose(swapped["std"], expected_std, atol=1e-7)
                )
                round_trip = (
                    swapped["windows"] * swapped["std"] + swapped["mean"]
                )
                self.assertTrue(np.allclose(round_trip, recovered_raw, atol=2e-6))
                self.assertTrue(
                    np.allclose(
                        swapped["windows"][expected_stat].mean(
                            axis=(0, 2), dtype=np.float64
                        ),
                        0.0,
                        atol=1e-6,
                    )
                )
                self.assertTrue(
                    np.allclose(
                        swapped["windows"][expected_stat].std(
                            axis=(0, 2), dtype=np.float64
                        ),
                        1.0,
                        atol=1e-6,
                    )
                )

    def test_metadata_records_mapping_statistics_and_input_hashes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            input_npz, input_meta, _ = build_fixture(root)
            input_npz_hash = sha256_file(input_npz)
            input_meta_hash = sha256_file(input_meta)
            output_npz, output_meta = create_swapped_dataset(
                input_npz, input_meta, root / "output"
            )
            meta = json.loads(output_meta.read_text(encoding="utf-8"))

            self.assertTrue(meta["diagnostic_only"])
            self.assertEqual(
                meta["analysis_role"], "diagnostic_only_old_new_semantic_swap"
            )
            self.assertEqual(meta["class_permutation"], CLASS_PERMUTATION.tolist())
            self.assertEqual(meta["inverse_class_permutation"], ORIGINAL_TO_NEW.tolist())
            self.assertEqual(meta["activity_names"]["1"], "Jumping Up")
            self.assertEqual(meta["activity_names"]["2"], "Sitting")
            self.assertEqual(meta["activity_names"]["3"], "Standing")
            self.assertEqual(meta["activity_names"]["7"], "Walking Forward")
            self.assertEqual(meta["original_activity_names"]["1"], "Walking Forward")
            self.assertEqual(
                meta["normalization"]["stat_original_labels_0based"],
                list(range(6, 12)),
            )
            self.assertEqual(meta["normalization"]["stat_subjects"], [1, 2])
            self.assertEqual(meta["normalization"]["stat_window_count"], 12)
            self.assertEqual(meta["input_artifacts"]["npz_sha256"], input_npz_hash)
            self.assertEqual(meta["input_artifacts"]["meta_sha256"], input_meta_hash)
            self.assertEqual(meta["output_npz_sha256"], sha256_file(output_npz))
            self.assertEqual(
                [item["new_role"] for item in meta["class_mapping"]],
                ["old"] * 6 + ["novel"] * 6,
            )

    def test_inconsistent_input_labels_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            input_npz, input_meta, _ = build_fixture(root)
            with np.load(input_npz, allow_pickle=True) as archive:
                arrays = {
                    name: np.array(archive[name], copy=True) for name in archive.files
                }
            arrays["labels_1based"][0] = 12
            meta = json.loads(input_meta.read_text(encoding="utf-8"))
            with self.assertRaisesRegex(ValueError, "labels_1based"):
                transform_arrays(arrays, meta)

    def test_reapplication_and_existing_output_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            input_npz, input_meta, _ = build_fixture(root)
            output_dir = root / "output"
            output_npz, output_meta = create_swapped_dataset(
                input_npz, input_meta, output_dir
            )
            with self.assertRaisesRegex(FileExistsError, "already exists"):
                create_swapped_dataset(input_npz, input_meta, output_dir)

            with np.load(output_npz, allow_pickle=True) as archive:
                swapped_arrays = {
                    name: np.array(archive[name], copy=True) for name in archive.files
                }
            swapped_meta = json.loads(output_meta.read_text(encoding="utf-8"))
            with self.assertRaisesRegex(ValueError, "swap provenance"):
                transform_arrays(swapped_arrays, swapped_meta)


if __name__ == "__main__":
    unittest.main()
