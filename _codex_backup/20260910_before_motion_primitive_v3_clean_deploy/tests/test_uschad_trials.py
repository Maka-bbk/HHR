from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from scipy.io import savemat

import data.uschad_trials as trial_module
from data.uschad_trials import (
    DEFAULT_USCHAD_ROOT,
    IGNORE_INDEX,
    SubjectSplit,
    USCHADProtocolError,
    build_uschad_fold_datasets,
    discover_trial_manifest,
    fixed_subject_manifest,
    pad_trial_batch,
    resolve_activity_split,
    stratified_train_label_assignment,
    subject_split_for_fold,
    validate_subject_split,
)


def _write_trial(
    root: Path,
    *,
    subject: int,
    activity: int,
    trial: int,
    length: int,
    offset: float,
    internal_subject: int | None = None,
    internal_activity: int | None = None,
    internal_trial: int | None = None,
) -> Path:
    directory = root / f"Subject{subject}"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"a{activity}t{trial}.mat"
    time = np.arange(length, dtype=np.float32)[:, None]
    channels = np.arange(6, dtype=np.float32)[None, :]
    sensor = offset + time + channels * np.float32(0.25)
    payload = {
        "subject": str(subject if internal_subject is None else internal_subject),
        "activity_number": str(
            activity if internal_activity is None else internal_activity
        ),
        "activity": f"fixture-activity-{activity}",
        "trial": str(trial if internal_trial is None else internal_trial),
        "sensor_readings": sensor,
    }
    savemat(path, payload)
    return path


def _write_fold1_fixture(root: Path) -> None:
    # Fold 1: subject 1 is train, subject 2 is validation, subject 10 is test.
    # All five train stratum trials are present so the required 4/1 assignment
    # remains testable even with require_complete=False.
    for trial in range(1, 6):
        _write_trial(
            root,
            subject=1,
            activity=1,
            trial=trial,
            length=2 + trial,
            offset=float(trial),
        )
    _write_trial(
        root, subject=2, activity=1, trial=1, length=5, offset=1000.0
    )
    _write_trial(
        root, subject=10, activity=1, trial=1, length=6, offset=2000.0
    )
    _write_trial(
        root, subject=10, activity=7, trial=1, length=4, offset=3000.0
    )


class SubjectAndClassProtocolTests(unittest.TestCase):
    def test_fixed_manifest_is_balanced_and_subject_disjoint(self) -> None:
        manifest = fixed_subject_manifest()
        self.assertEqual(len(manifest["folds"]), 7)
        self.assertEqual(
            set(manifest["coverage"]["train_folds_per_subject"].values()), {5}
        )
        self.assertEqual(
            set(manifest["coverage"]["validation_folds_per_subject"].values()),
            {1},
        )
        self.assertEqual(
            set(manifest["coverage"]["test_folds_per_subject"].values()), {1}
        )
        fold1 = subject_split_for_fold(1)
        self.assertEqual(fold1.validation_subjects, (2, 13))
        self.assertEqual(fold1.test_subjects, (10, 11))
        self.assertFalse(
            set(fold1.train_subjects)
            & set(fold1.validation_subjects)
            & set(fold1.test_subjects)
        )

    def test_overlap_and_noncanonical_partition_fail_closed(self) -> None:
        leaking = SubjectSplit(
            fold=1,
            train_subjects=(1, 2, 3, 4, 5, 6, 7, 8, 9, 12),
            validation_subjects=(2, 13),
            test_subjects=(10, 11),
        )
        with self.assertRaisesRegex(USCHADProtocolError, "leakage"):
            validate_subject_split(leaking)

        canonical = subject_split_for_fold(1)
        drifted = SubjectSplit(
            fold=1,
            train_subjects=canonical.train_subjects,
            validation_subjects=canonical.test_subjects,
            test_subjects=canonical.validation_subjects,
        )
        with self.assertRaisesRegex(USCHADProtocolError, "fixed seven-fold"):
            validate_subject_split(drifted)

    def test_old_new_partition_supports_default_and_swap(self) -> None:
        old, new = resolve_activity_split()
        self.assertEqual(old, tuple(range(1, 7)))
        self.assertEqual(new, tuple(range(7, 13)))
        swapped_old, swapped_new = resolve_activity_split(range(7, 13))
        self.assertEqual(swapped_old, tuple(range(7, 13)))
        self.assertEqual(swapped_new, tuple(range(1, 7)))
        with self.assertRaisesRegex(ValueError, "overlap"):
            resolve_activity_split((1, 2), (2, *range(3, 13)))


class RawManifestTests(unittest.TestCase):
    def test_known_label_anomaly_is_reported_or_excluded_never_relabelled(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _write_trial(
                root,
                subject=14,
                activity=3,
                trial=1,
                length=4,
                offset=0.0,
            )
            _write_trial(
                root,
                subject=14,
                activity=3,
                trial=2,
                length=5,
                offset=1.0,
                internal_activity=2,
            )
            reported, audit = discover_trial_manifest(
                root, require_complete=False, anomaly_policy="report"
            )
            anomaly = next(item for item in reported if item.trial_number == 2)
            self.assertEqual(anomaly.physical_activity_id, 3)
            self.assertEqual(anomaly.internal_activity_id, 2)
            self.assertTrue(anomaly.known_label_anomaly)
            self.assertIn("internal_activity_number=2", anomaly.metadata_issues[0])
            self.assertFalse(
                audit["known_label_anomaly"]["silently_relabelled"]
            )

            excluded, excluded_audit = discover_trial_manifest(
                root, require_complete=False, anomaly_policy="exclude"
            )
            self.assertEqual([item.trial_number for item in excluded], [1])
            self.assertFalse(excluded_audit["known_label_anomaly"]["retained"])

    @unittest.skipUnless(
        DEFAULT_USCHAD_ROOT.is_dir(), "raw D:/WorkDir USC-HAD is unavailable"
    )
    def test_real_raw_inventory_is_complete_and_metadata_only_discovery(self) -> None:
        descriptors, audit = discover_trial_manifest(DEFAULT_USCHAD_ROOT)
        self.assertEqual(len(descriptors), 840)
        self.assertEqual(audit["on_disk_mat_count"], 840)
        self.assertTrue(audit["complete_grid_observed"])
        self.assertEqual(audit["sensor_arrays_loaded_during_discovery"], 0)
        issues = {
            row["trial_key"]: row for row in audit["metadata_issue_records"]
        }
        self.assertEqual(
            set(issues),
            {"s03_a08_t05", "s05_a08_t03", "s13_a11_t04", "s14_a03_t02"},
        )
        anomaly = next(
            item for item in descriptors if item.trial_key == "s14_a03_t02"
        )
        self.assertEqual(anomaly.physical_activity_id, 3)
        self.assertEqual(anomaly.internal_activity_id, 2)


class FoldDatasetAndCollateTests(unittest.TestCase):
    def test_stratified_assignment_is_seeded_four_labelled_one_unlabelled(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for trial in range(1, 6):
                _write_trial(
                    root,
                    subject=1,
                    activity=1,
                    trial=trial,
                    length=trial + 1,
                    offset=float(trial),
                )
            descriptors, _ = discover_trial_manifest(root, require_complete=False)
            first = stratified_train_label_assignment(descriptors, seed=17)
            repeated = stratified_train_label_assignment(
                list(reversed(descriptors)), seed=17
            )
            self.assertEqual(len(first.labelled_trial_ids), 4)
            self.assertEqual(len(first.unlabelled_trial_ids), 1)
            self.assertEqual(first.labelled_trial_ids, repeated.labelled_trial_ids)
            variants = {
                stratified_train_label_assignment(descriptors, seed=seed).labelled_trial_ids
                for seed in range(10)
            }
            self.assertGreater(len(variants), 1)

    def test_fold_builder_fits_train_only_and_masks_j0_t_labels(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _write_fold1_fixture(root)
            original_loader = trial_module._load_sensor_readings
            with patch.object(
                trial_module, "_load_sensor_readings", wraps=original_loader
            ) as sensor_loader:
                bundle = build_uschad_fold_datasets(
                    root,
                    fold=1,
                    label_regime="J0-T",
                    label_seed=23,
                    require_complete=False,
                )
            loaded_paths = [Path(call.args[0]) for call in sensor_loader.call_args_list]
            self.assertEqual(len(loaded_paths), 5)
            self.assertEqual({path.parent.name for path in loaded_paths}, {"Subject1"})
            self.assertEqual(bundle.normalization.trial_count, 5)
            self.assertEqual(bundle.protocol_audit["trial_counts"]["train_old"], 5)
            self.assertEqual(
                bundle.protocol_audit["trial_counts"]["train_old_labelled_assignment"],
                4,
            )
            self.assertEqual(
                bundle.protocol_audit["trial_counts"]["train_old_unlabelled_assignment"],
                1,
            )
            self.assertEqual(
                set(bundle.train.descriptors[index].subject_id for index in range(5)),
                {1},
            )
            self.assertEqual(
                {item.subject_id for item in bundle.validation.descriptors}, {2}
            )
            self.assertEqual(
                {item.subject_id for item in bundle.outer_test.descriptors}, {10}
            )
            self.assertTrue(bundle.outer_test.outer_test_locked)
            with self.assertRaisesRegex(USCHADProtocolError, "locked"):
                _ = bundle.outer_test[0]
            self.assertEqual(bundle.outer_test.sensor_trials_loaded, 0)

            train_items = [bundle.train[index] for index in range(len(bundle.train))]
            self.assertEqual(sum(item["trajectory_label_mask"] for item in train_items), 4)
            hidden = next(item for item in train_items if not item["label_visible"])
            self.assertEqual(hidden["supervision_target"], IGNORE_INDEX)
            validation_item = bundle.validation[0]
            self.assertTrue(validation_item["label_visible"])

            bundle.outer_test.unlock_outer_test_for_evaluation("selected-checkpoint-sha256")
            test_items = [bundle.outer_test[index] for index in range(2)]
            self.assertTrue(all(not item["label_visible"] for item in test_items))
            self.assertEqual({item["is_old_class"] for item in test_items}, {False, True})

    def test_j0_u_exposes_no_supervision_and_collate_preserves_full_lengths(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _write_fold1_fixture(root)
            bundle = build_uschad_fold_datasets(
                root,
                fold=1,
                label_regime="J0-U",
                label_seed=31,
                require_complete=False,
            )
            items = [bundle.train[0], bundle.train[4]]
            self.assertTrue(all(not item["label_visible"] for item in items))
            self.assertTrue(
                all(item["supervision_target"] == IGNORE_INDEX for item in items)
            )
            collated = pad_trial_batch(items)
            self.assertEqual(
                tuple(collated["physical_trials"].shape),
                (2, 6, max(item["length"] for item in items)),
            )
            self.assertEqual(collated["lengths"].tolist(), [3, 7])
            self.assertEqual(collated["sample_mask"].sum(dim=1).tolist(), [3, 7])
            self.assertTrue(
                torch.all(
                    collated["supervision_targets"]
                    == torch.full((2,), IGNORE_INDEX, dtype=torch.long)
                )
            )
            self.assertFalse(collated["trajectory_label_mask"].any())
            # The shorter physical trial is right-padded rather than cropped or
            # repeated; every original sample remains under the validity mask.
            self.assertTrue(
                torch.all(collated["physical_trials"][0, :, 3:] == 0.0)
            )


if __name__ == "__main__":
    unittest.main()
