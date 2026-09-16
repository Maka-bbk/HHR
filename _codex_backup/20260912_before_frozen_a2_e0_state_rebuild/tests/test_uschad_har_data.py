"""Current NPZ-to-trial data-path contracts for motion-primitive HHR."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from data.uschad_har import (
    USCHADTrialDataset,
    USCHADWindowDataset,
    convert_window_dataset_to_trials,
    recompute_subject_train_normalization,
    uschad_trial_collate,
)


def _write_npz(path: Path) -> None:
    # Deliberately interleave and reverse windows inside trials.  The trial
    # wrapper must restore time order from window_start_indices, never file row
    # order.
    trial_ids = np.asarray([200, 100, 200, 101, 100, 200], dtype=np.int64)
    window_indices = np.asarray([2, 1, 0, 0, 0, 1], dtype=np.int64)
    starts = window_indices * 4
    subjects = np.asarray([2, 1, 2, 1, 1, 2], dtype=np.int64)
    labels = np.asarray([0, 0, 0, 1, 0, 0], dtype=np.int64)
    trial_numbers = np.asarray([1, 1, 1, 2, 1, 1], dtype=np.int64)

    values = np.asarray([1002.0, 2.0, 1000.0, 100.0, 0.0, 1001.0])
    raw = np.repeat(values[:, None, None], 6, axis=1)
    raw = np.repeat(raw, 8, axis=2).astype(np.float32)
    np.savez(
        path,
        windows=raw,
        windows_raw=raw,
        mean=np.zeros((1, 6, 1), dtype=np.float32),
        std=np.ones((1, 6, 1), dtype=np.float32),
        labels=labels,
        labels_1based=labels + 1,
        subject_ids=subjects,
        trial_numbers=trial_numbers,
        trial_global_ids=trial_ids,
        window_indices=window_indices,
        window_start_indices=starts,
        stat_mask=np.ones(len(raw), dtype=bool),
        channel_names=np.asarray(
            ["acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"],
            dtype=object,
        ),
    )


class USCHADHARDataTests(unittest.TestCase):
    def _source(self, directory: str) -> Path:
        path = Path(directory) / "uschad_windows.npz"
        _write_npz(path)
        return path

    def test_trial_wrapper_restores_order_and_collates_variable_lengths(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            trials = convert_window_dataset_to_trials(
                USCHADWindowDataset(str(self._source(directory))), None
            )
            self.assertIsInstance(trials, USCHADTrialDataset)
            self.assertEqual(trials.trial_global_ids.tolist(), [100, 101, 200])
            first, singleton, last = trials[0], trials[1], trials[2]
            self.assertEqual(first[0]["windows"][:, 0, 0].tolist(), [0.0, 2.0])
            self.assertEqual(last[0]["windows"][:, 0, 0].tolist(), [1000.0, 1001.0, 1002.0])
            batch, labels, trial_ids = uschad_trial_collate([first, last])
            self.assertEqual(tuple(batch["windows"].shape), (2, 3, 6, 8))
            self.assertEqual(batch["lengths"].tolist(), [2, 3])
            self.assertEqual(batch["mask"].sum(dim=1).tolist(), [2, 3])
            self.assertEqual(labels.tolist(), [0, 0])
            self.assertEqual(trial_ids.tolist(), [100, 200])
            self.assertEqual(singleton[0]["positions"].shape, (1, 3))

    def test_partial_trial_selection_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            windows = USCHADWindowDataset(str(self._source(directory)))
            # Row 1 is only one of trial 100's two windows.  Trial-level HHR
            # must never silently reinterpret that fragment as a full trial.
            windows.select_indices([1])
            with self.assertRaisesRegex(RuntimeError, "incomplete"):
                convert_window_dataset_to_trials(windows, None)

    def test_fold_normalization_uses_only_train_subject_old_class(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = self._source(directory)
            training = USCHADWindowDataset(str(source))
            evaluation = USCHADWindowDataset(str(source))
            audit = recompute_subject_train_normalization(
                training,
                evaluation,
                train_subjects=[1],
                train_classes=[0],
            )
            # Only raw values 0 and 2 qualify; subject 2 and class 1 must not
            # influence the fold statistics.
            np.testing.assert_allclose(audit["mean"], 1.0)
            np.testing.assert_allclose(audit["std"], 1.0)
            self.assertEqual(audit["stat_window_count"], 2)
            self.assertEqual(audit["stat_subjects"], [1])
            self.assertEqual(audit["stat_classes"], [0])
            np.testing.assert_array_equal(training.stat_mask, evaluation.stat_mask)
            np.testing.assert_allclose(training.data, evaluation.data)


if __name__ == "__main__":
    unittest.main()
