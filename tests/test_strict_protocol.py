from __future__ import annotations

import dataclasses
import os
import unittest
from pathlib import Path

import numpy as np

from experiments.motion_primitive.frozen_e0_state import PrimitiveTrial, WindowTrial
from experiments.motion_primitive.strict_protocol import (
    RegisteredProtocol,
    SensorTrial,
    SessionStream,
    SubjectSplit,
    build_registered_protocol,
    registered_subject_split,
)


FORBIDDEN_LEARNER_FIELDS = {
    "label",
    "labels",
    "activity",
    "activity_id",
    "activity_label",
    "activity_name",
    "class_id",
    "target",
    "targets",
    "y",
}


def _real_w256_npz() -> Path | None:
    configured = os.environ.get("HHR_USCHAD_NPZ")
    project_root = Path(__file__).resolve().parents[1]
    candidates = [
        Path(configured).expanduser() if configured else None,
        project_root / "processed/uschad_w256_s128_train17stats/uschad_windows.npz",
        Path("D:/WorkDir/HHR/processed/uschad_w256_s128_train17stats/uschad_windows.npz"),
        Path("/mnt/d/WorkDir/HHR/processed/uschad_w256_s128_train17stats/uschad_windows.npz"),
        Path("C:/Users/BJ/Desktop/HCGCD/HHR/processed/uschad_w256_s128_train17stats/uschad_windows.npz"),
        Path("/mnt/c/Users/BJ/Desktop/HCGCD/HHR/processed/uschad_w256_s128_train17stats/uschad_windows.npz"),
    ]
    for candidate in candidates:
        if candidate is not None and candidate.is_file():
            return candidate.resolve()
    return None


class LabelIsolationContractTests(unittest.TestCase):
    def test_every_learner_facing_dataclass_is_label_free(self) -> None:
        expected_sensor_fields = {
            "trial_id", "subject_id", "windows", "raw_windows", "window_starts"
        }
        expected_session_fields = {
            "session", "incoming", "evaluation", "maximum_new_classes",
            "incoming_trial_ids_sha256", "evaluation_trial_ids_sha256",
        }
        self.assertEqual({field.name for field in dataclasses.fields(SensorTrial)}, expected_sensor_fields)
        self.assertEqual({field.name for field in dataclasses.fields(SessionStream)}, expected_session_fields)

        for cls in (SensorTrial, SessionStream, WindowTrial, PrimitiveTrial):
            fields = {field.name.lower() for field in dataclasses.fields(cls)}
            self.assertFalse(
                fields & FORBIDDEN_LEARNER_FIELDS,
                f"{cls.__name__} leaks scorer-only identity fields: "
                f"{sorted(fields & FORBIDDEN_LEARNER_FIELDS)}",
            )

    def test_registered_subject_splits_are_complete_disjoint_10_2_2(self) -> None:
        observed_test_pairs = []
        for fold in range(1, 8):
            split = registered_subject_split(fold)
            self.assertIsInstance(split, SubjectSplit)
            self.assertEqual((len(split.train), len(split.validation), len(split.outer_test)), (10, 2, 2))
            groups = (set(split.train), set(split.validation), set(split.outer_test))
            self.assertFalse(groups[0] & groups[1])
            self.assertFalse(groups[0] & groups[2])
            self.assertFalse(groups[1] & groups[2])
            self.assertEqual(set.union(*groups), set(range(1, 15)))
            observed_test_pairs.append(tuple(split.outer_test))
        self.assertEqual(len(set(observed_test_pairs)), 7)
        self.assertEqual(set().union(*(set(pair) for pair in observed_test_pairs)), set(range(1, 15)))


class RealUSCHADProtocolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.npz_path = _real_w256_npz()
        if cls.npz_path is None:
            raise unittest.SkipTest(
                "No real w256/s128 USC-HAD NPZ was found; set HHR_USCHAD_NPZ to enable this blocker."
            )
        cls.protocol = build_registered_protocol(
            cls.npz_path,
            fold=1,
            seed=0,
            window_size=256,
            stride=128,
            shuffle_novel_classes=False,
        )

    def test_exact_offline_and_three_session_trial_counts(self) -> None:
        protocol = self.protocol
        self.assertIsInstance(protocol, RegisteredProtocol)
        self.assertEqual(
            (
                len(protocol.offline_train),
                len(protocol.offline_validation),
                len(protocol.offline_outer_test),
            ),
            (300, 60, 60),
        )
        self.assertEqual([len(session.incoming) for session in protocol.sessions], [22, 26, 30])
        self.assertEqual([len(session.evaluation) for session in protocol.sessions], [58, 52, 42])

    def test_online_stream_has_no_reuse_or_train_evaluation_leakage(self) -> None:
        cumulative_incoming: set[int] = set()
        for expected_session, session in enumerate(self.protocol.sessions, start=1):
            self.assertEqual(session.session, expected_session)
            incoming = {trial.trial_id for trial in session.incoming}
            evaluation = {trial.trial_id for trial in session.evaluation}
            self.assertEqual(len(incoming), len(session.incoming))
            self.assertFalse(incoming & cumulative_incoming)
            cumulative_incoming.update(incoming)
            self.assertFalse(evaluation & cumulative_incoming)

        offline_fit_ids = {trial.trial_id for trial in self.protocol.offline_train}
        offline_validation_ids = {trial.trial_id for trial in self.protocol.offline_validation}
        self.assertFalse(offline_fit_ids & offline_validation_ids)
        self.assertFalse(offline_fit_ids & cumulative_incoming)

    def test_visible_class_growth_is_available_only_through_truth_store(self) -> None:
        for session_index, session in enumerate(self.protocol.sessions, start=1):
            for trial in session.incoming + session.evaluation:
                self.assertFalse(hasattr(trial, "label"))
                self.assertFalse(hasattr(trial, "activity_name"))
            labels, names, subjects = self.protocol.truth.join(
                [trial.trial_id for trial in session.evaluation]
            )
            self.assertEqual(set(labels.tolist()), set(range(6 + 2 * session_index)))
            self.assertEqual(len(labels), len(session.evaluation))
            self.assertEqual(len(names), len(session.evaluation))
            self.assertEqual(subjects.shape, labels.shape)
            self.assertEqual(set(subjects.tolist()), set(self.protocol.split.outer_test))


if __name__ == "__main__":
    unittest.main()
