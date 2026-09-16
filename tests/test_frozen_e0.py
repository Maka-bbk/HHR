"""Numerical regression gates for frozen A2/E0/state trajectory construction."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from experiments.motion_primitive.frozen_e0 import (
    ABSOLUTE_DURATION_DESCRIPTOR_NAMES,
    DURATION_INVARIANT_DESCRIPTOR_PROFILE,
    DURATION_SOFT_SUBJECT_A025_PROFILE,
    DURATION_SOFT_SUBJECT_A050_PROFILE,
    DURATION_SOFT_SUBJECT_A075_PROFILE,
    DescriptorTransform,
    FULL_DEBIASED_DESCRIPTOR_PROFILE,
    GRAVITY_DESCRIPTOR_PROFILE,
    PrimitiveTrajectory,
    SIGNED_VERTICAL_NAMES,
    SUBJECT_DEBIASED_DESCRIPTOR_PROFILE,
    build_e0_trajectories,
    descriptor_matrix,
    fit_descriptor_transform,
    fit_subject_nuisance_projection,
    fit_frozen_e0_codebook,
    load_frozen_artifacts,
    raw_descriptor_dimension,
    reconstruct_trial_channels,
    gravity_aligned_trial_state,
    save_frozen_artifacts,
    statistic_names,
    trajectory_descriptor,
    window_ownership_partition,
    WindowGrid,
)


class FrozenE0Tests(unittest.TestCase):
    def test_historical_descriptor_dimensions_are_exact(self) -> None:
        self.assertEqual(raw_descriptor_dimension(32, include_state=False), 3275)
        self.assertEqual(raw_descriptor_dimension(32, include_state=True), 3739)

    def test_versioned_descriptor_dimensions_are_exact(self) -> None:
        self.assertEqual(
            raw_descriptor_dimension(
                32, include_state=True, descriptor_profile=GRAVITY_DESCRIPTOR_PROFILE
            ),
            3739 + len(SIGNED_VERTICAL_NAMES),
        )
        self.assertEqual(
            raw_descriptor_dimension(
                32,
                include_state=True,
                descriptor_profile=DURATION_INVARIANT_DESCRIPTOR_PROFILE,
            ),
            3701,
        )
        self.assertEqual(
            raw_descriptor_dimension(
                32,
                include_state=True,
                descriptor_profile=FULL_DEBIASED_DESCRIPTOR_PROFILE,
            ),
            3701 + len(SIGNED_VERTICAL_NAMES),
        )
        self.assertEqual(
            raw_descriptor_dimension(
                32,
                include_state=True,
                descriptor_profile=SUBJECT_DEBIASED_DESCRIPTOR_PROFILE,
            ),
            3739,
        )
        for profile in (
            DURATION_SOFT_SUBJECT_A025_PROFILE,
            DURATION_SOFT_SUBJECT_A050_PROFILE,
            DURATION_SOFT_SUBJECT_A075_PROFILE,
        ):
            with self.subTest(profile=profile):
                self.assertEqual(
                    raw_descriptor_dimension(
                        32, include_state=True, descriptor_profile=profile
                    ),
                    3701,
                )

    def test_overlapping_windows_use_nonoverlapping_ownership_duration(self) -> None:
        starts, ends = window_ownership_partition(
            np.asarray([0, 128, 256], dtype=np.int64), 256
        )
        np.testing.assert_array_equal(starts, [0, 192, 320])
        np.testing.assert_array_equal(ends, [192, 320, 512])
        self.assertEqual(int(np.sum(ends - starts)), 512)

    @staticmethod
    def _trajectory(primitive_num: int = 32) -> PrimitiveTrajectory:
        tokens = np.asarray([0, 1, 0], dtype=np.int64)
        statistics = np.zeros((3, 25), dtype=np.float64)
        statistics[:, 0] = np.asarray([1.92, 1.28, 1.92])
        statistics[:, 1:7] = np.asarray([[1] * 6, [2] * 6, [3] * 6])
        statistics[:, 13:19] = np.asarray([[4] * 6, [5] * 6, [6] * 6])
        return PrimitiveTrajectory(
            trial_id=10,
            subject_id=2,
            starts=np.asarray([0, 192, 320], dtype=np.int64),
            ends=np.asarray([192, 320, 512], dtype=np.int64),
            tokens=tokens,
            distances=np.asarray([0.1, 0.2, 0.3], dtype=np.float32),
            embeddings=np.eye(3, 4, dtype=np.float32),
            statistics=statistics,
            statistic_names=statistic_names(),
        ).validate(primitive_num)

    def test_descriptor_preserves_every_e0_window_and_full_state(self) -> None:
        vector, names = trajectory_descriptor(
            self._trajectory(), 32, include_state=True
        )
        self.assertEqual(vector.shape, (3739,))
        self.assertEqual(len(names), 3739)
        np.testing.assert_allclose(vector[:32].sum(), 1.0)
        np.testing.assert_allclose(vector[32:64].sum(), 1.0)
        self.assertEqual(int(np.count_nonzero(vector[-32:])), 2)
        self.assertIn("transition_any__32", names)
        self.assertIn("state_by_child_token__0", names)

    def test_e0_fit_is_trial_equal_and_roundtrips(self) -> None:
        rng = np.random.default_rng(19)
        content = rng.normal(size=(40, 8)).astype(np.float32)
        trial_ids = np.r_[np.zeros(30, dtype=np.int64), np.ones(10, dtype=np.int64)]
        subject_ids = np.r_[np.ones(30, dtype=np.int64), np.full(10, 2, dtype=np.int64)]
        codebook = fit_frozen_e0_codebook(
            content,
            trial_ids,
            subject_ids,
            primitive_num=4,
            pca_dim=3,
            seed=5,
            strict_historical=False,
        )
        tokens, distances, embedded = codebook.assign(content)
        self.assertEqual(tokens.shape, (40,))
        self.assertEqual(distances.shape, (40,))
        np.testing.assert_allclose(np.linalg.norm(embedded, axis=1), 1.0, atol=1e-5)
        self.assertEqual(codebook.fit_trial_count, 2)

        raw = rng.normal(size=(5, raw_descriptor_dimension(4))).astype(np.float64)
        transform = fit_descriptor_transform(
            raw,
            maximum_components=3,
            protected_columns=np.asarray([raw.shape[1] - 2, raw.shape[1] - 1]),
            protected_distance_weight=0.2,
        )
        transform, _ = fit_subject_nuisance_projection(
            transform,
            raw,
            np.asarray([0, 0, 1, 1, 1], dtype=np.int64),
            maximum_rank=1,
            projection_strength=0.5,
        )
        names = tuple(f"d{i}" for i in range(raw.shape[1]))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "artifacts.npz"
            save_frozen_artifacts(path, codebook, transform, names, {"fold": 1})
            restored, restored_transform, restored_names, metadata = load_frozen_artifacts(path)
            np.testing.assert_array_equal(
                restored.assign(content)[0], codebook.assign(content)[0]
            )
            np.testing.assert_allclose(
                restored_transform.transform(raw), transform.transform(raw), atol=1e-6
            )
            np.testing.assert_array_equal(
                restored_transform.protected_columns, transform.protected_columns
            )
            np.testing.assert_allclose(
                restored_transform.nuisance_basis, transform.nuisance_basis, atol=1e-12
            )
            self.assertEqual(restored_transform.nuisance_projection_strength, 0.5)
            self.assertEqual(restored_names, names)
            self.assertEqual(metadata, {"fold": 1})

    def test_raw_window_state_is_not_computed_from_ownership_crop(self) -> None:
        rng = np.random.default_rng(7)
        windows = rng.normal(size=(6, 6, 8)).astype(np.float32)
        labels = np.r_[np.zeros(3, dtype=np.int64), np.ones(3, dtype=np.int64)]
        trials = np.r_[np.zeros(3, dtype=np.int64), np.ones(3, dtype=np.int64)]
        starts = np.tile(np.asarray([0, 4, 8], dtype=np.int64), 2)
        grid = WindowGrid(
            windows=windows,
            labels=labels,
            labels_1based=labels + 1,
            subject_ids=np.r_[np.ones(3, dtype=np.int64), np.full(3, 2, dtype=np.int64)],
            trial_ids=trials,
            trial_numbers=np.ones(6, dtype=np.int64),
            window_indices=np.tile(np.arange(3, dtype=np.int64), 2),
            starts=starts,
            stored_mean=np.zeros((1, 6, 1), dtype=np.float32),
            stored_std=np.ones((1, 6, 1), dtype=np.float32),
            window_size_samples=8,
            stride_samples=4,
        )
        content = rng.normal(size=(6, 6)).astype(np.float32)
        codebook = fit_frozen_e0_codebook(
            content,
            trials,
            grid.subject_ids,
            primitive_num=2,
            pca_dim=2,
            seed=0,
            strict_historical=False,
        )
        trajectories = build_e0_trajectories(
            grid, content, codebook, sample_rate_hz=4.0
        )
        first = trajectories[0]
        np.testing.assert_allclose(first.statistics[:, 0], [1.5, 1.0, 1.5])
        np.testing.assert_allclose(
            first.statistics[:, 1:7], windows[:3].mean(axis=2), atol=1e-7
        )
        matrix, names = descriptor_matrix(trajectories, 2, include_state=True)
        self.assertEqual(matrix.shape, (2, raw_descriptor_dimension(2)))
        self.assertEqual(len(names), raw_descriptor_dimension(2))

    @staticmethod
    def _gravity_trajectory(*, subject_id: int = 2, reverse: bool = False) -> PrimitiveTrajectory:
        length = 128
        phase = np.r_[np.ones(length // 2), -np.ones(length // 2)] * 0.08
        if reverse:
            phase *= -1.0
        raw = np.zeros((1, 6, length), dtype=np.float32)
        raw[0, 0] = 1.0 + phase
        gravity = gravity_aligned_trial_state(raw, np.asarray([0], dtype=np.int64))
        statistics = np.zeros((1, len(statistic_names())), dtype=np.float64)
        statistics[0, 0] = length / 100.0
        return PrimitiveTrajectory(
            trial_id=20,
            subject_id=int(subject_id),
            starts=np.asarray([0], dtype=np.int64),
            ends=np.asarray([length], dtype=np.int64),
            tokens=np.asarray([0], dtype=np.int64),
            distances=np.asarray([0.2], dtype=np.float32),
            embeddings=np.ones((1, 4), dtype=np.float32),
            statistics=statistics,
            statistic_names=statistic_names(),
            gravity_aligned=gravity,
        ).validate(2)

    def test_signed_vertical_trend_reverses_and_is_rotation_invariant(self) -> None:
        up = self._gravity_trajectory(reverse=False).gravity_aligned
        down = self._gravity_trajectory(reverse=True).gravity_aligned
        self.assertGreater(float(up.descriptor[-1]), 0.0)
        self.assertLess(float(down.descriptor[-1]), 0.0)
        np.testing.assert_allclose(up.descriptor, -down.descriptor, atol=1e-7)

        length = 128
        phase = np.r_[np.ones(length // 2), -np.ones(length // 2)] * 0.08
        raw = np.zeros((1, 6, length), dtype=np.float32)
        raw[0, 0] = 1.0 + phase
        angle = np.deg2rad(37.0)
        rotation = np.asarray(
            [
                [np.cos(angle), -np.sin(angle), 0.0],
                [np.sin(angle), np.cos(angle), 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        rotated = raw.copy().astype(np.float64)
        rotated[0, :3] = rotation @ raw[0, :3]
        rotated[0, 3:] = rotation @ raw[0, 3:]
        original_state = gravity_aligned_trial_state(raw, np.asarray([0]))
        rotated_state = gravity_aligned_trial_state(rotated, np.asarray([0]))
        np.testing.assert_allclose(
            original_state.descriptor, rotated_state.descriptor, atol=1e-7
        )

    def test_raw_trial_reconstruction_rejects_inconsistent_overlap(self) -> None:
        raw = np.zeros((2, 6, 8), dtype=np.float32)
        raw[1, :, :4] = 1.0
        with self.assertRaisesRegex(RuntimeError, "Overlapping raw windows disagree"):
            reconstruct_trial_channels(raw, np.asarray([0, 4], dtype=np.int64))

    def test_duration_invariant_profile_drops_absolute_fields_and_resists_scaling(self) -> None:
        original = self._trajectory(32)
        scaled_statistics = original.statistics.copy()
        scaled_statistics[:, 0] *= 3.0
        scaled = PrimitiveTrajectory(
            trial_id=original.trial_id,
            subject_id=original.subject_id,
            starts=original.starts * 3,
            ends=original.ends * 3,
            tokens=original.tokens.copy(),
            distances=original.distances.copy(),
            embeddings=original.embeddings.copy(),
            statistics=scaled_statistics,
            statistic_names=original.statistic_names,
        ).validate(32)
        first, names = trajectory_descriptor(
            original,
            32,
            include_state=True,
            descriptor_profile=DURATION_INVARIANT_DESCRIPTOR_PROFILE,
        )
        second, scaled_names = trajectory_descriptor(
            scaled,
            32,
            include_state=True,
            descriptor_profile=DURATION_INVARIANT_DESCRIPTOR_PROFILE,
        )
        self.assertEqual(names, scaled_names)
        self.assertFalse(ABSOLUTE_DURATION_DESCRIPTOR_NAMES.intersection(names))
        self.assertNotIn("quantization_distance_max", names)
        self.assertFalse(any(name.startswith("state_child_token_presence__") for name in names))
        np.testing.assert_allclose(first, second, atol=1e-12)
        for profile in (
            DURATION_SOFT_SUBJECT_A025_PROFILE,
            DURATION_SOFT_SUBJECT_A050_PROFILE,
            DURATION_SOFT_SUBJECT_A075_PROFILE,
        ):
            with self.subTest(profile=profile):
                soft_values, soft_names = trajectory_descriptor(
                    original,
                    32,
                    include_state=True,
                    descriptor_profile=profile,
                )
                self.assertEqual(soft_names, names)
                np.testing.assert_array_equal(soft_values, first)

    def test_raw_descriptor_never_reads_subject_identity(self) -> None:
        first = self._gravity_trajectory(subject_id=2)
        second = self._gravity_trajectory(subject_id=99)
        for profile in (
            GRAVITY_DESCRIPTOR_PROFILE,
            FULL_DEBIASED_DESCRIPTOR_PROFILE,
        ):
            with self.subTest(profile=profile):
                first_values, first_names = trajectory_descriptor(
                    first, 2, descriptor_profile=profile
                )
                second_values, second_names = trajectory_descriptor(
                    second, 2, descriptor_profile=profile
                )
                self.assertEqual(first_names, second_names)
                np.testing.assert_array_equal(first_values, second_values)

    def test_source_subject_projection_reduces_subject_centroid_offset(self) -> None:
        rng = np.random.default_rng(20260915)
        rows = []
        subjects = []
        classes = []
        for subject, subject_shift in enumerate((-3.0, -1.0, 1.0, 3.0)):
            for class_id, class_shift in enumerate((-2.0, 2.0)):
                for _ in range(12):
                    rows.append(
                        [
                            class_shift + rng.normal(scale=0.03),
                            subject_shift + rng.normal(scale=0.03),
                            rng.normal(scale=0.1),
                        ]
                    )
                    subjects.append(subject)
                    classes.append(class_id)
        values = np.asarray(rows, dtype=np.float64)
        subject_values = np.asarray(subjects, dtype=np.int64)
        class_values = np.asarray(classes, dtype=np.int64)
        transform = fit_descriptor_transform(values, maximum_components=3)
        before = transform.transform(values)
        fitted, audit = fit_subject_nuisance_projection(
            transform,
            values,
            subject_values,
            maximum_rank=1,
            explained_variance=0.9,
        )
        after = fitted.transform(values)

        def dispersion(matrix: np.ndarray, labels: np.ndarray) -> float:
            centroids = np.stack([matrix[labels == value].mean(0) for value in np.unique(labels)])
            return float(np.mean(np.linalg.norm(centroids - centroids.mean(0), axis=1)))

        self.assertEqual(audit["selected_rank"], 1)
        self.assertLess(dispersion(after, subject_values), 0.25 * dispersion(before, subject_values))
        self.assertGreater(dispersion(after, class_values), 0.5 * dispersion(before, class_values))

    def test_soft_subject_projection_matches_registered_partial_formula(self) -> None:
        rng = np.random.default_rng(20260917)
        values = []
        subjects = []
        for subject, shift in enumerate((-2.0, -0.5, 0.5, 2.0)):
            for _ in range(12):
                values.append(
                    [
                        shift + rng.normal(scale=0.03),
                        rng.normal(scale=0.4),
                        rng.normal(scale=0.4),
                    ]
                )
                subjects.append(subject)
        matrix = np.asarray(values, dtype=np.float64)
        subject_values = np.asarray(subjects, dtype=np.int64)
        transform = fit_descriptor_transform(matrix, maximum_components=3)
        fitted, audit = fit_subject_nuisance_projection(
            transform,
            matrix,
            subject_values,
            maximum_rank=1,
            explained_variance=0.9,
            projection_strength=0.5,
        )
        base = transform._normalized_primary(matrix)
        basis = np.asarray(fitted.nuisance_basis, dtype=np.float64)
        expected = base - 0.5 * (base @ basis.T) @ basis
        expected /= np.linalg.norm(expected, axis=1, keepdims=True)
        np.testing.assert_allclose(fitted.transform(matrix), expected, atol=1e-7)
        self.assertEqual(fitted.nuisance_projection_strength, 0.5)
        self.assertEqual(audit["projection_strength"], 0.5)
        self.assertEqual(
            audit["mode"], "offline_old6_balanced_subject_centroid_svd_soft"
        )

        hard, hard_audit = fit_subject_nuisance_projection(
            transform,
            matrix,
            subject_values,
            maximum_rank=1,
            explained_variance=0.9,
            projection_strength=1.0,
        )
        self.assertEqual(
            hard_audit["mode"], "offline_old6_balanced_subject_centroid_svd_hard"
        )
        self.assertGreater(
            float(np.max(np.abs(fitted.transform(matrix) - hard.transform(matrix)))),
            1e-4,
        )
        for invalid in (0.0, -0.1, 1.1):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                fit_subject_nuisance_projection(
                    transform,
                    matrix,
                    subject_values,
                    maximum_rank=1,
                    projection_strength=invalid,
                )

    def test_subject_projection_cannot_modify_protected_block(self) -> None:
        rng = np.random.default_rng(20260916)
        rows = []
        subjects = []
        for subject, shift in enumerate((-2.0, -0.5, 0.5, 2.0)):
            for trial in range(10):
                rows.append(
                    [
                        shift + rng.normal(scale=0.05),
                        rng.normal(scale=0.3),
                        rng.normal(scale=0.3),
                        0.2 * trial + rng.normal(scale=0.02),
                        (-1.0) ** trial + rng.normal(scale=0.02),
                    ]
                )
                subjects.append(subject)
        values = np.asarray(rows, dtype=np.float64)
        subject_values = np.asarray(subjects, dtype=np.int64)
        transform = fit_descriptor_transform(
            values,
            maximum_components=3,
            protected_columns=np.asarray([3, 4], dtype=np.int64),
            protected_distance_weight=0.2,
        )
        before = transform.transform(values)
        fitted, _ = fit_subject_nuisance_projection(
            transform,
            values,
            subject_values,
            maximum_rank=1,
            explained_variance=0.9,
        )
        after = fitted.transform(values)

        self.assertEqual(
            fitted.nuisance_basis.shape,
            (1, transform.primary_output_dim),
        )
        np.testing.assert_allclose(before[:, -2:], after[:, -2:], atol=2e-7)
        before_direction = before[:, -2:] / np.maximum(
            np.linalg.norm(before[:, -2:], axis=1, keepdims=True), 1e-12
        )
        after_direction = after[:, -2:] / np.maximum(
            np.linalg.norm(after[:, -2:], axis=1, keepdims=True), 1e-12
        )
        np.testing.assert_allclose(before_direction, after_direction, atol=2e-7)


if __name__ == "__main__":
    unittest.main()
