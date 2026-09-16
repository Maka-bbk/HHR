from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from experiments.motion_primitive.frozen_e0_state import (
    HISTORICAL_STATE_SCHEMA_SHA256,
    STATE_DESCRIPTOR_RAW_DIM,
    PrimitiveTrial,
    WindowTrial,
    _window_partition,
    fit_e0_codebook,
    inverse_trial_frequency_weights,
    l2_normalize,
    statistic_names,
    trajectory_descriptor,
)
from experiments.motion_primitive.strict_artifacts import (
    load_e0_codebook,
    save_e0_codebook,
)


HISTORICAL_SCHEMA_LITERAL = (
    "dbdceb949b66e900400dc9908286f5ac371db4d04b56aaf6e90f8d206c7e70cc"
)


def _primitive_trial(
    tokens: list[int],
    durations: list[int],
    *,
    trial_id: int = 1,
) -> PrimitiveTrial:
    count = len(tokens)
    ends = np.cumsum(np.asarray(durations, dtype=np.int64))
    starts = np.r_[0, ends[:-1]].astype(np.int64)
    statistics = np.zeros((count, len(statistic_names())), dtype=np.float64)
    statistics[:, 0] = np.asarray(durations, dtype=np.float64) / 100.0
    # Make state blocks non-degenerate without letting them depend on labels.
    for row in range(count):
        statistics[row, 1:] = np.arange(1, statistics.shape[1], dtype=np.float64) + row
    embeddings = np.zeros((count, 64), dtype=np.float64)
    embeddings[np.arange(count), np.asarray(tokens, dtype=np.int64)] = 1.0
    return PrimitiveTrial(
        trial_id=trial_id,
        subject_id=3,
        starts=starts,
        ends=ends,
        child_tokens=np.asarray(tokens, dtype=np.int64),
        child_distances=np.linspace(0.05, 0.15, count, dtype=np.float64),
        child_embeddings=embeddings,
        child_statistics=statistics,
        statistic_names=statistic_names(),
        event_kinds=(),
    ).validate(32)


def _synthetic_codebook_trials() -> tuple[WindowTrial, ...]:
    """Create a full-rank, cluster-rich, label-free E0 fit set."""

    rng = np.random.default_rng(20260912)
    directions = rng.normal(size=(32, 80))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    trials: list[WindowTrial] = []
    for trial_offset in range(4):
        content = directions + rng.normal(scale=0.003, size=directions.shape)
        raw = rng.normal(size=(32, 6, 16)).astype(np.float32)
        trials.append(
            WindowTrial(
                trial_id=100 + trial_offset,
                subject_id=1 + trial_offset % 2,
                window_starts=np.arange(32, dtype=np.int64) * 8,
                raw_windows=raw,
                content_embeddings=content.astype(np.float32),
            ).validate(window_size=16)
        )
    return tuple(trials)


class FrozenE0StateContractTests(unittest.TestCase):
    def test_state_descriptor_is_3739_and_matches_historical_schema_sha(self) -> None:
        vector, names = trajectory_descriptor(
            _primitive_trial([0, 1, 0], [2, 3, 4]), include_state=True
        )
        observed_sha = hashlib.sha256("\n".join(names).encode("utf-8")).hexdigest()
        self.assertEqual(STATE_DESCRIPTOR_RAW_DIM, 3739)
        self.assertEqual(vector.shape, (3739,))
        self.assertEqual(len(names), 3739)
        # Keep the literal here as an independent tripwire: changing both the
        # implementation constant and the generated schema must still fail.
        self.assertEqual(HISTORICAL_STATE_SCHEMA_SHA256, HISTORICAL_SCHEMA_LITERAL)
        self.assertEqual(observed_sha, HISTORICAL_SCHEMA_LITERAL)
        self.assertTrue(np.all(np.isfinite(vector)))

    def test_window_centres_define_exact_voronoi_ownership(self) -> None:
        starts, ends = _window_partition(
            np.asarray([0, 128, 256], dtype=np.int64), window_size=256
        )
        np.testing.assert_array_equal(starts, np.asarray([0, 192, 320]))
        np.testing.assert_array_equal(ends, np.asarray([192, 320, 512]))
        np.testing.assert_array_equal(ends - starts, np.asarray([192, 128, 192]))
        self.assertEqual(int(starts[0]), 0)
        self.assertEqual(int(ends[-1]), 512)
        self.assertTrue(np.array_equal(starts[1:], ends[:-1]))

    def test_count_duration_and_transition_blocks_are_exact(self) -> None:
        vector, names = trajectory_descriptor(
            _primitive_trial([0, 1, 0], [2, 3, 4]), include_state=True
        )
        by_name = dict(zip(names, vector.tolist()))
        self.assertAlmostEqual(by_name["child_count_fraction__0"], 2.0 / 3.0)
        self.assertAlmostEqual(by_name["child_count_fraction__1"], 1.0 / 3.0)
        self.assertAlmostEqual(by_name["child_duration_fraction__0"], 6.0 / 9.0)
        self.assertAlmostEqual(by_name["child_duration_fraction__1"], 3.0 / 9.0)
        self.assertAlmostEqual(by_name["transition_any__1"], 0.5)   # 0 -> 1
        self.assertAlmostEqual(by_name["transition_any__32"], 0.5)  # 1 -> 0
        transition = vector[192 : 192 + 32 * 32]
        self.assertAlmostEqual(float(transition.sum()), 1.0)
        self.assertEqual(int(np.count_nonzero(transition)), 2)

    def test_reversing_a_trajectory_changes_order_features_not_bag_counts(self) -> None:
        forward, forward_names = trajectory_descriptor(
            _primitive_trial([0, 1, 2], [2, 2, 2], trial_id=10), include_state=True
        )
        reverse, reverse_names = trajectory_descriptor(
            _primitive_trial([2, 1, 0], [2, 2, 2], trial_id=11), include_state=True
        )
        self.assertEqual(forward_names, reverse_names)
        np.testing.assert_allclose(forward[:64], reverse[:64], atol=0.0, rtol=0.0)
        transition_slice = slice(192, 192 + 32 * 32)
        self.assertFalse(np.array_equal(forward[transition_slice], reverse[transition_slice]))
        self.assertFalse(np.array_equal(forward, reverse))
        index = {name: offset for offset, name in enumerate(forward_names)}
        self.assertEqual(forward[index["transition_any__1"]], 0.5)   # 0 -> 1
        self.assertEqual(reverse[index["transition_any__1"]], 0.0)
        self.assertEqual(reverse[index["transition_any__32"]], 0.5)  # 1 -> 0

    def test_inverse_frequency_weights_give_every_trial_equal_total_mass(self) -> None:
        trial_ids = np.asarray([7, 7, 7, 7, 9, 9], dtype=np.int64)
        weights = inverse_trial_frequency_weights(trial_ids)
        self.assertAlmostEqual(float(weights.mean()), 1.0)
        self.assertAlmostEqual(float(weights[trial_ids == 7].sum()), 3.0)
        self.assertAlmostEqual(float(weights[trial_ids == 9].sum()), 3.0)

        values = np.asarray(
            [[1.0, 0.0]] * 4 + [[0.0, 1.0]] * 2,
            dtype=np.float32,
        )
        weighted_mean = np.average(l2_normalize(values), axis=0, weights=weights)
        np.testing.assert_allclose(weighted_mean, [0.5, 0.5], atol=1e-7)

    def test_k32_pca64_fit_and_strict_artifact_round_trip(self) -> None:
        trials = _synthetic_codebook_trials()
        codebook = fit_e0_codebook(trials, seed=5, primitive_num=32, pca_dim=64)
        self.assertEqual(codebook.primitive_num, 32)
        self.assertEqual(codebook.pca_dim, 64)
        self.assertEqual(codebook.fit_trial_count, 4)
        self.assertEqual(codebook.fit_window_count, 128)
        self.assertEqual(codebook.fit_subject_count, 2)
        self.assertEqual(codebook.fit_used_k, 32)
        self.assertEqual(codebook.pca_components.shape, (64, 80))
        self.assertEqual(codebook.cluster_centers.shape, (32, 64))

        probe = np.concatenate([trial.content_embeddings[:3] for trial in trials])
        expected_tokens, expected_distances, expected_embeddings = codebook.assign(probe)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "e0_codebook.npz"
            metadata = save_e0_codebook(path, codebook)
            self.assertEqual(metadata["schema"], "hhr_frozen_e0_codebook_v1")
            self.assertEqual(metadata["fit_scope"], "offline_train_subjects_old6_only")
            self.assertIs(metadata["online_mutable"], False)
            restored = load_e0_codebook(path)
            actual_tokens, actual_distances, actual_embeddings = restored.assign(probe)
            np.testing.assert_array_equal(actual_tokens, expected_tokens)
            np.testing.assert_allclose(actual_distances, expected_distances, atol=1e-7)
            np.testing.assert_allclose(actual_embeddings, expected_embeddings, atol=1e-7)
            self.assertEqual(restored.state_sha256, codebook.state_sha256)

            sidecar = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
            self.assertEqual(sidecar["state_sha256"], codebook.state_sha256)


if __name__ == "__main__":
    unittest.main()
