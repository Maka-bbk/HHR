from __future__ import annotations

import argparse
import hashlib
import inspect
import unittest

import numpy as np

from experiments.motion_primitive.frozen_e0_state import (
    HISTORICAL_STATE_SCHEMA_SHA256,
    PrimitiveTrial,
    statistic_names,
    trajectory_descriptor,
)
from experiments.motion_primitive import strict_session3_k12_proxy as proxy
from experiments.motion_primitive import strict_session3_k12_proxy_cv as cv


class IncomingTransformTests(unittest.TestCase):
    def test_constant_zscore_pca32_l2_is_fit_only_from_supplied_rows(self) -> None:
        rng = np.random.default_rng(7)
        fit = rng.normal(size=(78, 50))
        fit[:, :3] = np.asarray([1.0, -2.0, 9.0])
        state = proxy.fit_incoming_descriptor_transform(fit)
        self.assertEqual(state.fit_row_count, 78)
        self.assertEqual(state.output_dim, 32)
        self.assertEqual(len(state.keep_columns), 47)
        features = state.transform(fit)
        self.assertEqual(features.shape, (78, 32))
        np.testing.assert_allclose(np.linalg.norm(features, axis=1), 1.0, atol=1e-6)
        evaluation = rng.normal(size=(42, 50)) * 1e6
        before = state.state_sha256
        self.assertEqual(state.transform(evaluation).shape, (42, 32))
        self.assertEqual(state.state_sha256, before)
        self.assertNotIn("evaluation", inspect.signature(proxy.fit_incoming_descriptor_transform).parameters)

    def test_raw_descriptor_schema_hash_uses_newline_join(self) -> None:
        trial = PrimitiveTrial(
            trial_id=1,
            subject_id=1,
            starts=np.asarray([0, 128], dtype=np.int64),
            ends=np.asarray([128, 384], dtype=np.int64),
            child_tokens=np.asarray([0, 1], dtype=np.int64),
            child_distances=np.asarray([0.1, 0.2]),
            child_embeddings=np.ones((2, 64)),
            child_statistics=np.ones((2, len(statistic_names()))),
            statistic_names=statistic_names(),
        ).validate(32)
        _, names = trajectory_descriptor(trial, 32, include_state=True)
        observed = hashlib.sha256("\n".join(names).encode("utf-8")).hexdigest()
        self.assertEqual(observed, HISTORICAL_STATE_SCHEMA_SHA256)


class ContractTests(unittest.TestCase):
    def _args(self, **updates: object) -> argparse.Namespace:
        values = {
            "profile": "frozen_a2_e0_state_k32",
            "fold": 1,
            "seed": 0,
            "cluster_count": 12,
            "descriptor_pca_dim": 32,
            "constant_tolerance": 1e-10,
            "kmeans_n_init": 50,
            "kmeans_max_iter": 300,
            "encode_batch_size": 512,
            "device": "cpu",
            "allow_smoke_a2": False,
            "resume": False,
        }
        values.update(updates)
        return argparse.Namespace(**values)

    def test_single_member_is_pinned_to_k12_pca32(self) -> None:
        proxy.validate_args(self._args())
        with self.assertRaisesRegex(ValueError, "K=12"):
            proxy.validate_args(self._args(cluster_count=10))
        with self.assertRaisesRegex(ValueError, "PCA32"):
            proxy.validate_args(self._args(descriptor_pca_dim=16))

    def test_cv_is_pinned_to_full_grid(self) -> None:
        folds, seeds = cv.validate_canonical_grid(tuple(range(1, 8)), (0, 5, 50, 500))
        self.assertEqual(len(folds) * len(seeds), 28)
        with self.assertRaisesRegex(ValueError, "requires"):
            cv.validate_canonical_grid((1,), (0, 5, 50, 500))

    def test_contract_constants_match_registered_final_session(self) -> None:
        self.assertEqual(proxy.INCOMING_SESSION_COUNTS, (22, 26, 30))
        self.assertEqual(proxy.INCOMING_TOTAL, 78)
        self.assertEqual(proxy.EVALUATION_COUNT, 42)
        self.assertEqual(proxy.CLUSTER_COUNT, 12)


if __name__ == "__main__":
    unittest.main()
