from __future__ import annotations

import numpy as np
import unittest

from experiments.motion_primitive.frozen_hierarchical_readout import (
    apply_frozen_leaf_overrides,
    assert_frozen_readout_schema_is_label_free,
    fit_frozen_coarse_readout,
    frozen_readout_diagnostics,
)


def _readout():
    train = np.asarray(
        [[0.0, 0.0], [0.1, 0.0], [2.0, 2.0], [2.1, 2.0]], dtype=np.float32
    )
    test = np.asarray(
        [[0.0, 0.1], [2.0, 2.1], [0.2, 0.0], [2.2, 2.0]], dtype=np.float32
    )
    return fit_frozen_coarse_readout(
        train,
        test,
        train_trial_ids=[1, 2, 3, 4],
        test_trial_ids=[11, 12, 13, 14],
        cluster_count=2,
        seed=7,
        n_init=5,
    )


class FrozenHierarchicalReadoutTests(unittest.TestCase):
    def test_frozen_readout_schema_is_label_free(self) -> None:
        assert_frozen_readout_schema_is_label_free()

    def test_only_routed_predictions_receive_appended_leaf_ids(self) -> None:
        readout = _readout()
        result = apply_frozen_leaf_overrides(readout, {11: 1, 14: 2})
        fallback = ~result.routed_mask
        self.assertTrue(
            np.array_equal(
                result.refined_predictions[fallback],
                result.coarse_predictions[fallback],
            )
        )
        self.assertEqual(result.refined_predictions[0], 2)
        self.assertEqual(result.refined_predictions[3], 3)
        diagnostics = frozen_readout_diagnostics(readout, result)
        self.assertEqual(diagnostics["coarse_fit_call_count"], 1)
        self.assertEqual(diagnostics["fallback_raw_mismatch_count"], 0)
        self.assertEqual(diagnostics["changed_trial_ids"], [11, 14])

    def test_disabled_gate_reproduces_coarse_predictions_bitwise(self) -> None:
        readout = _readout()
        result = apply_frozen_leaf_overrides(readout, {})
        self.assertTrue(
            np.array_equal(result.refined_predictions, result.coarse_predictions)
        )
        self.assertFalse(np.any(result.routed_mask))

    def test_leaf_ids_cannot_collide_with_coarse_clusters(self) -> None:
        with self.assertRaisesRegex(ValueError, "outside coarse"):
            apply_frozen_leaf_overrides(
                _readout(),
                {11: 1},
                leaf_ids_by_gate_state={1: 1, 2: 2},
            )

    def test_train_and_test_trial_ids_must_not_overlap(self) -> None:
        features = np.eye(3, dtype=np.float32)
        with self.assertRaisesRegex(RuntimeError, "overlap"):
            fit_frozen_coarse_readout(
                features,
                features,
                train_trial_ids=[1, 2, 3],
                test_trial_ids=[3, 4, 5],
                cluster_count=2,
            )


if __name__ == "__main__":
    unittest.main()
