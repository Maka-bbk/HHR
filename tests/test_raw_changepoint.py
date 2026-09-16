import copy
import unittest

import numpy as np
import torch
import torch.nn as nn

from experiments.motion_primitive.motion_encoder import MotionPrimitiveEncoder
from experiments.motion_primitive.raw_changepoint import (
    RAW_COMPONENTS_PER_SCALE,
    consensus_boundary_masks,
    fit_trial_equal_robust_component_scaler,
    fit_trial_equal_score_thresholds,
    raw_boundary_anchor_samples,
    raw_component_names,
    raw_multiscale_boundary_components,
    transform_raw_boundary_components,
)
from experiments.motion_primitive.train_motion_encoder import (
    _set_backbone_batchnorm_policy,
)


class RawKinematicBoundaryTests(unittest.TestCase):
    def test_constant_zero_trial_has_zero_components(self):
        components = raw_multiscale_boundary_components(
            np.zeros((6, 96), dtype=np.float64),
            np.asarray([0, 32, 64]),
            window_size=32,
            scales=(1.0,),
            frequency_bins=8,
        )
        np.testing.assert_array_equal(components, np.zeros_like(components))

    def test_multiscale_components_align_to_window_boundaries_and_detect_step(self):
        samples = 256
        time = np.arange(samples, dtype=np.float64)
        raw = np.zeros((6, samples), dtype=np.float64)
        raw[2, :128] = 9.81
        raw[0, 128:] = 9.81
        raw[3, :128] = np.sin(2.0 * np.pi * time[:128] / 32.0)
        raw[3, 128:] = 2.0 * np.sin(2.0 * np.pi * time[128:] / 8.0)
        starts = np.arange(0, 225, 32, dtype=np.int64)

        components = raw_multiscale_boundary_components(
            raw, starts, window_size=32, scales=(1.0, 2.0), frequency_bins=8
        )
        names = raw_component_names((1.0, 2.0))
        self.assertEqual(components.shape, (len(starts) - 1, len(names)))
        self.assertEqual(len(names), 2 * len(RAW_COMPONENTS_PER_SCALE))
        np.testing.assert_array_equal(
            raw_boundary_anchor_samples(starts, 32),
            np.asarray([32, 64, 96, 128, 160, 192, 224]),
        )
        centre = 3
        acc_mean_column = names.index("stride_x1__acc_mean_l2")
        gravity_column = names.index("stride_x1__gravity_direction_angle_rad")
        gyro_fft_column = names.index("stride_x1__gyro_fft_amplitude_l2")
        self.assertGreater(components[centre, acc_mean_column], components[0, acc_mean_column])
        self.assertGreater(components[centre, gravity_column], 1.0)
        self.assertGreater(components[centre, gyro_fft_column], 0.01)
        self.assertTrue(np.all(np.isfinite(components)))

    def test_trial_equal_robust_fit_is_invariant_to_boundary_replication(self):
        names = ("one", "two")
        short = np.asarray([[0.0, 2.0]])
        long_nine = np.repeat(np.asarray([[10.0, 6.0]]), 9, axis=0)
        long_ninety = np.repeat(np.asarray([[10.0, 6.0]]), 90, axis=0)
        first = fit_trial_equal_robust_component_scaler([short, long_nine], names)
        second = fit_trial_equal_robust_component_scaler([short, long_ninety], names)
        np.testing.assert_allclose(first["q25"], second["q25"])
        np.testing.assert_allclose(first["centre"], second["centre"])
        np.testing.assert_allclose(first["q75"], second["q75"])
        np.testing.assert_allclose(first["scale"], second["scale"])
        self.assertEqual(first["train_boundary_bearing_trial_count"], 2)

    def test_validation_is_transform_only_and_cannot_change_train_thresholds(self):
        train = [
            np.asarray([[0.0, 0.0], [1.0, 2.0]]),
            np.asarray([[2.0, 1.0], [3.0, 3.0]]),
        ]
        scaler = fit_trial_equal_robust_component_scaler(train, ("a", "b"))
        train_scores = [transform_raw_boundary_components(item, scaler) for item in train]
        thresholds = fit_trial_equal_score_thresholds(train_scores, 0.5, 0.9)
        frozen_scaler = copy.deepcopy(scaler)
        frozen_thresholds = copy.deepcopy(thresholds)

        validation = np.asarray([[1.0e9, -1.0e9], [5.0e8, 4.0e8]])
        transformed = transform_raw_boundary_components(validation, scaler)
        self.assertEqual(transformed.shape, (2,))
        self.assertEqual(scaler, frozen_scaler)
        self.assertEqual(thresholds, frozen_thresholds)

    def test_consensus_can_legitimately_produce_no_change_anchor(self):
        thresholds = {"low_threshold": 0.2, "high_threshold": 0.8}
        raw = np.asarray([0.0, 1.0, 0.5])
        frozen = np.asarray([0.0, 0.5, 1.0])
        stable, change, diagnostic = consensus_boundary_masks(
            raw,
            frozen,
            thresholds,
            thresholds,
            "raw_frozen_consensus",
        )
        np.testing.assert_array_equal(stable, [True, False, False])
        np.testing.assert_array_equal(change, [False, False, False])
        self.assertEqual(diagnostic["selected_change_count"], 0)


class BackboneBatchNormPolicyTests(unittest.TestCase):
    @staticmethod
    def model():
        return MotionPrimitiveEncoder(
            in_channels=6,
            backbone_dim=8,
            base_channels=2,
            backbone_layers=(1, 1, 1),
            backbone_dropout=0.0,
            segmentation_dim=4,
            content_dim=8,
            content_residual=True,
            augmentation_dim=3,
            projection_hidden_dim=7,
            num_classes=2,
            trial_hidden_dim=5,
            trial_peak_quantile=0.9,
            trial_dropout=0.0,
            predictor_hidden_dim=5,
        )

    def test_frozen_policy_only_sets_backbone_batchnorm_to_eval(self):
        model = self.model().train()
        count = _set_backbone_batchnorm_policy(model, "frozen")
        batchnorms = [item for item in model.backbone.modules() if isinstance(item, nn.BatchNorm1d)]
        self.assertEqual(count, len(batchnorms))
        self.assertGreater(count, 0)
        self.assertTrue(all(not item.training for item in batchnorms))
        self.assertTrue(all(item.weight.requires_grad for item in batchnorms))
        self.assertTrue(model.segmentation_head.training)
        self.assertTrue(model.content_head.training)

        model.train()
        _set_backbone_batchnorm_policy(model, "update")
        self.assertTrue(all(item.training for item in batchnorms))


if __name__ == "__main__":
    unittest.main()
