import unittest

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from experiments.motion_primitive.motion_augmentation import (
    MotionAugmentationConfig,
    augment_raw_trial,
    augment_raw_trial_with_metadata,
    make_augmented_trial_view,
    normalize_and_slice_windows,
)
from experiments.motion_primitive.motion_encoder import (
    MotionPrimitiveEncoder,
    batched_feature_change_scores,
    boundary_loss_terms,
    boundary_ranking_loss,
    masked_temporal_prediction_loss,
    stable_boundary_loss,
    symmetric_info_nce,
    variance_covariance_terms,
)
from experiments.motion_primitive.motion_checkpoint import (
    motion_state_dict_sha256,
    validate_motion_encoder_checkpoint_integrity,
)
from experiments.motion_primitive.run_experiment import (
    build_frozen_motion_encoder,
    encode_motion_windows,
)
from experiments.motion_primitive.segmentation import (
    build_segmented_features,
    feature_change_scores,
)


def tiny_architecture():
    return {
        "in_channels": 6,
        "backbone_dim": 8,
        "base_channels": 2,
        "backbone_layers": [1, 1, 1],
        "backbone_dropout": 0.0,
        "segmentation_dim": 4,
        "segmentation_residual": True,
        "content_dim": 6,
        "content_residual": True,
        "augmentation_dim": 3,
        "projection_hidden_dim": 7,
        "num_classes": 2,
        "trial_hidden_dim": 5,
        "trial_peak_quantile": 0.9,
        "trial_dropout": 0.0,
        "predictor_hidden_dim": 5,
    }


def tiny_encoder():
    return MotionPrimitiveEncoder(**tiny_architecture())


def disabled_augmentation():
    return MotionAugmentationConfig(
        noise_std_ratio=0.0,
        acc_scale_range=(1.0, 1.0),
        gyro_scale_range=(1.0, 1.0),
        time_shift_max_samples=0,
        time_mask_min_samples=0,
        time_mask_max_samples=0,
        rotation_max_degrees=0.0,
    )


class MotionPrimitiveEncoderShapeTests(unittest.TestCase):
    def test_segmentation_residual_starts_as_exact_normalized_backbone(self):
        architecture = tiny_architecture()
        architecture["segmentation_dim"] = architecture["backbone_dim"]
        torch.manual_seed(2)
        model = MotionPrimitiveEncoder(**architecture).eval()
        encoded = model.encode_windows(torch.randn(5, 6, 32))
        torch.testing.assert_close(
            encoded["segmentation_raw"], encoded["backbone"], rtol=0, atol=0
        )
        torch.testing.assert_close(
            encoded["segmentation"],
            F.normalize(encoded["backbone"], dim=-1),
            rtol=0,
            atol=0,
        )

    def test_flat_and_padded_dual_head_shapes_and_padding(self):
        torch.manual_seed(3)
        model = tiny_encoder().eval()

        flat = model(torch.randn(3, 6, 32))
        expected_flat_shapes = {
            "backbone": (3, 8),
            "segmentation_raw": (3, 4),
            "segmentation": (3, 4),
            "content": (3, 6),
            "content_delta": (3, 6),
            "augmentation_raw": (3, 3),
            "augmentation": (3, 3),
        }
        self.assertEqual(set(flat), set(expected_flat_shapes))
        for name, shape in expected_flat_shapes.items():
            self.assertEqual(tuple(flat[name].shape), shape)
        torch.testing.assert_close(
            torch.linalg.vector_norm(flat["segmentation"], dim=-1),
            torch.ones(3),
        )
        torch.testing.assert_close(
            torch.linalg.vector_norm(flat["augmentation"], dim=-1),
            torch.ones(3),
        )

        valid = torch.tensor([[True, True, True], [True, False, False]])
        padded_windows = torch.randn(2, 3, 6, 32)
        # Invalid windows must not enter the backbone. NaNs make accidental
        # processing immediately visible in the output.
        padded_windows[~valid] = torch.nan
        padded = model(padded_windows, valid)
        expected_padded_shapes = {
            "backbone": (2, 3, 8),
            "segmentation_raw": (2, 3, 4),
            "segmentation": (2, 3, 4),
            "content": (2, 3, 6),
            "content_delta": (2, 3, 6),
            "augmentation_raw": (2, 3, 3),
            "augmentation": (2, 3, 3),
            "trial_embedding": (2, 6),
            "trial_logits": (2, 2),
            "valid_mask": (2, 3),
        }
        self.assertEqual(set(padded), set(expected_padded_shapes))
        for name, shape in expected_padded_shapes.items():
            self.assertEqual(tuple(padded[name].shape), shape)
        torch.testing.assert_close(padded["valid_mask"], valid)
        for name in expected_flat_shapes:
            self.assertTrue(torch.isfinite(padded[name]).all(), name)
            torch.testing.assert_close(
                padded[name][~valid], torch.zeros_like(padded[name][~valid])
            )

    def test_padded_mask_must_be_left_aligned(self):
        model = tiny_encoder().eval()
        invalid = torch.tensor([[True, False, True]])
        with self.assertRaisesRegex(ValueError, "left aligned"):
            model(torch.randn(1, 3, 6, 32), invalid)


class MotionEncoderLossTests(unittest.TestCase):
    def test_torch_change_scores_match_numpy_for_each_unpadded_trial(self):
        generator = torch.Generator().manual_seed(19)
        lengths = [7, 4]
        values = F.normalize(torch.randn(2, 7, 5, generator=generator), dim=-1)
        valid = torch.zeros(2, 7, dtype=torch.bool)
        for row, length in enumerate(lengths):
            valid[row, :length] = True
        scores, boundary_valid = batched_feature_change_scores(
            values, context_windows=2, valid_mask=valid
        )

        for row, length in enumerate(lengths):
            expected = feature_change_scores(
                values[row, :length].numpy(), context_windows=2
            )
            np.testing.assert_allclose(
                scores[row, : length - 1].numpy(), expected, rtol=2e-6, atol=2e-6
            )
            self.assertTrue(boundary_valid[row, : length - 1].all())
            self.assertFalse(boundary_valid[row, length - 1 :].any())
            torch.testing.assert_close(
                scores[row, length - 1 :],
                torch.zeros_like(scores[row, length - 1 :]),
            )

    def test_info_nce_checks_unique_trials_handles_one_trial_and_detects_shuffle(self):
        view_one = torch.eye(4)
        view_two = torch.eye(4)
        matched = symmetric_info_nce(
            view_one, view_two, temperature=0.05, trial_ids=torch.arange(4)
        )
        shuffled = symmetric_info_nce(
            view_one,
            view_two[torch.tensor([1, 0, 3, 2])],
            temperature=0.05,
            trial_ids=torch.arange(4),
        )
        self.assertLess(float(matched), float(shuffled))
        self.assertLess(float(matched), 1e-5)

        with self.assertRaisesRegex(ValueError, "one anchor window per trial"):
            symmetric_info_nce(
                view_one, view_two, trial_ids=torch.tensor([4, 4, 5, 6])
            )

        singleton_one = torch.randn(1, 5, requires_grad=True)
        singleton_two = torch.randn(1, 5, requires_grad=True)
        singleton_loss = symmetric_info_nce(
            singleton_one, singleton_two, trial_ids=torch.tensor([17])
        )
        self.assertEqual(float(singleton_loss.detach()), 0.0)
        singleton_loss.backward()
        torch.testing.assert_close(singleton_one.grad, torch.zeros_like(singleton_one))
        torch.testing.assert_close(singleton_two.grad, torch.zeros_like(singleton_two))

    def test_variance_covariance_penalizes_constant_raw_projection(self):
        collapsed = torch.full((8, 3), 7.0)
        # Every combination of three independent signs has zero off-diagonal
        # covariance and sample std > 1, so it is a non-collapsed reference.
        spread = torch.tensor(
            [
                [-1.0, -1.0, -1.0],
                [-1.0, -1.0, 1.0],
                [-1.0, 1.0, -1.0],
                [-1.0, 1.0, 1.0],
                [1.0, -1.0, -1.0],
                [1.0, -1.0, 1.0],
                [1.0, 1.0, -1.0],
                [1.0, 1.0, 1.0],
            ]
        )
        collapsed_terms = variance_covariance_terms(collapsed)
        spread_terms = variance_covariance_terms(spread)
        self.assertGreater(float(collapsed_terms["variance"]), 0.98)
        self.assertEqual(float(collapsed_terms["covariance"]), 0.0)
        self.assertLess(float(spread_terms["loss"]), 1e-6)
        self.assertGreater(
            float(collapsed_terms["loss"]), float(spread_terms["loss"])
        )

    def test_stable_and_ranking_gradients_have_the_intended_direction(self):
        stable_scores = torch.tensor([[0.7, 0.2]], requires_grad=True)
        stable = torch.tensor([[True, False]])
        stable_loss = stable_boundary_loss(stable_scores, stable)
        stable_loss.backward()
        # Gradient descent therefore lowers a selected stable-boundary score.
        self.assertGreater(float(stable_scores.grad[0, 0]), 0.0)
        self.assertEqual(float(stable_scores.grad[0, 1]), 0.0)

        ranked_scores = torch.tensor([[0.1, 0.9]], requires_grad=True)
        change = torch.tensor([[True, False]])
        stable = torch.tensor([[False, True]])
        ranking_loss = boundary_ranking_loss(
            ranked_scores, change, stable, margin=0.2
        )
        ranking_loss.backward()
        # Gradient descent raises the change score and lowers the stable score.
        self.assertLess(float(ranked_scores.grad[0, 0]), 0.0)
        self.assertGreater(float(ranked_scores.grad[0, 1]), 0.0)

        good = boundary_ranking_loss(
            torch.tensor([[0.9, 0.1]]), change, stable, margin=0.2
        )
        self.assertEqual(float(good), 0.0)
        self.assertGreater(float(ranking_loss.detach()), float(good))

    def test_boundary_objective_rejects_all_merge_and_all_cut(self):
        x = torch.tensor([1.0, 0.0])
        y = torch.tensor([0.0, 1.0])
        ideal = torch.stack([x, x, x, y, y, y]).unsqueeze(0)
        all_merge = x.repeat(6, 1).unsqueeze(0)
        all_cut = torch.stack([x, -x, x, -x, x, -x]).unsqueeze(0)
        change = torch.tensor([[False, False, True, False, False]])
        stable = ~change

        def terms(features):
            return boundary_loss_terms(
                features,
                features,
                stable_mask=stable,
                change_mask=change,
                context_windows=1,
                rank_margin=0.2,
            )

        ideal_terms = terms(ideal)
        merge_terms = terms(all_merge)
        cut_terms = terms(all_cut)
        self.assertEqual(float(ideal_terms["loss"]), 0.0)
        self.assertGreater(float(merge_terms["ranking"]), 0.0)
        self.assertEqual(float(merge_terms["stable"]), 0.0)
        self.assertGreater(float(cut_terms["stable"]), 0.0)
        self.assertGreater(float(merge_terms["loss"]), float(ideal_terms["loss"]))
        self.assertGreater(float(cut_terms["loss"]), float(ideal_terms["loss"]))

    def test_empty_masked_prediction_loss_is_graph_linked_zero(self):
        prediction = torch.randn(2, 3, 4, requires_grad=True)
        target = torch.randn(2, 3, 4, requires_grad=True)
        empty = torch.zeros(2, 3, dtype=torch.bool)
        valid = torch.tensor([[True, True, True], [True, True, False]])
        loss = masked_temporal_prediction_loss(
            prediction, target, empty, valid_mask=valid
        )
        self.assertEqual(float(loss.detach()), 0.0)
        loss.backward()
        torch.testing.assert_close(prediction.grad, torch.zeros_like(prediction))
        torch.testing.assert_close(target.grad, torch.zeros_like(target))


class MotionAugmentationTests(unittest.TestCase):
    def setUp(self):
        time = torch.arange(24, dtype=torch.float32)
        self.raw = torch.stack([time + 100.0 * channel for channel in range(6)])
        self.mean = torch.arange(6, dtype=torch.float32) * 10.0
        self.std = torch.arange(1, 7, dtype=torch.float32)

    def test_identity_preserves_trial_length_and_window_temporal_order(self):
        identity = augment_raw_trial(
            self.raw, self.mean, self.std, disabled_augmentation()
        )
        torch.testing.assert_close(identity, self.raw, rtol=0.0, atol=0.0)
        self.assertEqual(tuple(identity.shape), tuple(self.raw.shape))

        starts = [0, 5, 12]
        windows = normalize_and_slice_windows(
            identity, self.mean, self.std, starts=starts, window_size=8
        )
        self.assertEqual(tuple(windows.shape), (3, 6, 8))
        normalized = (self.raw - self.mean[:, None]) / self.std[:, None]
        for index, start in enumerate(starts):
            torch.testing.assert_close(
                windows[index], normalized[:, start : start + 8]
            )

    def test_shift_uses_reflection_instead_of_circular_wraparound(self):
        config = MotionAugmentationConfig(
            noise_std_ratio=0.0,
            acc_scale_range=(1.0, 1.0),
            gyro_scale_range=(1.0, 1.0),
            time_shift_max_samples=3,
            time_mask_min_samples=0,
            time_mask_max_samples=0,
            rotation_max_degrees=0.0,
        )
        augmented = None
        metadata = None
        for seed in range(20):
            candidate, candidate_metadata = augment_raw_trial_with_metadata(
                self.raw,
                self.mean,
                self.std,
                config,
                generator=torch.Generator().manual_seed(seed),
            )
            if candidate_metadata.time_shift_samples != 0:
                augmented, metadata = candidate, candidate_metadata
                break
        self.assertIsNotNone(augmented, "test seeds unexpectedly sampled only zero shifts")
        shift = metadata.time_shift_samples
        self.assertFalse(torch.equal(augmented, torch.roll(self.raw, shift, dims=-1)))
        if shift > 0:
            torch.testing.assert_close(augmented[:, shift:], self.raw[:, :-shift])
        else:
            amount = -shift
            torch.testing.assert_close(augmented[:, :-amount], self.raw[:, amount:])

    def test_full_trial_augmentation_keeps_overlapping_samples_identical(self):
        config = MotionAugmentationConfig(
            noise_std_ratio=0.02,
            acc_scale_range=(0.97, 1.03),
            gyro_scale_range=(0.96, 1.04),
            time_shift_max_samples=2,
            time_mask_min_samples=3,
            time_mask_max_samples=3,
            rotation_max_degrees=3.0,
        )
        windows = make_augmented_trial_view(
            self.raw,
            self.mean,
            self.std,
            starts=[0, 4, 8],
            window_size=8,
            config=config,
            generator=torch.Generator().manual_seed(1234),
        )
        torch.testing.assert_close(windows[0, :, 4:], windows[1, :, :4])
        torch.testing.assert_close(windows[1, :, 4:], windows[2, :, :4])

    def test_time_mask_is_forced_inside_anchor(self):
        config = MotionAugmentationConfig(
            noise_std_ratio=0.0,
            acc_scale_range=(1.0, 1.0),
            gyro_scale_range=(1.0, 1.0),
            time_shift_max_samples=0,
            time_mask_min_samples=3,
            time_mask_max_samples=3,
            rotation_max_degrees=0.0,
        )
        augmented, metadata = augment_raw_trial_with_metadata(
            self.raw,
            self.mean,
            self.std,
            config,
            generator=torch.Generator().manual_seed(8),
            time_mask_anchor=(5, 11),
        )
        start = metadata.time_mask_start_sample
        length = metadata.time_mask_length_samples
        self.assertIsNotNone(start)
        self.assertGreaterEqual(start, 5)
        self.assertLessEqual(start + length, 11)
        self.assertEqual(length, 3)
        torch.testing.assert_close(
            augmented[:, start : start + length],
            self.mean[:, None].expand(-1, length),
        )
        outside = torch.ones(self.raw.shape[1], dtype=torch.bool)
        outside[start : start + length] = False
        torch.testing.assert_close(augmented[:, outside], self.raw[:, outside])

    def test_one_so3_rotation_is_shared_by_accelerometer_and_gyroscope(self):
        generator = torch.Generator().manual_seed(71)
        acceleration = torch.randn(3, 24, generator=generator)
        raw = torch.cat((acceleration, 2.0 * acceleration), dim=0)
        config = MotionAugmentationConfig(
            noise_std_ratio=0.0,
            acc_scale_range=(1.0, 1.0),
            gyro_scale_range=(1.0, 1.0),
            time_shift_max_samples=0,
            time_mask_min_samples=0,
            time_mask_max_samples=0,
            rotation_max_degrees=15.0,
        )
        rotated = augment_raw_trial(
            raw,
            torch.zeros(6),
            torch.ones(6),
            config,
            generator=torch.Generator().manual_seed(72),
        )
        self.assertFalse(torch.allclose(rotated, raw))
        torch.testing.assert_close(rotated[3:], 2.0 * rotated[:3])
        torch.testing.assert_close(
            torch.linalg.vector_norm(rotated[:3], dim=0),
            torch.linalg.vector_norm(raw[:3], dim=0),
            rtol=2e-5,
            atol=2e-6,
        )


class MotionCheckpointAndRoutingTests(unittest.TestCase):
    def checkpoint_for(self, model):
        canonical = {
            key: value.detach().clone()
            for key, value in model.state_dict().items()
        }
        compatibility = {
            key: value.detach().clone()
            for key, value in canonical.items()
        }
        return {
            "checkpoint_type": "motion_primitive_encoder",
            "schema_version": 1,
            "architecture": tiny_architecture(),
            "model": compatibility,
            "model_state_dict": canonical,
            "model_state_dict_sha256": motion_state_dict_sha256(canonical),
        }

    def test_schema_alias_and_digest_integrity_fail_closed(self):
        source = tiny_encoder()
        valid = self.checkpoint_for(source)
        canonical = validate_motion_encoder_checkpoint_integrity(valid)
        self.assertEqual(set(canonical), set(source.state_dict()))

        bad_schema = self.checkpoint_for(source)
        bad_schema["schema_version"] = 2
        with self.assertRaisesRegex(RuntimeError, "schema_version=1"):
            build_frozen_motion_encoder(
                bad_schema,
                {
                    "checkpoint_type": "motion_primitive_encoder",
                    "feature_roles": {"codebook": "content", "boundary": "segmentation"},
                },
            )

        missing_alias = self.checkpoint_for(source)
        missing_alias.pop("model_state_dict")
        with self.assertRaisesRegex(RuntimeError, "requires dictionary aliases"):
            validate_motion_encoder_checkpoint_integrity(missing_alias)

        divergent_alias = self.checkpoint_for(source)
        key = next(iter(divergent_alias["model"]))
        divergent_alias["model"][key] = divergent_alias["model"][key].clone()
        divergent_alias["model"][key].view(-1)[0] += 1
        with self.assertRaisesRegex(RuntimeError, "not exactly equivalent"):
            build_frozen_motion_encoder(
                divergent_alias,
                {
                    "checkpoint_type": "motion_primitive_encoder",
                    "feature_roles": {"codebook": "content", "boundary": "segmentation"},
                },
            )

        stale_digest = self.checkpoint_for(source)
        key = next(iter(stale_digest["model_state_dict"]))
        for alias in ("model", "model_state_dict"):
            stale_digest[alias][key] = stale_digest[alias][key].clone()
            stale_digest[alias][key].view(-1)[0] += 1
        with self.assertRaisesRegex(RuntimeError, "(?i)sha256 mismatch"):
            build_frozen_motion_encoder(
                stale_digest,
                {
                    "checkpoint_type": "motion_primitive_encoder",
                    "feature_roles": {"codebook": "content", "boundary": "segmentation"},
                },
            )

    def test_motion_checkpoint_load_is_strict_frozen_and_reproducible(self):
        torch.manual_seed(41)
        source = tiny_encoder()
        checkpoint = self.checkpoint_for(source)
        metadata = {
            "checkpoint_type": "motion_primitive_encoder",
            "feature_roles": {"codebook": "content", "boundary": "segmentation"},
        }
        loaded = build_frozen_motion_encoder(checkpoint, metadata)
        self.assertFalse(loaded.training)
        self.assertTrue(all(not parameter.requires_grad for parameter in loaded.parameters()))
        for name, expected in source.state_dict().items():
            torch.testing.assert_close(loaded.state_dict()[name], expected)

        state_keys = list(source.state_dict())
        for mutation in ("missing", "unexpected"):
            with self.subTest(mutation=mutation):
                bad_checkpoint = self.checkpoint_for(source)
                bad_checkpoint["model"] = dict(bad_checkpoint["model"])
                if mutation == "missing":
                    bad_checkpoint["model"].pop(state_keys[0])
                else:
                    bad_checkpoint["model"]["unexpected.weight"] = torch.zeros(1)
                with self.assertRaises(RuntimeError):
                    build_frozen_motion_encoder(bad_checkpoint, metadata)

    def test_run_experiment_keeps_content_and_segmentation_roles_separate(self):
        class RoleSentinelEncoder(nn.Module):
            backbone_dim = 2
            content_dim = 3
            segmentation_dim = 4

            def encode_windows(self, windows):
                count = windows.shape[0]
                return {
                    "backbone": windows.new_full((count, 2), 2.0),
                    "content": windows.new_full((count, 3), 7.0),
                    "segmentation": windows.new_full((count, 4), 11.0),
                }

        encoded = encode_motion_windows(
            RoleSentinelEncoder(),
            np.zeros((5, 6, 16), dtype=np.float32),
            torch.device("cpu"),
            batch_size=2,
        )
        self.assertEqual(set(encoded), {"backbone", "content", "segmentation"})
        np.testing.assert_array_equal(encoded["backbone"], np.full((5, 2), 2.0))
        np.testing.assert_array_equal(encoded["content"], np.full((5, 3), 7.0))
        np.testing.assert_array_equal(encoded["segmentation"], np.full((5, 4), 11.0))


class MotionSegmentationIntegrationTests(unittest.TestCase):
    def test_motion_encoder_changepoint_is_a_variable_length_method(self):
        codebook = np.asarray(
            [
                [10.0, 0.0],
                [12.0, 0.0],
                [0.0, 20.0],
                [0.0, 22.0],
                [0.0, 24.0],
            ],
            dtype=np.float32,
        )
        boundary = np.asarray(
            [
                [1.0, 0.0],
                [1.0, 0.0],
                [0.0, 1.0],
                [0.0, 1.0],
                [0.0, 1.0],
            ],
            dtype=np.float32,
        )
        segmented = build_segmented_features(
            codebook_window_features=codebook,
            boundary_window_features=boundary,
            trial_ids=np.asarray([9, 9, 9, 9, 9]),
            window_starts=np.asarray([0, 4, 8, 12, 16]),
            window_size=8,
            method="motion_encoder_changepoint",
            context_windows=1,
            threshold=0.5,
            min_segment_windows=1,
            normalize_segment_features=True,
        )
        self.assertEqual(segmented.method, "motion_encoder_changepoint")
        np.testing.assert_array_equal(segmented.segment_window_counts, [2, 3])
        np.testing.assert_array_equal(
            segmented.window_segment_ids, [0, 0, 1, 1, 1]
        )
        np.testing.assert_allclose(
            segmented.segment_features,
            np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
        )
        self.assertEqual(
            segmented.trial_statistics[0]["change_point_window_offsets"], [2]
        )

        with self.assertRaisesRegex(ValueError, "calibrated threshold"):
            build_segmented_features(
                codebook,
                boundary,
                np.asarray([9, 9, 9, 9, 9]),
                np.asarray([0, 4, 8, 12, 16]),
                8,
                "motion_encoder_changepoint",
                1,
                None,
                1,
                True,
            )


if __name__ == "__main__":
    unittest.main()
