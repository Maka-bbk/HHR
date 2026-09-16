import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from data.uschad import (
    USCHADWindowDataset,
    recompute_subject_train_normalization,
)
from models.trial_pooling import (
    TrialMeanRobustMaxPool,
    masked_feature_quantile,
)
from train_happy import experiment_metadata, validate_checkpoint_metadata


class DummyLogger:
    def info(self, _message):
        pass

    def warning(self, _message):
        pass


class RobustPoolingTests(unittest.TestCase):
    def test_masked_quantile_ignores_padding_and_matches_torch(self):
        torch.manual_seed(7)
        features = torch.randn(3, 100, 8, requires_grad=True)
        mask = torch.zeros(3, 100, dtype=torch.bool)
        mask[0, :1] = True
        mask[1, :3] = True
        mask[2, :100] = True
        features_with_large_padding = features.detach().clone()
        features_with_large_padding[~mask] = 1e6
        features_with_large_padding.requires_grad_(True)

        observed = masked_feature_quantile(
            features_with_large_padding, mask, 0.90
        )
        expected = torch.stack(
            [torch.quantile(features_with_large_padding[i, mask[i]], 0.90, dim=0)
             for i in range(3)]
        )
        self.assertTrue(torch.allclose(observed, expected))
        self.assertTrue(torch.all(observed[0] < 1e5))

        observed.sum().backward()
        self.assertIsNotNone(features_with_large_padding.grad)
        self.assertTrue(
            torch.isfinite(features_with_large_padding.grad).all().item()
        )
        self.assertEqual(
            int(features_with_large_padding.grad[~mask].abs().sum().item()), 0
        )

    def test_pool_output_shape_and_gradients(self):
        torch.manual_seed(11)
        features = torch.randn(3, 100, 256, requires_grad=True)
        mask = torch.zeros(3, 100, dtype=torch.bool)
        mask[0, :1] = True
        mask[1, :3] = True
        mask[2, :100] = True
        pool = TrialMeanRobustMaxPool(
            feature_dim=256,
            quantile=0.90,
            fusion_dim=64,
            dropout=0.0,
        )
        output, weights = pool(features, mask)
        self.assertEqual(tuple(output.shape), (3, 256))
        self.assertIsNone(weights)
        self.assertTrue(torch.isfinite(output).all().item())
        output.square().mean().backward()
        self.assertTrue(torch.isfinite(features.grad).all().item())
        self.assertEqual(int(features.grad[~mask].abs().sum().item()), 0)


class FoldNormalizationTests(unittest.TestCase):
    def test_reconstructed_raw_and_fold_statistics(self):
        rng = np.random.default_rng(19)
        raw = rng.normal(size=(12, 2, 4)).astype(np.float32)
        subjects = np.repeat(np.asarray([1, 2, 3], dtype=np.int64), 4)
        labels = np.tile(np.asarray([0, 0, 1, 1], dtype=np.int64), 3)
        saved_mean = raw.mean(axis=(0, 2), keepdims=True)
        saved_std = raw.std(axis=(0, 2), keepdims=True)
        windows = ((raw - saved_mean) / saved_std).astype(np.float32)
        original_stat_mask = np.ones(len(raw), dtype=bool)

        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "fixture.npz"
            np.savez_compressed(
                path,
                windows=windows,
                mean=saved_mean,
                std=saved_std,
                labels=labels,
                labels_1based=labels + 1,
                subject_ids=subjects,
                trial_numbers=np.arange(len(raw), dtype=np.int64),
                trial_global_ids=np.arange(len(raw), dtype=np.int64),
                window_indices=np.zeros(len(raw), dtype=np.int64),
                window_start_indices=np.zeros(len(raw), dtype=np.int64),
                stat_mask=original_stat_mask,
            )
            training_set = USCHADWindowDataset(path)
            test_set = USCHADWindowDataset(path)
            report = recompute_subject_train_normalization(
                training_set,
                test_set,
                train_subjects=[1, 2],
                train_classes=[0],
                eps=1e-6,
            )

        expected_mask = np.isin(subjects, [1, 2]) & (labels == 0)
        expected_mean = raw[expected_mask].mean(
            axis=(0, 2), keepdims=True, dtype=np.float64
        ).astype(np.float32)
        expected_std = raw[expected_mask].std(
            axis=(0, 2), keepdims=True, dtype=np.float64
        ).astype(np.float32)
        self.assertEqual(
            report["raw_source"], "reconstructed_from_npz_windows_mean_std"
        )
        self.assertTrue(np.array_equal(training_set.stat_mask, expected_mask))
        self.assertTrue(np.array_equal(test_set.stat_mask, expected_mask))
        self.assertEqual(training_set.normalization_stat_subjects, [1, 2])
        self.assertEqual(training_set.normalization_stat_classes, [0])
        self.assertTrue(
            np.allclose(training_set.normalization_mean, expected_mean, atol=1e-6)
        )
        self.assertTrue(
            np.allclose(training_set.normalization_std, expected_std, atol=1e-6)
        )
        self.assertTrue(
            np.array_equal(
                training_set.normalization_mean, test_set.normalization_mean
            )
        )
        self.assertTrue(
            np.array_equal(
                training_set.normalization_std, test_set.normalization_std
            )
        )
        reconstructed = (
            training_set.data * training_set.normalization_std
            + training_set.normalization_mean
        )
        self.assertTrue(np.allclose(reconstructed, raw, atol=2e-6))
        self.assertFalse(np.any(training_set.stat_mask[subjects == 3]))


def build_checkpoint_args(train_subjects, val_subjects, test_subjects, fold):
    return SimpleNamespace(
        uschad_npz_path="fixture.npz",
        uschad_window_size=256,
        uschad_sample_unit="trial",
        uschad_split_mode="subject",
        uschad_recompute_norm_from_train_subjects=True,
        uschad_norm_eps=1e-6,
        uschad_train_subjects=train_subjects,
        offline_val_subjects=val_subjects,
        uschad_test_subjects=test_subjects,
        uschad_cv_fold=fold,
        trial_pooling="mean_robust_max",
        trial_view_mode="full_random_crop",
        trial_crop_ratio=2.0 / 3.0,
        trial_min_windows=2,
        trial_robust_max_quantile=0.90,
        trial_pool_fusion_dim=64,
        trial_pool_fusion_dropout=0.0,
        trial_attention_dim=64,
        trial_attention_dropout=0.1,
        trial_attention_temperature=1.0,
        trial_attention_mean_mix=0.5,
        har_in_channels=6,
        har_feat_dim=256,
        har_base_channels=64,
        har_dropout=0.0,
        har_aug_mode="weak_strong",
        har_weak_jitter_std=0.0,
        har_weak_scale_std=0.1,
        har_strong_jitter_std=0.0,
        har_strong_scale_std=0.2,
        har_time_mask_ratio=0.0,
        projection_hidden_dim=2048,
        projection_bottleneck_dim=256,
        logger=DummyLogger(),
    )


class CheckpointFoldTests(unittest.TestCase):
    def test_cross_fold_checkpoint_is_rejected(self):
        fold_one_args = build_checkpoint_args(
            "1,3,4,5,6,7,8,9,12,14", "2,13", "11,10", 1
        )
        checkpoint = {"experiment_metadata": experiment_metadata(fold_one_args)}
        fold_two_args = build_checkpoint_args(
            "1,4,5,6,7,8,10,11,12,14", "3,9", "2,13", 2
        )
        with self.assertRaisesRegex(RuntimeError, "configuration mismatch"):
            validate_checkpoint_metadata(
                checkpoint, fold_two_args, "fold_one_model_best.pt"
            )


if __name__ == "__main__":
    unittest.main()
