import json
import tempfile
import unittest
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from experiments.motion_primitive.core import (
    assign_to_codebook,
    association_summary,
    build_trial_sequence,
    cross_subject_nearest_neighbor_accuracy,
    distance_matrix_from_sequences,
    fit_weighted_pca,
    inverse_trial_frequency_weights,
    normalized_levenshtein,
    normalized_weighted_levenshtein,
    run_length_encode,
    shuffle_valid_rle_tokens,
)
from experiments.motion_primitive.run_experiment import (
    build_trajectory_plot_arrays,
    build_frozen_encoder,
    build_split_masks,
    load_npz_arrays,
    save_trajectory_sequence_plot,
)
from experiments.motion_primitive.run_subject_cv import (
    aggregate_rows,
    read_decomposition_metrics,
    validate_existing_run,
)
from experiments.motion_primitive.segmentation import (
    build_segmented_features,
    codebook_segment_weights,
    feature_change_scores,
    select_changepoints,
    train_masked_denoising_adapter,
    transform_with_adapter,
)
from models.resnet1d import ResNet1D


class MotionPrimitiveCoreTests(unittest.TestCase):
    def test_run_length_sequence_keeps_duration_and_observed_span(self):
        tokens = np.asarray([3, 3, 3, 7, 7, 2], dtype=np.int64)
        starts = np.arange(len(tokens), dtype=np.int64) * 128
        run_tokens, run_lengths = run_length_encode(tokens)
        np.testing.assert_array_equal(run_tokens, [3, 7, 2])
        np.testing.assert_array_equal(run_lengths, [3, 2, 1])

        sequence = build_trial_sequence(
            tokens, starts, window_size=256, sample_rate_hz=100.0
        )
        self.assertEqual(sequence["rle_tokens"], [3, 7, 2])
        self.assertEqual(sequence["run_lengths"], [3, 2, 1])
        self.assertAlmostEqual(sequence["runs"][0]["observed_span_seconds"], 5.12)
        self.assertAlmostEqual(sequence["observed_trial_span_seconds"], 8.96)

    def test_normalized_levenshtein(self):
        self.assertEqual(normalized_levenshtein([], []), 0.0)
        self.assertEqual(normalized_levenshtein([1, 2], [1, 2]), 0.0)
        self.assertAlmostEqual(normalized_levenshtein([1, 2], [1, 3]), 0.5)
        self.assertAlmostEqual(normalized_levenshtein([1], [1, 2]), 0.5)

    def test_weighted_levenshtein_hard_cost_matches_legacy(self):
        hard_costs = np.ones((8, 8), dtype=np.float64)
        np.fill_diagonal(hard_costs, 0.0)
        pairs = [
            ([], []),
            ([], [1, 2]),
            ([1, 2], []),
            ([1, 2], [1, 2]),
            ([1, 2], [1, 3]),
            ([1], [1, 2]),
            ([7, 2, 7, 4], [7, 3, 4]),
        ]
        for left, right in pairs:
            with self.subTest(left=left, right=right):
                self.assertAlmostEqual(
                    normalized_weighted_levenshtein(left, right, hard_costs),
                    normalized_levenshtein(left, right),
                )

    def test_weighted_levenshtein_supports_soft_and_position_costs(self):
        soft_costs = np.ones((3, 3), dtype=np.float64)
        np.fill_diagonal(soft_costs, 0.0)
        soft_costs[1, 2] = 0.25
        self.assertAlmostEqual(
            normalized_weighted_levenshtein([0, 1], [0, 2], soft_costs),
            0.125,
        )

        observed_positions = set()

        def position_cost(left_token, right_token, left_index, right_index):
            observed_positions.add(
                (left_token, right_token, left_index, right_index)
            )
            return 0.2 if left_index == right_index else 1.0

        self.assertAlmostEqual(
            normalized_weighted_levenshtein([4, 4], [5, 5], position_cost),
            0.2,
        )
        self.assertIn((4, 5, 0, 0), observed_positions)
        self.assertIn((4, 5, 1, 1), observed_positions)
        self.assertEqual(
            normalized_weighted_levenshtein([1], [], position_cost), 1.0
        )

    def test_weighted_levenshtein_validates_costs_and_token_bounds(self):
        valid = np.asarray([[0.0, 1.0], [1.0, 0.0]], dtype=np.float64)
        self.assertEqual(normalized_weighted_levenshtein([], [], valid), 0.0)
        with self.assertRaisesRegex(ValueError, "shape"):
            normalized_weighted_levenshtein([0], [0], np.ones((2, 3)))
        with self.assertRaisesRegex(ValueError, "shape"):
            normalized_weighted_levenshtein([], [], np.empty((0, 0)))
        with self.assertRaisesRegex(ValueError, "finite and non-negative"):
            invalid = valid.copy()
            invalid[0, 1] = np.nan
            normalized_weighted_levenshtein([0], [1], invalid)
        with self.assertRaisesRegex(ValueError, "finite and non-negative"):
            invalid = valid.copy()
            invalid[0, 1] = -0.1
            normalized_weighted_levenshtein([0], [1], invalid)
        with self.assertRaisesRegex(ValueError, "outside"):
            normalized_weighted_levenshtein([-1], [], valid)
        with self.assertRaisesRegex(ValueError, "outside"):
            normalized_weighted_levenshtein([2], [0], valid)

        for invalid_cost in [np.nan, -0.1]:
            with self.subTest(callback_cost=invalid_cost):
                with self.assertRaisesRegex(
                    ValueError, "finite and non-negative"
                ):
                    normalized_weighted_levenshtein(
                        [0], [1], lambda *_: invalid_cost
                    )
        with self.assertRaisesRegex(ValueError, "scalar"):
            normalized_weighted_levenshtein(
                [0], [1], lambda *_: np.asarray([0.5])
            )

    def test_rle_shuffle_preserves_runs_without_equal_neighbors(self):
        tokens = [1, 2, 1, 3, 1, 4]
        rng = np.random.default_rng(17)
        for _ in range(100):
            shuffled = shuffle_valid_rle_tokens(tokens, rng)
            self.assertEqual(Counter(shuffled), Counter(tokens))
            self.assertEqual(len(shuffled), len(tokens))
            self.assertTrue(
                all(left != right for left, right in zip(shuffled, shuffled[1:]))
            )
        with self.assertRaisesRegex(ValueError, "not valid RLE"):
            shuffle_valid_rle_tokens([1, 1, 2], rng)
        first_rng = np.random.default_rng(99)
        second_rng = np.random.default_rng(99)
        self.assertEqual(
            [shuffle_valid_rle_tokens(tokens, first_rng) for _ in range(20)],
            [shuffle_valid_rle_tokens(tokens, second_rng) for _ in range(20)],
        )

    def test_cross_subject_1nn_splits_credit_across_distance_ties(self):
        matrix = np.asarray(
            [
                [0.0, 0.2, 0.1, 0.1],
                [0.2, 0.0, 0.1, 0.1],
                [0.1, 0.1, 0.0, 0.2],
                [0.1, 0.1, 0.2, 0.0],
            ]
        )
        accuracy, nearest = cross_subject_nearest_neighbor_accuracy(
            matrix,
            labels=[0, 1, 0, 1],
            subject_ids=[1, 1, 2, 2],
        )
        self.assertEqual(accuracy, 0.5)
        self.assertTrue(np.all(nearest >= 0))

    def test_per_trial_weights_give_each_trial_equal_mass(self):
        trial_ids = np.asarray([10, 10, 10, 20], dtype=np.int64)
        weights = inverse_trial_frequency_weights(trial_ids)
        self.assertAlmostEqual(float(weights[trial_ids == 10].sum()), 2.0)
        self.assertAlmostEqual(float(weights[trial_ids == 20].sum()), 2.0)
        self.assertAlmostEqual(float(weights.mean()), 1.0)

    def test_weighted_pca_and_cosine_assignment_shapes(self):
        values = np.asarray(
            [[1.0, 0.0, 0.0], [0.9, 0.1, 0.0], [0.0, 1.0, 0.0], [0.0, 0.9, 0.1]],
            dtype=np.float32,
        )
        pca = fit_weighted_pca(values, n_components=2, sample_weights=np.ones(4))
        transformed = pca.transform(values)
        self.assertEqual(transformed.shape, (4, 2))
        tokens, distances, centers = assign_to_codebook(
            values[:, :2], np.asarray([[1.0, 0.0], [0.0, 1.0]]), "cosine"
        )
        np.testing.assert_array_equal(tokens, [0, 0, 1, 1])
        self.assertEqual(distances.shape, (4,))
        self.assertEqual(centers.shape, (2, 2))

    def test_same_activity_sequences_are_closer_across_subjects(self):
        sequences = [
            [1, 2, 3],
            [7, 8, 9],
            [1, 2, 3],
            [7, 8, 9],
            [1, 2, 3],
            [7, 8, 9],
        ]
        labels = np.asarray([0, 1, 0, 1, 0, 1], dtype=np.int64)
        subjects = np.asarray([1, 1, 2, 2, 3, 3], dtype=np.int64)
        matrix = distance_matrix_from_sequences(sequences)
        summary = association_summary(matrix, labels, subjects, [0, 1])
        self.assertEqual(summary["same_mean"], 0.0)
        self.assertGreater(summary["different_mean"], summary["same_mean"])
        self.assertEqual(summary["cross_subject_1nn_activity_accuracy"], 1.0)

    def test_frozen_encoder_strictly_loads_window_checkpoint_prefix(self):
        source = ResNet1D(in_channels=6, feat_dim=16, base_channels=8)
        checkpoint = {
            "model": {f"0.{key}": value.clone() for key, value in source.state_dict().items()}
        }
        metadata = {
            "har_in_channels": 6,
            "har_feat_dim": 16,
            "har_base_channels": 8,
            "har_dropout": 0.0,
        }
        observed = build_frozen_encoder(checkpoint, metadata)
        self.assertFalse(observed.training)
        self.assertTrue(all(not parameter.requires_grad for parameter in observed.parameters()))
        for key, value in source.state_dict().items():
            self.assertTrue(torch.equal(value, observed.state_dict()[key]))


class MotionPrimitiveSegmentationTests(unittest.TestCase):
    def test_change_scores_and_minimum_gap_find_obvious_boundary(self):
        features = np.asarray(
            [[1.0, 0.0]] * 4 + [[0.0, 1.0]] * 4, dtype=np.float32
        )
        scores = feature_change_scores(features, context_windows=2)
        self.assertEqual(int(np.argmax(scores)) + 1, 4)
        boundaries = select_changepoints(
            scores, threshold=0.5, min_segment_windows=2
        )
        np.testing.assert_array_equal(boundaries, [4])

        flat = feature_change_scores(
            np.asarray([[1.0, 0.0]] * 8, dtype=np.float32), context_windows=2
        )
        self.assertEqual(select_changepoints(flat, 0.0, 2).tolist(), [])

    def test_segment_features_cover_trials_and_expand_without_crossing(self):
        codebook = np.asarray(
            [
                [1.0, 0.0],
                [1.0, 0.0],
                [0.0, 1.0],
                [0.0, 1.0],
                [1.0, 1.0],
                [1.0, 1.0],
            ],
            dtype=np.float32,
        )
        boundary = codebook.copy()
        trial_ids = np.asarray([10, 10, 10, 10, 20, 20], dtype=np.int64)
        starts = np.asarray([0, 128, 256, 384, 0, 128], dtype=np.int64)
        segmented = build_segmented_features(
            codebook,
            boundary,
            trial_ids,
            starts,
            window_size=256,
            method="ssl_feature_changepoint",
            context_windows=1,
            threshold=0.5,
            min_segment_windows=2,
            normalize_segment_features=True,
        )
        self.assertEqual(segmented.segment_window_counts.tolist(), [2, 2, 2])
        self.assertEqual(segmented.window_segment_ids.tolist(), [0, 0, 1, 1, 2, 2])
        self.assertEqual(segmented.segment_trial_ids.tolist(), [10, 10, 20])
        self.assertEqual(sum(segmented.segment_window_counts), len(codebook))
        self.assertTrue(
            np.all(
                segmented.segment_last_window_positions
                >= segmented.segment_first_window_positions
            )
        )

        segment_tokens = np.asarray([4, 9, 7], dtype=np.int64)
        expanded = segment_tokens[segmented.window_segment_ids]
        np.testing.assert_array_equal(expanded, [4, 4, 9, 9, 7, 7])

    def test_fixed_window_segmentation_is_exact_legacy_grid(self):
        features = np.eye(4, dtype=np.float32)
        segmented = build_segmented_features(
            features,
            features,
            trial_ids=np.asarray([1, 1, 2, 2]),
            window_starts=np.asarray([0, 128, 0, 128]),
            window_size=256,
            method="fixed_window",
            context_windows=2,
            threshold=None,
            min_segment_windows=2,
            normalize_segment_features=True,
        )
        np.testing.assert_array_equal(segmented.window_segment_ids, [0, 1, 2, 3])
        np.testing.assert_allclose(segmented.segment_features, features)
        np.testing.assert_array_equal(segmented.segment_window_counts, [1, 1, 1, 1])

    def test_segment_per_trial_weights_equalize_trial_mass(self):
        features = np.asarray(
            [[1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [0.0, 0.9]],
            dtype=np.float32,
        )
        segmented = build_segmented_features(
            features,
            features,
            trial_ids=np.asarray([1, 1, 1, 2]),
            window_starts=np.asarray([0, 128, 256, 0]),
            window_size=256,
            method="fixed_window",
            context_windows=1,
            threshold=None,
            min_segment_windows=1,
            normalize_segment_features=True,
        )
        weights = codebook_segment_weights(segmented, "per_trial")
        self.assertAlmostEqual(float(weights[segmented.segment_trial_ids == 1].sum()), 2.0)
        self.assertAlmostEqual(float(weights[segmented.segment_trial_ids == 2].sum()), 2.0)

        variable_features = np.asarray(
            [[1.0, 0.0]] * 2
            + [[0.0, 1.0]] * 4
            + [[1.0, 1.0]] * 2,
            dtype=np.float32,
        )
        variable = build_segmented_features(
            variable_features,
            variable_features,
            trial_ids=np.asarray([1] * 6 + [2] * 2),
            window_starts=np.asarray([0, 128, 256, 384, 512, 640, 0, 128]),
            window_size=256,
            method="ssl_feature_changepoint",
            context_windows=1,
            threshold=0.5,
            min_segment_windows=2,
            normalize_segment_features=True,
        )
        variable_weights = codebook_segment_weights(variable, "per_trial")
        self.assertAlmostEqual(
            float(variable_weights[variable.segment_trial_ids == 1].sum()),
            float(variable_weights[variable.segment_trial_ids == 2].sum()),
        )
        trial_one = variable_weights[variable.segment_trial_ids == 1]
        self.assertAlmostEqual(float(trial_one[1] / trial_one[0]), 2.0)

    def test_masked_denoising_adapter_is_train_only_shape_safe_and_deterministic(self):
        values = np.asarray(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.9, 0.1, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.9, 0.1, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.9, 0.1],
            ],
            dtype=np.float32,
        )
        kwargs = dict(
            fit_embeddings=values,
            sample_weights=np.ones(len(values), dtype=np.float32),
            output_dim=2,
            epochs=2,
            batch_size=3,
            learning_rate=1e-3,
            mask_ratio=0.1,
            noise_std=0.01,
            device=torch.device("cpu"),
            seed=17,
        )
        first = train_masked_denoising_adapter(**kwargs)
        second = train_masked_denoising_adapter(**kwargs)
        first_values = transform_with_adapter(first, values, torch.device("cpu"), 3)
        second_values = transform_with_adapter(second, values, torch.device("cpu"), 3)
        self.assertEqual(first_values.shape, (6, 2))
        self.assertTrue(np.isfinite(first_values).all())
        np.testing.assert_allclose(first_values, second_values, atol=1e-6)
        self.assertFalse(first.training["uses_activity_labels"])

    def test_trajectory_plot_arrays_keep_duration_and_explicit_boundaries(self):
        record = {
            "trial_key": "S01-A01-T1",
            "activity_label_1based": 1,
            "activity_name": "walk",
            "subject_id": 1,
            "trial_number": 1,
            "primitive_segmentation": {
                "segments": [
                    {"first_window_offset": 0},
                    {"first_window_offset": 2},
                ]
            },
            "sequence_variants": {"full": {"raw_tokens": [4, 4, 9, 9, 9]}},
        }
        _, matrix, boundaries = build_trajectory_plot_arrays([record])
        np.testing.assert_array_equal(matrix[0], [4, 4, 9, 9, 9])
        self.assertTrue(boundaries[0, 2])
        self.assertEqual(int(boundaries.sum()), 1)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trajectory.png"
            saved = save_trajectory_sequence_plot(
                path,
                [record],
                primitive_num=32,
                stride_seconds=1.28,
                primitive_segmentation="ssl_feature_changepoint",
            )
            self.assertTrue(saved)
            self.assertGreater(path.stat().st_size, 1000)


class MotionPrimitiveDataSafetyTests(unittest.TestCase):
    def test_split_mask_uses_train_old_only_and_disjoint_eval_subject(self):
        arrays = {
            "labels": np.asarray([0, 6, 0, 6], dtype=np.int64),
            "labels_1based": np.asarray([1, 7, 1, 7], dtype=np.int64),
            "subject_ids": np.asarray([1, 1, 2, 2], dtype=np.int64),
            "trial_numbers": np.asarray([1, 1, 1, 1], dtype=np.int64),
            "trial_global_ids": np.asarray([10, 11, 20, 21], dtype=np.int64),
        }
        fit, evaluation, details = build_split_masks(
            arrays, [1], [2], old_class_count=6, anomaly_policy="report"
        )
        np.testing.assert_array_equal(fit, [True, False, False, False])
        np.testing.assert_array_equal(evaluation, [False, False, True, True])
        self.assertEqual(details["subject_overlap"], [])
        with self.assertRaises(RuntimeError):
            build_split_masks(arrays, [1, 2], [2], 6, "report")

    def test_npz_per_window_activity_names_are_collapsed_by_label(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tiny.npz"
            np.savez(
                path,
                windows=np.zeros((4, 2, 3), dtype=np.float32),
                labels=np.asarray([0, 0, 1, 1]),
                labels_1based=np.asarray([1, 1, 2, 2]),
                subject_ids=np.asarray([1, 2, 1, 2]),
                trial_numbers=np.asarray([1, 1, 1, 1]),
                trial_global_ids=np.asarray([10, 20, 11, 21]),
                window_indices=np.zeros(4, dtype=np.int64),
                window_start_indices=np.zeros(4, dtype=np.int64),
                mean=np.zeros((1, 2, 1), dtype=np.float32),
                std=np.ones((1, 2, 1), dtype=np.float32),
                activity_names=np.asarray(["walk", "walk", "run", "run"], dtype=object),
            )
            arrays = load_npz_arrays(path)
            self.assertEqual(arrays["activity_names"].tolist(), ["walk", "run"])

    def test_cv_aggregation_recovers_multiple_primitive_trial_ratio(self):
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            records = [
                {
                    "sequence_variants": {
                        "full": {"unique_primitive_count": 1, "run_count": 1}
                    }
                },
                {
                    "sequence_variants": {
                        "full": {"unique_primitive_count": 3, "run_count": 5}
                    }
                },
            ]
            payload = "\n".join(json.dumps(record) for record in records) + "\n"
            (run_dir / "trial_primitive_sequences.jsonl").write_text(
                payload, encoding="utf-8"
            )
            metrics = read_decomposition_metrics(run_dir)
            self.assertEqual(metrics["decomposition_trial_count"], 2)
            self.assertEqual(metrics["unique_primitives_mean"], 2.0)
            self.assertEqual(metrics["run_count_mean"], 3.0)
            self.assertEqual(metrics["multiple_primitive_trial_ratio"], 0.5)
            self.assertEqual(metrics["multiple_run_trial_ratio"], 0.5)

    def test_cv_aggregation_rejects_unbalanced_or_duplicate_seed_grid(self):
        with self.assertRaisesRegex(RuntimeError, "Duplicate"):
            aggregate_rows([{"fold": 1, "seed": 0}, {"fold": 1, "seed": 0}])
        with self.assertRaisesRegex(RuntimeError, "Unbalanced"):
            aggregate_rows(
                [
                    {"fold": 1, "seed": 0},
                    {"fold": 2, "seed": 0},
                    {"fold": 2, "seed": 5},
                ]
            )

    def test_skip_existing_requires_matching_configuration(self):
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory) / "run"
            run_dir.mkdir()
            checkpoint = Path(directory) / "model_best.pt"
            checkpoint.touch()
            arguments = {
                "seed": 5,
                "primitive_num": 32,
                "pca_dim": 64,
                "label_permutations": 1000,
                "order_shuffles": 50,
                "batch_size": 512,
                "device": "auto",
                "anomaly_policy": "report",
                "old_class_count": 6,
                "embedding_normalization": "l2",
                "codebook_weighting": "per_trial",
                "kmeans_n_init": 20,
                "kmeans_max_iter": 300,
                "edge_trim_ratio": 0.10,
                "sample_rate_hz": 100.0,
            }
            config = {
                "checkpoint": str(checkpoint.resolve()),
                "arguments": arguments,
                "order_shuffle_control": {
                    "algorithm": "valid_rle_permutation_v2"
                },
            }
            (run_dir / "experiment_config.json").write_text(
                json.dumps(config), encoding="utf-8"
            )
            for name in [
                "summary.json",
                "trial_primitive_sequences.jsonl",
                "activity_sequence_distance_matrix.csv",
            ]:
                (run_dir / name).write_text("{}", encoding="utf-8")
            args = SimpleNamespace(
                primitive_num=32,
                pca_dim=64,
                label_permutations=1000,
                order_shuffles=50,
                batch_size=512,
                device="auto",
                anomaly_policy="report",
            )
            validate_existing_run(run_dir, checkpoint, fold=1, seed=5, args=args)
            args.pca_dim = 16
            with self.assertRaisesRegex(RuntimeError, "incompatible"):
                validate_existing_run(run_dir, checkpoint, fold=1, seed=5, args=args)
            args.pca_dim = 64
            args.primitive_segmentation = "ssl_feature_changepoint"
            args.ssl_feature_dim = 64
            args.ssl_epochs = 25
            args.ssl_learning_rate = 1e-3
            args.ssl_mask_ratio = 0.15
            args.ssl_noise_std = 0.02
            args.changepoint_context_windows = 2
            args.changepoint_score_quantile = 0.90
            args.changepoint_min_segment_windows = 2
            with self.assertRaisesRegex(RuntimeError, "incompatible"):
                validate_existing_run(run_dir, checkpoint, fold=1, seed=5, args=args)


if __name__ == "__main__":
    unittest.main()
