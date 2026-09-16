"""Closed-loop tests for the canonical offline A2-MP trajectory route."""

from __future__ import annotations

import logging
import hashlib
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

import numpy as np
import torch

from experiments.motion_primitive.joint_losses import (
    MaskedTemporalPredictor,
    MotionPrimitiveLossConfig,
    _physical_changepoint_anchors,
)
from experiments.motion_primitive.train_offline import (
    build_model,
    load_trial_window_encoder,
    parse_args,
    training_step,
    validate_args,
)
from experiments.motion_primitive.offline_trainer import (
    _usage_row,
    cross_subject_positive_coverage,
)


class OfflineA2MPTests(unittest.TestCase):
    @staticmethod
    def _changepoint_outputs(descriptors: torch.Tensor):
        pair_mask = torch.ones(
            descriptors.shape[0], descriptors.shape[1] - 1, dtype=torch.bool
        )
        return [
            {
                "window_physical_descriptors": descriptors.clone(),
                "boundary_pair_mask": pair_mask.clone(),
            },
            {
                "window_physical_descriptors": descriptors.clone(),
                "boundary_pair_mask": pair_mask.clone(),
            },
        ]

    def test_changepoint_uniform_above_floor_is_not_forced_to_change(self) -> None:
        descriptors = torch.tensor(
            [[[1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 1.0]]]
        )
        stable, change, scores = _physical_changepoint_anchors(
            self._changepoint_outputs(descriptors),
            MotionPrimitiveLossConfig(),
        )
        self.assertTrue(torch.all(scores > 0.01))
        self.assertFalse(torch.any(change))
        self.assertFalse(torch.any(stable & change))

    def test_changepoint_salient_peak_survives_and_masks_are_disjoint(self) -> None:
        descriptors = torch.tensor(
            [[[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]]]
        )
        stable, change, _ = _physical_changepoint_anchors(
            self._changepoint_outputs(descriptors),
            MotionPrimitiveLossConfig(),
        )
        torch.testing.assert_close(change, torch.tensor([[False, True, False]]))
        self.assertFalse(torch.any(stable & change))

    def test_cross_subject_positive_coverage_excludes_same_trial_view(self) -> None:
        coverage = cross_subject_positive_coverage(
            torch.tensor([0, 0, 0, 1]),
            torch.tensor([1, 1, 2, 3]),
        )
        self.assertEqual(coverage["eligible_anchor_count"], 3)
        self.assertEqual(coverage["anchor_count"], 4)
        self.assertAlmostEqual(coverage["eligible_anchor_fraction"], 0.75)
        self.assertEqual(coverage["directed_pair_count"], 4)
        self.assertEqual(coverage["batch_has_cross_subject_positive"], 1)

        uncovered = cross_subject_positive_coverage(
            torch.tensor([0, 0, 1, 1]),
            torch.tensor([1, 1, 2, 2]),
        )
        self.assertEqual(uncovered["eligible_anchor_count"], 0)
        self.assertEqual(uncovered["batch_has_cross_subject_positive"], 0)

    def test_initialization_is_an_explicit_experimental_decision(self) -> None:
        with tempfile.NamedTemporaryFile(suffix=".npz") as source:
            common = [
                "--npz-path", source.name,
                "--output-dir", str(Path(source.name).parent / "out"),
                "--train-subjects", "1,2",
                "--val-subjects", "3",
                "--test-subjects", "4",
            ]
            with self.assertRaises(SystemExit):
                parse_args(common)
            args = validate_args(parse_args(common + ["--encoder-initialization", "random"]))
            self.assertEqual(args.encoder_initialization, "random")
            self.assertEqual((args.window_size, args.window_stride), (256, 128))
            self.assertEqual(args.selection_head, "trajectory")
            self.assertFalse(hasattr(args, "trial_pooling"))
            self.assertFalse(hasattr(args, "pooled_aux_weight"))
            self.assertFalse(hasattr(args, "fusion_trajectory_weight"))

    def test_registered_a2_mp_weights_are_really_active(self) -> None:
        config = MotionPrimitiveLossConfig().validated()
        self.assertEqual(config.changepoint_weight, 1.0)
        self.assertEqual(config.content_boundary_alignment_weight, 0.10)
        self.assertEqual(config.noncollapse_weight, 0.05)
        self.assertEqual(config.temporal_prediction_weight, 0.50)
        audit = config.audit_dict()
        self.assertTrue(audit["a2_mother_route"])
        self.assertFalse(audit["complete_trial_pooling_used"])
        self.assertFalse(audit["instance_infonce_used"])

    def test_one_backward_reaches_encoder_boundary_codebook_trajectory_and_predictor(self) -> None:
        torch.manual_seed(9)
        with tempfile.NamedTemporaryFile(suffix=".npz") as source:
            args = validate_args(
                parse_args(
                    [
                        "--npz-path", source.name,
                        "--output-dir", str(Path(source.name).parent / "out"),
                        "--train-subjects", "1,2",
                        "--val-subjects", "3",
                        "--test-subjects", "4",
                        "--encoder-initialization", "random",
                        "--feature-dim", "8",
                        "--base-channels", "2",
                        "--codebook-size", "4",
                        "--trajectory-input-dim", "7",
                        "--trajectory-hidden-dim", "7",
                        "--run-state-dim", "3",
                        "--batch-size", "4",
                        "--epochs", "2",
                    ]
                )
            )
        model = build_model(args)
        predictor = MaskedTemporalPredictor(8, 6)
        optimizer = torch.optim.SGD(
            list(model.parameters()) + list(predictor.parameters()), lr=0.01
        )
        mask = torch.ones(4, 4, dtype=torch.bool)
        starts = torch.tensor([[0, 128, 256, 384]]).expand(4, -1)
        labels = torch.tensor([0, 0, 1, 1])
        subjects = torch.tensor([1, 2, 1, 2])
        clean = torch.randn(4, 4, 6, 256)
        views = []
        for windows in (clean, clean * 1.02):
            views.append(
                {
                    "windows": windows,
                    "positions": model.relative_positions(starts, mask, 256),
                    "mask": mask,
                    "lengths": mask.sum(dim=1),
                }
            )
        config = MotionPrimitiveLossConfig().validated()
        metrics = training_step(
            model,
            views,
            labels,
            optimizer,
            args,
            epoch_index=0,
            loss_config=config,
            subject_ids=subjects,
            temporal_predictor=predictor,
        )
        for name in (
            "changepoint",
            "content_boundary_alignment",
            "noncollapse",
            "temporal_prediction",
            "trajectory_ce",
            "vq_commitment",
        ):
            self.assertIn(name, metrics)
            self.assertTrue(torch.isfinite(torch.tensor(metrics[name])), name)
        modules = (
            model.window_encoder,
            model.boundary_head,
            model.codebook,
            model.trajectory_classifier,
            predictor,
        )
        for module in modules:
            total = sum(
                float(parameter.grad.abs().sum())
                for parameter in module.parameters()
                if parameter.grad is not None
            )
            self.assertGreater(total, 0.0, type(module).__name__)

    def test_a2_backbone_prefix_is_accepted_for_warmstart(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            npz = root / "data.npz"
            npz.touch()
            # Construct source/target before validation because validation also
            # verifies that the checkpoint already exists.
            random_args = validate_args(
                parse_args(
                    [
                        "--npz-path", str(npz), "--output-dir", str(root / "r"),
                        "--train-subjects", "1,2", "--val-subjects", "3",
                        "--test-subjects", "4", "--encoder-initialization", "random",
                        "--feature-dim", "8", "--base-channels", "2",
                    ]
                )
            )
            source = build_model(random_args)
            checkpoint = root / "a2.pt"
            torch.save(
                {
                    "model_state_dict": {
                        "backbone." + key: value.clone()
                        for key, value in source.window_encoder.state_dict().items()
                    },
                    "experiment_metadata": {
                        "uschad_window_size": 256,
                        "har_in_channels": 6,
                        "har_feat_dim": 8,
                        "har_base_channels": 2,
                        "har_dropout": 0.0,
                        "uschad_split_mode": "subject",
                        "uschad_recompute_norm_from_train_subjects": True,
                        "uschad_norm_eps": 1.0e-6,
                        "uschad_train_subjects": [1, 2],
                        "offline_val_subjects": [3],
                        "uschad_test_subjects": [4],
                        "uschad_cv_fold": -1,
                        "seed": 0,
                    },
                    "npz_sha256": hashlib.sha256(npz.read_bytes()).hexdigest(),
                    "split_audit": {
                        "old_class_ids_0based": [0, 1, 2, 3, 4, 5],
                        "window_size_samples": 256,
                        "npz_sha256": hashlib.sha256(npz.read_bytes()).hexdigest(),
                        "train_subjects": [1, 2],
                        "validation_subjects": [3],
                        "outer_test_subjects_metadata_only": [4],
                    },
                },
                checkpoint,
            )
            warm_args = validate_args(
                parse_args(
                    [
                        "--npz-path", str(npz), "--output-dir", str(root / "w"),
                        "--train-subjects", "1,2", "--val-subjects", "3",
                        "--test-subjects", "4", "--encoder-initialization", "warmstart",
                        "--trial-encoder-checkpoint", str(checkpoint),
                        "--feature-dim", "8", "--base-channels", "2",
                    ]
                )
            )
            target = build_model(warm_args)
            load_trial_window_encoder(target, str(checkpoint), warm_args, logging.getLogger())
            for key, expected in source.window_encoder.state_dict().items():
                torch.testing.assert_close(target.window_encoder.state_dict()[key], expected)

    def test_warmstart_provenance_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            npz = root / "data.npz"
            npz.write_bytes(b"current-dataset")
            args = validate_args(
                parse_args(
                    [
                        "--npz-path", str(npz), "--output-dir", str(root / "out"),
                        "--train-subjects", "1,2", "--val-subjects", "3",
                        "--test-subjects", "4", "--encoder-initialization", "random",
                        "--feature-dim", "8", "--base-channels", "2",
                    ]
                )
            )
            source = build_model(args)
            state = {
                "backbone." + key: value.clone()
                for key, value in source.window_encoder.state_dict().items()
            }
            metadata = {
                "uschad_window_size": 256, "har_in_channels": 6,
                "har_feat_dim": 8, "har_base_channels": 2, "har_dropout": 0.0,
                "uschad_recompute_norm_from_train_subjects": True,
                "uschad_norm_eps": 1.0e-6, "uschad_train_subjects": [1, 2],
                "offline_val_subjects": [3], "uschad_test_subjects": [4],
                "uschad_cv_fold": -1, "seed": 0,
            }
            current_hash = hashlib.sha256(npz.read_bytes()).hexdigest()
            base_audit = {
                "old_class_ids_0based": [0, 1, 2, 3, 4, 5],
                "window_size_samples": 256,
                "npz_sha256": current_hash,
                "train_subjects": [1, 2],
                "validation_subjects": [3],
                "outer_test_subjects_metadata_only": [4],
            }
            warm_args = parse_args(
                [
                    "--npz-path", str(npz), "--output-dir", str(root / "warm"),
                    "--train-subjects", "1,2", "--val-subjects", "3",
                    "--test-subjects", "4", "--encoder-initialization", "warmstart",
                    "--trial-encoder-checkpoint", str(root / "checkpoint.pt"),
                    "--feature-dim", "8", "--base-channels", "2",
                ]
            )
            checkpoint = root / "checkpoint.pt"
            cases = {
                "missing": {"model_state_dict": state, "experiment_metadata": metadata},
                "old-class": {
                    "model_state_dict": state, "experiment_metadata": metadata,
                    "npz_sha256": current_hash,
                    "split_audit": {**base_audit, "old_class_ids_0based": [0, 1]},
                },
                "npz": {
                    "model_state_dict": state, "experiment_metadata": metadata,
                    "npz_sha256": "0" * 64,
                    "split_audit": {**base_audit, "npz_sha256": "0" * 64},
                },
            }
            for name, payload in cases.items():
                with self.subTest(name=name):
                    torch.save(payload, checkpoint)
                    validated = validate_args(warm_args)
                    with self.assertRaises(RuntimeError):
                        load_trial_window_encoder(
                            build_model(validated), str(checkpoint), validated,
                            logging.getLogger(),
                        )

    def test_clean_anchor_ramp_and_dead_beta_are_fail_closed(self) -> None:
        parser_args = [
            "--npz-path", __file__, "--output-dir", str(Path(__file__).parent / "out"),
            "--train-subjects", "1,2", "--val-subjects", "3",
            "--test-subjects", "4", "--encoder-initialization", "random",
        ]
        with self.assertRaises(SystemExit):
            parse_args(parser_args + ["--har-weak-jitter-std", "0.01"])
        with self.assertRaises(SystemExit):
            parse_args(parser_args + ["--har-weak-scale-std", "0.01"])
        with self.assertRaises(SystemExit):
            parse_args(parser_args + ["--codebook-commitment-beta", "0.25"])
        with self.assertRaises(ValueError):
            validate_args(parse_args(parser_args + ["--motion-ramp-end-epoch", "1"]))
        with self.assertRaises(ValueError):
            validate_args(parse_args(parser_args + ["--encoder-lr-scale", "0.1"]))

    def test_capacity_is_not_reported_as_realized_code_count(self) -> None:
        diagnostics = _usage_row(np.asarray([0, 0, 2, 2, 2]), capacity_k=4)
        self.assertEqual(diagnostics["capacity_k"], 4)
        self.assertEqual(diagnostics["used_k"], 2)
        self.assertEqual(diagnostics["used_ids"], [0, 2])
        self.assertAlmostEqual(diagnostics["dead_fraction"], 0.5)
        self.assertLessEqual(diagnostics["perplexity_effective_k"], 2.0)


if __name__ == "__main__":
    unittest.main()
