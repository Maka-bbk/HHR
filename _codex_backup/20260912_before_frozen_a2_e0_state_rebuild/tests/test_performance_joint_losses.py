"""Regression tests for the trajectory-only A2-MP objective."""

from __future__ import annotations

import unittest

import torch

from experiments.motion_primitive.joint_losses import (
    MotionPrimitiveLossConfig,
    _physical_changepoint_anchors,
    compose_joint_loss,
    sample_temporal_prediction_mask,
)


class TrajectoryLossTests(unittest.TestCase):
    def test_default_has_a2_constraints_and_no_infonce_or_pooling(self) -> None:
        config = MotionPrimitiveLossConfig()
        self.assertGreater(config.changepoint_weight, 0.0)
        self.assertGreater(config.content_boundary_alignment_weight, 0.0)
        self.assertGreater(config.noncollapse_weight, 0.0)
        self.assertGreater(config.temporal_prediction_weight, 0.0)
        self.assertFalse(hasattr(config, "happy_infonce_weight"))
        self.assertFalse(hasattr(config, "pooled_aux_weight"))

    def test_minimal_composition_needs_only_trajectory_and_vq_outputs(self) -> None:
        torch.manual_seed(7)
        labels = torch.tensor([0, 1])
        logits = []
        outputs = []
        for _ in range(2):
            value = torch.randn(2, 2, requires_grad=True)
            logits.append(value)
            outputs.append(
                {
                    "trajectory_logits": value,
                    "trajectory_embedding": torch.randn(2, 4, requires_grad=True),
                    "commitment_loss": torch.zeros((), requires_grad=True),
                    "codebook_embedding_loss": torch.zeros((), requires_grad=True),
                    "boundary_pair_mask": torch.zeros(2, 0, dtype=torch.bool),
                    "final_boundary_probabilities": torch.zeros(2, 0),
                }
            )
        config = MotionPrimitiveLossConfig(
            trajectory_supcon_weight=0.0,
            changepoint_weight=0.0,
            content_boundary_alignment_weight=0.0,
            noncollapse_weight=0.0,
            temporal_prediction_weight=0.0,
            effective_minimum_duration_weight=0.0,
            vq_commitment_weight=0.0,
            vq_codebook_weight=0.0,
            transition_budget_weight=0.0,
        )
        result = compose_joint_loss(
            outputs,
            labels,
            torch.ones(2, dtype=torch.bool),
            config,
            epoch=0,
            total_epochs=1,
        )
        result.total.backward()
        for value in logits:
            self.assertGreater(float(value.grad.abs().sum()), 0.0)

    def test_temporal_mask_is_nonempty_and_never_selects_padding(self) -> None:
        torch.manual_seed(3)
        valid = torch.tensor([[True, True, True, False], [True, False, False, False]])
        selected = sample_temporal_prediction_mask(valid, 0.2)
        self.assertTrue(torch.all(selected <= valid))
        self.assertTrue(torch.all(selected.sum(dim=1) == 1))

    def test_null_gate_allows_zero_changes_but_keeps_salient_change(self) -> None:
        pair_mask = torch.ones(1, 3, dtype=torch.bool)

        def outputs(descriptors: torch.Tensor):
            return [
                {
                    "window_physical_descriptors": descriptors.clone(),
                    "boundary_pair_mask": pair_mask.clone(),
                }
                for _ in range(2)
            ]

        low_amplitude = torch.tensor(
            [[[1.0, 0.0], [1.0, 0.001], [1.0, -0.001], [1.0, 0.0005]]]
        )
        _, low_change, low_score = _physical_changepoint_anchors(
            outputs(low_amplitude), MotionPrimitiveLossConfig()
        )
        self.assertLess(float(low_score.max()), 0.01)
        self.assertEqual(int(low_change.sum()), 0)

        salient = torch.tensor(
            [[[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]]]
        )
        _, salient_change, salient_score = _physical_changepoint_anchors(
            outputs(salient), MotionPrimitiveLossConfig()
        )
        self.assertGreater(float(salient_score.max()), 0.9)
        self.assertEqual(salient_change.tolist(), [[False, True, False]])


if __name__ == "__main__":
    unittest.main()
