"""Isolation gates for the trajectory-only motion-primitive model."""

from __future__ import annotations

import unittest

import torch

from models.motion_primitive_cgcd import MotionPrimitiveCGCDModel, MotionPrimitiveConfig
from models.resnet1d import ResNet1D


class JointIsolationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.model = MotionPrimitiveCGCDModel(
            MotionPrimitiveConfig(
                window_size=32,
                window_stride=16,
                feature_dim=8,
                base_channels=2,
                old_class_count=3,
                codebook_size=4,
                trajectory_input_dim=7,
                trajectory_hidden_dim=7,
                run_state_dim=3,
            )
        )

    def _batch(self):
        mask = torch.tensor([[True, True, True], [True, True, False]])
        starts = torch.tensor([[0, 16, 32], [0, 16, -1]])
        return {
            "windows": torch.randn(2, 3, 6, 32),
            "positions": self.model.relative_positions(starts, mask, 32),
            "mask": mask,
            "lengths": mask.sum(1),
        }

    def test_joint_model_owns_exactly_one_resnet1d(self) -> None:
        encoders = [module for module in self.model.modules() if isinstance(module, ResNet1D)]
        self.assertEqual(len(encoders), 1)
        self.assertIs(encoders[0], self.model.window_encoder)

    def test_trajectory_forward_cannot_call_complete_trial_pooling(self) -> None:
        registered_types = {
            (type(module).__module__, type(module).__name__)
            for module in self.model.modules()
        }
        self.assertNotIn(
            ("models.trial_pooling", "TrialMeanRobustMaxPool"), registered_types
        )
        output = self.model.forward_trajectory(self._batch(), hard_codebook=False)
        self.assertIn("trajectory_logits", output)
        self.assertNotIn("happy_logits", output)
        self.assertNotIn("fused_logits", output)

    def test_trajectory_ce_reaches_resnet_codebook_and_readout(self) -> None:
        output = self.model.forward_trajectory(self._batch(), hard_codebook=False)
        output["run_features"].retain_grad()
        output["normalized_codebook"].retain_grad()
        loss = torch.nn.functional.cross_entropy(
            output["trajectory_logits"], torch.tensor([0, 1])
        )
        loss.backward()
        for module in (
            self.model.window_encoder,
            self.model.codebook,
            self.model.trajectory_input,
            self.model.trajectory_encoder,
            self.model.trajectory_classifier,
        ):
            gradient = sum(
                float(parameter.grad.abs().sum())
                for parameter in module.parameters()
                if parameter.grad is not None
            )
            self.assertGreater(gradient, 0.0, type(module).__name__)
        self.assertIsNotNone(output["run_features"].grad)
        self.assertGreater(
            float(
                output["run_features"].grad[..., : self.model.config.feature_dim]
                .abs()
                .sum()
            ),
            0.0,
        )
        self.assertIsNotNone(output["normalized_codebook"].grad)
        self.assertGreater(
            float(output["normalized_codebook"].grad.abs().sum()), 0.0
        )


if __name__ == "__main__":
    unittest.main()
