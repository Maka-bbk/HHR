import unittest

import numpy as np
import torch

from experiments.motion_primitive.one_stage_losses import (
    OneStageLossConfig,
    compose_one_stage_loss,
)
from experiments.motion_primitive.train_one_stage import _loss_inputs
from models.motion_trajectory import MotionTrajectoryConfig, MotionTrajectoryModel


class OneStageModelLossIntegrationTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(31)
        self.config = MotionTrajectoryConfig(
            in_channels=6,
            frame_size=16,
            frame_stride=8,
            local_hidden_dim=8,
            local_embedding_dim=8,
            state_hidden_dim=6,
            state_dim=4,
            codebook_size=4,
            trajectory_input_dim=8,
            trajectory_hidden_dim=8,
            trajectory_layers=1,
            trajectory_mask_ratio=0.25,
            num_classes=3,
        ).validated()
        self.model_trials = torch.randn(2, 40, 6)
        self.physical_trials = torch.randn(2, 40, 6)
        self.lengths = torch.tensor([40, 35], dtype=torch.long)
        self.batch = {
            "trial_ids": torch.tensor([10, 11], dtype=torch.long),
            "supervision_targets": torch.tensor([1, -100], dtype=torch.long),
            "trajectory_label_mask": torch.tensor([True, False]),
        }
        # Trial 11 ends with one partial model frame; its unmatched final
        # boundary is deliberately left uncertain by the raw anchor map.
        self.raw_targets = {
            10: (
                np.asarray([True, False, True]),
                np.asarray([False, True, False]),
            ),
            11: (
                np.asarray([True, False]),
                np.asarray([False, True]),
            ),
        }

    @staticmethod
    def _gradient_sum(module: torch.nn.Module) -> float:
        return float(
            sum(
                parameter.grad.abs().sum()
                for parameter in module.parameters()
                if parameter.grad is not None
            )
        )

    def test_j0_u_reaches_encoder_codebook_boundary_and_trajectory_but_not_classifier(self):
        model = MotionTrajectoryModel(self.config)
        model.train()
        outputs = model(
            self.model_trials,
            self.lengths,
            state_trials=self.physical_trials,
            hard_codebook=False,
            apply_random_trajectory_mask=True,
        )
        loss_inputs = _loss_inputs(
            outputs,
            self.batch,
            self.raw_targets,
            "J0-U",
            torch.device("cpu"),
        )
        result = compose_one_stage_loss(
            loss_inputs,
            OneStageLossConfig.j0_u(
                old_class_count=3,
                unlabelled_index=-100,
                minimum_segment_windows=2,
            ),
        )
        result.total.backward()
        self.assertGreater(self._gradient_sum(model.local_encoder), 0.0)
        self.assertGreater(float(model.codebook.vectors.grad.abs().sum()), 0.0)
        self.assertGreater(self._gradient_sum(model.boundary_head), 0.0)
        self.assertGreater(self._gradient_sum(model.trajectory_encoder), 0.0)
        self.assertIsNone(model.trajectory_classifier.weight.grad)
        self.assertEqual(tuple(outputs["boundary_logits"].shape), (2, 3))
        self.assertEqual(tuple(outputs["context_state_reconstruction"].shape), (2, 4, 18))

    def test_j0_t_classifier_receives_only_explicit_trial_ce(self):
        model = MotionTrajectoryModel(self.config)
        model.train()
        outputs = model(
            self.model_trials,
            self.lengths,
            state_trials=self.physical_trials,
            apply_random_trajectory_mask=True,
        )
        inputs = _loss_inputs(
            outputs,
            self.batch,
            self.raw_targets,
            "J0-T",
            torch.device("cpu"),
        )
        result = compose_one_stage_loss(
            inputs,
            OneStageLossConfig.j0_t(
                old_class_count=3,
                unlabelled_index=-100,
                minimum_segment_windows=2,
            ),
        )
        result.total.backward()
        self.assertEqual(result.labelled_old_trial_count, 1)
        self.assertGreater(self._gradient_sum(model.trajectory_classifier), 0.0)

    def test_classifier_api_has_no_continuous_local_feature_argument(self):
        model = MotionTrajectoryModel(self.config).eval()
        with torch.no_grad():
            outputs = model(
                self.model_trials,
                self.lengths,
                state_trials=self.physical_trials,
                hard_codebook=True,
                apply_random_trajectory_mask=False,
            )
            reversed_order = torch.arange(
                outputs["token_mask"].shape[1] - 1,
                -1,
                -1,
                dtype=torch.long,
            ).repeat(2, 1)
            # Both rows have four valid frames in this fixture.
            shuffled = model.encode_trajectory(
                outputs["trajectory_assignments"],
                outputs["token_mask"],
                outputs["state_features"],
                outputs["boundary_probabilities"],
                outputs["transition_probabilities"],
                outputs["duration_proxy"],
                apply_random_mask=False,
                token_permutation=reversed_order,
            )
        self.assertEqual(
            model.trajectory_feature_dim,
            self.config.codebook_size + self.config.state_dim + 3,
        )
        self.assertNotEqual(
            float(
                torch.max(
                    torch.abs(
                        outputs["trajectory_embedding"]
                        - shuffled["trajectory_embedding"]
                    )
                )
            ),
            0.0,
        )


if __name__ == "__main__":
    unittest.main()
