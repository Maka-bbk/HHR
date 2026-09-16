import json
import unittest
from dataclasses import replace

import torch
import torch.nn.functional as F

from experiments.motion_primitive.one_stage_losses import (
    J0_TRAJECTORY,
    J0_UNSUPERVISED,
    OneStageLossConfig,
    OneStageLossInputs,
    boundary_minimum_duration_loss,
    codebook_utilization_floor_loss,
    compose_one_stage_loss,
    raw_kinematic_boundary_supervision_loss,
    trajectory_old_class_cross_entropy,
    vq_losses,
)


def _inputs(*, with_trajectory: bool = False) -> OneStageLossInputs:
    torch.manual_seed(17)
    batch, length, feature_dim, code_count = 2, 4, 3, 3
    valid_window = torch.tensor(
        [[True, True, True, True], [True, True, True, False]]
    )
    valid_boundary = torch.tensor(
        [[True, True, True], [True, True, False]]
    )
    stable = torch.tensor(
        [[True, False, False], [False, True, False]]
    )
    # A complete trial is allowed to contain no q90 raw-change anchor.  This
    # fixture deliberately has none in either trial.
    change = torch.zeros_like(stable)
    trajectory_mask = torch.tensor(
        [[True, False, True, False], [False, True, False, False]]
    )
    pseudo_targets = torch.full((batch, length), -1, dtype=torch.long)
    pseudo_targets[trajectory_mask] = torch.tensor([0, 2, 1])
    state_targets = torch.randn(batch, length, 2, requires_grad=True)
    kwargs = {}
    if with_trajectory:
        kwargs = {
            "trajectory_logits": torch.tensor(
                [[0.2, 1.4, -0.5], [4.0, -2.0, 0.0]], requires_grad=True
            ),
            "trajectory_labels": torch.tensor([1, -1], dtype=torch.long),
            "labelled_old_trial_mask": torch.tensor([True, False]),
        }
    return OneStageLossInputs(
        encoded_states=torch.randn(batch, length, feature_dim, requires_grad=True),
        quantized_states=torch.randn(batch, length, feature_dim, requires_grad=True),
        reconstructed_content=torch.randn(
            batch, length, feature_dim, requires_grad=True
        ),
        content_targets=torch.randn(batch, length, feature_dim, requires_grad=True),
        next_content_predictions=torch.randn(
            batch, length - 1, feature_dim, requires_grad=True
        ),
        next_content_targets=torch.randn(
            batch, length - 1, feature_dim, requires_grad=True
        ),
        assignment_logits=torch.randn(batch, length, code_count, requires_grad=True),
        valid_window_mask=valid_window,
        boundary_logits=torch.randn(batch, length - 1, requires_grad=True),
        valid_boundary_mask=valid_boundary,
        raw_stable_mask=stable,
        raw_change_mask=change,
        reconstructed_states=torch.randn(batch, length, 2, requires_grad=True),
        state_targets=state_targets,
        masked_token_logits=torch.randn(
            batch, length, code_count, requires_grad=True
        ),
        pseudo_token_targets=pseudo_targets,
        masked_state_predictions=torch.randn(batch, length, 2, requires_grad=True),
        masked_trajectory_mask=trajectory_mask,
        **kwargs,
    )


class J0UnsupervisedLossTests(unittest.TestCase):
    def test_j0_u_without_labels_trains_local_codebook_and_trajectory_paths(self):
        inputs = _inputs()
        result = compose_one_stage_loss(inputs, OneStageLossConfig.j0_u())
        self.assertTrue(torch.isfinite(result.total))
        self.assertEqual(result.labelled_old_trial_count, 0)
        self.assertEqual(result.metrics["raw_change_anchor_count"], 0)
        result.total.backward()

        # Commitment reaches the local encoder; codebook fitting reaches the
        # quantized/codebook path; masked modelling reaches trajectory heads
        # and therefore the trajectory encoder upstream in an integrated model.
        self.assertGreater(float(inputs.encoded_states.grad.abs().sum()), 0.0)
        self.assertGreater(float(inputs.quantized_states.grad.abs().sum()), 0.0)
        self.assertGreater(
            float(inputs.reconstructed_content.grad.abs().sum()), 0.0
        )
        self.assertGreater(
            float(inputs.next_content_predictions.grad.abs().sum()), 0.0
        )
        self.assertGreater(float(inputs.masked_token_logits.grad.abs().sum()), 0.0)
        self.assertGreater(
            float(inputs.masked_state_predictions.grad.abs().sum()), 0.0
        )
        # Label-free descriptor targets and pseudo-token targets are targets,
        # not trainable shortcuts.
        self.assertIsNone(inputs.state_targets.grad)
        self.assertIsNone(inputs.content_targets.grad)
        self.assertIsNone(inputs.next_content_targets.grad)
        self.assertFalse(inputs.pseudo_token_targets.requires_grad)

        audit = result.to_audit()
        json.dumps(audit, allow_nan=False)
        self.assertEqual(audit["config"]["profile"], J0_UNSUPERVISED)
        self.assertFalse(audit["config"]["happy_checkpoint_used"])
        self.assertFalse(audit["config"]["window_classification_used"])
        self.assertFalse(audit["config"]["infonce_used"])
        self.assertFalse(audit["config"]["activity_supcon_used"])
        self.assertEqual(
            audit["config"]["objective_scopes"]["trajectory_ce"], "disabled"
        )
        self.assertFalse(
            audit["physical_activity_labels_consumed_outside_trajectory_ce"]
        )

    def test_j0_u_rejects_even_all_sentinel_activity_labels(self):
        inputs = _inputs()
        inputs.trajectory_logits = torch.zeros(2, 3)
        inputs.trajectory_labels = torch.full((2,), -1, dtype=torch.long)
        inputs.labelled_old_trial_mask = torch.zeros(2, dtype=torch.bool)
        with self.assertRaisesRegex(ValueError, "J0-U forbids"):
            compose_one_stage_loss(inputs, OneStageLossConfig.j0_u())

    def test_raw_boundary_masks_may_have_no_change_for_a_trial(self):
        logits = torch.tensor([[0.4, -0.3], [1.0, -1.0]], requires_grad=True)
        valid = torch.ones_like(logits, dtype=torch.bool)
        stable = torch.tensor([[True, False], [False, True]])
        change = torch.zeros_like(stable)
        loss, metrics = raw_kinematic_boundary_supervision_loss(
            logits, valid, stable, change
        )
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(metrics["raw_change_anchor_count"], 0)
        loss.backward()
        self.assertGreater(float(logits.grad.abs().sum()), 0.0)

    def test_non_raw_boundary_source_and_label_derived_state_are_rejected(self):
        inputs = _inputs()
        inputs.boundary_anchor_source = "raw_frozen_consensus"
        with self.assertRaisesRegex(ValueError, "raw kinematics only"):
            compose_one_stage_loss(inputs, OneStageLossConfig.j0_u())

        inputs = _inputs()
        inputs.state_target_source = "physical_activity_label"
        with self.assertRaisesRegex(ValueError, "label-free"):
            compose_one_stage_loss(inputs, OneStageLossConfig.j0_u())

    def test_model_frame_boundary_and_vq_vocabulary_alignment_are_enforced(self):
        inputs = _inputs()
        inputs.boundary_logits = torch.zeros(2, 4)
        with self.assertRaisesRegex(ValueError, r"\[B,L-1\]"):
            compose_one_stage_loss(inputs, OneStageLossConfig.j0_u())

        inputs = _inputs()
        inputs.masked_token_logits = torch.zeros(2, 4, 4)
        with self.assertRaisesRegex(ValueError, "vocabulary"):
            compose_one_stage_loss(inputs, OneStageLossConfig.j0_u())


class CodebookAndStructureLossTests(unittest.TestCase):
    def test_utilization_is_a_floor_not_uniformity_maximization(self):
        valid = torch.ones(1, 8, dtype=torch.bool)
        # Deliberately non-uniform but well above the low entropy floor.
        probabilities = torch.tensor([0.70, 0.10, 0.10, 0.10])
        logits = probabilities.log().reshape(1, 1, 4).repeat(1, 8, 1)
        loss, metrics = codebook_utilization_floor_loss(
            logits,
            valid,
            normalized_entropy_floor=0.35,
        )
        self.assertEqual(float(loss), 0.0)
        self.assertGreater(metrics["normalized_codebook_entropy"], 0.35)
        self.assertLess(metrics["normalized_codebook_entropy"], 1.0)

        collapsed = torch.tensor([12.0, -12.0, -12.0, -12.0]).reshape(1, 1, 4)
        collapsed = collapsed.repeat(1, 8, 1)
        collapse_loss, collapse_metrics = codebook_utilization_floor_loss(
            collapsed,
            valid,
            normalized_entropy_floor=0.35,
        )
        self.assertGreater(float(collapse_loss), 0.0)
        self.assertLess(collapse_metrics["normalized_codebook_entropy"], 0.35)

    def test_vq_stop_gradient_routes_are_separate(self):
        encoded = torch.tensor([[[1.0, 2.0]]], requires_grad=True)
        quantized = torch.tensor([[[3.0, -1.0]]], requires_grad=True)
        valid = torch.ones(1, 1, dtype=torch.bool)
        commitment, codebook = vq_losses(encoded, quantized, valid)
        commitment.backward(retain_graph=True)
        self.assertGreater(float(encoded.grad.abs().sum()), 0.0)
        self.assertIsNone(quantized.grad)
        encoded.grad = None
        codebook.backward()
        self.assertIsNone(encoded.grad)
        self.assertGreater(float(quantized.grad.abs().sum()), 0.0)

    def test_minimum_duration_accepts_prefix_padding_and_rejects_holes(self):
        logits = torch.ones(2, 4)
        prefix = torch.tensor(
            [[True, True, True, True], [True, True, False, False]]
        )
        loss = boundary_minimum_duration_loss(
            logits, prefix, minimum_segment_windows=2
        )
        self.assertGreater(float(loss), 0.0)
        hole = torch.tensor(
            [[True, False, True, False], [True, True, False, False]]
        )
        with self.assertRaisesRegex(ValueError, "contiguous prefix"):
            boundary_minimum_duration_loss(
                logits, hole, minimum_segment_windows=2
            )


class J0TrajectoryLossTests(unittest.TestCase):
    def test_j0_t_ce_uses_only_explicit_old_labelled_trials(self):
        inputs = _inputs(with_trajectory=True)
        config = OneStageLossConfig.j0_t(old_class_count=3)
        result = compose_one_stage_loss(inputs, config)
        expected = F.cross_entropy(
            inputs.trajectory_logits[:1], inputs.trajectory_labels[:1]
        )
        self.assertTrue(torch.allclose(result.components["trajectory_ce"], expected))
        self.assertEqual(result.labelled_old_trial_count, 1)
        result.total.backward()
        self.assertGreater(float(inputs.trajectory_logits.grad[0].abs().sum()), 0.0)
        self.assertEqual(float(inputs.trajectory_logits.grad[1].abs().sum()), 0.0)

    def test_hidden_label_on_unlabelled_trial_is_rejected(self):
        logits = torch.zeros(2, 3)
        labels = torch.tensor([1, 2], dtype=torch.long)
        selected = torch.tensor([True, False])
        with self.assertRaisesRegex(ValueError, "hidden physical labels"):
            trajectory_old_class_cross_entropy(
                logits,
                labels,
                selected,
                old_class_count=3,
            )

    def test_profiles_enforce_trajectory_supervision_contract(self):
        with self.assertRaisesRegex(ValueError, "J0-U forbids"):
            OneStageLossConfig(
                profile=J0_UNSUPERVISED, trajectory_ce_weight=1.0
            ).validated()
        with self.assertRaisesRegex(ValueError, "J0-T requires"):
            OneStageLossConfig(
                profile=J0_TRAJECTORY, trajectory_ce_weight=0.0
            ).validated()
        config = OneStageLossConfig.j0_t(old_class_count=3)
        ablated = config.with_ablation(raw_boundary_weight=0.0)
        self.assertEqual(ablated.raw_boundary_weight, 0.0)
        with self.assertRaisesRegex(ValueError, "Unknown ablation"):
            config.with_ablation(not_a_loss=0.0)


if __name__ == "__main__":
    unittest.main()
