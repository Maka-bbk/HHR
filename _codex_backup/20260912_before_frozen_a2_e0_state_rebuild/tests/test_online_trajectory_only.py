from __future__ import annotations

import unittest

import torch
import torch.nn.functional as F

from experiments.motion_primitive import online_runner as single
from experiments.motion_primitive import online_cv_runner as cv
from experiments.motion_primitive.motion_online import (
    MotionPrimitiveOnlineConfig,
    audit_primitive_retention,
    build_online_sgd,
    clone_motion_primitive_model,
    compute_online_trajectory_loss,
    expand_online_model,
    extract_trajectory_embeddings,
    grouped_memax_losses,
    initialise_new_trajectory_rows_,
    online_training_step,
    select_kmeans_new_trajectory_centres,
)
from experiments.motion_primitive.trajectory_distillation import (
    CrossViewDistillationLoss,
)
from models.motion_primitive_cgcd import MotionPrimitiveCGCDModel, MotionPrimitiveConfig


def _architecture(*, classes: int = 8, codebook_size: int = 4):
    return MotionPrimitiveConfig(
        in_channels=2, window_size=32, window_stride=16, feature_dim=8,
        base_channels=4, old_class_count=classes, codebook_size=codebook_size,
        trajectory_input_dim=6, trajectory_hidden_dim=7,
    )


def _config(**updates):
    values = {
        "epochs_per_session": 1, "warmup_teacher_epochs": 0,
        "batch_size": 3, "evaluation_batch_size": 3,
        "num_workers": 0, "evaluation_num_workers": 0,
        "learning_rate": 0.01, "weight_decay": 0.0,
    }
    values.update(updates)
    return MotionPrimitiveOnlineConfig(**values)


def _views(batch: int = 3):
    generator = torch.Generator().manual_seed(101)
    return [
        {
            "windows": torch.randn(batch, 3, 2, 32, generator=generator),
            "positions": torch.randn(batch, 3, 3, generator=generator),
            "mask": torch.ones(batch, 3, dtype=torch.bool),
            "lengths": torch.full((batch,), 3, dtype=torch.long),
        }
        for _ in range(2)
    ]


class TrajectoryOnlyOnlineTests(unittest.TestCase):
    def test_cross_view_distillation_matches_legacy_value_and_gradient(self):
        torch.manual_seed(97)
        student = torch.randn(6, 8, dtype=torch.float64, requires_grad=True)
        teacher = torch.randn(6, 8, dtype=torch.float64, requires_grad=True)
        observed = CrossViewDistillationLoss(
            2, 4, 2, 0.07, 0.04, 0.10
        )(student, teacher, 1)
        observed.backward()
        observed_gradient = student.grad.detach().clone()
        self.assertIsNone(teacher.grad)

        reference_student = student.detach().clone().requires_grad_(True)
        student_views = (reference_student / 0.10).chunk(2)
        teacher_views = F.softmax(teacher.detach() / 0.04, dim=-1).chunk(2)
        reference_terms = []
        for teacher_index, teacher_view in enumerate(teacher_views):
            for student_index, student_view in enumerate(student_views):
                if student_index != teacher_index:
                    reference_terms.append(
                        torch.sum(
                            -teacher_view * F.log_softmax(student_view, dim=-1),
                            dim=-1,
                        ).mean()
                    )
        reference = torch.stack(reference_terms).mean()
        reference.backward()
        torch.testing.assert_close(
            observed.detach(), reference.detach(), atol=1e-12, rtol=1e-12
        )
        torch.testing.assert_close(
            observed_gradient, reference_student.grad, atol=1e-12, rtol=1e-12
        )

    def test_grouped_memax_remains_finite_for_underflowed_class_groups(self):
        logits = torch.tensor(
            [[50.0, -50.0, -50.0, -50.0, -50.0, -50.0, -50.0, -50.0]]
            * 4,
            requires_grad=True,
        )
        losses = grouped_memax_losses(logits, 6, temperature=0.1)
        self.assertTrue(all(bool(torch.isfinite(value)) for value in losses))
        sum(losses).backward()
        self.assertTrue(bool(torch.isfinite(logits.grad).all()))

    def test_online_step_has_no_pool_or_fused_modules_and_updates_trajectory(self):
        torch.manual_seed(103)
        model = MotionPrimitiveCGCDModel(_architecture())
        self.assertFalse(hasattr(model, "happy_pool"))
        self.assertFalse(hasattr(model, "happy_head"))
        previous = clone_motion_primitive_model(model).eval()
        config = _config()
        optimizer = build_online_sgd(model, config)
        criterion = CrossViewDistillationLoss(0, 1, 2, 0.07, 0.04, 0.1)
        result = online_training_step(
            model, previous, _views(), optimizer, criterion, config,
            seen_class_count=6, epoch_index=0,
        )
        self.assertEqual(float(result.total), float(result.trajectory_total))
        self.assertGreater(
            sum(float(p.grad.abs().sum()) for p in model.trajectory_classifier.parameters()
                if p.grad is not None),
            0.0,
        )
        self.assertGreater(
            sum(float(p.grad.abs().sum()) for p in model.window_encoder.parameters()
                if p.grad is not None),
            0.0,
        )

    def test_loss_uses_previous_trajectory_logits_and_features(self):
        class ZeroCriterion:
            def __call__(self, logits, teacher_logits, epoch_index):
                del teacher_logits, epoch_index
                return logits.sum() * 0

        outputs, previous = [], []
        for _ in range(2):
            outputs.append({
                "trajectory_logits": torch.randn(3, 8, requires_grad=True),
                "trajectory_embedding": torch.randn(3, 7, requires_grad=True),
                "primitive_features": torch.randn(3, 2, 8, requires_grad=True),
                "token_mask": torch.ones(3, 2, dtype=torch.bool),
                "normalized_codebook": F.normalize(
                    torch.randn(4, 8, requires_grad=True), dim=-1
                ),
                "commitment_loss": torch.zeros((), requires_grad=True),
                "codebook_embedding_loss": torch.zeros((), requires_grad=True),
            })
            previous.append({
                "trajectory_logits": torch.randn(3, 6),
                "trajectory_embedding": torch.randn(3, 7),
                "primitive_features": torch.randn(3, 2, 8),
                "token_mask": torch.ones(3, 2, dtype=torch.bool),
                "normalized_codebook": F.normalize(torch.randn(4, 8), dim=-1),
            })
        config = _config(
            grouped_memax_old_new_weight=0, grouped_memax_old_in_weight=0,
            grouped_memax_new_in_weight=0, trajectory_cluster_weight=0,
            trajectory_logit_distillation_weight=1,
            trajectory_feature_distillation_weight=1,
            trajectory_view_consistency_weight=0, vq_commitment_weight=0,
            vq_codebook_weight=0,
        )
        first, parts = compute_online_trajectory_loss(
            outputs, previous, ZeroCriterion(), config,
            seen_class_count=6, epoch_index=0,
        )
        changed = [dict(item) for item in previous]
        changed[0]["trajectory_logits"] = -changed[0]["trajectory_logits"]
        changed[0]["trajectory_embedding"] = -changed[0]["trajectory_embedding"]
        second, _ = compute_online_trajectory_loss(
            outputs, changed, ZeroCriterion(), config,
            seen_class_count=6, epoch_index=0,
        )
        self.assertFalse(torch.allclose(first, second))
        self.assertIn("old_trajectory_logit_distillation", parts)
        self.assertIn("old_trajectory_feature_distillation", parts)
        self.assertIn("old_primitive_feature_distillation", parts)
        self.assertIn("old_codebook_row_anchor", parts)

    def test_primitive_and_codebook_retention_losses_are_independently_ablated(self):
        class ZeroCriterion:
            def __call__(self, logits, teacher_logits, epoch_index):
                del teacher_logits, epoch_index
                return logits.sum() * 0

        outputs, previous = [], []
        for _ in range(2):
            outputs.append({
                "trajectory_logits": torch.zeros(2, 8, requires_grad=True),
                "trajectory_embedding": torch.ones(2, 7, requires_grad=True),
                "primitive_features": F.normalize(
                    torch.randn(2, 3, 8, requires_grad=True), dim=-1
                ),
                "token_mask": torch.ones(2, 3, dtype=torch.bool),
                "normalized_codebook": F.normalize(
                    torch.randn(4, 8, requires_grad=True), dim=-1
                ),
                "commitment_loss": torch.zeros((), requires_grad=True),
                "codebook_embedding_loss": torch.zeros((), requires_grad=True),
            })
            previous.append({
                "trajectory_logits": torch.zeros(2, 6),
                "trajectory_embedding": torch.ones(2, 7),
                "primitive_features": F.normalize(torch.randn(2, 3, 8), dim=-1),
                "token_mask": torch.ones(2, 3, dtype=torch.bool),
                "normalized_codebook": F.normalize(torch.randn(4, 8), dim=-1),
            })
        common = dict(
            grouped_memax_old_new_weight=0,
            grouped_memax_old_in_weight=0,
            grouped_memax_new_in_weight=0,
            trajectory_cluster_weight=0,
            trajectory_logit_distillation_weight=0,
            trajectory_feature_distillation_weight=0,
            trajectory_view_consistency_weight=0,
            vq_commitment_weight=0,
            vq_codebook_weight=0,
        )
        changed = [dict(item) for item in previous]
        changed[0]["primitive_features"] = -changed[0]["primitive_features"]
        changed[0]["normalized_codebook"] = -changed[0]["normalized_codebook"]

        disabled = _config(
            **common,
            primitive_feature_distillation_weight=0,
            old_codebook_anchor_weight=0,
        )
        baseline, _ = compute_online_trajectory_loss(
            outputs, previous, ZeroCriterion(), disabled,
            seen_class_count=6, epoch_index=0,
        )
        ablated, _ = compute_online_trajectory_loss(
            outputs, changed, ZeroCriterion(), disabled,
            seen_class_count=6, epoch_index=0,
        )
        torch.testing.assert_close(baseline, ablated, atol=0, rtol=0)

        primitive_only = _config(
            **common,
            primitive_feature_distillation_weight=1,
            old_codebook_anchor_weight=0,
        )
        primitive_before, _ = compute_online_trajectory_loss(
            outputs, previous, ZeroCriterion(), primitive_only,
            seen_class_count=6, epoch_index=0,
        )
        primitive_after, _ = compute_online_trajectory_loss(
            outputs, changed, ZeroCriterion(), primitive_only,
            seen_class_count=6, epoch_index=0,
        )
        self.assertFalse(torch.allclose(primitive_before, primitive_after))

        anchor_only = _config(
            **common,
            primitive_feature_distillation_weight=0,
            old_codebook_anchor_weight=1,
        )
        anchor_before, _ = compute_online_trajectory_loss(
            outputs, previous, ZeroCriterion(), anchor_only,
            seen_class_count=6, epoch_index=0,
        )
        anchor_after, _ = compute_online_trajectory_loss(
            outputs, changed, ZeroCriterion(), anchor_only,
            seen_class_count=6, epoch_index=0,
        )
        self.assertFalse(torch.allclose(anchor_before, anchor_after))

    def test_retention_audit_is_identity_for_equal_models_and_ignores_labels(self):
        class ProvenanceDataset:
            trial_global_ids = torch.tensor([17, 18]).numpy()
            subject_ids = torch.tensor([4, 5]).numpy()

            def __len__(self):
                return 2

        class Loader:
            dataset = ProvenanceDataset()

            def __init__(self, label):
                self.batch = (
                    _views(batch=2)[0],
                    torch.full((2,), label, dtype=torch.long),
                    torch.tensor([17, 18]),
                    torch.zeros(2, dtype=torch.long),
                )

            def __iter__(self):
                yield self.batch

        current = MotionPrimitiveCGCDModel(_architecture())
        previous = clone_motion_primitive_model(current)
        first = audit_primitive_retention(
            current, previous, Loader(1), torch.device("cpu")
        )
        second = audit_primitive_retention(
            current, previous, Loader(11), torch.device("cpu")
        )
        self.assertEqual(first, second)
        self.assertAlmostEqual(first["local_feature_cosine_mean"], 1.0, places=6)
        self.assertAlmostEqual(first["old_token_exact_agreement"], 1.0, places=6)
        self.assertAlmostEqual(first["new_code_usage_fraction"], 0.0, places=6)
        self.assertFalse(first["activity_labels_used"])
        self.assertTrue(first["augmentation_disabled"])

    def test_new_class_initialization_is_in_trajectory_space(self):
        torch.manual_seed(111)
        previous = MotionPrimitiveCGCDModel(_architecture(classes=6))
        current = expand_online_model(previous, 8, _config())
        batch = _views(batch=8)[0]
        centres = select_kmeans_new_trajectory_centres(
            current, previous, [(batch, torch.arange(8))], 8, _config(),
            torch.device("cpu"),
        )
        self.assertEqual(tuple(centres.shape), (2, 7))
        initialise_new_trajectory_rows_(current, 6, centres)
        torch.testing.assert_close(
            F.normalize(current.trajectory_classifier.weight[6:], dim=-1), centres,
            atol=1e-6, rtol=1e-6,
        )

    def test_embedding_extraction_is_trajectory_only(self):
        model = MotionPrimitiveCGCDModel(_architecture())
        embeddings = extract_trajectory_embeddings(
            model, [(_views(batch=2)[0], torch.tensor([1, 7]))], torch.device("cpu")
        )
        self.assertEqual(tuple(embeddings.shape), (2, 7))
        torch.testing.assert_close(embeddings.norm(dim=-1), torch.ones(2), atol=1e-6, rtol=0)

    def test_runner_defaults_are_adaptive_k32_and_trajectory_only(self):
        args = single.build_parser().parse_args([
            "--offline-run-dir", "offline", "--checkpoint", "checkpoint.pt",
            "--output-dir", "online",
        ])
        self.assertEqual(args.codebook_expansion, "residual_adaptive")
        self.assertEqual(args.expected_offline_codebook_size, 32)
        self.assertEqual(args.primitive_feature_distillation_weight, 1.0)
        self.assertEqual(args.old_codebook_anchor_weight, 1.0)
        self.assertEqual(args.codebook_minimum_cluster_trials, 3)
        self.assertEqual(args.codebook_minimum_cluster_subjects, 2)
        self.assertEqual(single.report_head_for_profile(single.PROFILE_JOINT), "trajectory")
        options = single.build_parser()._option_string_actions
        for forbidden in (
            "--pooled-auxiliary-weight", "--proto-aug-weight",
            "--fusion-trajectory-weight", "--initialize-new-head-with-kmeans",
        ):
            self.assertNotIn(forbidden, options)
        self.assertEqual(cv.selected_checkpoint_name(cv.PROFILE_JOINT),
                         "checkpoint_best_trajectory.pt")
        with self.assertRaises(ValueError):
            cv.parse_profiles("happy_primitive_joint")
        with self.assertRaises(ValueError):
            cv.parse_profiles("happy_control")


if __name__ == "__main__":
    unittest.main()
