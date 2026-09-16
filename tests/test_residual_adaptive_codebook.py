from __future__ import annotations

import io
import sys
import unittest
from pathlib import Path

import torch
import torch.nn.functional as F

from experiments.motion_primitive.motion_online import (
    CodebookExpansionDecision,
    MotionPrimitiveOnlineConfig,
    build_online_sgd,
    expand_motion_codebook_,
    select_kmeans_codebook_centres,
    select_residual_adaptive_codebook_centres,
)
from experiments.motion_primitive import online_runner as single
from experiments.motion_primitive import online_cv_runner as cv
from models.motion_primitive_cgcd import (
    MotionPrimitiveCGCDModel,
    MotionPrimitiveConfig,
)


def _decision(features: torch.Tensor, old: torch.Tensor, **updates):
    row_count = int(len(features))
    arguments = {
        "max_delta": 3,
        "trial_ids": torch.arange(row_count, dtype=torch.long),
        "subject_ids": torch.arange(row_count, dtype=torch.long) % 2,
        "residual_quantile": 0.0,
        "minimum_residual_support": 4,
        "minimum_cluster_support": 2,
        "minimum_cluster_trials": 2,
        "minimum_cluster_subjects": 2,
        "minimum_relative_improvement": 0.10,
        "complexity_penalty": 0.01,
        "random_state": 17,
    }
    arguments.update(updates)
    return select_residual_adaptive_codebook_centres(features, old, **arguments)


def _architecture(*, codebook_size: int) -> MotionPrimitiveConfig:
    return MotionPrimitiveConfig(
        in_channels=2,
        window_size=32,
        window_stride=16,
        feature_dim=8,
        base_channels=4,
        old_class_count=6,
        codebook_size=codebook_size,
        trajectory_input_dim=6,
        trajectory_hidden_dim=7,
        run_state_dim=3,
    )


def _trajectory_batch(model: MotionPrimitiveCGCDModel) -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(43)
    one_window = torch.randn(1, 1, 2, 32, generator=generator)
    windows = one_window.expand(1, 3, -1, -1).clone()
    mask = torch.ones(1, 3, dtype=torch.bool)
    starts = torch.tensor([[0, 16, 32]], dtype=torch.long)
    return {
        "windows": windows,
        "positions": model.relative_positions(starts, mask, 32),
        "mask": mask,
        "lengths": mask.sum(dim=1),
    }


@torch.no_grad()
def _force_all_windows_to_old_code_zero(
    model: MotionPrimitiveCGCDModel,
    batch: dict[str, torch.Tensor],
) -> torch.Tensor:
    model.eval()
    initial = model(batch, hard_codebook=True)
    direction = F.normalize(initial["primitive_features"][0, 0], dim=0)
    model.codebook.vectors[0].copy_(direction)
    model.codebook.vectors[1:].copy_(-direction.expand_as(model.codebook.vectors[1:]))
    observed = model(batch, hard_codebook=True)
    if not torch.equal(
        observed["hard_tokens"], torch.zeros_like(observed["hard_tokens"])
    ):
        raise AssertionError("Test fixture could not force all windows to old code zero.")
    return direction


def _loader_with_label(label: int):
    class ProvenanceDataset:
        trial_global_ids = torch.tensor([0, 1]).numpy()
        subject_ids = torch.tensor([4, 5]).numpy()

        def __len__(self):
            return 2

    class Loader:
        dataset = ProvenanceDataset()

        def __init__(self, batch):
            self.batch = batch

        def __iter__(self):
            yield self.batch

    generator = torch.Generator().manual_seed(29)
    prepared = {
        "windows": torch.randn(2, 3, 2, 32, generator=generator),
        "positions": torch.zeros(2, 3, 3),
        "mask": torch.ones(2, 3, dtype=torch.bool),
        "lengths": torch.full((2,), 3, dtype=torch.long),
    }
    return Loader(
        (
            prepared,
            torch.full((2,), int(label), dtype=torch.long),
            torch.arange(2),
            torch.zeros(2, 1, dtype=torch.long),
        )
    )


class ResidualAdaptiveDecisionTests(unittest.TestCase):
    def test_low_residual_selects_zero_and_audits_every_candidate(self):
        old = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        features = old.repeat_interleave(8, dim=0)
        decision = _decision(features, old)
        self.assertIsNone(decision.centres)
        self.assertEqual(decision.selected_delta, 0)
        self.assertEqual(decision.audit["selection_reason"], "no_positive_quantization_residual")
        self.assertEqual(decision.audit["codebook_size_before"], 2)
        self.assertEqual(decision.audit["codebook_size_after"], 2)
        self.assertEqual(
            [row["delta"] for row in decision.audit["candidate_evaluations"]],
            [1, 2, 3],
        )

    def test_two_clear_high_residual_modes_select_two_unit_centres(self):
        old = torch.tensor(
            [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]
        )
        first = torch.tensor([0.0, 0.0, 1.0, 0.0]).repeat(20, 1)
        second = torch.tensor([0.0, 0.0, 0.0, 1.0]).repeat(20, 1)
        features = torch.cat((first, second), dim=0)
        decision = _decision(features, old, minimum_cluster_support=5)
        self.assertEqual(decision.selected_delta, 2)
        self.assertIsNotNone(decision.centres)
        torch.testing.assert_close(
            decision.centres.norm(dim=-1), torch.ones(2), atol=1e-6, rtol=0
        )
        self.assertEqual(decision.audit["codebook_size_after"], 4)
        self.assertGreaterEqual(
            min(decision.audit["candidate_evaluations"][1]["cluster_distinct_trial_supports"]),
            2,
        )
        self.assertGreaterEqual(
            min(decision.audit["candidate_evaluations"][1]["cluster_distinct_subject_supports"]),
            2,
        )

    def test_many_windows_from_one_trial_and_subject_cannot_expand(self):
        old = torch.tensor(
            [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]
        )
        features = torch.tensor([0.0, 0.0, 1.0, 0.0]).repeat(128, 1)
        decision = _decision(
            features,
            old,
            trial_ids=torch.full((128,), 77, dtype=torch.long),
            subject_ids=torch.full((128,), 4, dtype=torch.long),
            minimum_residual_support=32,
            minimum_cluster_support=8,
        )
        self.assertEqual(decision.selected_delta, 0)
        self.assertEqual(
            decision.audit["selection_reason"], "insufficient_residual_trial_support"
        )
        self.assertEqual(decision.audit["residual_distinct_trial_support"], 1)
        self.assertEqual(decision.audit["residual_distinct_subject_support"], 1)
        for candidate in decision.audit["candidate_evaluations"]:
            self.assertIn(
                "insufficient_residual_trial_support",
                candidate["rejection_reasons"],
            )
            self.assertIn(
                "insufficient_residual_subject_support",
                candidate["rejection_reasons"],
            )

    def test_insufficient_residual_support_rejects_all_candidates(self):
        old = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        features = torch.tensor([[0.0, 0.0, 1.0]]).repeat(5, 1)
        decision = _decision(
            features,
            old,
            minimum_residual_support=6,
            minimum_cluster_support=1,
        )
        self.assertEqual(decision.selected_delta, 0)
        self.assertTrue(
            all(
                "insufficient_residual_support" in row["rejection_reasons"]
                for row in decision.audit["candidate_evaluations"]
            )
        )

    def test_small_child_cluster_rejects_that_candidate(self):
        old = torch.tensor(
            [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]
        )
        major = torch.tensor([0.0, 0.0, 1.0, 0.0]).repeat(9, 1)
        minor = torch.tensor([0.0, 0.0, 0.0, 1.0]).repeat(1, 1)
        decision = _decision(
            torch.cat((major, minor)),
            old,
            max_delta=2,
            minimum_residual_support=10,
            minimum_cluster_support=3,
        )
        self.assertEqual(decision.selected_delta, 1)
        candidate_two = decision.audit["candidate_evaluations"][1]
        self.assertFalse(candidate_two["accepted"])
        self.assertIn(
            "minimum_cluster_support_not_met",
            candidate_two["rejection_reasons"],
        )


class PolicyAndRunnerContractTests(unittest.TestCase):
    def _single_args(self, policy: str):
        return single.build_parser().parse_args(
            [
                "--offline-run-dir",
                "offline",
                "--checkpoint",
                "checkpoint.pt",
                "--output-dir",
                "online",
                "--codebook-expansion",
                policy,
                "--codebook-adaptive-max-delta",
                "2",
                "--codebook-residual-quantile",
                "0.5",
                "--codebook-minimum-residual-support",
                "4",
                "--codebook-minimum-cluster-support",
                "2",
                "--codebook-minimum-cluster-trials",
                "2",
                "--codebook-minimum-cluster-subjects",
                "2",
                "--codebook-minimum-relative-improvement",
                "0.2",
                "--codebook-complexity-penalty",
                "0.03",
            ]
        )

    def test_policy_decision_is_identical_when_activity_labels_change(self):
        torch.manual_seed(31)
        model = MotionPrimitiveCGCDModel(_architecture(codebook_size=32))
        policy = single.build_codebook_expansion_policy(
            self._single_args("residual_adaptive"), single.PROFILE_JOINT
        )
        first = policy(1, model, _loader_with_label(1), torch.device("cpu"))
        second = policy(1, model, _loader_with_label(11), torch.device("cpu"))
        self.assertEqual(first.selected_delta, second.selected_delta)
        self.assertEqual(first.audit, second.audit)
        if first.centres is not None:
            torch.testing.assert_close(first.centres, second.centres, atol=0, rtol=0)
        self.assertFalse(first.audit["activity_labels_used"])

    def test_residual_policy_fails_closed_when_session_one_is_not_k32(self):
        model = MotionPrimitiveCGCDModel(_architecture(codebook_size=4))
        policy = single.build_codebook_expansion_policy(
            self._single_args("residual_adaptive"), single.PROFILE_JOINT
        )
        with self.assertRaisesRegex(RuntimeError, "K=32"):
            policy(1, model, None, torch.device("cpu"))

    def test_fixed_delta_keeps_original_kmeans_centres(self):
        torch.manual_seed(37)
        model = MotionPrimitiveCGCDModel(_architecture(codebook_size=4))
        args = self._single_args("fixed_delta")
        args.codebook_fixed_delta = 2
        args.codebook_kmeans_random_state = 9
        loader = _loader_with_label(3)
        expected = select_kmeans_codebook_centres(
            model, loader, 2, device=torch.device("cpu"), random_state=9
        )
        policy = single.build_codebook_expansion_policy(args, single.PROFILE_JOINT)
        observed = policy(1, model, loader, torch.device("cpu"))
        self.assertIsInstance(observed, CodebookExpansionDecision)
        self.assertEqual(observed.selected_delta, 2)
        torch.testing.assert_close(observed.centres, expected, atol=0, rtol=0)
        self.assertEqual(observed.audit["policy"], "fixed_delta")

    def test_cv_member_command_forwards_every_adaptive_parameter(self):
        args = cv.build_parser().parse_args(
            [
                "--offline-cv-root",
                "offline",
                "--output-root",
                "online",
                "--python-executable",
                sys.executable,
                "--codebook-expansion",
                "residual_adaptive",
                "--codebook-adaptive-max-delta",
                "7",
                "--codebook-residual-quantile",
                "0.8",
                "--codebook-minimum-residual-support",
                "21",
                "--codebook-minimum-cluster-support",
                "6",
                "--codebook-minimum-cluster-trials",
                "4",
                "--codebook-minimum-cluster-subjects",
                "2",
                "--codebook-minimum-relative-improvement",
                "0.17",
                "--codebook-complexity-penalty",
                "0.023",
            ]
        )
        command = cv.build_member_command(
            args,
            profile=cv.PROFILE_JOINT,
            fold=1,
            seed=0,
            offline_run_dir=Path("offline-member"),
            checkpoint=Path("checkpoint.pt"),
            output_dir=Path("online-member"),
        )
        values = {
            command[index]: command[index + 1]
            for index in range(len(command) - 1)
            if command[index].startswith("--")
            and not command[index + 1].startswith("--")
        }
        self.assertEqual(values["--codebook-expansion"], "residual_adaptive")
        self.assertEqual(values["--expected-offline-codebook-size"], "32")
        self.assertEqual(values["--codebook-adaptive-max-delta"], "7")
        self.assertEqual(values["--codebook-residual-quantile"], "0.8")
        self.assertEqual(values["--codebook-minimum-residual-support"], "21")
        self.assertEqual(values["--codebook-minimum-cluster-support"], "6")
        self.assertEqual(values["--codebook-minimum-cluster-trials"], "4")
        self.assertEqual(values["--codebook-minimum-cluster-subjects"], "2")
        self.assertEqual(values["--codebook-minimum-relative-improvement"], "0.17")
        self.assertEqual(values["--codebook-complexity-penalty"], "0.023")

    def test_codebook_migration_keeps_fixed_d_trajectory_input_exact(self):
        torch.manual_seed(41)
        model = MotionPrimitiveCGCDModel(_architecture(codebook_size=4)).eval()
        batch = _trajectory_batch(model)
        direction = _force_all_windows_to_old_code_zero(model, batch)
        before = model(batch, hard_codebook=True)
        input_state = {
            name: value.detach().clone()
            for name, value in model.trajectory_input.state_dict().items()
        }
        input_shapes = {
            name: tuple(value.shape)
            for name, value in model.trajectory_input.state_dict().items()
        }

        observed_old_k, observed_new_k = expand_motion_codebook_(
            model, -direction.unsqueeze(0)
        )
        after = model(batch, hard_codebook=True)

        self.assertEqual((observed_old_k, observed_new_k), (4, 5))
        self.assertEqual(model.config.run_feature_dim, 8 + 6 + 3)
        self.assertEqual(
            model.trajectory_input[1].in_features, model.config.run_feature_dim
        )
        self.assertEqual(
            {
                name: tuple(value.shape)
                for name, value in model.trajectory_input.state_dict().items()
            },
            input_shapes,
        )
        for name, value in model.trajectory_input.state_dict().items():
            torch.testing.assert_close(value, input_state[name], atol=0, rtol=0)
        torch.testing.assert_close(after["hard_tokens"], before["hard_tokens"], atol=0, rtol=0)
        torch.testing.assert_close(
            after["trajectory_embedding"], before["trajectory_embedding"], atol=0, rtol=0
        )
        torch.testing.assert_close(
            after["trajectory_logits"], before["trajectory_logits"], atol=0, rtol=0
        )

    def test_repeated_expansion_preserves_old_path_and_strict_roundtrip(self):
        torch.manual_seed(47)
        model = MotionPrimitiveCGCDModel(_architecture(codebook_size=4)).eval()
        batch = _trajectory_batch(model)
        direction = _force_all_windows_to_old_code_zero(model, batch)
        reference = model(batch, hard_codebook=True)
        trajectory_state = {
            name: value.detach().clone()
            for name, value in model.trajectory_input.state_dict().items()
        }

        expand_motion_codebook_(model, -direction.unsqueeze(0))
        expand_motion_codebook_(model, torch.stack((-direction, -direction)))
        expand_motion_codebook_(model, torch.stack((-direction, -direction, -direction)))
        self.assertEqual(model.codebook_size, 10)
        expanded = model(batch, hard_codebook=True)
        self.assertTrue(torch.equal(expanded["hard_tokens"], reference["hard_tokens"]))
        torch.testing.assert_close(
            expanded["trajectory_embedding"], reference["trajectory_embedding"], atol=0, rtol=0
        )
        torch.testing.assert_close(
            expanded["trajectory_logits"], reference["trajectory_logits"], atol=0, rtol=0
        )
        for name, value in model.trajectory_input.state_dict().items():
            torch.testing.assert_close(value, trajectory_state[name], atol=0, rtol=0)

        optimizer = build_online_sgd(model, MotionPrimitiveOnlineConfig())
        optimizer_parameters = {
            id(parameter)
            for group in optimizer.param_groups
            for parameter in group["params"]
        }
        self.assertIn(id(model.codebook.vectors), optimizer_parameters)
        self.assertEqual(tuple(model.codebook.vectors.shape), (10, 8))

        payload = {
            "schema": "hhr_motion_primitive_online_checkpoint_v3",
            "architecture": model.config.audit_dict(),
            "model": model.state_dict(),
        }
        buffer = io.BytesIO()
        torch.save(payload, buffer)
        buffer.seek(0)
        restored_payload = torch.load(buffer, map_location="cpu", weights_only=False)
        restored_config = MotionPrimitiveConfig(
            **restored_payload["architecture"]
        ).validated()
        restored = MotionPrimitiveCGCDModel(restored_config).eval()
        incompatible = restored.load_state_dict(restored_payload["model"], strict=True)
        self.assertEqual(incompatible.missing_keys, [])
        self.assertEqual(incompatible.unexpected_keys, [])
        self.assertIsNot(restored, model)
        restored_output = restored(batch, hard_codebook=True)
        for key in ("hard_tokens", "trajectory_embedding", "trajectory_logits"):
            torch.testing.assert_close(restored_output[key], expanded[key], atol=0, rtol=0)


if __name__ == "__main__":
    unittest.main()
