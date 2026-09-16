import unittest
import copy
import hashlib
import json
import random
import tempfile
from dataclasses import replace
from pathlib import Path
from unittest import mock

import numpy as np
import torch

from experiments.motion_primitive.one_stage_losses import (
    OneStageLossConfig,
    compose_one_stage_loss,
)
from experiments.motion_primitive.train_one_stage import (
    REQUIRED_COMPLETION_ARTIFACTS,
    SCHEMA_VERSION,
    _capture_rng_state,
    _checkpoint_eligibility,
    _commit_history_record,
    _configure_codebook_trainability,
    _load_verified_completed_summary,
    _loss_config_for_codebook_update,
    _loss_inputs,
    _optimizer_to_device,
    _remaining_epoch_numbers,
    _restore_training_checkpoint,
    _restore_rng_state,
    _selection_key,
    _validate_resume_artifact_state,
    _validate_args,
    build_parser,
)
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
            local_encoder_type="light_cnn",
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

    def test_training_trajectory_assignment_is_discrete_straight_through(self):
        model = MotionTrajectoryModel(self.config).train()
        outputs = model(
            self.model_trials,
            self.lengths,
            state_trials=self.physical_trials,
            hard_codebook=False,
            apply_random_trajectory_mask=False,
        )
        valid = outputs["trajectory_assignments"][outputs["token_mask"]]
        self.assertEqual(
            outputs["trajectory_assignment_forward"],
            "hard_one_hot_straight_through_deterministic",
        )
        self.assertTrue(bool(torch.all((valid == 0.0) | (valid == 1.0))))
        torch.testing.assert_close(
            valid.sum(dim=-1),
            torch.ones(valid.shape[0], dtype=valid.dtype),
        )
        self.assertTrue(outputs["trajectory_assignments"].requires_grad)
        outputs["trajectory_logits"].sum().backward()
        self.assertGreater(self._gradient_sum(model.local_encoder), 0.0)

    def test_primary_training_and_inference_use_the_same_deterministic_tokens(self):
        model = MotionTrajectoryModel(self.config)
        model.train()
        training = model(
            self.model_trials,
            self.lengths,
            state_trials=self.physical_trials,
            hard_codebook=False,
            apply_random_trajectory_mask=False,
        )
        model.eval()
        with torch.no_grad():
            inference = model(
                self.model_trials,
                self.lengths,
                state_trials=self.physical_trials,
                hard_codebook=True,
                apply_random_trajectory_mask=False,
            )
        torch.testing.assert_close(
            training["trajectory_assignments"],
            inference["trajectory_assignments"],
        )
        torch.testing.assert_close(training["hard_tokens"], inference["hard_tokens"])

    def test_resnet1d_local_encoder_preserves_the_one_stage_output_contract(self):
        config = replace(self.config, local_encoder_type="resnet1d").validated()
        model = MotionTrajectoryModel(config).train()
        outputs = model(
            self.model_trials,
            self.lengths,
            state_trials=self.physical_trials,
            hard_codebook=False,
            apply_random_trajectory_mask=True,
        )
        self.assertEqual(
            tuple(outputs["local_embeddings"].shape),
            (2, 4, config.local_embedding_dim),
        )
        outputs["local_embeddings"].sum().backward()
        self.assertGreater(self._gradient_sum(model.local_encoder), 0.0)

    def test_unused_codes_are_revived_from_candidate_features(self):
        model = MotionTrajectoryModel(self.config)
        model.codebook.reset_ema_state(torch.tensor([8.0, 1.0, 1.0, 3.0]))
        before = model.codebook.vectors.detach().clone()
        revived = model.codebook.revive_unused_codes(
            torch.tensor([8, 0, 0, 3]),
            torch.randn(12, self.config.local_embedding_dim),
        )
        self.assertEqual(revived, [1, 2])
        self.assertFalse(torch.equal(before[1:3], model.codebook.vectors[1:3]))
        torch.testing.assert_close(
            model.codebook.ema_cluster_size[1:3], torch.full((2,), 3.0)
        )
        torch.testing.assert_close(
            torch.nn.functional.normalize(
                model.codebook.ema_vector_sum[1:3], dim=-1
            ),
            torch.nn.functional.normalize(model.codebook.vectors[1:3], dim=-1),
        )

    def test_ema_updates_only_assigned_centres_and_keeps_finite_directions(self):
        model = MotionTrajectoryModel(self.config)
        with torch.no_grad():
            model.codebook.vectors.copy_(
                torch.nn.functional.normalize(model.codebook.vectors, dim=-1)
            )
        before = model.codebook.vectors.detach().clone()
        embeddings = torch.randn(2, 3, self.config.local_embedding_dim)
        mask = torch.tensor([[True, True, False], [True, True, True]])
        hard_ids = torch.tensor([[0, 0, -1], [2, 2, 2]])
        updated = model.codebook.ema_update(
            embeddings, mask, hard_ids, decay=0.5
        )
        self.assertEqual(updated, [0, 2])
        self.assertFalse(torch.equal(before[0], model.codebook.vectors[0]))
        self.assertFalse(torch.equal(before[2], model.codebook.vectors[2]))
        torch.testing.assert_close(before[1], model.codebook.vectors[1])
        torch.testing.assert_close(before[3], model.codebook.vectors[3])
        self.assertTrue(torch.isfinite(model.codebook.vectors).all())
        torch.testing.assert_close(
            model.codebook.vectors.norm(dim=-1), torch.ones(4), atol=1e-6, rtol=1e-6
        )

    def test_ema_gives_each_trial_equal_total_mass_regardless_of_length(self):
        short = MotionTrajectoryModel(self.config)
        long = MotionTrajectoryModel(self.config)
        long.load_state_dict(short.state_dict())
        first = torch.zeros(self.config.local_embedding_dim)
        first[0] = 1.0
        second = torch.zeros(self.config.local_embedding_dim)
        second[1] = 1.0
        short_embeddings = torch.stack((first, second)).view(2, 1, -1)
        short_mask = torch.ones(2, 1, dtype=torch.bool)
        short_ids = torch.zeros(2, 1, dtype=torch.long)
        long_embeddings = torch.stack(
            (first, first, first, second, second, second)
        ).view(2, 3, -1)
        long_mask = torch.ones(2, 3, dtype=torch.bool)
        long_ids = torch.zeros(2, 3, dtype=torch.long)
        short.codebook.ema_update(
            short_embeddings, short_mask, short_ids, decay=0.0
        )
        long.codebook.ema_update(long_embeddings, long_mask, long_ids, decay=0.0)
        torch.testing.assert_close(
            short.codebook.vectors[0], long.codebook.vectors[0], atol=1e-6, rtol=1e-6
        )

    def test_primary_classifier_is_primitive_only_and_state_invariant(self):
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
            changed_state = model.encode_trajectory(
                outputs["trajectory_assignments"],
                outputs["token_mask"],
                outputs["state_features"] + 100.0,
                outputs["boundary_probabilities"],
                outputs["transition_probabilities"],
                outputs["duration_proxy"],
                apply_random_mask=False,
            )
        self.assertEqual(
            model.trajectory_feature_dim,
            self.config.codebook_size + 3,
        )
        self.assertEqual(self.config.trajectory_input_mode, "primitive_only")
        torch.testing.assert_close(
            outputs["trajectory_embedding"],
            changed_state["trajectory_embedding"],
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

    def test_state_ingress_requires_an_explicit_attribution_mode(self):
        config = replace(
            self.config,
            trajectory_input_mode="primitive_plus_state",
        ).validated()
        model = MotionTrajectoryModel(config).eval()
        with torch.no_grad():
            outputs = model(
                self.model_trials,
                self.lengths,
                state_trials=self.physical_trials,
                hard_codebook=True,
                apply_random_trajectory_mask=False,
            )
            changed_state = model.encode_trajectory(
                outputs["trajectory_assignments"],
                outputs["token_mask"],
                outputs["state_features"] + torch.randn_like(outputs["state_features"]),
                outputs["boundary_probabilities"],
                outputs["transition_probabilities"],
                outputs["duration_proxy"],
                apply_random_mask=False,
            )
        self.assertEqual(
            model.trajectory_feature_dim,
            config.codebook_size + config.state_dim + 3,
        )
        self.assertGreater(
            float(
                torch.max(
                    torch.abs(
                        outputs["trajectory_embedding"]
                        - changed_state["trajectory_embedding"]
                    )
                )
            ),
            0.0,
        )


class OneStageRunnerGuardTests(unittest.TestCase):
    @staticmethod
    def _args(*extra: str):
        return build_parser().parse_args(
            [
                "--output-dir",
                "unused",
                "--profile",
                "J0-U",
                "--fold",
                "1",
                *extra,
            ]
        )

    def test_zero_trajectory_mask_ratio_is_rejected_at_cli_boundary(self):
        with self.assertRaisesRegex(ValueError, "strictly between zero and one"):
            _validate_args(self._args("--trajectory-mask-ratio", "0"))

    def test_primary_defaults_are_primitive_only_with_common_selection(self):
        args = self._args()
        self.assertEqual(args.trajectory_input_mode, "primitive_only")
        self.assertEqual(args.local_encoder, "resnet1d")
        self.assertEqual(args.checkpoint_selection, "common_unsupervised")
        self.assertEqual(args.codebook_init, "learnable")
        self.assertEqual(args.codebook_update, "gradient")
        self.assertEqual(args.loss_ramp_epochs, 10)
        self.assertFalse(args.revive_unused_codes)
        self.assertAlmostEqual(args.codebook_ema_decay, 0.99)
        self.assertAlmostEqual(args.minimum_hard_effective_code_fraction, 0.20)
        self.assertFalse(args.use_gumbel_training)

    def test_ema_codebook_has_one_update_owner_and_is_not_in_optimizer(self):
        model = MotionTrajectoryModel(
            MotionTrajectoryConfig(
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
        )
        loss_config = _loss_config_for_codebook_update(
            OneStageLossConfig.j0_u(old_class_count=3), "ema"
        )
        parameters = _configure_codebook_trainability(model, "ema")
        optimizer = torch.optim.AdamW(parameters, lr=1e-3)
        self.assertFalse(model.codebook.vectors.requires_grad)
        self.assertEqual(loss_config.vq_codebook_weight, 0.0)
        self.assertEqual(loss_config.codebook_diversity_weight, 0.0)
        self.assertTrue(
            all(
                parameter is not model.codebook.vectors
                for group in optimizer.param_groups
                for parameter in group["params"]
            )
        )

    def test_primary_gradient_codebook_is_optimizer_owned_and_normalized(self):
        model = MotionTrajectoryModel(
            MotionTrajectoryConfig(
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
        )
        config = OneStageLossConfig.j0_u(old_class_count=3)
        self.assertIs(_loss_config_for_codebook_update(config, "gradient"), config)
        parameters = _configure_codebook_trainability(model, "gradient")
        self.assertTrue(model.codebook.vectors.requires_grad)
        self.assertTrue(any(parameter is model.codebook.vectors for parameter in parameters))
        model.codebook.normalize_learnable_prototypes_()
        torch.testing.assert_close(
            model.codebook.vectors.norm(dim=-1),
            torch.ones(model.codebook_size),
        )

    def test_legacy_warmup_flag_is_only_a_loss_ramp_alias(self):
        args = self._args("--warmup-epochs", "7")
        self.assertEqual(args.loss_ramp_epochs, 7)
        self.assertFalse(hasattr(args, "warmup_epochs"))

    def test_learnable_initialization_cannot_silently_switch_to_ema(self):
        with self.assertRaisesRegex(ValueError, "learnable requires"):
            _validate_args(self._args("--codebook-update", "ema"))

    def test_checkpoint_gate_rejects_soft_full_but_hard_collapsed_codebook(self):
        eligible, reasons = _checkpoint_eligibility(
            {
                "hard_occupancy_fraction": 1.0 / 32.0,
                "hard_effective_code_fraction": 1.0 / 32.0,
                "hard_max_code_share": 1.0,
                "local_feature_effective_rank": 3.1,
                "local_feature_centroid_norm": 0.99996,
                "mean_assignment_margin": 0.00118,
                "effective_code_count": 31.87,
            },
            minimum_hard_code_fraction=0.25,
            minimum_hard_effective_code_fraction=0.20,
            maximum_hard_code_share=0.50,
            minimum_local_feature_effective_rank=4.0,
            maximum_local_feature_centroid_norm=0.98,
            minimum_assignment_margin=0.01,
        )
        self.assertFalse(eligible)
        self.assertEqual(len(reasons), 6)

    def test_checkpoint_gate_rejects_many_occupied_but_low_effective_usage(self):
        eligible, reasons = _checkpoint_eligibility(
            {
                "hard_occupancy_fraction": 11.0 / 32.0,
                "hard_effective_code_fraction": 4.56 / 32.0,
                "hard_max_code_share": 0.34,
                "local_feature_effective_rank": 16.5,
                "local_feature_centroid_norm": 0.94,
                "mean_assignment_margin": 0.027,
            },
            minimum_hard_code_fraction=0.25,
            minimum_hard_effective_code_fraction=0.20,
            maximum_hard_code_share=0.50,
            minimum_local_feature_effective_rank=4.0,
            maximum_local_feature_centroid_norm=0.98,
            minimum_assignment_margin=0.01,
        )
        self.assertFalse(eligible)
        self.assertEqual(len(reasons), 1)
        self.assertIn("hard_effective_code_fraction", reasons[0])

    def test_common_selection_does_not_use_validation_activity_labels(self):
        validation = {
            "unsupervised_loss": 2.5,
            "effective_code_count": 7.0,
            "hard_effective_code_count": 5.0,
            "trajectory_macro_f1": 0.99,
            "loss": 20.0,
        }
        self.assertEqual(
            _selection_key("J0-U", validation, "common_unsupervised"),
            _selection_key("J0-T", validation, "common_unsupervised"),
        )
        self.assertNotEqual(
            _selection_key("J0-T", validation, "profile_default"),
            _selection_key("J0-T", validation, "common_unsupervised"),
        )

    def test_rng_state_restores_python_numpy_torch_and_loader_generator(self):
        generator = torch.Generator().manual_seed(17)
        loader = torch.utils.data.DataLoader([0, 1], generator=generator)
        with mock.patch("torch.cuda.is_available", return_value=False):
            state = _capture_rng_state(loader)
            expected = (
                __import__("random").random(),
                float(np.random.random()),
                float(torch.rand(())),
                torch.randperm(9, generator=loader.generator),
            )
            _restore_rng_state(state, loader)
            observed = (
                __import__("random").random(),
                float(np.random.random()),
                float(torch.rand(())),
                torch.randperm(9, generator=loader.generator),
            )
        self.assertEqual(expected[:3], observed[:3])
        torch.testing.assert_close(expected[3], observed[3])

    def test_completed_artifacts_are_hash_and_identity_checked(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            identity = "identity-a"
            for relative in REQUIRED_COMPLETION_ARTIFACTS:
                (root / relative).write_bytes(f"artifact:{relative}".encode("utf-8"))
            checkpoint = root / "checkpoint_best.pt"
            checkpoint.write_bytes(b"checkpoint")
            checkpoint_hash = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
            summary = {
                "run_identity_sha256": identity,
                "selected_checkpoint_sha256": checkpoint_hash,
            }
            summary_path = root / "summary.json"
            summary_path.write_text(json.dumps(summary), encoding="utf-8")
            summary_hash = hashlib.sha256(summary_path.read_bytes()).hexdigest()
            artifact_hashes = {
                relative: hashlib.sha256((root / relative).read_bytes()).hexdigest()
                for relative in REQUIRED_COMPLETION_ARTIFACTS
            }
            (root / "complete.json").write_text(
                json.dumps(
                    {
                        "run_identity_sha256": identity,
                        "summary_sha256": summary_hash,
                        "selected_checkpoint_sha256": checkpoint_hash,
                        "artifact_sha256": artifact_hashes,
                    }
                ),
                encoding="utf-8",
            )
            observed = _load_verified_completed_summary(root, identity)
            self.assertEqual(observed, summary)
            with self.assertRaisesRegex(RuntimeError, "requested run identity"):
                _load_verified_completed_summary(root, "identity-b")
            summary_path.write_text(json.dumps({**summary, "tampered": True}), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "summary.json failed"):
                _load_verified_completed_summary(root, identity)

    @staticmethod
    def _make_tiny_training(seed: int):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        features = torch.arange(48, dtype=torch.float32).reshape(12, 4) / 47.0
        targets = torch.linspace(-0.5, 0.5, 12).reshape(-1, 1)
        dataset = torch.utils.data.TensorDataset(
            torch.arange(12), features, targets
        )
        generator = torch.Generator().manual_seed(seed)
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=3,
            shuffle=True,
            generator=generator,
            num_workers=0,
        )
        model = torch.nn.Sequential(
            torch.nn.Linear(4, 7),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.25),
            torch.nn.Linear(7, 1),
        )
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=2)
        return model, optimizer, scheduler, loader

    @staticmethod
    def _tiny_epoch(model, optimizer, scheduler, loader):
        order = []
        stochastic_trace = []
        model.train()
        for indices, features, targets in loader:
            order.extend(int(item) for item in indices)
            random_term = random.random()
            numpy_term = float(np.random.random())
            torch_term = torch.rand((), dtype=features.dtype)
            stochastic_trace.append((random_term, numpy_term, float(torch_term)))
            optimizer.zero_grad(set_to_none=True)
            prediction = model(features)
            loss = torch.nn.functional.mse_loss(prediction, targets)
            loss = loss + 1e-4 * (random_term + numpy_term + torch_term)
            loss.backward()
            optimizer.step()
        scheduler.step()
        return order, stochastic_trace

    @staticmethod
    def _assert_nested_equal(test_case, expected, observed):
        if isinstance(expected, torch.Tensor):
            torch.testing.assert_close(expected, observed, rtol=0.0, atol=0.0)
        elif isinstance(expected, dict):
            test_case.assertEqual(set(expected), set(observed))
            for key in expected:
                OneStageRunnerGuardTests._assert_nested_equal(
                    test_case, expected[key], observed[key]
                )
        elif isinstance(expected, (list, tuple)):
            test_case.assertEqual(len(expected), len(observed))
            for left, right in zip(expected, observed):
                OneStageRunnerGuardTests._assert_nested_equal(
                    test_case, left, right
                )
        else:
            test_case.assertEqual(expected, observed)

    def test_two_epochs_equal_one_epoch_then_exact_resume(self):
        seed = 20260908
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkpoint_path = root / "checkpoint_last.pt"
            history_path = root / "history.jsonl"
            run_identity = "tiny-primitive-only-resume"

            model_a, optimizer_a, scheduler_a, loader_a = self._make_tiny_training(seed)
            first_order, first_random = self._tiny_epoch(
                model_a, optimizer_a, scheduler_a, loader_a
            )
            first_record = {
                "epoch": 1,
                "trajectory_input_mode": "primitive_only",
                "batch_order": first_order,
                "stochastic_trace": first_random,
            }
            torch.save(
                {
                    "schema": SCHEMA_VERSION,
                    "run_identity_sha256": run_identity,
                    "epoch": 1,
                    "model": model_a.state_dict(),
                    "optimizer": optimizer_a.state_dict(),
                    "scheduler": scheduler_a.state_dict(),
                    "rng_state": _capture_rng_state(loader_a),
                    "history_record": first_record,
                    "history_record_sha256": hashlib.sha256(
                        json.dumps(
                            first_record,
                            ensure_ascii=False,
                            sort_keys=True,
                            separators=(",", ":"),
                            allow_nan=False,
                        ).encode("utf-8")
                    ).hexdigest(),
                    "history_sha256": hashlib.sha256(
                        json.dumps(
                            [first_record],
                            ensure_ascii=False,
                            sort_keys=True,
                            separators=(",", ":"),
                            allow_nan=False,
                        ).encode("utf-8")
                    ).hexdigest(),
                    "stopped_early": False,
                },
                checkpoint_path,
            )
            expected_order, expected_random = self._tiny_epoch(
                model_a, optimizer_a, scheduler_a, loader_a
            )
            expected_model = copy.deepcopy(model_a.state_dict())
            expected_optimizer = copy.deepcopy(optimizer_a.state_dict())
            expected_scheduler = copy.deepcopy(scheduler_a.state_dict())

            model_b, optimizer_b, scheduler_b, loader_b = self._make_tiny_training(seed)
            # Prove restoration does not accidentally depend on the process's
            # current random state.
            random.random()
            np.random.random()
            torch.rand(13)
            torch.randperm(12, generator=loader_b.generator)
            restored = _restore_training_checkpoint(
                checkpoint_path,
                run_identity=run_identity,
                model=model_b,
                optimizer=optimizer_b,
                scheduler=scheduler_b,
                train_loader=loader_b,
                history_path=history_path,
                device=torch.device("cpu"),
            )
            self.assertEqual(restored["epoch"], 1)
            observed_order, observed_random = self._tiny_epoch(
                model_b, optimizer_b, scheduler_b, loader_b
            )

            self.assertEqual(expected_order, observed_order)
            self.assertEqual(expected_random, observed_random)
            self._assert_nested_equal(self, expected_model, model_b.state_dict())
            self._assert_nested_equal(
                self, expected_optimizer, optimizer_b.state_dict()
            )
            self._assert_nested_equal(
                self, expected_scheduler, scheduler_b.state_dict()
            )
            committed = [
                json.loads(line)
                for line in history_path.read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(
                committed,
                [json.loads(json.dumps(first_record))],
            )

    def test_checkpoint_history_commit_repairs_only_missing_final_record(self):
        with tempfile.TemporaryDirectory() as temporary:
            history = Path(temporary) / "history.jsonl"
            first = {"epoch": 1, "loss": 2.0}
            second = {"epoch": 2, "loss": 1.5}
            _commit_history_record(history, first)
            _commit_history_record(history, first)
            _commit_history_record(history, second)
            self.assertEqual(
                [json.loads(line) for line in history.read_text().splitlines()],
                [first, second],
            )
            with self.assertRaisesRegex(RuntimeError, "conflicts"):
                _commit_history_record(history, {"epoch": 2, "loss": 99.0})
            with self.assertRaisesRegex(RuntimeError, "ahead"):
                _commit_history_record(history, first)

    def test_partial_or_gapped_history_fails_closed(self):
        with tempfile.TemporaryDirectory() as temporary:
            history = Path(temporary) / "history.jsonl"
            history.write_text('{"epoch":1}\n{"epoch":', encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "invalid JSON"):
                _commit_history_record(history, {"epoch": 2})
        with tempfile.TemporaryDirectory() as temporary:
            history = Path(temporary) / "history.jsonl"
            with self.assertRaisesRegex(RuntimeError, "gap"):
                _commit_history_record(history, {"epoch": 2})

    def test_early_stopped_checkpoint_has_no_remaining_training_epochs(self):
        self.assertEqual(tuple(_remaining_epoch_numbers(7, 80, True)), ())
        self.assertEqual(tuple(_remaining_epoch_numbers(7, 9, False)), (7, 8, 9))

    def test_resume_without_last_checkpoint_rejects_orphan_training_artifacts(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            last = root / "checkpoint_last.pt"
            history = root / "history.jsonl"
            best = root / "checkpoint_best.pt"
            _validate_resume_artifact_state(last, history, best)
            history.write_text("", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "checkpoint_last.pt is missing"):
                _validate_resume_artifact_state(last, history, best)
            last.write_bytes(b"present")
            _validate_resume_artifact_state(last, history, best)

    def test_every_completed_artifact_is_fail_closed_on_missing_or_tamper(self):
        def materialize(root: Path):
            identity = "artifact-integrity"
            for relative in REQUIRED_COMPLETION_ARTIFACTS:
                (root / relative).write_bytes(f"artifact:{relative}".encode())
            checkpoint_hash = hashlib.sha256(
                (root / "checkpoint_best.pt").read_bytes()
            ).hexdigest()
            summary = {
                "run_identity_sha256": identity,
                "selected_checkpoint_sha256": checkpoint_hash,
            }
            (root / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
            artifact_hashes = {
                relative: hashlib.sha256((root / relative).read_bytes()).hexdigest()
                for relative in REQUIRED_COMPLETION_ARTIFACTS
            }
            complete = {
                "run_identity_sha256": identity,
                "summary_sha256": hashlib.sha256(
                    (root / "summary.json").read_bytes()
                ).hexdigest(),
                "selected_checkpoint_sha256": checkpoint_hash,
                "artifact_sha256": artifact_hashes,
            }
            (root / "complete.json").write_text(json.dumps(complete), encoding="utf-8")
            return identity

        for relative in REQUIRED_COMPLETION_ARTIFACTS:
            with self.subTest(relative=relative, mutation="missing"):
                with tempfile.TemporaryDirectory() as temporary:
                    root = Path(temporary)
                    identity = materialize(root)
                    (root / relative).unlink()
                    with self.assertRaises(RuntimeError):
                        _load_verified_completed_summary(root, identity)
            with self.subTest(relative=relative, mutation="tampered"):
                with tempfile.TemporaryDirectory() as temporary:
                    root = Path(temporary)
                    identity = materialize(root)
                    with (root / relative).open("ab") as handle:
                        handle.write(b"tamper")
                    with self.assertRaises(RuntimeError):
                        _load_verified_completed_summary(root, identity)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is unavailable")
    def test_cpu_loaded_optimizer_state_moves_to_cuda_and_steps(self):
        cpu_model = torch.nn.Linear(3, 2)
        cpu_optimizer = torch.optim.AdamW(cpu_model.parameters(), lr=1e-3)
        cpu_optimizer.zero_grad(set_to_none=True)
        cpu_model(torch.randn(4, 3)).sum().backward()
        cpu_optimizer.step()
        state = copy.deepcopy(cpu_optimizer.state_dict())

        cuda_device = torch.device("cuda")
        cuda_model = torch.nn.Linear(3, 2).to(cuda_device)
        cuda_optimizer = torch.optim.AdamW(cuda_model.parameters(), lr=1e-3)
        cuda_optimizer.load_state_dict(state)
        _optimizer_to_device(cuda_optimizer, cuda_device)
        for parameter_state in cuda_optimizer.state.values():
            for value in parameter_state.values():
                if isinstance(value, torch.Tensor):
                    self.assertEqual(value.device.type, "cuda")
        cuda_optimizer.zero_grad(set_to_none=True)
        cuda_model(torch.randn(4, 3, device=cuda_device)).sum().backward()
        cuda_optimizer.step()


if __name__ == "__main__":
    unittest.main()
