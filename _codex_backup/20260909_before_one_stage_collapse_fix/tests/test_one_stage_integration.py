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
    _commit_history_record,
    _load_verified_completed_summary,
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
            "hard_one_hot_straight_through_gumbel",
        )
        self.assertTrue(bool(torch.all((valid == 0.0) | (valid == 1.0))))
        torch.testing.assert_close(
            valid.sum(dim=-1),
            torch.ones(valid.shape[0], dtype=valid.dtype),
        )
        self.assertTrue(outputs["trajectory_assignments"].requires_grad)
        outputs["trajectory_logits"].sum().backward()
        self.assertGreater(self._gradient_sum(model.local_encoder), 0.0)

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
        self.assertEqual(args.checkpoint_selection, "common_unsupervised")

    def test_common_selection_does_not_use_validation_activity_labels(self):
        validation = {
            "unsupervised_loss": 2.5,
            "effective_code_count": 7.0,
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
