import copy
import json
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn

from experiments.motion_primitive.motion_augmentation import MotionAugmentationConfig
from experiments.motion_primitive.motion_encoder import MotionPrimitiveEncoder
from experiments.motion_primitive.motion_checkpoint import (
    COMMAND_ARGUMENTS_V1,
    motion_state_dict_sha256,
)
from experiments.motion_primitive.train_motion_encoder import (
    TrialExample,
    _validate_source_npz_binding,
    build_parser,
    load_train_validation_trials,
    resolve_configuration,
    train_one_epoch,
)


def _tiny_encoder() -> MotionPrimitiveEncoder:
    return MotionPrimitiveEncoder(
        in_channels=6,
        backbone_dim=8,
        base_channels=2,
        backbone_layers=[1, 1, 1],
        backbone_dropout=0.0,
        segmentation_dim=4,
        content_dim=8,
        content_residual=True,
        augmentation_dim=4,
        projection_hidden_dim=8,
        num_classes=2,
        trial_hidden_dim=4,
        trial_peak_quantile=0.9,
        trial_dropout=0.0,
        predictor_hidden_dim=4,
    )


def _disabled_augmentation() -> MotionAugmentationConfig:
    return MotionAugmentationConfig(
        noise_std_ratio=0.0,
        acc_scale_range=(1.0, 1.0),
        gyro_scale_range=(1.0, 1.0),
        time_shift_max_samples=0,
        time_mask_min_samples=0,
        time_mask_max_samples=0,
        rotation_max_degrees=0.0,
    )


def _trial(trial_id: int, label: int) -> TrialExample:
    generator = torch.Generator().manual_seed(100 + trial_id)
    return TrialExample(
        trial_id=trial_id,
        subject_id=trial_id + 1,
        label=label,
        label_1based=label + 1,
        trial_number=1,
        starts=torch.tensor([0, 16, 32], dtype=torch.long),
        raw_trial=torch.randn(6, 64, generator=generator),
        clean_windows=torch.randn(3, 6, 32, generator=generator),
    )


class FrozenBatchNormProtocolTests(unittest.TestCase):
    def test_one_train_step_keeps_backbone_bn_running_statistics_fixed(self):
        torch.manual_seed(5)
        model = _tiny_encoder()
        for parameter in model.backbone.parameters():
            parameter.requires_grad_(False)
        teacher = copy.deepcopy(model).eval()
        for parameter in teacher.parameters():
            parameter.requires_grad_(False)
        optimizer = torch.optim.AdamW(
            [parameter for parameter in model.parameters() if parameter.requires_grad],
            lr=1e-3,
        )
        augmentation = asdict(_disabled_augmentation())
        config = {
            "backbone_bn_policy": "frozen",
            "augmentation": augmentation,
            "cp_aligned_augmentation": augmentation,
            "window_aug_consistency": "none",
            "window_aug_one_window_per_trial": True,
            "loss_weights": {
                "window_augmentation": 0.0,
                "noncollapse": 0.0,
                "changepoint": 0.0,
                "content_boundary_alignment": 0.0,
                "temporal_prediction": 0.0,
                "trial_auxiliary": 1.0,
                "cross_subject": 0.0,
            },
        }
        args = SimpleNamespace(
            trial_batch_size=2,
            noncollapse_target_std=1.0,
            noncollapse_variance_weight=1.0,
            noncollapse_covariance_weight=1.0,
            noncollapse_windows_per_trial=3,
            prediction_mask_ratio=0.2,
            prediction_loss="cosine",
            prediction_huber_delta=1.0,
            gradient_clip_norm=5.0,
            ema_momentum=0.99,
        )
        batch_norms = [
            module
            for module in model.backbone.modules()
            if isinstance(module, nn.BatchNorm1d)
        ]
        self.assertTrue(batch_norms)
        before = [
            (module.running_mean.clone(), module.running_var.clone())
            for module in batch_norms
        ]
        train_one_epoch(
            model,
            teacher,
            [_trial(0, 0), _trial(1, 1)],
            optimizer,
            torch.device("cpu"),
            torch.zeros(6),
            torch.ones(6),
            config,
            args,
            torch.Generator().manual_seed(9),
        )
        for module, (mean_before, variance_before) in zip(batch_norms, before):
            torch.testing.assert_close(module.running_mean, mean_before, rtol=0, atol=0)
            torch.testing.assert_close(module.running_var, variance_before, rtol=0, atol=0)


class AblationIdentityProtocolTests(unittest.TestCase):
    @staticmethod
    def _legacy_metadata() -> dict:
        return {
            "har_in_channels": 6,
            "har_feat_dim": 256,
            "har_base_channels": 64,
            "har_dropout": 0.0,
        }

    def test_a3_rejects_frozen_legacy_anchor(self):
        args = build_parser().parse_args(
            ["--ablation-profile", "A3", "--cp-anchor-source", "frozen_legacy"]
        )
        with self.assertRaisesRegex(ValueError, "A3 identity conflict.*cp_anchor_source"):
            resolve_configuration(args, self._legacy_metadata())

    def test_formal_profiles_use_conservative_trial_auxiliary_weight(self):
        args = build_parser().parse_args(["--ablation-profile", "A3"])
        resolved = resolve_configuration(args, self._legacy_metadata())
        self.assertAlmostEqual(resolved["loss_weights"]["trial_auxiliary"], 0.1)

    def test_checkpoint_argument_schema_tracks_every_trainer_option(self):
        parser_keys = set(vars(build_parser().parse_args([])))
        self.assertEqual(parser_keys, set(COMMAND_ARGUMENTS_V1))

    def test_infonce_profile_rejects_zero_augmentation_weight(self):
        args = build_parser().parse_args(
            ["--ablation-profile", "A3", "--window-aug-weight", "0"]
        )
        with self.assertRaisesRegex(
            ValueError, "A3 identity conflict.*window_aug_weight"
        ):
            resolve_configuration(args, self._legacy_metadata())

    def test_custom_explicitly_allows_frozen_legacy_anchor(self):
        args = build_parser().parse_args(
            [
                "--ablation-profile", "CUSTOM",
                "--window-aug-consistency", "infonce",
                "--window-aug-profile", "basic",
                "--cp-weight", "1",
                "--content-boundary-alignment-weight", "0.1",
                "--rotation-max-degrees", "0",
                "--cp-anchor-source", "frozen_legacy",
            ]
        )
        resolved = resolve_configuration(args, self._legacy_metadata())
        self.assertEqual(resolved["ablation_profile"], "CUSTOM")
        self.assertEqual(resolved["cp_anchor"]["source"], "frozen_legacy")


class SourceCheckpointDatasetBindingTests(unittest.TestCase):
    def test_exact_recorded_npz_path_is_accepted(self):
        with tempfile.TemporaryDirectory() as temporary:
            npz_path = Path(temporary) / "windows.npz"
            audit = _validate_source_npz_binding(
                {"uschad_npz_path": str(npz_path)}, npz_path
            )
            self.assertTrue(audit["source_checkpoint_npz_path_matches_requested"])
            self.assertEqual(
                Path(audit["requested_npz_path_resolved"]), npz_path.resolve()
            )

    def test_different_same_shape_dataset_path_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with self.assertRaisesRegex(RuntimeError, "identity mismatch"):
                _validate_source_npz_binding(
                    {"uschad_npz_path": str(root / "original.npz")},
                    root / "swapped.npz",
                )

    def test_missing_recorded_npz_path_is_rejected(self):
        with self.assertRaisesRegex(RuntimeError, "lacks uschad_npz_path"):
            _validate_source_npz_binding({}, Path("windows.npz"))


def _write_anomaly_label_fixture(path: Path, include_original_labels: bool) -> None:
    # Trial 100 is the physical Subject14/a3t2 anomaly after a diagnostic label
    # permutation.  Trial 101 has protocol label 3 but physical activity 9 and
    # must not be mistaken for the anomaly when original labels are available.
    labels = np.asarray([8, 2, 0], dtype=np.int64)
    labels_1based = labels + 1
    original_labels_1based = np.asarray([3, 9, 1], dtype=np.int64)
    subject_ids = np.asarray([14, 14, 1], dtype=np.int64)
    trial_numbers = np.asarray([2, 2, 1], dtype=np.int64)
    trial_ids = np.asarray([100, 101, 200], dtype=np.int64)
    time = np.arange(8, dtype=np.float32)
    windows = np.stack(
        [
            np.stack(
                [time * (channel + 1) + row for channel in range(6)]
            )
            for row in range(3)
        ]
    ).astype(np.float32)
    payload = {
        "windows": windows,
        "labels": labels,
        "labels_1based": labels_1based,
        "subject_ids": subject_ids,
        "trial_numbers": trial_numbers,
        "trial_global_ids": trial_ids,
        "window_indices": np.zeros(3, dtype=np.int64),
        "window_start_indices": np.zeros(3, dtype=np.int64),
        "mean": np.zeros((1, 6, 1), dtype=np.float32),
        "std": np.ones((1, 6, 1), dtype=np.float32),
        "channel_names": np.asarray(
            ["acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"],
            dtype=object,
        ),
    }
    if include_original_labels:
        payload["original_labels_1based"] = original_labels_1based
    np.savez_compressed(path, **payload)


class AnomalyPhysicalLabelProtocolTests(unittest.TestCase):
    @staticmethod
    def _split() -> dict:
        return {
            "train_subjects": [14],
            "validation_subjects": [1],
            "test_subjects": [2],
        }

    def test_report_and_exclude_use_original_physical_labels_after_swap(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "swapped.npz"
            _write_anomaly_label_fixture(path, include_original_labels=True)

            reported, _, _, _, report_audit = load_train_validation_trials(
                path,
                self._split(),
                old_class_count=12,
                norm_eps=1e-6,
                anomaly_policy="report",
            )
            self.assertEqual(report_audit["anomaly_label_source"], "original_labels_1based")
            self.assertEqual(
                report_audit["known_anomaly_counts"]["train_windows_before_policy"],
                1,
            )
            self.assertEqual({record.trial_id for record in reported}, {100, 101})
            self.assertEqual(
                {record.trial_id: record.label_1based for record in reported},
                {100: 9, 101: 3},
            )

            excluded, _, _, _, exclude_audit = load_train_validation_trials(
                path,
                self._split(),
                old_class_count=12,
                norm_eps=1e-6,
                anomaly_policy="exclude",
            )
            self.assertEqual(exclude_audit["anomaly_label_source"], "original_labels_1based")
            self.assertEqual(
                exclude_audit["known_anomaly_counts"]["train_windows_before_policy"],
                1,
            )
            self.assertEqual([record.trial_id for record in excluded], [101])
            self.assertEqual(excluded[0].label_1based, 3)

    def test_ordinary_npz_falls_back_to_protocol_labels_for_anomalies(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "ordinary.npz"
            _write_anomaly_label_fixture(path, include_original_labels=False)
            excluded, _, _, _, audit = load_train_validation_trials(
                path,
                self._split(),
                old_class_count=12,
                norm_eps=1e-6,
                anomaly_policy="exclude",
            )
            self.assertEqual(audit["anomaly_label_source"], "labels_1based")
            self.assertEqual(
                audit["known_anomaly_counts"]["train_windows_before_policy"], 1
            )
            self.assertEqual([record.trial_id for record in excluded], [100])
            self.assertEqual(excluded[0].label_1based, 9)


def _encoder_payload(root: Path, profile: str, suffix: str) -> Path:
    candidate = root / "fold_01" / profile.lower() / suffix / "motion_encoder_final.pt"
    candidate.parent.mkdir(parents=True, exist_ok=True)
    state = {"probe": torch.tensor([1.0, 2.0])}
    torch.save(
        {
            "checkpoint_type": "motion_primitive_encoder",
            "schema_version": 1,
            "model": state,
            "model_state_dict": state,
            "model_state_dict_sha256": motion_state_dict_sha256(state),
            "experiment_metadata": {
                "uschad_cv_fold": 1,
                "motion_encoder_seed": 500,
                "smoke_test": False,
            },
            "resolved_training_config": {
                "ablation_profile": profile,
                "backbone_bn_policy": "frozen",
                "cp_anchor": {"source": "raw_frozen_consensus"},
            },
            "selection": {"policy": "final_epoch"},
            "source_checkpoint": {
                "path": str(root / "legacy" / "fold_01" / "seed_500_offline" / "model_best.pt"),
                "legacy_experiment_metadata": {"uschad_cv_fold": 1},
            },
        },
        candidate,
    )
    return candidate.resolve()


def _canonical_trainer_payload(root: Path, profile: str) -> Path:
    candidate = (
        root
        / f"fold_01_seed_500_{profile}_20260903-120000"
        / "motion_encoder_final.pt"
    )
    candidate.parent.mkdir(parents=True, exist_ok=True)
    state = {"probe": torch.tensor([3.0, 4.0])}
    torch.save(
        {
            "checkpoint_type": "motion_primitive_encoder",
            "schema_version": 1,
            "model": state,
            "model_state_dict": state,
            "model_state_dict_sha256": motion_state_dict_sha256(state),
            "experiment_metadata": {
                "uschad_cv_fold": 1,
                "motion_encoder_seed": 500,
                "smoke_test": False,
            },
            "resolved_training_config": {
                "ablation_profile": profile,
                "backbone_bn_policy": "frozen",
                "cp_anchor": {"source": "raw_frozen_consensus"},
            },
            "selection": {"policy": "final_epoch"},
            "source_checkpoint": {
                "path": str(
                    root
                    / "legacy"
                    / "fold_01"
                    / "seed_500_offline"
                    / "model_best.pt"
                ),
                "legacy_experiment_metadata": {"uschad_cv_fold": 1},
            },
        },
        candidate,
    )
    return candidate.resolve()


def _formal_grid_payload(seed: int = 500) -> dict:
    state = {"probe": torch.tensor([float(seed)])}
    npz_hash = "a" * 64
    command_arguments = vars(build_parser().parse_args([])).copy()
    command_arguments.update(
        {
            "source_checkpoint": f"legacy/seed_{seed}_offline/model_best.pt",
            "npz_path": "data/uschad_windows.npz",
            "output_dir": f"output/seed_{seed}",
            "seed": seed,
            "device": "cuda:0" if seed == 500 else "cuda:1",
            "self_test": False,
            "ablation_profile": "A3",
        }
    )
    return {
        "checkpoint_type": "motion_primitive_encoder",
        "schema_version": 1,
        "model": state,
        "model_state_dict": state,
        "model_state_dict_sha256": motion_state_dict_sha256(state),
        "architecture": {"test_architecture": "same-across-grid"},
        "implementation_fingerprint": {
            "algorithm": "sha256_path_and_file_sha256_v1",
            "files": {"motion_encoder.py": "b" * 64},
            "combined_sha256": "c" * 64,
        },
        "npz_sha256": npz_hash,
        "data": {"npz_sha256": npz_hash},
        "resolved_training_config": {
            "ablation_profile": "A3",
            "backbone_bn_policy": "frozen",
            "cp_anchor": {"source": "raw_frozen_consensus"},
        },
        "command_arguments": command_arguments,
        "selection": {
            "policy": "final_epoch",
            "file_role": "canonical_final",
            "completed_epochs": 30,
            "selected_epoch_1based": 30,
            "outer_test_queries": 0,
        },
        "split_audit": {
            "npz_sha256": npz_hash,
            "smoke_test": False,
            "outer_test_sensor_windows_selected": 0,
            "outer_test_model_forward_calls": 0,
        },
    }


@unittest.skip(
    "Historical run_subject_cv grid contracts were replaced by strict_encoder_cv_runner."
)
class EncoderGridIdentityTests(unittest.TestCase):
    def test_persisted_grid_identity_accepts_json_tuple_list_normalisation(self):
        in_memory = {
            "schema": "motion_encoder_cv_grid_identity_v1",
            "training_identity": {
                "architecture": {"backbone_layers": (2, 2, 2)},
                "augmentation": {"scale_range": (0.95, 1.05)},
            },
            "members": [{"fold": 1, "seed": 0}],
        }
        persisted = json.loads(json.dumps(in_memory))

        self.assertNotEqual(persisted, in_memory)
        self.assertTrue(_json_documents_equal(persisted, in_memory))

        changed = copy.deepcopy(persisted)
        changed["members"][0]["seed"] = 5
        self.assertFalse(_json_documents_equal(changed, in_memory))

    def test_identity_ignores_member_paths_seed_and_device(self):
        first = _formal_grid_payload(seed=500)
        second = _formal_grid_payload(seed=1000)
        self.assertEqual(
            motion_encoder_grid_identity(first),
            motion_encoder_grid_identity(second),
        )

    def test_grid_rejects_any_loss_or_schedule_parameter_drift(self):
        with tempfile.TemporaryDirectory(prefix="motion_encoder_grid_identity_") as temporary:
            root = Path(temporary)
            first_path = root / "first.pt"
            second_path = root / "second.pt"
            torch.save(_formal_grid_payload(seed=500), first_path)
            for field, value in {
                "learning_rate": 2e-4,
                "window_aug_temperature": 0.3,
                "cp_high_quantile": 0.8,
                "prediction_mask_ratio": 0.3,
                "vicreg_covariance_weight": 2.0,
            }.items():
                second = _formal_grid_payload(seed=1000)
                second["command_arguments"][field] = value
                torch.save(second, second_path)
                with self.subTest(field=field), self.assertRaisesRegex(
                    RuntimeError, "Heterogeneous motion-encoder training identities"
                ):
                    validate_motion_encoder_grid_identity(
                        {(1, 500): first_path, (2, 1000): second_path}, "A3"
                    )

    def test_grid_accepts_homogeneous_formal_members_and_rejects_smoke(self):
        with tempfile.TemporaryDirectory(prefix="motion_encoder_grid_formal_") as temporary:
            root = Path(temporary)
            first_path = root / "first.pt"
            second_path = root / "second.pt"
            torch.save(_formal_grid_payload(seed=500), first_path)
            torch.save(_formal_grid_payload(seed=1000), second_path)
            audit = validate_motion_encoder_grid_identity(
                {(1, 500): first_path, (2, 1000): second_path}, "A3"
            )
            self.assertEqual(audit["schema"], "motion_encoder_cv_grid_identity_v1")
            self.assertEqual(len(audit["members"]), 2)

            smoke = _formal_grid_payload(seed=1000)
            smoke["split_audit"]["smoke_test"] = True
            torch.save(smoke, second_path)
            with self.assertRaisesRegex(RuntimeError, "not eligible for the formal CV grid"):
                validate_motion_encoder_grid_identity(
                    {(1, 500): first_path, (2, 1000): second_path}, "A3"
                )


@unittest.skip(
    "Historical run_subject_cv discovery was replaced by identity-locked strict CV."
)
class EncoderDiscoveryAndPairingTests(unittest.TestCase):
    def test_canonical_trainer_directory_is_discovered(self):
        with tempfile.TemporaryDirectory(
            prefix="motion_encoder_canonical_discovery_"
        ) as temporary:
            root = Path(temporary)
            expected = _canonical_trainer_payload(root, "A3")
            self.assertEqual(
                find_motion_encoder_checkpoint(root, 1, 500, "A3"), expected
            )

    def test_mixed_profiles_are_filtered_and_duplicate_identity_is_rejected(self):
        with tempfile.TemporaryDirectory(prefix="motion_encoder_discovery_") as temporary:
            root = Path(temporary)
            a3 = _encoder_payload(root, "A3", "run_one")
            _encoder_payload(root, "A4", "run_two")
            self.assertEqual(find_motion_encoder_checkpoint(root, 1, 500, "A3"), a3)
            with self.assertRaisesRegex(RuntimeError, "profile=A2.*matched 0"):
                find_motion_encoder_checkpoint(root, 1, 500, "A2")
            _encoder_payload(root, "A3", "duplicate")
            with self.assertRaisesRegex(RuntimeError, "profile=A3.*matched 2"):
                find_motion_encoder_checkpoint(root, 1, 500, "A3")

    @staticmethod
    def _cv_args(segmentation: str, root: Path) -> SimpleNamespace:
        return SimpleNamespace(
            motion_encoder_root=str(root),
            expected_encoder_profile="A3",
            primitive_segmentation=segmentation,
            primitive_num=32,
            pca_dim=64,
            label_permutations=10,
            order_shuffles=10,
            batch_size=16,
            device="cpu",
            anomaly_policy="report",
            changepoint_context_windows=2,
            changepoint_score_quantile=0.90,
            changepoint_min_segment_windows=2,
        )

    @staticmethod
    def _write_existing_run(
        run_dir: Path,
        checkpoint: Path,
        segmentation: str,
        checkpoint_hash: str,
    ) -> None:
        run_dir.mkdir(parents=True, exist_ok=True)
        for name in (
            "summary.json",
            "trial_primitive_sequences.jsonl",
            "activity_sequence_distance_matrix.csv",
        ):
            (run_dir / name).write_text("{}\n", encoding="utf-8")
        if segmentation == "motion_encoder_changepoint":
            (run_dir / "segmentation_statistics.json").write_text("{}\n", encoding="utf-8")
            (run_dir / "segment_embeddings_and_tokens.npz").write_bytes(b"npz-placeholder")
            (run_dir / "activity_trial_token_sequences.png").write_bytes(b"png-placeholder")
        arguments = {
            "seed": 500,
            "primitive_num": 32,
            "pca_dim": 64,
            "label_permutations": 10,
            "order_shuffles": 10,
            "batch_size": 16,
            "device": "cpu",
            "anomaly_policy": "report",
            "old_class_count": 6,
            "embedding_normalization": "l2",
            "codebook_weighting": "per_trial",
            "kmeans_n_init": 20,
            "kmeans_max_iter": 300,
            "edge_trim_ratio": 0.10,
            "sample_rate_hz": 100.0,
            "primitive_segmentation": segmentation,
        }
        if segmentation == "motion_encoder_changepoint":
            arguments.update(
                {
                    "changepoint_context_windows": 2,
                    "changepoint_score_quantile": 0.90,
                    "changepoint_min_segment_windows": 2,
                }
            )
        config = {
            "checkpoint": str(checkpoint.resolve()),
            "checkpoint_sha256": checkpoint_hash,
            "checkpoint_type": "motion_primitive_encoder",
            "checkpoint_schema_version": 1,
            "encoder_training": {"ablation_profile": "A3"},
            "arguments": arguments,
            "segmentation": {"method": segmentation},
            "codebook": {
                "primitive_num": 32,
                "assignment_metric": "cosine",
            },
            "feature_roles": {
                "codebook": "motion_encoder_content_head",
                "boundary": "motion_encoder_segmentation_head",
            },
            "order_shuffle_control": {"algorithm": "valid_rle_permutation_v2"},
        }
        (run_dir / "experiment_config.json").write_text(
            json.dumps(config, indent=2), encoding="utf-8"
        )

    def test_fixed_and_changepoint_controls_require_same_checkpoint_sha(self):
        with tempfile.TemporaryDirectory(prefix="motion_encoder_pairing_") as temporary:
            root = Path(temporary)
            checkpoint = root / "motion_encoder_final.pt"
            checkpoint.write_bytes(b"canonical-motion-encoder")
            expected_hash = sha256_file(checkpoint)
            fixed = root / "fixed"
            changepoint = root / "changepoint"
            self._write_existing_run(fixed, checkpoint, "fixed_window", expected_hash)
            self._write_existing_run(
                changepoint,
                checkpoint,
                "motion_encoder_changepoint",
                expected_hash,
            )
            validate_existing_run(
                fixed, checkpoint, 1, 500, self._cv_args("fixed_window", root)
            )
            validate_existing_run(
                changepoint,
                checkpoint,
                1,
                500,
                self._cv_args("motion_encoder_changepoint", root),
            )

            config_path = changepoint / "experiment_config.json"
            bad_config = json.loads(config_path.read_text(encoding="utf-8"))
            bad_config["checkpoint_sha256"] = "0" * 64
            config_path.write_text(json.dumps(bad_config, indent=2), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "checkpoint content mismatch"):
                validate_existing_run(
                    changepoint,
                    checkpoint,
                    1,
                    500,
                    self._cv_args("motion_encoder_changepoint", root),
                )


if __name__ == "__main__":
    unittest.main()
