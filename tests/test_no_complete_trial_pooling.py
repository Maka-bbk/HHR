import ast
import inspect
import unittest

import torch

from experiments.motion_primitive.profiles import PROFILE, PROFILES
from models.motion_primitive_cgcd import MotionPrimitiveCGCDModel, MotionPrimitiveConfig


class NoCompleteTrialPoolingContractTests(unittest.TestCase):
    def _model(self) -> MotionPrimitiveCGCDModel:
        return MotionPrimitiveCGCDModel(
            MotionPrimitiveConfig(
                window_size=32,
                window_stride=16,
                feature_dim=32,
                base_channels=8,
                old_class_count=3,
                codebook_size=4,
                trajectory_input_dim=16,
                trajectory_hidden_dim=12,
                run_state_dim=3,
            )
        )

    def test_public_project_exposes_only_motion_primitive_profile(self) -> None:
        self.assertEqual(PROFILES, (PROFILE,))
        self.assertEqual(PROFILE, "frozen_a2_e0_state_k32")

    def test_public_architecture_contains_no_pooled_branch_parameters(self) -> None:
        fields = set(MotionPrimitiveConfig.__dataclass_fields__)
        self.assertTrue(
            fields.isdisjoint(
                {
                    "pool_quantile",
                    "pool_fusion_dim",
                    "pool_dropout",
                    "projection_hidden_dim",
                    "projection_bottleneck_dim",
                    "codebook_commitment_beta",
                }
            ),
            fields,
        )

    def test_model_registers_no_complete_trial_pool_or_pooled_head(self) -> None:
        model = self._model()
        self.assertIs(type(model), MotionPrimitiveCGCDModel)
        self.assertEqual(type(model).__module__, "models.motion_primitive_cgcd")
        registered_types = {
            (type(module).__module__, type(module).__name__)
            for module in model.modules()
        }
        self.assertNotIn(
            ("models.trial_pooling", "TrialMeanRobustMaxPool"), registered_types
        )
        self.assertNotIn(("models.utils_simgcd", "DINOHead"), registered_types)
        self.assertIn(
            ("models.motion_primitives", "SoftMotionCodebook"), registered_types
        )
        self.assertFalse(hasattr(model, "happy_pool"))
        self.assertFalse(hasattr(model, "happy_head"))
        self.assertFalse(hasattr(model, "forward_joint"))
        forbidden = ("happy_", "pooling.", "0.", "1.")
        self.assertFalse(
            any(key.startswith(forbidden) for key in model.state_dict()),
            sorted(model.state_dict()),
        )

    def test_canonical_model_has_no_happy_or_pooling_import_dependency(self) -> None:
        tree = ast.parse(inspect.getsource(inspect.getmodule(MotionPrimitiveCGCDModel)))
        imports = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imports.append(node.module)
        forbidden = (
            "models.happy_motion_trajectory",
            "models.motion_trajectory",
            "models.trial_pooling",
            "models.utils_simgcd",
        )
        self.assertFalse(
            any(
                name == prefix or name.startswith(prefix + ".")
                for name in imports
                for prefix in forbidden
            ),
            imports,
        )

    def test_forward_returns_only_trajectory_classification(self) -> None:
        model = self._model().eval()
        mask = torch.tensor([[True, True, True], [True, True, False]])
        starts = torch.tensor([[0, 16, 32], [0, 16, -1]])
        batch = {
            "windows": torch.randn(2, 3, 6, 32),
            "positions": model.relative_positions(starts, mask, 32),
            "mask": mask,
            "lengths": mask.sum(dim=1),
        }
        with torch.no_grad():
            output = model(batch, hard_codebook=True)
        self.assertEqual(tuple(output["trajectory_logits"].shape), (2, 3))
        self.assertIn("primitive_runs", output)
        self.assertEqual(
            output["run_features"].shape[-1],
            model.config.feature_dim + 6 + model.config.run_state_dim,
        )
        self.assertEqual(
            output["primitive_classifier_input"],
            "run_level_codebook_embedding_boundary_transition_duration_position_physical_state",
        )
        torch.testing.assert_close(
            output["run_features"][..., : model.config.feature_dim],
            output["run_codebook_embeddings"],
        )
        self.assertFalse(any(key.startswith("happy_") for key in output))
        self.assertNotIn("fused_logits", output)
        self.assertNotIn("pooled_logits", output)


if __name__ == "__main__":
    unittest.main()
