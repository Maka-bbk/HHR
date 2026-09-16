"""Import-closure gate for the standalone motion-primitive project."""

from __future__ import annotations

import ast
import subprocess
import sys
import textwrap
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CANONICAL_SOURCES = (
    "data/uschad.py",
    "data/uschad_har.py",
    "models/resnet1d.py",
    "models/batch_utils.py",
    "models/motion_primitives.py",
    "models/motion_primitive_cgcd.py",
    "experiments/motion_primitive/profiles.py",
    "experiments/motion_primitive/trajectory_contrastive.py",
    "experiments/motion_primitive/trajectory_distillation.py",
    "experiments/motion_primitive/joint_losses.py",
    "experiments/motion_primitive/offline_trainer.py",
    "experiments/motion_primitive/offline_cv_runner.py",
    "experiments/motion_primitive/motion_online.py",
    "experiments/motion_primitive/online_runner.py",
    "experiments/motion_primitive/online_cv_runner.py",
    "experiments/motion_primitive/export_trajectory_visuals.py",
)
FORBIDDEN_IMPORTS = (
    "torchvision",
    "timm",
    "data.get_datasets",
    "models.vision_transformer",
    "models.trial_pooling",
    "models.utils_simgcd",
    "models.happy_motion_trajectory",
    "models.motion_trajectory",
    "experiments.motion_primitive.happy_joint_losses",
    "experiments.motion_primitive.happy_online",
)


def _forbidden(name: str) -> bool:
    return any(name == item or name.startswith(item + ".") for item in FORBIDDEN_IMPORTS)


class CleanImportTests(unittest.TestCase):
    def test_current_api_exposes_no_happy_compatibility_aliases(self) -> None:
        forbidden_names = (
            "happy_primitive_joint",
            "HappyJointLossConfig",
            "HappyJointLossResult",
            "compose_happy_motion_joint_loss",
            "audited_happy_profile",
        )
        sources = (
            "experiments/motion_primitive/profiles.py",
            "experiments/motion_primitive/joint_losses.py",
            "experiments/motion_primitive/offline_trainer.py",
        )
        for relative in sources:
            text = (PROJECT_ROOT / relative).read_text(encoding="utf-8")
            for name in forbidden_names:
                self.assertNotIn(name, text, f"{relative}: {name}")

    def test_canonical_sources_do_not_import_image_pooling_or_happy_modules(self) -> None:
        violations: list[str] = []
        for relative in CANONICAL_SOURCES:
            path = PROJECT_ROOT / relative
            self.assertTrue(path.is_file(), relative)
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            imports: list[str] = []
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    imports.extend(alias.name for alias in node.names)
                elif isinstance(node, ast.ImportFrom) and node.module:
                    imports.append(node.module)
            violations.extend(
                f"{relative}: {name}" for name in imports if _forbidden(name)
            )
        self.assertEqual(violations, [])

    def test_fresh_public_import_closure_survives_forbidden_module_blocker(self) -> None:
        child = textwrap.dedent(
            f"""
            import importlib.abc
            import sys

            forbidden = {FORBIDDEN_IMPORTS!r}
            def blocked(name):
                return any(name == item or name.startswith(item + '.') for item in forbidden)

            class RejectLegacy(importlib.abc.MetaPathFinder):
                def find_spec(self, fullname, path=None, target=None):
                    if blocked(fullname):
                        raise RuntimeError('forbidden import: ' + fullname)
                    return None

            sys.meta_path.insert(0, RejectLegacy())
            import data.uschad_har
            import models.motion_primitives
            import models.motion_primitive_cgcd
            import experiments.motion_primitive.offline_trainer
            import experiments.motion_primitive.offline_cv_runner
            import experiments.motion_primitive.motion_online
            import experiments.motion_primitive.online_runner
            import experiments.motion_primitive.online_cv_runner
            import experiments.motion_primitive.export_trajectory_visuals
            loaded = sorted(name for name in sys.modules if blocked(name))
            assert not loaded, loaded
            print('CLEAN_IMPORT_CLOSURE_OK')
            """
        )
        result = subprocess.run(
            [sys.executable, "-c", child],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(
            result.returncode,
            0,
            msg=f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}",
        )
        self.assertIn("CLEAN_IMPORT_CLOSURE_OK", result.stdout)

    def test_runtime_requirements_exclude_image_stack(self) -> None:
        for filename in ("requirements.txt", "requirements-har.txt"):
            lines = (PROJECT_ROOT / filename).read_text(encoding="utf-8").splitlines()
            packages = [line.partition("#")[0].strip().lower() for line in lines]
            self.assertFalse(any(line.startswith("torchvision") for line in packages))
            self.assertFalse(any(line.startswith("timm") for line in packages))


if __name__ == "__main__":
    unittest.main()
