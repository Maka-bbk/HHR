import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

from experiments.motion_primitive.run_adaptive_codebook_cv import (
    ONLINE_PROTOCOL,
    GridMember,
    _assert_fields,
    _prepare_directory,
    _validate_or_refresh_incompatible_online,
    build_analysis_command,
    build_encoder_command,
    build_fixed_command,
    build_online_command,
    fixed_run_name,
    online_run_name,
    parse_args,
    parse_int_list,
)


class AdaptiveCodebookCVWrapperTests(unittest.TestCase):
    def test_canonical_grid_parsing_is_sorted_and_unique(self):
        self.assertEqual(parse_int_list("7,1,7,3"), (1, 3, 7))
        self.assertEqual(parse_int_list("500,0,50,5"), (0, 5, 50, 500))

    def test_command_builders_preserve_member_identity(self):
        member = GridMember("A3", 2, 50)
        encoder = build_encoder_command(
            "python", Path("source.pt"), Path("data.npz"), Path("encoder"), member, "cuda"
        )
        fixed = build_fixed_command(
            "python", Path("encoder.pt"), Path("data.npz"), Path("fixed"), member, "cuda"
        )
        online = build_online_command(
            "python", Path("fixed"), Path("data.npz"), Path("online"), member, 10000
        )
        analysis = build_analysis_command(
            "python", Path("online"), Path("analysis"), (1, 2), (0, 5), 10000, 20260904
        )

        self.assertEqual(encoder[encoder.index("--ablation-profile") + 1], "A3")
        self.assertEqual(encoder[encoder.index("--seed") + 1], "50")
        self.assertEqual(fixed[fixed.index("--primitive-segmentation") + 1], "fixed_window")
        self.assertEqual(fixed[fixed.index("--primitive-num") + 1], "32")
        self.assertEqual(online[online.index("--fold") + 1], "2")
        self.assertEqual(online[online.index("--seed") + 1], "50")
        for key in ONLINE_PROTOCOL:
            flag = "--" + key.replace("_", "-")
            self.assertIn(flag, online)
        self.assertEqual(analysis[analysis.index("--expected-folds") + 1], "1,2")
        self.assertEqual(analysis[analysis.index("--expected-seeds") + 1], "0,5")

    def test_directory_names_are_unambiguous(self):
        member = GridMember("A0", 6, 500)
        self.assertEqual(
            fixed_run_name(member),
            "fold_06_seed_500_k32_motion_encoder_fixed_window_a0",
        )
        self.assertEqual(
            online_run_name(member), "fold_06_seed_500_A0_fixed_adaptive_v2"
        )

    def test_incomplete_directory_is_quarantined_not_deleted(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run = root / "run"
            run.mkdir()
            payload = run / "partial.txt"
            payload.write_text("recoverable", encoding="utf-8")

            completed = _prepare_directory(run, root, "done.json", dry_run=False)

            self.assertFalse(completed)
            self.assertFalse(run.exists())
            quarantined = list((root / "_incomplete").glob("run.*"))
            self.assertEqual(len(quarantined), 1)
            self.assertEqual(
                (quarantined[0] / "partial.txt").read_text(encoding="utf-8"),
                "recoverable",
            )

    def test_dry_run_does_not_move_incomplete_directory(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run = root / "run"
            run.mkdir()
            (run / "partial.txt").write_text("keep", encoding="utf-8")

            completed = _prepare_directory(run, root, "done.json", dry_run=True)

            self.assertFalse(completed)
            self.assertTrue((run / "partial.txt").is_file())
            self.assertFalse((root / "_incomplete").exists())

    def test_completion_marker_requires_validation_instead_of_quarantine(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run = root / "run"
            run.mkdir()
            (run / "done.json").write_text("{}", encoding="utf-8")
            self.assertTrue(_prepare_directory(run, root, "done.json", dry_run=False))
            self.assertTrue(run.is_dir())

    def test_incompatible_completed_online_run_stays_fail_closed_by_default(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run = root / "run"
            run.mkdir()
            (run / "done.json").write_text("{}", encoding="utf-8")
            validator = Mock(side_effect=RuntimeError("old runner fingerprint"))

            with self.assertRaisesRegex(RuntimeError, "old runner fingerprint"):
                _validate_or_refresh_incompatible_online(
                    run,
                    root,
                    validator,
                    refresh_incompatible=False,
                    dry_run=False,
                )

            self.assertTrue(run.is_dir())
            self.assertFalse((root / "_incomplete").exists())

    def test_explicit_refresh_quarantines_incompatible_online_run(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run = root / "run"
            run.mkdir()
            (run / "done.json").write_text("{}", encoding="utf-8")
            validator = Mock(side_effect=RuntimeError("old runner fingerprint"))

            reusable = _validate_or_refresh_incompatible_online(
                run,
                root,
                validator,
                refresh_incompatible=True,
                dry_run=False,
            )

            self.assertFalse(reusable)
            self.assertFalse(run.exists())
            quarantined = list((root / "_incomplete").glob("run.*"))
            self.assertEqual(len(quarantined), 1)
            self.assertTrue((quarantined[0] / "done.json").is_file())

    def test_identity_mismatch_fails_closed(self):
        with self.assertRaisesRegex(RuntimeError, "identity mismatch"):
            _assert_fields({"seed": 5, "profile": "A0"}, {"seed": 50}, "run")

    def test_cli_defaults_are_full_grid_and_analysis_terminal_stage(self):
        args = parse_args(
            [
                "--cv-root", "cv",
                "--npz-path", "data.npz",
                "--encoder-root", "encoders",
                "--work-root", "work",
                "--dry-run",
            ]
        )
        self.assertEqual(args.folds, "1,2,3,4,5,6,7")
        self.assertEqual(args.seeds, "0,5,50,500")
        self.assertEqual(args.stop_after, "analysis")
        self.assertTrue(args.dry_run)
        self.assertFalse(args.refresh_incompatible_online)


if __name__ == "__main__":
    unittest.main()
