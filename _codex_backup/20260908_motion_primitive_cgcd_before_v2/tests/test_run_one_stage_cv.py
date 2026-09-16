import tempfile
import unittest
from pathlib import Path

from experiments.motion_primitive.run_one_stage_cv import (
    _aggregate,
    build_parser,
    run,
)


class OneStageCVRunnerTests(unittest.TestCase):
    def test_dry_run_builds_exact_seven_fold_two_profile_grid(self):
        with tempfile.TemporaryDirectory() as temporary:
            args = build_parser().parse_args(
                [
                    "--output-root",
                    str(Path(temporary) / "grid"),
                    "--data-root",
                    str(Path(temporary) / "data"),
                    "--profiles",
                    "J0-U,J0-T",
                    "--folds",
                    "1,2,3,4,5,6,7",
                    "--seeds",
                    "50",
                    "--dry-run",
                ]
            )
            result = run(args)
            self.assertTrue(result["dry_run"])
            self.assertEqual(len(result["commands"]), 14)
            joined = [" ".join(command) for command in result["commands"]]
            self.assertEqual(sum("--profile J0-U" in item for item in joined), 7)
            self.assertEqual(sum("--profile J0-T" in item for item in joined), 7)
            self.assertTrue(all("--resume" not in item for item in joined))

    def test_output_root_rejects_changed_grid_identity(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "grid"
            first = build_parser().parse_args(
                ["--output-root", str(root), "--folds", "1", "--dry-run"]
            )
            run(first)
            changed = build_parser().parse_args(
                [
                    "--output-root",
                    str(root),
                    "--folds",
                    "1",
                    "--codebook-size",
                    "16",
                    "--dry-run",
                ]
            )
            with self.assertRaisesRegex(RuntimeError, "different CV identity"):
                run(changed)

    def test_fold_block_aggregate_averages_seeds_before_statistics(self):
        rows = []
        for profile, offset in (("J0-U", 0.0), ("J0-T", 0.1)):
            for fold in (1, 2):
                for seed in (5, 50):
                    row = {
                        "profile": profile,
                        "fold": fold,
                        "seed": seed,
                        "order_h_score_drop": 0.02 + offset,
                        "effective_code_count": 8.0,
                        "token_subject_nmi": 0.1,
                    }
                    row.update(
                        {
                            metric: 0.4 + 0.01 * fold + offset
                            for metric in (
                                "all_accuracy",
                                "old_accuracy",
                                "new_accuracy",
                                "h_score",
                                "macro_f1",
                            )
                        }
                    )
                    rows.append(row)
        result = _aggregate(
            rows,
            ["J0-U", "J0-T"],
            [1, 2],
            [5, 50],
            100,
            7,
        )
        self.assertAlmostEqual(
            result["profiles"]["J0-T"]["h_score"]["mean"]
            - result["profiles"]["J0-U"]["h_score"]["mean"],
            0.1,
        )
        self.assertEqual(
            result["paired_J0_T_vs_J0_U"]["h_score"]["permutation_count"],
            4,
        )


if __name__ == "__main__":
    unittest.main()
