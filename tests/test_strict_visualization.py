"""Artifact-bound tests for strict post-hoc motion-primitive figures."""

from __future__ import annotations

import builtins
import csv
import inspect
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from experiments.motion_primitive.strict_metrics import strict_three_layer_metrics
from experiments.motion_primitive.strict_visualization import (
    PROFILE,
    _load_plotting,
    _resample_tokens,
    activity_primitive_matrix,
    generate_visualizations,
    load_raw_trial_signals,
    load_visualization_artifacts,
    sha256_file,
)
import experiments.motion_primitive.strict_visualization as strict_visualization


def _write_json(path: Path, value) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )


def _write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _usage(tokens: list[int]) -> dict:
    counts = np.bincount(np.asarray(tokens, dtype=np.int64), minlength=32)
    probabilities = counts / counts.sum()
    positive = probabilities > 0
    entropy = -float(np.sum(probabilities[positive] * np.log(probabilities[positive])))
    used = int(np.count_nonzero(counts))
    return {
        "capacity_k": 32,
        "used_k": used,
        "dead_k": 32 - used,
        "dead_fraction": (32 - used) / 32.0,
        "effective_k": float(np.exp(entropy)),
        "counts": counts.tolist(),
        "fractions": probabilities.tolist(),
    }


class SyntheticStrictRun:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.offline = root / "offline"
        self.online = root / "online"
        self.output = root / "visuals"
        self.offline.mkdir()
        self.online.mkdir()
        self.npz = root / "uschad_windows.npz"
        self._build_npz()
        self._build_offline()
        self._build_online()

    @staticmethod
    def _hex(value: int) -> str:
        return f"{value:064x}"

    def _build_npz(self) -> None:
        trial_ids: list[int] = []
        starts: list[int] = []
        windows: list[np.ndarray] = []
        for trial_id in (101, 102, 201, 202, 301, 302):
            for index, start in enumerate((0, 4)):
                time = np.arange(8, dtype=np.float32)
                window = np.stack([
                    np.sin(0.2 * time + channel + 0.01 * trial_id) + index
                    for channel in range(6)
                ]).astype(np.float32)
                windows.append(window)
                trial_ids.append(trial_id)
                starts.append(start)
        np.savez_compressed(
            self.npz,
            windows=np.stack(windows),
            trial_global_ids=np.asarray(trial_ids, dtype=np.int64),
            window_start_indices=np.asarray(starts, dtype=np.int64),
            mean=np.zeros(6, dtype=np.float32),
            std=np.ones(6, dtype=np.float32),
        )

    def _build_offline(self) -> None:
        representation = self._hex(100)
        old_anchor = self._hex(101)
        old_state = self._hex(102)
        old_registry = {
            "schema": "hhr_strict_registry_state_v1",
            "state_sha256": old_state,
            "old_anchor_sha256": old_anchor,
            "representation_sha256": representation,
            "session_completed": 0,
            "entries": [],
            "unknown_buffer": [],
        }
        _write_json(self.offline / "old_registry.json", old_registry)
        usage = {
            "offline_train": _usage([0, 1, 0, 1]),
            "offline_validation": _usage([0, 2]),
            "offline_outer_test": _usage([1, 2, 3]),
        }
        _write_json(self.offline / "codebook_usage.json", usage)
        manifest = {
            "schema": "hhr_frozen_a2_e0_state_offline_v1",
            "profile": PROFILE,
            "fold": 1,
            "seed": 5,
            "npz_path": str(self.npz.resolve()),
            "npz_sha256": sha256_file(self.npz),
            "representation_sha256": representation,
            "codebook": {"primitive_num": 32},
            "old_registry": {
                "path": "old_registry.json",
                "artifact_sha256": sha256_file(self.offline / "old_registry.json"),
                "state_sha256": old_state,
                "old_anchor_sha256": old_anchor,
            },
        }
        _write_json(self.offline / "manifest.json", manifest)
        _write_json(self.offline / "complete.json", {
            "schema": "hhr_frozen_a2_e0_state_offline_v1",
            "profile": PROFILE,
            "manifest_sha256": sha256_file(self.offline / "manifest.json"),
            "complete": True,
        })

    def _build_online(self) -> None:
        trajectories: list[dict] = []
        previous = self._hex(102)
        representation = self._hex(100)
        old_anchor = self._hex(101)
        for session, trial_pair in enumerate(((101, 102), (201, 202), (301, 302)), start=1):
            class_count = 6 + 2 * session
            targets = np.asarray([0, class_count - 1], dtype=np.int64)
            raw = np.asarray([0, class_count - 2], dtype=np.int64)
            state = self._hex(200 + session)
            raw_path = self.online / f"raw_predictions_session_{session}.npz"
            np.savez_compressed(
                raw_path,
                trial_ids=np.asarray(trial_pair, dtype=np.int64),
                subject_ids=np.asarray([10, 11], dtype=np.int64),
                descriptors=np.eye(2, 3, dtype=np.float32),
                raw_registry_predictions=raw,
                registry_state_sha256=np.asarray(state),
                representation_sha256=np.asarray(representation),
            )
            metrics = strict_three_layer_metrics(
                targets,
                raw,
                class_count=class_count,
                old_class_count=6,
                seen_class_count_before_session=6 + 2 * (session - 1),
                registered_class_ids=tuple(range(class_count)),
            )
            metrics.update({
                "session": session,
                "raw_predictions_path": raw_path.name,
                "raw_predictions_sha256": sha256_file(raw_path),
                "raw_predictions_frozen_before_truth_join": True,
                "registry_state_sha256": state,
                "previous_registry_state_sha256": previous,
                "representation_sha256": representation,
                "expected_class_count": class_count,
                "registered_class_count": class_count,
                "registered_novel_count": class_count - 6,
                "registered_this_session": [class_count - 2, class_count - 1],
                "unknown_buffer_size": session,
                "incoming_codebook_usage": _usage([session, session + 1]),
                "evaluation_codebook_usage": _usage([0, session, session + 2]),
            })
            _write_json(self.online / f"metrics_session_{session}.json", metrics)
            primary = metrics["layers"]["old_fixed_novel_hungarian"]["aligned_predictions"]
            _write_csv(self.online / f"predictions_session_{session}.csv", [
                {
                    "trial_id": trial_pair[index],
                    "subject_id": 10 + index,
                    "activity_label": int(targets[index]),
                    "activity_name": f"activity-{int(targets[index])}",
                    "raw_registry_prediction": int(raw[index]),
                    "primary_aligned_prediction": int(primary[index]),
                }
                for index in range(2)
            ])
            for index, trial_id in enumerate(trial_pair):
                tokens = [session + index, session + index + 1]
                trajectories.append({
                    "session": session,
                    "role": "evaluation",
                    "trial_id": trial_id,
                    "subject_id": 10 + index,
                    "window_count": 2,
                    "used_primitive_count": 2,
                    "primitive_sequence": tokens,
                    "ownership_start_samples": [0, 6],
                    "ownership_end_samples_exclusive": [6, 12],
                    "quantization_distances": [0.1, 0.2],
                    "original_window_start_samples": [0, 4],
                    "raw_window_channel_means": [[0.0] * 6, [0.0] * 6],
                    "raw_registry_prediction": int(raw[index]),
                })
            discovery = {
                "state_sha256": state,
                "previous_state_sha256": previous,
                "initial_routing": [
                    {"trial_id": trial_pair[0], "accepted_known": True},
                    {"trial_id": trial_pair[1], "accepted_known": False},
                ],
                "discovery": {
                    "unknown_trial_ids": [trial_pair[1]],
                    "candidates": [{"accepted": True}],
                },
                "registered_ids": [class_count - 2, class_count - 1],
                "unresolved_trial_ids": [trial_pair[1]],
                "activity_labels_used": False,
            }
            _write_json(self.online / f"discovery_session_{session}.json", discovery)
            registry = {
                "schema": "hhr_strict_registry_state_v1",
                "session_completed": session,
                "state_sha256": state,
                "previous_state_sha256": previous,
                "old_anchor_sha256": old_anchor,
                "representation_sha256": representation,
                "entries": [{"registry_id": value} for value in range(class_count)],
                "unknown_buffer": [{"trial_id": trial_pair[1]}],
            }
            _write_json(self.online / f"registry_session_{session}.json", registry)
            previous = state
        with (self.online / "trajectories_label_free.jsonl").open("w", encoding="utf-8") as handle:
            for row in trajectories:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        _write_csv(self.online / "online_runs.csv", [
            {"session": session, "h_score": 0.5} for session in range(1, 4)
        ])
        summary = {
            "schema": "hhr_frozen_a2_e0_state_online_v1",
            "profile": PROFILE,
            "fold": 1,
            "seed": 5,
            "session_count": 3,
            "online_activity_labels_used_by_learner": False,
            "test_labels_used_only_for_scoring": True,
        }
        _write_json(self.online / "online_summary.json", summary)
        _write_json(self.online / "complete.json", {
            **summary,
            "online_summary_sha256": sha256_file(self.online / "online_summary.json"),
            "complete": True,
        })


class StrictVisualizationTests(unittest.TestCase):
    def test_module_has_no_online_learner_import_or_call(self) -> None:
        source = inspect.getsource(strict_visualization)
        self.assertNotIn("strict_registry import", source)
        self.assertNotIn("advance_registry_session", source)
        self.assertNotIn("route_label_free_trials", source)

    def test_raw_signal_reload_never_requires_activity_arrays(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run = SyntheticStrictRun(Path(directory))
            signals = load_raw_trial_signals(run.npz, [101])
            signal, starts = signals[101]
            self.assertEqual(signal.shape, (6, 12))
            np.testing.assert_array_equal(starts, [0, 4])

    def test_trajectory_raster_uses_physical_duration_not_token_index(self) -> None:
        raster = _resample_tokens([3, 7], [0, 90], [90, 100], columns=10)
        np.testing.assert_array_equal(raster, [3] * 9 + [7])

    def test_activity_heatmap_is_trial_then_subject_equal(self) -> None:
        rows = [
            {"activity_id": 0, "activity_name": "walk", "subject_id": 1,
             "primitive_sequence": [0] * 9 + [1]},
            {"activity_id": 0, "activity_name": "walk", "subject_id": 1,
             "primitive_sequence": [0] * 9 + [1]},
            {"activity_id": 0, "activity_name": "walk", "subject_id": 2,
             "primitive_sequence": [1]},
        ]
        activities, _, matrix = activity_primitive_matrix(rows, 2)
        self.assertEqual(activities, [0])
        np.testing.assert_allclose(matrix[0], [0.45, 0.55], atol=1e-12)

    def test_artifact_loader_fails_on_raw_prediction_tampering(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run = SyntheticStrictRun(Path(directory))
            path = run.online / "raw_predictions_session_2.npz"
            with path.open("ab") as handle:
                handle.write(b"tamper")
            with self.assertRaisesRegex(RuntimeError, "raw prediction SHA256"):
                load_visualization_artifacts(run.offline, run.online)

    def test_missing_matplotlib_has_clear_error(self) -> None:
        real_import = builtins.__import__

        def blocked(name, *args, **kwargs):
            if name == "matplotlib" or name.startswith("matplotlib."):
                raise ModuleNotFoundError("blocked for test")
            return real_import(name, *args, **kwargs)

        with mock.patch("builtins.__import__", side_effect=blocked):
            with self.assertRaisesRegex(RuntimeError, "matplotlib>=3.7"):
                _load_plotting()

    def test_complete_visualization_is_posthoc_and_non_mutating(self) -> None:
        try:
            _load_plotting()
        except RuntimeError as error:
            self.skipTest(str(error))
        with tempfile.TemporaryDirectory() as directory:
            run = SyntheticStrictRun(Path(directory))
            before = {
                str(path): sha256_file(path)
                for path in list(run.offline.iterdir()) + list(run.online.iterdir())
                if path.is_file()
            }
            result = generate_visualizations(
                run.offline,
                run.online,
                run.output,
                maximum_trial_plots=2,
                dpi=60,
            )
            resumed = generate_visualizations(
                run.offline,
                run.online,
                run.output,
                maximum_trial_plots=2,
                dpi=60,
                resume=True,
            )
            self.assertEqual(resumed["rendering_config"], result["rendering_config"])
            with self.assertRaisesRegex(RuntimeError, "different plot parameters"):
                generate_visualizations(
                    run.offline,
                    run.online,
                    run.output,
                    maximum_trial_plots=2,
                    dpi=61,
                    resume=True,
                )
            after = {path: sha256_file(Path(path)) for path in before}
            self.assertEqual(before, after)
            self.assertTrue(result["posthoc_scoring_only"])
            self.assertFalse(result["learner_api_called"])
            self.assertFalse(result["registry_or_model_writeback"])
            self.assertFalse(result["source_npz_activity_labels_read"])
            self.assertFalse(result["cross_fold_token_id_averaging_performed"])
            self.assertFalse(result["e0_codebook_capacity_changed_online"])
            self.assertEqual(result["online_growth_object"], "activity_class_registry")
            self.assertEqual(result["unique_trial_plot_count"], 2)
            expected = (
                "used_k_summary.png",
                "activity_primitive_heatmap.png",
                "trajectories_and_predictions.png",
                "confusion_three_layers_session_1.png",
                "confusion_three_layers_session_2.png",
                "confusion_three_layers_session_3.png",
                "registry_unknown_evolution.png",
                "trial_signal_and_primitives/trial_trajectory_index.csv",
            )
            for relative in expected:
                self.assertTrue((run.output / relative).is_file(), relative)
            index_rows = _read_csv(run.output / "trial_signal_and_primitives" / "trial_trajectory_index.csv")
            self.assertIn("complete_primitive_sequence", index_rows[0])
            used = json.loads((run.output / "used_k_summary.json").read_text(encoding="utf-8"))
            self.assertEqual(used["capacity_k"], 32)


if __name__ == "__main__":
    unittest.main()
