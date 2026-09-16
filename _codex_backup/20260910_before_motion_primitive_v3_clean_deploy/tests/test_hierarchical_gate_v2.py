from __future__ import annotations

from dataclasses import fields
import inspect
import unittest

import numpy as np

from experiments.motion_primitive.hierarchical_gate_v2 import (
    HierarchicalGateV2Model,
    MINIMUM_CHILD_COUNT,
    MINIMUM_CHILD_FRACTION,
    MINIMUM_SUBJECTS_PER_CHILD,
    assert_gate_v2_schema_is_label_free,
    assign_hierarchical_gate_v2,
    fit_hierarchical_gate_v2,
    hierarchical_gate_v2_diagnostics,
    leave_one_candidate_out_stability_audit,
)
from experiments.motion_primitive.online_secondary_codebook import (
    UnlabelledTokenOccurrence,
)


def occurrence(
    trial_id: int,
    residual: tuple[float, float],
    gravity: tuple[float, float, float],
    *,
    token: int = 7,
) -> UnlabelledTokenOccurrence:
    direction = np.asarray(gravity, dtype=np.float32)
    direction /= np.linalg.norm(direction)
    return UnlabelledTokenOccurrence(
        trial_global_id=int(trial_id),
        coarse_token=int(token),
        token_fraction=0.75,
        duration_samples=100,
        mean_quantization_residual=np.asarray(residual, dtype=np.float32),
        gravity_direction=direction,
    )


def balanced_fixture() -> tuple[
    list[UnlabelledTokenOccurrence], dict[int, np.ndarray], dict[int, int]
]:
    """Eight static and eight dynamic candidates; static gravity is 4:4."""

    items: list[UnlabelledTokenOccurrence] = []
    motion: dict[int, np.ndarray] = {}
    subjects: dict[int, int] = {}
    for index in range(8):
        trial_id = 100 + index
        gravity = (1.0, 0.0, 0.0) if index < 4 else (0.0, 1.0, 0.0)
        items.append(occurrence(trial_id, (0.0, 0.0), gravity))
        motion[trial_id] = np.asarray([0.01, 0.01], dtype=np.float32)
        # Each gravity child retains two subjects after any one deletion.
        subjects[trial_id] = 1 + (index % 2)
    for index in range(8):
        trial_id = 200 + index
        items.append(occurrence(trial_id, (4.0, 4.0), (0.0, 0.0, 1.0)))
        motion[trial_id] = np.asarray([2.0, 4.0], dtype=np.float32)
        subjects[trial_id] = 1 + (index % 2)
    return items, motion, subjects


def singleton_dynamic_fixture() -> tuple[
    list[UnlabelledTokenOccurrence], dict[int, np.ndarray], dict[int, int]
]:
    """Match the observed A3 failure mode: residual support is exactly 12:1."""

    items: list[UnlabelledTokenOccurrence] = []
    motion: dict[int, np.ndarray] = {}
    subjects: dict[int, int] = {}
    for index in range(12):
        trial_id = 300 + index
        gravity = (1.0, 0.0, 0.0) if index < 6 else (0.0, 1.0, 0.0)
        items.append(occurrence(trial_id, (0.0, 0.0), gravity))
        motion[trial_id] = np.asarray([0.01, 0.01], dtype=np.float32)
        subjects[trial_id] = 1 + (index % 2)
    items.append(occurrence(399, (5.0, 5.0), (0.0, 0.0, 1.0)))
    motion[399] = np.asarray([3.0, 5.0], dtype=np.float32)
    subjects[399] = 1
    return items, motion, subjects


def three_dynamic_fixture() -> tuple[
    list[UnlabelledTokenOccurrence], dict[int, np.ndarray], dict[int, int]
]:
    """Full fit passes 10:3, but every dynamic deletion creates 10:2."""

    items: list[UnlabelledTokenOccurrence] = []
    motion: dict[int, np.ndarray] = {}
    subjects: dict[int, int] = {}
    for index in range(10):
        trial_id = 400 + index
        gravity = (1.0, 0.0, 0.0) if index < 5 else (0.0, 1.0, 0.0)
        items.append(occurrence(trial_id, (0.0, 0.0), gravity))
        motion[trial_id] = np.asarray([0.01, 0.01], dtype=np.float32)
        subjects[trial_id] = 1 + (index % 2)
    for index in range(3):
        trial_id = 500 + index
        items.append(occurrence(trial_id, (4.0, 4.0), (0.0, 0.0, 1.0)))
        motion[trial_id] = np.asarray([2.0, 4.0], dtype=np.float32)
        subjects[trial_id] = 1 + (index % 2)
    return items, motion, subjects


class HierarchicalGateV2Tests(unittest.TestCase):
    def test_v2_schema_and_fitting_interfaces_have_no_activity_identity(self) -> None:
        assert_gate_v2_schema_is_label_free()
        forbidden = ("label", "activity", "class", "name")
        model_fields = {item.name.lower() for item in fields(HierarchicalGateV2Model)}
        self.assertFalse(
            any(fragment in name for name in model_fields for fragment in forbidden)
        )
        for function in (
            fit_hierarchical_gate_v2,
            leave_one_candidate_out_stability_audit,
        ):
            parameters = {value.lower() for value in inspect.signature(function).parameters}
            self.assertFalse(
                any(
                    fragment in parameter
                    for parameter in parameters
                    for fragment in forbidden
                )
            )

    def test_registered_minimums_cannot_be_weakened(self) -> None:
        items, motion, subjects = balanced_fixture()
        invalid = (
            {"minimum_child_count": MINIMUM_CHILD_COUNT - 1},
            {"minimum_child_fraction": MINIMUM_CHILD_FRACTION - 0.01},
            {"minimum_subjects_per_child": MINIMUM_SUBJECTS_PER_CHILD - 1},
        )
        for kwargs in invalid:
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(ValueError):
                    fit_hierarchical_gate_v2(
                        items,
                        motion,
                        subjects,
                        require_leave_one_out=False,
                        restarts=10,
                        seed=17,
                        **kwargs,
                    )

        for kwargs in (
            {"minimum_loo_enabled_fraction": 0.99},
            {"minimum_loo_partition_agreement": 0.89},
            {"minimum_loo_heldout_agreement": 0.89},
        ):
            with self.subTest(loo_kwargs=kwargs):
                with self.assertRaises(ValueError):
                    fit_hierarchical_gate_v2(
                        items,
                        motion,
                        subjects,
                        static_radius_quantile=1.0,
                        restarts=10,
                        seed=17,
                        **kwargs,
                    )

    def test_observed_twelve_to_one_residual_split_is_rejected(self) -> None:
        items, motion, subjects = singleton_dynamic_fixture()
        model = fit_hierarchical_gate_v2(
            items,
            motion,
            subjects,
            static_radius_quantile=1.0,
            require_leave_one_out=False,
            restarts=30,
            seed=17,
        )
        self.assertIs(model.gate_enabled, False)
        self.assertIn(
            "residual_child_below_minimum_support", model.gate_disable_reasons
        )
        self.assertEqual(
            sorted(model.support_audit["residual_child_counts"]), [1, 12]
        )
        self.assertEqual(model.support_audit["residual_required_per_child"], 3)
        assignments = assign_hierarchical_gate_v2(model, items)
        np.testing.assert_array_equal(
            assignments.gate_states, np.zeros(len(items), dtype=np.int64)
        )
        self.assertFalse(np.any(assignments.confident_static))

    def test_balanced_train_only_structure_passes_support_and_subject_checks(self) -> None:
        items, motion, subjects = balanced_fixture()
        model = fit_hierarchical_gate_v2(
            items,
            motion,
            subjects,
            static_radius_quantile=1.0,
            require_leave_one_out=False,
            restarts=30,
            seed=23,
        )
        self.assertIs(model.gate_enabled, True)
        self.assertEqual(model.gate_disable_reasons, ())
        audit = model.support_audit
        self.assertEqual(sorted(audit["residual_child_counts"]), [8, 8])
        self.assertEqual(sorted(audit["gravity_child_counts"]), [4, 4])
        self.assertGreaterEqual(min(audit["residual_child_subject_counts"]), 2)
        self.assertGreaterEqual(min(audit["gravity_child_subject_counts"]), 2)
        self.assertIs(audit["physical_consistency"]["passed"], True)
        self.assertIs(audit["support_and_physical_thresholds_use_train_only"], True)
        assignments = assign_hierarchical_gate_v2(model, items)
        self.assertEqual(set(assignments.gate_states[:8].tolist()), {1, 2})
        np.testing.assert_array_equal(
            assignments.gate_states[8:], np.zeros(8, dtype=np.int64)
        )

    def test_subject_support_is_enforced_only_when_mapping_is_available(self) -> None:
        items, motion, _ = balanced_fixture()
        # Residual groups contain both subjects, while each gravity child is
        # deliberately confined to one train subject.
        subjects = {
            int(item.trial_global_id): (
                1
                if int(item.trial_global_id) < 104
                else 2
                if int(item.trial_global_id) < 108
                else 1 + (int(item.trial_global_id) % 2)
            )
            for item in items
        }
        with_subjects = fit_hierarchical_gate_v2(
            items,
            motion,
            subjects,
            static_radius_quantile=1.0,
            require_leave_one_out=False,
            restarts=30,
            seed=23,
        )
        self.assertIs(with_subjects.gate_enabled, False)
        self.assertIn(
            "gravity_child_below_minimum_subjects",
            with_subjects.gate_disable_reasons,
        )
        without_subjects = fit_hierarchical_gate_v2(
            items,
            motion,
            None,
            static_radius_quantile=1.0,
            require_leave_one_out=False,
            restarts=30,
            seed=23,
        )
        self.assertIs(without_subjects.gate_enabled, True)
        self.assertIs(without_subjects.support_audit["subjects_available"], False)

    def test_subject_mapping_must_match_train_candidates_exactly(self) -> None:
        items, motion, subjects = balanced_fixture()
        string_keyed = {str(key): value for key, value in subjects.items()}
        accepted = fit_hierarchical_gate_v2(
            items,
            motion,
            string_keyed,
            static_radius_quantile=1.0,
            require_leave_one_out=False,
            restarts=10,
        )
        self.assertIs(accepted.support_audit["subjects_available"], True)
        subjects.pop(int(items[0].trial_global_id))
        with self.assertRaisesRegex(ValueError, "exactly match"):
            fit_hierarchical_gate_v2(
                items,
                motion,
                subjects,
                require_leave_one_out=False,
                restarts=10,
            )

    def test_leave_one_candidate_out_audit_passes_stable_balanced_fixture(self) -> None:
        items, motion, subjects = balanced_fixture()
        audit = leave_one_candidate_out_stability_audit(
            items,
            motion,
            subjects,
            static_radius_quantile=1.0,
            minimum_partition_agreement=1.0,
            minimum_heldout_agreement=1.0,
            restarts=20,
            seed=29,
        )
        self.assertIs(audit["passed"], True)
        self.assertEqual(audit["replicate_count"], len(items))
        self.assertEqual(audit["enabled_fraction"], 1.0)
        self.assertEqual(audit["minimum_residual_static_partition_agreement"], 1.0)
        self.assertEqual(audit["minimum_gravity_partition_agreement"], 1.0)
        self.assertEqual(audit["heldout_route_agreement"], 1.0)
        self.assertEqual(
            audit["heldout_gravity_agreement_on_jointly_routed"], 1.0
        )
        self.assertIs(audit["uses_only_train_candidates"], True)
        self.assertIs(audit["uses_test_occurrences"], False)

    def test_leave_one_out_disables_a_full_fit_supported_only_by_three_dynamics(self) -> None:
        items, motion, subjects = three_dynamic_fixture()
        structural = fit_hierarchical_gate_v2(
            items,
            motion,
            subjects,
            static_radius_quantile=1.0,
            require_leave_one_out=False,
            restarts=30,
            seed=31,
        )
        self.assertIs(structural.gate_enabled, True)
        self.assertEqual(
            sorted(structural.support_audit["residual_child_counts"]), [3, 10]
        )
        audited = fit_hierarchical_gate_v2(
            items,
            motion,
            subjects,
            static_radius_quantile=1.0,
            require_leave_one_out=True,
            restarts=30,
            seed=31,
        )
        self.assertIs(audited.gate_enabled, False)
        self.assertIn("leave_one_out_stability_failed", audited.gate_disable_reasons)
        self.assertLess(audited.leave_one_out_audit["enabled_fraction"], 1.0)
        failed_ids = {
            int(item["omitted_trial_id"])
            for item in audited.leave_one_out_audit["replicates"]
            if not item["structural_gate_enabled"]
        }
        self.assertTrue(failed_ids & {500, 501, 502})
        assignments = assign_hierarchical_gate_v2(audited, items)
        np.testing.assert_array_equal(assignments.gate_states, np.zeros(len(items)))

    def test_diagnostics_close_the_fail_closed_loop(self) -> None:
        items, motion, subjects = singleton_dynamic_fixture()
        model = fit_hierarchical_gate_v2(
            items,
            motion,
            subjects,
            static_radius_quantile=1.0,
            restarts=20,
            seed=37,
        )
        diagnostics = hierarchical_gate_v2_diagnostics(model, items)
        self.assertIs(diagnostics["gate_enabled"], False)
        self.assertIn(
            "residual_child_below_minimum_support",
            diagnostics["gate_disable_reasons"],
        )
        self.assertEqual(diagnostics["fit_gate_state_counts"], [len(items), 0, 0])
        self.assertIs(diagnostics["all_v2_thresholds_are_train_only"], True)
        self.assertEqual(
            diagnostics["leave_one_out_audit"]["status"],
            "not_run_full_fit_failed_structural_checks",
        )


if __name__ == "__main__":
    unittest.main()
