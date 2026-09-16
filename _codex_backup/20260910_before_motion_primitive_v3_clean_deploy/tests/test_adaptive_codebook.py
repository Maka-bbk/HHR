from __future__ import annotations

from dataclasses import fields
import inspect
import unittest

import numpy as np

from experiments.motion_primitive.adaptive_codebook import (
    AdaptiveCodebookConfig,
    AdaptiveOccurrence,
    adaptive_codebook_diagnostics,
    adaptive_duration_histograms,
    adaptive_state_by_trial,
    assert_adaptive_schema_is_label_free,
    assign_adaptive_codebook,
    audit_codebook_invariants,
    build_adaptive_occurrence,
    codebook_center_hash,
    fit_adaptive_codebook,
    _fit_transform_fold_descriptors,
)
from experiments.motion_primitive.online_secondary_codebook import (
    UnlabelledTokenOccurrence,
)


def occurrence(
    trial_id: int,
    subject_id: int,
    parent_token: int,
    descriptor: tuple[float, ...],
    token_fraction: float = 0.75,
) -> AdaptiveOccurrence:
    return AdaptiveOccurrence(
        trial_global_id=int(trial_id),
        subject_id=int(subject_id),
        parent_token=int(parent_token),
        token_fraction=float(token_fraction),
        duration_samples=128,
        descriptor=np.asarray(descriptor, dtype=np.float32),
        descriptor_block_sizes=(len(descriptor),),
    )


def replicated_two_mode_parent(
    parent_token: int,
    trial_offset: int = 0,
) -> list[AdaptiveOccurrence]:
    result = []
    trial_id = int(trial_offset)
    for subject_id in (4, 5):
        for center in (0.0, 10.0):
            for noise in (-0.10, 0.00, 0.10):
                result.append(
                    occurrence(
                        trial_id,
                        subject_id,
                        parent_token,
                        (center + noise, 0.5 * noise),
                    )
                )
                trial_id += 1
    return result


def subject_confounded_parent(
    parent_token: int,
    trial_offset: int = 100,
) -> list[AdaptiveOccurrence]:
    result = []
    trial_id = int(trial_offset)
    for subject_id, center in ((4, 0.0), (5, 10.0)):
        for noise in (-0.10, 0.00, 0.10):
            result.append(
                occurrence(
                    trial_id,
                    subject_id,
                    parent_token,
                    (center + noise, 0.5 * noise),
                )
            )
            trial_id += 1
    return result


class AdaptiveCodebookTests(unittest.TestCase):
    def setUp(self) -> None:
        self.base_centers = np.arange(32 * 4, dtype=np.float32).reshape(32, 4)

    def test_fitting_schema_and_signature_are_label_free(self) -> None:
        assert_adaptive_schema_is_label_free()
        forbidden = ("label", "activity", "class", "name")
        names = {item.name.lower() for item in fields(AdaptiveOccurrence)}
        self.assertFalse(
            any(fragment in name for name in names for fragment in forbidden)
        )
        parameters = set(inspect.signature(fit_adaptive_codebook).parameters)
        self.assertEqual(
            parameters,
            {"base_centers", "train_occurrences", "session_index", "config"},
        )
        self.assertFalse(
            parameters
            & {
                "test_occurrences",
                "labels",
                "activity_labels",
                "activity_names",
                "novel_class_ids",
            }
        )

    def test_public_builder_uses_fixed_blocks_without_test_fitted_scaling(self) -> None:
        raw = UnlabelledTokenOccurrence(
            trial_global_id=7,
            coarse_token=16,
            token_fraction=0.80,
            duration_samples=256,
            mean_quantization_residual=np.asarray([1.0, 2.0], dtype=np.float32),
            gravity_direction=np.asarray([0.0, 0.0, 2.0], dtype=np.float32),
        )
        built = build_adaptive_occurrence(
            raw,
            subject_id=4,
            motion_components=np.asarray([0.0, np.e - 1.0]),
        )
        self.assertEqual(built.descriptor_block_sizes, (2, 3, 2))
        np.testing.assert_allclose(
            built.descriptor,
            np.asarray([1.0, 2.0, 0.0, 0.0, 1.0, 0.0, 1.0]),
            atol=1e-6,
        )
        self.assertFalse(built.descriptor.flags.writeable)

    def test_extreme_held_subject_cannot_change_fold_complement_scaler(self) -> None:
        train = np.asarray(
            [[0.0, 1.0], [0.2, 1.2], [9.8, 2.0], [10.0, 2.2]],
            dtype=np.float64,
        )
        train_subjects = np.asarray([4, 4, 5, 5], dtype=np.int64)
        ordinary_held = np.asarray([[1.0, 1.0], [9.0, 2.0]], dtype=np.float64)
        extreme_held = np.asarray(
            [[-1.0e12, 1.0e12], [1.0e12, -1.0e12]], dtype=np.float64
        )
        ordinary = _fit_transform_fold_descriptors(
            train, ordinary_held, train_subjects, (1, 1)
        )
        extreme = _fit_transform_fold_descriptors(
            train, extreme_held, train_subjects, (1, 1)
        )
        np.testing.assert_array_equal(ordinary[2], extreme[2])
        np.testing.assert_array_equal(ordinary[3], extreme[3])
        np.testing.assert_array_equal(ordinary[0], extreme[0])
        self.assertFalse(np.array_equal(ordinary[1], extreme[1]))

    def test_cross_subject_mode_is_selected_and_subject_mode_is_rejected(self) -> None:
        valid = replicated_two_mode_parent(16, trial_offset=0)
        confounded = subject_confounded_parent(7, trial_offset=100)
        model = fit_adaptive_codebook(
            self.base_centers,
            valid + confounded,
            session_index=2,
        )
        self.assertTrue(model.gate_enabled)
        self.assertEqual(model.selected_parent_token, 16)
        self.assertEqual(model.K_total, 34)
        self.assertEqual(model.expansions[0].child_token_ids, (32, 33))

        audits = {item.parent_token: item for item in model.candidate_audits}
        self.assertTrue(audits[16].accepted)
        self.assertEqual(audits[16].child_subject_counts, (2, 2))
        self.assertGreaterEqual(audits[16].loo_distortion_reduction, 0.15)
        self.assertGreaterEqual(audits[16].loo_stability_subject_balanced, 0.80)
        self.assertEqual(audits[16].required_child_trial_count, 3)
        self.assertEqual(audits[16].loso_evaluable_holdout_count, 2)
        self.assertGreaterEqual(audits[16].loso_stability_subject_balanced, 0.80)
        self.assertLessEqual(audits[16].child_subject_nmi, 0.25)
        for total in dict(audits[16].subject_weight_totals).values():
            self.assertAlmostEqual(total, 0.5)

        self.assertFalse(audits[7].accepted)
        self.assertIn(
            "child_subject_support_below_minimum", audits[7].rejection_reasons
        )
        self.assertIn(
            "child_partition_is_subject_confounded", audits[7].rejection_reasons
        )
        self.assertIn(
            "fewer_than_two_evaluable_loso_holdouts",
            audits[7].rejection_reasons,
        )

    def test_minimum_child_fraction_raises_required_support(self) -> None:
        values = []
        trial_id = 0
        # A cross-subject 13:3 split satisfies the absolute support of three,
        # but not ceil(0.25 * 16) == four.
        for subject_id, small_count, large_count in ((4, 2, 6), (5, 1, 7)):
            for index in range(large_count):
                values.append(
                    occurrence(
                        trial_id,
                        subject_id,
                        12,
                        (0.02 * index, 0.0),
                    )
                )
                trial_id += 1
            for index in range(small_count):
                values.append(
                    occurrence(
                        trial_id,
                        subject_id,
                        12,
                        (10.0 + 0.02 * index, 0.0),
                    )
                )
                trial_id += 1
        config = AdaptiveCodebookConfig(minimum_child_fraction=0.25)
        model = fit_adaptive_codebook(
            self.base_centers, values, session_index=2, config=config
        )
        audit = model.candidate_audits[0]
        self.assertEqual(audit.child_trial_counts, (13, 3))
        self.assertEqual(audit.required_child_trial_count, 4)
        self.assertIn(
            "child_trial_support_below_minimum", audit.rejection_reasons
        )
        self.assertFalse(model.gate_enabled)

    def test_homogeneous_or_under_supported_parent_keeps_k32(self) -> None:
        homogeneous = [
            occurrence(index, 4 + index % 2, 9, (1.0, 1.0))
            for index in range(8)
        ]
        sparse = [
            occurrence(100 + index, 4 + index % 2, 3, (float(index), 0.0))
            for index in range(4)
        ]
        model = fit_adaptive_codebook(
            self.base_centers,
            homogeneous + sparse,
            session_index=2,
        )
        self.assertFalse(model.gate_enabled)
        self.assertIsNone(model.selected_parent_token)
        self.assertEqual(model.K_total, 32)
        audits = {item.parent_token: item for item in model.candidate_audits}
        self.assertIn("two_child_fit_unavailable", audits[9].rejection_reasons[0])
        self.assertIn(
            "parent_trial_support_below_minimum", audits[3].rejection_reasons
        )

    def test_assignment_and_histogram_preserve_low_confidence_parent_fallback(self) -> None:
        model = fit_adaptive_codebook(
            self.base_centers,
            replicated_two_mode_parent(16),
            session_index=2,
        )
        inference = [
            occurrence(1000, 4, 16, (0.02, 0.00)),
            occurrence(1001, 5, 16, (5.00, 0.00)),
            occurrence(1002, 4, 16, (0.02, 0.00), token_fraction=0.20),
            occurrence(1003, 4, 5, (0.02, 0.00)),
        ]
        assignments = assign_adaptive_codebook(model, inference)
        by_trial = {
            int(trial_id): int(token)
            for trial_id, token in zip(
                assignments.trial_ids.tolist(), assignments.output_tokens.tolist()
            )
        }
        self.assertIn(by_trial[1000], (32, 33))
        self.assertEqual(by_trial[1001], 16)
        self.assertEqual(by_trial[1002], 16)
        self.assertEqual(by_trial[1003], 5)

        state_map = adaptive_state_by_trial(model, assignments)
        self.assertIn(state_map[1000], (1, 2))
        self.assertEqual(state_map[1001], 0)
        self.assertEqual(state_map[1002], 0)
        self.assertNotIn(1003, state_map)
        durations = {
            1000: {16: 80.0, 5: 20.0},
            1001: {16: 100.0},
            1002: {16: 20.0, 5: 80.0},
            1003: {5: 100.0},
        }
        histograms = adaptive_duration_histograms(
            [1000, 1001, 1002, 1003], durations, model, state_map
        )
        self.assertEqual(histograms.shape, (4, 34))
        np.testing.assert_allclose(histograms.sum(axis=1), np.ones(4), atol=1e-7)
        self.assertAlmostEqual(histograms[0, by_trial[1000]], 0.8)
        self.assertAlmostEqual(histograms[0, 16], 0.0)
        self.assertAlmostEqual(histograms[1, 16], 1.0)
        self.assertAlmostEqual(histograms[2, 16], 0.2)
        self.assertAlmostEqual(histograms[3, 5], 1.0)

    def test_parent_selection_tie_breaks_by_smaller_token_and_is_order_invariant(self) -> None:
        parent_5 = replicated_two_mode_parent(5, trial_offset=100)
        parent_4 = replicated_two_mode_parent(4, trial_offset=200)
        forward = fit_adaptive_codebook(
            self.base_centers,
            parent_5 + parent_4,
            session_index=2,
        )
        reverse = fit_adaptive_codebook(
            self.base_centers,
            list(reversed(parent_5 + parent_4)),
            session_index=2,
        )
        self.assertEqual(forward.selected_parent_token, 4)
        self.assertEqual(reverse.selected_parent_token, 4)
        self.assertEqual(
            forward.expansions[0].child_medoid_trial_ids,
            reverse.expansions[0].child_medoid_trial_ids,
        )
        self.assertEqual(
            forward.expansions[0].fit_child_ids,
            reverse.expansions[0].fit_child_ids,
        )

    def test_center_hash_and_append_only_invariants_detect_external_mutation(self) -> None:
        external = self.base_centers.copy()
        before = codebook_center_hash(external)
        model = fit_adaptive_codebook(
            external,
            replicated_two_mode_parent(16),
            session_index=2,
        )
        clean = audit_codebook_invariants(model, external)
        self.assertTrue(clean["all_invariants_passed"])
        self.assertEqual(before, clean["base_center_hash"])
        self.assertEqual(clean["appended_child_token_ids"], [32, 33])
        with self.assertRaises(ValueError):
            model.base_centers[0, 0] = -1.0

        external[0, 0] = -1.0
        changed = audit_codebook_invariants(model, external)
        self.assertFalse(changed["all_invariants_passed"])
        self.assertFalse(changed["checks"]["current_base_hash_matches_frozen"])
        self.assertFalse(changed["checks"]["current_base_centers_exactly_equal"])
        self.assertEqual(codebook_center_hash(model.base_centers), before)

    def test_diagnostics_close_capacity_and_train_only_audit(self) -> None:
        model = fit_adaptive_codebook(
            self.base_centers,
            replicated_two_mode_parent(16),
            session_index=2,
        )
        diagnostics = adaptive_codebook_diagnostics(model)
        self.assertEqual(diagnostics["K_total"], 34)
        self.assertTrue(diagnostics["gate_enabled"])
        self.assertEqual(diagnostics["selected_parent_token"], 16)
        self.assertEqual(model.config.minimum_child_fraction, 0.15)
        self.assertFalse(diagnostics["fit_uses_activity_labels_or_names"])
        self.assertFalse(diagnostics["subject_id_used_as_descriptor"])
        self.assertFalse(diagnostics["test_occurrences_used_for_fit"])
        self.assertTrue(
            diagnostics["invariant_audit"]["all_invariants_passed"]
        )


if __name__ == "__main__":
    unittest.main()
