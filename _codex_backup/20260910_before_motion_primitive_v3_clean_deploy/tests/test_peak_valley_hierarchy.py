from __future__ import annotations

import math
import tempfile
import unittest
from collections import Counter
from dataclasses import replace
from pathlib import Path

import numpy as np

import experiments.motion_primitive.peak_valley_hierarchy as hierarchy_module
import experiments.motion_primitive.run_peak_valley_hierarchy_cv as cv_module
from experiments.motion_primitive.peak_valley_hierarchy import (
    ChildPrimitiveBatch,
    CodebookConfig,
    ParentGateConfig,
    PeakValleyConfig,
    SampleSegment,
    SegmentedTrial,
    TokenizedTrial,
    TrialSignal,
    TurningEvent,
    assign_codebook,
    fit_codebook,
    fit_matched_random_parent_catalog,
    fit_parent_catalog,
    fit_segmenter,
    load_codebook_state,
    load_parent_catalog_state,
    load_peak_valley_state,
    matched_random_segmentations,
    overlay_parents,
    save_state_json,
    segmentation_from_boundaries,
    segment_trial,
)


SEGMENTER_HASH = "a" * 64
CODEBOOK_HASH = "b" * 64
_CASE = unittest.TestCase()


def _motion_config(
    *, minimum_seconds: float = 0.10, min_axis_votes: int = 2
) -> PeakValleyConfig:
    return PeakValleyConfig(
        sample_rate_hz=100.0,
        smoothing_seconds=0.03,
        extrema_min_distance_seconds=0.20,
        prominence_mad_multiplier=0.50,
        axis_vote_tolerance_seconds=0.03,
        min_axis_votes=min_axis_votes,
        min_segment_seconds=minimum_seconds,
        feature_confirmation=False,
        shape_points=16,
    )


def _periodic_trial(
    trial_id: str = "periodic", subject_id: str = "subject-1", length: int = 600
) -> TrialSignal:
    samples = np.arange(length, dtype=np.float64)
    signal = np.zeros((6, length), dtype=np.float64)
    wave = np.sin(2.0 * np.pi * samples / 100.0)
    signal[0] = wave
    signal[1] = 0.75 * wave
    signal[2] = 1.25 * wave
    return TrialSignal(trial_id, subject_id, signal)


def _child_batch(
    trial_id: str,
    subject_id: str,
    shape_rows: list[list[float]],
    *,
    segmentation_hash: str = SEGMENTER_HASH,
) -> ChildPrimitiveBatch:
    shape = np.asarray(shape_rows, dtype=np.float64)
    count = len(shape)
    starts = np.arange(count, dtype=np.int64) * 10
    return ChildPrimitiveBatch(
        trial_id=trial_id,
        subject_id=subject_id,
        segment_indices=np.arange(count, dtype=np.int64),
        start_samples=starts,
        end_samples_exclusive=starts + 10,
        shape_features=shape,
        statistic_features=np.ones((count, 1), dtype=np.float64),
        statistic_names=("duration_seconds",),
        segmentation_state_sha256=segmentation_hash,
    ).validate()


def _tokenized_trial(
    trial_id: str,
    subject_id: str,
    tokens: list[int],
    event_kinds: list[str],
    *,
    codebook_hash: str = CODEBOOK_HASH,
    primitive_num: int = 3,
) -> TokenizedTrial:
    token_array = np.asarray(tokens, dtype=np.int64)
    count = len(token_array)
    assert len(event_kinds) == count - 1
    boundaries = np.arange(count + 1, dtype=np.int64) * 10
    events = tuple(
        TurningEvent(
            sample_index=int(boundaries[index + 1]),
            kind=kind,
            canonical_prominence=1.0,
            canonical_normalized_prominence=1.0,
            axis_indices=(0, 1),
            axis_sample_indices=(int(boundaries[index + 1]),) * 2,
            axis_prominences=(1.0, 1.0),
            axis_normalized_prominences=(1.0, 1.0),
        )
        for index, kind in enumerate(event_kinds)
    )
    segments = tuple(
        SampleSegment(
            segment_index=index,
            start_sample=int(boundaries[index]),
            end_sample_exclusive=int(boundaries[index + 1]),
            left_event_kind=None if index == 0 else event_kinds[index - 1],
            right_event_kind=None if index == count - 1 else event_kinds[index],
        )
        for index in range(count)
    )
    segmentation = SegmentedTrial(
        trial_id=trial_id,
        subject_id=subject_id,
        sample_count=int(boundaries[-1]),
        boundaries=boundaries,
        events=events,
        segments=segments,
        state_sha256=SEGMENTER_HASH,
        source="synthetic_test",
        diagnostics={"complete_partition": True},
    ).validate()
    batch = _child_batch(
        trial_id,
        subject_id,
        [[float(token), 1.0] for token in tokens],
    )
    return TokenizedTrial(
        segmentation=segmentation,
        child_features=batch,
        primitive_tokens=token_array,
        nearest_center_distances=np.zeros(count, dtype=np.float64),
        codebook_embeddings=np.c_[token_array, np.ones(count)].astype(np.float64),
        codebook_state_sha256=codebook_hash,
        primitive_num=primitive_num,
    ).validate()


def _parent_fit_trials(*, codebook_hash: str = CODEBOOK_HASH) -> list[TokenizedTrial]:
    trials: list[TokenizedTrial] = []
    # (peak, 0, 1) appears in two trials for each of three subjects.  Removing
    # any one supporting subject therefore leaves 4 trials and 2 subjects.
    for subject in ("s1", "s2", "s3"):
        for repetition in range(2):
            if subject == "s1" and repetition == 0:
                tokens = [0, 1, 2]
                kinds = ["peak", "valley"]
            else:
                tokens = [0, 1]
                kinds = ["peak"]
            trials.append(
                _tokenized_trial(
                    f"{subject}-trial-{repetition}",
                    subject,
                    tokens,
                    kinds,
                    codebook_hash=codebook_hash,
                )
            )
    # This subject is in the fit set but has no peak event.  Its target-key LOSO
    # fold is an explicit no-op, and it still belongs to the all-fit-subject
    # denominator.
    trials.append(
        _tokenized_trial(
            "s4-unrelated",
            "s4",
            [2, 2],
            ["valley"],
            codebook_hash=codebook_hash,
        )
    )
    return trials


def _parent_config() -> ParentGateConfig:
    return ParentGateConfig(
        minimum_occurrences=4,
        minimum_trials=4,
        minimum_subjects=2,
        minimum_npmi=0.0,
        minimum_mdl_gain_bits=0.0,
        minimum_loso_stability=1.0,
    )


def _parent_control_trials() -> list[TokenizedTrial]:
    trials = []
    for subject in ("s1", "s2", "s3"):
        # The reference motif has two trials per subject (six total, four after
        # leave-one-supporting-subject-out); the negative motif has only three.
        for repetition in range(2):
            trials.append(
                _tokenized_trial(
                    f"{subject}-reference-{repetition}",
                    subject,
                    [0, 1],
                    ["peak"],
                    primitive_num=4,
                )
            )
        trials.append(
            _tokenized_trial(
                f"{subject}-negative",
                subject,
                [2, 3],
                ["peak"],
                primitive_num=4,
            )
        )
    return trials


def _flag_value(command: list[str], flag: str) -> str:
    return command[command.index(flag) + 1]


def test_sample_level_multi_axis_peak_valley_detection() -> None:
    trial = _periodic_trial()
    state = fit_segmenter([trial], _motion_config())
    segmented = segment_trial(trial, state)

    assert {event.kind for event in segmented.events} == {"peak", "valley"}
    assert all(event.axis_votes >= 2 for event in segmented.events)
    assert all(set(event.axis_indices) == {0, 1, 2} for event in segmented.events)
    assert segmented.diagnostics["boundary_resolution"] == "one raw sample"
    # Boundaries such as sample 25 prove that the detector is not restricted
    # to the historical 128-sample window-stride grid.
    assert any(int(boundary) % 128 != 0 for boundary in segmented.boundaries[1:-1])


def test_single_axis_glitch_is_rejected_by_multi_axis_vote() -> None:
    length = 600
    samples = np.arange(length, dtype=np.float64)
    signal = np.zeros((6, length), dtype=np.float64)
    signal[0] = 0.05 * np.sin(2.0 * np.pi * samples / 100.0)
    signal[0] += 5.0 * np.exp(-0.5 * np.square((samples - 317.0) / 3.0))
    trial = TrialSignal("single-axis-glitch", "subject-1", signal)

    segmented = segment_trial(trial, fit_segmenter([trial], _motion_config()))

    assert segmented.boundaries.tolist() == [0, length]
    assert segmented.diagnostics["canonical_extremum_count"] > 0
    assert (
        segmented.diagnostics["canonical_candidates_below_min_axis_votes"]
        == segmented.diagnostics["canonical_extremum_count"]
    )


def test_zero_axis_votes_is_pure_canonical_envelope_segmentation() -> None:
    length = 600
    samples = np.arange(length, dtype=np.float64)
    signal = np.zeros((6, length), dtype=np.float64)
    signal[0] = np.sin(2.0 * np.pi * samples / 100.0)
    trial = TrialSignal("canonical-only", "subject-1", signal)
    config = _motion_config(min_axis_votes=0)

    segmented = segment_trial(trial, fit_segmenter([trial], config))

    assert segmented.diagnostics["consensus_mode"] == (
        "canonical_motion_envelope_without_axis_vote"
    )
    assert segmented.diagnostics["axis_vote_bypassed"] is True
    assert segmented.diagnostics["axis_extremum_count"] == 0
    assert len(segmented.events) > 0
    assert all(event.axis_votes == 0 for event in segmented.events)
    assert all(event.source == "canonical_motion_envelope_no_axis_vote" for event in segmented.events)


def test_flat_or_short_trial_remains_one_complete_segment() -> None:
    for length in (12, 240):
        with _CASE.subTest(length=length):
            trial = TrialSignal(
                f"flat-{length}",
                "subject-1",
                np.zeros((6, length), dtype=np.float64),
            )
            state = fit_segmenter([trial], _motion_config(minimum_seconds=0.25))

            segmented = segment_trial(trial, state)

            assert segmented.boundaries.tolist() == [0, length]
            assert len(segmented.segments) == 1
            assert segmented.segments[0].sample_count == length
            assert segmented.events == ()
            assert segmented.diagnostics["complete_partition"] is True


def test_detected_partition_is_complete_and_respects_minimum_duration() -> None:
    trial = _periodic_trial()
    state = fit_segmenter([trial], _motion_config(minimum_seconds=0.30))

    segmented = segment_trial(trial, state)

    assert segmented.boundaries[0] == 0
    assert segmented.boundaries[-1] == trial.signal.shape[1]
    assert np.all(np.diff(segmented.boundaries) >= state.config.min_segment_samples)
    assert sum(segment.sample_count for segment in segmented.segments) == len(
        trial.signal[0]
    )


def test_matched_random_boundaries_preserve_counts_and_are_deterministic() -> None:
    trials = [
        TrialSignal("trial-a", "s1", np.zeros((6, 200), dtype=np.float64)),
        TrialSignal("trial-b", "s2", np.zeros((6, 200), dtype=np.float64)),
    ]
    state = fit_segmenter(trials, _motion_config(minimum_seconds=0.20))
    references = [
        segmentation_from_boundaries(
            trials[0], state, [0, 30, 65, 100, 145, 175, 200]
        ),
        segmentation_from_boundaries(trials[1], state, [0, 40, 80, 120, 160, 200]),
    ]

    first = matched_random_segmentations(trials, references, state, seed=91)
    repeated = matched_random_segmentations(
        list(reversed(trials)), list(reversed(references)), state, seed=91
    )
    first_by_id = {item.trial_id: item for item in first}
    repeated_by_id = {item.trial_id: item for item in repeated}

    assert set(first_by_id) == set(repeated_by_id)
    for reference in references:
        observed = first_by_id[reference.trial_id]
        assert len(observed.segments) == len(reference.segments)
        assert np.all(np.diff(observed.boundaries) >= state.config.min_segment_samples)
        assert np.array_equal(
            observed.boundaries, repeated_by_id[reference.trial_id].boundaries
        )
        assert [event.kind for event in observed.events] == [
            event.kind for event in repeated_by_id[reference.trial_id].events
        ]
        matching = observed.diagnostics["control_matching"]
        assert matching["matched_segment_count"] is True
        assert matching["matched_minimum_segment_length_constraint"] is True
        assert matching["matched_event_kind_histogram"] is True
        assert matching["matched_reference_segment_length_distribution"] is False
        assert matching["joint_legal_partition_sampling_is_uniform"] is True


def test_uniform_weak_composition_has_no_multinomial_middle_bias() -> None:
    rng = np.random.default_rng(20260906)
    observed = Counter(
        tuple(hierarchy_module._uniform_weak_composition(2, 2, rng).tolist())
        for _ in range(6000)
    )

    assert set(observed) == {(0, 2), (1, 1), (2, 0)}
    # A multinomial draw would put about half the mass on (1,1).  Uniform
    # stars-and-bars should put one third on each legal composition.
    for count in observed.values():
        _CASE.assertAlmostEqual(count / 6000.0, 1.0 / 3.0, delta=0.025)


def test_codebook_assignment_rejects_subject_state_and_span_mismatches() -> None:
    tokenized = _tokenized_trial("strict", "s1", [0, 1], ["peak"])
    batch = tokenized.child_features
    segmentation = tokenized.segmentation
    codebook = fit_codebook(
        [batch],
        CodebookConfig(
            primitive_num=2,
            pca_dim=0,
            l2_normalize=False,
            kmeans_n_init=2,
            random_seed=3,
        ),
    )

    assigned = assign_codebook(batch, segmentation, codebook)
    assert assigned.subject_id == "s1"

    with _CASE.assertRaisesRegex(ValueError, "subject IDs differ"):
        assign_codebook(replace(batch, subject_id="s2"), segmentation, codebook)
    with _CASE.assertRaisesRegex(ValueError, "state fingerprints differ"):
        assign_codebook(
            replace(batch, segmentation_state_sha256="c" * 64),
            segmentation,
            codebook,
        )
    with _CASE.assertRaisesRegex(ValueError, "spans differ"):
        assign_codebook(
            replace(batch, end_samples_exclusive=np.asarray([9, 20])),
            segmentation,
            codebook,
        )

    other_state_batch = replace(batch, segmentation_state_sha256="d" * 64)
    other_state_codebook = fit_codebook(
        [other_state_batch],
        CodebookConfig(
            primitive_num=2,
            pca_dim=0,
            l2_normalize=False,
            kmeans_n_init=2,
            random_seed=3,
        ),
    )
    with _CASE.assertRaisesRegex(ValueError, "fitted codebook"):
        assign_codebook(batch, segmentation, other_state_codebook)

    malformed_tokenized = replace(
        tokenized,
        child_features=replace(batch, subject_id="s2"),
    )
    with _CASE.assertRaisesRegex(ValueError, "subject IDs differ"):
        malformed_tokenized.validate()


def test_all_three_learned_states_save_load_with_identical_hashes() -> None:
    trial = _periodic_trial(length=300)
    segmenter = fit_segmenter([trial], _motion_config())
    codebook = fit_codebook(
        [
            _child_batch("cb-1", "s1", [[0.0, 0.0], [0.1, 0.0]]),
            _child_batch("cb-2", "s2", [[2.0, 2.0], [2.1, 2.0]]),
        ],
        CodebookConfig(
            primitive_num=2,
            pca_dim=0,
            l2_normalize=False,
            kmeans_n_init=2,
            random_seed=7,
        ),
    )
    parent = fit_parent_catalog(
        _parent_fit_trials(codebook_hash=codebook.state_hash()), _parent_config()
    )

    loaders_and_states = (
        (load_peak_valley_state, segmenter),
        (load_codebook_state, codebook),
        (load_parent_catalog_state, parent),
    )
    with tempfile.TemporaryDirectory() as directory:
        for index, (loader, state) in enumerate(loaders_and_states):
            path = Path(directory) / f"state-{index}.json"
            save_state_json(path, state)
            loaded = loader(path)
            assert loaded.state_hash() == state.state_hash()
            assert loaded.to_dict() == state.to_dict()

    obsolete_parent = parent.to_dict()
    obsolete_parent["schema_version"] = 3
    obsolete_parent["algorithm_revision"] = "sample_peak_valley_hierarchy_v3"
    with _CASE.assertRaisesRegex(ValueError, "schema/revision mismatch"):
        hierarchy_module.ParentCatalogState.from_dict(obsolete_parent)


def test_subject_trial_equal_weights_equalize_subjects_and_trials() -> None:
    batches = [
        _child_batch("s1-short", "s1", [[0.0, 0.0]]),
        _child_batch("s1-long", "s1", [[1.0, 0.0]] * 3),
        _child_batch("s2-only", "s2", [[2.0, 0.0]] * 2),
    ]

    weights = hierarchy_module._segment_weights(batches, "subject_trial_equal")

    np.testing.assert_allclose(weights, [1.5, 0.5, 0.5, 0.5, 1.5, 1.5])
    _CASE.assertAlmostEqual(float(weights[0]), float(weights[1:4].sum()))
    _CASE.assertAlmostEqual(float(weights[:4].sum()), float(weights[4:].sum()))
    _CASE.assertAlmostEqual(float(weights.mean()), 1.0)


def test_balanced_npmi_matches_hand_calculation_and_p_joint_one_is_zero() -> None:
    Row = hierarchy_module._ParentEventRow
    target = ("peak", 0, 0)
    rows = [
        # Subject s1 has one four-boundary trial: every row has mass 1/8.
        Row(target, "s1-t1", "s1", 0),
        Row(target, "s1-t1", "s1", 2),
        Row(("peak", 0, 1), "s1-t1", "s1", 4),
        Row(("peak", 1, 0), "s1-t1", "s1", 6),
        # Subject s2 has two one-boundary trials: each row has mass 1/4.
        Row(target, "s2-t1", "s2", 0),
        Row(("peak", 0, 1), "s2-t2", "s2", 0),
    ]
    observed = hierarchy_module._association_for_key(rows, target, primitive_num=3)
    p_joint = 0.5
    p_left = 0.875
    p_right = 0.625
    expected_ratio = p_joint / (p_left * p_right)

    _CASE.assertAlmostEqual(observed["pmi_bits"], math.log2(expected_ratio))
    _CASE.assertAlmostEqual(
        observed["npmi"], math.log(expected_ratio) / -math.log(p_joint)
    )

    certain = hierarchy_module._association_for_key(
        [
            Row(target, "certain-1", "s1", 0),
            Row(target, "certain-2", "s2", 0),
        ],
        target,
        primitive_num=3,
    )
    _CASE.assertAlmostEqual(certain["pmi_bits"], 0.0)
    _CASE.assertAlmostEqual(certain["npmi"], 0.0)


def _random_parent_trials(seed: int, primitive_num: int = 5) -> list[TokenizedTrial]:
    rng = np.random.default_rng(seed)
    trials = []
    for subject_index in range(5):
        for repetition in range(3):
            child_count = int(rng.integers(9, 17))
            tokens = rng.integers(0, primitive_num, size=child_count).tolist()
            kinds = rng.choice(
                np.asarray(["peak", "valley"], dtype=object),
                size=child_count - 1,
            ).tolist()
            trials.append(
                _tokenized_trial(
                    f"random-{seed}-s{subject_index}-t{repetition}",
                    f"s{subject_index}",
                    tokens,
                    kinds,
                    primitive_num=primitive_num,
                )
            )
    return list(reversed(trials))


def _slow_parent_snapshot(
    trials: list[TokenizedTrial],
    config: ParentGateConfig,
    parent_symbol_count: int,
) -> dict:
    ordered = hierarchy_module._ordered_tokenized(trials)
    rows = hierarchy_module._parent_rows(ordered)
    primitive_num = ordered[0].primitive_num
    all_subjects = sorted(
        {
            hierarchy_module._identifier_sort_key(trial.subject_id)
            for trial in ordered
        }
    )
    result = {}
    for key in sorted({row.key for row in rows}):
        metrics = hierarchy_module._association_for_key(
            rows,
            key,
            primitive_num,
            parent_symbol_count=parent_symbol_count,
        )
        supporting_subjects = sorted(
            {
                hierarchy_module._identifier_sort_key(row.subject_id)
                for row in rows
                if row.key == key
            }
        )
        kind_subjects = {
            hierarchy_module._identifier_sort_key(row.subject_id)
            for row in rows
            if row.key[0] == key[0]
        }
        loso_rows = []
        for subject in all_subjects:
            reduced = [
                row
                for row in rows
                if hierarchy_module._identifier_sort_key(row.subject_id) != subject
            ]
            reduced_metrics = hierarchy_module._association_for_key(
                reduced,
                key,
                primitive_num,
                parent_symbol_count=parent_symbol_count,
            )
            base_checks = {
                "occurrences": (
                    reduced_metrics["occurrences"] >= config.minimum_occurrences
                ),
                "trials": reduced_metrics["trial_count"] >= config.minimum_trials,
                "subjects": (
                    reduced_metrics["subject_count"] >= config.minimum_subjects
                ),
                "npmi": reduced_metrics["npmi"] >= config.minimum_npmi,
                "mdl_gain": (
                    reduced_metrics["mdl_gain_bits"]
                    >= config.minimum_mdl_gain_bits
                ),
            }
            loso_rows.append((subject, reduced_metrics, base_checks))
        supporter_loso_rows = [
            item for item in loso_rows if item[0] in supporting_subjects
        ]
        effectful_loso_rows = [
            item for item in loso_rows if item[0] in kind_subjects
        ]
        supporter_pass_count = sum(
            all(item[2].values()) for item in supporter_loso_rows
        )
        effectful_pass_count = sum(
            all(item[2].values()) for item in effectful_loso_rows
        )
        supporter_stability = (
            supporter_pass_count / len(supporter_loso_rows)
        )
        effectful_stability = (
            effectful_pass_count / len(effectful_loso_rows)
        )
        checks = {
            "occurrences": metrics["occurrences"] >= config.minimum_occurrences,
            "trials": metrics["trial_count"] >= config.minimum_trials,
            "subjects": metrics["subject_count"] >= config.minimum_subjects,
            "npmi": metrics["npmi"] >= config.minimum_npmi,
            "mdl_gain": metrics["mdl_gain_bits"] >= config.minimum_mdl_gain_bits,
            "loso_supporter_stability": (
                supporter_stability >= config.minimum_loso_stability
            ),
            "loso_effectful_stability": (
                effectful_stability >= config.minimum_loso_stability
            ),
        }
        result[key] = {
            "metrics": metrics,
            "subjects": all_subjects,
            "supporting_subjects": tuple(supporting_subjects),
            "kind_subjects": tuple(sorted(kind_subjects)),
            "supporting_subject_count": len(supporting_subjects),
            "supporter_pass_count": supporter_pass_count,
            "supporter_stability": supporter_stability,
            "marginal_only_subject_count": len(
                kind_subjects.difference(supporting_subjects)
            ),
            "effectful_subject_count": len(kind_subjects),
            "effectful_pass_count": effectful_pass_count,
            "effectful_stability": effectful_stability,
            "no_kind_noop_subject_count": len(
                set(all_subjects).difference(kind_subjects)
            ),
            "loso": loso_rows,
            "checks": checks,
        }
    return result


def _assert_association_equivalent(expected: dict, observed: dict) -> None:
    assert set(expected) == set(observed)
    for key in expected:
        left = expected[key]
        right = observed[key]
        if isinstance(left, (int, np.integer)):
            assert int(right) == int(left), key
        elif math.isinf(float(left)):
            assert float(right) == float(left), key
        else:
            _CASE.assertAlmostEqual(float(right), float(left), places=14, msg=key)


def test_parent_association_index_matches_slow_reference_and_registration() -> None:
    config = ParentGateConfig(
        minimum_occurrences=4,
        minimum_trials=3,
        minimum_subjects=2,
        minimum_npmi=-0.25,
        minimum_mdl_gain_bits=0.0,
        minimum_loso_stability=0.5,
    ).validate()
    for seed in (3, 19, 71):
        trials = _random_parent_trials(seed)
        ordered = hierarchy_module._ordered_tokenized(trials)
        rows = hierarchy_module._parent_rows(ordered)
        keys = sorted({row.key for row in rows})
        all_subjects = sorted(
            {
                hierarchy_module._identifier_sort_key(trial.subject_id)
                for trial in ordered
            }
        )
        for parent_symbol_count in (1, 9):
            index = hierarchy_module._ParentAssociationIndex(
                rows,
                primitive_num=5,
                parent_symbol_count=parent_symbol_count,
            )
            for key in keys:
                _assert_association_equivalent(
                    hierarchy_module._association_for_key(
                        rows,
                        key,
                        primitive_num=5,
                        parent_symbol_count=parent_symbol_count,
                    ),
                    index.association(key),
                )
                for subject in all_subjects:
                    reduced = [
                        row
                        for row in rows
                        if hierarchy_module._identifier_sort_key(row.subject_id)
                        != subject
                    ]
                    _assert_association_equivalent(
                        hierarchy_module._association_for_key(
                            reduced,
                            key,
                            primitive_num=5,
                            parent_symbol_count=parent_symbol_count,
                        ),
                        index.association(key, omitted_subject=subject),
                    )

        optimistic = _slow_parent_snapshot(trials, config, parent_symbol_count=1)
        expected_capacity = max(
            1,
            sum(all(item["checks"].values()) for item in optimistic.values()),
        )
        expected = _slow_parent_snapshot(
            trials, config, parent_symbol_count=expected_capacity
        )
        catalog = fit_parent_catalog(trials, config)
        assert (
            catalog.selection_metadata["mdl_parent_symbol_capacity"]
            == expected_capacity
        )
        assert {item.key for item in catalog.registered_patterns} == {
            key for key, item in expected.items() if all(item["checks"].values())
        }
        for pattern in catalog.patterns:
            snapshot = expected[pattern.key]
            _assert_association_equivalent(
                snapshot["metrics"],
                {
                    "occurrences": pattern.occurrences,
                    "nonoverlap_occurrences": pattern.nonoverlap_occurrences,
                    "trial_count": pattern.trial_count,
                    "subject_count": pattern.subject_count,
                    "pmi_bits": pattern.pmi_bits,
                    "npmi": pattern.npmi,
                    "mdl_parent_symbol_count": pattern.mdl_parent_symbol_count,
                    "mdl_child_symbol_bits": pattern.mdl_child_symbol_bits,
                    "mdl_parent_symbol_bits": pattern.mdl_parent_symbol_bits,
                    "mdl_event_bits": pattern.mdl_event_bits,
                    "mdl_savings_per_nonoverlap_bits": (
                        pattern.mdl_savings_per_nonoverlap_bits
                    ),
                    "dictionary_cost_bits": pattern.mdl_dictionary_cost_bits,
                    "mdl_gain_bits": pattern.mdl_gain_bits,
                },
            )
            _CASE.assertAlmostEqual(
                pattern.loso_supporter_stability,
                snapshot["supporter_stability"],
                places=14,
            )
            _CASE.assertAlmostEqual(
                pattern.loso_effectful_stability,
                snapshot["effectful_stability"],
                places=14,
            )
            assert pattern.loso_total_fit_subject_count == len(all_subjects)
            assert pattern.loso_supporter_evaluation_count == snapshot[
                "supporting_subject_count"
            ]
            assert pattern.loso_supporter_pass_count == snapshot[
                "supporter_pass_count"
            ]
            assert pattern.loso_marginal_only_subject_count == snapshot[
                "marginal_only_subject_count"
            ]
            assert pattern.loso_effectful_evaluation_count == snapshot[
                "effectful_subject_count"
            ]
            assert pattern.loso_effectful_pass_count == snapshot[
                "effectful_pass_count"
            ]
            assert pattern.loso_no_kind_noop_subject_count == snapshot[
                "no_kind_noop_subject_count"
            ]
            for fold, (subject, reduced_metrics, base_checks) in zip(
                pattern.loso_folds, snapshot["loso"]
            ):
                supports_key = subject in snapshot["supporting_subjects"]
                has_kind = subject in snapshot["kind_subjects"]
                assert fold["omitted_subject_key"] == subject
                assert fold["subject_supports_key"] is supports_key
                assert fold["subject_has_event_kind"] is has_kind
                assert fold["included_in_supporter_stability"] is supports_key
                assert fold["included_in_effectful_stability"] is has_kind
                assert fold["base_gate_checks"] == base_checks
                assert fold["passed"] is all(base_checks.values())
                for field in ("occurrences", "trial_count", "subject_count"):
                    assert fold[field] == reduced_metrics[field]
                for field in ("npmi", "mdl_gain_bits"):
                    expected_value = reduced_metrics[field]
                    if math.isfinite(expected_value):
                        _CASE.assertAlmostEqual(
                            fold[field], expected_value, places=14
                        )
                    else:
                        assert fold[field] is None
            assert pattern.gate_checks == snapshot["checks"]


def test_aaa_and_aaaa_use_nonoverlapping_occurrences_for_mdl() -> None:
    Row = hierarchy_module._ParentEventRow
    key = ("peak", 1, 1)
    # AAA contributes adjacent offsets 0 and 1, hence one non-overlapping pair;
    # AAAA contributes offsets 0, 1, 2, hence two.
    rows = [
        Row(key, "AAA", "s1", 0),
        Row(key, "AAA", "s1", 1),
        Row(key, "AAAA", "s2", 0),
        Row(key, "AAAA", "s2", 1),
        Row(key, "AAAA", "s2", 2),
    ]

    observed = hierarchy_module._association_for_key(rows, key, primitive_num=4)

    assert observed["occurrences"] == 5
    assert observed["nonoverlap_occurrences"] == 3
    # K=4/P=1: child bits=2, parent-token bits=ceil(log2(5))=3,
    # event bits=1. Replacing 2 children+event by the parent saves 2 bits;
    # defining the dictionary entry costs 5 bits.
    assert observed["mdl_child_symbol_bits"] == 2
    assert observed["mdl_parent_symbol_bits"] == 3
    assert observed["mdl_event_bits"] == 1
    assert observed["dictionary_cost_bits"] == 5
    _CASE.assertAlmostEqual(observed["mdl_savings_per_nonoverlap_bits"], 2.0)
    _CASE.assertAlmostEqual(observed["mdl_gain_bits"], 3 * 2 - 5)


def test_mdl_parent_alphabet_cost_accounts_for_registered_capacity() -> None:
    Row = hierarchy_module._ParentEventRow
    key = ("peak", 1, 2)
    rows = [Row(key, f"trial-{index}", f"s{index}", 0) for index in range(4)]

    one_parent = hierarchy_module._association_for_key(
        rows, key, primitive_num=32, parent_symbol_count=1
    )
    thirty_three_parents = hierarchy_module._association_for_key(
        rows, key, primitive_num=32, parent_symbol_count=33
    )

    # K=32 crosses a bit-width boundary as soon as parents are introduced.
    assert one_parent["mdl_child_symbol_bits"] == 5
    assert one_parent["mdl_parent_symbol_bits"] == 6
    assert one_parent["mdl_event_bits"] == 1
    assert one_parent["mdl_savings_per_nonoverlap_bits"] == 5
    assert one_parent["mdl_gain_bits"] == 9
    # K+P=65 requires seven bits per parent/child sequence symbol.
    assert thirty_three_parents["mdl_parent_symbol_bits"] == 7
    assert thirty_three_parents["mdl_savings_per_nonoverlap_bits"] == 4
    assert thirty_three_parents["mdl_gain_bits"] == 5


def test_parent_registration_uses_cross_subject_support_and_base_gate_loso() -> None:
    catalog = fit_parent_catalog(_parent_fit_trials(), _parent_config())
    target = next(item for item in catalog.patterns if item.key == ("peak", 0, 1))
    rare = next(item for item in catalog.patterns if item.key == ("valley", 1, 2))

    assert target.parent_id is not None
    assert target.subject_count == 3
    assert target.trial_count == 6
    # s4 has no peak event.  Its fold remains in the audit table but is excluded
    # from both stability denominators.
    assert len(catalog.fit_subject_keys) == 4
    assert target.loso_total_fit_subject_count == 4
    assert target.loso_supporter_evaluation_count == 3
    assert target.loso_supporter_pass_count == 3
    _CASE.assertAlmostEqual(target.loso_supporter_stability, 1.0)
    assert target.loso_marginal_only_subject_count == 0
    assert target.loso_effectful_evaluation_count == 3
    assert target.loso_effectful_pass_count == 3
    _CASE.assertAlmostEqual(target.loso_effectful_stability, 1.0)
    assert target.loso_no_kind_noop_subject_count == 1
    _CASE.assertAlmostEqual(target.loso_stability, 1.0)
    assert all(
        set(fold["base_gate_checks"])
        == {"occurrences", "trials", "subjects", "npmi", "mdl_gain"}
        for fold in target.loso_folds
    )
    assert all(fold["passed"] for fold in target.loso_folds)
    no_kind_fold = next(
        fold
        for fold in target.loso_folds
        if fold["omission_effect"] == "no_kind_noop"
    )
    assert no_kind_fold["included_in_supporter_stability"] is False
    assert no_kind_fold["included_in_effectful_stability"] is False
    serialized = target.to_dict()
    assert "loso_stability" not in serialized
    assert {
        "loso_supporter_stability",
        "loso_supporter_pass_count",
        "loso_supporter_evaluation_count",
        "loso_effectful_stability",
        "loso_effectful_pass_count",
        "loso_effectful_evaluation_count",
        "loso_total_fit_subject_count",
        "loso_no_kind_noop_subject_count",
    }.issubset(serialized)
    invalid_target = replace(
        target,
        loso_effectful_evaluation_count=(
            target.loso_effectful_evaluation_count + 1
        ),
    )
    invalid_catalog = replace(
        catalog,
        patterns=tuple(
            invalid_target if item.key == target.key else item
            for item in catalog.patterns
        ),
    )
    with _CASE.assertRaisesRegex(ValueError, "Effectful LOSO count"):
        invalid_catalog.validate()
    assert rare.parent_id is None
    assert not rare.gate_checks["subjects"]
    assert not rare.gate_checks["trials"]


def test_marginal_only_failure_blocks_and_no_kind_is_excluded() -> None:
    trials = []
    for subject in ("s1", "s2", "s3"):
        for repetition in range(2):
            trials.append(
                _tokenized_trial(
                    f"{subject}-target-{repetition}",
                    subject,
                    [0, 1],
                    ["peak"],
                )
            )
    for repetition in range(2):
        trials.append(
            _tokenized_trial(
                f"s4-other-peak-{repetition}",
                "s4",
                [2, 2],
                ["peak"],
            )
        )
    trials.append(_tokenized_trial("s5-valley", "s5", [1, 2], ["valley"]))
    config = ParentGateConfig(
        minimum_occurrences=4,
        minimum_trials=4,
        minimum_subjects=2,
        minimum_npmi=0.1,
        minimum_mdl_gain_bits=0.0,
        minimum_loso_stability=0.9,
    )

    catalog = fit_parent_catalog(trials, config)
    target = next(item for item in catalog.patterns if item.key == ("peak", 0, 1))
    folds = {
        fold["omitted_subject_key"]: fold for fold in target.loso_folds
    }
    s4_key = hierarchy_module._identifier_sort_key("s4")
    s5_key = hierarchy_module._identifier_sort_key("s5")

    assert target.parent_id is None
    assert target.gate_checks["loso_supporter_stability"] is True
    assert target.gate_checks["loso_effectful_stability"] is False
    assert target.loso_total_fit_subject_count == 5
    assert target.loso_supporter_evaluation_count == 3
    assert target.loso_supporter_pass_count == 3
    _CASE.assertAlmostEqual(target.loso_supporter_stability, 1.0)
    assert target.loso_marginal_only_subject_count == 1
    assert target.loso_no_kind_noop_subject_count == 1
    assert target.loso_effectful_evaluation_count == 4
    assert target.loso_effectful_pass_count == 3
    _CASE.assertAlmostEqual(target.loso_effectful_stability, 0.75)
    _CASE.assertAlmostEqual(target.loso_stability, 0.75)

    assert folds[s4_key]["subject_supports_key"] is False
    assert folds[s4_key]["subject_has_event_kind"] is True
    assert folds[s4_key]["omission_effect"] == "kind_marginals_only"
    assert folds[s4_key]["included_in_supporter_stability"] is False
    assert folds[s4_key]["included_in_effectful_stability"] is True
    assert folds[s4_key]["base_gate_checks"]["npmi"] is False
    _CASE.assertAlmostEqual(folds[s4_key]["npmi"], 0.0)

    assert folds[s5_key]["subject_supports_key"] is False
    assert folds[s5_key]["subject_has_event_kind"] is False
    assert folds[s5_key]["omission_effect"] == "no_kind_noop"
    assert folds[s5_key]["included_in_supporter_stability"] is False
    assert folds[s5_key]["included_in_effectful_stability"] is False
    assert folds[s5_key]["passed"] is True
    _CASE.assertAlmostEqual(folds[s5_key]["npmi"], target.npmi)

    metadata = catalog.selection_metadata
    assert metadata["fit_subject_count"] == 5
    assert metadata["loso_subject_universe"] == (
        "all caller-supplied offline-fit subjects"
    )
    assert metadata["no_kind_subjects_are_explicit_noops"] is True
    assert metadata["no_kind_subjects_in_stability_denominators"] is False
    assert metadata["loso_registration_rule"] == (
        "supporter_stability and effectful_stability must each be at least "
        "minimum_loso_stability"
    )


def test_marginal_passes_cannot_dilute_two_supporter_failures() -> None:
    trials = []
    for subject in ("support-1", "support-2"):
        for repetition in range(2):
            trials.append(
                _tokenized_trial(
                    f"{subject}-target-{repetition}",
                    subject,
                    [0, 1],
                    ["peak"],
                )
            )
    for subject_index in range(8):
        trials.append(
            _tokenized_trial(
                f"background-{subject_index}",
                f"background-{subject_index}",
                [2, 2],
                ["peak"],
            )
        )
    config = ParentGateConfig(
        minimum_occurrences=4,
        minimum_trials=4,
        minimum_subjects=2,
        minimum_npmi=-1.0,
        minimum_mdl_gain_bits=0.0,
        minimum_loso_stability=0.8,
    )

    catalog = fit_parent_catalog(trials, config)
    target = next(item for item in catalog.patterns if item.key == ("peak", 0, 1))

    assert target.parent_id is None
    assert target.loso_total_fit_subject_count == 10
    assert target.loso_supporter_evaluation_count == 2
    assert target.loso_supporter_pass_count == 0
    _CASE.assertAlmostEqual(target.loso_supporter_stability, 0.0)
    assert target.loso_marginal_only_subject_count == 8
    assert target.loso_effectful_evaluation_count == 10
    assert target.loso_effectful_pass_count == 8
    _CASE.assertAlmostEqual(target.loso_effectful_stability, 0.8)
    assert target.loso_no_kind_noop_subject_count == 0
    assert target.gate_checks["loso_supporter_stability"] is False
    assert target.gate_checks["loso_effectful_stability"] is True


def test_negative_parent_control_verifies_fit_hash_and_matches_same_kind_globally() -> None:
    trials = _parent_control_trials()
    config = ParentGateConfig(
        minimum_occurrences=4,
        minimum_trials=4,
        minimum_subjects=2,
        minimum_npmi=-1.0,
        minimum_mdl_gain_bits=0.0,
        minimum_loso_stability=1.0,
    )
    reference = fit_parent_catalog(trials, config)
    assert [item.key for item in reference.registered_patterns] == [("peak", 0, 1)]

    control = fit_matched_random_parent_catalog(
        list(reversed(trials)), reference, config, seed=91
    )

    assert [item.key for item in control.registered_patterns] == [("peak", 2, 3)]
    metadata = control.selection_metadata
    assert metadata["uniform_random_catalog"] is False
    assert metadata["same_event_kind_required"] is True
    assert metadata["reference_fit_data_verified"] is True
    assert metadata["assignment_algorithm"] == "scipy.optimize.linear_sum_assignment"
    assert metadata["matching"][0]["reference_event_kind"] == "peak"
    assert metadata["matching"][0]["negative_event_kind"] == "peak"

    altered = list(trials)
    altered[-1] = _tokenized_trial(
        altered[-1].trial_id,
        altered[-1].subject_id,
        [3, 2],
        ["peak"],
        primitive_num=4,
    )
    with _CASE.assertRaisesRegex(ValueError, "different tokenized trials"):
        fit_matched_random_parent_catalog(altered, reference, config, seed=91)


def test_e3_frequency_only_and_e4_strict_parent_gates_are_distinct() -> None:
    # Four occurrences in one trial satisfy E3's deliberately naive frequency
    # baseline, while the exact same fit data must fail E4 cross-trial and
    # cross-subject support.  This prevents the two experimental arms from
    # silently collapsing to the same implementation.
    one_subject = [
        _tokenized_trial(
            "frequent-one-trial",
            "only-subject",
            [0, 1, 0, 1, 0, 1, 0, 1],
            ["peak"] * 7,
        )
    ]
    frequency = hierarchy_module.fit_frequency_parent_catalog(
        one_subject, minimum_occurrences=4
    )
    strict = fit_parent_catalog(one_subject, _parent_config())
    key = ("peak", 0, 1)
    frequency_pattern = next(item for item in frequency.patterns if item.key == key)
    strict_pattern = next(item for item in strict.patterns if item.key == key)

    assert frequency.catalog_mode == "frequency_only_fit_baseline"
    assert "loso_registration_rule" not in frequency.selection_metadata
    assert frequency_pattern.parent_id is not None
    assert frequency_pattern.gate_checks["frequency_only_registration"] is True
    assert strict.catalog_mode == "gated_fit_only"
    assert strict_pattern.parent_id is None
    assert not strict_pattern.gate_checks["trials"]
    assert not strict_pattern.gate_checks["subjects"]


def test_parent_overlay_never_changes_children_or_fit_catalog() -> None:
    catalog = fit_parent_catalog(_parent_fit_trials(), _parent_config())
    catalog_before = catalog.to_dict()
    matching = _tokenized_trial(
        "eval-match", "eval-s1", [0, 1], ["peak"]
    )
    nonmatching = _tokenized_trial(
        "eval-other", "eval-s2", [2, 0], ["peak"]
    )
    input_tokens_before = matching.primitive_tokens.copy()

    matching_overlay = overlay_parents(matching, catalog, role="held_out")
    nonmatching_overlay = overlay_parents(nonmatching, catalog, role="held_out")

    assert matching_overlay.non_destructive is True
    assert np.array_equal(matching_overlay.child_tokens, input_tokens_before)
    assert np.array_equal(matching.primitive_tokens, input_tokens_before)
    assert len(matching_overlay.parent_occurrences) == 1
    assert nonmatching_overlay.parent_occurrences == ()
    # Held-out/evaluation content can change the overlays, never the fit-only
    # registered catalog or its fingerprint.
    assert catalog.to_dict() == catalog_before
    assert catalog.state_hash() == catalog_before["state_sha256"]


def test_parent_overlay_rejects_codebook_fingerprint_mismatch() -> None:
    catalog = fit_parent_catalog(_parent_fit_trials(), _parent_config())
    wrong_codebook = _tokenized_trial(
        "eval-wrong-codebook",
        "eval-s1",
        [0, 1],
        ["peak"],
        codebook_hash="c" * 64,
    )

    with _CASE.assertRaisesRegex(ValueError, "different codebooks"):
        overlay_parents(wrong_codebook, catalog, role="held_out")


def test_cv_defaults_runner_identity_and_a2_training_command() -> None:
    args = cv_module.build_parser().parse_args(
        [
            "--cv-root",
            "cv-root",
            "--npz-path",
            "windows.npz",
            "--encoder-root",
            "encoders",
            "--output-root",
            "results",
        ]
    )

    assert args.profiles == "A0,A2,A3"
    assert args.child_feature_source == "content_embedding"
    protocol = cv_module._protocol_settings(args)
    assert protocol["e0_segmentation"] == "legacy_npz_windows"
    assert protocol["child_feature_source"] == "content_embedding"

    runner = cv_module._runner_command(
        args,
        Path("A2-checkpoint.pt"),
        Path("one-run"),
        "A2",
        1,
        5,
        ("E0", "E2", "E4"),
    )
    assert _flag_value(runner, "--expected-profile") == "A2"
    assert _flag_value(runner, "--arms") == "E0,E2,E4"
    assert _flag_value(runner, "--child-feature-source") == "content_embedding"

    trainer = cv_module._trainer_command(
        "python",
        Path("source.pt"),
        Path("windows.npz"),
        Path("A2-output"),
        "A2",
        5,
        "cuda",
    )
    assert _flag_value(trainer, "--ablation-profile") == "A2"
    # Keep the common formal raw identity value.  A2 disables InfoNCE through
    # its consistency method, not by passing a profile-specific raw zero.
    assert _flag_value(trainer, "--window-aug-weight") == "1"
    assert _flag_value(trainer, "--content-boundary-alignment-weight") == "0.1"


def load_tests(
    loader: unittest.TestLoader,
    standard_tests: unittest.TestSuite,
    pattern: str | None,
) -> unittest.TestSuite:
    del loader, standard_tests, pattern
    functions = [
        value
        for name, value in sorted(globals().items())
        if name.startswith("test_") and callable(value)
    ]
    return unittest.TestSuite(unittest.FunctionTestCase(item) for item in functions)


if __name__ == "__main__":
    unittest.main()
