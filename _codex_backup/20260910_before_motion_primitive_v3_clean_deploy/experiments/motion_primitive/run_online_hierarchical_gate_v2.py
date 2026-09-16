"""Run the frozen-readout and adaptive K=32-or-34 trajectory experiment.

This is an isolated trajectory-clustering proxy, not the formal Happy-CGCD
online learner.  It compares three interventions while fitting the trial-level
KMeans readout exactly once:

* F0: the frozen K=32 motion-primitive vocabulary;
* F1: the registered residual-to-gravity K=32-or-34 mechanism diagnostic;
* F2: a train-only adaptive append-only K=32-or-34 vocabulary.

The old 32 centres never move or change id.  Activity labels and names are
joined only after codebook selection, routing and raw cluster predictions have
been persisted in a label-free artifact.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.uschad import get_uschad_datasets  # noqa: E402
from experiments.motion_primitive.adaptive_codebook import (  # noqa: E402
    AdaptiveAssignments,
    AdaptiveCodebookConfig,
    AdaptiveOccurrence,
    adaptive_codebook_diagnostics,
    adaptive_duration_histograms,
    adaptive_state_by_trial,
    assign_adaptive_codebook,
    audit_codebook_invariants,
    build_adaptive_occurrence,
    codebook_center_hash,
    fit_adaptive_codebook,
)
from experiments.motion_primitive.core import run_length_encode  # noqa: E402
from experiments.motion_primitive.frozen_hierarchical_readout import (  # noqa: E402
    apply_frozen_leaf_overrides,
    fit_frozen_coarse_readout,
    frozen_readout_diagnostics,
)
from experiments.motion_primitive.hierarchical_gate import (  # noqa: E402
    GATE_STATES,
    assign_hierarchical_gate,
    fit_hierarchical_gate,
    hierarchical_duration_histograms,
    hierarchical_gate_diagnostics,
)
from experiments.motion_primitive.hierarchical_gate_v2 import (  # noqa: E402
    assign_hierarchical_gate_v2,
    fit_hierarchical_gate_v2,
    hierarchical_gate_v2_diagnostics,
)
from experiments.motion_primitive.online_secondary_codebook import (  # noqa: E402
    class_mean_histogram_distances,
    duration_histograms,
    global_hungarian_metrics,
    select_dominant_token,
    validate_session_manifest,
)
from experiments.motion_primitive.run_online_hierarchical_gate import (  # noqa: E402
    _build_motion_components,
    _posthoc_gate_sit_stand,
    _validate_registered_upstream,
)
from experiments.motion_primitive.run_online_secondary_codebook import (  # noqa: E402
    LabelFreeSourceSignalRepository,
    _paired_bootstrap_comparisons,
    _positive_int,
    _require_new_output_dir,
    _trial_ids,
    _truth_maps,
    _write_csv,
    _union_trial_ids,
    _union_trial_labels,
    _union_trial_subjects,
    build_token_occurrences,
    build_trial_token_durations,
    load_segment_artifacts,
)
from experiments.motion_primitive.trajectory_ablation import (  # noqa: E402
    SourceSignalRepository,
    jsonable,
    sha256_file,
)


ARMS = (
    "F0_frozen_K32",
    "F1_registered_K32_or_K34",
    "F2_adaptive_K32_or_K34",
)

# Canonical seven-fold outer subject split used by the existing Happy-CGCD
# checkpoints.  The fold id is not trusted merely because it appears in a
# directory name: every downstream run is checked against these metadata
# partitions before the Session-2 stream is reconstructed.
CANONICAL_SUBJECT_IDS = tuple(range(1, 15))
CANONICAL_EVAL_SUBJECTS_BY_FOLD = {
    1: (10, 11),
    2: (2, 13),
    3: (3, 9),
    4: (1, 7),
    5: (8, 12),
    6: (4, 5),
    7: (6, 14),
}
CANONICAL_VALIDATION_SUBJECTS_BY_FOLD = {
    1: (2, 13),
    2: (3, 9),
    3: (1, 7),
    4: (8, 12),
    5: (4, 5),
    6: (6, 14),
    7: (10, 11),
}
V2_OLD_CLASS_IDS = tuple(range(6))
V2_NOVEL_CLASS_ORDER = tuple(range(6, 12))
V2_SESSION2_ACTIVITY_IDS = tuple(range(10))
V2_SESSION_COUNTS = {
    "session_1_incremental_train_count": 22,
    "session_2_incremental_train_count": 26,
    "session_2_cumulative_train_count": 48,
    "session_2_test_count": 52,
}

PROTOCOL = {
    "name": "motion_primitive_adaptive_codebook_frozen_readout_v2",
    "scope": "isolated_session2_unlabelled_trajectory_clustering_proxy",
    "is_formal_happy_cgcd": False,
    "segmentation": "fixed_window_only",
    "primary_question": (
        "Can a frozen old-class K=32 primitive vocabulary be expanded only when "
        "a parent token contains a stable, cross-subject two-mode structure?"
    ),
    "arms": {
        ARMS[0]: "frozen K=32 vocabulary and the single shared coarse readout",
        ARMS[1]: (
            "registered residual-to-gravity q+2 mechanism reference; exploratory "
            "because the separate v2 safety audit may reject its 12:1 residual split"
        ),
        ARMS[2]: (
            "fully train-only adaptive parent selection and append-only K in {32,34}; "
            "low-confidence occurrences retain their parent token"
        ),
    },
    "capacity_policy": {
        "base_centres_frozen": True,
        "old_token_ids_frozen": True,
        "children_per_accepted_parent": 2,
        "maximum_expanded_parents_per_session": 1,
        "K_total": [32, 34],
    },
    "readout_policy": (
        "fit the K=10 trial-level coarse KMeans once on cumulative-online-train "
        "K32 histograms; refined arms copy its raw test predictions and replace "
        "only explicitly routed trials with appended leaf cluster ids 10 and 11"
    ),
    "important_id_boundary": (
        "primitive child token ids 32/33 and trial-level leaf cluster ids 10/11 "
        "belong to different namespaces"
    ),
    "primary_comparison": "F2_adaptive_K32_or_K34 minus F0_frozen_K32",
    "registered_parent_ineligibility_policy": (
        "If no parent token reaches the pre-registered dominant-duration support, "
        "do not relax the threshold: disable F1 expansion and reproduce F0 exactly."
    ),
}


def _array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(np.asarray(value))
    digest = hashlib.sha256()
    digest.update(
        json.dumps(
            {"shape": list(array.shape), "dtype": array.dtype.str},
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    )
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _exact_integer(value, context: str) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise RuntimeError(f"{context} must be an integer, not a boolean.")
    try:
        numeric = float(value)
    except (TypeError, ValueError) as error:
        raise RuntimeError(f"{context} must be an integer.") from error
    if not np.isfinite(numeric) or not numeric.is_integer():
        raise RuntimeError(f"{context} must be an integer.")
    return int(numeric)


def _unique_integer_list(mapping: Mapping, key: str, context: str) -> list[int]:
    raw = mapping.get(key)
    if not isinstance(raw, list) or not raw:
        raise RuntimeError(f"{context}.{key} must be a non-empty list.")
    values = [_exact_integer(value, f"{context}.{key}") for value in raw]
    if len(values) != len(set(values)):
        raise RuntimeError(f"{context}.{key} contains duplicate ids: {values}.")
    return sorted(values)


def _canonical_partition(fold: int) -> dict[str, list[int]]:
    fold = int(fold)
    if fold not in CANONICAL_EVAL_SUBJECTS_BY_FOLD:
        raise RuntimeError(
            "Adaptive-codebook v2 accepts only the registered seven outer folds; "
            f"observed fold={fold}."
        )
    evaluation = sorted(CANONICAL_EVAL_SUBJECTS_BY_FOLD[fold])
    validation = sorted(CANONICAL_VALIDATION_SUBJECTS_BY_FOLD[fold])
    fitting = sorted(set(CANONICAL_SUBJECT_IDS) - set(evaluation) - set(validation))
    return {"fit": fitting, "validation": validation, "evaluation": evaluation}


def _validate_v2_protocol_metadata(
    config: Mapping,
    split: Mapping,
    *,
    seed: int,
    expected_fold: int | None = None,
) -> dict:
    """Validate one formal upstream run without using historical fold06 guards.

    This is deliberately local to the v2 experiment.  The earlier fold06
    utilities retain their original fail-closed behaviour and therefore remain
    byte-for-byte compatible with the historical experiments.
    """

    metadata = config.get("checkpoint_metadata")
    arguments = config.get("arguments")
    if not isinstance(metadata, Mapping) or not isinstance(arguments, Mapping):
        raise RuntimeError("Upstream run lacks checkpoint_metadata or arguments.")
    if config.get("checkpoint_type") != "motion_primitive_encoder":
        raise RuntimeError("Adaptive-codebook v2 requires a motion_primitive_encoder run.")
    if _exact_integer(
        config.get("checkpoint_schema_version"), "checkpoint_schema_version"
    ) != 1:
        raise RuntimeError("Adaptive-codebook v2 requires checkpoint schema version 1.")
    if metadata.get("smoke_test") is not False:
        raise RuntimeError("Adaptive-codebook v2 rejects smoke-test encoders.")
    if metadata.get("outer_test_used_during_encoder_training") is not False:
        raise RuntimeError("Outer-test data entered motion-encoder training.")
    if metadata.get("uschad_recompute_norm_from_train_subjects") is not True:
        raise RuntimeError("Fold-train-only USC-HAD normalization is required.")

    fold = _exact_integer(metadata.get("uschad_cv_fold"), "uschad_cv_fold")
    canonical = _canonical_partition(fold)
    if expected_fold is not None and fold != int(expected_fold):
        raise RuntimeError(
            f"Explicit --fold={int(expected_fold)} disagrees with upstream fold={fold}."
        )
    requested_seed = int(seed)
    recorded_seed = _exact_integer(arguments.get("seed"), "arguments.seed")
    if recorded_seed != requested_seed:
        raise RuntimeError(
            "Session sampling seed must equal the upstream primitive-run seed: "
            f"{requested_seed} != {recorded_seed}."
        )
    for field in ("motion_encoder_seed", "seed"):
        if field not in metadata:
            raise RuntimeError(f"checkpoint_metadata.{field} is required.")
        value = _exact_integer(metadata[field], f"checkpoint_metadata.{field}")
        if value != requested_seed:
            raise RuntimeError(
                f"checkpoint_metadata.{field}={value} disagrees with seed={requested_seed}."
            )

    if _exact_integer(metadata.get("old_class_count"), "metadata old_class_count") != len(
        V2_OLD_CLASS_IDS
    ):
        raise RuntimeError("Motion encoder does not use the registered six old classes.")
    if _exact_integer(arguments.get("old_class_count"), "arguments old_class_count") != len(
        V2_OLD_CLASS_IDS
    ):
        raise RuntimeError("Primitive run does not use the registered six old classes.")
    old_ids = _unique_integer_list(split, "old_class_ids_0based", "split_audit")
    if old_ids != list(V2_OLD_CLASS_IDS):
        raise RuntimeError(
            f"split_audit old classes drifted: {old_ids} != {list(V2_OLD_CLASS_IDS)}."
        )

    fit_subjects = _unique_integer_list(split, "fit_subjects", "split_audit")
    eval_subjects = _unique_integer_list(split, "eval_subjects", "split_audit")
    metadata_fit = _unique_integer_list(
        metadata, "uschad_train_subjects", "checkpoint_metadata"
    )
    metadata_eval = _unique_integer_list(
        metadata, "uschad_test_subjects", "checkpoint_metadata"
    )
    validation_subjects = _unique_integer_list(
        metadata, "offline_val_subjects", "checkpoint_metadata"
    )
    if fit_subjects != metadata_fit or eval_subjects != metadata_eval:
        raise RuntimeError(
            "Primitive split subjects disagree with motion-checkpoint metadata: "
            f"fit={fit_subjects}/{metadata_fit}, eval={eval_subjects}/{metadata_eval}."
        )
    observed = {
        "fit": fit_subjects,
        "validation": validation_subjects,
        "evaluation": eval_subjects,
    }
    if observed != canonical:
        raise RuntimeError(
            f"Fold {fold} subject partition drifted: observed={observed}, "
            f"expected={canonical}."
        )
    if (
        set(fit_subjects) & set(validation_subjects)
        or set(fit_subjects) & set(eval_subjects)
        or set(validation_subjects) & set(eval_subjects)
    ):
        raise RuntimeError("Fit/validation/evaluation subjects are not disjoint.")
    if sorted(fit_subjects + validation_subjects + eval_subjects) != list(
        CANONICAL_SUBJECT_IDS
    ):
        raise RuntimeError("The fold partition does not cover each USC-HAD subject once.")

    safety_flags = {
        "arguments.allow_split_override": arguments.get("allow_split_override"),
        "arguments.allow_unverified_npz_normalization": arguments.get(
            "allow_unverified_npz_normalization"
        ),
        "split.checkpoint_split_override_used": split.get(
            "checkpoint_split_override_used"
        ),
        "split.checkpoint_split_override_explicitly_allowed": split.get(
            "checkpoint_split_override_explicitly_allowed"
        ),
        "split.unverified_npz_normalization_explicitly_allowed": split.get(
            "unverified_npz_normalization_explicitly_allowed"
        ),
    }
    unsafe = [name for name, value in safety_flags.items() if value is not False]
    if unsafe:
        raise RuntimeError(f"Unsafe upstream split/normalization flags: {unsafe}.")
    if arguments.get("anomaly_policy") != "report" or split.get("anomaly_policy") != "report":
        raise RuntimeError(
            "The canonical multi-fold v2 grid requires anomaly_policy='report'; "
            "an exclusion sensitivity run needs a separately defined manifest."
        )
    return {
        "fold": fold,
        "seed": requested_seed,
        "fit_subjects": fit_subjects,
        "validation_subjects": validation_subjects,
        "eval_subjects": eval_subjects,
        "canonical_partition_verified": True,
    }


def _read_json_object(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"Expected a JSON object: {path}.")
    return value


def _validate_v2_run(
    run_dir: Path,
    npz_path: Path,
    *,
    seed: int,
    expected_fold: int | None = None,
) -> tuple[dict, dict, Path, Path, dict]:
    required = {
        "config": Path(run_dir) / "experiment_config.json",
        "split": Path(run_dir) / "split_audit.json",
        "windows": Path(run_dir) / "window_embeddings_and_tokens.npz",
        "codebook": Path(run_dir) / "primitive_codebook.npz",
    }
    missing = [str(path) for path in required.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Upstream run lacks required files: {missing}.")
    config = _read_json_object(required["config"])
    split = _read_json_object(required["split"])
    identity = _validate_v2_protocol_metadata(
        config, split, seed=int(seed), expected_fold=expected_fold
    )
    actual_npz_hash = sha256_file(Path(npz_path))
    recorded_npz_hash = str(config.get("npz_sha256", ""))
    if recorded_npz_hash != actual_npz_hash:
        raise RuntimeError(
            "Source NPZ SHA256 differs from the upstream primitive run: "
            f"{actual_npz_hash} != {recorded_npz_hash}."
        )
    return (
        config,
        split,
        required["windows"],
        required["codebook"],
        identity,
    )


def _validate_session2_protocol_v2(
    audit: dict,
    *,
    fold: int,
    novel_order: Sequence[int],
    labels_by_feature_trial: Mapping[int, int],
    subjects_by_feature_trial: Mapping[int, int],
    expected_eval_subjects: Sequence[int],
) -> dict:
    observed_order = tuple(int(value) for value in novel_order)
    if observed_order != V2_NOVEL_CLASS_ORDER:
        raise RuntimeError(
            "Registered novel-class order drifted: "
            f"{list(observed_order)} != {list(V2_NOVEL_CLASS_ORDER)}."
        )
    observed_counts = {key: int(audit.get(key, -1)) for key in V2_SESSION_COUNTS}
    if observed_counts != V2_SESSION_COUNTS:
        raise RuntimeError(
            f"Fold {fold} Session-2 trial counts drifted: "
            f"{observed_counts} != {V2_SESSION_COUNTS}."
        )
    expected_feature_ids = set(
        int(value) for value in audit["cumulative_train_trial_ids"]
    ) | set(int(value) for value in audit["session_2_test_trial_ids"])
    label_ids = set(int(value) for value in labels_by_feature_trial)
    subject_ids = set(int(value) for value in subjects_by_feature_trial)
    if label_ids != expected_feature_ids or subject_ids != expected_feature_ids:
        raise RuntimeError(
            "Session protocol metadata and feature-trial manifest disagree: "
            f"label_missing={sorted(expected_feature_ids - label_ids)}, "
            f"label_extra={sorted(label_ids - expected_feature_ids)}, "
            f"subject_missing={sorted(expected_feature_ids - subject_ids)}, "
            f"subject_extra={sorted(subject_ids - expected_feature_ids)}."
        )
    expected_subjects = sorted(int(value) for value in expected_eval_subjects)
    observed_subjects = sorted(
        set(int(value) for value in subjects_by_feature_trial.values())
    )
    if observed_subjects != expected_subjects:
        raise RuntimeError(
            f"Fold {fold} Session-2 subjects drifted: "
            f"{observed_subjects} != {expected_subjects}."
        )
    observed_activities = sorted(
        set(int(value) for value in labels_by_feature_trial.values())
    )
    if observed_activities != list(V2_SESSION2_ACTIVITY_IDS):
        raise RuntimeError(
            f"Fold {fold} Session-2 activities drifted: "
            f"{observed_activities} != {list(V2_SESSION2_ACTIVITY_IDS)}."
        )
    future_trials = sorted(
        trial_id
        for trial_id, label in labels_by_feature_trial.items()
        if int(label) not in V2_SESSION2_ACTIVITY_IDS
    )
    if future_trials:
        raise RuntimeError(f"Future-class trials entered Session 2: {future_trials}.")
    for subject in expected_subjects:
        activities = sorted(
            {
                int(labels_by_feature_trial[trial_id])
                for trial_id, trial_subject in subjects_by_feature_trial.items()
                if int(trial_subject) == int(subject)
            }
        )
        if activities != list(V2_SESSION2_ACTIVITY_IDS):
            raise RuntimeError(
                f"Subject {subject} lacks complete Session-2 class support: {activities}."
            )
    audit.update(
        {
            "builder": "data.uschad.get_uschad_datasets/v2_multifold",
            "evaluated_session": 2,
            "uschad_cv_fold": int(fold),
            "registered_protocol_verified": True,
            "registered_novel_class_order": list(V2_NOVEL_CLASS_ORDER),
            "allowed_session_2_activity_ids": list(V2_SESSION2_ACTIVITY_IDS),
            "feature_trial_count": len(expected_feature_ids),
            "future_feature_trial_ids": [],
            "future_feature_trial_count": 0,
            "feature_subject_ids": observed_subjects,
            "subject_by_feature_trial_protocol_only": {
                str(trial_id): int(subjects_by_feature_trial[trial_id])
                for trial_id in sorted(expected_feature_ids)
            },
        }
    )
    return audit


def build_session2_manifest_v2(
    npz_path: Path,
    *,
    fold: int,
    fit_subjects: Sequence[int],
    eval_subjects: Sequence[int],
    validation_subjects: Sequence[int],
    seed: int,
    window_size_samples: int = 256,
) -> dict:
    """Reconstruct the canonical Session-2 stream for any registered fold.

    The label-aware loader datasets remain local to this function.  Only the
    audited trial ids and protocol-only subject map cross the return boundary.
    """

    loader_args = SimpleNamespace(
        uschad_npz_path=str(Path(npz_path).resolve()),
        num_novel_class_per_session=2,
        num_novel_classes_per_session=2,
        uschad_sample_unit="trial",
        n_views=2,
        trial_view_mode="full_random_crop",
        trial_crop_ratio=2.0 / 3.0,
        trial_min_windows=2,
        har_aug_mode="weak_strong",
        har_weak_jitter_std=0.0,
        har_weak_scale_std=0.1,
        har_strong_jitter_std=0.0,
        har_strong_scale_std=0.2,
        har_time_mask_ratio=0.0,
        uschad_split_mode="subject",
        uschad_train_subjects=",".join(str(int(value)) for value in fit_subjects),
        uschad_test_subjects=",".join(str(int(value)) for value in eval_subjects),
        offline_val_subjects=",".join(
            str(int(value)) for value in validation_subjects
        ),
        uschad_recompute_norm_from_train_subjects=True,
        uschad_norm_eps=1e-6,
        har_in_channels=6,
        uschad_window_size=int(window_size_samples),
        logger=None,
    )
    loader_config = {
        "continual_session_num": 3,
        "online_novel_unseen_num": 5,
        "online_old_seen_num": 2,
        "online_novel_seen_num": 2,
        "sample_unit": "trial",
    }
    datasets, novel_order = get_uschad_datasets(
        train_transform=None,
        test_transform=None,
        config_dict=loader_config,
        train_classes=range(len(V2_OLD_CLASS_IDS)),
        prop_train_labels=0.8,
        split_train_val=False,
        is_shuffle=False,
        seed=int(seed),
        args=loader_args,
    )
    old_sessions = datasets["online_old_dataset_unlabelled_list"]
    novel_sessions = datasets["online_novel_dataset_unlabelled_list"]
    test_sessions = datasets["online_test_dataset_list"]
    session_one = _union_trial_ids(old_sessions[0], novel_sessions[0])
    session_two = _union_trial_ids(old_sessions[1], novel_sessions[1])
    session_two_test = _trial_ids(test_sessions[1])
    labels = _union_trial_labels(
        old_sessions[0],
        novel_sessions[0],
        old_sessions[1],
        novel_sessions[1],
        test_sessions[1],
    )
    subjects = _union_trial_subjects(
        old_sessions[0],
        novel_sessions[0],
        old_sessions[1],
        novel_sessions[1],
        test_sessions[1],
    )
    audit = validate_session_manifest(session_one, session_two, session_two_test)
    audit.update(
        {
            "seed": int(seed),
            "novel_class_order_protocol_only": np.asarray(
                novel_order, dtype=np.int64
            ).tolist(),
            "fit_subjects": sorted(int(value) for value in fit_subjects),
            "eval_subjects": sorted(int(value) for value in eval_subjects),
            "validation_subjects": sorted(
                int(value) for value in validation_subjects
            ),
            "session_1_incremental_train_count": len(session_one),
            "session_2_incremental_train_count": len(session_two),
            "session_2_cumulative_train_count": len(
                audit["cumulative_train_trial_ids"]
            ),
            "session_2_test_count": len(session_two_test),
        }
    )
    _validate_session2_protocol_v2(
        audit,
        fold=int(fold),
        novel_order=np.asarray(novel_order, dtype=np.int64).tolist(),
        labels_by_feature_trial=labels,
        subjects_by_feature_trial=subjects,
        expected_eval_subjects=eval_subjects,
    )
    return audit


def _state_map(assignments) -> dict[int, int]:
    return {
        int(trial_id): int(state)
        for trial_id, state in zip(
            assignments.trial_ids.tolist(), assignments.gate_states.tolist()
        )
    }


def _subject_map_from_manifest(
    manifest: Mapping, required_trial_ids: Sequence[int]
) -> dict[int, int]:
    raw = manifest.get("subject_by_feature_trial_protocol_only")
    if not isinstance(raw, Mapping):
        raise RuntimeError("Session manifest lacks the protocol-only subject map.")
    result = {int(key): int(value) for key, value in raw.items()}
    missing = sorted(set(int(value) for value in required_trial_ids) - set(result))
    if missing:
        raise RuntimeError(f"Session subject map lacks trials: {missing}.")
    return result


def _candidate_parent_ids(
    trial_ids: Sequence[int],
    trial_token_durations: Mapping[int, Mapping[int, float]],
    minimum_fraction: float,
) -> dict[int, list[int]]:
    result: dict[int, list[int]] = {}
    for trial_id_value in trial_ids:
        trial_id = int(trial_id_value)
        durations = trial_token_durations[trial_id]
        total = float(sum(float(value) for value in durations.values()))
        if not np.isfinite(total) or total <= 0:
            raise ValueError(f"Trial {trial_id} has no positive token duration.")
        for token_value, duration_value in durations.items():
            if float(duration_value) / total + 1e-12 >= float(minimum_fraction):
                result.setdefault(int(token_value), []).append(trial_id)
    return {token: sorted(ids) for token, ids in sorted(result.items())}


def _build_adaptive_training_occurrences(
    artifacts,
    source: LabelFreeSourceSignalRepository,
    train_ids: Sequence[int],
    trial_token_durations: Mapping[int, Mapping[int, float]],
    subjects_by_trial: Mapping[int, int],
    minimum_fraction: float,
) -> list[AdaptiveOccurrence]:
    occurrences: list[AdaptiveOccurrence] = []
    for token, token_trial_ids in _candidate_parent_ids(
        train_ids, trial_token_durations, minimum_fraction
    ).items():
        unlabelled = build_token_occurrences(
            artifacts, source, token_trial_ids, selected_token=int(token)
        )
        motion = _build_motion_components(
            artifacts, source, token_trial_ids, selected_token=int(token)
        )
        if set(unlabelled) != set(token_trial_ids) or set(motion) != set(token_trial_ids):
            raise RuntimeError("Adaptive occurrence construction dropped a train trial.")
        for trial_id in token_trial_ids:
            occurrences.append(
                build_adaptive_occurrence(
                    unlabelled[trial_id],
                    subject_id=subjects_by_trial[trial_id],
                    motion_components=motion[trial_id],
                )
            )
    if not occurrences:
        raise RuntimeError("No parent-token occurrence reached the minimum duration.")
    return occurrences


def _build_selected_adaptive_occurrences(
    artifacts,
    source: LabelFreeSourceSignalRepository,
    trial_ids: Sequence[int],
    subjects_by_trial: Mapping[int, int],
    selected_parent: int | None,
) -> list[AdaptiveOccurrence]:
    if selected_parent is None:
        return []
    unlabelled = build_token_occurrences(
        artifacts, source, trial_ids, selected_token=int(selected_parent)
    )
    selected_ids = sorted(int(value) for value in unlabelled)
    if not selected_ids:
        return []
    motion = _build_motion_components(
        artifacts, source, selected_ids, selected_token=int(selected_parent)
    )
    return [
        build_adaptive_occurrence(
            unlabelled[trial_id],
            subject_id=subjects_by_trial[trial_id],
            motion_components=motion[trial_id],
        )
        for trial_id in selected_ids
    ]


def _empty_adaptive_assignments() -> AdaptiveAssignments:
    return AdaptiveAssignments(
        trial_ids=np.asarray([], dtype=np.int64),
        parent_tokens=np.asarray([], dtype=np.int64),
        output_tokens=np.asarray([], dtype=np.int64),
        child_ids=np.asarray([], dtype=np.int64),
        assigned_child_distances=np.asarray([], dtype=np.float64),
        routed_to_child=np.asarray([], dtype=bool),
    )


REGISTERED_NO_PARENT_REASON = "no_parent_met_registered_dominant_trial_support"


def _select_registered_candidate_or_fallback(
    trial_token_durations: Mapping[int, Mapping[int, float]],
    *,
    minimum_fraction: float,
    minimum_support: int,
) -> dict:
    """Select the registered F1 parent or record a legitimate K32 fallback.

    The historical selector intentionally raises when no token meets its
    support threshold.  Across a full subject-CV grid that outcome is not data
    corruption: it means the registered intervention is ineligible in this
    run.  The threshold is kept unchanged and the arm must remain exactly F0.
    """

    try:
        selected = select_dominant_token(
            trial_token_durations,
            minimum_fraction=float(minimum_fraction),
            minimum_support=int(minimum_support),
        )
    except RuntimeError as error:
        if not str(error).startswith(
            "No coarse token reached the registered dominant-trial support:"
        ):
            raise
        try:
            observed = select_dominant_token(
                trial_token_durations,
                minimum_fraction=float(minimum_fraction),
                minimum_support=1,
            )
        except RuntimeError as observed_error:
            if not str(observed_error).startswith(
                "No coarse token reached the registered dominant-trial support:"
            ):
                raise
            observed = {
                "selected_token": None,
                "selected_support": 0,
                "support_by_token": {},
                "dominant_trial_ids": [],
            }
        return {
            "status": "no_eligible_parent_fallback_to_f0",
            "eligible": False,
            "selected_token": None,
            "selected_support": None,
            "minimum_fraction": float(minimum_fraction),
            "minimum_support": int(minimum_support),
            "support_by_token": dict(observed["support_by_token"]),
            "maximum_observed_support": int(observed["selected_support"]),
            "best_observed_token": observed["selected_token"],
            "best_observed_dominant_trial_ids": list(
                observed["dominant_trial_ids"]
            ),
            "dominant_trial_ids": [],
            "fallback_reason": REGISTERED_NO_PARENT_REASON,
            "threshold_was_lowered": False,
            "selection_uses_labels": False,
        }
    result = dict(selected)
    result.update(
        {
            "status": "eligible_parent_selected",
            "eligible": True,
            "maximum_observed_support": int(selected["selected_support"]),
            "best_observed_token": int(selected["selected_token"]),
            "best_observed_dominant_trial_ids": list(
                selected["dominant_trial_ids"]
            ),
            "fallback_reason": None,
            "threshold_was_lowered": False,
        }
    )
    return result


def _registered_no_parent_audits(candidate: Mapping, args: argparse.Namespace) -> tuple[dict, dict]:
    """Return explicit label-free diagnostics for an ineligible F1 parent."""

    common = {
        "coarse_token": None,
        "gate_enabled": False,
        "gate_disable_reason": REGISTERED_NO_PARENT_REASON,
        "fit_trial_count": 0,
        "fit_trial_ids": [],
        "fit_gate_state_counts": [0, 0, 0],
        "gate_state_meanings": {str(key): value for key, value in GATE_STATES.items()},
    }
    operational = {
        **common,
        "static_radius_quantile": float(args.static_radius_quantile),
        "minimum_token_fraction": float(args.minimum_token_fraction),
        "minimum_motion_energy_ratio": float(args.minimum_motion_energy_ratio),
        "minimum_motion_energy_gap": float(args.minimum_motion_energy_gap),
        "registered_parent_minimum_support": int(args.registered_minimum_support),
        "maximum_observed_parent_support": int(
            candidate["maximum_observed_support"]
        ),
        "confident_static_fit_trial_ids": [],
        "confident_static_fit_count": 0,
        "fit_uses_activity_labels": False,
        "static_child_naming_uses_train_only_raw_motion": True,
        "test_motion_energy_used_for_assignment": False,
    }
    safety = {
        **common,
        "gate_disable_reasons": [REGISTERED_NO_PARENT_REASON],
        "support_audit": {
            "status": "not_run_no_eligible_registered_parent",
            "parent_selection_passed": False,
            "minimum_parent_support": int(args.registered_minimum_support),
            "maximum_observed_parent_support": int(
                candidate["maximum_observed_support"]
            ),
            "support_by_token": dict(candidate["support_by_token"]),
            "threshold_was_lowered": False,
            "uses_only_train_candidates": True,
        },
        "leave_one_out_audit": {
            "status": "not_run_no_eligible_registered_parent",
            "passed": False,
            "replicate_count": 0,
        },
        "fit_uses_activity_labels_or_names": False,
        "all_v2_thresholds_are_train_only": True,
    }
    return operational, safety


def _actual_segment_tokens(
    artifacts,
    trial_id: int,
    selected_parent: int | None,
    state_by_trial: Mapping[int, int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    positions = np.flatnonzero(artifacts.trial_ids == int(trial_id))
    if len(positions) == 0:
        raise RuntimeError(f"Trial {trial_id} is absent from segment artifacts.")
    positions = positions[np.argsort(artifacts.starts[positions], kind="stable")]
    base_tokens = artifacts.tokens[positions].astype(np.int64, copy=True)
    tokens = base_tokens.copy()
    if selected_parent is not None and np.any(base_tokens == int(selected_parent)):
        if int(trial_id) not in state_by_trial:
            raise RuntimeError(
                f"Trial {trial_id} contains selected parent {selected_parent} "
                "but has no frozen route state."
            )
        state = int(state_by_trial[int(trial_id)])
        if state not in GATE_STATES:
            raise RuntimeError(f"Trial {trial_id} has invalid route state {state}.")
        if state > 0:
            tokens[base_tokens == int(selected_parent)] = (
                len(artifacts.centers) + state - 1
            )
    return tokens, artifacts.starts[positions], artifacts.ends[positions]


def _trajectory_invariant_audit(
    artifacts,
    trial_ids: Sequence[int],
    selected_parent: int | None,
    state_by_trial: Mapping[int, int],
) -> dict:
    routed_ids: list[int] = []
    fallback_ids: list[int] = []
    changed_ids: list[int] = []
    for trial_id_value in trial_ids:
        trial_id = int(trial_id_value)
        positions = np.flatnonzero(artifacts.trial_ids == trial_id)
        positions = positions[np.argsort(artifacts.starts[positions], kind="stable")]
        base = artifacts.tokens[positions].astype(np.int64)
        refined, _, _ = _actual_segment_tokens(
            artifacts, trial_id, selected_parent, state_by_trial
        )
        state = int(state_by_trial.get(trial_id, 0))
        changed = refined != base
        if state == 0:
            fallback_ids.append(trial_id)
            if np.any(changed):
                raise RuntimeError("A fallback trajectory changed.")
        else:
            routed_ids.append(trial_id)
            if selected_parent is None or not np.any(changed):
                raise RuntimeError("A routed trajectory did not change its parent token.")
            if np.any(base[changed] != int(selected_parent)):
                raise RuntimeError("A routed trajectory changed a non-parent token.")
            expected = len(artifacts.centers) + state - 1
            if np.any(refined[changed] != expected):
                raise RuntimeError("A routed trajectory used a non-appended child id.")
        if np.any(changed):
            changed_ids.append(trial_id)
    if sorted(routed_ids) != sorted(changed_ids):
        raise RuntimeError("Exactly the routed trajectories must change.")
    return {
        "routed_trial_count": len(routed_ids),
        "fallback_trial_count": len(fallback_ids),
        "changed_trial_count": len(changed_ids),
        "routed_trial_ids": routed_ids,
        "fallback_trial_ids": fallback_ids,
        "changed_trial_ids": changed_ids,
        "fallback_trajectories_exactly_equal": True,
        "only_selected_parent_positions_changed": True,
    }


def _h_score(old_accuracy: float | None, new_accuracy: float | None) -> float | None:
    if old_accuracy is None or new_accuracy is None:
        return None
    denominator = float(old_accuracy) + float(new_accuracy)
    return (
        2.0 * float(old_accuracy) * float(new_accuracy) / denominator
        if denominator > 0
        else 0.0
    )


def _complete_clustering_metrics(
    y_true: np.ndarray, raw_predictions: np.ndarray
) -> dict:
    metrics = global_hungarian_metrics(
        y_true, raw_predictions, old_class_count=6
    )
    metrics["h_score"] = _h_score(
        metrics["old_accuracy"], metrics["new_accuracy"]
    )
    metrics["adjusted_rand_index"] = float(
        adjusted_rand_score(y_true, raw_predictions)
    )
    metrics["normalized_mutual_information"] = float(
        normalized_mutual_info_score(y_true, raw_predictions)
    )
    return metrics


def _attribution_audit(
    trial_ids: Sequence[int],
    y_true: np.ndarray,
    coarse_raw: np.ndarray,
    refined_raw: np.ndarray,
    coarse_aligned: np.ndarray,
    refined_aligned: np.ndarray,
    state_by_trial: Mapping[int, int],
) -> dict:
    states = np.asarray(
        [int(state_by_trial.get(int(trial_id), 0)) for trial_id in trial_ids],
        dtype=np.int64,
    )
    routed = states > 0
    fallback = ~routed
    raw_changed = refined_raw != coarse_raw
    if np.any(raw_changed & fallback):
        raise RuntimeError("A fallback raw cluster prediction changed.")
    coarse_correct = coarse_aligned == y_true
    refined_correct = refined_aligned == y_true
    gains = routed & ~coarse_correct & refined_correct
    losses = routed & coarse_correct & ~refined_correct
    alignment_only = fallback & (coarse_aligned != refined_aligned)
    return {
        "routed_trial_count": int(np.sum(routed)),
        "fallback_trial_count": int(np.sum(fallback)),
        "changed_raw_prediction_count": int(np.sum(raw_changed)),
        "fallback_raw_mismatch_count": int(np.sum(raw_changed & fallback)),
        "fallback_alignment_only_changed_count": int(np.sum(alignment_only)),
        "fallback_alignment_only_changed_trial_ids": np.asarray(trial_ids)[
            alignment_only
        ].astype(int).tolist(),
        "routed_correct_gain_count": int(np.sum(gains)),
        "routed_correct_loss_count": int(np.sum(losses)),
        "routed_correct_net_change": int(np.sum(gains) - np.sum(losses)),
        "routed_trial_ids": np.asarray(trial_ids)[routed].astype(int).tolist(),
        "raw_prediction_change_is_routed_only": True,
    }


def _save_trajectory_plot(
    path: Path,
    artifacts,
    test_ids: Sequence[int],
    parents_by_arm: Mapping[str, int | None],
    states_by_arm: Mapping[str, Mapping[int, int]],
    labels: Mapping[int, int],
    subjects: Mapping[int, int],
    names: Mapping[int, str],
) -> None:
    ordered = sorted(
        (int(value) for value in test_ids),
        key=lambda trial_id: (labels[trial_id], subjects[trial_id], trial_id),
    )
    fig, axes = plt.subplots(
        len(ARMS),
        1,
        figsize=(19, max(13.0, 0.22 * len(ordered) * len(ARMS))),
        sharex=True,
    )
    palette = plt.get_cmap("tab20")
    child_colors = {len(artifacts.centers): "black", len(artifacts.centers) + 1: "magenta"}
    for axis, arm in zip(np.atleast_1d(axes), ARMS):
        for row, trial_id in enumerate(ordered):
            tokens, starts, ends = _actual_segment_tokens(
                artifacts,
                trial_id,
                parents_by_arm[arm],
                states_by_arm[arm],
            )
            for token, start, end in zip(tokens, starts, ends):
                axis.barh(
                    row,
                    (int(end) - int(start)) / 100.0,
                    left=int(start) / 100.0,
                    height=0.82,
                    color=child_colors.get(
                        int(token), palette((int(token) % 20) / 19.0)
                    ),
                    linewidth=0,
                )
        axis.set_title(
            f"{arm}; parent={parents_by_arm[arm]}; "
            f"routed={sum(int(v) > 0 for v in states_by_arm[arm].values())}"
        )
        axis.set_yticks(np.arange(len(ordered)))
        axis.set_yticklabels(
            [
                f"{names[t]} | S{subjects[t]} | T{t}"
                for t in ordered
            ],
            fontsize=5,
        )
        axis.set_ylabel("test trial")
        axis.invert_yaxis()
    np.atleast_1d(axes)[-1].set_xlabel("visible time (seconds)")
    fig.suptitle(
        "Session-2 motion-primitive trajectories; black/magenta are appended children",
        y=0.998,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _save_confusions(
    path: Path, arm_results: Mapping[str, dict], class_names: Sequence[str]
) -> None:
    fig, axes = plt.subplots(1, len(ARMS), figsize=(21, 6.8))
    for axis, arm in zip(np.atleast_1d(axes), ARMS):
        matrix = np.asarray(
            arm_results[arm]["global_metrics"]["confusion_counts"], dtype=np.float64
        )
        normalized = np.divide(
            matrix,
            matrix.sum(axis=1, keepdims=True),
            out=np.zeros_like(matrix),
            where=matrix.sum(axis=1, keepdims=True) > 0,
        )
        image = axis.imshow(normalized, vmin=0.0, vmax=1.0, cmap="Blues")
        metrics = arm_results[arm]["global_metrics"]
        axis.set_title(
            f"{arm}\nAll={metrics['all_accuracy']:.3f}, "
            f"Old={metrics['old_accuracy']:.3f}, New={metrics['new_accuracy']:.3f}"
        )
        axis.set_xticks(
            range(len(class_names)), class_names, rotation=55, ha="right", fontsize=6
        )
        axis.set_yticks(range(len(class_names)), class_names, fontsize=6)
        axis.set_xlabel("globally aligned prediction")
        axis.set_ylabel("true activity")
        fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    fig.suptitle("Session-2 test confusion matrices")
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _save_activity_heatmaps(
    path: Path,
    matrices: Mapping[str, np.ndarray],
    class_names: Sequence[str],
) -> None:
    upper = max(float(np.max(value)) for value in matrices.values())
    fig, axes = plt.subplots(1, len(ARMS), figsize=(21, 6.8))
    for axis, arm in zip(np.atleast_1d(axes), ARMS):
        matrix = np.asarray(matrices[arm], dtype=np.float64)
        image = axis.imshow(matrix, vmin=0.0, vmax=max(upper, 1e-6), cmap="magma")
        axis.set_title(arm)
        axis.set_xticks(
            range(len(class_names)), class_names, rotation=55, ha="right", fontsize=6
        )
        axis.set_yticks(range(len(class_names)), class_names, fontsize=6)
        for row in range(len(matrix)):
            for column in range(len(matrix)):
                axis.text(
                    column,
                    row,
                    f"{matrix[row, column]:.2f}",
                    ha="center",
                    va="center",
                    fontsize=4,
                    color=(
                        "white"
                        if matrix[row, column] > 0.55 * max(upper, 1e-6)
                        else "black"
                    ),
                )
        fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    fig.suptitle("Activity mean-trajectory Jensen-Shannon distance heatmaps")
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def run(args: argparse.Namespace) -> dict:
    run_dir = Path(args.run_dir).resolve()
    npz_path = Path(args.npz_path).resolve()
    output_dir = _require_new_output_dir(Path(args.output_dir))
    bootstrap_resamples = int(args.bootstrap_resamples)
    seed = int(args.seed)
    expected_fold = getattr(args, "fold", None)
    config, split, _, codebook_path, protocol_identity = _validate_v2_run(
        run_dir,
        npz_path,
        seed=seed,
        expected_fold=(None if expected_fold is None else int(expected_fold)),
    )
    fold = int(protocol_identity["fold"])
    run_scope = (
        f"fold{fold:02d}_seed{seed}_session2_fixed_window_frozen_readout_v2"
    )
    segmentation = str(config.get("segmentation", {}).get("method", ""))
    if segmentation != "fixed_window":
        raise RuntimeError(
            "The v2 minimum experiment is pre-registered for fixed_window only; "
            f"observed {segmentation!r}."
        )
    upstream_audit = _validate_registered_upstream(config, load_segment_artifacts(run_dir, codebook_path))
    validation_subjects = protocol_identity["validation_subjects"]
    manifest = build_session2_manifest_v2(
        npz_path=npz_path,
        fold=fold,
        fit_subjects=split["fit_subjects"],
        eval_subjects=split["eval_subjects"],
        validation_subjects=validation_subjects,
        seed=seed,
    )
    train_ids = [int(value) for value in manifest["cumulative_train_trial_ids"]]
    test_ids = [int(value) for value in manifest["session_2_test_trial_ids"]]
    all_ids = train_ids + test_ids
    subjects_protocol = _subject_map_from_manifest(manifest, all_ids)

    artifacts = load_segment_artifacts(run_dir, codebook_path)
    base_centers_hash_before = codebook_center_hash(artifacts.centers)
    source = LabelFreeSourceSignalRepository(npz_path)
    visible_end_by_trial = {
        trial_id: int(max(source.trial_window_starts(trial_id)) + source.window_size)
        for trial_id in all_ids
    }
    trial_token_durations = build_trial_token_durations(
        artifacts, all_ids, visible_end_by_trial=visible_end_by_trial
    )

    coarse_train_histograms = duration_histograms(
        train_ids, trial_token_durations, primitive_num=len(artifacts.centers)
    )
    coarse_test_histograms = duration_histograms(
        test_ids, trial_token_durations, primitive_num=len(artifacts.centers)
    )
    frozen_readout = fit_frozen_coarse_readout(
        coarse_train_histograms,
        coarse_test_histograms,
        train_trial_ids=train_ids,
        test_trial_ids=test_ids,
        cluster_count=10,
        seed=seed,
        n_init=50,
        max_iter=300,
    )

    registered_candidate = _select_registered_candidate_or_fallback(
        {trial_id: trial_token_durations[trial_id] for trial_id in train_ids},
        minimum_fraction=float(args.minimum_token_fraction),
        minimum_support=int(args.registered_minimum_support),
    )
    selected_registered_parent = registered_candidate["selected_token"]
    registered_parent = (
        None
        if selected_registered_parent is None
        else int(selected_registered_parent)
    )
    if registered_parent is None:
        registered_fit_occurrences = []
        registered_states = {trial_id: 0 for trial_id in test_ids}
        registered_v2_states = {trial_id: 0 for trial_id in test_ids}
        (
            registered_fit_diagnostics,
            registered_v2_safety_audit,
        ) = _registered_no_parent_audits(registered_candidate, args)
        registered_gate_enabled = False
    else:
        registered_occurrence_map = build_token_occurrences(
            artifacts, source, all_ids, selected_token=registered_parent
        )
        registered_fit_ids = [
            int(value) for value in registered_candidate["dominant_trial_ids"]
        ]
        registered_fit_occurrences = [
            registered_occurrence_map[trial_id] for trial_id in registered_fit_ids
        ]
        registered_all_occurrences = [
            registered_occurrence_map[trial_id]
            for trial_id in sorted(registered_occurrence_map)
        ]
        registered_motion = _build_motion_components(
            artifacts, source, registered_fit_ids, registered_parent
        )
        registered_fit_subjects = {
            trial_id: subjects_protocol[trial_id] for trial_id in registered_fit_ids
        }
        registered_v1_model = fit_hierarchical_gate(
            registered_fit_occurrences,
            registered_motion,
            static_radius_quantile=float(args.static_radius_quantile),
            minimum_token_fraction=float(args.minimum_token_fraction),
            minimum_motion_energy_ratio=float(args.minimum_motion_energy_ratio),
            minimum_motion_energy_gap=float(args.minimum_motion_energy_gap),
            restarts=50,
            seed=seed,
        )
        registered_v2_model = fit_hierarchical_gate_v2(
            registered_fit_occurrences,
            registered_motion,
            registered_fit_subjects,
            static_radius_quantile=float(args.static_radius_quantile),
            minimum_token_fraction=float(args.minimum_token_fraction),
            minimum_motion_energy_ratio=float(args.minimum_motion_energy_ratio),
            minimum_motion_energy_gap=float(args.minimum_motion_energy_gap),
            minimum_child_count=int(args.minimum_child_trials),
            minimum_child_fraction=float(args.minimum_child_fraction),
            minimum_subjects_per_child=int(args.minimum_child_subjects),
            require_leave_one_out=True,
            minimum_loo_enabled_fraction=float(
                args.minimum_gate_loo_enabled_fraction
            ),
            minimum_loo_partition_agreement=float(args.minimum_gate_loo_agreement),
            minimum_loo_heldout_agreement=float(args.minimum_gate_loo_agreement),
            restarts=50,
            seed=seed,
        )
        registered_v1_assignments = assign_hierarchical_gate(
            registered_v1_model, registered_all_occurrences
        )
        registered_all_states = _state_map(registered_v1_assignments)
        registered_states = {
            trial_id: registered_all_states[trial_id]
            for trial_id in test_ids
            if trial_id in registered_all_states
        }
        registered_fit_diagnostics = hierarchical_gate_diagnostics(
            registered_v1_model, registered_fit_occurrences
        )
        registered_v2_assignments = assign_hierarchical_gate_v2(
            registered_v2_model, registered_all_occurrences
        )
        registered_v2_all_states = _state_map(registered_v2_assignments)
        registered_v2_states = {
            trial_id: registered_v2_all_states[trial_id]
            for trial_id in test_ids
            if trial_id in registered_v2_all_states
        }
        registered_v2_safety_audit = hierarchical_gate_v2_diagnostics(
            registered_v2_model, registered_fit_occurrences
        )
        registered_gate_enabled = bool(registered_v1_model.gate_enabled)

    adaptive_config = AdaptiveCodebookConfig(
        base_token_count=len(artifacts.centers),
        minimum_token_fraction=float(args.minimum_token_fraction),
        minimum_parent_trials=int(args.minimum_parent_trials),
        minimum_parent_subjects=int(args.minimum_parent_subjects),
        minimum_parent_trials_per_subject=1,
        minimum_child_trials=int(args.minimum_child_trials),
        minimum_child_fraction=float(args.minimum_child_fraction),
        minimum_child_subjects=int(args.minimum_child_subjects),
        minimum_silhouette=float(args.minimum_silhouette),
        minimum_loo_distortion_reduction=float(
            args.minimum_loo_distortion_reduction
        ),
        minimum_loo_stability=float(args.minimum_loo_stability),
        minimum_loso_stability=float(args.minimum_loso_stability),
        maximum_subject_nmi=float(args.maximum_subject_nmi),
        confidence_radius_quantile=float(args.confidence_radius_quantile),
        max_expanded_parents_per_session=1,
        max_new_children_per_session=2,
    )
    adaptive_train_occurrences = _build_adaptive_training_occurrences(
        artifacts,
        source,
        train_ids,
        trial_token_durations,
        subjects_protocol,
        minimum_fraction=float(args.minimum_token_fraction),
    )
    adaptive_model = fit_adaptive_codebook(
        artifacts.centers,
        adaptive_train_occurrences,
        session_index=2,
        config=adaptive_config,
    )
    adaptive_all_occurrences = _build_selected_adaptive_occurrences(
        artifacts,
        source,
        all_ids,
        subjects_protocol,
        selected_parent=adaptive_model.selected_parent_token,
    )
    adaptive_assignments = (
        assign_adaptive_codebook(adaptive_model, adaptive_all_occurrences)
        if adaptive_all_occurrences
        else _empty_adaptive_assignments()
    )
    adaptive_all_states = adaptive_state_by_trial(
        adaptive_model, adaptive_assignments
    )
    adaptive_states = {
        trial_id: adaptive_all_states[trial_id]
        for trial_id in test_ids
        if trial_id in adaptive_all_states
    }
    adaptive_invariants = audit_codebook_invariants(
        adaptive_model, current_base_centers=artifacts.centers
    )
    if not adaptive_invariants["all_invariants_passed"]:
        raise RuntimeError("Adaptive append-only codebook invariant failed.")
    base_centers_hash_after = codebook_center_hash(artifacts.centers)
    if base_centers_hash_after != base_centers_hash_before:
        raise RuntimeError("The frozen K32 parent centres changed during online fitting.")

    f0_states = {trial_id: 0 for trial_id in test_ids}
    predictions = {
        ARMS[0]: np.asarray(frozen_readout.test_predictions, dtype=np.int64).copy()
    }
    registered_leaf = apply_frozen_leaf_overrides(
        frozen_readout, registered_states
    )
    adaptive_leaf = apply_frozen_leaf_overrides(
        frozen_readout, adaptive_states
    )
    predictions[ARMS[1]] = registered_leaf.refined_predictions.copy()
    predictions[ARMS[2]] = adaptive_leaf.refined_predictions.copy()
    states_by_arm = {
        ARMS[0]: f0_states,
        ARMS[1]: registered_states,
        ARMS[2]: adaptive_states,
    }
    parents_by_arm = {
        ARMS[0]: None,
        ARMS[1]: registered_parent,
        ARMS[2]: adaptive_model.selected_parent_token,
    }

    histograms = {
        ARMS[0]: coarse_test_histograms,
        ARMS[1]: (
            coarse_test_histograms.copy()
            if registered_parent is None
            else hierarchical_duration_histograms(
                test_ids,
                trial_token_durations,
                primitive_num=len(artifacts.centers),
                selected_token=registered_parent,
                gate_state_by_trial=registered_states,
            )
        ),
        ARMS[2]: adaptive_duration_histograms(
            test_ids,
            trial_token_durations,
            adaptive_model,
            gate_state_by_trial=adaptive_states,
        ),
    }
    trajectory_audits = {
        arm: _trajectory_invariant_audit(
            artifacts,
            test_ids,
            parents_by_arm[arm],
            states_by_arm[arm],
        )
        for arm in ARMS
    }
    readout_audits = {
        ARMS[0]: {
            "coarse_fit_call_count": 1,
            "fallback_raw_mismatch_count": 0,
            "changed_raw_prediction_count": 0,
            "routed_trial_count": 0,
        },
        ARMS[1]: frozen_readout_diagnostics(frozen_readout, registered_leaf),
        ARMS[2]: frozen_readout_diagnostics(frozen_readout, adaptive_leaf),
    }
    if any(
        int(readout_audits[arm]["fallback_raw_mismatch_count"]) != 0
        for arm in ARMS
    ):
        raise RuntimeError("Frozen-readout fallback audit failed.")
    if int(readout_audits[ARMS[0]]["coarse_fit_call_count"]) != 1:
        raise RuntimeError("The coarse trial readout was not fitted exactly once.")

    output_dir.mkdir(parents=True)
    frozen_path = output_dir / "frozen_label_free_v2_decisions.npz"
    np.savez_compressed(
        frozen_path,
        train_trial_ids=np.asarray(train_ids, dtype=np.int64),
        test_trial_ids=np.asarray(test_ids, dtype=np.int64),
        coarse_train_predictions=frozen_readout.train_predictions,
        coarse_test_predictions=frozen_readout.test_predictions,
        registered_test_predictions=predictions[ARMS[1]],
        adaptive_test_predictions=predictions[ARMS[2]],
        registered_gate_states=np.asarray(
            [registered_states.get(trial_id, 0) for trial_id in test_ids],
            dtype=np.int64,
        ),
        registered_v2_safe_gate_states=np.asarray(
            [registered_v2_states.get(trial_id, 0) for trial_id in test_ids],
            dtype=np.int64,
        ),
        adaptive_gate_states=np.asarray(
            [adaptive_states.get(trial_id, 0) for trial_id in test_ids],
            dtype=np.int64,
        ),
        registered_parent_token=np.asarray(
            [-1 if registered_parent is None else registered_parent],
            dtype=np.int64,
        ),
        adaptive_parent_token=np.asarray(
            [
                -1
                if adaptive_model.selected_parent_token is None
                else adaptive_model.selected_parent_token
            ],
            dtype=np.int64,
        ),
        primitive_base_center_hash=np.asarray([base_centers_hash_before]),
    )

    # Ground truth is unavailable to every fit/route call above.  It is joined
    # only after the raw decisions have been frozen on disk.
    evaluation_source = SourceSignalRepository(npz_path)
    labels, subjects, names = _truth_maps(evaluation_source, test_ids)
    for trial_id in test_ids:
        if int(subjects[trial_id]) != int(subjects_protocol[trial_id]):
            raise RuntimeError("Post-freeze subject metadata disagrees with the protocol.")
    y_true = np.asarray([labels[trial_id] for trial_id in test_ids], dtype=np.int64)
    y_subject = np.asarray(
        [subjects[trial_id] for trial_id in test_ids], dtype=np.int64
    )
    class_ids = sorted(set(y_true.tolist()))
    class_names = [
        names[next(trial_id for trial_id in test_ids if labels[trial_id] == class_id)]
        for class_id in class_ids
    ]

    arm_results: dict[str, dict] = {}
    distance_matrices: dict[str, np.ndarray] = {}
    aligned: dict[str, np.ndarray] = {}
    for arm in ARMS:
        metrics = _complete_clustering_metrics(y_true, predictions[arm])
        aligned[arm] = np.asarray(metrics["aligned_predictions"], dtype=np.int64)
        distance = class_mean_histogram_distances(
            histograms[arm], y_true, class_ids
        )
        distance_matrices[arm] = distance
        if arm == ARMS[0]:
            posthoc = None
            vocabulary = {
                "base_token_count": len(artifacts.centers),
                "K_total": len(artifacts.centers),
                "selected_parent_token": None,
                "appended_child_token_ids": [],
            }
        elif arm == ARMS[1]:
            posthoc = _posthoc_gate_sit_stand(
                test_ids,
                registered_states,
                labels,
                subjects,
                sitting_label=7,
                standing_label=8,
            )
            vocabulary = {
                "base_token_count": len(artifacts.centers),
                "K_total": (
                    len(artifacts.centers) + 2
                    if registered_gate_enabled
                    else len(artifacts.centers)
                ),
                "selected_parent_token": registered_parent,
                "appended_child_token_ids": (
                    [len(artifacts.centers), len(artifacts.centers) + 1]
                    if registered_gate_enabled
                    else []
                ),
            }
        else:
            posthoc = _posthoc_gate_sit_stand(
                test_ids,
                adaptive_states,
                labels,
                subjects,
                sitting_label=7,
                standing_label=8,
            )
            vocabulary = {
                "base_token_count": len(artifacts.centers),
                "K_total": int(adaptive_model.K_total),
                "selected_parent_token": adaptive_model.selected_parent_token,
                "appended_child_token_ids": adaptive_invariants[
                    "appended_child_token_ids"
                ],
            }
        arm_results[arm] = {
            "primitive_vocabulary": vocabulary,
            "shared_frozen_trial_readout": readout_audits[arm],
            "trajectory_invariant_audit": trajectory_audits[arm],
            "global_metrics": metrics,
            "sit_stand_posthoc": posthoc,
            "activity_mean_histogram_distance_matrix": distance,
        }

    for arm in ARMS[1:]:
        arm_results[arm]["attribution_audit"] = _attribution_audit(
            test_ids,
            y_true,
            predictions[ARMS[0]],
            predictions[arm],
            aligned[ARMS[0]],
            aligned[arm],
            states_by_arm[arm],
        )

    bootstrap, bootstrap_rows = _paired_bootstrap_comparisons(
        trial_ids=test_ids,
        y_true=y_true,
        subjects=y_subject,
        comparisons={
            f"{arm}_minus_{ARMS[0]}": (aligned[ARMS[0]], aligned[arm])
            for arm in ARMS[1:]
        },
        resamples=bootstrap_resamples,
        seed=seed,
        scope=run_scope,
    )

    _write_csv(
        output_dir / "session_manifest.csv",
        [
            {"split": split_name, "session": session, "trial_global_id": int(trial_id)}
            for split_name, session, ids in (
                ("online_train", 1, manifest["session_1_train_trial_ids"]),
                ("online_train", 2, manifest["session_2_train_trial_ids"]),
                ("online_test", 2, manifest["session_2_test_trial_ids"]),
            )
            for trial_id in ids
        ],
    )
    prediction_rows = []
    for row, trial_id in enumerate(test_ids):
        item = {
            "trial_global_id": trial_id,
            "subject_id": subjects[trial_id],
            "activity_label_0based": labels[trial_id],
            "activity_name": names[trial_id],
            "registered_gate_state": registered_states.get(trial_id, 0),
            "registered_v2_safe_gate_state": registered_v2_states.get(trial_id, 0),
            "adaptive_gate_state": adaptive_states.get(trial_id, 0),
        }
        for arm in ARMS:
            item[f"{arm}_raw_cluster"] = int(predictions[arm][row])
            item[f"{arm}_aligned_prediction"] = int(aligned[arm][row])
        prediction_rows.append(item)
    _write_csv(output_dir / "session2_predictions.csv", prediction_rows)

    trajectory_rows = []
    for trial_id in test_ids:
        item = {
            "trial_global_id": trial_id,
            "subject_id": subjects[trial_id],
            "activity_label_0based": labels[trial_id],
            "activity_name": names[trial_id],
        }
        for arm in ARMS:
            tokens, starts, ends = _actual_segment_tokens(
                artifacts,
                trial_id,
                parents_by_arm[arm],
                states_by_arm[arm],
            )
            rle, _ = run_length_encode(tokens.astype(int).tolist())
            item[f"{arm}_rle_tokens"] = json.dumps(rle.astype(int).tolist())
            item[f"{arm}_segment_tokens"] = json.dumps(tokens.astype(int).tolist())
            item[f"{arm}_durations_samples"] = json.dumps(
                (ends - starts).astype(int).tolist()
            )
        trajectory_rows.append(item)
    _write_csv(output_dir / "session2_adaptive_trajectories.csv", trajectory_rows)
    _write_csv(
        output_dir / "trial_bootstrap_confidence_intervals.csv", bootstrap_rows
    )

    _save_trajectory_plot(
        output_dir / "session2_adaptive_trajectories.png",
        artifacts,
        test_ids,
        parents_by_arm,
        states_by_arm,
        labels,
        subjects,
        names,
    )
    _save_confusions(
        output_dir / "session2_global_confusions.png", arm_results, class_names
    )
    _save_activity_heatmaps(
        output_dir / "session2_activity_distance_heatmaps.png",
        distance_matrices,
        class_names,
    )

    implementation_paths = {
        "runner": Path(__file__).resolve(),
        "adaptive_codebook": PROJECT_ROOT
        / "experiments"
        / "motion_primitive"
        / "adaptive_codebook.py",
        "frozen_readout": PROJECT_ROOT
        / "experiments"
        / "motion_primitive"
        / "frozen_hierarchical_readout.py",
        "hierarchical_gate_v2": PROJECT_ROOT
        / "experiments"
        / "motion_primitive"
        / "hierarchical_gate_v2.py",
        "historical_hierarchical_gate_v1": PROJECT_ROOT
        / "experiments"
        / "motion_primitive"
        / "hierarchical_gate.py",
    }
    run_protocol = {
        **PROTOCOL,
        "scope": run_scope,
        "outer_fold": fold,
        "run_seed": seed,
    }
    result = {
        "protocol": run_protocol,
        "arguments": {
            "run_dir": str(run_dir),
            "npz_path": str(npz_path),
            "output_dir": str(output_dir),
            "fold": fold,
            "seed": seed,
            "bootstrap_resamples": bootstrap_resamples,
            "minimum_token_fraction": float(args.minimum_token_fraction),
            "registered_minimum_support": int(args.registered_minimum_support),
            "minimum_parent_trials": int(args.minimum_parent_trials),
            "minimum_parent_subjects": int(args.minimum_parent_subjects),
            "minimum_child_trials": int(args.minimum_child_trials),
            "minimum_child_fraction": float(args.minimum_child_fraction),
            "minimum_child_subjects": int(args.minimum_child_subjects),
            "minimum_silhouette": float(args.minimum_silhouette),
            "minimum_loo_distortion_reduction": float(
                args.minimum_loo_distortion_reduction
            ),
            "minimum_loo_stability": float(args.minimum_loo_stability),
            "minimum_loso_stability": float(args.minimum_loso_stability),
            "maximum_subject_nmi": float(args.maximum_subject_nmi),
            "confidence_radius_quantile": float(args.confidence_radius_quantile),
            "static_radius_quantile": float(args.static_radius_quantile),
            "minimum_motion_energy_ratio": float(
                args.minimum_motion_energy_ratio
            ),
            "minimum_motion_energy_gap": float(args.minimum_motion_energy_gap),
            "minimum_gate_loo_enabled_fraction": float(
                args.minimum_gate_loo_enabled_fraction
            ),
            "minimum_gate_loo_agreement": float(args.minimum_gate_loo_agreement),
        },
        "input_audit": {
            "uschad_cv_fold": fold,
            "run_seed": seed,
            "fit_subjects": protocol_identity["fit_subjects"],
            "validation_subjects": protocol_identity["validation_subjects"],
            "eval_subjects": protocol_identity["eval_subjects"],
            "canonical_subject_partition_verified": protocol_identity[
                "canonical_partition_verified"
            ],
            "registered_upstream": upstream_audit,
            "primitive_segmentation": segmentation,
            "encoder_ablation_profile": config.get("encoder_training", {}).get(
                "ablation_profile"
            ),
            "run_config_sha256": sha256_file(run_dir / "experiment_config.json"),
            "segment_artifact_sha256": sha256_file(
                run_dir / "segment_embeddings_and_tokens.npz"
            ),
            "codebook_artifact_sha256": sha256_file(codebook_path),
            "npz_sha256": sha256_file(npz_path),
            "base_center_hash_before": base_centers_hash_before,
            "base_center_hash_after": base_centers_hash_after,
            "base_centers_unchanged": (
                base_centers_hash_before == base_centers_hash_after
            ),
            "label_free_raw_source": source.source_audit(all_ids),
            "postfreeze_metadata_source": evaluation_source.source_audit(all_ids),
            "implementation_fingerprint": {
                name: {
                    "path": str(path),
                    "sha256": sha256_file(path),
                }
                for name, path in implementation_paths.items()
            },
        },
        "session_manifest_audit": manifest,
        "label_firewall": {
            "adaptive_fit_uses_activity_labels_or_names": False,
            "registered_gate_fit_uses_activity_labels_or_names": False,
            "trial_readout_fit_uses_activity_labels_or_names": False,
            "test_occurrences_used_to_fit_adaptive_codebook": False,
            "test_population_used_to_calibrate_descriptor_scaler": False,
            "subject_id_used_as_descriptor": False,
            "production_manifest_validates_registered_session_schedule_only": True,
            "production_manifest_is_fold_generic_and_metadata_bound": True,
            "raw_predictions_persisted_before_scoring_metadata_join": True,
            "evaluation_ground_truth_join_time": (
                "after codebook selection, route states, and raw test predictions "
                "were frozen to frozen_label_free_v2_decisions.npz"
            ),
        },
        "readout_audit": {
            "coarse_fit_call_count": 1,
            "coarse_cluster_count": int(frozen_readout.cluster_count),
            "coarse_center_hash": codebook_center_hash(frozen_readout.centers),
            "coarse_train_prediction_hash": _array_sha256(
                frozen_readout.train_predictions
            ),
            "coarse_test_prediction_hash": _array_sha256(
                frozen_readout.test_predictions
            ),
            "test_trial_order_hash": _array_sha256(
                np.asarray(test_ids, dtype=np.int64)
            ),
            "all_refined_fallback_raw_mismatch_counts_are_zero": True,
        },
        "registered_candidate": registered_candidate,
        "registered_gate_fit": registered_fit_diagnostics,
        "registered_gate_v2_safety_audit": registered_v2_safety_audit,
        "adaptive_selection": adaptive_codebook_diagnostics(adaptive_model),
        "codebook_audit": adaptive_invariants,
        "arms": arm_results,
        "trial_bootstrap_confidence_intervals": bootstrap,
        "frozen_label_free_decisions": {
            "file": frozen_path.name,
            "sha256": sha256_file(frozen_path),
            "contains_activity_labels_or_names": False,
        },
        "generated_files": [
            "online_adaptive_codebook_v2_results.json",
            "frozen_label_free_v2_decisions.npz",
            "session_manifest.csv",
            "session2_predictions.csv",
            "session2_adaptive_trajectories.csv",
            "session2_adaptive_trajectories.png",
            "session2_global_confusions.png",
            "session2_activity_distance_heatmaps.png",
            "trial_bootstrap_confidence_intervals.csv",
        ],
    }
    (output_dir / "online_adaptive_codebook_v2_results.json").write_text(
        json.dumps(jsonable(result), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return result


def _unit_interval(value: str) -> float:
    parsed = float(value)
    if not np.isfinite(parsed) or not 0.0 < parsed <= 1.0:
        raise argparse.ArgumentTypeError("value must lie in (0,1]")
    return parsed


def _fraction_at_most_half(value: str) -> float:
    parsed = float(value)
    if not np.isfinite(parsed) or not 0.0 < parsed <= 0.5:
        raise argparse.ArgumentTypeError("value must lie in (0,0.5]")
    return parsed


def _minimum_two(value: str) -> int:
    parsed = int(value)
    if parsed < 2:
        raise argparse.ArgumentTypeError("value must be at least two")
    return parsed


def _minimum_three(value: str) -> int:
    parsed = int(value)
    if parsed < 3:
        raise argparse.ArgumentTypeError("value must be at least three")
    return parsed


def _ratio_at_least_one(value: str) -> float:
    parsed = float(value)
    if not np.isfinite(parsed) or parsed < 1.0:
        raise argparse.ArgumentTypeError("value must be finite and at least one")
    return parsed


def _nonnegative(value: str) -> float:
    parsed = float(value)
    if not np.isfinite(parsed) or parsed < 0.0:
        raise argparse.ArgumentTypeError("value must be finite and non-negative")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Seven-fold fixed-window frozen-readout experiment with registered "
            "and adaptive append-only K32/K34 motion-primitive vocabularies."
        )
    )
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--npz-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--fold",
        type=int,
        choices=sorted(CANONICAL_EVAL_SUBJECTS_BY_FOLD),
        default=None,
        help=(
            "Optional explicit outer-fold assertion. When omitted, the fold is "
            "read from checkpoint metadata and still checked against the canonical grid."
        ),
    )
    parser.add_argument("--seed", type=int, default=500)
    parser.add_argument("--bootstrap-resamples", type=_positive_int, default=10000)
    parser.add_argument("--minimum-token-fraction", type=_unit_interval, default=0.50)
    parser.add_argument("--registered-minimum-support", type=_positive_int, default=10)
    parser.add_argument("--minimum-parent-trials", type=_minimum_three, default=6)
    parser.add_argument("--minimum-parent-subjects", type=_minimum_two, default=2)
    parser.add_argument("--minimum-child-trials", type=_minimum_three, default=3)
    parser.add_argument(
        "--minimum-child-fraction", type=_fraction_at_most_half, default=0.15
    )
    parser.add_argument("--minimum-child-subjects", type=_minimum_two, default=2)
    parser.add_argument("--minimum-silhouette", type=_unit_interval, default=0.25)
    parser.add_argument(
        "--minimum-loo-distortion-reduction", type=_unit_interval, default=0.15
    )
    parser.add_argument("--minimum-loo-stability", type=_unit_interval, default=0.80)
    parser.add_argument("--minimum-loso-stability", type=_unit_interval, default=0.80)
    parser.add_argument("--maximum-subject-nmi", type=_unit_interval, default=0.25)
    parser.add_argument(
        "--confidence-radius-quantile", type=_unit_interval, default=0.95
    )
    parser.add_argument(
        "--static-radius-quantile", type=_unit_interval, default=0.95
    )
    parser.add_argument(
        "--minimum-motion-energy-ratio", type=_ratio_at_least_one, default=2.0
    )
    parser.add_argument(
        "--minimum-motion-energy-gap", type=_nonnegative, default=0.25
    )
    parser.add_argument(
        "--minimum-gate-loo-enabled-fraction", type=_unit_interval, default=1.0
    )
    parser.add_argument(
        "--minimum-gate-loo-agreement", type=_unit_interval, default=0.90
    )
    return parser.parse_args()


def main() -> None:
    result = run(parse_args())
    print(
        json.dumps(
            jsonable(
                {
                    "output_dir": result["arguments"]["output_dir"],
                    "fold": result["arguments"]["fold"],
                    "encoder": result["input_audit"]["encoder_ablation_profile"],
                    "registered_parent": result["registered_candidate"][
                        "selected_token"
                    ],
                    "registered_v2_gate_enabled": result[
                        "registered_gate_v2_safety_audit"
                    ]["gate_enabled"],
                    "adaptive_K_total": result["adaptive_selection"]["K_total"],
                    "adaptive_selected_parent": result["adaptive_selection"][
                        "selected_parent_token"
                    ],
                    "metrics": {
                        arm: result["arms"][arm]["global_metrics"] for arm in ARMS
                    },
                }
            ),
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
