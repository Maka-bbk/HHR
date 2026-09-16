"""Run one registered frozen-encoder W128/K64 descriptor experiment part.

The six existing fold/seed A2 checkpoints are reused verbatim.  Each fold/seed
fits its E0 codebook exactly once in the legacy member, and every other member
in that experiment part reuses the frozen state.  The three original parts stay
factor-isolated.  The separately invoked duration-soft-subject part uses the
duration-invariant profile as its direct baseline and compares three registered
soft projection strengths without changing encoder or codebook state.
The dedicated seven-fold confirmation part compares duration invariance with
soft subject projection strengths 0.25 and 0.50 while keeping folds 1--3 as
the screening partition and folds 4--7 as the untouched confirmation partition.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from experiments.motion_primitive.frozen_e0 import (
    DESCRIPTOR_PROFILES,
    DURATION_INVARIANT_DESCRIPTOR_PROFILE,
    DURATION_SOFT_SUBJECT_A025_PROFILE,
    DURATION_SOFT_SUBJECT_A050_PROFILE,
    DURATION_SOFT_SUBJECT_A075_PROFILE,
    GRAVITY_DESCRIPTOR_PROFILE,
    LEGACY_DESCRIPTOR_PROFILE,
    SUBJECT_DEBIASED_DESCRIPTOR_PROFILE,
    descriptor_profile_spec,
)
from experiments.motion_primitive.strict_artifacts import write_csv, write_json
from experiments.motion_primitive.strict_cv_common import (
    bootstrap_fold_mean,
    canonical_hash,
    member_directory,
    parse_integer_grid,
)
from experiments.motion_primitive.strict_protocol import sha256_file
from experiments.motion_primitive.window_codebook_batch_proxy import (
    validate_completed_output as validate_member_completed_output,
)


SCHEMA = "hhr_trajectory_descriptor_ablation_cv_v5"
SUITE_SCHEMA = "hhr_trajectory_descriptor_ablation_suite_v1"
MEMBER_SCHEMA = "hhr_window_codebook_batch_proxy_v3"
GRAVITY_EXPERIMENT_PART = "gravity"
SUBJECT_DEBIAS_EXPERIMENT_PART = "subject_debias"
DURATION_EXPERIMENT_PART = "duration"
DURATION_SOFT_SUBJECT_EXPERIMENT_PART = "duration_soft_subject"
DURATION_SOFT_SUBJECT_CONFIRM_7FOLD_EXPERIMENT_PART = (
    "duration_soft_subject_confirm_7fold"
)
ALL_EXPERIMENT_PARTS = "all"
EXPERIMENT_PART_PROFILES: dict[str, tuple[str, ...]] = {
    GRAVITY_EXPERIMENT_PART: (
        LEGACY_DESCRIPTOR_PROFILE,
        GRAVITY_DESCRIPTOR_PROFILE,
    ),
    SUBJECT_DEBIAS_EXPERIMENT_PART: (
        LEGACY_DESCRIPTOR_PROFILE,
        SUBJECT_DEBIASED_DESCRIPTOR_PROFILE,
    ),
    DURATION_EXPERIMENT_PART: (
        LEGACY_DESCRIPTOR_PROFILE,
        DURATION_INVARIANT_DESCRIPTOR_PROFILE,
    ),
    DURATION_SOFT_SUBJECT_EXPERIMENT_PART: (
        LEGACY_DESCRIPTOR_PROFILE,
        DURATION_INVARIANT_DESCRIPTOR_PROFILE,
        DURATION_SOFT_SUBJECT_A025_PROFILE,
        DURATION_SOFT_SUBJECT_A050_PROFILE,
        DURATION_SOFT_SUBJECT_A075_PROFILE,
    ),
    DURATION_SOFT_SUBJECT_CONFIRM_7FOLD_EXPERIMENT_PART: (
        LEGACY_DESCRIPTOR_PROFILE,
        DURATION_INVARIANT_DESCRIPTOR_PROFILE,
        DURATION_SOFT_SUBJECT_A025_PROFILE,
        DURATION_SOFT_SUBJECT_A050_PROFILE,
    ),
}
# ``all`` remains the already-completed three-single-factor suite.  The new
# combination experiment is intentionally opt-in so resuming ``all`` cannot
# silently expand the historical 36-member protocol.
REGISTERED_EXPERIMENT_PARTS = (
    GRAVITY_EXPERIMENT_PART,
    SUBJECT_DEBIAS_EXPERIMENT_PART,
    DURATION_EXPERIMENT_PART,
)
CLI_EXPERIMENT_PARTS = (*tuple(EXPERIMENT_PART_PROFILES), ALL_EXPERIMENT_PARTS)
DEFAULT_FOLDS = (1, 2, 3)
DEFAULT_SEEDS = (0, 5)
CONFIRMATION_FOLDS = tuple(range(1, 8))
CONFIRMATION_SEEDS = (0, 5)
SCREENING_FOLDS = (1, 2, 3)
CONFIRMATION_HOLDOUT_FOLDS = (4, 5, 6, 7)
WINDOW_SIZE = 128
WINDOW_STRIDE = 64
PRIMITIVE_NUM = 64
OUTER_TRIAL_COUNT = 120
PERFORMANCE_METRICS = (
    "all_accuracy",
    "old_accuracy",
    "new_accuracy",
    "h_score",
    "macro_f1",
    "ari",
    "nmi",
)
BIAS_DIAGNOSTICS = (
    "max_abs_feature_log_duration_correlation",
    "outer_subject_centroid_dispersion_diagnostic",
)
PART_CONTRASTS: dict[str, tuple[tuple[str, dict[str, float]], ...]] = {
    GRAVITY_EXPERIMENT_PART: (
        (
            "signed_gravity_minus_legacy",
            {GRAVITY_DESCRIPTOR_PROFILE: 1.0, LEGACY_DESCRIPTOR_PROFILE: -1.0},
        ),
    ),
    SUBJECT_DEBIAS_EXPERIMENT_PART: (
        (
            "subject_debias_minus_legacy",
            {
                SUBJECT_DEBIASED_DESCRIPTOR_PROFILE: 1.0,
                LEGACY_DESCRIPTOR_PROFILE: -1.0,
            },
        ),
    ),
    DURATION_EXPERIMENT_PART: (
        (
            "duration_invariant_minus_legacy",
            {
                DURATION_INVARIANT_DESCRIPTOR_PROFILE: 1.0,
                LEGACY_DESCRIPTOR_PROFILE: -1.0,
            },
        ),
    ),
    DURATION_SOFT_SUBJECT_EXPERIMENT_PART: (
        (
            "soft_a025_minus_duration_invariant",
            {
                DURATION_SOFT_SUBJECT_A025_PROFILE: 1.0,
                DURATION_INVARIANT_DESCRIPTOR_PROFILE: -1.0,
            },
        ),
        (
            "soft_a050_minus_duration_invariant",
            {
                DURATION_SOFT_SUBJECT_A050_PROFILE: 1.0,
                DURATION_INVARIANT_DESCRIPTOR_PROFILE: -1.0,
            },
        ),
        (
            "soft_a075_minus_duration_invariant",
            {
                DURATION_SOFT_SUBJECT_A075_PROFILE: 1.0,
                DURATION_INVARIANT_DESCRIPTOR_PROFILE: -1.0,
            },
        ),
    ),
    DURATION_SOFT_SUBJECT_CONFIRM_7FOLD_EXPERIMENT_PART: (
        (
            "duration_invariant_minus_legacy",
            {
                DURATION_INVARIANT_DESCRIPTOR_PROFILE: 1.0,
                LEGACY_DESCRIPTOR_PROFILE: -1.0,
            },
        ),
        (
            "soft_a025_minus_duration_invariant",
            {
                DURATION_SOFT_SUBJECT_A025_PROFILE: 1.0,
                DURATION_INVARIANT_DESCRIPTOR_PROFILE: -1.0,
            },
        ),
        (
            "soft_a050_minus_duration_invariant",
            {
                DURATION_SOFT_SUBJECT_A050_PROFILE: 1.0,
                DURATION_INVARIANT_DESCRIPTOR_PROFILE: -1.0,
            },
        ),
    ),
}


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON artifact is not an object: {path}.")
    return value


def profiles_for_part(value: str) -> tuple[str, ...]:
    part = str(value).strip()
    try:
        profiles = EXPERIMENT_PART_PROFILES[part]
    except KeyError as error:
        raise ValueError(
            f"Unknown experiment part {part!r}; expected "
            f"{list(EXPERIMENT_PART_PROFILES)}."
        ) from error
    if profiles[0] != LEGACY_DESCRIPTOR_PROFILE:
        raise RuntimeError("Every registered part must start with the codebook producer.")
    if len(profiles) != len(set(profiles)) or not set(profiles) <= set(DESCRIPTOR_PROFILES):
        raise RuntimeError("A registered experiment part has an invalid profile grid.")
    return profiles


def _checkpoint_path(root: Path, fold: int, seed: int) -> Path:
    return root / "a2" / f"fold_{int(fold):02d}_seed_{int(seed)}" / "motion_encoder_final.pt"


def _member_command(
    args: argparse.Namespace,
    *,
    profile: str,
    fold: int,
    seed: int,
    checkpoint: Path,
    output: Path,
    reuse_codebook_run_dir: Path | None,
) -> list[str]:
    command = [
        sys.executable,
        str(PROJECT_ROOT / "experiments" / "motion_primitive" / "window_codebook_batch_proxy.py"),
        "--a2-checkpoint", str(checkpoint),
        "--npz-path", str(Path(args.npz_path).expanduser().resolve()),
        "--output-dir", str(output),
        "--fold", str(int(fold)),
        "--seed", str(int(seed)),
        "--window-size", str(WINDOW_SIZE),
        "--window-stride", str(WINDOW_STRIDE),
        "--primitive-num", str(PRIMITIVE_NUM),
        "--pca-dim", "64",
        "--activity-cluster-count", "12",
        "--descriptor-pca-dim", "32",
        "--descriptor-profile", str(profile),
        "--signed-vertical-distance-weight", str(float(args.signed_vertical_distance_weight)),
        "--subject-nuisance-max-rank", str(int(args.subject_nuisance_max_rank)),
        "--subject-nuisance-explained-variance", str(float(args.subject_nuisance_explained_variance)),
        "--kmeans-n-init", str(int(args.kmeans_n_init)),
        "--kmeans-max-iter", str(int(args.kmeans_max_iter)),
        "--encode-batch-size", str(int(args.encode_batch_size)),
        "--device", str(args.device),
    ]
    if reuse_codebook_run_dir is not None:
        command.extend(
            ["--reuse-codebook-run-dir", str(reuse_codebook_run_dir.resolve())]
        )
    if bool(args.resume):
        command.append("--resume")
    return command


def _quoted(command: Sequence[str]) -> str:
    return " ".join(f'"{item}"' for item in command)


def _run_command(command: Sequence[str], *, stage: str) -> None:
    print("[command] " + _quoted(command), flush=True)
    try:
        subprocess.run(list(command), cwd=PROJECT_ROOT, check=True)
    except subprocess.CalledProcessError as error:
        raise RuntimeError(
            f"Descriptor-ablation stage {stage!r} failed with exit code "
            f"{error.returncode}; the child traceback is printed above."
        ) from error


def _implementation_hashes() -> dict[str, str]:
    names = (
        "experiments/motion_primitive/run_trajectory_descriptor_ablation_cv.py",
        "experiments/motion_primitive/window_codebook_batch_proxy.py",
        "experiments/motion_primitive/frozen_e0.py",
    )
    return {name: sha256_file(PROJECT_ROOT / name) for name in names}


def _identity(
    args: argparse.Namespace,
    experiment_part: str,
    profiles: Sequence[str],
    folds: Sequence[int],
    seeds: Sequence[int],
) -> dict[str, Any]:
    npz = Path(args.npz_path).expanduser().resolve()
    encoder_root = Path(args.encoder_source_root).expanduser().resolve()
    checkpoints: list[dict[str, Any]] = []
    for fold in folds:
        for seed in seeds:
            checkpoint = _checkpoint_path(encoder_root, fold, seed)
            if not checkpoint.is_file():
                raise FileNotFoundError(checkpoint)
            checkpoints.append(
                {
                    "fold": int(fold),
                    "seed": int(seed),
                    "path": str(checkpoint),
                    "sha256": sha256_file(checkpoint),
                }
            )
    return {
        "schema": SCHEMA,
        "experiment_role": "frozen_encoder_trajectory_descriptor_ablation",
        "experiment_part": str(experiment_part),
        "npz_path": str(npz),
        "npz_sha256": sha256_file(npz),
        "encoder_source_root": str(encoder_root),
        "encoder_checkpoints": checkpoints,
        "window_size": WINDOW_SIZE,
        "window_stride": WINDOW_STRIDE,
        "primitive_num": PRIMITIVE_NUM,
        "profiles": list(profiles),
        "profile_subject_nuisance_projection_strengths": {
            str(profile): float(
                descriptor_profile_spec(profile).subject_nuisance_projection_strength
            )
            for profile in profiles
        },
        "folds": list(folds),
        "seeds": list(seeds),
        "member_count": int(len(profiles) * len(folds) * len(seeds)),
        "signed_vertical_distance_weight": float(args.signed_vertical_distance_weight),
        "subject_nuisance_max_rank": int(args.subject_nuisance_max_rank),
        "subject_nuisance_explained_variance": float(
            args.subject_nuisance_explained_variance
        ),
        "kmeans_n_init": int(args.kmeans_n_init),
        "kmeans_max_iter": int(args.kmeans_max_iter),
        "encode_batch_size": int(args.encode_batch_size),
        "protocol": {
            "experiment_parts_are_run_in_separate_output_roots": True,
            "encoder_and_codebook_change_between_profiles": False,
            "codebook_fit_once_per_fold_seed": True,
            "codebook_producer_profile": LEGACY_DESCRIPTOR_PROFILE,
            "nonlegacy_profiles_reuse_completed_legacy_codebook": True,
            "shared_codebook_state_sha256_required": True,
            "trajectory_coordinate_transform_fit_scope": "all_outer_unlabeled_descriptors",
            "subject_nuisance_basis_fit_scope": "offline_old6_source_subject_metadata",
            "outer_activity_truth_used_by_learner": False,
            "outer_subject_ids_used_by_nuisance_fit": False,
            "global_hungarian": "post_hoc_scoring_only",
            "known_activity_cluster_count": 12,
            "not_deployable_online_cgcd": True,
        },
        "implementation_sha256": _implementation_hashes(),
    }


def _prepare_manifest(root: Path, identity: Mapping[str, Any]) -> str:
    body = dict(identity)
    digest = canonical_hash(body)
    expected = {**body, "identity_sha256": digest}
    path = root / "grid_manifest.json"
    if path.is_file():
        observed = _read_json(path)
        observed_body = {key: value for key, value in observed.items() if key != "identity_sha256"}
        if observed.get("identity_sha256") != canonical_hash(observed_body):
            raise RuntimeError("Existing descriptor-ablation manifest hash is invalid.")
        if observed != expected:
            raise RuntimeError(
                "Output root records another descriptor ablation; use a new --output-root."
            )
    else:
        if root.exists() and any(root.iterdir()):
            raise RuntimeError("A non-empty output root has no descriptor-ablation identity.")
        root.mkdir(parents=True, exist_ok=True)
        write_json(path, expected)
    return digest


def _validated_member(
    output: Path, *, profile: str, fold: int, seed: int
) -> dict[str, Any]:
    path = output / "complete.json"
    identity_path = output / "run_identity.json"
    if not path.is_file() or not identity_path.is_file():
        raise RuntimeError(f"Descriptor member is incomplete: {output}.")
    member_identity = _read_json(identity_path)
    complete = validate_member_completed_output(
        output,
        expected_identity=member_identity,
    )
    expected = {
        "schema": MEMBER_SCHEMA,
        "descriptor_profile": str(profile),
        "fold": int(fold),
        "seed": int(seed),
        "window_size": WINDOW_SIZE,
        "window_stride": WINDOW_STRIDE,
        "primitive_num": PRIMITIVE_NUM,
        "outer_trial_count": OUTER_TRIAL_COUNT,
        "complete": True,
    }
    for key, value in expected.items():
        if complete.get(key) != value:
            raise RuntimeError(
                f"Descriptor member field {key!r} differs: {complete.get(key)!r} != {value!r}."
            )
    expected_strength = float(
        descriptor_profile_spec(profile).subject_nuisance_projection_strength
    )
    if not math.isclose(
        float(complete.get("subject_nuisance_projection_strength", float("nan"))),
        expected_strength,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise RuntimeError(
            "Descriptor member subject-nuisance projection strength differs."
        )
    expected_origin = (
        "fit_offline_old6_current_legacy_member"
        if profile == LEGACY_DESCRIPTOR_PROFILE
        else "reuse_completed_same_fold_seed_legacy_member"
    )
    if complete.get("codebook_origin_mode") != expected_origin:
        raise RuntimeError(
            f"Descriptor member codebook origin differs: "
            f"{complete.get('codebook_origin_mode')!r} != {expected_origin!r}."
        )
    codebook_sha = str(complete.get("codebook_state_sha256", ""))
    if len(codebook_sha) != 64:
        raise RuntimeError("Descriptor member has no valid codebook state SHA256.")
    source_identity_sha = str(
        complete.get("codebook_source_run_identity_sha256", "")
    )
    if profile == LEGACY_DESCRIPTOR_PROFILE and source_identity_sha:
        raise RuntimeError("The legacy codebook producer unexpectedly records a source member.")
    if profile != LEGACY_DESCRIPTOR_PROFILE and len(source_identity_sha) != 64:
        raise RuntimeError("A non-legacy member lacks its legacy source identity SHA256.")
    row: dict[str, Any] = {
        "descriptor_profile": str(profile),
        "fold": int(fold),
        "seed": int(seed),
        "window_size": WINDOW_SIZE,
        "window_stride": WINDOW_STRIDE,
        "primitive_num": PRIMITIVE_NUM,
        "member_dir": str(output.resolve()),
        "member_complete_sha256": sha256_file(path),
        "member_run_identity_sha256": str(complete["run_identity_sha256"]),
        "codebook_state_sha256": codebook_sha,
        "codebook_origin_mode": expected_origin,
        "codebook_source_run_identity_sha256": source_identity_sha,
        "class_recall_json": json.dumps(
            complete.get("class_recall", {}), ensure_ascii=False, sort_keys=True
        ),
    }
    for metric in (*PERFORMANCE_METRICS, *BIAS_DIAGNOSTICS):
        value = float(complete.get(metric, float("nan")))
        if not math.isfinite(value):
            raise RuntimeError(f"Member metric {metric!r} is not finite.")
        row[metric] = value
    row["subject_nuisance_rank"] = int(complete.get("subject_nuisance_rank", 0))
    row["subject_nuisance_projection_strength"] = expected_strength
    row["pre_subject_debias_transform_state_sha256"] = str(
        complete.get("pre_subject_debias_transform_state_sha256", "")
    )
    if len(row["pre_subject_debias_transform_state_sha256"]) != 64:
        raise RuntimeError("Descriptor member lacks a valid pre-debias transform SHA256.")
    row["gravity_reliable_fraction"] = float(
        complete.get("gravity_reliable_fraction", float("nan"))
    )
    return row


def _validate_shared_codebooks(
    rows: Sequence[Mapping[str, Any]],
    *,
    profiles: Sequence[str],
    folds: Sequence[int],
    seeds: Sequence[int],
) -> dict[str, Any]:
    pairs: list[dict[str, Any]] = []
    for fold in folds:
        for seed in seeds:
            local = [
                row
                for row in rows
                if int(row["fold"]) == int(fold) and int(row["seed"]) == int(seed)
            ]
            if {str(row["descriptor_profile"]) for row in local} != set(profiles):
                raise RuntimeError(
                    f"Shared-codebook audit lacks the full profile set for fold={fold}, seed={seed}."
                )
            hashes = {str(row["codebook_state_sha256"]) for row in local}
            if len(hashes) != 1:
                raise RuntimeError(
                    f"Descriptor profiles do not share one exact codebook for "
                    f"fold={fold}, seed={seed}: {sorted(hashes)}."
                )
            legacy = next(
                row
                for row in local
                if row["descriptor_profile"] == LEGACY_DESCRIPTOR_PROFILE
            )
            legacy_identity_sha = str(legacy["member_run_identity_sha256"])
            for row in local:
                if row["descriptor_profile"] == LEGACY_DESCRIPTOR_PROFILE:
                    if row["codebook_origin_mode"] != "fit_offline_old6_current_legacy_member":
                        raise RuntimeError("The legacy member is not the shared codebook producer.")
                else:
                    if (
                        row["codebook_origin_mode"]
                        != "reuse_completed_same_fold_seed_legacy_member"
                        or row["codebook_source_run_identity_sha256"]
                        != legacy_identity_sha
                    ):
                        raise RuntimeError(
                            "A non-legacy member does not reference the exact same-fold/seed "
                            "legacy producer identity."
                        )
            pairs.append(
                {
                    "fold": int(fold),
                    "seed": int(seed),
                    "codebook_state_sha256": next(iter(hashes)),
                    "legacy_member_run_identity_sha256": legacy_identity_sha,
                    "profile_count": len(local),
                }
            )
    return {
        "fit_count": len(pairs),
        "expected_fit_count": len(folds) * len(seeds),
        "one_fit_per_fold_seed": True,
        "all_profiles_share_exact_state_sha256": True,
        "pairs": pairs,
    }


def _validate_duration_soft_subject_coordinates(
    rows: Sequence[Mapping[str, Any]],
    *,
    experiment_part: str,
    folds: Sequence[int],
    seeds: Sequence[int],
) -> dict[str, Any]:
    """Prove that duration and soft arms differ only after nuisance fitting."""

    if str(experiment_part) not in (
        DURATION_SOFT_SUBJECT_EXPERIMENT_PART,
        DURATION_SOFT_SUBJECT_CONFIRM_7FOLD_EXPERIMENT_PART,
    ):
        return {"required": False, "verified": True, "pairs": []}
    compared_profiles = (
        (
            DURATION_INVARIANT_DESCRIPTOR_PROFILE,
            DURATION_SOFT_SUBJECT_A025_PROFILE,
            DURATION_SOFT_SUBJECT_A050_PROFILE,
        )
        if str(experiment_part)
        == DURATION_SOFT_SUBJECT_CONFIRM_7FOLD_EXPERIMENT_PART
        else (
            DURATION_INVARIANT_DESCRIPTOR_PROFILE,
            DURATION_SOFT_SUBJECT_A025_PROFILE,
            DURATION_SOFT_SUBJECT_A050_PROFILE,
            DURATION_SOFT_SUBJECT_A075_PROFILE,
        )
    )
    pairs: list[dict[str, Any]] = []
    for fold in folds:
        for seed in seeds:
            local = [
                row
                for row in rows
                if int(row["fold"]) == int(fold)
                and int(row["seed"]) == int(seed)
                and str(row["descriptor_profile"]) in compared_profiles
            ]
            if {str(row["descriptor_profile"]) for row in local} != set(
                compared_profiles
            ):
                raise RuntimeError(
                    "Duration/soft coordinate audit lacks a full same-member profile set."
                )
            hashes = {
                str(row["pre_subject_debias_transform_state_sha256"])
                for row in local
            }
            if len(hashes) != 1:
                raise RuntimeError(
                    "Duration and soft-subject arms do not share the exact "
                    "pre-debias descriptor coordinates."
                )
            pairs.append(
                {
                    "fold": int(fold),
                    "seed": int(seed),
                    "pre_subject_debias_transform_state_sha256": next(iter(hashes)),
                }
            )
    return {
        "required": True,
        "verified": True,
        "profiles": list(compared_profiles),
        "pairs": pairs,
    }


def _fold_means(
    rows: Sequence[Mapping[str, Any]],
    *,
    profile: str,
    metric: str,
    folds: Sequence[int],
    seeds: Sequence[int],
) -> list[float]:
    values: list[float] = []
    for fold in folds:
        local = [
            float(row[metric])
            for row in rows
            if row["descriptor_profile"] == profile and int(row["fold"]) == int(fold)
        ]
        if len(local) != len(seeds):
            raise RuntimeError(f"Profile {profile} fold {fold} lacks both seeds.")
        values.append(float(np.mean(local)))
    return values


def _contrast_fold_values(
    rows: Sequence[Mapping[str, Any]],
    *,
    coefficients: Mapping[str, float],
    metric: str,
    folds: Sequence[int],
    seeds: Sequence[int],
) -> tuple[list[float], dict[str, list[float]]]:
    lookup = {
        (str(row["descriptor_profile"]), int(row["fold"]), int(row["seed"])): float(
            row[metric]
        )
        for row in rows
    }
    same_member: dict[str, list[float]] = {}
    fold_values: list[float] = []
    for fold in folds:
        values = []
        for seed in seeds:
            try:
                value = sum(
                    float(weight) * lookup[(str(profile), int(fold), int(seed))]
                    for profile, weight in coefficients.items()
                )
            except KeyError as error:
                raise RuntimeError(
                    f"Contrast lacks a member for fold={fold}, seed={seed}."
                ) from error
            values.append(float(value))
        same_member[str(int(fold))] = values
        fold_values.append(float(np.mean(values)))
    return fold_values, same_member


def _confirmation_partition_reports(
    rows: Sequence[Mapping[str, Any]],
    *,
    experiment_part: str,
    folds: Sequence[int],
    seeds: Sequence[int],
    bootstrap_seed: int,
    bootstrap_replicates: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Separate screened folds from folds untouched by alpha selection."""

    if str(experiment_part) != DURATION_SOFT_SUBJECT_CONFIRM_7FOLD_EXPERIMENT_PART:
        return {}, []
    if tuple(folds) != CONFIRMATION_FOLDS:
        raise RuntimeError("The confirmation partition report requires folds 1..7.")
    partitions = (
        ("screening_folds_1_3", SCREENING_FOLDS, "used_for_alpha_screening"),
        (
            "confirmation_holdout_folds_4_7",
            CONFIRMATION_HOLDOUT_FOLDS,
            "untouched_by_alpha_screening_primary_confirmation",
        ),
        ("all_folds_1_7", CONFIRMATION_FOLDS, "combined_stability_summary"),
    )
    report: dict[str, Any] = {}
    csv_rows: list[dict[str, Any]] = []
    for partition_index, (partition_name, partition_folds, role) in enumerate(partitions):
        contrast_report: dict[str, Any] = {}
        for contrast_index, (contrast_name, coefficients) in enumerate(
            PART_CONTRASTS[DURATION_SOFT_SUBJECT_CONFIRM_7FOLD_EXPERIMENT_PART]
        ):
            metric_report: dict[str, Any] = {}
            for metric_index, metric in enumerate(
                (*PERFORMANCE_METRICS, *BIAS_DIAGNOSTICS)
            ):
                fold_values, same_member = _contrast_fold_values(
                    rows,
                    coefficients=coefficients,
                    metric=metric,
                    folds=partition_folds,
                    seeds=seeds,
                )
                aggregate = {
                    **bootstrap_fold_mean(
                        fold_values,
                        seed=(
                            int(bootstrap_seed)
                            + 20_000
                            + partition_index * 1_000
                            + contrast_index * 100
                            + metric_index
                        ),
                        replicates=int(bootstrap_replicates),
                    ),
                    "profile_coefficients": dict(coefficients),
                    "same_fold_same_seed_values": same_member,
                    "fold_mean_values_after_averaging_seeds": fold_values,
                }
                metric_report[metric] = aggregate
                csv_rows.append(
                    {
                        "partition": partition_name,
                        "partition_role": role,
                        "folds_json": json.dumps(list(partition_folds)),
                        "contrast": contrast_name,
                        "metric": metric,
                        "mean": aggregate["mean"],
                        "ci95_low": aggregate["ci95_low"],
                        "ci95_high": aggregate["ci95_high"],
                        "fold_values_json": json.dumps(fold_values),
                    }
                )
            contrast_report[contrast_name] = metric_report
        report[partition_name] = {
            "role": role,
            "folds": list(partition_folds),
            "fold_count": len(partition_folds),
            "contrasts": contrast_report,
        }
    return report, csv_rows


def _reports(
    root: Path,
    rows: Sequence[Mapping[str, Any]],
    *,
    experiment_part: str,
    profiles: Sequence[str],
    folds: Sequence[int],
    seeds: Sequence[int],
    identity_sha256: str,
    bootstrap_seed: int,
    bootstrap_replicates: int,
) -> dict[str, Any]:
    expected = {
        (profile, int(fold), int(seed))
        for profile in profiles for fold in folds for seed in seeds
    }
    observed = [
        (str(row["descriptor_profile"]), int(row["fold"]), int(row["seed"]))
        for row in rows
    ]
    if len(observed) != len(set(observed)) or set(observed) != expected:
        raise RuntimeError("Descriptor-ablation member grid is duplicate or incomplete.")
    shared_codebook_audit = _validate_shared_codebooks(
        rows,
        profiles=profiles,
        folds=folds,
        seeds=seeds,
    )
    duration_soft_coordinate_audit = _validate_duration_soft_subject_coordinates(
        rows,
        experiment_part=experiment_part,
        folds=folds,
        seeds=seeds,
    )
    ordered = sorted(
        (dict(row) for row in rows),
        key=lambda row: (
            list(profiles).index(str(row["descriptor_profile"])),
            int(row["fold"]),
            int(row["seed"]),
        ),
    )
    write_csv(root / "descriptor_ablation_runs.csv", ordered)

    configurations: dict[str, Any] = {}
    summary_rows: list[dict[str, Any]] = []
    metrics = (*PERFORMANCE_METRICS, *BIAS_DIAGNOSTICS)
    for profile_index, profile in enumerate(profiles):
        profile_metrics: dict[str, Any] = {}
        for metric_index, metric in enumerate(metrics):
            fold_values = _fold_means(
                ordered,
                profile=profile,
                metric=metric,
                folds=folds,
                seeds=seeds,
            )
            aggregate = {
                **bootstrap_fold_mean(
                    fold_values,
                    seed=int(bootstrap_seed) + profile_index * 100 + metric_index,
                    replicates=int(bootstrap_replicates),
                ),
                "fold_means_after_averaging_seeds": fold_values,
            }
            profile_metrics[metric] = aggregate
            summary_rows.append(
                {
                    "descriptor_profile": profile,
                    "metric": metric,
                    **{key: value for key, value in aggregate.items() if key != "fold_means_after_averaging_seeds"},
                    "fold_means_json": json.dumps(fold_values),
                }
            )
        configurations[profile] = {
            "subject_nuisance_projection_strength": float(
                descriptor_profile_spec(profile).subject_nuisance_projection_strength
            ),
            "metrics": profile_metrics,
        }
    write_csv(root / "descriptor_ablation_summary.csv", summary_rows)

    paired: dict[str, Any] = {}
    paired_rows: list[dict[str, Any]] = []
    for profile_index, profile in enumerate(profiles):
        if profile == LEGACY_DESCRIPTOR_PROFILE:
            continue
        profile_result: dict[str, Any] = {}
        for metric_index, metric in enumerate(PERFORMANCE_METRICS):
            fold_deltas: list[float] = []
            same_member: dict[str, list[float]] = {}
            for fold in folds:
                deltas = []
                for seed in seeds:
                    candidate = next(
                        float(row[metric]) for row in ordered
                        if row["descriptor_profile"] == profile
                        and int(row["fold"]) == int(fold)
                        and int(row["seed"]) == int(seed)
                    )
                    baseline = next(
                        float(row[metric]) for row in ordered
                        if row["descriptor_profile"] == LEGACY_DESCRIPTOR_PROFILE
                        and int(row["fold"]) == int(fold)
                        and int(row["seed"]) == int(seed)
                    )
                    deltas.append(candidate - baseline)
                same_member[str(fold)] = deltas
                fold_deltas.append(float(np.mean(deltas)))
            aggregate = {
                **bootstrap_fold_mean(
                    fold_deltas,
                    seed=int(bootstrap_seed) + 1000 + profile_index * 100 + metric_index,
                    replicates=int(bootstrap_replicates),
                ),
                "same_fold_same_seed_deltas": same_member,
                "fold_mean_paired_deltas_after_averaging_seeds": fold_deltas,
            }
            profile_result[metric] = aggregate
            paired_rows.append(
                {
                    "candidate_profile": profile,
                    "baseline_profile": LEGACY_DESCRIPTOR_PROFILE,
                    "metric": metric,
                    "mean_delta": aggregate["mean"],
                    "ci95_low": aggregate["ci95_low"],
                    "ci95_high": aggregate["ci95_high"],
                    "fold_deltas_json": json.dumps(fold_deltas),
                }
            )
        paired[profile] = profile_result
    write_csv(root / "descriptor_paired_vs_legacy.csv", paired_rows)
    write_json(root / "descriptor_paired_vs_legacy.json", paired)

    targeted: dict[str, Any] = {}
    targeted_rows: list[dict[str, Any]] = []
    for contrast_index, (contrast_name, coefficients) in enumerate(
        PART_CONTRASTS[str(experiment_part)]
    ):
        if not set(coefficients) <= set(profiles):
            raise RuntimeError(
                f"Contrast {contrast_name!r} is not contained in experiment part "
                f"{experiment_part!r}."
            )
        contrast_metrics: dict[str, Any] = {}
        for metric_index, metric in enumerate((*PERFORMANCE_METRICS, *BIAS_DIAGNOSTICS)):
            fold_values, same_member = _contrast_fold_values(
                ordered,
                coefficients=coefficients,
                metric=metric,
                folds=folds,
                seeds=seeds,
            )
            aggregate = {
                **bootstrap_fold_mean(
                    fold_values,
                    seed=(
                        int(bootstrap_seed)
                        + 10_000
                        + contrast_index * 100
                        + metric_index
                    ),
                    replicates=int(bootstrap_replicates),
                ),
                "profile_coefficients": dict(coefficients),
                "same_fold_same_seed_values": same_member,
                "fold_mean_values_after_averaging_seeds": fold_values,
            }
            contrast_metrics[metric] = aggregate
            targeted_rows.append(
                {
                    "experiment_part": str(experiment_part),
                    "contrast": contrast_name,
                    "metric": metric,
                    "mean": aggregate["mean"],
                    "ci95_low": aggregate["ci95_low"],
                    "ci95_high": aggregate["ci95_high"],
                    "profile_coefficients_json": json.dumps(
                        coefficients, ensure_ascii=False, sort_keys=True
                    ),
                    "fold_values_json": json.dumps(fold_values),
                }
            )
        targeted[contrast_name] = contrast_metrics
    write_csv(root / "descriptor_targeted_contrasts.csv", targeted_rows)
    write_json(root / "descriptor_targeted_contrasts.json", targeted)

    confirmation_partitions, confirmation_partition_rows = (
        _confirmation_partition_reports(
            ordered,
            experiment_part=experiment_part,
            folds=folds,
            seeds=seeds,
            bootstrap_seed=bootstrap_seed,
            bootstrap_replicates=bootstrap_replicates,
        )
    )
    if confirmation_partitions:
        write_csv(
            root / "descriptor_confirmation_partitions.csv",
            confirmation_partition_rows,
        )
        write_json(
            root / "descriptor_confirmation_partitions.json",
            confirmation_partitions,
        )

    summary = {
        "schema": SCHEMA,
        "grid_identity_sha256": str(identity_sha256),
        "experiment_role": "frozen_encoder_trajectory_descriptor_ablation",
        "experiment_part": str(experiment_part),
        "fixed_representation": "A2_E0_W128_S64_K64",
        "profiles": list(profiles),
        "folds": list(folds),
        "fold_count": len(folds),
        "seeds": list(seeds),
        "member_count": len(rows),
        "statistical_unit": "held_out_subject_fold_after_averaging_two_seeds",
        "primary_metric": "h_score",
        "global_hungarian_is_scoring_only": True,
        "known_k12_batch_proxy_not_online_cgcd": True,
        "three_fold_confidence_intervals_are_exploratory": len(folds) == 3,
        "bootstrap_intervals_describe_split_stability_not_population_inference": True,
        "shared_codebook_audit": shared_codebook_audit,
        "duration_soft_pre_debias_coordinate_audit": (
            duration_soft_coordinate_audit
        ),
        "configurations": configurations,
        "paired_against_legacy": paired,
        "targeted_contrasts": targeted,
        "confirmation_partitions": confirmation_partitions,
    }
    write_json(root / "descriptor_ablation_summary.json", summary)
    artifact_names = [
        "grid_manifest.json",
        "descriptor_ablation_runs.csv",
        "descriptor_ablation_summary.csv",
        "descriptor_ablation_summary.json",
        "descriptor_paired_vs_legacy.csv",
        "descriptor_paired_vs_legacy.json",
        "descriptor_targeted_contrasts.csv",
        "descriptor_targeted_contrasts.json",
    ]
    if confirmation_partitions:
        artifact_names.extend(
            [
                "descriptor_confirmation_partitions.csv",
                "descriptor_confirmation_partitions.json",
            ]
        )
    artifacts = {
        name: sha256_file(root / name)
        for name in artifact_names
    }
    complete = {**summary, "artifact_sha256": artifacts, "complete": True}
    write_json(root / "complete.json", complete)
    return complete


def validate_args(
    args: argparse.Namespace,
) -> tuple[
    argparse.Namespace, str, tuple[str, ...], tuple[int, ...], tuple[int, ...]
]:
    experiment_part = str(args.experiment_part)
    profiles = profiles_for_part(experiment_part)
    folds = parse_integer_grid(args.folds, minimum=1, maximum=7)
    seeds = parse_integer_grid(args.seeds, minimum=0)
    expected_folds = (
        CONFIRMATION_FOLDS
        if experiment_part == DURATION_SOFT_SUBJECT_CONFIRM_7FOLD_EXPERIMENT_PART
        else DEFAULT_FOLDS
    )
    expected_seeds = (
        CONFIRMATION_SEEDS
        if experiment_part == DURATION_SOFT_SUBJECT_CONFIRM_7FOLD_EXPERIMENT_PART
        else DEFAULT_SEEDS
    )
    if tuple(folds) != tuple(expected_folds) or tuple(seeds) != tuple(expected_seeds):
        raise ValueError(
            f"Descriptor experiment part {experiment_part!r} is fixed to folds "
            f"{tuple(expected_folds)} and seeds {tuple(expected_seeds)}."
        )
    if not Path(args.npz_path).expanduser().resolve().is_file():
        raise FileNotFoundError(Path(args.npz_path).expanduser().resolve())
    if min(
        int(args.kmeans_n_init),
        int(args.kmeans_max_iter),
        int(args.encode_batch_size),
        int(args.subject_nuisance_max_rank),
        int(args.bootstrap_replicates),
    ) < 1:
        raise ValueError("Batch, KMeans, nuisance-rank and bootstrap values must be positive.")
    if not 0.0 < float(args.signed_vertical_distance_weight) < 1.0:
        raise ValueError("signed-vertical-distance-weight must lie in (0,1).")
    if not 0.0 < float(args.subject_nuisance_explained_variance) <= 1.0:
        raise ValueError("subject-nuisance-explained-variance must lie in (0,1].")
    return args, experiment_part, profiles, tuple(folds), tuple(seeds)


def _args_for_part(args: argparse.Namespace, part: str) -> argparse.Namespace:
    values = dict(vars(args))
    values["experiment_part"] = str(part)
    values["output_root"] = str(
        Path(args.output_root).expanduser().resolve() / str(part)
    )
    return argparse.Namespace(**values)


def _suite_identity(
    args: argparse.Namespace,
) -> tuple[dict[str, Any], dict[str, argparse.Namespace]]:
    children: dict[str, argparse.Namespace] = {}
    child_identities: dict[str, dict[str, Any]] = {}
    for part in REGISTERED_EXPERIMENT_PARTS:
        child = _args_for_part(args, part)
        child, observed_part, profiles, folds, seeds = validate_args(child)
        children[part] = child
        identity = _identity(child, observed_part, profiles, folds, seeds)
        child_identities[part] = {
            "output_root": str(Path(child.output_root).resolve()),
            "member_count": int(len(profiles) * len(folds) * len(seeds)),
            "grid_identity_sha256": canonical_hash(identity),
            "grid_identity": identity,
        }
    identity = {
        "schema": SUITE_SCHEMA,
        "experiment_role": "three_independent_single_factor_descriptor_ablations",
        "parts": list(REGISTERED_EXPERIMENT_PARTS),
        "part_count": len(REGISTERED_EXPERIMENT_PARTS),
        "member_count_per_part": 12,
        "total_member_count": 12 * len(REGISTERED_EXPERIMENT_PARTS),
        "children": child_identities,
        "implementation_sha256": _implementation_hashes(),
    }
    return identity, children


def _validate_suite_complete(root: Path, *, identity_sha256: str) -> dict[str, Any]:
    summary_path = root / "suite_summary.json"
    complete_path = root / "complete.json"
    if not summary_path.is_file() or not complete_path.is_file():
        raise RuntimeError(f"Descriptor suite is incomplete: {root}.")
    summary = _read_json(summary_path)
    complete = _read_json(complete_path)
    if summary.get("schema") != SUITE_SCHEMA or complete.get("schema") != SUITE_SCHEMA:
        raise RuntimeError("Descriptor suite schema differs.")
    if complete.get("complete") is not True:
        raise RuntimeError("Descriptor suite completion flag is false.")
    if complete.get("grid_identity_sha256") != str(identity_sha256):
        raise RuntimeError("Descriptor suite identity SHA256 differs.")
    if complete.get("summary_sha256") != sha256_file(summary_path):
        raise RuntimeError("Descriptor suite summary SHA256 differs.")
    for key, value in summary.items():
        if complete.get(key) != value:
            raise RuntimeError(f"Descriptor suite summary field differs: {key}.")
    hashes = complete.get("artifact_sha256")
    expected_names = {
        "grid_manifest.json",
        "suite_summary.json",
        *(f"{part}/complete.json" for part in REGISTERED_EXPERIMENT_PARTS),
    }
    if not isinstance(hashes, Mapping) or set(hashes) != expected_names:
        raise RuntimeError("Descriptor suite artifact inventory is incomplete.")
    for name, expected_sha in hashes.items():
        path = root / str(name)
        if not path.is_file() or sha256_file(path) != expected_sha:
            raise RuntimeError(f"Descriptor suite artifact SHA256 differs: {name}.")
    return complete


def _run_suite(args: argparse.Namespace) -> dict[str, Any]:
    root = Path(args.output_root).expanduser().resolve()
    identity, children = _suite_identity(args)
    identity_sha = canonical_hash(identity)
    if bool(args.dry_run):
        commands: list[list[str]] = []
        child_dry_runs: dict[str, Any] = {}
        for part in REGISTERED_EXPERIMENT_PARTS:
            result = run(children[part])
            if result.get("member_count") != 12 or result.get("dry_run") is not True:
                raise RuntimeError(f"Suite dry-run child {part!r} is not a 12-member grid.")
            child_dry_runs[part] = {
                "output_root": str(Path(children[part].output_root).resolve()),
                "member_count": int(result["member_count"]),
                "profiles": list(result["profiles"]),
            }
            commands.extend(result["commands"])
        return {
            "schema": SUITE_SCHEMA,
            "dry_run": True,
            "parts": list(REGISTERED_EXPERIMENT_PARTS),
            "member_count_per_part": 12,
            "member_count": len(commands),
            "children": child_dry_runs,
            "commands": commands,
        }

    identity_sha = _prepare_manifest(root, identity)
    complete_path = root / "complete.json"
    if complete_path.is_file():
        if not bool(args.resume):
            raise FileExistsError(f"Completed descriptor suite exists: {root}.")
        _validate_suite_complete(root, identity_sha256=identity_sha)

    child_summaries: dict[str, Any] = {}
    for part in REGISTERED_EXPERIMENT_PARTS:
        result = run(children[part])
        child_complete_path = Path(children[part].output_root) / "complete.json"
        if result.get("complete") is not True or not child_complete_path.is_file():
            raise RuntimeError(f"Descriptor suite child {part!r} did not complete.")
        expected_child_identity = str(
            identity["children"][part]["grid_identity_sha256"]
        )
        if result.get("grid_identity_sha256") != expected_child_identity:
            raise RuntimeError(
                f"Descriptor suite child {part!r} returned another grid identity."
            )
        child_summaries[part] = {
            "output_root": str(Path(children[part].output_root).resolve()),
            "member_count": int(result["member_count"]),
            "grid_identity_sha256": str(result["grid_identity_sha256"]),
            "complete_sha256": sha256_file(child_complete_path),
        }
    summary = {
        "schema": SUITE_SCHEMA,
        "grid_identity_sha256": identity_sha,
        "parts": list(REGISTERED_EXPERIMENT_PARTS),
        "part_count": len(REGISTERED_EXPERIMENT_PARTS),
        "member_count_per_part": 12,
        "member_count": sum(
            int(value["member_count"]) for value in child_summaries.values()
        ),
        "independent_single_factor_children": True,
        "children": child_summaries,
    }
    write_json(root / "suite_summary.json", summary)
    artifact_names = (
        "grid_manifest.json",
        "suite_summary.json",
        *(f"{part}/complete.json" for part in REGISTERED_EXPERIMENT_PARTS),
    )
    complete = {
        **summary,
        "summary_sha256": sha256_file(root / "suite_summary.json"),
        "artifact_sha256": {
            name: sha256_file(root / name) for name in artifact_names
        },
        "complete": True,
    }
    write_json(complete_path, complete)
    return complete


def run(args: argparse.Namespace) -> dict[str, Any]:
    if str(args.experiment_part) == ALL_EXPERIMENT_PARTS:
        return _run_suite(args)
    args, experiment_part, profiles, folds, seeds = validate_args(args)
    root = Path(args.output_root).expanduser().resolve()
    identity = _identity(args, experiment_part, profiles, folds, seeds)
    identity_sha = canonical_hash(identity)
    if bool(args.dry_run):
        commands = []
        encoder_root = Path(args.encoder_source_root).expanduser().resolve()
        for profile in profiles:
            for fold in folds:
                for seed in seeds:
                    output = member_directory(root / "members" / profile, fold, seed)
                    legacy_output = member_directory(
                        root / "members" / LEGACY_DESCRIPTOR_PROFILE, fold, seed
                    )
                    commands.append(
                        _member_command(
                            args,
                            profile=profile,
                            fold=fold,
                            seed=seed,
                            checkpoint=_checkpoint_path(encoder_root, fold, seed),
                            output=output,
                            reuse_codebook_run_dir=(
                                None
                                if profile == LEGACY_DESCRIPTOR_PROFILE
                                else legacy_output
                            ),
                        )
                    )
        return {
            "schema": SCHEMA,
            "dry_run": True,
            "experiment_part": experiment_part,
            "profiles": list(profiles),
            "member_count": len(commands),
            "commands": commands,
        }

    identity_sha = _prepare_manifest(root, identity)
    complete_path = root / "complete.json"
    if complete_path.is_file() and not bool(args.resume):
        raise FileExistsError(f"Completed descriptor ablation exists: {root}.")
    encoder_root = Path(args.encoder_source_root).expanduser().resolve()
    rows: list[dict[str, Any]] = []
    for profile in profiles:
        for fold in folds:
            for seed in seeds:
                checkpoint = _checkpoint_path(encoder_root, fold, seed)
                output = member_directory(root / "members" / profile, fold, seed)
                legacy_output = member_directory(
                    root / "members" / LEGACY_DESCRIPTOR_PROFILE, fold, seed
                )
                command = _member_command(
                    args,
                    profile=profile,
                    fold=fold,
                    seed=seed,
                    checkpoint=checkpoint,
                    output=output,
                    reuse_codebook_run_dir=(
                        None
                        if profile == LEGACY_DESCRIPTOR_PROFILE
                        else legacy_output
                    ),
                )
                _run_command(
                    command,
                    stage=f"profile={profile} fold={fold} seed={seed}",
                )
                rows.append(
                    _validated_member(
                        output, profile=profile, fold=fold, seed=seed
                    )
                )
                write_csv(root / "descriptor_ablation_runs.partial.csv", rows)
    return _reports(
        root,
        rows,
        experiment_part=experiment_part,
        profiles=profiles,
        folds=folds,
        seeds=seeds,
        identity_sha256=identity_sha,
        bootstrap_seed=int(args.bootstrap_seed),
        bootstrap_replicates=int(args.bootstrap_replicates),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Reuse registered frozen W128 A2 encoders for one W128/K64 "
            "trajectory-descriptor experiment part. Historical parts remain "
            "3 folds x 2 seeds; the explicit confirmation part is 7 folds x 2 seeds."
        )
    )
    parser.add_argument(
        "--npz-path",
        default=str(
            PROJECT_ROOT
            / "processed"
            / "uschad_w128_s64_train17stats"
            / "uschad_windows.npz"
        ),
    )
    parser.add_argument("--encoder-source-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument(
        "--experiment-part",
        required=True,
        choices=CLI_EXPERIMENT_PARTS,
        help=(
            "gravity: legacy vs signed gravity only; subject_debias: the no-gravity "
            "legacy vs source-subject nuisance-projection experiment; duration: "
            "legacy vs removal of registered absolute-duration fields; "
            "duration_soft_subject: duration-invariant direct baseline vs fixed "
            "soft strengths 0.25/0.50/0.75 (legacy remains codebook producer); "
            "duration_soft_subject_confirm_7fold: isolated seven-fold confirmation "
            "of duration vs soft strengths 0.25/0.50; "
            "all: run only the original three independent 12-member parts under "
            "one output root."
        ),
    )
    parser.add_argument("--folds", default="1,2,3")
    parser.add_argument("--seeds", default="0,5")
    parser.add_argument("--signed-vertical-distance-weight", type=float, default=0.15)
    parser.add_argument("--subject-nuisance-max-rank", type=int, default=4)
    parser.add_argument(
        "--subject-nuisance-explained-variance", type=float, default=0.90
    )
    parser.add_argument("--kmeans-n-init", type=int, default=50)
    parser.add_argument("--kmeans-max-iter", type=int, default=300)
    parser.add_argument("--encode-batch-size", type=int, default=1024)
    parser.add_argument("--bootstrap-seed", type=int, default=20260915)
    parser.add_argument("--bootstrap-replicates", type=int, default=10_000)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> dict[str, Any]:
    result = run(build_parser().parse_args(argv))
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return result


if __name__ == "__main__":
    main()


__all__ = [
    "ALL_EXPERIMENT_PARTS",
    "CLI_EXPERIMENT_PARTS",
    "DEFAULT_FOLDS",
    "DEFAULT_SEEDS",
    "CONFIRMATION_FOLDS",
    "CONFIRMATION_HOLDOUT_FOLDS",
    "CONFIRMATION_SEEDS",
    "DURATION_EXPERIMENT_PART",
    "DURATION_SOFT_SUBJECT_CONFIRM_7FOLD_EXPERIMENT_PART",
    "DURATION_SOFT_SUBJECT_EXPERIMENT_PART",
    "EXPERIMENT_PART_PROFILES",
    "GRAVITY_EXPERIMENT_PART",
    "MEMBER_SCHEMA",
    "PRIMITIVE_NUM",
    "REGISTERED_EXPERIMENT_PARTS",
    "SCHEMA",
    "SCREENING_FOLDS",
    "SUBJECT_DEBIAS_EXPERIMENT_PART",
    "SUITE_SCHEMA",
    "WINDOW_SIZE",
    "WINDOW_STRIDE",
    "build_parser",
    "main",
    "profiles_for_part",
    "run",
    "validate_args",
]
