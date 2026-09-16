"""Variable-resolution A2/E0 trajectory batch-GCD mechanism probe.

This experiment intentionally has no old/novel gate and no sequential online
updates.  A fold-specific A2 encoder and an E0 codebook are fitted only from
offline old-class data.  Afterwards all 120 trials of the two outer subjects
are represented and clustered together as one unlabeled transductive batch.

The activity cluster count (12) and the motion-primitive codebook capacity
(32/64/128) are different quantities.  Activity truth is opened only after raw
cluster IDs have been written and hashed; Hungarian alignment is scorer-only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import sys
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import sklearn
import torch
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from experiments.motion_primitive.frozen_e0 import (
    ABSOLUTE_DURATION_DESCRIPTOR_NAMES,
    DESCRIPTOR_PROFILES,
    FrozenE0Codebook,
    LEGACY_DESCRIPTOR_PROFILE,
    PrimitiveTrajectory,
    SIGNED_VERTICAL_NAMES,
    DescriptorTransform,
    descriptor_matrix,
    descriptor_profile_spec,
    fit_descriptor_transform,
    fit_frozen_e0_codebook,
    fit_subject_nuisance_projection,
    gravity_aligned_trial_state,
    load_frozen_artifacts,
    raw_descriptor_dimension,
    save_frozen_artifacts,
    statistic_names,
    window_ownership_partition,
    window_statistics,
)
from experiments.motion_primitive.motion_checkpoint import motion_state_dict_sha256
from experiments.motion_primitive.strict_artifacts import (
    encode_sensor_trials,
    load_frozen_a2_encoder,
    write_csv,
    write_json,
)
from experiments.motion_primitive.strict_cv_common import canonical_hash
from experiments.motion_primitive.strict_metrics import strict_three_layer_metrics
from experiments.motion_primitive.strict_protocol import (
    SensorTrial,
    build_registered_protocol,
    sha256_file,
)


SCHEMA = "hhr_window_codebook_batch_proxy_v3"
IDENTITY_SCHEMA = "hhr_window_codebook_batch_proxy_identity_v3"
PROFILE = "a2_e0_state_variable_k_transductive_batch_gcd"
ARM_BY_PAIR = frozenset({(64, 128), (128, 64), (128, 128), (256, 32)})
OLD_CLASS_COUNT = 6
TOTAL_CLASS_COUNT = 12
OUTER_TRIAL_COUNT = 120
LOGGER = logging.getLogger("hhr_window_codebook_batch_proxy")


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON artifact is not an object: {path}.")
    return value


def _subject_identity_paths(value: Any, path: tuple[str, ...] = ()) -> list[str]:
    """Find explicit subject-identity fields in a pre-truth artifact."""

    found: list[str] = []
    if isinstance(value, Mapping):
        for raw_key, item in value.items():
            key = str(raw_key)
            tokens = tuple(
                token
                for token in key.lower().replace("-", "_").split("_")
                if token
            )
            if (
                any(token in {"subject", "subjects"} for token in tokens)
                and any(
                    token in {"id", "ids", "identity", "identities"}
                    for token in tokens
                )
            ):
                found.append(".".join((*path, key)))
            found.extend(_subject_identity_paths(item, (*path, key)))
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            found.extend(_subject_identity_paths(item, (*path, str(index))))
    return found


def _array_sha256(*arrays: np.ndarray) -> str:
    digest = hashlib.sha256()
    for index, value in enumerate(arrays):
        array = np.ascontiguousarray(np.asarray(value))
        if array.dtype.kind == "f" and not np.all(np.isfinite(array)):
            raise ValueError("Cannot hash a non-finite array.")
        digest.update(str(index).encode("ascii"))
        digest.update(array.dtype.str.encode("ascii"))
        digest.update(str(array.shape).encode("ascii"))
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _ids_sha256(values: Sequence[int]) -> str:
    return _array_sha256(np.asarray(values, dtype="<i8"))


def _atomic_npz(path: Path, **arrays: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    temporary.replace(path)


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False, allow_nan=False) + "\n")
    temporary.replace(path)


def _resolve_device(value: str) -> torch.device:
    if str(value).lower() == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")
    return device


def _codebook_state_sha256(state: FrozenE0Codebook) -> str:
    state.validate(strict_historical=(state.primitive_num, state.pca_dim) == (32, 64))
    metadata = np.asarray(
        [
            state.primitive_num,
            state.pca_dim,
            state.input_dim,
            state.seed,
            state.fit_window_count,
            state.fit_trial_count,
            state.fit_subject_count,
            state.kmeans_n_iter,
        ],
        dtype="<i8",
    )
    return _array_sha256(
        metadata,
        np.asarray(state.pca_mean, dtype="<f4"),
        np.asarray(state.pca_components, dtype="<f4"),
        np.asarray(state.cluster_centers, dtype="<f4"),
    )


def _transform_state_sha256(state: DescriptorTransform) -> str:
    state_arrays = [
        np.asarray(state.keep_columns, dtype="<i8"),
        np.asarray(state.mean, dtype="<f8"),
        np.asarray(state.scale, dtype="<f8"),
        np.asarray([], dtype="<f8") if state.pca_mean is None else np.asarray(state.pca_mean, dtype="<f8"),
        np.empty((0, 0), dtype="<f8")
        if state.pca_components is None
        else np.asarray(state.pca_components, dtype="<f8"),
        np.asarray([], dtype="<i8")
        if state.protected_columns is None
        else np.asarray(state.protected_columns, dtype="<i8"),
        np.asarray([], dtype="<f8")
        if state.protected_mean is None
        else np.asarray(state.protected_mean, dtype="<f8"),
        np.asarray([], dtype="<f8")
        if state.protected_scale is None
        else np.asarray(state.protected_scale, dtype="<f8"),
        np.asarray([state.protected_distance_weight], dtype="<f8"),
        np.empty((0, state.output_dim), dtype="<f8")
        if state.nuisance_basis is None
        else np.asarray(state.nuisance_basis, dtype="<f8"),
        np.asarray([], dtype="<f8")
        if state.nuisance_singular_values is None
        else np.asarray(state.nuisance_singular_values, dtype="<f8"),
        np.asarray(
            [
                state.nuisance_explained_fraction,
                state.nuisance_fit_subject_count,
                state.nuisance_fit_trial_count,
                state.nuisance_projection_strength,
            ],
            dtype="<f8",
        ),
    ]
    return _array_sha256(*state_arrays)


def _implementation_hashes() -> dict[str, str]:
    names = (
        "experiments/motion_primitive/window_codebook_batch_proxy.py",
        "experiments/motion_primitive/frozen_e0.py",
        "experiments/motion_primitive/core.py",
        "experiments/motion_primitive/strict_protocol.py",
        "experiments/motion_primitive/strict_artifacts.py",
        "experiments/motion_primitive/strict_metrics.py",
        "experiments/motion_primitive/motion_checkpoint.py",
        "experiments/motion_primitive/motion_encoder.py",
        "models/resnet1d.py",
    )
    return {name: sha256_file(PROJECT_ROOT / name) for name in names}


def _identity(args: argparse.Namespace) -> dict[str, Any]:
    npz_path = Path(args.npz_path).expanduser().resolve()
    checkpoint = Path(args.a2_checkpoint).expanduser().resolve()
    if not npz_path.is_file():
        raise FileNotFoundError(npz_path)
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    descriptor_spec = descriptor_profile_spec(args.descriptor_profile)
    npz_sha256 = sha256_file(npz_path)
    checkpoint_sha256 = sha256_file(checkpoint)
    codebook_origin = _validated_reuse_codebook_reference(
        args,
        npz_sha256=npz_sha256,
        checkpoint_sha256=checkpoint_sha256,
    )
    return {
        "schema": IDENTITY_SCHEMA,
        "profile": PROFILE,
        "fold": int(args.fold),
        "seed": int(args.seed),
        "npz_path": str(npz_path),
        "npz_sha256": npz_sha256,
        "a2_checkpoint_path": str(checkpoint),
        "a2_checkpoint_sha256": checkpoint_sha256,
        "codebook_origin": codebook_origin,
        "representation": {
            "window_size": int(args.window_size),
            "window_stride": int(args.window_stride),
            "window_seconds": float(args.window_size) / 100.0,
            "stride_seconds": float(args.window_stride) / 100.0,
            "primitive_num": int(args.primitive_num),
            "primitive_pca_dim": int(args.pca_dim),
            "raw_state_descriptor_dim": raw_descriptor_dimension(
                int(args.primitive_num),
                include_state=True,
                descriptor_profile=descriptor_spec.name,
            ),
            "descriptor_pca_dim": int(args.descriptor_pca_dim),
            "descriptor_profile": descriptor_spec.name,
            "signed_vertical": bool(descriptor_spec.signed_vertical),
            "duration_invariant": bool(descriptor_spec.duration_invariant),
            "subject_debias": bool(descriptor_spec.subject_debias),
            "signed_vertical_feature_count": (
                len(SIGNED_VERTICAL_NAMES) if descriptor_spec.signed_vertical else 0
            ),
            "signed_vertical_distance_weight": (
                float(args.signed_vertical_distance_weight)
                if descriptor_spec.signed_vertical else 0.0
            ),
            "subject_nuisance_max_rank": (
                int(args.subject_nuisance_max_rank)
                if descriptor_spec.subject_debias else 0
            ),
            "subject_nuisance_explained_variance": (
                float(args.subject_nuisance_explained_variance)
                if descriptor_spec.subject_debias else 0.0
            ),
            "subject_nuisance_projection_strength": (
                float(descriptor_spec.subject_nuisance_projection_strength)
                if descriptor_spec.subject_debias else 0.0
            ),
        },
        "batch_gcd": {
            "outer_trial_count": OUTER_TRIAL_COUNT,
            "activity_cluster_count": int(args.activity_cluster_count),
            "fit_scope": "all_outer_subject_trials_unlabeled_transductive",
            "gate": "none",
            "sequential_sessions": False,
            "activity_class_count_known": True,
            "hungarian": "post_hoc_scoring_only",
            "kmeans_n_init": int(args.kmeans_n_init),
            "kmeans_max_iter": int(args.kmeans_max_iter),
        },
        "runtime": {
            "encode_batch_size": int(args.encode_batch_size),
            "requested_device": str(args.device),
            "resolved_device": str(_resolve_device(args.device)),
            "allow_smoke_a2": bool(args.allow_smoke_a2),
            "numpy": np.__version__,
            "sklearn": sklearn.__version__,
            "torch": torch.__version__,
        },
        "implementation_sha256": _implementation_hashes(),
    }


def _verified_identity(path: Path) -> dict[str, Any]:
    value = _read_json(path)
    body = {key: item for key, item in value.items() if key != "identity_sha256"}
    if canonical_hash(body) != value.get("identity_sha256"):
        raise RuntimeError(f"Identity SHA256 does not verify: {path}.")
    return value


def _validated_reuse_codebook_reference(
    args: argparse.Namespace,
    *,
    npz_sha256: str,
    checkpoint_sha256: str,
) -> dict[str, Any]:
    """Resolve and fully validate the sole external codebook provenance.

    The registered descriptor ablation fits the E0 codebook exactly once in
    its legacy member.  Every non-legacy member must reuse that completed
    member byte-for-byte; matching seeds are deliberately not accepted as a
    substitute for matching state hashes.
    """

    raw_source = str(getattr(args, "reuse_codebook_run_dir", "") or "").strip()
    descriptor_profile = str(args.descriptor_profile)
    if not raw_source:
        if descriptor_profile != LEGACY_DESCRIPTOR_PROFILE:
            raise ValueError(
                "A non-legacy descriptor profile must provide "
                "--reuse-codebook-run-dir pointing to its completed same-fold/seed "
                "legacy_state_v1 member."
            )
        return {
            "mode": "fit_offline_old6_current_legacy_member",
            "source_member_dir": None,
        }
    if descriptor_profile == LEGACY_DESCRIPTOR_PROFILE:
        raise ValueError("legacy_state_v1 is the codebook producer and cannot reuse another member.")

    source = Path(raw_source).expanduser().resolve()
    current_output = Path(args.output_dir).expanduser().resolve()
    if source == current_output:
        raise ValueError("A member cannot reuse its own output directory as a codebook source.")
    identity_path = source / "run_identity.json"
    complete_path = source / "complete.json"
    representation_path = source / "frozen_representation.npz"
    if not all(path.is_file() for path in (identity_path, complete_path, representation_path)):
        raise FileNotFoundError(
            "The reuse-codebook source is not a completed legacy member: "
            f"{source}."
        )

    source_identity = _verified_identity(identity_path)
    source_complete = validate_completed_output(
        source,
        expected_identity=source_identity,
    )
    if source_identity.get("schema") != IDENTITY_SCHEMA:
        raise RuntimeError("The reuse-codebook source identity schema differs.")
    if source_complete.get("schema") != SCHEMA:
        raise RuntimeError("The reuse-codebook source completion schema differs.")
    if source_complete.get("descriptor_profile") != LEGACY_DESCRIPTOR_PROFILE:
        raise RuntimeError("Only a legacy_state_v1 member may produce the shared codebook.")
    if source_complete.get("codebook_origin_mode") != "fit_offline_old6_current_legacy_member":
        raise RuntimeError("The shared-codebook source did not fit its own registered legacy codebook.")

    representation = source_identity.get("representation")
    if not isinstance(representation, Mapping):
        raise RuntimeError("The shared-codebook source has no representation identity.")
    expected_fields = {
        "fold": int(args.fold),
        "seed": int(args.seed),
        "npz_sha256": str(npz_sha256),
        "a2_checkpoint_sha256": str(checkpoint_sha256),
    }
    for key, expected in expected_fields.items():
        if source_identity.get(key) != expected:
            raise RuntimeError(
                f"Shared-codebook source identity field {key!r} differs: "
                f"{source_identity.get(key)!r} != {expected!r}."
            )
    expected_representation = {
        "window_size": int(args.window_size),
        "window_stride": int(args.window_stride),
        "primitive_num": int(args.primitive_num),
        "primitive_pca_dim": int(args.pca_dim),
        "descriptor_profile": LEGACY_DESCRIPTOR_PROFILE,
    }
    for key, expected in expected_representation.items():
        if representation.get(key) != expected:
            raise RuntimeError(
                f"Shared-codebook source representation field {key!r} differs: "
                f"{representation.get(key)!r} != {expected!r}."
            )
    if source_identity.get("implementation_sha256") != _implementation_hashes():
        raise RuntimeError("The shared-codebook source was produced by another implementation.")

    artifact_hashes = source_complete.get("artifact_sha256")
    if not isinstance(artifact_hashes, Mapping):
        raise RuntimeError("The shared-codebook source has no artifact hash inventory.")
    representation_sha256 = str(artifact_hashes.get("frozen_representation.npz", ""))
    if not representation_sha256 or sha256_file(representation_path) != representation_sha256:
        raise RuntimeError("The shared frozen representation SHA256 does not verify.")
    codebook_sha256 = str(source_complete.get("codebook_state_sha256", ""))
    if len(codebook_sha256) != 64:
        raise RuntimeError("The shared-codebook source does not expose a valid state SHA256.")
    codebook, _, _, metadata = load_frozen_artifacts(representation_path)
    if _codebook_state_sha256(codebook) != codebook_sha256:
        raise RuntimeError("The loaded shared codebook state differs from its completion record.")
    if metadata.get("descriptor_profile") != LEGACY_DESCRIPTOR_PROFILE:
        raise RuntimeError("The shared frozen artifact is not the legacy descriptor member.")

    return {
        "mode": "reuse_completed_same_fold_seed_legacy_member",
        "source_member_dir": str(source),
        "source_run_identity_sha256": str(source_identity["identity_sha256"]),
        "source_complete_sha256": sha256_file(complete_path),
        "source_frozen_representation_sha256": representation_sha256,
        "expected_codebook_state_sha256": codebook_sha256,
    }


def _prepare_identity(output: Path, args: argparse.Namespace) -> dict[str, Any]:
    body = _identity(args)
    expected = {**body, "identity_sha256": canonical_hash(body)}
    path = output / "run_identity.json"
    if path.is_file():
        if _verified_identity(path) != expected:
            raise RuntimeError(f"Output records another experiment identity: {output}.")
        return expected
    if output.exists():
        allowed = {"cv_member_manifest.json"}
        unexpected = [item.name for item in output.iterdir() if item.name not in allowed]
        if unexpected:
            raise RuntimeError(
                f"Unidentified non-empty member directory cannot be adopted: {unexpected}."
            )
    output.mkdir(parents=True, exist_ok=True)
    write_json(path, expected)
    return expected


def outer_batch_trials(protocol: Any) -> tuple[SensorTrial, ...]:
    """Recover the complete outer-subject pool without opening activity truth."""

    by_id: dict[int, SensorTrial] = {}
    for session in tuple(protocol.sessions):
        for trial in tuple(session.incoming):
            trial_id = int(trial.trial_id)
            if trial_id in by_id:
                raise RuntimeError("A trial was repeated between registered incoming cohorts.")
            by_id[trial_id] = trial
    for trial in tuple(protocol.sessions[-1].evaluation):
        trial_id = int(trial.trial_id)
        if trial_id in by_id:
            raise RuntimeError("Final evaluation overlaps cumulative incoming trials.")
        by_id[trial_id] = trial
    ordered = tuple(by_id[value] for value in sorted(by_id))
    if len(ordered) != OUTER_TRIAL_COUNT:
        raise RuntimeError(
            f"Expected all {OUTER_TRIAL_COUNT} outer-subject trials, observed {len(ordered)}."
        )
    return ordered


def validate_scorer_truth(
    trials: Sequence[SensorTrial],
    targets: Sequence[int] | np.ndarray,
    subjects: Sequence[int] | np.ndarray,
    *,
    expected_outer_subjects: Sequence[int],
) -> dict[str, Any]:
    """Fail closed on scorer-only truth alignment after predictions are frozen."""

    target_values = np.asarray(targets, dtype=np.int64)
    subject_values = np.asarray(subjects, dtype=np.int64)
    if target_values.shape != (OUTER_TRIAL_COUNT,) or subject_values.shape != (
        OUTER_TRIAL_COUNT,
    ):
        raise RuntimeError("Scorer truth does not contain exactly 120 aligned rows.")
    if len(trials) != OUTER_TRIAL_COUNT:
        raise RuntimeError("Scorer received a non-canonical outer trial batch.")

    learner_subjects = np.asarray(
        [int(item.subject_id) for item in trials], dtype=np.int64
    )
    if not np.array_equal(subject_values, learner_subjects):
        raise RuntimeError("TruthStore subject IDs are not row-aligned with SensorTrial.")

    expected_subjects = tuple(sorted(int(value) for value in expected_outer_subjects))
    if len(expected_subjects) != 2 or len(set(expected_subjects)) != 2:
        raise RuntimeError("Scorer expected exactly two distinct outer subjects.")
    if set(subject_values.tolist()) != set(expected_subjects):
        raise RuntimeError("Scorer truth does not cover exactly the two outer subjects.")
    if set(target_values.tolist()) != set(range(TOTAL_CLASS_COUNT)):
        raise RuntimeError("Scorer truth does not cover exactly activity labels 0..11.")

    class_counts = np.bincount(target_values, minlength=TOTAL_CLASS_COUNT)
    if not np.array_equal(
        class_counts, np.full(TOTAL_CLASS_COUNT, 10, dtype=np.int64)
    ):
        raise RuntimeError(
            f"Scorer truth must contain ten trials per activity: {class_counts.tolist()}."
        )
    subject_class_counts: dict[str, list[int]] = {}
    for subject in expected_subjects:
        local = np.bincount(
            target_values[subject_values == int(subject)], minlength=TOTAL_CLASS_COUNT
        )
        if not np.array_equal(
            local, np.full(TOTAL_CLASS_COUNT, 5, dtype=np.int64)
        ):
            raise RuntimeError(
                "Scorer truth must contain five trials per activity for every "
                f"outer subject; subject={subject}, counts={local.tolist()}."
            )
        subject_class_counts[str(subject)] = local.astype(int).tolist()
    return {
        "row_subject_ids_match_learner_trials": True,
        "class_counts": class_counts.astype(int).tolist(),
        "subject_class_counts": subject_class_counts,
    }


def _tokenize_trial(
    encoded: Any,
    codebook: FrozenE0Codebook,
    *,
    include_gravity: bool = False,
) -> PrimitiveTrajectory:
    raw_windows = np.asarray(encoded.raw_windows, dtype=np.float32)
    window_starts = np.asarray(encoded.window_starts, dtype=np.int64)
    starts, ends = window_ownership_partition(
        window_starts,
        int(raw_windows.shape[2]),
    )
    tokens, distances, embedded = codebook.assign(encoded.content_embeddings)
    return PrimitiveTrajectory(
        trial_id=int(encoded.trial_id),
        subject_id=int(encoded.subject_id),
        starts=starts,
        ends=ends,
        tokens=tokens,
        distances=distances,
        embeddings=embedded,
        statistics=window_statistics(
            raw_windows, starts, ends, sample_rate_hz=100.0
        ),
        statistic_names=statistic_names(),
        gravity_aligned=(
            gravity_aligned_trial_state(raw_windows, window_starts)
            if include_gravity else None
        ),
    ).validate(codebook.primitive_num)


def _primitive_trials(
    encoder: torch.nn.Module,
    trials: Sequence[SensorTrial],
    codebook: FrozenE0Codebook,
    *,
    device: torch.device,
    batch_size: int,
    include_gravity: bool = False,
) -> list[PrimitiveTrajectory]:
    encoded = encode_sensor_trials(
        encoder, trials, device=device, batch_size=int(batch_size)
    )
    return [
        _tokenize_trial(item, codebook, include_gravity=include_gravity)
        for item in encoded
    ]


def _fit_codebook(
    encoded_trials: Sequence[Any],
    *,
    primitive_num: int,
    pca_dim: int,
    seed: int,
) -> FrozenE0Codebook:
    content = np.concatenate(
        [np.asarray(item.content_embeddings, dtype=np.float32) for item in encoded_trials]
    )
    trial_ids = np.concatenate(
        [np.full(len(item.content_embeddings), int(item.trial_id), dtype=np.int64) for item in encoded_trials]
    )
    subject_ids = np.concatenate(
        [np.full(len(item.content_embeddings), int(item.subject_id), dtype=np.int64) for item in encoded_trials]
    )
    return fit_frozen_e0_codebook(
        content,
        trial_ids,
        subject_ids,
        primitive_num=int(primitive_num),
        pca_dim=int(pca_dim),
        seed=int(seed),
        strict_historical=(int(primitive_num), int(pca_dim)) == (32, 64),
    )


def _load_reused_codebook(
    origin: Mapping[str, Any],
    *,
    primitive_num: int,
    pca_dim: int,
    seed: int,
) -> FrozenE0Codebook:
    if origin.get("mode") != "reuse_completed_same_fold_seed_legacy_member":
        raise RuntimeError("A reused codebook has an invalid provenance mode.")
    source = Path(str(origin.get("source_member_dir", ""))).expanduser().resolve()
    artifact = source / "frozen_representation.npz"
    source_complete = source / "complete.json"
    expected_complete_sha = str(origin.get("source_complete_sha256", ""))
    if (
        not source_complete.is_file()
        or not expected_complete_sha
        or sha256_file(source_complete) != expected_complete_sha
    ):
        raise RuntimeError("The shared legacy completion record changed before loading.")
    expected_artifact_sha = str(origin.get("source_frozen_representation_sha256", ""))
    if not artifact.is_file() or sha256_file(artifact) != expected_artifact_sha:
        raise RuntimeError("The shared legacy frozen representation changed before loading.")
    codebook, _, _, _ = load_frozen_artifacts(artifact)
    codebook.validate(
        strict_historical=(int(primitive_num), int(pca_dim)) == (32, 64)
    )
    expected = (
        int(primitive_num),
        int(pca_dim),
        int(seed),
    )
    observed = (
        int(codebook.primitive_num),
        int(codebook.pca_dim),
        int(codebook.seed),
    )
    if observed != expected:
        raise RuntimeError(
            f"Shared codebook K/PCA/seed differs: observed={observed}, expected={expected}."
        )
    actual_state_sha = _codebook_state_sha256(codebook)
    if actual_state_sha != origin.get("expected_codebook_state_sha256"):
        raise RuntimeError("The shared legacy codebook state SHA256 changed before use.")
    return codebook


def _primitive_usage(
    trials: Sequence[PrimitiveTrajectory], primitive_num: int
) -> dict[str, Any]:
    tokens = np.concatenate([np.asarray(item.tokens, dtype=np.int64) for item in trials])
    counts = np.bincount(tokens, minlength=int(primitive_num)).astype(np.int64)
    fractions = counts / max(float(counts.sum()), 1.0)
    active = fractions > 0
    entropy = float(-np.sum(fractions[active] * np.log(fractions[active])))
    used = int(np.count_nonzero(counts))
    unique_per_trial = np.asarray(
        [len(np.unique(item.tokens)) for item in trials], dtype=np.float64
    )
    distance = np.concatenate([np.asarray(item.distances, dtype=np.float64) for item in trials])
    return {
        "capacity_k": int(primitive_num),
        "used_k": used,
        "dead_k": int(primitive_num) - used,
        "dead_fraction": float((int(primitive_num) - used) / int(primitive_num)),
        "effective_k": float(np.exp(entropy)),
        "token_count": int(counts.sum()),
        "counts": counts.tolist(),
        "fractions": fractions.tolist(),
        "unique_k_per_trial_mean": float(unique_per_trial.mean()),
        "unique_k_per_trial_min": int(unique_per_trial.min()),
        "unique_k_per_trial_max": int(unique_per_trial.max()),
        "quantization_distance_mean": float(distance.mean()),
        "quantization_distance_std": float(distance.std()),
    }


def _gravity_quality(trajectories: Sequence[PrimitiveTrajectory]) -> dict[str, Any]:
    states = [item.gravity_aligned for item in trajectories]
    if any(state is None for state in states):
        raise RuntimeError("A tokenized trajectory lacks its gravity audit state.")
    reliable = np.asarray([bool(state.reliable) for state in states], dtype=bool)
    norms = np.asarray([float(state.gravity_norm_g) for state in states], dtype=np.float64)
    angles = np.asarray(
        [float(state.edge_angle_degrees) for state in states], dtype=np.float64
    )
    return {
        "trial_count": int(len(states)),
        "reliable_count": int(reliable.sum()),
        "reliable_fraction": float(reliable.mean()),
        "gravity_norm_g_median": float(np.median(norms)),
        "gravity_norm_g_min": float(norms.min()),
        "gravity_norm_g_max": float(norms.max()),
        "edge_angle_degrees_median": float(np.median(angles)),
        "edge_angle_degrees_q95": float(np.quantile(angles, 0.95)),
    }


def _maximum_duration_correlation(
    features: np.ndarray, trajectories: Sequence[PrimitiveTrajectory]
) -> float:
    matrix = np.asarray(features, dtype=np.float64)
    duration = np.log1p(
        np.asarray([int(item.ends[-1]) for item in trajectories], dtype=np.float64)
    )
    duration -= duration.mean()
    feature_centered = matrix - matrix.mean(axis=0, keepdims=True)
    denominator = np.linalg.norm(feature_centered, axis=0) * np.linalg.norm(duration)
    correlations = np.divide(
        duration @ feature_centered,
        denominator,
        out=np.zeros(matrix.shape[1], dtype=np.float64),
        where=denominator > 1e-12,
    )
    return float(np.max(np.abs(correlations)))


def _subject_centroid_dispersion(
    features: np.ndarray,
    subjects: Sequence[int] | np.ndarray,
) -> float:
    matrix = np.asarray(features, dtype=np.float64)
    subject_values = np.asarray(subjects, dtype=np.int64)
    centroids = np.stack(
        [matrix[subject_values == value].mean(axis=0) for value in np.unique(subject_values)]
    )
    effects = centroids - centroids.mean(axis=0, keepdims=True)
    return float(np.mean(np.linalg.norm(effects, axis=1)))


def _trajectory_rows(
    trials: Sequence[SensorTrial], trajectories: Sequence[PrimitiveTrajectory]
) -> list[dict[str, Any]]:
    rows = []
    for sensor, trajectory in zip(trials, trajectories):
        if int(sensor.trial_id) != int(trajectory.trial_id):
            raise RuntimeError("Sensor/trajectory row order differs.")
        rows.append(
            {
                "role": "outer_all_unlabeled_transductive_batch",
                "trial_id": int(sensor.trial_id),
                "window_count": int(len(trajectory.tokens)),
                "primitive_occurrence_count": int(len(trajectory.tokens)),
                "unique_primitive_type_count": int(len(np.unique(trajectory.tokens))),
                "used_primitive_ids": sorted(np.unique(trajectory.tokens).astype(int).tolist()),
                "primitive_sequence": trajectory.tokens.astype(int).tolist(),
                "ownership_start_samples": trajectory.starts.astype(int).tolist(),
                "ownership_end_samples_exclusive": trajectory.ends.astype(int).tolist(),
                "original_window_start_samples": np.asarray(sensor.window_starts, dtype=int).tolist(),
                "quantization_distances": trajectory.distances.astype(float).tolist(),
                "signed_vertical_descriptor": (
                    trajectory.gravity_aligned.descriptor.astype(float).tolist()
                    if trajectory.gravity_aligned is not None else []
                ),
                "gravity_norm_g_audit_only": (
                    float(trajectory.gravity_aligned.gravity_norm_g)
                    if trajectory.gravity_aligned is not None else None
                ),
                "gravity_edge_angle_degrees_audit_only": (
                    float(trajectory.gravity_aligned.edge_angle_degrees)
                    if trajectory.gravity_aligned is not None else None
                ),
                "gravity_reliable_audit_only": (
                    bool(trajectory.gravity_aligned.reliable)
                    if trajectory.gravity_aligned is not None else False
                ),
            }
        )
    return rows


def trajectory_time_raster_seconds(
    trajectories: Sequence[PrimitiveTrajectory],
    *,
    sample_rate_hz: float = 100.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Rasterize token ownership spans on a common physical-time axis.

    Each raster column represents one sensor sample.  The returned edge vector
    is expressed in seconds, so window grids with different lengths/strides can
    be plotted against the same physical unit.  NaN is used only after a trial's
    final observable ownership interval.
    """

    if not trajectories:
        raise ValueError("At least one trajectory is required for time rasterization.")
    rate = float(sample_rate_hz)
    if not np.isfinite(rate) or rate <= 0.0:
        raise ValueError("sample_rate_hz must be finite and positive.")

    rows: list[np.ndarray] = []
    maximum_end = 0
    for trajectory in trajectories:
        starts = np.asarray(trajectory.starts, dtype=np.int64)
        ends = np.asarray(trajectory.ends, dtype=np.int64)
        tokens = np.asarray(trajectory.tokens, dtype=np.int64)
        if (
            starts.ndim != 1
            or ends.shape != starts.shape
            or tokens.shape != starts.shape
            or not len(starts)
        ):
            raise ValueError("Trajectory ownership/token arrays are malformed.")
        if (
            int(starts[0]) != 0
            or np.any(ends <= starts)
            or np.any(starts[1:] != ends[:-1])
        ):
            raise ValueError(
                "Trajectory ownership must be a contiguous partition beginning at zero."
            )
        durations = ends - starts
        row = np.repeat(tokens, durations).astype(np.float32, copy=False)
        if len(row) != int(ends[-1]):
            raise RuntimeError("Trajectory time raster does not match its final endpoint.")
        rows.append(row)
        maximum_end = max(maximum_end, int(ends[-1]))

    raster = np.full((len(rows), maximum_end), np.nan, dtype=np.float32)
    for index, row in enumerate(rows):
        raster[index, : len(row)] = row
    time_edges_seconds = np.arange(maximum_end + 1, dtype=np.float64) / rate
    return raster, time_edges_seconds


def _save_visuals(
    output: Path,
    *,
    trajectories: Sequence[PrimitiveTrajectory],
    targets: np.ndarray,
    activity_names: Sequence[str],
    subjects: np.ndarray,
    global_metrics: Mapping[str, Any],
    primitive_num: int,
) -> tuple[str, ...]:
    visual_dir = output / "visualizations"
    visual_dir.mkdir(parents=True, exist_ok=True)
    class_names = []
    for class_id in range(TOTAL_CLASS_COUNT):
        found = [str(name) for name, target in zip(activity_names, targets) if int(target) == class_id]
        class_names.append(found[0] if found else f"class_{class_id}")

    fractions = np.stack(
        [
            np.bincount(item.tokens, minlength=int(primitive_num)).astype(np.float64)
            / max(float(len(item.tokens)), 1.0)
            for item in trajectories
        ]
    )
    heatmap = np.zeros((TOTAL_CLASS_COUNT, int(primitive_num)), dtype=np.float64)
    for class_id in range(TOTAL_CLASS_COUNT):
        per_subject = []
        for subject in sorted(np.unique(subjects[targets == class_id]).astype(int).tolist()):
            selected = (targets == class_id) & (subjects == subject)
            per_subject.append(fractions[selected].mean(axis=0))
        heatmap[class_id] = np.mean(per_subject, axis=0)
    fig, axis = plt.subplots(figsize=(max(12.0, primitive_num / 8.0), 6.5))
    image = axis.imshow(heatmap, aspect="auto", interpolation="nearest", cmap="magma")
    axis.set_xlabel(f"Motion primitive ID (capacity K={primitive_num}; member-local IDs)")
    axis.set_ylabel("Activity")
    axis.set_yticks(np.arange(TOTAL_CLASS_COUNT), labels=class_names)
    fig.colorbar(image, ax=axis, label="Subject-balanced token fraction")
    fig.tight_layout()
    heatmap_path = visual_dir / "activity_primitive_heatmap.png"
    fig.savefig(heatmap_path, dpi=180)
    plt.close(fig)

    order = np.lexsort(
        (
            np.asarray([item.trial_id for item in trajectories], dtype=np.int64),
            subjects,
            targets,
        )
    )
    ordered_trajectories = [trajectories[index] for index in order.tolist()]
    sequence, time_edges_seconds = trajectory_time_raster_seconds(
        ordered_trajectories,
        sample_rate_hz=100.0,
    )
    cmap = plt.get_cmap("turbo").copy()
    cmap.set_bad("white")
    fig, axis = plt.subplots(figsize=(15.0, 9.0))
    image = axis.imshow(
        sequence,
        aspect="auto",
        interpolation="nearest",
        cmap=cmap,
        vmin=0,
        vmax=max(1, int(primitive_num) - 1),
        extent=(
            float(time_edges_seconds[0]),
            float(time_edges_seconds[-1]),
            float(len(order)) - 0.5,
            -0.5,
        ),
    )
    sorted_targets = targets[order]
    boundaries = np.flatnonzero(np.r_[True, sorted_targets[1:] != sorted_targets[:-1], True])
    midpoints = 0.5 * (boundaries[:-1] + boundaries[1:] - 1)
    axis.set_yticks(midpoints, labels=class_names)
    for boundary in boundaries[1:-1]:
        axis.axhline(boundary - 0.5, color="black", linewidth=0.4)
    axis.set_xlabel(
        "Physical time since trial start (seconds; 100 Hz raster; white = trial ended)"
    )
    axis.set_ylabel("Activity-grouped outer trials")
    fig.colorbar(image, ax=axis, label="Member-local primitive ID")
    fig.tight_layout()
    sequence_path = visual_dir / "trajectory_sequences.png"
    fig.savefig(sequence_path, dpi=180)
    plt.close(fig)

    confusion = np.asarray(
        global_metrics["confusion_matrix_with_unknown_column"], dtype=np.int64
    )[:, :TOTAL_CLASS_COUNT]
    fig, axis = plt.subplots(figsize=(9.0, 8.0))
    image = axis.imshow(confusion, interpolation="nearest", cmap="Blues")
    axis.set_xticks(np.arange(TOTAL_CLASS_COUNT), labels=class_names, rotation=45, ha="right")
    axis.set_yticks(np.arange(TOTAL_CLASS_COUNT), labels=class_names)
    axis.set_xlabel("Global-Hungarian prediction (scoring only)")
    axis.set_ylabel("True activity")
    for row in range(TOTAL_CLASS_COUNT):
        for column in range(TOTAL_CLASS_COUNT):
            axis.text(column, row, str(int(confusion[row, column])), ha="center", va="center", fontsize=7)
    fig.colorbar(image, ax=axis, label="Trial count")
    fig.tight_layout()
    confusion_path = visual_dir / "global_hungarian_confusion.png"
    fig.savefig(confusion_path, dpi=180)
    plt.close(fig)
    return (
        str(heatmap_path.relative_to(output)),
        str(sequence_path.relative_to(output)),
        str(confusion_path.relative_to(output)),
    )


def _output_artifacts() -> tuple[str, ...]:
    return (
        "activity_cluster.json",
        "activity_cluster.npz",
        "codebook_usage.json",
        "descriptor_bias_audit.json",
        "fit_manifest.json",
        "frozen_representation.npz",
        "leakage_audit.json",
        "metrics.json",
        "predictions.csv",
        "raw_predictions.npz",
        "summary.json",
        "trajectories_label_free.jsonl",
        "visualizations/activity_primitive_heatmap.png",
        "visualizations/global_hungarian_confusion.png",
        "visualizations/trajectory_sequences.png",
    )


def validate_completed_output(
    output: Path, *, expected_identity: Mapping[str, Any]
) -> dict[str, Any]:
    identity_path = output / "run_identity.json"
    complete_path = output / "complete.json"
    summary_path = output / "summary.json"
    if not all(item.is_file() for item in (identity_path, complete_path, summary_path)):
        raise RuntimeError(f"Batch-proxy member is incomplete: {output}.")
    if _verified_identity(identity_path) != dict(expected_identity):
        raise RuntimeError("Member run identity changed after execution.")
    complete = _read_json(complete_path)
    summary = _read_json(summary_path)
    if complete.get("schema") != SCHEMA or summary.get("schema") != SCHEMA:
        raise RuntimeError("Member schema differs.")
    if complete.get("complete") is not True:
        raise RuntimeError("Member completion flag is false.")
    if complete.get("run_identity_sha256") != expected_identity.get("identity_sha256"):
        raise RuntimeError("Member completion identity SHA256 mismatch.")
    if complete.get("summary_sha256") != sha256_file(summary_path):
        raise RuntimeError("Member summary SHA256 mismatch.")
    for key, value in summary.items():
        if complete.get(key) != value:
            raise RuntimeError(f"Member summary/completion field differs: {key}.")
    hashes = complete.get("artifact_sha256")
    if not isinstance(hashes, Mapping) or set(hashes) != set(_output_artifacts()):
        raise RuntimeError("Member artifact inventory is incomplete.")
    for name, expected in hashes.items():
        path = output / str(name)
        if not path.is_file() or sha256_file(path) != expected:
            raise RuntimeError(f"Member artifact SHA256 mismatch: {name}.")
    if int(summary.get("outer_trial_count", -1)) != OUTER_TRIAL_COUNT:
        raise RuntimeError("Completed member does not contain all 120 outer trials.")
    codebook_sha = str(summary.get("codebook_state_sha256", ""))
    if len(codebook_sha) != 64:
        raise RuntimeError("Completed member does not expose a valid codebook state SHA256.")
    transform_sha = str(summary.get("descriptor_transform_state_sha256", ""))
    pre_transform_sha = str(
        summary.get("pre_subject_debias_transform_state_sha256", "")
    )
    if len(transform_sha) != 64 or len(pre_transform_sha) != 64:
        raise RuntimeError(
            "Completed member does not expose valid descriptor-transform SHA256 values."
        )
    expected_representation = expected_identity.get("representation", {})
    expected_strength = float(
        expected_representation.get("subject_nuisance_projection_strength", 0.0)
    )
    if not math.isclose(
        float(summary.get("subject_nuisance_projection_strength", float("nan"))),
        expected_strength,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise RuntimeError("Completed member subject-nuisance strength differs.")
    raw_prediction_path = output / "raw_predictions.npz"
    raw_prediction_sha = sha256_file(raw_prediction_path)
    if str(summary.get("raw_predictions_sha256", "")) != raw_prediction_sha:
        raise RuntimeError(
            "Final raw predictions no longer match the pre-truth frozen SHA256."
        )
    with np.load(raw_prediction_path, allow_pickle=False) as raw:
        forbidden_raw_keys = _subject_identity_paths({name: None for name in raw.files})
        if forbidden_raw_keys:
            raise RuntimeError(
                "Pre-truth raw predictions contain forbidden outer subject identity "
                f"fields: {forbidden_raw_keys}."
            )
        if str(np.asarray(raw["codebook_state_sha256"]).item()) != codebook_sha:
            raise RuntimeError("Raw-prediction and summary codebook SHA256 differ.")
        if str(np.asarray(raw["transform_state_sha256"]).item()) != transform_sha:
            raise RuntimeError("Raw-prediction and summary transform SHA256 differ.")
        if (
            str(
                np.asarray(raw["pre_subject_debias_transform_state_sha256"]).item()
            )
            != pre_transform_sha
        ):
            raise RuntimeError("Raw-prediction and summary pre-debias SHA256 differ.")
        if not math.isclose(
            float(np.asarray(raw["subject_nuisance_projection_strength"]).item()),
            expected_strength,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise RuntimeError("Raw-prediction subject-nuisance strength differs.")
    codebook, restored_transform, _, _ = load_frozen_artifacts(
        output / "frozen_representation.npz"
    )
    if _codebook_state_sha256(codebook) != codebook_sha:
        raise RuntimeError("Frozen-representation and summary codebook SHA256 differ.")
    if _transform_state_sha256(restored_transform) != transform_sha:
        raise RuntimeError("Frozen-representation and summary transform SHA256 differ.")
    fit_manifest = _read_json(output / "fit_manifest.json")
    forbidden_manifest_paths = _subject_identity_paths(fit_manifest)
    if forbidden_manifest_paths:
        raise RuntimeError(
            "Pre-truth fit manifest contains forbidden outer subject identity "
            f"fields: {forbidden_manifest_paths}."
        )
    if fit_manifest.get("descriptor_transform_state_sha256") != transform_sha:
        raise RuntimeError("Fit manifest and summary transform SHA256 differ.")
    if (
        fit_manifest.get("pre_subject_debias_transform_state_sha256")
        != pre_transform_sha
    ):
        raise RuntimeError("Fit manifest and summary pre-debias SHA256 differ.")
    with (output / "trajectories_label_free.jsonl").open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            forbidden_row_paths = _subject_identity_paths(row)
            if forbidden_row_paths:
                raise RuntimeError(
                    "Pre-truth label-free trajectories contain forbidden outer subject "
                    f"identity fields: {forbidden_row_paths}."
                )
    bias_audit = _read_json(output / "descriptor_bias_audit.json")
    if bias_audit.get("raw_predictions_sha256_before_subject_diagnostic") != raw_prediction_sha:
        raise RuntimeError("Subject diagnostics do not reference the frozen pre-truth predictions.")
    leakage_audit = _read_json(output / "leakage_audit.json")
    if leakage_audit.get("raw_predictions_sha256") != raw_prediction_sha:
        raise RuntimeError("Leakage audit does not reference the frozen pre-truth predictions.")
    return complete


def validate_args(args: argparse.Namespace) -> argparse.Namespace:
    descriptor_spec = descriptor_profile_spec(args.descriptor_profile)
    reuse_source = str(getattr(args, "reuse_codebook_run_dir", "") or "").strip()
    if descriptor_spec.name == LEGACY_DESCRIPTOR_PROFILE and reuse_source:
        raise ValueError("legacy_state_v1 must fit the one shared codebook itself.")
    if descriptor_spec.name != LEGACY_DESCRIPTOR_PROFILE and not reuse_source:
        raise ValueError(
            "Every non-legacy descriptor profile must reuse the completed "
            "same-fold/seed legacy_state_v1 codebook."
        )
    pair = (int(args.window_size), int(args.primitive_num))
    if pair not in ARM_BY_PAIR:
        raise ValueError(f"Unsupported scale/vocabulary pair {pair}; expected {sorted(ARM_BY_PAIR)}.")
    if int(args.window_stride) * 2 != int(args.window_size):
        raise ValueError("Every experiment arm requires exactly 50% window overlap.")
    if int(args.pca_dim) != 64:
        raise ValueError("This controlled experiment keeps primitive PCA fixed at 64.")
    if int(args.activity_cluster_count) != TOTAL_CLASS_COUNT:
        raise ValueError("This known-class-count batch proxy is fixed to 12 activity clusters.")
    if int(args.descriptor_pca_dim) != 32:
        raise ValueError("This controlled experiment keeps trajectory PCA fixed at 32.")
    if int(args.fold) not in range(1, 8) or int(args.seed) < 0:
        raise ValueError("Fold/seed is outside the supported range.")
    if min(
        int(args.kmeans_n_init),
        int(args.kmeans_max_iter),
        int(args.encode_batch_size),
        int(args.subject_nuisance_max_rank),
    ) < 1:
        raise ValueError("KMeans and encoding batch parameters must be positive.")
    if not 0.0 < float(args.signed_vertical_distance_weight) < 1.0:
        raise ValueError("signed-vertical-distance-weight must lie in (0,1).")
    if not 0.0 < float(args.subject_nuisance_explained_variance) <= 1.0:
        raise ValueError("subject-nuisance-explained-variance must lie in (0,1].")
    _resolve_device(args.device)
    return args


def run(args: argparse.Namespace) -> dict[str, Any]:
    args = validate_args(args)
    descriptor_spec = descriptor_profile_spec(args.descriptor_profile)
    output = Path(args.output_dir).expanduser().resolve()
    if output.exists() and not bool(args.resume):
        raise FileExistsError(f"Output already exists: {output}.")
    identity = _prepare_identity(output, args)
    if (output / "complete.json").is_file():
        if not bool(args.resume):
            raise FileExistsError(f"Completed output already exists: {output}.")
        return validate_completed_output(output, expected_identity=identity)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.FileHandler(output / "batch_proxy.log", encoding="utf-8"),
            logging.StreamHandler(),
        ],
        force=True,
    )
    protocol = build_registered_protocol(
        args.npz_path,
        fold=int(args.fold),
        seed=int(args.seed),
        window_size=int(args.window_size),
        stride=int(args.window_stride),
        shuffle_novel_classes=False,
    )
    if protocol.npz_sha256 != identity["npz_sha256"]:
        raise RuntimeError("Protocol NPZ differs from the run identity.")
    device = _resolve_device(args.device)
    encoder, encoder_audit = load_frozen_a2_encoder(
        args.a2_checkpoint,
        expected_npz_sha256=protocol.npz_sha256,
        expected_fold=int(args.fold),
        expected_seed=int(args.seed),
        expected_split=protocol.split,
        expected_window_size=int(args.window_size),
        allow_smoke=bool(args.allow_smoke_a2),
    )
    initial_encoder_sha = str(encoder_audit["model_state_dict_sha256"])

    encoded_offline = encode_sensor_trials(
        encoder,
        protocol.offline_train,
        device=device,
        batch_size=int(args.encode_batch_size),
    )
    codebook_origin = identity.get("codebook_origin")
    if not isinstance(codebook_origin, Mapping):
        raise RuntimeError("Run identity lacks codebook provenance.")
    if codebook_origin.get("mode") == "fit_offline_old6_current_legacy_member":
        LOGGER.info(
            "fitting the shared offline old6 E0 codebook: W=%d S=%d primitive_K=%d",
            int(args.window_size),
            int(args.window_stride),
            int(args.primitive_num),
        )
        codebook = _fit_codebook(
            encoded_offline,
            primitive_num=int(args.primitive_num),
            pca_dim=int(args.pca_dim),
            seed=int(args.seed),
        )
    else:
        LOGGER.info(
            "reusing the completed same-fold/seed legacy E0 codebook: %s",
            codebook_origin.get("source_member_dir"),
        )
        codebook = _load_reused_codebook(
            codebook_origin,
            primitive_num=int(args.primitive_num),
            pca_dim=int(args.pca_dim),
            seed=int(args.seed),
        )
    offline_trajectories = [
        _tokenize_trial(
            item, codebook, include_gravity=descriptor_spec.signed_vertical
        )
        for item in encoded_offline
    ]

    batch_trials = outer_batch_trials(protocol)
    offline_ids = {int(item.trial_id) for item in protocol.offline_train}
    outer_ids = [int(item.trial_id) for item in batch_trials]
    if offline_ids & set(outer_ids):
        raise RuntimeError("Offline and outer batch trial IDs overlap.")
    LOGGER.info("encoding one transductive outer batch with %d complete trials", len(batch_trials))
    outer_trajectories = _primitive_trials(
        encoder,
        batch_trials,
        codebook,
        device=device,
        batch_size=int(args.encode_batch_size),
        include_gravity=descriptor_spec.signed_vertical,
    )
    raw_descriptors, descriptor_names = descriptor_matrix(
        outer_trajectories,
        int(args.primitive_num),
        include_state=True,
        descriptor_profile=descriptor_spec.name,
    )
    expected_raw_dim = raw_descriptor_dimension(
        int(args.primitive_num),
        include_state=True,
        descriptor_profile=descriptor_spec.name,
    )
    if raw_descriptors.shape != (OUTER_TRIAL_COUNT, expected_raw_dim):
        raise RuntimeError("Outer trajectory descriptor shape changed.")
    forbidden_duration = sorted(
        ABSOLUTE_DURATION_DESCRIPTOR_NAMES.intersection(descriptor_names)
    )
    if descriptor_spec.duration_invariant and forbidden_duration:
        raise RuntimeError(
            f"Duration-invariant descriptor retained forbidden fields {forbidden_duration}."
        )
    protected_columns = np.asarray(
        [
            index
            for index, name in enumerate(descriptor_names)
            if name in SIGNED_VERTICAL_NAMES
        ],
        dtype=np.int64,
    )
    expected_protected = len(SIGNED_VERTICAL_NAMES) if descriptor_spec.signed_vertical else 0
    if len(protected_columns) != expected_protected:
        raise RuntimeError("Signed-vertical protected block is incomplete.")
    transform = fit_descriptor_transform(
        raw_descriptors,
        maximum_components=int(args.descriptor_pca_dim),
        protected_columns=protected_columns,
        protected_distance_weight=float(args.signed_vertical_distance_weight),
    )
    pre_subject_debias_transform_sha = _transform_state_sha256(transform)
    features_before_subject_debias = transform.transform(raw_descriptors)
    nuisance_audit: dict[str, Any] = {
        "mode": "none",
        "target_subject_ids_required": False,
        "activity_labels_used": False,
        "selected_rank": 0,
        "projection_strength": 0.0,
    }
    if descriptor_spec.subject_debias:
        offline_raw_descriptors, offline_names = descriptor_matrix(
            offline_trajectories,
            int(args.primitive_num),
            include_state=True,
            descriptor_profile=descriptor_spec.name,
        )
        if offline_names != descriptor_names:
            raise RuntimeError("Offline and outer descriptor schemas differ.")
        offline_subjects = np.asarray(
            [item.subject_id for item in offline_trajectories], dtype=np.int64
        )
        source_counts = np.unique(offline_subjects, return_counts=True)[1]
        if len(set(source_counts.astype(int).tolist())) != 1:
            raise RuntimeError(
                "Source subject-centroid debiasing requires a balanced old-class mixture."
            )
        transform, nuisance_audit = fit_subject_nuisance_projection(
            transform,
            offline_raw_descriptors,
            offline_subjects,
            maximum_rank=int(args.subject_nuisance_max_rank),
            explained_variance=float(args.subject_nuisance_explained_variance),
            projection_strength=float(
                descriptor_spec.subject_nuisance_projection_strength
            ),
        )
    features = transform.transform(raw_descriptors)
    if not np.all(np.isfinite(features)) or np.any(np.linalg.norm(features, axis=1) <= 1e-8):
        raise RuntimeError("Batch trajectory transform produced invalid features.")

    duration_correlation_before = _maximum_duration_correlation(
        features_before_subject_debias, outer_trajectories
    )
    duration_correlation_final = _maximum_duration_correlation(
        features, outer_trajectories
    )
    gravity_quality = (
        {
            "offline_train": _gravity_quality(offline_trajectories),
            "outer_all": _gravity_quality(outer_trajectories),
        }
        if descriptor_spec.signed_vertical
        else {
            "offline_train": {"computed": False, "reliable_fraction": 0.0},
            "outer_all": {"computed": False, "reliable_fraction": 0.0},
        }
    )

    model = KMeans(
        n_clusters=int(args.activity_cluster_count),
        n_init=int(args.kmeans_n_init),
        max_iter=int(args.kmeans_max_iter),
        random_state=int(args.seed),
        algorithm="lloyd",
    )
    raw_cluster_ids = model.fit_predict(features).astype(np.int64)
    if set(np.unique(raw_cluster_ids).tolist()) != set(range(TOTAL_CLASS_COUNT)):
        raise RuntimeError("Activity KMeans did not use all 12 requested clusters.")
    centers = np.asarray(model.cluster_centers_, dtype=np.float32)
    cluster_state_sha = _array_sha256(
        np.asarray(
            [TOTAL_CLASS_COUNT, int(args.seed), int(model.n_iter_)], dtype="<i8"
        ),
        centers,
    )

    codebook_sha = _codebook_state_sha256(codebook)
    transform_sha = _transform_state_sha256(transform)
    fit_manifest = {
        "schema": SCHEMA,
        "fold": int(args.fold),
        "seed": int(args.seed),
        "offline_codebook_fit_scope": (
            "300_train_subject_old_class_trials_only_current_legacy_member"
            if codebook_origin.get("mode") == "fit_offline_old6_current_legacy_member"
            else "reused_completed_same_fold_seed_legacy_member"
        ),
        "codebook_origin": dict(codebook_origin),
        "codebook_state_sha256": codebook_sha,
        "descriptor_transform_state_sha256": transform_sha,
        "offline_codebook_fit_trial_count": int(codebook.fit_trial_count),
        "offline_codebook_fit_window_count": int(codebook.fit_window_count),
        "offline_codebook_fit_subject_count": int(codebook.fit_subject_count),
        "outer_batch_fit_scope": "all_120_outer_subject_trials_unlabeled_transductive",
        "outer_batch_trial_ids": outer_ids,
        "outer_batch_trial_ids_sha256": _ids_sha256(outer_ids),
        "activity_labels_present_in_fit_api": False,
        "window_size": int(args.window_size),
        "window_stride": int(args.window_stride),
        "primitive_num": int(args.primitive_num),
        "descriptor_profile": descriptor_spec.name,
        "raw_descriptor_dim": int(raw_descriptors.shape[1]),
        "constant_filter_kept_dim": int(len(transform.keep_columns)),
        "protected_vertical_dim": int(len(protected_columns)),
        "subject_nuisance_rank": int(
            0 if transform.nuisance_basis is None else len(transform.nuisance_basis)
        ),
        "subject_nuisance_projection_strength": (
            float(transform.nuisance_projection_strength)
            if transform.nuisance_basis is not None else 0.0
        ),
        "pre_subject_debias_transform_state_sha256": (
            pre_subject_debias_transform_sha
        ),
        "descriptor_output_dim": int(transform.output_dim),
    }
    write_json(output / "fit_manifest.json", fit_manifest)
    usage = {
        "offline_train": _primitive_usage(offline_trajectories, int(args.primitive_num)),
        "outer_all": _primitive_usage(outer_trajectories, int(args.primitive_num)),
    }
    write_json(output / "codebook_usage.json", usage)
    _write_jsonl(
        output / "trajectories_label_free.jsonl",
        _trajectory_rows(batch_trials, outer_trajectories),
    )
    save_frozen_artifacts(
        output / "frozen_representation.npz",
        codebook,
        transform,
        descriptor_names,
        {
            "profile": PROFILE,
            "descriptor_profile": descriptor_spec.name,
            "fold": int(args.fold),
            "seed": int(args.seed),
            "window_size": int(args.window_size),
            "window_stride": int(args.window_stride),
            "primitive_num": int(args.primitive_num),
            "codebook_state_sha256": codebook_sha,
            "codebook_origin": dict(codebook_origin),
            "descriptor_transform_state_sha256": transform_sha,
            "pre_subject_debias_transform_state_sha256": (
                pre_subject_debias_transform_sha
            ),
            "descriptor_fit_scope": "all_outer_unlabeled_transductive_batch",
            "subject_nuisance_basis_fit_scope": (
                "offline_train_old6_balanced_subject_centroids"
                if descriptor_spec.subject_debias else "none"
            ),
            "subject_nuisance_projection_strength": (
                float(descriptor_spec.subject_nuisance_projection_strength)
                if descriptor_spec.subject_debias else 0.0
            ),
        },
    )
    restored_codebook, restored_transform, restored_names, _ = load_frozen_artifacts(
        output / "frozen_representation.npz"
    )
    if (
        _codebook_state_sha256(restored_codebook) != codebook_sha
        or _transform_state_sha256(restored_transform) != transform_sha
        or restored_names != descriptor_names
    ):
        raise RuntimeError("Frozen variable-K representation did not round-trip exactly.")
    _atomic_npz(
        output / "activity_cluster.npz",
        centers=centers,
        raw_cluster_ids=raw_cluster_ids,
        trial_ids=np.asarray(outer_ids, dtype=np.int64),
    )
    write_json(
        output / "activity_cluster.json",
        {
            "schema": SCHEMA,
            "descriptor_profile": descriptor_spec.name,
            "algorithm": "sklearn_KMeans_lloyd",
            "fit_scope": "all_120_outer_subject_trials_unlabeled_transductive",
            "known_activity_cluster_count": TOTAL_CLASS_COUNT,
            "n_init": int(args.kmeans_n_init),
            "max_iter": int(args.kmeans_max_iter),
            "iterations": int(model.n_iter_),
            "inertia": float(model.inertia_),
            "feature_dim": int(features.shape[1]),
            "state_sha256": cluster_state_sha,
        },
    )

    # Durable learner/scorer boundary: truth has not been joined above.
    _atomic_npz(
        output / "raw_predictions.npz",
        trial_ids=np.asarray(outer_ids, dtype=np.int64),
        trajectory_features=np.asarray(features, dtype=np.float32),
        raw_activity_cluster_ids=raw_cluster_ids,
        descriptor_profile=np.asarray(descriptor_spec.name),
        subject_nuisance_projection_strength=np.asarray(
            float(descriptor_spec.subject_nuisance_projection_strength)
            if descriptor_spec.subject_debias else 0.0,
            dtype=np.float64,
        ),
        primitive_num=np.asarray(int(args.primitive_num), dtype=np.int64),
        codebook_state_sha256=np.asarray(codebook_sha),
        transform_state_sha256=np.asarray(transform_sha),
        pre_subject_debias_transform_state_sha256=np.asarray(
            pre_subject_debias_transform_sha
        ),
        cluster_state_sha256=np.asarray(cluster_state_sha),
    )
    raw_prediction_sha = sha256_file(output / "raw_predictions.npz")

    # Scorer-only truth access starts here.
    targets, activity_names, subjects = protocol.truth.join(outer_ids)
    target_values = np.asarray(targets, dtype=np.int64)
    subject_values = np.asarray(subjects, dtype=np.int64)
    scorer_truth_audit = validate_scorer_truth(
        batch_trials,
        target_values,
        subject_values,
        expected_outer_subjects=protocol.split.outer_test,
    )
    descriptor_bias_audit = {
        "schema": SCHEMA,
        "descriptor_profile": descriptor_spec.name,
        "absolute_duration_fields_forbidden": bool(descriptor_spec.duration_invariant),
        "forbidden_absolute_duration_fields_present": forbidden_duration,
        "uniform_time_scale_invariance_is_unit_tested": bool(
            descriptor_spec.duration_invariant
        ),
        "duration_correlation_is_diagnostic_not_a_fit_input": True,
        "outer_max_abs_feature_log_duration_correlation_before_subject_debias": (
            duration_correlation_before
        ),
        "outer_max_abs_feature_log_duration_correlation_final": (
            duration_correlation_final
        ),
        "outer_subject_centroid_dispersion_before_diagnostic_only": (
            _subject_centroid_dispersion(
                features_before_subject_debias, subject_values
            )
        ),
        "outer_subject_centroid_dispersion_final_diagnostic_only": (
            _subject_centroid_dispersion(features, subject_values)
        ),
        "outer_subject_ids_used_to_fit_features_or_projection": False,
        "outer_subject_diagnostic_after_raw_predictions_frozen": True,
        "raw_predictions_sha256_before_subject_diagnostic": raw_prediction_sha,
        "coordinate_transform_fit_scope": "all_outer_unlabeled_descriptors",
        "subject_nuisance_basis_fit_scope": (
            "offline_old6_balanced_source_subject_centroids"
            if descriptor_spec.subject_debias else "none"
        ),
        "subject_nuisance_projection_strength": (
            float(descriptor_spec.subject_nuisance_projection_strength)
            if descriptor_spec.subject_debias else 0.0
        ),
        "signed_vertical_block_bypasses_structural_pca": bool(
            descriptor_spec.signed_vertical
        ),
        "signed_vertical_distance_weight": (
            float(args.signed_vertical_distance_weight)
            if descriptor_spec.signed_vertical else 0.0
        ),
        "gravity_quality": gravity_quality,
        "subject_nuisance_projection": nuisance_audit,
        "claim_boundary": (
            "The coordinate transform is fitted on all outer unlabeled descriptors; "
            "only the nuisance basis uses offline source-subject metadata. The "
            "profile removes registered absolute-duration fields and lowers a "
            "source-estimated subject subspace, but does not prove that duration "
            "or identity is statistically unrecoverable."
        ),
    }
    write_json(output / "descriptor_bias_audit.json", descriptor_bias_audit)
    score_bundle = strict_three_layer_metrics(
        target_values,
        raw_cluster_ids,
        class_count=TOTAL_CLASS_COUNT,
        old_class_count=OLD_CLASS_COUNT,
        seen_class_count_before_session=TOTAL_CLASS_COUNT,
        registered_class_ids=tuple(range(TOTAL_CLASS_COUNT)),
    )
    global_metrics = score_bundle["layers"]["global_hungarian_upper_bound"]
    ari = float(adjusted_rand_score(target_values, raw_cluster_ids))
    nmi = float(normalized_mutual_info_score(target_values, raw_cluster_ids))
    confusion = np.asarray(
        global_metrics["confusion_matrix_with_unknown_column"], dtype=np.int64
    )
    class_recall_values = np.diag(confusion[:, :TOTAL_CLASS_COUNT]) / np.maximum(
        confusion.sum(axis=1), 1
    )
    class_name_by_id = {
        int(label): str(name)
        for label, name in zip(target_values.tolist(), activity_names)
    }
    class_recall = {
        class_name_by_id[class_id]: float(class_recall_values[class_id])
        for class_id in range(TOTAL_CLASS_COUNT)
    }
    metrics = {
        "schema": SCHEMA,
        "protocol": "transductive_batch_gcd_known_activity_k12",
        "primary_layer": "global_hungarian_scoring_only",
        "raw_predictions_sha256": raw_prediction_sha,
        "raw_predictions_frozen_before_truth_join": True,
        "adjusted_rand_index": ari,
        "normalized_mutual_information": nmi,
        "layers": score_bundle["layers"],
    }
    write_json(output / "metrics.json", metrics)
    aligned = np.asarray(global_metrics["aligned_predictions"], dtype=np.int64)
    write_csv(
        output / "predictions.csv",
        [
            {
                "trial_id": int(trial.trial_id),
                "subject_id": int(subject_values[index]),
                "activity_label": int(target_values[index]),
                "activity_name": str(activity_names[index]),
                "raw_cluster_id": int(raw_cluster_ids[index]),
                "global_hungarian_prediction": int(aligned[index]),
                "global_hungarian_correct": bool(aligned[index] == target_values[index]),
            }
            for index, trial in enumerate(batch_trials)
        ],
    )
    visual_paths = _save_visuals(
        output,
        trajectories=outer_trajectories,
        targets=target_values,
        activity_names=activity_names,
        subjects=subject_values,
        global_metrics=global_metrics,
        primitive_num=int(args.primitive_num),
    )

    final_encoder_sha = motion_state_dict_sha256(
        {key: value.detach().cpu() for key, value in encoder.state_dict().items()}
    )
    if final_encoder_sha != initial_encoder_sha:
        raise RuntimeError("Frozen A2 parameters or BatchNorm buffers changed during the proxy.")
    leakage_audit = {
        "schema": SCHEMA,
        "offline_encoder_uses_old_class_train_and_validation_only": True,
        "primitive_codebook_fit_uses_offline_train_old6_only": True,
        "primitive_codebook_origin": dict(codebook_origin),
        "descriptor_profile": descriptor_spec.name,
        "absolute_duration_fields_removed": bool(descriptor_spec.duration_invariant),
        "trajectory_coordinate_transform_fit_scope": "all_outer_unlabeled_descriptors",
        "subject_nuisance_basis_fit_scope": (
            "offline_train_old6_source_subject_metadata_only"
            if descriptor_spec.subject_debias else "none"
        ),
        "subject_nuisance_projection_strength": (
            float(descriptor_spec.subject_nuisance_projection_strength)
            if descriptor_spec.subject_debias else 0.0
        ),
        "outer_subject_ids_used_by_subject_nuisance_fit": False,
        "outer_subject_ids_required_at_inference": False,
        "outer_subject_ids_written_to_pretruth_artifacts": False,
        "outer_subject_diagnostics_after_raw_predictions_frozen": True,
        "outer_batch_activity_labels_used_by_learner": False,
        "outer_batch_features_fit_transductively": True,
        "outer_batch_trial_count": OUTER_TRIAL_COUNT,
        "raw_predictions_frozen_before_truth_join": True,
        "raw_predictions_sha256": raw_prediction_sha,
        "global_hungarian_uses_truth_for_scoring_only": True,
        "global_hungarian_writeback": False,
        "known_activity_cluster_count_is_oracle": True,
        "scorer_truth_alignment": scorer_truth_audit,
        "gate_used": False,
        "sequential_online_sessions_used": False,
        "online_encoder_update_used": False,
        "online_codebook_update_used": False,
        "frozen_a2_initial_sha256": initial_encoder_sha,
        "frozen_a2_final_sha256": final_encoder_sha,
        "frozen_e0_codebook_sha256": codebook_sha,
    }
    write_json(output / "leakage_audit.json", leakage_audit)

    outer_usage = usage["outer_all"]
    summary = {
        "schema": SCHEMA,
        "profile": PROFILE,
        "descriptor_profile": descriptor_spec.name,
        "fold": int(args.fold),
        "seed": int(args.seed),
        "window_size": int(args.window_size),
        "window_stride": int(args.window_stride),
        "primitive_num": int(args.primitive_num),
        "primitive_pca_dim": int(args.pca_dim),
        "codebook_state_sha256": codebook_sha,
        "descriptor_transform_state_sha256": transform_sha,
        "codebook_origin_mode": str(codebook_origin.get("mode")),
        "codebook_source_run_identity_sha256": str(
            codebook_origin.get("source_run_identity_sha256", "")
        ),
        "raw_descriptor_dim": expected_raw_dim,
        "descriptor_kept_dim": int(len(transform.keep_columns)),
        "protected_vertical_dim": int(len(protected_columns)),
        "signed_vertical_distance_weight": (
            float(args.signed_vertical_distance_weight)
            if descriptor_spec.signed_vertical else 0.0
        ),
        "subject_nuisance_rank": int(
            0 if transform.nuisance_basis is None else len(transform.nuisance_basis)
        ),
        "subject_nuisance_projection_strength": (
            float(transform.nuisance_projection_strength)
            if transform.nuisance_basis is not None else 0.0
        ),
        "pre_subject_debias_transform_state_sha256": (
            pre_subject_debias_transform_sha
        ),
        "descriptor_output_dim": int(transform.output_dim),
        "activity_cluster_count": TOTAL_CLASS_COUNT,
        "outer_trial_count": OUTER_TRIAL_COUNT,
        "all_accuracy": float(global_metrics["all_accuracy"]),
        "old_accuracy": float(global_metrics["old_accuracy"]),
        "new_accuracy": float(global_metrics["new_accuracy"]),
        "h_score": float(global_metrics["h_score"]),
        "macro_f1": float(global_metrics["macro_f1"]),
        "ari": ari,
        "nmi": nmi,
        "class_recall": class_recall,
        "gravity_reliable_fraction": float(
            descriptor_bias_audit["gravity_quality"]["outer_all"]["reliable_fraction"]
        ),
        "max_abs_feature_log_duration_correlation": float(
            descriptor_bias_audit["outer_max_abs_feature_log_duration_correlation_final"]
        ),
        "outer_subject_centroid_dispersion_diagnostic": float(
            descriptor_bias_audit["outer_subject_centroid_dispersion_final_diagnostic_only"]
        ),
        "used_primitive_k": int(outer_usage["used_k"]),
        "effective_primitive_k": float(outer_usage["effective_k"]),
        "dead_primitive_fraction": float(outer_usage["dead_fraction"]),
        "raw_predictions_sha256": raw_prediction_sha,
        "raw_predictions_frozen_before_truth_join": True,
        "global_hungarian_scoring_only": True,
        "visualizations": list(visual_paths),
    }
    write_json(output / "summary.json", summary)
    artifact_hashes = {name: sha256_file(output / name) for name in _output_artifacts()}
    complete = {
        **summary,
        "run_identity_sha256": identity["identity_sha256"],
        "summary_sha256": sha256_file(output / "summary.json"),
        "artifact_sha256": artifact_hashes,
        "complete": True,
    }
    write_json(output / "complete.json", complete)
    LOGGER.info(
        "completed descriptor=%s W%d/K%d fold=%d seed=%d all=%.4f old=%.4f new=%.4f H=%.4f ARI=%.4f NMI=%.4f",
        descriptor_spec.name,
        int(args.window_size),
        int(args.primitive_num),
        int(args.fold),
        int(args.seed),
        float(global_metrics["all_accuracy"]),
        float(global_metrics["old_accuracy"]),
        float(global_metrics["new_accuracy"]),
        float(global_metrics["h_score"]),
        ari,
        nmi,
    )
    return complete


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "One variable-window/variable-codebook transductive batch-GCD member "
            "with no gate and no sequential online update."
        )
    )
    parser.add_argument("--a2-checkpoint", required=True)
    parser.add_argument("--npz-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--fold", type=int, required=True, choices=range(1, 8))
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--window-size", type=int, required=True, choices=(64, 128, 256))
    parser.add_argument("--window-stride", type=int, required=True, choices=(32, 64, 128))
    parser.add_argument("--primitive-num", type=int, required=True, choices=(32, 64, 128))
    parser.add_argument("--pca-dim", type=int, default=64, choices=(64,))
    parser.add_argument("--activity-cluster-count", type=int, default=12, choices=(12,))
    parser.add_argument("--descriptor-pca-dim", type=int, default=32, choices=(32,))
    parser.add_argument(
        "--descriptor-profile",
        default=LEGACY_DESCRIPTOR_PROFILE,
        choices=tuple(DESCRIPTOR_PROFILES),
    )
    parser.add_argument(
        "--reuse-codebook-run-dir",
        default=None,
        help=(
            "Completed same-fold/seed legacy_state_v1 member whose frozen E0 "
            "codebook must be reused; required for every non-legacy profile."
        ),
    )
    parser.add_argument("--signed-vertical-distance-weight", type=float, default=0.15)
    parser.add_argument("--subject-nuisance-max-rank", type=int, default=4)
    parser.add_argument(
        "--subject-nuisance-explained-variance", type=float, default=0.90
    )
    parser.add_argument("--kmeans-n-init", type=int, default=50)
    parser.add_argument("--kmeans-max-iter", type=int, default=300)
    parser.add_argument("--encode-batch-size", type=int, default=1024)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--allow-smoke-a2", action="store_true")
    parser.add_argument("--resume", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> dict[str, Any]:
    result = run(build_parser().parse_args(argv))
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return result


if __name__ == "__main__":
    main()


__all__ = [
    "ARM_BY_PAIR",
    "IDENTITY_SCHEMA",
    "OUTER_TRIAL_COUNT",
    "PROFILE",
    "SCHEMA",
    "build_parser",
    "main",
    "outer_batch_trials",
    "run",
    "trajectory_time_raster_seconds",
    "validate_args",
    "validate_completed_output",
    "validate_scorer_truth",
]
