"""Fixed-split-01 static/dynamic expert experiment for motion-primitive GCD.

One frozen fold-1 A2 encoder is reused while ``run_seed`` controls the E0 K64
codebook and all downstream unlabeled models.  Three paired predictions are
frozen before scorer truth is opened:

* ``B0_global_trajectory``: one duration-invariant trajectory KMeans12;
* ``E1_gate_static_expert``: an unlabeled static/dynamic gate plus an
  independent physical static expert. The known total K12 is divided between
  branches from their unlabeled member counts under USC-HAD's balanced-trial
  protocol; this internal allocation is not a third experiment arm.
* ``E2_gate_static_expert_soft_a025``: exactly the same gate, branch K and
  static predictions as E1, with only the dynamic trajectory coordinates
  changed to duration-invariant alpha=0.25 source-subject soft debiasing.

This remains known-K12 transductive batch GCD, not multi-session online CGCD.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

for _name, _value in {
    "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
}.items():
    os.environ[_name] = _value

import numpy as np
import sklearn
import torch
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, roc_auc_score
from threadpoolctl import threadpool_info, threadpool_limits

from experiments.motion_primitive.frozen_e0 import save_frozen_artifacts
from experiments.motion_primitive.motion_checkpoint import motion_state_dict_sha256
from experiments.motion_primitive.static_dynamic_expert import (
    DEFAULT_STATIC_BLOCK_WEIGHTS,
    STATIC_BLOCK_ORDER,
    fit_duration_invariant_trajectory_features,
    fit_duration_soft_subject_trajectory_features,
    fit_fixed_k_branch,
    fit_static_dynamic_gate,
    fit_static_expert,
    merge_branch_predictions,
    motion_primitive_static_matrix,
    physical_block_matrices,
    fit_count_proportional_branch_clusters,
)
from experiments.motion_primitive.strict_artifacts import (
    configure_deterministic_runtime,
    encode_sensor_trials,
    load_frozen_a2_encoder,
    write_csv,
    write_json,
)
from experiments.motion_primitive.strict_cv_common import canonical_hash
from experiments.motion_primitive.strict_metrics import strict_three_layer_metrics
from experiments.motion_primitive.strict_protocol import build_registered_protocol, sha256_file
from experiments.motion_primitive.window_codebook_batch_proxy import (
    _codebook_state_sha256,
    _fit_codebook,
    _primitive_trials,
    _tokenize_trial,
    _transform_state_sha256,
    outer_batch_trials,
    validate_scorer_truth,
)


SCHEMA = "hhr_static_dynamic_expert_fixedsplit01_v2"
IDENTITY_SCHEMA = "hhr_static_dynamic_expert_fixedsplit01_identity_v2"
FIXED_SPLIT = 1
WINDOW_SIZE = 128
WINDOW_STRIDE = 64
PRIMITIVE_NUM = 64
PRIMITIVE_PCA_DIM = 64
TOTAL_CLASS_COUNT = 12
OLD_CLASS_COUNT = 6
OUTER_TRIAL_COUNT = 120
ARMS = (
    "B0_global_trajectory",
    "E1_gate_static_expert",
    "E2_gate_static_expert_soft_a025",
)
SOFT_SUBJECT_PROJECTION_STRENGTH = 0.25
ARTIFACT_NAMES = (
    "frozen_base_representation.npz",
    "expert_state.npz",
    "branch_allocation.json",
    "gate_manifest.json",
    "raw_predictions.npz",
    "metrics.json",
    "posttruth_gate_diagnostic.json",
    "predictions.csv",
    "leakage_audit.json",
    "summary.json",
)
LOGGER = logging.getLogger("hhr_static_dynamic_batch_proxy")


@contextmanager
def _exclusive_output_lock(output: str | Path, *, lock_name: str = ".member.lock"):
    """Hold a non-blocking process lock for one output directory."""

    root = Path(output).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    lock_path = root / str(lock_name)
    handle = lock_path.open("a+b")
    locked = False
    try:
        if os.name == "posix":
            import fcntl

            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as error:
                raise RuntimeError(f"Output directory is already locked: {root}.") from error
        else:
            import msvcrt

            handle.seek(0, os.SEEK_END)
            if handle.tell() == 0:
                handle.write(b"0")
                handle.flush()
            handle.seek(0)
            try:
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError as error:
                raise RuntimeError(f"Output directory is already locked: {root}.") from error
        locked = True
        yield lock_path
    finally:
        if locked:
            if os.name == "posix":
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            else:
                import msvcrt

                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        handle.close()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON artifact is not an object: {path}.")
    return value


def _atomic_npz(path: Path, **arrays: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    temporary.replace(path)


def _array_sha256(*arrays: np.ndarray) -> str:
    digest = hashlib.sha256()
    for index, value in enumerate(arrays):
        array = np.ascontiguousarray(np.asarray(value))
        digest.update(str(index).encode("ascii"))
        digest.update(array.dtype.str.encode("ascii"))
        digest.update(str(array.shape).encode("ascii"))
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _seed(seed: int, offset: int) -> int:
    return int((int(seed) * 1_000_003 + int(offset)) % (2**32 - 1))


def _resolve_device(value: str) -> torch.device:
    if str(value).lower() == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")
    return device


def _implementation_hashes() -> dict[str, str]:
    names = (
        "experiments/motion_primitive/static_dynamic_batch_proxy.py",
        "experiments/motion_primitive/static_dynamic_expert.py",
        "experiments/motion_primitive/window_codebook_batch_proxy.py",
        "experiments/motion_primitive/frozen_e0.py",
        "experiments/motion_primitive/frozen_e0_state.py",
        "experiments/motion_primitive/core.py",
        "experiments/motion_primitive/strict_protocol.py",
        "experiments/motion_primitive/strict_artifacts.py",
        "experiments/motion_primitive/strict_cv_common.py",
        "experiments/motion_primitive/strict_metrics.py",
        "experiments/motion_primitive/motion_checkpoint.py",
        "experiments/motion_primitive/motion_encoder.py",
        "models/resnet1d.py",
    )
    return {name: sha256_file(PROJECT_ROOT / name) for name in names}


def _static_weights(args: argparse.Namespace) -> dict[str, float]:
    result = {
        "posture": float(args.static_posture_weight),
        "signed_gravity": float(args.static_gravity_weight),
        "energy": float(args.static_energy_weight),
        "motion_primitive": float(args.static_motion_primitive_weight),
    }
    if set(result) != set(STATIC_BLOCK_ORDER):
        raise RuntimeError("Internal static block schema changed.")
    if any(value <= 0.0 for value in result.values()) or not np.isclose(
        sum(result.values()), 1.0, rtol=0.0, atol=1e-12
    ):
        raise ValueError("Static expert weights must be positive and sum exactly to 1.")
    return result


def validate_args(args: argparse.Namespace) -> argparse.Namespace:
    if int(args.run_seed) < 0 or int(args.encoder_seed) < 0:
        raise ValueError("Run and encoder seeds must be non-negative.")
    if min(
        int(args.encode_batch_size),
        int(args.gate_n_init),
        int(args.kmeans_n_init),
        int(args.kmeans_max_iter),
        int(args.static_motion_primitive_pca_dim),
        int(args.subject_nuisance_max_rank),
    ) < 1:
        raise ValueError("Batch size and initialization/iteration counts must be positive.")
    if not 0.0 < float(args.subject_nuisance_explained_variance) <= 1.0:
        raise ValueError("Subject-nuisance explained variance must lie in (0,1].")
    if not np.isclose(
        float(args.subject_nuisance_projection_strength),
        SOFT_SUBJECT_PROJECTION_STRENGTH,
        rtol=0.0,
        atol=1e-12,
    ):
        raise ValueError("This experiment locks soft subject projection strength to 0.25.")
    _static_weights(args)
    _resolve_device(args.device)
    return args


def _identity(args: argparse.Namespace) -> dict[str, Any]:
    checkpoint = Path(args.a2_checkpoint).expanduser().resolve()
    npz_path = Path(args.npz_path).expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    if not npz_path.is_file():
        raise FileNotFoundError(npz_path)
    return {
        "schema": IDENTITY_SCHEMA,
        "protocol": "fixed_split_01_known_k12_transductive_batch_gcd",
        "fixed_split": FIXED_SPLIT,
        "run_seed": int(args.run_seed),
        "encoder_fold": FIXED_SPLIT,
        "encoder_seed": int(args.encoder_seed),
        "seed_semantics": "run_seed_controls_codebook_gate_and_clusters_not_encoder",
        "a2_checkpoint_path": str(checkpoint),
        "a2_checkpoint_sha256": sha256_file(checkpoint),
        "npz_path": str(npz_path),
        "npz_sha256": sha256_file(npz_path),
        "representation": {
            "window_size": WINDOW_SIZE,
            "window_stride": WINDOW_STRIDE,
            "primitive_num": PRIMITIVE_NUM,
            "primitive_pca_dim": PRIMITIVE_PCA_DIM,
            "dynamic_descriptor": "duration_invariant_v1_state",
            "soft_dynamic_descriptor": "duration_soft_subject_a025_v1_state",
            "soft_subject_nuisance_max_rank": int(args.subject_nuisance_max_rank),
            "soft_subject_nuisance_explained_variance": float(
                args.subject_nuisance_explained_variance
            ),
            "soft_subject_nuisance_projection_strength": float(
                args.subject_nuisance_projection_strength
            ),
            "soft_coordinate_fit_scope": "gate_selected_outer_dynamic_unlabeled",
            "soft_nuisance_basis_fit_scope": (
                "offline_train_old6_balanced_source_subject_centroids"
            ),
            "static_block_weights": _static_weights(args),
            "static_motion_primitive_pca_dim": int(args.static_motion_primitive_pca_dim),
        },
        "learner": {
            "arms": list(ARMS),
            "gate": "robust_scaled_diagonal_gmm2_low_energy_is_static",
            "gate_n_init": int(args.gate_n_init),
            "known_total_activity_clusters": TOTAL_CLASS_COUNT,
            "branch_count_selection": "gate_count_proportional_under_balanced_trial_protocol",
            "E1_E2_share_gate_branch_k_and_static_predictions": True,
            "minimum_dynamic_clusters": OLD_CLASS_COUNT,
            "kmeans_n_init": int(args.kmeans_n_init),
            "kmeans_max_iter": int(args.kmeans_max_iter),
            "outer_fit_scope": "all_120_outer_trials_unlabeled_transductive",
            "truth_opened_only_after_raw_prediction_sha256": True,
        },
        "runtime": {
            "encode_batch_size": int(args.encode_batch_size),
            "device": str(_resolve_device(args.device)),
            "allow_smoke_a2": bool(args.allow_smoke_a2),
            "numpy": np.__version__,
            "sklearn": sklearn.__version__,
            "torch": torch.__version__,
        },
        "implementation_sha256": _implementation_hashes(),
    }


def _prepare_identity(output: Path, args: argparse.Namespace) -> dict[str, Any]:
    body = _identity(args)
    expected = {**body, "identity_sha256": canonical_hash(body)}
    path = output / "run_identity.json"
    if path.is_file():
        observed = _read_json(path)
        if observed != expected:
            raise RuntimeError(f"Output records another experiment identity: {output}.")
        return expected
    if output.exists():
        unexpected = [item.name for item in output.iterdir() if item.name != ".member.lock"]
        if unexpected:
            raise RuntimeError(
                f"Non-empty output has no matching run identity: {output}; {unexpected}."
            )
    output.mkdir(parents=True, exist_ok=True)
    write_json(path, expected)
    return expected


def validate_completed_output(
    output: str | Path,
    *,
    expected_identity: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    root = Path(output).expanduser().resolve()
    identity = _read_json(root / "run_identity.json")
    body = {key: value for key, value in identity.items() if key != "identity_sha256"}
    if identity.get("identity_sha256") != canonical_hash(body):
        raise RuntimeError("Run identity hash does not verify.")
    if expected_identity is not None and dict(expected_identity) != identity:
        raise RuntimeError("Completed output identity differs from the requested run.")
    complete = _read_json(root / "complete.json")
    if complete.get("complete") is not True or complete.get("schema") != SCHEMA:
        raise RuntimeError("Completion marker is invalid.")
    if complete.get("run_identity_sha256") != identity.get("identity_sha256"):
        raise RuntimeError("Completion marker is not bound to the verified run identity.")
    inventory = complete.get("artifact_sha256")
    if not isinstance(inventory, Mapping) or set(inventory) != set(ARTIFACT_NAMES):
        raise RuntimeError("Completion marker lacks the exact registered artifact inventory.")
    for name, expected in inventory.items():
        path = root / str(name)
        if not path.is_file() or sha256_file(path) != str(expected):
            raise RuntimeError(f"Completed artifact changed: {path}.")
    summary_path = root / "summary.json"
    summary = _read_json(summary_path)
    if complete.get("summary_sha256") != sha256_file(summary_path):
        raise RuntimeError("Completion marker is not bound to summary.json.")
    for key, value in summary.items():
        if complete.get(key) != value:
            raise RuntimeError(f"Completion field {key!r} differs from summary.json.")
    raw_sha = sha256_file(root / "raw_predictions.npz")
    if complete.get("raw_predictions_sha256") != raw_sha:
        raise RuntimeError("Frozen raw prediction SHA256 no longer verifies.")
    return complete


def _score_arm(
    targets: np.ndarray,
    predictions: np.ndarray,
    activity_names: Sequence[str],
) -> dict[str, Any]:
    bundle = strict_three_layer_metrics(
        targets,
        predictions,
        class_count=TOTAL_CLASS_COUNT,
        old_class_count=OLD_CLASS_COUNT,
        seen_class_count_before_session=TOTAL_CLASS_COUNT,
        registered_class_ids=tuple(range(TOTAL_CLASS_COUNT)),
    )
    global_metrics = bundle["layers"]["global_hungarian_upper_bound"]
    confusion = np.asarray(global_metrics["confusion_matrix_with_unknown_column"], dtype=np.int64)
    recalls = np.diag(confusion[:, :TOTAL_CLASS_COUNT]) / np.maximum(confusion.sum(axis=1), 1)
    names_by_label = {
        int(label): str(name) for label, name in zip(targets.tolist(), activity_names)
    }
    return {
        "adjusted_rand_index": float(adjusted_rand_score(targets, predictions)),
        "normalized_mutual_information": float(
            normalized_mutual_info_score(targets, predictions)
        ),
        "class_recall": {
            names_by_label[label]: float(recalls[label]) for label in range(TOTAL_CLASS_COUNT)
        },
        "layers": bundle["layers"],
    }


def _summary_metrics(score: Mapping[str, Any]) -> dict[str, float]:
    layer = score["layers"]["global_hungarian_upper_bound"]
    return {
        "all_accuracy": float(layer["all_accuracy"]),
        "old_accuracy": float(layer["old_accuracy"]),
        "new_accuracy": float(layer["new_accuracy"]),
        "h_score": float(layer["h_score"]),
        "macro_f1": float(layer["macro_f1"]),
        "ari": float(score["adjusted_rand_index"]),
        "nmi": float(score["normalized_mutual_information"]),
    }


def _transform_arrays(prefix: str, transform: Any) -> dict[str, np.ndarray]:
    return {
        f"{prefix}_keep_columns": np.asarray(transform.keep_columns, dtype=np.int64),
        f"{prefix}_mean": np.asarray(transform.mean, dtype=np.float64),
        f"{prefix}_scale": np.asarray(transform.scale, dtype=np.float64),
        f"{prefix}_pca_mean": np.asarray([], dtype=np.float64)
        if transform.pca_mean is None
        else np.asarray(transform.pca_mean, dtype=np.float64),
        f"{prefix}_pca_components": np.empty((0, 0), dtype=np.float64)
        if transform.pca_components is None
        else np.asarray(transform.pca_components, dtype=np.float64),
        f"{prefix}_nuisance_basis": np.empty(
            (0, int(transform.primary_output_dim)), dtype=np.float64
        )
        if transform.nuisance_basis is None
        else np.asarray(transform.nuisance_basis, dtype=np.float64),
        f"{prefix}_nuisance_singular_values": np.asarray([], dtype=np.float64)
        if transform.nuisance_singular_values is None
        else np.asarray(transform.nuisance_singular_values, dtype=np.float64),
        f"{prefix}_nuisance_explained_fraction": np.asarray(
            float(transform.nuisance_explained_fraction), dtype=np.float64
        ),
        f"{prefix}_nuisance_fit_subject_count": np.asarray(
            int(transform.nuisance_fit_subject_count), dtype=np.int64
        ),
        f"{prefix}_nuisance_fit_trial_count": np.asarray(
            int(transform.nuisance_fit_trial_count), dtype=np.int64
        ),
        f"{prefix}_nuisance_projection_strength": np.asarray(
            float(transform.nuisance_projection_strength), dtype=np.float64
        ),
    }


def _run_deterministic(args: argparse.Namespace) -> dict[str, Any]:
    args = validate_args(args)
    output = Path(args.output_dir).expanduser().resolve()
    identity = _prepare_identity(output, args)
    if (output / "complete.json").is_file():
        if not bool(args.resume):
            raise FileExistsError(f"Completed output already exists: {output}.")
        return validate_completed_output(output, expected_identity=identity)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.FileHandler(output / "static_dynamic_expert.log", encoding="utf-8"),
            logging.StreamHandler(),
        ],
        force=True,
    )
    protocol = build_registered_protocol(
        args.npz_path,
        fold=FIXED_SPLIT,
        seed=int(args.run_seed),
        window_size=WINDOW_SIZE,
        stride=WINDOW_STRIDE,
        shuffle_novel_classes=False,
    )
    if protocol.npz_sha256 != identity["npz_sha256"]:
        raise RuntimeError("Protocol NPZ differs from the frozen run identity.")
    device = _resolve_device(args.device)
    encoder, encoder_audit = load_frozen_a2_encoder(
        args.a2_checkpoint,
        expected_npz_sha256=protocol.npz_sha256,
        expected_fold=FIXED_SPLIT,
        expected_seed=int(args.encoder_seed),
        expected_split=protocol.split,
        expected_window_size=WINDOW_SIZE,
        allow_smoke=bool(args.allow_smoke_a2),
    )
    if encoder_audit.get("checkpoint_sha256") != identity["a2_checkpoint_sha256"]:
        raise RuntimeError("A2 checkpoint changed between identity freeze and model load.")
    initial_encoder_sha = str(encoder_audit["model_state_dict_sha256"])

    LOGGER.info(
        "fixed split=%d encoder_seed=%d run_seed=%d: fitting offline old6 K%d codebook",
        FIXED_SPLIT,
        int(args.encoder_seed),
        int(args.run_seed),
        PRIMITIVE_NUM,
    )
    encoded_offline = encode_sensor_trials(
        encoder,
        protocol.offline_train,
        device=device,
        batch_size=int(args.encode_batch_size),
    )
    codebook = _fit_codebook(
        encoded_offline,
        primitive_num=PRIMITIVE_NUM,
        pca_dim=PRIMITIVE_PCA_DIM,
        seed=int(args.run_seed),
    )
    codebook_sha = _codebook_state_sha256(codebook)
    offline_trajectories = [
        _tokenize_trial(item, codebook, include_gravity=False)
        for item in encoded_offline
    ]
    offline_trial_ids = np.asarray(
        [int(item.trial_id) for item in offline_trajectories], dtype=np.int64
    )
    offline_subjects = np.asarray(
        [int(item.subject_id) for item in offline_trajectories], dtype=np.int64
    )
    source_subjects, source_counts = np.unique(offline_subjects, return_counts=True)
    if (
        len(offline_trajectories) != 300
        or len(np.unique(offline_trial_ids)) != 300
        or len(source_subjects) != 10
        or not np.array_equal(source_counts, np.full(10, 30, dtype=np.int64))
    ):
        raise RuntimeError(
            "Soft subject debiasing requires the registered 300=10x30 offline-old6 source set."
        )

    batch_trials = outer_batch_trials(protocol)
    outer_ids = np.asarray([int(item.trial_id) for item in batch_trials], dtype=np.int64)
    if len(outer_ids) != OUTER_TRIAL_COUNT:
        raise RuntimeError("Fixed-split outer batch does not contain 120 trials.")
    if set(offline_trial_ids.astype(int).tolist()) & set(outer_ids.astype(int).tolist()):
        raise RuntimeError("Offline source and outer target trial IDs overlap.")
    LOGGER.info("encoding %d unlabeled outer trials", len(batch_trials))
    trajectories = _primitive_trials(
        encoder,
        batch_trials,
        codebook,
        device=device,
        batch_size=int(args.encode_batch_size),
        include_gravity=True,
    )

    physical = physical_block_matrices(
        [item.raw_windows for item in batch_trials],
        [item.window_starts for item in batch_trials],
    )
    gate = fit_static_dynamic_gate(
        physical["gate"], seed=int(args.run_seed), n_init=int(args.gate_n_init)
    )
    static_rows = np.flatnonzero(gate.is_static)
    dynamic_rows = np.flatnonzero(~gate.is_static)
    LOGGER.info(
        "unlabeled gate assigned dynamic=%d static=%d",
        len(dynamic_rows),
        len(static_rows),
    )

    # B0: paired global trajectory baseline using the same encoder and codebook.
    global_features, global_transform, descriptor_names = (
        fit_duration_invariant_trajectory_features(
            trajectories, PRIMITIVE_NUM, maximum_components=32
        )
    )
    global_ids, global_centers, global_inertia = fit_fixed_k_branch(
        global_features,
        TOTAL_CLASS_COUNT,
        seed=_seed(args.run_seed, 3_001),
        n_init=int(args.kmeans_n_init),
        max_iter=int(args.kmeans_max_iter),
    )

    # Fit only the dynamic trajectory coordinates needed by E1. The static
    # branch below has its own physical expert; no gate-only third arm is fit.
    dynamic_trajectories = [trajectories[index] for index in dynamic_rows]
    dynamic_features, dynamic_transform, dynamic_names = (
        fit_duration_invariant_trajectory_features(
            dynamic_trajectories, PRIMITIVE_NUM, maximum_components=32
        )
    )
    if descriptor_names != dynamic_names:
        raise RuntimeError("Global and dynamic trajectory descriptor schemas differ.")

    # E2 changes only E1's dynamic representation.  Its nuisance basis is
    # fitted from the balanced offline-old6 source subject centroids; no outer
    # subject identity is read before scorer truth is opened.
    (
        soft_dynamic_features,
        soft_dynamic_transform,
        soft_dynamic_names,
        soft_nuisance_audit,
    ) = fit_duration_soft_subject_trajectory_features(
        dynamic_trajectories,
        offline_trajectories,
        PRIMITIVE_NUM,
        maximum_components=32,
        maximum_rank=int(args.subject_nuisance_max_rank),
        explained_variance=float(args.subject_nuisance_explained_variance),
        projection_strength=float(args.subject_nuisance_projection_strength),
    )
    if dynamic_names != soft_dynamic_names:
        raise RuntimeError("E1 and E2 dynamic trajectory descriptor schemas differ.")
    for name in ("keep_columns", "mean", "scale", "pca_mean", "pca_components"):
        left = getattr(dynamic_transform, name)
        right = getattr(soft_dynamic_transform, name)
        if left is None or right is None:
            if left is not None or right is not None:
                raise RuntimeError("E1/E2 pre-debias coordinate states differ.")
        elif not np.array_equal(np.asarray(left), np.asarray(right)):
            raise RuntimeError("E1/E2 pre-debias coordinate states differ.")

    # E1 static expert. The branch K allocation comes only from gate membership
    # counts; features are used only after K is fixed to fit each branch.
    motion_primitive_block, motion_primitive_names = motion_primitive_static_matrix(
        trajectories, PRIMITIVE_NUM
    )
    static_blocks = {
        "posture": physical["posture"][static_rows],
        "signed_gravity": physical["signed_gravity"][static_rows],
        "energy": physical["energy"][static_rows],
        "motion_primitive": motion_primitive_block[static_rows],
    }
    static_expert, static_expert_features = fit_static_expert(
        static_blocks,
        weights=_static_weights(args),
        motion_primitive_pca_dim=int(args.static_motion_primitive_pca_dim),
    )
    branch = fit_count_proportional_branch_clusters(
        dynamic_features,
        static_expert_features,
        total_clusters=TOTAL_CLASS_COUNT,
        minimum_dynamic_clusters=OLD_CLASS_COUNT,
        seed=int(args.run_seed),
        n_init=int(args.kmeans_n_init),
        max_iter=int(args.kmeans_max_iter),
    )
    gate_expert_ids = merge_branch_predictions(
        gate.is_static,
        branch.dynamic_labels,
        branch.static_labels,
        dynamic_k=branch.dynamic_k,
        total_clusters=TOTAL_CLASS_COUNT,
    )
    soft_dynamic_labels, soft_dynamic_centers, soft_dynamic_inertia = fit_fixed_k_branch(
        soft_dynamic_features,
        branch.dynamic_k,
        seed=_seed(args.run_seed, 1_000 + branch.static_k),
        n_init=int(args.kmeans_n_init),
        max_iter=int(args.kmeans_max_iter),
    )
    soft_gate_expert_ids = merge_branch_predictions(
        gate.is_static,
        soft_dynamic_labels,
        branch.static_labels,
        dynamic_k=branch.dynamic_k,
        total_clusters=TOTAL_CLASS_COUNT,
    )
    if not np.array_equal(
        gate_expert_ids[static_rows], soft_gate_expert_ids[static_rows]
    ):
        raise RuntimeError("E1/E2 static predictions must be bitwise identical.")

    save_frozen_artifacts(
        output / "frozen_base_representation.npz",
        codebook,
        global_transform,
        descriptor_names,
        {
            "schema": SCHEMA,
            "fixed_split": FIXED_SPLIT,
            "encoder_seed": int(args.encoder_seed),
            "run_seed": int(args.run_seed),
            "descriptor_profile": "duration_invariant_v1",
            "role": "paired_B0_global_baseline_and_shared_E0_codebook",
        },
    )
    expert_arrays: dict[str, Any] = {
        "gate_median": gate.state.median,
        "gate_scale": gate.state.scale,
        "gate_mixture_weights": gate.state.mixture_weights,
        "gate_mixture_means": gate.state.mixture_means,
        "gate_mixture_covariances": gate.state.mixture_covariances,
        "gate_static_component": np.asarray(gate.state.static_component, dtype=np.int64),
        "dynamic_k": np.asarray(branch.dynamic_k, dtype=np.int64),
        "static_k": np.asarray(branch.static_k, dtype=np.int64),
        "gate_dynamic_centers": branch.dynamic_centers,
        "soft_dynamic_centers": soft_dynamic_centers,
        "expert_static_centers": branch.static_centers,
        "static_expert_weights": static_expert.weights,
        "soft_subject_projection_strength": np.asarray(
            SOFT_SUBJECT_PROJECTION_STRENGTH, dtype=np.float64
        ),
    }
    expert_arrays.update(_transform_arrays("dynamic", dynamic_transform))
    expert_arrays.update(_transform_arrays("soft_dynamic", soft_dynamic_transform))
    for name in STATIC_BLOCK_ORDER:
        block = getattr(static_expert, name)
        expert_arrays.update(
            {
                f"static_{name}_keep_columns": block.keep_columns,
                f"static_{name}_median": block.median,
                f"static_{name}_scale": block.scale,
                f"static_{name}_pca_center": np.asarray([], dtype=np.float64)
                if block.pca_center is None
                else block.pca_center,
                f"static_{name}_pca_components": np.empty((0, 0), dtype=np.float64)
                if block.pca_components is None
                else block.pca_components,
            }
        )
    _atomic_npz(output / "expert_state.npz", **expert_arrays)

    branch_manifest = {
        "schema": SCHEMA,
        "selection_scope": "outer_unlabeled_gate_membership_counts_only",
        "strategy": "count_proportional_under_balanced_trial_protocol",
        "balanced_activity_trial_count_assumption": True,
        "total_known_activity_clusters": TOTAL_CLASS_COUNT,
        "minimum_dynamic_clusters": OLD_CLASS_COUNT,
        "selected_dynamic_k": int(branch.dynamic_k),
        "selected_static_k": int(branch.static_k),
        "allocation_table": list(branch.candidate_table),
        "gate_dynamic_trial_count": int(len(dynamic_rows)),
        "gate_static_trial_count": int(len(static_rows)),
        "activity_labels_used": False,
        "subject_ids_used_for_gate_or_k_allocation": False,
        "offline_source_subject_ids_used_for_dynamic_representation": True,
        "outer_subject_ids_used": False,
        "reported_as_separate_experiment_arm": False,
        "E1_E2_share_gate_branch_k_and_static_predictions": True,
        "E2_dynamic_inertia": float(soft_dynamic_inertia),
    }
    write_json(output / "branch_allocation.json", branch_manifest)
    write_json(
        output / "gate_manifest.json",
        {
            "schema": SCHEMA,
            "algorithm": "robust_scaled_diagonal_GMM2",
            "component_semantics": "lower_sum_log_energy_centroid_is_body_static",
            "fit_scope": "all_outer_trials_unlabeled_transductive",
            "activity_labels_used": False,
            "subject_ids_used": False,
            "static_component": int(gate.state.static_component),
            "state_sha256": gate.state.state_sha256,
        },
    )

    predictions = {
        "B0_global_trajectory": global_ids,
        "E1_gate_static_expert": gate_expert_ids,
        "E2_gate_static_expert_soft_a025": soft_gate_expert_ids,
    }
    _atomic_npz(
        output / "raw_predictions.npz",
        trial_ids=outer_ids,
        gate_is_static=gate.is_static.astype(np.uint8),
        gate_static_probability=gate.static_probability.astype(np.float32),
        gate_features=physical["gate"].astype(np.float32),
        dynamic_row_indices=dynamic_rows.astype(np.int64),
        static_row_indices=static_rows.astype(np.int64),
        global_trajectory_features=global_features.astype(np.float32),
        dynamic_trajectory_features=dynamic_features.astype(np.float32),
        soft_dynamic_trajectory_features=soft_dynamic_features.astype(np.float32),
        static_expert_features=static_expert_features.astype(np.float32),
        shared_static_branch_labels=branch.static_labels.astype(np.int64),
        E1_dynamic_branch_labels=branch.dynamic_labels.astype(np.int64),
        E2_dynamic_branch_labels=soft_dynamic_labels.astype(np.int64),
        B0_global_trajectory=global_ids.astype(np.int64),
        E1_gate_static_expert=gate_expert_ids.astype(np.int64),
        E2_gate_static_expert_soft_a025=soft_gate_expert_ids.astype(np.int64),
        encoder_seed=np.asarray(int(args.encoder_seed), dtype=np.int64),
        run_seed=np.asarray(int(args.run_seed), dtype=np.int64),
        codebook_state_sha256=np.asarray(codebook_sha),
        gate_state_sha256=np.asarray(gate.state.state_sha256),
        static_expert_state_sha256=np.asarray(static_expert.state_sha256),
        soft_dynamic_transform_state_sha256=np.asarray(
            _transform_state_sha256(soft_dynamic_transform)
        ),
    )
    raw_prediction_sha = sha256_file(output / "raw_predictions.npz")

    # The scorer boundary begins only after raw predictions are durable and hashed.
    targets, activity_names, subjects = protocol.truth.join(outer_ids)
    target_values = np.asarray(targets, dtype=np.int64)
    subject_values = np.asarray(subjects, dtype=np.int64)
    scorer_audit = validate_scorer_truth(
        batch_trials,
        target_values,
        subject_values,
        expected_outer_subjects=protocol.split.outer_test,
    )
    scores = {
        arm: _score_arm(target_values, values, activity_names)
        for arm, values in predictions.items()
    }
    write_json(
        output / "metrics.json",
        {
            "schema": SCHEMA,
            "protocol": "fixed_split_01_known_k12_transductive_batch_gcd",
            "metric_alignment": "global_hungarian_upper_bound",
            "metric_interpretation": "unlabeled_clustering_diagnostic_upper_bound",
            "raw_predictions_sha256": raw_prediction_sha,
            "raw_predictions_frozen_before_truth_join": True,
            "arms": scores,
        },
    )

    true_static = target_values >= 7
    gate_confusion = np.asarray(
        [
            [np.sum(~true_static & ~gate.is_static), np.sum(~true_static & gate.is_static)],
            [np.sum(true_static & ~gate.is_static), np.sum(true_static & gate.is_static)],
        ],
        dtype=np.int64,
    )
    gate_accuracy = float(np.mean(true_static == gate.is_static))
    posttruth = {
        "schema": SCHEMA,
        "diagnostic_only_after_raw_predictions_frozen": True,
        "raw_predictions_sha256": raw_prediction_sha,
        "body_static_definition": "Sitting_Standing_Sleeping_ElevatorUp_ElevatorDown",
        "gate_accuracy": gate_accuracy,
        "gate_roc_auc": float(roc_auc_score(true_static.astype(np.int64), gate.static_probability)),
        "gate_confusion_true_rows_dynamic_static_pred_columns_dynamic_static": gate_confusion.tolist(),
        "selected_dynamic_k": int(branch.dynamic_k),
        "selected_static_k": int(branch.static_k),
    }
    write_json(output / "posttruth_gate_diagnostic.json", posttruth)

    aligned = {
        arm: np.asarray(
            scores[arm]["layers"]["global_hungarian_upper_bound"]["aligned_predictions"],
            dtype=np.int64,
        )
        for arm in ARMS
    }
    write_csv(
        output / "predictions.csv",
        [
            {
                "trial_id": int(trial_id),
                "subject_id": int(subject_values[index]),
                "activity_label": int(target_values[index]),
                "activity_name": str(activity_names[index]),
                "gate_is_static": bool(gate.is_static[index]),
                "gate_static_probability": float(gate.static_probability[index]),
                **{
                    f"{arm}_raw_cluster": int(predictions[arm][index]) for arm in ARMS
                },
                **{
                    f"{arm}_aligned_prediction": int(aligned[arm][index]) for arm in ARMS
                },
            }
            for index, trial_id in enumerate(outer_ids)
        ],
    )

    final_encoder_sha = motion_state_dict_sha256(
        {key: value.detach().cpu() for key, value in encoder.state_dict().items()}
    )
    if final_encoder_sha != initial_encoder_sha:
        raise RuntimeError("Frozen A2 parameters or BatchNorm buffers changed.")
    leakage_audit = {
        "schema": SCHEMA,
        "fixed_split": FIXED_SPLIT,
        "encoder_seed": int(args.encoder_seed),
        "run_seed": int(args.run_seed),
        "encoder_and_run_seed_are_explicitly_distinct_roles": True,
        "offline_encoder_uses_old_class_train_and_validation_only": True,
        "primitive_codebook_fit_uses_offline_train_old6_only": True,
        "outer_gate_activity_labels_used": False,
        "outer_gate_subject_ids_used": False,
        "outer_expert_activity_labels_used": False,
        "outer_expert_subject_ids_used": False,
        "offline_old6_source_subject_ids_used_for_soft_nuisance_basis": True,
        "soft_nuisance_basis_fit_api_receives_activity_labels": False,
        "soft_nuisance_fit_cohort_selected_by_registered_old6_protocol": True,
        "soft_coordinate_transform_fit_scope": (
            "gate_selected_outer_dynamic_unlabeled_transductive"
        ),
        "soft_nuisance_basis_fit_scope": (
            "offline_train_old6_balanced_source_subject_centroids"
        ),
        "soft_outer_subject_ids_used": False,
        "soft_outer_subject_ids_required_at_inference": False,
        "soft_subject_projection_strength": SOFT_SUBJECT_PROJECTION_STRENGTH,
        "soft_subject_nuisance_selected_rank": int(
            soft_nuisance_audit["selected_rank"]
        ),
        "offline_source_trial_count": int(len(offline_trajectories)),
        "offline_source_subject_count": int(len(source_subjects)),
        "offline_source_trials_per_subject": int(source_counts[0]),
        "offline_and_outer_trial_ids_disjoint": True,
        "outer_features_fit_transductively": True,
        "known_total_cluster_count_is_oracle": True,
        "branch_cluster_counts_selected_without_truth": True,
        "raw_predictions_frozen_before_truth_join": True,
        "raw_predictions_sha256": raw_prediction_sha,
        "global_hungarian_is_scoring_only": True,
        "scorer_truth_alignment": scorer_audit,
        "sequential_online_sessions_used": False,
        "online_encoder_update_used": False,
        "online_codebook_update_used": False,
        "frozen_a2_initial_sha256": initial_encoder_sha,
        "frozen_a2_final_sha256": final_encoder_sha,
        "frozen_e0_codebook_sha256": codebook_sha,
    }
    write_json(output / "leakage_audit.json", leakage_audit)

    summaries = {arm: _summary_metrics(scores[arm]) for arm in ARMS}
    summary = {
        "schema": SCHEMA,
        "fixed_split": FIXED_SPLIT,
        "run_seed": int(args.run_seed),
        "encoder_fold": FIXED_SPLIT,
        "encoder_seed": int(args.encoder_seed),
        "seed_semantics": "one_frozen_encoder_with_downstream_run_seed",
        "window_size": WINDOW_SIZE,
        "window_stride": WINDOW_STRIDE,
        "primitive_num": PRIMITIVE_NUM,
        "metric_alignment": "global_hungarian_upper_bound",
        "metric_interpretation": "unlabeled_clustering_diagnostic_upper_bound",
        "outer_trial_count": OUTER_TRIAL_COUNT,
        "gate_dynamic_trial_count": int(len(dynamic_rows)),
        "gate_static_trial_count": int(len(static_rows)),
        "gate_posttruth_accuracy_diagnostic": gate_accuracy,
        "selected_dynamic_k": int(branch.dynamic_k),
        "selected_static_k": int(branch.static_k),
        "codebook_state_sha256": codebook_sha,
        "gate_state_sha256": gate.state.state_sha256,
        "global_transform_state_sha256": _transform_state_sha256(global_transform),
        "dynamic_transform_state_sha256": _transform_state_sha256(dynamic_transform),
        "soft_dynamic_pre_debias_transform_state_sha256": _transform_state_sha256(
            dynamic_transform
        ),
        "soft_dynamic_transform_state_sha256": _transform_state_sha256(
            soft_dynamic_transform
        ),
        "soft_subject_projection_strength": float(
            soft_dynamic_transform.nuisance_projection_strength
        ),
        "soft_subject_nuisance_selected_rank": int(
            len(soft_dynamic_transform.nuisance_basis)
        ),
        "soft_subject_nuisance_explained_fraction": float(
            soft_dynamic_transform.nuisance_explained_fraction
        ),
        "soft_subject_nuisance_audit": soft_nuisance_audit,
        "E1_E2_share_gate_branch_k_and_static_predictions": True,
        "static_expert_state_sha256": static_expert.state_sha256,
        "global_cluster_state_sha256": _array_sha256(global_centers, global_ids),
        "E1_branch_cluster_state_sha256": branch.state_sha256,
        "E2_dynamic_cluster_state_sha256": _array_sha256(
            soft_dynamic_centers, soft_dynamic_labels
        ),
        "global_inertia": float(global_inertia),
        "expert_static_inertia": float(branch.candidate_table[0]["static_inertia"]),
        "static_motion_primitive_feature_count": int(len(motion_primitive_names)),
        "arms": summaries,
        "raw_predictions_sha256": raw_prediction_sha,
        "raw_predictions_frozen_before_truth_join": True,
        "claim_boundary": (
            "Fixed split 01 only; four run seeds share one frozen encoder. "
            "This is known-K12 transductive batch GCD, not online multi-session CGCD."
        ),
    }
    write_json(output / "summary.json", summary)
    complete = {
        **summary,
        "run_identity_sha256": identity["identity_sha256"],
        "summary_sha256": sha256_file(output / "summary.json"),
        "artifact_sha256": {name: sha256_file(output / name) for name in ARTIFACT_NAMES},
        "complete": True,
    }
    write_json(output / "complete.json", complete)
    LOGGER.info(
        "completed fixed_split=1 encoder_seed=%d run_seed=%d gate=%.4f Kdyn=%d Kstat=%d "
        "B0_H=%.4f E1_H=%.4f E2_H=%.4f",
        int(args.encoder_seed),
        int(args.run_seed),
        gate_accuracy,
        int(branch.dynamic_k),
        int(branch.static_k),
        summaries["B0_global_trajectory"]["h_score"],
        summaries["E1_gate_static_expert"]["h_score"],
        summaries["E2_gate_static_expert_soft_a025"]["h_score"],
    )
    return complete


def run(args: argparse.Namespace) -> dict[str, Any]:
    runtime = configure_deterministic_runtime(int(args.run_seed))
    output = Path(args.output_dir).expanduser().resolve()
    with _exclusive_output_lock(output):
        with threadpool_limits(limits=1):
            pools = [
                {
                    "internal_api": str(item.get("internal_api")),
                    "user_api": str(item.get("user_api")),
                    "num_threads": int(item.get("num_threads", -1)),
                }
                for item in threadpool_info()
            ]
            if any(item["num_threads"] != 1 for item in pools):
                raise RuntimeError(f"A BLAS/OpenMP pool escaped the one-thread limit: {pools}.")
            if not runtime["torch_deterministic_algorithms"]:
                raise RuntimeError("Strict PyTorch deterministic mode was not enabled.")
            return _run_deterministic(args)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "One fixed-split-01 member with a label-free body-motion gate and "
            "an independent static activity expert."
        )
    )
    parser.add_argument("--a2-checkpoint", required=True)
    parser.add_argument("--npz-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--run-seed", type=int, required=True)
    parser.add_argument("--encoder-seed", type=int, default=0)
    parser.add_argument("--encode-batch-size", type=int, default=1024)
    parser.add_argument("--gate-n-init", type=int, default=20)
    parser.add_argument("--kmeans-n-init", type=int, default=50)
    parser.add_argument("--kmeans-max-iter", type=int, default=300)
    parser.add_argument("--subject-nuisance-max-rank", type=int, default=4)
    parser.add_argument(
        "--subject-nuisance-explained-variance", type=float, default=0.90
    )
    parser.add_argument(
        "--subject-nuisance-projection-strength",
        type=float,
        default=SOFT_SUBJECT_PROJECTION_STRENGTH,
        choices=(SOFT_SUBJECT_PROJECTION_STRENGTH,),
    )
    parser.add_argument("--static-motion-primitive-pca-dim", type=int, default=8)
    parser.add_argument(
        "--static-posture-weight",
        type=float,
        default=DEFAULT_STATIC_BLOCK_WEIGHTS["posture"],
    )
    parser.add_argument(
        "--static-gravity-weight",
        type=float,
        default=DEFAULT_STATIC_BLOCK_WEIGHTS["signed_gravity"],
    )
    parser.add_argument(
        "--static-energy-weight",
        type=float,
        default=DEFAULT_STATIC_BLOCK_WEIGHTS["energy"],
    )
    parser.add_argument(
        "--static-motion-primitive-weight",
        type=float,
        default=DEFAULT_STATIC_BLOCK_WEIGHTS["motion_primitive"],
    )
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
    "ARMS",
    "ARTIFACT_NAMES",
    "FIXED_SPLIT",
    "IDENTITY_SCHEMA",
    "SCHEMA",
    "SOFT_SUBJECT_PROJECTION_STRENGTH",
    "build_parser",
    "main",
    "run",
    "validate_args",
    "validate_completed_output",
]
