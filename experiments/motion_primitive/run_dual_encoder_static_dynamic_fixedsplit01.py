"""Run the fixed-split static/dynamic experiment with two encoder origins.

The first branch reuses the registered 2026-09-14 W128/S64 fold-1 seed-0 A2
checkpoint.  The second branch retrains the same ResNet1D warm-up -> A2 route
from a fresh seeded random initialization, then runs the identical downstream
B0/E1/E2 four-run-seed suite.  The two branches are executed serially to avoid
competing for one GPU; they remain one immutable comparison suite.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.motion_primitive import run_static_dynamic_expert_fixedsplit01 as downstream
from experiments.motion_primitive import pretrain_window_encoder, train_motion_encoder
from experiments.motion_primitive import strict_encoder_cv_runner as encoder_cv
from experiments.motion_primitive.static_dynamic_batch_proxy import (
    ARMS,
    _exclusive_output_lock,
)
from experiments.motion_primitive.strict_artifacts import write_json
from experiments.motion_primitive.strict_cv_common import canonical_hash
from experiments.motion_primitive.strict_protocol import sha256_file


SCHEMA = "hhr_dual_encoder_static_dynamic_fixedsplit01_v1"
IDENTITY_SCHEMA = "hhr_dual_encoder_static_dynamic_fixedsplit01_identity_v1"
ENCODER_BRANCHES = ("R0_reuse_0914v1", "R1_retrained_current")
EXPECTED_NPZ_SHA256 = "dc06f6c3d9e665c55534d4847bf45f08a128a854e61dfa4b1760e0ae4e4ff5b5"
EXPECTED_REUSED_A2_SHA256 = "72ea25aa56b12396d0961941b7b1c0d3f18db8c090406107da29b023e18946aa"
REGISTERED_0914_WARMUP_SHA256 = "602344d55883cd3d3a17cdad343a14494c8ea635ddd7650e6e676a61981f190f"
ARTIFACT_NAMES = (
    "encoder_branches.json",
    "branch_comparison.json",
    "member_rows.csv",
)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON artifact is not an object: {path}.")
    return value


def _run_command(command: Sequence[str], *, stage: str) -> None:
    print("[command] " + shlex.join(list(command)), flush=True)
    environment = os.environ.copy()
    environment.update(
        {
            "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
            "PYTHONHASHSEED": "0",
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
            "VECLIB_MAXIMUM_THREADS": "1",
        }
    )
    try:
        subprocess.run(
            list(command), cwd=PROJECT_ROOT, check=True, env=environment
        )
    except subprocess.CalledProcessError as error:
        raise RuntimeError(
            f"Dual-encoder stage {stage!r} failed with exit code {error.returncode}."
        ) from error


def _retrained_directories(output: Path) -> tuple[Path, Path]:
    root = output / "encoders" / "R1_retrained_current" / "w128_s64"
    return (
        root / "window_pretrain" / "fold_01_seed_0",
        root / "a2" / "fold_01_seed_0",
    )


def _encoder_namespace(args: argparse.Namespace) -> argparse.Namespace:
    """Adapt the dual-suite CLI to the hardened single-member encoder API."""

    return argparse.Namespace(
        protocol=encoder_cv.W128_CONFIRMATION_PROTOCOL,
        npz_path=str(Path(args.npz_path).expanduser().resolve()),
        window_epochs=int(args.window_epochs),
        window_batch_size=int(args.window_batch_size),
        window_eval_batch_size=int(args.window_eval_batch_size),
        window_learning_rate=float(args.window_learning_rate),
        window_weight_decay=float(args.window_weight_decay),
        window_weak_scale_std=float(args.window_weak_scale_std),
        window_strong_scale_std=float(args.window_strong_scale_std),
        window_selection_policy=str(args.window_selection_policy),
        a2_epochs=int(args.a2_epochs),
        a2_trial_batch_size=int(args.a2_trial_batch_size),
        a2_source_encode_batch_size=int(args.a2_source_encode_batch_size),
        a2_learning_rate=float(args.a2_learning_rate),
        a2_minimum_learning_rate=float(args.a2_minimum_learning_rate),
        a2_weight_decay=float(args.a2_weight_decay),
        num_workers=int(args.num_workers),
        device=str(args.device),
        smoke_max_windows=int(args.smoke_max_windows),
        smoke_max_train_trials=int(args.smoke_max_train_trials),
        smoke_max_val_trials=int(args.smoke_max_val_trials),
        resume=bool(args.resume),
    )


def _branch_output(output: Path, branch: str) -> Path:
    if str(branch) not in ENCODER_BRANCHES:
        raise ValueError(f"Unknown encoder branch: {branch}.")
    return output / "branches" / str(branch)


def build_downstream_command(
    args: argparse.Namespace,
    *,
    checkpoint: Path,
    branch: str,
) -> list[str]:
    command = [
        sys.executable,
        str(
            PROJECT_ROOT
            / "experiments/motion_primitive/run_static_dynamic_expert_fixedsplit01.py"
        ),
        "--npz-path",
        str(Path(args.npz_path).expanduser().resolve()),
        "--encoder-root",
        str(checkpoint.resolve()),
        "--output-root",
        str(_branch_output(Path(args.output_root).expanduser().resolve(), branch)),
        "--encoder-seed",
        "0",
        "--run-seeds",
        str(args.run_seeds),
        "--encode-batch-size",
        str(int(args.encode_batch_size)),
        "--gate-n-init",
        str(int(args.gate_n_init)),
        "--kmeans-n-init",
        str(int(args.kmeans_n_init)),
        "--kmeans-max-iter",
        str(int(args.kmeans_max_iter)),
        "--subject-nuisance-max-rank",
        str(int(args.subject_nuisance_max_rank)),
        "--subject-nuisance-explained-variance",
        str(float(args.subject_nuisance_explained_variance)),
        "--subject-nuisance-projection-strength",
        str(float(args.subject_nuisance_projection_strength)),
        "--static-motion-primitive-pca-dim",
        str(int(args.static_motion_primitive_pca_dim)),
        "--static-posture-weight",
        str(float(args.static_posture_weight)),
        "--static-gravity-weight",
        str(float(args.static_gravity_weight)),
        "--static-energy-weight",
        str(float(args.static_energy_weight)),
        "--static-motion-primitive-weight",
        str(float(args.static_motion_primitive_weight)),
        "--device",
        str(args.device),
    ]
    if bool(args.allow_smoke_a2):
        command.append("--allow-smoke-a2")
    if bool(args.resume):
        command.append("--resume")
    return command


def _implementation_hashes() -> dict[str, str]:
    names = (
        "experiments/motion_primitive/run_dual_encoder_static_dynamic_fixedsplit01.py",
        "experiments/motion_primitive/run_static_dynamic_expert_fixedsplit01.py",
        "experiments/motion_primitive/static_dynamic_batch_proxy.py",
        "experiments/motion_primitive/static_dynamic_expert.py",
        "experiments/motion_primitive/strict_encoder_cv_runner.py",
        "experiments/motion_primitive/pretrain_window_encoder.py",
        "experiments/motion_primitive/train_motion_encoder.py",
        "experiments/motion_primitive/motion_encoder.py",
        "experiments/motion_primitive/motion_checkpoint.py",
        "models/resnet1d.py",
        "models/window_pretrain.py",
    )
    return {name: sha256_file(PROJECT_ROOT / name) for name in names}


def _identity(
    args: argparse.Namespace,
    *,
    reused_checkpoint: Path,
) -> dict[str, Any]:
    npz_path = Path(args.npz_path).expanduser().resolve()
    if not npz_path.is_file():
        raise FileNotFoundError(npz_path)
    npz_sha = sha256_file(npz_path)
    reused_sha = sha256_file(reused_checkpoint)
    if npz_sha != EXPECTED_NPZ_SHA256:
        raise RuntimeError("The dual-encoder experiment is bound to the registered W128/S64 NPZ.")
    if reused_sha != EXPECTED_REUSED_A2_SHA256:
        raise RuntimeError("The reused branch is not the registered 20260914-v1 A2 checkpoint.")
    return {
        "schema": IDENTITY_SCHEMA,
        "protocol": "fixed_split_01_dual_encoder_known_k12_transductive_batch_gcd",
        "encoder_branches": list(ENCODER_BRANCHES),
        "encoder_seed": 0,
        "reused_checkpoint_path": str(reused_checkpoint),
        "reused_checkpoint_sha256": reused_sha,
        "registered_0914_warmup_sha256_reference": REGISTERED_0914_WARMUP_SHA256,
        "npz_path": str(npz_path),
        "npz_sha256": npz_sha,
        "retrained_encoder": {
            "random_initialization": True,
            "route": "ResNet1D_window_warmup_then_A2_final_frozen",
            "interpretation": (
                "same_registered_hyperparameters_current_hardened_implementation"
            ),
            "window_size": 128,
            "window_stride": 64,
            "fold": 1,
            "seed": 0,
            "window_epochs": int(args.window_epochs),
            "window_batch_size": int(args.window_batch_size),
            "window_eval_batch_size": int(args.window_eval_batch_size),
            "window_learning_rate": float(args.window_learning_rate),
            "window_weight_decay": float(args.window_weight_decay),
            "window_weak_scale_std": float(args.window_weak_scale_std),
            "window_strong_scale_std": float(args.window_strong_scale_std),
            "window_selection_policy": str(args.window_selection_policy),
            "a2_epochs": int(args.a2_epochs),
            "a2_trial_batch_size": int(args.a2_trial_batch_size),
            "a2_source_encode_batch_size": int(args.a2_source_encode_batch_size),
            "a2_learning_rate": float(args.a2_learning_rate),
            "a2_minimum_learning_rate": float(args.a2_minimum_learning_rate),
            "a2_weight_decay": float(args.a2_weight_decay),
            "cp_context_windows": int(args.cp_context_windows),
        },
        "downstream": {
            "arms": list(ARMS),
            "run_seeds": list(downstream.parse_integer_list(args.run_seeds)),
            "encode_batch_size": int(args.encode_batch_size),
            "gate_n_init": int(args.gate_n_init),
            "kmeans_n_init": int(args.kmeans_n_init),
            "kmeans_max_iter": int(args.kmeans_max_iter),
            "subject_nuisance_max_rank": int(args.subject_nuisance_max_rank),
            "subject_nuisance_explained_variance": float(
                args.subject_nuisance_explained_variance
            ),
            "subject_nuisance_projection_strength": float(
                args.subject_nuisance_projection_strength
            ),
            "static_motion_primitive_pca_dim": int(
                args.static_motion_primitive_pca_dim
            ),
            "static_block_weights": {
                "posture": float(args.static_posture_weight),
                "signed_gravity": float(args.static_gravity_weight),
                "energy": float(args.static_energy_weight),
                "motion_primitive": float(args.static_motion_primitive_weight),
            },
        },
        "runtime": {
            "device": str(args.device),
            "num_workers": int(args.num_workers),
            "allow_smoke_a2": bool(args.allow_smoke_a2),
            "smoke_max_windows": int(args.smoke_max_windows),
            "smoke_max_train_trials": int(args.smoke_max_train_trials),
            "smoke_max_val_trials": int(args.smoke_max_val_trials),
        },
        "implementation_sha256": _implementation_hashes(),
    }


def _prepare_identity(output: Path, body: Mapping[str, Any]) -> dict[str, Any]:
    expected = {**dict(body), "identity_sha256": canonical_hash(body)}
    path = output / "dual_suite_identity.json"
    if path.is_file():
        observed = _read_json(path)
        if observed != expected:
            raise RuntimeError(f"Output root records another dual-suite identity: {output}.")
        return expected
    if output.exists():
        unexpected = [
            item.name for item in output.iterdir() if item.name != ".dual_suite.lock"
        ]
        if unexpected:
            raise RuntimeError(
                f"Non-empty output root has no matching dual-suite identity: "
                f"{output}; {unexpected}."
            )
    output.mkdir(parents=True, exist_ok=True)
    write_json(path, expected)
    return expected


def _validate_branch_suite(
    directory: Path,
    *,
    checkpoint: Path,
) -> dict[str, Any]:
    complete_path = directory / "complete.json"
    identity_path = directory / "suite_identity.json"
    if not complete_path.is_file() or not identity_path.is_file():
        raise RuntimeError(f"Downstream encoder branch is incomplete: {directory}.")
    complete = _read_json(complete_path)
    identity = _read_json(identity_path)
    if complete.get("schema") != downstream.SCHEMA or complete.get("complete") is not True:
        raise RuntimeError(f"Downstream branch completion is invalid: {directory}.")
    if identity.get("a2_checkpoint_sha256") != sha256_file(checkpoint):
        raise RuntimeError("Downstream branch is bound to another encoder checkpoint.")
    identity_body = {
        key: value for key, value in identity.items() if key != "identity_sha256"
    }
    if identity.get("identity_sha256") != canonical_hash(identity_body):
        raise RuntimeError("Downstream branch identity hash does not verify.")
    if complete.get("suite_identity_sha256") != identity.get("identity_sha256"):
        raise RuntimeError("Downstream branch completion/identity binding differs.")
    if tuple(complete.get("run_seeds", ())) != downstream.DEFAULT_RUN_SEEDS:
        raise RuntimeError("Downstream branch run-seed grid differs.")
    if set(complete.get("arms", {})) != set(ARMS):
        raise RuntimeError("Downstream branch arm set differs.")
    inventory = complete.get("artifact_sha256")
    if not isinstance(inventory, Mapping) or set(inventory) != {
        "aggregate.json",
        "member_rows.csv",
    }:
        raise RuntimeError("Downstream branch lacks its exact artifact inventory.")
    for name, expected in inventory.items():
        path = directory / str(name)
        if not path.is_file() or sha256_file(path) != str(expected):
            raise RuntimeError(f"Downstream branch artifact changed: {path}.")
    member_identities = complete.get("member_run_identity_sha256")
    member_completions = complete.get("member_complete_sha256")
    expected_seeds = {str(int(value)) for value in downstream.DEFAULT_RUN_SEEDS}
    if not isinstance(member_identities, Mapping) or set(member_identities) != expected_seeds:
        raise RuntimeError("Downstream branch lacks exact member identity bindings.")
    if not isinstance(member_completions, Mapping) or set(member_completions) != expected_seeds:
        raise RuntimeError("Downstream branch lacks exact member completion bindings.")
    for run_seed in downstream.DEFAULT_RUN_SEEDS:
        member_dir = downstream.member_output_dir(directory, run_seed)
        member_complete = downstream.validate_completed_output(member_dir)
        if member_complete.get("run_identity_sha256") != member_identities[str(run_seed)]:
            raise RuntimeError("Downstream member identity binding changed.")
        if sha256_file(member_dir / "complete.json") != member_completions[str(run_seed)]:
            raise RuntimeError("Downstream member completion binding changed.")
    return complete


def _comparison(
    branch_results: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    reused = branch_results[ENCODER_BRANCHES[0]]
    retrained = branch_results[ENCODER_BRANCHES[1]]
    arms: dict[str, Any] = {}
    for arm in ARMS:
        metrics: dict[str, Any] = {}
        for metric in downstream.METRICS:
            reused_metric = reused["arms"][arm][metric]
            retrained_metric = retrained["arms"][arm][metric]
            reused_by_seed = {
                str(key): float(value)
                for key, value in reused_metric["values_by_run_seed"].items()
            }
            retrained_by_seed = {
                str(key): float(value)
                for key, value in retrained_metric["values_by_run_seed"].items()
            }
            if set(reused_by_seed) != set(retrained_by_seed):
                raise RuntimeError("Encoder branches have different downstream run seeds.")
            metrics[metric] = {
                "reuse_0914v1_mean": float(reused_metric["mean"]),
                "retrained_current_mean": float(retrained_metric["mean"]),
                "mean_delta_retrained_minus_reuse": float(
                    retrained_metric["mean"] - reused_metric["mean"]
                ),
                "paired_delta_by_run_seed": {
                    seed: float(retrained_by_seed[seed] - reused_by_seed[seed])
                    for seed in sorted(reused_by_seed, key=int)
                },
            }
        arms[arm] = metrics
    return {
        "schema": SCHEMA,
        "delta_definition": "R1_retrained_current_minus_R0_reuse_0914v1",
        "statistical_unit_warning": (
            "Both encoder branches use one fixed subject split and encoder seed 0; "
            "the four paired downstream seeds do not estimate cross-subject or "
            "encoder-initialization uncertainty."
        ),
        "arms": arms,
        "within_branch_contrasts": {
            branch: {
                "E1_minus_B0": result["paired_contrast"],
                "E2_minus_E1": result["soft_subject_paired_contrast"],
            }
            for branch, result in branch_results.items()
        },
    }


def _write_member_rows(
    path: Path,
    branch_results: Mapping[str, Mapping[str, Any]],
) -> None:
    rows: list[dict[str, Any]] = []
    for branch in ENCODER_BRANCHES:
        result = branch_results[branch]
        for arm in ARMS:
            for run_seed in downstream.DEFAULT_RUN_SEEDS:
                rows.append(
                    {
                        "encoder_branch": branch,
                        "encoder_seed": 0,
                        "run_seed": int(run_seed),
                        "arm": arm,
                        "metric_alignment": "global_hungarian_upper_bound",
                        **{
                            metric: float(
                                result["arms"][arm][metric]["values_by_run_seed"][
                                    str(int(run_seed))
                                ]
                            )
                            for metric in downstream.METRICS
                        },
                    }
                )
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _validate_complete(output: Path, identity: Mapping[str, Any]) -> dict[str, Any]:
    complete = _read_json(output / "complete.json")
    if complete.get("schema") != SCHEMA or complete.get("complete") is not True:
        raise RuntimeError("Dual-encoder completion marker is invalid.")
    if complete.get("dual_suite_identity_sha256") != identity.get("identity_sha256"):
        raise RuntimeError("Dual-encoder completion identity differs.")
    inventory = complete.get("artifact_sha256")
    if not isinstance(inventory, Mapping) or set(inventory) != set(ARTIFACT_NAMES):
        raise RuntimeError("Dual-encoder completion lacks the exact artifact inventory.")
    for name, expected in inventory.items():
        path = output / str(name)
        if not path.is_file() or sha256_file(path) != str(expected):
            raise RuntimeError(f"Dual-encoder artifact changed: {path}.")
    bindings = complete.get("branch_complete_sha256")
    if not isinstance(bindings, Mapping) or set(bindings) != set(ENCODER_BRANCHES):
        raise RuntimeError("Dual-encoder completion lacks exact branch bindings.")
    for branch, expected in bindings.items():
        path = _branch_output(output, branch) / "complete.json"
        if not path.is_file() or sha256_file(path) != str(expected):
            raise RuntimeError(f"Downstream branch completion changed: {path}.")
    encoder_manifest = _read_json(output / "encoder_branches.json")
    branch_encoders = encoder_manifest.get("branches")
    if not isinstance(branch_encoders, Mapping) or set(branch_encoders) != set(
        ENCODER_BRANCHES
    ):
        raise RuntimeError("Encoder branch manifest is incomplete.")
    checkpoints: dict[str, Path] = {}
    for branch, record in branch_encoders.items():
        if not isinstance(record, Mapping):
            raise RuntimeError("Encoder branch manifest record is invalid.")
        checkpoint = Path(str(record.get("checkpoint_path", ""))).expanduser().resolve()
        if not checkpoint.is_file() or sha256_file(checkpoint) != record.get(
            "checkpoint_sha256"
        ):
            raise RuntimeError(f"Encoder checkpoint changed for branch {branch}.")
        checkpoints[str(branch)] = checkpoint
    retrained_record = branch_encoders[ENCODER_BRANCHES[1]]
    window_dir, a2_dir = _retrained_directories(output)
    if sha256_file(window_dir / "complete.json") != retrained_record.get(
        "window_complete_sha256"
    ):
        raise RuntimeError("Retrained warm-up completion changed.")
    if sha256_file(a2_dir / "complete.json") != retrained_record.get(
        "a2_complete_sha256"
    ):
        raise RuntimeError("Retrained A2 completion changed.")
    for branch in ENCODER_BRANCHES:
        _validate_branch_suite(
            _branch_output(output, branch), checkpoint=checkpoints[branch]
        )
    return complete


def _run_unlocked(args: argparse.Namespace) -> dict[str, Any]:
    if int(args.encoder_seed) != 0:
        raise ValueError("The dual-encoder protocol locks encoder seed to 0.")
    downstream.parse_integer_list(args.run_seeds)
    registered_training = {
        "window_epochs": 60,
        "window_batch_size": 256,
        "window_eval_batch_size": 1024,
        "window_learning_rate": 0.1,
        "window_weight_decay": 5e-4,
        "window_weak_scale_std": 0.1,
        "window_strong_scale_std": 0.2,
        "a2_epochs": 30,
        "a2_trial_batch_size": 8,
        "a2_source_encode_batch_size": 1024,
        "a2_learning_rate": 1e-4,
        "a2_minimum_learning_rate": 1e-6,
        "a2_weight_decay": 1e-4,
        "cp_context_windows": 2,
        "num_workers": 0,
    }
    for name, expected in registered_training.items():
        observed = getattr(args, name)
        if isinstance(expected, float):
            matches = np.isclose(float(observed), expected, rtol=0.0, atol=1e-12)
        else:
            matches = int(observed) == expected
        if not bool(matches):
            raise ValueError(
                f"The dual-encoder comparison locks {name}={expected}; got {observed}."
            )
    if str(args.window_selection_policy) != "best_val_macro_f1":
        raise ValueError(
            "The registered 0914 protocol uses best_val_macro_f1 warm-up selection."
        )
    if bool(args.allow_smoke_a2) and not (
        int(args.smoke_max_windows) > 0
        and int(args.smoke_max_train_trials) > 0
        and int(args.smoke_max_val_trials) > 0
    ):
        raise ValueError("Smoke A2 requires explicit positive smoke limits.")

    output = Path(args.output_root).expanduser().resolve()
    npz_path = Path(args.npz_path).expanduser().resolve()
    reused_checkpoint = downstream.resolve_a2_checkpoint(
        args.reused_encoder_root, int(args.encoder_seed)
    )
    window_dir, a2_dir = _retrained_directories(output)
    encoder_args = _encoder_namespace(args)
    encoder_cv._validate_confirmation_hyperparameters(encoder_args)
    window_command = encoder_cv._window_command(encoder_args, 1, 0, window_dir)
    source_checkpoint = window_dir / "model_best.pt"
    a2_command = encoder_cv._a2_command(
        encoder_args, 1, 0, source_checkpoint, a2_dir
    )
    retrained_checkpoint = a2_dir / "motion_encoder_final.pt"
    branch_commands = [
        build_downstream_command(
            args,
            checkpoint=reused_checkpoint,
            branch=ENCODER_BRANCHES[0],
        ),
        build_downstream_command(
            args,
            checkpoint=retrained_checkpoint,
            branch=ENCODER_BRANCHES[1],
        ),
    ]
    body = _identity(args, reused_checkpoint=reused_checkpoint)
    if bool(args.dry_run):
        commands = [window_command, a2_command, *branch_commands]
        for command in commands:
            print("[command] " + shlex.join(command), flush=True)
        return {"dry_run": True, "identity": body, "commands": commands}

    identity = _prepare_identity(output, body)
    if (output / "complete.json").is_file():
        if not bool(args.resume):
            raise FileExistsError(f"Completed dual-encoder output exists: {output}.")
        return _validate_complete(output, identity)

    npz_hash = sha256_file(npz_path)
    window_arguments = encoder_cv._child_arguments(
        window_command, pretrain_window_encoder.build_parser()
    )
    window_identity = pretrain_window_encoder._run_identity(window_arguments)
    encoder_root = window_dir.parents[1]
    window_complete = encoder_cv._ensure_member(
        args=encoder_args,
        stage="retrained_current_window_pretrain",
        target=window_dir,
        validator=lambda: encoder_cv._validate_window_member(
            window_dir,
            fold=1,
            seed=0,
            npz_sha256=npz_hash,
            expected_identity=window_identity,
        ),
        command=window_command,
        root=encoder_root,
    )
    validated_source = window_dir / str(window_complete["checkpoint"])
    if validated_source.resolve() != source_checkpoint.resolve():
        raise RuntimeError("Validated warm-up checkpoint path differs from the A2 command.")
    a2_arguments = encoder_cv._child_arguments(
        a2_command, train_motion_encoder.build_parser()
    )
    a2_identity = train_motion_encoder.resolve_run_identity(a2_arguments)
    source_sha = sha256_file(validated_source)
    encoder_cv._ensure_member(
        args=encoder_args,
        stage="retrained_current_a2",
        target=a2_dir,
        validator=lambda: encoder_cv._validate_a2_member(
            a2_dir,
            fold=1,
            seed=0,
            npz_sha256=npz_hash,
            source_checkpoint_sha256=source_sha,
            expected_identity=a2_identity,
        ),
        command=a2_command,
        root=encoder_root,
    )
    validated_retrained = a2_dir / "motion_encoder_final.pt"
    if validated_retrained.resolve() != retrained_checkpoint.resolve():
        raise RuntimeError("Validated retrained A2 checkpoint path differs.")

    encoder_manifest = {
        "schema": SCHEMA,
        "encoder_seed": 0,
        "branches": {
            ENCODER_BRANCHES[0]: {
                "origin": "pretrained_20260914_v1",
                "checkpoint_path": str(reused_checkpoint),
                "checkpoint_sha256": sha256_file(reused_checkpoint),
                "trained_in_this_suite": False,
            },
            ENCODER_BRANCHES[1]: {
                "origin": "fresh_seeded_random_ResNet1D_then_A2",
                "checkpoint_path": str(validated_retrained),
                "checkpoint_sha256": sha256_file(validated_retrained),
                "trained_in_this_suite": True,
                "window_complete_sha256": sha256_file(window_dir / "complete.json"),
                "a2_complete_sha256": sha256_file(a2_dir / "complete.json"),
            },
        },
        "checkpoint_equal": bool(
            sha256_file(reused_checkpoint) == sha256_file(validated_retrained)
        ),
    }
    write_json(output / "encoder_branches.json", encoder_manifest)

    branch_results: dict[str, dict[str, Any]] = {}
    checkpoints = (reused_checkpoint, validated_retrained)
    for branch, checkpoint, command in zip(
        ENCODER_BRANCHES, checkpoints, branch_commands
    ):
        _run_command(command, stage=f"downstream {branch}")
        branch_results[branch] = _validate_branch_suite(
            _branch_output(output, branch), checkpoint=checkpoint
        )

    comparison = _comparison(branch_results)
    write_json(output / "branch_comparison.json", comparison)
    _write_member_rows(output / "member_rows.csv", branch_results)
    branch_bindings = {
        branch: sha256_file(_branch_output(output, branch) / "complete.json")
        for branch in ENCODER_BRANCHES
    }
    complete = {
        **comparison,
        "encoder_branches": encoder_manifest["branches"],
        "dual_suite_identity_sha256": identity["identity_sha256"],
        "branch_complete_sha256": branch_bindings,
        "artifact_sha256": {
            name: sha256_file(output / name) for name in ARTIFACT_NAMES
        },
        "complete": True,
    }
    write_json(output / "complete.json", complete)
    return complete


def run(args: argparse.Namespace) -> dict[str, Any]:
    if bool(args.dry_run):
        return _run_unlocked(args)
    output = Path(args.output_root).expanduser().resolve()
    with _exclusive_output_lock(output, lock_name=".dual_suite.lock"):
        return _run_unlocked(args)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compare the reused 20260914-v1 encoder with one freshly retrained "
            "W128/S64 fold-1 seed-0 encoder under the same B0/E1/E2 readout."
        )
    )
    parser.add_argument("--npz-path", required=True)
    parser.add_argument("--reused-encoder-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--encoder-seed", type=int, default=0, choices=(0,))
    parser.add_argument("--run-seeds", default="0,5,50,500")

    parser.add_argument("--window-epochs", type=int, default=60)
    parser.add_argument("--window-batch-size", type=int, default=256)
    parser.add_argument("--window-eval-batch-size", type=int, default=1024)
    parser.add_argument("--window-learning-rate", type=float, default=0.1)
    parser.add_argument("--window-weight-decay", type=float, default=5e-4)
    parser.add_argument("--window-weak-scale-std", type=float, default=0.1)
    parser.add_argument("--window-strong-scale-std", type=float, default=0.2)
    parser.add_argument(
        "--window-selection-policy",
        choices=("best_val_macro_f1",),
        default="best_val_macro_f1",
    )
    parser.add_argument("--a2-epochs", type=int, default=30)
    parser.add_argument("--a2-trial-batch-size", type=int, default=8)
    parser.add_argument("--a2-source-encode-batch-size", type=int, default=1024)
    parser.add_argument("--a2-learning-rate", type=float, default=1e-4)
    parser.add_argument("--a2-minimum-learning-rate", type=float, default=1e-6)
    parser.add_argument("--a2-weight-decay", type=float, default=1e-4)
    parser.add_argument("--cp-context-windows", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--smoke-max-windows", type=int, default=0)
    parser.add_argument("--smoke-max-train-trials", type=int, default=0)
    parser.add_argument("--smoke-max-val-trials", type=int, default=0)

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
        default=0.25,
        choices=(0.25,),
    )
    parser.add_argument("--static-motion-primitive-pca-dim", type=int, default=8)
    parser.add_argument("--static-posture-weight", type=float, default=0.35)
    parser.add_argument("--static-gravity-weight", type=float, default=0.35)
    parser.add_argument("--static-energy-weight", type=float, default=0.20)
    parser.add_argument("--static-motion-primitive-weight", type=float, default=0.10)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--allow-smoke-a2", action="store_true")
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
    "ARTIFACT_NAMES",
    "ENCODER_BRANCHES",
    "EXPECTED_NPZ_SHA256",
    "EXPECTED_REUSED_A2_SHA256",
    "IDENTITY_SCHEMA",
    "SCHEMA",
    "build_downstream_command",
    "build_parser",
    "main",
    "run",
]
