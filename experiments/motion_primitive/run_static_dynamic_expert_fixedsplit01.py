"""Run the current fixed-split-01 static-expert route over four run seeds."""

from __future__ import annotations

import argparse
import csv
import json
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import numpy as np
import sklearn
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.motion_primitive.static_dynamic_batch_proxy import (
    ARMS,
    FIXED_SPLIT,
    SCHEMA as MEMBER_SCHEMA,
    _exclusive_output_lock,
    _implementation_hashes as member_implementation_hashes,
    validate_completed_output,
)
from experiments.motion_primitive.strict_artifacts import write_json
from experiments.motion_primitive.strict_cv_common import canonical_hash
from experiments.motion_primitive.strict_protocol import sha256_file


SCHEMA = "hhr_static_dynamic_expert_fixedsplit01_4runseed_v2"
IDENTITY_SCHEMA = "hhr_static_dynamic_expert_fixedsplit01_4runseed_identity_v2"
DEFAULT_RUN_SEEDS = (0, 5, 50, 500)
METRICS = ("all_accuracy", "old_accuracy", "new_accuracy", "h_score", "macro_f1", "ari", "nmi")


def parse_integer_list(value: str) -> tuple[int, ...]:
    result = tuple(int(item.strip()) for item in str(value).split(",") if item.strip())
    if result != DEFAULT_RUN_SEEDS:
        raise ValueError(
            "The current fixed-split protocol locks --run-seeds to 0,5,50,500."
        )
    return result


def resolve_a2_checkpoint(root_value: str | Path, encoder_seed: int) -> Path:
    """Resolve the registered W128/S64 fold-1 final A2 checkpoint fail-closed."""

    root = Path(root_value).expanduser().resolve()
    if root.is_file():
        if root.name != "motion_encoder_final.pt":
            raise ValueError("An explicit encoder file must be motion_encoder_final.pt.")
        return root
    relative = Path("a2") / f"fold_01_seed_{int(encoder_seed)}" / "motion_encoder_final.pt"
    candidates = (
        root / "encoders" / "w128_s64" / relative,
        root / "w128_s64" / relative,
        root / relative,
        root / f"fold_01_seed_{int(encoder_seed)}" / "motion_encoder_final.pt",
    )
    existing = []
    for candidate in candidates:
        if candidate.is_file() and candidate not in existing:
            existing.append(candidate)
    if len(existing) != 1:
        raise FileNotFoundError(
            "Expected exactly one W128/S64 fold-1 A2 final checkpoint below "
            f"{root}; found {[str(path) for path in existing]}."
        )
    return existing[0]


def member_output_dir(output_root: Path, run_seed: int) -> Path:
    return output_root / "members" / f"fixed_split_01_run_seed_{int(run_seed)}"


def build_member_command(
    args: argparse.Namespace,
    *,
    checkpoint: Path,
    run_seed: int,
) -> list[str]:
    command = [
        sys.executable,
        str(PROJECT_ROOT / "experiments/motion_primitive/static_dynamic_batch_proxy.py"),
        "--a2-checkpoint",
        str(checkpoint),
        "--npz-path",
        str(Path(args.npz_path).expanduser().resolve()),
        "--output-dir",
        str(member_output_dir(Path(args.output_root).expanduser().resolve(), run_seed)),
        "--run-seed",
        str(int(run_seed)),
        "--encoder-seed",
        str(int(args.encoder_seed)),
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


def _identity(
    args: argparse.Namespace,
    *,
    checkpoint: Path,
    run_seeds: Sequence[int],
) -> dict[str, Any]:
    npz_path = Path(args.npz_path).expanduser().resolve()
    if not npz_path.is_file():
        raise FileNotFoundError(npz_path)
    suite_source = (
        PROJECT_ROOT / "experiments/motion_primitive/run_static_dynamic_expert_fixedsplit01.py"
    )
    return {
        "schema": IDENTITY_SCHEMA,
        "fixed_split": FIXED_SPLIT,
        "encoder_fold": FIXED_SPLIT,
        "encoder_seed": int(args.encoder_seed),
        "run_seeds": [int(seed) for seed in run_seeds],
        "seed_semantics": "four_downstream_run_seeds_share_one_frozen_encoder",
        "a2_checkpoint_path": str(checkpoint),
        "a2_checkpoint_sha256": sha256_file(checkpoint),
        "npz_path": str(npz_path),
        "npz_sha256": sha256_file(npz_path),
        "arms": list(ARMS),
        "parameters": {
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
            "static_motion_primitive_pca_dim": int(args.static_motion_primitive_pca_dim),
            "static_posture_weight": float(args.static_posture_weight),
            "static_gravity_weight": float(args.static_gravity_weight),
            "static_energy_weight": float(args.static_energy_weight),
            "static_motion_primitive_weight": float(args.static_motion_primitive_weight),
            "device": str(args.device),
            "allow_smoke_a2": bool(args.allow_smoke_a2),
        },
        "runtime": {
            "python": sys.version,
            "numpy": np.__version__,
            "sklearn": sklearn.__version__,
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "cuda_available": bool(torch.cuda.is_available()),
        },
        "implementation_sha256": {
            str(suite_source.relative_to(PROJECT_ROOT)).replace("\\", "/"): sha256_file(
                suite_source
            ),
            **member_implementation_hashes(),
        },
    }


def _prepare_identity(output: Path, body: Mapping[str, Any]) -> dict[str, Any]:
    expected = {**dict(body), "identity_sha256": canonical_hash(body)}
    path = output / "suite_identity.json"
    if path.is_file():
        observed = json.loads(path.read_text(encoding="utf-8"))
        if observed != expected:
            raise RuntimeError(f"Output root records another suite identity: {output}.")
        return expected
    if output.exists():
        unexpected = [item.name for item in output.iterdir() if item.name != ".suite.lock"]
        if unexpected:
            raise RuntimeError(
                f"Non-empty output root has no suite identity: {output}; {unexpected}."
            )
    output.mkdir(parents=True, exist_ok=True)
    write_json(path, expected)
    return expected


def _aggregate(members: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for arm in ARMS:
        arm_summary: dict[str, Any] = {}
        for metric in METRICS:
            values = np.asarray(
                [float(member["arms"][arm][metric]) for member in members], dtype=np.float64
            )
            arm_summary[metric] = {
                "mean": float(values.mean()),
                "sample_std_across_run_seeds": float(values.std(ddof=1)),
                "values_by_run_seed": {
                    str(int(member["run_seed"])): float(value)
                    for member, value in zip(members, values)
                },
            }
        result[arm] = arm_summary
    contrast: dict[str, Any] = {}
    soft_contrast: dict[str, Any] = {}
    for metric in METRICS:
        values = np.asarray(
            [
                float(member["arms"]["E1_gate_static_expert"][metric])
                - float(member["arms"]["B0_global_trajectory"][metric])
                for member in members
            ],
            dtype=np.float64,
        )
        contrast[metric] = {
            "mean_delta_E1_minus_B0": float(values.mean()),
            "sample_std_across_run_seeds": float(values.std(ddof=1)),
            "values_by_run_seed": {
                str(int(member["run_seed"])): float(value)
                for member, value in zip(members, values)
            },
        }
        soft_values = np.asarray(
            [
                float(member["arms"]["E2_gate_static_expert_soft_a025"][metric])
                - float(member["arms"]["E1_gate_static_expert"][metric])
                for member in members
            ],
            dtype=np.float64,
        )
        soft_contrast[metric] = {
            "mean_delta_E2_minus_E1": float(soft_values.mean()),
            "sample_std_across_run_seeds": float(soft_values.std(ddof=1)),
            "values_by_run_seed": {
                str(int(member["run_seed"])): float(value)
                for member, value in zip(members, soft_values)
            },
        }
    return {
        "arms": result,
        "paired_contrast": contrast,
        "soft_subject_paired_contrast": soft_contrast,
    }


def _write_rows(path: Path, members: Sequence[Mapping[str, Any]]) -> None:
    rows = []
    for member in members:
        for arm in ARMS:
            rows.append(
                {
                    "fixed_split": FIXED_SPLIT,
                    "encoder_seed": int(member["encoder_seed"]),
                    "run_seed": int(member["run_seed"]),
                    "arm": arm,
                    "metric_alignment": "global_hungarian_upper_bound",
                    "selected_dynamic_k": int(member["selected_dynamic_k"]),
                    "selected_static_k": int(member["selected_static_k"]),
                    "gate_accuracy_diagnostic": float(
                        member["gate_posttruth_accuracy_diagnostic"]
                    ),
                    **{metric: float(member["arms"][arm][metric]) for metric in METRICS},
                }
            )
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _run_suite_unlocked(args: argparse.Namespace) -> dict[str, Any]:
    run_seeds = parse_integer_list(args.run_seeds)
    if int(args.encoder_seed) != 0:
        raise ValueError("The current fixed-split protocol locks --encoder-seed to 0.")
    if int(args.subject_nuisance_max_rank) < 1:
        raise ValueError("Subject-nuisance maximum rank must be positive.")
    if not 0.0 < float(args.subject_nuisance_explained_variance) <= 1.0:
        raise ValueError("Subject-nuisance explained variance must lie in (0,1].")
    if not np.isclose(
        float(args.subject_nuisance_projection_strength), 0.25, rtol=0.0, atol=1e-12
    ):
        raise ValueError("This suite locks soft subject projection strength to 0.25.")
    checkpoint = resolve_a2_checkpoint(args.encoder_root, int(args.encoder_seed))
    output = Path(args.output_root).expanduser().resolve()
    body = _identity(args, checkpoint=checkpoint, run_seeds=run_seeds)
    if bool(args.dry_run):
        commands = [
            build_member_command(args, checkpoint=checkpoint, run_seed=seed)
            for seed in run_seeds
        ]
        for command in commands:
            print("[command] " + shlex.join(command), flush=True)
        return {"dry_run": True, "commands": commands, "identity": body}

    identity = _prepare_identity(output, body)
    complete_path = output / "complete.json"
    if complete_path.is_file():
        if not bool(args.resume):
            raise FileExistsError(f"Completed output already exists: {output}.")
        complete = json.loads(complete_path.read_text(encoding="utf-8"))
        if complete.get("complete") is not True or complete.get("schema") != SCHEMA:
            raise RuntimeError("Suite completion marker is invalid.")
        if complete.get("suite_identity_sha256") != identity["identity_sha256"]:
            raise RuntimeError("Suite completion identity differs from the current request.")
        inventory = complete.get("artifact_sha256")
        if not isinstance(inventory, Mapping) or set(inventory) != {
            "aggregate.json",
            "member_rows.csv",
        }:
            raise RuntimeError("Suite completion lacks the exact artifact inventory.")
        for name, expected in inventory.items():
            path = output / str(name)
            if not path.is_file() or sha256_file(path) != str(expected):
                raise RuntimeError(f"Suite artifact changed: {path}.")
        aggregate_path = output / "aggregate.json"
        aggregate = json.loads(aggregate_path.read_text(encoding="utf-8"))
        if complete.get("aggregate_sha256") != sha256_file(aggregate_path):
            raise RuntimeError("Suite completion is not bound to aggregate.json.")
        for key, value in aggregate.items():
            if complete.get(key) != value:
                raise RuntimeError(f"Suite completion field {key!r} differs from aggregate.json.")
        member_bindings = complete.get("member_run_identity_sha256")
        if not isinstance(member_bindings, Mapping) or set(member_bindings) != {
            str(int(seed)) for seed in run_seeds
        }:
            raise RuntimeError("Suite completion lacks exact member identity bindings.")
        member_complete_bindings = complete.get("member_complete_sha256")
        if not isinstance(member_complete_bindings, Mapping) or set(
            member_complete_bindings
        ) != {str(int(seed)) for seed in run_seeds}:
            raise RuntimeError("Suite completion lacks exact member completion bindings.")
        for run_seed in run_seeds:
            child_dir = member_output_dir(output, run_seed)
            child = validate_completed_output(child_dir)
            if int(child.get("encoder_seed", -1)) != int(args.encoder_seed):
                raise RuntimeError("A resumed member used another encoder seed.")
            if int(child.get("run_seed", -1)) != int(run_seed):
                raise RuntimeError("A resumed member run seed differs.")
            expected_child_identity = member_bindings[str(int(run_seed))]
            if child.get("run_identity_sha256") != expected_child_identity:
                raise RuntimeError("A resumed member identity differs from the suite binding.")
            if sha256_file(child_dir / "complete.json") != member_complete_bindings[
                str(int(run_seed))
            ]:
                raise RuntimeError(
                    "A resumed member completion differs from the suite binding."
                )
        return complete

    members: list[dict[str, Any]] = []
    member_identity_sha256: dict[str, str] = {}
    member_complete_sha256: dict[str, str] = {}
    for run_seed in run_seeds:
        command = build_member_command(args, checkpoint=checkpoint, run_seed=run_seed)
        print("[command] " + shlex.join(command), flush=True)
        subprocess.run(command, cwd=PROJECT_ROOT, check=True)
        member_dir = member_output_dir(output, run_seed)
        complete = validate_completed_output(member_dir)
        if complete.get("schema") != MEMBER_SCHEMA:
            raise RuntimeError("Member schema differs from the registered suite.")
        if int(complete.get("fixed_split", -1)) != FIXED_SPLIT:
            raise RuntimeError("A member escaped fixed split 01.")
        if int(complete.get("encoder_seed", -1)) != int(args.encoder_seed):
            raise RuntimeError("A member used another encoder seed.")
        if int(complete.get("run_seed", -1)) != int(run_seed):
            raise RuntimeError("A member run seed differs from its command.")
        member_identity_sha256[str(int(run_seed))] = str(
            complete["run_identity_sha256"]
        )
        member_complete_sha256[str(int(run_seed))] = sha256_file(
            member_dir / "complete.json"
        )
        members.append(complete)

    checkpoint_hashes = {
        json.loads((member_output_dir(output, seed) / "run_identity.json").read_text(encoding="utf-8"))[
            "a2_checkpoint_sha256"
        ]
        for seed in run_seeds
    }
    if checkpoint_hashes != {sha256_file(checkpoint)}:
        raise RuntimeError("The four downstream runs did not share exactly one encoder checkpoint.")
    aggregate = {
        "schema": SCHEMA,
        "fixed_split": FIXED_SPLIT,
        "encoder_seed": int(args.encoder_seed),
        "run_seeds": list(run_seeds),
        "member_run_identity_sha256": member_identity_sha256,
        "member_complete_sha256": member_complete_sha256,
        "metric_alignment": "global_hungarian_upper_bound",
        "metric_interpretation": "unlabeled_clustering_diagnostic_upper_bound",
        "statistical_unit_warning": (
            "The four run seeds share one subject split and one frozen encoder; "
            "their standard deviation is algorithmic variability, not a cross-subject CI."
        ),
        "cross_fold_confidence_interval": "not_estimable_from_one_fixed_split",
        **_aggregate(members),
    }
    write_json(output / "aggregate.json", aggregate)
    _write_rows(output / "member_rows.csv", members)
    artifact_names = ("aggregate.json", "member_rows.csv")
    complete = {
        **aggregate,
        "suite_identity_sha256": identity["identity_sha256"],
        "aggregate_sha256": sha256_file(output / "aggregate.json"),
        "artifact_sha256": {name: sha256_file(output / name) for name in artifact_names},
        "complete": True,
    }
    write_json(complete_path, complete)
    return complete


def run(args: argparse.Namespace) -> dict[str, Any]:
    if bool(args.dry_run):
        return _run_suite_unlocked(args)
    output = Path(args.output_root).expanduser().resolve()
    with _exclusive_output_lock(output, lock_name=".suite.lock"):
        return _run_suite_unlocked(args)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Current HHR route: fixed split 01, one frozen A2 encoder, and four "
            "downstream seeds for duration, gate+static expert, and its A025 "
            "soft-subject dynamic extension."
        )
    )
    parser.add_argument("--npz-path", required=True)
    parser.add_argument(
        "--encoder-root",
        required=True,
        help=(
            "The 20260914 experiment root, its encoders/w128_s64 directory, or an "
            "explicit motion_encoder_final.pt."
        ),
    )
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--encoder-seed", type=int, default=0, choices=(0,))
    parser.add_argument(
        "--run-seeds", default=",".join(str(value) for value in DEFAULT_RUN_SEEDS)
    )
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
    "DEFAULT_RUN_SEEDS",
    "IDENTITY_SCHEMA",
    "SCHEMA",
    "build_member_command",
    "build_parser",
    "main",
    "member_output_dir",
    "parse_integer_list",
    "resolve_a2_checkpoint",
    "run",
]
