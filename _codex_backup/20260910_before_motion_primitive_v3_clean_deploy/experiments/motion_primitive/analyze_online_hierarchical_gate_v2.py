"""Audit and aggregate frozen-readout adaptive-codebook v2 experiments.

The v2 experiment deliberately separates two questions:

* did a local codebook expansion change only explicitly routed trials; and
* did the resulting frozen cluster vector improve post-hoc CGCD metrics?

The first question is checked on raw cluster ids and is a hard invariant.  The
second uses the aligned predictions written by the runner.  Per-arm Hungarian
alignment can change the semantic label attached to an unchanged raw cluster,
so those alignment-only changes are reported separately and are never treated
as evidence that the local route itself changed a fallback trial.

This module is evaluation-only.  Activity labels are loaded from the frozen
prediction artifact after all fitting/routing audits have been validated.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import adjusted_rand_score, f1_score, normalized_mutual_info_score

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.motion_primitive.trajectory_ablation import jsonable, sha256_file


BASELINE_ARM = "F0_frozen_K32"
REGISTERED_ARM = "F1_registered_K32_or_K34"
ADAPTIVE_ARM = "F2_adaptive_K32_or_K34"
ARMS = (BASELINE_ARM, REGISTERED_ARM, ADAPTIVE_ARM)
CHALLENGERS = (REGISTERED_ARM, ADAPTIVE_ARM)

ARM_ALIASES = {
    BASELINE_ARM: (
        BASELINE_ARM,
        "F0_frozen_coarse",
        "F0_coarse",
        "G0_frozen_coarse",
        "G0_coarse",
    ),
    REGISTERED_ARM: (
        REGISTERED_ARM,
        "F1_registered_K34",
        "F1_frozen_K34",
        "F1_registered",
        "F1_frozen_hierarchical",
        "G1_registered_K34",
    ),
    ADAPTIVE_ARM: (
        ADAPTIVE_ARM,
        "F2_adaptive",
        "F2_adaptive_hierarchical",
        "G2_adaptive_K32_or_K34",
        "G3_frozen_hierarchical",
    ),
}

RESULT_FILENAMES = (
    "online_adaptive_codebook_v2_results.json",
    "online_hierarchical_gate_v2_results.json",
)
DISCOVERY_IGNORED_DIRECTORY_NAMES = frozenset({"_incomplete"})
PREDICTION_FILENAMES = (
    "session2_predictions.csv",
    "adaptive_codebook_v2_predictions.csv",
    "predictions.csv",
)

QUALITY_METRICS = (
    "all_accuracy",
    "old_accuracy",
    "new_accuracy",
    "h_score",
    "macro_f1",
    "ari",
    "nmi",
)
PERCENTAGE_METRICS = {
    "all_accuracy",
    "old_accuracy",
    "new_accuracy",
    "h_score",
    "macro_f1",
}


@dataclass(frozen=True)
class AuditedRun:
    """One fully audited v2 result directory."""

    directory: Path
    result_path: Path
    result: dict
    fold: int
    seed: int
    profile: str
    segmentation: str
    session: int
    trial_ids: np.ndarray
    subjects: np.ndarray
    truth: np.ndarray
    actual_arm_names: Mapping[str, str]
    raw_predictions: Mapping[str, np.ndarray]
    aligned_predictions: Mapping[str, np.ndarray]
    routed_masks: Mapping[str, np.ndarray]
    metrics: Mapping[str, Mapping[str, float]]
    arm_audits: Mapping[str, Mapping[str, Any]]
    intervention_audits: Mapping[str, Mapping[str, Any]]

    @property
    def config_id(self) -> str:
        return (
            f"profile={self.profile}|segmentation={self.segmentation}|"
            f"session={self.session}"
        )


def _read_json(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"Result JSON must contain an object: {path}")
    return value


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise RuntimeError(f"Prediction CSV is empty: {path}")
    return rows


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Cannot write an empty table: {path}")
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _nested(mapping: Mapping[str, Any], *keys: str) -> Any:
    value: Any = mapping
    for key in keys:
        if not isinstance(value, Mapping) or key not in value:
            return None
        value = value[key]
    return value


def _first_present(mapping: Mapping[str, Any], paths: Iterable[Sequence[str]]) -> Any:
    for path in paths:
        value = _nested(mapping, *path)
        if value is not None and value != "":
            return value
    return None


def _deep_find(mapping: Mapping[str, Any], names: set[str]) -> Any:
    """Find the first exact key recursively, preserving JSON insertion order."""

    for key, value in mapping.items():
        if str(key) in names and value is not None and value != "":
            return value
    for value in mapping.values():
        if isinstance(value, Mapping):
            found = _deep_find(value, names)
            if found is not None:
                return found
    return None


def _deep_key_values(
    value: Any, names: set[str], path: tuple[str, ...] = ()
) -> list[tuple[tuple[str, ...], Any]]:
    """Collect every exact-key occurrence, including mappings inside lists.

    Security/provenance flags must be checked exhaustively.  Returning the
    first recursive match is insufficient when both registered and adaptive
    arms report the same audit key.
    """

    found: list[tuple[tuple[str, ...], Any]] = []
    if isinstance(value, Mapping):
        for key, nested in value.items():
            nested_path = (*path, str(key))
            if str(key) in names:
                found.append((nested_path, nested))
            found.extend(_deep_key_values(nested, names, nested_path))
    elif isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray)
    ):
        for index, nested in enumerate(value):
            found.extend(_deep_key_values(nested, names, (*path, str(index))))
    return found


def _finite_float(value: Any, context: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise RuntimeError(f"{context} must be numeric.") from error
    if not np.isfinite(result):
        raise RuntimeError(f"{context} must be finite.")
    return result


def _integer(value: Any, context: str) -> int:
    numeric = _finite_float(value, context)
    if not numeric.is_integer():
        raise RuntimeError(f"{context} must be an integer.")
    return int(numeric)


def _strict_bool(value: Any, context: str) -> bool:
    if type(value) is bool:
        return value
    if isinstance(value, (int, np.integer)) and int(value) in (0, 1):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1"}:
            return True
        if normalized in {"false", "0"}:
            return False
    raise RuntimeError(f"{context} must be a strict boolean.")


def _optional_int(value: Any, context: str) -> int | None:
    if value is None or value == "" or str(value).strip().lower() in {"none", "null"}:
        return None
    return _integer(value, context)


def _json_cell(value: Any) -> str:
    if value is None:
        return ""
    return json.dumps(jsonable(value), ensure_ascii=False, sort_keys=True)


def _find_prediction_path(directory: Path) -> Path:
    for filename in PREDICTION_FILENAMES:
        candidate = directory / filename
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        f"No v2 prediction CSV found in {directory}; expected one of "
        f"{list(PREDICTION_FILENAMES)}."
    )


def discover_result_paths(inputs: Sequence[Path]) -> list[Path]:
    """Discover active v2 JSON artifacts without reading quarantined runs.

    Resume-safe runners retain replaced or interrupted results below an
    ``_incomplete`` directory.  Those files are deliberately recoverable, but
    they are not members of the active experiment grid and must never enter a
    recursive aggregate alongside their replacements.
    """

    discovered: set[Path] = set()
    for input_value in inputs:
        path = Path(input_value).resolve()
        if path.is_file():
            if path.name not in RESULT_FILENAMES:
                raise ValueError(
                    f"Explicit result file must be named one of {RESULT_FILENAMES}: {path}"
                )
            discovered.add(path)
            continue
        if not path.is_dir():
            raise FileNotFoundError(path)
        for filename in RESULT_FILENAMES:
            direct = path / filename
            if direct.is_file():
                discovered.add(direct.resolve())
            for candidate in path.rglob(filename):
                relative = candidate.relative_to(path)
                if any(
                    part.casefold() in DISCOVERY_IGNORED_DIRECTORY_NAMES
                    for part in relative.parts[:-1]
                ):
                    continue
                discovered.add(candidate.resolve())
    if not discovered:
        raise RuntimeError(
            "No v2 result JSON was discovered beneath the supplied input roots."
        )
    return sorted(discovered, key=lambda item: str(item).lower())


def _resolve_arm_names(result: Mapping[str, Any]) -> dict[str, str]:
    arms = _mapping(result.get("arms"))
    resolved: dict[str, str] = {}
    for canonical, aliases in ARM_ALIASES.items():
        matches = [alias for alias in aliases if alias in arms]
        if len(matches) != 1:
            raise RuntimeError(
                f"Expected exactly one alias for {canonical}, found {matches}; "
                f"available arms={sorted(arms)}."
            )
        resolved[canonical] = matches[0]
    if len(set(resolved.values())) != len(ARMS):
        raise RuntimeError("Multiple canonical arms resolved to the same result arm.")
    return resolved


def _identity_integer(result: Mapping[str, Any], directory: Path, field: str) -> int:
    if field == "fold":
        value = _first_present(
            result,
            (
                ("fold",),
                ("arguments", "fold"),
                ("arguments", "cv_fold"),
                ("arguments", "uschad_cv_fold"),
                ("input_audit", "fold"),
                ("input_audit", "uschad_cv_fold"),
                ("input_audit", "checkpoint_metadata", "uschad_cv_fold"),
            ),
        )
        if value is None:
            value = _deep_find(
                _mapping(result.get("input_audit")),
                {"fold", "cv_fold", "uschad_cv_fold"},
            )
        if value is None:
            candidates = [
                str(directory),
                str(_nested(result, "arguments", "run_dir") or ""),
                str(_nested(result, "input_audit", "registered_upstream") or ""),
            ]
            for candidate in candidates:
                match = re.search(r"fold[_-]?0*(\d+)", candidate, flags=re.I)
                if match:
                    value = match.group(1)
                    break
    elif field == "seed":
        value = _first_present(
            result,
            (
                ("seed",),
                ("arguments", "seed"),
                ("session_manifest_audit", "seed"),
                ("input_audit", "seed"),
            ),
        )
        if value is None:
            match = re.search(r"seed[_-]?(-?\d+)", str(directory), flags=re.I)
            value = match.group(1) if match else None
    else:
        raise ValueError(field)
    if value is None:
        raise RuntimeError(f"Cannot determine {field} for {directory}.")
    return _integer(value, f"{directory} {field}")


def _identity_text(result: Mapping[str, Any], directory: Path, field: str) -> str:
    if field == "profile":
        value = _first_present(
            result,
            (
                ("profile",),
                ("arguments", "profile"),
                ("arguments", "encoder_profile"),
                ("input_audit", "encoder_ablation_profile"),
                ("input_audit", "profile"),
            ),
        )
        if value is None:
            match = re.search(r"(?:^|[_/\\-])(A[0-9]+)(?:[_/\\-]|$)", str(directory))
            value = match.group(1) if match else None
    elif field == "segmentation":
        value = _first_present(
            result,
            (
                ("segmentation",),
                ("arguments", "primitive_segmentation"),
                ("input_audit", "primitive_segmentation"),
            ),
        )
        if value is None:
            lowered = str(directory).lower()
            value = "fixed_window" if "fixed" in lowered else None
    else:
        raise ValueError(field)
    if value is None or not str(value).strip():
        raise RuntimeError(f"Cannot determine {field} for {directory}.")
    return str(value).strip()


def _session_number(result: Mapping[str, Any]) -> int:
    value = _first_present(
        result,
        (
            ("session",),
            ("arguments", "session"),
            ("arguments", "session_number"),
            ("session_manifest_audit", "evaluated_session"),
        ),
    )
    return 2 if value is None else _integer(value, "evaluated session")


def _column_values(
    rows: Sequence[Mapping[str, str]], candidates: Sequence[str], context: str
) -> tuple[str, list[str]]:
    for candidate in candidates:
        if all(candidate in row and str(row[candidate]).strip() != "" for row in rows):
            return candidate, [str(row[candidate]) for row in rows]
    raise RuntimeError(f"Prediction CSV lacks {context}; tried columns {list(candidates)}.")


def _prediction_columns(
    rows: Sequence[Mapping[str, str]], canonical: str, actual: str, kind: str
) -> np.ndarray:
    prefixes = list(dict.fromkeys((actual, canonical, *ARM_ALIASES[canonical])))
    suffixes = (
        ("raw_cluster", "raw_prediction", "raw")
        if kind == "raw"
        else ("aligned_prediction", "aligned")
    )
    candidates = [f"{prefix}_{suffix}" for prefix in prefixes for suffix in suffixes]
    _, values = _column_values(rows, candidates, f"{canonical} {kind} predictions")
    return np.asarray([_integer(value, f"{canonical} {kind}") for value in values])


def _arm_mapping(container: Mapping[str, Any], canonical: str, actual: str) -> Mapping[str, Any]:
    for key in (actual, canonical, *ARM_ALIASES[canonical]):
        value = container.get(key)
        if isinstance(value, Mapping):
            return value
    return {}


def _route_mask_from_csv(
    rows: Sequence[Mapping[str, str]], canonical: str, actual: str
) -> np.ndarray | None:
    short = canonical.split("_", 1)[0]
    prefixes = list(dict.fromkeys((actual, canonical, short, *ARM_ALIASES[canonical])))
    bool_candidates = [
        f"{prefix}_{suffix}"
        for prefix in prefixes
        for suffix in ("routed", "route", "routed_to_leaf", "route_mask")
    ]
    for candidate in bool_candidates:
        if all(candidate in row and str(row[candidate]).strip() != "" for row in rows):
            return np.asarray(
                [_strict_bool(row[candidate], candidate) for row in rows], dtype=bool
            )
    state_candidates = [
        f"{prefix}_{suffix}"
        for prefix in prefixes
        for suffix in ("gate_state", "state")
    ]
    if canonical == REGISTERED_ARM:
        state_candidates.extend(("registered_gate_state", "registered_state"))
    elif canonical == ADAPTIVE_ARM:
        state_candidates.extend(("adaptive_gate_state", "adaptive_state"))
    for candidate in state_candidates:
        if all(candidate in row and str(row[candidate]).strip() != "" for row in rows):
            states = np.asarray(
                [_integer(row[candidate], candidate) for row in rows], dtype=np.int64
            )
            if np.any(states < 0):
                raise RuntimeError(f"{candidate} contains a negative gate state.")
            return states > 0
    return None


def _route_mask(
    result: Mapping[str, Any],
    rows: Sequence[Mapping[str, str]],
    trial_ids: np.ndarray,
    canonical: str,
    actual: str,
    k_total: int,
) -> np.ndarray:
    from_csv = _route_mask_from_csv(rows, canonical, actual)
    if from_csv is not None:
        return from_csv

    readout = _mapping(result.get("readout_audit"))
    arm_readout = _arm_mapping(readout, canonical, actual)
    ids = _deep_find(arm_readout, {"routed_trial_ids", "route_trial_ids"})
    if ids is None:
        descriptor = _arm_descriptor(result, canonical)
        ids = _deep_find(descriptor, {"routed_trial_ids", "route_trial_ids"})
    if ids is not None:
        try:
            routed_ids = {int(value) for value in ids}
        except (TypeError, ValueError) as error:
            raise RuntimeError(f"{canonical} routed_trial_ids is invalid.") from error
        unknown = routed_ids - set(trial_ids.tolist())
        if unknown:
            raise RuntimeError(f"{canonical} route audit includes non-test ids {sorted(unknown)}.")
        return np.isin(trial_ids, np.asarray(sorted(routed_ids), dtype=np.int64))

    if canonical == ADAPTIVE_ARM and int(k_total) == 32:
        return np.zeros(len(trial_ids), dtype=bool)
    raise RuntimeError(
        f"{canonical} lacks an independent routed mask. Raw prediction changes "
        "cannot be used to infer routing because that would hide fallback mismatches."
    )


def _arm_descriptor(result: Mapping[str, Any], canonical: str) -> Mapping[str, Any]:
    if canonical == REGISTERED_ARM:
        registered = _mapping(result.get("registered_gate_fit"))
        safety = _mapping(result.get("registered_gate_v2_safety_audit"))
        if safety:
            # The operational F1 arm is the previously registered v1
            # mechanism.  The v2 object is a safety/generalisation audit and
            # is therefore preferred only for support/stability reporting.
            merged = dict(safety)
            merged["operational_registered_gate_fit"] = registered
            return merged
        return registered
    if canonical == ADAPTIVE_ARM:
        adaptive = _mapping(result.get("adaptive_selection"))
        for key in ("selected_gate_fit", "gate_fit", "selected_candidate"):
            nested = adaptive.get(key)
            if isinstance(nested, Mapping):
                # Put the selected candidate first.  A candidate_audits list
                # may contain rejected parents before the selected parent;
                # recursive extraction must never report one of those as the
                # expansion that was actually applied.
                merged = dict(nested)
                for outer_key, outer_value in adaptive.items():
                    if outer_key not in merged:
                        merged[outer_key] = outer_value
                return merged
        selected_parent = _deep_find(adaptive, {"selected_parent_token"})
        candidates = adaptive.get("candidate_audits")
        if selected_parent is not None and isinstance(candidates, Sequence):
            selected_parent_int = _integer(
                selected_parent, "adaptive selected_parent_token"
            )
            matches = [
                item
                for item in candidates
                if isinstance(item, Mapping)
                and _optional_int(
                    item.get("parent_token"), "adaptive candidate parent_token"
                )
                == selected_parent_int
            ]
            if len(matches) != 1:
                raise RuntimeError(
                    "adaptive_selection must contain exactly one candidate audit "
                    f"for selected parent {selected_parent_int}, found {len(matches)}."
                )
            merged = dict(matches[0])
            for outer_key, outer_value in adaptive.items():
                if outer_key not in merged:
                    merged[outer_key] = outer_value
            return merged
        return adaptive
    return {}


def _codebook_arm_audit(
    result: Mapping[str, Any], canonical: str, actual: str
) -> Mapping[str, Any]:
    audit = _mapping(result.get("codebook_audit"))
    nested = _arm_mapping(audit, canonical, actual)
    return nested if nested else audit


def _extract_k_total(result: Mapping[str, Any], canonical: str, actual: str) -> int:
    arm_result = _arm_mapping(_mapping(result.get("arms")), canonical, actual)
    codebook = _codebook_arm_audit(result, canonical, actual)
    descriptor = _arm_descriptor(result, canonical)
    for source in (arm_result, codebook, descriptor):
        value = _deep_find(
            source,
            {"k_total", "K_total", "total_codebook_size", "codebook_size"},
        )
        if value is not None:
            return _integer(value, f"{canonical} K_total")
    if canonical == BASELINE_ARM:
        return 32
    expanded = _deep_find(
        descriptor,
        {
            "expansion_applied",
            "expanded",
            "selected_for_expansion",
            "gate_enabled",
        },
    )
    if expanded is not None:
        return 34 if _strict_bool(expanded, f"{canonical} expansion enabled") else 32
    raise RuntimeError(f"{canonical} does not report K_total or expansion_applied.")


def _operational_gate_enabled(
    result: Mapping[str, Any], canonical: str
) -> bool | None:
    """Return the arm's operational split decision, not an auxiliary audit."""

    if canonical == REGISTERED_ARM:
        descriptor = _mapping(result.get("registered_gate_fit"))
    elif canonical == ADAPTIVE_ARM:
        descriptor = _mapping(result.get("adaptive_selection"))
    else:
        return False
    value = _deep_find(
        descriptor,
        {"gate_enabled", "expansion_applied", "selected_for_expansion"},
    )
    return None if value is None else _strict_bool(value, f"{canonical} gate_enabled")


def _extract_parent_token(
    result: Mapping[str, Any], canonical: str, actual: str, k_total: int
) -> int | None:
    if canonical == BASELINE_ARM:
        return None
    sources = (
        _arm_descriptor(result, canonical),
        _codebook_arm_audit(result, canonical, actual),
        _arm_mapping(_mapping(result.get("arms")), canonical, actual),
    )
    value = None
    for source in sources:
        value = _deep_find(
            source,
            {
                "expanded_parent_token",
                "selected_parent_token",
                "parent_token",
                "coarse_token",
                "selected_token",
            },
        )
        if value is not None:
            break
    parent = _optional_int(value, f"{canonical} expanded parent token")
    if int(k_total) > 32 and parent is None:
        raise RuntimeError(f"{canonical} expanded to K={k_total} without a parent token.")
    if parent is not None and not 0 <= parent < 32:
        raise RuntimeError(f"{canonical} parent token {parent} lies outside frozen K32.")
    return parent


def _support_and_stability_summary(
    result: Mapping[str, Any], canonical: str
) -> dict[str, Any]:
    descriptor = _arm_descriptor(result, canonical)
    support = _mapping(_deep_find(descriptor, {"support_audit"}))
    loo = _mapping(
        _deep_find(
            descriptor,
            {"leave_one_out_audit", "leave_one_candidate_out_audit", "stability_audit"},
        )
    )

    def find(names: set[str], *sources: Mapping[str, Any]) -> Any:
        for source in sources:
            value = _deep_find(source, names)
            if value is not None:
                return value
        return None

    fit_count = find({"fit_trial_count", "candidate_support", "selected_support"}, support, descriptor)
    enabled_fraction = find({"enabled_fraction", "enable_fraction"}, loo)
    route_agreement = find(
        {"heldout_route_agreement", "route_agreement", "held_out_route_agreement"}, loo
    )
    coassignment = find(
        {"coassignment_ari", "co_assignment_ari", "median_coassignment_ari"}, loo
    )
    child_trials = find({"child_trial_counts"}, descriptor)
    required_child_trials = find({"required_child_trial_count"}, descriptor)
    child_subjects = find({"child_subject_counts"}, descriptor)
    return {
        "fit_trial_count": "" if fit_count is None else _integer(fit_count, "fit_trial_count"),
        "residual_child_counts": _json_cell(
            find({"residual_child_counts", "child_counts"}, support)
        ),
        "residual_required_per_child": (
            ""
            if find({"residual_required_per_child", "minimum_child_support"}, support)
            is None
            else _integer(
                find({"residual_required_per_child", "minimum_child_support"}, support),
                "residual_required_per_child",
            )
        ),
        "residual_child_subject_counts": _json_cell(
            find({"residual_child_subject_counts"}, support)
        ),
        "gravity_child_counts": _json_cell(find({"gravity_child_counts"}, support)),
        "gravity_required_per_child": (
            ""
            if find({"gravity_required_per_child"}, support) is None
            else _integer(
                find({"gravity_required_per_child"}, support),
                "gravity_required_per_child",
            )
        ),
        "gravity_child_subject_counts": _json_cell(
            find({"gravity_child_subject_counts"}, support)
        ),
        "physical_consistency": _json_cell(find({"physical_consistency"}, support)),
        "loo_status": str(find({"status"}, loo) or ""),
        "loo_passed": _json_cell(find({"passed"}, loo)),
        "loo_replicate_count": (
            ""
            if find({"replicate_count", "leave_one_out_count"}, loo) is None
            else _integer(
                find({"replicate_count", "leave_one_out_count"}, loo),
                "leave-one-out replicate_count",
            )
        ),
        "loo_enabled_fraction": (
            "" if enabled_fraction is None else _finite_float(enabled_fraction, "enabled_fraction")
        ),
        "loo_route_agreement": (
            "" if route_agreement is None else _finite_float(route_agreement, "route_agreement")
        ),
        "loo_coassignment_ari": (
            "" if coassignment is None else _finite_float(coassignment, "coassignment_ari")
        ),
        "adaptive_child_trial_counts": _json_cell(child_trials),
        "adaptive_required_child_trial_count": (
            ""
            if required_child_trials is None
            else _integer(required_child_trials, "required_child_trial_count")
        ),
        "adaptive_child_subject_counts": _json_cell(child_subjects),
        "adaptive_silhouette_subject_balanced": (
            ""
            if find({"silhouette_subject_balanced"}, descriptor) is None
            else _finite_float(
                find({"silhouette_subject_balanced"}, descriptor),
                "silhouette_subject_balanced",
            )
        ),
        "adaptive_loo_distortion_reduction": (
            ""
            if find({"loo_distortion_reduction"}, descriptor) is None
            else _finite_float(
                find({"loo_distortion_reduction"}, descriptor),
                "loo_distortion_reduction",
            )
        ),
        "adaptive_loo_stability_subject_balanced": (
            ""
            if find({"loo_stability_subject_balanced"}, descriptor) is None
            else _finite_float(
                find({"loo_stability_subject_balanced"}, descriptor),
                "loo_stability_subject_balanced",
            )
        ),
        "adaptive_loso_stability_subject_balanced": (
            ""
            if find({"loso_stability_subject_balanced"}, descriptor) is None
            else _finite_float(
                find({"loso_stability_subject_balanced"}, descriptor),
                "loso_stability_subject_balanced",
            )
        ),
        "adaptive_child_subject_nmi": (
            ""
            if find({"child_subject_nmi"}, descriptor) is None
            else _finite_float(
                find({"child_subject_nmi"}, descriptor), "child_subject_nmi"
            )
        ),
        "support_audit": _json_cell(support),
        "leave_one_out_audit": _json_cell(loo),
    }


def _validate_label_firewall(result: Mapping[str, Any], directory: Path) -> dict:
    firewall = _mapping(result.get("label_firewall"))
    if not firewall:
        raise RuntimeError(f"{directory} lacks label_firewall.")
    unsafe_flags = {
        "gate_fit_uses_activity_labels_or_names",
        "adaptive_selection_uses_activity_labels_or_names",
        "codebook_fit_uses_activity_labels_or_names",
        "trial_kmeans_fit_uses_activity_labels",
        "test_trials_used_for_fit",
        "test_occurrences_used_for_fit",
        "future_trials_used_for_fit",
        "fit_uses_activity_labels_or_names",
        "fit_uses_activity_labels",
        "routing_uses_activity_labels",
        "uses_activity_labels_or_names",
        "uses_test_occurrences",
    }
    unsafe_reports = _deep_key_values(result, unsafe_flags)
    for path, value in unsafe_reports:
        context = ".".join(path)
        if _strict_bool(value, context):
            raise RuntimeError(f"{directory} reports unsafe fitting: {context}=true.")
    train_only_flags = {
        "all_v2_thresholds_are_train_only",
        "support_and_physical_thresholds_use_train_only",
        "uses_only_train_candidates",
    }
    train_only_reports = _deep_key_values(result, train_only_flags)
    for path, value in train_only_reports:
        context = ".".join(path)
        if not _strict_bool(value, context):
            raise RuntimeError(f"{directory} reports non-train-only fitting: {context}=false.")
    future_count = _deep_find(
        firewall, {"future_feature_trial_count", "future_trial_count"}
    )
    if future_count is not None and _integer(future_count, "future trial count") != 0:
        raise RuntimeError(f"{directory} includes future-session feature trials.")
    joined_after = _deep_find(
        firewall,
        {
            "labels_joined_only_after_predictions_frozen",
            "labels_joined_after_predictions_frozen",
            "metadata_aware_repository_created_after_predictions",
        },
    )
    if joined_after is not None and not _strict_bool(joined_after, "label join audit"):
        raise RuntimeError(f"{directory} joined labels before predictions were frozen.")
    return {
        "unsafe_fit_flags_checked": sorted(unsafe_flags),
        "unsafe_fit_flag_report_count": len(unsafe_reports),
        "train_only_flag_report_count": len(train_only_reports),
        "future_feature_trial_count": 0 if future_count is None else int(future_count),
        "postfreeze_label_join_reported": joined_after is not None,
    }


def _validate_manifest(
    result: Mapping[str, Any], trial_ids: np.ndarray, directory: Path
) -> dict:
    manifest = _mapping(result.get("session_manifest_audit"))
    if not manifest:
        raise RuntimeError(f"{directory} lacks session_manifest_audit.")
    train_ids_value = _first_present(
        manifest,
        (("cumulative_train_trial_ids",), ("cumulative_online_train_trial_ids",)),
    )
    test_ids_value = _first_present(
        manifest,
        (("session_2_test_trial_ids",), ("test_trial_ids",)),
    )
    if train_ids_value is None or test_ids_value is None:
        raise RuntimeError(f"{directory} has an incomplete session manifest audit.")
    train_ids = [int(value) for value in train_ids_value]
    test_ids = [int(value) for value in test_ids_value]
    if not train_ids or not test_ids:
        raise RuntimeError(f"{directory} session manifest contains an empty split.")
    if len(train_ids) != len(set(train_ids)) or len(test_ids) != len(set(test_ids)):
        raise RuntimeError(f"{directory} session manifest repeats trial ids.")
    overlap = set(train_ids) & set(test_ids)
    if overlap:
        raise RuntimeError(f"{directory} train/test trial leakage: {sorted(overlap)}")
    if not np.array_equal(trial_ids, np.asarray(test_ids, dtype=np.int64)):
        raise RuntimeError(f"{directory} prediction order differs from test manifest.")
    future_count = manifest.get("future_feature_trial_count")
    if future_count is not None and int(future_count) != 0:
        raise RuntimeError(f"{directory} includes future-session features.")
    return {
        "manifest_reported": True,
        "cumulative_train_trial_count": len(train_ids),
        "test_trial_count": len(test_ids),
        "train_test_overlap": 0,
    }


def _validate_codebook_invariance(result: Mapping[str, Any], directory: Path) -> dict:
    audit = _mapping(result.get("codebook_audit"))
    if not audit:
        raise RuntimeError(f"{directory} lacks codebook_audit.")
    unsafe_boolean_names = {
        "old_centers_changed",
        "parent_centers_changed",
        "old_token_ids_reindexed",
        "parent_token_ids_reindexed",
    }
    safe_boolean_names = {
        "old_centers_unchanged",
        "parent_centers_unchanged",
        "old_token_ids_unchanged",
        "append_only",
        "all_invariants_passed",
    }
    for path, value in _deep_key_values(audit, unsafe_boolean_names):
        context = ".".join(("codebook_audit", *path))
        if _strict_bool(value, context):
            raise RuntimeError(f"{directory} violates frozen K32: {context}=true.")
    for path, value in _deep_key_values(audit, safe_boolean_names):
        context = ".".join(("codebook_audit", *path))
        if not _strict_bool(value, context):
            raise RuntimeError(f"{directory} violates frozen K32: {context}=false.")
    base_k = _deep_find(audit, {"base_k", "frozen_parent_count", "parent_k"})
    if base_k is not None and _integer(base_k, "codebook base K") != 32:
        raise RuntimeError(f"{directory} codebook base K is not 32.")
    return {
        "base_k": 32 if base_k is None else int(base_k),
        "append_only_reported": _deep_find(audit, {"append_only"}) is not None,
        "parent_invariance_reported": any(
            _deep_find(audit, {field}) is not None
            for field in safe_boolean_names | unsafe_boolean_names
        ),
    }


def _fit_call_count(result: Mapping[str, Any], directory: Path) -> int:
    audit = _mapping(result.get("readout_audit"))
    if not audit:
        raise RuntimeError(f"{directory} lacks readout_audit.")
    value = _first_present(
        audit,
        (
            ("coarse_fit_call_count",),
            ("shared_coarse_fit_call_count",),
            ("global_fit_call_count",),
            ("shared_coarse_readout", "coarse_fit_call_count"),
            ("shared_coarse_readout", "fit_call_count"),
        ),
    )
    if value is None:
        raise RuntimeError(f"{directory} readout_audit lacks the shared fit call count.")
    count = _integer(value, "coarse readout fit call count")
    if count != 1:
        raise RuntimeError(
            f"{directory} must fit the shared coarse readout exactly once, got {count}."
        )
    return count


def _metrics(
    truth: np.ndarray,
    raw: np.ndarray,
    aligned: np.ndarray,
    old_class_count: int,
) -> dict[str, float]:
    if truth.shape != raw.shape or truth.shape != aligned.shape or truth.ndim != 1:
        raise ValueError("Truth/raw/aligned predictions must be equal 1D arrays.")
    old_mask = truth < int(old_class_count)
    new_mask = ~old_mask
    if not np.any(old_mask) or not np.any(new_mask):
        raise RuntimeError("Both old and new trials are required for H-score.")
    correct = aligned == truth
    old_accuracy = float(np.mean(correct[old_mask]))
    new_accuracy = float(np.mean(correct[new_mask]))
    denominator = old_accuracy + new_accuracy
    return {
        "all_accuracy": float(np.mean(correct)),
        "old_accuracy": old_accuracy,
        "new_accuracy": new_accuracy,
        "h_score": float(2.0 * old_accuracy * new_accuracy / denominator)
        if denominator > 0.0
        else 0.0,
        "macro_f1": float(
            f1_score(
                truth,
                aligned,
                labels=sorted(set(truth.tolist())),
                average="macro",
                zero_division=0,
            )
        ),
        "ari": float(adjusted_rand_score(truth, raw)),
        "nmi": float(normalized_mutual_info_score(truth, raw, average_method="arithmetic")),
    }


def _validate_reported_metrics(
    result: Mapping[str, Any],
    canonical: str,
    actual: str,
    computed: Mapping[str, float],
    raw: np.ndarray,
    aligned: np.ndarray,
    context: str,
) -> None:
    arm_result = _arm_mapping(_mapping(result.get("arms")), canonical, actual)
    reported = _mapping(arm_result.get("global_metrics"))
    for field in ("all_accuracy", "old_accuracy", "new_accuracy", "macro_f1"):
        if field in reported and not np.isclose(
            _finite_float(reported[field], f"{context}/{field}"),
            computed[field],
            atol=1e-12,
        ):
            raise RuntimeError(f"{context}/{field} cannot be reproduced from the CSV.")
    for field, aliases in {
        "ari": ("ari", "adjusted_rand_index"),
        "nmi": ("nmi", "normalized_mutual_information"),
        "h_score": ("h_score", "old_new_h_score"),
    }.items():
        for alias in aliases:
            if alias in reported and not np.isclose(
                _finite_float(reported[alias], f"{context}/{alias}"),
                computed[field],
                atol=1e-12,
            ):
                raise RuntimeError(f"{context}/{alias} cannot be reproduced from the CSV.")
    for field, observed in (
        ("raw_predictions", raw),
        ("raw_cluster_predictions", raw),
        ("aligned_predictions", aligned),
    ):
        if field in reported:
            stored = np.asarray(reported[field], dtype=np.int64)
            if not np.array_equal(stored, observed):
                raise RuntimeError(f"{context}/{field} disagrees with the CSV.")


def _reported_fallback_mismatch(
    result: Mapping[str, Any], canonical: str, actual: str
) -> int | None:
    audit = _mapping(result.get("readout_audit"))
    arm_audit = _arm_mapping(audit, canonical, actual)
    value = _deep_find(
        arm_audit if arm_audit else audit,
        {"fallback_raw_mismatch_count"},
    )
    return None if value is None else _integer(value, "fallback_raw_mismatch_count")


def load_audited_run(result_path: Path, old_class_count: int = 6) -> AuditedRun:
    result_path = Path(result_path).resolve()
    directory = result_path.parent
    result = _read_json(result_path)
    actual_arm_names = _resolve_arm_names(result)

    # All fitting/provenance checks happen before truth columns are consumed.
    fit_call_count = _fit_call_count(result, directory)
    firewall_audit = _validate_label_firewall(result, directory)
    codebook_invariance = _validate_codebook_invariance(result, directory)

    prediction_path = _find_prediction_path(directory)
    rows = _read_csv(prediction_path)
    _, trial_values = _column_values(
        rows, ("trial_global_id", "trial_id"), "trial ids"
    )
    trial_ids = np.asarray(
        [_integer(value, "trial id") for value in trial_values], dtype=np.int64
    )
    if len(set(trial_ids.tolist())) != len(trial_ids):
        raise RuntimeError(f"{directory} repeats prediction trial ids.")
    manifest_audit = _validate_manifest(result, trial_ids, directory)

    raw = {
        canonical: _prediction_columns(rows, canonical, actual_arm_names[canonical], "raw")
        for canonical in ARMS
    }
    aligned = {
        canonical: _prediction_columns(
            rows, canonical, actual_arm_names[canonical], "aligned"
        )
        for canonical in ARMS
    }
    k_total = {
        canonical: _extract_k_total(result, canonical, actual_arm_names[canonical])
        for canonical in ARMS
    }
    if k_total[BASELINE_ARM] != 32:
        raise RuntimeError(f"{directory} baseline K must be 32, got {k_total}.")
    for canonical in CHALLENGERS:
        if k_total[canonical] not in (32, 34):
            raise RuntimeError(
                f"{directory}/{canonical} K must be 32 or 34, "
                f"got {k_total[canonical]}."
            )
        gate_enabled = _operational_gate_enabled(result, canonical)
        if gate_enabled is not None:
            expected_k = 34 if gate_enabled else 32
            if k_total[canonical] != expected_k:
                raise RuntimeError(
                    f"{directory}/{canonical} reports gate_enabled={gate_enabled} "
                    f"but K_total={k_total[canonical]} (expected {expected_k})."
                )

    routed = {BASELINE_ARM: np.zeros(len(trial_ids), dtype=bool)}
    for canonical in CHALLENGERS:
        routed[canonical] = _route_mask(
            result,
            rows,
            trial_ids,
            canonical,
            actual_arm_names[canonical],
            k_total[canonical],
        )

    interventions: dict[str, dict[str, Any]] = {}
    for canonical in CHALLENGERS:
        raw_changed = raw[canonical] != raw[BASELINE_ARM]
        aligned_changed = aligned[canonical] != aligned[BASELINE_ARM]
        fallback = ~routed[canonical]
        mismatch_count = int(np.sum(raw_changed & fallback))
        reported_mismatch = _reported_fallback_mismatch(
            result, canonical, actual_arm_names[canonical]
        )
        if reported_mismatch is not None and reported_mismatch != mismatch_count:
            raise RuntimeError(
                f"{directory}/{canonical} reported fallback mismatch "
                f"{reported_mismatch}, reproduced {mismatch_count}."
            )
        if mismatch_count != 0:
            offending = trial_ids[raw_changed & fallback].astype(int).tolist()
            raise RuntimeError(
                f"{directory}/{canonical} changed {mismatch_count} fallback raw "
                f"predictions; offending trial ids={offending}."
            )
        if k_total[canonical] == 32 and (
            np.any(routed[canonical]) or np.any(raw_changed)
        ):
            raise RuntimeError(
                f"{directory}/{canonical} reports K=32 but still routes or changes trials."
            )
        interventions[canonical] = {
            "routed_trial_count": int(np.sum(routed[canonical])),
            "fallback_trial_count": int(np.sum(fallback)),
            "raw_changed_count": int(np.sum(raw_changed)),
            "routed_without_raw_change_count": int(np.sum(routed[canonical] & ~raw_changed)),
            "fallback_raw_mismatch_count": mismatch_count,
            "aligned_changed_count": int(np.sum(aligned_changed)),
            "alignment_only_change_count": int(np.sum(~raw_changed & aligned_changed)),
            "fallback_alignment_only_change_count": int(
                np.sum(fallback & ~raw_changed & aligned_changed)
            ),
            "raw_and_aligned_change_count": int(np.sum(raw_changed & aligned_changed)),
            "routed_trial_ids": trial_ids[routed[canonical]].astype(int).tolist(),
            "raw_changed_trial_ids": trial_ids[raw_changed].astype(int).tolist(),
            "alignment_only_trial_ids": trial_ids[~raw_changed & aligned_changed]
            .astype(int)
            .tolist(),
        }

    # Labels are joined only after the label-free audit and raw-vector checks above.
    _, subject_values = _column_values(rows, ("subject_id",), "subject ids")
    _, truth_values = _column_values(
        rows, ("activity_label_0based", "activity_label", "label"), "ground truth"
    )
    subjects = np.asarray(
        [_integer(value, "subject id") for value in subject_values], dtype=np.int64
    )
    truth = np.asarray(
        [_integer(value, "activity label") for value in truth_values], dtype=np.int64
    )
    if len(subjects) != len(trial_ids) or len(truth) != len(trial_ids):
        raise RuntimeError(f"{directory} metadata length differs from prediction length.")

    computed_metrics: dict[str, dict[str, float]] = {}
    for canonical in ARMS:
        values = _metrics(
            truth,
            raw[canonical],
            aligned[canonical],
            old_class_count=int(old_class_count),
        )
        _validate_reported_metrics(
            result,
            canonical,
            actual_arm_names[canonical],
            values,
            raw[canonical],
            aligned[canonical],
            f"{directory}/{canonical}",
        )
        computed_metrics[canonical] = values

    arm_audits: dict[str, dict[str, Any]] = {}
    for canonical in ARMS:
        parent = _extract_parent_token(
            result,
            canonical,
            actual_arm_names[canonical],
            k_total[canonical],
        )
        arm_audits[canonical] = {
            "k_total": int(k_total[canonical]),
            "expanded_parent_token": parent,
            **_support_and_stability_summary(result, canonical),
        }

    baseline_correct = aligned[BASELINE_ARM] == truth
    for canonical in CHALLENGERS:
        challenger_correct = aligned[canonical] == truth
        route = routed[canonical]
        fallback = ~route
        delta = challenger_correct.astype(np.int64) - baseline_correct.astype(np.int64)
        interventions[canonical].update(
            {
                "routed_correct_delta_count": int(np.sum(delta[route])),
                "routed_baseline_accuracy": (
                    float(np.mean(baseline_correct[route])) if np.any(route) else None
                ),
                "routed_challenger_accuracy": (
                    float(np.mean(challenger_correct[route])) if np.any(route) else None
                ),
                "routed_accuracy_delta": (
                    float(np.mean(delta[route])) if np.any(route) else 0.0
                ),
                "fallback_correct_delta_count": int(np.sum(delta[fallback])),
                "fallback_accuracy_delta": (
                    float(np.mean(delta[fallback])) if np.any(fallback) else 0.0
                ),
            }
        )

    fold = _identity_integer(result, directory, "fold")
    seed = _identity_integer(result, directory, "seed")
    profile = _identity_text(result, directory, "profile")
    segmentation = _identity_text(result, directory, "segmentation")
    session = _session_number(result)

    result.setdefault("analyzer_input_audit", {})
    result["analyzer_input_audit"] = {
        "coarse_readout_fit_call_count": fit_call_count,
        "label_firewall": firewall_audit,
        "session_manifest": manifest_audit,
        "codebook_invariance": codebook_invariance,
        "prediction_csv": str(prediction_path),
        "prediction_csv_sha256": sha256_file(prediction_path),
    }
    return AuditedRun(
        directory=directory,
        result_path=result_path,
        result=result,
        fold=fold,
        seed=seed,
        profile=profile,
        segmentation=segmentation,
        session=session,
        trial_ids=trial_ids,
        subjects=subjects,
        truth=truth,
        actual_arm_names=actual_arm_names,
        raw_predictions=raw,
        aligned_predictions=aligned,
        routed_masks=routed,
        metrics=computed_metrics,
        arm_audits=arm_audits,
        intervention_audits=interventions,
    )


def _parse_integer_set(value: str) -> set[int]:
    normalized = str(value or "").replace(",", " ").strip()
    if not normalized:
        return set()
    values = [_integer(item, "expected grid value") for item in normalized.split()]
    if len(values) != len(set(values)):
        raise ValueError("Expected grid contains duplicate values.")
    return set(values)


def _grid_method_identity(run: AuditedRun) -> dict[str, Any]:
    """Return the fold/seed-independent v2 method identity for one run."""

    arguments = _mapping(run.result.get("arguments"))
    if not arguments:
        raise RuntimeError(f"{run.directory} lacks runner arguments.")
    dynamic_arguments = {"run_dir", "output_dir", "fold", "seed"}
    method_arguments = {
        str(key): jsonable(value)
        for key, value in arguments.items()
        if str(key) not in dynamic_arguments
    }

    raw_fingerprint = _mapping(
        _nested(run.result, "input_audit", "implementation_fingerprint")
    )
    required_components = {
        "runner",
        "adaptive_codebook",
        "frozen_readout",
        "hierarchical_gate_v2",
        "historical_hierarchical_gate_v1",
    }
    if set(raw_fingerprint) != required_components:
        raise RuntimeError(
            f"{run.directory} implementation fingerprint components differ: "
            f"observed={sorted(raw_fingerprint)}, expected={sorted(required_components)}."
        )
    implementation_hashes: dict[str, str] = {}
    for name in sorted(required_components):
        digest = str(_mapping(raw_fingerprint[name]).get("sha256", "")).lower()
        if re.fullmatch(r"[0-9a-f]{64}", digest) is None:
            raise RuntimeError(
                f"{run.directory} has an invalid implementation SHA256 for {name}."
            )
        implementation_hashes[name] = digest

    protocol = dict(_mapping(run.result.get("protocol")))
    if protocol.get("name") != "motion_primitive_adaptive_codebook_frozen_readout_v2":
        raise RuntimeError(f"{run.directory} has an unexpected v2 protocol name.")
    for field in ("scope", "outer_fold", "run_seed"):
        protocol.pop(field, None)
    identity = {
        "method_arguments": method_arguments,
        "implementation_hashes": implementation_hashes,
        "protocol": jsonable(protocol),
    }
    canonical = json.dumps(identity, sort_keys=True, separators=(",", ":"))
    identity["identity_sha256"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return identity


def _validate_grid(
    runs: Sequence[AuditedRun], expected_folds: set[int], expected_seeds: set[int]
) -> dict:
    grouped: dict[str, list[AuditedRun]] = defaultdict(list)
    for run in runs:
        grouped[run.config_id].append(run)
    details = {}
    for config_id, members in sorted(grouped.items()):
        observed_pairs = [(run.fold, run.seed) for run in members]
        if len(observed_pairs) != len(set(observed_pairs)):
            raise RuntimeError(f"Duplicate fold/seed in {config_id}: {observed_pairs}")
        observed_folds = set(run.fold for run in members)
        observed_seeds = set(run.seed for run in members)
        if expected_folds and observed_folds != expected_folds:
            raise RuntimeError(
                f"{config_id} fold grid mismatch: observed={sorted(observed_folds)}, "
                f"expected={sorted(expected_folds)}."
            )
        if expected_seeds and observed_seeds != expected_seeds:
            raise RuntimeError(
                f"{config_id} seed grid mismatch: observed={sorted(observed_seeds)}, "
                f"expected={sorted(expected_seeds)}."
            )
        if expected_folds and expected_seeds:
            expected_pairs = {
                (fold, seed) for fold in expected_folds for seed in expected_seeds
            }
            if set(observed_pairs) != expected_pairs:
                missing = sorted(expected_pairs - set(observed_pairs))
                extra = sorted(set(observed_pairs) - expected_pairs)
                raise RuntimeError(
                    f"{config_id} incomplete Cartesian grid: missing={missing}, extra={extra}."
                )
        identities = [_grid_method_identity(run) for run in members]
        identity_hashes = {item["identity_sha256"] for item in identities}
        if len(identity_hashes) != 1:
            by_run = {
                f"fold={run.fold},seed={run.seed}": identity["identity_sha256"]
                for run, identity in zip(members, identities)
            }
            raise RuntimeError(
                f"{config_id} mixes runner arguments, protocol definitions, or "
                f"implementation versions across its grid: {by_run}."
            )
        details[config_id] = {
            "run_count": len(members),
            "folds": sorted(observed_folds),
            "seeds": sorted(observed_seeds),
            "fold_seed_pairs": [list(value) for value in sorted(observed_pairs)],
            "method_identity_sha256": identities[0]["identity_sha256"],
            "method_arguments": identities[0]["method_arguments"],
            "implementation_hashes": identities[0]["implementation_hashes"],
        }
    return {
        "configuration_count": len(grouped),
        "expected_folds": sorted(expected_folds),
        "expected_seeds": sorted(expected_seeds),
        "configurations": details,
    }


def _base_identity(run: AuditedRun) -> dict[str, Any]:
    return {
        "config_id": run.config_id,
        "profile": run.profile,
        "segmentation": run.segmentation,
        "session": run.session,
        "fold": run.fold,
        "seed": run.seed,
    }


def _run_tables(
    runs: Sequence[AuditedRun], old_class_count: int
) -> tuple[list[dict], list[dict], list[dict]]:
    metric_rows: list[dict] = []
    effect_rows: list[dict] = []
    audit_rows: list[dict] = []
    for run in sorted(runs, key=lambda item: (item.config_id, item.fold, item.seed)):
        identity = _base_identity(run)
        for arm in ARMS:
            metric_rows.append(
                {
                    **identity,
                    "arm": arm,
                    **run.metrics[arm],
                    **run.arm_audits[arm],
                    "trial_count": len(run.trial_ids),
                    "old_trial_count": int(np.sum(run.truth < int(old_class_count))),
                    "new_trial_count": int(np.sum(run.truth >= int(old_class_count))),
                    "result_directory": str(run.directory),
                }
            )
        for challenger in CHALLENGERS:
            effect = {
                **identity,
                "challenger": challenger,
                "reference": BASELINE_ARM,
            }
            for metric in QUALITY_METRICS:
                effect[f"delta_{metric}"] = (
                    run.metrics[challenger][metric] - run.metrics[BASELINE_ARM][metric]
                )
            effect.update(run.intervention_audits[challenger])
            effect["k_total"] = run.arm_audits[challenger]["k_total"]
            effect["expanded_parent_token"] = run.arm_audits[challenger][
                "expanded_parent_token"
            ]
            effect_rows.append(effect)
            audit_rows.append(
                {
                    **identity,
                    "challenger": challenger,
                    **run.intervention_audits[challenger],
                    **run.arm_audits[challenger],
                    "input_result": str(run.result_path),
                    "input_result_sha256": sha256_file(run.result_path),
                }
            )
    return metric_rows, effect_rows, audit_rows


def _paired_encoder_runs(
    runs: Sequence[AuditedRun],
) -> tuple[list[tuple[AuditedRun, AuditedRun]], dict[str, Any]]:
    """Pair A0/A3 at the same fold, seed, segmentation and session.

    Encoder interaction is a paired difference-in-differences estimand.  It is
    invalid if the test population or fitting hyperparameters differ between
    A0 and A3, so every mismatch is fail-closed rather than silently dropped.
    """

    allowed_profiles = {"A0", "A3"}
    groups: dict[tuple[str, int, int, int], dict[str, AuditedRun]] = defaultdict(dict)
    for run in runs:
        if run.profile not in allowed_profiles:
            raise RuntimeError(
                "Encoder interaction requires exactly the A0 and A3 profiles; "
                f"observed unsupported profile {run.profile!r}."
            )
        key = (run.segmentation, int(run.session), int(run.fold), int(run.seed))
        if run.profile in groups[key]:
            raise RuntimeError(
                f"Duplicate {run.profile} encoder run for paired identity {key}."
            )
        groups[key][run.profile] = run

    ignored_argument_fields = {"run_dir", "output_dir"}
    pairs: list[tuple[AuditedRun, AuditedRun]] = []
    audit_pairs: list[dict[str, Any]] = []
    for key, by_profile in sorted(groups.items()):
        if set(by_profile) != allowed_profiles:
            missing = sorted(allowed_profiles - set(by_profile))
            raise RuntimeError(
                f"Encoder interaction pair {key} is incomplete; missing {missing}."
            )
        a0 = by_profile["A0"]
        a3 = by_profile["A3"]
        if not np.array_equal(a0.trial_ids, a3.trial_ids):
            raise RuntimeError(f"A0/A3 test trial ids differ for paired identity {key}.")
        if not np.array_equal(a0.subjects, a3.subjects):
            raise RuntimeError(f"A0/A3 test subjects differ for paired identity {key}.")
        if not np.array_equal(a0.truth, a3.truth):
            raise RuntimeError(f"A0/A3 ground truth differs for paired identity {key}.")
        a0_manifest = _mapping(a0.result.get("session_manifest_audit"))
        a3_manifest = _mapping(a3.result.get("session_manifest_audit"))
        a0_train_ids = [
            int(value) for value in a0_manifest.get("cumulative_train_trial_ids", [])
        ]
        a3_train_ids = [
            int(value) for value in a3_manifest.get("cumulative_train_trial_ids", [])
        ]
        if not a0_train_ids or a0_train_ids != a3_train_ids:
            raise RuntimeError(
                f"A0/A3 cumulative online-train manifests differ for paired "
                f"identity {key}."
            )

        def fit_arguments(run: AuditedRun) -> dict[str, Any]:
            return {
                str(name): jsonable(value)
                for name, value in _mapping(run.result.get("arguments")).items()
                if str(name) not in ignored_argument_fields
            }

        a0_arguments = fit_arguments(a0)
        a3_arguments = fit_arguments(a3)
        if json.dumps(a0_arguments, sort_keys=True) != json.dumps(
            a3_arguments, sort_keys=True
        ):
            raise RuntimeError(
                f"A0/A3 fitting arguments differ for paired identity {key}: "
                f"A0={a0_arguments}, A3={a3_arguments}."
            )
        a0_method_identity = _grid_method_identity(a0)
        a3_method_identity = _grid_method_identity(a3)
        if (
            a0_method_identity["identity_sha256"]
            != a3_method_identity["identity_sha256"]
        ):
            raise RuntimeError(
                f"A0/A3 method implementation or protocol differs for paired "
                f"identity {key}: A0={a0_method_identity['identity_sha256']}, "
                f"A3={a3_method_identity['identity_sha256']}."
            )
        pairs.append((a0, a3))
        audit_pairs.append(
            {
                "segmentation": key[0],
                "session": key[1],
                "fold": key[2],
                "seed": key[3],
                "trial_count": len(a0.trial_ids),
                "test_trial_ids_equal": True,
                "test_subjects_equal": True,
                "ground_truth_equal": True,
                "cumulative_online_train_trial_ids_equal": True,
                "fitting_arguments_equal": True,
                "method_implementation_and_protocol_equal": True,
                "method_identity_sha256": a0_method_identity[
                    "identity_sha256"
                ],
                "a0_result": str(a0.result_path),
                "a3_result": str(a3.result_path),
            }
        )
    if not pairs:
        raise RuntimeError("No complete A0/A3 encoder interaction pair was found.")
    return pairs, {
        "required_profiles": ["A0", "A3"],
        "pairing_fields": ["segmentation", "session", "fold", "seed"],
        "pair_count": len(pairs),
        "all_pairs_complete": True,
        "all_test_manifests_equal_within_pair": True,
        "all_cumulative_online_train_manifests_equal_within_pair": True,
        "all_fitting_arguments_equal_within_pair": True,
        "all_method_implementations_and_protocols_equal_within_pair": True,
        "pairs": audit_pairs,
    }


def _run_encoder_interaction_rows(
    pairs: Sequence[tuple[AuditedRun, AuditedRun]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for a0, a3 in pairs:
        pair_id = (
            f"profiles=A3_vs_A0|segmentation={a0.segmentation}|"
            f"session={a0.session}"
        )
        for challenger in CHALLENGERS:
            a0_intervention = a0.intervention_audits[challenger]
            a3_intervention = a3.intervention_audits[challenger]
            if int(a0_intervention["fallback_raw_mismatch_count"]) != 0 or int(
                a3_intervention["fallback_raw_mismatch_count"]
            ) != 0:
                raise RuntimeError(
                    f"{pair_id}/{challenger} contains a fallback raw mismatch."
                )
            row: dict[str, Any] = {
                "interaction_config_id": pair_id,
                "segmentation": a0.segmentation,
                "session": int(a0.session),
                "fold": int(a0.fold),
                "seed": int(a0.seed),
                "challenger": challenger,
                "reference": BASELINE_ARM,
                "interaction_formula": (
                    "(A3_challenger-A3_F0)-(A0_challenger-A0_F0)"
                ),
            }
            for metric in QUALITY_METRICS:
                a0_delta = (
                    a0.metrics[challenger][metric] - a0.metrics[BASELINE_ARM][metric]
                )
                a3_delta = (
                    a3.metrics[challenger][metric] - a3.metrics[BASELINE_ARM][metric]
                )
                row[f"a0_delta_{metric}"] = float(a0_delta)
                row[f"a3_delta_{metric}"] = float(a3_delta)
                row[f"interaction_delta_{metric}"] = float(a3_delta - a0_delta)
            for field in (
                "routed_trial_count",
                "raw_changed_count",
                "alignment_only_change_count",
                "routed_correct_delta_count",
                "fallback_correct_delta_count",
            ):
                a0_value = int(a0_intervention[field])
                a3_value = int(a3_intervention[field])
                row[f"a0_{field}"] = a0_value
                row[f"a3_{field}"] = a3_value
                row[f"a3_minus_a0_{field}"] = a3_value - a0_value
            row.update(
                {
                    "a0_k_total": int(a0.arm_audits[challenger]["k_total"]),
                    "a3_k_total": int(a3.arm_audits[challenger]["k_total"]),
                    "a0_expanded_parent_token": a0.arm_audits[challenger][
                        "expanded_parent_token"
                    ],
                    "a3_expanded_parent_token": a3.arm_audits[challenger][
                        "expanded_parent_token"
                    ],
                    "fallback_raw_mismatch_count": 0,
                    "a0_result_directory": str(a0.directory),
                    "a3_result_directory": str(a3.directory),
                }
            )
            rows.append(row)
    return rows


def _mean(values: Sequence[Any]) -> float:
    array = np.asarray(values, dtype=np.float64)
    if len(array) == 0 or not np.all(np.isfinite(array)):
        raise RuntimeError("Fold aggregation received empty or non-finite values.")
    return float(np.mean(array))


def _fold_metric_rows(run_rows: Sequence[Mapping[str, Any]]) -> list[dict]:
    groups: dict[tuple, list[Mapping[str, Any]]] = defaultdict(list)
    for row in run_rows:
        key = (
            row["config_id"],
            row["profile"],
            row["segmentation"],
            int(row["session"]),
            int(row["fold"]),
            row["arm"],
        )
        groups[key].append(row)
    output = []
    for key, members in sorted(groups.items()):
        config_id, profile, segmentation, session, fold, arm = key
        output.append(
            {
                "config_id": config_id,
                "profile": profile,
                "segmentation": segmentation,
                "session": session,
                "fold": fold,
                "arm": arm,
                "seed_count": len(members),
                "seeds": _json_cell(sorted(int(item["seed"]) for item in members)),
                **{
                    metric: _mean([item[metric] for item in members])
                    for metric in QUALITY_METRICS
                },
                "mean_k_total": _mean([item["k_total"] for item in members]),
                "expanded_parent_tokens": _json_cell(
                    sorted(
                        {
                            int(item["expanded_parent_token"])
                            for item in members
                            if item["expanded_parent_token"] not in (None, "")
                        }
                    )
                ),
            }
        )
    return output


def _fold_effect_rows(run_rows: Sequence[Mapping[str, Any]]) -> list[dict]:
    groups: dict[tuple, list[Mapping[str, Any]]] = defaultdict(list)
    for row in run_rows:
        key = (
            row["config_id"],
            row["profile"],
            row["segmentation"],
            int(row["session"]),
            int(row["fold"]),
            row["challenger"],
        )
        groups[key].append(row)
    output = []
    for key, members in sorted(groups.items()):
        config_id, profile, segmentation, session, fold, challenger = key
        output.append(
            {
                "config_id": config_id,
                "profile": profile,
                "segmentation": segmentation,
                "session": session,
                "fold": fold,
                "challenger": challenger,
                "reference": BASELINE_ARM,
                "seed_count": len(members),
                "seeds": _json_cell(sorted(int(item["seed"]) for item in members)),
                **{
                    f"delta_{metric}": _mean(
                        [item[f"delta_{metric}"] for item in members]
                    )
                    for metric in QUALITY_METRICS
                },
                "mean_k_total": _mean([item["k_total"] for item in members]),
                "mean_routed_trial_count": _mean(
                    [item["routed_trial_count"] for item in members]
                ),
                "mean_raw_changed_count": _mean(
                    [item["raw_changed_count"] for item in members]
                ),
                "mean_alignment_only_change_count": _mean(
                    [item["alignment_only_change_count"] for item in members]
                ),
                "mean_routed_correct_delta_count": _mean(
                    [item["routed_correct_delta_count"] for item in members]
                ),
                "mean_fallback_correct_delta_count": _mean(
                    [item["fallback_correct_delta_count"] for item in members]
                ),
                "fallback_raw_mismatch_count": int(
                    sum(int(item["fallback_raw_mismatch_count"]) for item in members)
                ),
                "expanded_parent_tokens": _json_cell(
                    sorted(
                        {
                            int(item["expanded_parent_token"])
                            for item in members
                            if item["expanded_parent_token"] not in (None, "")
                        }
                    )
                ),
            }
        )
    return output


def _fold_encoder_interaction_rows(
    run_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    groups: dict[tuple, list[Mapping[str, Any]]] = defaultdict(list)
    for row in run_rows:
        key = (
            row["interaction_config_id"],
            row["segmentation"],
            int(row["session"]),
            int(row["fold"]),
            row["challenger"],
        )
        groups[key].append(row)
    output: list[dict[str, Any]] = []
    for key, members in sorted(groups.items()):
        pair_id, segmentation, session, fold, challenger = key
        if any(int(item["fallback_raw_mismatch_count"]) != 0 for item in members):
            raise RuntimeError(f"{pair_id}/{challenger} contains a fallback mismatch.")
        row: dict[str, Any] = {
            "interaction_config_id": pair_id,
            "segmentation": segmentation,
            "session": session,
            "fold": fold,
            "challenger": challenger,
            "reference": BASELINE_ARM,
            "interaction_formula": (
                "(A3_challenger-A3_F0)-(A0_challenger-A0_F0)"
            ),
            "seed_count": len(members),
            "seeds": _json_cell(sorted(int(item["seed"]) for item in members)),
        }
        for metric in QUALITY_METRICS:
            for prefix in ("a0_delta", "a3_delta", "interaction_delta"):
                field = f"{prefix}_{metric}"
                row[field] = _mean([item[field] for item in members])
        for field in (
            "routed_trial_count",
            "raw_changed_count",
            "alignment_only_change_count",
            "routed_correct_delta_count",
            "fallback_correct_delta_count",
        ):
            for prefix in ("a0", "a3", "a3_minus_a0"):
                name = f"{prefix}_{field}"
                row[f"mean_{name}"] = _mean([item[name] for item in members])
        row.update(
            {
                "mean_a0_k_total": _mean(
                    [item["a0_k_total"] for item in members]
                ),
                "mean_a3_k_total": _mean(
                    [item["a3_k_total"] for item in members]
                ),
                "a0_expanded_parent_tokens": _json_cell(
                    sorted(
                        {
                            int(item["a0_expanded_parent_token"])
                            for item in members
                            if item["a0_expanded_parent_token"] not in (None, "")
                        }
                    )
                ),
                "a3_expanded_parent_tokens": _json_cell(
                    sorted(
                        {
                            int(item["a3_expanded_parent_token"])
                            for item in members
                            if item["a3_expanded_parent_token"] not in (None, "")
                        }
                    )
                ),
                "fallback_raw_mismatch_count": 0,
            }
        )
        output.append(row)
    return output


def _bootstrap_fold_mean(
    values: Sequence[float], replicates: int, seed: int
) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or len(array) == 0 or not np.all(np.isfinite(array)):
        raise ValueError("Fold values must be a finite non-empty vector.")
    if int(replicates) < 1:
        raise ValueError("bootstrap_replicates must be positive.")
    mean = float(np.mean(array))
    if len(array) == 1:
        lower = upper = mean
        bootstrap_mean = mean
    else:
        rng = np.random.default_rng(int(seed))
        positions = rng.integers(0, len(array), size=(int(replicates), len(array)))
        bootstrap_values = np.mean(array[positions], axis=1)
        lower, upper = np.quantile(bootstrap_values, [0.025, 0.975])
        bootstrap_mean = float(np.mean(bootstrap_values))
    return {
        "fold_mean": mean,
        "fold_sample_std": float(np.std(array, ddof=1)) if len(array) > 1 else 0.0,
        "fold_bootstrap_mean": bootstrap_mean,
        "fold_bootstrap_ci95_lower": float(lower),
        "fold_bootstrap_ci95_upper": float(upper),
        "positive_fold_count": int(np.sum(array > 0.0)),
        "zero_fold_count": int(np.sum(array == 0.0)),
        "negative_fold_count": int(np.sum(array < 0.0)),
        "fold_count": int(len(array)),
        "independent_unit": "subject_fold_after_seed_average",
    }


def _aggregate_metrics(
    fold_rows: Sequence[Mapping[str, Any]], replicates: int, seed: int
) -> list[dict]:
    groups: dict[tuple, list[Mapping[str, Any]]] = defaultdict(list)
    for row in fold_rows:
        key = (
            row["config_id"],
            row["profile"],
            row["segmentation"],
            int(row["session"]),
            row["arm"],
        )
        groups[key].append(row)
    output = []
    counter = 0
    for key, members in sorted(groups.items()):
        config_id, profile, segmentation, session, arm = key
        for metric in QUALITY_METRICS:
            summary = _bootstrap_fold_mean(
                [float(item[metric]) for item in members],
                replicates=replicates,
                seed=int(seed) + counter,
            )
            counter += 1
            output.append(
                {
                    "config_id": config_id,
                    "profile": profile,
                    "segmentation": segmentation,
                    "session": session,
                    "arm": arm,
                    "metric": metric,
                    **summary,
                }
            )
    return output


def _aggregate_effects(
    fold_rows: Sequence[Mapping[str, Any]], replicates: int, seed: int
) -> list[dict]:
    groups: dict[tuple, list[Mapping[str, Any]]] = defaultdict(list)
    for row in fold_rows:
        key = (
            row["config_id"],
            row["profile"],
            row["segmentation"],
            int(row["session"]),
            row["challenger"],
        )
        groups[key].append(row)
    output = []
    counter = 10000
    for key, members in sorted(groups.items()):
        config_id, profile, segmentation, session, challenger = key
        if any(int(item["fallback_raw_mismatch_count"]) != 0 for item in members):
            raise RuntimeError(f"{config_id}/{challenger} contains a fallback mismatch.")
        for metric in QUALITY_METRICS:
            summary = _bootstrap_fold_mean(
                [float(item[f"delta_{metric}"]) for item in members],
                replicates=replicates,
                seed=int(seed) + counter,
            )
            counter += 1
            output.append(
                {
                    "config_id": config_id,
                    "profile": profile,
                    "segmentation": segmentation,
                    "session": session,
                    "challenger": challenger,
                    "reference": BASELINE_ARM,
                    "metric": metric,
                    **summary,
                    "mean_k_total": _mean(
                        [float(item["mean_k_total"]) for item in members]
                    ),
                    "mean_routed_trial_count": _mean(
                        [float(item["mean_routed_trial_count"]) for item in members]
                    ),
                    "mean_alignment_only_change_count": _mean(
                        [
                            float(item["mean_alignment_only_change_count"])
                            for item in members
                        ]
                    ),
                    "mean_routed_correct_delta_count": _mean(
                        [
                            float(item["mean_routed_correct_delta_count"])
                            for item in members
                        ]
                    ),
                    "mean_fallback_correct_delta_count": _mean(
                        [
                            float(item["mean_fallback_correct_delta_count"])
                            for item in members
                        ]
                    ),
                    "fallback_raw_mismatch_count": 0,
                }
            )
    return output


def _aggregate_encoder_interactions(
    fold_rows: Sequence[Mapping[str, Any]], replicates: int, seed: int
) -> list[dict[str, Any]]:
    groups: dict[tuple, list[Mapping[str, Any]]] = defaultdict(list)
    for row in fold_rows:
        key = (
            row["interaction_config_id"],
            row["segmentation"],
            int(row["session"]),
            row["challenger"],
        )
        groups[key].append(row)
    output: list[dict[str, Any]] = []
    counter = 20000
    for key, members in sorted(groups.items()):
        pair_id, segmentation, session, challenger = key
        if any(int(item["fallback_raw_mismatch_count"]) != 0 for item in members):
            raise RuntimeError(f"{pair_id}/{challenger} contains a fallback mismatch.")
        for metric in QUALITY_METRICS:
            summary = _bootstrap_fold_mean(
                [float(item[f"interaction_delta_{metric}"]) for item in members],
                replicates=replicates,
                seed=int(seed) + counter,
            )
            counter += 1
            output.append(
                {
                    "interaction_config_id": pair_id,
                    "segmentation": segmentation,
                    "session": session,
                    "challenger": challenger,
                    "reference": BASELINE_ARM,
                    "metric": metric,
                    "interaction_formula": (
                        "(A3_challenger-A3_F0)-(A0_challenger-A0_F0)"
                    ),
                    "a0_effect_fold_mean": _mean(
                        [float(item[f"a0_delta_{metric}"]) for item in members]
                    ),
                    "a3_effect_fold_mean": _mean(
                        [float(item[f"a3_delta_{metric}"]) for item in members]
                    ),
                    **summary,
                    "mean_a0_k_total": _mean(
                        [float(item["mean_a0_k_total"]) for item in members]
                    ),
                    "mean_a3_k_total": _mean(
                        [float(item["mean_a3_k_total"]) for item in members]
                    ),
                    "mean_a0_routed_trial_count": _mean(
                        [
                            float(item["mean_a0_routed_trial_count"])
                            for item in members
                        ]
                    ),
                    "mean_a3_routed_trial_count": _mean(
                        [
                            float(item["mean_a3_routed_trial_count"])
                            for item in members
                        ]
                    ),
                    "fallback_raw_mismatch_count": 0,
                }
            )
    return output


def _save_effect_plot(path: Path, aggregate_effects: Sequence[Mapping[str, Any]]) -> None:
    labels = sorted(
        {
            (str(row["config_id"]), str(row["challenger"]))
            for row in aggregate_effects
        }
    )
    indexed = {
        (str(row["config_id"]), str(row["challenger"]), str(row["metric"])): row
        for row in aggregate_effects
    }
    fig, axes = plt.subplots(2, 4, figsize=(24, max(8, 0.55 * len(labels) + 4)))
    axes_flat = axes.ravel()
    colors = ("#3366cc", "#dc3912", "#109618", "#ff9900", "#990099", "#0099c6", "#dd4477")
    for metric_index, metric in enumerate(QUALITY_METRICS):
        axis = axes_flat[metric_index]
        rows = [indexed[(config, challenger, metric)] for config, challenger in labels]
        values = np.asarray([row["fold_mean"] for row in rows]) * 100.0
        lower = np.asarray([row["fold_bootstrap_ci95_lower"] for row in rows]) * 100.0
        upper = np.asarray([row["fold_bootstrap_ci95_upper"] for row in rows]) * 100.0
        positions = np.arange(len(rows))
        axis.hlines(positions, lower, upper, color=colors[metric_index], linewidth=1.6)
        axis.scatter(values, positions, color=colors[metric_index], s=30, zorder=3)
        axis.axvline(0.0, color="black", linewidth=1, linestyle="--")
        axis.set_title(metric.replace("_", " "))
        axis.set_xlabel("challenger - frozen K32 (x100)")
        axis.grid(axis="x", alpha=0.25)
        axis.set_yticks(positions)
        if metric_index % 4 == 0:
            axis.set_yticklabels(
                [f"{config}\n{challenger}" for config, challenger in labels], fontsize=8
            )
        else:
            axis.set_yticklabels([])
        axis.invert_yaxis()
    axes_flat[-1].axis("off")
    fig.suptitle(
        "Frozen-readout adaptive-codebook effects\n"
        "Seeds averaged within fold; intervals bootstrap subject-fold means"
    )
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def analyze_runs(
    runs: Sequence[AuditedRun],
    *,
    old_class_count: int,
    bootstrap_replicates: int,
    bootstrap_seed: int,
    expected_folds: set[int] | None = None,
    expected_seeds: set[int] | None = None,
) -> dict:
    if not runs:
        raise ValueError("At least one audited run is required.")
    grid_audit = _validate_grid(
        runs,
        expected_folds=expected_folds or set(),
        expected_seeds=expected_seeds or set(),
    )
    encoder_pairs, encoder_pairing_audit = _paired_encoder_runs(runs)
    run_metrics, run_effects, intervention_audits = _run_tables(
        runs, old_class_count=int(old_class_count)
    )
    run_encoder_interactions = _run_encoder_interaction_rows(encoder_pairs)
    fold_metrics = _fold_metric_rows(run_metrics)
    fold_effects = _fold_effect_rows(run_effects)
    fold_encoder_interactions = _fold_encoder_interaction_rows(
        run_encoder_interactions
    )
    aggregate_metrics = _aggregate_metrics(
        fold_metrics, replicates=int(bootstrap_replicates), seed=int(bootstrap_seed)
    )
    aggregate_effects = _aggregate_effects(
        fold_effects,
        replicates=int(bootstrap_replicates),
        seed=int(bootstrap_seed),
    )
    aggregate_encoder_interactions = _aggregate_encoder_interactions(
        fold_encoder_interactions,
        replicates=int(bootstrap_replicates),
        seed=int(bootstrap_seed),
    )
    return {
        "protocol": {
            "name": "frozen_readout_adaptive_codebook_batch_analysis_v2",
            "baseline_arm": BASELINE_ARM,
            "registered_arm": REGISTERED_ARM,
            "adaptive_arm": ADAPTIVE_ARM,
            "quality_metrics": list(QUALITY_METRICS),
            "old_class_count": int(old_class_count),
            "bootstrap_replicates": int(bootstrap_replicates),
            "bootstrap_seed": int(bootstrap_seed),
            "independent_statistical_unit": "subject_fold_after_within_fold_seed_average",
            "alignment_caveat": (
                "Each arm's aligned prediction may change when its global Hungarian "
                "mapping changes. Raw fallback equality is the local-intervention "
                "invariant; alignment-only changes are reported separately."
            ),
            "multiplicity_adjustment": "none",
            "encoder_interaction_formula": (
                "(A3_challenger-A3_F0)-(A0_challenger-A0_F0)"
            ),
            "encoder_interaction_pairing": (
                "same segmentation, session, held-out-subject fold, seed, "
                "test manifest, and fitting arguments"
            ),
        },
        "grid_audit": grid_audit,
        "encoder_pairing_audit": encoder_pairing_audit,
        "run_metrics": run_metrics,
        "run_effects": run_effects,
        "run_encoder_interactions": run_encoder_interactions,
        "intervention_audits": intervention_audits,
        "fold_metrics_after_seed_average": fold_metrics,
        "fold_effects_after_seed_average": fold_effects,
        "fold_encoder_interactions_after_seed_average": fold_encoder_interactions,
        "aggregate_metrics_across_folds": aggregate_metrics,
        "aggregate_effects_across_folds": aggregate_effects,
        "aggregate_encoder_interactions_across_folds": (
            aggregate_encoder_interactions
        ),
    }


def _require_new_output_dir(path: Path) -> Path:
    resolved = Path(path).resolve()
    if resolved.exists():
        raise FileExistsError(f"Output directory must not already exist: {resolved}")
    return resolved


def run(args: argparse.Namespace) -> dict:
    input_values = [Path(value) for value in args.input_root] + [
        Path(value) for value in args.run_dir
    ]
    if not input_values:
        raise ValueError("Provide at least one --input-root or --run-dir.")
    result_paths = discover_result_paths(input_values)
    runs = [
        load_audited_run(path, old_class_count=int(args.old_class_count))
        for path in result_paths
    ]
    result = analyze_runs(
        runs,
        old_class_count=int(args.old_class_count),
        bootstrap_replicates=int(args.bootstrap_replicates),
        bootstrap_seed=int(args.seed),
        expected_folds=_parse_integer_set(args.expected_folds),
        expected_seeds=_parse_integer_set(args.expected_seeds),
    )
    output_dir = _require_new_output_dir(Path(args.output_dir))
    output_dir.mkdir(parents=True)

    _write_csv(output_dir / "adaptive_codebook_v2_run_metrics.csv", result["run_metrics"])
    _write_csv(output_dir / "adaptive_codebook_v2_run_effects.csv", result["run_effects"])
    _write_csv(
        output_dir / "adaptive_codebook_v2_run_encoder_interactions.csv",
        result["run_encoder_interactions"],
    )
    _write_csv(
        output_dir / "adaptive_codebook_v2_intervention_audits.csv",
        result["intervention_audits"],
    )
    _write_csv(
        output_dir / "adaptive_codebook_v2_fold_metrics.csv",
        result["fold_metrics_after_seed_average"],
    )
    _write_csv(
        output_dir / "adaptive_codebook_v2_fold_effects.csv",
        result["fold_effects_after_seed_average"],
    )
    _write_csv(
        output_dir / "adaptive_codebook_v2_fold_encoder_interactions.csv",
        result["fold_encoder_interactions_after_seed_average"],
    )
    _write_csv(
        output_dir / "adaptive_codebook_v2_aggregate_metrics.csv",
        result["aggregate_metrics_across_folds"],
    )
    _write_csv(
        output_dir / "adaptive_codebook_v2_aggregate_effects.csv",
        result["aggregate_effects_across_folds"],
    )
    _write_csv(
        output_dir / "adaptive_codebook_v2_aggregate_encoder_interactions.csv",
        result["aggregate_encoder_interactions_across_folds"],
    )
    _save_effect_plot(
        output_dir / "adaptive_codebook_v2_fold_effects.png",
        result["aggregate_effects_across_folds"],
    )

    result["arguments"] = {
        "input_roots": [str(Path(value).resolve()) for value in input_values],
        "output_dir": str(output_dir),
        "old_class_count": int(args.old_class_count),
        "bootstrap_replicates": int(args.bootstrap_replicates),
        "seed": int(args.seed),
        "expected_folds": sorted(_parse_integer_set(args.expected_folds)),
        "expected_seeds": sorted(_parse_integer_set(args.expected_seeds)),
    }
    result["input_results"] = [
        {
            "path": str(path),
            "sha256": sha256_file(path),
        }
        for path in result_paths
    ]
    result["implementation_fingerprint"] = {
        "analyzer_path": str(Path(__file__).resolve()),
        "analyzer_sha256": sha256_file(Path(__file__).resolve()),
    }
    result["generated_files"] = [
        "adaptive_codebook_v2_analysis.json",
        "adaptive_codebook_v2_run_metrics.csv",
        "adaptive_codebook_v2_run_effects.csv",
        "adaptive_codebook_v2_run_encoder_interactions.csv",
        "adaptive_codebook_v2_intervention_audits.csv",
        "adaptive_codebook_v2_fold_metrics.csv",
        "adaptive_codebook_v2_fold_effects.csv",
        "adaptive_codebook_v2_fold_encoder_interactions.csv",
        "adaptive_codebook_v2_aggregate_metrics.csv",
        "adaptive_codebook_v2_aggregate_effects.csv",
        "adaptive_codebook_v2_aggregate_encoder_interactions.csv",
        "adaptive_codebook_v2_fold_effects.png",
    ]
    (output_dir / "adaptive_codebook_v2_analysis.json").write_text(
        json.dumps(jsonable(result), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Audit and aggregate frozen-readout K32/K34 adaptive-codebook v2 runs."
        )
    )
    parser.add_argument(
        "--input-root",
        action="append",
        default=[],
        help="Result root to search recursively; may be supplied more than once.",
    )
    parser.add_argument(
        "--run-dir",
        action="append",
        default=[],
        help="One explicit run directory; may be supplied more than once.",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--old-class-count", type=int, default=6)
    parser.add_argument("--bootstrap-replicates", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260904)
    parser.add_argument(
        "--expected-folds",
        default="",
        help='Optional strict fold grid, e.g. "1,2,3,4,5,6,7".',
    )
    parser.add_argument(
        "--expected-seeds",
        default="",
        help='Optional strict seed grid, e.g. "0,5,50,500".',
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if int(args.old_class_count) < 1:
        raise ValueError("old_class_count must be positive.")
    if int(args.bootstrap_replicates) < 1:
        raise ValueError("bootstrap_replicates must be positive.")
    result = run(args)
    concise = {
        "output_dir": result["arguments"]["output_dir"],
        "grid_audit": result["grid_audit"],
        "aggregate_effects_across_folds": result[
            "aggregate_effects_across_folds"
        ],
        "aggregate_encoder_interactions_across_folds": result[
            "aggregate_encoder_interactions_across_folds"
        ],
    }
    print(json.dumps(jsonable(concise), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
