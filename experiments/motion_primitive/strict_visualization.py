"""Post-hoc visual audit for the frozen A2/E0/state CGCD route.

This module is deliberately separated from the Online learner.  It reads only
durable run artifacts after raw predictions have been frozen, verifies their
hashes, and never imports or calls the registry update API.  Ground-truth
activity identity is read from the scorer-side prediction CSV files only for
annotation and confusion-matrix labels.

Matplotlib is imported lazily and forced onto the non-interactive ``Agg``
backend, so importing the experiment package does not require a GUI.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence

import numpy as np


OFFLINE_SCHEMA = "hhr_frozen_a2_e0_state_offline_v1"
ONLINE_SCHEMA = "hhr_frozen_a2_e0_state_online_v1"
METRICS_SCHEMA = "hhr_strict_cgcd_metrics_v1"
PROFILE = "frozen_a2_e0_state_k32"
LAYER_ORDER = (
    "direct_registry",
    "old_fixed_novel_hungarian",
    "global_hungarian_upper_bound",
)
LAYER_TITLES = {
    "direct_registry": "Direct registry IDs",
    "old_fixed_novel_hungarian": "Old-fixed novel Hungarian",
    "global_hungarian_upper_bound": "Global Hungarian upper bound",
}


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"JSON artifact {path} must contain an object.")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise TypeError(f"{path}:{line_number} is not a JSON object.")
            rows.append(value)
    return rows


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _write_json(path: Path, payload: Any) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    temporary.replace(path)


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    rows = list(rows)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if str(key) not in fields:
                fields.append(str(key))
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows([{key: row.get(key, "") for key in fields} for row in rows])
    temporary.replace(path)


def _load_plotting():
    try:
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
        from matplotlib.colors import ListedColormap
    except Exception as error:  # pragma: no cover - exact import error is environment-specific
        raise RuntimeError(
            "Strict experiment visualisation requires the optional dependency "
            "matplotlib>=3.7. Install requirements-har.txt; no GUI backend is required."
        ) from error
    return plt, ListedColormap


def _primitive_colormap(codebook_size: int):
    plt, listed_colormap = _load_plotting()
    return listed_colormap(plt.get_cmap("turbo")(np.linspace(0.0, 1.0, codebook_size)))


def _require_file(path: Path) -> Path:
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def _forbidden_learner_labels(row: Mapping[str, Any]) -> set[str]:
    forbidden = {
        "label", "labels", "activity", "activity_id", "activity_label",
        "activity_name", "target", "targets", "truth", "ground_truth",
    }
    return {str(key) for key in row if str(key).lower() in forbidden}


@dataclass(frozen=True)
class ScoredSession:
    session: int
    metrics: Mapping[str, Any]
    predictions: tuple[Mapping[str, Any], ...]
    evaluation_trajectories: tuple[Mapping[str, Any], ...]
    discovery: Mapping[str, Any]
    registry: Mapping[str, Any]


@dataclass(frozen=True)
class VisualizationArtifacts:
    offline_dir: Path
    online_dir: Path
    source_npz: Path
    manifest: Mapping[str, Any]
    offline_usage: Mapping[str, Any]
    sessions: tuple[ScoredSession, ...]
    all_trajectories: tuple[Mapping[str, Any], ...]
    codebook_size: int
    protected_hashes: Mapping[str, str]


def _protected_input_files(offline_dir: Path, online_dir: Path, session_count: int) -> list[Path]:
    candidates = [
        offline_dir / "manifest.json",
        offline_dir / "complete.json",
        offline_dir / "codebook_usage.json",
        offline_dir / "e0_codebook.npz",
        offline_dir / "e0_codebook.json",
        offline_dir / "state_descriptor_transform.npz",
        offline_dir / "state_descriptor_transform.json",
        offline_dir / "old_registry.json",
        online_dir / "online_summary.json",
        online_dir / "complete.json",
        online_dir / "trajectories_label_free.jsonl",
        online_dir / "online_runs.csv",
    ]
    for session in range(1, session_count + 1):
        candidates.extend((
            online_dir / f"raw_predictions_session_{session}.npz",
            online_dir / f"metrics_session_{session}.json",
            online_dir / f"predictions_session_{session}.csv",
            online_dir / f"discovery_session_{session}.json",
            online_dir / f"registry_session_{session}.json",
        ))
    return [path for path in candidates if path.is_file()]


def _hash_map(paths: Iterable[Path]) -> dict[str, str]:
    return {str(path.resolve()): sha256_file(path) for path in sorted(paths, key=str)}


def _validate_layer(layer: Mapping[str, Any], class_count: int, sample_count: int) -> None:
    matrix = np.asarray(layer.get("confusion_matrix_with_unknown_column"), dtype=np.int64)
    if matrix.shape != (class_count, class_count + 1):
        raise RuntimeError(
            f"Scoring layer confusion shape {matrix.shape} differs from "
            f"({class_count},{class_count + 1})."
        )
    aligned = np.asarray(layer.get("aligned_predictions"), dtype=np.int64)
    if aligned.shape != (sample_count,):
        raise RuntimeError("Scoring layer prediction count differs from the raw artifact.")
    if int(matrix.sum()) != sample_count:
        raise RuntimeError("Scoring layer confusion count differs from the raw artifact.")


def load_visualization_artifacts(
    offline_run_dir: str | Path,
    online_run_dir: str | Path,
) -> VisualizationArtifacts:
    """Load and fail-closed validate durable post-hoc plotting inputs."""

    offline_dir = Path(offline_run_dir).expanduser().resolve()
    online_dir = Path(online_run_dir).expanduser().resolve()
    manifest_path = _require_file(offline_dir / "manifest.json")
    offline_complete = _read_json(_require_file(offline_dir / "complete.json"))
    manifest = _read_json(manifest_path)
    if manifest.get("schema") != OFFLINE_SCHEMA or manifest.get("profile") != PROFILE:
        raise RuntimeError("Offline artifacts are not the frozen A2/E0/state/K32 route.")
    if offline_complete.get("schema") != OFFLINE_SCHEMA or offline_complete.get("complete") is not True:
        raise RuntimeError("Offline run is incomplete or has another schema.")
    if offline_complete.get("manifest_sha256") != sha256_file(manifest_path):
        raise RuntimeError("Offline manifest SHA256 does not verify.")
    online_complete_path = _require_file(online_dir / "complete.json")
    online_complete = _read_json(online_complete_path)
    summary_path = _require_file(online_dir / "online_summary.json")
    summary = _read_json(summary_path)
    if summary.get("schema") != ONLINE_SCHEMA or online_complete.get("schema") != ONLINE_SCHEMA:
        raise RuntimeError("Online artifacts have an unexpected schema.")
    if summary.get("profile") != PROFILE or online_complete.get("profile") != PROFILE:
        raise RuntimeError("Online artifacts use another experiment profile.")
    if online_complete.get("complete") is not True:
        raise RuntimeError("Online run is incomplete.")
    if online_complete.get("online_summary_sha256") != sha256_file(summary_path):
        raise RuntimeError("Online summary SHA256 does not verify.")
    if summary.get("online_activity_labels_used_by_learner") is not False:
        raise RuntimeError("Online run does not prove learner-side label isolation.")
    if summary.get("test_labels_used_only_for_scoring") is not True:
        raise RuntimeError("Online run does not declare scorer-only test labels.")
    session_count = int(summary.get("session_count", 0))
    if session_count != 3:
        raise RuntimeError(f"Strict visual audit requires all three sessions, got {session_count}.")
    trajectory_path = _require_file(online_dir / "trajectories_label_free.jsonl")
    trajectories = _read_jsonl(trajectory_path)
    if not trajectories:
        raise RuntimeError("Online trajectory artifact is empty.")
    for row in trajectories:
        leaked = _forbidden_learner_labels(row)
        if leaked:
            raise RuntimeError(
                "Learner-facing trajectory artifact contains scorer identity fields: "
                f"{sorted(leaked)}."
            )

    old_registry_path = _require_file(
        offline_dir / str(manifest.get("old_registry", {}).get("path", "old_registry.json"))
    )
    if manifest.get("old_registry", {}).get("artifact_sha256") != sha256_file(old_registry_path):
        raise RuntimeError("Offline old-registry artifact SHA256 does not verify.")
    old_registry = _read_json(old_registry_path)
    previous_registry_hash = str(old_registry.get("state_sha256", ""))
    old_anchor_hash = str(old_registry.get("old_anchor_sha256", ""))
    representation_hash = str(old_registry.get("representation_sha256", ""))
    if previous_registry_hash != manifest.get("old_registry", {}).get("state_sha256"):
        raise RuntimeError("Offline old-registry state differs from its manifest.")
    if old_anchor_hash != manifest.get("old_registry", {}).get("old_anchor_sha256"):
        raise RuntimeError("Offline old-registry semantic anchor differs from its manifest.")
    if representation_hash != manifest.get("representation_sha256"):
        raise RuntimeError("Offline old registry is bound to another representation.")

    sessions: list[ScoredSession] = []
    for session in range(1, session_count + 1):
        metrics_path = _require_file(online_dir / f"metrics_session_{session}.json")
        metrics = _read_json(metrics_path)
        if metrics.get("schema") != METRICS_SCHEMA or int(metrics.get("session", -1)) != session:
            raise RuntimeError(f"Session {session} metrics schema/identity is invalid.")
        if metrics.get("raw_predictions_frozen_before_truth_join") is not True:
            raise RuntimeError("Metrics do not prove prediction persistence before truth access.")
        if metrics.get("scoring_only_no_registry_writeback") is not True:
            raise RuntimeError("Metrics do not prove scorer/registry separation.")
        if tuple(metrics.get("layers", {}).keys()) != LAYER_ORDER:
            # Dict insertion order is part of the persisted schema, but accept
            # an equivalent key set to remain robust to pretty-printers.
            if set(metrics.get("layers", {})) != set(LAYER_ORDER):
                raise RuntimeError(f"Session {session} lacks one of the three scoring layers.")
        raw_path = _require_file(online_dir / str(metrics.get("raw_predictions_path", "")))
        if metrics.get("raw_predictions_sha256") != sha256_file(raw_path):
            raise RuntimeError(f"Session {session} raw prediction SHA256 does not verify.")
        with np.load(raw_path, allow_pickle=False) as archive:
            raw_trial_ids = np.asarray(archive["trial_ids"], dtype=np.int64)
            raw_predictions = np.asarray(archive["raw_registry_predictions"], dtype=np.int64)
            stored_registry_hash = str(np.asarray(archive["registry_state_sha256"]).item())
            stored_representation_hash = str(np.asarray(archive["representation_sha256"]).item())
        if raw_trial_ids.ndim != 1 or raw_predictions.shape != raw_trial_ids.shape:
            raise RuntimeError("Raw prediction artifact has inconsistent vectors.")
        if stored_registry_hash != metrics.get("registry_state_sha256"):
            raise RuntimeError("Raw predictions are bound to another registry state.")
        if stored_representation_hash != metrics.get("representation_sha256"):
            raise RuntimeError("Raw predictions are bound to another representation.")
        class_count = int(metrics.get("expected_class_count", 0))
        for layer_name in LAYER_ORDER:
            _validate_layer(metrics["layers"][layer_name], class_count, len(raw_trial_ids))
        direct = np.asarray(metrics["layers"]["direct_registry"]["aligned_predictions"], dtype=np.int64)
        if not np.array_equal(direct, raw_predictions):
            raise RuntimeError("Direct scoring layer differs from frozen raw predictions.")

        prediction_rows = _read_csv(
            _require_file(online_dir / f"predictions_session_{session}.csv")
        )
        prediction_ids = np.asarray([int(row["trial_id"]) for row in prediction_rows], dtype=np.int64)
        prediction_raw = np.asarray(
            [int(row["raw_registry_prediction"]) for row in prediction_rows], dtype=np.int64
        )
        if not np.array_equal(prediction_ids, raw_trial_ids) or not np.array_equal(prediction_raw, raw_predictions):
            raise RuntimeError("Scorer CSV order/values differ from the frozen raw prediction artifact.")
        primary = np.asarray(
            metrics["layers"]["old_fixed_novel_hungarian"]["aligned_predictions"], dtype=np.int64
        )
        csv_primary = np.asarray(
            [int(row["primary_aligned_prediction"]) for row in prediction_rows], dtype=np.int64
        )
        if not np.array_equal(primary, csv_primary):
            raise RuntimeError("Primary aligned scorer CSV differs from metrics JSON.")
        evaluation = tuple(
            row for row in trajectories
            if int(row.get("session", -1)) == session and row.get("role") == "evaluation"
        )
        trajectory_ids = np.asarray([int(row["trial_id"]) for row in evaluation], dtype=np.int64)
        if not np.array_equal(trajectory_ids, raw_trial_ids):
            raise RuntimeError("Evaluation trajectory order differs from raw predictions.")
        for row, raw_prediction in zip(evaluation, raw_predictions.tolist()):
            if int(row.get("raw_registry_prediction", -999999)) != int(raw_prediction):
                raise RuntimeError("Evaluation trajectory embeds a different raw prediction.")
        discovery = _read_json(
            _require_file(online_dir / f"discovery_session_{session}.json")
        )
        registry = _read_json(
            _require_file(online_dir / f"registry_session_{session}.json")
        )
        if discovery.get("activity_labels_used") is not False:
            raise RuntimeError("Discovery artifact does not prove label-free registration.")
        if int(registry.get("session_completed", -1)) != session:
            raise RuntimeError(f"Registry session marker differs for session {session}.")
        if registry.get("previous_state_sha256") != previous_registry_hash:
            raise RuntimeError(f"Registry hash chain breaks before session {session}.")
        if registry.get("state_sha256") != metrics.get("registry_state_sha256"):
            raise RuntimeError(f"Registry artifact/metrics state differs in session {session}.")
        if discovery.get("state_sha256") != registry.get("state_sha256"):
            raise RuntimeError(f"Discovery update/registry state differs in session {session}.")
        if registry.get("old_anchor_sha256") != old_anchor_hash:
            raise RuntimeError("Frozen old semantic registry changed online.")
        if registry.get("representation_sha256") != representation_hash:
            raise RuntimeError("Frozen representation binding changed online.")
        previous_registry_hash = str(registry["state_sha256"])
        sessions.append(ScoredSession(
            session=session,
            metrics=metrics,
            predictions=tuple(prediction_rows),
            evaluation_trajectories=evaluation,
            discovery=discovery,
            registry=registry,
        ))

    source_npz = Path(str(manifest.get("npz_path", ""))).expanduser().resolve()
    _require_file(source_npz)
    if manifest.get("npz_sha256") != sha256_file(source_npz):
        raise RuntimeError("Source NPZ SHA256 differs from the offline manifest.")
    codebook_size = int(manifest.get("codebook", {}).get("primitive_num", 0))
    if codebook_size != 32:
        raise RuntimeError(f"Strict E0 visualisation expects K32, got K={codebook_size}.")
    offline_usage = _read_json(_require_file(offline_dir / "codebook_usage.json"))
    protected = _hash_map(_protected_input_files(offline_dir, online_dir, session_count))
    return VisualizationArtifacts(
        offline_dir=offline_dir,
        online_dir=online_dir,
        source_npz=source_npz,
        manifest=manifest,
        offline_usage=offline_usage,
        sessions=tuple(sessions),
        all_trajectories=tuple(trajectories),
        codebook_size=codebook_size,
        protected_hashes=protected,
    )


def _session_join(session: ScoredSession) -> list[dict[str, Any]]:
    joined: list[dict[str, Any]] = []
    layers = session.metrics["layers"]
    for index, (prediction, trajectory) in enumerate(
        zip(session.predictions, session.evaluation_trajectories)
    ):
        joined.append({
            **dict(trajectory),
            "activity_id": int(prediction["activity_label"]),
            "activity_name": str(prediction["activity_name"]),
            "direct_prediction": int(layers["direct_registry"]["aligned_predictions"][index]),
            "primary_prediction": int(layers["old_fixed_novel_hungarian"]["aligned_predictions"][index]),
            "global_prediction": int(layers["global_hungarian_upper_bound"]["aligned_predictions"][index]),
        })
    return joined


def _overlap_average(windows: np.ndarray, starts: np.ndarray) -> np.ndarray:
    windows = np.asarray(windows, dtype=np.float64)
    starts = np.asarray(starts, dtype=np.int64)
    if windows.ndim != 3 or windows.shape[1] != 6 or starts.shape != (len(windows),):
        raise ValueError("Raw signal reconstruction expects windows [L,6,T] and L starts.")
    length = int(starts[-1]) + int(windows.shape[2])
    signal = np.zeros((6, length), dtype=np.float64)
    count = np.zeros(length, dtype=np.float64)
    for window, start in zip(windows, starts):
        end = int(start) + int(windows.shape[2])
        signal[:, int(start):end] += window
        count[int(start):end] += 1.0
    if np.any(count <= 0):
        raise RuntimeError("Saved windows do not cover a contiguous original trial.")
    return signal / count[None, :]


def load_raw_trial_signals(
    npz_path: str | Path,
    trial_ids: Sequence[int],
) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """Reconstruct six raw channels without opening NPZ activity labels."""

    requested = {int(value) for value in trial_ids}
    with np.load(Path(npz_path), allow_pickle=False) as archive:
        required = ("windows", "trial_global_ids", "window_start_indices", "mean", "std")
        missing = [name for name in required if name not in archive]
        if missing:
            raise RuntimeError(f"Source NPZ lacks raw-plot fields {missing}.")
        stored = np.asarray(archive["windows"], dtype=np.float32)
        ids = np.asarray(archive["trial_global_ids"], dtype=np.int64)
        starts = np.asarray(archive["window_start_indices"], dtype=np.int64)
        mean = np.asarray(archive["mean"], dtype=np.float32).reshape(1, 6, 1)
        std = np.asarray(archive["std"], dtype=np.float32).reshape(1, 6, 1)
    if stored.ndim != 3 or stored.shape[1] != 6 or ids.shape != starts.shape or ids.shape != (len(stored),):
        raise RuntimeError("Source NPZ raw-window arrays have inconsistent shapes.")
    raw = stored * std + mean
    result: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for trial_id in sorted(requested):
        rows = np.flatnonzero(ids == trial_id)
        if not len(rows):
            raise KeyError(f"Source NPZ has no trial {trial_id}.")
        order = np.argsort(starts[rows], kind="stable")
        rows = rows[order]
        local_starts = starts[rows]
        if local_starts[0] != 0 or np.any(np.diff(local_starts) <= 0):
            raise RuntimeError(f"Trial {trial_id} windows are incomplete or unordered.")
        result[trial_id] = (_overlap_average(raw[rows], local_starts), local_starts.copy())
    return result


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_") or "trial"


def _prediction_text(value: int) -> str:
    return "Unknown" if int(value) < 0 else f"R{int(value)}"


def _latest_unique_trial_rows(artifacts: VisualizationArtifacts) -> list[dict[str, Any]]:
    selected: dict[int, dict[str, Any]] = {}
    for session in artifacts.sessions:
        for row in _session_join(session):
            selected[int(row["trial_id"])] = row
    # Incoming trials are label-free and may not occur in any evaluation split.
    for row in artifacts.all_trajectories:
        if row.get("role") != "incoming":
            continue
        trial_id = int(row["trial_id"])
        selected.setdefault(trial_id, dict(row))
    return sorted(
        selected.values(),
        key=lambda row: (
            999 if "activity_id" not in row else int(row["activity_id"]),
            int(row["subject_id"]), int(row["trial_id"]),
        ),
    )


def _plot_trial(
    row: Mapping[str, Any],
    signal: np.ndarray,
    output_path: Path,
    *,
    codebook_size: int,
    sample_rate_hz: float,
    dpi: int,
) -> None:
    plt, _ = _load_plotting()
    colour = _primitive_colormap(codebook_size)
    tokens = np.asarray(row["primitive_sequence"], dtype=np.int64)
    starts = np.asarray(row["ownership_start_samples"], dtype=np.int64)
    ends = np.asarray(row["ownership_end_samples_exclusive"], dtype=np.int64)
    if tokens.ndim != 1 or not len(tokens) or starts.shape != tokens.shape or ends.shape != tokens.shape:
        raise RuntimeError("Trajectory token and ownership arrays have inconsistent shapes.")
    if starts[0] != 0 or np.any(starts[1:] != ends[:-1]) or int(ends[-1]) != signal.shape[1]:
        raise RuntimeError("Trajectory ownership does not cover the reconstructed raw trial.")
    seconds = np.arange(signal.shape[1], dtype=np.float64) / float(sample_rate_hz)
    figure, axes = plt.subplots(
        7, 1, figsize=(15, 11), sharex=True,
        gridspec_kw={"height_ratios": [1, 1, 1, 1, 1, 1, 0.34]},
        constrained_layout=True,
    )
    channel_names = ("acc-x", "acc-y", "acc-z", "gyro-x", "gyro-y", "gyro-z")
    for channel, axis in enumerate(axes[:6]):
        axis.plot(seconds, signal[channel], color="#172554", linewidth=0.72)
        axis.set_ylabel(channel_names[channel], fontsize=8)
        for token, start, end in zip(tokens, starts, ends):
            axis.axvspan(
                float(start) / sample_rate_hz,
                float(end) / sample_rate_hz,
                color=colour(int(token)), alpha=0.10, linewidth=0,
            )
    token_samples = np.empty(signal.shape[1], dtype=np.int64)
    for token, start, end in zip(tokens, starts, ends):
        token_samples[int(start):int(end)] = int(token)
    axes[6].imshow(
        token_samples[None, :], aspect="auto", interpolation="nearest",
        cmap=colour, vmin=-0.5, vmax=codebook_size - 0.5,
        extent=(0.0, signal.shape[1] / sample_rate_hz, 0.0, 1.0),
    )
    axes[6].set_yticks([0.5], labels=["E0 token"], fontsize=8)
    axes[6].set_xlabel("Time (s)")
    used = sorted(set(tokens.astype(int).tolist()))
    activity = str(row.get("activity_name", "unscored incoming trial"))
    predictions = ""
    if "direct_prediction" in row:
        predictions = (
            f" | direct={_prediction_text(int(row['direct_prediction']))}"
            f", primary={_prediction_text(int(row['primary_prediction']))}"
            f", global={_prediction_text(int(row['global_prediction']))}"
        )
    figure.suptitle(
        f"{activity} | S{int(row['subject_id']):02d} trial {int(row['trial_id'])} | "
        f"{len(tokens)} primitives, {len(used)} used types {used}{predictions}",
        fontsize=10,
    )
    figure.savefig(output_path, dpi=int(dpi))
    plt.close(figure)


def plot_all_trial_trajectories(
    artifacts: VisualizationArtifacts,
    output_dir: Path,
    *,
    maximum_plots: int = 0,
    sample_rate_hz: float = 100.0,
    dpi: int = 170,
) -> tuple[list[str], list[dict[str, Any]]]:
    rows = _latest_unique_trial_rows(artifacts)
    if int(maximum_plots) < 0:
        raise ValueError("maximum_plots must be zero (all) or positive.")
    if int(maximum_plots) > 0:
        rows = rows[: int(maximum_plots)]
    output_dir.mkdir(parents=True, exist_ok=True)
    signals = load_raw_trial_signals(
        artifacts.source_npz, [int(row["trial_id"]) for row in rows]
    )
    paths: list[str] = []
    index_rows: list[dict[str, Any]] = []
    for row in rows:
        trial_id = int(row["trial_id"])
        signal, saved_starts = signals[trial_id]
        expected_starts = np.asarray(row["original_window_start_samples"], dtype=np.int64)
        if not np.array_equal(saved_starts, expected_starts):
            raise RuntimeError(f"Trial {trial_id} trajectory is bound to another window grid.")
        role = _safe_name(str(row.get("role", "unknown")))
        path = output_dir / f"trial_{trial_id:05d}_session_{int(row['session'])}_{role}.png"
        _plot_trial(
            row, signal, path,
            codebook_size=artifacts.codebook_size,
            sample_rate_hz=float(sample_rate_hz), dpi=int(dpi),
        )
        tokens = [int(value) for value in row["primitive_sequence"]]
        index_rows.append({
            "trial_id": trial_id,
            "subject_id": int(row["subject_id"]),
            "session": int(row["session"]),
            "role": str(row["role"]),
            "activity_label": row.get("activity_id", ""),
            "activity_name": row.get("activity_name", ""),
            "primitive_occurrence_count": len(tokens),
            "unique_primitive_type_count": len(set(tokens)),
            "used_primitive_ids": " ".join(map(str, sorted(set(tokens)))),
            "complete_primitive_sequence": " ".join(map(str, tokens)),
            "direct_prediction": row.get("direct_prediction", ""),
            "primary_prediction": row.get("primary_prediction", ""),
            "global_prediction": row.get("global_prediction", ""),
            "image": str(path.name),
        })
        paths.append(str(path))
    _write_csv(output_dir / "trial_trajectory_index.csv", index_rows)
    return paths, index_rows


def activity_primitive_matrix(
    rows: Sequence[Mapping[str, Any]],
    codebook_size: int,
) -> tuple[list[int], list[str], np.ndarray]:
    """Return trial-normalised then subject-equal activity token profiles."""

    if not rows or int(codebook_size) < 1:
        raise ValueError("Activity heatmap needs scored rows and a positive codebook size.")
    activities = sorted({int(row["activity_id"]) for row in rows})
    matrix = np.zeros((len(activities), int(codebook_size)), dtype=np.float64)
    labels: list[str] = []
    for row_index, activity_id in enumerate(activities):
        members = [row for row in rows if int(row["activity_id"]) == activity_id]
        labels.append(f"{activity_id}: {members[0]['activity_name']}")
        # A long trial or a subject with more retained trials must not dominate
        # an activity row.  First normalise each trial, then average trials
        # within subject, and finally average subjects with equal mass.
        subject_rows: list[np.ndarray] = []
        for subject_id in sorted({int(row["subject_id"]) for row in members}):
            trial_rows: list[np.ndarray] = []
            for row in members:
                if int(row["subject_id"]) != subject_id:
                    continue
                counts = np.bincount(
                    np.asarray(row["primitive_sequence"], dtype=np.int64),
                    minlength=int(codebook_size),
                ).astype(np.float64)
                if len(counts) != int(codebook_size):
                    raise ValueError("A trajectory contains a token outside the codebook.")
                trial_rows.append(counts / max(float(counts.sum()), 1.0))
            subject_rows.append(np.mean(np.stack(trial_rows), axis=0))
        matrix[row_index] = np.mean(np.stack(subject_rows), axis=0)
    return activities, labels, matrix


def plot_activity_primitive_heatmap(
    artifacts: VisualizationArtifacts,
    output_path: Path,
    csv_path: Path,
    *,
    dpi: int = 180,
) -> None:
    plt, _ = _load_plotting()
    rows = _session_join(artifacts.sessions[-1])
    activities, labels, matrix = activity_primitive_matrix(
        rows, artifacts.codebook_size
    )
    _write_csv(csv_path, [
        {"activity_id": activity, "activity_name": label.split(": ", 1)[-1],
         "aggregation": "trial_fraction_then_subject_equal_mean", **{
            f"P{primitive}": float(matrix[row, primitive])
            for primitive in range(artifacts.codebook_size)
        }}
        for row, (activity, label) in enumerate(zip(activities, labels))
    ])
    figure, axis = plt.subplots(
        figsize=(13, max(5.5, 0.48 * len(labels) + 1.8)), constrained_layout=True
    )
    image = axis.imshow(matrix, aspect="auto", cmap="magma", vmin=0.0)
    axis.set_yticks(np.arange(len(labels)), labels=labels, fontsize=8)
    axis.set_xticks(np.arange(artifacts.codebook_size), labels=[f"P{i}" for i in range(artifacts.codebook_size)], rotation=90, fontsize=7)
    axis.set_xlabel("E0 motion primitive")
    axis.set_title("Final-session activity × primitive fraction (trial/subject equal)")
    figure.colorbar(image, ax=axis, fraction=0.025, pad=0.01, label="Within-activity fraction")
    figure.savefig(output_path, dpi=int(dpi))
    plt.close(figure)


def _resample_tokens(
    tokens: Sequence[int],
    starts: Sequence[int],
    ends: Sequence[int],
    columns: int = 220,
) -> np.ndarray:
    values = np.asarray(tokens, dtype=np.int64)
    starts_array = np.asarray(starts, dtype=np.int64)
    ends_array = np.asarray(ends, dtype=np.int64)
    if not len(values) or starts_array.shape != values.shape or ends_array.shape != values.shape:
        raise ValueError("Cannot draw an empty or malformed primitive sequence.")
    if starts_array[0] != 0 or np.any(starts_array[1:] != ends_array[:-1]):
        raise ValueError("Primitive ownership must be a contiguous partition.")
    # Sample physical progress rather than token index.  This preserves the
    # longer edge ownership regions created by overlapping E0 windows.
    positions = (np.arange(columns, dtype=np.float64) + 0.5) * float(ends_array[-1]) / columns
    indices = np.searchsorted(ends_array, positions, side="right")
    indices = np.minimum(indices, len(values) - 1)
    return values[indices]


def plot_trajectories_and_predictions(
    artifacts: VisualizationArtifacts,
    output_path: Path,
    *,
    per_activity: int = 2,
    dpi: int = 180,
) -> None:
    plt, _ = _load_plotting()
    colour = _primitive_colormap(artifacts.codebook_size)
    rows = _session_join(artifacts.sessions[-1])
    selected: list[dict[str, Any]] = []
    for activity in sorted({int(row["activity_id"]) for row in rows}):
        candidates = sorted(
            (row for row in rows if int(row["activity_id"]) == activity),
            key=lambda row: (int(row["subject_id"]), int(row["trial_id"])),
        )
        chosen: list[dict[str, Any]] = []
        seen_subjects: set[int] = set()
        for row in candidates:
            if int(row["subject_id"]) not in seen_subjects:
                chosen.append(row)
                seen_subjects.add(int(row["subject_id"]))
            if len(chosen) >= int(per_activity):
                break
        selected.extend(chosen or candidates[:1])
    matrix = np.stack([
        _resample_tokens(
            row["primitive_sequence"],
            row["ownership_start_samples"],
            row["ownership_end_samples_exclusive"],
        )
        for row in selected
    ])
    figure, axis = plt.subplots(
        figsize=(17, max(6.0, 0.43 * len(selected) + 1.8)), constrained_layout=True
    )
    image = axis.imshow(
        matrix, aspect="auto", interpolation="nearest", cmap=colour,
        vmin=-0.5, vmax=artifacts.codebook_size - 0.5,
    )
    axis.set_yticks(np.arange(len(selected)), labels=[
        f"{row['activity_name']} | S{int(row['subject_id']):02d} T{int(row['trial_id'])} | "
        f"n={len(row['primitive_sequence'])}, used={len(set(row['primitive_sequence']))} | "
        f"D={_prediction_text(row['direct_prediction'])} "
        f"P={_prediction_text(row['primary_prediction'])} "
        f"G={_prediction_text(row['global_prediction'])}"
        for row in selected
    ], fontsize=7)
    axis.set_xticks(np.linspace(0, matrix.shape[1] - 1, 6), labels=("0", ".2", ".4", ".6", ".8", "1"))
    axis.set_xlabel("Normalised trial progress")
    axis.set_title("Final-session frozen E0 trajectories and three-layer predictions")
    figure.colorbar(image, ax=axis, fraction=0.025, pad=0.01, label="Primitive ID")
    figure.savefig(output_path, dpi=int(dpi))
    plt.close(figure)


def _activity_name_map(session: ScoredSession) -> dict[int, str]:
    result: dict[int, str] = {}
    for row in session.predictions:
        result[int(row["activity_label"])] = str(row["activity_name"])
    return result


def plot_three_layer_confusions(
    session: ScoredSession,
    output_path: Path,
    *,
    dpi: int = 180,
) -> None:
    plt, _ = _load_plotting()
    class_count = int(session.metrics["expected_class_count"])
    names = _activity_name_map(session)
    row_labels = [f"{index}: {names.get(index, 'class')}" for index in range(class_count)]
    figure, axes = plt.subplots(1, 3, figsize=(24, 7.5), constrained_layout=True)
    image = None
    for axis, layer_name in zip(axes, LAYER_ORDER):
        layer = session.metrics["layers"][layer_name]
        counts = np.asarray(layer["confusion_matrix_with_unknown_column"], dtype=np.int64)
        totals = counts.sum(axis=1, keepdims=True)
        fractions = np.divide(
            counts, totals, out=np.zeros_like(counts, dtype=np.float64), where=totals > 0
        )
        image = axis.imshow(fractions, cmap="Blues", vmin=0.0, vmax=1.0, aspect="auto")
        if layer_name == "direct_registry":
            # Novel registry IDs are append-only identifiers, not activity
            # semantics.  Calling them by activity names would falsely imply
            # that a deployable mapping was known online.
            column_labels = [f"R{index}" for index in range(class_count)] + ["Unknown"]
        else:
            column_labels = [f"A{index}: {names.get(index, 'class')}" for index in range(class_count)] + ["Unknown"]
        axis.set_xticks(np.arange(class_count + 1), labels=column_labels, rotation=55, ha="right", fontsize=6)
        axis.set_yticks(np.arange(class_count), labels=row_labels, fontsize=6)
        axis.set_xlabel("Predicted")
        axis.set_ylabel("True")
        qualifier = (
            "deployable IDs; no test-label alignment"
            if layer_name == "direct_registry"
            else (
                "primary score; novel IDs test-aligned only for scoring"
                if layer_name == "old_fixed_novel_hungarian"
                else "diagnostic upper bound; all IDs test-aligned"
            )
        )
        axis.set_title(
            f"{LAYER_TITLES[layer_name]}\n{qualifier}\n"
            f"all={float(layer['all_accuracy']):.3f}, old={float(layer['old_accuracy']):.3f}, "
            f"new={float(layer['new_accuracy']):.3f}, H={float(layer['h_score']):.3f}",
            fontsize=9,
        )
        for row in range(class_count):
            for column in range(class_count + 1):
                if counts[row, column] > 0:
                    axis.text(
                        column, row, f"{fractions[row, column]:.2f}\n{counts[row, column]}",
                        ha="center", va="center", fontsize=5,
                        color="white" if fractions[row, column] > 0.55 else "black",
                    )
    if image is not None:
        figure.colorbar(image, ax=axes, fraction=0.018, pad=0.01, label="Row-normalised fraction")
    figure.suptitle(f"Strict CGCD session {session.session}: three scoring layers")
    figure.savefig(output_path, dpi=int(dpi))
    plt.close(figure)


def build_used_k_rows(artifacts: VisualizationArtifacts) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    def append(scope: str, session: int | str, role: str, usage: Mapping[str, Any]) -> None:
        counts = [int(value) for value in usage.get("counts", [])]
        rows.append({
            "scope": scope,
            "session": session,
            "role": role,
            "capacity_k": int(usage["capacity_k"]),
            "actual_used_k": int(usage["used_k"]),
            "dead_k": int(usage["dead_k"]),
            "dead_fraction": float(usage["dead_fraction"]),
            "effective_k": float(usage["effective_k"]),
            "token_count": int(sum(counts)),
            "primitive_counts": " ".join(map(str, counts)),
        })

    for role, usage in artifacts.offline_usage.items():
        append("offline", "offline", str(role), usage)
    for session in artifacts.sessions:
        append("online", session.session, "incoming", session.metrics["incoming_codebook_usage"])
        append("online", session.session, "evaluation", session.metrics["evaluation_codebook_usage"])
    return rows


def plot_used_k_summary(
    rows: Sequence[Mapping[str, Any]],
    output_path: Path,
    *,
    dpi: int = 180,
) -> None:
    plt, _ = _load_plotting()
    labels = [
        f"{row['scope']}:{row['session']}\n{row['role']}" for row in rows
    ]
    used = np.asarray([int(row["actual_used_k"]) for row in rows])
    dead = np.asarray([int(row["dead_k"]) for row in rows])
    effective = np.asarray([float(row["effective_k"]) for row in rows])
    x = np.arange(len(rows))
    figure, axis = plt.subplots(figsize=(14, 6), constrained_layout=True)
    axis.bar(x, used, color="#2563eb", label="Actual used K")
    axis.bar(x, dead, bottom=used, color="#cbd5e1", label="Dead codes")
    axis.set_xticks(x, labels=labels, rotation=35, ha="right", fontsize=8)
    axis.set_ylabel("Codebook entries")
    axis.set_ylim(0, max(33, int((used + dead).max()) + 1))
    second = axis.twinx()
    second.plot(x, effective, color="#dc2626", marker="o", label="Effective K")
    second.set_ylabel("Entropy-based effective K")
    handles, handle_labels = axis.get_legend_handles_labels()
    handles2, labels2 = second.get_legend_handles_labels()
    axis.legend(handles + handles2, handle_labels + labels2, loc="lower right")
    axis.set_title("Actual K32 utilisation across offline and online artifacts")
    figure.savefig(output_path, dpi=int(dpi))
    plt.close(figure)


def build_registry_evolution_rows(artifacts: VisualizationArtifacts) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    previous_k = 6
    for session in artifacts.sessions:
        metrics = session.metrics
        discovery = session.discovery.get("discovery", {})
        candidates = discovery.get("candidates", [])
        initial = session.discovery.get("initial_routing", [])
        rows.append({
            "session": session.session,
            "registry_k_before": previous_k,
            "registry_k_after": int(metrics["registered_class_count"]),
            "novel_registry_k_after": int(metrics["registered_novel_count"]),
            "registered_this_session": len(metrics.get("registered_this_session", [])),
            "initial_unknown_count": sum(not bool(item.get("accepted_known")) for item in initial),
            "discovery_pool_count": len(discovery.get("unknown_trial_ids", [])),
            "candidate_count": len(candidates),
            "accepted_candidate_count": sum(bool(item.get("accepted")) for item in candidates),
            "rejected_candidate_count": sum(not bool(item.get("accepted")) for item in candidates),
            "unknown_buffer_after": int(metrics["unknown_buffer_size"]),
        })
        previous_k = int(metrics["registered_class_count"])
    return rows


def plot_registry_evolution(
    rows: Sequence[Mapping[str, Any]],
    output_path: Path,
    *,
    dpi: int = 180,
) -> None:
    plt, _ = _load_plotting()
    sessions = np.asarray([int(row["session"]) for row in rows])
    figure, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    axes[0].plot(sessions, [row["registry_k_before"] for row in rows], "o--", label="Activity registry K before")
    axes[0].plot(sessions, [row["registry_k_after"] for row in rows], "o-", label="Activity registry K after")
    axes[0].bar(sessions, [row["registered_this_session"] for row in rows], alpha=0.35, label="New rows appended")
    axes[0].set_xticks(sessions)
    axes[0].set_xlabel("Online session")
    axes[0].set_ylabel("Activity-class registry entries")
    axes[0].set_title("Append-only activity registry (E0 codebook remains K32)")
    axes[0].legend(fontsize=8)
    axes[1].bar(
        sessions - 0.2, [row["initial_unknown_count"] for row in rows],
        width=0.2, label="Incoming initially unknown",
    )
    axes[1].bar(
        sessions, [row["discovery_pool_count"] for row in rows],
        width=0.2, label="Unknown discovery pool",
    )
    axes[1].bar(
        sessions + 0.2, [row["unknown_buffer_after"] for row in rows],
        width=0.2, label="Unresolved buffer after",
    )
    axes[1].set_xticks(sessions)
    axes[1].set_xlabel("Online session")
    axes[1].set_ylabel("Trial count")
    axes[1].set_title("Unknown buffer and discovery flow")
    axes[1].legend(fontsize=8)
    figure.savefig(output_path, dpi=int(dpi))
    plt.close(figure)


def generate_visualizations(
    offline_run_dir: str | Path,
    online_run_dir: str | Path,
    output_dir: str | Path,
    *,
    maximum_trial_plots: int = 0,
    sample_rate_hz: float = 100.0,
    dpi: int = 180,
    resume: bool = False,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Generate all strict figures without mutating any training artifact."""

    artifacts = load_visualization_artifacts(offline_run_dir, online_run_dir)
    output = Path(output_dir).expanduser().resolve()
    manifest_path = output / "visualization_manifest.json"
    rendering_config = {
        "maximum_trial_plots": int(maximum_trial_plots),
        "sample_rate_hz": float(sample_rate_hz),
        "dpi": int(dpi),
        "activity_heatmap_aggregation": "trial_fraction_then_subject_equal_mean",
        "trajectory_progress_basis": "ownership_duration_samples",
    }
    if manifest_path.is_file() and bool(resume):
        previous = _read_json(manifest_path)
        if previous.get("input_artifact_sha256") != dict(artifacts.protected_hashes):
            raise RuntimeError("Existing visualisation belongs to changed input artifacts.")
        if previous.get("rendering_config") != rendering_config:
            raise RuntimeError(
                "Existing visualisation uses different plot parameters; choose a new "
                "output directory or rerun with --overwrite."
            )
        return previous
    if output.exists() and any(output.iterdir()) and not bool(overwrite):
        raise FileExistsError(
            f"Visualisation output is not empty: {output}. Use --resume or --overwrite."
        )
    output.mkdir(parents=True, exist_ok=True)
    _load_plotting()
    created: list[str] = []

    used_rows = build_used_k_rows(artifacts)
    _write_csv(output / "used_k_summary.csv", used_rows)
    used_summary = {
        "capacity_k": artifacts.codebook_size,
        "rows": used_rows,
        "final_evaluation_actual_used_k": int(used_rows[-1]["actual_used_k"]),
        "final_evaluation_effective_k": float(used_rows[-1]["effective_k"]),
    }
    _write_json(output / "used_k_summary.json", used_summary)
    plot_used_k_summary(used_rows, output / "used_k_summary.png", dpi=dpi)
    created.extend(("used_k_summary.csv", "used_k_summary.json", "used_k_summary.png"))

    plot_activity_primitive_heatmap(
        artifacts,
        output / "activity_primitive_heatmap.png",
        output / "activity_primitive_heatmap.csv",
        dpi=dpi,
    )
    created.extend(("activity_primitive_heatmap.png", "activity_primitive_heatmap.csv"))
    plot_trajectories_and_predictions(
        artifacts, output / "trajectories_and_predictions.png", dpi=dpi
    )
    created.append("trajectories_and_predictions.png")

    for session in artifacts.sessions:
        name = f"confusion_three_layers_session_{session.session}.png"
        plot_three_layer_confusions(session, output / name, dpi=dpi)
        created.append(name)

    evolution_rows = build_registry_evolution_rows(artifacts)
    _write_csv(output / "registry_unknown_evolution.csv", evolution_rows)
    plot_registry_evolution(
        evolution_rows, output / "registry_unknown_evolution.png", dpi=dpi
    )
    created.extend(("registry_unknown_evolution.csv", "registry_unknown_evolution.png"))

    trial_paths, trial_rows = plot_all_trial_trajectories(
        artifacts,
        output / "trial_signal_and_primitives",
        maximum_plots=int(maximum_trial_plots),
        sample_rate_hz=float(sample_rate_hz),
        dpi=dpi,
    )
    created.append("trial_signal_and_primitives/trial_trajectory_index.csv")
    created.extend(
        f"trial_signal_and_primitives/{Path(path).name}" for path in trial_paths
    )

    after_hashes = _hash_map(Path(path) for path in artifacts.protected_hashes)
    if after_hashes != dict(artifacts.protected_hashes):
        raise RuntimeError("A protected offline/online artifact changed during plotting.")
    result = {
        "schema": "hhr_strict_cgcd_visualization_v1",
        "profile": PROFILE,
        "fold": int(artifacts.manifest["fold"]),
        "seed": int(artifacts.manifest["seed"]),
        "posthoc_scoring_only": True,
        "learner_api_called": False,
        "registry_or_model_writeback": False,
        "labels_read_only_from_frozen_scorer_artifacts": True,
        "source_npz_activity_labels_read": False,
        "token_id_scope": "single_fold_seed_member_only",
        "cross_fold_token_id_averaging_performed": False,
        "e0_codebook_capacity_changed_online": False,
        "online_growth_object": "activity_class_registry",
        "matplotlib_backend": "Agg",
        "input_artifact_sha256": dict(artifacts.protected_hashes),
        "actual_used_k_final_evaluation": used_summary["final_evaluation_actual_used_k"],
        "unique_trial_plot_count": len(trial_rows),
        "maximum_trial_plots": int(maximum_trial_plots),
        "rendering_config": rendering_config,
        "files": created,
        "complete": True,
    }
    _write_json(manifest_path, result)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate post-hoc figures for a strict frozen A2/E0 CGCD member."
    )
    parser.add_argument("--offline-run-dir", required=True)
    parser.add_argument("--online-run-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--maximum-trial-plots", type=int, default=0,
        help="0 plots every unique observed trial; a positive value is a deterministic cap.",
    )
    parser.add_argument("--sample-rate-hz", type=float, default=100.0)
    parser.add_argument("--dpi", type=int, default=180)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> dict[str, Any]:
    args = build_parser().parse_args(argv)
    if args.resume and args.overwrite:
        raise ValueError("--resume and --overwrite are mutually exclusive.")
    if int(args.maximum_trial_plots) < 0 or float(args.sample_rate_hz) <= 0 or int(args.dpi) < 50:
        raise ValueError("Plot limit/rate/dpi arguments are invalid.")
    result = generate_visualizations(
        args.offline_run_dir,
        args.online_run_dir,
        args.output_dir,
        maximum_trial_plots=int(args.maximum_trial_plots),
        sample_rate_hz=float(args.sample_rate_hz),
        dpi=int(args.dpi),
        resume=bool(args.resume),
        overwrite=bool(args.overwrite),
    )
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return result


if __name__ == "__main__":
    main()


__all__ = [
    "LAYER_ORDER",
    "ScoredSession",
    "VisualizationArtifacts",
    "activity_primitive_matrix",
    "build_registry_evolution_rows",
    "build_used_k_rows",
    "generate_visualizations",
    "load_raw_trial_signals",
    "load_visualization_artifacts",
    "plot_activity_primitive_heatmap",
    "plot_all_trial_trajectories",
    "plot_registry_evolution",
    "plot_three_layer_confusions",
    "plot_trajectories_and_predictions",
    "plot_used_k_summary",
]
