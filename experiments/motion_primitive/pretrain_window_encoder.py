"""Independent HAR-only ResNet1D window warm-up for the A2 route."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import logging
import math
import os
import random
import sys
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
from sklearn.metrics import confusion_matrix, f1_score
from torch import nn
from torch.optim import SGD
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, Dataset

from experiments.motion_primitive.motion_checkpoint import motion_state_dict_sha256
from experiments.motion_primitive.strict_protocol import build_registered_protocol
from models.resnet1d import ResNet1D
from models.window_pretrain import (
    SupervisedContrastiveLoss,
    WindowProjectionHead,
    info_nce_loss,
)


LOGGER = logging.getLogger("hhr_window_pretrain")
SCHEMA = "hhr_har_window_pretrain_v2"
IDENTITY_SCHEMA = "hhr_har_window_pretrain_identity_v2"


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    return value


def _write_json(path: Path, payload: Any) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(_jsonable(payload), ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    temporary.replace(path)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_checkpoint(path: Path) -> dict[str, Any]:
    try:
        value = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        value = torch.load(path, map_location="cpu")
    if not isinstance(value, dict):
        raise TypeError(f"Checkpoint {path} is not a dictionary.")
    return value


def _backbone_state_dict_sha256(state: dict[str, torch.Tensor]) -> str:
    """Hash exactly the ResNet1D state consumed by the downstream A2 stage."""

    backbone = {
        key.removeprefix("0."): value
        for key, value in state.items()
        if key.startswith("0.")
    }
    if not backbone:
        raise RuntimeError("Warm-up state lacks the sequential ResNet1D prefix '0.'.")
    return motion_state_dict_sha256(backbone)


def _validate_checkpoint_state_hashes(
    checkpoint: Mapping[str, Any],
) -> tuple[str, str]:
    """Verify both state aliases and the full/backbone tensor digests."""

    state = checkpoint.get("model_state_dict")
    alias = checkpoint.get("model")
    if not isinstance(state, Mapping) or not isinstance(alias, Mapping):
        raise RuntimeError("Warm-up checkpoint lacks model state aliases.")
    if set(state) != set(alias):
        raise RuntimeError("Warm-up checkpoint model state aliases differ in keys.")
    for key in state:
        left = state[key]
        right = alias[key]
        if not isinstance(left, torch.Tensor) or not isinstance(right, torch.Tensor):
            raise TypeError(f"Warm-up checkpoint state {key!r} is not a tensor.")
        if (
            left.dtype != right.dtype
            or left.layout != right.layout
            or tuple(left.shape) != tuple(right.shape)
            or not torch.equal(left.detach().cpu(), right.detach().cpu())
        ):
            raise RuntimeError(
                "Warm-up checkpoint model state aliases differ in values."
            )
    canonical = dict(state)
    observed_full = motion_state_dict_sha256(canonical)
    recorded_full = checkpoint.get("model_state_dict_sha256")
    if recorded_full != observed_full:
        raise RuntimeError(
            "Warm-up checkpoint full tensor-state SHA256 mismatch: "
            f"recorded={recorded_full!r}, observed={observed_full!r}."
        )
    observed_backbone = _backbone_state_dict_sha256(canonical)
    recorded_backbone = checkpoint.get("backbone_state_dict_sha256")
    if recorded_backbone != observed_backbone:
        raise RuntimeError(
            "Warm-up checkpoint backbone tensor-state SHA256 mismatch: "
            f"recorded={recorded_backbone!r}, observed={observed_backbone!r}."
        )
    return observed_full, observed_backbone


def _validate_determinism_record(
    record: Mapping[str, Any], *, require_python_hash_seed: bool = False
) -> None:
    """Fail closed when an artifact only claims, but did not run, strict mode."""

    required = {
        "enabled": True,
        "torch_deterministic_algorithms": True,
        "cudnn_benchmark": False,
        "cublas_workspace_config": ":4096:8",
    }
    observed = {key: record.get(key) for key in required}
    if observed != required:
        raise RuntimeError(
            "Warm-up deterministic runtime record is invalid: "
            f"{observed!r} != {required!r}."
        )
    if require_python_hash_seed and str(record.get("python_hash_seed")) != "0":
        raise RuntimeError("Warm-up child process did not use PYTHONHASHSEED=0.")
    if record.get("cuda_device_name") is not None:
        cuda_required = {
            "cudnn_deterministic": True,
            "cudnn_allow_tf32": True,
            "cuda_matmul_allow_tf32": False,
        }
        cuda_observed = {key: record.get(key) for key in cuda_required}
        if cuda_observed != cuda_required:
            raise RuntimeError(
                "Warm-up CUDA deterministic runtime record is invalid: "
                f"{cuda_observed!r} != {cuda_required!r}."
            )


def _implementation_fingerprint() -> dict[str, Any]:
    """Bind a resumable warm-up member to the code that produced it."""

    relative_paths = (
        "experiments/motion_primitive/pretrain_window_encoder.py",
        "experiments/motion_primitive/strict_protocol.py",
        "models/resnet1d.py",
        "models/window_pretrain.py",
    )
    files = {
        relative: _file_sha256(PROJECT_ROOT / relative)
        for relative in relative_paths
    }
    digest = hashlib.sha256()
    for relative, file_hash in sorted(files.items()):
        digest.update(relative.encode("utf-8"))
        digest.update(file_hash.encode("ascii"))
    return {
        "algorithm": "sha256_path_and_file_sha256_v1",
        "files": files,
        "combined_sha256": digest.hexdigest(),
    }


def _resolved_device_name(value: str) -> str:
    device = torch.device(
        value if value != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")
    return str(device)


def _seed_worker(_worker_id: int) -> None:
    """Seed every RNG a DataLoader worker may use from PyTorch's worker seed."""

    worker_seed = int(torch.initial_seed() % (2**32))
    random.seed(worker_seed)
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)


def _seed_everything(seed: int, deterministic: bool) -> tuple[torch.Generator, dict[str, Any]]:
    """Configure a reproducible warm-up before CUDA creates any BLAS handles."""

    seed = int(seed)
    deterministic = bool(deterministic)
    if deterministic:
        # CUDA 10.2+ requires this workspace mode for deterministic cuBLAS calls.
        # Setting it here is early enough for direct script use because no CUDA
        # tensor or model has been created before this function is called.
        # Use one canonical workspace profile rather than inheriting an
        # arbitrary parent value.  The parent runners set the same value before
        # Python starts; this assignment also protects direct CLI execution.
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Set both branches explicitly so repeated in-process test calls cannot
    # inherit a previous strict/non-strict state.
    torch.use_deterministic_algorithms(deterministic)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = deterministic
        if deterministic:
            # Preserve the historical cuDNN convolution precision route. TF32
            # selection is a numeric-mode choice, not a source of randomness;
            # deterministic kernels are enforced independently above.
            torch.backends.cudnn.allow_tf32 = True
    if deterministic:
        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = False
        torch.set_float32_matmul_precision("highest")

    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed + 17)
    runtime = {
        "enabled": deterministic,
        "torch_deterministic_algorithms": bool(
            torch.are_deterministic_algorithms_enabled()
        ),
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
        "cuda_matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "python_hash_seed": os.environ.get("PYTHONHASHSEED"),
        "torch_version": str(torch.__version__),
        "cuda_version": None if torch.version.cuda is None else str(torch.version.cuda),
        "cudnn_version": (
            None if not torch.backends.cudnn.is_available()
            else int(torch.backends.cudnn.version())
        ),
        "cuda_device_name": (
            None if not torch.cuda.is_available() else str(torch.cuda.get_device_name(0))
        ),
    }
    return generator, runtime


def _run_identity(args: argparse.Namespace) -> dict[str, Any]:
    """Identity used to prevent a resumed directory from mixing experiments."""

    ignored = {"output_dir", "resume"}
    arguments = {
        key: value for key, value in vars(args).items() if key not in ignored
    }
    arguments["npz_path"] = str(Path(args.npz_path).expanduser().resolve())
    return {
        "schema": IDENTITY_SCHEMA,
        "fold": int(args.fold),
        "seed": int(args.seed),
        "npz_path": arguments["npz_path"],
        "npz_sha256": _file_sha256(Path(arguments["npz_path"])),
        "arguments": _jsonable(arguments),
        "resolved_device": _resolved_device_name(str(args.device)),
        "implementation_fingerprint": _implementation_fingerprint(),
    }


class WindowDataset(Dataset):
    def __init__(
        self,
        windows: np.ndarray,
        labels: np.ndarray,
        *,
        training: bool,
        weak_scale_std: float = 0.1,
        strong_scale_std: float = 0.2,
    ) -> None:
        self.windows = np.asarray(windows, dtype=np.float32)
        self.labels = np.asarray(labels, dtype=np.int64)
        self.training = bool(training)
        self.weak_scale_std = float(weak_scale_std)
        self.strong_scale_std = float(strong_scale_std)
        if self.windows.ndim != 3 or self.windows.shape[1] != 6:
            raise ValueError("WindowDataset expects [N,6,T].")
        if self.labels.shape != (len(self.windows),):
            raise ValueError("Window labels differ from window count.")

    @staticmethod
    def _scale(window: torch.Tensor, standard_deviation: float) -> torch.Tensor:
        scale = torch.randn(
            window.shape[0], 1, dtype=window.dtype, device=window.device
        ) * float(standard_deviation) + 1.0
        return window * scale

    def __getitem__(self, index: int):
        window = torch.from_numpy(self.windows[int(index)]).float()
        label = int(self.labels[int(index)])
        if not self.training:
            return window, label
        return (
            self._scale(window.clone(), self.weak_scale_std),
            self._scale(window.clone(), self.strong_scale_std),
        ), label

    def __len__(self) -> int:
        return len(self.windows)


def _flatten_trials(protocol, role: str) -> tuple[np.ndarray, np.ndarray]:
    trials = getattr(protocol, role)
    windows = np.concatenate([item.windows for item in trials], axis=0)
    trial_ids = np.concatenate([
        np.full(len(item.windows), int(item.trial_id), dtype=np.int64) for item in trials
    ])
    labels, _, _ = protocol.truth.join(trial_ids)
    if not set(np.unique(labels).tolist()).issubset(set(range(6))):
        raise RuntimeError("Window warm-up received a non-old activity.")
    return windows.astype(np.float32), labels


def _parameter_groups(model: nn.Module, weight_decay: float) -> list[dict]:
    regularized: list[nn.Parameter] = []
    unregularized: list[nn.Parameter] = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        (unregularized if name.endswith(".bias") or parameter.ndim == 1 else regularized).append(parameter)
    return [
        {"params": regularized, "weight_decay": float(weight_decay)},
        {"params": unregularized, "weight_decay": 0.0},
    ]


def _metrics(labels: np.ndarray, predictions: np.ndarray) -> dict[str, Any]:
    labels = np.asarray(labels, dtype=np.int64)
    predictions = np.asarray(predictions, dtype=np.int64)
    matrix = confusion_matrix(labels, predictions, labels=list(range(6)))
    class_accuracy = np.diag(matrix) / np.maximum(matrix.sum(axis=1), 1)
    return {
        "num_samples": int(len(labels)),
        "overall_accuracy": float(np.mean(labels == predictions)),
        "mean_class_accuracy": float(class_accuracy.mean()),
        "macro_f1": float(f1_score(labels, predictions, labels=list(range(6)), average="macro", zero_division=0)),
        "per_class_accuracy": class_accuracy.tolist(),
        "confusion_matrix": matrix.tolist(),
    }


@torch.inference_mode()
def _evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> dict[str, Any]:
    model.eval()
    labels: list[np.ndarray] = []
    predictions: list[np.ndarray] = []
    for windows, target in loader:
        _, logits = model(windows.to(device))
        labels.append(target.numpy())
        predictions.append(logits.argmax(dim=1).cpu().numpy())
    return _metrics(np.concatenate(labels), np.concatenate(predictions))


def _checkpoint(
    model: nn.Module,
    optimizer: SGD,
    *,
    epoch: int,
    metrics: dict[str, Any],
    args: argparse.Namespace,
    protocol,
    identity: dict[str, Any],
    determinism: dict[str, Any],
) -> dict[str, Any]:
    state = copy.deepcopy(model.state_dict())
    metadata = {
        "uschad_npz_path": str(Path(args.npz_path).expanduser().resolve()),
        "uschad_npz_sha256": protocol.npz_sha256,
        "uschad_window_size": int(args.window_size),
        "uschad_window_stride": int(args.window_stride),
        "uschad_sample_unit": "window",
        "uschad_split_mode": "subject",
        "uschad_recompute_norm_from_train_subjects": True,
        "uschad_norm_eps": 1e-6,
        "uschad_train_subjects": list(protocol.split.train),
        "offline_val_subjects": list(protocol.split.validation),
        "uschad_test_subjects": list(protocol.split.outer_test),
        "uschad_cv_fold": int(args.fold),
        "har_in_channels": 6,
        "har_feat_dim": 256,
        "har_base_channels": 64,
        "har_dropout": 0.0,
        "har_aug_mode": "weak_strong",
        "har_weak_jitter_std": 0.0,
        "har_weak_scale_std": float(args.weak_scale_std),
        "har_strong_jitter_std": 0.0,
        "har_strong_scale_std": float(args.strong_scale_std),
        "har_time_mask_ratio": 0.0,
        "projection_hidden_dim": 2048,
        "projection_bottleneck_dim": 256,
        "outer_test_used_during_window_pretrain": False,
        "determinism": copy.deepcopy(determinism),
    }
    return {
        "schema": SCHEMA,
        "run_identity": copy.deepcopy(identity),
        "model": state,
        "model_state_dict": state,
        "model_state_dict_sha256": motion_state_dict_sha256(state),
        "backbone_state_dict_sha256": _backbone_state_dict_sha256(state),
        "optimizer": optimizer.state_dict(),
        "epoch": int(epoch),
        "validation_metrics": metrics,
        "validation_score": float(metrics["macro_f1"]),
        "selection_metric": "macro_f1",
        "determinism": copy.deepcopy(determinism),
        "loss_weights": {"cls": 1.0, "cluster": 0.0, "contrast": 0.65, "supcon": 0.35},
        "experiment_metadata": metadata,
        "fold_normalization": {
            "mean": protocol.fold_mean,
            "std": protocol.fold_std,
            "fit_scope": "train_subjects_old_classes_only",
        },
        "arguments": vars(args),
    }


def train(args: argparse.Namespace) -> Path:
    output = Path(args.output_dir).expanduser().resolve()
    generator, determinism = _seed_everything(
        int(args.seed), bool(args.deterministic)
    )
    identity = _run_identity(args)
    if output.exists() and not bool(args.resume):
        raise FileExistsError(f"Output already exists: {output}")
    complete = output / "complete.json"
    if bool(args.resume) and complete.is_file():
        completion = json.loads(complete.read_text(encoding="utf-8"))
        if completion.get("schema") != SCHEMA:
            raise RuntimeError("Completed window-pretrain schema is incompatible.")
        expected_checkpoint_name = (
            "model_best.pt"
            if str(args.selection_policy) == "best_val_macro_f1"
            else "model_last.pt"
        )
        checkpoint_name = str(completion.get("checkpoint", ""))
        selection = completion.get("selection")
        if not isinstance(selection, Mapping) or selection.get("policy") != str(
            args.selection_policy
        ):
            raise RuntimeError(
                "Completed window-pretrain checkpoint-selection policy differs."
            )
        if checkpoint_name != expected_checkpoint_name:
            raise RuntimeError(
                "Completed window-pretrain canonical checkpoint differs from its "
                "selection policy."
            )
        checkpoint_path = (output / checkpoint_name).resolve()
        if checkpoint_path.parent != output.resolve():
            raise RuntimeError(
                "Completed window-pretrain checkpoint escapes its output directory."
            )
        observed_identity = completion.get("identity")
        if observed_identity != identity:
            raise RuntimeError(
                "Completed window-pretrain directory records another run identity; "
                "use a new output directory."
            )
        if (
            completion.get("complete") is not True
            or not checkpoint_path.name
            or not checkpoint_path.is_file()
        ):
            raise RuntimeError("Window-pretrain completion artifacts are incomplete.")
        if _file_sha256(checkpoint_path) != completion.get("checkpoint_sha256"):
            raise RuntimeError("Window-pretrain checkpoint SHA256 mismatch.")
        checkpoint = _load_checkpoint(checkpoint_path)
        if checkpoint.get("run_identity") != identity:
            raise RuntimeError("Window-pretrain checkpoint records another run identity.")
        if int(checkpoint.get("epoch", -1)) != int(
            completion.get("selected_epoch", -2)
        ):
            raise RuntimeError(
                "Window-pretrain selected checkpoint epoch differs from completion."
            )
        if float(checkpoint.get("validation_score", math.nan)) != float(
            completion.get("selected_validation_macro_f1", math.nan)
        ):
            raise RuntimeError(
                "Window-pretrain selected validation score differs from completion."
            )
        observed_full, observed_backbone = _validate_checkpoint_state_hashes(
            checkpoint
        )
        if completion.get("model_state_dict_sha256") != observed_full:
            raise RuntimeError(
                "Window-pretrain completion selected tensor-state SHA256 mismatch."
            )
        if completion.get("backbone_state_dict_sha256") != observed_backbone:
            raise RuntimeError(
                "Window-pretrain completion selected backbone SHA256 mismatch."
            )
        if bool(args.deterministic):
            checkpoint_runtime = checkpoint.get("determinism")
            completion_runtime = completion.get("determinism")
            if checkpoint_runtime != completion_runtime:
                raise RuntimeError(
                    "Window-pretrain checkpoint/completion deterministic runtime differs."
                )
            _validate_determinism_record(
                checkpoint_runtime if isinstance(checkpoint_runtime, Mapping) else {}
            )
        return checkpoint_path
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(
            f"Output directory is incomplete: {output}. Preserve it under another "
            "name before restarting this member."
        )
    output.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[logging.FileHandler(output / "train.log", encoding="utf-8"), logging.StreamHandler()],
        force=True,
    )
    device = torch.device(_resolved_device_name(str(args.device)))
    protocol = build_registered_protocol(
        args.npz_path,
        fold=int(args.fold),
        seed=int(args.seed),
        window_size=int(args.window_size),
        stride=int(args.window_stride),
        shuffle_novel_classes=False,
    )
    train_windows, train_labels = _flatten_trials(protocol, "offline_train")
    val_windows, val_labels = _flatten_trials(protocol, "offline_validation")
    if int(args.smoke_max_windows) > 0:
        count = min(int(args.smoke_max_windows), len(train_windows))
        train_windows, train_labels = train_windows[:count], train_labels[:count]
        val_count = min(max(12, count // 4), len(val_windows))
        val_windows, val_labels = val_windows[:val_count], val_labels[:val_count]
    train_data = WindowDataset(
        train_windows,
        train_labels,
        training=True,
        weak_scale_std=float(args.weak_scale_std),
        strong_scale_std=float(args.strong_scale_std),
    )
    val_data = WindowDataset(val_windows, val_labels, training=False)
    train_loader = DataLoader(
        train_data,
        batch_size=int(args.batch_size),
        shuffle=True,
        num_workers=int(args.num_workers),
        pin_memory=device.type == "cuda",
        drop_last=True,
        generator=generator,
        worker_init_fn=_seed_worker,
    )
    val_loader = DataLoader(
        val_data,
        batch_size=int(args.eval_batch_size),
        shuffle=False,
        num_workers=int(args.num_workers),
        worker_init_fn=_seed_worker,
    )
    backbone = ResNet1D(in_channels=6, feat_dim=256, base_channels=64, dropout=0.0)
    model = nn.Sequential(
        backbone,
        WindowProjectionHead(256, 6, hidden_dim=2048, bottleneck_dim=256),
    ).to(device)
    optimizer = SGD(
        _parameter_groups(model, float(args.weight_decay)),
        lr=float(args.learning_rate),
        momentum=0.9,
    )
    minimum_ratio = 1e-3
    scheduler = LambdaLR(
        optimizer,
        lr_lambda=lambda epoch: minimum_ratio + (1.0 - minimum_ratio) * 0.5 * (
            1.0 + math.cos(math.pi * min(max(float(epoch) / int(args.epochs), 0.0), 1.0))
        ),
    )
    supcon = SupervisedContrastiveLoss()
    best_score = -math.inf
    best_epoch = -1
    history_path = output / "history.jsonl"
    for epoch in range(1, int(args.epochs) + 1):
        model.train()
        running = {"loss": 0.0, "cls": 0.0, "contrast": 0.0, "supcon": 0.0, "count": 0}
        for (weak, strong), labels in train_loader:
            labels = labels.to(device)
            joined = torch.cat([weak.to(device), strong.to(device)], dim=0)
            projection, logits = model(joined)
            repeated_labels = labels.repeat(2)
            cls_loss = nn.functional.cross_entropy(logits / 0.1, repeated_labels)
            # Historical Happy HAR used temperature=1.0 for instance InfoNCE.
            # The separate 0.1 value above belongs only to class logits.
            contrast_loss = info_nce_loss(projection, temperature=1.0)
            first, second = projection.chunk(2)
            paired = torch.stack([first, second], dim=1)
            paired = nn.functional.normalize(paired, dim=-1)
            supcon_loss = supcon(paired, labels)
            loss = cls_loss + 0.65 * contrast_loss + 0.35 * supcon_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            batch = len(labels)
            for name, value in (("loss", loss), ("cls", cls_loss), ("contrast", contrast_loss), ("supcon", supcon_loss)):
                running[name] += float(value.detach()) * batch
            running["count"] += batch
        scheduler.step()
        validation = _evaluate(model, val_loader, device)
        record = {
            "epoch": epoch,
            "lr": float(optimizer.param_groups[0]["lr"]),
            **{name: running[name] / max(running["count"], 1) for name in ("loss", "cls", "contrast", "supcon")},
            "validation": validation,
        }
        payload = _checkpoint(
            model,
            optimizer,
            epoch=epoch,
            metrics=validation,
            args=args,
            protocol=protocol,
            identity=identity,
            determinism=determinism,
        )
        record["model_state_dict_sha256"] = payload["model_state_dict_sha256"]
        record["backbone_state_dict_sha256"] = payload[
            "backbone_state_dict_sha256"
        ]
        with history_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(_jsonable(record), ensure_ascii=False, allow_nan=False) + "\n")
        torch.save(payload, output / "model_last.pt")
        if validation["macro_f1"] > best_score:
            best_score = float(validation["macro_f1"])
            best_epoch = epoch
            torch.save(payload, output / "model_best.pt")
        LOGGER.info(
            "epoch=%d/%d loss=%.5f val_macro_f1=%.4f best=%.4f@%d",
            epoch, int(args.epochs), record["loss"], validation["macro_f1"], best_score, best_epoch,
        )
    best_path = output / "model_best.pt"
    last_path = output / "model_last.pt"
    best_checkpoint = _load_checkpoint(best_path)
    last_checkpoint = _load_checkpoint(last_path)
    _validate_checkpoint_state_hashes(best_checkpoint)
    _validate_checkpoint_state_hashes(last_checkpoint)
    selected_path = (
        best_path
        if str(args.selection_policy) == "best_val_macro_f1"
        else last_path
    )
    selected_checkpoint = (
        best_checkpoint if selected_path == best_path else last_checkpoint
    )
    digest = _file_sha256(selected_path)
    _write_json(output / "complete.json", {
        "schema": SCHEMA,
        "identity": identity,
        "fold": int(args.fold),
        "seed": int(args.seed),
        "best_epoch": int(best_epoch),
        "best_validation_macro_f1": float(best_score),
        "last_epoch": int(last_checkpoint["epoch"]),
        "last_validation_macro_f1": float(last_checkpoint["validation_score"]),
        "selected_epoch": int(selected_checkpoint["epoch"]),
        "selected_validation_macro_f1": float(
            selected_checkpoint["validation_score"]
        ),
        "checkpoint": selected_path.name,
        "checkpoint_sha256": digest,
        "model_state_dict_sha256": str(
            selected_checkpoint["model_state_dict_sha256"]
        ),
        "backbone_state_dict_sha256": str(
            selected_checkpoint["backbone_state_dict_sha256"]
        ),
        "best_checkpoint": best_path.name,
        "best_checkpoint_sha256": _file_sha256(best_path),
        "best_model_state_dict_sha256": str(
            best_checkpoint["model_state_dict_sha256"]
        ),
        "best_backbone_state_dict_sha256": str(
            best_checkpoint["backbone_state_dict_sha256"]
        ),
        "last_checkpoint": last_path.name,
        "last_checkpoint_sha256": _file_sha256(last_path),
        "last_model_state_dict_sha256": str(
            last_checkpoint["model_state_dict_sha256"]
        ),
        "last_backbone_state_dict_sha256": str(
            last_checkpoint["backbone_state_dict_sha256"]
        ),
        "selection": {
            "policy": str(args.selection_policy),
            "unit": "old_class_validation_windows",
            "outer_test_used": False,
            "tie_break": "earliest_epoch",
        },
        "determinism": determinism,
        "smoke_test": bool(int(args.smoke_max_windows) > 0),
        "outer_test_queries": 0,
        "complete": True,
    })
    return selected_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="HAR-only ResNet1D window warm-up for A2.")
    parser.add_argument("--npz-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--fold", type=int, required=True, choices=range(1, 8))
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--window-size", type=int, default=256, choices=(64, 128, 256))
    parser.add_argument("--window-stride", type=int, default=128, choices=(32, 64, 128))
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--eval-batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=0.1)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--weak-scale-std", type=float, default=0.1)
    parser.add_argument("--strong-scale-std", type=float, default=0.2)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--deterministic",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Require deterministic PyTorch/cuDNN/cuBLAS execution. Disable only "
            "for an explicitly labelled speed ablation."
        ),
    )
    parser.add_argument(
        "--selection-policy",
        choices=("best_val_macro_f1", "final_epoch"),
        default="best_val_macro_f1",
        help=(
            "Select only from old-class inner validation. best_val_macro_f1 "
            "preserves the historical route; final_epoch is a diagnostic control."
        ),
    )
    parser.add_argument("--smoke-max-windows", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    path = train(build_parser().parse_args(argv))
    print(path, flush=True)


if __name__ == "__main__":
    main()
