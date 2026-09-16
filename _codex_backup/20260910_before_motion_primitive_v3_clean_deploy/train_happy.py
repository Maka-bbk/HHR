import argparse
import json
import os
import math
import time
import random
from tqdm import tqdm
from copy import deepcopy

from sklearn.cluster import KMeans
import numpy as np
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
from torch.optim import SGD, lr_scheduler

from project_utils.general_utils import set_seed, init_experiment, AverageMeter
from project_utils.cluster_and_log_utils import log_accs_from_preds
from project_utils.cluster_utils import cluster_acc

from data.get_datasets import get_class_splits, get_datasets
from data.uschad import parse_subject_ids, uschad_trial_collate



from models.utils_simgcd import DINOHead, get_params_groups, SupConLoss, info_nce_logits, DistillLoss
from models.utils_simgcd_pro import get_kmeans_centroid_for_new_head
from models.utils_proto_aug import ProtoAugManager
from models.resnet1d import ResNet1D
from models.trial_pooling import build_trial_backbone, move_trial_batch_to_device
from config import exp_root_happy

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

def build_transforms(args):
    """
    Build train/test transforms for USC-HAD only.

    Current branch:
        USC-HAD only.
        No Happy image augmentation.
        HAR view generation is handled by data/uschad.py.

    Important:
        train_transform = None
        test_transform = None

    Because:
        data/uschad.py will internally create:
            train: [view1, view2]
            test:  single tensor [C, T]
    """

    if args.dataset_name != 'uschad':
        raise ValueError(
            "This branch is USC-HAD only. "
            "Please use --dataset_name uschad."
        )

    if args.image_aug_mode == 'on':
        raise ValueError(
            "image_aug_mode=on is invalid in USC-HAD-only branch. "
            "USC-HAD is time-series data, not image data. "
            "Use --image_aug_mode auto or --image_aug_mode off."
        )

    args.logger.info(
        "USC-HAD-only branch: skip all Happy image augmentations. "
        f"HAR augmentation is controlled by --har_aug_mode={args.har_aug_mode}."
    )

    return None, None
def build_backbone(args):
    """
    Build USC-HAD ResNet1D backbone.

    Input:
        [B, C, T]

    Default:
        [B, 6, 256]

    Output:
        [B, har_feat_dim]

    Default:
        [B, 256]
    """

    if args.dataset_name != 'uschad':
        raise ValueError(
            "This branch only supports USC-HAD. "
            "Please use --dataset_name uschad."
        )

    args.logger.info("Building USC-HAD ResNet1D backbone.")

    window_encoder = ResNet1D(
        in_channels=args.har_in_channels,
        feat_dim=args.har_feat_dim,
        base_channels=args.har_base_channels,
        dropout=args.har_dropout,
    )

    if args.uschad_sample_unit == 'trial':
        backbone = build_trial_backbone(window_encoder, args)
    else:
        backbone = window_encoder

    args.feat_dim = args.har_feat_dim
    args.num_mlp_layers = 3

    args.logger.info("Using ResNet1D backbone.")
    args.logger.info(f"har_in_channels = {args.har_in_channels}")
    args.logger.info(f"har_feat_dim = {args.har_feat_dim}")
    args.logger.info(f"har_base_channels = {args.har_base_channels}")
    args.logger.info(f"har_dropout = {args.har_dropout}")
    args.logger.info(f"args.feat_dim = {args.feat_dim}")
    args.logger.info(f"uschad_sample_unit = {args.uschad_sample_unit}")
    if args.uschad_sample_unit == 'trial':
        args.logger.info(f"trial_pooling = {args.trial_pooling}")
        args.logger.info(f"trial_view_mode = {args.trial_view_mode}")
        if args.trial_view_mode == 'full_random_crop':
            args.logger.info(f"trial_crop_ratio = {args.trial_crop_ratio}")
            args.logger.info(f"trial_min_windows = {args.trial_min_windows}")
        else:
            args.logger.info("trial crop parameters are inactive for full_full")
        if args.trial_pooling == 'mean_robust_max':
            args.logger.info(
                f"trial_robust_max_quantile = {args.trial_robust_max_quantile}"
            )
            args.logger.info(f"trial_pool_fusion_dim = {args.trial_pool_fusion_dim}")
            args.logger.info(
                f"trial_pool_fusion_dropout = {args.trial_pool_fusion_dropout}"
            )
        elif args.trial_pooling == 'gated_attention':
            args.logger.info(f"trial_attention_dim = {args.trial_attention_dim}")
            args.logger.info(f"trial_attention_dropout = {args.trial_attention_dropout}")
            args.logger.info(
                f"trial_attention_temperature = {args.trial_attention_temperature}"
            )
            args.logger.info(
                f"trial_attention_mean_mix = {args.trial_attention_mean_mix}"
            )

    return backbone

def seed_dataloader_worker(worker_id):
    """
    Seed each DataLoader worker.

    Why:
        USC-HAD HAR augmentation uses torch / numpy / python random-like sources.
        If workers are not explicitly seeded, changing num_workers may change
        the augmentation random stream and make experiments harder to reproduce.
    """

    worker_seed = torch.initial_seed() % 2**32

    np.random.seed(worker_seed)
    random.seed(worker_seed)
    torch.manual_seed(worker_seed)


def build_dataloader_generator(seed):
    """
    Build a torch.Generator for DataLoader shuffle order and worker base seeds.
    """

    generator = torch.Generator()
    generator.manual_seed(int(seed))
    return generator


def prepare_model_inputs(images, args):
    """Move window tensors or padded trial dictionaries to CUDA."""
    if args.uschad_sample_unit == 'trial':
        if isinstance(images, list):
            return [
                move_trial_batch_to_device(view, torch.device('cuda'))
                for view in images
            ]
        return move_trial_batch_to_device(images, torch.device('cuda'))

    if isinstance(images, list):
        return torch.cat(images, dim=0).cuda(non_blocking=True)
    return images.cuda(non_blocking=True)


def forward_prepared_model(model, prepared_inputs):
    """Return backbone features, projected features, and logits in view order."""
    if isinstance(prepared_inputs, list):
        features_list = []
        projected_list = []
        logits_list = []
        for view in prepared_inputs:
            features = model[0](view)
            projected, logits = model[1](features)
            features_list.append(features)
            projected_list.append(projected)
            logits_list.append(logits)
        return (
            torch.cat(features_list, dim=0),
            torch.cat(projected_list, dim=0),
            torch.cat(logits_list, dim=0),
        )

    features = model[0](prepared_inputs)
    projected, logits = model[1](features)
    return features, projected, logits


def forward_prepared_backbone(backbone, prepared_inputs):
    if isinstance(prepared_inputs, list):
        return torch.cat([backbone(view) for view in prepared_inputs], dim=0)
    return backbone(prepared_inputs)


def build_sgd_optimizer(model, args):
    """Use a lower LR for the pretrained trial window encoder when requested."""
    grouped = {}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        is_trial_encoder = (
            args.uschad_sample_unit == 'trial'
            and name.startswith('0.window_encoder.')
        )
        no_weight_decay = name.endswith('.bias') or parameter.ndim == 1
        key = (is_trial_encoder, no_weight_decay)
        grouped.setdefault(key, []).append(parameter)

    parameter_groups = []
    for (is_trial_encoder, no_weight_decay), parameters in grouped.items():
        group = {
            'params': parameters,
            'lr': (
                args.lr * args.trial_encoder_lr_scale
                if is_trial_encoder else args.lr
            ),
        }
        if no_weight_decay:
            group['weight_decay'] = 0.0
        parameter_groups.append(group)

    if len(parameter_groups) == 0:
        raise RuntimeError("No trainable parameters were found for SGD.")
    return SGD(
        parameter_groups,
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )


def build_cosine_lr_scheduler(optimizer, num_epochs, minimum_ratio=1e-3):
    """Apply one cosine multiplier so parameter-group LR ratios stay fixed."""
    num_epochs = int(num_epochs)
    minimum_ratio = float(minimum_ratio)
    if num_epochs < 1:
        raise ValueError("Cosine LR scheduling requires at least one epoch.")
    if not 0.0 <= minimum_ratio <= 1.0:
        raise ValueError("Cosine minimum LR ratio must be in [0, 1].")

    def cosine_multiplier(epoch):
        progress = min(max(float(epoch) / num_epochs, 0.0), 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return minimum_ratio + (1.0 - minimum_ratio) * cosine

    return lr_scheduler.LambdaLR(optimizer, lr_lambda=cosine_multiplier)


def configure_trial_encoder_for_epoch(student, epoch_number, args):
    if args.uschad_sample_unit != 'trial':
        return False
    freeze = epoch_number <= args.trial_encoder_freeze_epochs
    encoder = student[0].window_encoder
    for parameter in encoder.parameters():
        parameter.requires_grad = not freeze
    if freeze:
        # student.train() must not update pretrained BatchNorm statistics while frozen.
        encoder.eval()
    return freeze


def canonical_subject_ids(value):
    """Return a stable subject-id list for logs and checkpoint identity."""
    parsed = parse_subject_ids(value)
    if parsed is None:
        return []
    return sorted(set(int(subject_id) for subject_id in parsed))


def load_trial_window_encoder(backbone, checkpoint_path, args, logger):
    """Strictly load only ResNet1D weights from a window or trial checkpoint."""
    if checkpoint_path is None or str(checkpoint_path).strip() == '':
        return False
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"Trial window-encoder checkpoint not found: {checkpoint_path}"
        )

    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    metadata = checkpoint.get('experiment_metadata')
    if not isinstance(metadata, dict):
        raise RuntimeError(
            f"Window encoder checkpoint lacks experiment_metadata: {checkpoint_path}. "
            "Regenerate the window checkpoint with the current code so its window "
            "size and encoder architecture can be verified."
        )
    expected_metadata = {
        'uschad_window_size': int(args.uschad_window_size),
        'har_in_channels': int(args.har_in_channels),
        'har_feat_dim': int(args.har_feat_dim),
        'har_base_channels': int(args.har_base_channels),
        'har_dropout': float(args.har_dropout),
    }
    if args.uschad_recompute_norm_from_train_subjects:
        expected_metadata.update({
            'uschad_split_mode': str(args.uschad_split_mode),
            'uschad_recompute_norm_from_train_subjects': True,
            'uschad_norm_eps': float(args.uschad_norm_eps),
            'uschad_train_subjects': canonical_subject_ids(
                args.uschad_train_subjects
            ),
            'offline_val_subjects': canonical_subject_ids(
                args.offline_val_subjects
            ),
            'uschad_test_subjects': canonical_subject_ids(
                args.uschad_test_subjects
            ),
            'uschad_cv_fold': int(args.uschad_cv_fold),
        })
    metadata_mismatches = []
    for key, expected_value in expected_metadata.items():
        if key not in metadata:
            metadata_mismatches.append(f"{key}=<missing> expected {expected_value!r}")
            continue
        observed_value = metadata[key]
        if isinstance(expected_value, float):
            equal = math.isclose(
                float(observed_value), expected_value, rel_tol=0, abs_tol=1e-12
            )
        else:
            equal = observed_value == expected_value
        if not equal:
            metadata_mismatches.append(
                f"{key}={observed_value!r} expected {expected_value!r}"
            )
    if metadata_mismatches:
        raise RuntimeError(
            f"Window encoder checkpoint/data mismatch for {checkpoint_path}: "
            + "; ".join(metadata_mismatches)
        )

    source_state = checkpoint.get('model', checkpoint)
    target_state = backbone.window_encoder.state_dict()
    extracted = {}
    prefixes = ['0.window_encoder.', 'window_encoder.', '0.', '']
    for target_key, target_value in target_state.items():
        matches = []
        for prefix in prefixes:
            source_key = prefix + target_key
            if source_key in source_state:
                matches.append((source_key, source_state[source_key]))
        if len(matches) == 0:
            raise RuntimeError(
                f"Checkpoint {checkpoint_path} is missing window encoder key {target_key}."
            )
        source_key, source_value = matches[0]
        if tuple(source_value.shape) != tuple(target_value.shape):
            raise RuntimeError(
                f"Window encoder shape mismatch for {source_key}: "
                f"checkpoint={tuple(source_value.shape)}, expected={tuple(target_value.shape)}."
            )
        extracted[target_key] = source_value

    backbone.window_encoder.load_state_dict(extracted, strict=True)
    logger.info(
        f"Loaded ResNet1D window encoder from {checkpoint_path} "
        f"({len(extracted)} tensors, strict=True)."
    )
    return True


def experiment_metadata(args):
    metadata = {
        'uschad_npz_path': str(args.uschad_npz_path),
        'uschad_window_size': int(args.uschad_window_size),
        'uschad_sample_unit': str(args.uschad_sample_unit),
        'uschad_split_mode': str(args.uschad_split_mode),
        'uschad_recompute_norm_from_train_subjects': bool(
            args.uschad_recompute_norm_from_train_subjects
        ),
        'uschad_norm_eps': float(args.uschad_norm_eps),
        'uschad_train_subjects': canonical_subject_ids(
            args.uschad_train_subjects
        ),
        'offline_val_subjects': canonical_subject_ids(args.offline_val_subjects),
        'uschad_test_subjects': canonical_subject_ids(args.uschad_test_subjects),
        'uschad_cv_fold': int(args.uschad_cv_fold),
        'har_in_channels': int(args.har_in_channels),
        'har_feat_dim': int(args.har_feat_dim),
        'har_base_channels': int(args.har_base_channels),
        'har_dropout': float(args.har_dropout),
        'har_aug_mode': str(args.har_aug_mode),
        'har_weak_jitter_std': float(args.har_weak_jitter_std),
        'har_weak_scale_std': float(args.har_weak_scale_std),
        'har_strong_jitter_std': float(args.har_strong_jitter_std),
        'har_strong_scale_std': float(args.har_strong_scale_std),
        'har_time_mask_ratio': float(args.har_time_mask_ratio),
        'projection_hidden_dim': int(args.projection_hidden_dim),
        'projection_bottleneck_dim': int(args.projection_bottleneck_dim),
    }
    if args.uschad_sample_unit == 'trial':
        metadata.update({
            'trial_pooling': str(args.trial_pooling),
            'trial_view_mode': str(args.trial_view_mode),
            'trial_crop_ratio': float(args.trial_crop_ratio),
            'trial_min_windows': int(args.trial_min_windows),
        })
        if args.trial_pooling == 'mean_robust_max':
            metadata.update({
                'trial_robust_max_quantile': float(
                    args.trial_robust_max_quantile
                ),
                'trial_pool_fusion_dim': int(args.trial_pool_fusion_dim),
                'trial_pool_fusion_dropout': float(
                    args.trial_pool_fusion_dropout
                ),
            })
        elif args.trial_pooling == 'gated_attention':
            metadata.update({
                'trial_attention_dim': int(args.trial_attention_dim),
                'trial_attention_dropout': float(args.trial_attention_dropout),
                'trial_attention_temperature': float(
                    args.trial_attention_temperature
                ),
                'trial_attention_mean_mix': float(args.trial_attention_mean_mix),
            })
    return metadata


def validate_checkpoint_metadata(checkpoint, args, checkpoint_path):
    expected = experiment_metadata(args)
    observed = checkpoint.get('experiment_metadata')
    if observed is None:
        if args.uschad_sample_unit == 'trial':
            raise RuntimeError(
                f"Trial checkpoint lacks experiment_metadata: {checkpoint_path}. "
                "Use a checkpoint produced by the trial-aware implementation."
            )
        args.logger.warning(
            f"Legacy checkpoint has no experiment_metadata: {checkpoint_path}."
        )
        return

    keys = ['uschad_window_size', 'uschad_sample_unit']
    new_structural_keys = [
        'har_in_channels',
        'har_feat_dim',
        'har_base_channels',
        'har_dropout',
        'projection_hidden_dim',
        'projection_bottleneck_dim',
    ]
    if args.uschad_sample_unit == 'trial':
        # Trial checkpoints are only produced by this implementation, so every
        # behavior-changing field must be present and equal.
        keys.extend([
            'trial_pooling',
            'trial_view_mode',
            'trial_crop_ratio',
            'trial_min_windows',
        ])
        if args.trial_pooling == 'mean_robust_max':
            keys.extend([
                'trial_robust_max_quantile',
                'trial_pool_fusion_dim',
                'trial_pool_fusion_dropout',
            ])
        elif args.trial_pooling == 'gated_attention':
            keys.extend([
                'trial_attention_dim',
                'trial_attention_dropout',
                'trial_attention_temperature',
                'trial_attention_mean_mix',
            ])
        keys.extend(new_structural_keys)
    else:
        # Preserve compatibility with pre-trial window checkpoints. If a newer
        # field is recorded, still verify it instead of silently ignoring it.
        keys.extend(key for key in new_structural_keys if key in observed)
    normalization_keys = [
        'uschad_recompute_norm_from_train_subjects',
    ]
    recompute_expected = bool(args.uschad_recompute_norm_from_train_subjects)
    recompute_observed = bool(
        observed.get('uschad_recompute_norm_from_train_subjects', False)
    )
    if recompute_expected or recompute_observed:
        normalization_keys.extend([
            'uschad_split_mode',
            'uschad_norm_eps',
            'uschad_train_subjects',
            'offline_val_subjects',
            'uschad_test_subjects',
            'uschad_cv_fold',
        ])
    keys.extend(
        key for key in normalization_keys
        if recompute_expected or key in observed
    )
    mismatches = []
    for key in keys:
        if key not in observed:
            mismatches.append(f"{key}=<missing> expected {expected[key]!r}")
            continue
        observed_value = observed[key]
        expected_value = expected[key]
        if isinstance(expected_value, float):
            equal = math.isclose(
                float(observed_value), float(expected_value), rel_tol=0, abs_tol=1e-12
            )
        else:
            equal = observed_value == expected_value
        if not equal:
            mismatches.append(
                f"{key}={observed_value!r} expected {expected_value!r}"
            )
    if mismatches:
        raise RuntimeError(
            f"Checkpoint/data configuration mismatch for {checkpoint_path}: "
            + "; ".join(mismatches)
        )
    args.logger.info(f"Checkpoint metadata verified: {checkpoint_path}")


def prediction_diagnostics(model, logits, args):
    temperature = float(args.trial_confidence_temperature)
    probabilities = torch.softmax(logits / temperature, dim=1)
    confidence = probabilities.max(dim=1).values
    normalizer = math.log(max(2, probabilities.size(1)))
    predictive_entropy = -torch.sum(
        probabilities * torch.log(probabilities.clamp_min(1e-12)), dim=1
    ) / normalizer
    diagnostics = {
        'confidence_sum': float(confidence.sum().item()),
        'predictive_entropy_sum': float(predictive_entropy.sum().item()),
        'count': int(len(logits)),
        'temperature': temperature,
    }

    if (
        args.uschad_sample_unit == 'trial'
        and args.trial_pooling == 'gated_attention'
    ):
        weights = model[0].last_attention_weights
        mask = model[0].last_attention_mask
        if weights is None or mask is None:
            raise RuntimeError("Trial attention diagnostics were not produced.")
        lengths = mask.sum(dim=1).to(weights.dtype)
        raw_entropy = -torch.sum(
            weights * torch.log(weights.clamp_min(1e-12)), dim=1
        )
        attention_entropy = torch.where(
            lengths > 1,
            raw_entropy / torch.log(lengths.clamp_min(2.0)),
            torch.ones_like(raw_entropy),
        )
        effective_windows = 1.0 / torch.sum(weights.pow(2), dim=1).clamp_min(1e-12)
        diagnostics['attention_entropy_sum'] = float(attention_entropy.sum().item())
        diagnostics['effective_windows_sum'] = float(effective_windows.sum().item())
    return diagnostics


def merge_prediction_diagnostics(records):
    total = sum(record['count'] for record in records)
    if total <= 0:
        raise RuntimeError("No prediction diagnostics were collected.")
    merged = {
        'confidence_temperature': float(records[0].get('temperature', 1.0)),
        'mean_confidence': sum(record['confidence_sum'] for record in records) / total,
        'mean_predictive_entropy': (
            sum(record['predictive_entropy_sum'] for record in records) / total
        ),
    }
    if 'attention_entropy_sum' in records[0]:
        merged['mean_attention_entropy'] = (
            sum(record['attention_entropy_sum'] for record in records) / total
        )
        merged['mean_effective_windows'] = (
            sum(record['effective_windows_sum'] for record in records) / total
        )
    return merged


def online_checkpoint_path(args, session_number):
    """Return a checkpoint name that reflects how the epoch was selected."""
    suffix = (
        '_final.pt'
        if args.online_checkpoint_selection == 'final_epoch'
        else '_best.pt'
    )
    return args.model_path[:-3] + f'_session-{int(session_number)}{suffix}'


def derive_cgcd_session_config(args):
    """
    Derive continual_session_num from num_novel_classes_per_session.

    CGCD setting:
        - args.num_old_classes controls initial old classes.
        - args.num_novel_classes_per_session controls how many new classes appear
          in each online session.
        - args.continual_session_num is derived automatically.

    Strict rule:
        total novel classes must be divisible by num_novel_classes_per_session.
    """

    num_old_classes = int(args.num_labeled_classes)
    num_novel_classes = int(args.num_unlabeled_classes)
    num_novel_classes_per_session = int(args.num_novel_classes_per_session)

    if num_old_classes <= 0:
        raise ValueError(
            f"num_old_classes / num_labeled_classes must be positive, "
            f"but got {num_old_classes}."
        )

    if num_novel_classes <= 0:
        raise ValueError(
            f"num_unlabeled_classes must be positive for CGCD online sessions, "
            f"but got {num_novel_classes}."
        )

    if num_novel_classes_per_session < 1:
        raise ValueError(
            f"--num_novel_classes_per_session must be at least 1, "
            f"but got {num_novel_classes_per_session}."
        )

    if num_novel_classes % num_novel_classes_per_session != 0:
        raise ValueError(
            f"Invalid CGCD session config: total novel classes "
            f"({num_novel_classes}) cannot be divided by "
            f"--num_novel_classes_per_session ({num_novel_classes_per_session}). "
            f"Please choose a divisor of {num_novel_classes}. "
            f"For example, if total novel classes is 6, valid values are 1, 2, 3, or 6."
        )

    derived_continual_session_num = num_novel_classes // num_novel_classes_per_session

    user_continual_session_num = int(getattr(args, "continual_session_num", -1))

    if user_continual_session_num > 0 and user_continual_session_num != derived_continual_session_num:
        print(
            "[CGCD SESSION CONFIG WARNING] "
            f"--continual_session_num={user_continual_session_num} is ignored. "
            f"Derived continual_session_num={derived_continual_session_num} from "
            f"num_unlabeled_classes={num_novel_classes} and "
            f"num_novel_classes_per_session={num_novel_classes_per_session}."
        )

    args.continual_session_num = derived_continual_session_num

    # Keep the old singular name because the existing online code uses it.
    args.num_novel_class_per_session = num_novel_classes_per_session

    return args


def validate_training_args(args):
    """Fail early for configurations that make loss or validation behavior ambiguous."""
    if args.epochs_offline < 1 or args.epochs_online_per_session < 1:
        raise ValueError(
            "--epochs_offline and --epochs_online_per_session must both be positive."
        )
    if args.lr <= 0:
        raise ValueError(f"--lr must be positive, got {args.lr}.")
    if args.har_in_channels < 1 or args.har_feat_dim < 1 or args.har_base_channels < 1:
        raise ValueError(
            "--har_in_channels, --har_feat_dim, and --har_base_channels must be positive."
        )
    if not 0.0 <= args.har_dropout < 1.0:
        raise ValueError("--har_dropout must be in [0, 1).")
    augmentation_stds = {
        '--har_weak_jitter_std': args.har_weak_jitter_std,
        '--har_weak_scale_std': args.har_weak_scale_std,
        '--har_strong_jitter_std': args.har_strong_jitter_std,
        '--har_strong_scale_std': args.har_strong_scale_std,
    }
    negative_augmentation_stds = {
        name: value for name, value in augmentation_stds.items() if value < 0
    }
    if negative_augmentation_stds:
        raise ValueError(
            "HAR jitter/scale standard deviations must be non-negative: "
            f"{negative_augmentation_stds}."
        )
    if not 0.0 <= args.har_time_mask_ratio <= 1.0:
        raise ValueError("--har_time_mask_ratio must be in [0, 1].")
    if args.temperature <= 0:
        raise ValueError(f"--temperature must be positive, got {args.temperature}.")
    if args.teacher_temp <= 0 or args.warmup_teacher_temp <= 0:
        raise ValueError(
            "--teacher_temp and --warmup_teacher_temp must both be positive."
        )
    if args.warmup_teacher_temp_epochs < 0:
        raise ValueError("--warmup_teacher_temp_epochs must be non-negative.")
    active_epochs = (
        args.epochs_offline
        if args.train_session == 'offline'
        else args.epochs_online_per_session
    )
    if args.warmup_teacher_temp_epochs > active_epochs:
        raise ValueError(
            "--warmup_teacher_temp_epochs cannot exceed the active training epochs; "
            "otherwise DistillLoss has an invalid schedule."
        )
    if args.offline_early_stop_patience < 0:
        raise ValueError("--offline_early_stop_patience must be non-negative.")
    if args.projection_hidden_dim < 1 or args.projection_bottleneck_dim < 1:
        raise ValueError(
            "--projection_hidden_dim and --projection_bottleneck_dim must be positive."
        )
    if args.offline_bn_freeze_epoch > args.epochs_offline:
        raise ValueError(
            "--offline_bn_freeze_epoch cannot exceed --epochs_offline."
        )
    if args.offline_lr_drop_epoch > args.epochs_offline:
        raise ValueError(
            "--offline_lr_drop_epoch cannot exceed --epochs_offline."
        )
    if args.offline_lr_drop_epoch > 0 and not 0.0 < args.offline_lr_drop_factor < 1.0:
        raise ValueError(
            "--offline_lr_drop_factor must be between 0 and 1 when LR drop is enabled."
        )
    aux_start_enabled = args.offline_aux_decay_start_epoch > 0
    aux_end_enabled = args.offline_aux_decay_end_epoch > 0
    if aux_start_enabled != aux_end_enabled:
        raise ValueError(
            "Set both --offline_aux_decay_start_epoch and "
            "--offline_aux_decay_end_epoch, or disable both."
        )
    if aux_start_enabled:
        if args.offline_aux_decay_end_epoch <= args.offline_aux_decay_start_epoch:
            raise ValueError(
                "--offline_aux_decay_end_epoch must be greater than "
                "--offline_aux_decay_start_epoch."
            )
        if args.offline_aux_decay_end_epoch > args.epochs_offline:
            raise ValueError(
                "--offline_aux_decay_end_epoch cannot exceed --epochs_offline."
            )
        if not 0.0 <= args.offline_aux_final_scale <= 1.0:
            raise ValueError(
                "--offline_aux_final_scale must be in [0, 1]."
            )
    diagnostic_epochs = parse_epoch_numbers(args.offline_gradient_diagnostic_epochs)
    if diagnostic_epochs and max(diagnostic_epochs) > args.epochs_offline:
        raise ValueError(
            "--offline_gradient_diagnostic_epochs contains an epoch greater than "
            "--epochs_offline."
        )
    if args.n_views != 2:
        raise ValueError(
            "This USC-HAD Happy implementation currently requires --n_views 2; "
            "the offline losses explicitly pair two views."
        )
    if args.batch_size < 2:
        raise ValueError("Contrastive training requires --batch_size at least 2.")
    if args.eval_batch_size < 1:
        raise ValueError("--eval_batch_size must be positive.")
    if args.num_workers < 0 or args.num_workers_test < 0:
        raise ValueError("DataLoader worker counts must be non-negative.")
    if args.uschad_sample_unit == 'trial':
        if not 0.0 < args.trial_crop_ratio <= 1.0:
            raise ValueError("--trial_crop_ratio must be in (0, 1].")
        if args.trial_min_windows < 1:
            raise ValueError("--trial_min_windows must be at least 1.")
        if args.trial_pooling == 'mean_robust_max':
            if not 0.5 <= args.trial_robust_max_quantile < 1.0:
                raise ValueError(
                    "--trial_robust_max_quantile must be in [0.5, 1.0)."
                )
            if args.trial_pool_fusion_dim < 1:
                raise ValueError("--trial_pool_fusion_dim must be positive.")
            if not 0.0 <= args.trial_pool_fusion_dropout < 1.0:
                raise ValueError(
                    "--trial_pool_fusion_dropout must be in [0, 1)."
                )
        elif args.trial_pooling == 'gated_attention':
            if args.trial_attention_dim < 1:
                raise ValueError("--trial_attention_dim must be positive.")
            if not 0.0 <= args.trial_attention_dropout < 1.0:
                raise ValueError("--trial_attention_dropout must be in [0, 1).")
            if args.trial_attention_temperature <= 0:
                raise ValueError("--trial_attention_temperature must be positive.")
            if not 0.0 <= args.trial_attention_mean_mix <= 1.0:
                raise ValueError("--trial_attention_mean_mix must be in [0, 1].")
        if not 0.0 < args.trial_encoder_lr_scale <= 1.0:
            raise ValueError("--trial_encoder_lr_scale must be in (0, 1].")
        if args.trial_encoder_freeze_epochs < 0:
            raise ValueError("--trial_encoder_freeze_epochs must be non-negative.")
        if args.trial_encoder_freeze_epochs > args.epochs_offline:
            raise ValueError(
                "--trial_encoder_freeze_epochs cannot exceed --epochs_offline."
            )
        if (
            args.train_session == 'offline'
            and args.trial_encoder_freeze_epochs > 0
            and str(args.trial_encoder_checkpoint).strip() == ''
        ):
            raise ValueError(
                "Freezing the trial encoder requires --trial_encoder_checkpoint; "
                "otherwise a random encoder would be frozen."
            )
        trial_counts = [
            args.online_old_seen_trials,
            args.online_novel_unseen_trials,
            args.online_novel_seen_trials,
        ]
        if any(count < 1 for count in trial_counts):
            raise ValueError(
                "All --online_*_trials counts must be positive in trial mode."
            )
    else:
        window_counts = [
            args.online_old_seen_num,
            args.online_novel_unseen_num,
            args.online_novel_seen_num,
        ]
        if any(count < 1 for count in window_counts):
            raise ValueError("All --online_*_num window counts must be positive.")
        if str(args.trial_encoder_checkpoint).strip() != '':
            raise ValueError(
                "--trial_encoder_checkpoint is only valid with --uschad_sample_unit trial."
            )
    if args.trial_confidence_temperature <= 0:
        raise ValueError("--trial_confidence_temperature must be positive.")
    if not math.isfinite(args.uschad_norm_eps) or args.uschad_norm_eps <= 0:
        raise ValueError("--uschad_norm_eps must be finite and positive.")
    if args.uschad_cv_fold < -1:
        raise ValueError("--uschad_cv_fold must be -1 or a non-negative fold id.")
    if args.uschad_recompute_norm_from_train_subjects:
        if args.uschad_split_mode != 'subject':
            raise ValueError(
                "--uschad_recompute_norm_from_train_subjects requires "
                "--uschad_split_mode subject."
            )
        split_subjects = {
            'train': set(canonical_subject_ids(args.uschad_train_subjects)),
            'validation': set(canonical_subject_ids(args.offline_val_subjects)),
            'test': set(canonical_subject_ids(args.uschad_test_subjects)),
        }
        if not split_subjects['train']:
            raise ValueError(
                "Fold-specific normalization requires explicit "
                "--uschad_train_subjects."
            )
        if not split_subjects['test']:
            raise ValueError(
                "Fold-specific normalization requires explicit "
                "--uschad_test_subjects so validation subjects are not absorbed "
                "into the test split."
            )
        for left_name, right_name in [
            ('train', 'validation'),
            ('train', 'test'),
            ('validation', 'test'),
        ]:
            overlap = split_subjects[left_name] & split_subjects[right_name]
            if overlap:
                raise ValueError(
                    "USC-HAD fold subject overlap: "
                    f"{left_name}/{right_name}={sorted(overlap)}."
                )
    if args.online_checkpoint_selection == 'auto':
        args.online_checkpoint_selection = (
            'final_epoch' if args.uschad_sample_unit == 'trial' else 'test_best'
        )
    if args.train_session == 'offline' and str(args.offline_val_subjects).strip() == '':
        raise ValueError(
            "Offline training now requires --offline_val_subjects so model selection "
            "does not repeatedly use the final test set."
        )
    return args


def compute_classification_metrics(y_true, y_pred, num_classes):
    """Compute reproducible closed-set classification metrics from predictions."""
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    num_classes = int(num_classes)

    if y_true.ndim != 1 or y_pred.ndim != 1 or len(y_true) != len(y_pred):
        raise ValueError(
            "Classification metrics require equally sized 1D y_true/y_pred arrays, "
            f"but got {y_true.shape} and {y_pred.shape}."
        )
    if len(y_true) == 0:
        raise ValueError("Classification metrics cannot be computed from an empty test set.")
    if num_classes <= 0:
        raise ValueError(f"num_classes must be positive, got {num_classes}.")

    labels = np.arange(num_classes, dtype=np.int64)
    confusion = np.zeros((num_classes, num_classes), dtype=np.int64)
    valid = (
        (y_true >= 0) & (y_true < num_classes) &
        (y_pred >= 0) & (y_pred < num_classes)
    )
    if not np.all(valid):
        invalid_count = int((~valid).sum())
        raise ValueError(
            f"Found {invalid_count} predictions/targets outside [0, {num_classes - 1}] "
            "while computing classification metrics."
        )

    np.add.at(confusion, (y_true, y_pred), 1)
    support = confusion.sum(axis=1)
    predicted = confusion.sum(axis=0)
    true_positive = np.diag(confusion).astype(np.float64)

    per_class_accuracy = np.divide(
        true_positive,
        support,
        out=np.zeros(num_classes, dtype=np.float64),
        where=support > 0,
    )
    f1_denominator = support + predicted
    per_class_f1 = np.divide(
        2.0 * true_positive,
        f1_denominator,
        out=np.zeros(num_classes, dtype=np.float64),
        where=f1_denominator > 0,
    )
    present_classes = support > 0

    return {
        "num_samples": int(len(y_true)),
        "labels": labels.tolist(),
        "overall_accuracy": float(true_positive.sum() / len(y_true)),
        "mean_class_accuracy": float(per_class_accuracy[present_classes].mean()),
        "macro_f1": float(per_class_f1[present_classes].mean()),
        "per_class_accuracy": per_class_accuracy.tolist(),
        "per_class_f1": per_class_f1.tolist(),
        "support": support.astype(np.int64).tolist(),
        "confusion_matrix": confusion.tolist(),
    }


def align_cluster_predictions(y_true, y_pred):
    """Map arbitrary cluster ids to semantic labels with the GCD ACC mapping."""
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)

    if y_true.ndim != 1 or y_pred.ndim != 1 or len(y_true) != len(y_pred):
        raise ValueError(
            "Hungarian alignment requires equally sized 1D y_true/y_pred arrays, "
            f"but got {y_true.shape} and {y_pred.shape}."
        )
    if len(y_true) == 0:
        raise ValueError("Hungarian alignment cannot be computed from an empty test set.")

    _, assignment, _ = cluster_acc(y_true, y_pred, return_ind=True)
    pred_to_true = {
        int(predicted_label): int(true_label)
        for predicted_label, true_label in assignment
    }
    missing_predicted_labels = sorted(set(y_pred.tolist()) - set(pred_to_true))
    if missing_predicted_labels:
        raise RuntimeError(
            "Hungarian assignment did not cover predicted labels: "
            f"{missing_predicted_labels}"
        )

    aligned_pred = np.asarray(
        [pred_to_true[int(predicted_label)] for predicted_label in y_pred],
        dtype=np.int64,
    )
    assignment_pairs = [
        [int(predicted_label), int(true_label)]
        for predicted_label, true_label in assignment
    ]
    return aligned_pred, assignment_pairs


def log_and_save_classification_metrics(metrics, phase_name, args):
    """Log and persist metrics associated with the configured checkpoint policy."""
    args.logger.info(
        f"[{phase_name} Classification Metrics] samples={metrics['num_samples']} | "
        f"overall_accuracy={metrics['overall_accuracy']:.4f} | "
        f"mean_class_accuracy={metrics['mean_class_accuracy']:.4f} | "
        f"macro_f1={metrics['macro_f1']:.4f}"
    )
    args.logger.info(
        f"[{phase_name} Classification Metrics] labels={metrics['labels']} | "
        f"support={metrics['support']}"
    )
    diagnostics = metrics.get("prediction_diagnostics")
    if diagnostics is not None:
        diagnostic_text = (
            f"mean_confidence={diagnostics['mean_confidence']:.4f} | "
            f"mean_predictive_entropy={diagnostics['mean_predictive_entropy']:.4f}"
        )
        if "mean_attention_entropy" in diagnostics:
            diagnostic_text += (
                f" | mean_attention_entropy={diagnostics['mean_attention_entropy']:.4f}"
                f" | mean_effective_windows={diagnostics['mean_effective_windows']:.4f}"
            )
        args.logger.info(f"[{phase_name} Diagnostics] {diagnostic_text}")
    matrix_rows = [" ".join(str(value) for value in row) for row in metrics["confusion_matrix"]]
    args.logger.info(
        f"[{phase_name} Confusion Matrix] rows=true, cols=pred\n" + "\n".join(matrix_rows)
    )

    metrics_path = os.path.join(
        args.log_dir,
        phase_name.lower().replace(" ", "_") + "_classification_metrics.json",
    )
    with open(metrics_path, "w", encoding="utf-8") as handle:
        json.dump(metrics, handle, ensure_ascii=False, indent=2)
    args.logger.info(f"[{phase_name} Classification Metrics] saved to {metrics_path}")
    return metrics_path
'''offline train and test'''
'''====================================================================================================================='''
def configure_backbone_trainable(args, backbone):
    """
    Configure trainable parameters for USC-HAD ResNet1D.

    Current branch:
        train all ResNet1D parameters.

    Reason:
        USC-HAD has no DINO-pretrained ViT.
        ResNet1D is trained from scratch.
    """

    args.logger.info("USC-HAD ResNet1D backbone: train all parameters.")

    for param in backbone.parameters():
        param.requires_grad = True

    trainable_params = sum(p.numel() for p in backbone.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in backbone.parameters())

    args.logger.info(
        f"Backbone trainable params: {trainable_params} / {total_params}"
    )

    return backbone
def resolve_offline_loss_weights(args):
    """Resolve explicit offline loss weights while preserving legacy defaults."""
    explicit = {
        "cls": args.offline_cls_weight,
        "cluster": args.offline_cluster_weight,
        "contrast": args.offline_contrast_weight,
        "supcon": args.offline_supcon_weight,
    }
    supplied = [value is not None for value in explicit.values()]
    if any(supplied) and not all(supplied):
        missing = [name for name, value in explicit.items() if value is None]
        raise ValueError(
            "Set all four explicit offline loss weights together; missing "
            f"{missing}. This avoids an accidental mixed legacy/new configuration."
        )
    if not any(supplied):
        explicit = {
            "cls": float(args.sup_weight),
            "cluster": 1.0 - float(args.sup_weight),
            "contrast": 1.0 - float(args.sup_weight),
            "supcon": float(args.sup_weight),
        }
    if any(value < 0 for value in explicit.values()):
        raise ValueError(f"Offline loss weights must be non-negative, got {explicit}.")
    if sum(explicit.values()) <= 0:
        raise ValueError("At least one offline loss weight must be positive.")
    return explicit


def parse_epoch_numbers(value):
    """Parse a comma/space separated list of 1-based epoch numbers."""
    if value is None or str(value).strip() == "":
        return set()
    tokens = str(value).replace(",", " ").split()
    epochs = {int(token) for token in tokens}
    if any(epoch < 1 for epoch in epochs):
        raise ValueError(
            "Epoch lists use 1-based positive integers, got "
            f"{sorted(epochs)}."
        )
    return epochs


def auxiliary_loss_scale(epoch_number, start_epoch, end_epoch, final_scale):
    """Return the late-training scale for cluster and InfoNCE losses."""
    if start_epoch <= 0 and end_epoch <= 0:
        return 1.0
    if epoch_number <= start_epoch:
        return 1.0
    if epoch_number >= end_epoch:
        return float(final_scale)
    progress = (epoch_number - start_epoch) / float(end_epoch - start_epoch)
    return 1.0 + progress * (float(final_scale) - 1.0)


def effective_offline_loss_weights(base_weights, epoch_number, args):
    """Apply optional late decay to cluster and InfoNCE weights."""
    scale = auxiliary_loss_scale(
        epoch_number,
        args.offline_aux_decay_start_epoch,
        args.offline_aux_decay_end_epoch,
        args.offline_aux_final_scale,
    )
    weights = dict(base_weights)
    weights["cluster"] *= scale
    weights["contrast"] *= scale
    return weights, scale


def freeze_batchnorm_running_stats(model):
    """Freeze BatchNorm running statistics while leaving affine parameters trainable."""
    count = 0
    for module in model.modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            module.eval()
            count += 1
    return count


def maybe_apply_offline_lr_drop(optimizer, scheduler, epoch_number, args, applied):
    """Scale the live cosine trajectory once at the configured 1-based epoch."""
    drop_epoch = int(args.offline_lr_drop_epoch)
    if applied or drop_epoch <= 0 or epoch_number != drop_epoch:
        return applied

    factor = float(args.offline_lr_drop_factor)
    for group in optimizer.param_groups:
        group["lr"] *= factor
        if "initial_lr" in group:
            group["initial_lr"] *= factor
    scheduler.base_lrs = [lr * factor for lr in scheduler.base_lrs]
    scheduler._last_lr = [group["lr"] for group in optimizer.param_groups]
    return True


def _gradient_norm(gradients):
    squared_norm = sum(
        float(gradient.detach().float().pow(2).sum().item())
        for gradient in gradients
        if gradient is not None
    )
    return math.sqrt(squared_norm)


def _gradient_cosine(gradients_a, gradients_b):
    norm_a = _gradient_norm(gradients_a)
    norm_b = _gradient_norm(gradients_b)
    if norm_a == 0.0 or norm_b == 0.0:
        return None
    dot = sum(
        float((gradient_a.detach().float() * gradient_b.detach().float()).sum().item())
        for gradient_a, gradient_b in zip(gradients_a, gradients_b)
        if gradient_a is not None and gradient_b is not None
    )
    return dot / (norm_a * norm_b)


def compute_backbone_gradient_diagnostics(losses, backbone, weights):
    """Measure each loss gradient on the backbone without changing ``.grad``."""
    parameters = [parameter for parameter in backbone.parameters() if parameter.requires_grad]
    gradients = {}
    for name, component_loss in losses.items():
        gradients[name] = torch.autograd.grad(
            component_loss,
            parameters,
            retain_graph=True,
            allow_unused=True,
        )

    raw_norms = {name: _gradient_norm(value) for name, value in gradients.items()}
    weighted_norms = {
        name: abs(float(weights[name])) * raw_norm
        for name, raw_norm in raw_norms.items()
    }
    names = list(losses)
    pairwise_cosines = {}
    for index, name_a in enumerate(names):
        for name_b in names[index + 1:]:
            pairwise_cosines[f"{name_a}__{name_b}"] = _gradient_cosine(
                gradients[name_a], gradients[name_b]
            )
    return {
        "raw_gradient_norms": raw_norms,
        "weighted_gradient_norms": weighted_norms,
        "pairwise_gradient_cosines": pairwise_cosines,
    }


def selection_value(metrics, metric_name):
    if metric_name not in metrics:
        raise KeyError(
            f"Selection metric {metric_name!r} is unavailable. "
            f"Available metrics: {sorted(metrics.keys())}"
        )
    return float(metrics[metric_name])


def train_offline(student, train_loader, val_loader, test_loader, args):

    optimizer = build_sgd_optimizer(student, args)

    exp_lr_scheduler = build_cosine_lr_scheduler(
        optimizer,
        args.epochs_offline,
    )

    cluster_criterion = DistillLoss(
                        args.warmup_teacher_temp_epochs,
                        args.epochs_offline,
                        args.n_views,
                        args.warmup_teacher_temp,
                        args.teacher_temp,
                        student_temp=args.temperature,
                    )
    base_loss_weights = resolve_offline_loss_weights(args)
    args.logger.info(
        "Offline base loss weights: "
        f"cls={base_loss_weights['cls']}, cluster={base_loss_weights['cluster']}, "
        f"contrast={base_loss_weights['contrast']}, supcon={base_loss_weights['supcon']}"
    )
    args.logger.info(
        f"Offline selection: validation {args.offline_selection_metric}; "
        "the final test set is evaluated once after checkpoint selection."
    )

    best_val_score = -float("inf")
    best_epoch = None
    best_val_metrics = None
    best_checkpoint_path = args.model_path[:-3] + "_best.pt"
    epoch_metrics_path = os.path.join(args.log_dir, "offline_epoch_metrics.jsonl")
    gradient_metrics_path = os.path.join(
        args.log_dir, "offline_gradient_diagnostics.jsonl"
    )
    gradient_diagnostic_epochs = parse_epoch_numbers(
        args.offline_gradient_diagnostic_epochs
    )
    lr_drop_applied = False

    for epoch in range(args.epochs_offline):
        epoch_number = epoch + 1
        lr_drop_was_applied = lr_drop_applied
        lr_drop_applied = maybe_apply_offline_lr_drop(
            optimizer,
            exp_lr_scheduler,
            epoch_number,
            args,
            lr_drop_applied,
        )
        if lr_drop_applied and not lr_drop_was_applied:
            args.logger.info(
                f"Applied offline LR drop at epoch {epoch_number}: "
                f"factor={args.offline_lr_drop_factor}, "
                f"lr={optimizer.param_groups[0]['lr']:.8g}."
            )

        loss_weights, aux_loss_scale = effective_offline_loss_weights(
            base_loss_weights, epoch_number, args
        )
        loss_record = AverageMeter()
        cls_loss_record = AverageMeter()
        cluster_loss_record = AverageMeter()
        contrastive_loss_record = AverageMeter()
        sup_con_loss_record = AverageMeter()
        train_acc_record = AverageMeter()

        student.train()
        trial_encoder_frozen = configure_trial_encoder_for_epoch(
            student, epoch_number, args
        )
        if args.uschad_sample_unit == 'trial' and (
            epoch_number == 1
            or epoch_number == args.trial_encoder_freeze_epochs + 1
        ):
            args.logger.info(
                f"Trial window encoder frozen={trial_encoder_frozen} at "
                f"offline epoch {epoch_number}."
            )
        bn_running_stats_frozen = (
            args.offline_bn_freeze_epoch > 0
            and epoch_number >= args.offline_bn_freeze_epoch
        )
        if bn_running_stats_frozen:
            frozen_bn_count = freeze_batchnorm_running_stats(student)
            if epoch_number == args.offline_bn_freeze_epoch:
                args.logger.info(
                    f"Froze running statistics for {frozen_bn_count} BatchNorm modules "
                    f"at epoch {epoch_number}; affine parameters remain trainable."
                )
        args.logger.info(
            f"Epoch {epoch_number} effective loss weights: "
            f"cls={loss_weights['cls']:.6g}, cluster={loss_weights['cluster']:.6g}, "
            f"contrast={loss_weights['contrast']:.6g}, "
            f"supcon={loss_weights['supcon']:.6g}, aux_scale={aux_loss_scale:.6g}."
        )
        for batch_idx, batch in enumerate(train_loader):

            images, class_labels, uq_idxs = batch   # NOTE!!! no mask lab in this setting
            mask_lab = torch.ones_like(class_labels)   # NOTE!!! all samples are labeled

            class_labels, mask_lab = class_labels.cuda(non_blocking=True), mask_lab.cuda(non_blocking=True).bool()
            prepared_inputs = prepare_model_inputs(images, args)
            _, student_proj, student_out = forward_prepared_model(
                student, prepared_inputs
            )
            teacher_out = student_out.detach()

            # clustering, sup
            sup_logits = torch.cat(
                [f[mask_lab] for f in (student_out / args.temperature).chunk(2)], dim=0
            )
            sup_labels = torch.cat([class_labels[mask_lab] for _ in range(2)], dim=0)
            cls_loss = nn.CrossEntropyLoss()(sup_logits, sup_labels)

            # clustering, unsup
            cluster_loss = cluster_criterion(student_out, teacher_out, epoch)
            avg_probs = (student_out / args.temperature).softmax(dim=1).mean(dim=0)
            me_max_loss = - torch.sum(torch.log(avg_probs**(-avg_probs))) + math.log(float(len(avg_probs)))
            cluster_loss += args.memax_weight * me_max_loss

            # represent learning, unsup
            contrastive_logits, contrastive_labels = info_nce_logits(features=student_proj)
            contrastive_loss = torch.nn.CrossEntropyLoss()(contrastive_logits, contrastive_labels)

            # representation learning, sup
            student_proj = torch.cat([f[mask_lab].unsqueeze(1) for f in student_proj.chunk(2)], dim=1)
            student_proj = torch.nn.functional.normalize(student_proj, dim=-1)
            sup_con_labels = class_labels[mask_lab]
            sup_con_loss = SupConLoss()(student_proj, labels=sup_con_labels)

            # Total loss
            loss = (
                loss_weights["cluster"] * cluster_loss
                + loss_weights["cls"] * cls_loss
                + loss_weights["contrast"] * contrastive_loss
                + loss_weights["supcon"] * sup_con_loss
            )

            if batch_idx == 0 and epoch_number in gradient_diagnostic_epochs:
                gradient_record = compute_backbone_gradient_diagnostics(
                    {
                        "cls": cls_loss,
                        "cluster": cluster_loss,
                        "contrast": contrastive_loss,
                        "supcon": sup_con_loss,
                    },
                    student[0],
                    loss_weights,
                )
                gradient_record.update(
                    {
                        "epoch": epoch_number,
                        "batch_index": batch_idx,
                        "loss_values": {
                            "cls": float(cls_loss.item()),
                            "cluster": float(cluster_loss.item()),
                            "contrast": float(contrastive_loss.item()),
                            "supcon": float(sup_con_loss.item()),
                        },
                        "effective_loss_weights": dict(loss_weights),
                        "auxiliary_loss_scale": float(aux_loss_scale),
                    }
                )
                with open(gradient_metrics_path, "a", encoding="utf-8") as handle:
                    handle.write(json.dumps(gradient_record, ensure_ascii=False) + "\n")
                args.logger.info(
                    "Backbone gradient diagnostics: "
                    + json.dumps(gradient_record, ensure_ascii=False)
                )

            # logs
            pstr = ''
            pstr += f'cls_loss: {cls_loss.item():.4f} '
            pstr += f'cluster_loss: {cluster_loss.item():.4f} '
            pstr += f'sup_con_loss: {sup_con_loss.item():.4f} '
            pstr += f'contrastive_loss: {contrastive_loss.item():.4f} '

            loss_record.update(loss.item(), class_labels.size(0))
            cls_loss_record.update(cls_loss.item(), class_labels.size(0))
            cluster_loss_record.update(cluster_loss.item(), class_labels.size(0))
            contrastive_loss_record.update(contrastive_loss.item(), class_labels.size(0))
            sup_con_loss_record.update(sup_con_loss.item(), class_labels.size(0))
            train_acc_record.update(
                (sup_logits.argmax(dim=1) == sup_labels).float().mean().item(),
                sup_labels.size(0),
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if batch_idx % args.print_freq == 0:
                args.logger.info('Epoch: [{}][{}/{}]\t loss {:.5f}\t {}'
                            .format(epoch, batch_idx, len(train_loader), loss.item(), pstr))

        args.logger.info(
            "Train Epoch: {} Avg Loss: {:.4f} | train_sup_acc: {:.4f} | "
            "cls: {:.4f} cluster: {:.4f} contrast: {:.4f} supcon: {:.4f}".format(
                epoch,
                loss_record.avg,
                train_acc_record.avg,
                cls_loss_record.avg,
                cluster_loss_record.avg,
                contrastive_loss_record.avg,
                sup_con_loss_record.avg,
            )
        )

        args.logger.info('Evaluating validation set for checkpoint selection...')
        _, _, _, val_metrics = test_offline(
            student, val_loader, epoch=epoch, save_name='Validation ACC', args=args
        )
        val_score = selection_value(val_metrics, args.offline_selection_metric)
        args.logger.info(
            f"Validation {args.offline_selection_metric}: {val_score:.4f} | "
            f"overall={val_metrics['overall_accuracy']:.4f} | "
            f"macro_f1={val_metrics['macro_f1']:.4f}"
        )

        save_dict = {
            'model': student.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch + 1,
            'validation_metrics': val_metrics,
            'validation_score': val_score,
            'selection_metric': args.offline_selection_metric,
            'loss_weights': base_loss_weights,
            'effective_loss_weights': loss_weights,
            'auxiliary_loss_scale': float(aux_loss_scale),
            'projection_hidden_dim': int(args.projection_hidden_dim),
            'projection_bottleneck_dim': int(args.projection_bottleneck_dim),
            'experiment_metadata': experiment_metadata(args),
        }

        torch.save(save_dict, args.model_path)
        args.logger.info("model saved to {}.".format(args.model_path))

        epoch_record = {
            "epoch": epoch + 1,
            "lr": float(optimizer.param_groups[0]["lr"]),
            "train_loss": float(loss_record.avg),
            "train_supervised_accuracy": float(train_acc_record.avg),
            "cls_loss": float(cls_loss_record.avg),
            "cluster_loss": float(cluster_loss_record.avg),
            "contrastive_loss": float(contrastive_loss_record.avg),
            "supcon_loss": float(sup_con_loss_record.avg),
            "base_loss_weights": dict(base_loss_weights),
            "effective_loss_weights": dict(loss_weights),
            "auxiliary_loss_scale": float(aux_loss_scale),
            "batchnorm_running_stats_frozen": bool(bn_running_stats_frozen),
            "trial_encoder_frozen": bool(trial_encoder_frozen),
            "optimizer_group_lrs": [
                float(group["lr"]) for group in optimizer.param_groups
            ],
            "lr_drop_applied": bool(lr_drop_applied),
            "validation_score": float(val_score),
            "validation_metrics": val_metrics,
        }
        with open(epoch_metrics_path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(epoch_record, ensure_ascii=False) + "\n")

        if val_score > best_val_score:
            args.logger.info(
                f"Best validation {args.offline_selection_metric}: {val_score:.4f} at epoch {epoch + 1}."
            )
            torch.save(save_dict, best_checkpoint_path)
            args.logger.info("model saved to {}.".format(best_checkpoint_path))
            best_val_score = val_score
            best_epoch = epoch + 1
            best_val_metrics = val_metrics

        exp_lr_scheduler.step()
        args.logger.info(f'Exp Name: {args.exp_name}')
        args.logger.info(
            f'Best validation metric so far: {best_val_score:.4f} at epoch {best_epoch}'
        )
        args.logger.info('\n')
        if (
            args.offline_early_stop_patience > 0
            and (epoch + 1 - best_epoch) >= args.offline_early_stop_patience
        ):
            args.logger.info(
                "Early stopping offline training after {} epochs without validation improvement."
                .format(args.offline_early_stop_patience)
            )
            break

    if best_val_metrics is None or best_epoch is None:
        raise RuntimeError("Offline training finished without a best-checkpoint metric snapshot.")

    best_checkpoint = torch.load(best_checkpoint_path)
    student.load_state_dict(best_checkpoint['model'])
    args.logger.info(
        f"Loaded validation-selected offline checkpoint from {best_checkpoint_path}; "
        f"epoch={best_epoch}."
    )
    if args.offline_skip_final_test:
        best_val_metrics['selection_metric'] = args.offline_selection_metric
        best_val_metrics['selection_score'] = float(best_val_score)
        best_val_metrics['selection_epoch'] = int(best_epoch)
        log_and_save_classification_metrics(
            best_val_metrics,
            "Offline Best Validation",
            args,
        )
        args.logger.info("Skipped final test evaluation as requested.")
        return
    args.logger.info('Final one-time evaluation on disjoint test set...')
    _, _, _, best_classification_metrics = test_offline(
        student, test_loader, epoch=best_epoch - 1, save_name='Final Test ACC', args=args
    )
    best_classification_metrics['selection_metric'] = args.offline_selection_metric
    best_classification_metrics['selection_score'] = float(best_val_score)
    best_classification_metrics['selection_epoch'] = int(best_epoch)
    best_classification_metrics['validation_metrics'] = best_val_metrics
    best_checkpoint['classification_metrics'] = best_classification_metrics
    torch.save(best_checkpoint, best_checkpoint_path)
    log_and_save_classification_metrics(
        best_classification_metrics,
        "Offline Best",
        args,
    )


def test_offline(model, test_loader, epoch, save_name, args):

    model.eval()

    preds, targets = [], []
    diagnostic_records = []
    mask = np.array([])
    # First extract all features
    for batch_idx, (images, label, _) in enumerate(tqdm(test_loader)):
        prepared_inputs = prepare_model_inputs(images, args)
        with torch.no_grad():
            _, _, logits = forward_prepared_model(model, prepared_inputs)
            preds.append(logits.argmax(1).cpu().numpy())
            targets.append(label.cpu().numpy())
            diagnostic_records.append(prediction_diagnostics(model, logits, args))
            mask = np.append(mask, np.array([True if x.item() in range(len(args.train_classes)) else False for x in label]))

    preds = np.concatenate(preds)
    targets = np.concatenate(targets)

    # -----------------------
    # EVALUATE
    # -----------------------
    all_acc, old_acc, new_acc = log_accs_from_preds(y_true=targets, y_pred=preds, mask=mask,
                                                    T=epoch, eval_funcs=args.eval_funcs, save_name=save_name,
                                                    args=args)

    classification_metrics = compute_classification_metrics(
        targets,
        preds,
        num_classes=args.num_labeled_classes,
    )
    classification_metrics["gcd_all_accuracy"] = float(all_acc)
    classification_metrics["gcd_old_accuracy"] = float(old_acc)
    classification_metrics["sample_unit"] = args.uschad_sample_unit
    classification_metrics["prediction_diagnostics"] = merge_prediction_diagnostics(
        diagnostic_records
    )

    return all_acc, old_acc, new_acc, classification_metrics
'''====================================================================================================================='''



'''online train and test'''
'''====================================================================================================================='''
def train_online(student, student_pre, proto_aug_manager, train_loader, test_loader, current_session, args):

    optimizer = build_sgd_optimizer(student, args)

    exp_lr_scheduler = build_cosine_lr_scheduler(
        optimizer,
        args.epochs_online_per_session,
    )

    cluster_criterion = DistillLoss(
                        args.warmup_teacher_temp_epochs,
                        args.epochs_online_per_session,
                        args.n_views,
                        args.warmup_teacher_temp,
                        args.teacher_temp,
                    )

    # best acc log
    best_test_acc_all = -1.0
    best_test_acc_old = 0
    best_test_acc_new = 0

    best_test_acc_soft_all = 0
    best_test_acc_seen = 0
    best_test_acc_unseen = 0
    best_classification_metrics = None
    selected_epoch = None
    selected_checkpoint_path = online_checkpoint_path(args, current_session)

    for epoch in range(args.epochs_online_per_session):
        loss_record = AverageMeter()

        student.train()
        student_pre.eval()
        for batch_idx, batch in enumerate(train_loader):

            images, class_labels, uq_idxs, _ = batch   # NOTE!!!   mask lab in this setting
            mask_lab = torch.zeros_like(class_labels)   # NOTE!!! all samples are unlabeled

            class_labels, mask_lab = class_labels.cuda(non_blocking=True), mask_lab.cuda(non_blocking=True).bool()
            prepared_inputs = prepare_model_inputs(images, args)
            feats, student_proj, student_out = forward_prepared_model(
                student, prepared_inputs
            )
            teacher_out = student_out.detach()

            # clustering, unsup
            cluster_loss = cluster_criterion(student_out, teacher_out, epoch)
            avg_probs = (student_out / 0.1).softmax(dim=1).mean(dim=0)
            #me_max_loss = - torch.sum(torch.log(avg_probs**(-avg_probs))) + math.log(float(len(avg_probs)))
            #cluster_loss += args.memax_weight * me_max_loss

            # 1. inter old and new
            avg_probs_old_in = avg_probs[:args.num_seen_classes]
            avg_probs_new_in = avg_probs[args.num_seen_classes:]

            #avg_probs_old_new = torch.tensor([torch.sum(avg_probs_old_in), torch.sum(avg_probs_new_in)], requires_grad=True, device=device)
            #me_max_loss_old_new = - torch.sum(torch.log(avg_probs_old_new**(-avg_probs_old_new))) + math.log(float(len(avg_probs_old_new)))
            avg_probs_old_marginal, avg_probs_new_marginal = torch.sum(avg_probs_old_in), torch.sum(avg_probs_new_in)
            me_max_loss_old_new =  avg_probs_old_marginal * torch.log(avg_probs_old_marginal) + avg_probs_new_marginal * torch.log(avg_probs_new_marginal) + math.log(2)

            # 2. old (intra) & new (intra)
            avg_probs_old_in_norm = avg_probs_old_in / torch.sum(avg_probs_old_in)   # norm
            avg_probs_new_in_norm = avg_probs_new_in / torch.sum(avg_probs_new_in)   # norm
            me_max_loss_old_in = - torch.sum(torch.log(avg_probs_old_in_norm**(-avg_probs_old_in_norm))) + math.log(float(len(avg_probs_old_in_norm)))
            if args.num_novel_class_per_session > 1:
                me_max_loss_new_in = - torch.sum(torch.log(avg_probs_new_in_norm**(-avg_probs_new_in_norm))) + math.log(float(len(avg_probs_new_in_norm)))
            else:
                me_max_loss_new_in = torch.tensor(0.0, device=device)
            # overall me-max loss
            cluster_loss += args.memax_old_new_weight * me_max_loss_old_new + \
                args.memax_old_in_weight * me_max_loss_old_in + args.memax_new_in_weight * me_max_loss_new_in


            # represent learning, unsup
            contrastive_logits, contrastive_labels = info_nce_logits(features=student_proj)
            contrastive_loss = torch.nn.CrossEntropyLoss()(contrastive_logits, contrastive_labels)

            proto_aug_loss = proto_aug_manager.compute_proto_aug_hardness_aware_loss(
                student,
                batch_size=feats.size(0),
            )
            feats = torch.nn.functional.normalize(feats, dim=-1)
            with torch.no_grad():
                feats_pre = forward_prepared_backbone(
                    student_pre[0], prepared_inputs
                )
                feats_pre = torch.nn.functional.normalize(feats_pre, dim=-1)
            feat_distill_loss = (feats-feats_pre).pow(2).sum() / len(feats)

            # Total loss
            loss = 0
            loss += 1 * cluster_loss
            loss += 1 * contrastive_loss
            loss += args.proto_aug_weight * proto_aug_loss
            loss += args.feat_distill_weight * feat_distill_loss

            # logs
            pstr = ''
            pstr += f'me_max_loss_old_new: {me_max_loss_old_new.item():.4f} '
            pstr += f'me_max_loss_old_in: {me_max_loss_old_in.item():.4f} '
            pstr += f'me_max_loss_new_in: {me_max_loss_new_in.item():.4f} '
            pstr += f'cluster_loss: {cluster_loss.item():.4f} '
            pstr += f'contrastive_loss: {contrastive_loss.item():.4f} '
            pstr += f'proto_aug_loss: {proto_aug_loss.item():.4f} '
            pstr += f'feat_distill_loss: {feat_distill_loss.item():.4f} '

            loss_record.update(loss.item(), class_labels.size(0))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if batch_idx % args.print_freq == 0:
                args.logger.info('Epoch: [{}][{}/{}]\t loss {:.5f}\t {}'
                            .format(epoch, batch_idx, len(train_loader), loss.item(), pstr))
                new_true_ratio = len(class_labels[class_labels>=args.num_seen_classes]) / len(class_labels)
                logits = student_out / 0.1
                preds = logits.argmax(1)
                new_pred_ratio = len(preds[preds>=args.num_seen_classes]) / len(preds)
                args.logger.info(f'Avg old prob: {torch.sum(avg_probs_old_in).item():.4f} | Avg new prob: {torch.sum(avg_probs_new_in).item():.4f} | Pred new ratio: {new_pred_ratio:.4f} | Ground-truth new ratio: {new_true_ratio:.4f}')

        args.logger.info('Train Epoch: {} Avg Loss: {:.4f} '.format(epoch, loss_record.avg))

        # Step schedule
        exp_lr_scheduler.step()

        if args.online_checkpoint_selection == 'test_best':
            args.logger.info('Testing on disjoint test set for legacy checkpoint selection...')
            all_acc_test, old_acc_test, new_acc_test, \
                all_acc_soft_test, seen_acc_test, unseen_acc_test, classification_metrics = test_online(
                    student, test_loader, epoch=epoch, save_name='Test ACC', args=args
                )
            args.logger.info('Test Accuracies (Hard): All {:.4f} | Old {:.4f} | New {:.4f}'.format(all_acc_test, old_acc_test, new_acc_test))
            args.logger.info('Test Accuracies (Soft): All {:.4f} | Seen {:.4f} | Unseen {:.4f}'.format(all_acc_soft_test, seen_acc_test, unseen_acc_test))
            save_dict = {
                'model': student.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch + 1,
                'classification_metrics': classification_metrics,
                'checkpoint_selection': args.online_checkpoint_selection,
                'experiment_metadata': experiment_metadata(args),
            }
            if all_acc_test > best_test_acc_all:
                classification_metrics['checkpoint_selection'] = 'test_best'
                classification_metrics['selection_epoch'] = int(epoch + 1)
                save_dict['classification_metrics'] = classification_metrics
                args.logger.info(f'Best ACC on All Classes on test set of session-{current_session}: {all_acc_test:.4f}...')
                torch.save(save_dict, selected_checkpoint_path)
                args.logger.info(f"model saved to {selected_checkpoint_path}.")
                best_test_acc_all = all_acc_test
                best_test_acc_old = old_acc_test
                best_test_acc_new = new_acc_test
                best_test_acc_soft_all = all_acc_soft_test
                best_test_acc_seen = seen_acc_test
                best_test_acc_unseen = unseen_acc_test
                best_classification_metrics = classification_metrics
                selected_epoch = epoch + 1

            args.logger.info(f'Exp Name: {args.exp_name}')
            args.logger.info(f'Metrics with best model on test set (Hard) of session-{current_session}: All (Hard): {best_test_acc_all:.4f} Old: {best_test_acc_old:.4f} New: {best_test_acc_new:.4f}')
            args.logger.info(f'Metrics with best model on test set (Hard) of session-{current_session}: All (Soft): {best_test_acc_soft_all:.4f} Seen: {best_test_acc_seen:.4f} Unseen: {best_test_acc_unseen:.4f}')
        else:
            args.logger.info(
                'Final-test evaluation deferred until the fixed final epoch; '
                'the test set is not used for epoch selection.'
            )
        args.logger.info('\n')


    if args.online_checkpoint_selection == 'final_epoch':
        final_epoch = args.epochs_online_per_session - 1
        args.logger.info(
            f'Evaluating session-{current_session} final checkpoint once on the test set.'
        )
        best_test_acc_all, best_test_acc_old, best_test_acc_new, \
            best_test_acc_soft_all, best_test_acc_seen, best_test_acc_unseen, \
            best_classification_metrics = test_online(
                student,
                test_loader,
                epoch=final_epoch,
                save_name='Final Test ACC',
                args=args,
            )
        best_classification_metrics['checkpoint_selection'] = 'final_epoch'
        best_classification_metrics['selection_epoch'] = int(
            args.epochs_online_per_session
        )
        selected_epoch = args.epochs_online_per_session
        save_dict = {
            'model': student.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': args.epochs_online_per_session,
            'classification_metrics': best_classification_metrics,
            'checkpoint_selection': args.online_checkpoint_selection,
            'experiment_metadata': experiment_metadata(args),
        }
        torch.save(save_dict, selected_checkpoint_path)
        args.logger.info(f'Final-epoch model saved to {selected_checkpoint_path}.')

    # Log metrics from the configured online checkpoint-selection policy.
    selection_name = (
        "Final" if args.online_checkpoint_selection == 'final_epoch' else "Best"
    )
    if not os.path.exists(selected_checkpoint_path):
        raise RuntimeError(
            f"Online {selection_name.lower()} checkpoint was not created for "
            f"session-{current_session}: "
            f"{selected_checkpoint_path}. Please check epochs_online_per_session "
            "and evaluation results."
        )

    if best_classification_metrics is None:
        raise RuntimeError(
            f"Online training finished without a {selection_name.lower()}-checkpoint "
            f"metric snapshot for session-{current_session}."
        )
    if selected_epoch is None:
        raise RuntimeError(
            f"Online session-{current_session} did not record its selected epoch."
        )
    log_and_save_classification_metrics(
        best_classification_metrics,
        f"Online Session {current_session} {selection_name}",
        args,
    )

    args.best_test_acc_all_list.append(best_test_acc_all)
    args.best_test_acc_old_list.append(best_test_acc_old)
    args.best_test_acc_new_list.append(best_test_acc_new)
    args.best_test_acc_soft_all_list.append(best_test_acc_soft_all)
    args.best_test_acc_seen_list.append(best_test_acc_seen)
    args.best_test_acc_unseen_list.append(best_test_acc_unseen)
    args.best_test_classification_metrics_list.append(best_classification_metrics)


def test_online(model, test_loader, epoch, save_name, args):

    model.eval()

    preds, targets = [], []
    diagnostic_records = []
    mask_hard = np.array([])
    mask_soft = np.array([])
    for batch_idx, (images, label, _) in enumerate(tqdm(test_loader)):
        prepared_inputs = prepare_model_inputs(images, args)
        with torch.no_grad():
            _, _, logits = forward_prepared_model(model, prepared_inputs)
            preds.append(logits.argmax(1).cpu().numpy())
            targets.append(label.cpu().numpy())
            diagnostic_records.append(prediction_diagnostics(model, logits, args))
            mask_hard = np.append(mask_hard, np.array([True if x.item() in range(len(args.train_classes))
                                         else False for x in label]))
            mask_soft = np.append(mask_soft, np.array([True if x.item() in range(args.num_seen_classes)
                                         else False for x in label]))

    preds = np.concatenate(preds)
    targets = np.concatenate(targets)

    # -----------------------
    # EVALUATE
    # -----------------------
    all_acc, old_acc, new_acc = log_accs_from_preds(y_true=targets, y_pred=preds, mask=mask_hard,
                                                    T=epoch, eval_funcs=args.eval_funcs, save_name=save_name,
                                                    args=args)

    all_acc_soft, seen_acc, unseen_acc = log_accs_from_preds(y_true=targets, y_pred=preds, mask=mask_soft,
                                                    T=epoch, eval_funcs=args.eval_funcs, save_name=save_name,
                                                    args=args)

    num_classes = max(
        int(getattr(args, "mlp_out_dim_cur", 0)),
        int(targets.max()) + 1,
        int(preds.max()) + 1,
    )
    aligned_preds, hungarian_assignment = align_cluster_predictions(targets, preds)
    classification_metrics = compute_classification_metrics(
        targets,
        aligned_preds,
        num_classes=num_classes,
    )
    classification_metrics["prediction_alignment"] = "hungarian_v2_global"
    classification_metrics["hungarian_assignment_pred_to_true"] = hungarian_assignment
    classification_metrics["gcd_all_accuracy"] = float(all_acc)
    classification_metrics["gcd_old_accuracy"] = float(old_acc)
    classification_metrics["gcd_new_accuracy"] = float(new_acc)
    classification_metrics["gcd_soft_all_accuracy"] = float(all_acc_soft)
    classification_metrics["gcd_seen_accuracy"] = float(seen_acc)
    classification_metrics["gcd_unseen_accuracy"] = float(unseen_acc)
    classification_metrics["sample_unit"] = args.uschad_sample_unit
    classification_metrics["prediction_diagnostics"] = merge_prediction_diagnostics(
        diagnostic_records
    )

    return all_acc, old_acc, new_acc, all_acc_soft, seen_acc, unseen_acc, classification_metrics
'''====================================================================================================================='''


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='cluster', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--eval_batch_size', default=256, type=int,
                        help='Batch size for test, prototype, and new-head initialization DataLoaders.')
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--num_workers_test', default=4, type=int)
    parser.add_argument('--eval_funcs',nargs='+', help='Which eval functions to use',default=['v2'],)
    parser.add_argument('--dataset_name', type=str, default='cifar100', help='options: cifar10, cifar100, tiny_imagenet, cub, imagenet_100')
    parser.add_argument('--use_ssb_splits', action='store_true', default=True)

    parser.add_argument('--grad_from_block', type=int, default=11)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--gamma', type=float, default=0.1)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=5e-5)
    parser.add_argument('--exp_root', type=str, default=exp_root_happy)
    parser.add_argument('--transform', type=str, default='imagenet')
    parser.add_argument('--image_aug_mode',type=str,default='off',choices=['auto', 'on', 'off'],
        help=(
            "Control Happy image augmentation. "
            "auto: image datasets use image augmentation and USC-HAD skips it; "
            "on: force image augmentation; "
            "off: disable image augmentation for image datasets."
        ),
    )

    parser.add_argument('--har_aug_mode',type=str,default='none',choices=['none', 'weak_strong'],
        help=(
            "Control USC-HAD time-series augmentation. "
            "none: no HAR augmentation, but still return two cloned views for Happy training; "
            "weak_strong: return weak and strong augmented HAR views."
        ),
    )
    parser.add_argument('--uschad_npz_path',type=str,default='',
        help='Path to processed USC-HAD npz file, e.g. ./processed/uschad_v0/uschad_windows.npz',
    )

    parser.add_argument('--uschad_window_size',type=int,default=256,
        help='Expected USC-HAD npz temporal window size. Must match npz windows.shape[2].',
    )
    parser.add_argument('--uschad_sample_unit', type=str, default='window',
        choices=['window', 'trial'],
        help='Classification sample unit. trial groups every complete trial into one bag.')
    parser.add_argument('--trial_pooling', type=str, default='mean_robust_max',
        choices=['mean', 'mean_robust_max', 'gated_attention'],
        help='Pooling used only when --uschad_sample_unit trial.')
    parser.add_argument('--trial_view_mode', type=str, default='full_random_crop',
        choices=['full_full', 'full_random_crop'],
        help='Two-view policy for trial training; evaluation always uses the full trial.')
    parser.add_argument('--trial_crop_ratio', type=float, default=2.0 / 3.0,
        help='Contiguous crop ratio for the second full_random_crop trial view.')
    parser.add_argument('--trial_min_windows', type=int, default=2,
        help='Minimum number of windows retained by the random trial crop.')
    parser.add_argument('--trial_robust_max_quantile', type=float, default=0.90,
        help=(
            'Feature-wise quantile used as the robust maximum by '
            'mean_robust_max pooling. A value of 0.90 filters the most extreme '
            'approximately 10 percent of window features.'
        ))
    parser.add_argument('--trial_pool_fusion_dim', type=int, default=64,
        help='Low-rank hidden dimension for mean/robust-maximum feature fusion.')
    parser.add_argument('--trial_pool_fusion_dropout', type=float, default=0.0,
        help='Dropout in mean/robust-maximum fusion; 0 is recommended for small folds.')
    parser.add_argument('--trial_attention_dim', type=int, default=64,
        help='Hidden dimension of gated MIL attention.')
    parser.add_argument('--trial_attention_dropout', type=float, default=0.1,
        help='Dropout applied to attention inputs.')
    parser.add_argument('--trial_attention_temperature', type=float, default=1.0,
        help='Softmax temperature for trial attention scores.')
    parser.add_argument('--trial_attention_mean_mix', type=float, default=0.5,
        help='Attention share rho in (1-rho)*mean + rho*attention pooling.')
    parser.add_argument('--trial_encoder_checkpoint', type=str, default='',
        help='Optional window-level model_best.pt used to initialize ResNet1D.')
    parser.add_argument('--trial_encoder_freeze_epochs', type=int, default=0,
        help='Offline epochs that train only trial pooling/head; requires an encoder checkpoint.')
    parser.add_argument('--trial_encoder_lr_scale', type=float, default=0.1,
        help='Window-encoder LR divided relative to pooling/head LR in trial mode.')
    parser.add_argument('--trial_confidence_temperature', type=float, default=1.0,
        help='Reporting-only softmax temperature for confidence/entropy diagnostics.')
    parser.add_argument('--uschad_split_mode',type=str,default='trial',choices=['trial', 'subject'],
        help=('USC-HAD split mode. ''trial: train/test are split by trials and subjects may overlap. ''subject: train/test subjects are disjoint.'),
    )
    parser.add_argument(
        '--uschad_recompute_norm_from_train_subjects',
        action='store_true',
        default=False,
        help=(
            'Reconstruct raw windows and recompute per-channel z-score statistics '
            'from only old-class windows of the explicit training subjects.'
        ),
    )
    parser.add_argument('--uschad_norm_eps', type=float, default=1e-6,
        help='Minimum per-channel std for fold-specific USC-HAD normalization.')
    parser.add_argument('--uschad_cv_fold', type=int, default=-1,
        help='Optional subject-level cross-validation fold id recorded in checkpoints.')

    parser.add_argument(
        '--uschad_train_subjects',type=str,default='',help=(
            'Explicit train subject ids for USC-HAD subject split, ''e.g. "1,2,3,4,5,6,7,8,9,10". ''Empty means random subject split.'),
    )

    parser.add_argument('--uschad_test_subjects',type=str,default='',help=(
            'Explicit test subject ids for USC-HAD subject split, ''e.g. "11,12,13,14". ''Empty means random subject split.'),
    )
    parser.add_argument('--offline_val_subjects', type=str, default='', help=(
        'Optional subject-disjoint validation subjects for offline checkpoint selection, '
        'e.g. "8,9". Required for offline training.'),
    )
    parser.add_argument('--har_in_channels',type=int,default=6,
        help='Number of USC-HAD input channels. V0 raw setting is 6. V1 raw+mag setting is 8.',
    )
    parser.add_argument('--har_feat_dim',type=int,default=256,
        help='Output feature dimension of ResNet1D. Default is 256.',
    )
    parser.add_argument('--har_base_channels',type=int,default=64,
        help='Base channel number of ResNet1D. Default is 64.',
    )
    parser.add_argument('--har_dropout',type=float,default=0.0,
        help='Dropout rate in ResNet1D BasicBlock. Default is 0.0.',
    )
    parser.add_argument('--har_weak_jitter_std',type=float,default=0.01,
        help='Jitter std for weak HAR augmentation.',
    )

    parser.add_argument('--har_weak_scale_std',type=float,default=0.05,
        help='Scaling std for weak HAR augmentation.',
    )

    parser.add_argument('--har_strong_jitter_std',type=float,default=0.02,
        help='Jitter std for strong HAR augmentation.',
    )

    parser.add_argument('--har_strong_scale_std',type=float, default=0.10,
        help='Scaling std for strong HAR augmentation.',
    )

    parser.add_argument('--har_time_mask_ratio',type=float,default=0.10,
        help='Time masking ratio for strong HAR augmentation.',
    )
    parser.add_argument('--temperature', type=float, default=0.1,
        help='Student/classifier temperature for offline CE, entropy, and DistillLoss. Default preserves the previous hard-coded 0.1.')
    parser.add_argument('--sup_weight', type=float, default=0.35)
    parser.add_argument('--offline_cls_weight', type=float, default=None,
        help='Explicit offline CE weight. Set all four --offline_*_weight values together.')
    parser.add_argument('--offline_cluster_weight', type=float, default=None,
        help='Explicit offline distillation-plus-MeMax weight.')
    parser.add_argument('--offline_contrast_weight', type=float, default=None,
        help='Explicit offline InfoNCE weight.')
    parser.add_argument('--offline_supcon_weight', type=float, default=None,
        help='Explicit offline supervised contrastive weight.')
    parser.add_argument('--offline_bn_freeze_epoch', type=int, default=-1,
        help=('Freeze BatchNorm running mean/variance from this 1-based offline epoch; '
              'affine parameters remain trainable. Values <= 0 disable it.'))
    parser.add_argument('--offline_lr_drop_epoch', type=int, default=-1,
        help=('Multiply the live offline LR trajectory once at this 1-based epoch. '
              'Values <= 0 disable it.'))
    parser.add_argument('--offline_lr_drop_factor', type=float, default=0.1,
        help='Multiplicative LR factor used by --offline_lr_drop_epoch.')
    parser.add_argument('--offline_aux_decay_start_epoch', type=int, default=-1,
        help=('First 1-based epoch of the optional linear cluster/InfoNCE weight decay; '
              'set together with --offline_aux_decay_end_epoch.'))
    parser.add_argument('--offline_aux_decay_end_epoch', type=int, default=-1,
        help='1-based epoch at which cluster/InfoNCE weights reach their final scale.')
    parser.add_argument('--offline_aux_final_scale', type=float, default=0.0,
        help='Final multiplier in [0, 1] for cluster/InfoNCE weights after decay.')
    parser.add_argument('--projection_hidden_dim', type=int, default=2048,
        help='Hidden width of every offline/online DINO projection head.')
    parser.add_argument('--projection_bottleneck_dim', type=int, default=256,
        help='Contrastive bottleneck width of every offline/online DINO projection head.')
    parser.add_argument('--offline_gradient_diagnostic_epochs', type=str, default='',
        help=('Comma/space separated 1-based epochs whose first batch records per-loss '
              'backbone gradient norms and pairwise cosine similarities.'))
    parser.add_argument('--offline_selection_metric', type=str, default='gcd_old_accuracy',
        choices=['gcd_old_accuracy', 'overall_accuracy', 'mean_class_accuracy', 'macro_f1'],
        help='Validation metric used to select model_best.pt in offline training.')
    parser.add_argument('--offline_early_stop_patience', type=int, default=0,
        help='Stop offline training after this many epochs without validation improvement; 0 disables early stopping.')
    parser.add_argument('--offline_skip_final_test', action='store_true', default=False,
        help='Do not evaluate the held-out test set after offline training; use for validation-only screening.')
    parser.add_argument('--n_views', default=2, type=int)
    parser.add_argument('--contrast_unlabel_only', action='store_true', default=False)

    '''group-wise entropy regularization'''
    # memax weight for offline session
    parser.add_argument('--memax_weight', type=float, default=1)
    # memax weight for online session
    parser.add_argument('--memax_old_new_weight', type=float, default=2)
    parser.add_argument('--memax_old_in_weight', type=float, default=1)
    parser.add_argument('--memax_new_in_weight', type=float, default=1)
    parser.add_argument('--warmup_teacher_temp', default=0.07, type=float, help='Initial value for the teacher temperature.')
    parser.add_argument('--teacher_temp', default=0.04, type=float, help='Final value (after linear warmup) of the teacher temperature.')
    #parser.add_argument('--teacher_temp_final', default=0.05, type=float, help='Final value (online session) of the teacher temperature.')
    parser.add_argument('--warmup_teacher_temp_epochs', default=30, type=int, help='Number of warmup epochs for the teacher temperature.')

    '''clustering-guided initialization'''
    parser.add_argument('--init_new_head', action='store_true', default=False)

    '''PASS params'''
    parser.add_argument('--proto_aug_weight', type=float, default=1.0)
    parser.add_argument('--feat_distill_weight', type=float, default=1.0)
    parser.add_argument('--radius_scale', type=float, default=1.0)

    '''hardness-aware sampling temperature'''
    parser.add_argument('--hardness_temp', type=float, default=0.1)

    # Continual GCD params
    parser.add_argument('--num_old_classes', type=int, default=-1)
    parser.add_argument('--prop_train_labels', type=float, default=0.8)
    parser.add_argument('--train_session', type=str, default='offline', help='options: offline, online')
    parser.add_argument('--load_offline_id', type=str, default=None)
    parser.add_argument('--epochs_offline', default=100, type=int)
    parser.add_argument('--epochs_online_per_session', default=30, type=int)

    parser.add_argument('--num_novel_classes_per_session',default=2,type=int,
        help=('Number of newly introduced novel classes in each online CGCD session. ''It must be >= 1 and must divide the total number of novel classes. '
            'For USC-HAD with 6 novel classes, recommended values are 1, 2, 3, or 6; ''2 is recommended for 3 online sessions.'),)

    parser.add_argument('--continual_session_num',default=-1,type=int,
        help=('Deprecated compatibility argument. ''Do not use it as the main control parameter. ''The real continual_session_num will be derived from '
              'num_unlabeled_classes // num_novel_classes_per_session.'),)

    parser.add_argument('--online_novel_unseen_num', default=400, type=int)
    parser.add_argument('--online_old_seen_num', default=50, type=int)
    parser.add_argument('--online_novel_seen_num', default=50, type=int)
    parser.add_argument('--online_old_seen_trials', default=2, type=int,
        help='Per-old-class trial count per online session in trial mode.')
    parser.add_argument('--online_novel_unseen_trials', default=5, type=int,
        help='Trial count when a novel class first appears in trial mode.')
    parser.add_argument('--online_novel_seen_trials', default=2, type=int,
        help='Trial count for a previously introduced novel class in trial mode.')
    parser.add_argument('--online_checkpoint_selection', type=str, default='auto',
        choices=['auto', 'test_best', 'final_epoch'],
        help=('auto uses final_epoch for trial mode and legacy test_best for window mode. '
              'final_epoch evaluates the final test set only once per session.'))
    # shuffle dataset classes
    parser.add_argument('--shuffle_classes', action='store_true', default=False)
    parser.add_argument('--seed', default=0, type=int)

    # others
    parser.add_argument('--print_freq', default=10, type=int)
    parser.add_argument('--exp_name', default='simgcd-pro-v5', type=str)


    # ----------------------
    # INIT
    # ----------------------
    args = parser.parse_args()
    device = torch.device('cuda:0')
    set_seed(args.seed)
    args = get_class_splits(args)

    args.num_labeled_classes = len(args.train_classes)
    args.num_unlabeled_classes = len(args.unlabeled_classes)

    args = derive_cgcd_session_config(args)
    args = validate_training_args(args)

    # Keep the user-specified root before appending train_session.
    # This is needed when online training loads checkpoints from the offline root.
    args.base_exp_root = args.exp_root

    args.exp_root = args.exp_root + '_' + args.train_session
    args.exp_name = 'happy' + '-' + args.train_session

    if args.train_session == 'offline':
        args.base_exp_id = 'Old' + str(args.num_labeled_classes) + '_' + 'Ratio' + str(args.prop_train_labels)
        if args.uschad_sample_unit == 'trial':
            args.base_exp_id += (
                '_SampleUnitTrial'
                + '_Pool' + str(args.trial_pooling)
                + '_View' + str(args.trial_view_mode)
            )
        else:
            args.base_exp_id += '_SampleUnitWindow'


    elif args.train_session == 'online':
        args.base_exp_id = 'Old' + str(args.num_labeled_classes) + '_' + 'Ratio' + str(args.prop_train_labels) \
                         + '_' + 'NovelPerSession' + str(args.num_novel_class_per_session) \
                         + '_' + 'ContinualNum' + str(args.continual_session_num)
        if args.uschad_sample_unit == 'trial':
            args.base_exp_id += (
                '_SampleUnitTrial'
                + '_Pool' + str(args.trial_pooling)
                + '_View' + str(args.trial_view_mode)
                + '_OldTrials' + str(args.online_old_seen_trials)
                + '_UnseenTrials' + str(args.online_novel_unseen_trials)
                + '_SeenTrials' + str(args.online_novel_seen_trials)
            )
        else:
            args.base_exp_id += (
                '_SampleUnitWindow'
                + '_OldSeenNum' + str(args.online_old_seen_num)
                + '_UnseenNum' + str(args.online_novel_unseen_num)
                + '_SeenNum' + str(args.online_novel_seen_num)
            )

    else:
        raise NotImplementedError

    init_experiment(args, runner_name=['Happy'])
    args.logger.info(f'Using evaluation function {args.eval_funcs[0]} to print results')
    args.logger.info("========== DATALOADER SEED CONFIG ==========")
    args.logger.info(f"seed = {args.seed}")
    args.logger.info(f"num_workers = {args.num_workers}")
    args.logger.info(f"num_workers_test = {args.num_workers_test}")
    args.logger.info("DataLoader generator is explicitly set.")
    args.logger.info("DataLoader worker_init_fn is explicitly set.")
    args.logger.info("============================================")
    args.logger.info("========== OFFLINE DIAGNOSTIC CONFIG ==========")
    args.logger.info(f"temperature = {args.temperature}")
    args.logger.info(
        f"teacher_temp: warmup={args.warmup_teacher_temp}, final={args.teacher_temp}, "
        f"warmup_epochs={args.warmup_teacher_temp_epochs}"
    )
    args.logger.info(f"offline_val_subjects = {args.offline_val_subjects}")
    args.logger.info(f"offline_selection_metric = {args.offline_selection_metric}")
    args.logger.info(f"offline_early_stop_patience = {args.offline_early_stop_patience}")
    args.logger.info(f"offline_skip_final_test = {args.offline_skip_final_test}")
    args.logger.info(
        "explicit offline loss weights: "
        f"cls={args.offline_cls_weight}, cluster={args.offline_cluster_weight}, "
        f"contrast={args.offline_contrast_weight}, supcon={args.offline_supcon_weight}"
    )
    args.logger.info(
        "offline stability controls: "
        f"bn_freeze_epoch={args.offline_bn_freeze_epoch}, "
        f"lr_drop_epoch={args.offline_lr_drop_epoch}, "
        f"lr_drop_factor={args.offline_lr_drop_factor}, "
        f"aux_decay={args.offline_aux_decay_start_epoch}->"
        f"{args.offline_aux_decay_end_epoch}, "
        f"aux_final_scale={args.offline_aux_final_scale}"
    )
    args.logger.info(
        "projection head: "
        f"hidden_dim={args.projection_hidden_dim}, "
        f"bottleneck_dim={args.projection_bottleneck_dim}"
    )
    args.logger.info(
        "offline_gradient_diagnostic_epochs = "
        f"{sorted(parse_epoch_numbers(args.offline_gradient_diagnostic_epochs))}"
    )
    args.logger.info(
        "USC-HAD split/normalization config: "
        f"mode={args.uschad_split_mode}, cv_fold={args.uschad_cv_fold}, "
        f"train_subjects={canonical_subject_ids(args.uschad_train_subjects)}, "
        f"val_subjects={canonical_subject_ids(args.offline_val_subjects)}, "
        f"test_subjects={canonical_subject_ids(args.uschad_test_subjects)}, "
        "recompute_from_train_old_classes="
        f"{args.uschad_recompute_norm_from_train_subjects}, "
        f"norm_eps={args.uschad_norm_eps}"
    )
    if args.uschad_sample_unit == 'trial':
        sample_config = (
            "USC-HAD sample config: "
            f"unit=trial, window={args.uschad_window_size}, "
            f"pooling={args.trial_pooling}, view_mode={args.trial_view_mode}"
        )
        if args.trial_view_mode == 'full_random_crop':
            sample_config += (
                f", crop_ratio={args.trial_crop_ratio}, "
                f"min_windows={args.trial_min_windows}"
            )
        else:
            sample_config += ", crop parameters inactive"
        args.logger.info(sample_config)
    else:
        args.logger.info(
            "USC-HAD sample config: "
            f"unit=window, window={args.uschad_window_size}; "
            "trial pooling/view options are inactive"
        )
    args.logger.info(
        "HAR augmentation config: "
        f"mode={args.har_aug_mode}, "
        f"weak_jitter={args.har_weak_jitter_std}, "
        f"weak_scale={args.har_weak_scale_std}, "
        f"strong_jitter={args.har_strong_jitter_std}, "
        f"strong_scale={args.har_strong_scale_std}, "
        f"time_mask_ratio={args.har_time_mask_ratio}"
    )
    if args.uschad_sample_unit == 'trial':
        if args.trial_pooling == 'mean_robust_max':
            args.logger.info(
                "Trial robust maximum config: "
                f"quantile={args.trial_robust_max_quantile}, "
                f"filtered_upper_fraction="
                f"{1.0 - args.trial_robust_max_quantile:.6g}, "
                f"fusion_dim={args.trial_pool_fusion_dim}, "
                f"fusion_dropout={args.trial_pool_fusion_dropout}"
            )
        elif args.trial_pooling == 'gated_attention':
            args.logger.info(
                "Trial attention config: "
                f"dim={args.trial_attention_dim}, "
                f"dropout={args.trial_attention_dropout}, "
                f"temperature={args.trial_attention_temperature}, "
                f"mean_mix={args.trial_attention_mean_mix}"
            )
        args.logger.info(
            "Trial encoder config: "
            f"checkpoint={args.trial_encoder_checkpoint or '<none>'}, "
            f"freeze_epochs={args.trial_encoder_freeze_epochs}, "
            f"lr_scale={args.trial_encoder_lr_scale}"
        )
    if args.train_session == 'online':
        if args.uschad_sample_unit == 'trial':
            args.logger.info(
                "Online sampling counts (trials/class): "
                f"old={args.online_old_seen_trials}, "
                f"novel_unseen={args.online_novel_unseen_trials}, "
                f"novel_seen={args.online_novel_seen_trials}; "
                f"checkpoint_selection={args.online_checkpoint_selection}"
            )
        else:
            args.logger.info(
                "Online sampling counts (windows/class): "
                f"old={args.online_old_seen_num}, "
                f"novel_unseen={args.online_novel_unseen_num}, "
                f"novel_seen={args.online_novel_seen_num}; "
                f"checkpoint_selection={args.online_checkpoint_selection}"
            )
    args.logger.info("=================================================")

    # ----------------------
    # BASE MODEL
    # ----------------------
    args.interpolation = 3
    args.crop_pct = 0.875

    backbone = build_backbone(args)
    backbone = configure_backbone_trainable(args, backbone)
    if args.uschad_sample_unit == 'trial':
        load_trial_window_encoder(
            backbone,
            args.trial_encoder_checkpoint,
            args,
            args.logger,
        )

    args.mlp_out_dim = args.num_labeled_classes  # NOTE!!!

    args.logger.info('model build')

    # ----------------------
    # PROJECTION HEAD
    # ----------------------
    projector = DINOHead(
        in_dim=args.feat_dim,
        out_dim=args.mlp_out_dim,
        nlayers=args.num_mlp_layers,
        hidden_dim=args.projection_hidden_dim,
        bottleneck_dim=args.projection_bottleneck_dim,
    )
    model = nn.Sequential(backbone, projector)

    model.to(device)

    # --------------------
    # CONTRASTIVE TRANSFORM
    # --------------------
    train_transform, test_transform = build_transforms(args)


    # ----------------------
    # 1. OFFLINE TRAIN
    # ----------------------
    if args.train_session == 'offline':
        args.logger.info('========== offline training with labeled old data (old) ==========')
        args.logger.info('loading dataset...')
        offline_session_train_dataset, offline_session_val_dataset, offline_session_test_dataset,\
            _online_session_train_dataset_list, _online_session_test_dataset_list,\
                datasets, dataset_split_config_dict, novel_targets_shuffle = get_datasets(
                    args.dataset_name, train_transform, test_transform, args)

        # saving dataset dict
        print('save dataset dict...')
        save_dataset_dict_path = os.path.join(args.log_dir, 'offline_dataset_dict.txt')
        f_dataset_dict = open(save_dataset_dict_path, 'w')
        f_dataset_dict.write('offline_dataset_split_dict: \n')
        f_dataset_dict.write(str(dataset_split_config_dict))
        f_dataset_dict.write('\nnovel_targets_shuffle: \n')
        f_dataset_dict.write(str(novel_targets_shuffle))
        f_dataset_dict.close()

        offline_train_generator = build_dataloader_generator(args.seed + 10)
        offline_val_generator = build_dataloader_generator(args.seed + 11)
        offline_test_generator = build_dataloader_generator(args.seed + 12)
        collate_fn = (
            uschad_trial_collate
            if args.uschad_sample_unit == 'trial' else None
        )
        offline_drop_last = True
        if args.uschad_sample_unit == 'trial':
            if len(offline_session_train_dataset) < 2:
                raise RuntimeError("Trial-level offline training needs at least 2 trials.")
            offline_drop_last = (
                len(offline_session_train_dataset) % args.batch_size == 1
            )

        offline_session_train_loader = DataLoader(
            offline_session_train_dataset,
            num_workers=args.num_workers,
            batch_size=args.batch_size,
            shuffle=True,
            drop_last=offline_drop_last,
            pin_memory=True,
            generator=offline_train_generator,
            worker_init_fn=seed_dataloader_worker,
            collate_fn=collate_fn,
        )

        offline_session_test_loader = DataLoader(
            offline_session_test_dataset,
            num_workers=args.num_workers_test,
            batch_size=args.eval_batch_size,
            shuffle=False,
            pin_memory=False,
            generator=offline_test_generator,
            worker_init_fn=seed_dataloader_worker,
            collate_fn=collate_fn,
        )

        if offline_session_val_dataset is None or len(offline_session_val_dataset) == 0:
            raise RuntimeError(
                "offline_session_val_dataset is empty. Check --offline_val_subjects "
                "and ensure they are disjoint from --uschad_train_subjects and "
                "--uschad_test_subjects."
            )
        offline_session_val_loader = DataLoader(
            offline_session_val_dataset,
            num_workers=args.num_workers_test,
            batch_size=args.eval_batch_size,
            shuffle=False,
            pin_memory=False,
            generator=offline_val_generator,
            worker_init_fn=seed_dataloader_worker,
            collate_fn=collate_fn,
        )

        # ----------------------
        # TRAIN
        # ----------------------
        train_offline(
            model,
            offline_session_train_loader,
            offline_session_val_loader,
            offline_session_test_loader,
            args,
        )


    # ----------------------
    # 2. ONLINE TRAIN
    # ----------------------
    elif args.train_session == 'online':
        args.logger.info('\n\n==================== online continual GCD with unlabeled data (old + novel) ====================')
        args.logger.info('loading dataset...')
        _offline_session_train_dataset, _offline_session_val_dataset, _offline_session_test_dataset,\
            online_session_train_dataset_list, online_session_test_dataset_list,\
                datasets, dataset_split_config_dict, novel_targets_shuffle = get_datasets(
                    args.dataset_name, train_transform, test_transform, args)

        # saving dataset dict
        print('save dataset dict...')
        save_dataset_dict_path = os.path.join(args.log_dir, 'online_dataset_dict.txt')
        f_dataset_dict = open(save_dataset_dict_path, 'w')
        f_dataset_dict.write('online_dataset_split_dict: \n')
        f_dataset_dict.write(str(dataset_split_config_dict))
        f_dataset_dict.write('\nnovel_targets_shuffle: \n')
        f_dataset_dict.write(str(novel_targets_shuffle))
        f_dataset_dict.write('\nnum_novel_class_per_session: \n')
        f_dataset_dict.write(str(args.num_novel_class_per_session))
        f_dataset_dict.write('\ncontinual_session_num: \n')
        f_dataset_dict.write(str(args.continual_session_num))
        f_dataset_dict.close()


        # ----------------------
        # CONTINUAL SESSIONS
        # ----------------------
        args.logger.info('number of novel class per session: {}'.format(args.num_novel_class_per_session))
        args.logger.info('derived continual session num: {}'.format(args.continual_session_num))

        '''v5: ProtoAug Manager'''
        proto_aug_manager = ProtoAugManager(args.feat_dim, args.n_views*args.batch_size, args.hardness_temp, args.radius_scale, device, args.logger)

        # best test acc list across continual sessions
        args.best_test_acc_all_list = []
        args.best_test_acc_old_list = []
        args.best_test_acc_new_list = []
        args.best_test_acc_soft_all_list = []
        args.best_test_acc_seen_list = []
        args.best_test_acc_unseen_list = []
        args.best_test_classification_metrics_list = []

        start_session = 0

        '''Continual GCD sessions'''
        #for session in range(args.continual_session_num):
        for session in range(start_session, args.continual_session_num):
            args.logger.info('\n\n========== begin online continual session-{} ==============='.format(session+1))
            # dataset for the current session
            online_session_train_dataset = online_session_train_dataset_list[session]
            online_session_test_dataset = online_session_test_dataset_list[session]

            online_train_generator = build_dataloader_generator(args.seed + 10000 + session * 10)
            online_test_generator = build_dataloader_generator(args.seed + 10000 + session * 10 + 1)
            collate_fn = (
                uschad_trial_collate
                if args.uschad_sample_unit == 'trial' else None
            )
            online_drop_last = True
            if args.uschad_sample_unit == 'trial':
                if len(online_session_train_dataset) < 2:
                    raise RuntimeError(
                        f"Online session-{session + 1} needs at least 2 trial samples."
                    )
                online_drop_last = (
                    len(online_session_train_dataset) % args.batch_size == 1
                )

            online_session_train_loader = DataLoader(
                online_session_train_dataset,
                num_workers=args.num_workers,
                batch_size=args.batch_size,
                shuffle=True,
                drop_last=online_drop_last,
                pin_memory=True,
                generator=online_train_generator,
                worker_init_fn=seed_dataloader_worker,
                collate_fn=collate_fn,
            )

            online_session_test_loader = DataLoader(
                online_session_test_dataset,
                num_workers=args.num_workers_test,
                batch_size=args.eval_batch_size,
                shuffle=False,
                pin_memory=False,
                generator=online_test_generator,
                worker_init_fn=seed_dataloader_worker,
                collate_fn=collate_fn,
            )

            # number of seen (offline old + previous online new) classes till the beginning of this session
            args.num_seen_classes = args.num_labeled_classes + args.num_novel_class_per_session * session
            args.logger.info('number of seen class (old + seen novel) at the beginning of current session: {}'.format(args.num_seen_classes))
            if args.dataset_name == 'cifar100':
                args.num_cur_novel_classes = len(np.unique(online_session_train_dataset.novel_unlabelled_dataset.targets))
            elif args.dataset_name == 'tiny_imagenet':
                novel_cls_labels = [t for i, (p, t) in enumerate(online_session_train_dataset.novel_unlabelled_dataset.data)]
                args.num_cur_novel_classes = len(np.unique(novel_cls_labels))
            elif args.dataset_name == 'aircraft':
                novel_cls_labels = [t for i, (p, t) in enumerate(online_session_train_dataset.novel_unlabelled_dataset.samples)]
                args.num_cur_novel_classes = len(np.unique(novel_cls_labels))
            elif args.dataset_name == 'scars':
                args.num_cur_novel_classes = len(np.unique(online_session_train_dataset.novel_unlabelled_dataset.target))   # NOTE!!! target
            else:
                args.num_cur_novel_classes = args.num_novel_class_per_session * (session+1)
            args.logger.info('number of all novel class (seen novel + unseen novel) in current session: {}'.format(args.num_cur_novel_classes))

            '''tunable params in backbone'''
            ####################################################################################################################
            backbone = configure_backbone_trainable(args, backbone)
            ####################################################################################################################

            '''load ckpts from last session (session>0) or offline session (session=0)'''
            ####################################################################################################################
            args.logger.info('loading checkpoints of model_pre...')
            if session == 0:
                projector_pre = DINOHead(
                    in_dim=args.feat_dim,
                    out_dim=args.num_labeled_classes,
                    nlayers=args.num_mlp_layers,
                    hidden_dim=args.projection_hidden_dim,
                    bottleneck_dim=args.projection_bottleneck_dim,
                )
                model_pre = nn.Sequential(backbone, projector_pre)

                if args.load_offline_id is None or str(args.load_offline_id).strip() == "":
                    raise RuntimeError(
                        "Online training requires an offline best checkpoint, "
                        "but --load_offline_id was not provided. "
                        "Please pass the offline experiment id, for example: "
                        "--load_offline_id Old6_Ratio0.8_YYYYMMDD-HHMMSS"
                    )

                offline_exp_root = args.base_exp_root + '_' + 'offline'
                load_dir_online = os.path.join(
                    offline_exp_root,
                    args.dataset_name,
                    args.load_offline_id,
                    'checkpoints',
                    'model_best.pt',
                )

                if not os.path.exists(load_dir_online):
                    raise FileNotFoundError(
                        f"Offline best checkpoint not found: {load_dir_online}\n"
                        f"Please check --exp_root, --dataset_name, and --load_offline_id.\n"
                        f"Expected path format:\n"
                        f"  <exp_root>_offline/{args.dataset_name}/<load_offline_id>/checkpoints/model_best.pt"
                    )

                args.logger.info('loading offline best checkpoint from: ' + load_dir_online)
                load_dict = torch.load(load_dir_online)
                validate_checkpoint_metadata(load_dict, args, load_dir_online)
                model_pre.load_state_dict(load_dict['model'])
                args.logger.info('successfully loaded offline best checkpoint!')
            else:        # session > 0:
                projector_pre = DINOHead(
                    in_dim=args.feat_dim,
                    out_dim=args.num_seen_classes,
                    nlayers=args.num_mlp_layers,
                    hidden_dim=args.projection_hidden_dim,
                    bottleneck_dim=args.projection_bottleneck_dim,
                )
                model_pre = nn.Sequential(backbone, projector_pre)
                load_dir_online = online_checkpoint_path(args, session)
                args.logger.info(
                    'loading selected checkpoint from last online session: '
                    + load_dir_online
                )
                if not os.path.exists(load_dir_online):
                    raise FileNotFoundError(
                        f"Previous online selected checkpoint not found: {load_dir_online}. "
                        f"Please check that session-{session} completed successfully."
                    )
                load_dict = torch.load(load_dir_online)
                validate_checkpoint_metadata(load_dict, args, load_dir_online)
                model_pre.load_state_dict(load_dict['model'])
                args.logger.info('successfully loaded checkpoints!')
            ####################################################################################################################

            '''incremental parametric classifier in SimGCD'''
            ####################################################################################################################
            ####################################################################################################################
            backbone_cur = deepcopy(backbone)   # NOTE!!!
            backbone_cur.load_state_dict(model_pre[0].state_dict())   # NOTE!!!
            args.mlp_out_dim_cur = args.num_labeled_classes + args.num_cur_novel_classes   # total num of classes in the current session
            args.logger.info('number of all class (old + all new) in current session: {}'.format(args.mlp_out_dim_cur))
            projector_cur = DINOHead(
                in_dim=args.feat_dim,
                out_dim=args.mlp_out_dim_cur,
                nlayers=args.num_mlp_layers,
                hidden_dim=args.projection_hidden_dim,
                bottleneck_dim=args.projection_bottleneck_dim,
            )
            args.logger.info('transferring classification head of seen classes...')
            projector_cur.last_layer.weight_v.data[:args.num_seen_classes] = projector_pre.last_layer.weight_v.data[:args.num_seen_classes]   # NOTE!!!
            projector_cur.last_layer.weight_g.data[:args.num_seen_classes] = projector_pre.last_layer.weight_g.data[:args.num_seen_classes]   # NOTE!!!
            projector_cur.last_layer.weight.data[:args.num_seen_classes] = projector_pre.last_layer.weight.data[:args.num_seen_classes]   # NOTE!!!
            # initialize new class heads
            #############################################
            online_session_train_dataset_for_new_head_init = deepcopy(online_session_train_dataset)
            online_session_train_dataset_for_new_head_init.old_unlabelled_dataset.transform = test_transform   # NOTE!!!
            online_session_train_dataset_for_new_head_init.novel_unlabelled_dataset.transform = test_transform   # NOTE!!!
            new_head_init_generator = build_dataloader_generator(args.seed + 10000 + session * 10 + 2)

            online_session_train_loader_for_new_head_init = DataLoader(
                online_session_train_dataset_for_new_head_init,
                num_workers=args.num_workers_test,
                batch_size=args.eval_batch_size,
                shuffle=False,
                pin_memory=False,
                generator=new_head_init_generator,
                worker_init_fn=seed_dataloader_worker,
                collate_fn=collate_fn,
            )
            if args.init_new_head:
                new_head = get_kmeans_centroid_for_new_head(model_pre, online_session_train_loader_for_new_head_init, args, device)   # torch.Size([10, 768])
                norm_new_head_weight_v = torch.norm(projector_cur.last_layer.weight_v.data[args.num_seen_classes:], dim=-1).mean()
                norm_new_head_weight = torch.norm(projector_cur.last_layer.weight.data[args.num_seen_classes:], dim=-1).mean()
                new_head_weight_v = new_head * norm_new_head_weight_v
                new_head_weight = new_head * norm_new_head_weight
                args.logger.info('initializing classification head of unseen novel classes...')
                projector_cur.last_layer.weight_v.data[args.num_seen_classes:] = new_head_weight_v.data   # NOTE!!!   # copy
                projector_cur.last_layer.weight.data[args.num_seen_classes:] = new_head_weight.data   # NOTE!!!
            ##############################################

            model_cur = nn.Sequential(backbone_cur, projector_cur)   # NOTE!!! backbone_cur
            args.logger.info('incremental classifier heads from {} to {}'.format(len(model_pre[1].last_layer.weight_v), len(model_cur[1].last_layer.weight_v)))
            model_cur.to(device)
            ####################################################################################################################
            ####################################################################################################################

            '''compute prototypes offline (session = 0)'''
            if session == 0:
                args.logger.info('Before Train: compute offline prototypes and radius from {} classes with the best model...'.format(args.num_labeled_classes))
                offline_session_train_dataset_for_proto_aug = deepcopy(_offline_session_train_dataset)
                offline_session_train_dataset_for_proto_aug.transform = test_transform
                offline_proto_generator = build_dataloader_generator(args.seed + 10000 + session * 10 + 3)

                offline_session_train_loader_for_proto_aug = DataLoader(
                    offline_session_train_dataset_for_proto_aug,
                    num_workers=args.num_workers_test,
                    batch_size=args.eval_batch_size,
                    shuffle=False,
                    pin_memory=False,
                    generator=offline_proto_generator,
                    worker_init_fn=seed_dataloader_worker,
                    collate_fn=collate_fn,
                )
                # NOTE!!! use model_pre && offline_session_train_loader
                proto_aug_manager.update_prototypes_offline(model_pre, offline_session_train_loader_for_proto_aug, args.num_labeled_classes)
                save_path = os.path.join(args.model_dir, 'ProtoAugDict' + '_offline' + f'.pt')
                args.logger.info('Saving ProtoAugDict to {}.'.format(save_path))
                proto_aug_manager.save_proto_aug_dict(save_path)

            # ----------------------
            # TRAIN
            # ----------------------
            train_online(model_cur, model_pre, proto_aug_manager, online_session_train_loader, online_session_test_loader, session+1, args)

            '''compute prototypes online after train (session > 0)'''
            #############################################################################################################
            selection_name = (
                'final' if args.online_checkpoint_selection == 'final_epoch' else 'best'
            )
            args.logger.info(
                'After Train: update online prototypes from {} to {} classes with '
                'the {} model...'.format(
                    args.num_seen_classes,
                    args.num_labeled_classes + args.num_cur_novel_classes,
                    selection_name,
                )
            )
            load_dir_online_selected = online_checkpoint_path(args, session + 1)
            args.logger.info(
                'loading selected checkpoint from current online session: '
                + load_dir_online_selected
            )
            if not os.path.exists(load_dir_online_selected):
                raise FileNotFoundError(
                    f"Current online selected checkpoint not found: "
                    f"{load_dir_online_selected}. Session-{session + 1} did not "
                    "produce its configured checkpoint."
                )
            load_dict = torch.load(load_dir_online_selected)
            validate_checkpoint_metadata(load_dict, args, load_dir_online_selected)
            model_cur.load_state_dict(load_dict['model'])
            proto_aug_manager.update_prototypes_online(model_cur, online_session_train_loader_for_new_head_init, 
                                                       args.num_seen_classes, args.num_labeled_classes + args.num_cur_novel_classes)
            save_path = os.path.join(args.model_dir, 'ProtoAugDict' + '_session-' + str(session+1) + f'.pt')
            args.logger.info('Saving ProtoAugDict to {}.'.format(save_path))
            proto_aug_manager.save_proto_aug_dict(save_path)

            '''save results dict after each session'''
            selected_acc_list_dict = {
                'checkpoint_selection': args.online_checkpoint_selection,
                'selected_test_acc_all_list': args.best_test_acc_all_list,
                'selected_test_acc_old_list': args.best_test_acc_old_list,
                'selected_test_acc_new_list': args.best_test_acc_new_list,
                'selected_test_acc_soft_all_list': args.best_test_acc_soft_all_list,
                'selected_test_acc_seen_list': args.best_test_acc_seen_list,
                'selected_test_acc_unseen_list': args.best_test_acc_unseen_list,
                'selected_test_classification_metrics_list': args.best_test_classification_metrics_list,
                # Legacy aliases retained inside the payload for older readers.
                'best_test_acc_all_list': args.best_test_acc_all_list,
                'best_test_acc_old_list': args.best_test_acc_old_list,
                'best_test_acc_new_list': args.best_test_acc_new_list,
                'best_test_acc_soft_all_list': args.best_test_acc_soft_all_list,
                'best_test_acc_seen_list': args.best_test_acc_seen_list,
                'best_test_acc_unseen_list': args.best_test_acc_unseen_list,
                'best_test_classification_metrics_list': args.best_test_classification_metrics_list,
            }
            selection_file_tag = (
                'final' if args.online_checkpoint_selection == 'final_epoch' else 'best'
            )
            save_results_path = os.path.join(
                args.model_dir,
                selection_file_tag + '_acc_list_session-' + str(session + 1) + '.pt',
            )
            args.logger.info(
                'Saving selected online results to {}.'.format(save_results_path)
            )
            torch.save(selected_acc_list_dict, save_results_path)

        # print final results
        args.logger.info('\n\n==================== print final results over {} continual sessions ===================='.format(args.continual_session_num))
        for session in range(args.continual_session_num):
            classification_metrics = args.best_test_classification_metrics_list[session]
            args.logger.info(
                f'Session-{session+1}: All (Hard): {args.best_test_acc_all_list[session]:.4f} '
                f'Old: {args.best_test_acc_old_list[session]:.4f} New: {args.best_test_acc_new_list[session]:.4f} '
                f'| All (Soft): {args.best_test_acc_soft_all_list[session]:.4f} '
                f'Seen: {args.best_test_acc_seen_list[session]:.4f} '
                f'Unseen: {args.best_test_acc_unseen_list[session]:.4f} '
                f'| MeanClassAcc: {classification_metrics["mean_class_accuracy"]:.4f} '
                f'MacroF1: {classification_metrics["macro_f1"]:.4f}'
            )
        for session in range(args.continual_session_num):
            classification_metrics = args.best_test_classification_metrics_list[session]
            print(
                f'Session-{session+1}: All (Hard): {args.best_test_acc_all_list[session]:.4f} '
                f'Old: {args.best_test_acc_old_list[session]:.4f} New: {args.best_test_acc_new_list[session]:.4f} '
                f'| All (Soft): {args.best_test_acc_soft_all_list[session]:.4f} '
                f'Seen: {args.best_test_acc_seen_list[session]:.4f} '
                f'Unseen: {args.best_test_acc_unseen_list[session]:.4f} '
                f'| MeanClassAcc: {classification_metrics["mean_class_accuracy"]:.4f} '
                f'MacroF1: {classification_metrics["macro_f1"]:.4f}'
            )

    else:
        raise NotImplementedError
