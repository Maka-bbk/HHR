#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python}"
GPU="${GPU:-0}"
NPZ_PATH="${NPZ_PATH:-processed/uschad_w256_s128_train17stats/uschad_windows.npz}"
RUN_ID="${RUN_ID:-trial_attention_w256_screen}"
SEEDS=(0 5 50 500)
CONFIGS=(
  mean_full_full
  gated_attention_full_full
  mean_full_random_crop
  gated_attention_full_random_crop
)
WINDOW_EPOCHS=60
TRIAL_EPOCHS=60
ENCODER_ROOT=""
FINAL_TEST=0
SKIP_DATA_CHECK=0

usage() {
  cat <<'EOF'
Usage: bash scripts/run_trial_attention_ablation_w256.sh [options]

Options:
  --seeds "0 5 50 500"       Seeds to run.
  --configs "mean_full_full gated_attention_full_full"
                              Trial configurations to run.
  --gpu ID                    CUDA device id. Default: 0.
  --npz PATH                  Corrected USC-HAD w256/s128 NPZ path.
  --run-id ID                 Unique result subdirectory name.
  --window-epochs N           Window encoder pretraining epochs. Default: 60.
  --trial-epochs N            Trial training epochs. Default: 60.
  --encoder-root PATH         Reuse window checkpoints from an earlier result
                              root and skip window pretraining.
  --final-test                Evaluate the held-out test set once. This requires
                              exactly one trial configuration.
  --skip-data-check           Skip both trial-view dataset preflight checks.
  -h, --help                  Show this help.

Only full_full and full_random_crop are implemented. There are no fixed
prefix/suffix or first/last two-thirds configurations.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --seeds)
      read -r -a SEEDS <<< "$2"
      shift 2
      ;;
    --configs)
      read -r -a CONFIGS <<< "$2"
      shift 2
      ;;
    --gpu)
      GPU="$2"
      shift 2
      ;;
    --npz)
      NPZ_PATH="$2"
      shift 2
      ;;
    --run-id)
      RUN_ID="$2"
      shift 2
      ;;
    --window-epochs)
      WINDOW_EPOCHS="$2"
      shift 2
      ;;
    --trial-epochs)
      TRIAL_EPOCHS="$2"
      shift 2
      ;;
    --encoder-root)
      ENCODER_ROOT="$2"
      shift 2
      ;;
    --final-test)
      FINAL_TEST=1
      shift
      ;;
    --skip-data-check)
      SKIP_DATA_CHECK=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ "${#SEEDS[@]}" -lt 1 || "${#CONFIGS[@]}" -lt 1 ]]; then
  echo "At least one seed and one configuration are required." >&2
  exit 2
fi
if [[ "$WINDOW_EPOCHS" -lt 1 || "$TRIAL_EPOCHS" -lt 1 ]]; then
  echo "Epoch counts must be positive." >&2
  exit 2
fi
WINDOW_WARMUP_EPOCHS=30
TRIAL_WARMUP_EPOCHS=30
TRIAL_FREEZE_EPOCHS=5
if [[ "$WINDOW_EPOCHS" -lt "$WINDOW_WARMUP_EPOCHS" ]]; then
  WINDOW_WARMUP_EPOCHS="$WINDOW_EPOCHS"
fi
if [[ "$TRIAL_EPOCHS" -lt "$TRIAL_WARMUP_EPOCHS" ]]; then
  TRIAL_WARMUP_EPOCHS="$TRIAL_EPOCHS"
fi
if [[ "$TRIAL_EPOCHS" -lt "$TRIAL_FREEZE_EPOCHS" ]]; then
  TRIAL_FREEZE_EPOCHS="$TRIAL_EPOCHS"
fi
if [[ "$FINAL_TEST" -eq 1 && "${#CONFIGS[@]}" -ne 1 ]]; then
  echo "--final-test requires exactly one --configs value." >&2
  exit 2
fi
if [[ ! -f "$NPZ_PATH" ]]; then
  echo "NPZ not found: $NPZ_PATH" >&2
  echo "Generate the w256/s128 train-subject-normalized NPZ first." >&2
  exit 1
fi
if [[ -n "$ENCODER_ROOT" && ! -d "$ENCODER_ROOT" ]]; then
  echo "Encoder result root not found: $ENCODER_ROOT" >&2
  exit 1
fi

for config in "${CONFIGS[@]}"; do
  case "$config" in
    mean_full_full|gated_attention_full_full|mean_full_random_crop|gated_attention_full_random_crop)
      ;;
    *)
      echo "Unknown trial configuration: $config" >&2
      usage >&2
      exit 2
      ;;
  esac
done

RESULT_ROOT="results/trial_attention_ablation/${RUN_ID}"
if [[ -d "$RESULT_ROOT" ]] && [[ -n "$(find "$RESULT_ROOT" -type f -print -quit)" ]]; then
  echo "Result directory is not empty: $RESULT_ROOT" >&2
  echo "Use a new --run-id to avoid mixing repeated runs." >&2
  exit 1
fi
mkdir -p "$RESULT_ROOT"

COMMON_DATA=(
  --dataset_name uschad
  --uschad_npz_path "$NPZ_PATH"
  --uschad_window_size 256
  --har_in_channels 6
  --uschad_split_mode subject
  --uschad_train_subjects "1,2,3,4,5,6,7"
  --offline_val_subjects "8,9"
  --uschad_test_subjects "10,11,12,13,14"
  --num_old_classes 6
  --prop_train_labels 0.8
  --num_novel_classes_per_session 2
  --eval_funcs v2
  --image_aug_mode off
)

# Fixed best scale-only augmentation from the preceding window experiments.
COMMON_AUG=(
  --har_aug_mode weak_strong
  --har_weak_jitter_std 0
  --har_strong_jitter_std 0
  --har_weak_scale_std 0.10
  --har_strong_scale_std 0.20
  --har_time_mask_ratio 0
)

# paper_eq2 removes the redundant offline clustering term while preserving
# CE, instance InfoNCE, and supervised contrastive learning.
COMMON_OPT=(
  --har_dropout 0.0
  --momentum 0.9
  --weight_decay 5e-4
  --temperature 0.1
  --memax_weight 1
  --warmup_teacher_temp 0.07
  --teacher_temp 0.04
  --offline_cls_weight 1.0
  --offline_cluster_weight 0.0
  --offline_contrast_weight 0.65
  --offline_supcon_weight 0.35
  --offline_selection_metric macro_f1
  --offline_early_stop_patience 0
  --num_workers 0
  --num_workers_test 0
)

if [[ "$SKIP_DATA_CHECK" -eq 0 ]]; then
  for view_mode in full_full full_random_crop; do
    "$PYTHON_BIN" -B check_uschad_dataset.py \
      --uschad_npz_path "$NPZ_PATH" \
      --uschad_window_size 256 \
      --har_in_channels 6 \
      --uschad_sample_unit trial \
      --trial_view_mode "$view_mode" \
      --trial_crop_ratio 0.6666666666666666 \
      --trial_min_windows 2 \
      --uschad_split_mode subject \
      --uschad_train_subjects "1,2,3,4,5,6,7" \
      --offline_val_subjects "8,9" \
      --uschad_test_subjects "10,11,12,13,14" \
      --online_old_seen_trials 2 \
      --online_novel_unseen_trials 5 \
      --online_novel_seen_trials 2 \
      --har_aug_mode weak_strong \
      --har_weak_jitter_std 0 \
      --har_strong_jitter_std 0 \
      --har_weak_scale_std 0.10 \
      --har_strong_scale_std 0.20 \
      --har_time_mask_ratio 0 \
      --batch_size 8 \
      --num_workers 0 \
      --seed "${SEEDS[0]}"
  done
fi

TRIAL_TEST_ARGS=(--offline_skip_final_test)
if [[ "$FINAL_TEST" -eq 1 ]]; then
  TRIAL_TEST_ARGS=()
fi

find_encoder_checkpoint() {
  local root="$1"
  local seed="$2"
  local search_root="${root}/window_pretrain/seed_${seed}_offline/uschad"
  local checkpoints=()
  if [[ -d "$search_root" ]]; then
    mapfile -t checkpoints < <(
      find "$search_root" -type f -path '*/checkpoints/model_best.pt' -print
    )
  fi
  if [[ "${#checkpoints[@]}" -ne 1 ]]; then
    echo "Expected exactly one seed-${seed} window checkpoint under $search_root; found ${#checkpoints[@]}." >&2
    return 1
  fi
  printf '%s\n' "${checkpoints[0]}"
}

for seed in "${SEEDS[@]}"; do
  if [[ -z "$ENCODER_ROOT" ]]; then
    echo "================================================================"
    echo "Window encoder pretraining: seed=${seed}, window=256"
    echo "================================================================"
    CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON_BIN" -B train_happy.py \
      "${COMMON_DATA[@]}" \
      "${COMMON_AUG[@]}" \
      "${COMMON_OPT[@]}" \
      --train_session offline \
      --uschad_sample_unit window \
      --online_old_seen_num 1 \
      --online_novel_unseen_num 1 \
      --online_novel_seen_num 1 \
      --batch_size 64 \
      --eval_batch_size 64 \
      --epochs_offline "$WINDOW_EPOCHS" \
      --warmup_teacher_temp_epochs "$WINDOW_WARMUP_EPOCHS" \
      --offline_skip_final_test \
      --lr 0.1 \
      --seed "$seed" \
      --exp_root "${RESULT_ROOT}/window_pretrain/seed_${seed}"
    encoder_checkpoint="$(find_encoder_checkpoint "$RESULT_ROOT" "$seed")"
  else
    encoder_checkpoint="$(find_encoder_checkpoint "$ENCODER_ROOT" "$seed")"
  fi

  echo "Using seed-${seed} window encoder: $encoder_checkpoint"

  for config in "${CONFIGS[@]}"; do
    case "$config" in
      mean_full_full)
        pooling=mean
        view_mode=full_full
        ;;
      gated_attention_full_full)
        pooling=gated_attention
        view_mode=full_full
        ;;
      mean_full_random_crop)
        pooling=mean
        view_mode=full_random_crop
        ;;
      gated_attention_full_random_crop)
        pooling=gated_attention
        view_mode=full_random_crop
        ;;
    esac

    echo "================================================================"
    echo "Trial ablation: config=${config}, seed=${seed}"
    echo "pooling=${pooling}, view_mode=${view_mode}"
    echo "================================================================"
    CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON_BIN" -B train_happy.py \
      "${COMMON_DATA[@]}" \
      "${COMMON_AUG[@]}" \
      "${COMMON_OPT[@]}" \
      "${TRIAL_TEST_ARGS[@]}" \
      --train_session offline \
      --uschad_sample_unit trial \
      --trial_pooling "$pooling" \
      --trial_view_mode "$view_mode" \
      --trial_crop_ratio 0.6666666666666666 \
      --trial_min_windows 2 \
      --trial_attention_dim 64 \
      --trial_attention_dropout 0.1 \
      --trial_attention_temperature 1.0 \
      --trial_attention_mean_mix 0.5 \
      --trial_encoder_checkpoint "$encoder_checkpoint" \
      --trial_encoder_freeze_epochs "$TRIAL_FREEZE_EPOCHS" \
      --trial_encoder_lr_scale 0.1 \
      --online_old_seen_trials 2 \
      --online_novel_unseen_trials 5 \
      --online_novel_seen_trials 2 \
      --batch_size 16 \
      --eval_batch_size 16 \
      --epochs_offline "$TRIAL_EPOCHS" \
      --warmup_teacher_temp_epochs "$TRIAL_WARMUP_EPOCHS" \
      --lr 0.01 \
      --seed "$seed" \
      --exp_root "${RESULT_ROOT}/${config}/seed_${seed}"
  done
done

"$PYTHON_BIN" -B scripts/summarize_trial_attention_ablation.py \
  --root "$RESULT_ROOT" \
  --output-prefix trial_attention

echo "Completed trial-attention experiment."
echo "Results: ${RESULT_ROOT}"
