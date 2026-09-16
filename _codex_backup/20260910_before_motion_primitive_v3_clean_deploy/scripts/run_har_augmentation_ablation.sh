#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python}"
GPU="${GPU:-0}"
NPZ_PATH="${NPZ_PATH:-processed/uschad_w512_s256/uschad_windows.npz}"
RUN_ID="${RUN_ID:-har_augmentation_w512_4seeds}"
SEEDS=(0 5 50 500)
CONFIGS=(
  none
  jitter_l jitter_m jitter_h
  scale_l scale_m scale_h
  mask_l mask_m mask_h
  jitter_scale jitter_mask scale_mask
  full
)
SKIP_DATA_CHECK=0

usage() {
  cat <<'EOF'
Usage: bash scripts/run_har_augmentation_ablation.sh [options]

Options:
  --seeds "0 5 50 500"     Seeds to run.
  --configs "none scale_m"  Augmentation configurations to run.
  --gpu ID                  CUDA device id. Default: 0.
  --npz PATH                USC-HAD NPZ path.
  --run-id ID               Result subdirectory name.
  --skip-data-check         Skip the pre-run leakage and shape check.
  -h, --help                Show this help.

The script is validation-only and never evaluates the final test set.
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

RESULT_ROOT="results/har_augmentation_ablation/${RUN_ID}"
mkdir -p "$RESULT_ROOT"

if [[ "$SKIP_DATA_CHECK" -eq 0 ]]; then
  "$PYTHON_BIN" -B check_uschad_dataset.py \
    --uschad_npz_path "$NPZ_PATH" \
    --uschad_window_size 512 \
    --uschad_split_mode subject \
    --uschad_train_subjects "1,2,3,4,5,6,7" \
    --offline_val_subjects "8,9" \
    --uschad_test_subjects "10,11,12,13,14" \
    --online_old_seen_num 10 \
    --online_novel_unseen_num 20 \
    --online_novel_seen_num 10 \
    --batch_size 8 \
    --num_workers 0 \
    --seed 0
fi

# Fixed best no-augmentation base. Only HAR augmentation arguments may be
# overridden by the configurations below.
COMMON=(
  --dataset_name uschad
  --train_session offline
  --uschad_npz_path "$NPZ_PATH"
  --uschad_window_size 512
  --har_in_channels 6
  --uschad_split_mode subject
  --uschad_train_subjects "1,2,3,4,5,6,7"
  --offline_val_subjects "8,9"
  --uschad_test_subjects "10,11,12,13,14"
  --batch_size 64
  --eval_batch_size 64
  --num_workers 0
  --num_workers_test 0
  --epochs_offline 100
  --offline_early_stop_patience 0
  --offline_selection_metric macro_f1
  --offline_skip_final_test
  --image_aug_mode off
  --har_dropout 0.0
  --lr 0.1
  --momentum 0.9
  --weight_decay 5e-4
  --temperature 0.1
  --sup_weight 0.35
  --memax_weight 1
  --warmup_teacher_temp 0.07
  --teacher_temp 0.04
  --warmup_teacher_temp_epochs 30
)

run_config() {
  local config="$1"
  shift

  for seed in "${SEEDS[@]}"; do
    echo "============================================================"
    echo "Running augmentation=${config}, seed=${seed}"
    echo "============================================================"

    CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON_BIN" -B train_happy.py \
      "${COMMON[@]}" \
      --seed "$seed" \
      --exp_root "${RESULT_ROOT}/${config}" \
      "$@"
  done
}

for config in "${CONFIGS[@]}"; do
  case "$config" in
    none)
      run_config none \
        --har_aug_mode none
      ;;
    jitter_l)
      run_config jitter_l \
        --har_aug_mode weak_strong \
        --har_weak_jitter_std 0.005 \
        --har_strong_jitter_std 0.01 \
        --har_weak_scale_std 0 \
        --har_strong_scale_std 0 \
        --har_time_mask_ratio 0
      ;;
    jitter_m)
      run_config jitter_m \
        --har_aug_mode weak_strong \
        --har_weak_jitter_std 0.01 \
        --har_strong_jitter_std 0.02 \
        --har_weak_scale_std 0 \
        --har_strong_scale_std 0 \
        --har_time_mask_ratio 0
      ;;
    jitter_h)
      run_config jitter_h \
        --har_aug_mode weak_strong \
        --har_weak_jitter_std 0.02 \
        --har_strong_jitter_std 0.04 \
        --har_weak_scale_std 0 \
        --har_strong_scale_std 0 \
        --har_time_mask_ratio 0
      ;;
    scale_l)
      run_config scale_l \
        --har_aug_mode weak_strong \
        --har_weak_jitter_std 0 \
        --har_strong_jitter_std 0 \
        --har_weak_scale_std 0.025 \
        --har_strong_scale_std 0.05 \
        --har_time_mask_ratio 0
      ;;
    scale_m)
      run_config scale_m \
        --har_aug_mode weak_strong \
        --har_weak_jitter_std 0 \
        --har_strong_jitter_std 0 \
        --har_weak_scale_std 0.05 \
        --har_strong_scale_std 0.10 \
        --har_time_mask_ratio 0
      ;;
    scale_h)
      run_config scale_h \
        --har_aug_mode weak_strong \
        --har_weak_jitter_std 0 \
        --har_strong_jitter_std 0 \
        --har_weak_scale_std 0.10 \
        --har_strong_scale_std 0.20 \
        --har_time_mask_ratio 0
      ;;
    mask_l)
      run_config mask_l \
        --har_aug_mode weak_strong \
        --har_weak_jitter_std 0 \
        --har_strong_jitter_std 0 \
        --har_weak_scale_std 0 \
        --har_strong_scale_std 0 \
        --har_time_mask_ratio 0.05
      ;;
    mask_m)
      run_config mask_m \
        --har_aug_mode weak_strong \
        --har_weak_jitter_std 0 \
        --har_strong_jitter_std 0 \
        --har_weak_scale_std 0 \
        --har_strong_scale_std 0 \
        --har_time_mask_ratio 0.10
      ;;
    mask_h)
      run_config mask_h \
        --har_aug_mode weak_strong \
        --har_weak_jitter_std 0 \
        --har_strong_jitter_std 0 \
        --har_weak_scale_std 0 \
        --har_strong_scale_std 0 \
        --har_time_mask_ratio 0.20
      ;;
    jitter_scale)
      run_config jitter_scale \
        --har_aug_mode weak_strong \
        --har_weak_jitter_std 0.01 \
        --har_strong_jitter_std 0.02 \
        --har_weak_scale_std 0.05 \
        --har_strong_scale_std 0.10 \
        --har_time_mask_ratio 0
      ;;
    jitter_mask)
      run_config jitter_mask \
        --har_aug_mode weak_strong \
        --har_weak_jitter_std 0.01 \
        --har_strong_jitter_std 0.02 \
        --har_weak_scale_std 0 \
        --har_strong_scale_std 0 \
        --har_time_mask_ratio 0.10
      ;;
    scale_mask)
      run_config scale_mask \
        --har_aug_mode weak_strong \
        --har_weak_jitter_std 0 \
        --har_strong_jitter_std 0 \
        --har_weak_scale_std 0.05 \
        --har_strong_scale_std 0.10 \
        --har_time_mask_ratio 0.10
      ;;
    full)
      run_config full \
        --har_aug_mode weak_strong \
        --har_weak_jitter_std 0.01 \
        --har_strong_jitter_std 0.02 \
        --har_weak_scale_std 0.05 \
        --har_strong_scale_std 0.10 \
        --har_time_mask_ratio 0.10
      ;;
    *)
      echo "Unknown augmentation configuration: $config" >&2
      exit 2
      ;;
  esac
done

"$PYTHON_BIN" -B scripts/summarize_noaug_regularization.py \
  --root "$RESULT_ROOT" \
  --output-prefix augmentation_ablation

echo "Completed validation-only HAR augmentation ablation."
echo "Results: ${RESULT_ROOT}"
