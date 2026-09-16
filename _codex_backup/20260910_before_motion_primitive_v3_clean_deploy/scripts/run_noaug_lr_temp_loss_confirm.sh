#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python}"
GPU="${GPU:-0}"
NPZ_PATH="${NPZ_PATH:-processed/uschad_w512_s256/uschad_windows.npz}"
RUN_ID="${RUN_ID:-noaug_lr_temp_loss_4seeds}"
SEEDS=(0 5 50 500)
SKIP_DATA_CHECK=0

usage() {
  cat <<'EOF'
Usage: bash scripts/run_noaug_lr_temp_loss_confirm.sh [options]

Options:
  --seeds "0 5 50 500"  Seeds to run. Default: "0 5 50 500".
  --gpu ID               CUDA device id. Default: 0.
  --npz PATH             USC-HAD NPZ path.
  --run-id ID            Result subdirectory name.
  --skip-data-check      Skip the pre-run leakage and shape check.
  -h, --help             Show this help.

This script is validation-only and never evaluates the final test set.
It runs three configurations on the weight_decay=5e-4 no-augmentation base:
  lr_005
  flat_teacher_temp
  supervised_weighted
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --seeds)
      read -r -a SEEDS <<< "$2"
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

RESULT_ROOT="results/noaug_parameter_confirm/${RUN_ID}"
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
  --har_aug_mode none
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
    echo "Running config=${config}, seed=${seed}"
    echo "============================================================"

    CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON_BIN" -B train_happy.py \
      "${COMMON[@]}" \
      --seed "$seed" \
      --exp_root "${RESULT_ROOT}/${config}" \
      "$@"
  done
}

# Learning-rate ablation. The existing reference uses lr=0.1.
run_config lr_005 \
  --lr 0.05

# Teacher-temperature ablation. The existing reference uses 0.07 -> 0.04
# over 30 epochs; this configuration keeps teacher temperature at 0.07.
run_config flat_teacher_temp \
  --warmup_teacher_temp 0.07 \
  --teacher_temp 0.07 \
  --warmup_teacher_temp_epochs 0

# Loss-weight ablation. The existing reference uses legacy weights:
# cls/cluster/contrast/SupCon = 0.35/0.65/0.65/0.35.
run_config supervised_weighted \
  --offline_cls_weight 0.65 \
  --offline_cluster_weight 0.35 \
  --offline_contrast_weight 0.35 \
  --offline_supcon_weight 0.65

"$PYTHON_BIN" -B scripts/summarize_noaug_regularization.py \
  --root "$RESULT_ROOT"

echo "Completed validation-only parameter confirmation."
echo "New results: ${RESULT_ROOT}"
echo "Reference: results/noaug_regularization_screen/noaug_confirm_4seeds_20260728/wd_5e4_offline"
