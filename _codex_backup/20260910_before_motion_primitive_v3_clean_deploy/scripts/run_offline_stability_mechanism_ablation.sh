#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python}"
GPU="${GPU:-0}"
NPZ_PATH="${NPZ_PATH:-processed/uschad_w512_s256_train17stats/uschad_windows.npz}"
RUN_ID="${RUN_ID:-stability_mechanisms_w512}"
SEEDS=(0 5 50 500)
CONFIGS=(control bn_freeze10 late_lr_drop15 aux_decay15_30 head512)
DIAGNOSTIC_EPOCHS="5,10,15,30,60,100"

usage() {
  cat <<'EOF'
Usage: bash scripts/run_offline_stability_mechanism_ablation.sh [options]

Options:
  --seeds "0 5 50 500"       Seeds to run.
  --configs "control head512"
                              Single-variable configurations to run.
  --gpu ID                    CUDA device id. Default: 0.
  --npz PATH                  Corrected USC-HAD w512 NPZ path.
  --run-id ID                 Unique result subdirectory name.
  -h, --help                  Show this help.

Configurations:
  control             Released-code static loss and original cosine LR.
  bn_freeze10         Freeze BatchNorm running statistics from epoch 10.
  late_lr_drop15      Preserve the original schedule, then multiply LR by 0.1
                      once at epoch 15.
  aux_decay15_30      Linearly decay cluster and InfoNCE weights from their
                      original values at epoch 15 to zero at epoch 30.
  head512             Reduce projection hidden width from 2048 to 512.

All runs are validation-only, use four released-code loss weights, and record
first-batch gradient diagnostics at epochs 5, 10, 15, 30, 60, and 100.
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
if [[ ! -f "$NPZ_PATH" ]]; then
  echo "NPZ not found: $NPZ_PATH" >&2
  exit 1
fi

RESULT_ROOT="results/offline_stability_mechanisms/${RUN_ID}"
if [[ -d "$RESULT_ROOT" ]] && [[ -n "$(find "$RESULT_ROOT" -type f -print -quit)" ]]; then
  echo "Result directory is not empty: $RESULT_ROOT" >&2
  echo "Use a new --run-id to avoid mixing repeated runs." >&2
  exit 1
fi
mkdir -p "$RESULT_ROOT"

"$PYTHON_BIN" -B check_uschad_dataset.py \
  --uschad_npz_path "$NPZ_PATH" \
  --uschad_window_size 512 \
  --har_in_channels 6 \
  --uschad_split_mode subject \
  --uschad_train_subjects "1,2,3,4,5,6,7" \
  --offline_val_subjects "8,9" \
  --uschad_test_subjects "10,11,12,13,14" \
  --online_old_seen_num 20 \
  --online_novel_unseen_num 50 \
  --online_novel_seen_num 20 \
  --har_aug_mode weak_strong \
  --har_weak_jitter_std 0 \
  --har_strong_jitter_std 0 \
  --har_weak_scale_std 0.10 \
  --har_strong_scale_std 0.20 \
  --har_time_mask_ratio 0 \
  --batch_size 8 \
  --num_workers 0 \
  --seed 0

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
  --num_old_classes 6
  --prop_train_labels 0.8
  --num_novel_classes_per_session 2
  --online_old_seen_num 20
  --online_novel_unseen_num 50
  --online_novel_seen_num 20
  --batch_size 64
  --eval_batch_size 64
  --num_workers 0
  --num_workers_test 0
  --epochs_offline 100
  --offline_early_stop_patience 0
  --offline_selection_metric macro_f1
  --offline_skip_final_test
  --offline_gradient_diagnostic_epochs "$DIAGNOSTIC_EPOCHS"
  --eval_funcs v2
  --image_aug_mode off
  --har_aug_mode weak_strong
  --har_weak_jitter_std 0
  --har_strong_jitter_std 0
  --har_weak_scale_std 0.10
  --har_strong_scale_std 0.20
  --har_time_mask_ratio 0
  --har_dropout 0.0
  --lr 0.1
  --momentum 0.9
  --weight_decay 5e-4
  --temperature 0.1
  --memax_weight 1
  --warmup_teacher_temp 0.07
  --teacher_temp 0.04
  --warmup_teacher_temp_epochs 30
  --offline_cls_weight 0.35
  --offline_cluster_weight 0.65
  --offline_contrast_weight 0.65
  --offline_supcon_weight 0.35
)

run_config() {
  local config="$1"
  shift
  local extra_args=("$@")

  for seed in "${SEEDS[@]}"; do
    echo "================================================================"
    echo "config=${config}, seed=${seed}, epochs=100"
    echo "extra args: ${extra_args[*]}"
    echo "================================================================"
    CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON_BIN" -B train_happy.py \
      "${COMMON[@]}" \
      "${extra_args[@]}" \
      --seed "$seed" \
      --exp_root "${RESULT_ROOT}/${config}"
  done
}

for config in "${CONFIGS[@]}"; do
  case "$config" in
    control)
      run_config "$config"
      ;;
    bn_freeze10)
      run_config "$config" --offline_bn_freeze_epoch 10
      ;;
    late_lr_drop15)
      run_config "$config" --offline_lr_drop_epoch 15 --offline_lr_drop_factor 0.1
      ;;
    aux_decay15_30)
      run_config "$config" \
        --offline_aux_decay_start_epoch 15 \
        --offline_aux_decay_end_epoch 30 \
        --offline_aux_final_scale 0.0
      ;;
    head512)
      run_config "$config" --projection_hidden_dim 512
      ;;
    *)
      echo "Unknown configuration: $config" >&2
      usage >&2
      exit 2
      ;;
  esac
done

"$PYTHON_BIN" -B scripts/summarize_noaug_regularization.py \
  --root "$RESULT_ROOT" \
  --output-prefix stability_mechanisms

echo "Completed offline stability-mechanism ablation."
echo "Results: ${RESULT_ROOT}"
