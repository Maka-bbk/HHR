#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python}"
GPU="${GPU:-0}"
NPZ_PATH="${NPZ_PATH:-processed/uschad_w512_s256_train17stats/uschad_windows.npz}"
RUN_ID="${RUN_ID:-loss_screen_seed0}"
SEEDS=(0)
CONFIGS=(
  released_code
  released_no_cluster_sum2
  paper_eq2
  paper_no_infonce_sum2
  paper_no_supcon_sum2
  ce_only_sum2
)
EPOCHS=100
EARLY_STOP_PATIENCE=0
FINAL_TEST=0
SKIP_DATA_CHECK=0

usage() {
  cat <<'EOF'
Usage: bash scripts/run_offline_early_overfit_ablation.sh [options]

Options:
  --seeds "0 5 50 500"       Seeds to run. Default: "0".
  --configs "paper_eq2 ce_only"
                              Loss configurations to run.
  --gpu ID                    CUDA device id. Default: 0.
  --npz PATH                  Corrected USC-HAD w512 NPZ path.
  --run-id ID                 Result subdirectory name.
  --epochs N                  Offline epochs. Default: 100.
  --early-stop-patience N     0 disables early stopping. Default: 0.
  --final-test                Evaluate the held-out test set. Requires exactly
                              one configuration; use only after validation selection.
  --skip-data-check           Skip the dataset/leakage preflight.
  -h, --help                  Show this help.

Configurations (cls / cluster / InfoNCE / SupCon):
  released_code        0.35 / 0.65 / 0.65 / 0.35
  released_no_cluster_sum2
                       0.518519 / 0.00 / 0.962963 / 0.518519
  paper_eq2            1.00 / 0.00 / 0.65 / 0.35
  paper_no_infonce_sum2
                       1.481481 / 0.00 / 0.00 / 0.518519
  paper_no_supcon_sum2 1.212121 / 0.00 / 0.787879 / 0.00
  ce_only_sum2         2.00 / 0.00 / 0.00 / 0.00

The *_sum2 ablations preserve the remaining loss ratios and rescale their
coefficient sum to 2.0, matching released_code and paper_eq2. This controls
the crude total-loss scaling confound; per-loss gradient norms can still differ.
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
    --epochs)
      EPOCHS="$2"
      shift 2
      ;;
    --early-stop-patience)
      EARLY_STOP_PATIENCE="$2"
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

if [[ "$EPOCHS" -lt 1 ]]; then
  echo "--epochs must be at least 1." >&2
  exit 2
fi
if [[ "$EARLY_STOP_PATIENCE" -lt 0 ]]; then
  echo "--early-stop-patience must be non-negative." >&2
  exit 2
fi
if [[ "$FINAL_TEST" -eq 1 && "${#CONFIGS[@]}" -ne 1 ]]; then
  echo "--final-test requires exactly one --configs value." >&2
  exit 2
fi
if [[ ! -f "$NPZ_PATH" ]]; then
  echo "NPZ not found: $NPZ_PATH" >&2
  echo "Generate the train-subject-only normalization NPZ first." >&2
  exit 1
fi

RESULT_ROOT="results/offline_early_overfit/${RUN_ID}"
if [[ -d "$RESULT_ROOT" ]] && [[ -n "$(find "$RESULT_ROOT" -type f -print -quit)" ]]; then
  echo "Result directory is not empty: $RESULT_ROOT" >&2
  echo "Use a new --run-id to avoid mixing repeated runs." >&2
  exit 1
fi
mkdir -p "$RESULT_ROOT"

if [[ "$SKIP_DATA_CHECK" -eq 0 ]]; then
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
  --epochs_offline "$EPOCHS"
  --offline_early_stop_patience "$EARLY_STOP_PATIENCE"
  --offline_selection_metric macro_f1
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
)

TEST_ARGS=(--offline_skip_final_test)
if [[ "$FINAL_TEST" -eq 1 ]]; then
  TEST_ARGS=()
fi

run_config() {
  local config="$1"
  local cls_weight="$2"
  local cluster_weight="$3"
  local contrast_weight="$4"
  local supcon_weight="$5"

  for seed in "${SEEDS[@]}"; do
    echo "================================================================"
    echo "config=${config}, seed=${seed}, epochs=${EPOCHS}, patience=${EARLY_STOP_PATIENCE}"
    echo "loss weights: cls=${cls_weight}, cluster=${cluster_weight}, contrast=${contrast_weight}, supcon=${supcon_weight}"
    echo "================================================================"

    CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON_BIN" -B train_happy.py \
      "${COMMON[@]}" \
      "${TEST_ARGS[@]}" \
      --offline_cls_weight "$cls_weight" \
      --offline_cluster_weight "$cluster_weight" \
      --offline_contrast_weight "$contrast_weight" \
      --offline_supcon_weight "$supcon_weight" \
      --seed "$seed" \
      --exp_root "${RESULT_ROOT}/${config}"
  done
}

for config in "${CONFIGS[@]}"; do
  case "$config" in
    released_code)
      run_config "$config" 0.35 0.65 0.65 0.35
      ;;
    released_no_cluster_sum2)
      run_config "$config" 0.518519 0.0 0.962963 0.518519
      ;;
    paper_eq2)
      run_config "$config" 1.0 0.0 0.65 0.35
      ;;
    paper_no_infonce_sum2)
      run_config "$config" 1.481481 0.0 0.0 0.518519
      ;;
    paper_no_supcon_sum2)
      run_config "$config" 1.212121 0.0 0.787879 0.0
      ;;
    ce_only_sum2)
      run_config "$config" 2.0 0.0 0.0 0.0
      ;;
    *)
      echo "Unknown loss configuration: $config" >&2
      usage >&2
      exit 2
      ;;
  esac
done

"$PYTHON_BIN" -B scripts/summarize_noaug_regularization.py \
  --root "$RESULT_ROOT" \
  --output-prefix early_overfit

echo "Completed offline early-overfit experiment."
echo "Results: ${RESULT_ROOT}"
