#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python}"
GPU="${GPU:-0}"
NPZ_PATH="${NPZ_PATH:-processed/uschad_w512_s256/uschad_windows.npz}"
RUN_ID="${RUN_ID:-final_scale_h_w512}"
SEEDS=(0 5 50 500)
SKIP_DATA_CHECK=0

usage() {
  cat <<'EOF'
Usage: bash scripts/run_final_scale_h_offline_online.sh [options]

Options:
  --seeds "0 5 50 500"  Seeds to run.
  --gpu ID               CUDA device id. Default: 0.
  --npz PATH             USC-HAD w512 NPZ path.
  --run-id ID            Result subdirectory name.
  --skip-data-check      Skip the dataset/leakage preflight for each seed.
  -h, --help             Show this help.
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

RESULT_ROOT="results/final_scale_h_offline_online/${RUN_ID}"
mkdir -p "$RESULT_ROOT"

COMMON=(
  --dataset_name uschad
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
  --eval_funcs v2
  --image_aug_mode off
  --har_aug_mode weak_strong
  --har_weak_jitter_std 0
  --har_strong_jitter_std 0
  --har_weak_scale_std 0.10
  --har_strong_scale_std 0.20
  --har_time_mask_ratio 0
  --har_dropout 0.0
  --momentum 0.9
  --weight_decay 5e-4
  --temperature 0.1
  --sup_weight 0.35
  --memax_weight 1
)

for seed in "${SEEDS[@]}"; do
  echo "================================================================"
  echo "Final scale_h pipeline: seed=${seed}"
  echo "================================================================"

  if [[ "$SKIP_DATA_CHECK" -eq 0 ]]; then
    "$PYTHON_BIN" -B check_uschad_dataset.py \
      --uschad_npz_path "$NPZ_PATH" \
      --uschad_window_size 512 \
      --uschad_split_mode subject \
      --uschad_train_subjects "1,2,3,4,5,6,7" \
      --offline_val_subjects "8,9" \
      --uschad_test_subjects "10,11,12,13,14" \
      --online_old_seen_num 20 \
      --online_novel_unseen_num 50 \
      --online_novel_seen_num 20 \
      --batch_size 8 \
      --num_workers 0 \
      --seed "$seed"
  fi

  seed_root="${RESULT_ROOT}/seed_${seed}"

  CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON_BIN" -B train_happy.py \
    "${COMMON[@]}" \
    --train_session offline \
    --epochs_offline 100 \
    --offline_early_stop_patience 0 \
    --offline_selection_metric macro_f1 \
    --lr 0.1 \
    --warmup_teacher_temp 0.07 \
    --teacher_temp 0.04 \
    --warmup_teacher_temp_epochs 30 \
    --seed "$seed" \
    --exp_root "$seed_root"

  offline_dir=""
  for candidate in "${seed_root}_offline/uschad"/Old6_Ratio0.8_*; do
    if [[ -d "$candidate" ]] && {
      [[ -z "$offline_dir" ]] || [[ "$candidate" -nt "$offline_dir" ]]
    }; then
      offline_dir="$candidate"
    fi
  done

  if [[ -z "$offline_dir" ]]; then
    echo "No offline experiment directory found under ${seed_root}_offline/uschad" >&2
    exit 1
  fi

  offline_id="$(basename "$offline_dir")"
  offline_checkpoint="${offline_dir}/checkpoints/model_best.pt"
  if [[ ! -f "$offline_checkpoint" ]]; then
    echo "Offline best checkpoint not found: $offline_checkpoint" >&2
    exit 1
  fi

  echo "Using offline best checkpoint: $offline_checkpoint"

  CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON_BIN" -B train_happy.py \
    "${COMMON[@]}" \
    --train_session online \
    --load_offline_id "$offline_id" \
    --epochs_online_per_session 30 \
    --lr 0.01 \
    --warmup_teacher_temp 0.05 \
    --teacher_temp 0.05 \
    --warmup_teacher_temp_epochs 10 \
    --memax_old_new_weight 1 \
    --memax_old_in_weight 1 \
    --memax_new_in_weight 1 \
    --proto_aug_weight 1 \
    --feat_distill_weight 1 \
    --radius_scale 1.0 \
    --hardness_temp 0.1 \
    --init_new_head \
    --seed "$seed" \
    --exp_root "$seed_root"
done

echo "Completed final scale_h offline + online pipeline."
echo "Results: ${RESULT_ROOT}"
