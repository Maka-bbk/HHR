#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python}"
GPU="${GPU:-0}"
NPZ_PATH="${NPZ_PATH:-processed/uschad_w256_s128_train17stats/uschad_windows.npz}"
RUN_ID="${RUN_ID:-trial_mean_robust_max_subject_cv_w256}"
WINDOW_SIZE=256
SEEDS=(0)
FOLDS=(1 2 3 4 5 6 7)
WINDOW_EPOCHS=60
TRIAL_EPOCHS=60
ONLINE_EPOCHS=30
ENCODER_ROOT=""
OUTER_TEST=0
INCLUDE_ONLINE=0
WINDOW_ONLY=0
SKIP_EXISTING=0
SKIP_DATA_CHECK=0
TRIAL_POOLING="mean_robust_max"
ROBUST_QUANTILE=0.90
FUSION_DIM=64
FUSION_DROPOUT=0.0
ATTENTION_DIM=64
ATTENTION_DROPOUT=0.1
ATTENTION_TEMPERATURE=1.0
ATTENTION_MEAN_MIX=0.5
NORM_EPS=1e-6

# Each subject appears exactly once as outer test and once as validation.
TEST_PAIRS=("11,10" "2,13" "3,9" "7,1" "12,8" "5,4" "14,6")
VAL_PAIRS=("2,13" "3,9" "7,1" "12,8" "5,4" "14,6" "11,10")

usage() {
  cat <<'EOF'
Usage: bash scripts/run_trial_mean_robust_max_subject_cv_w256.sh [options]

Options:
  --seeds "0 5 50 500"   Seeds to run. Default: "0".
  --folds "1 2 3 4 5 6 7"
                          Subject folds to run. Default: all seven.
  --gpu ID                CUDA device id. Default: 0.
  --npz PATH              Master USC-HAD w256/s128 NPZ.
  --window-size N         Expected NPZ/encoder window size: 128 or 256.
                          Default: 256. This does not infer from the filename.
  --run-id ID             Unique result directory name.
  --window-epochs N       Fold-specific window pretraining epochs. Default: 60.
  --trial-epochs N        Trial pooling epochs. Default: 60.
  --online-epochs N       Epochs in each online session. Default: 30.
  --encoder-root PATH     Reuse fold/seed-matched window checkpoints.
  --trial-pooling NAME    mean, mean_robust_max, or gated_attention.
                          Default: mean_robust_max.
  --robust-quantile Q     Feature-wise robust maximum quantile. Default: 0.90.
  --fusion-dim N          Low-rank mean/peak fusion width. Default: 64.
  --fusion-dropout P      Mean/peak fusion dropout. Default: 0.0.
  --attention-dim N       Gated-attention hidden width. Default: 64.
  --attention-dropout P   Gated-attention input dropout. Default: 0.1.
  --attention-temp T      Gated-attention softmax temperature. Default: 1.0.
  --attention-mean-mix R  Attention share in mean/attention fusion. Default: 0.5.
  --norm-eps E            Fold z-score minimum std. Default: 1e-6.
  --outer-test            Evaluate every fold's outer test after validation
                          selects the checkpoint. Omit for validation-only runs.
  --include-online        Continue each fold/seed from its offline model_best.pt
                          through all online sessions using final_epoch selection.
  --window-only           Stop after fold-specific window pretraining. This is
                          the minimal source-checkpoint entry for motion encoders.
  --skip-existing         With --window-only, resume a partially completed run by
                          reusing only uniquely identified, fully completed window
                          checkpoints. Incomplete/ambiguous targets fail closed.
  --skip-data-check       Skip fold dataset preflight checks.
  -h, --help              Show this help.

The architecture, pooling-specific parameters, and optimizer settings are fixed before an
outer-test run. Do not tune them from the resulting outer-test aggregate and
then report that same aggregate as an unbiased estimate.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --seeds)
      read -r -a SEEDS <<< "$2"
      shift 2
      ;;
    --folds)
      read -r -a FOLDS <<< "$2"
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
    --window-size)
      WINDOW_SIZE="$2"
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
    --online-epochs)
      ONLINE_EPOCHS="$2"
      shift 2
      ;;
    --encoder-root)
      ENCODER_ROOT="$2"
      shift 2
      ;;
    --trial-pooling)
      TRIAL_POOLING="$2"
      shift 2
      ;;
    --robust-quantile)
      ROBUST_QUANTILE="$2"
      shift 2
      ;;
    --fusion-dim)
      FUSION_DIM="$2"
      shift 2
      ;;
    --fusion-dropout)
      FUSION_DROPOUT="$2"
      shift 2
      ;;
    --attention-dim)
      ATTENTION_DIM="$2"
      shift 2
      ;;
    --attention-dropout)
      ATTENTION_DROPOUT="$2"
      shift 2
      ;;
    --attention-temp)
      ATTENTION_TEMPERATURE="$2"
      shift 2
      ;;
    --attention-mean-mix)
      ATTENTION_MEAN_MIX="$2"
      shift 2
      ;;
    --norm-eps)
      NORM_EPS="$2"
      shift 2
      ;;
    --outer-test)
      OUTER_TEST=1
      shift
      ;;
    --include-online)
      INCLUDE_ONLINE=1
      shift
      ;;
    --window-only)
      WINDOW_ONLY=1
      shift
      ;;
    --skip-existing)
      SKIP_EXISTING=1
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

if [[ "${#SEEDS[@]}" -lt 1 || "${#FOLDS[@]}" -lt 1 ]]; then
  echo "At least one seed and one fold are required." >&2
  exit 2
fi
case "$WINDOW_SIZE" in
  128|256) ;;
  *)
    echo "Window size must be 128 or 256; got: $WINDOW_SIZE" >&2
    exit 2
    ;;
esac
if [[ "$WINDOW_ONLY" -eq 1 && "$INCLUDE_ONLINE" -eq 1 ]]; then
  echo "--window-only cannot be combined with --include-online." >&2
  exit 2
fi
if [[ "$WINDOW_ONLY" -eq 1 && -n "$ENCODER_ROOT" ]]; then
  echo "--window-only must train new checkpoints and cannot use --encoder-root." >&2
  exit 2
fi
if [[ "$SKIP_EXISTING" -eq 1 && "$WINDOW_ONLY" -ne 1 ]]; then
  echo "--skip-existing is supported only with --window-only." >&2
  exit 2
fi
if [[ "$WINDOW_EPOCHS" -lt 1 || "$TRIAL_EPOCHS" -lt 1 || "$ONLINE_EPOCHS" -lt 1 ]]; then
  echo "Epoch counts must be positive." >&2
  exit 2
fi
case "$TRIAL_POOLING" in
  mean|mean_robust_max|gated_attention) ;;
  *)
    echo "Trial pooling must be mean, mean_robust_max, or gated_attention; got: $TRIAL_POOLING" >&2
    exit 2
    ;;
esac
seen_folds=" "
for fold in "${FOLDS[@]}"; do
  if [[ ! "$fold" =~ ^[1-7]$ ]]; then
    echo "Fold ids must be integers from 1 through 7; got: $fold" >&2
    exit 2
  fi
  if [[ "$seen_folds" == *" $fold "* ]]; then
    echo "Duplicate fold id: $fold" >&2
    exit 2
  fi
  seen_folds+="$fold "
done
seen_seeds=" "
for seed in "${SEEDS[@]}"; do
  if [[ ! "$seed" =~ ^-?[0-9]+$ ]]; then
    echo "Seed must be an integer; got: $seed" >&2
    exit 2
  fi
  if [[ "$seen_seeds" == *" $seed "* ]]; then
    echo "Duplicate seed: $seed" >&2
    exit 2
  fi
  seen_seeds+="$seed "
done
if [[ ! -f "$NPZ_PATH" ]]; then
  echo "NPZ not found: $NPZ_PATH" >&2
  exit 1
fi
if [[ -n "$ENCODER_ROOT" && ! -d "$ENCODER_ROOT" ]]; then
  echo "Encoder result root not found: $ENCODER_ROOT" >&2
  exit 1
fi

RESULT_ROOT="results/trial_pooling_subject_cv/${RUN_ID}"
if [[ -d "$RESULT_ROOT" ]] && [[ -n "$(find "$RESULT_ROOT" -mindepth 1 -print -quit)" ]]; then
  if [[ "$SKIP_EXISTING" -ne 1 ]]; then
    echo "Result directory is not empty: $RESULT_ROOT" >&2
    echo "Use a new --run-id to avoid mixing runs, or use --window-only --skip-existing to resume." >&2
    exit 1
  fi
  echo "Resume requested; every existing fold/seed target will be validated before reuse."
fi
mkdir -p "$RESULT_ROOT"

WINDOW_WARMUP_EPOCHS=30
TRIAL_WARMUP_EPOCHS=30
TRIAL_FREEZE_EPOCHS=5
ONLINE_WARMUP_EPOCHS=10
if [[ "$WINDOW_EPOCHS" -lt "$WINDOW_WARMUP_EPOCHS" ]]; then
  WINDOW_WARMUP_EPOCHS="$WINDOW_EPOCHS"
fi
if [[ "$TRIAL_EPOCHS" -lt "$TRIAL_WARMUP_EPOCHS" ]]; then
  TRIAL_WARMUP_EPOCHS="$TRIAL_EPOCHS"
fi
if [[ "$TRIAL_EPOCHS" -lt "$TRIAL_FREEZE_EPOCHS" ]]; then
  TRIAL_FREEZE_EPOCHS="$TRIAL_EPOCHS"
fi
if [[ "$ONLINE_EPOCHS" -lt "$ONLINE_WARMUP_EPOCHS" ]]; then
  ONLINE_WARMUP_EPOCHS="$ONLINE_EPOCHS"
fi

build_train_subjects() {
  local test_csv="$1"
  local val_csv="$2"
  local result=""
  local subject
  for subject in {1..14}; do
    if [[ ",$test_csv," == *",$subject,"* ]]; then
      continue
    fi
    if [[ ",$val_csv," == *",$subject,"* ]]; then
      continue
    fi
    if [[ -n "$result" ]]; then
      result+=","
    fi
    result+="$subject"
  done
  printf '%s\n' "$result"
}

find_encoder_checkpoint() {
  local root="$1"
  local fold_tag="$2"
  local seed="$3"
  local search_root="${root}/fold_${fold_tag}/window_pretrain/seed_${seed}_offline/uschad"
  local checkpoints=()
  if [[ -d "$search_root" ]]; then
    mapfile -t checkpoints < <(
      find "$search_root" -type f -path '*/checkpoints/model_best.pt' -print
    )
  fi
  if [[ "${#checkpoints[@]}" -ne 1 ]]; then
    echo "Expected exactly one fold-${fold_tag}/seed-${seed} window checkpoint under $search_root; found ${#checkpoints[@]}." >&2
    return 1
  fi
  printf '%s\n' "${checkpoints[0]}"
}

find_completed_window_checkpoint() {
  local target_root="$1"
  local expected_epochs="$2"
  local expected_window_size="$3"
  local expected_fold="$4"
  local expected_train_subjects="$5"
  local expected_val_subjects="$6"
  local expected_test_subjects="$7"
  local expected_npz="$8"
  local checkpoints=()

  if [[ ! -d "$target_root" ]]; then
    echo "Existing window target is not a directory: $target_root" >&2
    return 1
  fi
  mapfile -t checkpoints < <(
    find "$target_root" -type f -path '*/checkpoints/model_best.pt' -print
  )
  if [[ "${#checkpoints[@]}" -ne 1 ]]; then
    echo "Refusing to resume incomplete/ambiguous window target $target_root: expected exactly one model_best.pt; found ${#checkpoints[@]}." >&2
    return 1
  fi

  local best_checkpoint="${checkpoints[0]}"
  local experiment_dir="${best_checkpoint%/checkpoints/model_best.pt}"
  local final_checkpoint="${experiment_dir}/checkpoints/model.pt"
  local epoch_metrics="${experiment_dir}/offline_epoch_metrics.jsonl"
  local best_metrics="${experiment_dir}/offline_best_validation_classification_metrics.json"
  local required
  for required in "$final_checkpoint" "$epoch_metrics" "$best_metrics"; do
    if [[ ! -s "$required" ]]; then
      echo "Refusing to resume incomplete window target; required completed-run artifact is missing or empty: $required" >&2
      return 1
    fi
  done

  "$PYTHON_BIN" -B - \
    "$best_checkpoint" "$final_checkpoint" "$epoch_metrics" "$best_metrics" \
    "$expected_epochs" "$expected_window_size" "$expected_fold" \
    "$expected_train_subjects" "$expected_val_subjects" \
    "$expected_test_subjects" "$expected_npz" <<'PY'
import json
import sys
from pathlib import Path

import torch

(
    best_path,
    final_path,
    epoch_metrics_path,
    best_metrics_path,
    expected_epochs,
    expected_window_size,
    expected_fold,
    expected_train_subjects,
    expected_val_subjects,
    expected_test_subjects,
    expected_npz,
) = sys.argv[1:]
expected_epochs = int(expected_epochs)
expected_window_size = int(expected_window_size)
expected_fold = int(expected_fold)


def load_checkpoint(path):
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location="cpu")
    if not isinstance(checkpoint, dict) or not isinstance(checkpoint.get("model"), dict):
        raise RuntimeError(f"Invalid checkpoint payload: {path}")
    return checkpoint


def subject_ids(value):
    return sorted(int(item) for item in value.split(",") if item)


def validate_metadata(checkpoint, path):
    metadata = checkpoint.get("experiment_metadata")
    if not isinstance(metadata, dict):
        raise RuntimeError(f"Checkpoint lacks experiment_metadata: {path}")
    expected = {
        "uschad_sample_unit": "window",
        "uschad_window_size": expected_window_size,
        "uschad_cv_fold": expected_fold,
        "uschad_train_subjects": subject_ids(expected_train_subjects),
        "offline_val_subjects": subject_ids(expected_val_subjects),
        "uschad_test_subjects": subject_ids(expected_test_subjects),
        "uschad_recompute_norm_from_train_subjects": True,
    }
    mismatches = [
        f"{key}={metadata.get(key)!r} expected {value!r}"
        for key, value in expected.items()
        if metadata.get(key) != value
    ]
    recorded_npz = metadata.get("uschad_npz_path")
    if not recorded_npz or Path(recorded_npz).resolve() != Path(expected_npz).resolve():
        mismatches.append(
            f"uschad_npz_path={recorded_npz!r} expected {expected_npz!r}"
        )
    if mismatches:
        raise RuntimeError(
            f"Checkpoint metadata mismatch for {path}: " + "; ".join(mismatches)
        )


best = load_checkpoint(best_path)
final = load_checkpoint(final_path)
validate_metadata(best, best_path)
validate_metadata(final, final_path)
if int(final.get("epoch", -1)) != expected_epochs:
    raise RuntimeError(
        f"Final checkpoint epoch={final.get('epoch')!r}; expected {expected_epochs}."
    )
best_epoch = int(best.get("epoch", -1))
if not 1 <= best_epoch <= expected_epochs:
    raise RuntimeError(
        f"Best checkpoint epoch={best.get('epoch')!r}; expected 1..{expected_epochs}."
    )

with open(epoch_metrics_path, "r", encoding="utf-8") as handle:
    epoch_records = [json.loads(line) for line in handle if line.strip()]
observed_epochs = [int(record.get("epoch", -1)) for record in epoch_records]
if observed_epochs != list(range(1, expected_epochs + 1)):
    raise RuntimeError(
        "Offline epoch metrics are incomplete or non-contiguous: "
        f"observed={observed_epochs}, expected=1..{expected_epochs}."
    )
with open(best_metrics_path, "r", encoding="utf-8") as handle:
    best_metrics = json.load(handle)
if int(best_metrics.get("selection_epoch", -1)) != best_epoch:
    raise RuntimeError(
        "Best-validation metrics do not match model_best.pt: "
        f"selection_epoch={best_metrics.get('selection_epoch')!r}, "
        f"checkpoint_epoch={best_epoch}."
    )
PY
  printf '%s\n' "$best_checkpoint"
}

find_offline_experiment_dir() {
  local seed_root="$1"
  local search_root="${seed_root}_offline/uschad"
  local experiments=()
  if [[ -d "$search_root" ]]; then
    mapfile -t experiments < <(
      find "$search_root" -mindepth 1 -maxdepth 1 -type d \
        -name "Old6_Ratio0.8_SampleUnitTrial_Pool${TRIAL_POOLING}_Viewfull_random_crop_*" \
        -print
    )
  fi
  if [[ "${#experiments[@]}" -ne 1 ]]; then
    echo "Expected exactly one offline ${TRIAL_POOLING} experiment under $search_root; found ${#experiments[@]}." >&2
    return 1
  fi
  if [[ ! -f "${experiments[0]}/checkpoints/model_best.pt" ]]; then
    echo "Offline best checkpoint not found: ${experiments[0]}/checkpoints/model_best.pt" >&2
    return 1
  fi
  printf '%s\n' "${experiments[0]}"
}

COMMON_AUG=(
  --har_aug_mode weak_strong
  --har_weak_jitter_std 0
  --har_strong_jitter_std 0
  --har_weak_scale_std 0.10
  --har_strong_scale_std 0.20
  --har_time_mask_ratio 0
)

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

TRIAL_TEST_ARGS=(--offline_skip_final_test)
if [[ "$OUTER_TEST" -eq 1 ]]; then
  TRIAL_TEST_ARGS=()
fi

for fold in "${FOLDS[@]}"; do
  fold_index=$((fold - 1))
  fold_tag="$(printf '%02d' "$fold")"
  test_subjects="${TEST_PAIRS[$fold_index]}"
  val_subjects="${VAL_PAIRS[$fold_index]}"
  train_subjects="$(build_train_subjects "$test_subjects" "$val_subjects")"

  echo "================================================================"
  echo "Subject CV fold ${fold_tag}: train=${train_subjects}"
  echo "Subject CV fold ${fold_tag}: validation=${val_subjects}"
  echo "Subject CV fold ${fold_tag}: outer_test=${test_subjects}"
  echo "================================================================"

  COMMON_DATA=(
    --dataset_name uschad
    --uschad_npz_path "$NPZ_PATH"
    --uschad_window_size "$WINDOW_SIZE"
    --har_in_channels 6
    --uschad_split_mode subject
    --uschad_train_subjects "$train_subjects"
    --offline_val_subjects "$val_subjects"
    --uschad_test_subjects "$test_subjects"
    --uschad_recompute_norm_from_train_subjects
    --uschad_norm_eps "$NORM_EPS"
    --uschad_cv_fold "$fold"
    --num_old_classes 6
    --prop_train_labels 0.8
    --num_novel_classes_per_session 2
    --eval_funcs v2
    --image_aug_mode off
  )

  if [[ "$SKIP_DATA_CHECK" -eq 0 ]]; then
    "$PYTHON_BIN" -B check_uschad_dataset.py \
      --uschad_npz_path "$NPZ_PATH" \
      --uschad_window_size "$WINDOW_SIZE" \
      --har_in_channels 6 \
      --uschad_sample_unit trial \
      --trial_view_mode full_random_crop \
      --trial_crop_ratio 0.6666666666666666 \
      --trial_min_windows 2 \
      --uschad_split_mode subject \
      --uschad_train_subjects "$train_subjects" \
      --offline_val_subjects "$val_subjects" \
      --uschad_test_subjects "$test_subjects" \
      --uschad_recompute_norm_from_train_subjects \
      --uschad_norm_eps "$NORM_EPS" \
      --uschad_cv_fold "$fold" \
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
  fi

  for seed in "${SEEDS[@]}"; do
    if [[ -z "$ENCODER_ROOT" ]]; then
      window_target="${RESULT_ROOT}/fold_${fold_tag}/window_pretrain/seed_${seed}_offline"
      if [[ "$SKIP_EXISTING" -eq 1 && ( -e "$window_target" || -L "$window_target" ) ]]; then
        encoder_checkpoint="$(find_completed_window_checkpoint \
          "$window_target" "$WINDOW_EPOCHS" "$WINDOW_SIZE" "$fold" \
          "$train_subjects" "$val_subjects" "$test_subjects" "$NPZ_PATH")"
        echo "Reusing completed window checkpoint: ${encoder_checkpoint}"
      else
        echo "Window encoder pretraining: fold=${fold_tag}, seed=${seed}"
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
          --exp_root "${RESULT_ROOT}/fold_${fold_tag}/window_pretrain/seed_${seed}"
        encoder_checkpoint="$(find_encoder_checkpoint "$RESULT_ROOT" "$fold_tag" "$seed")"
      fi
    else
      encoder_checkpoint="$(find_encoder_checkpoint "$ENCODER_ROOT" "$fold_tag" "$seed")"
    fi

    if [[ "$WINDOW_ONLY" -eq 1 ]]; then
      echo "Window-only checkpoint ready: ${encoder_checkpoint}"
      continue
    fi

    echo "Trial pooling (${TRIAL_POOLING}): fold=${fold_tag}, seed=${seed}"
    echo "Window encoder: ${encoder_checkpoint}"
    seed_root="${RESULT_ROOT}/fold_${fold_tag}/${TRIAL_POOLING}_full_random_crop/seed_${seed}"
    CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON_BIN" -B train_happy.py \
      "${COMMON_DATA[@]}" \
      "${COMMON_AUG[@]}" \
      "${COMMON_OPT[@]}" \
      "${TRIAL_TEST_ARGS[@]}" \
      --train_session offline \
      --uschad_sample_unit trial \
      --trial_pooling "$TRIAL_POOLING" \
      --trial_robust_max_quantile "$ROBUST_QUANTILE" \
      --trial_pool_fusion_dim "$FUSION_DIM" \
      --trial_pool_fusion_dropout "$FUSION_DROPOUT" \
      --trial_attention_dim "$ATTENTION_DIM" \
      --trial_attention_dropout "$ATTENTION_DROPOUT" \
      --trial_attention_temperature "$ATTENTION_TEMPERATURE" \
      --trial_attention_mean_mix "$ATTENTION_MEAN_MIX" \
      --trial_view_mode full_random_crop \
      --trial_crop_ratio 0.6666666666666666 \
      --trial_min_windows 2 \
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
      --exp_root "$seed_root"

    if [[ "$INCLUDE_ONLINE" -eq 1 ]]; then
      offline_dir="$(find_offline_experiment_dir "$seed_root")"
      offline_id="$(basename "$offline_dir")"
      offline_checkpoint="${offline_dir}/checkpoints/model_best.pt"
      echo "Online continuation: fold=${fold_tag}, seed=${seed}"
      echo "Loading offline best checkpoint: ${offline_checkpoint}"
      CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON_BIN" -B train_happy.py \
        "${COMMON_DATA[@]}" \
        "${COMMON_AUG[@]}" \
        "${COMMON_OPT[@]}" \
        --train_session online \
        --load_offline_id "$offline_id" \
        --uschad_sample_unit trial \
        --trial_pooling "$TRIAL_POOLING" \
        --trial_robust_max_quantile "$ROBUST_QUANTILE" \
        --trial_pool_fusion_dim "$FUSION_DIM" \
        --trial_pool_fusion_dropout "$FUSION_DROPOUT" \
        --trial_attention_dim "$ATTENTION_DIM" \
        --trial_attention_dropout "$ATTENTION_DROPOUT" \
        --trial_attention_temperature "$ATTENTION_TEMPERATURE" \
        --trial_attention_mean_mix "$ATTENTION_MEAN_MIX" \
        --trial_view_mode full_random_crop \
        --trial_crop_ratio 0.6666666666666666 \
        --trial_min_windows 2 \
        --trial_encoder_freeze_epochs 0 \
        --trial_encoder_lr_scale 0.1 \
        --online_old_seen_trials 2 \
        --online_novel_unseen_trials 5 \
        --online_novel_seen_trials 2 \
        --online_checkpoint_selection final_epoch \
        --batch_size 16 \
        --eval_batch_size 16 \
        --epochs_online_per_session "$ONLINE_EPOCHS" \
        --lr 0.01 \
        --warmup_teacher_temp 0.05 \
        --teacher_temp 0.05 \
        --warmup_teacher_temp_epochs "$ONLINE_WARMUP_EPOCHS" \
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
    fi
  done
done

if [[ "$WINDOW_ONLY" -eq 0 ]]; then
  "$PYTHON_BIN" -B scripts/summarize_trial_pooling_subject_cv.py \
    --root "$RESULT_ROOT" \
    --expected-pooling "$TRIAL_POOLING" \
    --output-prefix "${TRIAL_POOLING}_subject_cv"

  if [[ "$INCLUDE_ONLINE" -eq 1 ]]; then
    "$PYTHON_BIN" -B scripts/summarize_trial_pooling_online_subject_cv.py \
      --root "$RESULT_ROOT" \
      --expected-pooling "$TRIAL_POOLING" \
      --output-prefix "${TRIAL_POOLING}_subject_cv_online"
  fi
fi

echo "Completed subject CV (window_size=${WINDOW_SIZE}, window_only=${WINDOW_ONLY}, skip_existing=${SKIP_EXISTING}, include_online=${INCLUDE_ONLINE})."
echo "Results: ${RESULT_ROOT}"
