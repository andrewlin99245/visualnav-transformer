#!/bin/bash
set -euo pipefail

CHECKPOINT="${1:-${CHECKPOINT:-/project/6003584/tsungen/models/vint_checkpoints/checkpoints/vint.pth}}"
VECTOR_DATA_FOLDER="${2:-${VECTOR_DATA_FOLDER:-}}"
VECTOR_SPLIT_FOLDER="${3:-${VECTOR_SPLIT_FOLDER:-}}"
VECTOR_DATASET_NAME="${4:-${VECTOR_DATASET_NAME:-go_stanford}}"
EVAL_DATA_FOLDER="${5:-${EVAL_DATA_FOLDER:-}}"
EVAL_SPLIT_FOLDER="${6:-${EVAL_SPLIT_FOLDER:-}}"
EVAL_DATASET_NAME="${7:-${EVAL_DATASET_NAME:-scand}}"
MAX_SAMPLES="${8:-${MAX_SAMPLES:-0}}"
BATCH_SIZE="${9:-${BATCH_SIZE:-64}}"

REPO_DIR="/project/6003584/tsungen/visualnav-transformer"
if [[ "$MAX_SAMPLES" -le 0 ]]; then
  SAMPLE_TAG="full"
else
  SAMPLE_TAG="${MAX_SAMPLES}s"
fi

OUTPUT_NAME="${OUTPUT_NAME:-${VECTOR_DATASET_NAME}_to_${EVAL_DATASET_NAME}_sira_${SAMPLE_TAG}}"
INPUT_DIR="${INPUT_DIR:-/home/tsungen/projects/def-beltrame/vnm_datasets/processed_datasets}"
WORKDIR="${WORKDIR:-${SLURM_TMPDIR:-/tmp/visualnav_transformer_sira}}"

if command -v module >/dev/null 2>&1; then
  module load gcc opencv/4.11.0
  module load python/3.11.5
fi

mkdir -p "$WORKDIR/data"
mkdir -p "$WORKDIR/data/data_splits"

ensure_dataset() {
  local dataset_name="$1"
  local dataset_dir="$WORKDIR/data/$dataset_name"
  local split_dir="$WORKDIR/data/data_splits/$dataset_name"
  local archive_path="$INPUT_DIR/$dataset_name.tar.gz"

  if [[ ! -d "$dataset_dir" ]]; then
    if [[ ! -f "$archive_path" ]]; then
      echo "Dataset archive not found: $archive_path"
      exit 1
    fi
    echo "Extracting $dataset_name..."
    cp "$archive_path" "$WORKDIR/data/"
    tar -xf "$WORKDIR/data/$dataset_name.tar.gz" -C "$WORKDIR/data/"
    rm -f "$WORKDIR/data/$dataset_name.tar.gz"
  fi

  if [[ ! -d "$split_dir" ]]; then
    echo "Generating $dataset_name splits..."
    (
      cd "$REPO_DIR/train"
      python data_split.py -i "$dataset_dir" -d "$dataset_name" -o "$WORKDIR/data/data_splits"
    )
  fi
}

if [[ -z "$VECTOR_DATA_FOLDER" ]]; then
  ensure_dataset "$VECTOR_DATASET_NAME"
  VECTOR_DATA_FOLDER="$WORKDIR/data/$VECTOR_DATASET_NAME"
fi

if [[ -z "$VECTOR_SPLIT_FOLDER" ]]; then
  ensure_dataset "$VECTOR_DATASET_NAME"
  VECTOR_SPLIT_FOLDER="$WORKDIR/data/data_splits/$VECTOR_DATASET_NAME/test"
fi

if [[ -z "$EVAL_DATA_FOLDER" ]]; then
  ensure_dataset "$EVAL_DATASET_NAME"
  EVAL_DATA_FOLDER="$WORKDIR/data/$EVAL_DATASET_NAME"
fi

if [[ -z "$EVAL_SPLIT_FOLDER" ]]; then
  ensure_dataset "$EVAL_DATASET_NAME"
  EVAL_SPLIT_FOLDER="$WORKDIR/data/data_splits/$EVAL_DATASET_NAME/test"
fi

if [[ -z "$VECTOR_DATA_FOLDER" || -z "$VECTOR_SPLIT_FOLDER" ]]; then
  echo "Usage: bash experiments/run_vint_sira.sh [checkpoint] <vector_data_folder> <vector_split_folder> [vector_dataset_name] [eval_data_folder] [eval_split_folder] [eval_dataset_name] [max_samples] [batch_size]"
  echo "Defaults:"
  echo "  checkpoint=$CHECKPOINT"
  echo "  vector_dataset_name=$VECTOR_DATASET_NAME"
  echo "  eval_dataset_name=$EVAL_DATASET_NAME"
  exit 1
fi

cd "$REPO_DIR"
if [[ -n "${VENV_DIR:-}" ]]; then
  source "$VENV_DIR/bin/activate"
fi

python experiments/sira_correlation.py \
  --checkpoint "$CHECKPOINT" \
  --vector-data-folder "$VECTOR_DATA_FOLDER" \
  --vector-split-folder "$VECTOR_SPLIT_FOLDER" \
  --vector-dataset-name "$VECTOR_DATASET_NAME" \
  --eval-data-folder "$EVAL_DATA_FOLDER" \
  --eval-split-folder "$EVAL_SPLIT_FOLDER" \
  --eval-dataset-name "$EVAL_DATASET_NAME" \
  --max-samples "$MAX_SAMPLES" \
  --batch-size "$BATCH_SIZE" \
  --output-dir "$REPO_DIR/experiments/results/$OUTPUT_NAME"
