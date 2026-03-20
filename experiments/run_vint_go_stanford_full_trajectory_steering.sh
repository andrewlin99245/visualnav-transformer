#!/bin/bash
set -euo pipefail

CHECKPOINT="${1:-${CHECKPOINT:-/project/6003584/tsungen/models/vint_checkpoints/checkpoints/vint.pth}}"
DATA_FOLDER="${2:-${DATA_FOLDER:-}}"
SPLIT_FOLDER="${3:-${SPLIT_FOLDER:-}}"
MAX_EXAMPLES="${4:-${MAX_EXAMPLES:-0}}"
BATCH_SIZE="${5:-${BATCH_SIZE:-64}}"

REPO_DIR="/project/6003584/tsungen/visualnav-transformer"
INPUT_DIR="${INPUT_DIR:-/home/tsungen/projects/def-beltrame/vnm_datasets/processed_datasets}"
WORKDIR="${WORKDIR:-${SLURM_TMPDIR:-/tmp/visualnav_transformer_full_trajectory}}"
DATASET_NAME="${DATASET_NAME:-go_stanford}"
SPLIT_NAME="${SPLIT_NAME:-test}"
COEFF="${COEFF:-0.05}"
LAYERS="${LAYERS:-all}"
SAVE_CHUNK_SIZE="${SAVE_CHUNK_SIZE:-2048}"
RESIDUAL_DTYPE="${RESIDUAL_DTYPE:-bfloat16}"

if [[ "$MAX_EXAMPLES" -le 0 ]]; then
  SAMPLE_TAG="full"
else
  SAMPLE_TAG="${MAX_EXAMPLES}s"
fi

OUTPUT_NAME="${OUTPUT_NAME:-${DATASET_NAME}_${SPLIT_NAME}_full_trajectory_steering_${SAMPLE_TAG}_c${COEFF//./p}}"

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

if [[ -z "$DATA_FOLDER" || -z "$SPLIT_FOLDER" ]]; then
  ensure_dataset "$DATASET_NAME"
fi

if [[ -z "$DATA_FOLDER" ]]; then
  DATA_FOLDER="$WORKDIR/data/$DATASET_NAME"
fi

if [[ -z "$SPLIT_FOLDER" ]]; then
  SPLIT_FOLDER="$WORKDIR/data/data_splits/$DATASET_NAME/$SPLIT_NAME"
fi

cd "$REPO_DIR"
if [[ -n "${VENV_DIR:-}" ]]; then
  source "$VENV_DIR/bin/activate"
fi

python experiments/full_trajectory_steering.py \
  --checkpoint "$CHECKPOINT" \
  --data-folder "$DATA_FOLDER" \
  --split-folder "$SPLIT_FOLDER" \
  --dataset-name "$DATASET_NAME" \
  --max-examples "$MAX_EXAMPLES" \
  --batch-size "$BATCH_SIZE" \
  --coeff "$COEFF" \
  --layers "$LAYERS" \
  --save-chunk-size "$SAVE_CHUNK_SIZE" \
  --residual-dtype "$RESIDUAL_DTYPE" \
  --output-dir "$REPO_DIR/experiments/results/$OUTPUT_NAME"
