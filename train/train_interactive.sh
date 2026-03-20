#!/bin/bash
# Interactive session version of train_hpc.sh
# Usage: bash train_interactive.sh [0|1]  (0=baseline, 1=sira)

set -e

TASK_ID=${1:-1}  # default to SIRA
CONFIGS=(config/vint_ft_baseline.yaml config/vint_ft_sira.yaml config/vint_eval_pretrained.yaml)
SOURCE_CONFIG=${CONFIGS[$TASK_ID]}
echo "Using config: $SOURCE_CONFIG"

# Load modules
module load gcc opencv/4.11.0
module load python/3.11.5

source /project/6003584/tsungen/venv/bin/activate

WORKDIR=$SLURM_TMPDIR
mkdir -p $WORKDIR/data

# Extract datasets if not already done
if [ ! -d "$WORKDIR/data/go_stanford" ]; then
    echo "Extracting go_stanford..."
    INPUT_DIR="/home/tsungen/projects/def-beltrame/vnm_datasets/processed_datasets"
    cp $INPUT_DIR/go_stanford.tar.gz $WORKDIR/data/
    tar -xf $WORKDIR/data/go_stanford.tar.gz -C $WORKDIR/data/
    rm $WORKDIR/data/go_stanford.tar.gz
fi

if [ ! -d "$WORKDIR/data/scand" ]; then
    echo "Extracting scand..."
    INPUT_DIR="/home/tsungen/projects/def-beltrame/vnm_datasets/processed_datasets"
    cp $INPUT_DIR/scand.tar.gz $WORKDIR/data/
    tar -xf $WORKDIR/data/scand.tar.gz -C $WORKDIR/data/
    rm $WORKDIR/data/scand.tar.gz
fi

# Generate data splits if not already done
cd /project/6003584/tsungen/visualnav-transformer/train
mkdir -p $WORKDIR/data/data_splits

if [ ! -d "$WORKDIR/data/data_splits/go_stanford" ]; then
    echo "Generating go_stanford splits..."
    python data_split.py -i $WORKDIR/data/go_stanford -d go_stanford -o $WORKDIR/data/data_splits
fi

if [ ! -d "$WORKDIR/data/data_splits/scand" ]; then
    echo "Generating scand splits..."
    python data_split.py -i $WORKDIR/data/scand -d scand -o $WORKDIR/data/data_splits
fi

# Generate runtime config with paths
TMPCONFIG=$WORKDIR/vint_runtime.yaml
python - "$SOURCE_CONFIG" "$TMPCONFIG" "$WORKDIR" <<'EOF'
import yaml, sys
source_config, tmpconfig, workdir = sys.argv[1], sys.argv[2], sys.argv[3]
with open(source_config) as f:
    cfg = yaml.safe_load(f)
for ds in cfg['datasets']:
    cfg['datasets'][ds]['data_folder'] = f'{workdir}/data/{ds}'
    if 'train' in cfg['datasets'][ds]:
        cfg['datasets'][ds]['train'] = f'{workdir}/data/data_splits/{ds}/train/'
    cfg['datasets'][ds]['test'] = f'{workdir}/data/data_splits/{ds}/test/'
with open(tmpconfig, 'w') as f:
    yaml.dump(cfg, f, default_flow_style=False)
EOF

python train.py -c $TMPCONFIG
