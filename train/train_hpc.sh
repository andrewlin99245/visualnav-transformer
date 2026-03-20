#!/bin/bash

#SBATCH --time=6:30:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=128000M
#SBATCH --account=def-beltrame
#SBATCH --gres=gpu:h100:1
#SBATCH --array=0-1

# Select config based on array index: 0=baseline, 1=SIRA
CONFIGS=(config/vint_ft_baseline.yaml config/vint_ft_sira.yaml)
SOURCE_CONFIG=${CONFIGS[$SLURM_ARRAY_TASK_ID]}

# Load modules
module load gcc opencv/4.11.0
module load python/3.11.5

# Set up venv on local SSD
python -m venv $SLURM_TMPDIR/.venv
source $SLURM_TMPDIR/.venv/bin/activate
pip install -r /project/6003584/tsungen/visualnav-transformer/train/requirements.txt --extra-index-url https://rospypi.github.io/simple/
pip install -e /project/6003584/tsungen/visualnav-transformer/train/

# Copy and extract datasets
mkdir -p $SLURM_TMPDIR/data
INPUT_DIR="/home/tsungen/projects/def-beltrame/vnm_datasets/processed_datasets"
cp $INPUT_DIR/go_stanford.tar.gz $SLURM_TMPDIR/data/
cp $INPUT_DIR/scand.tar.gz $SLURM_TMPDIR/data/

tar -xf $SLURM_TMPDIR/data/go_stanford.tar.gz -C $SLURM_TMPDIR/data/
rm $SLURM_TMPDIR/data/go_stanford.tar.gz
tar -xf $SLURM_TMPDIR/data/scand.tar.gz -C $SLURM_TMPDIR/data/
rm $SLURM_TMPDIR/data/scand.tar.gz

# Generate data splits
cd /project/6003584/tsungen/visualnav-transformer/train
mkdir -p $SLURM_TMPDIR/data/data_splits

python data_split.py \
    -i $SLURM_TMPDIR/data/go_stanford \
    -d go_stanford \
    -o $SLURM_TMPDIR/data/data_splits

python data_split.py \
    -i $SLURM_TMPDIR/data/scand \
    -d scand \
    -o $SLURM_TMPDIR/data/data_splits

# Generate runtime config with SLURM_TMPDIR paths
TMPCONFIG=$SLURM_TMPDIR/vint_runtime.yaml
python - "$SOURCE_CONFIG" "$TMPCONFIG" "$SLURM_TMPDIR" <<'EOF'
import yaml, sys
source_config, tmpconfig, tmpdir = sys.argv[1], sys.argv[2], sys.argv[3]
with open(source_config) as f:
    cfg = yaml.safe_load(f)
for ds in cfg['datasets']:
    cfg['datasets'][ds]['data_folder'] = f'{tmpdir}/data/{ds}'
    if 'train' in cfg['datasets'][ds]:
        cfg['datasets'][ds]['train'] = f'{tmpdir}/data/data_splits/{ds}/train/'
    cfg['datasets'][ds]['test'] = f'{tmpdir}/data/data_splits/{ds}/test/'
with open(tmpconfig, 'w') as f:
    yaml.dump(cfg, f, default_flow_style=False)
EOF

# Make pretrained checkpoint available to load_run (expects logs/vint-pretrained/latest.pth)
mkdir -p /project/6003584/tsungen/visualnav-transformer/train/logs/vint-pretrained

python train.py -c $TMPCONFIG
