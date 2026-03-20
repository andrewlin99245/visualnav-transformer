#!/bin/bash
set -euo pipefail

REPO_DIR="/project/6003584/tsungen/visualnav-transformer"
VENV_DIR="${VENV_DIR:-/project/6003584/tsungen/venv}"
DIFFUSION_POLICY_DIR="${DIFFUSION_POLICY_DIR:-/project/6003584/tsungen/diffusion_policy}"
DIFFUSION_POLICY_REPO="${DIFFUSION_POLICY_REPO:-https://github.com/real-stanford/diffusion_policy.git}"
SNAPSHOT_DIR="${SNAPSHOT_DIR:-$REPO_DIR/train/env_snapshots}"

if [[ ! -d "$VENV_DIR" ]]; then
  echo "Virtualenv not found: $VENV_DIR"
  exit 1
fi

source "$VENV_DIR/bin/activate"
mkdir -p "$SNAPSHOT_DIR"

SNAPSHOT_FILE="$SNAPSHOT_DIR/$(date +%Y%m%d_%H%M%S)_before_nomad.txt"
python -m pip freeze > "$SNAPSHOT_FILE"
echo "Saved package snapshot to: $SNAPSHOT_FILE"

# Keep the existing ViNT env stable: only apply the minimal compatibility pins
# needed for the repo's legacy NoMaD stack.
python -m pip install --upgrade --no-deps "huggingface_hub==0.11.1"
python -m pip install --upgrade \
  "zarr<3" \
  "numcodecs<0.16"
python -m pip install -e "$REPO_DIR/train" --no-deps

if [[ ! -d "$DIFFUSION_POLICY_DIR/.git" ]]; then
  git clone "$DIFFUSION_POLICY_REPO" "$DIFFUSION_POLICY_DIR"
fi

python -m pip install -e "$DIFFUSION_POLICY_DIR" --no-deps

python - <<'PY'
checks = [
    ("ViNT model", "from vint_train.models.vint.vint import ViNT"),
    ("ViNT dataset", "from vint_train.data.vint_dataset import ViNT_Dataset"),
    ("NoMaD model", "from vint_train.models.nomad.nomad import NoMaD"),
    ("NoMaD encoder", "from vint_train.models.nomad.nomad_vint import NoMaD_ViNT"),
    ("Diffusers scheduler", "from diffusers.schedulers.scheduling_ddpm import DDPMScheduler"),
    ("Diffusion policy", "from diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D"),
]

for label, stmt in checks:
    exec(stmt, {})
    print(f"OK {label}")
PY
