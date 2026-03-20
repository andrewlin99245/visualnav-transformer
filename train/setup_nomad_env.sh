#!/bin/bash
set -euo pipefail

REPO_DIR="/project/6003584/tsungen/visualnav-transformer"
TRAIN_DIR="$REPO_DIR/train"
VENV_DIR="${VENV_DIR:-/project/6003584/tsungen/venv-nomad}"
DIFFUSION_POLICY_DIR="${DIFFUSION_POLICY_DIR:-/project/6003584/tsungen/diffusion_policy}"
DIFFUSION_POLICY_REPO="${DIFFUSION_POLICY_REPO:-https://github.com/real-stanford/diffusion_policy.git}"
PYTHON_MODULE="${PYTHON_MODULE:-python/3.11.5}"
OPENCV_MODULE="${OPENCV_MODULE:-opencv/4.11.0}"
GCC_MODULE="${GCC_MODULE:-gcc}"
ROSPYPI_INDEX="${ROSPYPI_INDEX:-https://rospypi.github.io/simple/}"

if command -v module >/dev/null 2>&1; then
  module load "$GCC_MODULE" "$OPENCV_MODULE"
  module load "$PYTHON_MODULE"
fi

if [[ ! -d "$VENV_DIR" ]]; then
  python -m venv "$VENV_DIR"
fi

source "$VENV_DIR/bin/activate"

python -m pip install --upgrade pip setuptools wheel
python -m pip install -r "$TRAIN_DIR/requirements.txt" --extra-index-url "$ROSPYPI_INDEX"
# Make core runtime deps available inside the venv itself instead of relying on
# the loaded cluster Python stack.
python -m pip install --upgrade "typing_extensions==4.15.0"
python -m pip install -e "$TRAIN_DIR" --no-deps

if [[ ! -d "$DIFFUSION_POLICY_DIR/.git" ]]; then
  git clone "$DIFFUSION_POLICY_REPO" "$DIFFUSION_POLICY_DIR"
fi

python -m pip install -e "$DIFFUSION_POLICY_DIR" --no-deps
python - <<'PY'
import site
from pathlib import Path

repo = Path("/project/6003584/tsungen/diffusion_policy")
pth = Path(site.getsitepackages()[0]) / "diffusion_policy_repo.pth"
pth.write_text(f"{repo}\n")
print(f"Wrote {pth}")
PY

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

echo "NoMaD env ready at: $VENV_DIR"
