#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${ENV_NAME:-arc-sae-sweep}"
PYTHON_VERSION="${PYTHON_VERSION:-3.11}"

if ! command -v conda >/dev/null 2>&1; then
  curl -fsSL -o /tmp/miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
  bash /tmp/miniconda.sh -b -p "$HOME/miniconda3"
  source "$HOME/miniconda3/etc/profile.d/conda.sh"
else
  source "$(conda info --base)/etc/profile.d/conda.sh"
fi

if ! conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  conda create -y -n "$ENV_NAME" "python=$PYTHON_VERSION"
fi

conda activate "$ENV_NAME"
python -m pip install --upgrade pip
EXTRAS="${EXTRAS:-wandb,spark,sae}"
if [[ "${INSTALL_SGLANG:-0}" == "1" ]] && [[ "$EXTRAS" != *sglang* ]]; then
  EXTRAS="${EXTRAS},sglang"
fi
python -m pip install -e ".[${EXTRAS}]"

python - <<'PY'
import torch
print("torch", torch.__version__)
print("cuda_available", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device_count", torch.cuda.device_count())
PY
