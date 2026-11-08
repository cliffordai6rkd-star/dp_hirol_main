#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
CONDA_ENV_NAME="${PINN_CONDA_ENV_NAME:-pinn}"
TORCH_INDEX_URL="https://download.pytorch.org/whl/cu124"

if ! command -v conda >/dev/null 2>&1; then
    echo "Error: Conda was not found. Install Miniconda or Anaconda first." >&2
    exit 1
fi

if conda env list | awk -v env_name="${CONDA_ENV_NAME}" '$1 == env_name { found = 1 } END { exit(found ? 0 : 1) }'; then
    echo "Reusing existing Conda environment: ${CONDA_ENV_NAME}"
else
    conda create --yes --name "${CONDA_ENV_NAME}" python=3.10 pip
fi

CONDA_BASE="$(conda info --base)"
# shellcheck disable=SC1091
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV_NAME}"

python -m pip install --upgrade "pip>=24" "setuptools>=71,<81" wheel
python -m pip install \
    --index-url "${TORCH_INDEX_URL}" \
    "torch==2.6.0" \
    "torchvision==0.21.0" \
    "torchaudio==2.6.0" \
    "torchcodec==0.2.1"
python -m pip install -e "${PROJECT_DIR}[physics,test]"

if ! python -c 'import torch; assert torch.cuda.is_available(), "CUDA is unavailable"; print(f"CUDA ready: torch={torch.__version__}, runtime={torch.version.cuda}, gpu={torch.cuda.get_device_name(0)}")'; then
    echo "Error: NVIDIA GPU training is required, but PyTorch cannot access CUDA." >&2
    echo "Check the NVIDIA driver and GPU visibility, then run setup.sh again." >&2
    exit 1
fi

echo
echo "PINN GPU environment is ready."
echo "Activate it with: conda activate ${CONDA_ENV_NAME}"
echo "Run tests with:    python -m pytest -q tests"
