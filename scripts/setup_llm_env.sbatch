#!/usr/bin/env bash
#SBATCH -J saiot-env
#SBATCH -p gpuq
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH -o logs/%j.out
#
# Builds the conda env used by the real-LLM runs and PROVES the torch build works on this GPU.
#   sbatch scripts/setup_llm_env.sbatch
# The V100 is compute capability 7.0 (sm_70). The cu126 wheels still ship sm_70 kernels; newer CUDA
# builds may not, so TORCH_INDEX is the knob to step down if the proof at the end fails.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
ENV_NAME="${ENV_NAME:-saiot-llm}"
TORCH_INDEX="${TORCH_INDEX:-https://download.pytorch.org/whl/cu126}"
export PYTHONNOUSERSITE=1
source /cm/shared/apps/anaconda3/2025.12-2/etc/profile.d/conda.sh
if ! conda env list | grep -qE "/${ENV_NAME}\$"; then
  conda create -y -n "$ENV_NAME" -c conda-forge --override-channels python=3.11 pip git
fi
conda activate "$ENV_NAME"
python -m pip install --upgrade pip
python -m pip install torch --index-url "$TORCH_INDEX"
python -m pip install "transformers>=4.45" accelerate
python -m pip install -e ".[dev]"
mkdir -p reproducibility
python -m pip freeze --exclude-editable > reproducibility/llm_requirements.lock.txt
echo "== proof on $(hostname) =="
python scripts/gpu_proof.py
