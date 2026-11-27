#!/usr/bin/env bash
#SBATCH -J saiot-dl
#SBATCH -p gpuq
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH -o logs/%j.out
#
# Pre-download model weights into HF_HOME (compute nodes have outbound internet; the head node must not do this).
#   MODELS="Qwen/Qwen2.5-7B-Instruct" sbatch scripts/hf_download.sbatch
# Gated models (e.g. meta-llama/*) need HF_TOKEN in the submitting environment. Run this BEFORE the job array:
# the array tasks run with HF_HUB_OFFLINE=1 so that six concurrent jobs never race on one download.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
source scripts/llm_env.sh
MODELS="${MODELS:-Qwen/Qwen2.5-7B-Instruct}"
MIN_FREE_GB="${MIN_FREE_GB:-30}"
mkdir -p "$HF_HOME"
free_gb=$(df -BG --output=avail "$HF_HOME" | tail -1 | tr -dc '0-9')
echo "HF_HOME=$HF_HOME free=${free_gb}G (minimum ${MIN_FREE_GB}G)"
if [ "$free_gb" -lt "$MIN_FREE_GB" ]; then echo "BLOCKED: under ${MIN_FREE_GB} GB free, not downloading"; exit 3; fi
for m in $MODELS; do
  echo "== downloading $m =="
  python - "$m" <<'PY'
import sys
from pathlib import Path
from huggingface_hub import snapshot_download
path = Path(snapshot_download(sys.argv[1], allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model", "*.jinja"]))
size = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
print(f"snapshot: {path}\nrevision: {path.name}\nsize_gb: {size / 1e9:.2f}")
PY
done
