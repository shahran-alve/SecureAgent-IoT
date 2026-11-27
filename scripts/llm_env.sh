# Common environment for the real-LLM Slurm jobs. Source it; do not execute it.
#   ENV_NAME  conda env built by scripts/setup_llm_env.sbatch (default: saiot-llm)
#   HF_HOME   Hugging Face cache; must be on the network-mounted home so every compute node sees it
ENV_NAME="${ENV_NAME:-saiot-llm}"
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export PYTHONNOUSERSITE=1            # keep ~/.local site-packages out of the env
export PYTHONUNBUFFERED=1            # progress lines reach logs/<job>.out immediately
export TOKENIZERS_PARALLELISM=false
source /cm/shared/apps/anaconda3/2025.12-2/etc/profile.d/conda.sh
conda activate "$ENV_NAME"
