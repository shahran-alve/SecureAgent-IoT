# Candidate models of the real-LLM run, in order. Source this file; do not execute it.
# The first one is the pilot model and is required. The ids were NOT verified when this list was written:
# scripts/llm_preflight.py looks each one up on the Hub, and a candidate that does not resolve, is gated without
# access, is over 13B parameters or has no chat template is skipped with the reason. No replacement id is guessed.
# Override with the CANDIDATES variable (space separated). The preflight and the job array must be given the SAME
# list: the array index is computed from a candidate's position in it.
DEFAULT_CANDIDATES="Qwen/Qwen2.5-7B-Instruct ibm-granite/granite-4.1-8b-instruct google/gemma-4-12b-it microsoft/Phi-4-mini-instruct mistralai/Mistral-Nemo-Instruct-2407"
export CANDIDATES="${CANDIDATES:-$DEFAULT_CANDIDATES}"
