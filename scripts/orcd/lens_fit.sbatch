#!/usr/bin/env bash
# Registered Qwen3-32B lens refit (extension/PREREGISTRATION_V4.md). Engineering work on
# public data: reads no registered evaluation item and writes no gate result.
#SBATCH --job-name=jspace-validity-lensfit
#SBATCH --partition=mit_preemptable
#SBATCH --gpus=h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-gpu=160G
#SBATCH --time=24:00:00
#SBATCH --requeue
#SBATCH --output=logs/orcd/%x-%j.out
#SBATCH --error=logs/orcd/%x-%j.err

set -euo pipefail

readonly JSPACE_ROOT="${JSPACE_ROOT:-${HOME}/scratch/jspace-validity}"
cd "$JSPACE_ROOT"
if ! module load miniforge/24.3.0-0 2>/dev/null; then
  module load miniforge
fi

export HF_HOME="${HF_HOME:-$JSPACE_ROOT/.hf}"
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

printf 'job_id=%s\n' "${SLURM_JOB_ID:-unknown}"
printf 'code_commit=%s\n' "$(git rev-parse HEAD)"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
.venv/bin/python -u -m jspace.lens_fit --model qwen3-32b
