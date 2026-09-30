#!/usr/bin/env bash
# Engineering-only model/lens smoke test. It cannot read registered evaluation data.
# Submit with --export=ALL,JSPACE_PROFILE=qwen3-8b (or qwen3-14b).
#SBATCH --job-name=jspace-validity-smoke
#SBATCH --partition=mit_normal_gpu
#SBATCH --gpus=l40s:1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-gpu=96G
#SBATCH --time=00:30:00
#SBATCH --output=logs/orcd/%x-%j.out
#SBATCH --error=logs/orcd/%x-%j.err

set -euo pipefail
: "${JSPACE_PROFILE:?submit with JSPACE_PROFILE=qwen3-8b or qwen3-14b}"
case "$JSPACE_PROFILE" in
  qwen3-8b | qwen3-14b | qwen3-1.7b | qwen3-32b) ;;  # v2 + v3 (PREREGISTRATION_V3.md)
  *) echo "profile is outside the registered studies: $JSPACE_PROFILE" >&2; exit 2 ;;
esac

readonly JSPACE_ROOT="${JSPACE_ROOT:-${HOME}/scratch/jspace-validity}"
cd "$JSPACE_ROOT"
if ! git diff --quiet HEAD --; then
  echo "smoke jobs require a clean tracked checkout" >&2
  exit 1
fi
if ! module load miniforge/24.3.0-0 2>/dev/null; then
  module load miniforge
fi

export HF_HOME="${HF_HOME:-$JSPACE_ROOT/.hf}"
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export JSPACE_DEVICE=cuda
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

printf 'job_id=%s\n' "${SLURM_JOB_ID:-unknown}"
printf 'code_commit=%s\n' "$(git rev-parse HEAD)"
printf 'profile=%s\n' "$JSPACE_PROFILE"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
.venv/bin/python -u -m jspace.scale_smoke --model "$JSPACE_PROFILE"
