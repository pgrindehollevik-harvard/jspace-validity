#!/usr/bin/env bash
# Stage one allowed public model and stock lens from the ORCD login node.
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 qwen3-8b|qwen3-14b|qwen3-1.7b|qwen3-32b" >&2
  exit 2
fi
case "$1" in
  qwen3-8b | qwen3-14b | qwen3-1.7b | qwen3-32b) ;;  # v2 + v3 (PREREGISTRATION_V3.md)
  *) echo "profile is outside the registered studies: $1" >&2; exit 2 ;;
esac

readonly JSPACE_ROOT="${JSPACE_ROOT:-${HOME}/scratch/jspace-validity}"
cd "$JSPACE_ROOT"
if ! module load miniforge/24.3.0-0 2>/dev/null; then
  module load miniforge
fi
export HF_HOME="${HF_HOME:-$JSPACE_ROOT/.hf}"
export HF_HUB_DISABLE_TELEMETRY=1

.venv/bin/python -u -m jspace.stage_assets --model "$1"
