#!/usr/bin/env bash
# Materialize and verify the preregistered public-data subset on the login node.
set -euo pipefail

readonly JSPACE_ROOT="${JSPACE_ROOT:-${HOME}/scratch/jspace-validity}"
cd "$JSPACE_ROOT"
if ! module load miniforge/24.3.0-0 2>/dev/null; then
  module load miniforge
fi
export HF_HOME="${HF_HOME:-$JSPACE_ROOT/.hf}"
export HF_HUB_DISABLE_TELEMETRY=1

.venv/bin/python -u -m jspace.scale_data
