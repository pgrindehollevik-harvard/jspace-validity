#!/usr/bin/env bash
# Small allocation-only CUDA probe. It reads no registered dataset.
#SBATCH --job-name=jspace-validity-probe
#SBATCH --partition=mit_normal_gpu
#SBATCH --gpus=l40s:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:10:00
#SBATCH --output=logs/orcd/%x-%j.out
#SBATCH --error=logs/orcd/%x-%j.err

set -euo pipefail
readonly JSPACE_ROOT="${JSPACE_ROOT:-${HOME}/scratch/jspace-validity}"
cd "$JSPACE_ROOT"
if ! module load miniforge/24.3.0-0 2>/dev/null; then
  module load miniforge
fi

printf 'job_id=%s\n' "${SLURM_JOB_ID:-unknown}"
printf 'partition=%s\n' "${SLURM_JOB_PARTITION:-unknown}"
printf 'requested_gpus=%s\n' "${SLURM_GPUS_ON_NODE:-unknown}"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

.venv/bin/python - <<'PY'
import json
import time

import torch

if not torch.cuda.is_available():
    raise SystemExit("Slurm allocated no CUDA-visible GPU")
torch.cuda.reset_peak_memory_stats()
started = time.perf_counter()
matrix = torch.randn((1024, 1024), device="cuda", dtype=torch.bfloat16)
result = matrix @ matrix
torch.cuda.synchronize()
elapsed = time.perf_counter() - started
print(
    json.dumps(
        {
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "device_count": torch.cuda.device_count(),
            "device": torch.cuda.get_device_name(0),
            "total_memory_gib": round(
                torch.cuda.get_device_properties(0).total_memory / 2**30, 2
            ),
            "peak_cuda_gib": round(torch.cuda.max_memory_allocated() / 2**30, 3),
            "matmul_seconds": round(elapsed, 4),
            "checksum": round(result[0, 0].item(), 4),
        },
        indent=2,
    )
)
PY
