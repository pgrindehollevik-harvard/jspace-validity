#!/usr/bin/env bash
# Read-only accounting summary for one completed or running Slurm job.
set -euo pipefail

if [[ $# -ne 1 || ! "$1" =~ ^[0-9]+$ ]]; then
  echo "usage: $0 JOB_ID" >&2
  exit 2
fi
readonly JOB_ID="$1"

sacct -X -j "$JOB_ID" \
  -o JobIDRaw,JobName%32,Partition,Submit,Eligible,Start,End,Elapsed,State,ExitCode,ReqTRES%80,AllocTRES%80
if command -v jobstats >/dev/null 2>&1; then
  jobstats "$JOB_ID"
fi
