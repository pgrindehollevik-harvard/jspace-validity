# ORCD execution guide

The repository includes Slurm templates used for the released experiments on MIT ORCD.
They are cluster-specific examples, not a credential or account setup guide: configure
SSH, Duo, allocation access, and any site-specific modules yourself. No password, token,
private key, hostname, or control socket is required by or stored in this repository.

Set `JSPACE_ROOT` to your own checkout. The default shown below is deliberately generic:

```bash
export JSPACE_ROOT="$HOME/scratch/jspace-validity"
git clone <your-clone-url> "$JSPACE_ROOT"
cd "$JSPACE_ROOT"
```

## 1. Inspect the scheduler first

On the login node, check the partitions and your own queue before submitting anything:

```bash
sinfo -O "Partition,Nodes:10,CPUsState,Gres:30,GresUsed:30,StateCompact" -e
squeue --me
```

The committed templates request L40S or H200 GPUs on ORCD's `mit_normal_gpu` and
`mit_preemptable` partitions. Adjust only the site configuration needed for your own
allocation; do not silently change a frozen experimental profile.

## 2. Bootstrap and stage public assets

Bootstrap the locked Python environment and the pinned public Jacobian-lens companion
checkout:

```bash
bash scripts/orcd/bootstrap.sh
scripts/orcd/stage_assets.sh qwen3-1.7b
scripts/orcd/stage_assets.sh qwen3-14b
scripts/orcd/stage_assets.sh qwen3-32b
scripts/orcd/stage_scale_data.sh
```

Staging downloads public model and lens assets to the configured cache. It performs no
model inference and writes no scientific result artifact.

## 3. Probe, smoke-test, then run a gate

Run an allocation probe before a model job:

```bash
sbatch scripts/orcd/probe_gpu.sbatch
squeue --me
```

After the probe, submit a model/lens smoke test. It cannot read registered evaluation
data or write a result artifact:

```bash
sbatch --export=ALL,JSPACE_PROFILE=qwen3-14b scripts/orcd/scale_smoke.sbatch
```

Only then submit a frozen gate:

```bash
sbatch --export=ALL,JSPACE_PROFILE=qwen3-14b scripts/orcd/scale_gate.sbatch
```

The runner validates pinned code, configuration, data, and companion revisions. It
records scientific provenance in its result artifact but, by default, does not emit a
hostname or scheduler job identifier. Set `JSPACE_RECORD_EXECUTION_IDS=1` only for a
private operational run where you explicitly want those fields.

## 4. Analyze locally

Result artifacts are intended to be committed separately from the frozen protocol. To
recreate the released analysis after syncing results back to a clean checkout:

```bash
uv run jspace-analyze
uv run python -m jspace.v3_summary
make -C paper public
```

The 32B lens-fit recipe is described in `extension/PREREGISTRATION_V4.md`; it is not a
routine retry mechanism. Read the preregistration and deviations log before attempting a
new fit or a new scientific run.
