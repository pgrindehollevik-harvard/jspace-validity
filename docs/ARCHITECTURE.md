# Architecture

The repository keeps scientific choices, execution state, and analysis separate.

## Frozen inputs

`extension/config.json` is the machine-readable source of model roles, artifact commits,
band transfer, samples, thresholds, and compute caps. `jspace.scale_config` reads it but
does not infer or override scientific settings. The runner records its SHA-256 alongside
the staged-data digest and code commit.

## Runtime boundary

- `profiles.py` contains only pinned artifact identities and model shapes.
- `core.py` defines backend-neutral setup types and registered generation constants.
- `scale_runtime.py` loads one allowed profile and validates its shape and lens coverage.
- `ablation.py` implements J-space projection and the per-slot norm-matched random control.
- `generate.py` owns the paired clean/intervened stream and resumable decoding.

The runtime has no band search, calibration ladder, lens fallback, or excluded model
profile. The transferred band comes only from the frozen configuration.

## Evaluation boundary

- `lens_validation.py` implements the registered J-lens versus logit-lens prerequisite.
- `scale_data.py` materializes the exact public-data subset before GPU execution and
  verifies both its internal digest and the preregistered digest on every load.
- `scale_gate.py` executes ordered, checkpointed stages and stops when a registered
  prerequisite fails.
- `metrics.py` is a pure transformation from checkpoint state to metrics and gate checks.

Before loading registered items, the runner also verifies that the companion evaluation
checkout is at the pinned commit with no modified tracked files. Checkpoint resumption
requires the same profile, code commit, configuration digest, data digest, and companion
identity.

Checkpoints live under ignored `extension/state/`. Final results live under
`extension/results/<profile>/` and contain full provenance plus explicit unavailable
stages when an early stop occurs.

## Analysis boundary

`scale_analyze.py` requires both preregistered model result files, combines them with the
hash-verified historical baseline, and produces the fixed table, bootstrap intervals, and
figure. It preserves early-stopped quantities as missing rather than imputing or silently
dropping them.
