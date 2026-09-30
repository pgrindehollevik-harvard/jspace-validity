# Exploratory reanalysis (post hoc)

`frozen_reanalysis.json` is produced by

```bash
uv run python -m jspace.frozen_explore
```

Its inputs are three immutable artifacts — the registered 14B result and the two
historical 4B calibration files — whose SHA-256 digests are recorded inside the output.
Everything here was computed after the registered outcomes were known: it is exploratory,
hypothesis-generating, and feeds no registered gate decision. The locked confirmatory
analysis lives in `extension/analysis/` and is unchanged.

The script is deterministic (bootstrap seed 20260825, 10,000 draws) and asserts agreement
between its recomputed exact McNemar values and the registered ones before writing output.
The cross-model flip comparison assumes both studies stage the pinned order-of-operations
evaluation file in file order; within-4B comparisons need no such assumption.
