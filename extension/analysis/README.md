# Registered scale-gate analysis

These files are the locked post-processing outputs of `uv run jspace-analyze`.
The command combines the two registered extension artifacts with the historical Qwen3-4B
light-band result and uses the preregistered bootstrap seed (`20260825`) and 10,000 draws.

- `summary.json` retains the complete machine-readable rows and confidence intervals.
- `scale_table.csv` is the manuscript-ready tabular extract.
- `scale_gate.pdf` and `scale_gate.png` are equivalent renderings of the reserved figure.

Qwen3-8B has missing downstream values because its stock lens failed the registered
validation prerequisite. Qwen3-14B passed lens validation and the clean-control floor,
but failed the positive-effect, random-specificity, and Wikitext-selectivity conditions;
the ordered stop rule therefore skipped coherence generation. Neither extension model
passed the complete gate. The 4B row is historical context, not confirmatory evidence.

The result artifacts used here have SHA-256 digests
`308ced39d6f7d85a6c00f390c79e36012f7954bde8d9ad2da22aeef2cd359fb1`
(8B) and `e8287016b93cadc8a0e7eb9ebc6e5167b739ae23c7cd537aef2e002b5b846451`
(14B).
