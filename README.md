# Worse Than Random: J-Space Ablation Is Anti-Selective at Every Measured Qwen3 Scale

**Peter Flo · Harvard University**

**Accepted at the NeurIPS 2026 Workshop on Interpretability as a Science**

[Read the paper](paper.pdf) · [Release notes](RELEASE.md) · [Citation](CITATION.cff)

This repository contains the code, preregistrations, analysis, and released result
artifacts for a study of whether J-space ablation selectively disrupts internal reasoning
while leaving ordinary language processing intact. The source study argues that J-space
directions form a workspace-like, verbalizable channel. We test a necessary condition for
that interpretation on the open Qwen3 family: targeted J-space ablation should be less
damaging to ordinary next-token prediction than a matched random-direction intervention.

It is not, at any scale where that paired comparison was measured.

| Qwen3 scale | status | paired WikiText sequences where J is worse | selectivity index | role |
| --- | --- | ---: | ---: | --- |
| 1.7B | measured | 50 / 50 | 3.38 | confirmatory |
| 4B | measured | historical calibration | 2.17 | historical context |
| 8B | stopped at lens prerequisite | not measured | not applicable | preregistered stop |
| 14B | measured | 49 / 50 | 1.57 | exploratory transfer |
| 32B | measured with a recipe-faithful refit lens | 42 / 50 | 1.16 | confirmatory |

The selectivity index is J-arm damage divided by matched-random-arm damage. Values above
one mean that J-ablation damages ordinary text more than the local norm-matched random
control. The 1.7B and 32B endpoints were separately preregistered and confirm this
anti-selective signature; 4B and 14B provide historical or exploratory context. The
results do **not** test the source study's frontier-scale models, so they leave open a
falsifiable possibility that selectivity may emerge beyond 32B.

## What is in the repository

- `paper.pdf`: public, deanonymized rendering of the accepted manuscript.
- `paper/`: manuscript source, figures, bibliography, and a public compute summary.
- `extension/`: frozen v2, v3, and v4 preregistrations; configurations; raw result
  artifacts; and locked analyses.
- `baseline/qwen3-4b/`: hash-verified historical 4B artifacts used only as context.
- `src/jspace/`: the intervention, paired clean/intervened generation, gate runner,
  refit recipe, and analysis code.
- `tests/unit/`: model-free regression tests.
- `docs/`: architecture and cluster-execution guidance.

Model weights, fitted-lens binaries, public datasets, and the source study's companion
repository are deliberately not redistributed. The code pins their revisions and stages
them from their original public locations.

## Reproduce the released analyses

The commands below validate the software and recompute the analyses that use committed
result artifacts. They do not download a model or require a GPU.

```bash
uv sync --frozen --dev
uv lock --check
uv run ruff check src tests/unit
uv run ruff format --check src tests
uv run pytest
uv run jspace-analyze
uv run python -m jspace.v3_summary
make -C paper public
```

`jspace-analyze` regenerates the locked v2 table and figure under
`extension/analysis/`; `v3_summary` regenerates the endpoint summaries under
`extension/analysis_v3/`. The public manuscript is written to `paper.pdf`.

## Running the GPU experiments

GPU jobs require separately staged public assets. The committed Slurm templates were
used on MIT ORCD but contain no account, host, socket, or credential configuration.
Start with [the cluster execution guide](docs/ORCD.md), set `JSPACE_ROOT` for your own
checkout, and inspect the frozen preregistration before running a gate. The gate runner
refuses dirty tracked code and records the exact pinned data and configuration digests.

## Provenance and scope

The 4B historical baseline originates in the earlier
[`jspace-4b`](https://github.com/pgrindehollevik-harvard/jspace-4b) preregistration.
The final multi-scale results, including 1.7B and 32B, live in this repository. Exact
source revisions and artifact hashes are recorded in [PROVENANCE.md](PROVENANCE.md).
Public-release redactions are documented in [RELEASE.md](RELEASE.md); they remove only
operational identifiers, not scientific outcomes, configurations, or analyses.

## Citation

Please use [CITATION.cff](CITATION.cff) or the following:

```bibtex
@inproceedings{flo2026worse,
  title = {Worse Than Random: J-Space Ablation Is Anti-Selective at Every Measured Qwen3 Scale},
  author = {Flo, Peter},
  booktitle = {NeurIPS 2026 Workshop on Interpretability as a Science},
  year = {2026}
}
```

## License

MIT. See [LICENSE](LICENSE).
