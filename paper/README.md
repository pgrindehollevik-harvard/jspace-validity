# Paper source

This directory contains the public, deanonymized source for:

> **Worse Than Random: J-Space Ablation Is Anti-Selective at Every Measured Qwen3 Scale**
>
> Peter Flo · Harvard University
>
> Accepted at the NeurIPS 2026 Workshop on Interpretability as a Science

Build the public PDF from this directory with:

```bash
make public
```

The result is written to `../paper.pdf`, with intermediates under `build/`. The public
renderer is `public.tex`; it enables the author line, workshop acceptance note, and PDF
metadata while sharing the exact manuscript body and figures in `main.tex`.

`make review` builds the archival anonymous rendering from the same source. It is retained
only to make the source structure transparent; the public release uses `make public`.

The paper figures are committed under `figures/`. `CLAIM_LEDGER.md` maps empirical prose
to immutable result artifacts, and `COMPUTE_AUDIT.md` gives the public compute summary.
References are recorded in `references.bib`.
