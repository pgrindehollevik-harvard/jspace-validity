# Public release notes

This is the source tree for a fresh-history, public research release of the final
multi-scale study. It is intentionally separate from the private working repository and
its review history.

## What changed for the public release

- The manuscript has a deanonymized public rendering with Peter Flo's byline and the
  workshop acceptance notice.
- The root README, citation metadata, and paper build instructions describe the final
  1.7B, 4B, 8B, 14B, and 32B evidence rather than the earlier v2-only transfer study.
- Review-only files and the account-specific deployment helper are excluded.
- Hostnames, scheduler job identifiers, and queue timestamps are removed from public
  result artifacts and compute documentation.

## What did not change

No model, dataset, prompt selection, intervention, frozen configuration, statistical
endpoint, result value, figure, or scientific conclusion was altered for release. The
committed result artifacts retain metrics, paired hit vectors, per-sequence agreements,
displacement summaries, hashes, software versions, device classes, and preregistration
provenance. The omitted operational identifiers are not needed to reproduce or evaluate
the analysis.

The public PDF is generated from `paper/public.tex`; run `make -C paper public` to
rebuild `paper.pdf`.
