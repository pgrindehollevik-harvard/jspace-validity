# Version-3 preregistration: the reserved scale endpoints (Qwen3-1.7B and 32B)

**Status:** freeze this document, `extension/config_v3.json`, and the runner in git
before any 1.7B or 32B staging, smoke test, or inference. This registration is written
with full knowledge of the version-2 outcomes (8B lens failure; 14B non-coherence gate
failure) and of the exploratory reanalysis of the frozen artifacts (paired
anti-selectivity; displacement attenuation; flip-set anatomy). The hypotheses below are
stated with that knowledge and are new confirmatory tests only under this registration.
Version-2 artifacts (`extension/PREREGISTRATION.md`, `extension/config.json`, both
registered results, the locked analysis) are unchanged and remain authoritative for v2.

## 1. Why these models, and why now

Version-2 section 3 reserved exactly these two models: "Qwen3-1.7B and Qwen3-32B are
outside the confirmatory experiment. Adding either model ... requires a separately
recorded follow-up; none may be selected because the 8B/14B outcome is inconvenient."
This registration is that separately recorded follow-up. Per version-2 section 9 we
state plainly that it is prompted by observed outcomes: the v2 gate failures and the
exploratory anti-selectivity observation motivate testing (a) whether the complete gate
passes at the remaining pre-authorized scales and (b) whether the anti-selectivity
signature replicates as a preregistered prediction. Both hypotheses are reportable in
either direction; neither model may be dropped after data are seen.

## 2. Hypotheses

**H1 (primary, both models): confirmatory anti-selectivity.** At each scale, J-ablation
damages ordinary text more than the per-slot norm-matched random intervention. Endpoint:
the selectivity index SI = (1 - a_J)/(1 - a_R) over the identical 50-sequence WikiText
protocol, with a_J and a_R the mean top-1 agreements with the clean model. A model
confirms H1 only if BOTH (i) the exact two-sided sign test on the 50 paired per-sequence
(J - random) deltas rejects at alpha = .025 with the median delta negative, and (ii) the
10,000-draw paired sequence percentile-bootstrap 95% CI for SI (seed 20260825) lies
entirely above 1. Alpha is Bonferroni-split (.025 per model, two models). The estimator
is the one published verbatim in the manuscript's exploratory appendix and implemented
in `jspace.frozen_explore`. The SI endpoint requires only the stock lens artifact, not
the stage-1 ranking prerequisite, so it is measured whether or not stage 1 passes.

**H2 (secondary, Qwen3-32B): upward scale rescue.** The fixed intervention passes the
complete version-2 gate at 32B. Given the v2 outcomes we expect failure, and a failure
falsifies the scale-emergence prediction across the full pre-authorized range 1.7B-32B;
a pass localizes an onset between 14B and 32B and licenses the registered coherence
stage.

**H3 (secondary, Qwen3-1.7B): downward boundary.** The identical gate at 1.7B, for
completeness of the family curve. No direction is predicted.

**H4 (secondary, conditional): flip-core overlap.** For any v3 model with a positive
clean-to-J control drop and at least one down-flip, we test whether its down-flip set
overlaps the historical 4B nine-item core {2, 14, 17, 18, 34, 35, 39, 46, 48} (dataset
file-order indices) beyond chance, by the exact hypergeometric upper-tail probability
over that model's clean-correct items, alpha = .05. No direction is predicted; the
exploratory anchor is that the two 14B down-flips were disjoint from the core.

## 3. Models and immutable artifacts

- Primary: `Qwen/Qwen3-32B` at `9216db5781bf21249d130ec9da846c4624c16137`
  (64 layers, hidden 5120).
- Secondary: `Qwen/Qwen3-1.7B` at `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e`
  (28 layers, hidden 2048).
- Stock lenses: `neuronpedia/jacobian-lens` at the v2-pinned revision
  `0731326edff4ae730ffc5356fe1a4728c748b3a6`, files
  `qwen3-32b/jlens/Salesforce-wikitext/Qwen3-32B_jacobian_lens.pt` and
  `qwen3-1.7b/jlens/Salesforce-wikitext/Qwen3-1.7B_jacobian_lens.pt`.
- Companion evaluation code/data, datasets, staged-data digest, thresholds, k, clean
  exemption, seeds, and statistics: identical to version 2 and to
  `extension/config_v3.json`, whose data block is value-identical to the frozen v2
  configuration (staged-data digest `a7f2abfe...81c6` must reproduce).

No other model, band, k, lens, refit, or dataset is authorized by this registration.

## 4. Fixed intervention and band transfer

Unchanged from version 2 in every respect. The observed 4B band 14..19 maps as the
half-open normalized interval [14, 20)/36 with floored boundaries, fixing layers 10..14
at 1.7B (28 layers) and 24..34 at 32B (64 layers). k = 10; clean top-10 exemption;
per-slot norm-matched random control with the same seeding scheme.

## 5. Ordered execution and the one registered procedural change

Each model runs the version-2 stages in the version-2 order with one registered
difference, enabled by `measure_noncoherence_unconditionally` in
`extension/config_v3.json`: the lens-validation and clean-floor outcomes no longer stop
the run before the control and WikiText arms. Every non-coherence stage is measured
unconditionally so that H1 is evaluable even where a stock lens fails its stage-1
prerequisite (as at 8B in v2, where the stop left no downstream data). The gate verdict
itself is unchanged and fully conjunctive: coherence generation runs only if every
earlier condition (lens prerequisite, clean floor, both McNemar conditions, the
2x-random ratio, and WikiText selectivity) passes, and `passes_gate` requires all seven
registered checks. A stopped or failed conjunction remains a reportable result.

## 6. Compute and hardware

One NVIDIA L40S for 1.7B. One NVIDIA H200 for 32B, declared in advance because the 32B
bfloat16 weights (~65 GB) exceed L40S memory (44 GB); this is a capacity declaration,
not the v2 OOM-fallback clause. Cap: 3 GPU-hours per model, 6 total. Engineering smoke
tests (model+lens load, short arithmetic generation, no registered data, no result
artifact) precede each registered job, exactly as in v2. Jobs checkpoint after every
item; infrastructure failures may resume from checkpoints; scientific failures may not
be tuned away.

## 7. Analysis and reporting

Gate verdicts use the unchanged locked metric logic. H1 uses the published SI estimator
with seed 20260825. The locked v2 analysis (`jspace-analyze` and
`extension/analysis/`) is not modified; v3 results are summarized by a separate script
against the raw v3 result artifacts. The manuscript will label v3 as a registration
made after v2 outcomes were known, will report all v3 outcomes regardless of direction,
and will not reinterpret any v2 verdict. Any post-freeze change goes in
`extension/DEVIATIONS.md` before rerunning and must state whether it was prompted by
observed outcomes.
