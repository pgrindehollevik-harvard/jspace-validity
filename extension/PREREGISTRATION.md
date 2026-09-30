# Version-2 preregistration: a fixed-intervention scale test

**Status:** freeze this document, `extension/config.json`, and the runner in git before
the first 8B or 14B scientific job. The existing 4B observations were known while this
document was written and are therefore historical evidence, not new confirmatory data.

## 1. Question and workshop-sized claim

The 4B study found that the light-band J-space intervention had a specific causal effect
on the order-operations positive control: accuracy fell by 16.4 points under J-ablation
and by 0 points under a per-slot norm-matched random intervention. It nevertheless changed
ordinary next-token predictions too often (66.1% top-1 agreement, below the registered 80%
selectivity threshold), so the planned chain-of-thought grid correctly stopped.

The natural extension asks one narrow question: **does the same intervention become a
valid, selective causal probe at a larger scale within the Qwen3 family?** This is a
measurement-validity study. It does not run the old GSM8K/MATH/AIME difficulty grid and
will not claim that chain of thought protects reasoning unless this new gate succeeds.

The intended Interpretability as a Science workshop contribution is the combination of
(i) a preregistered stop rule, (ii) a matched-damage causal control, and (iii) a scale test
showing where the intervention does or does not license a mechanistic claim.

## 2. Hypotheses and possible conclusions

**H1 (primary, Qwen3-14B):** the fixed light-band intervention passes the complete validity
gate in section 7. In particular, it lowers positive-control accuracy beyond the matched
random intervention while preserving more than 80% of clean top-1 predictions on held-out
Wikitext and producing fewer than 10% degenerate generations.

**H2 (secondary, Qwen3-8B):** the identical gate localizes the onset. A pass at 8B places
the boundary between 4B and 8B; an 8B failure followed by a 14B pass places it between 8B
and 14B. Failures at both sizes falsify the scale-emergence prediction in the tested range.

If a stock lens fails the lens-validation prerequisite, we report a measurement failure
for that model and make no claim about the model's internal mechanism. If 14B passes but
8B does not, that is evidence of a scale-associated boundary under this stock-lens
pipeline, not proof that parameter count rather than lens quality caused the boundary.

## 3. Models and immutable artifacts

- Primary: `Qwen/Qwen3-14B` at
  `40c069824f4251a91eefaf281ebe4c544efd3e18`.
- Secondary: `Qwen/Qwen3-8B` at
  `b968826d9c46dd6066d109eabc6255188de91218`.
- Stock lenses: `neuronpedia/jacobian-lens` at
  `0731326edff4ae730ffc5356fe1a4728c748b3a6`, using the exact matching 8B and 14B
  Wikitext lens files declared in `src/jspace/profiles.py`.
- Companion evaluation code/data: `anthropics/jacobian-lens` at
  `581d398613e5602a5af361e1c34d3a92ea82ba8e`.
- Historical comparator only: the already observed Qwen3-4B light-band rung in
  `results/calibration.json`.

Qwen3-1.7B and Qwen3-32B are outside the confirmatory experiment. Adding either model,
refitting a lens, or running the original difficulty grid requires a separately recorded
follow-up; none may be selected because the 8B/14B outcome is inconvenient.

## 4. Fixed intervention

The implementation is unchanged from the 4B study: at each token position and selected
layer, remove the joint projection onto the top ten J-lens directions after exempting the
clean stream's top ten next-token ids. The control removes the same per-slot displacement
norm along random directions. Decoding, prompts, and seeds are identical across sizes.

There is one band and no calibration ladder. The observed 4B light band 14..19 is treated
as the half-open depth interval `[14,20)` out of 36 blocks. For a target with `L` blocks,
both interval boundaries are multiplied by `L/36` and floored; the returned block range is
inclusive. This fixes 8B at 14..19 and 14B at 15..21. `k=10` is fixed. No result-dependent
band, layer, `k`, or prompt choice is allowed.

## 5. Data fixed before the run

1. **Lens validation:** all 55 order-operations and all 93 multihop items from the pinned
   companion repository, evaluated with the existing single-token synonym and pass@k
   protocol. These items only test whether the J-lens improves on the vanilla logit lens.
2. **Positive control:** all 55 order-operations items. Each gets an 8-token greedy
   completion in clean, J-space, and norm-matched-random conditions. The target substring
   rule and exact paired hit vector are retained.
3. **Selectivity:** 50 Wikitext-103 training sequences from pinned revision
   `b08601e04326c79dfdd32d625aee71d232d685c3`. Apply the existing deterministic rule:
   skip the first 2,000 rows with at least 600 characters, take the next 50, truncate to
   2,000 characters and then 128 tokens. The endpoint is teacher-forced top-1 agreement
   with the clean model, separately for J and random interventions.
4. **Coherence:** 30 GSM8K training problems from pinned revision
   `740312add88f781978c0658806c59bc2815b9866`, sampled with Python seed 100. Generate
   non-thinking CoT with seed 1 and a 640-token cap. Degenerate means cap hit or maximum
   four-gram repetition fraction above 0.5, matching v1.

The login-node staging step materializes the exact Wikitext/GSM8K subset into an ignored
local cache and records its SHA-256. GPU jobs are offline and include that digest in every
result.

## 6. Ordered stop rules

Each model proceeds independently and in this order:

1. Stop if the stock J-lens does not weakly beat the logit lens on pass@1 and pass@5,
   with a strict improvement on at least one, for at least one of the two lens evals.
2. Stop if clean order-operations accuracy is below 60%.
3. Run the J and random positive-control arms and both Wikitext arms. Stop before long
   coherence generation if any non-coherence gate condition in section 7 fails.
4. Otherwise run the 30 coherence generations and finalize the gate.

A stopped model remains a reportable result. Jobs checkpoint after every control item,
Wikitext sequence, and generation chunk. Infrastructure failures may be retried from the
checkpoint; scientific failures may not be tuned away.

## 7. Complete validity gate and analysis

A model passes only when every condition below is true:

- lens validation passes section 6.1;
- clean positive-control accuracy is at least 60%;
- clean-to-J drop is positive with two-sided exact McNemar `p < .05`;
- J accuracy is lower than random accuracy with two-sided exact McNemar `p < .05`;
- `drop_J >= 2 * max(drop_random, 0)`;
- Wikitext top-1 agreement under J-ablation is strictly above 80%;
- degenerate-generation rate is strictly below 10%.

We report all paired hit vectors; accuracy differences; exact p-values; per-sequence
Wikitext agreement; random-arm agreement; mean per-layer relative displacement norms;
degenerate flags; 95% problem/sequence bootstrap intervals; runtime; throughput; and peak
GPU memory. H1 is a single primary 14B conjunction, so its component conditions are not
treated as separate discoveries. The 8B test is explicitly secondary. The known 4B value
is shown descriptively and is never re-labelled as preregistered evidence.

## 8. Compute and hardware boundaries

The confirmatory cap is six GPU-hours per model and twelve total. One L40S is the default
for both sizes. If and only if a compatibility smoke test or scientific job OOMs before
producing a metric, it may be rerun unchanged on one H200; this is an infrastructure
fallback, not a model-dependent intervention choice. Runs use bfloat16 and one GPU.

The profile-specific smoke test may load artifacts and generate a short arithmetic answer,
but may not inspect a registered dataset or write a scientific result. The old full grid,
32B model, model-specific band search, lens refit, and any calibration ladder are outside
this compute authorization.

## 9. Deviations and authorship safeguards

Any post-freeze change goes in `extension/DEVIATIONS.md` before rerunning and must state
whether it was prompted by observed outcomes. The manuscript will distinguish prediction,
observation, and interpretation. Every citation must point to a source actually opened by
an author; the citation manifest and all generated prose receive a human final pass.
