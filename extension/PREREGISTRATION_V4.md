# Version-4 preregistration: completing the blocked 32B arm with a recipe-refit lens

**Status:** freeze this document, `extension/config_v4.json`, and the runner in git
before any lens fitting or 32B inference under this registration. It is written with
full knowledge of the v2 and v3 outcomes, including the 1.7B anti-selectivity
confirmation and the v3 finding that the upstream 32B lens artifact is an interrupted
fit checkpoint. The v2 and v3 registrations, configurations, and results are unchanged.
Version 2 section 3 reserved lens refits for "a separately recorded follow-up"; this is
that follow-up for exactly one model.

## 1. The instrument

The stock lens repository ships no finished Qwen3-32B lens (its head equals the pinned
revision; the only artifact is a checkpoint halted after 80 prompts). We therefore fit a
lens from scratch with the released recipe at the pinned companion code
(`jlens.fitting.fit`): wikitext-103-raw-v1 train at the pinned dataset revision, prompts
in dataset order (first 1,000 rows whose stripped text is at least 600 characters,
truncated to 2,000 characters), `dim_batch` 128, `max_seq_len` 128, bfloat16, at most
1,000 prompts. The released early stop (0.002 per-prompt mean relative change, minimum
100 prompts, window 10) is reimplemented as a registered block rule: stop at the first
multiple of 10 prompts, at or beyond 100, where the running mean changed by less than
0.02 (maximum over layers) across the preceding 10 prompts. `torch.compile` is disabled.
The fit prompt stream (eligible ranks 0-999) is disjoint by construction from the
registered evaluation slice, which skips the first 2,000 eligible rows. The finished
lens is saved in the stock format; its SHA-256, prompt count, and convergence trace are
recorded in a manifest before any gate inference, and the gate result records the
override file and digest. Fitting is engineering work on public data: it reads no
registered evaluation item and writes no gate result. Checkpoint resume after
preemption or wall limits is an infrastructure retry.

## 2. Hypotheses (identical endpoints to version 3, one model)

- **H1 (primary, confirmatory): anti-selectivity at 32B.** Same endpoint and thresholds
  as v3 H1: exact two-sided sign test on the 50 paired per-sequence WikiText differences
  and a 10,000-draw paired bootstrap selectivity-index 95% interval above 1, both at
  alpha .025, measured unconditionally with respect to the lens prerequisite.
- **H2 (secondary): the complete v2 gate at 32B** with the refit lens. A pass localizes
  a scale onset between 14B and 32B; a fail extends the negative result to the largest
  reserved scale under a recipe-complete instrument.
- **H3 (secondary, conditional): flip-core overlap**, with the marked class defined as
  the 4B core items that are clean-correct at 32B (the corrected population definition
  recorded in the deviation log), alpha .05, no predicted direction.
- **Interpretation guard:** the refit lens is not the stock artifact; outcomes are
  reported as "recipe-refit lens" results and never merged into stock-lens rows.

## 3. Protocol, models, and compute

Identical to v3 in every other respect: same gate order and thresholds, same
unconditional non-coherence measurement, same datasets and staged-data digest, band
24-34 (64 layers) under the frozen transfer rule, k = 10, same seeds and statistics.
Model `Qwen/Qwen3-32B` at `9216db5781bf21249d130ec9da846c4624c16137` on one NVIDIA H200
(65 GB bf16 weights exceed L40S memory). Caps: 30 GPU-hours for the fit, 3 for the gate.
Engineering smoke (model plus refit lens load, short arithmetic generation, no
registered data) precedes the gate. No other model, band, k, or lens is authorized.

## 4. Reporting

All outcomes are reported once the run completes, in both directions, including a lens
that fails the stage-1 prerequisite. At the authors' direction the manuscript
incorporates v4 results only after the run is complete; until then the paper reports
the v3 blockage unchanged. Any post-freeze change goes in `extension/DEVIATIONS.md`
before rerunning and must state whether it was prompted by observed outcomes.
