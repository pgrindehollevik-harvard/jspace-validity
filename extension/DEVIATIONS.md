# Version-2, version-3, and version-4 deviations

## 2026-08-27 (v4): fit cap overrun and final chunk width

The registered lens fit converged by its registered block rule at 760 prompts but
consumed 32.33 GPU-hours against the registered 30-hour cap: the driver's wall guard was
set to the scheduler job limit (35 hours) rather than the registered cap, an estimation
and configuration error noticed post hoc and not prompted by any outcome. The fit ran at
Jacobian chunk width 16 under the previously logged env-override mechanism (the released
128 and the halved 64 both exceeded H200 memory); the manifest records the width, the
prompt-stream digest, the convergence trace, and the finished lens digest. The gate-stage
cap (3 hours) was respected with 8.5 minutes used.

## 2026-08-25 (v4): halve the Jacobian chunk width for H200 memory

The second lens-fit job ran out of GPU memory on its first prompt: the released recipe's
`dim_batch=128` vectorizes the Jacobian forward 128-wide, which fit the 179 GB B200 the
upstream lenses were fit on but exceeds the 140 GB H200. The chunk width only partitions
the same Jacobian computation and is mathematically equivalent at any value; the driver
now defaults to 64 (env-overridable), records the value used in the fit manifest, and
enables the PyTorch expandable-segments allocator. Prompted by the pre-fit out-of-memory
traceback, not by any observed outcome; no prompt had been fitted.

## 2026-08-25 (v4): stage the fit prompts on the login node

The first lens-fit job failed in seconds, before loading the model or any prompt: the
offline compute node cannot resolve the Hugging Face dataset builder
(`OfflineModeIsEnabled`). Following the repository's standard staging pattern (and the
analogous v2 lens-loading deviation), the driver now stages the registered prompt stream
to a local JSON file on the login node and offline jobs read that file; its SHA-256 is
recorded in the fit manifest. The prompt rule, count, order, and truncation are unchanged
from the registration. Prompted by the pre-fit infrastructure traceback, not by any
observed outcome; no fitting had occurred.

## 2026-08-25 (v3): post-data corrections to the summary script (no endpoint changed)

Two changes to `src/jspace/v3_summary.py` after the 1.7B artifact existed, both disclosed
in the manuscript appendix: (1) descriptive j-drop and specificity intervals following the
v2 reporting convention were added; they feed no registered endpoint, and the H1/H4
outputs are byte-identical to the pre-data script (independently verified). (2) The H4
hypergeometric implementation was corrected to match the registered wording: the null
draws down-flips from the model's clean-correct items, so the marked class is the core
items present in that population (four of nine at 1.7B), not the full nine-item core. The
reported tail probability changes from .74 to .42; the conclusion (no overlap signal) is
unchanged. Prompted by an independent verification pass, not by any preferred outcome.

## 2026-08-25 (v3): restage the data cache under the v3 configuration identity

The first registered 1.7B gate job stopped in seven seconds, before loading any model,
lens, or registered data: `load_staged_data` embeds the whole-configuration digest at
staging time, and the cache staged for version 2 carried the v2 digest while the job ran
under `extension/config_v3.json`. The cache was restaged on the login node with the v3
configuration; the staging command verified that the scientific payload reproduced the
preregistered digest `a7f2abfe...81c6` exactly (identical WikiText slice and GSM8K
sample). Prompted by a pre-metric infrastructure guard, not by any observed outcome; the
gate was then resubmitted unchanged.

## 2026-08-25 (v3): the 32B arm is blocked by an upstream lens artifact defect

The registered 32B engineering smoke failed in 37 seconds, before any registered data
access: the pinned stock lens file
`qwen3-32b/jlens/Salesforce-wikitext/Qwen3-32B_jacobian_lens.pt` at revision
`0731326e...` is not a `JacobianLens` artifact but an interrupted `fit()` checkpoint
(keys `jacobian_sum`, `n_done`, `next_idx`, `source_layers`; `n_done = 80` prompts; its
convergence log ends at prompt 80 with delta about 0.03, far from the released recipe's
0.002 early stop). The repository's `main` revision is identical to the pinned revision,
so no finished stock 32B lens exists upstream. Finalizing or refitting the lens ourselves
is outside the version-3 authorization, so the 32B arm is reported as blocked: no
registered 32B measurement exists, and no v3 hypothesis is evaluated at 32B. Prompted by
the artifact defect, not by any observed outcome.

## 2026-08-25 (v2): resolve the staged lens without Hub metadata

An engineering smoke run failed before generation because the public
`JacobianLens.from_pretrained` helper attempted to query Hugging Face repository metadata
while `HF_HUB_OFFLINE=1` was active. The matching 8B lens file had already been staged at
the preregistered repository revision.

The runtime now resolves that exact file through `hf_hub_download` with
`local_files_only=True` during offline jobs and passes its path to `JacobianLens.load`.
This changes no lens bytes, model, data, intervention, seed, threshold, stop rule, or
analysis. The change was prompted only by the pre-metric infrastructure traceback: the
failed smoke could not load registered data or write a scientific result.
