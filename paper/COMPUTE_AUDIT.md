# Compute audit: public summary

This summary retains resources and accounting relevant to scientific reproducibility while
omitting hostnames, scheduler identifiers, and queue timestamps from the public release.
All reported runs used one NVIDIA GPU and bfloat16 model weights.

| study stage | device class | allocated GPU time | result |
| --- | --- | ---: | --- |
| v2 engineering and registered gates | L40S | 10.63 minutes | 8B and 14B gate artifacts |
| v3 engineering and registered gate | L40S / H200 | 3.47 minutes | 1.7B gate artifact; stock 32B lens blocker |
| v4 32B lens construction | H200 | 32.33 hours | recipe-faithful 32B lens refit |
| v4 32B smoke and registered gate | H200 | 8.5 minutes | 32B gate artifact |

Total allocation across the project was approximately 32.9 GPU-hours. Registered gate
inference accounts for 19.4 GPU-minutes; the one-time 32B lens construction accounts for
nearly all remaining allocation. The v4 fit exceeded its registered 30-hour cap by 2.33
hours because its wall guard used the scheduler limit. This execution deviation is
preserved in `extension/DEVIATIONS.md`.

Every registered gate log was reviewed after completion. No registered gate reported an
exception, warning, out-of-memory event, non-finite value, or numerical-error marker.
Exact device class, peak CUDA allocation, runtime measurements, software versions, and
scientific results remain in the public artifacts; only operational identifiers are
redacted.
