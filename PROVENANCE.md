# Provenance

## Preregistration source

- Source repository: `pgrindehollevik-harvard/jspace-4b`
- Frozen tag: `prereg-v2`
- Source commit: `fbeac5f2262b977d1b2c893c50b8151bba5ed804`
- Configuration SHA-256:
  `18581126da8946d50a7e5cef4d60cb2cc95af7807e25f3fa86b9422889a59b5f`
- Preregistration SHA-256:
  `354ce44fb1e00984a036cf0295a84637cae6c5cb105389877985080697bb548a`

The frozen preregistration, configuration, and deviation log in `extension/` were copied
byte-for-byte from that tag. The repository move does not change a hypothesis, threshold,
model, dataset, band, seed, stopping rule, or analysis endpoint.

The historical-result path embedded in the frozen configuration is source-repository
relative. In this standalone repository, the locked analysis resolves the same copied
artifacts under `baseline/qwen3-4b/`; their byte identities are listed below and in the
baseline manifest.

## Artifact identities

- Qwen3-8B: `b968826d9c46dd6066d109eabc6255188de91218`
- Qwen3-14B: `40c069824f4251a91eefaf281ebe4c544efd3e18`
- Stock Jacobian lenses: `0731326edff4ae730ffc5356fe1a4728c748b3a6`
- Jacobian-lens code/evaluation data:
  `581d398613e5602a5af361e1c34d3a92ea82ba8e`
- Staged Wikitext/GSM8K subset SHA-256:
  `a7f2abfe81ba735ebd7b11341d99502e0f6762a0f82608c58488f6090d8181c6`

## Pre-inference state

Before the frozen tag, infrastructure checks loaded public artifacts and generated short
arithmetic completions. One engineering-only Qwen3-8B smoke run was submitted from the
source repository. Its program could not load a registered evaluation dataset or write a
scientific result. No 8B or 14B scale-gate job had been submitted.

## Historical baseline

The three files under `baseline/qwen3-4b/` are exact copies of the previously observed
4B positive-control, checkpoint, and lens-validation artifacts. Their hashes are recorded
in `baseline/qwen3-4b/MANIFEST.md`. They are included only for the locked cross-scale
analysis and remain explicitly historical.
