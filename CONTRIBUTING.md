# Contributing

Use one branch and one pull request per logical change. Keep infrastructure, scientific
results, analysis, and manuscript edits separate so each can be reviewed on its own.

Before opening a pull request:

```bash
uv lock --check
uv run ruff check src tests/unit
uv run ruff format --check src tests
uv run pytest
```

The preregistration and `extension/config.json` are immutable after the public
`prereg-v2` tag. A necessary scientific change must be documented in
`extension/DEVIATIONS.md` before rerunning, with its timing and relationship to observed
outcomes. Infrastructure-only fixes should preserve configuration and data digests.

Do not commit credentials, local caches, checkpoints, scheduler logs, generated paper
build products, or internal drafting notes. Result commits must include the code commit,
configuration digest, data digest, hardware identity, and complete stop status. Keep
operational scheduler identifiers outside public result artifacts.
