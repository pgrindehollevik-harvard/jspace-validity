## Scope

<!-- State the single purpose of this change and what is intentionally out of scope. -->

## Scientific integrity

- [ ] This change does not alter a frozen hypothesis, endpoint, threshold, sample, seed,
      stop rule, or model-specific intervention choice.
- [ ] Any scientific deviation is recorded in `extension/DEVIATIONS.md`, including
      whether observed outcomes motivated it.
- [ ] No registered result was inspected to choose this implementation.
- [ ] New factual claims and citations were checked against primary sources.

## Verification

- [ ] `uv lock --check`
- [ ] `uv run ruff check src tests/unit`
- [ ] `uv run ruff format --check src tests`
- [ ] `uv run pytest`
- [ ] Relevant GPU/integration evidence is attached or explicitly not applicable.
