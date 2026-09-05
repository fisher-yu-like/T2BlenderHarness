# Owner Challenge Set v1

This train-only challenge set is generated deterministically by
`videoact.owner_challenges.build_default_challenge_set()` and materialized
with:

```text
uv run python scripts/validate_owner_challenges.py --write-default
uv run python scripts/validate_owner_challenges.py
```

It contains five positive/negative pairs for each of the seven primary
Harness owners. The generated manifest is an executable challenge artifact,
not frozen test data.
