# Balance study — isolation contract

This study asks why adding input features to the check-in representation does not translate into
downstream gains, and whether a *balanced* embedding fixes it.

## Isolation

Nothing outside these four paths is modified:

    scripts/balance/          study scripts
    research/balance/         study-local model code
    docs/studies/balance/     plans and reports
    docs/results/balance/     result JSONs

Engines produced here are named `balance_*` and are registered in the engine enum through the same
serialized lock the integrity study used, so v18 engines are untouched. Representation checkpoints go
to `results/balance/<state>/<cell>/`, never into a v17, v18 or `check2hgi_dk_*` directory; the builder
refuses to write into those.

A git worktree was the intended container. Worktree creation is blocked in this environment (the
sandbox denies writes under `.git/`), so the isolation is by path convention plus the builder's
refusal, and it is asserted in code rather than assumed.
