# Recoverability frontier v0.5.2 — protocol amendment F

Date: 2026-07-15

Status: fixed before hybrid development execution.

## Hybrid search

The inferred homophone-label multiset remains fixed. Each deterministic outer restart performs:

1. exact pair-block coordinate optimisation;
2. swap-only simulated annealing, which preserves the same inventory;
3. a second exact pair-block polish.

The best candidate is selected solely by the unchanged train-only quadgram objective.

## Development scope

English 384-character development trials are the primary gate because Hebrew already reaches 99.5117% under fixed-inventory pair blocks.

Development schedule grid:

- 3 outer restarts, 5 block sweeps, 50,000 annealing iterations × 2 inner restarts;
- 6 outer restarts, 8 block sweeps, 100,000 × 3;
- 12 outer restarts, 12 block sweeps, 200,000 × 4.

Proceed only if English reaches at least 70% mean recovery. Any later test must use a new untouched source block.
