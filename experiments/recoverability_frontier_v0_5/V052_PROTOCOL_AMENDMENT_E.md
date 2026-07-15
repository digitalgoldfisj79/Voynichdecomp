# Recoverability frontier v0.5.2 — protocol amendment E

Date: 2026-07-15

Status: fixed before execution.

## Evidence

English has the smallest plaintext alphabet in the six-language battery, so the failure is not caused by alphabet dimensionality.

At 384 characters:

- mean initial inferred-inventory overlap: 95.8729%;
- flexible annealing final overlap: 90.3431%;
- flexible pair-block development final overlap: 89.3783%.

The current search damages an already strong inventory estimate. Meanwhile, exact-inventory single-symbol search fails because coordinated assignments are required.

## Frozen diagnostic

Use the pair-block optimiser while preserving the complete inferred plaintext-label multiset exactly:

- random restarts permute the same inferred label multiset;
- each pair-block reassignment preserves the current count of both labels;
- no label can be introduced, removed, split or merged;
- the quadgram objective and all corpus boundaries are unchanged.

Run English and Hebrew development data at 384 characters using the same pair-block schedule grid.

Proceed only if both languages reach at least 70% mean recovery. Any later validation must use a fresh untouched test block.
