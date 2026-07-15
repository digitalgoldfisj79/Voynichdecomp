# Recoverability frontier v0.5.2 — protocol amendment G

Date: 2026-07-15

Status: fixed before execution.

## Inventory-distance audit

For English 384-character ciphertexts, the inferred homophone-count inventory is close to the true observed inventory:

- development: 1–3 one-slot transfers, mean 1.75;
- failed test block: 1–4 transfers, mean 2.65.

A transfer decreases one plaintext label's homophone count by one and increases another label's count by one, subject to the known family multiplicity caps.

## Nested inventory beam

The new search separates the two optimisation levels:

1. **Outer beam:** explore bounded one-slot inventory transfers to depth at most four.
2. **Inner optimisation:** for every outer candidate, fully re-optimise the cipher-symbol assignments while preserving that candidate inventory, using exact pair-block coordinate optimisation under the unchanged train-only quadgram objective.

A candidate inventory is compared only after inner optimisation. This avoids evaluating inventory changes through the low-scoring partially relabelled intermediates that defeated single-level annealing.

## Development schedule grid

English 384-character development trials only:

1. depth 3, beam 4, 8 proposed inventory moves per state, 4 inner block sweeps;
2. depth 4, beam 6, 12 proposed moves, 6 sweeps;
3. depth 4, beam 10, 20 proposed moves, 8 sweeps.

The initial state receives a fixed-inventory block polish before beam expansion. Inventory signatures are deduplicated at every depth.

## Gate

Proceed to a fresh untouched test block only if English development mean recovery reaches at least 70%. The language model, family definition and plaintext metric are unchanged.
