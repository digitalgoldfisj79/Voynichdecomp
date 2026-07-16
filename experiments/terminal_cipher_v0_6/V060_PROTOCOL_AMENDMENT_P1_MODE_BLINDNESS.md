# v0.6 protocol amendment P1 — mode-blind Family P

Date: 2026-07-16

Status: **FROZEN BEFORE ANY FULLY BLIND DEVELOPMENT RESULT**

No test data or Voynich text has been inspected.

## Defect found

The initial Family P harness supplied the true operating mode (`periodic` or `line_reset`) to the candidate-period search. That is valid for component-oracle work but not for the registered fully blind stage.

The first oracle result remains evidence that the two component channels are individually recoverable, but it is superseded as the formal gate because periodic trials did not carry independent observed line boundaries.

## Correction

Every trial now receives an independently generated observed line partition, regardless of its true operating mode. The partition is generated from a separate deterministic seed and therefore cannot alter the wheel alphabet, period or shifts.

Encryption then uses either:

- continuous periodic phase; or
- phase reset at each observed line start.

The fully blind solver must jointly compare all 22 structural candidates:

- 2 operating modes;
- periods 2–12.

Candidate selection uses the same train-only trigram-plus-unigram objective and the same BIC-like period penalty. No extra preference is assigned to either operating mode.

## Corrected oracle gate

Sixteen English development trials at 384 characters: 8 periodic and 8 line-reset.

Both conditions are required:

1. true schedule, unknown mixed wheel:
   - mean recovery at least 95%;
   - minimum recovery at least 90%;
2. true mixed wheel, unknown mode, period and shifts:
   - mean recovery at least 95%;
   - minimum recovery at least 90%;
   - exact mode-plus-period recovery in at least 14/16 trials.

Failure blocks fully blind development.

## Fully blind development gate

The first fully blind schedule is frozen at:

- 250,000 proposals × 24 restarts per structural candidate;
- all 22 mode-period candidates;
- 16 English development ciphertexts of length 384.

It passes only if all conditions hold:

- mean plaintext recovery at least 80%;
- median recovery at least 90%;
- at least 14/16 trials recover at least 80%;
- operating-mode accuracy at least 14/16;
- exact mode-plus-period accuracy at least 12/16.

One development-only amendment is permitted. It may alter search architecture, proposal mix, restart count or training scale, but not the corpus, ciphertexts, family definition, objective data, gate or test split.

A passing development configuration is frozen and evaluated once on a fresh untouched test block with periods drawn from 5, 7, 9, 10 and 12. No post-test amendment is permitted.