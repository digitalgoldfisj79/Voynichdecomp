# Recoverability frontier v0.5.2 — protocol amendment B

Date: 2026-07-15

Status: fixed after the flexible-inventory smoke, before the next execution.

## Evidence motivating the amendment

The flexible inventory search improved the previously inspected smoke test from 57.2049% to 59.7222%, but still failed. More search consistently reduced recovery while improving the search objective:

- 100,000 × 20 development: 71.2240%;
- 300,000 × 35 development: 69.1406%;
- 700,000 × 50 development: 60.0260%.

This pattern indicates objective-model overfitting rather than insufficient search depth.

## Frozen objective change

Replace the smoothed character trigram objective with a smoothed character quadgram objective plus the same small unigram term. The model remains fitted on each language's `train` partition only.

No cipher, inventory, search-move or gate definition changes are permitted in this execution.

## Fresh smoke holdout

The previously inspected smoke used test source-chunk replicates 0–5. The quadgram execution must use a disjoint untouched block beginning at replicate 20. Development remains on the corpus `dev` partition.

The quadgram smoke passes only if the untouched English/Turkish holdout achieves:

- at least 70% overall mean recovery;
- at least 60% in each language;
- improvement over 59.7222%.

Failure stops v0.5.2 before any six-language run.
