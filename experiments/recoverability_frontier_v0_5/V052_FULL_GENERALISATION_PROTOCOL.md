# Recoverability frontier v0.5.2 — six-language homophonic generalisation

Date: 2026-07-15

Status: **frozen before execution**

## Fixed architecture

The passing confirmation architecture is unchanged:

- first-occurrence recurrence canonicalisation;
- bounded variable homophone inventory;
- train-only smoothed character quadgram model, alpha `0.05`;
- unigram score weight `0.12`;
- `700,000` annealing iterations × `50` restarts;
- no nulls and no channel noise.

No further development or schedule selection is allowed.

## Test design

- languages: English, German, Finnish, Turkish, Hebrew and Arabic;
- lengths: 96, 192 and 384 normalized characters;
- 20 unseen source chunks and keys per language × length;
- test replicate block begins at 64;
- total: 360 trials;
- six independent CPU-XL language shards run in parallel;
- each shard preserves its complete row-level JSON in the immutable job log.

Replicates 64–83 are disjoint from all earlier homophonic smoke and confirmation blocks.

## Frozen gate

Proceed to null-bearing homophonic development only if:

- overall mean recovery is at least 70%;
- every language reaches at least 50%;
- 96-character mean recovery is at least 60%;
- longer texts do not show a material collapse;
- no post-test model, inventory or schedule changes are made.

## Boundary

Passing establishes bounded synthetic homophonic recoverability only. It does not establish blind family identification, generic cipher detection or any Voynich conclusion.
