# v0.6 Family T — component-oracle result

Date: 2026-07-16

Verdict: **T1 PASS; T2 PASS; FULLY BLIND T3 IS AUTHORISED.**

No test data or Voynich text was scored.

## Job

Hugging Face job: `Digitalgoldfish79/6a589123b1669a49bf077088`

Scientific SHA-256: `60db0399ec02b6abdf87a7cffcf8bfe4a75df1c75a6f0605e544bec2f881083b`

## Configuration

- English development split;
- 16 ciphertexts of length 384;
- 8 global and 8 line-reset ragged columnar transpositions;
- widths 4–10;
- no padding or exposed rectangle dimensions;
- fresh monoalphabetic substitution composed before transposition;
- permutation search: `200,000 × 32` per mode-width candidate;
- validated monoalphabetic search: `700,000 × 50`.

## T1 — true substitution key; unknown mode, width and column order

- mean plaintext recovery: **99.9674%**;
- median: **100%**;
- minimum: **99.4792%**;
- mode recovery: **16/16**;
- width recovery: **16/16**;
- exact canonical column order: **15/16**.

The single non-exact permutation still recovers 99.48% of the plaintext.

## T2 — true transposition; unknown substitution key

- mean plaintext recovery: **99.5117%**;
- median: **99.6094%**;
- minimum: **98.1771%**;
- all 16 trials exceed 95%;
- exact plaintexts: 4/16.

## Decision

Both component gates pass. Family T proceeds to a fully blind coordinate solver alternating:

1. frequency-seeded substitution estimation;
2. mode-width-specific column-order optimisation;
3. validated monoalphabetic refinement;
4. full mode-width model comparison.

The test split remains sealed.