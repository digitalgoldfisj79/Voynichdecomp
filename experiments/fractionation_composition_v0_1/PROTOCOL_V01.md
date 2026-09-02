# Fractionation Composition v0.1 — synthetic identifiability gate

Status: preregistered development gate. Voynich remains sealed until the gate passes.

## Question

Can a detector recover the obligatory two-role phase structure of bounded fractionation/regrouping after controlling for token length, symbol counts, and a substantial prefix/core/suffix positional grammar?

This does **not** ask whether Voynich is a Polybius cipher, and a positive result would not identify plaintext. It asks whether the proposed family has an observable signature that can be distinguished from known slot-like confounds before Voynich is examined.

## Positive arms

1. `frac_pair`: simple coordinate pair expansion, equivalent in spirit to the v0.5 fractionated baseline.
2. `frac_block_token`: coordinate fractions regrouped in blocks of width 2–8, resetting at token boundaries.
3. `frac_block_stream`: same regrouping on the continuous stream, with observed token boundaries imposed afterwards.
4. `frac_homophonic`: block-regrouped coordinate fractions with 1–2 homophones per coordinate and bounded row/column surface-symbol overlap.

Grid dimensions are derived from the plaintext alphabet as in v0.5. Labels and parameters are randomized deterministically per replicate.

## Negative/control arms

1. `slot_control`: deliberately PGCS-like prefix/core/suffix restricted generator, overlapping inventories and recurrence, with exact 2x token lengths.
2. `expanded_mono`: two surface emissions per plaintext character from one shared homophonic channel, but no separate coordinate roles.
3. `expanded_transposition`: the expanded monographic control with bounded block transposition.
4. `markov_control`: recurrent first-order surface generator with exact 2x token lengths.

The expanded-monographic controls are intentionally difficult because verbose substitution can create superficial periodicity without coordinate fractionation.

## Corpora and leakage control

Use the exact six pinned Universal Dependencies corpora from `recoverability_frontier_v0_5/corpus_manifest_v050.json`: English, German, Finnish, Turkish, Hebrew and Arabic. All downloaded bytes must match the pinned SHA-256 values.

Development uses only the UD `dev` split. The `test` split remains sealed until the detector specification is frozen after development.

## Detector

For candidate block widths b=1..8, compute normalized mutual information between surface symbol identity and the binary component-role phase induced by b. Evaluate both:

- token-reset phase; and
- continuous-stream phase.

The sample statistic is the maximum normalized MI across these 16 candidates. It is symbol-renaming invariant.

## Matched null

For each sample, generate 99 within-sample permutation nulls. Preserve exactly:

- token count and each token length;
- global surface-symbol counts;
- symbol counts within L0, L1, R0, R1 positional strata;
- symbol counts within three coarse interior thirds.

This preserves strong edge restrictions and a substantial prefix/core/suffix grammar while destroying fine periodic component phase. The same max-over-16 statistic is recomputed for every null, so block/mode search is included in the null.

Report observed statistic, matched-null mean, matched-null SD, residual, z score and empirical permutation p.

## Development gate

Across all six languages and replicates:

- >=90% of positive-arm samples must have z >= 3;
- <=10% of control samples may have z >= 3;
- mean positive/control z separation must be >=2 control SDs.

If any condition fails: `STOP_NON_IDENTIFIABLE`. Do not run Voynich.

If all conditions pass on development: freeze the detector and run the same gate unchanged on the previously untouched UD `test` split. Voynich remains sealed until the locked synthetic test also passes.

## Interpretation ceiling

Passing both gates establishes only that this bounded fractionation-composition family is structurally detectable against these controls. A later Voynich positive would establish compatibility with the detected phase signature, not historical identification, decryption, or plaintext recovery.
