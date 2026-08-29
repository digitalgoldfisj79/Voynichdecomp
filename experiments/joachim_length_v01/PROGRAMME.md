# JLC v0.1 — Joachim Length-Dependent Cipher Test

Date: 2026-08-29
Status: preregistered before execution on the branch.

## Claims under test

The supplied hypothesis makes three separable empirical claims:

1. **Counter/core claim:** after holding a consonantal/core skeleton fixed, changing the number/placement of EVA `e`/`i` units (and hence total token length) changes external token context.
2. **Counter specificity claim:** one-unit near-neighbour pairs created by insertion/deletion of `e` or `i` have a larger external-context split between short and long forms than frequency- and length-matched one-unit pairs involving other inserted units.
3. **Short/long regime claim:** the contextual split between adjacent token lengths has a discontinuity around a short/long boundary at 3/4 glyphs rather than merely changing smoothly with length.

No plaintext assignments (`ked=nd`, vowel bridges, etc.) are assumed or used.

## Frozen observables

Primary corpus: ZL3b running text. Primary length representation: greedy EVA units used in WLCP v0.1. Sensitivity: raw transcription characters.

External context excludes the token itself and comprises previous-token final unit + length bin and next-token initial unit + length bin. Line edges use explicit BOS/EOS symbols. Position is controlled by six within-line bins. Currier and manuscript section are also conditioning variables.

## T1 — e/i skeleton conditional-context test

Define an operational core skeleton by deleting only EVA units exactly equal to `e` or `i`. Retain skeletons observed at >=2 total lengths and with >=12 usable occurrences. Measure conditional mutual information

`I(total_length ; external_context | skeleton, Currier, section, position_bin)`.

Matched null: permute total-length labels within each conditioning stratum. 1,000 deterministic permutations. Report observed, null mean, effect, null SD, z and empirical p. Replicate independently in Currier A and B and under raw-character representation.

This establishes or rejects the descriptive core/length-context coupling. It does not establish ciphering.

## T2 — e/i specificity against other one-unit edits

Enumerate one-unit insertion/deletion token pairs. Select lexically disjoint pairs using frequency only. Tag a pair `EI` iff the inserted/deleted EVA unit is `e` or `i`; otherwise `OTHER`.

Match each EI pair 1:1 to an OTHER pair of the same short-token length with nearest log total frequency. No token type may occur in more than one matched block.

For each group aggregate the conditional mutual information between pair member (short/long) and external context conditioned on pair ID, Currier, section and position bin. Primary specificity statistic is `CMI(EI)-CMI(OTHER)`.

Matched null: independently swap the EI/OTHER assignment inside each matched lexical block. 2,000 deterministic permutations. This preserves pair identities, lengths, frequencies and contextual data while testing whether `e/i` status itself is special.

Replicate in A/B and raw-character representations.

## T3 — threshold/discontinuity test

For every one-unit disjoint pair, group by the short member's length L. On a deterministic 50% training-folio split calculate bias-corrected contextual CMI by L. Choose between candidate boundaries k=3 and k=4 using only the training split, where the score is the local curvature

`D(k) - 0.5*(D(k-1)+D(k+1))`.

Freeze the selected k. On untouched test folios calculate the same curvature. Matched null: permute short/long member labels within pair×Currier×section×position strata independently inside each length group, preserving every marginal count; 2,000 permutations. A positive >2-null-SD test effect is required.

The second candidate boundary is reported as a sensitivity result but is not promoted after selection.

## Decision rule

- A claim with |z| < 2 is reported: **the metric does not resolve this**.
- A `length-dependent cipher` interpretation requires T1 + T2 + T3 to pass directionally in full corpus, Currier A, Currier B and the alternative representation.
- If T1 passes but T2 fails, the result is ordinary length/morphology-context coupling, not evidence that `e/i` are counters.
- If T1/T2 pass but T3 fails, the proposed discrete 3/4-vs-longer lookup regime is unsupported.
- Even if all three pass, this programme establishes a structural signature only. Cipher interpretation would still require a frozen generator that predicts the signature out of sample and beats matched non-cipher generators.

Audit order: circularity → leakage → confounds → matched nulls → control fairness → measurement degeneracy → representation dependence → decision-rule fragility → audit completeness → interpretation.
