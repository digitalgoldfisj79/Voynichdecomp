# JLCD v0.1 — Joachim Length-Conditioned Cipher Test

Date: 2026-08-29
Status: preregistered before execution on the branch `experiment/joachim-length-v01-20260829`.

## Claim under test

The supplied hypothesis states that Voynich token interpretation depends on total token length; specifically that repeated `e` / `i` material can act as counters which move an otherwise similar core into another length-dependent lookup class, and that a short/long regime boundary exists around 3–4 glyphs.

The programme does not test proposed plaintext assignments such as `ked = nd`, and it does not assume the claimed vowel-bridge mechanism is correct.

## Primary falsification tests

### T1 — e/i-stripped-core conditional context test

Operational core: remove all EVA units `e` and `i` from a token, retaining the ordered remaining units. Within occurrences sharing that core, test whether total token length predicts the immediate *external* context after conditioning on:

- core identity;
- Currier A/B;
- manuscript section;
- token position class within line (initial / medial / final);
- coarse line-length bin.

External context is frozen as `(previous-token terminal unit, previous-token length bin, next-token initial unit, next-token length bin)` with explicit boundary symbols.

Statistic: conditional mutual information `I(total_length ; external_context | matched_stratum)` in bits/occurrence.

Null: permute exact total lengths only within the complete matched strata above. This preserves core, Currier, section, line position, line-length composition and external-context frequencies.

Primary requirement: |z| >= 2 in the full corpus and same-direction |z| >= 2 independently in Currier A and Currier B, under both greedy EVA-unit length and raw transcription-character length.

### T2 — claimed 3/4-glyph regime boundary

Using the same e/i-stripped cores and matched strata, scan fixed thresholds k = 2..8 for the binary variable `L <= k` versus `L > k`.

The 3/4 claim is supported only if:

1. k=3 or k=4 is among the two strongest standardized thresholds in the full corpus;
2. its full-corpus |z| >= 2;
3. the same k has same-direction |z| >= 2 in both Currier A and Currier B;
4. it survives raw-character representation;
5. a joint max-|z| permutation correction across k=2..8 has p <= 0.05.

A general monotone length effect without this change-point pattern does not support the claimed short/long lookup split.

### T3 — e/i specificity

Repeat T1 after stripping each pair drawn from the eight most frequent EVA units. For every strip pair report the bias-corrected CMI effect (`observed - null mean`), null SD, z and eligible occurrence count.

The e/i counter claim receives specificity support only if the `{e,i}` pair is in the top quartile by bias-corrected effect and is not explained solely by having a materially larger eligible sample. This is a secondary test; no cipher claim is promoted on specificity alone.

### T4 — `ked` / `keed` supplied example

Report occurrence counts, Currier/section distribution, line-position distribution and immediate-context summaries for exact EVA strings `ked` and `keed`. This is descriptive only. No single lexical example can pass a programme gate.

## Decision states

- JLCD-0: e/i-stripped-core length does not predict external context after matched controls.
- JLCD-1: an independent within-core length/context effect exists, but e/i specificity and/or 3/4 change-point tests fail. This is structural evidence, not cipher evidence.
- JLCD-2: e/i-specific effect plus the preregistered 3/4 change-point survives full, A/B, representation and familywise controls. This supports the proposed *mechanistic signature* but does not validate plaintext mappings.

## Hard identifiability boundary

Even a positive JLCD result cannot distinguish a cipher from a non-cipher generator that explicitly conditions variant selection on token length and line context. A cipher interpretation therefore requires an independently frozen generator/decoder that predicts the held-out JLCD statistics against matched non-cipher generators.

## Audit order

circularity -> leakage -> confounds -> matched nulls -> control fairness -> measurement degeneracy -> representation dependence -> decision-rule fragility -> audit completeness -> interpretation.

All phases write pickle checkpoints. `RESULTS_JLCD_v01.md` begins with `RETRACTED FINDINGS` and retains any later corrections at the top.
