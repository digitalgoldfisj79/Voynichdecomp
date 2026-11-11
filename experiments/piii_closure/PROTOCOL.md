# PIII-CLOSURE frozen protocol

Seed: 20260714.

## Question

Can the unchanged 24-letter by 131-row table of Trithemius, *Polygraphia* III, under the published Hermes policies or conservative historical row-use policies, reproduce held-out Voynich surface organisation rather than merely selected entropy values?

## Frozen arms

- `iid10`, `iid24`, `iid131`: exact Hermes random-row policies.
- `cycle131`: rows used in table order, reset at every physical line.
- `line_fixed`: one random table row reused throughout a line.
- `sticky24`: bounded sensitivity, p(stay)=0.75 within the first 24 rows.
- `permute10`: inventory-matched control; plaintext-letter columns independently permuted within every row.

The PIII table is never altered using Voynich data. No Voynich-to-PIII dictionary is fitted. Davis hand labels are sealed and unused.

## Inputs

Primary VMS transcription: ZLZI. Sensitivities: ZLZB and TTIA. Text-only pages and non-`[a-z]` tokens are excluded. External plaintext streams are Melanchthon, *Secreta Secretorum* (Latin), *Picatrix* (Latin), and *Rettorica* (Old Italian), normalised exactly as Hermes: ASCII, lower case, `j→i`, `v→u`, PIII alphabet only.

## Favourable layout assumption

Each generated corpus uses the real Voynich folio and line token-count template, and row state resets at each line. PIII is therefore not penalised for plaintext wrapping, but cannot claim to explain line lengths. One plaintext character produces one PIII codeword.

## Independence

Chunks never cross folios. Five grouped folds use whole approximate quire blocks; no folio is split. Exact physical bifolium mapping is not present in the source bundle and is a declared limitation. Section results are diagnostics only.

## Tests

1. Grouped classifier two-sample test (C2ST) on 34 frozen alphabet-label-invariant features over approximately 120-token chunks.
2. Whole-corpus forward discrepancy on 19 critical metrics, standardised by 200 approximate-quire bootstraps of VMS.
3. Calibration: random VMS labels median AUC ≤0.60; within-line word shuffle median AUC ≥0.80; within-token character shuffle median AUC ≥0.80.
4. Policy identifiability among `iid10`, `cycle131`, `line_fixed`, and `permute10`: grouped macro AUC ≥0.80.
5. Alternate-transcription median AUC must not reverse the result.

## Decision

An arm is surface-compatible only if calibration passes, median C2ST AUC ≤0.60, median absolute critical z ≤2, no critical |z| >4, and both transcription sensitivities have median AUC ≤0.65. If all five primary arms fail, verdict **FAIL for the exact PIII table and these policies**. Calibration failure yields **UNRESOLVED**.

## Scope

A FAIL does not reject a newly invented or Voynich-fitted morpho-local nomenclator, scribe-specific changing codebooks, alternative segmentation, or a separate post-encryption surface realiser. Direct exact-token MDL is not admitted because no cross-alphabet token mapping is fitted; fitting one would change the hypothesis and add a large codebook.
