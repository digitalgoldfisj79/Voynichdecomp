# v0.6 Family S1 — segmentation-oracle result

Date: 2026-07-16

Verdict: **PASS. SEGMENTED POLYGRAPHIC ORACLE S2 IS AUTHORISED.**

No test data or Voynich text was scored.

## Job

Hugging Face job: `Digitalgoldfish79/6a588d77b1669a49bf077023`

Scientific SHA-256: `29aa04c41e3c01ac56712e98e6376f609d27405091548ca866200910d3c14cf1`

## Configuration

- English development split;
- 16 independent plaintexts of length 384;
- 63-unit plaintext inventory: characters plus 16 digraphs and 8 trigraphs;
- fresh non-prefix-free one-to-three-symbol codebook per trial;
- no emitted group separators or word boundaries;
- true codebook supplied to an exact train-only character-LM segmentation lattice.

## Results

- mean boundary F1: **99.6719%**;
- median boundary F1: **99.8432%**;
- minimum boundary F1: **97.9969%**;
- mean plaintext recovery: **99.1889%**;
- median plaintext recovery: **99.4792%**;
- minimum plaintext recovery: **95.5959%**;
- all 16 trials exceeded 90% plaintext recovery;
- exact plaintext recovery: 7/16.

## Decision

Ambiguous variable-length group boundaries are strongly identifiable when the codebook is known. Family S therefore proceeds to S2, which supplies the true group boundaries but hides the fresh mapping from code groups to plaintext polygraphic units.

The test split remains sealed. The fully joint S3 stage remains forbidden unless S2 also passes.