# Programme status

**Programme:** Compression-Transfer Distance Programme v0.1  
**Status:** scientifically complete; negative.  
**Stage 1 source-language decision:** `STAGE1_FAIL`.  
**Stage 2 surface-class decision:** `STAGE2_SURFACE_FAIL`.  
**Voynich:** sealed; never loaded or scored.  
**Stage 3 Voynich execution:** prohibited.  
**Permitted claims from v0.1:** none about Voynich source language or surface family.

## Implementation

Completed before formal execution:

- preregistered escalation and stopping rules;
- directional compressor cross-entropy and self-normalized excess cost;
- order-retaining normalized compression distance;
- fixed-width Unicode and label-invariant recurrence representations;
- document-level manifests, provenance and SHA validation;
- deterministic chunking and reference construction;
- independent arithmetic checks;
- consensus and null-control machinery;
- reproducible Stage 1 and Stage 2 corpus builders.

## Stage 1 — source-language calibration

The frozen period-tolerant panel contained 96 documents: 12 each in Arabic, English, Finnish, German, Greek, Hebrew, Latin and Turkish. Acquisition, split and duplicate gates passed.

The intact zlib result appeared strong, but the registered non-space character shuffle retained 100% accuracy in every language. The method was therefore recognizing alphabet, codepoint inventory and unigram profile rather than order-sensitive language structure.

Formal result: `STAGE1_FAIL`. Source-language use under v0.1 is closed.

Reports:

- `results/STAGE1_PERIOD_TOLERANT_RESULT.json`;
- `results/STAGE1_PERIOD_TOLERANT_RESULT.md`.

## Stage 2 — surface-class calibration

The frozen Stage 2 panel passed acquisition qualification:

- 148 documents;
- 13 registered families;
- plaintext, monoalphabetic, homophonic, nomenclator, substitution-plus-transposition, Family P, null-bearing, polygraphic/fractionating, CoReMA procedural, human meaningless, structured-generator, Polygraphia-table and matched-null families;
- fresh independent keys per synthetic document;
- source-document split inheritance across transformed families;
- generator-disjoint test regimes;
- participant-disjoint human controls and manuscript-disjoint CoReMA controls;
- no exact or thresholded near-duplicate failures;
- acquisition freeze payload: `aa4c61d541f05621297dcf6956e38132a38ab92200634b20c16e185152e17c73`.

The primary representation was `codepoint_u32_ws`. A qualifying support set required the primary representation to pass under at least two mandatory compressors.

| Compressor | Top-1 | Macro accuracy | Worst recall | Generator-disjoint accuracy | Matched-null FPR | Median own rank | Cell |
|---|---:|---:|---:|---:|---:|---:|---|
| zlib9 | 0.2951 | 0.3798 | 0.0000 | 0.4677 | 1.0000 | 4 | FAIL |
| bz2_9 | 0.2186 | 0.2821 | 0.0000 | 0.2581 | 1.0000 | 6 | FAIL |

Both cells passed 64/64 independent arithmetic checks.

After these two mandatory primary failures, only LZMA remained. The maximum possible primary support was therefore one compressor against a requirement of two. A qualifying cross-cell support set had become logically impossible, so all scientifically redundant remaining cells were stopped.

Formal result: `STAGE2_SURFACE_FAIL`.

The method separated some grossly distinctive surfaces—particularly plaintext, CoReMA procedural text, human meaningless writing, structured generation and polygraphic output—but collapsed the majority of fresh-key cipher families. Both tested compressors misclassified every matched-null probe as a non-null family.

Reports:

- `results/STAGE2_SURFACE_EARLY_STOP_RESULT.json`;
- `results/STAGE2_SURFACE_EARLY_STOP_RESULT.md`.

Stage 2 scientific payload:

```text
32af2b85f96e4bae24185a2427ff9a388086fef5e02cd2967f253c44a9aa981f
```

## Final interpretation

Compression transfer under v0.1 did not validate either source-language recovery or reliable surface-family recognition. No Voynich matrix was produced. The programme therefore closes without a Voynich compatibility claim.

The negative calibration remains methodologically informative: strong-looking compression results can reflect alphabet, inventory, unigram or gross renderer differences while failing the recoverability and matched-null tests required for interpretation.
