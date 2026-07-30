# Compression-Transfer v0.1 — Stage 2 surface result

**Decision:** `STAGE2_SURFACE_FAIL`  
**Basis:** logical early stop after two primary mandatory-compressor failures  
**Voynich:** sealed; never loaded or scored

## Acquisition qualification

The frozen Stage 2 panel passed acquisition qualification before compression scoring:

- 148 documents across 13 surface families;
- train, development and test partitions in every family;
- fresh independently generated keys for every synthetic cipher document;
- source-document splits inherited across all transformed families;
- generator-disjoint test regimes for Family P, polygraphic/fractionating ciphers, structured generators and matched nulls;
- participant-disjoint human meaningless-writing composites;
- manuscript-disjoint CoReMA procedural documents;
- no exact or thresholded near-duplicate failures.

Acquisition freeze payload:

```text
aa4c61d541f05621297dcf6956e38132a38ab92200634b20c16e185152e17c73
```

## Frozen stopping rule

The primary representation was `codepoint_u32_ws`. A qualifying cross-cell support set required that representation to pass with at least two mandatory compressors. The mandatory compressors were zlib, bzip2 and LZMA.

After zlib and bzip2 both failed the primary cell, only LZMA remained. The maximum possible primary support was therefore one compressor against a requirement of two. No remaining result could create a qualifying support set, so the programme stopped without running scientifically redundant cells.

## Primary-cell results

| Compressor | Top-1 | Macro accuracy | Worst recall | Generator-disjoint accuracy | Matched-null FPR | Median own rank | Median NCD order gap | Cell |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| zlib9 | 0.2951 | 0.3798 | 0.0000 | 0.4677 | 1.0000 | 4 | 0.00114 | FAIL |
| bz2_9 | 0.2186 | 0.2821 | 0.0000 | 0.2581 | 1.0000 | 6 | 0.01294 | FAIL |

Both cells passed 64/64 independent arithmetic cross-checks.

## Class-level pattern

The method separated several grossly distinctive surfaces but did not recover the registered cipher-family partition.

Zlib recall was 1.0 for plaintext, CoReMA procedural text, human meaningless writing and the structured generator, and 0.9375 for the polygraphic/fractionating family. It was 0 for monoalphabetic, homophonic, nomenclator, substitution-plus-transposition, Family P, null-bearing cipher, Polygraphia-style table material and the matched null.

Bzip2 recall was 1.0 for plaintext, CoReMA procedural text and the polygraphic/fractionating family, and 0.6667 for human meaningless writing. It was 0 for every other family.

Most importantly, both compressors assigned every matched-null probe to a non-null family, producing a false-positive rate of 1.0 against a registered maximum of 0.05.

## Formal interpretation

Compression distance under programme v0.1 does not support reliable recognition of the registered cipher, notation, procedural and generator surface families. The method is dominated by broad representational contrasts and collapses many fresh-key cipher families together.

Accordingly:

- Stage 2 surface calibration: `FAIL`;
- Stage 3 Voynich execution: prohibited;
- Voynich source-language claims: already closed by the Stage 1 result;
- Voynich surface-family claims from this compression programme: closed;
- Compression-Transfer Distance Programme v0.1: scientifically complete and negative.

The negative result does not imply that the registered surface families are intrinsically indistinguishable. It establishes that this compressor-transfer method, with the frozen representations and gates, did not distinguish them reliably enough to be used on the sealed target.

## Evidence and execution record

- Stage 2 acquisition job: `6a6b504cb36a6516e96a2b0d`;
- independent primary zlib job: `6a6b51c0b36a6516e96a2b2c`;
- formal parallel matrix job, stopped after logical impossibility: `6a6b52ecb36a6516e96a2b49`;
- closeout reproduction job: `6a6b548223ed89c748ec7a99`.

The attempted Hugging Face private-dataset upload failed with an authorization error after the scientific outputs had been generated. The formal decision, exact acquisition freeze, cell metrics, arithmetic checks and scientific payload hashes are preserved in GitHub. No claim is made that a separate Hugging Face result repository was created.

Scientific payload SHA-256:

```text
32af2b85f96e4bae24185a2427ff9a388086fef5e02cd2967f253c44a9aa981f
```
