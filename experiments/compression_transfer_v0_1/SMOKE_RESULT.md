# Compression-transfer v0.1 smoke result

**Date:** 2026-07-30  
**Status:** engineering qualification pass; no scientific calibration claim.

## Fixture

Four deterministic synthetic Markov sources, with two train, two development and two test documents per source. The fixture intentionally gives each source a distinct alphabet and transition system.

## Result

On the registered primary smoke representation `codepoint_u32_ws`:

- zlib: 8/8 test probes classified to their own source;
- bzip2: 8/8 test probes classified to their own source;
- median own-source rank: 1 for both compressors.

The label-invariant `token_recurrence_u32` control classified 2/8 probes for each compressor. This is expected because the fixture's strongest distinction is alphabet identity; it prevents the smoke result being misdescribed as structural validation.

Independent row-level validation passed 194/194 checks. The deterministic scientific payload hash is:

`12f4b35579f1ea58a81413cb647530cb0208ff22871054ac4aa41cdfd6833358`

Consensus across the four compressor/representation cells accepted 7/8 probes and classified all accepted probes correctly. This consensus result is descriptive only; the Stage 0 gate is defined on the primary representation.

## Boundary

This confirms code execution, deterministic serialization, metric arithmetic, row emission and tree production. It does not establish language recognition, cipher recognition, historical transfer or Voynich applicability.
