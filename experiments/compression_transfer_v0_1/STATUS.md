# Programme status

**Status:** Stage 1 period-tolerant calibration executed; formal decision `STAGE1_FAIL`.  
**Source-language use under v0.1:** closed.  
**Voynich:** sealed; never loaded or scored.  
**Stage 2 surface-class calibration:** permitted as a logically separate programme phase.  
**Stage 3 Voynich execution:** not permitted.

## Implementation

Completed before formal execution:

- protocol and exact escalation/stop rules;
- directional compressor cross-entropy;
- self-normalized excess cost;
- order-retaining NCD;
- fixed-width Unicode and label-invariant recurrence representations;
- document-level manifests and SHA validation;
- deterministic chunking, reference construction and UPGMA output;
- row-level results and independent arithmetic validation;
- consensus evaluator and null-control generator;
- synthetic engineering smoke fixture and CI workflow.

## Stage 1 corpus acquisition

The frozen period-tolerant panel passed acquisition qualification:

- 96 documents;
- 12 documents in each of Arabic, English, Finnish, German, Greek, Hebrew, Latin and Turkish;
- train/dev/test partitions in every class;
- no exact or thresholded near-duplicate failures;
- acquisition freeze payload: `158749612acf8d7b6ac48c1dd465f54e6b6cc213a71a3dcb17f5e1b0aa1699c3`.

## Intact-text diagnostic

Primary representation: `codepoint_u32_ws`; probe length: 4,096 units; 113 held-out probes.

- zlib9 top-1: 0.9912; macro accuracy: 0.9922; worst-language recall: 0.9375;
- bz2_9 top-1: 0.6726; macro accuracy: 0.6901; worst-language recall: 0.0000;
- 128/128 cached-versus-frozen arithmetic cross-checks passed.

These intact-text scores are diagnostic only because the registered shuffled-control gate subsequently failed.

## Decisive registered null result

The deterministic non-space character shuffle, seed 1731, destroys character order while retaining each document's character inventory and whitespace positions.

Registered gate:

```text
shuffled-control macro language accuracy <= 0.50
```

Observed with zlib9:

```text
macro accuracy        1.0000
micro accuracy        1.0000
worst-language recall 1.0000
median own rank       1
```

All eight language recalls were 1.0 after character order was destroyed. The null-result payload is:

```text
bc4fcc4c7612001b4171a3045d0bd3ae3eaa0201035ce78bc1817104fe968586
```

## Formal decision

The Stage 1 null gate fails maximally. Under the frozen stop rule, source-language use of compression distance is closed under programme v0.1. The high intact zlib score is compatible with alphabet, codepoint-inventory and unigram-profile recognition; it is not evidence of order-sensitive language recovery.

The longer LZMA, sensitivity-length and optional-compressor runs were stopped after the decisive failure because they could not reverse the formal decision.

Full machine-readable and narrative reports:

- `results/STAGE1_PERIOD_TOLERANT_RESULT.json`;
- `results/STAGE1_PERIOD_TOLERANT_RESULT.md`.

## Remaining permitted phase

Only Stage 2 surface-class calibration may proceed under v0.1. A Stage 2 pass could support surface compatibility only; it cannot reopen source-language interpretation or authorize a source-family claim about Voynich.
