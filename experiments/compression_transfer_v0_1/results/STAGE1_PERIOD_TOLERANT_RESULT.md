# Stage 1 period-tolerant result

**Programme:** Compression-Transfer Distance Programme v0.1  
**Date:** 2026-07-30  
**Formal decision:** `STAGE1_FAIL`  
**Voynich:** `SEALED`

## Acquisition qualification

The frozen Stage 1 period-tolerant panel was acquired and independently frozen before scoring:

- 96 documents;
- 12 documents in each of Arabic, English, Finnish, German, Greek, Hebrew, Latin and Turkish;
- train/dev/test partitions present in every class;
- no exact or thresholded near-duplicate failures;
- acquisition qualification: `PASS`.

Acquisition freeze payload:

```text
158749612acf8d7b6ac48c1dd465f54e6b6cc213a71a3dcb17f5e1b0aa1699c3
```

Manifest:

```text
5304b526ffda19272ef310fae0433730a3d60663706f808f81db05c2b7733f98
```

## Intact-text diagnostic

Primary representation: `codepoint_u32_ws`  
Probe length: 4,096 units  
Held-out probes: 113

| Compressor | Top-1 | Macro accuracy | Median own rank | Worst recall | Median NCD order gap |
|---|---:|---:|---:|---:|---:|
| zlib9 | 0.9912 | 0.9922 | 1 | 0.9375 | 0.00093 |
| bz2_9 | 0.6726 | 0.6901 | 1 | 0.0000 | 0.00866 |

The cached arithmetic evaluation was checked against the frozen directional-cost implementation on 64 observations per compressor; all 128 checks agreed exactly.

The two-compressor unanimous consensus had 0.6726 coverage. Accuracy among accepted probes was 1.0, but worst-language recall was 0.0. This is diagnostic only: the final mandatory-compressor consensus was not completed because a later locked gate failed decisively.

## Decisive null result

The registered deterministic non-space character shuffle was generated with seed 1731. It destroys character order while preserving each document's character inventory and whitespace positions.

Registered requirement:

```text
shuffled-control macro language accuracy <= 0.50
```

Observed zlib9 result:

```text
macro accuracy       1.0000
micro accuracy       1.0000
worst-language recall 1.0000
median own rank      1
```

Every one of the eight language classes retained recall 1.0 after character order was destroyed.

Null-result payload:

```text
bc4fcc4c7612001b4171a3045d0bd3ae3eaa0201035ce78bc1817104fe968586
```

## Formal interpretation

The Stage 1 null gate fails maximally. The high intact zlib score is therefore compatible with alphabet, codepoint inventory and unigram-profile recognition; it is not evidence that the compressor recovered order-sensitive language structure.

Under the frozen protocol, failure of any Stage 1 gate closes source-language use of compression distance under this programme version. Accordingly:

- Stage 1 source-language calibration: `FAIL`;
- source-family claims about Voynich under v0.1: `CLOSED`;
- Voynich execution: not permitted;
- Stage 2 surface-class calibration: still permitted, because it is logically separate from source transfer.

The longer LZMA, sensitivity and optional-compressor runs were cancelled after the decisive gate failure. They could not reverse the registered decision.

## Additional limitation discovered

The acquired panel contains mostly one document per author. It therefore does not cleanly operationalize a within-author baseline against which to measure the registered leave-author-out accuracy drop. This is an additional v0.1 design defect, but it is not needed to reach the formal failure: the shuffled-control result already closes Stage 1.

## Jobs

- acquisition qualification: `6a6b23ffb36a6516e96a2532`;
- intact cached evaluation: `6a6b2b01b36a6516e96a25f7`;
- decisive shuffle control: `6a6b2d7923ed89c748ec7417`.

Stopped after the failed gate:

- `6a6b252eb36a6516e96a2538`;
- `6a6b2688b36a6516e96a2558`;
- `6a6b27b7b36a6516e96a2569`;
- `6a6b2d1423ed89c748ec740e`.
