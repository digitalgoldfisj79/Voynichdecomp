# External recoverability calibration — v0.5

**Execution:** Hugging Face Job `6a6458dcdb23d7a7ec1cbab8`  
**Result SHA-256:** `83299da49877ce6183b2d895dc3533f82083da0c815a33121c9a907004ca9ad1`  
**Protocol status:** frozen before formal execution

## Decision summary

| Gate | Result |
|---|---|
| A — representation-neutral historical recognition | **PASS** |
| B1 — Ammerbach paired-channel alignment recovery | **FAIL** |
| B2 — neume-boundary recovery in both families | **FAIL** |
| Downstream open-set adjudication admissible | **NO** |

## A. Representation leakage audit

Historical notation remained highly recognisable after literal identities and explicit transcription conventions were removed. Across frequency-rank, within-event equality, run-pattern, event-rank and length maps, grouped ensemble ROC AUC ranged from **0.9896 to 0.9997**. All five no-length variants exceeded 0.98.

This passes Gate A. The v0.4 classifier was not relying solely on literal symbol inventories or event lengths.

The result is still a broad historical-notation-versus-control distinction, not proof that the exact operational variables are recoverable.

## B1. Ammerbach paired-channel alignment

The task was to locate the true zero offset between the duration/special and pitch/rest channels among offsets −6…+6 under cross-book transfer.

| Train → test | Surface mode | Zero-offset top-1 | MRR |
|---|---|---:|---:|
| bookA → bookB | literal | 0.4892 | 0.6398 |
| bookA → bookB | frequency-rank | 0.4475 | 0.5949 |
| bookB → bookA | literal | 0.5492 | 0.6739 |
| bookB → bookA | frequency-rank | 0.5325 | 0.6537 |

The frozen threshold was 0.70 in every direction, including identity-neutral transfer. Gate B1 fails.

The channels are associated, but the implemented local association score does not recover their exact operational alignment reliably enough across books.

## B2. Neume event-boundary recovery

| Family | Boundary F1 | Renewal baseline | Gain | Exact-event recovery |
|---|---:|---:|---:|---:|
| Aquitanian | 0.6685 | 0.6583 | 0.0102 | 0.4121 |
| Square | 0.7546 | 0.5966 | 0.1580 | 0.5772 |

Square notation passes the intended threshold. Aquitanian notation does not reach F1 0.70 and barely exceeds the length-distribution baseline. Because the gate required both historical families, Gate B2 fails.

## Formal interpretation

The external programme demonstrates robust identity-neutral **recognisability**, but not sufficiently reliable cross-manuscript **recoverability** of known operational structure.

Accordingly, the frozen protocol does not permit a formal recoverability-signature comparison with Voynich. This is a calibration failure at the stronger evidential gate, not a rejection of every possible operational notation model.
