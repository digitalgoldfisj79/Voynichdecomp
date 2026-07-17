# AMENDMENT 006 — v1.5.1 preflight panel correction

**Date:** 2026-07-17  
**Status:** prospectively frozen before rerun metrics  
**Voynich Phase I opened:** no  
**Davis labels loaded:** no  
**f115r boundary loaded:** no

## Trigger

The first v1.5 preflight job `6a5a1200bee6ee1cf4ecd370` successfully completed source assembly, immutable-source checks, data parsing, deterministic writer splitting, patch extraction, morphology-aware HOG-MAE pretraining, NetRVLAD metric training, validation checkpoint selection, checkpoint reload, page encoding, PCA and nuisance residualization. It then stopped before emitting terminal results because the reduced preflight split contained eight test writers while the frozen synthetic calibration requests K=2–10. Sampling ten distinct writers from an eight-writer preflight test set is impossible.

This is a preflight-panel sizing defect. It does not alter the v1.5 formal model, training schedule, representation selector, K range, metric, threshold or full-corpus split.

## Single permitted correction

For preflight mode only, change:

`MAX_WRITERS = 48 if PREFLIGHT else None`

to:

`MAX_WRITERS = 80 if PREFLIGHT else None`

Under the frozen 70%/15%/remainder writer split this yields 56 training writers, 12 validation writers and 12 test writers, sufficient for K=2–10. All other preflight reductions remain unchanged. In full mode the value remains `None`; therefore all eligible Historical-WI writers are still admitted exactly as preregistered in v1.5.

## Immutable derivation

- parent v1.5 source bytes: `23391`
- parent v1.5 SHA-256: `e064648d07e28eac56a2f46012012d5e472aacc4e44dfa81c7018235b220b934`
- derived v1.5.1 source bytes: `23391`
- derived v1.5.1 SHA-256: `fd8f93893a488b59d41eba4395de82e5690ebb491bc8bbe6c1de581a2884cdd8`
- permitted substitutions: exactly one, as stated above

The rerun must refuse execution unless both parent and derived byte counts and SHA-256 values match. No statistic from the failed preflight authorizes scientific inference.
