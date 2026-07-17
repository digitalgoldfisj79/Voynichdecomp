# STOP RECORD — Blind stroke palaeography v1

**Date:** 2026-07-17  
**Terminal state:** `STOPPED_BEFORE_VOYNICH`  
**Voynich Phase I opened:** no  
**Davis labels loaded:** no  
**f115r target boundary loaded:** no

## Reason

A post-freeze implementation audit found that the v1 external-control runner implemented only five of the eight frozen calibration gates. It did not yet perform the full perturbation-retention test, held-out visual-family test, continuous-drift false-discrete test, or no-signal abstention test. It also used a proxy raw representation rather than the exact out-of-fold selected representation for permutation and known-K calibration.

The v1 blind model-selection runner likewise did not yet implement the frozen held-out exact-word-type and held-out visual-family gates in the same fold-local pipeline.

Proceeding would therefore create a formally invalid path to opening Voynich Phase I. The one permitted bounded implementation repair had already been consumed by the Historical-WI physical-page derivative grouping correction. The v1 programme is stopped rather than silently amended a second time.

## Scientific consequence

No positive, negative or abstention result about Voynich scribal structure exists from v1. The completed work remains valid as infrastructure and provenance:

- repository and model access tests;
- external corpus acquisition and parser smoke tests;
- physical bifolium and quire registry;
- frozen blinding rules;
- failed-job ledger;
- source and environment checks.

The programme continues as v2 under a fresh pre-registration. V2 preserves the scientific question and thresholds, completes all external and Phase-I gates before freezing, and starts with no implementation-repair allowance consumed.