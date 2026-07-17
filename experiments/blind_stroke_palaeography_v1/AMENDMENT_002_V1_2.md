# AMENDMENT 002 — Stop v1.1 and preregister external calibration v1.2

**Date:** 2026-07-17  
**Status:** new preregistered version; not a second repair to v1.1  
**Voynich Phase I opened:** no  
**Davis labels loaded:** no  
**f115r target boundary loaded:** no

## Trigger

Repaired Historical-WI smoke job `6a59f2e8bee6ee1cf4ecd085` completed archive acquisition and the canonicalized corpus audit, then stopped before preprocessing, feature extraction or any performance metric.

The audit established a corpus fact that had been obscured by the duplicate colour/binarized derivatives: Historical-WI contains exactly **three physical pages per writer**, not six and not five. The v1.1 launcher requested five physical pages per writer, leaving zero eligible writers and producing the terminal error:

`RuntimeError: too few writers after parsing/selection`

Two queued replicas were cancelled because they had the identical immutable source and invocation and therefore could not produce a different outcome.

## Why v1.1 is closed

`AMENDMENT_001.md` had already consumed the single bounded pre-Voynich repair and explicitly required at least five physical pages per writer. That requirement is impossible for this corpus. Quietly changing it would violate the freeze. Version 1.1 is therefore closed without a calibration result.

No writer-retrieval score, nuisance comparison, permutation statistic, selected representation, K-recovery statistic, Voynich partition or Davis comparison was observed under v1.1.

## v1.2 preregistered change

Historical-WI uses all three available physical pages per selected writer and **three deterministic writer-balanced page-group folds**. For each writer, one physical page is held out in each fold. Colour and binary derivatives of the same physical page retain the canonical shared page identifier, and only the colour derivative is used as the independent retrieval item when both exist.

The fold count is computed as the largest valid grouped count up to five:

`n_folds = min(5, minimum physical pages represented for any selected writer)`

A minimum of two is required. For Historical-WI this deterministically evaluates to three. This is equivalent to leave-one-physical-page-out for the corpus and is consistent with the original protocol's stated Historical-WI split.

All other elements remain unchanged:

- external corpora and archive checksums;
- DINOv3 and historical-TrOCR models;
- image units and preprocessing;
- feature families and ensembles;
- fold-local family residualization and nuisance removal;
- same-page gallery exclusion;
- writer selection seed;
- permutation and synthetic-K procedures;
- confirmation thresholds;
- prohibition on opening Voynich Phase I before the complete external gate passes.

## Immutable source derivation

Version 1.2 is derived from the frozen v1.1 calibration source by an audited launcher that:

1. reconstructs the five v1.1 source parts from immutable commit `8f5cd23cd1f8415c21c3a9367c3e280eafe58bbd`;
2. verifies parent byte length `37401` and SHA-256 `f93fc90c0527266d71d876962050923b9f7e4020c77dc8c7fad83019b80ac883`;
3. applies only the prospectively declared fold-count, default-page-count and schema-version substitutions;
4. verifies derived byte length `37929` and SHA-256 `edb7fe7b3405e2c41678ab035c267f563657b8e7a379425e9b26a89675b73607` before execution.

The launcher is `code/external_calibration_v1_2_launcher.py`.

## Scientific consequence

The change reduces the number of independently held-out pages per writer from an impossible five to the complete available set of three. It does not respond to a calibration metric and cannot tune a Voynich result, because neither existed when v1.2 was frozen.
