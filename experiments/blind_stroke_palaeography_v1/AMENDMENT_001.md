# AMENDMENT 001 — External-control page grouping and smoke-fold repair

**Date:** 2026-07-17  
**Status:** the single bounded pre-Voynich implementation repair permitted by the frozen protocol  
**Voynich Phase I opened:** no  
**Davis labels loaded:** no  
**f115r target boundary loaded:** no

## Trigger

Non-confirmatory Historical-WI smoke job `6a59e6ca85d9643ce16d752d` completed archive acquisition, parsing, preprocessing, DINOv3 extraction and historical-TrOCR extraction, then failed before reporting any calibration metric.

The failure exposed two implementation defects:

1. The colour and binarized derivatives of the same Historical-WI physical page were parsed as different page identifiers. For example, `100-3-IMG_MAX_1007803.jpg` and `100-IMG_MAX_1007803.png` refer to the same physical page but would have been eligible to cross page-group folds. This violates the frozen page-leakage prohibition.
2. The smoke invocation requested only two pages per writer while the frozen evaluator constructs five page-group folds. Some folds therefore had no test sample and the code failed at an empty array stack. The formal configuration already requested sufficient pages; the smoke configuration did not.

No Voynich representation, clustering result, selected K, Davis comparison or external-control performance statistic was observed before this defect was diagnosed.

## Exact bounded repair

`external_calibration.py` v3 makes only these changes:

- Historical-WI derivative filenames are canonicalized to a common physical-page identifier of the form `writer::IMG_MAX::physical_id`.
- Where both colour and binarized derivatives exist, panel construction preferentially retains the colour derivative and does not treat the binary derivative as an independent page or retrieval item.
- The implementation is re-smoked with at least five physical pages per writer. The grouped-fold evaluator, representations, metrics, seeds, model families, thresholds and formal sample sizes are otherwise unchanged.

The repaired source is:

- byte length: `37401`
- SHA-256: `f93fc90c0527266d71d876962050923b9f7e4020c77dc8c7fad83019b80ac883`

A parser unit check confirms that both derivative example filenames map to the same writer and physical-page identifier.

## Scientific consequence

The repair removes a route to inflated writer recovery and makes the external calibration more conservative. It does not respond to a favourable or unfavourable metric and cannot tune the Voynich result because the blind Voynich analysis remains unopened.

No further implementation repair is permitted under the frozen one-repair rule unless the programme is formally stopped and re-preregistered as a new version.