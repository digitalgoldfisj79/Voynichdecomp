# Blind stroke-level palaeography v1

Branch: `experiment/blind-stroke-palaeography-v1-20260717-full`

This workspace implements a blinded, externally calibrated image-based study of scribal structure in the Voynich Manuscript. It is scientifically and operationally separate from the cryptanalysis programme.

## Hard exclusions

- No global whole-word or connected-component K-means reruns.
- No Davis hand map in Phase-I code, filenames, plots, prompts, or data frames.
- No crop-random validation.
- No interpretation of stable clusters as scribes without external calibration and nuisance survival.
- No f115r boundary information during model selection.

## Execution order

1. `code/preflight.py`
2. freeze protocol, folds, feature/model registries, and selection code
3. external known-writer calibration
4. blinded Voynich Phase I
5. sealed Davis adjudication
6. reserved f115r change-point test

All failed jobs and amendments are retained in `RUNNING_LEDGER.md`.