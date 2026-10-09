# LANG2 S0 adjudication (append-only audit, 2026-10-09)

## RETRACTED / CORRECTED FINDINGS (TOP)
1. RETRACTED preliminary claim: the original GER2 physical-fold extraction could not be reproduced. This was an audit-filter discrepancy, not a genuine conflict: the first local audit used substring `P` and an incorrect pair inclusion set. Once the original `u == '+P0'` and the pinned bifolium map were applied, exact reproduction PASSED. Retain the failed preflight in the log as a provenance record; do not cite the preflight token total as the GER2 population.
2. RETRACTED C-LANG cheap language-source adjudications: known positive control failed. Latin and Padua source language are unresolved, not ruled out.
3. RETRACTED treatment of the old PGCS S6 Latin-leading language ranking as independent evidence: language score is contingent on the unqualified PGCS representation; no fair direct comparison follows.
4. No German inference from GER2: ALEM failed the preregistered validation threshold despite a sealed result exceeding 2 null SD.

## S0 results — actual execution
- Frozen Voynich file verified SHA: `26e7490e099b1074ed2ce19356d0ea493aa1791826004e1c551d3f4f9bf8574f`.
- Reconstructed all 34,087 original pinned fold-loader events.
- Exact canonical rows SHA passed: `74f7310ea35922dc5ed71012f0825ef9480d051a1fa516c79fc5ca53f88a925f`.
- Exact physical bifolium map SHA passed: `e774001ca046d88f24f56bf70f29213007d9cac3500d50f414e1782688d74888`.
- Frozen GER2 strict +P0, line-entry exclusion and legacy folio eligibility reproduced across five folds: `f0=2931, f1=4651, f2=4399, f3=5470, f4=4312`; total `21763` tokens.
- Type panel reproduced: discovery=735; fold4 unseen validation=735; strict-final sealed=735.
- Local verifier and atomic pickle checkpoint: `lang2_fold_reproduction.py`, `lang2_fold_checkpoint.pkl`; result: `lang2_fold_reproduction.json`.
- Exact historical source controls: 51,999 alphabetic Circa Instans token events from archived pickle (52,004 unfiltered stored entries); Padua PAGE XML 51,459 locally tokenized alphabetic events (263 pages); Padua is mixed Latin+Italian/Venetian, NOT pure Tuscan. Both original-file SHA-256 matches PASSED.
- No independent multi-witness panel yet certified for Latin/Tuscan/French; S1 positive controls, S2 and S3 Voynich tests remain LOCKED.

## Structural necessary-condition check (calibration-only)
A fixed bijective one-character substitution preserves raw Levenshtein pairwise distances exactly. The deterministic test passed on genuine Latin, Padua and VMS vocabularies (650 types each): all ED1/ED<=2 pair counts identical before and after the permutation, zero collisions. A hypothesis requiring that such a mapping itself *creates* Voynich's excess low-edit-distance graph is mathematically falsified. A non-isometric shared transform remains untested and viable; this test does not favor a source language or prove that any renderer is historical.

## Data acquisition priorities, without overclaiming
- Extract independent date-proximate medieval Latin medical/scientific manuscripts; Circa Instans alone is one textual tradition. Existing documented A1B, Egerton and Palladius Latin controls are relevant but not automatically independent 15th-century manuscripts of a comparable genre.
- Obtain at least three independently dated Tuscan/central-Italian **medical or technical** vernacular texts. Padua is a mixed Latin/Venetian HTR source and cannot be renamed Tuscan.
- Recover comparable French/French-Italian medical text. CHrOMed includes identified medieval texts; CoMMA PEN-alignment data may serve as HTR-to-normalisation positive controls but is not evidence of a historical Voynich renderer.
- Hold German corpora fixed and validate that corpus selection and model power are matched before any rank ordering.

## Next scientific licence
S0 corpus/witness manifest and prospectively blind **multilingual S1 positive-control calibration** only. Do not spend GPU on S2 candidate language arms until S1 detection and false-positive gates are passed in all primary languages. S3 requires its own independent paired-source calibration and entropy-matched many-to-one nulls. The active R4 sequence-recovery controller remains unmodified.