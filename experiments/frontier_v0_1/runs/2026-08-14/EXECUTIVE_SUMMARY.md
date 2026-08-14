# Voynich Frontier Programme v0.1 — formal execution 2026-08-14

Execution commit: `499a68f7086a862e0b79886e44c0492ed661c192`  
GitHub Actions run: `31833392027`  
No preregistered scientific threshold was relaxed.

## Gate 0 — PASS

- 37,465 canonical records;
- exactly nine registered section labels;
- 23 transliterators in the slim container;
- 50 physical bifolia in exactly five inherited folds;
- no target opened by Gate 0.

The fold manifest was recovered mechanically from the archived frozen `events_structural.pkl` (SHA-256 `dec8708b380c7b85e40967240f21468c2a636ce5ac3e3761b9a4793cf3258eec`) and cross-checked against `events_stage1.pkl` before execution.

## U1 uncertainty lattice — COVERAGE PASS

- reference lines: 5,162;
- admitted lines: 5,154;
- coverage: 0.9984502131 (99.845%);
- family coverage: ZL 5,162; IT 5,143; VT 5,151; RF 5,157; GC 5,135.

This is an availability/admissibility pass only. The substantive U1 effect test was **NOT OPENED** because v0.1 has no calibrated effect adapter.

Important caveat: the frozen `GCGI` representative is encoded in a different surface alphabet from the EVA-family readings. A generated example represents the same line as `1os ayo2oe` under GC and `chosaroshol` under ZL/IT/VT/RF. A later U1 effect adapter therefore requires a frozen common representation or an explicitly representation-invariant statistic. Exact-string disagreement is not itself interpretable as transcription uncertainty.

## U2 — NOT OPENED

The page-to-folio mapping gate is implemented, but the frozen 28-row mapping panel and independent folio-label panel were not recovered as executable inputs. No D'Imperio anomaly score was computed.

## U3 — ABSTAIN / SEALED

Synthetic calibration and leave-feature-family-out qualification are prerequisites and are not implemented in v0.1. No Currier/hand/section target association was opened.

## U4 — SEALED

The inherited surface-closure result remains controlling: 4/5 absolute adequacy folds but only 3/5 dynamic-over-static folds, hence overall closure FAIL. Manuscript payload testing remains sealed. This is not evidence against encipherment; it is refusal to test payload against an inadequate null.

## U5 / U6 — NOT OPENED

The blind recovery/recognition and VTPS target instruments are not qualified in this build.

## Formal position

**GATE 0 PASSED; U1 COVERAGE PASSED; SUBSTANTIVE TARGET TESTS REMAIN SEALED OR UNQUALIFIED.**

The next scientific build should implement the U1 common-representation/calibrated effect adapter, recover/freeze the U2 mapping inputs, and complete U3 synthetic/LOFO qualification. U4 may open only after a surface realiser satisfies the already frozen closure criterion.
