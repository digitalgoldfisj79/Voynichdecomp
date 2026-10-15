# v143 visible carry preflight — NO-GO
14 October 2026

## RETRACTIONS AND NARROWED FINDINGS
- The V1 experiment compared different eligible sample populations, so comparisons between its arms are invalid. Preserved as V1 only.
- A visible predecessor-suffix association appears in **all** arms, including unchanged v142b. It is **not specific** to carry.
- v143 carries a preceding **decoded vowel**, not a known function of visible suffixes. The visible-end proxy is a candidate only.
- Prior V142/V143 Voynich surface failures, previous V10 synthetic identification failure and original B-only bridge-state null remain unchanged.
- No new Voynich A or B test has been run, as the synthetic discrimination gate failed.

## Frozen test
All 3 Hartlieb ciphertexts encode exactly the same source, and each aligned token retains identical length and first glyph. The held-out instrument predicts the middle glyph from its first glyph and length, with or without the preceding ciphertext word's last 3 characters. Five block-based folds; identical 9,790 admitted positions across arms; 60 matched within-shell/block permutation nulls, paired across arms. The held-out score is gain of the enriched model in bits/event. Effect = score minus permutation-null mean. The null SD is a permutation SD, not a historical uncertainty estimate.

| Arm | Held-out gain | Effect against null | Null SD | Ratio |
|---|---:|---:|---:|---:|
| unchanged v142b | +0.137331 | +0.263961 | 0.006658 | +39.65 |
| v143 carry | +0.112085 | +0.204900 | 0.004968 | +41.25 |
| v143 statefree | +0.029530 | +0.105155 | 0.004803 | +21.89 |

| Paired contrast | Effect difference | Paired null SD | Ratio |
|---|---:|---:|---:|
| carry − unchanged v142b | -0.059061 | 0.005721 | -10.32 |
| carry − index-statefree | +0.099745 | 0.005365 | +18.59 |

## Verification questions
1. **Same events?** Yes, 9,790 aligned Hartlieb positions; first glyph and length exactly invariant across all arms.
2. **Matched null?** Yes, 60 within-shell, held-out-block shuffles with identical reassignment indices for each arm.
3. **Does carry uniquely explain the signal?** No. Its proxy effect is lower than unchanged v142b. Statefree also produces a positive association.
4. **Does a negative result rule out historical carry?** No. Visible last-three characters were not a required representation of the transmitted plaintext-vowel state.
5. **Is it valid to run Currier A/B now?** No. A manuscript test of this proxy would be an underqualified repeat of our CONNECT problem.

## Audit order
1. Circularity: synthetic-generated corpus; scoring blocked held-out, not historical validation.
2. Leakage: no source, decoded vowels or encoder ledger used in prediction.
3. Confounds: interior glyph diversity and visible suffix marginals differ across arms, despite invariant lengths/heads and same source.
4. Matched nulls: preserves shells, within-block composition and word lengths; permutation SD only.
5. Control fairness: V1 retracted; V2 aligns token positions and feature conditioning.
6. Measurement degeneracy: null SD nonzero; strong positive associations occur in all controls, hence non-specific.
7. Representation dependence: raw glyphs and fixed three-character ending; no re-atomisation on the result.
8. Decision-rule fragility: no tuning after the carry-vs-baseline failure.
9. Audit completeness: data files, code, contracts, numerical results, atomic pickle checkpoints and SHA-256 manifest in ZIP.

## Decision
**STOP before testing Voynich.** We have a measurable dependency but it does not specifically diagnose carry. Our general identifiability limit is unchanged: an autonomous generator can simulate any output-only law produced by an encryption mechanism. Progress requires a narrower, externally grounded or independently discriminatory prediction.
