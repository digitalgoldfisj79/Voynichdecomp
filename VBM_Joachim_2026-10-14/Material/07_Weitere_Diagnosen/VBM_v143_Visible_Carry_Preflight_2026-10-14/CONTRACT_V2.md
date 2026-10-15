# Visible carry proxy preflight V2, frozen before scoring

## RETRACTIONS / NARROWED PRIOR RESULTS
1. V1 showed positive observable association in **all three** synthetic arms (carry, statefree, unchanged v142b). Therefore the mere association is **not diagnostic** of cross-word carry. V1 also compared different admitted test subsets and is NOT an arm-comparable estimate. Preserve `RESULTS.json` from V1 separately.
2. v143 source-aware roundtrip/paired detection remains a synthetic result; it does not identify the Voynich mechanism.
3. Previous v142/v143 surface NO-GO and V10 failures are unchanged.

## Invariance and fair contrast
- Every arm uses exactly the same Hartlieb source, number of ciphertext words, **first glyph** and **whole-word length** per token.
- Take only token positions i where 5 <= ciphertext word length <= 20 and i>0. For all arms, shell is (length, first glyph) and the outcome is the middle character of the current word. Predictor is the last three *visible ciphertext characters* of the previous word.
- Train/testing split: 5 folds defined by source token index `floor(i/1000) % 5`. No plaintext, decoded labels or codebook are used in the model.
- Admit a test position only if the shell in **all three** training arms has >=20 observations with >=2 outcome characters; the identical positions are scored in all three.
- Baseline P(Y | shell) add-3 global-smoothed. Alternative P(Y | shell, previous three-character suffix) add-15 to the baseline.
- Null: permute previous suffix labels within (shell, held-out block, fold) among test positions, using the **same permutation of indices in all three arms**. 60 replicates and deterministic seed 23091. Preserves shell, lengths, sample, folds, local block composition and arm alignment. The test is transfer of fitted effect to held-out observations, not origin identification.
- Primary contrasts: carry minus baseline and carry minus statefree gain in held-out bits per event, centered against the corresponding **paired null difference**. Compare to paired-null SD. Demand effect >= 2 SD AND neither negative control be independently elevated in a way that makes the proposed suffix relation non-specific. If failure, no Voynich trial.
- Sensitivity: this proxy is not an entailment of v143 without its key, so failure does not refute cross-word cipher generally. A positive result cannot distinguish plaintext-driven carry from an autonomous context-conditioned process.
