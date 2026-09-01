# VBM Joachim Exact v9 — Q2D oracle diagnostic

Date: 2026-09-01
Status: **FROZEN BEFORE Q2D OUTPUT**

Q2 failed formal CAL and remains closed. Q2D is a synthetic-only post-closeout diagnostic. It cannot qualify the instrument, reopen VAL, or authorize any Voynich language fit.

## Question

Why did the blind global-codebook solver fail known synthetic positives?

Two possibilities are distinguished:

1. **optimisation failure**: the true synthetic key has a substantially better language-model objective than ordinary wrong keys, but the coordinate solver cannot reach it;
2. **objective non-identifiability**: the true key is not strongly preferred by the language objective, or simple one-entry deviations from truth can improve the objective.

## Frozen diagnostic

Use the six formal-CAL global positives (`DE_GLOBAL` and `IT_GLOBAL`, reps 0–2) from the frozen Q2 generator.

For each replicate:

- decode HOLDOUT with the known generating key and score under the native 4-gram LM (`TRUE_HOLD_LM`);
- score 200 deterministic random dictionaries from the same candidate inventories on the same HOLDOUT;
- report random median, 95th percentile, maximum, and `ORACLE_ADV = TRUE_HOLD_LM - random_median`;
- report the rank/percentile of the true key among the 201 scores;
- score the actual known plaintext HOLDOUT under both DE and IT LMs to audit whether raw cross-language likelihood is calibrated in the direction of the generating language;
- on FIT+SELECT, start exactly at the true key and perform a **single exhaustive coordinate-neighbourhood scan without accepting changes**: for each occurring bridge type test the four alternative vowels; for each occurring nucleus type test the 31 alternative consonant runs. Count how many dictionary entries possess at least one single-entry alternative with a better global per-character LM objective, and report the best single-entry improvement.

No learned Voynich mapping and no H1/C1 material is accessed.

## Interpretation

- If the true key dominates the random-key distribution but many true entries have improving one-step alternatives, the language objective is not truth-identifying at finite sample even though it contains signal.
- If the true key dominates random keys and is nearly a coordinate local optimum, the main failure is optimisation/search.
- If the true key does not strongly dominate random keys, the objective itself is too weak for this architecture.
- Cross-language raw-score reversal on known plaintext is evidence that Q2's uncalibrated DE-vs-IT selector is unsuitable; this is diagnostic only and does not rescue Q2.

Q2 remains failed regardless of Q2D outcome.
