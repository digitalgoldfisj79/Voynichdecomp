# Anonymous word-equality instrument v0.1 — RESULT

Status: **IMPLEMENTATION / SAME-SOURCE HOLDOUT PASS**. This qualifies only image-level whole-token equality extraction on the consumed 1541 Bovo print family. It does not qualify glyph lengths, Yiddish population transfer, R/L/T, or Voynich inference.

Protocol freeze commit: `ecff87f9c4c06533905f99dffdbeb9a6d5bfeb3b`.
Nuisance matching addendum freeze commit: `90f3739498472edbfd397a5eecb4c0b9f7917235`.

## BUILD

Bovo 1541 Archive leaf n51, stanzas 29–31; Penn orthographic reference 1507w-bovo stanzas 29–31.

- retained image words: 157
- Penn reconstructed orthographic words: 159
- tuned complete-link cosine threshold tau: **0.410**
- image TTR 0.719745 vs Penn 0.666667
- image hapax fraction 0.522293 vs Penn 0.522013
- image repeated-token fraction 0.477707 vs Penn 0.477987
- image max-frequency fraction 0.038217 vs Penn 0.062893
- image equal-pair density 0.005390 vs Penn 0.010509
- registered BUILD loss 0.103912

## HOLDOUT

Bovo 1541 Archive leaf n54, stanzas 39–41; no holdout statistic entered the threshold selection.

- image N=179; Penn N=175; relative token-count error **0.022857** <=0.03 PASS
- |ΔTTR| **0.013121** <=0.05 PASS
- |Δhapax fraction| **0.039298** <=0.05 PASS
- |Δrepeated-token fraction| **0.039298** <=0.05 PASS
- |Δmax-frequency fraction| **0.012067** <=0.02 PASS
- |Δequal-pair density| **0.003012** <=0.010 PASS
- BUILD and HOLDOUT nondegeneracy gates PASS.

Therefore same-source holdout gate PASS.

## Registered nuisance robustness

Pages n82, n86, n120; eight fixed scan perturbations each =24 cells.

- **24/24 cells PASS** (registered gate >=22/24).
- Every cell retained exactly the original token count and matched-position fraction 1.000.
- Equality-partition ARI =1.000 in 22/24 cells.
- n86 blur sigma 0.8: ARI 0.968221 PASS.
- n120 blur sigma 0.8: ARI 0.990841 PASS.

No fresh 1599 page and no Voynich transcription was loaded during tuning or qualification.

## Bound / interpretation

This result establishes that a label-free visual detector can recover a stable approximation to the whole-token equality partition in one consumed sixteenth-century Yiddish print and transfer across held-out pages and scan nuisances from that same print. It does **not** yet establish transfer across printer/typeface/source. The next gate must therefore be an independently frozen source-transfer audit on the untouched Basel 1599 witness. That witness remains a 1501–1600 transfer-panel source under the overarching Yiddish protocol; it cannot qualify the 1350–1500 core period by itself.
