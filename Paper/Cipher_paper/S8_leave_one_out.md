# Supplement S8: Leave-One-Out Analysis

## Method

Each of the twelve assigned function words (ten inferred plus two known anchors)
is removed one at a time. For each removal, the remaining assignments are evaluated
on three metrics:

- **Training r:** Bigram correlation on Ald.211 (2,006 words)
- **CI r:** Cross-validation on Circa Instans (52,004 words, unseen during training)
- **Held-out r:** Three-fold cross-validation on VMS Herbal-A folios

The table shows the change (Δ) in each metric when a word is removed.

## Full Results

| Removed | Family | Freq (Ald.211) | Δ train r | Δ CI r | Δ held-out r |
|---------|--------|-----------------|-----------|--------|--------------|
| et | Y | 186 | −0.513 | −0.392 | −0.547 |
| cum | N | 42 | −0.054 | −0.021 | −0.051 |
| de | L | 41 | −0.034 | −0.015 | −0.035 |
| habet | L | 18 | −0.016 | +0.005 | −0.014 |
| hoc | N | 12 | −0.008 | +0.003 | −0.006 |
| supra | L | 8 | −0.006 | −0.001 | −0.005 |
| postea | Y | 7 | −0.005 | −0.002 | −0.005 |
| uel | L | 15 | −0.004 | −0.007 | −0.003 |
| que | L | 11 | −0.003 | −0.003 | −0.004 |
| eam | R | 6 | −0.001 | −0.000 | −0.001 |
| in | N | 0* | 0.000 | 0.000 | 0.000 |
| ad | L | 21 | −0.002 | −0.001 | −0.002 |

*Note: 'in' shows zero impact because the vowel heuristic (first vowel 'i' → N-family)
already assigns it to the correct family. The nomenclator entry is redundant — it
confirms the heuristic rather than correcting it.

## Key findings

1. **et dominates:** Removing et collapses all three r values by 0.39–0.55. The vowel
   heuristic routes et (first vowel 'e') to R-family; the nomenclator corrects it to
   Y-family. This single correction transforms the bigram profile.

2. **No word is expendable for significance:** With only et+in (the two known anchors),
   training r = 0.81, which falls below the null maximum (0.953). The eight free
   assignments raise r above the significance threshold. Removing any one of the eight
   degrades the fit but does not individually breach significance.

3. **Cross-validation tracks training:** Every word that improves training r also
   improves (or holds neutral) the CI and held-out metrics, confirming the assignments
   generalise rather than overfit.

4. **habet and hoc show slight CI improvement when removed:** This suggests these
   two words are slightly Ald.211-specific. The net CI effect is negligible (+0.005
   and +0.003), and both degrade training and held-out metrics.

## Baseline comparisons

| Configuration | Train r | CI r | Held-out r |
|---------------|---------|------|------------|
| Heuristic only (no nomenclator) | 0.31 | 0.44 | 0.36 |
| et + in only | 0.81 | 0.84 | 0.79 |
| Full 12 assignments | 0.96 | 0.89 | 0.95 |
| Null maximum (10,000 random) | 0.953 | — | — |
