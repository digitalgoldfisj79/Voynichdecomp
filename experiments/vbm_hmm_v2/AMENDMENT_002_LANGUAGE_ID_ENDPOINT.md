# VBM HMM v2 — Amendment 002: Language-identification endpoint

Date: 2026-08-11

Engineering smoke under the broad homophone envelope showed that exact plaintext-letter labels can remain non-identifiable while held-out language marginal likelihood still robustly identifies the generating language. In three full-scale Bavarian smoke controls spanning balanced, classical frequency-proportional and extreme irregular homophony, Bavarian ranked first over German and Italian with held-out mean margins 0.1331, 0.0615 and 0.0765 nats/event respectively.

No formal 36-control sweep and no Voynich HMM target score has yet begun.

Therefore v2 is frozen as a **language-identification instrument**. Plaintext recovery remains diagnostic and cannot be used to claim decipherment.

## Formal Q0-HS-LID sweep

Use all 36 controls: 3 truth languages × the already frozen 12 homophone regimes. Fit Bavarian, German and Italian moment-initialized HMMs to every control.

For each candidate retain independent A/B ensemble holdout scores. Define:
- `winner_A` = language with highest A-ensemble holdout score;
- `winner_B` = language with highest B-ensemble holdout score;
- `winner_mean` = language with highest mean(A,B) score;
- `margin_mean` = winner_mean minus second-best mean score.

A control is a qualified language identification iff:
1. winner_A = winner_B = winner_mean = truth language;
2. margin_mean >= 0.02 nats/event;
3. truth-language |score_A - score_B| <= 0.10 nats/event.

EM iteration-limit flags and plaintext recovery are diagnostics, not gates; ensemble held-out reproducibility replaces arbitrary emission-map convergence.

Q0-HS-LID passes iff:
- >=32/36 controls qualify;
- each truth language qualifies on >=10/12 regimes;
- all 12 `FREQ_PROP` controls (FLAT and SKEW across languages) qualify;
- for each language, both `UNIFORM/FLAT` and `DIRICHLET_SKEW/SKEW` qualify.

For each truth language freeze `MARGIN_FLOOR_L` = 5th percentile of its 12 correct-vs-best-wrong mean margins, including any nonqualifying rows. This makes subsequent target gates sensitive to the entire preregistered homophone envelope rather than a favorable subset.

## Q1 structured negatives

Only if Q0-HS-LID passes.

40 typed structured negatives: 8 each IID, permuted order-2 Markov, motif, copy/mutate, slot grammar. Fit all three language HMMs.

A negative is a false language positive if:
- A and B select the same language L;
- mean margin >= max(0.02, MARGIN_FLOOR_L);
- selected-language |score_A-score_B| <=0.10.

Gate: <=1/40 false positives and <=1 in any generator class.

## Voynich H1

Only if Q0-HS-LID and Q1 negatives pass.

Fit emissions on FIT-A only for all three candidate languages. Apply each fitted A/B model without refitting to pooled VBM_H1 and separately to each of its six folios.

Language L is an H1 candidate iff:
- winner_A = winner_B = winner_mean = L on pooled H1;
- pooled mean margin >= max(0.02, MARGIN_FLOOR_L);
- selected-language pooled |score_A-score_B| <=0.10;
- L ranks first by mean score on at least 5/6 individual H1 folios.

A Bavarian candidate requires L=Bavarian.

No decoded plaintext is a programme result under v2.

## Confirmation

VBM_C1 remains sealed unless H1 produces a candidate. If opened, the already fitted FIT-A emissions are applied without refitting. Confirmation requires the same pooled A/B winner, margin and score-gap gates and rank1 on >=5/6 C1 folios.

## Stop rules

No changing the 12-regime homophone sweep, language panel, moment initializer, HMM, likelihood-ratio gates, H1 split or negative generators after the formal Q0-HS-LID sweep begins. A failure closes this v2 language-ID instrument without touching Voynich H1.
