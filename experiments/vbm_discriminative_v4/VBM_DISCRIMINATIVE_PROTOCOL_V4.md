# VBM discriminative evidence v4 — protocol

Date: 2026-08-11
Namespace: `VBMDISCV4`
Parent: `experiment/vbm-amadi-homophone-v3-20260811`

## Purpose

v3 showed that a source-grounded Amadi homophone HMM can identify Bavarian/German/Italian positive controls, but its raw language-margin endpoint is not specific against Voynich-shaped non-language generators. v4 therefore tests a different quantity: whether the language+homophony model predicts held-out surface data better than a deliberately strong non-language VBM null.

No v2/v3 typed HMM has been fitted to Voynich H1. VBM_C1 remains sealed.

## Frozen VBM language model

Representation is unchanged from v1/v3:
- 21 core surface units -> consonant latent letters only;
- 123 cross-boundary bridge surface units -> vowel latent letters only;
- `VR=qo` for retained `qo...`, otherwise first retained character;
- VL = final retained character;
- right-edge `eed` and `ed` are composite C2 units;
- remaining maximal `e+` chains are composite core units.

Language panel is frozen: Bavarian primary, German nearest rival, Italian source-native comparator.

Homophone calibration is source-grounded from the solved Amadi cipher:
`a:e:i:o:u = 3:2:3:4:2`, scaled by largest remainder to 123 bridge surfaces as `26:18:26:35:18`.
Two bridge schedules remain admitted: FLAT and source-literal CYCLE. The seven surplus core homophones are swept across the six v3 regimes: ANTI_SQRT, UNIFORM, SQRT_FREQ, FREQ_PROP, SUPER_FREQ, DIRICHLET_SKEW.

The language likelihood is the held-out observation log likelihood from the moment-initialised typed soft HMM, averaged over independent A/B ensembles.

## Frozen non-language null family

Every null is trained on exactly the same fit surface sequences and evaluated prospectively on the same held-out sequences. The null is allowed to choose the **best held-out score** across the registered family; this deliberately advantages the null and needs no post-hoc complexity reward.

Null family:
1. categorical IID with Jeffreys smoothing;
2. hierarchical Dirichlet/backoff surface Markov models of maximum orders 1, 2, 3, 4, 5 and 6;
3. typed hierarchical Markov models that factor C/V event type from the within-type surface identity, maximum orders 1–5;
4. periodic slot models of periods 2–8, with Jeffreys-smoothed `p(surface | phase)`;
5. typed periodic slot models of periods 2–8, modelling phase-specific C/V probability and conditional within-type symbol probability.

The null score is `max(null_family heldout log likelihood / event)`.

## Primary statistic

For each language L:

`Delta_L = language_HMM_score_L - best_null_score`.

Language evidence exists only if the winning language has positive calibrated Delta and separately wins the three-language HMM comparison.

## Q0 positive calibration

Fresh namespace, fresh control spans and fresh hidden maps. 36 controls = 3 truth languages × 6 core-homophone allocations × 2 bridge schedules. Each uses ~18k fit + ~7k heldout plaintext characters.

For each row, fit all three language HMMs and the full null family.

A positive-control row qualifies iff:
- A and B HMM ensembles and their mean all select the truth language;
- truth-vs-best-wrong HMM margin >= 0.02 nats/event;
- truth-language A/B score gap <= 0.10;
- `Delta_truth > 0`.

Q0 passes iff:
- >=10/12 rows qualify for each language;
- >=5/6 CYCLE rows qualify for each language;
- both Bavarian FREQ_PROP rows qualify;
- >=34/36 rows qualify overall.

After Q0, freeze for each language the 5th percentile of positive `Delta_truth` among that language's controls as `DELTA_FLOOR_L`, and the 5th percentile truth-vs-wrong HMM margin as `LANG_MARGIN_FLOOR_L`.

## Q1 adversarial non-language calibration

Generate at least 200 fresh negatives before target use, 40 each from:
- matched IID;
- row-permuted typed Markov;
- motif/mutation;
- long-copy/mutation;
- slot grammar.

Each generator is calibrated to FIT-A surface frequencies, sequence lengths and C/V event fraction. Negative rows are evaluated exactly like positives.

A false language positive requires all of:
- same winner in A and B and mean;
- winner HMM margin >= `max(0.02, LANG_MARGIN_FLOOR_winner)`;
- winner `Delta >= DELTA_FLOOR_winner`;
- winner A/B score gap <=0.10.

Q1 passes iff false-positive rate <=1% overall and <=2.5% in every generator family. With 200 negatives this means <=2 total and <=1 of 40 in any family.

Early stop: cancel the paid run once either bound is mathematically unrecoverable.

## Q2 cross-transfer calibration

Only if Q0 and Q1 pass.

For 18 fresh positive controls (6 per language, spanning both bridge schedules and broad core allocations), split the fit corpus deterministically into three disjoint source blocks. Fit the language HMM emission model separately on each block and score the same untouched holdout. Fit the registered null independently on each source block as well.

Define `Delta_transfer` as the median of the three block-specific language-minus-null deltas. A row transfers iff all three blocks select the truth language by mean HMM score and `Delta_transfer > 0`.

Q2 passes iff >=16/18 rows transfer and >=5/6 Bavarian rows transfer. Freeze the 5th percentile Bavarian `Delta_transfer` as the target transfer floor.

## Target H1

Only if Q0, Q1 and Q2 all pass.

Fit on the original 181-folio FIT-A and score only the already-defined six-folio `VBM_H1 = f28v f31v f88r f5r f34r f81v`.

A target candidate requires:
1. the same winning language in A and B and mean;
2. HMM language margin >= the winner's frozen language-margin floor;
3. `Delta_winner >= DELTA_FLOOR_winner`;
4. A/B score gap <=0.10;
5. three-way FIT-A folio-block transfer: same winning language from all three independent FIT blocks and median transfer Delta >= the frozen target transfer floor.

Only then may `VBM_C1 = f85r1 f53v f33r f10r f23r f111r` be opened.

## Stop rules

- No target scoring before Q0/Q1/Q2 pass.
- No changing the null family after seeing a target score.
- No dropping a null because it is too competitive.
- No optimizer-budget increase after target access.
- Plaintext/homophone-map recovery is diagnostic only and cannot establish a positive result.
