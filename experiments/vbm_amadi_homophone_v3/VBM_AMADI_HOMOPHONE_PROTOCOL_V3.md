# VBM v3 — Source-Grounded Amadi Homophone Language-ID Protocol

Date: 2026-08-11
Namespace: `VBMAMADIV3`
Parent v2 closeout: `da477e1289197ae452150e4837ad1757c529bfc9`

## Independent historical trigger

After formal v2 was closed without touching Voynich H1, the Amadi/Scheers source audit recovered the actual homophone allocation in the solved 12-letter homophonic cipher:

- a: 3 cipher signs
- e: 2
- i: 3
- o: 4
- u: 2

Amadi's prose says the alternative vowel signs are to be used 'one after another'. This is an independently specified irregular homophone schedule. It is not the v2 `UNIFORM` family and is not a post-hoc choice of the v2 `FREQ_PROP` family.

v3 therefore tests this exact historical vowel-allocation shape prospectively.

## Representation and target split

Unchanged from VBM v1/v2:
- 21 core surface units constrained to consonant plaintext states;
- 123 bridge surface units constrained to vowel plaintext states;
- `VR=qo` for retained `qo...` words, otherwise first retained character;
- VL = final retained character;
- right-edge `eed`, then `ed`, are composite C2 units;
- other maximal `e+` runs are composite core units;
- FIT-A = original 181 folios;
- VBM_H1 = `f28v f31v f88r f5r f34r f81v`;
- VBM_C1 = `f85r1 f53v f33r f10r f23r f111r`, still sealed.

Language panel unchanged: Bavarian primary, Standard German rival, Italian comparator.

Inference unchanged from v2: moment-factorised typed soft HMM, language bigram transitions fixed, emissions learned from fit data, independent A/B ensembles, held-out marginal log likelihood is the endpoint. Exact plaintext/homophone labels remain diagnostic only.

## Source-grounded bridge allocation

Scale Amadi's `a:e:i:o:u = 3:2:3:4:2` ratio to 123 VBM bridge signs by deterministic largest remainder:

- a = 26
- e = 18
- i = 26
- o = 35
- u = 18

Total = 123.

No alternative bridge-count allocation is admitted in v3.

### Bridge use schedules

Two source-compatible use profiles are tested:

1. `FLAT`: each occurrence chooses equiprobably among that vowel's assigned bridge signs. This is the stochastic counterpart of cycling over a long text.
2. `CYCLE`: signs assigned to a vowel are used deterministically one after another, wrapping at the end, exactly reflecting Amadi's instruction.

The initial phase of every vowel cycle is a fresh deterministic hidden key.

## Core-side uncertainty sweep

Amadi's example adds homophones specifically to vowels; VBM nevertheless has 21 core surface units for 14 normalized consonant states, leaving seven surplus core units whose historical interpretation is not source-fixed. Rather than choosing a favorable allocation, v3 sweeps six preregistered allocations of these seven surplus units:

1. `ANTI_SQRT` — consonant frequency^-0.5
2. `UNIFORM`
3. `SQRT_FREQ` — frequency^0.5
4. `FREQ_PROP` — frequency^1.0
5. `SUPER_FREQ` — frequency^1.5
6. `DIRICHLET_SKEW` — Dirichlet(0.20) × frequency^0.5

Every consonant receives at least one core sign; the seven surplus signs use deterministic largest remainder. Conditional use of assigned core homophones is FLAT.

Crossing 6 core allocations × 2 bridge schedules gives **12 controls per truth language**, 36 total.

## Q0 — source-grounded positive-control sweep

For every truth language × core allocation × bridge schedule:
- fresh 18,000-character fit span;
- fresh 7,000-character holdout span;
- fresh symbol assignments under namespace `VBMAMADIV3`;
- fit Bavarian, German and Italian HMMs independently.

For each candidate retain A and B ensemble holdout scores.

A control qualifies iff:
1. winner_A = winner_B = winner_mean = truth language;
2. mean truth-vs-runner-up margin >= 0.02 nats/event;
3. truth-language |score_A-score_B| <= 0.10 nats/event.

Q0 passes iff:
- >=32/36 controls qualify;
- each truth language >=10/12;
- Bavarian >=10/12;
- for each truth language, at least 5/6 core allocations qualify under `CYCLE`;
- both `FREQ_PROP` core controls (FLAT-bridge and CYCLE-bridge) qualify for Bavarian.

For each language freeze `MARGIN_FLOOR_L` = 5th percentile of all 12 true-vs-best-wrong mean margins, including failures.

If Q0 fails, stop; Voynich H1 is not scored.

## Q1 — structured negatives

Only if Q0 passes.

40 typed negatives, 8 each:
- IID
- permuted order-2 Markov
- motif
- copy/mutate
- slot grammar

Each negative preserves approximately the empirical FIT-A core/bridge event fraction and surface-frequency profile, but has no natural-language generating process.

Fit all three candidate HMMs.

A negative is a false language positive iff:
- A and B select the same language L;
- mean winner margin >= max(0.02, MARGIN_FLOOR_L);
- selected-language |score_A-score_B| <=0.10.

Q1 passes iff <=1/40 false positives and <=1 in any generator class.

If Q1 fails, stop; Voynich H1 is not scored.

## H1 — first typed Voynich HMM target

Only if Q0 and Q1 pass.

For each candidate language:
- fit A/B emissions on FIT-A only;
- apply without refitting to pooled VBM_H1;
- also score each H1 folio separately.

Language L is an H1 candidate iff:
1. winner_A = winner_B = winner_mean = L on pooled H1;
2. pooled mean margin >= max(0.02, MARGIN_FLOOR_L);
3. selected-language pooled |score_A-score_B| <=0.10;
4. L ranks first by mean score on >=5/6 individual H1 folios.

A Bavarian candidate requires L=Bavarian.

No posterior plaintext may be interpreted or reported as decipherment under v3.

If H1 has no candidate, VBM_C1 remains sealed.

## C1 confirmation

Only if H1 has a candidate. Apply the already fitted FIT-A emission matrices to VBM_C1 without refitting.

Confirmation requires:
- same pooled winning language in A/B/mean;
- margin >= max(0.02,MARGIN_FLOOR_L);
- selected-language score gap <=0.10;
- rank1 on >=5/6 C1 folios.

## Stop rules

No change after Q0 begins to:
- bridge count ratio 3:2:3:4:2;
- bridge schedules FLAT/CYCLE;
- six core allocations;
- HMM/moment initializer;
- language panel;
- thresholds;
- target split;
- negative generators.

A v3 failure is binding for this source-grounded Amadi homophone language-ID instrument. Any future programme requires new independent source evidence or a materially different inference family.
