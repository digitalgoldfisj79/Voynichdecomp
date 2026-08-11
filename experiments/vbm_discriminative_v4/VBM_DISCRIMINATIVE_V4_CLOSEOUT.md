# VBM discriminative evidence v4 / v4.1 — closeout

Date: 2026-08-11
Branch: `experiment/vbm-discriminative-evidence-v4-20260811`

## Binding conclusion

The proposed language-vs-flexible-null programme was implemented and calibrated prospectively. The primary `Delta = HMM language likelihood - best registered non-language null likelihood` statistic behaved strongly on natural-language positive controls, but the full preregistered language-identification qualification did not reproduce robustly enough across the broad homophone envelope.

**VBM DISCRIMINATIVE EVIDENCE v4/v4.1: CALIBRATION NOT QUALIFIED; CLOSED BEFORE Q1/TARGET.**

No v4/v4.1 Voynich H1 HMM fit was generated. No posterior plaintext was inspected. `VBM_C1 = f85r1 f53v f33r f10r f23r f111r` remains sealed.

---

## v4 model

The VBM representation stayed frozen:
- 21 core surface units -> consonant latent letters;
- 123 cross-boundary bridge surface units -> vowel latent letters;
- source-grounded Amadi vowel-homophone ratio `a:e:i:o:u = 3:2:3:4:2`, scaled to `26:18:26:35:18` bridge symbols;
- FLAT and literal Amadi-style CYCLE bridge use;
- six core-homophone allocation regimes: ANTI_SQRT, UNIFORM, SQRT_FREQ, FREQ_PROP, SUPER_FREQ, DIRICHLET_SKEW;
- languages Bavarian/German/Italian;
- moment-initialised typed soft HMM, independent A/B ensembles.

Registered non-language null family:
1. Jeffreys-smoothed categorical IID;
2. hierarchical/backoff surface Markov orders 1–6;
3. typed hierarchical Markov orders 1–5;
4. periodic surface slot models periods 2–8;
5. typed periodic slot models periods 2–8.

For every row, the null was allowed to take the best held-out likelihood across the entire family.

Primary statistic:
`Delta_L = heldout_HMM_L - best_heldout_null`.

---

## v4 Q0

36 fresh controls, one per language × core regime × bridge schedule.

Original frozen gate:
- each language >=10/12 qualified;
- each language CYCLE >=5/6;
- both Bavarian FREQ_PROP rows qualified;
- overall >=34/36.

Result:
- Bavarian 10/12, CYCLE 6/6, both FREQ_PROP pass;
- German 11/12, CYCLE 5/6;
- Italian 12/12, CYCLE 6/6;
- overall **33/36**.

Therefore v4 Q0 formally failed the aggregate gate by one row.

Importantly, all 36 truth-language Deltas were positive. The three row failures were language-margin or A/B language-ranking failures, not failures of language-vs-null predictive evidence.

Computed but nonbinding v4 floors (not reused later):
- Bavarian Delta p05 0.19884390; language-margin p05 0.05217066;
- German Delta p05 0.23850194; language-margin p05 0.08317762;
- Italian Delta p05 0.30929004; language-margin p05 0.15467595.

Because the aggregate >=34/36 rule was not implied by the simultaneously stated per-language robustness criteria, v4 was closed and a doubled fresh replication was preregistered before any target access.

---

## v4.1 Q0 replication

Namespace `VBMDISCV41`.

72 entirely fresh controls were preregistered: two independent replicas per each of the 36 cells. No v4 spans, keys, floors or fitted states were reused.

Binding v4.1 gate:
- each language >=20/24 qualified;
- each language >=10/12 CYCLE;
- all four Bavarian FREQ_PROP rows qualified.

No redundant aggregate threshold was added.

### Bavarian slice — complete

**20/24 qualified — PASS exactly.**

- CYCLE: **12/12 PASS**.
- FREQ_PROP: **4/4 PASS**.
- Four failures, all FLAT:
  - ANTI_SQRT/FLAT rep0;
  - ANTI_SQRT/FLAT rep1;
  - SUPER_FREQ/FLAT rep1;
  - DIRICHLET_SKEW/FLAT rep0.

Every Bavarian truth Delta emitted was positive. Approximate observed range among the 24 rows: **+0.1565 to +0.2923 nats/event**.

The contrast between 12/12 CYCLE and 8/12 FLAT is a calibration observation only; it is not target evidence. It is nevertheless consistent with the independent Amadi instruction to use a vowel's alternatives one after another.

### German slice — complete

**19/24 qualified — FAIL by one row.**

- CYCLE: **11/12 PASS**.
- Five failures:
  - ANTI_SQRT/FLAT rep0;
  - ANTI_SQRT/FLAT rep1;
  - SUPER_FREQ/FLAT rep0;
  - DIRICHLET_SKEW/FLAT rep1;
  - DIRICHLET_SKEW/CYCLE rep0.

Again, every emitted German truth Delta was positive. Approximate observed range: **+0.2260 to +0.3557 nats/event**.

Because 19/24 is below the frozen 20/24 requirement after all 24 German rows had completed, the Q0.1 gate became mathematically impossible regardless of the Italian slice. The paid CPU-XL job was immediately cancelled before running the remaining Italian work.

### Italian slice

Not completed because the global Q0.1 pass had already become impossible. No inference should be made from its absence.

---

## Scientific interpretation

The v4 programme separates two claims that were conflated in v3:

1. **Natural-language/homophonic model vs flexible non-language null.** This part performed very strongly in all emitted v4/v4.1 positive controls: the truth-language Delta stayed positive even in cells where the exact language-ranking gate failed.
2. **Bavarian vs German vs Italian identification under extreme homophone allocations.** This remains insufficiently robust. Fresh v4.1 German controls missed the preregistered threshold by one row, while Bavarian passed exactly at threshold.

Therefore the programme does **not** justify a Voynich language test yet. In particular, running H1 now and reporting a Bavarian Delta would be invalid because the full instrument was not prospectively qualified.

The most useful result is diagnostic: the flexible null largely solves the v3 specificity problem, but the remaining bottleneck is **close-language discrimination / ensemble stability under hostile homophone allocation**, not language-vs-null evidence itself.

A successor should not add HMM starts or relax the broad-homophone gate. It should target Bavarian-vs-German discrimination directly, ideally using an independent feature family or likelihood ratio conditioned on having already passed the language-vs-null gate.

---

## Target integrity

- v4 Q1 200-negative calibration: **not run** because Q0/Q0.1 did not qualify.
- v4 Q2 cross-transfer: **not run**.
- Voynich `VBM_H1 = f28v f31v f88r f5r f34r f81v`: **no v4/v4.1 HMM target fit generated**.
- `VBM_C1`: **sealed**.
- No candidate plaintext exists.

---

## Compute ledger

- v4 smoke: HF `6a7b7f30f6d0f3ee953aa100` — completed.
- v4 formal Q0: HF `6a7b7f9227caad61c6eac534` — completed; 33/36, formal fail.
- v4.1 doubled Q0: HF `6a7b8230f6d0f3ee953aa12d` — explicitly cancelled immediately after German completed at 19/24 and the global gate became impossible.

No target job was launched.
