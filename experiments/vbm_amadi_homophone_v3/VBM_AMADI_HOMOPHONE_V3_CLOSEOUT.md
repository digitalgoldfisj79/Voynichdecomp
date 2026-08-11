# VBM v3 — Source-Grounded Amadi Homophone Language-ID Closeout

Date: 2026-08-11
Branch: `experiment/vbm-amadi-homophone-v3-20260811`

## Executive result

The v3 instrument was built after v2 was formally closed without touching Voynich H1. Its trigger was independent source evidence from the solved Amadi homophonic cipher: vowel homophone counts `a:e:i:o:u = 3:2:3:4:2`, with alternative signs to be used one after another.

This source-grounded v3 passed its positive-control sweep strongly but failed its preregistered structured-negative gate before any typed Voynich H1 HMM fit was allowed.

Binding conclusion:

**VBM v3 SOURCE-GROUNDED HOMOPHONE LANGUAGE-ID: POSITIVE-CONTROL QUALIFIED, BUT NOT SPECIFIC; CLOSED BEFORE TARGET.**

No Voynich H1 HMM language likelihood was generated. No posterior plaintext was interpreted. `VBM_C1` remains sealed.

---

## 1. Source-grounded model

The VBM representation remained frozen from v1:
- 21 core surface units -> consonant states only;
- 123 cross-boundary bridge units -> vowel states only;
- `VR=qo` for retained `qo...` words, otherwise first retained character;
- VL = final retained character;
- right-edge `eed` and `ed` composite C2 units;
- other maximal `e+` runs composite core units.

The Amadi vowel ratio was scaled exactly to 123 bridge signs by deterministic largest remainder:
- a = 26
- e = 18
- i = 26
- o = 35
- u = 18

Two bridge-use schedules were preregistered:
- `FLAT` equiprobable use within a vowel's assigned signs;
- `CYCLE` deterministic one-after-another use, with a hidden starting phase.

Because Amadi's example source-fixes extra homophones for vowels but not the seven surplus VBM core signs, six consonant-side allocations were swept prospectively: `ANTI_SQRT`, `UNIFORM`, `SQRT_FREQ`, `FREQ_PROP`, `SUPER_FREQ`, `DIRICHLET_SKEW`.

Inference used the v2 moment-factorised typed soft HMM; the endpoint was held-out marginal language likelihood, not homophone-map identity or plaintext recovery.

---

## 2. Q0 positive-control sweep

36 controls = 3 truth languages × 6 core allocations × 2 bridge schedules.

Each control used a fresh 18k-character fit span, fresh 7k-character holdout span and fresh hidden sign assignment. Bavarian, German and Italian HMMs were fitted independently.

A row qualified only if A ensemble, B ensemble and mean likelihood all selected the truth language, mean margin was >=0.02 nats/event, and truth-language A/B score gap was <=0.10.

### Formal Q0 result

**35/36 qualified — PASS.**

- Bavarian: **11/12**; CYCLE **5/6**.
- German: **12/12**; CYCLE **6/6**.
- Italian: **12/12**; CYCLE **6/6**.
- Both Bavarian `FREQ_PROP` core controls passed.

Frozen 5th-percentile true-vs-best-wrong margin floors:
- Bavarian: **0.03265678670240173** nats/event.
- German: **0.08360344975587261**.
- Italian: **0.15600209198750553**.

Bavarian median margin: **0.08257164773431747**.

The sole Q0 failure was Bavarian `SUPER_FREQ + CYCLE`: mean still ranked Bavarian but A selected German, B selected Bavarian; margin 0.016483 and Bavarian A/B score gap 0.100314.

Thus source-grounded Amadi-style vowel homophony made Bavarian language identity recoverable across a very broad consonant-homophone envelope, including uniform core allocation that had failed in v2 when vowel homophones themselves were uniform.

This was sufficient to unlock structured negatives, not target data.

---

## 3. Q1 structured-negative gate

Preregistered 40 negatives:
- 8 IID;
- 8 corrupted/row-permuted Markov;
- 8 motif;
- 8 copy/mutate;
- 8 slot-grammar.

They preserved approximately the FIT-A surface-frequency and core/bridge profile but had no natural-language plaintext generator.

A false language positive required A/B agreement on winning language L, mean language margin >= max(0.02, Q0 margin floor for L), and selected-language A/B score gap <=0.10.

Gate: <=1/40 total and <=1 per generator class.

### Engineering note

The first Q1 launch failed before any scientific row because core/vowel conditional-frequency NumPy slices were views and their normalization mutated the parent probability vector. This was corrected only by `.copy()` on those two slices; generators, seeds, thresholds and model remained unchanged.

### Binding Q1 result

The corrected run was cancelled as soon as the gate became mathematically irrecoverable.

Eight IID rows completed:
- one false positive: IID rep1 selected Bavarian, margin **0.03390677** > Bavarian floor **0.03265679**, score gap **0.00087637**.

The first four Markov rows then emitted:
- rep0: negative;
- rep1: **false positive**, Bavarian margin **0.03288246**, score gap **0.00038796**;
- rep2: **false positive**, Bavarian margin **0.07692274**, score gap **0.03142776**;
- rep3: negative because A/B winning language disagreed.

At that point there were already **3 false positives**, including **2 in the Markov class**, versus allowed maxima of one total and one per class. No later result could restore the gate. The paid job was immediately cancelled.

Therefore:

**Q1 STRUCTURED-NEGATIVE GATE: FAIL.**

---

## 4. Scientific interpretation

v3 establishes a useful but narrower result:

1. The source-grounded Amadi vowel-homophone architecture is sufficiently informative for the soft HMM to distinguish Bavarian, German and Italian on natural positive controls across nearly the full preregistered core-homophone envelope.
2. Exact homophone/plaintext labels remain substantially non-identifiable and are not a valid result endpoint.
3. More importantly, the same Bavarian likelihood margin can be generated by non-language IID/Markov surface processes calibrated to Voynich-like surface statistics.
4. Therefore a Bavarian preference from this HMM on Voynich would not currently have adequate specificity. Running H1 anyway would create an uninterpretable target result and violate the protocol.

This means the previous S0 C/V result remains the cleanest Bavarian-specific observation: higher-order VBM C/V topology preferred Bavarian over German/Italian on six held-out folios. v3 does **not** upgrade that to a language identification.

---

## 5. Target integrity

- No v2 or v3 typed HMM was fitted to Voynich H1.
- `VBM_H1` remains unconsumed by the HMM language-ID endpoint (although its C/V topology was opened in v1).
- `VBM_C1 = f85r1 f53v f33r f10r f23r f111r` remains fully sealed.
- No candidate plaintext exists under v3.

Any next programme must improve **specificity against non-language surface processes**, not merely improve optimizer convergence, add starts or narrow the homophone family.

A legitimate successor would require a new statistic/inference family that prospectively discriminates natural homophonic language from matched synthetic sequence generators before target use—for example a calibrated likelihood-ratio / posterior-predictive test that explicitly models both the language-HMM alternative and a flexible non-language VBM null.

---

## Compute closeout

- v2 formal broad sweep: `6a7b7275f6d0f3ee953aa084` — completed; v2 closed without target.
- v3 first Q0 launch: `6a7b756d27caad61c6eac440` — cancelled after pre-result assertion error.
- v3 corrected Q0: `6a7b75d7f6d0f3ee953aa09c` — completed; 35/36 pass.
- v3 first negative launch: `6a7b780227caad61c6eac47e` — engineering error before scientific rows.
- v3 corrected negative launch: `6a7b789c27caad61c6eac495` — explicitly cancelled when gate became irreversibly false at 3 false positives.
