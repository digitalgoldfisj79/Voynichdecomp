# VBM HMM v2 — Broad Homophone Sweep Closeout

Date: 2026-08-11
Branch: `experiment/vbm-hmm-inference-v2-20260811`

## Binding result

The formal 36-control language-identification sweep completed under the frozen 6 allocation × 2 within-homophone usage grid.

- Overall qualified: **32/36**.
- Bavarian: **10/12**.
- German: **10/12**.
- Italian: **12/12**.
- All **6 FREQ_PROP** controls (3 languages × FLAT/SKEW) qualified.
- Formal Q0-HS-LID verdict: **FAIL**, because the preregistered hostile-anchor requirement demanded both `UNIFORM/FLAT` and `DIRICHLET_SKEW/SKEW` for every language; Bavarian `UNIFORM/FLAT` failed.

No structured-negative stage and no Voynich HMM H1 target fit were authorised. `VBM_C1` remains sealed.

## Frozen homophone sweep

Allocation of surplus core/bridge symbols among typed plaintext letters:

1. `ANTI_SQRT` — frequency^-0.5
2. `UNIFORM`
3. `SQRT_FREQ` — frequency^0.5
4. `FREQ_PROP` — frequency^1.0
5. `SUPER_FREQ` — frequency^1.5
6. `DIRICHLET_SKEW` — Dirichlet(0.20) × frequency^0.5

Each was crossed with:
- `FLAT` equiprobable use within a letter's homophones;
- `SKEW` Dirichlet(0.25) conditional use.

Each truth language therefore had 12 controls; every control used a fresh hidden map and fresh 18k-fit / 7k-holdout plaintext span, and all three candidate language HMMs were fitted.

## Formal language-ID criterion

A control qualified iff:
- A ensemble winner = B ensemble winner = mean winner = truth language;
- mean truth-vs-runner-up margin >=0.02 nats/event;
- truth-language A/B holdout score gap <=0.10 nats/event.

Exact homophone-map/plaintext recovery was diagnostic only, because the soft HMM smoke demonstrated severe homophone-label non-identifiability even when marginal language likelihood was reproducible.

## Bavarian 12-regime result

| allocation | use | margin nats/event | verdict |
|---|---|---:|---|
| ANTI_SQRT | FLAT | 0.073015 | PASS |
| ANTI_SQRT | SKEW | 0.042493 | PASS |
| UNIFORM | FLAT | 0.031903 | **FAIL — German wins A/B/mean** |
| UNIFORM | SKEW | 0.050904 | **FAIL — A German, B Bavarian** |
| SQRT_FREQ | FLAT | 0.079207 | PASS |
| SQRT_FREQ | SKEW | 0.058463 | PASS |
| FREQ_PROP | FLAT | 0.068280 | PASS |
| FREQ_PROP | SKEW | 0.074583 | PASS |
| SUPER_FREQ | FLAT | 0.091410 | PASS |
| SUPER_FREQ | SKEW | 0.041731 | PASS |
| DIRICHLET_SKEW | FLAT | 0.072808 | PASS |
| DIRICHLET_SKEW | SKEW | 0.088509 | PASS |

Bavarian median margin = **0.0705443** nats/event.
All ten non-UNIFORM Bavarian controls qualified.

## German result

German qualified **10/12**. Its two failures were both `SUPER_FREQ` controls; all other allocation families, including both `UNIFORM` and both `FREQ_PROP`, qualified.

German median margin = **0.0852714** nats/event.

## Italian result

Italian qualified **12/12** across the entire homophone envelope.

Italian median margin = **0.166896** nats/event.

## Frozen diagnostic margin floors

5th percentile of all 12 true-vs-best-wrong margins, including failures:

- Bavarian: **0.0373084050**
- German: **0.0358822536**
- Italian: **0.1571089084**

These were not used on Voynich because Q0 failed.

## Scientific interpretation

The new soft-HMM/moment instrument demonstrates that language identity can survive extensive homophony even when individual homophone labels cannot be recovered. The result is broad but not universal. In particular:

- Bavarian is recoverable under anti-frequency, sqrt-frequency, frequency-proportional, super-frequency, and strongly irregular allocations, but not under the deliberately balanced `UNIFORM` allocation in this formal sample.
- German has a different failure boundary, at `SUPER_FREQ`.
- Italian is substantially easier to identify against this panel.
- The fact that all `FREQ_PROP` controls pass is diagnostic only; v2 cannot be narrowed to that family after observing the result.

Therefore **VBM HMM v2 is formally closed without target use**.

## Independent source discovery after v2 closeout

A subsequent audit of the Amadi/Scheers source material found an independently specified historical homophone allocation in the solved Amadi cipher: vowel homophone counts `a:e:i:o:u = 3:2:3:4:2`, with the alternatives described as being used one after another. This allocation is neither `UNIFORM` nor the post-sweep `FREQ_PROP` subset. It provides an external basis for a prospectively separate v3 rather than relaxing v2.

## Compute

Formal Q0 HF job: `6a7b7275f6d0f3ee953aa084`, completed normally in 465 s. No paid job was left running at closeout.
