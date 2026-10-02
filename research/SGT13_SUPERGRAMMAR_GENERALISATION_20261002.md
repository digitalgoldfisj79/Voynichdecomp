# SGT13 / Super-Grammar generalisation suite — 2026-10-02

Canonical Supabase handoff: `voynich_sgt13_supergrammar_generalisation_suite_20261002_v01`.

## Result

The wheel does **not** generalise as one global controller. Instead the manuscript factorises into several small local mechanisms.

### Negative cross-layer tests
- ED1-3 first-token remainder bridge: effect -0.000561 bits / null SD .004743 = **-0.12 SD**.
- line-final atom after full controls: +0.000253 / .003322 = **+0.08 SD**.
- lag-2 exact recurrence: +0.001731 / .004496 = **+0.38 SD**.
- i-run terminator after prev1: +0.000933 / .003722 = **+0.25 SD**; prev2 = **-0.19 SD**.
- token-internal order-2 graphotactics: adding wheel move worsens held-out prediction by **-0.05257 bits/character**, negative in 5/5 folds.
- q-conditioned k/t: borderline matched-CMI screens fail held-out; frozen d/q/y/X state worsens prediction by **-0.003715 bits/event**, 4/5 folds negative.
- SPACE junction: raw CMI is +2.72 SD but frozen d/q/y/X state worsens held-out by **-0.01717 bits/event**, 4/5 folds negative. No coupling promoted.

### Core wheel portability
Continuation action = clockwise / anticlockwise / stay / exit.

After matching source anchor, line position and other metadata:
- Currier: **-0.70 SD**, unresolved.
- Davis hand: **+0.83 SD**, unresolved.
- section: **+0.50 SD**, unresolved.

Thus the core d/q/y action kernel is portable beneath surface register labels.

### Spoke routing
For the exact next opener, after previous opener + position + other metadata:
- Currier: **-0.80 SD**, unresolved.
- hand: **-0.83 SD**, unresolved.
- section: **+3.84 SD**, resolved.

Section effect localises to:
- coarse next class d/q/y/X: **+2.34 SD**;
- exact off-wheel destination: **+2.97 SD**;
- off-wheel re-entry class: **+2.79 SD**.

Interpretation: universal core wheel; section-conditioned peripheral routing.

### Independent line-packing state
Voynich line lengths show strong local memory, but it is not driven by the opener wheel:
- previous length -> current: **+18.18 SD** under full controls.
- second-order length residual: **+3.00 SD** under heavy controls; simpler paragraph-only test **+19.43 SD**.
- length trend memory: **+11.08 SD**.
- Enikel 1397 control trend memory: **+29.58 SD**.

Thus line-packing is another simple local mechanism, with an ordinary-manuscript analogue.

## Architecture

```
PARAGRAPH RESET
    |
    v
UNIVERSAL LEFT-MARGIN CORE
    d -> q -> y -> d
    |
    +-- section-conditioned exit/re-entry spokes
    |
    v
CURRENT LINE OPENER
    |
    +-- token-internal order-2 graphotactics
    +-- q/k-t and i-run local selectors
    +-- SPACE junction morphology
    +-- banded recurrence
    +-- independent line-packing geometry
```

The previous opener does not directly control those downstream layers once the current opener is fixed.

The natural next falsifier is a joint modular forward simulator assembled only from already-supported operators and tested on held-out observables not used to fit the individual modules.
