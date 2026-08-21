# VBM five-state × production factorisation — v0.2 closeout

Protocol SHA-256: `8a728bc681a9a70460285ea157f542b78dc89f6dd0abbc8a473526ab2e941846`

## Calibration
- Positive 5×2 synthetic: K5 recovery=1.000; selected K=5; K5 NMI=1.000; fold-stability NMI=1.000; **PASS**.
- Hostile true-10 synthetic: K5 recovery=0.564; selected K=10; false-rescue rejection **PASS**.
- Target was opened only after both calibration directions passed.

## Unconstrained dimensionality
- Extending the unchanged v0.1 representation to K=30 locates the raw minimum at **K=27**; the one-SE choice is **K=19**.
- K29→K30 change is +0.00346 bits/event; the curve is effectively flat in the high-20s, so K=27 is not interpreted as 27 literal symbols.

## Primary 5 × (Currier × line-position) test
- Production-conditioned raw best: **K=23**; one-SE: **K=13**.
- K=5 gain beyond production state: **1.1993 bits/event** versus best **1.4046**.
- Five-state recovery: **85.4%**, below the frozen 90% survival gate.
- Five-class fold stability: median NMI **0.750** (passes the 0.70 stability gate).
- Matched production-label shuffles: **1.2250 ± 0.0083 bits/event**; observed five-state gain is **z=-3.09** relative to null — the wrong direction.
- Currier-A vs Currier-B independent K5 fits: NMI **0.625** across 54 shared bridge types (passes invariance floor).

## Secondary production arms
- linepos: K5 recovery **82.3%** vs K23; fold-stability NMI 0.574.
- currier: K5 recovery **83.3%** vs K23; fold-stability NMI 0.658.
- hand: K5 recovery **85.5%** vs K23; fold-stability NMI 0.658.
- section: K5 recovery **89.7%** vs K23; fold-stability NMI 0.550.

## Frozen gate
- recovery: **FAIL**
- selected_K: **FAIL**
- fold_stability: **PASS**
- shuffle_z: **FAIL**
- simple_arm: **PASS**
- cross_currier: **PASS**

## Binding verdict
**FIVE-STATE VBM DOES NOT SURVIVE THE PRODUCTION-FACTOR TEST.**

The boundary bridge remains a strong empirical object. What fails is the parsimonious rescue in which its higher dimensionality is merely five invariant values multiplied by independently known Currier/line-position production state. The real production labels do not recover the missing structure; under the matched shuffle they are actually less useful than randomized labels. No Bavarian/MHG or vowel-letter interpretation is licensed by this run.
