# VBM BG conditional v5 — Q2 H1 result

Date: 2026-08-11
HF job: `6a7b8c3bf6d0f3ee953aa191`

## Binding result

**Q2 H1 FAIL.**

The result is sharply split between topology and full surface predictive evidence.

### Independent Bavarian/German topology gate — PASS strongly

Frozen threshold:
`TAU_BG = 1.6272712366587183`.

Observed aggregate H1 logit:
**14.205598317083886**.

Per-folio logits:
- f28v: +12.97443821
- f31v: +17.93054030
- f88r: +14.87258343
- f5r: +14.55868576
- f34r: +12.42700957
- f81v: +11.97404250

Thus:
- aggregate gate: PASS;
- Bavarian-direction folios: **6/6**;
- median folio logit: **13.76656198**;
- cross-folio gate: PASS.

This substantially reproduces and strengthens the previously observed H1 Bavarian C/V-topology signal under the independent v5 classifier.

### Language-vs-flexible-null gate — FAIL decisively

Frozen threshold:
`TAU_LANG = +0.21806702545882306` nats/event.

Observed:
`DELTA_LANG = -0.9145712099335848` nats/event.

The best registered non-language model was `markov_hier_o2` with held-out score:
**-2.1170686589658834**.

Homophonic HMM scores:
- Bavarian: **-3.0453000849505765**;
- German: **-3.031639868899468**.

The German HMM is numerically the better of the two by a very small margin, but this is non-evidence: both language models are vastly worse than the flexible non-language null. Exact language ranking is not a binding endpoint in v5.

The HMM A/B fits were nonconverged on the target (Bavarian score gap 0.0287; German 0.0499, with low decode agreement). The preregistered protocol does not permit extra starts or target-driven tuning. More importantly, even the better language score would need an improvement of about **1.13264 nats/event** relative to the frozen threshold to pass the language-vs-null gate.

### Surface geometry

Frozen target geometry reproduced exactly:
- core surface units: 21;
- bridge surface units: 123;
- FIT bridge occurrence coverage: 0.9950440958.

FIT surface HMM input:
- 181 FIT folios;
- 3,867 retained segments;
- 104,888 retained events;
- 121 dropped segments.

H1 surface HMM holdout:
- 96 retained segments;
- 2,365 retained events;
- 5 dropped segments.

The independent topology representation contained 2,526 H1 events: 1,898 core and 628 bridge, bridge fraction 0.2486144.

## Scientific interpretation

v5 resolves the central ambiguity left by v1–v4:

1. The VBM-induced binary C/V topology is robustly and independently Bavarian-like relative to German. This survives cross-corpus calibration and is positive on every H1 folio.
2. That topology signal **does not extend to full surface-sequence evidence for a homophonic Bavarian/German language model**. Once a flexible non-language surface null is allowed, the null wins by a very large margin.

Therefore the H1 result is not evidence that the Voynich text is Bavarian under this VBM cipher model. The surviving observation is narrower: the chosen VBM segmentation induces a higher-order C/V pattern unusually compatible with Bavarian, while the full 144-symbol surface process remains much better predicted by a generic second-order statistical model than by the qualified homophonic language mechanism.

## Consequence

Per the frozen v5 protocol, Q2 requires both the topology and language-vs-null gates. Because the language-vs-null gate fails, **v5 stops here**.

`VBM_C1 = f85r1 f53v f33r f10r f23r f111r` remains sealed and must not be run under v5.

No plaintext or target decode map was inspected.
