# VBM Bavarian–German conditional discriminator v5 — closeout

Date: 2026-08-11
Branch: `experiment/vbm-bg-conditional-v5-20260811`

## Binding conclusion

**VBM BG CONDITIONAL v5: Q0 PASS; Q1 PASS; H1 FAILS LANGUAGE-vs-NULL; C1 REMAINS SEALED.**

v5 successfully qualified two instruments prospectively before target use:

1. an independent Bavarian-vs-German C/V-topology discriminator with cross-corpus transfer;
2. a homophonic language-vs-flexible-nonlanguage predictive detector.

On H1 these instruments disagreed sharply. The topology result was extremely Bavarian-directional, but the full surface-sequence language model was decisively outperformed by a generic second-order surface Markov null. Because v5 preregistered the conjunction of both gates, H1 failed and the programme stopped without opening C1.

No plaintext or target decode map was inspected.

---

## Q0 — independent Bavarian/German topology discriminator

HF job `6a7b886c27caad61c6eac5f4`.

Features were independent of the HMM: C/V Markov log-likelihood-ratio features orders 1–6 plus consonant- and vowel-run likelihood ratios.

Discovery validation:
- balanced accuracy **1.0000**;
- Bavarian recall **48/48**;
- German recall **41/41**.

Independent transfer:
- non-Wikipedia MaiBaam Bavarian: **12/13** correct;
- German PUD: **48/48** correct;
- balanced accuracy **0.96153846**.

Frozen target threshold:

`TAU_BG = 1.6272712366587183`.

The small explicit South-Bavarian diagnostic subset was too small for a binding regional inference. Q0 therefore qualifies broad Bavarian-vs-German topology discrimination, not a South-Bavarian/Tyrolean claim.

---

## Q1 — language vs flexible non-language null

HF job `6a7b88f427caad61c6eac608`.

Statistic:

`DELTA_LANG = max(HMM_Bavarian, HMM_German) - best_registered_nonlanguage_null`.

Only source-grounded CYCLE homophone scheduling was admitted.

Calibration namespace:
- 12/12 positive controls pass;
- 0/24 negative controls false-positive;
- balanced accuracy **1.0000**.

Frozen threshold:

`TAU_LANG = 0.21806702545882306` nats/event.

Untouched validation namespace:
- 12/12 positives above threshold;
- 0/24 negatives above threshold;
- Bavarian recall **6/6**;
- German recall **6/6**;
- balanced accuracy **1.0000**.

Validation negative-family medians:
- typed IID: -0.034722;
- typed first-order Markov: +0.014879;
- typed periodic slot: -0.036710;
- block-shuffle3: +0.117023.

Hardest validation negative: +0.129339, still well below the frozen threshold.

Thus Q1 was fully qualified before H1 access.

---

## Q2 — H1

HF job `6a7b8c3bf6d0f3ee953aa191`.

H1 folios:
`f28v f31v f88r f5r f34r f81v`.

### Topology gate

Frozen threshold:
`TAU_BG = 1.6272712367`.

Observed aggregate H1 Bavarian logit:
**14.2055983171** — PASS.

Per folio:
- f28v +12.97443821
- f31v +17.93054030
- f88r +14.87258343
- f5r +14.55868576
- f34r +12.42700957
- f81v +11.97404250

Thus **6/6 folios** are Bavarian-directional and the median folio logit is **13.76656198**. The topology signal is not a marginal pass; it is far above the independently frozen transfer floor.

### Full surface language-vs-null gate

Frozen threshold:
`TAU_LANG = +0.2180670255` nats/event.

Observed:

`DELTA_LANG = -0.9145712099` nats/event — **FAIL decisively**.

Best non-language model:
- `markov_hier_o2` score **-2.1170686590**.

Language HMMs:
- Bavarian **-3.0453000850**;
- German **-3.0316398689**.

German is numerically the better language HMM by only ~0.01366 nats/event, but this ranking is non-evidence because both language models lose to the non-language null by roughly nine-tenths of a nat/event. The better language model is approximately **1.13264 nats/event below the binding pass threshold relative to the null**.

Both target language HMM ensembles were nonconverged under the frozen optimiser. No extra starts or target-driven tuning were permitted. This does not rescue the result: the registered instrument as qualified on controls fails H1 by a very large predictive margin.

### Surface geometry

Frozen geometry reproduced:
- 21 core units;
- 123 bridge units;
- FIT bridge occurrence coverage **0.9950440958**.

FIT HMM input:
- 181 folios;
- 3,867 retained segments;
- 104,888 retained events.

H1 HMM holdout:
- 96 retained segments;
- 2,365 retained events.

Independent topology representation:
- 2,526 events;
- 1,898 core;
- 628 bridge;
- bridge fraction 0.2486144.

---

## Scientific interpretation

The programme now cleanly separates two phenomena that earlier VBM work could not separate.

### What survives

The VBM segmentation induces a binary C/V topology that is strongly Bavarian-like relative to German. This signal:
- was originally observed on H1 in v1;
- survives a new feature family independent of the HMM;
- transfers from Bavarian Wikipedia/German GSD training to independent MaiBaam/PUD material;
- appears in the Bavarian direction on all six H1 folios;
- exceeds the prospectively frozen transfer-derived strength threshold by a very large margin.

This is a real structural observation about the VBM representation.

### What does not survive

The stronger claim that the 144-symbol VBM surface stream behaves like a homophonically encoded Bavarian/German language fails badly once compared directly against a flexible non-language surface process. A generic second-order Markov null predicts held-out H1 far better than either qualified language HMM.

Therefore v5 does **not** support the claim that Voynich is Bavarian under this VBM cipher mechanism. The surviving result is narrower and representation-level: **VBM creates a higher-order C/V topology with unusually Bavarian-like structure, but the full surface process is not supported as a homophonic Bavarian/German language encoding by the qualified predictive test.**

This distinction is important: it suggests that the Bavarian C/V resemblance may arise from structural regularities in Voynich/VBM segmentation that need not correspond to latent Bavarian plaintext.

---

## C1 disposition

`VBM_C1 = f85r1 f53v f33r f10r f23r f111r` was not scored under v5.

Because H1 failed the preregistered language-vs-null gate, Q3 was not authorised. C1 remains sealed for any genuinely different future hypothesis.

---

## Compute ledger

- Q0 topology: `6a7b886c27caad61c6eac5f4` — completed.
- Q1 language/null: `6a7b88f427caad61c6eac608` — completed.
- Q2 H1: `6a7b8c3bf6d0f3ee953aa191` — completed.
- Q3 C1: not launched.

No target plaintext or decode mapping was inspected.
