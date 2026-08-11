# VBM v6.1 — label-invariant key-transfer closeout

Date: 2026-08-11
Branch: `experiment/vbm-key-transfer-v61-20260811`

## Binding formal result

**VBM v6.1 KEY-TRANSFER: SYNTHETIC INSTRUMENT QUALIFIED; VOYNICH FIT PASS; C1 PASS. POSTHOC MATCHED-SURFACE AUDIT SHOWS THE EFFECT IS NOT CIPHER-SPECIFIC.**

The preregistered v6.1 experiment passed every binding gate. It therefore establishes a strong, reproducible cross-folio surface-symbol identity effect in the VBM representation under the topology-preserving permutation intervention.

However, a posthoc FIT-only audit using an empirical joint surface-bigram model reproduced most of the same permutation excess. The formal v6.1 pass must therefore **not** be interpreted as evidence that a latent cipher transducer has been identified. The correct scientific interpretation is narrower: VBM surface identities and pair statistics transfer very strongly across folios, while the additional latent-language/transducer contribution remains unqualified.

No plaintext or per-symbol decode map was inspected.

---

## Instrument

v6.1 retained v6 identity-transfer excess (ITE):

`ITE = held-out observed score under the frozen train transducer - median score after independent topology-preserving within-type label permutation of each held-out folio`.

The intervention preserves C/V topology, segmentation, boundaries and core/bridge positions while destroying reusable surface-symbol identity.

The brittle v6 argmax latent-map stability endpoint was replaced prospectively by **emission-kernel stability (EKS)**:

`K(E) = E.T @ diag(pi) @ E`

with the binding EKS score equal to the Frobenius cosine between normalized kernels fitted independently on two disjoint training halves.

The transducer was the typed moment-factorisation model without post-moment Baum-Welch. Candidate latent languages were Bavarian and German.

Protocol: `VBM_KEY_TRANSFER_V61_PROTOCOL.md`.

---

## Synthetic smoke

HF job `6a7b933b27caad61c6eac6d6`.

Median smoke results:

| family | ITE | EKS |
|---|---:|---:|
| BAV_GLOBAL | 0.837686 | 0.946084 |
| GER_GLOBAL | 1.393271 | 0.844981 |
| BAV_GLOBAL_SWAP | 0.805753 | 0.905757 |
| BAV_FRESH | -0.004438 | 0.592099 |
| GER_FRESH | 0.002953 | 0.566559 |
| STABLE_MARKOV | 0.556142 | 0.942392 |

The smoke showed the intended conjunction: EKS separated reusable from fresh keys, while ITE was needed to reject a stable non-language Markov grammar.

---

## Formal CAL and untouched VAL

HF job `6a7b936f27caad61c6eac6da`.

### Calibration extrema

- weakest reusable-key positive ITE: **0.8577435465**
- strongest negative/adversarial ITE: **0.6150983204**
- weakest reusable-key positive EKS: **0.7571863911**
- strongest fresh-key EKS: **0.5929097839**

Frozen thresholds before Voynich access:

- `TAU_ITE = 0.7364209334512819`
- `TAU_EKS = 0.6750480875047778`

### Untouched validation

Reusable-key positives: **12/12 pass both thresholds**.

- BAV_GLOBAL: 4/4
- GER_GLOBAL: 4/4
- BAV_GLOBAL_SWAP: 4/4

Negative/adversarial false positives: **0/12**.

- BAV_FRESH: 0/4
- GER_FRESH: 0/4
- STABLE_MARKOV: 0/4

Thus the instrument was fully qualified prospectively before Voynich FIT access.

Freeze record: `V61_Q0_FREEZE.md`.

---

## Voynich FIT

HF job `6a7b947827caad61c6eac6f2`.

H1 was not reused. C1 was still sealed during this stage.

Six deterministic held-folio folds:

| fold | selected language | ITE | EKS |
|---:|---|---:|---:|
| 0 | German | 6.615210 | 0.963140 |
| 1 | Bavarian | 6.217176 | 0.896472 |
| 2 | German | 6.428126 | 0.959643 |
| 3 | German | 6.375233 | 0.888853 |
| 4 | German | 6.497391 | 0.896082 |
| 5 | Bavarian | 5.837639 | 0.906441 |

Aggregate FIT result:

- median ITE: **6.4016796828** vs threshold 0.7364209335
- median EKS: **0.9014567295** vs threshold 0.6750480875
- ITE > 0: **6/6 folds**
- EKS above threshold: **6/6 folds**
- latent-language selections: German **4/6**, Bavarian **2/6**

All preregistered FIT gates passed. German was frozen as the FIT plurality before C1 was opened.

Freeze record: `V61_FIT_FREEZE_BEFORE_C1.md`.

---

## One-time C1 test

HF job `6a7b94c527caad61c6eac6f6`.

C1 folios:
`f85r1 f53v f33r f10r f23r f111r`.

Frozen criteria were satisfied:

- selected language: **German**
- frozen FIT plurality: **German**
- C1 score: **-7.2969595426**
- topology-preserving permutation median: **-13.7131480647**
- best of 64 permutations: **-11.8313230513**
- C1 ITE: **6.4161885221** vs threshold 0.7364209335
- observed rank among observed + 64 permutations: **1/65**

Formal C1 verdict: **PASS**.

C1 is now consumed and is no longer an untouched validation set for a successor experiment.

---

## Posthoc audit 1 — ordinary surface statistics

Because topology-preserving label permutation also destroys stable symbol-frequency and transition identities, a FIT-only audit tested whether simple empirical surface models generate the same effect.

Script: `v61_posthoc_surface_null_audit.py`.
HF job: `6a7b950d27caad61c6eac6f8`.

Across the same six FIT folds:

- empirical surface unigram median permutation excess: **1.9476853939**
- empirical first-order conditional Markov median permutation excess: **3.9324184916**

Thus a substantial identity-transfer effect exists without any latent-language or cipher model.

---

## Posthoc audit 2 — matched empirical joint-pair null

The v6.1 transducer scores a joint surface-pair probability matrix. A second FIT-only audit therefore used the exact same scoring geometry with an empirical smoothed joint-bigram matrix fitted directly from training surface data.

Script: `v61_posthoc_matched_pair_audit.py`.
HF job: `6a7b954ff6d0f3ee953aa1eb`.

| fold | latent ITE | empirical pair ITE | latent - pair |
|---:|---:|---:|---:|
| 0 | 6.615210 | 5.766754 | +0.848456 |
| 1 | 6.217176 | 5.875358 | +0.341817 |
| 2 | 6.428126 | 5.793278 | +0.634848 |
| 3 | 6.375233 | 5.828849 | +0.546384 |
| 4 | 6.497391 | 5.818171 | +0.679220 |
| 5 | 5.837639 | 5.877661 | **-0.040022** |

Aggregates:

- median v6.1 latent ITE: **6.4016796828**
- median empirical joint-pair ITE: **5.8235096643**
- median paired residual `latent ITE - empirical pair ITE`: **+0.5906163741**
- latent residual positive: **5/6 folds**

Approximately 91% of the raw median v6.1 ITE magnitude is reproduced by a completely empirical surface joint-bigram model. The +0.5906 residual is posthoc and uncalibrated; it is hypothesis-generating only.

---

## Scientific interpretation

### What v6.1 establishes

The VBM representation contains extremely strong reusable cross-folio **surface-symbol identity and surface-pair structure**. This effect:

- survives six held-folio FIT folds;
- reproduces strongly on the formerly sealed C1 set;
- is far stronger than fresh-key language controls under the original intervention;
- is accompanied by reproducible label-invariant emission geometry.

The effect is real and highly reproducible.

### What v6.1 does not establish

v6.1 does **not** establish that the reusable structure is a cipher key or latent-language transducer. The matched empirical pair audit shows that ordinary stable surface bigram statistics reproduce most of the permutation contrast.

Accordingly:

- the German 4/6 FIT plurality and German C1 selection are **not evidence that Voynich plaintext is German**;
- the formal C1 pass is a pass for the preregistered v6.1 statistic, not a cipher-specific validation;
- no decipherment claim is licensed;
- no plaintext or mapping should be extracted from v6.1.

This result is consistent with the v5 finding that a flexible surface Markov model predicts held-out H1 substantially better than the Bavarian/German homophonic HMM. Both experiments now point to strong surface regularity as the dominant explanation that must be controlled before making a cipher inference.

### Live successor hypothesis

The scientifically meaningful unresolved statistic is now the **latent excess over a matched empirical surface null**, prospectively defined rather than inspected posthoc. A successor should use something of the form:

`RITE = ITE_latent - ITE_best_matched_surface_null`.

The matched null family should include at minimum empirical joint bigrams and richer surface processes capable of matching the actual Voynich symbol-frequency concentration and transition structure. Same-key cipher positives must be shown prospectively to retain positive RITE while target-like stable non-language generators do not.

The current FIT-only posthoc median residual of +0.590616 is not a result to threshold or interpret; it only motivates such a successor.

---

## Data disposition

- H1: previously consumed; not reused in v6.1.
- FIT: consumed by v6.1 and the posthoc audits.
- C1: opened once and consumed by v6.1.
- No plaintext or per-symbol decode mapping inspected.

A future target-level confirmatory experiment will require a genuinely new prospective holdout or another independent representation/source; C1 cannot be reused as pristine confirmation.

## Compute ledger

- smoke `6a7b933b27caad61c6eac6d6` — completed.
- CAL/VAL `6a7b936f27caad61c6eac6da` — completed.
- FIT `6a7b947827caad61c6eac6f2` — completed.
- C1 `6a7b94c527caad61c6eac6f6` — completed.
- posthoc surface audit `6a7b950d27caad61c6eac6f8` — completed.
- posthoc matched-pair audit `6a7b954ff6d0f3ee953aa1eb` — completed.
