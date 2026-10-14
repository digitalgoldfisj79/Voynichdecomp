# Causal hidden-state ciphertext transducer — result — 2026-09-28

- key: `voynich_hidden_state_transducer_result_20260928`
- status: active
- created: 2026-09-28T11:58:51.960285+00:00
- updated: 2026-10-10T16:27:25.988447+00:00

---

# Causal hidden-state ciphertext transducer — result — 2026-09-28

## RETRACTED / DOWNGRADED
- REJECTED AS A RESCUE: a causal finite-state transducer whose hidden state is inferred prospectively from past ciphertext does not turn the frozen SG64 realization families into a latent lexical/plaintext layer.
- DO NOT promote the isolated TEST accuracy gain (+2.11 null SD). The same model worsens probabilistic log loss, its CALIB-selected families do not replicate, downstream lexical gain is indistinguishable from random compression, and positional grammar worsens.
- DOWNGRADE: SG64 family variation is not explained as ordinary finite-state encoder noise recoverable from ciphertext history at K=2,3,4,6,8.
- SG64 realization families remain real; the failed claim is that they are removable emissions around a simpler lexical sequence under this causal HMM representation.

FROZEN PROTOCOL
Canonical preregistration: voynich_hidden_state_transducer_protocol_20260928.
Model z_t~A(z_t|z_t-1); y_t~B(y_t|f_t,z_t), where f_t is frozen SG64 family and y_t exact surface token.
State inferred causally from earlier ciphertext only; no layout/paragraph/folio/morphology/root/current-future token/oracle/test features.
TRAIN 10 discovery Stars folios; CALIB f104v,f106r,f107r,f107v,f112v; TEST f104r,f105v,f106v,f111r,f114r.
K={2,3,4,6,8}; Baum-Welch MAP TRAIN-only, five deterministic starts/K, emission shrinkage lambda=8, transition/initial pseudocount 1.

CALIB
Every K worsens family-only causal log loss.
K8 least bad: state 2.01390 vs baseline 1.99736; benefit -.01654 bits/token = -2.08 SD.
K4 -.01909=-2.11 SD; K6 -.02053=-2.09; K3 -.02513=-2.14; K2 -.04492=-2.14.
K8 exact accuracy gain +.935 pp = +1.68 SD unresolved.
Frozen CALIB rule selected 13 families: chal,cham,cheal,cheam,lkain,lkal,lkam,ltal,qokady,qokchdy,qotedy,shal,sheal.

SEALED TEST ALL AMBIGUOUS
n=1177.
Exact accuracy 47.9184% ->49.3628%; gain +1.4444 pp / null SD .6850 = +2.11 SD.
Probabilistic log loss 1.98234 ->2.00317; benefit -.02083 bits/token / .01045 = -1.99 SD, wrong direction.
Thus isolated hard-decision positive is contradicted by proper scoring rule.

SEALED TEST SELECTED 13
n=189.
Accuracy 52.9101% ->52.9101%, gain 0.
Log-loss benefit -.01759 / .01333 = -1.32 SD unresolved/worse.
Family-level CALIB predictability does not replicate.

STATE DIAGNOSTICS
K8 mean self-transition .56124; mean transition entropy 2.22070 bits (max3).
TEST posterior occupancy nearly uniform [.1233,.1248,.1264,.1243,.1222,.1259,.1289,.1243].
MAP states 0,1,4 never dominate any TEST folio; weakly differentiated state solution.
Paragraph-reset sensitivity does not rescue: TEST log-loss benefit -.02083 = -1.99 SD; accuracy +1.02 pp =1.70 SD unresolved.

DOWNSTREAM BIGRAM
TRAIN->TEST within-line add-.5:
raw 10.59804; selective strip 10.51203; gain +.08601 bits/token / .05014 = +1.72 SD unresolved.
500 matched random family collapses: mean gain +.08860, SD .04213.
Actual-minus-random -.00259/.04213=-.06 SD; p(random>=actual)=.479.
No transducer-specific lexical gain.

POSITIONAL TEST
P1 raw -2.9584 pp -> strip -3.5798; paired change -.6213 pp /.2871=-2.16 SD WRONG WAY.
P3 raw +.4585 -> +1.6892; change +1.2306/1.8969=.65 SD unresolved; magnitude grows.
E-2 raw +10.9324 -> +12.7663; change +1.8339/2.5140=.73 SD unresolved; magnitude grows.
0/3 anomaly magnitudes shrink.

ADJUDICATION
A hard accuracy nominal pass; B fail; C fail; D decisive fail; E decisive fail.
Preregistered decision: FAIL.

LICENSED CONCLUSION
The tested evidence is positively inconsistent with Voynich surface variation being a modest removable encoding layer around an ordinary lexical sequence. Surface grammar behaves as part of the mechanism.
Remaining broad classes:
A created/meaning-bearing formal surface system;
B much stronger hidden transduction/cipher with state not recoverable from finite causal ciphertext history in the frozen SG64 representation.
B is not excluded, but is increasingly less parsimonious.

RESEARCH STOP
Do not continue by arbitrarily increasing K, adding languages, or inventing post-hoc families. Future text-only work should require genuinely new information/mechanism/crib.
Working model: deliberately generated, meaning-capable surface system with grammar intrinsic to production mechanism; conventional/modestly stateful encoded-natural-language models substantially weakened.

