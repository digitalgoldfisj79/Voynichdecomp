# State-aware transducer result — encoded vs created — 2026-09-28

- key: `voynich_state_aware_transducer_result_20260928`
- status: active
- created: 2026-09-28T11:29:44.735152+00:00
- updated: 2026-10-10T16:27:25.988447+00:00

---

# State-aware transducer result — 2026-09-28

## RETRACTED / DOWNGRADED
- REJECTED AS RESCUE: the preregistered observable-state transducer does not recover a prospectively useful latent-to-surface mapping on held-out Stars folios.
- DOWNGRADE: local line/paragraph state does not explain enough SG64 realization to justify stripping it as an encoding layer.
- DO NOT interpret the selective-strip bigram gain as decryption evidence; matched random compression does better on average.
- SG64 realization families themselves remain real. What fails is the stronger removable-state-noise interpretation.

## Frozen design
Protocol: voynich_state_aware_transducer_protocol_20260928.
TRAIN: 10 pre-existing DISCOVERY Stars folios.
CALIB: f104v,f106r,f107r,f107v,f112v.
TEST: f104r,f105v,f106v,f111r,f114r.
TEST untouched at model/selection time.

Frozen inputs: latent family identity + line_initial/final + word-position bin + relative-line quartile + first/middle/last paragraph-line position + paragraph-number bin.
Excluded exact morphology/root/init/final, folio identity, oracle state, test labels, surface-neighbour identity.

Model:
P0(y|f)=(n_fy+.5)/(n_f+.5K)
P(y|f,s)=(n_fsy+8 P0)/(n_fs+8), lambda=8 frozen.

## Gate A/B prospective surface prediction
CALIB, n=1176 ambiguous tokens:
accuracy +0.2551 pp / folio-cluster null SD .4885 pp = +.52 SD unresolved.
log-loss benefit -.01052 bits/token / .00629 = -1.67 SD unresolved/worse.
Frozen CALIB rule selected 16 families.

TEST all ambiguous, n=1177:
accuracy gain -1.7842 pp /1.0646 = -1.68 SD unresolved/worse.
log-loss benefit -.01398 bits/token /.01565 = -.89 SD unresolved/worse.

TEST selected 16 families, n=298:
accuracy gain +.3356 pp /.7504 = +.45 SD unresolved.
log-loss benefit +.03195 bits/token /.02867 = +1.11 SD unresolved.
Selection does not prospectively replicate.

## Gate C/E held-out lexical prediction
Within-line add-.5 TRAIN->TEST bigram:
raw 10.59804 bits/token.
selective strip 10.49307.
gain +.10497 bits/token / folio-cluster null SD .05094 = +2.06 SD.

Matched generic compression:
500 deterministic random family selections preserving selected-family count and TEST token-mass decile distribution.
random mean gain +.12743 bits/token, SD .04815.
actual excess over random = -.02246 bits/token /.04815 = -.47 SD; p(random>=actual)=.645.
Therefore the apparent absolute bigram improvement is generic compression, not transducer-specific.

## Gate D TEST positional grammar
54 TEST star spans >=18; E-2 n=40.

P1:
raw -2.9584 pp /1.6234=-1.82 SD.
strip -4.9884 /1.7988=-2.77 SD.
paired change -2.0300 pp /.5476=-3.71 SD WRONG WAY.

P3:
raw +.4585 pp /3.5486=.13 SD.
strip -1.5005 /3.6680=-.41 SD.
paired change -1.9590 pp /.5430=-3.61 SD. Absolute anomaly does not shrink.

E-2:
raw +10.9324 pp /5.9580=1.84 SD.
strip +16.2048 /7.3942=2.19 SD.
paired change +5.2724 pp /4.4375=1.19 SD unresolved, magnitude grows.

0/3 anomaly magnitudes shrink; P1 has >2-SD wrong-way change.

## Audit
Circularity: family transform frozen from SG64; model/features/smoothing and TEST gate preregistered before outcomes.
Leakage: CALIB selects families, TEST untouched. Family inventory historical SG64, so not pristine unknown vocabulary.
Compression: 500 matched family-count/token-mass random collapses.
Topology: next-token metric restricted within lines.
RLS enabled on all new public tables.

## Adjudication
FAIL:
A fail; B fail; C apparent pass invalidated by E; D decisive fail; E fail.

Licensed inference:
Observable manuscript line/paragraph state is insufficient to turn frozen SG64 realization families into a prospectively useful latent-natural-language transducer. The surface realization system remains structurally real rather than removable nuisance state.

This shifts evidence further away from simple/modestly stateful latent-natural-language encoding and toward either:
1. created/formal surface system; or
2. substantially more stateful hidden transduction/cipher not represented by observable layout/paragraph state.
Does not distinguish those two and is not a universal cipher exclusion.

Tables:
vms_state_transducer_tokens_v01
vms_state_transducer_family_prior_v01
vms_state_transducer_state_count_v01
vms_state_transducer_state_total_v01
vms_state_transducer_predictions_v01
vms_state_transducer_selected_families_v01
vms_state_transducer_selective_tokens_v01
vms_state_transducer_test_entries_v01
vms_state_transducer_random_family_sets_v01
vms_state_transducer_random_bigram_v01

