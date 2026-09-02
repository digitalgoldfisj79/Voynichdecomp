# VBM v13 — Shared e-Operator Ciphertext Test closeout

Date: 2026-09-02
Job: `6a97cda10718b0f6d890ff69`
Hardware: HF `cpu-upgrade`
Status: **COMPLETED**
Protocol: `VBM_V13_E_OPERATOR_PROTOCOL.md`

## Frozen verdict

`V13_NO_SHARED_E_OPERATOR_EVIDENCE`

The synthetic method qualification passed strongly, but the Voynich ciphertext failed the preregistered shared-operator gate.

No plaintext or language likelihood was used. H1/C1 remained excluded. No GPU was used.

## Synthetic qualification

Known-answer `SHARED_OPERATOR` calibration:

- 6/6 replicates had median DELTA > 0 and empirical p = `0.000999`;
- z ranged from `2.8275` to `3.4699`.

Matched `IDIOSYNCRATIC_OPERATOR` controls:

- 0/6 false positives at p <= .01;
- all observed median DELTA values were negative.

Therefore the method qualified exactly as preregistered and had demonstrated power to distinguish a shared repeated operator from equally sized skeleton-specific changes.

## Voynich eligibility

Using the unchanged Joachim-exact Q0b parser and V11 exclusions:

- eligible TRAIN nuclei: 148;
- TRAIN non-empty nucleus events: 24,292;
- INTERNAL_HOLDOUT non-empty nucleus events: 6,687;
- eligible adjacent e-ladder edges: 20;
- untouched-evaluable edges: 18;
- untouched-evaluable skeletons: 16.

The minimum-data gate (>=12 edges, >=8 skeletons) therefore passed comfortably.

## Primary cross-skeleton / held-out result

For each evaluable skeleton, the multiplicative e-operator was estimated from all *other* TRAIN skeletons and applied to that skeleton's INTERNAL_HOLDOUT source-level context distribution. It was compared with the identity prediction that the next e-level simply retains the same context distribution.

Primary result:

- median skeleton DELTA: `-0.00230725`;
- sign-flip null median: `-0.00390776`;
- null mean: `-0.00432467`;
- null SD: `0.00357731`;
- z = `0.56395`;
- empirical p = `0.32077`.

Thus the estimated shared operator did **not** improve held-out next-level prediction over simple e-ladder similarity. The point estimate was itself negative.

## Independent TRAIN-half transfer

The two preregistered independent transfer directions were both negative:

- SKEL-A -> SKEL-B median DELTA: `-0.00526978`;
- SKEL-B -> SKEL-A median DELTA: `-0.00790726`.

This independently fails the requirement that the same directional transformation transfer across different sets of skeletons.

## Skeleton heterogeneity

The failure is not because every skeleton is null. Some individual skeletons showed positive directional effects, including approximately:

- `E`: +0.01553;
- `Ed`: +0.01532;
- `Ek`: +0.02439;
- `Eod`: +0.02489.

Others showed effects in the opposite direction, including:

- `kEod`: -0.01521;
- `lkEd`: -0.02073;
- `tE`: -0.01809;
- `tEo`: -0.02009.

This mixture is consistent with the V11 result that e-ladder relatives share contextual family resemblance, while contradicting the stronger claim that one common directed transformation is repeatedly applied across skeletons.

## Repeated-application diagnostic

Only two qualifying m -> m+2 cases existed. One (`E`) improved under two applications of the estimated operator; the other (`kE`) did not. This diagnostic was non-binding and too small to alter the primary conclusion.

## Interpretation

V13 narrows the V11/V12 picture.

V11-B remains a strong result: nuclei differing mainly in e multiplicity have unusually similar boundary contexts. However V13 shows that this similarity does not behave like one stable cross-skeleton multiplicative operator of the tested form. V12's global e-permutation operator was therefore not merely an imperfect implementation of an obvious shared ciphertext transformation; the ciphertext does not presently support the stronger shared-operator premise itself.

The defensible conclusion is:

> `e`-family variation is structured and context-preserving, but its effect appears skeleton-dependent rather than a single reusable repeated operator shared across nucleus families.

This result does not negate bridge-half factorisation from V11-C, which remains independently supported.

## Stopping rule

Per preregistration:

- synthetic qualification: **PASS**;
- data-quantity gate: **PASS**;
- primary median DELTA > 0: **FAIL**;
- p <= .01 and z >= 2.5: **FAIL**;
- A->B transfer > 0: **FAIL**;
- B->A transfer > 0: **FAIL**.

Therefore no V14 nucleus transducer is authorised from the shared-e-operator line. Any continuation would require a new independently motivated hypothesis, most plausibly one in which e-family effects are conditional on nucleus skeleton/morphological class rather than globally shared.
