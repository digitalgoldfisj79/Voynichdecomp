# VMS edge forensics Stage F v0.1 — CLOSEOUT

Date: 2026-09-07
Protocol: `vms_edge_forensics_orientation_20260907_v01`
Physical addendum: `vms_edge_physical_image_20260907_v01`

## RETRACTIONS / BOUNDS — READ FIRST

1. **ZERO residual text edges are promoted to `CANDIDATE_PRODUCTION_NEIGHBOUR` in v0.1.** The five pairwise relations remain, at most, text-state neighbours / contextual codicological hypotheses.
2. **Stage O direction/orientation remains SEALED and is not run.** The preregistered condition `>=1 Stage-F edge promoted` was not met.
3. **E1's strong upper-profile image result is NOT a positive.** It fails the preregistered scalar-confirmation requirement and therefore cannot be rescued by post-hoc interpretation or detector retuning.
4. **E3 is demoted by F1 deletion robustness.** It cannot be rescued by physical evidence in this protocol version.
5. Pelling's independent f84v↔f78r visual proposal coincides with E5, but under the preregistration it is visual/layout context rather than a calibrated material-contact discriminator; it does not promote E5.
6. Existing paint offsets predominantly document later/current contact states and are not treated as original production adjacency.
7. No threshold, feature family, image region, null hierarchy, or candidate set is changed after target opening.

# Frozen candidates

- E1 `q01_b3_6` ↔ `q03_b17_24` = (f3,f6) ↔ (f17,f24), residual 12/12.
- E2 `q06_b42_47` ↔ `q01_b1_8` = (f42,f47) ↔ (f1,f8), residual 12/12.
- E3 `q05_b36_37` ↔ `q03_b19_22` = (f36,f37) ↔ (f19,f22), residual 7/12.
- E4 `q13_b76_83` ↔ `q13_b77_82` = (f76,f83) ↔ (f77,f82), residual 12/12.
- E5 `q13_b75_84` ↔ `q13_b78_81` = (f75,f84) ↔ (f78,f81), residual 10/12.

# F1 deletion robustness

- E1: PASS; family deletion 3/3; transcription deletion 4/4.
- E2: PASS; family deletion 3/3; transcription deletion 4/4.
- E3: **FAIL**; family deletion 1/3; transcription deletion 1/4.
- E4: PASS; family deletion 3/3; transcription deletion 4/4.
- E5: PASS; family deletion 3/3; transcription deletion 4/4.

# F2 held-out text validation

No edge passes the frozen rule `closeness >=2 matched-null SD AND exact one-sided p<=.01`.

## E1
RARE observed 0.0722203 versus null 0.0199266 ± 0.0147644; effect +0.0522937 = **3.5419 null SD**, exact p=.0104712. **FAIL by the frozen p<=.01 criterion; do not round.**
LAYOUT effect −0.092618 against null SD 0.0658614 = **1.406 SD**; **the metric does not resolve this**.

## E2
RARE effect +0.0125127 against null SD 0.0145814 = **0.858 SD**; **the metric does not resolve this**.
LAYOUT effect −0.0727115 against null SD 0.139898 = **0.520 SD**; **the metric does not resolve this**.

## E3
RARE = 0.035 SD; LAYOUT = 0.544 SD. **The metrics do not resolve this.** F1 already fails.

## E4
RARE effect +0.0354748 against null SD 0.0156069 = **2.273 SD**, exact p=.08108; FAIL.
LAYOUT is in the wrong direction; FAIL.

## E5
RARE effect +0.0431237 against null SD 0.0156327 = **2.759 SD**, exact p=.05405; FAIL.
LAYOUT is in the wrong direction; FAIL.

Stage-F text workflow: GitHub Actions run `34156702811`; artifact digest `sha256:507d75b4e8338ae626ab22366ae8f24ae10bc472bca5e88cb130be7cc1d6b806`.

# Documentary / codicological audit

A versioned ledger is stored in `public.vms_edge_forensics_evidence_v01`.

Key adjudications:
- The catastrophic upper spill supplies chronology but no documented candidate-specific E1/E2 continuity.
- f17r's previously suggested flower-like contact imprint is not licensed as transfer evidence; later reassessment favours a drawing/sketch interpretation.
- f16v↔f17r, f46v↔f47r and many other offsets record later/current contact states, not original production neighbourhoods.
- f1r is heavily exposed/damaged and therefore a prospective image confound; this was handled by generic leave-one-surface-out testing rather than hand-exclusion.
- f8 and f47 were among physically sampled leaves and both are calf parchment, but species identity is manuscript-wide and does not identify common skin/batch/neighbourhood.
- E4/E5 endpoints contain strong internal bifolium cross-fold evidence, but that establishes functional bifolium status, not inter-bifolium adjacency.
- Pelling independently proposed f84v facing f78r, exactly the E5 pair, from visual/layout symmetry and a repeated motif. This is notable contextual corroboration but not a calibrated material-contact discriminator under the frozen rules.
- Stolfi notes a competing physical-shape affinity between `q13_b77_82` and `q13_b78_81`; this prevents using visual congruence as a clean E4/E5 discriminator.

# F3/F4 quantitative physical-image audit

The physical addendum was frozen before candidate ranks and reused the qualified S3 upper-edge detector unchanged over all eligible complete Q1–Q7 bifolia. All 110 page sides were fetched from Yale IIIF and hashed. Pair scores were written under opaque IDs before the reveal map.

GitHub Actions run: `34156996206`.
Artifact digest: `sha256:379567d9dfc71bde7ab5d2b74419d7068fcac0893f7615ed14e7721768537107`.

## E1 (f3/f6 ↔ f17/f24)
Matched null: same Currier + hand + illustration class (`A_LHI`), n=153 endpoint-disjoint alternatives.

- `TOP_SPECIFIC_PROFILE`: observed −4.92510 versus null −0.850966 ± 1.247222; effect **−4.074138 = 3.2666 null SD closer**, exact lower-tail p=.0064935. This component passes its individual threshold.
- `TOP_SPECIFIC_SCALAR`: observed −0.350007 versus null −0.386281 ± 0.723622; effect **+0.036274**, equivalent closeness **−0.050 SD**. **The metric does not resolve this and the direction does not confirm the profile channel.**
- raw bottom-edge distance is much *larger*, not spuriously closer: observed 6.42333 versus null 2.36760 ± 1.20439; closeness z=−3.367.
- leave-one-surface-out: all 8 deletions preserve the profile-effect sign; 7/8 remain >=2 SD. The one failure is deletion of f6r, leaving 0.629 SD.

Frozen composite decision: **`P2_QUANT_SIGNAL=False`** because scalar confirmation is mandatory. No retuning is allowed.

## E2 (f42/f47 ↔ f1/f8)
Matched null: same Currier + hand (`B_LH`), n=171 alternatives.

- `TOP_SPECIFIC_PROFILE`: effect −2.327495 against null SD 1.612093 = **1.4438 SD**, p=.1686. **The metric does not resolve this.**
- `TOP_SPECIFIC_SCALAR`: effect −0.201885 against null SD 0.742579 = **0.272 SD**. **The metric does not resolve this.**
- leave-one-surface-out is unstable: one deletion reverses direction and only 1/8 remains >=2 SD.

Frozen decision: **`P2_QUANT_SIGNAL=False`**.

# F5 promotion adjudication

The frozen rule requires:
1. `TEXT_ROBUST`;
2. held-out text pass OR independent high-confidence documentary physical observation;
3. an independent discriminating physical/codicological family positive;
4. no hard contradiction.

Adjudication:
- E1: condition 1 passes; condition 2 fails (RARE p=.01047 misses the frozen .01 gate; no high-confidence documentary physical adjacency); quantitative P2 composite also fails. **NOT PROMOTED.**
- E2: condition 1 passes; condition 2 fails; P2 fails. **NOT PROMOTED.**
- E3: condition 1 fails. **NOT PROMOTED.**
- E4: condition 1 passes; held-out p fails; internal cross-fold evidence is non-discriminating for E4 and a competing physical affinity exists. **NOT PROMOTED.**
- E5: condition 1 passes; held-out p fails; Pelling's exact visual proposal is contextual but not a calibrated material discriminator, and competing physical-shape affinity exists. **NOT PROMOTED.**

**Stage-F promoted edge count = 0/5.**

# Stage O stop rule

The preregistered trigger for orientation is not met.

> `DIRECTION INFERENCE NOT OPENED — ZERO STAGE-F PRODUCTION-NEIGHBOUR EDGES PROMOTED.`

No D1/D2 target scores are computed. No present folio numbering is used as directional truth. No orientation model is calibrated or opened on the Voynich target in this protocol version.

# Licensed result

The surviving pairs remain evidence for **bifolium-level production/text state similarity**, not demonstrated physical adjacency or chronological order.

The strongest unresolved lead is E1: it is highly stable textually and shows a strong upper-edge profile compatibility, but the independently frozen scalar physical channel does not confirm it and the held-out text p-value misses the preregistered gate by 0.000471. These are reasons to preserve E1 as a targeted future conservation-measurement candidate, **not reasons to weaken the decision rule**.

Future reopening requires genuinely new independent physical data or a predeclared new instrument, e.g. measured parchment thickness/shape, high-confidence hair/flesh mapping, animal-individual proteomic/DNA assignment, systematic calibrated offset atlas, or multispectral/material cues. A post-hoc retune of the present text/image scores is explicitly prohibited.
