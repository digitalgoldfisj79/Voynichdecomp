# Stage 5 closeout — Alpine–Venetian Corridor v0.1

Date: 2026-08-08
Run: `corridor_v01_20260808_run01`
Run UUID: `523d7ffe-dafd-46cb-ad5b-165f6fcea367`

## Verdict

**NONRESOLVING / UNDERPOWERED FOR H1.**

This is not a rejection of the Alpine–Venetian corridor hypothesis and not positive evidence for it. The confirmatory run terminates because the frozen inferential instruments do not survive their own gates.

No valid corridor-to-Voynich similarity score was generated in this run.

## 1. Final sealed visual corpus

The Stage 4 audit state was reconstructed from already sealed QA logs before any target similarity.

Final QA-passed visual corpus:

- 78 object crops;
- 15 manuscripts;
- 40 crops from 9 corridor-core manuscripts;
- 31 crops from 3 Lombardy controls;
- 7 crops from 3 Bavaria/Swabia controls.

Canonical manifest SHA-256:

`63a1dd441c11b92f7851d367c7cf58aca1a446e1b5ae25c1c9e93ee436504669`

Missing objects/manuscripts remained missing; no post-outcome hand selection was used.

## 2. Pixel/DINO family — excluded by the pre-target confound gate

Fixed backbone:

`facebook/dinov3-vit7b16-pretrain-lvd1689m`

Confound-estimation subset:

- 59 frozen crops;
- 10 manuscripts with at least two independent source pages;
- 59/59 crops acquired;
- zero acquisition failures;
- manuscript identity classified under leave-one-source-page-out prediction.

Results:

| Representation | Page-held-out macro OVR AUC | Top-1 | Frozen decision |
|---|---:|---:|---|
| `rgb_norm_v1` | 0.8460122517 | 0.6779661017 | FAIL |
| `gray_bgdiv_v1` | 0.7906096373 | 0.6271186441 | FAIL |
| `inkmask_v1` | 0.7905697030 | 0.5084745763 | FAIL |

Frozen failure threshold: AUC > 0.70.

**Decision:** all three representations fail. The DINO/pixel family is excluded. No DINO-to-Voynich similarity was computed.

Successful confound job: `6a776d6ada2af92a634ef9ec`.

## 3. Blind-description / geometry families — not promoted to a confirmatory result

The legacy description field was found pre-similarity to be pipeline-confounded: comparator descriptions in important control structural classes were often the placeholder `neutral morphology only`, whereas corridor map-recovery objects had substantive morphology descriptions.

A uniform identity-blind re-description repair was preregistered in Amendment 010. A Qwen2.5-VL-7B job (`6a777440da2af92a634efa6b`) was launched, then deliberately cancelled while it was still downloading model weights after the Stage 5 stopping logic was resolved. It produced **zero descriptions** and therefore exposed no target-facing result.

The repair was not continued because the original three-representation visual convergence rule cannot be satisfied once the DINO family is excluded: at most two visual representation families remain. Visual work is secondary/exploratory under Amendment 006 and cannot substitute for the missing primary palaeographic/codicological instrument.

Amendment 008's temporary class-specific hard-stop argument was retracted pre-target by Amendment 009; it is not part of this verdict.

## 4. Primary palaeographic arm — no scientifically unlocked instrument

Amendment 006 promotes like-for-like codicology plus dated/localised palaeography to the primary inferential role.

The repository/history audit found only:

`experiments/blind_stroke_palaeography_v1/results/V1_5_1_PREFLIGHT_RESULT.md`

for the v1.5.1 SAGHOG / Historical-WI instrument.

That result explicitly states:

- purpose: implementation/end-to-end preflight, not confirmatory external calibration;
- terminal mAP 0.299835;
- acquisition-nuisance mAP 0.572803;
- miniature training/permutation settings;
- scientific gates not passed;
- a full external Historical-WI run remains required;
- **no Voynich phase is unlocked**.

A recursive repository-tree audit found no later full Historical-WI/HisFrag20 calibration result in that programme directory. Therefore using the preflight to score Candus, Fontana or any control hand would violate the palaeography programme's own stopping gate.

No new palaeographic model is introduced in Stage 5 after seeing the DINO confound result.

## 5. Codicology — anchor information exists, but no independent score matrix

Useful anchors remain real:

- Bodleian Canon. Misc. 554: parchment; Padua; Candus; dated 20 February 1435; authenticated Candus writing only.
- Vat.lat.4082 / Petrus de Fita: secure Padua comparison hand for Hand II ff. 47–246, 1401–02, but paper and pre-window; excluded from the primary anchor set.

However, the run contains no independent codicological feature table/matrix for variables such as ruling, quire construction, prickings, page geometry, mise-en-page, ink/pen metrics, parchment preparation or comparable material features. The available date/substrate fields were already used in candidate selection/matching and cannot be recycled as an independent affinity result.

Therefore Canon. Misc. 554 is a high-quality **anchor**, not evidence of corridor enrichment by itself.

## 6. Documentary/prosopographic arm — zero qualifying edges

Verified person/manuscript nodes include Candus, Petrus de Fita, Franciscus Squaranus, Lambertus Nerden Croyl de Almania, Math. de Almania bassa and Wilhelmus Gherardi de Gouda.

Under Amendment 006, nodes without a qualifying cross-node edge score zero. The current database contains verified scribe/copyist links to individual manuscripts but no same-hand/person/exemplar/patron/workshop/teacher-pupil edge satisfying the frozen documentary criterion.

Ordinary movement into Venice, German origin, Padua affiliation, or present Trento holding carries zero positive weight.

**Documentary score: zero / no qualifying edge found.**

## 7. What the run establishes

The run establishes several robust negative/methodological facts:

1. DINOv3 crop embeddings remain strongly manuscript/acquisition-confounded even after greyscale/background division and ink masking in this corpus.
2. The previously available description records are not uniformly generated and cannot be used as a clean confirmatory text family without repair.
3. The existing stroke-palaeography programme has not yet passed the external calibration gate required to compare Voynich with newly identified dated hands.
4. The current codicological registry is insufficient for an independent material-affinity score.
5. No qualifying prosopographic cross-node edge has yet been established.

## 8. Scientific interpretation

The correct status is **NONRESOLVING**, not Tier 0.

Tier 0 would require a valid tested affinity instrument showing no enrichment. Here the main valid image instrument was excluded before target comparison, and the primary palaeographic/codicological instrument is not yet scientifically available.

The programme therefore neither supports nor rejects H1. It does, however, identify exactly what evidence would make a new run resolvable:

- a fully externally calibrated acquisition-robust palaeographic instrument, followed by preregistered comparison of authenticated dated/localised hands such as Candus against like-for-like controls; and/or
- an independently collected codicological feature matrix not reusing selection variables; and/or
- a verified documentary cross-node edge.

Those are new instrument/data-acquisition programmes and should be preregistered as a new run rather than added post hoc to v0.1.

## Compute closeout

The Qwen description repair job was cancelled before inference output. No Hugging Face job was left running at Stage 5 closeout.
