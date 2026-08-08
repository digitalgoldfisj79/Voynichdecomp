# Stage 5 confound-gate result

Date: 2026-08-08
Run: `corridor_v01_20260808_run01`
Job: `6a776d6ada2af92a634ef9ec`
Backbone: `facebook/dinov3-vit7b16-pretrain-lvd1689m`

## Acquisition

All 59 frozen confound crops were acquired successfully from all 10 eligible manuscripts. There were zero image-acquisition errors. The acquisition gate therefore passed.

## Page-held-out manuscript identification

Primary metric is macro one-vs-rest ROC AUC under leave-one-source-page-out prediction, as frozen in Amendment 007.

| Representation | Macro OVR AUC | Top-1 accuracy | Gate |
|---|---:|---:|---|
| `rgb_norm_v1` | 0.8460122517 | 0.6779661017 | **FAIL** |
| `gray_bgdiv_v1` | 0.7906096373 | 0.6271186441 | **FAIL** |
| `inkmask_v1` | 0.7905697030 | 0.5084745763 | **FAIL** |

Frozen thresholds:

- PASS: AUC <= 0.65
- CAUTION: 0.65 < AUC <= 0.70
- FAIL: AUC > 0.70

All three representations fail.

The leakage-prone crop-random two-fold diagnostics were, respectively, 0.86535, 0.81999 and 0.82745; these are diagnostic only and have no role in the gate decision.

## Decision

**DINO/pixel family: EXCLUDED.**

No DINO corridor-to-Voynich similarity may be computed or inspected under this run. The result means that even tightly cropped and aggressively normalised DINOv3 representations retain enough manuscript/acquisition identity to violate the preregistered confound threshold.

This is a pre-target exclusion. At the time of this result, `vms_similarity_computed=false`; no VMS DINO similarity has been generated.

Stage 5 may continue only with the independently defined non-pixel families (blind text-description and explicit geometry/morphology), plus the separately governed codicological/palaeographic and documentary-edge arms under Amendment 006.
