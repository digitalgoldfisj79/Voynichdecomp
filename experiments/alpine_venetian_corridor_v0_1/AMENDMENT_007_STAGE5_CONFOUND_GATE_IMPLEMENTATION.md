# Amendment 007 — Stage 5 final visual corpus and confound-gate implementation

Date: 2026-08-08
Programme: Alpine–Venetian Corridor v0.1
Run: `corridor_v01_20260808_run01`

## Timing / firewall

This amendment is frozen before any corridor-to-Voynich similarity has been computed or inspected. At freeze time `corridor_scores` contains zero rows and the run flag `vms_similarity_computed` is false.

Amendment 006 remains controlling for evidential hierarchy: image/description/geometry are secondary/exploratory; primary inference is codicology + dated/localised palaeography, and documentary nodes without cross-node edges score zero.

## Final visual object corpus

The Stage 4 QA decisions have been reconstructed into Supabase from the already sealed QA job logs. No object was re-triaged or promoted after outcome inspection.

Final accepted visual corpus before similarity:

- 78 QA-passed crops
- 15 manuscripts
- 40 crops / 9 `corridor_core` manuscripts
- 31 crops / 3 `control_lombardy` manuscripts
- 7 crops / 3 `control_bavaria_swabia` manuscripts

Canonical manifest SHA-256:

`63a1dd441c11b92f7851d367c7cf58aca1a446e1b5ae25c1c9e93ee436504669`

Missing manuscripts/classes remain missing. Pizzigano 1424 has no QA-passed local object and is not repaired by hand selection.

## Confound-gate population

The frozen protocol requires a manuscript/institution classifier on the final normalised image representations under grouped cross-validation before the pixel embedding arm can support H1.

To make `grouped cross-validation` operational without page leakage, the primary confound test uses only manuscripts with at least **two distinct source-page URLs** among QA-passed crops. This yields:

- 59 crops
- 10 manuscripts

Five one-page witnesses (19 crops total) are excluded from the confound-estimation subset only because manuscript identity cannot be evaluated with a held-out page when only one page exists. They remain in the sealed 78-object visual corpus if and only if a representation survives the confound gate.

## Frozen representations

Backbone is fixed by `config.json`:

`facebook/dinov3-vit7b16-pretrain-lvd1689m`

No finetuning.

Three frozen crop variants are evaluated independently:

1. `rgb_norm_v1`: tight source crop, RGB conversion, fixed square white padding, processor resize/normalisation.
2. `gray_bgdiv_v1`: grayscale crop; deterministic low-frequency background estimate by Gaussian blur; intensity divided by background and clipped; replicated to RGB; fixed square white padding.
3. `inkmask_v1`: thresholded dark-ink residual derived from the same background-divided grayscale crop; black foreground on white; fixed square white padding.

No full-page RGB embedding is used.

## Primary grouped classifier

Target: manuscript/candidate identity.

Group: exact source-page URL.

Classifier: L2 multinomial logistic regression (`C=1`, `class_weight='balanced'`, fixed seed 20260808) on frozen DINOv3 embeddings, with train-fold standardisation only.

Cross-validation: leave-one-source-page-out. Each held-out crop is predicted only by a classifier trained without every crop from that source page. Manuscripts with only one source page are absent from this gate by construction, so every class remains represented in training.

Primary metric: macro one-vs-rest ROC AUC computed once over all out-of-fold predictions. The class ordering is frozen globally and all folds use the same ordering.

A crop-random stratified 5-fold AUC is reported only as a leakage-prone diagnostic; it cannot be used to pass a representation.

## Gate decision

The protocol thresholds apply **per representation**:

- AUC `<= 0.65`: PASS; representation may enter exploratory image similarity.
- `0.65 < AUC <= 0.70`: CAUTION; representation may enter only with the mandatory institution/manuscript sensitivities.
- AUC `> 0.70`: FAIL; that representation is excluded from VMS image similarity.

If all three representations fail, the entire DINO/pixel family is excluded. A passing representation does not rescue a failing representation.

This representation-specific rule is stricter and more informative than averaging AUCs across variants and prevents a low-confound transform from concealing a high-confound one.

## Acquisition failure rule

Image acquisition retries are bounded. If fewer than 80% of the 59 frozen confound crops are successfully acquired, or fewer than 6 manuscripts remain with at least two distinct acquired source pages, the confound gate is `NONRESOLVING` and the pixel family cannot support any inference until the acquisition problem is repaired without changing object selection.

## No target access

The confound job contains no VMS images, VMS embeddings, corridor/control labels, production places, or similarity computation. Only after this job is complete and its decisions are recorded may any surviving representation be compared with the frozen VMS reference objects.
