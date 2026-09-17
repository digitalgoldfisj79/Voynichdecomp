# Voynich crown corpus

Zero-paid-API pipeline for the 1390–1440 crown control experiment.

Pipeline: Warburg advanced search -> object/IIIF manifest harvest -> image download -> CPU zero-shot crown detection -> provenance-blind crown crops -> A0 contact sheets -> artifact upload.

Known discussed comparators (Voynich zodiac crowns, Rudolf IV, Morgan M.853, Bellifortis K 465, Taccola Pal.766) are excluded from the null corpus and must be handled as external controls/confounds.

Primary morphology is frozen before scoring:
- S = triangular lower-circlet points whose sides themselves are stepped/serrated/wavy
- P = simple triangular points without S
- A = one dominant overhead longitudinal arch/hoop
- X = terminal cross at the apex
- TF = trefoil/fleur-de-lis terminals

Primary endpoint: S+A+X. Primary independent unit: manuscript/workshop cluster, not image.

No paid FAL/HF inference is used in the first workflow. GroundingDINO runs on the GitHub-hosted CPU and all detector failures are retained in the manifest for recall audit.