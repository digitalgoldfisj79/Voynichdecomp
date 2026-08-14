# U6-v0.2 — external writer-sensitive local visual instrument

Date frozen: 2026-08-14
Status: **NEW INSTRUMENT PREREGISTRATION — VOYNICH TARGET SEALED**

## Relation to VTPS v0.1

VTPS v0.1 is permanently closed under its DINOv3 instrument: after its one permitted internal repair, nuisance false-positive rates exceeded 5% and maximum physical-source detection at beta<=0.50 remained 34% (midfix) / 41% (suffix), far below the 80% gate. No v0.1 threshold, nuisance setting or DINO component is reopened here.

U6-v0.2 is a new independent instrument. It does not modify a failed v0.1 parameter. Its purpose is to test whether an externally writer-sensitive local handwriting representation can first be demonstrated on manuscripts with known writer identity and then, only if qualified, substituted into the already-frozen VTPS synthetic-coupling/null framework.

## External dataset

Primary external qualification corpus: **ScriptNet / ICDAR2017 Historical Writer Identification training-binarized set**, Zenodo DOI `10.5281/zenodo.1324999`, file `icdar17-historicalwi-training-binarized.zip`, published MD5 `3caa11320e857cf80694031a568ce6bf` and size approximately 64.4 MB.

The filename prefix before the first hyphen is the writer identifier. Distinct images of the same writer are treated as separate pages. No Voynich image, Davis hand label or Voynich text feature participates in external training or qualification.

The broader benchmark is explicitly a historical writer-identification corpus spanning 13th–20th-century pages; each benchmark writer contributed multiple pages. This is used only as external calibration, not as a historical comparandum for Voynich provenance.

## Writer split firewall

Writer IDs, not pages, are split deterministically before model fitting:

`bucket = int(SHA256('u6v02-writer|' + writer_id)[:8], 16) mod 10`

- buckets 0–5: TRAIN writers;
- buckets 6–7: CALIBRATION writers;
- buckets 8–9: LOCKED writers.

No writer appears in more than one partition. The locked-writer labels are used only once for the final external AUC calculation.

If fewer than 30 writers or fewer than 60 pages survive in any split, the instrument returns `FAIL_EXTERNAL_DATA_GATE` without any Voynich operation.

## Image representation

The external input is the published binarized image, decoded as foreground ink on a white field. Preprocessing is fixed:

1. convert to single-channel float ink mask in [0,1]; infer polarity from border versus central intensity and invert when necessary;
2. remove a 2% outer frame from every side;
3. resize preserving aspect ratio so median connected-component height on the page is 18 pixels when the statistic is resolvable; otherwise resize page height to 1200 pixels;
4. sample deterministic local ink-bearing windows of 96×320 pixels; a window is admissible when ink fraction is in [0.02,0.35];
5. resize each admitted window to 224×224 for the encoder; foreground remains the only channel, repeated three times.

The use of local windows rather than full pages is mandatory: it is intended to prevent page margins, parchment, scanning protocol and overall layout from serving as the writer signal and to move the external calibration closer to the local-word scale required by VTPS.

Each page contributes at most 12 windows selected from a deterministic SHA-derived candidate sequence. No window coordinates are selected using writer-classification accuracy.

## Encoder

Fixed architecture: torchvision `resnet18` with ImageNet-1K V1 weights as initialization. The final classifier is replaced by:

- 512 -> 128 linear embedding;
- L2 normalisation;
- training-only linear writer classifier over TRAIN writer IDs.

Optimisation:

- cross-entropy on TRAIN writer identities;
- AdamW, lr 1e-4, weight_decay 1e-4;
- batch size 64;
- fixed augmentations: random translation <=4 px, rotation <=2 degrees, and one binary dilation-or-erosion operation with probability 0.25; no colour augmentation;
- epochs 1..5 only;
- all ResNet parameters trainable;
- seed 20260814.

Epoch is selected once by maximum open-set same-writer/different-writer AUC on CALIBRATION writers; ties choose the earliest epoch. No hyperparameter other than epoch is selected from calibration.

## External open-set test

For every CALIBRATION/LOCKED page, page embedding is the L2-normalised arithmetic mean of its admitted local-window embeddings. The primary comparison deliberately requires distinct pages:

- positive pair: two different pages with the same writer prefix;
- negative pair: pages from different writer prefixes.

All available same-writer page pairs are used. Negative pairs are drawn deterministically to equal the positive-pair count and are frequency-matched on the number of admitted windows to within two windows where possible. Pair score is cosine similarity.

Primary external statistic: ROC AUC on **LOCKED writers**.

### External gate

The umbrella Frontier threshold is unchanged:

- **PASS_EXTERNAL** iff locked-writer AUC >= 0.80;
- otherwise `FAIL_EXTERNAL_AUC` and no Voynich crop or embedding may be opened under U6-v0.2.

Report calibration AUC, locked AUC, writer/page/window counts, and same-vs-different cosine distributions. Page-level top-1 writer classification is descriptive only and cannot rescue a failed AUC.

## Cross-domain nuisance checks

Before Stage B, perform the following on LOCKED external pages using the frozen selected encoder:

- `BACKGROUND_ONLY`: blank all foreground ink inside every local window while retaining its spatial support; same-writer AUC must be <=0.60;
- `PATCH_ORDER`: page embedding is unchanged under deterministic permutation of local-window order (exact numerical check);
- `IMAGE_ID`: no filename substring other than writer prefix is present in model input or features (code audit).

Failure of BACKGROUND_ONLY yields `FAIL_EXTERNAL_NUISANCE` even if writer AUC passes.

## Stage B — inherited VTPS synthetic qualification

Only after `PASS_EXTERNAL` and nuisance qualification may the encoder be applied to the already-located Voynich word-crop asset. Stage B **still does not open true retention labels**.

Stage B replaces only the visual embedding/scalar in the frozen VTPS synthetic calibration. All of the following remain inherited from v0.1:

- event definitions and eligibility;
- 5 physical-bifolium folds;
- text/position/page/hand nuisance design;
- synthetic sources `iid_null`, `page_only`, `hand_only`, `abstract_text_only`, `background_page_pc`, `immediate_visual`, `line_reset_visual`, `broad_visual`;
- beta grid {0.20,0.30,0.40,0.50,0.70};
- target-free threshold freezing;
- no actual retained/switched target calculation before calibration qualification.

New pair visual score: cosine between L2-normalised 128-D U6-v0.2 crop embeddings. Page/background nuisance removal, if needed, is not retuned: Stage B first uses **no learned page-subspace subtraction** because the external local-window training is designed to eliminate that nuisance. Any page/background null FPR >5% is a hard failure; no repair is available.

### Stage-B gate

The Frontier U6 thresholds remain unchanged:

- every nuisance-source FPR <=0.05;
- detection power >=0.80 for the prespecified physical sources by beta<=0.50.

Failure returns `FAIL_VTPS_CALIBRATION` / `ABSTAIN_UNRESOLVED`; actual retention labels remain unopened.

## Target opening

Only if both external and Stage-B gates pass may a later, separately recorded target run compute retained-vs-switched visual association. No target claim is part of this protocol stage.

## Interpretation

External writer AUC demonstrates that the visual scalar contains local writer-sensitive handwriting information that generalises to unseen writers; it does not establish date, region or manuscript identity. Stage-B success would establish that the scalar can detect a planted physical-state effect under the exact Voynich event/nuisance geometry. Neither is itself Voynich evidence.