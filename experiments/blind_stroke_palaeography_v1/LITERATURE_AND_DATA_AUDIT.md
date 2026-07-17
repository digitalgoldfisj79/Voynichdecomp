# Phase 0 literature and data audit

Date: 2026-07-17. This audit was completed before external calibration and before any new blind Voynich model selection.

## 1. Binding prior evidence

The completed DINOv3 baseline established that dense DINOv3 features are a viable correspondence backbone, but pooled embeddings are heavily folio/background sensitive. Raw whole-word nearest neighbours were overwhelmingly same-folio; ink normalization reduced but did not eliminate that leakage. Connected components were not validated as characters.

The completed Nomic programme closed global proposal clustering as a glyph-discovery route. The earlier page-level separation remains correctly described as **scribe-correlated production-style signal, cause unresolved** because layout/page-density baselines explain nearly as much and fold-local nuisance adjustment reduced the result below its frozen threshold.

Accordingly, this programme conditions on homologous local form, uses fold-local nuisance removal, validates on known writers, and treats all segmentations as proposals.

## 2. Palaeographic basis

Davis (2020) distinguishes hands through repeated execution of homologous forms: ductus, loop construction, angle, slope, compactness and connections. Her partition and the reported f115r transition are adjudication targets, not training information. They remain sealed during model selection.

Traditional writer-identification work supports combining allographic and textural evidence. Bulacu and Schomaker describe contour-direction and curvature distributions, run-length features, and grapheme-codebook/allographic evidence. These motivate F1 and fold-local visual-family residuals, but no published descriptor is assumed sufficient for Voynich.

Recent practical evaluations caution that benchmark writer-identification accuracy does not automatically transfer to real palaeographic attribution. The protocol therefore requires page-excluded retrieval, nuisance-only baselines, perturbation tests, known-K recovery and an abstention rule.

## 3. External control corpora

### Historical-WI

Official Zenodo record: `10.5281/zenodo.1324999`.

- 3,600 historical handwritten pages;
- 720 writers;
- five pages per writer;
- thirteenth to twentieth century;
- colour and binarized training/test archives.

This is the primary page-level writer control. Evaluation excludes every same-page crop or tile from the gallery.

### HisFrag20

Official Zenodo record: `10.5281/zenodo.3893807`.

- more than 120,000 historical fragments;
- approximately 100,000 training and 20,000 test fragments;
- filename convention `WID_PID_FID.jpg` gives writer, page and fragment provenance;
- manuscripts, letters, charters and legal documents in the underlying historical-retrieval material.

This is the primary fragment-level control. Same-page fragments are excluded from writer retrieval.

### Secondary medieval calibration

The Parzival database is a thirteenth-century Gothic manuscript written by three known writers, with page, line and word images. It is the preferred medieval-script secondary control if its research-use access can be obtained without changing the frozen primary gates. It cannot substitute for Historical-WI and HisFrag20.

The 1QIsa-a writer-identification data are also available as a compact degraded-manuscript sensitivity corpus. Because its central two-scribe conclusion is itself research output rather than a broad multi-writer benchmark, it is a secondary stress test only.

## 4. Representation choices

### Classical stroke/ductus

- skeleton topology and normalized branch statistics;
- endpoints and junctions;
- distance-transform stroke width;
- contour orientation and curvature;
- Fourier/projection descriptors;
- terminal, loop, crossing and run-length geometry;
- foreground texture.

### Fold-local allographic conditioning

Homologous-form families are defined from secure coordinate-aligned forms where available or learned from training-fold local patches. Family centroids are never estimated with held-out pages/bifolia. Shuffled-family alignment is a mandatory null.

### Self-supervised vision

DINOv3-B/16 is retained for dense patch correspondences. Whole-image or whole-word pooling is not the primary representation. Local token means, variances and mutual correspondences are used after foreground conditioning.

### Historical HTR

Primary model: `Riksarkivet/trocr-base-handwritten-hist-swe-2`. Its encoder is used as a historical-handwriting representation; recognition output and decoded text are irrelevant. Generic Microsoft TrOCR is an ablation.

### External metric adaptation

A Siamese writer encoder can be trained only on external writer labels. It must generalize to held-out pages and retained writers/corpora before becoming eligible for Voynich.

## 5. Leakage and nuisance controls

The principal known failure modes are:

- crop-random page leakage;
- page colour/parchment and scanning derivative;
- layout and line-position cues;
- glyph/content imbalance;
- vocabulary and word-length effects;
- physical quire/folio confounding;
- segmentation-specific artifacts;
- discrete clustering of continuous drift.

The frozen response is document-grouped folds, fold-local residualization, nuisance-only comparators, held-out exact word/visual families, physical-bifolium bootstrap, same-folio-neighbour exclusion, segmentation perturbation and explicit continuous alternatives.

## 6. Data feasibility status at freeze

Available operational assets include:

- Yale-IIIF-registered word crops and multi-scale proposal crops;
- a full-corpus DINOv3 embedding store;
- registration/crop manifests in the private Hugging Face dataset `Digitalgoldfish79/vdino3-crops`;
- current repository experiment lineage and coordinate-transfer implementation.

The original prior-result ZIP bytes were not recovered from File Library; their checksum stubs were recovered. The live dataset and repository are being inventoried and hashed independently. No claim of checksum equivalence is made until the original archives are found.

Exact registration coverage, missing foldouts, available physical-bifolium metadata, crop-path integrity and manifest field coverage are outputs of the running preflight/data audit. A missing physical-bifolium registry is scientifically material but repairable from codicological metadata before Phase I; it does not justify folio-random splitting.

## 7. Phase-0 decision

The programme is feasible. Two broad known-writer controls are openly available with page provenance, a medieval secondary control exists, and the Voynich visual infrastructure is sufficient to begin access and quality audits. The blind Voynich result remains closed until the external gates pass.
