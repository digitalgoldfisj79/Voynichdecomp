# Anonymous whole-word equality source-transfer v0.2 — FROZEN PROTOCOL

Status at freeze: Voynich SEALED. Fresh confirmation source images NOT accessed under this programme.

## Purpose

Repair one demonstrated portability defect in v0.1 and retest source transfer on a new source-disjoint historical-Yiddish print. This is an image-instrument/source-transfer qualification only. It does not identify letters, decipher text, classify language, or authorize Voynich scoring.

## Prior result and consumed development data

v0.1 passed same-print Bovo 1541 holdout and 24/24 scan-nuisance cells, but failed fresh Basel 1599 source transfer because the Bovo-specific absolute vertical admissibility band (`500 < line_start < 4600` on a 3600x5400 canonical page) omitted genuine lines on all six fixed Basel audit pages.

The following are now DEVELOPMENT and may never serve as fresh v0.2 confirmation:

- Bovo 1541: n51 BUILD, n54 HOLDOUT, n82/n86/n120 nuisance pages.
- Basel 1599 `Maʿaśe bait Daṿid bime paras`, e-rara 2611258, all acquired canvases; fixed v0.1 audit canvases 5/11/17/23/29/35 are explicitly DEVELOPMENT.

## v0.2: sole algorithmic repair

All v0.1 detector components remain fixed except the demonstrated absolute-y portability defect.

At canonical 3600x5400 resolution:

1. grayscale Otsu binarisation, ink=1;
2. analysis x-range 500:3150;
3. horizontal ink profile smoothed with a 15-row boxcar;
4. candidate text rows where smoothed ink >220;
5. retain row-runs height 60–120 px and with **line start y satisfying 0.03*H < y < 0.97*H**, where H is canonical page height; this replaces only v0.1's `500 < y < 4600`;
6. trim line x extent to columns with >1 ink pixel;
7. split words at white-column runs >25 px;
8. trim each word to its ink bounding box;
9. discard punctuation-only segments whose ink-bounding-box height is <55 px;
10. preserve page order, line order and RTL word order. No lexical correction.

No other segmentation constant may change after this freeze.

## Descriptor and equality detector — unchanged from v0.1

For each retained word crop:

- resize to height 48 px preserving aspect ratio;
- if width >240 px, scale to fit 240 px preserving aspect;
- centre on 64x256 blank canvas;
- Gaussian blur 3x3;
- HOG: 8 orientations, 8x8 pixels/cell, 2x2 cells/block;
- append log aspect ratio and normalized ink fraction scaled using the original Bovo BUILD scaler only;
- L2-normalize final vector.

Clustering remains complete-link agglomerative clustering on cosine distance with frozen **tau=0.410**. No retuning on Basel or Shmuel is permitted.

## DEVELOPMENT requalification before touching confirmation images

v0.2 must first pass both consumed-source checks:

### Bovo

Re-run n51/n54 under the repaired y rule. The original v0.1 HOLDOUT gates remain unchanged:

- token-count relative error <=3%;
- |delta TTR| <=0.05;
- |delta hapax fraction| <=0.05;
- |delta repeated-token fraction| <=0.05;
- |delta max-frequency fraction| <=0.02;
- |delta equal-pair density| <=0.010;
- TTR >=0.30; maximum cluster fraction <=0.15.

The 24 registered Bovo nuisance cells must still pass >=22/24.

### Basel 1599 development

Apply v0.2 to the already-consumed Basel source. On the six pre-existing audit canvases 5/11/17/23/29/35, visual overlays must show:

- zero missing substantive text lines;
- zero non-text lines included as substantive text;
- <=2 obvious word-boundary split/merge errors per page.

All six must pass. This is a DEVELOPMENT gate only and produces no confirmation evidence.

If either Bovo or Basel development gate fails, stop. Do not access the Shmuel confirmation images.

## Fresh confirmation source

Fixed before image access:

- Work: `Seyfer shmuel / dos bukh Shmuel in taytsher shprakh` (*Shmuel-bukh*).
- Witness: Augsburg, 1544.
- Digital object: Bayerische Staatsbibliothek / MDZ **`bsb10170884`**.
- IIIF manifest endpoint pattern: `https://api.digitale-sammlungen.de/iiif/presentation/v2/bsb10170884/manifest`.

This witness is source-disjoint from Bovo and Basel and has not been used for detector tuning. It is a **1501–1600 transfer-panel witness** under the overarching Yiddish qualification protocol. Composition history does not convert the 1544 witness into core 1350–1500 evidence.

## Confirmation acquisition — fixed before manifest/image access

After DEVELOPMENT passes:

1. fetch and hash the IIIF manifest;
2. let N be the manifest canvas count;
3. define the eligible body interval prospectively as canvas indices `ceil(0.15*N)` through `floor(0.85*N)`, inclusive;
4. acquire eligible canvases in manifest order at the largest practical IIIF resolution;
5. canonicalize each exactly as Basel v0.1: grayscale; isotropic resize to width 3600; if height >5400 centre-crop vertically to 5400, if <5400 centre-pad white to 5400; no deskew, text-region crop, sharpen, denoise or primary contrast change;
6. a page is automatically text-bearing iff v0.2 returns >=10 retained lines and >=80 retained words;
7. concatenate automatically text-bearing pages in manifest order until cumulative retained words >=4,128; the **first exactly 4,128 retained words** are the registered long confirmation cell. No manual page inclusion/exclusion is allowed.

If the eligible body interval cannot yield 4,128 words, confirmation fails quantity.

## Prospectively selected visual-audit pages

After the manifest is fetched but before any page image is viewed, form the set of automatically text-bearing pages in the eligible body interval. For each page compute SHA-256 of the UTF-8 string:

`shmuel1544-anon-v02-audit|bsb10170884|<canvas_id>`

Sort ascending by digest and take the first six distinct canvases. The selected indices/IDs and manifest hash must be written to an audit-sample file before any audit image is opened. No substitutions are allowed.

Each audit page passes only if overlay inspection records:

- zero missing substantive text lines;
- zero included non-text substantive lines;
- <=2 obvious word-boundary split/merge errors.

All six must pass.

## Equality-pair audit

Run frozen descriptor + tau on the 4,128-word confirmation cell.

Use RNG seed string `shmuel1544-anon-word-transfer-v02`.

### Predicted equal

Sample 60 unordered pairs from clusters of size >=2, preferring cross-page pairs and allowing at most 5 pairs from one cluster. Image audit labels each `same visual word`, `different`, or `uncertain`; uncertain fails.

Pass >=57/60 `same visual word`.

### Hard negative

Among different clusters, rank candidate word pairs by cosine distance ascending; take the first 60 distinct pairs after excluding any word used more than twice. Image audit labels as above.

Pass >=57/60 `different visual word`; uncertain fails.

No threshold/preprocessing change is allowed after these audits.

## Scan-nuisance robustness

On the six fixed audit canvases apply the eight v0.1 perturbations:

- contrast 0.90 / 1.10;
- Gaussian blur sigma 0.5 / 0.8;
- JPEG Q90 / Q75;
- resize 0.99 / 1.01 then back to canonical size.

48 cells total. A cell passes if:

- token-count relative difference <=2%;
- matched-position fraction >=0.98;
- ARI >=0.90.

Overall pass >=44/48.

## Nondegeneracy

On the 4,128-word confirmation cell require:

- TTR >=0.30;
- maximum cluster fraction <=0.15.

## Source-transfer decision

`SHMUEL1544_ANON_WORD_SOURCE_TRANSFER_V02_PASS` requires all:

1. DEVELOPMENT requalification passed before confirmation image access;
2. six confirmation visual-audit pages pass;
3. exactly 4,128 retained confirmation words available;
4. predicted-equal audit >=57/60;
5. hard-negative audit >=57/60;
6. nuisance robustness >=44/48;
7. nondegeneracy passes.

Any failure => `SHMUEL1544_ANON_WORD_SOURCE_TRANSFER_V02_FAIL` and this Shmuel witness is consumed for this detector version. No rescue tuning.

## Interpretation

A pass would establish cross-source transfer of the image-derived **whole-token equality** instrument to a second fresh sixteenth-century Yiddish witness. It still would not establish core-period transfer, Yiddish-vs-German/Hebrew discrimination, R/L/T completion, or Voynich inference. Voynich remains sealed until a later preregistered comparator/population gate explicitly admits the target.