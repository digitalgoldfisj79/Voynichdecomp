# BnF M19 Image Bridge v1.4 — Dense Segmental Protocol

Date: 2026-08-09
Parent: v1.2/v1.3 bounded image-gate negatives.
HF source revision: `Digitalgoldfish79/vdino3-crops@ea597db8ff2c06631c4c311d90c8cf0418f5e26c`.

## Rationale fixed before scoring

Two independent image-only observations motivate exactly one combination:
1. raw dense patch-mean DINO at K=19 gave the strongest hard-class result (stability 0.6385; silhouette 0.11931);
2. adjacent-component segmental merging gave reproducible boundaries (~0.83–0.84 F1) but unstable CLS class identities.

v1.4 therefore tests **raw dense DINO, K=19, visual-only segmental merging**. No other K/feature family is searched.

## Sealing and split

All v1.2 sealing rules remain binding: textual/transliteration fields (`word`, `eva_aligned`, `eva_glyph`, `word_len`, etc.) are forbidden before terminal audit. Use only image/provenance fields plus dense vectors.

Use the exact frozen v1.2 split: T12=112 folios, H12=45, C12=68, with Tfit/Tvis=90/22. C12 cannot be language-scored unless H12 passes.

## Visual segmentation

Input: `ccmerge/norm`, raw unit-normalized spatial patch-mean DINO vectors. K=19.

Adjacent components within each image-word region may merge into segments of length 1–3. Segment vector = normalized mean of member dense vectors. Fit/segment by exactly three alternations of centroid fitting and dynamic programming with cost `1 - max cosine + lambda`.

Scan lambda in {0.02,0.04,0.06,0.08,0.10,0.12} using Tfit/Tvis only. Two independent centroid fits are Hungarian matched. Gate:
- combined class/boundary stability >=0.75;
- accepted Tvis fraction >=0.75;
- each class in >=3 Tvis folios and >=25 accepted Tvis segments;
- mean segments/word between 2 and 10.

Choose maximum Tvis cosine silhouette among passing lambdas; ties within 0.005 choose larger lambda. If none passes: `DENSE SEGMENTAL IMAGE-UNDERPOWERED`; no language score.

## M19 qualification and Voynich gates

If visual gate passes, run the exact v1.2 segmental synthetic qualification: six fresh languages (Latin, Italian, German, French, Arabic, Spanish), synthetic microcomponents, boundary F1 >=0.85, 6/6 language correct, min language margin >=0.05, median numerical-map recovery >=0.95, minimum recovery >=0.85, independent-fit agreement >=0.90.

Then refit/freeze visual segmentation on all T12. Fit separate M19 maps under the eight frozen language models using T12 only. H12 primary gate is unchanged: rank 1, margin >=0.05 nats/mapped segment, map-fit agreement >=0.90, H12 coverage >=0.90.

Only if H12 passes, freeze the winning map and evaluate C12 with no refitting. Confirmation requires rank 1, margin >=0.05, coverage >=0.90 and positive candidate-vs-best-other margin in all four frozen C12 buckets.

No decoded strings are inspected before C12 confirmation.
