# Anonymous word-equality instrument v0.1 — frozen before clustering outcome

Purpose: qualify a source-image representation for one narrow M0 invariant: whole-token equality. Under a globally fixed bijection with preserved boundaries, exact equality/inequality of whole tokens is invariant. This instrument does **not** identify letters, recover a key, classify language, or score Voynich.

## Source roles

All tuning/qualification uses the already-consumed 1541 Bovo print only.

- BUILD: Archive leaf n51, printed stanzas 29–31. Penn 1507w-bovo stanzas 29–31 are the secondary orthographic reference.
- HOLDOUT: Archive leaf n54, printed stanzas 39–41. Penn stanzas 39–41 are not inspected by the clustering threshold optimiser.
- SECONDARY DIAGNOSTIC ONLY: n73 / stanzas 98–100. Before any clustering outcome, its source-image word count was observed to disagree materially with the Penn orthographic count; it is therefore excluded from pass/fail and may only diagnose source/edition mismatch.
- IMAGE-NUISANCE ROBUSTNESS: n82, n86, n120. No Penn reference is available/used for their stanza range.
- Fresh Basel 1599 and Voynich remain sealed.

## Frozen image segmentation

At native 3600×5400 scan resolution:

1. grayscale Otsu binarisation, ink=1;
2. analysis x-range 500:3150;
3. horizontal ink profile smoothed with a 15-row boxcar;
4. candidate text rows where smoothed ink >220;
5. retain row-runs height 60–120 px and y between 500 and 4600;
6. trim line x extent to columns with >1 ink pixel;
7. split words at white column-runs >25 px;
8. trim each word to its ink bounding box;
9. discard punctuation-only segments whose ink-bounding-box height is <55 px;
10. preserve page order, line order, and RTL word order. No lexical correction is allowed.

These constants were fixed after visual engineering on consumed Bovo pages and before any word-clustering outcome.

## Penn orthographic reference reconstruction

From `1507w-bovo.psd`, take leaf terminals, excluding ID/CODE, traces/empty categories, and editorial brace markers. Reconstruct orthographic words according to the corpus annotation convention: parsing splits marked with trailing/leading `@` are rejoined; underscore forms remain one orthographic word. No modernisation or lexical conflation is applied.

The already-observed segmentation preflight counts are recorded, not hidden: BUILD image=157 retained word segments vs Penn=159; HOLDOUT image=179 vs Penn=175. Token-count error is therefore a separate fixed gate and cannot be repaired by clustering.

## Word-image descriptor

For each retained word crop:

- resize to height 48 px preserving aspect ratio;
- if resulting width exceeds 240 px, scale to fit 240 px while preserving aspect;
- centre on a 64×256 blank canvas;
- apply a 3×3 Gaussian blur;
- compute HOG with 8 orientations, 8×8 pixels/cell, 2×2 cells/block;
- append log aspect ratio and normalized ink fraction, each scaled to unit variance using BUILD only;
- L2-normalize the final vector.

No OCR output, character labels, dictionary, Penn token identity, or language model enters this descriptor.

## Clustering and one allowed tuning axis

Cluster word descriptors using complete-link agglomerative clustering with cosine distance threshold `tau`.

The sole tuned quantity is `tau`, selected on BUILD from grid 0.02, 0.025, …, 0.50. For each `tau`, compute the image frequency-spectrum summary and the Penn BUILD summary:

- type-token ratio T/N;
- hapax fraction V1/N;
- repeated-token fraction (N−V1)/N;
- maximum type frequency / N;
- equal-pair density sum_i C(f_i,2) / C(N,2).

BUILD loss = absolute error(TTR) + absolute error(hapax fraction) + absolute error(repeated-token fraction) + absolute error(max-frequency fraction) + 5×absolute error(equal-pair density).

Choose minimum loss; ties choose the smaller `tau` (conservative against false merges). Once chosen, `tau` is frozen.

## HOLDOUT pass/fail

Using frozen segmentation, descriptor preprocessing learned only on BUILD, and frozen `tau`, n54 must satisfy all:

1. token-count relative error vs Penn <=3%;
2. |Δ TTR| <=0.05;
3. |Δ hapax fraction| <=0.05;
4. |Δ repeated-token fraction| <=0.05;
5. |Δ max-frequency fraction| <=0.02;
6. |Δ equal-pair density| <=0.010.

Failure of any gate => `ANON_WORD_EQUALITY_INSTRUMENT_FAIL`; do not touch fresh 1599.

## Nuisance robustness gate

Only if HOLDOUT passes: on n82/n86/n120, create eight deterministic non-semantic scan perturbations per page (24 cells total): contrast ±10%; Gaussian blur sigma 0.5 and 0.8; JPEG recompression quality 90 and 75; resize to 99% and 101% then back to native dimensions. Re-run segmentation and clustering independently with frozen parameters.

A cell passes if retained token count differs by <=2% and the equality partition, compared positionally when counts match, has adjusted Rand index >=0.90. Overall nuisance gate >=22/24 cells.

## Nondegeneracy

On BUILD and HOLDOUT separately, require TTR >=0.30 and max-frequency fraction <=0.15. This prevents trivial one-cluster or near-one-cluster solutions.

## Interpretation

A pass qualifies only the anonymous **whole-token equality** extraction method on this consumed printed source family. It does not qualify glyph-level word lengths, historical-Yiddish population transfer, language discrimination, R/L/T, or Voynich inference. Fresh 1599 remains unopened until this pass is recorded. Any later use of 1599 is a separate transfer-stage protocol and must retain its 1501–1600 transfer-panel label.
