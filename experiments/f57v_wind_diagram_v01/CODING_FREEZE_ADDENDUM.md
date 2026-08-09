# CODING FREEZE ADDENDUM — v0.1 execution
Date frozen: 2026-08-09
Status at freeze: no empirical target feature score has yet been generated.

## Primary coder
The fixed visual feature schema will be coded by **Qwen/Qwen2.5-VL-3B-Instruct** running locally in a Hugging Face GPU job. This is a machine coder, not a human expert.

Each source image is downloaded locally first, assigned an opaque random ID under fixed seed `20260809`, and supplied to the model as pixels only. The prompt contains the frozen feature definitions but no repository, shelfmark, folio, date, place, class label, source URL, transcription, or semantic description. The model is explicitly instructed not to read or interpret writing.

Each feature is forced to one of `{0, 0.5, 1}`. There is one scoring pass per image. Invalid JSON may be reparsed mechanically, but no feature definition, threshold, weight, crop, class label, or source selection may be changed in response to a score.

The target is scored in the same randomized pass and is not identified as the target to the coder.

## Panel freeze
The execution panel contains eight independent manuscript strata and is frozen before scoring:

1. Walters W.73 — 2 wind diagrams, 4 same-manuscript circular/cosmological controls.
2. British Library Harley MS 3667 — 1 wind diagram, 6 same-manuscript circular/computistical controls.
3. British Library Cotton MS Titus D XXVII — 1 wind diagram, 2 same-manuscript computistical controls.
4. British Library Harley MS 2688 — 1 wind diagram, 3 same-manuscript controls.
5. Vatican Pal.lat.1417 — 3 wind-diagram pages (1v–2v), 3 same-manuscript diagram controls.
6. Morgan MS M.721 — 2 distinct non-overlapping wind diagrams cropped from f.14v, 4 same-manuscript astronomical/cosmological controls.
7. NYPL MA 97 — 1 wind-head map, 4 same-manuscript map controls.
8. BSB Clm 210 — 1 twelve-winds diagram, 3 same-manuscript astronomical/calendar controls.

Totals: 12 wind-positive images, 29 controls, 8 independent strata. Late-medieval (1350–1500) positives: 3 (Morgan two diagrams; NYPL one). Central/Alpine sensitivity positives include Clm 210 (Salzburg) and Harley 2688 (catalogued as possibly Cologne/Western Germany or France; geographic ambiguity will be reported explicitly).

The unresolved user-supplied BnF screenshot remains quarantined. Bodl. MS 646 and BSB Clm 569 are not used in the primary run because their exact required image surfaces could not be acquired reproducibly before this freeze.

## Morgan f.14v crops
A preliminary pixels-only detector, run before primary feature scoring, found two distinct circular diagrams on f.14v. Frozen normalized crop boxes (x1,y1,x2,y2 on 0–1000 page coordinates), expanded by a fixed 20-unit margin from the detector boxes, are:
- M721-A: `[23,132,158,272]`
- M721-B: `[19,282,157,421]`

No crop will be adjusted after feature scores are observed.

## Positive-control gate
The preregistered positive-control qualification is evaluated before any target inference is released. If it fails, the formal primary result is `NO TEST`; target distances, even if technically computed, are exploratory only.

## Secondary vision model
The preregistered DINOv3 model was attempted first. The available Hugging Face DINOv3 repository returned a gated-repository 401 in the execution environment. Therefore the preregistered fallback clause ('closest current self-supervised vision model if DINOv3 is unavailable') will use `facebook/dinov2-small`. This substitution is frozen before target embedding analysis.