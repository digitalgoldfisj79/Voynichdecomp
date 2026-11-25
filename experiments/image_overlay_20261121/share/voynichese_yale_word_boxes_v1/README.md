# Voynichese word boxes on the Yale IIIF images (v1, 2026-11-21)

One rectangle per word of Beinecke MS 408, placed on the Yale Library's current IIIF images,
so any word can be cropped straight from Yale's server at full resolution.

The rectangles and readings are the Voynichese Project's (Apache License 2.0). They were drawn on
the Beinecke's first digitisation (2004), so they did not line up with the current Yale IIIF images,
which are the second digitisation (2014; per voynich.nu/gallery.html). Here they are re-projected
onto the 2014 canvases. Nothing has been corrected: errors in the original boxes or readings carry over.

## Files
- `voynichese_yale_word_boxes.csv`: 37,550 rows, 225 folios. One row per word.
- `folio_summary.csv`: one row per folio (box count, placement check, registration error, Yale canvas label).
- `registration_homographies.jsonl`: the per-folio transforms used (old Voynichese frame to Yale image).
- `LICENSE`, `NOTICE`: Apache 2.0 and the required attribution.

## Columns (`voynichese_yale_word_boxes.csv`)
| column | meaning |
|---|---|
| folio | folio id (Voynichese naming, e.g. f68r1, f101v2) |
| voynichese_index | word index in the Voynichese Project XML for that folio |
| voynichese_word | the Voynichese Project's EVA reading of the box |
| zl_line | line number in the Zandbergen–Landini (ZL) transliteration, assigned by aligning word sequences; blank if none |
| yale_canvas_label, iiif_service | the Yale image the box is on |
| canvas_w, canvas_h | full size of that Yale image in pixels |
| x, y, w, h | the box, in full-resolution Yale pixels, origin top left |
| iiif_crop_url | ready-made IIIF link to the crop |
| folio_placement_check | CONFIRM / FALSIFY / Inconclusive / unchecked: see below |
| folio_reg_median_err_px | median registration fit error for the folio, full-resolution pixels |

Foldout panels sit on Yale's combined canvases (e.g. f69v and f70r1 both on "69v and 70r"); the box
coordinates are on that combined image.

## How it was made
1. Boxes from `voynichese_data.zip` (XML, 1-px grid). The runtime files the Voynichese website itself
   loads are coarser (snapped to a 3-px grid) and clip 122 words at the edge; they were not used.
2. Each folio's old image was registered to the Yale image (SIFT + MAGSAC homography). Median fit
   error 2.4 px at full resolution (worst folio 5.4 px, f68r3).
3. Boxes were mapped through that transform. 317 words whose centre falls outside the registered area
   were dropped (the transform is not trustworthy there; most are on f101v2, f101r1, f89r2, f70r2), and
   19 more had zero size after clipping. 37,886 - 336 = 37,550.

## Checks
- **Placement** (per folio, `folio_placement_check`): the box set was shifted over a grid of offsets on a
  1000-px image and the darkest-ink position found. Planted shifts are recovered on 208 of 225 folios;
  on those, 206 sit within a quarter of a box height of where the ink is (CONFIRM), one does not (f68r1,
  FALSIFY, though it looks right by eye), one is in between (f18v). Median offset 0.06 box heights.
  The other 17 folios could not be tested (unchecked).
- **Right word under the box**: 98.0% of 1,464 hand-drawn glyph boxes from T. Timm's Archetype dataset
  (Zenodo 18827108) that fall inside a box fall in one whose word contains that glyph. 16 of 16 random
  crops fetched from `iiif_crop_url` showed the stated word.

## Known limits
- Boxes are tight: some clip the edge of the first or last glyph, or tall gallows. Pad by about 15% for crops.
- Rotated, circular and label text keep the original boxes' quality, which is weaker there.
- The placement check finds misplaced box sets, not single misplaced boxes.
- `zl_line` comes from an automatic alignment; it is blank for 492 boxes.
- One session's work, not independently re-derived.

## Citation
The Voynichese Project (2014), `voynichese_data.zip`, Apache 2.0. Re-projection: E. Bozzard (2026),
github.com/digitalgoldfisj79/Voynichdecomp, `experiments/image_overlay_20261121`.
