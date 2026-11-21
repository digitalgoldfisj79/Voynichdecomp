# Image overlay for voynichstats.com (2026-11-21)

Word boxes and per-glyph ink outlines on the Yale IIIF images of Beinecke MS 408, built for the
Folio Browser "Show on image" option. Everything here is provisional: one session, no independent re-derivation.

Read `RESULTS_image_overlay.md` first. Retractions are at the top.

## Contents
- `data/voynichese-boxes.json`: 37,550 word boxes on 225 folios, as fractions of the Yale canvas (x10000),
  with a transliteration line number and the Voynichese Project's reading. Source boxes: The Voynichese
  Project, `voynichese_data.zip`, Copyright (C) 2014, Apache License 2.0. Modified here: re-projected onto the
  Yale canvases through `data/reg_full_225.jsonl`, clipped, line numbers added.
- `data/glyphs.tar.gz`: one JSON per folio with an SVG outline of each glyph's own ink (units of 2 canvas px).
  Automatic, not hand-checked.
- `data/glyph_width_table.json`: relative character widths (used as a prior by the ink method; the width-only
  "slab" overlay built from it was withdrawn).
- `scripts/`: s1-s8 word boxes and checks; g0-g7 width-table splitter and its test against Timm's Archetype
  boxes (Zenodo 18827108); h1 fragment counts on the HF crop manifest; inkglyph.py + k1-k5 ink method.
- `results/`: per-folio tables. `trial/`: the renders that were judged by eye.

## Not included (regenerate with the scripts)
Yale images, intermediate pickles, Timm's dataset, the transliteration dumps from the VoynichStats MCP
(`dl/ZLZI_folios.json`, `dl/TTLI_folios.json`).
