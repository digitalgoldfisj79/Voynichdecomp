# Image overlay (word boxes on the Yale page images): running results

## Retracted findings
- 2026-11-21 GLYPH BANDS: "shipping bar met, ship as approximate" is WITHDRAWN as a judgement of fitness.
  The numbers stand (pooled hit 98.3%, median IoU 0.83 on iin/k/sh) but the bar was too lenient to mean
  anything visually: it scored horizontal overlap only, on three wide targets, against loose hand boxes.
  Ed judged the preview "shockingly bad"; a full-resolution render of f1r (look_f1r_a.png) confirms it:
  full-height slabs that overlap between lines (word boxes include ascenders and descenders), cuts that
  run through strokes, benches (ch) split in two, and visible drift inside words such as fachys.
  Lesson: an overlay has to be judged at full resolution by eye before any metric is trusted, and the
  metric must include vertical extent and single narrow characters.

## 2026-11-21 Pre-registration (written before any box was mapped or any image fetched)

Task: build per-folio word boxes in Yale canvas coordinates for the voynichstats Folio Browser
overlay. Sources: voynichese.com runtime boxes (script/{folio}.js, 636x900 frame),
voynichese_data.zip XML (1090x1500 frame, Apache 2.0 per the project's DataSets wiki page),
register/reg_full_225.jsonl homographies (HF Digitalgoldfish79/vdino3-crops).

G1 (box source identity). Claim: the runtime .js boxes are the same data as the Apache-licensed XML.
- CONFIRM: for every folio, same word count and same word sequence, and an affine fit XML->JS box
  corners has max residual <= 1.5 px (integer rounding in both frames).
- FALSIFY: any folio with a different word sequence, or residual > 1.5 px. Such folios are listed
  and excluded from the "same data" statement.

G2 (boxes land on ink). Claim: the mapped boxes sit on writing in the Yale image.
- Measure per folio: mean darkness (1 - grey/255 after dividing by a blurred background) inside the
  mapped boxes, on a Yale derivative with long side 1000 px.
- Null: the same box set translated by 200 random offsets (uniform, |dx|,|dy| between 1.5 and 6 box
  heights, clipped to the image). z = (observed - null mean) / null SD.
- CONFIRM per folio: z >= 3. FALSIFY per folio: z < 2. Between 2 and 3: "Inconclusive at this power".
- Naive baseline (rule 19 analogue): the same boxes mapped by plain width/height rescaling of the
  636x900 frame to the canvas, no homography. If the naive mapping scores as well as the
  homography, the registration adds nothing measurable on this metric and I say so.
- Power: plant a known shift (boxes moved by 0.5 and 1.0 median box heights vertically) and report
  how far the darkness score drops, in null SDs. If a 1.0-height shift drops the score by < 2 null
  SD on a folio, that folio's G2 result is "Inconclusive at this power".
- Known limit, stated in advance: G2 detects gross misplacement, not few-pixel error, and does not
  check that the RIGHT word is under each box.

P(survives 60d), stated before running: G1 same-data 85%; G2 confirm on >= 90% of folios 75%.

## 2026-11-21 G1 result: FALSIFIED on the residual criterion as pre-registered
- Word count and word sequence: identical in 225 of 225 folios (37,886 words in each source).
- Affine residual XML -> runtime: per-folio maximum has median 3.36 px, 224 of 225 folios exceed the
  1.5 px bar. So the runtime boxes are NOT a rounded copy of the XML boxes.
- Cause found by inspection (not predicted): every one of the 37,886 runtime boxes has all four
  coordinates on a 3-px grid, and 122 boxes in 10 folios are clamped to zero width/height at the
  viewport edge (105 of them on f101v2). The runtime file is a coarsened, viewport-clipped copy.
- Consequence: the XML (Apache 2.0, 1-px grid in a larger frame) is the better source. The earlier
  DINOv3 crop pipeline used the runtime boxes, so its crops inherited the 3-px snapping and lost the
  122 clamped words. This does not touch any embedding result; it is a geometry note.
- Registration file check: 225 rows, 225 passed, 201 distinct Yale image services. Median
  reprojection error in full-resolution pixels: median 2.36, worst 5.43 (f68r3). The "2.9 px worst
  (f82r)" figure I quoted from memory earlier in this chat was at derivative scale, not full
  resolution. Corrected here.

### Amendment to G2, written before any image was fetched
Add arm B: XML boxes mapped XML -> runtime frame by a per-folio scale+offset fitted on box centres
(trimmed least squares, unclamped boxes only), then through the same homography. Arm A = runtime
boxes through the homography (as pre-registered). Arm N = naive rescale. Because A boxes are
padded/snapped (larger), mean in-box darkness is biased towards B; A vs B is therefore reported
descriptively only. For arm B a global offset scan (integer pixel shifts on a 2000-px derivative,
30-folio sample, seed 20261121) checks whether the centre fit carries a systematic bias; if the
darkness-maximising shift is consistent across folios (same sign in >= 80% and median |shift| >= 1
legacy px) it is applied as a correction and logged, otherwise no correction.

## 2026-11-21 G2 result (random-offset null): Inconclusive at this power
- Arm B (XML boxes): in-box darkness z vs random-offset null, median 1.59 over 225 folios;
  44 CONFIRM, 5 FALSIFY, 176 Inconclusive. Arm A (runtime boxes) median z 0.78. Naive rescale median z -0.69.
- Power, measured: a planted shift of one full box height lowers the arm-B score by a median of only
  1.52 null SD (0.5 height: 0.77). The pre-registered power bar was 2. The instrument cannot see a
  one-line misplacement on most folios, so neither the 44 confirmations nor the 5 falsifications are
  usable. Cause: on a text page, boxes shifted by 1.5-6 box heights land on other lines of text, so the
  null is nearly as dark as the observation.
- What survives descriptively: B scores above the naive rescale in 218 of 225 folios.
- The instrument is retired, not the boxes. Replacement below.

## 2026-11-21 G2b pre-registration (peak-location instrument), written before running it
Instrument: for a box set, compute mean in-box darkness over a grid of integer shifts
(dx, dy in -16..16 px on the 1000-px derivative) and take the argmax shift. A correctly placed box
set peaks at (0,0); a set misplaced by d peaks at -d.
Phase 1 (known-answer, on the planted data): plant 5 shifts on arm B per folio:
(0,8) (0,-8) (8,0) (-8,0) (6,-6) px. Recovery = argmax of the planted set, minus the folio's own
unplanted argmax, within 2 px of minus the plant. A folio is "readable" if >= 4 of 5 plants recover.
Instrument accepted only if >= 80% of folios are readable; otherwise report failure and fall back to
visual inspection only.
Phase 2 (decision, readable folios only): arm B residual offset r = |argmax| in px, as a fraction of
the folio's median box height. CONFIRM r <= 0.25; FALSIFY r > 0.5; else Inconclusive. Unreadable
folios: "Inconclusive at this power".
Naive baseline: same statistic for arm N. Offset bias rule for arm B as in the amendment above
(same sign in >= 80% of readable folios and median |shift| >= 1 legacy px -> apply correction).
P(survives 60d): instrument accepted 65%; B CONFIRM on >= 90% of readable folios 70%.

## 2026-11-21 G2b result (peak-location instrument)
- Phase 1, instrument: 208 of 225 folios readable (>= 4 of 5 planted shifts recovered within 2 px);
  201 folios recovered all 5. Acceptance bar was 80%; observed 92.4%. Accepted. Run once, no tuning;
  no second held-out control was run, so this is a single-pass acceptance.
- Phase 2, arm B (XML boxes -> centre-fit -> homography): 206 CONFIRM, 1 FALSIFY (f68r1), 1 Inconclusive
  (f18v), 17 Inconclusive at this power (unreadable: f102v2 f3v f68v3 f69r f6r f71v f72r1 f72r2 f72r3
  f88v f89r1 f89r2 f89v1 f90r1 f93v f94v f96v).
- Effect size: arm B residual offset, readable folios, median 0.058 box heights (p90 0.187, max 0.955);
  in full-resolution pixels median 3.8, p90 11.4. Naive rescale: median 1.121 box heights (p90 1.441).
  The scan grid is 1 px on a 1000-px derivative (about 3.8 full-res px), so offsets below one step are
  not resolved; "median 3.8 px" is one grid step, an upper bound on the typical error, not a measurement of it.
- Offset bias rule: dy median -1.13 legacy px, negative in 62% of readable folios (positive 11%);
  dx median 0. The 80% same-sign bar is not met, so NO correction is applied. A sub-step upward bias
  of the centre fit is possible and unmeasured at this resolution.
- Visual check (4 folios drawn, 1000-px derivative): f1r boxes sit on the words; f68r1 boxes sit on the
  star labels (the FALSIFY there is not visible to the eye at this size and may be the instrument
  reacting to the sparse labels and star drawings; unverified which); f101v2 shows the text block
  correctly boxed but the XML words outside the runtime viewport map off the left edge of the Yale
  canvas. Those are extrapolations of the homography and are wrong. Rule adopted: a box is "unplaced"
  and not shipped if its centre in the runtime frame lies outside the 636x900 viewport.

## 2026-11-21 Line assignment, coverage and the shipped file
- Folio names: voynichese f101r1 / f101v2 are treated as the site's f101r / f101v (alias). Support for the
  alias: 134 of 141 placed f101r1 words and 101 of 108 placed f101v2 words align to ZLZI tokens of
  f101r / f101v. Not checked against the images.
- 37,886 XML words; 317 unplaced (centre outside the runtime viewport; 105 on f101v2, 68 on f101r1,
  40 on f89r2, 37 on f70r2, rest <= 17 per folio); 19 more dropped as zero-size after clipping.
  Shipped: 37,550 boxes, 225 folios, 1,164,753 bytes (voynichese-boxes.json, version 20261121-1).
- Line numbers come from a per-folio global alignment of the voynichese word sequence to ZLZI
  (difflib similarity, floor 0.45); 37,348 of 37,886 boxes received a line number.
- Coverage under the deterministic runtime rule (equal counts -> by position; else exact-string LCS
  anchors plus equal-length gaps): ZLZI 34,217 of 37,463 tokens get a box (91.3%); TTLI 33,450 of
  34,351 (97.4%). With the looser fuzzy rule the ZLZI figure was 95.5%; the deterministic rule was
  chosen because the site can reproduce it exactly in a test.
- Where a box pairs with a ZLZI token, the box's own word equals the token in 88.1% of pairs (fuzzy
  rule). The overlay colours by the active transliteration's token, not the box's word.
- Fixture numbers for the site tests: f1r ZLZI 207 tokens / 190 with box; f1v ZLZI 91 / 82;
  f76r ZLZI 558 / 518; f1r TTLI 183 / 181. Full table: overlay_expected_counts.csv.

P(survives 60d): "boxes are within half a box height of the word on >= 90% of ordinary text pages" 80%;
"ZLZI coverage 91.3% reproduces on the live site" 70% (depends on the site's line numbering matching the
MCP dump's); "f101 alias is right" 75%.

## 2026-11-21 Site build (Lovable commit 25831b6b, preview only, not published)
- Data file copied with matching sha256 (17fc4545...eefbc).
- The site's own tests, run from its raw records, reproduce my fixtures exactly: f1r ZLZI 190/207,
  f1v ZLZI 82/91, f76r ZLZI 518/558, f1r TTLI 181/183, whole corpus ZLZI 34,217/37,463 (91.3%).
  This confirms the site's line numbers agree with the MCP dump the file was built against.
- f1r, ZLZI, minimum 2: the site colours 65 boxes; my independent simulation (s8) also gives 65.
  Visual check of that simulation on a 2000-px Yale image: boxes sit on the words and the labels
  match the script (two adjacent "cthar", "daiin", "shol", "or", "y"). One page only.
- 38 of 225 folios draw nothing because the site shows no single Yale image for them (35 foldout
  panels) or a different image from the one the boxes were registered to (f90r1, f90r2, f95v2).
- Not seen by me: the overlay as rendered in the site preview. Lovable reports it checked f1r in a
  Chrome engine only.
- Corrections to what I said earlier in this chat: word_iiif_links.csv is NOT on Hugging Face (no
  tools/ folder, not in results/), so the boxes were rebuilt from the XML and reg_full_225.jsonl.

Status: provisional (rule 21). Needs re-derivation in a later session and a cold check (rule 20) before
glyph-level work is built on these boxes.

# GLYPH STEP (splitting word boxes into glyph bands)

## 2026-11-21 Pre-registration, written after parsing Timm's boxes but before any splitter was scored
Known-answer data: Timm's Archetype project (Zenodo 18827108, md5 9b19b4eb... verified): 1,684
hand-drawn graph rectangles; iin 854, k 432, sh 348, in 32, weirdos 18; 201 loci.
Task: given a word box and its EVA string, predict the horizontal band of each character.
Gold: the x-range of Timm's rectangle. Scored only where the graph's centre falls inside one of my
word boxes AND that box's word contains the allograph string exactly once (iin not preceded by i).
Metrics per graph: horizontal IoU of predicted band vs gold; hit = gold centre inside predicted band.

Phase 1 (development, free iteration, every variant logged):
- DEV set: iin + k graphs on a random half of folios (seed 20261121).
- Variants: V0 equal split (naive baseline, rule 19); V1 per-character width table fitted by
  non-negative least squares on word-box widths over the whole corpus (no Timm labels);
  V2 = V1 priors with cuts snapped to ink-profile minima by dynamic programming; each with and
  without trimming blank margins. Parameters tuned on DEV only.
- One variant is chosen on DEV.
Phase 1 acceptance (run once, after freezing and hashing the chosen splitter):
- HELD-OUT A: iin + k on the other half of folios. HELD-OUT B: all sh graphs (allograph never tuned on).
- Accept only if the chosen variant beats V0 on mean IoU by >= 2 folio-clustered bootstrap SE on
  HELD-OUT A and on HELD-OUT B separately. If it does not beat V0, V0-level performance is all that
  can be claimed and I say so.
Shipping rule (absolute, on HELD-OUT A+B pooled): hit rate >= 80% and median IoU >= 0.5 -> ship as
"approximate glyph position"; hit 60-80% -> ship only behind an "experimental" label; < 60% -> do not ship.
Stated limits: gold covers three allographs only; nothing here validates gallows other than k, bench
characters other than sh, or rare characters; words with repeated target strings are excluded, so
long repetitive words are under-sampled; Timm's rectangles are loose, which caps achievable IoU.
P(survives 60d), before running: chosen variant beats V0 on both held-out sets 55%; ship bar (>= 80% hit) 45%.

## 2026-11-21 Glyph Phase 1 development (DEV only: 376 iin + 192 k on half the folios; 30 variants logged in g3_dev_variants.csv)
- Linkage of Timm's graphs: 1,634 iin/k/sh graphs; 1,434 linked; 83 centre in no word box; 79 locus name not
  in my box set; 8 image aspect differs; 30 box word lacks or repeats the allograph. Side result: of the
  1,464 graphs whose centre fell inside a word box, 1,434 (98.0%) sit in a box whose word contains the
  labelled allograph exactly once. That is the first check that the RIGHT word is under each box, on
  ~200 folios, though only for words containing iin, k or sh.
- DEV results (mean IoU / hit): equal split 0.7246 / 0.9173; width table + pad 0.7773 / 0.9912;
  best ink-snapped variant 0.7710 / 0.9894. By allograph: k 0.5035 -> 0.6775 IoU, hit 0.7552 -> 0.9740;
  iin 0.8375 -> 0.8282 (equal split is already at ceiling for a 3-character word-final group).
- Ink-snapping and margin trimming did not improve on the plain width table on DEV. Chosen variant:
  mode=table, pad=on, no trim, no snap. It needs no image data, so the site can compute it for any EVA token.
- FROZEN before the held-out run: glyphsplit.py sha256 31dda0830ee3bc14..., width table pickle sha256 aa17823b55639d7d...

## 2026-11-21 Glyph Phase 1 acceptance: single held-out run of the frozen splitter (g4_heldout.csv)
| set | n | folios | width table IoU | equal split IoU | diff | boot SE | ratio | table hit | equal hit |
|---|---|---|---|---|---|---|---|---|---|
| HELD-OUT A (iin + k, unseen folios) | 552 | 94 | 0.7650 | 0.7121 | +0.0529 | 0.0039 | 13.46 | 0.9746 | 0.9058 |
| HELD-OUT B (sh, allograph never tuned on) | 314 | 141 | 0.8552 | 0.8478 | +0.0073 | 0.0054 | 1.35 | 0.9968 | 0.9968 |
| pooled A+B | 866 | 166 | 0.7977 | 0.7613 | +0.0364 | 0.0034 | 10.75 | 0.9827 | 0.9388 |
| A, k only | 191 | 91 | 0.6606 | 0.4903 | +0.1703 | 0.0093 | 18.31 | 0.9476 | 0.7487 |
| A, iin only | 361 | 94 | 0.8202 | 0.8294 | -0.0092 | 0.0019 | -4.95 | 0.9889 | 0.9889 |

- Acceptance as pre-registered ("beats equal split by >= 2 SE on A and on B separately"): NOT MET.
  A passes (ratio 13.46). B: Inconclusive at this power, effect +0.0073, bootstrap SE 0.0054, ratio 1.35;
  no planted-signal power run was done for B.
- So the claim "the width table is better than an equal split" holds for k only (+0.1703 IoU, hit
  0.7487 -> 0.9476). For iin the width table is slightly WORSE (-0.0092, ratio -4.95). For sh there is
  no measurable difference. The naive split is already near ceiling for iin and sh.
- Shipping rule (absolute, pooled A+B, chosen variant): hit 0.9827 >= 0.80 and median IoU 0.8286 >= 0.5
  -> ship as "approximate glyph position". The equal split would also have cleared this bar (hit 0.9388).
- Descriptive, post hoc (g5_diag.csv), frozen variant on the 866 held-out graphs: band-centre error median
  5.1 full-res px (p90 13.1) = 0.17 mean character widths (p90 0.44); 7.2% of graphs are off by more than
  half a character width (k: 10.5%). Error grows with word length: median 0.08 character widths for 1-3
  characters, 0.27 for 8+.
- NOT validated: every character other than k, and the groups iin and sh. In particular single narrow
  characters (o, a, e, i, y), other gallows (t, p, f), benches with gallows (cth, ckh), and rare characters
  whose table widths rest on few words (g, x, z, v). A 0.17-character median error is small for a
  3-character group and large for a single narrow character.
- Width table (relative units) and pad are in glyph_width_table.json; JSON port agrees with the frozen
  library to 9e-05 over 300 words.

P(survives 60d): "k improvement over equal split replicates on re-derivation" 85%; "pooled hit >= 95% for
iin/k/sh" 85%; "single narrow characters hit >= 80% when someone hand-boxes them" 40% (no data).

## 2026-11-21 Glyph bands on the site (Lovable commit dd60f232, preview only, not published)
- Site reproduces the five edge fixtures to 4 decimals.
- f1r, ZLZI: 904 raw-character bands / 755 EVA-grouped bands over 190 paired words on the site;
  my independent simulation gives 904 / 755. Words mode unchanged at 65 boxes. f76r draws 2,560 bands.
- Simulated f1r crop (sim_glyphs_f1r.png, 2000-px image, Timm's boxes outlined in black) viewed: bands follow
  the words and Timm's k / iin boxes sit over the matching bands. One page, by eye.
- Not seen by me: the site preview itself. Lovable reports Chromium checks only.
- The inline simulation code is reproduced in scripts/g6_sim_glyphs.py.
Status: provisional (rule 21); no cold demolition yet (rule 20).

## 2026-11-21 Fragment-count feasibility check on the HF manifest: pre-registration (before reading counts)
Question: how often does the stored connected-component segmentation (crop.py, kinds cc and ccmerge) break
a word into as many pieces as it has glyph units? Data: results/corpus_crop_manifest.jsonl, n_slots per word.
Unit definitions compared: raw EVA characters; EVA-grouped (ch sh cth ckh cph cfh as one).
- Route VIABLE if, for the better of cc / ccmerge under either unit definition, pieces == units in >= 60%
  of words and |pieces - units| <= 1 in >= 85%.
- Route NOT VIABLE as a direct mapping if exact agreement < 30%.
- Between: usable only with a merge/split step driven by the transliteration; say so.
Limit stated in advance: equal counts do not prove the pieces ARE the glyphs (a split in one place and a
merge in another cancel). The old fragments were cut from the 3-px runtime boxes at <= 448 px with a
40-piece cap. P(viable) before running: 25%.

## 2026-11-21 Fragment-count check: NOT VIABLE as a direct mapping (h1_fragment_counts.csv)
Full manifest read: 1,762,602 rows, 37,886 words, 225 folios (cc 415,587 pieces, ccmerge 332,085).
| segmentation | units | pieces == units | within 1 | fewer pieces | more pieces | median pieces |
|---|---|---|---|---|---|---|
| cc | raw | 2.98% | 8.95% | 3.23% | 93.79% | 11 |
| cc | grouped | 2.32% | 7.01% | 2.31% | 95.37% | 11 |
| ccmerge | raw | 5.96% | 18.16% | 15.36% | 78.69% | 9 |
| ccmerge | grouped | 5.43% | 16.29% | 12.86% | 81.71% | 9 |
- Best case 5.96% exact against a 30% floor: falsified. Exact agreement does not rise for short words
  (one-unit words: 4-11%), so this is not a long-word problem.
- Direction: the stored segmentation over-fragments (about twice as many pieces as characters).
- Not separated here: how much of the excess is real stroke breakage and how much is ink from
  neighbouring words and lines inside the 15% crop padding, or parchment speckle. The manifest cannot
  tell these apart; only re-running on tight boxes can.
- Calibration: P(viable) was 25% -> NOT viable. The one row I had read (12 pieces for an 11-character
  word) was unrepresentative.

## 2026-11-21 Ink-based glyph trial: plan, written before any code was run
Method under trial: threshold the ink on the page; assign each connected piece of ink to the word box that
contains most of it; cut pieces at thin joins; then group the pieces of a word, left to right, into its
characters by dynamic programming (expected widths from the width table, plus a "tall piece = gallows" cue).
Output: per glyph, its own ink pixels and a tight box.
Judgement: Ed's eye on full-resolution renders of ten varied pages. That is the test. No metric decides.
Secondary, development only: 2-D IoU against Timm's rectangles on the DEV set (iin + k, half the folios).
The held-out sets were used once today and are NOT reused in this trial.
Naive baseline shown alongside: width-table slabs at full box height (what is on the preview now).
Stated in advance: circular and rotated text is out of scope; painted areas overlapping text will pollute
the ink mask; faint ink will fragment. P(Ed passes the ten pages at first showing): 35%.

## 2026-11-21 Ink-based glyph trial: what was built and what the development numbers say
Three iterations, each judged by my own look at f1r at full resolution before the next (logged here, none hidden):
- v1: connected pieces cut at thin joins, grouped by DP on expected widths. Faint ink missed; gallows split
  between two characters.
- v2: hysteresis threshold for faint ink, neighbouring tall pieces fused, dip cuts, column-share fallback for
  characters left without ink. Gallows loops that overhang the next letter were still cut by column.
- v3 (inkglyph.py as delivered): grouping is done on the x-height band only; ink above and below the band
  (gallows tops, plumes, tails) is attached to the character it physically touches.
Ten pages rendered at full resolution (out/trial/*_ink.jpg): f1r f9v f26r f34r f42r f58r f75r f78r f88r f103r.
Per page, share of glyphs that needed the column-share fallback (touching glyphs the cuts did not separate):
f1r 11.5%, f9v 7.1%, f26r 7.4%, f34r 9.8%, f42r 6.1%, f58r 10.9%, f75r 13.5%, f78r 18.9%, f88r 7.1%, f103r 16.1%.
Glyphs with no ink at all: 0 to 0.8% per page.

DEV numbers (iin + k on half the folios, 568 graphs, 2000-px images; held-out sets NOT touched), 2-D IoU vs
Timm's rectangles:
| | ink method | width-table slabs |
|---|---|---|
| mean 2-D IoU, all | 0.587 | 0.622 |
| mean 2-D IoU, k | 0.574 | 0.572 |
| mean 2-D IoU, iin | 0.594 | 0.648 |
| share with IoU >= 0.5 | 0.831 | 0.852 |
| gold centre inside predicted box | 0.951 (x and y) | 0.991 (x only, not like for like) |
- On this metric the ink method is NOT better than the slabs, and for iin it is worse. No bootstrap SE was
  computed; I make no claim of a difference in either direction.
- Two reasons the metric may understate it, neither tested: Timm's rectangles are loose, so a tight box
  inside a loose one scores low by construction; and the DEV run used 2000-px images where strokes are
  about 2 px wide, while the renders use full resolution.
- One reason it may be telling the truth: about 5% of iin/k glyphs (about 10% for k) are put on the wrong
  ink. That is a real error rate and will show as wrongly coloured glyphs.
- My own eye on f78r and f103r crops: the same shapes carry the same colours across the page far more often
  than not (d, y, q, o, gallows), with visible errors where glyphs touch and where paint or stains sit
  under text. This is an impression from two crops, not a count.
Status: awaiting Ed's pass/fail. Nothing sent to the site.

## 2026-11-21 Corpus run of the ink method (k3_corpus_glyphs.py, inkglyph.py v3, full-resolution Yale images)
- 225 folios, 201 images. Readings segmented per box: the ZLZI token paired by the site's line rule, plus
  the voynichese word where it differs.
- First reading: 189,457 glyphs; 15.2% separated from a touching neighbour by the column-share fallback
  (per folio: median 10.1%, p90 34.3%, max 48.3% on f70r1); 2,633 glyphs (1.39%) with no ink found.
  The ten trial pages Ed saw ran 6-19%, so they were on the easy side of the corpus: one page in ten is
  at 34% or worse. Those pages have not been looked at.
- Outlines traced to polygons (simplified to 1.8 px), stored as SVG path data in units of 2 canvas px:
  23,925,061 bytes over 225 files (median 75,822 bytes per page, largest 309,375 on f111v); 6,976,276 bytes
  as glyphs.tar.gz. Round trip checked by eye on f78r (k4_roundtrip.py): the stored outlines redraw onto the ink.
- Site fixtures (ZLZI): 34,217 words with shapes, 170,812 glyph paths; f1r 190 words / 883 paths
  (739 with EVA grouping); f78r 272 / 1,311; f103r 490 / 2,566; f1r with TTLI 181 / 838.
- Per-folio table: glyph_corpus_by_folio.csv.
Status: provisional, unscored. No accuracy claim is made for this method yet.

## 2026-11-21 Ink outlines on the site (Lovable commit 8fb450f9, preview only, not published)
- 225 glyph files bundled unchanged (23,925,061 bytes). Every fixture reproduced exactly: f1r ZLZI 190 words /
  883 paths (739 grouped); f1r TTLI 181 / 838; f78r 272 / 1,311; f103r 490 / 2,566; corpus ZLZI 34,217 / 170,812.
  Words mode unchanged (65 boxes on f1r).
- Lovable checked Chromium only. Not seen by me in the preview. Full-resolution image swap at high zoom unverified.
