# vms-spatial exploration (eo777/vms-spatial @ f0d04b7, 6 Nov 2026) — running results, 2026-11-14

## RETRACTED FINDINGS
- 2026-11-14 08:15: B1 'ink coverage differs by Davis hand' and 'conjugate leaves share coverage' were mostly WRITING SIZE (coverage rises when the
  same pen writes smaller). Net of size neither holds at p < 0.01 (K1). What survives: hand 2 writes smaller; ink weight net of size is smooth along Path 1.

## Docs read in full: README, STREAM, SHEETS, MSI_LEAF_AUDIT, PITFALLS. Not yet read: COMPARE, LIGHTTABLE, DATA_MODEL, DEWARPING, code.
Key find: a "2026 bifolium drift study" with a ten-sheet Herbal A "1a candidate" copying chain:
A:1-8 > D:27-30 > F:42-47 > G:49-56 > A:2-7 > D:25-32 > C:20-21 > E:35-38 > A:4-5 > D:28-29 ("seven-sheet middle run is the main
copying-target chain"). 4 of the 6 crease bifolia (D:25-32, C:20-21, A:4-5, D:28-29) are in it. The study itself is not in the repo.

## PRE-REGISTRATION P1 (2026-11-14 ~06:50, before computing): leaf heights as an independent physical variable
Data: data/manuscript/leaves.json.gz leaf_height_mm (measured on Beinecke scans; his stated noise 2-3 mm; ordinary leaves 221-232 mm).
Sample: ordinary leaves of quires A-H (no panel, scale_method and top_method 'measured').
STEP 0 instrument check (known answers), must pass before anything else is read:
  (a) recto vs verso of the SAME leaf: Pearson r of heights; pass if r >= 0.5 and above the 99th percentile of 5,000 permutations.
  (b) conjugate leaves of the same sheet vs random leaf pairs in the same quire: mean |height difference| lower, p < 0.05.
  If (a) fails: report "Inconclusive at this power" for everything below.
STEP 1 his claim "f5 is cut much shorter than the others in quire 1": f5 height (mean r,v) vs the other quire-A leaves; report mm and rank.
STEP 2 bifolium height (mean of its pages) vs Golf Path 1 bifolium score, quires A-H: Spearman, null = permute heights WITHIN quire (10,000).
  CONFIRM (physical size tracks text order beyond quire): p < 0.01. FALSIFY: p >= 0.05. Same for the label-free order (L1).
STEP 3 BOTTOM-N6 height spread (SD) vs depth-matched random sets (20,000). CONFIRM p < 0.01; FALSIFY p >= 0.05.
Prediction: step 0a passes weakly (P=0.5); step 2 falsified (P=0.75); step 3 falsified (P=0.8).

## P1 RESULT: INSTRUMENT FAILED its known-answer check -> Inconclusive at this power for steps 1-3
Step 0a: recto vs verso height of the same leaf, r = 0.181 (58 leaves; needed >= 0.5; permutation 99th pct 0.29); median |r - v| = 3.9 mm
 against a between-leaf SD of 2.9 mm. The measurement noise is larger than the real variation. Step 0b: conjugates no closer than
 random same-quire leaves (1.09 vs 1.14 ± 0.16 mm, z 0.27).
Not interpretable, logged only: f5 = 229.2 mm (f5r 226.8, f5v 231.6), not short in quire A (227.2-229.4); height vs Path 1 rho -0.21
(within-quire null -0.31 ± 0.08); N6 height SD 1.45 vs 1.47 ± 0.41.
Note: his 'f5 is cut much shorter' is NOT visible in his own leaves product, but the product cannot resolve 2-3 mm, so this neither
supports nor contradicts him. A real test needs ruler-calibrated measurements (as in his MSI audit of f1r/f8r).

## Other things in vms-spatial worth using (descriptive; nothing run)
- data/manuscript/registrations.json.gz + spatial_review.json.gz: every ZL token tied to a word rectangle (38,325 of 38,484 located) and a
  per-page mapping from that frame onto the Beinecke IIIF scans (SIFT + RANSAC homography + per-line shifts, ~1 px). This is the piece our
  DINOv3 crop manifest lacks (no box coordinates): any word vector's image could be re-cut from these two files + Yale IIIF.
- ink_measures.json.gz: per-word ink baseline, x-height, writing angle measured on the scans. Candidate scribal-hand features that are NOT
  layout/density (relevant to the demoted P1 hand-as-shape finding).
- viewer/data/compare/rankings.json.gz: first-line page-overlay resemblance (layout Dice + JS), top 50 per page only; full matrix needs
  his compare.rank(). His own PITFALLS 25: 'describe resemblance, they do not test copying'; pairs were chosen by eye before controls.
- Stream / Light table viewers: lay sheets as a chain with adjustable overlap; his own caveat: 'A boundary you set by hand is where you
  put it ... not evidence that it does [line up]'.
- PITFALLS.md: 28 documented traps with his own withdrawn results (word-length artefact, boxes kept under shuffle, one-realisation nulls).

## PRE-REGISTRATION S1-S4 (2026-11-14 ~07:30, written before any statistic below was computed)
Family: about 12 tests in this block. Single-test thresholds below; anything with 0.01 <= p < 0.05 is reported as weak, not confirmed.
All nulls 10,000 draws unless stated; seeds 20261122+.
Toolkit: eo777 vmsspatial @ f0d04b7 (his instruments, his data products). Page-pair overlay scores from compare.rank(pool='all',
cutoff=3, scope='pages'): dice_current (occupied-area overlap of the opening lines after one translation = LAYOUT, no text),
spatial (position-constrained EVA similarity), js_joined (text; lower = more alike).
Sheet-pair score = mean over all eligible page pairs between the two sheets.

S1 THE CHAIN (obvious). Chain = A:1-8 > D:27-30 > F:42-47 > G:49-56 > A:2-7 > D:25-32 > C:20-21 > E:35-38 > A:4-5 > D:28-29.
 Statistic: mean sheet-pair score over the 9 joins. Null A: random orderings of the SAME ten sheets. Null B: ten random Currier-A hand-1
 herbal sheets (quires A-H) in random order. PRIMARY: dice_current, null A. CONFIRM p < 0.01; FALSIFY p >= 0.05.
 Text metrics (spatial, js_joined) are reported but are circular with Path 1 (fitted on text), so they cannot confirm anything.
S1b WHOLE ORDER. Mean adjacent sheet-pair dice along (i) published Path 1, (ii) label-free order, (iii) current binding order
 (sheets by quire, outer to inner). Null: permute sheets within stratum (section x hand of the sheet's majority). Same thresholds.
S2 TILT CARRY (obvious). Page tilt = compare.page_drift recorded angle. STEP 0 instrument check: odd-line vs even-line median angle per
 page, Pearson r >= 0.7 required. NUISANCE check: pages photographed in the same opening (verso n with recto n+1 in the bound book)
 vs other same-quire page pairs: if their tilt agreement is strong, tilt is largely the photograph.
 Statistic: mean |difference| of sheet median tilt across the 9 chain joins; nulls A and B as S1. And along Path 1 / label-free / binding
 orders with the stratified null. CONFIRM p < 0.01; FALSIFY p >= 0.05.
S3 INK FEATURES (less obvious). Per page, from ink_measures (unflagged words): x-height in mm (xheight / leaves 'mm'), writing angle
 (per-word shear 'angle' median), x-height spread (MAD/median). STEP 0: odd/even-word split-half r >= 0.7 per feature, else that feature is dropped.
 S3a positive control: Davis hands differ (permutation of hand labels across pages; eta^2). If no feature separates hands at p < 0.01,
   the ink features carry no scribe signal at page level and S3b-d are 'Inconclusive at this power'.
 S3b conjugate leaves (same sheet) closer in standardised feature space than random same-quire same-hand leaf pairs. CONFIRM p < 0.01.
 S3c recto/verso of one leaf closer than conjugates? descriptive.
 S3d smoothness along Path 1 within hand-1 herbal A sheets: mean adjacent distance vs random orderings of those sheets. CONFIRM p < 0.01.
S4 LAYOUT AS AN INDEPENDENT SIGNAL (less obvious).
 S4a same-sheet page pairs vs other same-quire same-hand page pairs: higher dice_current? (bifolium as a unit, from layout only). CONFIRM p < 0.01.
 S4b Mantel-type correlation between dice_current and eo777's combined JS distance over herbal hand-1 page pairs (permute page labels).
   |r| < 0.1 -> layout is close to independent of the text distance; report r and p.
 S4c crease leaves DAMAGED-6: mean pairwise dice vs depth x side x Currier-matched leaves (20,000). CONFIRM p < 0.01; FALSIFY p >= 0.05.
Predictions: S1 falsified (P 0.6); S2 step 0 passes, nuisance strong, chain falsified (P 0.7); S3a passes (P 0.6); S3b weak/none (P 0.6);
S4a confirmed (P 0.5); S4b small positive r; S4c falsified (P 0.75).

## AMENDMENT to S3 (07:50, before any ink statistic was computed; made after reading the ink field list only)
Added two scale-free features: ascender ratio = (mean line - ascender top)/x-height per word (words with an ascender), and ink coverage.
The registered 'writing angle' is a baseline rotation in the recorded frame, so it is confounded with page tilt (S2 found recto/verso
tilt r = -0.42 and rectos +0.12 deg vs versos -0.65 deg: a photographic side effect). It is kept but flagged.

## BOUND B1 (exploratory, written 08:00 before running): is ink coverage just glyph composition?
Coverage (ink fraction of the word box) passed split-half (r 0.80), separates Davis hands (eta^2 0.29, p 0.0002) and is smooth along Path 1
(z 3.06). But dense glyphs differ by Currier language, and Path 1 is built from glyph statistics, so this may be text again.
Bound: fit coverage ~ word length (chars) + counts of each EVA character, over all clean words (OLS); page residual = median word residual.
Rerun S3a (hands), S3b (conjugates), S3d (Path 1 stratified) on the residual. If hands eta^2 falls below p 0.01 or Path 1 z < 2,
the coverage signal is glyph composition, not ink/pen.

## RESULTS S1-S4 + B1 (2026-11-14). 12 registered tests + 1 exploratory bound. Power not measured for any (no planted-signal runs).
S2 TILT: instrument reliable (split-half r 0.957) but it is the photograph: recto vs verso of one leaf r = -0.42; rectos +0.12 deg, versos -0.65 deg.
  Chain: mean |tilt gap| 0.95 vs 0.98 ± 0.14 (z 0.25, p 0.40) -> FALSIFIED. Path 1 z 0.59; label-free 0.58; binding -0.59. No tilt carry.
S1 CHAIN, layout overlap (dice): 0.612 vs 0.603 ± 0.007 same-ten-sheets null (z 1.37, p 0.089) -> FALSIFIED (p >= 0.05); vs random Herbal A sheets z 0.27.
  Path 1 whole order, dice: z 1.62, p 0.063 (inconclusive at this power); label-free z -0.24; binding z 0.35.
  Text-based scores (circular, cannot confirm): position-constrained EVA similarity chain z 2.20 / 2.93; Path 1 z 2.79; JS Path 1 z 6.88.
S4a same-sheet page pairs, layout dice: z -0.39 (p 0.66) -> FALSIFIED. Opening-3-line text JS for conjugates z -0.01: the sheet effect is not
  visible in three opening lines. Recto/verso of a leaf: dice z 1.38, JS z 2.03 (weak).
S4b layout dice vs his JS text distance (82 Herbal A hand-1 pages, 3,321 pairs): r = -0.016 (null SD 0.064) -> layout is independent of text distance.
S4c crease leaves layout: DAMAGED-6 z -0.10 (p 0.51) -> FALSIFIED. PARTNER-6 z 2.29 (p 0.015): weak, one of ~12 tests, not confirmed.
S3 INK step 0: split-half r: x-height mm 0.55 FAIL, ascender ratio 0.35 FAIL, x-height spread 0.16 FAIL, writing angle 0.92 pass (but photographic:
  rectos +0.09, versos -0.87), coverage 0.80 pass. Only COVERAGE is a usable non-photographic feature (128 pages with >= 30 clean words).
  S3a hands: coverage eta^2 0.288 (p 0.0002): hand 1 0.377, hand 2 0.462, hand 3 0.444, hand 5 0.350. Positive control passes.
  S3b conjugates (coverage only): z 2.01, p 0.019 weak. S3d Path 1 stratified (section x hand): z 3.06, p 0.0026; label-free 1.97; binding 2.09;
  within Herbal A hand 1 only (15 sheets): Path 1 z 1.01, label-free -0.29, binding 1.58 -> nothing inside the chain's own region.
B1 BOUND (exploratory): coverage residual after removing word length + EVA character counts (model R^2 0.13; split-half r 0.83; no recto/verso
  offset): hands eta^2 0.28 (p 0.0002); within Herbal, hand 2 minus hand 1 = +0.053 (z 2.57, p 0.003); conjugate leaves closer than same-quire
  same-hand leaves: 0.029 vs 0.044 ± 0.006 (z 2.54, p 0.0036); smooth along Path 1 beyond section x hand: z 3.44 (p 0.0012);
  label-free z 2.34 (p 0.020); binding order z 0.59 (p 0.27).
  -> ANALYTICAL CLAIM, needs external test: ink coverage is a non-text, non-photographic page property that (i) differs by Davis hand,
     (ii) is shared by the two leaves of a sheet, (iii) changes gradually along eo777's order and NOT along the binding order.
  UNBOUNDED confounds: coverage = ink / box area, so it depends on script size, pen width, ink fading, parchment tone and the box-drawing
  convention; 'density' is exactly the confound that demoted the earlier hand-as-shape finding (P1). Photograph exposure not ruled out per image.
P(survives 60d): tilt = photograph 85%; chain has no layout/tilt support 70%; coverage differs by hand 60%;
  coverage shared by conjugates 40%; coverage smooth along Path 1 beyond section x hand 35%.

## PRE-REGISTRATION K1-K6 (2026-11-14 ~08:00, before computing): what is "ink coverage" and what can it tell us?
Data: vmsspatial ink_measures (clean words), bundle geometry (recorded boxes), leaves 'mm' scale, registration scale. Seeds 20261126+.
K1 DECOMPOSE. coverage = ink pixels / box area. Per word: box height mm (H), box width per character mm (Wc), ink amount per character
 (n_ink / chars, converted to mm^2 with the page scales; if the unit of n_ink cannot be established, use n_ink/chars and say so), and
 NET coverage = residual of log(coverage) on glyph counts + length + log H + log Wc (ink weight net of text and size).
 For each of H, Wc, ink-per-char, NET: split-half r (>= 0.7 to be used), then the three tests from B1
 (hand 2 vs hand 1 within herbal; conjugate leaves vs same-quire same-hand leaves; Path 1 smoothness with section x hand null).
 READING: signals in H or Wc only -> script/box SIZE. Signals survive in NET -> stroke weight or ink darkness. Threshold p < 0.01 each.
K2 PHOTOGRAPH. NET coverage for pages on the same photograph (same image_id) vs other same-quire same-hand page pairs.
 If same-photo pages are closer at p < 0.01, exposure contributes; then rerun K1 tests on NET residualised on image_id-level mean where possible.
K3 PEN PHYSICS (known-answer check that coverage measures ink load). A quill runs dry and is re-dipped, so ink weight along the writing
 sequence should be positively autocorrelated beyond what the page and line explain. Statistic: lag-1 correlation of word NET values in
 reading order within a line. Null: shuffle word order within each line (1,000). CONFIRM p < 0.01 and effect > 2 null SD.
 Also: mean NET by position in line (first..last decile) and by line number down the page (descriptive).
K4 HAND OR LANGUAGE? In the herbal pages hand 1 = Currier A and hand 2 = Currier B almost exactly, so hand and language cannot be separated
 there. Report NET by (hand x section) cells with >= 5 pages, and the hand-2 pages across sections (Herbal vs Bio) and hand 3 (Stars).
K5 GRADIENT. Within each section x hand stratum with >= 6 sheets: Spearman of sheet NET against Path 1 score; combined by Fisher/Stouffer.
 Distinguishes a monotone drift (a pen/ink wearing or a batch change) from mere local similarity.
K6 CREASE SIX: NET spread vs depth x Currier-matched sets (low power, reported only).
Predictions: K1 signal mostly in size (P 0.55); K2 some photo effect (P 0.5); K3 confirmed (P 0.6); K5 no monotone drift (P 0.6).
NOTE (08:05, before computing; from reading ink.py line 350): coverage = ink pixels / (ink width x x-height), NOT / box area. So its
denominator contains the x-height estimate, which FAILED split-half (0.55). Added variant NET_A = residual of log(ink area per character,
mm^2) on glyph counts + length + log Wc + log H, which avoids x-height entirely. NET (registered) and NET_A are both reported.

## RETRACTION (moved here for prominence; also see top): B1 claims "coverage differs by hand" and "conjugate leaves share coverage"
were mostly WRITING SIZE. After controlling size (K1): hand difference in ink weight z 2.09-2.47 (p 0.036 / 0.012), conjugates z 0.64-1.82. Neither holds at p < 0.01.

## RESULTS K1-K6 (pre-registered) + size bound (exploratory). 12,565 clean words, 128 pages, 41 sheets.
K1 WRITING SIZE (reliable: split-half 0.83 box height, 0.93 width per character):
  Herbal, hand 2 vs hand 1: box height -14% (z -5.04, p 0.0001); ink width per character -10% (z -4.67, p 0.0001).
  Size bound (glyph-adjusted width, model R^2 0.69): hand 2 is 8.5% narrower (z -4.39, p 0.0001).
  By hand x section (width vs overall): hand 1 herbal +9.3%, hand 1 pharma +5.7%; hand 2 herbal -0.4%, hand 2 bio +0.3%; hand 3 stars -4.3%, hand 3 herbal 0.0% (n=5).
  -> width is stable within hand 1 and hand 2 across sections. UNBOUNDED: width residual vs words on the page r = -0.56 (density); hand = Currier language in the herbal.
  Size is NOT smooth along Path 1 (height z 1.07; width z 1.82, p 0.038) and conjugates share width only weakly (z 2.01, p 0.022).
K1 INK WEIGHT net of glyphs and size (NET: split-half 0.83; NET_A, no x-height: 0.93):
  hand 2 heavier than hand 1 in herbal by ~11%: NET z 2.09 (p 0.036), NET_A z 2.47 (p 0.012) -> weak.
  conjugates: NET z 1.82 (p 0.036), NET_A z 0.64 (p 0.27) -> not supported.
  smooth along Path 1 (section x hand null): NET z 3.29 (p 0.0024), NET_A z 2.77 (p 0.009) -> CONFIRMED on both; label-free z 1.98 / 2.37 (weak);
  binding order z 1.01 / 0.24 (none).
K2 PHOTOGRAPH: NOT TESTABLE. 128 pages sit on 122 different images; only 2 same-photo pairs. Per-image exposure remains unbounded.
K3 PEN PHYSICS: neighbouring words on a line are more alike in ink weight than shuffled (NET_A lag-1 -0.076 vs -0.212 ± 0.019, z 7.18);
  consecutive lines z 6.67. Lag-2 is at the null. By the rule this is CONFIRMED, but it does not validate 'ink load': his binarisation
  uses a local window of 1.5 box heights, which neighbouring words share, so lag-1 similarity is partly built in. Ambiguous.
  Ink weight rises slightly down the page (top fifth -0.022, bottom two fifths about +0.01).
K4 by hand x section, ink weight NET_A: hand 1 herbal -0.039, pharma -0.038; hand 2 herbal +0.049, bio +0.092; hand 3 herbal -0.060 (n=5), stars +0.048.
  Within-hand section differences are as large as between-hand differences -> ink weight tracks section/campaign at least as much as scribe.
K5 GRADIENT: no monotone drift within strata (Stouffer z 0.79 NET, 1.13 NET_A; herbal hand 1 rho 0.24 / 0.47, n 17). The Path 1 result is local similarity, not a trend.
K6 crease six: NET z 0.11; NET_A z 1.65 (p 0.16). Nothing.
P(survives 60d): 'hand 2 writes ~9-14% smaller than hand 1 in the herbal' 65%; 'ink weight net of size is smooth along Path 1 beyond section x hand' 35%;
'ink weight is a scribe marker' 20%.
