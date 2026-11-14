## RETRACTIONS / DOWNGRADES / REVISIONS (kept at top) -- newest first
- 2026-11-14 13:50 REVISION of "the ink neither confirms nor contradicts his order" (my 10:40 summary): on the 2014 photographs ink DARKNESS is smoother
  along Path 1 than chance within section x hand (z 3.79; 3.49 after removing parchment brightness; focus/width-adjusted 2.39), and not along the binding
  order (z -0.08). The earlier "ink amount" downgrade (same-word z 2.22) stands as written; darkness is a different, calibrated measure. One campaign,
  one session, not yet demolished.
- 2026-11-14 13:20 REVISION of the 11:30 downgrade: on the independent 2014 photographs the conjugate (same-sheet) stroke-width result is z 3.73, and on
  the two-campaign mean of fore-edge-half values z 3.10. Registered SURVIVES rule met (narrowly). P(survives 60d) 30% -> 50%.
- 2026-11-14 13:20 WITHDRAWN: "down the page strokes narrow 2-3%" as a writing effect. 2004: -2.0% (z -3.1); 2014: +0.9% (z 1.5). 2004 photography only.
- 2026-11-14 13:20 CORRECTION to "along-line gradient is a scan effect": the recto/verso split carries TWO parts in both campaigns: a reading-direction
  part (+4.0% / +1.5%) and a gutter-distance part (-4.3% / -3.2%). The Bd3 rule called it scan-only; the two-campaign rule says a writing trend exists.
- 2026-11-14 11:30 DOWNGRADE: "conjugate leaves share stroke width at sheet level (z 4.25)". Stroke width on these scans reads wider towards the
  gutter (versos +8.6% along the line, z 12; rectos -0.5%). Using only the fore-edge half of each page the conjugate result is z 2.70 (null SD 0.009);
  gutter half z 4.05. Partly a page-opening artefact: both leaves of a sheet sit at the same depth in the quire. P(survives 60d) 50% -> 30%.
- 2026-11-14 11:15 VOID: Phase 4 G-A and G-B as first run (unstable sort scrambled word and line order; my bug). I had written "no gradient along
  the line" and "+2.0% down the page"; corrected values are +3.0% along the line (z 7.3, a scan effect) and -2.0% down the page (z -3.2).
- 2026-11-14 10:40 DOWNGRADE: "plain ink amount follows Path 1 (z 3.92), surviving size, width and contrast (z 3.02)". On identical word types only
  (T1) it is z 2.22 (null SD 0.054, 33 sheets); the matched-count all-word control is itself only z 2.73. Inconclusive at this power, not established.
- 2026-11-14 10:40 INSTRUMENT FAILURE (my design fault): T2 drift test was run on the manuscript without a known-answer power check first. Its measured
  power is 0-13% for a 2-SD drift (36% at best on binding). Its outputs carry no weight either way. Breaks Ed's instrument rule; logged as such.
- 2026-11-14 DARKNESS not accepted (Phase 1) and unusable on these scans (Phase 2 gate).

# Direct ink measurement on the 600 dpi SID TIFFs — running results (2026-11-14)

## RETRACTED FINDINGS
(none yet)

## Data
Drive folder "Voynich SID TIFFs (zodiackillerciphers mirror)" holds MANIFEST.csv only (file, bytes, sha256, source_url); the TIFFs are not on Drive.
Fetched from the manifest's source_url and verified against its sha256. f5r.sid.tif: 3981 x 5597 px, RGB, 600 dpi (23.62 px/mm), sha256 OK.

## PHASE 1 PRE-REGISTRATION (08:10, before building): instrument development on KNOWN-ANSWER data only. VMS text sealed.
Instrument (per word crop, luminance L): parchment level B = median of the brighter half of the crop; darkness map D = 1 - L/B
(exposure-gain invariant); ink mask = D > half of the crop's 98th-percentile D (half-maximum rule, so width should not depend on darkness);
stroke width = 2 x median distance-transform value on the mask's skeleton; ink darkness = median D on the skeleton.
Known-answer data: synthetic strokes of known width w and darkness d drawn on real blank-parchment patches (margins only, no text),
blurred (sigma 1 px), with noise, then exposure gain g applied.
ACCEPT only if, on the development set:
 A1 width: Spearman(measured, true w) >= 0.95, and measured width changes by < 10% of its range across the darkness levels at fixed w.
 A2 darkness: Spearman(measured, true d) >= 0.95, and changes by < 10% of its range across widths at fixed d.
 A3 gain invariance: both measures change by < 5% between g = 0.8 and g = 1.2.
Then HELD-OUT control pair, never used in tuning: (i) a different, darker/stained parchment patch from another page; (ii) blur sigma 1.6
(softer focus). Same A1-A3 thresholds, plus: report the bias each causes. Every variant tried gets logged below.

## PHASE 1 LOG
v1 (sha 189dda26...): DEV pass (width rho 0.990, darkness rho 0.986, cross-talk 2.0% / 2.4%, gain 0.16% / 0.09%).
   HELD-OUT (i) other parchment (f1r margins, darker): PASS (0.989 / 0.986; cross-talk 2.3% / 2.2%).
   HELD-OUT (ii) blur sigma 1.6: FAIL on A2 (darkness cross-talk with width 13.9% > 10%): under soft focus thin strokes read lighter.
   -> v1 REJECTED by the rule (must pass the whole held-out pair).
v2 plan (08:20): darkness is read only in the core of thick stroke parts (skeleton pixels whose distance-transform value >= 3 px,
   i.e. local stroke width >= 6 px); words with < 8 such pixels return no darkness. Width unchanged.
   Blur sigma 1.6 and f1r are now development data. NEW held-out pair for v2: (i) parchment from a third page (f58r);
   (ii) blur sigma 2.0 combined with gamma 0.9 (a non-linear tone change). Same thresholds.
v2 (sha 2f9116c7...): DEV sigma 1 pass; f1r pass; sigma 1.6 FAIL again (darkness cross-talk 13.1%). Core threshold of 3 px is not enough. REJECTED.
v3 plan (08:25): core threshold 4 px (darkness read only where the local stroke is >= 8 px = 0.34 mm wide). Domain of validity is
   therefore strokes >= 8 px; words without such strokes return no darkness. Held-out pair for v3 unchanged (f58r parchment; blur 2.0 + gamma 0.9).
   Consequence to check in Phase 2 BEFORE any hypothesis test: the share of real words that yield a darkness value. If < 50%, darkness is
   declared unusable and only stroke width is analysed.
v3 (sha 233bb3b7612c20bec5dc539fc142d8e45feed7a361261e0e29e0117e7fcf4002): DEV all three pass (sigma 1: cross-talk 2.0% / 1.3%; sigma 1.6: 2.9% / 6.3%; f1r: 2.0% / 1.5%).
   HELD-OUT (i) third parchment (f58r margins): PASS (width rho 0.991, darkness rho 0.986, cross-talk 1.8% / 0.8%, gain 0.6% / 1.4%).
   HELD-OUT (ii) blur sigma 2.0 + gamma 0.9: FAIL on A2 (darkness cross-talk with width 15.5%); width still passes (rho 0.984, cross-talk 3.3%,
   gain 0.05%) but its slope drops to 0.82 (soft focus compresses width differences).

## PHASE 1 VERDICT (2026-11-14 08:35)
- DARKNESS: NOT ACCEPTED. Three variants, each failed a held-out soft-focus control (13.9%, 13.1% in dev, 15.5%). Under blur a thin stroke
  reads lighter than a thick stroke of the same ink, so darkness cannot be separated from stroke width unless focus is known per page.
  It is valid only if page focus is constant and close to sigma <= 1 px, which has not been measured.
- STROKE WIDTH: passed A1 and A3 on every development and held-out set (5 sets; rho 0.984-0.991; cross-talk with darkness 1.8-3.3%;
  exposure-gain change <= 1.1%). Its code has not changed since v1.
  DEVIATION FROM THE REGISTERED RULE: acceptance was written as joint (width AND darkness). Accepting width alone is a post-hoc split.
  Flagged; Ed to decide whether that is acceptable before Phase 2.
- Not yet done: no manuscript word has been measured. Phase 2 (freeze + hash, VMS decision rule with planted-signal power, then run) not started.
Files: instrument.py (v3), phase1.py, phase1_dev.json (v1), phase1_heldout.json (v1), phase1_v3_dev.json, phase1_v3_heldout.json.
P(width instrument is fit for purpose on real words) = 60% (synthetic strokes are simpler than real glyphs: no hairlines, no ink pooling, no show-through).

## REGISTRATION CHECK (2026-11-14, after Ed's "Proceed logically") -- no hypothesis test, plumbing only
reg.py (v1: whole-darkness map, free scale 3.2-4.3): FAILED on f5r and f1r (scale at grid edge, offsets of 1000+ px; plant paint dominated).
reg2.py (v2: thin-stroke map with paint suppressed, line-structured template, scale within +-4% of 23.62 / page px-per-mm):
 f5r scale 3.718 offset (8,8) px; f1r 3.826 (0,4); f58r 3.786 (4,8). Ink inside boxes vs boxes shifted half a line: 1.36x, 1.30x, 1.19x.
 Visual check of overlays (ov_f1r.png, ov_f58r.png): boxes sit on the words. The SID TIFF is the recorded frame enlarged ~3.7-3.8x, near-zero offset.
 Limits: only 3 pages eyeballed; boxes are Voynichese-project word boxes, some overlap neighbours and include descenders from the line above.

## PHASE 1b PRE-REGISTRATION: sharpness (edge-softness) gauge -- written before any gauge code is run
Purpose: per-page covariate for focus, and gate for darkness. Known-answer data only (synthetic strokes on real blank parchment, as Phase 1).
Gauge: sigma_hat = stroke peak darkness / (2.5066 x steepest edge gradient), page value = median over words.
ACCEPT if, at "page" level (median of 30 synthetic words per condition):
 S1 Spearman(sigma_hat, true blur sigma in {0.8,1.0,1.3,1.6,2.0,2.5}) >= 0.95;
 S2 at fixed sigma, sigma_hat moves by < 15% of its range across stroke widths {6,8,10,12} and across darkness {0.3,0.5,0.7};
 S3 gain 0.8 vs 1.2 changes it by < 5%.
HELD-OUT pair (not used in tuning): f58r parchment; gamma 0.9. Same thresholds.
FALSIFIED (gauge rejected) if any of S1-S3 fails on the held-out pair. Then: no focus covariate, darkness stays unusable, width is analysed
with the caveat that soft focus compresses width differences (slope 0.82 at sigma 2).
Limit known in advance: on real ink the gauge reads focus PLUS ink feathering at stroke edges; it cannot separate them.

## PHASE 1b LOG (sharpness gauge)
s1 (smooth 0.5, max-filter 5): DEV f5r pass (width 13.6%, darkness 13.0%); DEV f1r FAIL (darkness cross-talk 16.1% > 15%). Rejected.
s2 (smooth 1.0, max-filter 3): darkness cross-talk fixed (3.4-3.8%) but width cross-talk 19.7-20.3%. FAIL. Rejected.
s3 (s2 + analytic correction for finite stroke width, solved per word): DEV pass on both (rho 0.980 / 0.979; width 14.3% / 14.1%; darkness 3.3% / 3.5%; gain < 0.1%).
   sha256 c927ca9e544bf478760b4108ca1dd089cf8341226c91d10d9fd8990efcb3b471
   HELD-OUT f58r parchment: PASS (rho 0.979; width 13.3%; darkness 3.2%; gain 3.2%). HELD-OUT gamma 0.9: PASS (rho 0.979; 14.1%; 3.7%; 3.0%).
   Readings for true sigma 0.8/1.0/1.3/1.6/2.0/2.5: 1.07/1.24/1.50/1.79/2.13/2.55 (reads ~0.2 high at the sharp end; monotone).
VERDICT: gauge s3 ACCEPTED as a per-page covariate. Margin is thin: width cross-talk 13-14% against a 15% limit, three variants tried.
P(gauge tracks real focus on manuscript pages) = 50% (real edges also carry ink feathering; untested on real ink).

## PHASE 2 PRE-REGISTRATION (2026-11-14) -- written BEFORE any manuscript word is measured
Instruments frozen: instrument.py v3 sha 233bb3b7...4002 (stroke width; darkness under gate), sharp.py s3 sha c927ca9e...b471, reg2.py and
measure_pages.py (hashes logged at run start). No edits after this point; any edit = new version, new pre-registration.
DEVIATION carried forward: width accepted alone although the Phase 1 rule was joint. Ed replied "Proceed logically"; not an explicit ruling.
DATA: single-page SID TIFFs (600 dpi) from zodiackillerciphers.com/voynich-sid-tif. Fold-outs/multi-panel files excluded. sha256 of every
 file I download is logged (tif_sha256.csv) for comparison with Ed's Drive MANIFEST; I do NOT check against the manifest in this run (deviation).
WORDS: Voynichese word boxes (vmsspatial recorded frame), only tokens inside eo777 ink units with no bad flag (same mask as K1-K6). Crop = box + 6 px.
PAGE EXCLUSION: registration ratio (ink in boxes / boxes shifted half a line) < 1.10, or |offset| > 80 px, or < 20 measured words.
MEASURES (page = mean over words of the word-level value; sheet = mean of its pages):
 W_net  PRIMARY: log stroke width, residual on glyph counts of the word (word-level OLS over all pages).
 W_raw  sensitivity: log stroke width, unadjusted.
 W_adj  sensitivity: W_net further residualised at page level on sigma_hat (focus gauge) and log box height (writing size).
 N_frac NAIVE BASELINE: share of crop pixels darker than 25% below background (no skeleton, no width model), glyph-adjusted the same way.
GATES (if either fails, STOP: instrument fails on real ink, no hypothesis result is reported as evidence):
 G1 coverage: >= 50% of eligible words return a width.
 G2 reliability: odd/even-word split-half correlation of page W_net, Spearman-Brown corrected, >= 0.80.
DARKNESS GATE: analysed only if >= 50% of words return darkness AND >= 80% of pages read sigma_hat <= 1.79 (= true blur 1.6, the largest blur
 at which darkness passed). Otherwise darkness is declared unusable on these scans. If analysed: secondary, same tests, adjusted for sigma_hat.
HYPOTHESES (unit = sheet; 10,000 permutations; seed 20261114):
 H1 Hand: herbal sheets, hand 2 vs hand 1 mean W_net. Null: shuffle hand labels over herbal sheets. Two-sided.
    CONFIRM: |z| >= 3 on W_net AND same sign with |z| >= 2 on W_adj.  FALSIFY: |z| < 2 on W_net AND planted 5% width shift detected >= 80%.
    Otherwise: inconclusive at this power.
 H2 Conjugates: mean |difference| between the two leaves of a sheet vs leaves re-paired at random within section x hand. Lower = signal.
    CONFIRM: z >= 3 on W_net and >= 2 on W_adj.  FALSIFY: z < 2 AND planted sheet effect (SD = half the leaf-level SD) detected >= 80%.
 H3 Order: mean |difference| between neighbouring sheets along (a) Path 1, (b) label-free order, (c) binding order; null = shuffle sheets
    within section x hand. Lower = signal. CONFIRM for an order: z >= 3 on W_net and >= 2 on W_adj.
    FALSIFY for an order: z < 2 AND planted linear drift (end-to-end = 1 within-stratum SD) detected >= 80%.
    "Path 1 beats binding" may be said only if Path 1 confirms and binding has z < 2; that is an analytical claim needing external test.
 Naive baseline rule: if N_frac gives the same or stronger result than W_net on a test, the finding is about "amount of ink", not stroke width.
POWER: planted effects are added to label-shuffled real page values (500 plants each); detection = same threshold as CONFIRM (z >= 3).
Known untestable confound: scanning batch/session (no metadata). A hand or order effect could be a scan-session effect.

Frozen hashes at run start:
233bb3b7612c20bec5dc539fc142d8e45feed7a361261e0e29e0117e7fcf4002  instrument.py
c927ca9e544bf478760b4108ca1dd089cf8341226c91d10d9fd8990efcb3b471  sharp.py
c310169f789cb2cc8c8c8246a3aa45c9c30f815182e319402d1883bf7edfd4e8  reg.py
5b137046c6ea806b762dff9552a3e7d0b317538dc21e8f96e03058a915b270f6  reg2.py
87d7d54671029ad1546a9e15590bea54e8eb79f58cf1bc81c4cb6afffea5519a  measure_pages.py
Plumbing fix before any word was measured (ink units are column lists, not row dicts); measure_pages.py new sha: 71cebbd4b864c78a5d9ba09b715a4245be3c6bc3477bcd81eae9cbc53d3730b0

## PHASE 2 RUN (2026-11-14) -- results as registered
187 single-page TIFFs measured (0 errors). 16 failed the registration rule (mostly dense versos in the Stars/Pharma quires: f102v1 f102v2 f103v f104v
f105v f107v f108v f35r f36r f38r f41r f65r f87v f89v2 f90v1 f95v1); 30 more had < 20 clean words or are not in the physical map.
ANALYSED: 141 pages, 10,488 words. Median stroke width 0.263 mm (5th-95th pct of pages 0.197-0.302). Word-level width is quantised (~1 px = 0.04 mm).
GATES: G1 coverage 100% PASS. G2 split-half reliability (Spearman-Brown) W_net 0.977 PASS (W_raw 0.976, N_frac 0.989). Recto/verso r = 0.50.
DARKNESS GATE: FAIL (only 63% of pages read sigma_hat <= 1.79; needed 80%). Darkness declared UNUSABLE on these scans, as registered.
 Page sigma_hat median 1.73 (5-95%: 1.40-2.02). Glyph composition explains 1.9% of word-level width variance.
H1 hand (herbal, 24 sheets hand 1 vs 5 sheets hand 2): INCONCLUSIVE AT THIS POWER. Hand 2 strokes +14.8% wider; effect 0.138 log units, null SD 0.070,
   ratio 1.95 (W_adj 2.09; naive N_frac 2.08). Power measured: a planted 15% shift is detected 5% of the time at z >= 3. Neither CONFIRM nor FALSIFY met.
H2 conjugates (37 sheets): CONFIRM criterion met. Mean |difference| 0.077 vs null 0.126, null SD 0.012, z 3.97 (W_adj 3.67).
   NAIVE BASELINE gives the same (N_frac z 3.81) -> by the registered rule this is about amount of ink, not stroke width specifically.
   Power for the planted effect was 3% (planted effect was set too small; my design fault) -- irrelevant since the criterion was met.
H3 order (48 sheets), stroke width W_net: Path 1 z -0.11; label-free z -1.88; binding z 1.77 (W_adj 2.24). None confirms.
   FALSIFY not available: planted 1-SD drift detected 0%, 3-SD drift 1.5-7.5%. The adjacent-difference test is blind to slow drift. INCONCLUSIVE.
   NAIVE BASELINE N_frac: Path 1 z 3.92 (null SD 0.020), label-free z 2.78, binding z -0.41.
   -> stroke width does NOT carry an order signal; plain ink amount does, along Path 1, on scans independent of eo777's ink data
      (his earlier NET result: z 3.29). N_frac was registered as a baseline, not a hypothesis: treat as awaiting bounding.
P(survives 60d): H2 conjugates share ink amount 45%; H1 hand difference 30%; "stroke width has no order signal" 55% (low power); N_frac along Path 1 35%.

## BOUNDING TESTS PRE-REGISTRATION (written before running; post-hoc relative to Phase 2, labelled as such)
B1 Is H2 sheet-specific or quire-level? Same statistic, null = re-pair leaves within the same quire. SHEET-SPECIFIC if z >= 2 on W_net; QUIRE-LEVEL
   (or scan-session level) if z < 2. Same for N_frac.
B2 What is the N_frac order signal? Page-level N_frac residualised step by step on (a) writing size (log box height, log box width per glyph),
   (b) + stroke width W_net, (c) + ink contrast (page median of the word 'peak' = 98th pct darkness). Smoothness along the three orders after each step.
   SURVIVES if z >= 2 on Path 1 after (a)+(b). If it falls below 2 only when (c) is added -> the signal is ink contrast against parchment
   (real ink darkness, parchment tone or focus -- not separable here). If it is gone after (a) -> it was writing size/density.
   Also: contrast itself along the three orders. 3,000 permutations, seed 20261115.

## BOUNDING RESULTS (2026-11-14)
B1: conjugate sharing is SHEET-SPECIFIC by the rule. Null = re-pair within quire (35 sheets, 9 quires): W_net obs 0.071 vs 0.109, null SD 0.009,
    z 4.25; N_frac z 4.09. (Within section x hand on the same 35 sheets: 4.04 / 3.55.)
B2: N_frac along Path 1 SURVIVES by the rule: raw z 4.07; after writing size 3.09; + stroke width 3.36; + ink contrast 3.02 (null SD 0.011-0.019).
    Label-free order: 2.75 / 2.55 / 3.10 / 1.66 (falls below 2 once contrast is added). Binding order: -0.40 / 0.14 / -0.41 / 1.40.
    Contrast alone: Path 1 z 2.37, label-free 2.18, binding -0.58. Focus gauge alone: 1.20 / 0.08 / 0.94. Writing size alone: 1.36 / 1.65 / 0.75.
    Size + width + contrast explain 78% of page-level N_frac; the Path 1 signal sits in the remaining 22%, which I cannot name.
    N_frac page correlations: contrast 0.83, stroke width 0.44, box height -0.42, focus gauge -0.40.
NOT YET BOUNDED (next): circularity. Path 1 is built from glyph statistics and N_frac is only adjusted for glyph COUNTS. Test = same word type only
 (e.g. daiin, ol, chedy): does ink amount of the identical word still vary smoothly along Path 1? Until then the N_frac order result is not load-bearing.
Analytical claims flagged (need external test): "sheets were written at one sitting"; "ink amount tracks writing order".
P(survives 60d): conjugates share stroke width at sheet level 50%; N_frac along Path 1 beyond size/width/contrast 30%.
Files: measure_pages.py, analysis2.py, bound2.py, reg2.py, sharp.py, instrument.py, phase2_results.json, bound2_results.json, phase2_pages.csv,
 phase2_sheets.csv, phase2_registration.csv, tif_sha256.csv, pages/*.pkl (one per page), phase2.pkl.

## PHASE 3 PRE-REGISTRATION (2026-11-14 10:20, after Ed: "half a job") -- written before running
Instruments and page measurements unchanged (pages/*.pkl, phase2.pkl). Seed 20261116. Unit = sheet. Null = shuffle within section x hand unless stated.
T1 SAME-WORD (circularity bound for the N_frac order result). Take the 30 commonest word types. For each token: log n_frac, residualised WITHIN
   word type on log box height and log box width, then z-scored within type. Page = mean (pages with >= 5 such tokens). Smoothness along Path 1,
   label-free, binding. CONTROL: all-word N_frac (size-adjusted the same way, glyph-count-adjusted) on a random subsample with the same token count
   per page, 20 draws, median z.
   SURVIVES (not circular) if same-word z >= 3 on Path 1. CIRCULAR if same-word z < 2 while the matched-count control keeps z >= 3.
   If both fall below 3: inconclusive at this power (token loss, not circularity).
T2 DRIFT-SENSITIVE ORDER TEST. Statistic = |Spearman| between sheet value (stratum mean removed) and position in the order. Measures: W_net,
   N_frac net of size+width. Three orders. Power: planted linear drift, end-to-end 1 and 2 within-stratum SD, 200 plants x 300 perms, detection z >= 3.
   CONFIRM z >= 3. FALSIFY z < 2 AND 1-SD drift detected >= 80%. Otherwise inconclusive at this power.
T3 CREASE SIX on ink (A:4-5, B:10-15, C:20-21, D:25-32, D:28-29, E:36-37). Statistic = mean pairwise |difference| among the six, for W_net and
   N_frac net of size+width. Null = 10,000 random sets of six sheets with the same section x hand make-up. Lower = signal.
   CONFIRM z >= 3. z < 2 = no ink support for the six being written together (power: planted shared offset of 1 within-stratum SD, 300 plants).

## PHASE 3 RESULTS (2026-11-14 10:40)
T1 same-word (30 commonest types, 2,441 tokens, 82 pages, 33 sheets; median 18 tokens/page): Path 1 z 2.22 (obs 0.663, null 0.782, null SD 0.054);
   label-free 1.55; binding 0.07. Matched-count all-word control: Path 1 median z 2.73 (range 1.51-3.99); label-free 2.69; binding -0.10.
   All words on the same 82 pages: 2.88 / 2.94 / -0.27. RULE OUTCOME: both below 3 -> INCONCLUSIVE AT THIS POWER (token and sheet loss). Not shown circular,
   not shown real. The common word types sit mostly in B-language pages, so 15 of 48 sheets drop out.
T2 drift test: Path 1 W_net z -1.64, N_frac(net) 2.68; label-free 0.19 / -1.40; binding -0.39 / 0.63. Power for 1-SD drift 0-5.5%, 2-SD 0-36%.
   INSTRUMENT FAILED its power requirement; no conclusion drawn.
T3 crease six (all six are herbal hand-1 sheets; pool 24): stroke width mean pairwise gap 0.116 vs 0.161, null SD 0.043, z 1.02; ink amount (net)
   0.141 vs 0.206, null SD 0.074, z 0.89. Planted shared offset detected 4-6%. INCONCLUSIVE AT THIS POWER: no ink support, and none could have been found.
OVERALL: with 48 sheets split over section x hand, and 6 creased sheets among 24, these scans cannot decide the order or crease questions by ink.
 The limit is sample size at sheet level, not the measurement (page reliability 0.977).
P(survives 60d): conjugates share stroke width at sheet level 50% (unchanged); ink amount follows Path 1 20% (was 30%).

## PHASE 4 PRE-REGISTRATION: ORDER-FREE GRADIENTS (2026-11-14 10:45, Ed: "tests on the rest of the manuscript irrespective of what he's saying")
No eo777 order used anywhere. Data = the frozen Phase 2 word measurements (10,488 words, 141 pages). Seed 20261117.
Measures: W_net (stroke width, glyph-adjusted) and N_net (ink amount, glyph- and size-adjusted at word level).
Each test gets a planted-signal power check on label-shuffled real data BEFORE its real result is read (script prints power first).
G-A ALONG THE LINE: slope of value on position in line (0 = first word, 1 = last), line means removed. Null: shuffle words within line. Plant 2% and 5%.
G-B DOWN THE PAGE: slope of line mean on line position (0 top, 1 bottom), page means removed. Null: shuffle lines within page. Plant 2% and 5%.
G-C WORD TO WORD: correlation between neighbouring words in a line (line means removed). Null: shuffle within line. (pen/ink carry-over, e.g. dip cycle)
G-D PAGE-PAIR CLASSES: mean |difference| of page values for (1) recto/verso of a leaf, (2) facing pages (verso of leaf n, recto of n+1, same quire),
    (3) conjugate pages on the same face of the flat sheet, (4) conjugate pages on opposite faces. Reference: 10,000 random sets of same-quire page
    pairs of equal count. This asks what the unit of writing was: facing-similar = written in the bound/nested book; same-sheet-similar and
    facing-not = written on flat sheets.
G-E THROUGH THE BOOK: Spearman of leaf value (section x hand mean removed) against folio number. Null: shuffle leaves within section x hand.
    Plant end-to-end drift of 1 and 2 within-stratum SD.
RULE for every test: GRADIENT EXISTS if |z| >= 3. NO GRADIENT (at the planted size) if |z| < 2 AND that planted size was detected >= 80%.
Otherwise: inconclusive at this power. G-D has no plant; it is read as z per class.

## PHASE 4 RESULTS (2026-11-14 11:00) -- power printed with each result
G-A along the line: stroke width +0.48% first-to-last word, null SD 0.0039, z 1.23; planted 2% caught 99% -> NO GRADIENT of 2% or more.
    Ink amount -0.07%, z -0.06; planted 2% caught 28%, 5% caught 100% -> NO GRADIENT of 5% or more.
G-B down the page: stroke width +2.0% top-to-bottom, effect 0.0201, null SD 0.0064, z 3.14 -> GRADIENT EXISTS by the rule (barely; 141 pages, 2,990 lines).
    Ink amount -3.1%, null SD 0.0152, z -2.03; planted 5% caught 54% -> inconclusive at this power.
G-C neighbouring words: width correlation -0.135 vs -0.284 expected, null SD 0.012, z 12.25; ink amount z 7.06 -> neighbours are more alike than
    other words in the same line. AWAITING BOUNDING: neighbouring crops overlap, and local focus varies smoothly.
G-D page pairs vs random same-quire pairs (z; positive = more alike): width: recto/verso 0.86, facing -0.12, conjugate same face 1.68, opposite face 1.47.
    Ink amount: 1.91, 1.29, 2.31, 0.78. Nothing reaches 3 -> inconclusive at this power (45-56 pairs per class). Facing pages show no width similarity.
    NOTE this is weaker than Phase 2 H2 (leaf-level, z 3.97-4.25): different unit (pages) and a reference that includes same-sheet pairs.
G-E through the book (79 leaves): width rho 0.026, null SD 0.078, z 1.29; ink amount z 1.37. Planted 2-SD drift caught 24% / 60% -> inconclusive at this power.

## PHASE 4 BOUNDING PRE-REGISTRATION (before running)
Bd1 (G-B): repeat with stroke width adjusted for the word's own focus reading (sigma_hat), and test sigma_hat itself down the page.
    Still "exists" if z >= 3; "weakened" 2-3; "explained by focus" < 2.
Bd2 (G-C): correlation at lag 2 (word and next-but-one; crops cannot share strokes), raw and focus-adjusted. REAL local continuity if lag-2 z >= 3.

## PHASE 4 CORRECTION + BOUNDING (2026-11-14 11:15)
ERROR FOUND (mine): phase4.py re-sorted with an unstable sort, scrambling word order within lines and line order within pages (only 0.7% of pages
 kept ascending line order). G-A and G-B numbers above are VOID. G-C, G-D, G-E did not use that sort. Line index checked against box y: runs top to bottom.
CORRECTED (stable sort; phase4_fixed_results.json):
 G-A along the line: stroke width +3.0% first-to-last word, effect 0.0297, null SD 0.0040, z 7.45 -> GRADIENT EXISTS. Ink amount -0.08%, z -0.11
     (5% detectable 100%, 2% only 28%) -> no gradient of 5% or more.
 G-B down the page: stroke width -2.0% top-to-bottom, effect -0.0204, null SD 0.0065, z -3.18 -> GRADIENT EXISTS (barely). Ink amount +2.4%, z 1.55 -> inconclusive.
Bd1: down-page width after focus adjustment -3.1%, z -5.02 (stronger). Focus reading itself falls 3.2% down the page, z -6.91.
Bd2: neighbour correlation lag 1 z 12.3 (focus-adjusted 14.8) but lag 2 z -2.15 and lag 3 z -5.17. By the rule: local continuity NOT shown; lag-1 is
     compatible with overlapping crops, and the negative longer lags are what a trend along the line produces.
NEXT BOUNDING, registered now (Bd3, G-A): split recto / verso (gutter is left on rectos, right on versos), drop first and last word of each line,
 and focus-adjust. WRITING effect if same sign with |z| >= 3 on rectos AND versos and after dropping edge words. SCAN/PAGE-CURVE effect if the sign
 flips between recto and verso.

## Bd3 RESULT (2026-11-14 11:25): the along-line gradient is a SCAN / PAGE-CURVE effect by the registered rule
Rectos: -0.5%, null SD 0.0050, z -0.97. Versos: +8.6%, null SD 0.0069, z 11.91. Edge words dropped: rectos -2.0% (z -2.92), versos +6.2% (z 6.05).
Sign flips with the side of the gutter: strokes read WIDER TOWARDS THE GUTTER. Focus reading also changes along the line (-1.6%, z -5.5) and box
height +2.6% (z 4.2). By hand (not a hand effect, hands differ in recto/verso mix): hand 1 +9.0%, hand 2 +3.1%, hand 3 +0.1%.
CONSEQUENCE: stroke width on these scans is contaminated by distance from the gutter. That threatens the Phase 2 conjugate finding, because both
leaves of a sheet sit at the same depth in the quire and so open equally flat.
Bd4 registered now: recompute leaf stroke width from the fore-edge half of each page only (words on the side away from the gutter), rerun the
conjugate test with the within-quire null. SURVIVES if z >= 3. ARTEFACT-COMPATIBLE if z < 2. Also: gutter-half only, for comparison.

## Bd4 RESULT (2026-11-14 11:30): conjugates, within-quire null, 35 sheets
All words z 4.23-4.29; fore-edge half only z 2.68-2.70 (gap 0.086 vs 0.110, null SD 0.009); gutter half z 4.05-4.19. Rule: neither SURVIVES (>= 3) nor
ARTEFACT-COMPATIBLE (< 2): weakened, inconclusive at this power.
PHASE 4 SUMMARY: order-free gradients in INK that hold up: none yet. Stroke width along the line = scan/gutter effect. Down the page (-2 to -3%,
z -3.2 to -5.0) cannot be separated from page curve/focus (focus reading itself falls 3.2% down the page, z -6.9). Ink amount: no gradient along the
line (5% detectable), inconclusive down the page and through the book. Page-pair classes and through-the-book trend: inconclusive at this power.
What is established instead: these 2004 SID scans carry a gutter-distance bias in apparent stroke width of several percent.
Consolidated code: phase4_bounding.py (replaces the three inline runs; reproduces them to the second decimal of z). Table: phase4_results_table.csv.
P(survives 60d): gutter bias in stroke width 80%; down-page narrowing is real writing 20%.

## PHASE 5 PLAN (2026-11-14 11:05, Ed: "Do what you must. There are many different versions of the images")
Image sets found (voynich.nu gallery, read as search highlight only, not the full page): (1) 2004 Beinecke scans (MrSID; the set used so far);
(2) 2014 Yale re-digitisation, full manuscript, served by Yale IIIF (f5r = 2636 x 3723 px, ~15.8 px/mm); (3) 2014 Lazarus multispectral, 10 pages only.
Use (2) as an INDEPENDENT photographic campaign. Logic: ink properties must agree between campaigns; page-curve effects need not.
## PHASE 1c PRE-REGISTRATION (known-answer only): does instrument v3 stroke width work on 15.8 px/mm images enlarged 1.5x (bicubic)?
Synthetic strokes (true width 5-12 px at 600 dpi scale) on real blank parchment, reduced to 2/3 size, then enlarged 1.5x, then measured.
ACCEPT if width Spearman >= 0.95, cross-talk with darkness < 10%, gain change < 5% (same A1/A3 as Phase 1) on f5r tiles AND on held-out f58r tiles
with extra blur sigma 1.0 before reduction. FAIL -> the 2014 set cannot be used with this instrument; stop.
PHASE 1c RESULT: PASS. Dev f5r: rho 0.987, cross-talk 1.9%, gain 0.4%, slope 0.94. Held-out f58r + extra blur: rho 0.988, 1.3%, 1.8%, slope 0.92.
 Stroke width from the 2014 images enlarged 1.5x is accepted (slightly compressed scale; fine for comparisons within the set).

## PHASE 5 PRE-REGISTRATION (before any 2014 image is measured)
2014 images: Yale IIIF full-size PNG per page, luminance, enlarged 1.5x bicubic, instrument v3 and gauge s3 unchanged. Word boxes placed with eo777's
recorded->Beinecke homography (registrations.json.gz, v4), not my FFT fit. Same word mask, same crop padding, same page exclusion rules.
Both campaigns are then processed by ONE loader (same glyph adjustment, same size adjustment for ink amount).
P5-1 AGREEMENT: page-level correlation between campaigns for W_net and N_net. INK-LIKE r >= 0.70; SCAN-DOMINATED r < 0.40; between = mixed.
P5-2 ALONG THE LINE, split into a = (recto slope + verso slope)/2 [reading direction: the scribe] and b = (recto - verso)/2 [distance from gutter:
     the page curve]. WRITING TREND if a has the same sign and |z| >= 3 in BOTH campaigns. Power: planted a = 2%.
P5-3 DOWN THE PAGE: WRITING TREND if same sign and |z| >= 3 in both campaigns.
P5-4 CONJUGATES (within-quire null): SURVIVES if z >= 3 on 2014 alone AND z >= 3 on the two-campaign mean of fore-edge-half values. < 2 on either = not supported.
P5-5 On the two-campaign page mean: page-pair classes (recto/verso, facing, same sheet) and through-the-book trend, same rules as Phase 4.
Any result seen in one campaign only is treated as a property of that campaign's photography.
measure_bein.py sha a55c2ab1ad1da9b998d8b8ca0c0a5b3c75f0350c1f1ab8b97be0febafc315784

## PHASE 5 RESULTS (2026-11-14 13:20) -- two independent photographic campaigns
2014 set: 187 pages measured, 182 pass registration, 151 analysed, 11,309 words. 2004: 141 pages, 10,488 words. In both: 138 pages.
Median stroke width 0.263 mm (2004) vs 0.213 mm (2014): ABSOLUTE width is not comparable across campaigns (2004 is blurrier: focus reading 1.70 vs 1.08).
P5-1 AGREEMENT: stroke width r = 0.755 (within section x hand 0.771) -> INK-LIKE (>= 0.70). Ink amount r = 0.909 (0.916) -> INK-LIKE.
P5-2 ALONG THE LINE, stroke width: reading-direction part a = +3.95% (null SD 0.0041, z 9.42) in 2004 and +1.52% (null SD 0.0036, z 4.27) in 2014;
     planted 2% caught 98% in each -> WRITING TREND by the rule. Gutter part b = -4.3% (z -10.1) and -3.2% (z -8.8): present in both.
     Raw: rectos -0.5% / -1.7%, versos +8.6% / +4.8%. The a/b split assumes the two parts simply add; size of a differs 2.6x between campaigns.
     Ink amount: a = -0.35% (z -0.44) and +0.27% (z 0.52; planted 2% caught 79%) -> no trend shown.
P5-3 DOWN THE PAGE: stroke width -2.0% (z -3.12) in 2004, +0.9% (z 1.53) in 2014 -> NOT a writing trend. Ink amount +8.0% (null SD 0.015, z 5.07) and
     +2.8% (null SD 0.010, z 2.83): same sign, second under 3 -> inconclusive at this power (planted 2% caught 5% / 17%).
P5-4 CONJUGATES, within-quire null. Stroke width: 2014 alone z 3.73 (gap 0.063 vs 0.095, null SD 0.0085, 37 sheets); 2004 z 4.28; two-campaign mean of
     fore-edge halves z 3.10; of all words z 4.42 -> SURVIVES. Ink amount: 2014 z 4.17; fore-edge-half mean z 2.34 -> not met on the second criterion.
P5-5 two-campaign mean. Page pairs (z): stroke width recto/verso 1.80, facing 0.19, same sheet same face 2.26, opposite face 1.64; ink amount 1.99, 1.53,
     2.25, 0.63 -> none reaches 3. Through the book (78 leaves): stroke width rho 0.03 (null SD 0.08, z 1.12); ink amount rho 0.11 (z 1.38); planted 2-SD
     drift caught 100% -> NO book-long drift of that size in binding order within section x hand.
P(survives 60d): pages differ in real stroke width/ink amount (cross-campaign r) 80%; sheets share stroke width 50%; strokes widen along the line 40%
 (additivity assumption, size mismatch); no book-long drift >= 2 SD in binding order 65%.
NOT DONE: darkness on the 2014 set (it is sharp enough: focus reading 1.08) -- needs its own known-answer calibration on enlarged images first.
Files: measure_bein.py, analysis5.py, phase1c.py, phase5_results.json, phase5_results_table.csv, pages_bein/*.pkl, phase5.pkl.

## PHASE 1d PRE-REGISTRATION (2026-11-14 13:15, Ed: "Run it"): ink darkness on the 2014 photographs -- known-answer only
Darkness values already sit in pages_bein/*.pkl (instrument v3 wrote them) but have NOT been read or analysed. They stay sealed until Phase 2d.
Test: synthetic strokes (true darkness 0.3-0.8, width 5-12 px) on real parchment, reduced to 2/3, enlarged 1.5x, measured with v3 darkness.
ACCEPT if Spearman(measured, true) >= 0.95, cross-talk with stroke width < 10% of range, gain 0.8 vs 1.2 change < 5%, on dev (f5r tiles) AND on the
held-out pair: (i) f58r tiles with extra blur 0.5; (ii) f58r tiles with gamma 0.9. Extra blur 1.0 is also reported, to find where it breaks.
The focus-gauge reading under each condition is logged so the pass region can be mapped onto real pages.
Non-outcome checks allowed before unsealing: share of words that return a darkness value, and the page focus readings.
PHASE 1d RESULT: v3 darkness PASSES on the enlarged-image path. Dev: rho 0.986, cross-talk 2.0%, gain 0.2%. Held-out blur 0.5: 0.986 / 2.4% / 0.6%.
 Held-out gamma 0.9: 0.986 / 1.6% / 1.2%. Extra blur 1.0: 0.986 / 5.0% / 0.5% (still passes). Gauge reading across these conditions 1.43-1.78.
 Non-outcome checks on the 2014 pages: 100% of words return a darkness value; page focus readings 5%/50%/95%/max = 0.86 / 1.14 / 1.47 / 1.74,
 i.e. every page is sharper than the blurriest condition that passed. Darkness ACCEPTED for the 2014 set. (sha of instrument unchanged.)
 What "darkness" is: how dark the core of the thicker stroke parts is relative to the parchment next to it (0 = parchment, 1 = black).

## PHASE 2d PRE-REGISTRATION: darkness on the 2014 photographs (written before unsealing). Seed 20261124.
D_net = log darkness, glyph-count adjusted (PRIMARY). D_adj = D_net further adjusted at word level for focus reading and log stroke width.
NAIVE BASELINE = log of the crop's 98th-percentile darkness ("peak"), glyph-adjusted. Page >= 20 words. One campaign only: no replication possible
except a descriptive correlation with the (invalid, blur-affected) 2004 darkness.
GATE: split-half reliability of page D_net >= 0.80, else stop.
D1 neighbours: correlation of darkness between a word and the next-but-one / next-but-two in the line (crops cannot overlap). INK CARRY-OVER if z >= 3 at lag 2.
D2 along the line: reading part a and gutter part b (as P5-2). D3 down the page. Trend if |z| >= 3; no trend if |z| < 2 with planted 2% caught >= 80%.
D4 conjugates, within-quire null, all words and fore-edge half: SHARED if z >= 3 on both.
D5 page pairs (recto/verso, facing, same sheet) and through-the-book trend (planted 2-SD drift for power).
D6 hands: herbal sheets, hand 2 vs hand 1 (label shuffle; planted 5% and 10%).
D7 orders: neighbour smoothness along Path 1, label-free, binding; null within section x hand. z >= 3 confirms for that order.
D8 crease six: mean pairwise gap vs random sixes of herbal hand-1 sheets. z >= 3 confirms.
Every test also run on D_adj and the naive baseline; a result counts only if D_net and D_adj agree (same sign, D_adj |z| >= 2).

## PHASE 2d RESULTS (2026-11-14 13:40): ink darkness, 2014 photographs, 151 pages, 11,309 words
GATE: split-half reliability 0.993 PASS. Median darkness 0.42 (pages 5-95%: 0.29-0.50). Correlation with the (blur-affected) 2004 darkness: r = 0.949.
D1 neighbours: lag 1 z 16.4, lag 2 z -1.98, lag 3 z -6.8 -> ink carry-over between words NOT shown (lag 1 = overlapping crops).
D2 along the line: reading part -0.18% (null SD 0.0029, z -0.61; planted 2% caught 100%) -> NO TREND. Gutter part +0.5% (z 1.7; D_adj z 5.1).
D3 down the page: -1.5% (effect -0.0155, null SD 0.0050, z -3.06; D_adj z -2.74) -> TREND by the rule: ink slightly lighter towards the foot. One campaign.
D4 conjugates, within-quire null (37 sheets): gap 0.057 vs 0.110, null SD 0.010, z 5.18; fore-edge half z 4.60; D_adj 4.43 / 4.24 -> SHARED.
D5 page pairs (z): same sheet same face 4.03; same sheet opposite face 2.48; recto/verso 2.27; FACING -0.57.
   Through the book (79 leaves): rho 0.23, null SD 0.079, z 2.41 (D_adj 2.95); planted 2-SD drift caught 100% -> inconclusive (between 2 and 3).
   By stratum: herbal hand 1 rho 0.36 (42 leaves), herbal hand 2 0.70 (10), balneo 0.07, stars -0.18.
D6 hands (herbal, 24 v 5 sheets): hand 2 +9.4% darker, null SD 0.061, ratio 1.49 -> INCONCLUSIVE AT THIS POWER (planted 10% caught 7%).
D7 orders (48 sheets): Path 1 gap 0.096 vs 0.134, null SD 0.0099, z 3.79 (D_adj 2.39; naive peak 3.63) -> CONFIRMS for Path 1 by the rule.
   Label-free z 2.81 (D_adj 1.57) -> not confirmed. Binding z -0.08 -> nothing.
D8 crease six: gap 0.095 vs 0.147, null SD 0.034, z 1.52 -> INCONCLUSIVE AT THIS POWER.
OPEN CONFOUND: darkness is measured against the parchment beside the stroke. The two halves of a sheet are one piece of skin, so shared parchment
tone could produce D4/D5 without any shared ink.
## BOUNDING Db1 (registered before running): adjust page darkness for page parchment brightness (median log background B) and rerun D4, D5 same-face,
D7. NOT PARCHMENT TONE if D4 stays z >= 3 and D7 Path 1 stays z >= 2. Also report whether parchment brightness itself is shared by conjugates and
smooth along the orders.

## Db1 RESULT (2026-11-14 13:50): NOT PARCHMENT TONE by the rule
Page darkness vs parchment brightness r = 0.23. After removing parchment brightness: conjugates z 4.92 (null SD 0.011); Path 1 z 3.49 (null SD 0.0094);
label-free 2.66; binding -0.37.
Parchment brightness itself: shared by conjugates z 3.70; smooth along BINDING order z 4.05 (null SD 0.0022); Path 1 z 1.01; label-free -0.09.
-> a dissociation: the skin follows the binding order, the ink darkness follows Path 1. (Parchment-along-binding also works as a positive control:
   the method can see a real physical gradient when there is one.)
Not yet bounded: (1) second campaign with valid darkness (2004 is too blurred; its darkness still correlates 0.95 with 2014 at page level);
(2) Path 1 leave-one-quire-out; (3) whether Path 1 neighbours are simply same-Currier-language pages inside section x hand strata;
(4) fresh-chat demolition. Analytical claims flagged: "sheets were inked at one sitting", "Path 1 tracks ink batches".
P(survives 60d): conjugates share ink darkness 60%; ink darkness smoother along Path 1 than chance 45%; parchment brightness follows binding order 75%;
ink lighter towards the foot of the page 30%; crease six: no estimate (inconclusive).
Files: phase1d.py, analysis6.py, phase2d_results.json, phase2d_results_table.csv, phase2d_pages.csv, phase2d_sheets.csv, phase2d_db1_results.json.

## PHASE 6 PRE-REGISTRATION (2026-11-14 13:30, Ed: "Run it"): batch or sequence? Sheet-level darkness, 2014 photographs. Seed 20261126.
Measures: D_net (primary), D_B (parchment-adjusted). Planted-signal checks run on label-shuffled values and are printed before the real result.
T-A LEAVE ONE QUIRE OUT: D7 Path 1 smoothness recomputed with each quire's sheets removed. ROBUST if every z >= 2 and the median >= 3. FRAGILE if any z < 1.5.
T-B DIRECTION inside herbal hand 1 only (the one stratum big enough): Spearman between position in the order and darkness, for Path 1, label-free,
    binding. Null: plain shuffle. Power: planted straight drift of 1 and 2 SD end to end. DIRECTIONAL if |z| >= 3; NOT DIRECTIONAL (at 2 SD) if |z| < 2
    and the 2-SD plant is caught >= 80%; else inconclusive at this power.
T-C SEQUENCE vs BATCH: neighbour smoothness along Path 1 against a null that only shuffles sheets WITHIN blocks of 6 consecutive Path 1 positions
    (and within section x hand). If only the rough grouping matters, local shuffling costs nothing. Known-answer check first: (a) planted pure batch
    signal (block offsets) must NOT fire (false alarms at z >= 3 reported); (b) planted smooth sequence signal (random walk, same variance) -- power.
    SEQUENCE if z >= 3. BATCH-ONLY if z < 2 and planted sequence caught >= 80%. Otherwise inconclusive at this power. Block sizes 4 and 8 logged as variants.

## PHASE 6 RESULTS (2026-11-14 13:45)
T-A leave one quire out (14 quires): Path 1 smoothness z ranges 2.76-3.52, median 3.36 (parchment-adjusted 2.37-3.40, median 3.13) -> ROBUST by the rule.
    No single quire carries it.
T-B direction inside herbal hand 1 (24 sheets): Path 1 rho 0.35, null SD 0.21, z 1.65; label-free rho -0.27 (z -1.31); binding rho 0.33 (z 1.58).
    Planted 2-SD drift caught 16%, 3-SD 62% -> INCONCLUSIVE AT THIS POWER. 24 sheets cannot show a direction unless it is very large.
T-C sequence vs batch (blocks of 6): gap 0.096 vs 0.115, null SD 0.009, z 2.04 (parchment-adjusted 1.68). Known-answer check: planted batch signal gave
    0% false alarms (good), planted sequence signal was caught only 8-10% (bad) -> INCONCLUSIVE AT THIS POWER. Variants: blocks of 4 z 3.12, of 8 z 2.83
    (not the registered primary; not to be quoted as the result).
Context: Path 1 neighbours are the same Currier language 98% of the time and the same quire only 32% (binding-order neighbours: 72% same quire).
VERDICT: the darkness-along-Path-1 result is robust to any one quire. Whether Path 1 is a SEQUENCE or just a correct GROUPING cannot be decided with
48 sheets by these tests. That is a data limit: the sheet is the unit of inking, and there are only 48 of them in his order.
P(survives 60d): darkness smoother along Path 1 than chance 50% (was 45%); "Path 1 is a true sequence" -- no estimate, untested at usable power.
Files: phase6.py, phase6_results.json, phase6_sheets.csv.

## PHASE 7 PRE-REGISTRATION (2026-11-14 13:35, Ed: "Do the page level work"). Darkness, 2014 photographs, pages. Seed 20261127.
Non-outcome check done first: page SD of D_net 0.156; within-sheet page SD 0.071; 151 pages, 48 sheets, 23 sheets with all four pages.
PL-0 USABLE? within-sheet page SD must be >= 3x the measurement noise SD (from odd/even word halves). Else stop.
PL-1 DOES PAGE ORDER CARRY INK INFORMATION BEYOND THE SHEET? Statistic: mean |difference| in darkness between neighbouring pages along eo777's page
     path (page_path_rank). Null: shuffle the page values WITHIN each sheet (sheet totals untouched, so the sheet-level result cannot leak in).
     Comparator order: plain reading order (leaf number, recto then verso). Power: planted smooth signal along the page order with SD = 0.5x and 1x the
     within-sheet SD. YES if z >= 3. NO if z < 2 and the 1x plant is caught >= 80%. Otherwise inconclusive at this power.
PL-2 DIRECTION at page level, herbal hand 1: Spearman of page darkness against page-path position. Null keeps sheets intact (sheet means shuffled
     between sheets, within-sheet offsets kept). Planted 2-SD and 3-SD drift. Same rule as T-B.
PL-3 WHAT VARIES INSIDE A SHEET (descriptive, sign-flip null over sheets): recto minus verso; first leaf minus second leaf; outer face minus inner face
     of the flat sheet. |z| >= 3 = a consistent within-sheet difference.

## PHASE 7 RESULTS (2026-11-14 13:50)
PL-0: measurement noise SD 0.0126; within-sheet page SD 0.0711; ratio 5.6 -> USABLE. Pages of one sheet really differ in darkness.
PL-1 (neighbour-difference statistic): page path z 0.14 (null SD 0.0047, 144 pages); reading order z -0.79; herbal hand 1: -0.15 / -0.55.
     Planted 1x signal caught 0.5-6% -> INSTRUMENT HAS NO POWER; inconclusive, carries no weight either way. (Same weakness as T2 and T-C: the
     neighbour-difference statistic is poor at smooth signals.)
PL-2 direction, herbal hand 1 (69 pages, 24 sheets): rho 0.19, null SD 0.18, z 0.94; planted 3-SD drift caught 88%, 2-SD 33% -> no drift of 3 SD or more;
     smaller drifts not testable.
PL-3 inside a sheet: recto minus verso -0.8% (z -0.54); first minus second leaf +0.4% (z 0.31); outer minus inner face -1.8% (z -1.02).
     No consistent within-sheet pattern.
## PHASE 7b: NEW INSTRUMENT for PL-1, developed on planted data only (real page order stays sealed for this statistic)
Statistic: correlation between a page's offset from its sheet mean and the offset of its neighbourhood (mean darkness of the k pages either side in
the order, taken from OTHER sheets only), null = shuffle page values within sheet. Phase 1: choose k from {3,5,8} by power on planted smooth signals
(0.5x and 1x within-sheet SD) and check false alarms on pure within-sheet-shuffled data. ACCEPT only if the 1x plant is caught >= 80% with false alarms
<= 2%. If not accepted: do NOT run on the real order; declare the page-level question unanswerable with these data.
Phase 2 rule (if accepted): YES if z >= 3; NO if z < 2; between = inconclusive. Run on page path and on reading order.
7b variant v1 (neighbourhood = raw darkness of other-sheet neighbours): k=3 false alarms 0%, planted 1x caught 2% -> REJECTED in development (between-sheet differences swamp it); k=5,8 run stopped unfinished. Variant v2: neighbourhood = other-sheet neighbours' OFFSETS from their own sheet means. Same acceptance rule.
7b variant v2 development: k=3/5/8 false alarms 0%, planted 1x caught 1.7% / 1.7% / 0% -> REJECTED. No variant accepted; per the rule the real page
order was NOT run with this statistic. (v1 real-order result does not exist; only the powerless neighbour-difference PL-1 above was run.)
WHY (checked after rejection, descriptive): see the span figures logged below.
SPAN CHECK: in eo777's page path the pages of one sheet sit within a median span of 8.5 positions out of 144 (random expectation 83.7, SD 6.1);
91% of sheets span fewer than 20 positions. His page order keeps each sheet's pages together, so it adds almost no information beyond the sheet order,
and a null that holds sheets fixed has nothing left to test. (This also means his text statistics independently group pages by physical sheet.)
PHASE 7 VERDICT: page-level darkness is measurable (within-sheet differences 5.6x noise) but CANNOT test page order beyond the sheet with these data.
No within-sheet pattern by side, leaf or face. No page-level drift of 3 SD or more in herbal hand 1. Sequence-vs-grouping stays open.
P(survives 60d): "his page path clusters pages by sheet" 85%; "pages of a sheet differ in darkness beyond noise" 80%.
Files: phase7.py, phase7b.py, phase7_results.json, phase7b_dev.json (v2 only; v1 dev stopped), phase7_pages.csv.

## PHASE 8 PRE-REGISTRATION (2026-11-14 13:55, Ed: "What about singulion"). Same design as horseshoe PRE-REGISTRATION 8 (2026-10-14, text), now on ink.
Checked chat history first: Davis's hypothesis = the MS began as a stack of independently written singulions; our Oct text test asked FACE vs SEQUENCE.
Data: 2014 darkness (D_net; D_B parchment-adjusted), sheets with all four single-panel pages. Pages ar, av (first leaf), br, bv (second leaf).
Singulion reading order ar, av, br, bv: distance 1 = (ar,av), (av,br), (br,bv); distance 2 = (ar,br), (av,bv); distance 3 = (ar,bv).
Per sheet: cross = mean of the two distance-2 gaps. D_inner = cross - |av - br|;  D_outer = cross - |ar - bv|;  D_leaf = cross - mean(|ar-av|, |br-bv|).
Sign-flip null over sheets, 100,000. Seed 20261129.
H_SEQUENCE (written/inked in singulion reading order): D_inner > 0, D_leaf > 0, D_outer < 0 (outer pair is furthest apart).
H_FACE (each face of the flat sheet inked as a unit): D_inner > 0 and D_outer > 0, D_leaf <= 0.
VERDICT on D_outer: FACE if z >= 2; SEQUENCE if z <= -2; otherwise NOT RESOLVED. Power first: plant a sequence drift (step per page = 0.5x and 1x
within-sheet SD) and a face offset (same sizes) into within-sheet-shuffled real values; report how often each verdict comes out.
Already known and consistent with ANY sheet-as-unit model (singulion or flat sheet): conjugates share darkness (z 5.2), facing pages do not (z -0.6).

## PHASE 8 RESULTS (2026-11-14 14:05): singulion test on ink darkness, 22 complete sheets
POWER (printed first): no structure -> wrong verdict 3% FACE, 1% SEQUENCE. Planted 1x within-sheet SD: sequence caught 90%, face caught 81%.
 Planted 0.5x: sequence 14%, face 10%. So only strong effects are visible.
D_outer (the deciding quantity): mean 0.017, null SD 0.012, z 1.44 (parchment-adjusted 0.93) -> NOT RESOLVED. Inconclusive at this power.
D_inner: 0.033, null SD 0.012, z 2.73 (adj 2.28): the inner pair (av, br) is the closest pair, 77% of sheets. Both hypotheses predict that.
D_leaf: 0.014, null SD 0.008, z 1.76 (adj 1.49).
Mean gaps: distance 1: ar-av 0.089, av-br 0.061, br-bv 0.073; distance 2: ar-br 0.084, av-bv 0.106; distance 3 (outer, ar-bv) 0.078.
 The outer pair is NOT the furthest apart, which is the opposite of what reading-order inking predicts; but not at z <= -2 or >= 2.
Mean darkness by position vs sheet mean: ar +1.2%, av +1.7%, br -0.4%, bv -2.5% (descriptive; first-vs-second leaf was z 0.31 over 36 sheets).
READING: a strong singulion-sequence gradient (one within-sheet SD per page) is unlikely (would have been caught 90%). A strong face effect likewise
(81%). Weaker versions of either cannot be seen with 22 sheets. What the ink does support is the shared premise: the sheet is the unit.
P(survives 60d): "no strong reading-order ink gradient inside a sheet" 60%; face vs sequence -- no estimate, unresolved.
Files: phase8.py, phase8_results.json.

## PHASE 9 PRE-REGISTRATION (2026-11-14 14:00, Ed: "singulion reading order within and across quires?"). Darkness, 2014. Seed 20261130.
If the book was a stack of singulions read ar,av,br,bv then the next sheet's ar, the join between consecutive sheets is (bv of one, ar of the next).
In the book as bound (nested), the joins are the facing pages (av of outer sheet, ar of next inner) and (bv of inner, br of outer).
S1 SHEET LEVEL (already run as D7 "binding"): is sheet darkness smooth along the present stacking order (quire by quire, outer sheet to inner)? Reported again.
   Added now: the same within quires only (neighbouring sheets of the same quire), null = shuffle sheets within quire.
S2 PAGE JOINS, within quires, sheets adjacent in nesting order: gap at the singulion join (bv_k, ar_k+1), at the reverse-stack join (bv_k+1, ar_k), and at
   the two bound facing joins, each compared with the mean gap of all other page pairs between the same two sheets. Null: shuffle pages within each
   sheet. Power: plant a shared offset (1x within-sheet SD) on the join pages. JOIN IS REAL if z >= 3; ABSENT if z < 2 with plant caught >= 80%.
S3 ACROSS QUIRES: same four joins between the last sheet of one quire and the first of the next (for the singulion stack: innermost sheet of quire q to
   outermost of q+1; for the bound book: outermost to outermost). Few joins; power reported.

## PHASE 9 RESULTS (2026-11-14 14:10)
S1 sheets next to each other in the present nesting order, within quires (34 pairs, 13 quires): gap 0.110 vs 0.114 for random same-quire sheets,
   null SD 0.0075, z 0.46. Whole-book stacking order (D7 "binding"): z -0.08. -> no ink continuity between neighbouring sheets as now arranged.
S2 page joins within quires: singulion join (bv -> next ar) is slightly WORSE than other page pairs of the two sheets: gain -0.026, null SD 0.016,
   z -1.55 (24 joins). Reverse stack: z 0.33 (23). Bound facing joins: z 0.93 (18).
   Power: planted "join pages made equal" caught 100% / 100% / 65%. That plant is the strongest possible; milder continuity was not planted -> ABSENT
   only in the strong sense for the singulion joins; bound facing joins inconclusive at this power.
S3 across quires: singulion stack join z -0.67 (5 joins); reverse 1.01 (4); bound facing 1.00 (2). Bound-book arrangement: -0.64 (7), -0.42 (6), 0.26 (4).
   Too few joins to say anything.
READING: in the order the sheets sit in today, the ink shows no reading-order continuity of either kind (singulion or nested), within or across quires.
Together with D7 (ink smooth along Path 1, z 3.8, not along binding) this says the present sheet order is not the inking order. It does not show what the
original order was, nor that the sheets were read as singulions.
P(survives 60d): "no strong ink continuity between neighbouring sheets in the present order" 65%.
Files: phase9.py, phase9_results.json.

## CONSOLIDATION TO SUPABASE (2026-11-14 14:05), project ymaqlcfjmdwncdbjprmw
New tables: vms_ink_findings_v01 (28 findings), vms_ink_page_values_v01 (154 pages). Added runs INK7, INK8, INK9, INK11, INK12; results now 203 rows;
pre-registrations 14; retractions 13; README_eo777_ink_session.md stored in vms_crease_files_v01 (index of everything).
NOT uploaded: word-level rows (21,797), full running files for order/spatial/ink, most code -- too large to pass through the SQL tool. A temporary
anon-callable ingest function to bulk-load them was BLOCKED by the permission system and not created. They are in eo777_ink_session_bundle.zip
(sha256 6c848a0d9118384cef80afb88aef3f751c10fb58922099083f07b2b909d9f671; this note itself was added after the bundle was built).
UPDATE 14:20: bulk files went to GitHub instead (digitalgoldfisj79/Voynichdecomp, branch eo777-ink-session-2026-11-14, first commit 08be9eb); Supabase
vms_crease_files_v01 holds pinned pointers. Second commit adds the intermediate pickles the scripts load (null_depth.pkl, pairs_cut3.pkl, phase2.pkl,
phase5.pkl, phase4_words.pkl), eo777 label-free fit outputs, run logs and three crease zoom crops. Not archived anywhere: source images (re-downloadable;
checksums in tif_sha256.csv and the page pickles), overlay PNGs, the Python environment, clones of eo777's repos.

## PHASE 10 PRE-REGISTRATION (2026-11-14 14:10, Ed: "proceed with follow up tests"). Singulion joins ALONG eo777's sheet order. Darkness, 2014. Seed 20261201.
Question: if the sheets are put in Path 1 order and each is read as a singulion (ar, av, br, bv), does the ink run on from one sheet into the next?
J1 JOINS between consecutive sheets in the order: singulion join (bv_k, ar_k+1); reverse-direction join (ar_k, bv_k+1, i.e. the order read backwards);
   comparator "nested" joins (av_k, ar_k+1) and (br_k+1, bv_k). Gain = mean gap of the other page pairs of the two sheets minus the join gap.
   Null: shuffle pages within each sheet (sheet means untouched, so the known sheet-level smoothness cannot leak in).
J2 DIRECTION: correlation, over sheets, between the within-sheet change (bv - ar) and the change between neighbours in the order (mean of next sheet
   minus mean of previous sheet). Positive = the inside of a sheet continues the drift of the order, as a singulion sequence requires.
   Null: shuffle pages within each sheet.
Orders: Path 1 (primary), label-free, present binding order (comparator).
Power (printed first), planted on within-sheet-shuffled values: (a) strongest: join pages made equal; (b) mild: a smooth random walk over the full
reading sequence (4 pages per sheet in order), SD = 1x and 2x the within-sheet SD.
RULE: REAL if z >= 3. ABSENT if z < 2 and the 1x mild plant is caught >= 80%. Otherwise inconclusive at this power (stating which plant sizes were visible).

## PHASE 10 RESULTS (2026-11-14 14:25)
POWER: strongest plant (join pages made equal) caught 100% on every order. Mild plants (smooth walk over the reading sequence, 1x and 2x the within-sheet
 SD 0.066) caught 0-2% by the join test and 0% by the direction test. So only a seamless join is visible; realistic continuity is not.
J1 Path 1 (28 joins): singulion join gain 0.013, null SD 0.017, z 0.80; reverse-direction join -0.034, null SD 0.016, z -2.18; nested joins z 1.43 (18).
   Label-free: 0.62 below / -0.35 / 0.33 (z -0.62, -0.35, 0.33). Binding: z -1.74, 0.75, -0.30.
J2 direction, Path 1 (30 sheets): corr +0.48, null SD 0.27, z 1.80. Label-free -0.06 (z -0.31). Binding -0.12 (z -0.65).
VERDICT: INCONCLUSIVE AT THIS POWER on every order. A seamless singulion join along Path 1 is ruled out (would be caught 100%); anything milder is untestable
 with 28 joins. The Path 1 direction correlation (+0.48) has the sign a singulion sequence predicts and is absent on the other two orders, but at ratio 1.8
 and with zero measured power it is a hint only (unverified).
P(survives 60d): "no seamless ink join between consecutive Path 1 sheets" 70%; singulion sequence along Path 1 -- no estimate.
Files: phase10.py, phase10_results.json, phase10_results_table.csv.
