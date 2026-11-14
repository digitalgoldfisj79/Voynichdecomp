# eo777 / npcompl33t thread-6299 crease claim — running results (2026-11-13)

## RETRACTED FINDINGS
- 2026-11-13 19:20 RETRACTED: "quire-centre position explains the bottom-crease adjacency in Golf" (R2 implication).
  The pre-registered depth-matched null gives p_D = 0.0008 (N6, path 1); it also survives depth x Currier (p = 0.003)
  and depth x hand (p = 0.0026). R1 (centre clustering) was overstated: on the CSV it is p_U = 0.028, not 0.0086.

## Status: POST-HOC. Not pre-registered. The quire-centre pattern was spotted while looking at his chart,
then tested. Treat as a hypothesis for a pre-registered rerun on golf-v21-assignments.csv, not a result.

## Inputs
- 9 images from the thread (uploaded by Ed). Golf row order transcribed BY EYE from his
  "Bottom creases / warping" chart, rows 1-22 and 30-37 (rows 23-29 not shown in the chart).
- Quire structure assumed: Q1-Q7 = eight-leaf quires f1-8 ... f49-56 (f12 missing). Unverified against Zandbergen here.

## R1 Golf order clusters quire-centre bifolia (descriptive, from his own chart)
All 7 centre bifolia (4-5, 12-13, 20-21, 28-29, 36-37, 44-45, 52-53) sit at ranks 6-18 of the 30 shown rows.
P(7 random ranks fall within a span of <=13 of 30) = 0.0086 (exact count).
P(survives 60d) = 55% (transcription by eye; hidden rows 23-29; needs rerun on the CSV).

## R2 Damaged leaves over-represent centre bifolia
Of the 9 listed bottom-damage leaves, 5 (f5, f21, f29, f37, f52) are on a centre bifolium. Base rate 2/8.
Binomial P(>=5 of 9 | p=0.25) = 0.049. Inconclusive at this power: n=9, the selection was by eye, and no planted-signal test was run.
P(survives 60d) = 35%.

## Implication (analytical claim — needs external test before relying on)
If R1 and R2 both hold, "damage set is consecutive in Golf" can come from BOTH correlating with
position in the quire, without any earlier side-by-side arrangement of the leaves.
Discriminating test: rerun the damage-vs-Golf adjacency with a null that permutes Golf rows WITHIN
nesting depth (centre / 2nd / 3rd / outer). If adjacency survives that null, his claim stands; if not, quire position explains it.

## Visual observations (eyeball, not measured)
- f5r: a real fold, with the vellum doubled back. f21r: a faint depression only.
- Tail-edge zoom: the V-dip seems to carry through the layers BETWEEN f5 and f21, which fits pressure
  in the CURRENT binding rather than the two leaves having been side by side.
- Crease distance from the spine varies widely across the 9 panels (roughly 15%-60% of page width,
  allowing for r/v mirroring). A single stacked event would put the creases at about the same distance from the fold.
- Fore-edge set: 33-40 and 34-39 are ALREADY adjacent nested bifolia in Q5. A crease on both leaves of a
  bifolium is expected whatever the order, because the two fore-edges lie together. He notes himself that the crease heights differ.

## PRE-REGISTRATION N1 (2026-11-13 ~19:10, written BEFORE computing anything below)
Data: github.com/eo777/voynich-clusters @ 778066a (28 Oct 2026), golf-v21-assignments.csv,
bifolium_path_rank / _2 / _3 / _4 (path 1 = default). Also his chart order as transcribed (sensitivity check only).
Universe: bifolia of quires A-H (f1-f66, 30 bifolia), re-ranked 1..30 in Golf order after dropping the non-herbal ones.
Depth = number of the quire's bifolia nested inside this one (0 = quire centre).
Damage sets (from his charts): BOTTOM-N6 = {4-5,10-15,20-21,25-32,28-29,36-37} (PRIMARY);
BOTTOM-N5 = N6 without the optional 25-32; BOTTOM-L9 = N6 + {9-16, 27-30, 52-53};
FORE-F4 = {33-40, 34-39, 50-55, 57-66}.
Statistic: span of Golf ranks (max-min+1). Secondary: mean pairwise |rank difference|.
Nulls, 20,000 draws, seed 20261113: U = k bifolia drawn at random from the universe;
D (matched-random) = each damaged bifolium replaced by a random bifolium of the SAME depth, drawn without replacement.
DECISION on PRIMARY (N6, path 1, null D):
  p_D >= 0.05 -> FALSIFIED: Golf adjacency of the damaged set is explained by quire position.
  p_D < 0.01  -> CONFIRMED: adjacency survives the quire-position confound.
  otherwise   -> Inconclusive.
POWER: the 1st-percentile span under D. If even a perfectly contiguous span (=k) cannot reach p<0.01 under D,
report "Inconclusive at this power" whatever the p-value.
Also re-test R1 (do the centre bifolia cluster in Golf?) on the CSV.
My prediction (P=0.6): p_U < 0.01 but p_D >= 0.05 (confound explains it).

## N1 RESULT (pre-registered primary): N6, path 1, null D: span 7 vs null 20.59 ± 3.89 (z = 3.49), p_D = 0.0008 -> CONFIRMED by the pre-registered rule.
Power OK: a perfectly contiguous span reaches p_D = 0.00005. My prediction (confound explains it) was WRONG.
R1 on the CSV: the centre bifolia span 17 of 30, p_U = 0.028 (my chart-based 0.0086 overstated this).
L9 (adding f9?, f30, f52): p_D = 0.25 (path 1). The result depends on which damaged leaves are nominated.
FORE-F4: p_D = 0.019 (path 1): inconclusive by the same rule.

## BOUND N2 (EXPLORATORY, written before running): the same statistic with matched nulls
D-lang = same depth x same bifolium Currier label; D-hand = same depth x same Davis hand.
If p rises above 0.05 under either, the N1 confirmation is explained by language or hand, not by physical adjacency.

## N2 RESULT (exploratory bound)
BOTTOM-N6 path 1: depth x Currier p = 0.0030 (span 7 vs 16.04 ± 2.79, z = 3.25); depth x hand p = 0.0026 (z = 3.28).
Chart order: p = 0.0025 for both. N5 (without f32): p = 0.007-0.008.
FORE-F4: depth x Currier p = 1.0; depth x hand p = 1.0 (chart order p = 0.67). The fore-edge set is EXPLAINED by Golf
already grouping the Herbal B / hand 2 bifolia together. No support for the fore-edge claim.

## STANDING (as of 2026-11-13)
- Bottom-crease set N6 is adjacent in Golf beyond depth, language and hand. P(survives 60d) = 45%.
  Main threat: was N6 (vs the 9 he listed) chosen knowing Golf? L9 does NOT survive (p_D = 0.25).
  Needs: blind crease scoring of all herbal leaves by someone who has not seen Golf, then rerun N1.
- Fore-edge claim: explained by language/hand. P(that this verdict survives 60d) = 80%.
- Physical: f5 fold is real; whether f21 is a matching depression vs stack-wide pressure is unresolved (eyeball only).

## VoynichStats quire_optimize (within-quire, A-E, cosine/word, 500 runs, seed 42): descriptive only
A: [3,1,2,4] exploratory; B: [4,3,1,2] exploratory; C: unchanged; D: [2,3,4,1] exploratory; E: unchanged.
It works WITHIN quires only, so it cannot test cross-quire adjacency. Logged for completeness.

## PRE-REGISTRATION M1 (2026-11-13 ~19:40, before computing): maths-only cross-quire test
Unit: bifolium = all pages of both its leaves (ZLZI, VoynichStats corpus_dump), quires A-H, 30 bifolia.
Similarity: raw word-count cosine (as quire_optimize). Statistic: mean pairwise cosine among BOTTOM-N6.
Nulls (20,000 draws, seed 20261115): D = same depth; DL = same depth x Currier; DH = same depth x hand.
CONFIRM (text alone makes N6 a group): p < 0.01 under DL. FALSIFY: p >= 0.05 under DL. Else inconclusive.
Naive control (rule 19 analogue): the same on character-bigram cosine; if it gives the same or a stronger result,
the signal is generic text profile, not vocabulary.
Note: Golf is itself built from text, so CONFIRM would mostly re-derive Golf, not add independent support for the creases.
Prediction: P(confirm) = 0.5.

## M1 RESULT (pre-registered): CONFIRM
Word cosine, BOTTOM-N6: mean pairwise 0.708 vs depth x Currier null 0.583 ± 0.039 (z = 3.21), p = 0.0012;
depth x hand p = 0.0011; depth only p = 0.00025.
Naive control (character-bigram cosine): z = 1.70, p = 0.047 under DL. Weaker, so the signal is vocabulary, not a generic text profile.
BOTTOM-L9 (all nine): word DL p = 0.019 (z = 2.21). The full nine are also somewhat grouped by text, unlike on Golf ranks (p = 0.25).
Length check: the N6 bifolia are SHORTER (mean 289 words vs 398 for the rest), which should LOWER cosine. So length does not explain it.
Interpretation: text alone, with no Golf algorithm, re-derives the grouping. This is NOT independent support for the
crease claim, because Golf is text too. It does show the grouping is not an artefact of Golf's path-fitting.
P(survives 60d) = 60%.

## PRE-REGISTRATION M2 (2026-11-13 ~19:45, before computing): leaf-level (folio) version
Unit: leaf = recto + verso ZLZI words, quires A-H. Word cosine, as in M1. Statistic: mean pairwise cosine of the set.
Sets: DAMAGED-6 = f5, f15, f21, f29, f32, f37; CONJUGATE-6 = f4, f10, f20, f25, f28, f36 (their partner leaves);
DAMAGED-5 = without f32; LISTED-9 = DAMAGED-6 + f9, f30, f52.
Null (20,000 draws, seed 20261116): matched on depth x side (A/B leaf) x leaf Currier (primary); also depth x side x hand.
CONFIRM (damaged leaves group on text): DAMAGED-6 p < 0.01. FALSIFY: p >= 0.05.
SHEET-vs-LEAF: if CONJUGATE-6 is also p < 0.01 with z within 1 of DAMAGED-6, the grouping belongs to the sheet,
not to the damaged leaves. If CONJUGATE-6 p >= 0.05, the grouping is specific to the damaged leaves.
Naive control: character-bigram cosine.
Prediction: P(DAMAGED-6 confirm) = 0.55; P(conjugates also grouped) = 0.5.

## M2 RESULT (pre-registered)
Word cosine, depth x side x Currier: DAMAGED-6 0.576 vs 0.457 ± 0.040, z = 2.99, p = 0.0008 -> CONFIRM.
CONJUGATE-6 0.548 vs 0.427 ± 0.047, z = 2.57, p = 0.006 -> also < 0.01, and z within 1 (difference 0.42)
-> by the pre-registered rule the grouping BELONGS TO THE SHEET, not to the damaged leaves.
(Hand null: damaged p = 0.00015, z = 2.98; conjugates p = 0.0125, z = 2.13: a gap of 0.85, still within 1.)
DAMAGED-5 p = 0.004; LISTED-9 p = 0.0076 (word, Currier null).
Naive bigram: damaged z = 2.09 (p = 0.009); conjugates z = 1.36 (p = 0.11). Weaker than word in both, as expected.
Caveat: the smallest stratum pool is 4 leaves, so the nulls are tightly constrained.
Consistent with the Oct 2026 finding that conjugate leaves share a text state.
Implication: text cannot tell which leaf of a sheet was damaged. The "all on the B side" observation is physical only.
P(survives 60d) = 60%.

## PRE-REGISTRATION M3 (2026-11-13 ~19:55, before computing): are the plant images grouped? (his "illustrations are similar")
Data: Supabase herbal_objects, VMS, embed_variant IS NULL (legacy raw-crop pipeline only: never mixed with v2-masked),
crop_qa IS NULL (spurious fragments excluded), 3072-d embedding. part = 'plant' (PRIMARY), 'root' (secondary).
Bifolium vector = mean of the L2-normalised plant vectors on its pages. Quires A-H only.
Statistic: mean pairwise cosine among the set's bifolium vectors. Nulls as in M1 (depth x Currier PRIMARY; depth; depth x hand), 20,000 draws, seed 20261117.
Sets: BOTTOM-N6 bifolia; FORE-F4 bifolia; leaf level: DAMAGED-6 vs CONJUGATE-6 (null depth x side x Currier).
CONFIRM (his plants look alike): BOTTOM-N6 p < 0.01. FALSIFY: p >= 0.05. Else inconclusive.
Power: a planted check is not possible without an image control. I report the null SD and the smallest z the null can produce instead.
Prediction: P(confirm) = 0.35 (image embeddings are dominated by page and style nuisance; see comparanda-vlm-plants).

## M3 RESULT (pre-registered): Inconclusive at this power (formally the FALSIFY branch, p >= 0.05)
Plant crops, bifolium level, depth x Currier: BOTTOM-N6 0.9589 vs 0.9586 ± 0.0049, z = 0.05, p = 0.47.
FORE-F4: z = 0.07, p = 0.41. Leaf level: DAMAGED-6 z = -0.95; CONJUGATE-6 z = -1.27. Roots: N6 z = 0.82 (p = 0.20); F4 z = -0.92.
All effects < 2x null SD -> Inconclusive at this power. Power NOT measured: no planted-signal control was run.
The instrument looks saturated: every bifolium-mean cosine sits at 0.90-0.97 with a null SD of about 0.005. Legacy raw-crop vectors
carry page/parchment/style nuisance (see comparanda-vlm-plants), and this probably swamps plant shape.
Do NOT report this as "the plants are not similar". Report it as "this embedding cannot see it either way".
Next: blind human rating in Ed's manuscript-by-bifolium view, or the v2-masked-gray (silhouette) pipeline after re-embedding the VMS crops.
P(that the inconclusive verdict survives 60d) = 75%.

## V1 Visual check, Beinecke IIIF full-resolution (manifest oid 2002046), eyeball only, NOT pre-registered
Prompted by the forum commenter's point that f39r has paint missing at the bottom of the crease, so the damage predates painting.
- f39r (1006150): the crease runs in from the fore-edge and crosses one leaf by about 5 mm. Along it there is a PALE LINE in the green.
  The same bright line shows on the bare parchment beside it, so at least part of it is a lighting highlight on a raised ridge.
  Highlight, abraded paint and a brush skip cannot be separated in one photo. Lean (unverified): paint worn or cracked on the ridge
  AFTER painting, because a brush rarely skips a ridge.
- f39v (1006151): the crease shows as a groove (dark line) and ends in bare parchment before reaching any paint: no test possible.
- f5r (1006084): the fold crosses the root outlines. The lower root line seems to stop at the fold and resume about 2 mm further on
  (about 25 px at roughly 0.09 mm/px, assuming a page width of about 23 cm). That would fit a strip hidden by the vellum folding over,
  i.e. a fold AFTER drawing. Suggestive only.
Needs raking light, or MSI (Lazarus 2014 set), to settle. P(f5 fold postdates ink) = 60%; P(f39r crease postdates paint) = 55%.

## PRE-REGISTRATION J1 (2026-11-14 ~06:35, before computing): crease sets on eo777's own published distances
Data: voynich-clusters@778066a research-data/pairwise/shared-combined-js.csv (also shared-word-js, shared-line-js), pool|<sheet> rows for
bifolia; f<N>r|para / f<N>v|para rows for leaves (leaf-leaf distance = mean of the page-page JS between the two leaves' pages).
Statistic: mean pairwise JS within the set (LOWER = more alike). Nulls as before, 20,000 draws, seed 20261119:
bifolia depth x Currier (primary), depth, depth x hand; leaves depth x side x Currier.
Sets: BOTTOM-N6 (primary), BOTTOM-L9, FORE-F4; leaves DAMAGED-6, PARTNER-6.
CONFIRM: N6 combined-JS p < 0.01 under depth x Currier. FALSIFY: p >= 0.05. Else inconclusive.
Prediction: P(confirm) = 0.75 (rank-based N1/N2 and word-cosine M1 both confirmed).

## PRE-REGISTRATION L1 (2026-11-14 ~06:35, before running): his label-free fit
Run: text-order/fit.py --format ivtff --units zattera --pool (50 pooled bifolia, ZL only), defaults (100 samples, 8 starts, 2 penalties),
no Golf labels, no direction, no order filters. Compare its scores with published Path 1 bifolium scores (ZL+RF, label-based starts).
|Spearman| >= 0.90 -> the bifolium order does NOT depend materially on Golf-label starts/filters.
|Spearman| < 0.70 -> it depends substantially on them. Between -> partial dependence.
Secondary: N6 rank span on the label-free order, depth x Currier null (as N2).
Known mismatch: published = ZL+RF average; this run = ZL only. Prediction: |rho| >= 0.90 with P = 0.6.

## J1 RESULT (pre-registered): Inconclusive at this power
BOTTOM-N6, combined JS, depth x Currier: 0.0830 vs 0.0964 ± 0.0073, z = 1.85 (< 2 null SD), p = 0.042 -> between the thresholds.
Depth only: p = 0.008 (z 2.22); depth x hand p = 0.040. Word-JS p = 0.030; line-JS p = 0.048. L9: p = 0.094. FORE-F4: p = 0.26.
Leaves (depth x side x Currier): DAMAGED-6 z = 1.06 (p 0.16); PARTNER-6 z = 1.32 (p 0.09). Power not measured (no planted signal).
My prediction (confirm, P=0.75) was WRONG.
Reading: the six sit close along his 1-D ordering (N2: z 3.25) and share vocabulary (M1: z 3.21), but in his full glyph-transition
distance space they are only marginally closer to each other than matched bifolia. The crease-set support is therefore WEAKER and
feature-dependent: strong on whole-word overlap and on path position, marginal on glyph-transition distance.
P(N6 'adjacent beyond depth/language/hand' survives 60d) revised 45% -> 35%.
